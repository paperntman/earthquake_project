import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import joblib
from datetime import datetime, timedelta
import requests
import json
import os

# --- 0. 전역 설정 ---
MODEL_PATH = 'earthquake_lstm_model.h5'
SCALER_PATH = 'earthquake_scaler.gz'
RAW_DATA_PATH = 'japan_earthquake_data_raw.csv'
OUTPUT_PATH = 'latest_forecast.json'
LOG_FILE_PATH = 'automation.log'
SEQUENCE_LENGTH = 12 
GRID_SIZE = 0.5
LAT_MIN, LON_MIN = 24, 122

# --- 1. 헬퍼 함수 정의 ---
def log_message(message):
    """로그 메시지를 시간과 함께 출력하고 파일에 기록하는 함수"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
        f.write(log_entry + "\n")

def calculate_b_value(magnitudes):
    """b-value를 계산하는 함수"""
    if len(magnitudes) < 30: return np.nan
    min_mag = 2.5
    return 1 / (np.log(10) * (np.mean(magnitudes) - min_mag))

# --- 2. 메인 실행 함수 ---
def run_complete_pipeline():
    log_message("===== 자동 업데이트 및 예측 파이프라인 시작 =====")

    # --- 데이터 업데이트 ---
    log_message("### 1/4: 데이터 업데이트 시작 ###")
    try:
        if os.path.exists(RAW_DATA_PATH):
            df_old = pd.read_csv(RAW_DATA_PATH, parse_dates=['time'])
            start_date = (df_old['time'].max() + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            df_old = pd.DataFrame()
            start_date = "2000-01-01"
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date <= end_date:
            log_message(f"USGS에서 {start_date} ~ {end_date} 신규 데이터 요청...")
            url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&orderby=time&starttime={start_date}&endtime={end_date}&minlatitude=24&maxlatitude=46&minlongitude=122&maxlongitude=148&minmagnitude=2.5"
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            data = response.json()
            if data['features']:
                new_quakes = [{'time': pd.to_datetime(f['properties']['time'], unit='ms'), 'magnitude': f['properties']['mag'], 'depth': f['geometry']['coordinates'][2], 'longitude': f['geometry']['coordinates'][0], 'latitude': f['geometry']['coordinates'][1], 'location': f['properties']['place']} for f in data['features']]
                df_new = pd.DataFrame(new_quakes)
                df = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=['time', 'latitude', 'longitude']).sort_values(by='time')
                df.to_csv(RAW_DATA_PATH, index=False, encoding='utf-8-sig')
                log_message(f"{len(df_new)}개 신규 데이터 추가. 총 데이터: {len(df)}개")
            else:
                log_message("신규 데이터 없음. 기존 데이터 사용.")
                df = df_old
        else:
             log_message("데이터가 이미 최신입니다. 기존 데이터를 사용합니다.")
             df = df_old
    except Exception as e:
        log_message(f"오류: 데이터 업데이트 실패 - {e}"); return
        
    # --- 특징 생성 ---
    log_message("### 2/4: 특징 생성 시작 ###")
    df['grid_id'] = (pd.cut(df['latitude'], bins=np.arange(LAT_MIN, 47.1, GRID_SIZE), labels=False, include_lowest=True).astype(str) + '_' + pd.cut(df['longitude'], bins=np.arange(LON_MIN, 148.1, GRID_SIZE), labels=False, include_lowest=True).astype(str))
    df.dropna(subset=['grid_id'], inplace=True)
    df = df.set_index('time').sort_index()
    monthly_agg = df.groupby('grid_id').resample('ME')['magnitude'].agg(['count', 'mean']).fillna(0).reset_index()
    for w in [3, 6, 12]: monthly_agg[f'count_{w}m_avg'] = monthly_agg.groupby('grid_id')['count'].transform(lambda x: x.rolling(w, min_periods=1).mean())
    feature_rows = []
    for grid_id, group in tqdm(monthly_agg.groupby('grid_id'), desc="과학적 힌트 계산"):
        grid_df_original = df[df['grid_id'] == grid_id]
        for _, row in group.iterrows():
            current_month_end = row['time']
            start_date_1y = current_month_end - pd.DateOffset(years=1)
            past_1y_quakes = grid_df_original.loc[start_date_1y:current_month_end, 'magnitude']
            b_val = calculate_b_value(past_1y_quakes)
            quakes_before_now = grid_df_original.loc[:current_month_end]
            days_since = 9999 if quakes_before_now.empty else (current_month_end - quakes_before_now.index.max()).days
            new_row = row.to_dict(); new_row['b_value'] = b_val; new_row['days_since_last_quake'] = days_since
            feature_rows.append(new_row)
    features_df = pd.DataFrame(feature_rows).dropna(subset=['b_value']).copy()
    features_df['days_since_last_quake'].fillna(9999, inplace=True)

    # --- AI 예측 ---
    log_message("### 3/4: AI 예측 시작 ###")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        log_message(f"오류: 모델/스케일러 로드 실패 - {e}"); return
    features_to_scale = ['count', 'mean', 'count_3m_avg', 'count_6m_avg', 'count_12m_avg', 'b_value', 'days_since_last_quake']
    predict_data = features_df.groupby('grid_id').tail(SEQUENCE_LENGTH)
    valid_grids = predict_data.groupby('grid_id').filter(lambda x: len(x) == SEQUENCE_LENGTH)
    grid_ids_to_predict = valid_grids['grid_id'].unique()
    X_predict, latest_features_for_grids = [], {}
    for grid_id in grid_ids_to_predict:
        grid_data = valid_grids[valid_grids['grid_id'] == grid_id]
        X_predict.append(scaler.transform(grid_data[features_to_scale]))
        latest_features_for_grids[grid_id] = grid_data.iloc[-1]
    X_predict = np.array(X_predict)
    predictions = model.predict(X_predict, verbose=0)

    # --- 결과 저장 ---
    log_message("### 4/4: 최종 결과 저장 ###")
    forecast_list = []
    for i, grid_id in enumerate(grid_ids_to_predict):
        prob = predictions[i][0] * 100
        details_row = latest_features_for_grids[grid_id]
        lat_idx, lon_idx = map(int, grid_id.split('_'))
        south, west = LAT_MIN + lat_idx * GRID_SIZE, LON_MIN + lon_idx * GRID_SIZE
        forecast_list.append({
            "grid_id": grid_id, "bounds": [[south, west], [south + GRID_SIZE, west + GRID_SIZE]],
            "risk_probability": round(float(prob), 2),
            "details": {"count": int(details_row['count']), "mean": round(details_row['mean'], 2), "count_12m_avg": round(details_row['count_12m_avg'], 2), "b_value": round(details_row['b_value'], 2), "days_since_last_quake": int(details_row['days_since_last_quake'])}
        })
    output_data = {"update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "forecasts": forecast_list}
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f: json.dump(output_data, f, ensure_ascii=False, indent=2)
    log_message(f"✅ 예측 완료! '{OUTPUT_PATH}' 파일이 성공적으로 생성되었습니다.")
    log_message("===== 파이프라인 종료 =====")

if __name__ == "__main__":
    run_complete_pipeline()