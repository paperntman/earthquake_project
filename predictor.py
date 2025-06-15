import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import joblib
from datetime import datetime
import json

MODEL_PATH = 'earthquake_lstm_model.h5'
SCALER_PATH = 'earthquake_scaler.gz'
RAW_DATA_PATH = 'japan_earthquake_data_raw.csv'
OUTPUT_PATH = 'latest_forecast.json'
SEQUENCE_LENGTH = 12 

def calculate_b_value(magnitudes):
    if len(magnitudes) < 30: return np.nan
    min_mag = 2.5
    return 1 / (np.log(10) * (np.mean(magnitudes) - min_mag))

def generate_and_predict():
    print("(1/4) 데이터 로드 및 특징 생성 시작...")
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=['time'])
    df['grid_id'] = (
        pd.cut(df['latitude'], bins=np.arange(24, 47.1, 0.5), labels=False, include_lowest=True).astype(str) + '_' +
        pd.cut(df['longitude'], bins=np.arange(122, 148.1, 0.5), labels=False, include_lowest=True).astype(str)
    )
    df.dropna(subset=['grid_id'], inplace=True)
    df = df.set_index('time').sort_index()

    monthly_agg = df.groupby('grid_id').resample('ME')['magnitude'].agg(['count', 'mean']).fillna(0)
    monthly_agg.reset_index(inplace=True)

    windows = [3, 6, 12]
    for w in windows:
        monthly_agg[f'count_{w}m_avg'] = monthly_agg.groupby('grid_id')['count'].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
    
    feature_rows = []
    for grid_id, group in tqdm(monthly_agg.groupby('grid_id'), desc="과학적 힌트 계산"):
        grid_df_original = df[df['grid_id'] == grid_id]
        for _, row in group.iterrows():
            current_month_end = row['time']
            start_date_1y = current_month_end - pd.DateOffset(years=1)
            mask_1y = (grid_df_original.index >= start_date_1y) & (grid_df_original.index <= current_month_end)
            past_1y_quakes = grid_df_original.loc[mask_1y, 'magnitude']
            b_val = calculate_b_value(past_1y_quakes)

            mask_before = grid_df_original.index <= current_month_end
            quakes_before_now = grid_df_original.loc[mask_before]
            days_since = 9999
            if not quakes_before_now.empty:
                last_quake_date = quakes_before_now.index.max()
                days_since = (current_month_end - last_quake_date).days
            
            new_row = row.to_dict()
            new_row['b_value'] = b_val
            new_row['days_since_last_quake'] = days_since
            feature_rows.append(new_row)

    features_df = pd.DataFrame(feature_rows)
    features_df = features_df.dropna(subset=['b_value']).copy()
    features_df['days_since_last_quake'].fillna(9999, inplace=True)
    
    print("\n✅ 특징 생성 완료!")

    print("\n(2/4) AI 모델 및 스케일러 로드 중...")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ 로드 완료!")

    print("\n(3/4) 최신 데이터로 예측 수행 중...")
    features_to_scale = ['count', 'mean', 'count_3m_avg', 'count_6m_avg', 'count_12m_avg', 'b_value', 'days_since_last_quake']
    predict_data = features_df.groupby('grid_id').tail(SEQUENCE_LENGTH)
    valid_grids = predict_data.groupby('grid_id').filter(lambda x: len(x) == SEQUENCE_LENGTH)
    grid_ids_to_predict = valid_grids['grid_id'].unique()
    
    X_predict, latest_features_for_grids = [], {}
    for grid_id in grid_ids_to_predict:
        grid_data = valid_grids[valid_grids['grid_id'] == grid_id]
        X_predict.append(scaler.transform(grid_data[features_to_scale]))
        latest_features_for_grids[grid_id] = grid_data.iloc[-1] # 마지막 행의 정보를 저장

    X_predict = np.array(X_predict)
    predictions = model.predict(X_predict, verbose=0)
    print("✅ 예측 완료!")

    # ... (파일 상단과 대부분의 코드는 이전과 동일) ...

    # (4/4) 예측 결과를 JSON 파일로 저장 중...
    print("\n(4/4) 예측 결과를 JSON 파일로 저장 중...")

    # 격자 경계를 계산하기 위한 기본 정보
    GRID_SIZE = 0.5
    LAT_MIN, LON_MIN = 24, 122

    forecast_list = []
    for i, grid_id in enumerate(grid_ids_to_predict):
        prob = predictions[i][0] * 100
        details_row = latest_features_for_grids[grid_id]
        
        # grid_id (예: "29_39")로부터 격자의 경계 계산
        lat_idx, lon_idx = map(int, grid_id.split('_'))
        
        south = LAT_MIN + lat_idx * GRID_SIZE
        north = south + GRID_SIZE
        west = LON_MIN + lon_idx * GRID_SIZE
        east = west + GRID_SIZE
        
        forecast_list.append({
            "grid_id": grid_id,
            # 'bounds' 필드를 새로 추가! Leaflet이 선호하는 형식: [[남, 서], [북, 동]]
            "bounds": [[south, west], [north, east]],
            "risk_probability": round(float(prob), 2),
            "details": {
                "count": int(details_row['count']),
                "mean": round(details_row['mean'], 2),
                "count_12m_avg": round(details_row['count_12m_avg'], 2),
                "b_value": round(details_row['b_value'], 2),
                "days_since_last_quake": int(details_row['days_since_last_quake'])
            }
        })
        
    output_data = {"update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "forecasts": forecast_list}

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 최종 결과가 '{OUTPUT_PATH}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    generate_and_predict()