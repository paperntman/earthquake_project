import pandas as pd
import numpy as np
from flask import Flask, jsonify, send_from_directory, request
import os
import sqlite3
import tensorflow as tf
import joblib
from datetime import datetime

# --- 0. 설정 및 애플리케이션 시작 시 초기화 ---
app = Flask(__name__, 
            static_folder='earthquake-risk-heatmap/dist/assets', 
            static_url_path='/assets')

# 전역 변수로 모델, 스케일러, 원본 데이터를 미리 로드하여 API 호출 시 속도 향상
model = None
scaler = None
raw_df = None

DB_PATH = 'features_database.db'
RAW_DATA_PATH = 'japan_earthquake_data_raw.csv'
MODEL_PATH = 'earthquake_lstm_model.h5'
SCALER_PATH = 'earthquake_scaler.gz'
SEQUENCE_LENGTH = 12

# 앱 컨텍스트 안에서 모델 로드 실행 (Flask 권장 방식)
with app.app_context():
    try:
        print("AI 모델과 스케일러를 로드합니다...")
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        raw_df = pd.read_csv(RAW_DATA_PATH, parse_dates=['time'])
        print("✅ 모델, 스케일러, 원본 데이터 로드 완료.")
    except Exception as e:
        print(f"⚠️ 경고: 시작 시 필요한 파일을 로드하지 못했습니다. ({e})")

# --- 1. 웹페이지 서빙 ---
dist_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                'earthquake-risk-heatmap', 'dist')
@app.route("/")
@app.route("/<path:path>")
def serve_vite_app(path=None):
    # Vite 빌드 결과인 index.html을 제공
    if not os.path.exists(os.path.join(dist_folder_path, 'index.html')):
        return "Frontend not found. Please run 'npm run build' in the frontend directory.", 404
    return send_from_directory(dist_folder_path, 'index.html')

# --- 2. 실시간 예측 API (타임머신 기능) ---
@app.route("/api/predict_at")
def predict_at_date():
    # 1. 사용자 요청에서 날짜 가져오기 (YYYY-MM 형식)
    target_month_str = request.args.get('date')
    if not target_month_str:
        return jsonify({"error": "날짜를 'YYYY-MM' 형식으로 입력해주세요."}), 400
    
    # 모델이나 데이터가 로드되지 않았으면 에러 반환
    if model is None or scaler is None or raw_df is None:
        return jsonify({"error": "서버가 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요."}), 503

    try:
        # 2. 예측에 필요한 기간 계산
        prediction_base_date = pd.to_datetime(target_month_str + '-01') - pd.DateOffset(days=1)
        
        # 이 날짜가 이제 특징을 가져오는 기간의 마지막이 됨
        target_month_end = prediction_base_date 
        
        # LSTM 시퀀스 길이를 고려하여 시작 기간 계산
        start_period = target_month_end - pd.DateOffset(months=SEQUENCE_LENGTH - 1)
        
        # 3. 데이터베이스에서 필요한 특징 데이터 조회 (기간이 한 달 앞당겨짐)
        conn = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True)
        query = f"""
            SELECT * FROM monthly_features 
            WHERE time BETWEEN '{start_period.strftime('%Y-%m-%d')}' AND '{target_month_end.strftime('%Y-%m-%d')}'
        """
        features_df = pd.read_sql_query(query, conn)
        conn.close()
        
        # 4. AI 예측 실행
        features_to_scale = ['count', 'mean', 'count_3m_avg', 'count_6m_avg', 'count_12m_avg', 'b_value', 'days_since_last_quake']
        
        # 🚨 여기가 추가된 부분: 예측에 사용할 특징들의 빈 값을 0으로 채운다.
        features_df[features_to_scale] = features_df[features_to_scale].fillna(0)
        
        predict_data = features_df.groupby('grid_id').filter(lambda x: len(x) == SEQUENCE_LENGTH)
        grid_ids_to_predict = predict_data['grid_id'].unique()
        
        X_predict, latest_features_for_grids = [], {}
        for grid_id in grid_ids_to_predict:
            grid_data = predict_data[predict_data['grid_id'] == grid_id].sort_values('time')
            X_predict.append(scaler.transform(grid_data[features_to_scale]))
            latest_features_for_grids[grid_id] = grid_data.iloc[-1]

        if not X_predict:
            return jsonify({"forecasts": [], "actual_quakes": []}), 200
            
        X_predict = np.array(X_predict)
        predictions = model.predict(X_predict, verbose=0)

        # 5. 실제 발생한 지진 데이터 조회
        actual_quakes_df = raw_df[raw_df['time'].dt.strftime('%Y-%m') == target_month_str]
        actual_quakes = actual_quakes_df[['latitude', 'longitude', 'magnitude', 'depth']].to_dict('records')
        
        # 6. 최종 결과 조합
        GRID_SIZE = 0.5; LAT_MIN = 24; LON_MIN = 122
        forecast_list = []
        for i, grid_id in enumerate(grid_ids_to_predict):
            prob = predictions[i][0] * 100
            details_row = latest_features_for_grids[grid_id]
            lat_idx, lon_idx = map(int, grid_id.split('_'))
            south, west = LAT_MIN + lat_idx * GRID_SIZE, LON_MIN + lon_idx * GRID_SIZE
            forecast_list.append({
                "grid_id": grid_id, "bounds": [[south, west], [south + GRID_SIZE, west + GRID_SIZE]],
                "risk_probability": round(float(prob), 2),
                "details": {"count": int(details_row['count']), "mean": round(details_row['mean'], 2),
                            "count_12m_avg": round(details_row['count_12m_avg'], 2), "b_value": round(details_row['b_value'], 2),
                            "days_since_last_quake": int(details_row['days_since_last_quake'])}
            })
            
        response_data = {
            "analyzed_month": target_month_str, # 'update_time' 대신 '분석된 월'을 명확히 전달
            "forecasts": forecast_list,
            "actual_quakes": actual_quakes
        }
        return jsonify(response_data)

    except Exception as e:
        # 실제 운영에서는 더 상세한 로깅이 필요
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"타임머신 예측 중 오류 발생: {str(e)}"}), 500

import json # 파일 상단에 import 되어있는지 확인, 없다면 추가

# --- 새로운 API: 최신 예보 데이터 전송 ---
LATEST_FORECAST_PATH = 'latest_forecast.json'

@app.route("/api/latest")
def get_latest_forecast():
    """
    미리 계산된 최신 예보 JSON 파일을 읽어 그대로 반환합니다.
    데이터베이스 조회나 AI 예측 없이 매우 빠르게 동작합니다.
    """
    try:
        # with open으로 파일을 열면 자동으로 닫아주어 안전합니다.
        with open(LATEST_FORECAST_PATH, 'r', encoding='utf-8') as f:
            latest_data = json.load(f)
        
        # 프론트엔드가 사용하는 데이터 형식과 맞추기 위해 키 이름을 변경합니다.
        # latest_forecast.json의 'update_time' -> 'analyzed_month'
        # 'update_time'이 'YYYY-MM-DD HH:MM:SS' 형식이므로 'YYYY-MM'으로 잘라줍니다.
        update_time_str = latest_data.get('update_time', '')
        if update_time_str:
            # datetime 객체로 변환했다가 다시 원하는 형식으로 포맷팅합니다.
            analyzed_dt = datetime.strptime(update_time_str, '%Y-%m-%d %H:%M:%S')
            # 해당 월의 예측이므로, 해당 월을 기준으로 표시합니다.
            # 예: 2024-05-15에 생성되었으면, 2024년 6월 예측이므로 2024-06이 분석 대상 월.
            # 하지만 run_daily_forecast.py 로직 상 마지막 달을 기준으로 하므로,
            # 생성된 날짜의 월을 분석 월로 봐도 무방합니다. 여기서는 단순하게 YYYY-MM으로 자릅니다.
            latest_data['analyzed_month'] = analyzed_dt.strftime('%Y-%m')
        else:
            latest_data['analyzed_month'] = "N/A"
            
        # 'update_time' 키는 프론트엔드에서 사용하지 않으므로 제거하거나 그대로 둬도 됩니다.
        # 일관성을 위해 제거하는 것이 더 깔끔할 수 있습니다.
        if 'update_time' in latest_data:
            del latest_data['update_time']
            
        # 프론트엔드에서 actual_quakes 키를 기대하므로, 빈 리스트를 추가해줍니다.
        # 최신 예보는 '미래'를 예측하는 것이므로 '실제 발생 지진'은 아직 없습니다.
        if 'actual_quakes' not in latest_data:
            latest_data['actual_quakes'] = []

        return jsonify(latest_data)
        
    except FileNotFoundError:
        # latest_forecast.json 파일이 아직 생성되지 않은 경우
        return jsonify({"error": "최신 예보 데이터가 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요."}), 404
    except Exception as e:
        # 파일은 있으나 JSON 형식이 잘못되었거나 다른 오류가 발생한 경우
        print(f"Error reading latest forecast file: {e}")
        return jsonify({"error": "최신 예보 데이터를 읽는 중 오류가 발생했습니다."}), 500

# --- 3. 서버 실행 ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=False)