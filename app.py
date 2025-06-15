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
        target_month_end = pd.to_datetime(target_month_str + '-01').to_period('M').end_time
        start_period = target_month_end - pd.DateOffset(months=SEQUENCE_LENGTH - 1)
        
        # 3. 데이터베이스에서 필요한 특징 데이터 조회
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

# --- 3. 서버 실행 ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=False)