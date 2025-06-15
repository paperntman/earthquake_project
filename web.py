from flask import Flask, jsonify, send_from_directory
import os
import json

# Vite 빌드 결과물이 있는 'dist' 폴더를 static_folder로 지정
app = Flask(__name__, 
            static_folder='earthquake-risk-heatmap/dist/assets', 
            static_url_path='/assets')

# Vite 빌드 결과가 있는 폴더의 절대 경로
dist_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                'earthquake-risk-heatmap', 
                                'dist')

# AI 예측 결과 파일 경로
FORECAST_FILE_PATH = 'latest_forecast.json'


# --- 웹페이지 및 API 라우팅 ---

# 1. 루트 주소 ('/') : 사용자가 처음 접속하면 index.html을 보여준다.
@app.route("/")
@app.route("/<path:path>")
def serve_react_app(path=None):
    return send_from_directory(dist_folder_path, 'index.html')


# 2. 우리의 핵심 API 주소 ('/api/forecast') - 여기가 수정된 부분!
@app.route("/api/forecast")
def get_forecast():
    try:
        # 미리 생성된 JSON 파일을 읽어서 내용을 그대로 반환한다.
        with open(FORECAST_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    
    except FileNotFoundError:
        # 만약 파일이 없다면, 에러 메시지를 반환한다.
        return jsonify({"error": "예측 데이터를 찾을 수 없습니다. predictor.py를 먼저 실행하세요."}), 404
    
    except Exception as e:
        # 기타 다른 에러가 발생했을 경우
        return jsonify({"error": f"데이터를 읽는 중 오류 발생: {str(e)}"}), 500


# --- 서버 실행 ---
if __name__ == "__main__":
    # debug=False로 설정하여 실제 서비스와 가깝게 운영
    app.run(host='0.0.0.0', port=80, debug=False)