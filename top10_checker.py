import json
import pandas as pd

FORECAST_FILE_PATH = 'latest_forecast.json'

def check_top_10_risky_zones():
    """
    최신 예측 결과 파일에서 위험 확률이 가장 높은 상위 10개 지역을 출력합니다.
    """
    print("--- 가장 위험한 지역 TOP 10 ---")
    
    try:
        with open(FORECAST_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"오류: '{FORECAST_FILE_PATH}' 파일을 찾을 수 없습니다.")
        print("먼저 자동화 스크립트(run_daily_forecast.py)를 실행하여 예측 파일을 생성해주세요.")
        return
    except json.JSONDecodeError:
        print(f"오류: '{FORECAST_FILE_PATH}' 파일이 올바른 JSON 형식이 아닙니다.")
        return

    # forecasts 리스트가 비어있는지 확인
    if not data.get('forecasts'):
        print("예측 데이터가 없습니다.")
        return

    # JSON 데이터를 pandas 데이터프레임으로 변환하여 다루기 쉽게 만든다.
    df = pd.DataFrame(data['forecasts'])
    
    # 'risk_probability'를 기준으로 내림차순 정렬
    top_10_df = df.sort_values(by='risk_probability', ascending=False).head(10)
    
    # 보기 좋게 출력
    print(f"데이터 기준 시각: {data.get('update_time', 'N/A')}\n")
    
    # to_string()을 사용하면 인덱스 없이 깔끔하게 출력 가능
    print(top_10_df.to_string(index=False))
    
    # 각 row의 상세 정보도 함께 출력
    print("\n--- TOP 10 지역 상세 정보 ---")
    for index, row in top_10_df.iterrows():
        print(f"\n[순위 {index + 1}] Grid ID: {row['grid_id']} (위험 확률: {row['risk_probability']}%)")
        # details가 딕셔너리 형태일 경우
        if isinstance(row.get('details'), dict):
            for key, value in row['details'].items():
                print(f"  - {key}: {value}")
        else:
             print("  - 상세 정보 없음")


if __name__ == "__main__":
    check_top_10_risky_zones()