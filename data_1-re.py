import pandas as pd
import requests
import time
from datetime import datetime
from tqdm import tqdm # 진행도 바를 위한 tqdm 임포트

# --- 설정 ---
# 시작 년도와 종료 년도 설정
START_YEAR = 2000
END_YEAR = datetime.now().year

# 일본 열도 주변의 위도/경도 범위 설정
LAT_MIN, LAT_MAX = 24, 46
LON_MIN, LON_MAX = 122, 148

# 다운로드할 최소 지진 규모
MIN_MAGNITUDE = 2.5

# --- 코드 시작 ---
print("USGS 서버에서 일본 지진 데이터를 1년 단위로 분할 다운로드합니다...")

# 각 년도별로 다운로드한 데이터프레임을 저장할 리스트
all_dataframes = []

# tqdm을 사용해 시작 년도부터 종료 년도까지 1년씩 반복
for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="연도별 다운로드 중"):
    
    # 해당 년도의 시작일과 종료일 설정
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # API URL 생성
    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query?"
        "format=geojson"
        f"&starttime={start_date}"
        f"&endtime={end_date}"
        f"&minlatitude={LAT_MIN}"
        f"&maxlatitude={LAT_MAX}"
        f"&minlongitude={LON_MIN}"
        f"&maxlongitude={LON_MAX}"
        f"&minmagnitude={MIN_MAGNITUDE}"
        "&orderby=time"
    )

    try:
        # API에 데이터 요청
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        data = response.json()
        
        # 'features'에 데이터가 있는지 확인
        if data['features']:
            earthquake_list = []
            for feature in data['features']:
                properties = feature['properties']
                geometry = feature['geometry']['coordinates']
                
                earthquake_list.append({
                    'time': properties['time'],
                    'magnitude': properties['mag'],
                    'depth': geometry[2],
                    'longitude': geometry[0],
                    'latitude': geometry[1],
                    'location': properties['place']
                })
            
            # 해당 년도의 데이터프레임을 리스트에 추가
            df_year = pd.DataFrame(earthquake_list)
            all_dataframes.append(df_year)
            # print(f" -> {year}년 데이터 {len(df_year)}개 다운로드 성공!") # tqdm 사용 시에는 생략 가능
    
    except requests.exceptions.RequestException as e:
        print(f"\n{year}년 데이터 다운로드 중 에러 발생: {e}. 해당 년도는 건너뜁니다.")
    
    # USGS 서버에 부담을 주지 않도록 1초 대기
    time.sleep(1)

# 모든 데이터프레임이 리스트에 잘 담겼는지 확인
if all_dataframes:
    # 리스트에 있는 모든 데이터프레임을 하나로 합치기
    print("\n다운로드한 모든 데이터를 하나로 합치는 중...")
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    # 시간 컬럼 변환
    final_df['time'] = pd.to_datetime(final_df['time'], unit='ms')

    # 파일로 저장
    output_filename = 'japan_earthquake_data_raw.csv'
    final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 완료! 총 {len(final_df)}개의 지진 데이터가 '{output_filename}' 파일로 저장됐어.")
    print("\n--- 데이터 샘플 (첫 5줄) ---")
    print(final_df.head())
    
    print("\n--- 데이터 정보 ---")
    final_df.info()

else:
    print("\n❌ 아쉽지만, 다운로드한 데이터가 하나도 없어. 인터넷 연결이나 설정을 다시 확인해봐.")