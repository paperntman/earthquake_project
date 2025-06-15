import pandas as pd
import time
from datetime import datetime

# --- 설정 ---
START_YEAR = 1970
END_DATE = datetime.now().strftime('%Y-%m-%d')


# --- 코드 시작 ---
print("기상청 지진 데이터 다운로드 및 정제를 시작할게! (최종 수정 버전)")

all_dataframes = []

# 우리가 최종적으로 사용할 컬럼 이름 (한글 원본)
# 사용자가 제공한 헤더를 기반으로 정확하게 작성
# "최대\n진도" 처럼 줄바꿈이 포함된 경우, 파이썬에서는 \n으로 표현
original_cols = ['발생시각', '규모', '깊이(km)', '위도', '경도', '위치']

# 새로 바꿀 컬럼 이름 (영어)
new_cols = ['time', 'magnitude', 'depth', 'latitude', 'longitude', 'location']

for year in range(START_YEAR, datetime.now().year + 1, 10):
    start_date = f"{year}-01-01"
    end_date_year = year + 9
    
    if end_date_year > datetime.now().year:
        end_date = END_DATE
    else:
        end_date = f"{end_date_year}-12-31"

    url = f"https://www.weather.go.kr/w/eqk-vol/search/korea.do?xls=1&dpType=a&startTm={start_date}&endTm={end_date}"
    
    print(f"다운로드 중: {start_date} 부터 {end_date} 까지...")

    try:
        # cp949 인코딩으로 HTML 테이블 읽기
        df_list = pd.read_html(url, encoding='cp949', header=0)
        
        if df_list:
            df = df_list[0]
            
            # --- 여기가 핵심! 필요한 컬럼만 선택하고 이름 바꾸기 ---
            # 1. 원본 데이터에서 필요한 컬럼만 선택
            df_subset = df[original_cols]
            
            # 2. 선택한 데이터프레임의 컬럼 이름을 영어로 변경
            df_subset.columns = new_cols
            
            all_dataframes.append(df_subset)
            print(" -> 성공! 필요한 데이터만 추출 완료.")
        else:
            print(" -> 해당 기간에 데이터가 없나봐.")
            
    except KeyError:
        # 1978~2000년 이전 데이터는 컬럼 이름이 살짝 다를 수 있어.
        # 그 경우를 대비해서 예외 처리를 해두면 좋아. 지금은 일단 넘어가자!
        print(f" -> 경고: {year}년대 데이터의 컬럼 이름이 달라서 건너뛸게. 이따가 따로 처리할 수도 있어.")
    except Exception as e:
        print(f" -> 에러 발생: {e}")

    time.sleep(1)


if all_dataframes:
    print("\n다운로드한 모든 데이터를 하나로 합치는 중...")
    # as_index=False를 추가해서 복사 경고를 방지
    final_df = pd.concat([df.copy() for df in all_dataframes], ignore_index=True)
    
    output_filename = "korea_earthquake_data_cleaned.csv"
    final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 좋아! 정제된 데이터가 '{output_filename}' 파일로 저장됐어.")
    print("\n--- 데이터 샘플 (첫 5줄) ---")
    print(final_df.head())
    
    print("\n--- 데이터 정보 (결측치 및 타입 확인) ---")
    final_df.info()

else:
    print("\n❌ 아쉽지만, 다운로드한 데이터가 하나도 없어. 인터넷 연결이나 기상청 웹사이트를 확인해봐야 할 것 같아.")