import pandas as pd
import numpy as np

try:
    df = pd.read_csv("korea_earthquake_data_cleaned.csv")
    print("파일을 성공적으로 읽었어! 이제 데이터 전처리를 다시 시작할게.\n")
except FileNotFoundError:
    print("앗, 'korea_earthquake_data_cleaned.csv' 파일을 찾을 수 없어. 이전 단계의 코드가 잘 실행됐는지 확인해봐!")
    exit()

# --- 1. 시간(time) 컬럼을 datetime 타입으로 바꾸기 ---
df['time'] = pd.to_datetime(df['time'])

# --- 2. 위도(latitude), 경도(longitude)를 숫자(float) 타입으로 바꾸기 (수정!) ---
# 먼저 ' N', ' E' 같은 문자를 제거
# (존재하지 않는 경우에도 에러 없이 동작함)
df['latitude'] = df['latitude'].astype(str).str.replace(' N', '')
df['longitude'] = df['longitude'].astype(str).str.replace(' E', '')

# errors='coerce'를 추가해서 '-' 같은 값을 만나면 NaN으로 바꿔버리자!
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')


# --- 3. 깊이(depth) 컬럼의 문제를 해결하자! (수정!) ---
# 여기에도 errors='coerce'를 적용해서 모든 문자열을 NaN으로 처리
df['depth'] = pd.to_numeric(df['depth'], errors='coerce')


# --- 4. 결측치(NaN) 처리 ---
# 먼저, 위도 또는 경도 정보가 없는(NaN) 행은 위치를 모르니 삭제하자
rows_before_drop = len(df)
df.dropna(subset=['latitude', 'longitude'], inplace=True)
rows_after_drop = len(df)
print(f"위치 정보가 없는 데이터 {rows_before_drop - rows_after_drop}개를 삭제했어.")

# 이제 남은 데이터 중에서, 깊이(depth)가 비어있는 값만 중간값으로 채우자
median_depth = df['depth'].median()
df['depth'].fillna(median_depth, inplace=True)
print(f"깊이(depth)의 비어있는 값들을 중간값({median_depth:.2f} km)으로 채웠어.\n")


# --- 최종 결과 확인 ---
print("--- 전처리 후 데이터 샘플 (첫 5줄) ---")
print(df.head())

print("\n--- 전처리 후 데이터 정보 (결측치 및 타입 확인) ---")
df.info()

df.to_csv("korea_earthquake_data_preprocessed.csv", index=False, encoding='utf-8-sig')
print("\n전처리가 끝난 데이터를 'korea_earthquake_data_preprocessed.csv' 파일로 저장했어.")