import pandas as pd
import numpy as np
from tqdm import tqdm # <--- 1. tqdm 라이브러리를 불러오자!

# --- 설정 (이전과 동일) ---
GRID_SIZE = 0.2
TIME_WINDOW = 'M'
LABEL_MAGNITUDE_THRESHOLD = 3.0
FEATURE_MONTHS = 1
LABEL_MONTHS = 1

# --- 함수 정의 (이전과 동일) ---
def magnitude_to_energy(m):
    return 10**(1.5 * m + 4.8)

# --- 코드 시작 ---
print("전처리된 데이터를 불러옵니다...")
try:
    df = pd.read_csv("korea_earthquake_data_preprocessed.csv", parse_dates=['time'])
except FileNotFoundError:
    print("앗, 'korea_earthquake_data_preprocessed.csv' 파일을 찾을 수 없어. 이전 단계 코드를 확인해봐!")
    exit()

# 1단계: 땅 나누기 (이전과 동일)
print(f"{GRID_SIZE}도 간격으로 격자를 생성합니다...")
lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
lat_bins = np.arange(lat_min, lat_max + GRID_SIZE, GRID_SIZE)
lon_bins = np.arange(lon_min, lon_max + GRID_SIZE, GRID_SIZE)
df['lat_grid'] = pd.cut(df['latitude'], bins=lat_bins, right=False, labels=False)
df['lon_grid'] = pd.cut(df['longitude'], bins=lon_bins, right=False, labels=False)
df.dropna(subset=['lat_grid', 'lon_grid'], inplace=True)
df['grid_id'] = df.apply(lambda row: f"{int(row.lat_grid):02d}-{int(row.lon_grid):02d}", axis=1)
df['energy'] = magnitude_to_energy(df['magnitude'])

# 2단계 & 3단계: 월별 힌트와 정답 만들기 (tqdm 적용!)
print("월별로 각 격자의 힌트(Feature)와 정답(Label)을 계산합니다...")
df.set_index('time', inplace=True)

features = []
grid_ids = df['grid_id'].unique()
time_periods = df.resample(TIME_WINDOW).mean(numeric_only=True).index

# --- 2. 여기가 수정된 부분이야! ---
# 그냥 for문 대신 tqdm()으로 감싸주기만 하면 돼.
for grid_id in tqdm(grid_ids, desc="격자별 데이터 처리 중"):
    for t_start in time_periods:
        # 이 아래 부분은 이전 코드와 완전히 동일해.
        t_end = t_start + pd.DateOffset(months=FEATURE_MONTHS)
        current_data = df[(df['grid_id'] == grid_id) & (df.index >= t_start) & (df.index < t_end)]
        
        quake_count = len(current_data)
        if quake_count > 0:
            mean_magnitude = current_data['magnitude'].mean()
            total_energy = current_data['energy'].sum()
        else:
            mean_magnitude = 0
            total_energy = 0

        label_start = t_end
        label_end = label_start + pd.DateOffset(months=LABEL_MONTHS)
        future_data = df[(df['grid_id'] == grid_id) & (df.index >= label_start) & (df.index < label_end)]
        
        if any(future_data['magnitude'] >= LABEL_MAGNITUDE_THRESHOLD):
            label = 1
        else:
            label = 0
            
        features.append({
            'time': t_start,
            'grid_id': grid_id,
            'quake_count': quake_count,
            'mean_magnitude': mean_magnitude,
            'total_energy_log': np.log10(total_energy + 1),
            'label': label
        })

final_df = pd.DataFrame(features)
output_filename = "grid_features_and_labels.csv"
final_df.to_csv(output_filename, index=False)

print(f"\n✅ 완료! AI 학습용 데이터가 '{output_filename}' 파일로 저장됐어.")
print("\n--- 생성된 데이터 샘플 (첫 10줄) ---")
print(final_df.head(10))

print("\n--- 정답(Label) 분포 확인 ---")
print(final_df['label'].value_counts())