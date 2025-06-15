import pandas as pd
import numpy as np
from tqdm import tqdm

# --- 설정: 일본 데이터에 맞게 수정! ---
GRID_SIZE = 0.5  # 일본은 영역이 넓으니 격자를 좀 더 크게 (0.5도 x 0.5도)
TIME_WINDOW = 'M' # 시간 단위는 월(Month)로 동일하게
LABEL_MAGNITUDE_THRESHOLD = 4.0 # '위험' 기준 규모를 4.0으로 상향!
FEATURE_MONTHS = 1
LABEL_MONTHS = 1

# --- 함수 정의 (동일) ---
def magnitude_to_energy(m):
    return 10**(1.5 * m + 4.8)

# --- 코드 시작 ---
print("다운로드한 일본 지진 데이터를 불러옵니다...")
try:
    df = pd.read_csv("japan_earthquake_data_raw.csv", parse_dates=['time'])
except FileNotFoundError:import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing # 멀티프로세싱 라이브러리 임포트
from functools import partial # 함수에 인자를 고정시킬 때 사용

# --- 설정 (이전과 동일) ---
GRID_SIZE = 0.5
TIME_WINDOW = 'M'
LABEL_MAGNITUDE_THRESHOLD = 4.0
FEATURE_MONTHS = 1
LABEL_MONTHS = 1

# --- 함수 정의 ---
def magnitude_to_energy(m):
    return 10**(1.5 * m + 4.8)

# --- 멀티프로세싱으로 실행될 '일꾼 함수' 정의 ---
# 이 함수는 grid_id 하나를 받아서 해당 격자의 모든 시간 데이터를 처리해
def process_grid(grid_id, df, time_periods):
    
    local_features = []
    # 전달받은 grid_id에 대해서만 루프를 돈다
    for t_start in time_periods:
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
        
        label = 1 if any(future_data['magnitude'] >= LABEL_MAGNITUDE_THRESHOLD) else 0
            
        local_features.append({
            'time': t_start,
            'grid_id': grid_id,
            'quake_count': quake_count,
            'mean_magnitude': mean_magnitude,
            'total_energy_log': np.log10(total_energy + 1),
            'label': label
        })
    return local_features

# --- 메인 코드 실행 부분 ---
# 멀티프로세싱 코드는 반드시 이 if 문 안에서 실행해야 해 (중요!)
if __name__ == '__main__':
    print("다운로드한 일본 지진 데이터를 불러옵니다...")
    try:
        df = pd.read_csv("japan_earthquake_data_raw.csv", parse_dates=['time'])
    except FileNotFoundError:
        print("앗, 'japan_earthquake_data_raw.csv' 파일을 찾을 수 없어.")
        exit()

    # 데이터 준비 (이전과 동일)
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
    df.set_index('time', inplace=True)

    grid_ids = df['grid_id'].unique()
    time_periods = df.resample(TIME_WINDOW).mean(numeric_only=True).index

    # --- 멀티프로세싱 시작 ---
    print("월별 힌트/정답 계산을 멀티프로세싱으로 시작합니다...")
    
    # 사용할 CPU 코어 수 결정 (전부 사용)
    num_cores = multiprocessing.cpu_count()
    print(f"사용할 CPU 코어 수: {num_cores}")
    
    # Pool을 생성해서 일꾼들을 준비시킴
    pool = multiprocessing.Pool(num_cores)
    
    # 각 일꾼에게 df와 time_periods 정보를 미리 고정시켜 전달 준비
    task_func = partial(process_grid, df=df, time_periods=time_periods)

    # imap을 사용해 작업을 분배하고 tqdm으로 진행 상황 표시
    # 여러 일꾼이 grid_ids 리스트에서 grid_id를 하나씩 가져가 task_func를 실행
    results = []
    for result in tqdm(pool.imap_unordered(task_func, grid_ids), total=len(grid_ids), desc="전체 격자 처리 중"):
        results.extend(result) # 각 일꾼의 결과물을 최종 리스트에 추가

    # 작업이 끝나면 Pool을 닫아준다
    pool.close()
    pool.join()

    # --- 결과 정리 ---
    final_df = pd.DataFrame(results)
    output_filename = "grid_features_and_labels_japan.csv"
    final_df.to_csv(output_filename, index=False)
    
    print(f"\n✅ 완료! AI 학습용 데이터가 '{output_filename}' 파일로 저장됐어.")
    print("\n--- 생성된 데이터 샘플 (첫 10줄) ---")
    print(final_df.head(10))

    print("\n--- 정답(Label) 분포 확인 ---")
    print(final_df['label'].value_counts())