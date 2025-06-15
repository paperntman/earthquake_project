# 파일 이름: precompute_features.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import sqlite3
from datetime import datetime

def calculate_b_value(magnitudes):
    """b-value를 계산하는 함수"""
    if len(magnitudes) < 30: return np.nan
    min_mag = 2.5
    return 1 / (np.log(10) * (np.mean(magnitudes) - min_mag))

def run_precomputation():
    """
    모든 기간에 대한 모든 특징을 미리 계산하여
    SQLite 데이터베이스에 저장하는 메인 함수.
    """
    print("### 모든 기간에 대한 특징 데이터 사전 계산 시작 ###")
    
    # --- 1. 원본 데이터 로드 ---
    print("(1/5) 원본 데이터를 로드합니다...")
    try:
        df = pd.read_csv("japan_earthquake_data_raw.csv", parse_dates=['time'])
    except FileNotFoundError:
        print("오류: 'japan_earthquake_data_raw.csv' 파일이 없습니다. 먼저 데이터를 다운로드하세요.")
        return
        
    df['grid_id'] = (
        pd.cut(df['latitude'], bins=np.arange(24, 47.1, 0.5), labels=False, include_lowest=True).astype(str) + '_' +
        pd.cut(df['longitude'], bins=np.arange(122, 148.1, 0.5), labels=False, include_lowest=True).astype(str)
    )
    df.dropna(subset=['grid_id'], inplace=True)
    df = df.set_index('time').sort_index()

    # --- 2. 월별 기본 특징 집계 ---
    print("(2/5) 월별 기본 특징을 집계합니다...")
    monthly_agg = df.groupby('grid_id').resample('ME')['magnitude'].agg(['count', 'mean']).fillna(0).reset_index()

    # --- 3. 이동 평균 특징 생성 ---
    print("(3/5) 이동 평균 특징을 생성합니다...")
    windows = [3, 6, 12]
    for w in windows:
        monthly_agg[f'count_{w}m_avg'] = monthly_agg.groupby('grid_id')['count'].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
    
    # --- 4. 과학적 특징 생성 ---
    print("(4/5) 과학적 특징(b-value, 경과일)을 생성합니다... (시간이 많이 소요될 수 있습니다)")
    feature_rows = []
    for grid_id, group in tqdm(monthly_agg.groupby('grid_id'), desc="격자별 계산"):
        grid_df_original = df[df['grid_id'] == grid_id]
        for _, row in group.iterrows():
            current_month_end = row['time']
            
            # b-value 계산
            start_date_1y = current_month_end - pd.DateOffset(years=1)
            past_1y_quakes = grid_df_original.loc[start_date_1y:current_month_end, 'magnitude']
            b_val = calculate_b_value(past_1y_quakes)

            # days_since_last_quake 계산
            quakes_before_now = grid_df_original.loc[:current_month_end]
            days_since = 9999 if quakes_before_now.empty else (current_month_end - quakes_before_now.index.max()).days
            
            new_row = row.to_dict()
            new_row['b_value'] = b_val
            new_row['days_since_last_quake'] = days_since
            feature_rows.append(new_row)

    final_features_df = pd.DataFrame(feature_rows)
    # 🚨 여기가 수정된 부분: b_value가 NaN이어도 버리지 않고, 경과일만 채운다!
    final_features_df['days_since_last_quake'].fillna(9999, inplace=True)
    
    final_features_df['time'] = final_features_df['time'].dt.strftime('%Y-%m-%d')
    
    # --- 5. 데이터베이스에 저장 ---
    DB_PATH = 'features_database.db'
    print(f"(5/5) 계산된 특징 데이터 {len(final_features_df)}개를 데이터베이스 '{DB_PATH}'에 저장합니다...")
    conn = sqlite3.connect(DB_PATH)
    
    final_features_df.to_sql('monthly_features', conn, if_exists='replace', index=False)
    
    print("빠른 조회를 위해 데이터베이스 인덱스를 생성합니다...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_grid_time ON monthly_features (grid_id, time)")
    conn.commit()
    conn.close()
    
    print(f"\n✅ 모든 특징 데이터가 '{DB_PATH}'에 성공적으로 저장되었습니다.")
    print("이제 이 DB 파일을 사용하여 app.py에서 실시간 예측 API를 만들 수 있습니다.")

if __name__ == '__main__':
    run_precomputation()