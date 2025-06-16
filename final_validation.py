import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # Input을 명시적으로 import 해도 좋습니다.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import requests
import time
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

# --- 0. 전역 설정 ---
TRAIN_START_YEAR = 1960
TRAIN_END_YEAR = 2000
VALIDATION_START_YEAR = 2001
SEQUENCE_LENGTH = 6
GRID_SIZE = 0.5
LAT_MIN, LON_MIN = 24, 122

# 캐시 파일 이름 정의
RAW_DATA_CACHE = "cached_raw_earthquake_data_extended.pkl" # 데이터 기간이 바뀌었으므로 새 캐시 파일 이름 권장
FEATURES_CACHE = "cached_features_data_extended.pkl"

# --- 1. 헬퍼 함수 ---
def log_message(message): print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
def calculate_b_value(magnitudes):
    if len(magnitudes) < 30: return np.nan
    return 1 / (np.log(10) * (np.mean(magnitudes[magnitudes >= 2.5]) - 2.5))

def download_data(start_year, end_year):
    """지정된 기간의 데이터를 1년 단위로 다운로드하는 함수"""
    all_yearly_data = []
    for year in tqdm(range(start_year, end_year + 1), desc=f"{start_year}-{end_year} 데이터 다운로드"):
        start, end = f"{year}-01-01", f"{year}-12-31"
        url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&orderby=time&starttime={start}&endtime={end}&minlatitude=24&maxlatitude=46&minlongitude=122&maxlongitude=148&minmagnitude=2.5"
        try:
            response = requests.get(url, timeout=180); response.raise_for_status(); data = response.json()
            if data['features']:
                year_quakes = [{'time': pd.to_datetime(f['properties']['time'], unit='ms'), 'magnitude': f['properties']['mag'], 'depth': f['geometry']['coordinates'][2], 'longitude': f['geometry']['coordinates'][0], 'latitude': f['geometry']['coordinates'][1], 'location': f['properties']['place']} for f in data['features']]
                all_yearly_data.append(pd.DataFrame(year_quakes))
            time.sleep(1)
        except Exception as e: log_message(f"{year}년 다운로드 실패: {e}")
    if not all_yearly_data: return pd.DataFrame()
    return pd.concat(all_yearly_data, ignore_index=True).sort_values(by='time')
# ### 1. 모델 생성 함수 정의 ###
# 하이퍼파라미터를 인자로 받아 Keras 모델을 생성하는 함수
def create_model(lstm_units=50, dropout_rate=0.2, input_shape=None):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=lstm_units, return_sequences=True),
        Dropout(rate=dropout_rate),
        LSTM(units=lstm_units),
        Dropout(rate=dropout_rate),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 2. 메인 실행 함수 ---
def run_final_validation():
    log_message("===== 최종 검증 파이프라인 시작 =====")
    
    raw_data_was_updated = False

    # --- 1/5: 데이터 로드 또는 다운로드 (캐싱 로직) ---
    log_message("### 1/5: 데이터 로드 또는 다운로드 ###")
    if os.path.exists(RAW_DATA_CACHE):
        log_message(f"'{RAW_DATA_CACHE}' 발견. 데이터 로딩 중...")
        full_raw_df = pd.read_pickle(RAW_DATA_CACHE)
        last_data_year = full_raw_df['time'].dt.year.max()
        current_year = datetime.now().year
        
        if last_data_year < current_year:
            log_message(f"캐시 데이터가 최신이 아닙니다 (최종: {last_data_year}년). {last_data_year + 1}년부터 현재까지 데이터 추가 다운로드...")
            new_raw_df = download_data(last_data_year + 1, current_year)
            if not new_raw_df.empty:
                full_raw_df = pd.concat([full_raw_df, new_raw_df], ignore_index=True).drop_duplicates().sort_values(by='time', ignore_index=True)
                full_raw_df.to_pickle(RAW_DATA_CACHE)
                log_message("✅ 캐시된 원본 데이터를 업데이트하고 저장했습니다.")
                raw_data_was_updated = True
            else:
                log_message("추가할 새로운 데이터가 없습니다.")
        else:
            log_message("✅ 캐시된 데이터가 최신입니다. 다운로드를 건너뜁니다.")
    else:
        log_message(f"캐시 파일 '{RAW_DATA_CACHE}' 없음. 전체 데이터 다운로드 시작...")
        full_raw_df = download_data(TRAIN_START_YEAR, datetime.now().year)
        full_raw_df.to_pickle(RAW_DATA_CACHE)
        log_message(f"✅ 전체 원본 데이터를 다운로드하여 '{RAW_DATA_CACHE}'에 저장했습니다.")
        raw_data_was_updated = True

    # --- 2/5: 특징 생성 또는 로드 (캐싱 로직) ---
    log_message("### 2/5: 특징 생성 또는 로드 ###")
    if not raw_data_was_updated and os.path.exists(FEATURES_CACHE):
        log_message(f"'{FEATURES_CACHE}' 발견. 특징 데이터 로딩 중...")
        features_df = pd.read_pickle(FEATURES_CACHE)
        log_message("✅ 캐시된 특징 데이터를 로드했습니다. 특징 생성을 건너뜁니다.")
    else:
        if raw_data_was_updated:
            log_message("원본 데이터가 업데이트되어 특징을 다시 생성합니다.")
        else:
            log_message(f"캐시 파일 '{FEATURES_CACHE}' 없음. 특징 생성 시작...")
            
        full_raw_df['grid_id'] = (pd.cut(full_raw_df['latitude'], bins=np.arange(24, 47.1, 0.5), labels=False, include_lowest=True).astype(str) + '_' + pd.cut(full_raw_df['longitude'], bins=np.arange(122, 148.1, 0.5), labels=False, include_lowest=True).astype(str))
        full_raw_df.dropna(subset=['grid_id'], inplace=True)
        full_raw_df = full_raw_df.set_index('time').sort_index()
        monthly_agg = full_raw_df.groupby('grid_id').resample('ME')['magnitude'].agg(['count', 'mean', 'max']).fillna(0).reset_index()
        for w in [3, 6, 12]: monthly_agg[f'count_{w}m_avg'] = monthly_agg.groupby('grid_id')['count'].transform(lambda x: x.rolling(w, min_periods=1).mean())
        feature_rows = []
        for grid_id, group in tqdm(monthly_agg.groupby('grid_id'), desc="특징 계산"):
            grid_df_original = full_raw_df[full_raw_df['grid_id'] == grid_id]
            for _, row in group.iterrows():
                end_date = row['time']; start_date_1y = end_date - pd.DateOffset(years=1)
                b_val = calculate_b_value(grid_df_original.loc[start_date_1y:end_date, 'magnitude'])
                quakes_before_now = grid_df_original.loc[:end_date]
                days_since = 9999 if quakes_before_now.empty else (end_date - quakes_before_now.index.max()).days
                new_row = row.to_dict(); new_row['b_value'] = b_val; new_row['days_since_last_quake'] = days_since
                feature_rows.append(new_row)
        features_df = pd.DataFrame(feature_rows).dropna(subset=['b_value']).copy()
        features_df['days_since_last_quake'].fillna(9999, inplace=True)
        features_df['label_mag'] = features_df.groupby('grid_id')['max'].shift(-1)
        features_df = features_df.dropna(subset=['label_mag']).copy()
        features_df['label'] = (features_df['label_mag'] >= 4.0).astype(int)
        
        features_df.to_pickle(FEATURES_CACHE)
        log_message(f"✅ 생성된 특징 데이터를 '{FEATURES_CACHE}'에 저장했습니다.")
    
    # --- 로직 오류 수정: 이 아래 블록 전체를 if/else 문 밖으로 이동시켰습니다. ---
    
    features_df = pd.read_pickle(FEATURES_CACHE) # 간단하게 로드하는 것으로 가정
    
    # --- 3/5: AI 모델 재학습 및 하이퍼파라미터 튜닝 ---
    log_message("### 3/5: AI 모델 재학습 및 하이퍼파라미터 튜닝 시작 ###")
    train_features_df = features_df[features_df['time'].dt.year <= TRAIN_END_YEAR].copy()
    features_to_scale = ['count', 'mean', 'count_3m_avg', 'count_6m_avg', 'count_12m_avg', 'b_value', 'days_since_last_quake']
    scaler = StandardScaler()
    train_features_df[features_to_scale] = scaler.fit_transform(train_features_df[features_to_scale])

    def create_sequences(data, features, label_col, seq_len):
        X, y = [], []
        for _, group in tqdm(data.groupby('grid_id'), desc="시퀀스 생성"):
            feature_data, label_data = group[features].values, group[label_col].values
            if len(group) <= seq_len: continue
            for i in range(len(group) - seq_len): X.append(feature_data[i:(i + seq_len)]); y.append(label_data[i + seq_len])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_features_df, features_to_scale, 'label', SEQUENCE_LENGTH)

    if X_train.shape[0] < 10:
        log_message(f"훈련 데이터 샘플이 너무 적습니다({X_train.shape[0]}개). 파이프라인을 중단합니다.")
        return
        
    # ### 2. 하이퍼파라미터 탐색 설정 ###
    log_message("하이퍼파라미터 랜덤 탐색을 설정합니다.")
    
    # KerasClassifier로 모델 래핑. input_shape를 여기서 고정.
    model_for_tuning = KerasClassifier(
        model=create_model,
        input_shape=(X_train.shape[1], X_train.shape[2]),
        verbose=0
    )
    
    # 탐색할 하이퍼파라미터 범위 정의
    param_dist = {
        'model__lstm_units': [10, 20, 30, 40, 50],
        'model__dropout_rate': [0.3, 0.4, 0.5],
        'batch_size': [32, 64, 128, 256, 512],
    }

    # RandomizedSearchCV 설정
    # n_iter: 시도할 조합의 수 (10번의 랜덤 조합 시도)
    # cv: 교차 검증 횟수 (3-fold cross-validation)
    random_search = RandomizedSearchCV(
        estimator=model_for_tuning,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        verbose=1,
        n_jobs=-1, # 모든 CPU 코어 사용
        random_state=42 # 결과 재현을 위한 시드
    )
    
    # 모델 학습 시 전달할 파라미터들 (epochs, class_weight 등)
    neg, pos = np.bincount(y_train)
    fit_params = {
        'epochs': 10,
        'class_weight': {0: (1/neg)*((neg+pos)/2.0), 1: (1/pos)*((neg+pos)/2.0)}
    }

    # ### 3. 탐색 실행 ###
    log_message("하이퍼파라미터 탐색을 시작합니다... (시간이 오래 걸릴 수 있습니다)")
    random_search.fit(X_train, y_train, **fit_params)

    # 최적의 결과 출력
    log_message(f"최적의 하이퍼파라미터: {random_search.best_params_}")
    log_message(f"최고 교차검증 점수: {random_search.best_score_:.4f}")

    # 최적의 모델을 최종 모델로 사용
    best_model = random_search.best_estimator_
    log_message("✅ 모델 튜닝 및 재학습 완료.")

    # 스케일러와 함께 모델 저장 (최적 모델 저장)
    best_model.model_.save("final_validation_model_tuned.h5")
    joblib.dump(scaler, "final_validation_scaler.gz")
    log_message("✅ 튜닝된 최종 검증 모델과 스케일러를 파일로 저장했습니다.")
    
    # --- 4/5: 미래 예측 (최적 모델 사용) ---
    log_message("### 4/5: 미래 예측 시뮬레이션 시작 ###")
    validation_features_df = features_df[features_df['time'].dt.year >= VALIDATION_START_YEAR].copy()
    validation_features_df[features_to_scale] = scaler.transform(validation_features_df[features_to_scale])
    X_val, y_val_actual = create_sequences(validation_features_df, features_to_scale, 'label', SEQUENCE_LENGTH)

    if X_val.shape[0] > 0:
        # predict는 best_estimator_를 사용
        y_val_pred = best_model.predict(X_val)
        log_message("✅ 미래 예측 완료.")
    else:
        log_message("경고: 검증할 시퀀스 데이터가 없습니다."); return

    # --- 5/5: 실제 미래와 비교 및 최종 평가/시각화 ---
    log_message("### 5/5: 실제 미래 데이터와 비교 평가 ###")
    # ... (이하 평가 및 시각화 코드는 기존과 동일) ...
    log_message("\n--- 최종 검증 결과 ---")
    print(classification_report(y_val_actual, y_val_pred, target_names=['안전(0)', '위험(1)'], digits=4))

    cm = confusion_matrix(y_val_actual, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted: Safe', 'Predicted: Danger'],
                yticklabels=['Actual: Safe', 'Actual: Danger'])
    plt.title(f'Final Validation - Confusion Matrix ({VALIDATION_START_YEAR}~ Future) - Tuned')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('final_validation_confusion_matrix_tuned.png')
    plt.close()
    log_message("✅ 최종 검증 혼동 행렬을 'final_validation_confusion_matrix_tuned.png'로 저장했습니다.")

    log_message("===== 최종 검증 파이프라인 종료 =====")

if __name__ == '__main__':
    run_final_validation()