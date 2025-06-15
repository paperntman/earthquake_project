# 파일 이름: final_model_save.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# (이전과 동일한 함수들)
def calculate_b_value(magnitudes):
    if len(magnitudes) < 30: return np.nan
    min_mag = 2.5
    return 1 / (np.log(10) * (np.mean(magnitudes) - min_mag))

def create_sequences(data, features, label_col, sequence_length):
    X, y = [], []
    for grid_id, group in tqdm(data.groupby('grid_id'), desc="LSTM 시퀀스 생성 중", leave=False):
        feature_data = group[features].values
        label_data = group[label_col].values
        if len(group) <= sequence_length: continue
        for i in range(len(group) - sequence_length):
            X.append(feature_data[i:(i + sequence_length)])
            y.append(label_data[i + sequence_length])
    return np.array(X), np.array(y)


def run_project_and_save_model():
    # ... (데이터 생성 로직은 이전과 동일)
    print("### 1. 데이터 생성 시작 ###")
    try:
        df = pd.read_csv("japan_earthquake_data_raw.csv", parse_dates=['time'])
    except FileNotFoundError:
        print("앗, 'japan_earthquake_data_raw.csv' 파일이 없습니다.")
        return
    df['grid_id'] = (
        pd.cut(df['latitude'], bins=np.arange(24, 47.1, 0.5), labels=False, include_lowest=True).astype(str) + '_' +
        pd.cut(df['longitude'], bins=np.arange(122, 148.1, 0.5), labels=False, include_lowest=True).astype(str)
    )
    df.dropna(subset=['grid_id'], inplace=True)
    df = df.set_index('time').sort_index()
    monthly_agg = df.groupby('grid_id').resample('ME')['magnitude'].agg(['count', 'mean', 'max']).fillna(0)
    monthly_agg.reset_index(inplace=True)
    monthly_agg['label_mag'] = monthly_agg.groupby('grid_id')['max'].shift(-1)
    windows = [3, 6, 12]
    for w in windows:
        monthly_agg[f'count_{w}m_avg'] = monthly_agg.groupby('grid_id')['count'].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
    feature_rows = []
    for grid_id, group in tqdm(monthly_agg.groupby('grid_id'), desc="과학적 힌트 계산"):
        grid_df_original = df[df['grid_id'] == grid_id]
        for _, row in group.iterrows():
            current_month_end = row['time']
            start_date_1y = current_month_end - pd.DateOffset(years=1)
            mask_1y = (grid_df_original.index >= start_date_1y) & (grid_df_original.index <= current_month_end)
            past_1y_quakes = grid_df_original.loc[mask_1y, 'magnitude']
            b_val = calculate_b_value(past_1y_quakes)
            mask_before = grid_df_original.index <= current_month_end
            quakes_before_now = grid_df_original.loc[mask_before]
            days_since = 9999
            if not quakes_before_now.empty:
                last_quake_date = quakes_before_now.index.max()
                days_since = (current_month_end - last_quake_date).days
            new_row = row.to_dict()
            new_row['b_value'] = b_val
            new_row['days_since_last_quake'] = days_since
            feature_rows.append(new_row)
    final_df = pd.DataFrame(feature_rows)
    final_df = final_df.dropna(subset=['label_mag', 'b_value']).copy()
    final_df['label'] = (final_df['label_mag'] >= 4.0).astype(int)
    print("\n✅ 데이터 생성 완료!")

    # ... (LSTM 데이터 준비 및 모델 훈련/평가 로직은 이전과 동일)
    print("\n### 2. LSTM 모델 준비 및 훈련 ###")
    features = ['count', 'mean', 'count_3m_avg', 'count_6m_avg', 'count_12m_avg', 'b_value', 'days_since_last_quake']
    scaler = StandardScaler()
    final_df[features] = scaler.fit_transform(final_df[features])
    SEQUENCE_LENGTH = 12 
    split_date = '2019-01-01'
    train_df = final_df[final_df['time'] < split_date]
    test_df = final_df[final_df['time'] >= split_date]
    X_train, y_train = create_sequences(train_df, features, 'label', SEQUENCE_LENGTH)
    X_test, y_test = create_sequences(test_df, features, 'label', SEQUENCE_LENGTH)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2), LSTM(50), Dropout(0.2), Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    neg, pos = np.bincount(final_df['label'])
    class_weight = {0: (1 / neg) * ((neg + pos) / 2.0), 1: (1 / pos) * ((neg + pos) / 2.0)}
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=64, class_weight=class_weight, validation_data=(X_test, y_test), verbose=1)
    
    print("\n### 3. 모델 평가 ###")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    print("\n[LSTM 모델 분류 리포트]")
    print(classification_report(y_test, y_pred, target_names=['안전(0)', '위험(1)'], digits=4))

    # ==================================================================
    # 4. 최종 모델 저장 (★★★ 여기가 추가된 부분 ★★★)
    # ==================================================================
    print("\n### 4. 최종 모델 저장 ###")
    model.save("earthquake_lstm_model.h5")
    print("\n✅ 모델이 'earthquake_lstm_model.h5' 파일로 성공적으로 저장되었습니다.")
    print("\n### 스케일러 저장 ###")
    # X_train을 정규화할 때 사용했던 scaler 객체를 저장
    joblib.dump(scaler, 'earthquake_scaler.gz') 
    print("\n✅ 스케일러가 'earthquake_scaler.gz' 파일로 성공적으로 저장되었습니다.")

if __name__ == '__main__':
    run_project_and_save_model()