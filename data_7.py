# 파일 이름: the_final_one.py
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# tqdm을 pandas와 함께 사용하기 위해 등록
tqdm.pandas()

def calculate_b_value(magnitudes):
    """b-value를 계산하는 함수"""
    if len(magnitudes) < 30: return np.nan
    min_mag = 2.5
    return 1 / (np.log(10) * (np.mean(magnitudes) - min_mag))

def run_the_whole_thing():
    # ==================================================================
    # 1. 데이터 생성 (가장 안정적이고 확실한 방법)
    # ==================================================================
    print("### 1. 데이터 생성 시작 (최종 안정화 버전) ###")
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
    # 모든 계산의 기준이 되는 인덱스를 시간순으로 완벽히 정렬
    df = df.set_index('time').sort_index()

    print("월별 데이터 집계 중...")
    monthly_agg = df.groupby('grid_id').resample('ME')['magnitude'].agg(['count', 'mean', 'max']).fillna(0)
    monthly_agg.reset_index(inplace=True)

    print("라벨(정답) 및 이동 평균 힌트 생성 중...")
    monthly_agg['label_mag'] = monthly_agg.groupby('grid_id')['max'].shift(-1)
    windows = [3, 6, 12]
    for w in windows:
        monthly_agg[f'count_{w}m_avg'] = monthly_agg.groupby('grid_id')['count'].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
        
    print("과학적 힌트 계산 중... 이 과정은 몇 분 정도 걸릴 수 있습니다.")
    # 계산 결과를 저장할 리스트 생성
    feature_rows = []
    # 그룹별로 루프를 돌며 가장 확실한 방법으로 계산
    for grid_id, group in tqdm(monthly_agg.groupby('grid_id'), desc="격자별 계산"):
        grid_df_original = df[df['grid_id'] == grid_id]
        for _, row in group.iterrows():
            current_month_end = row['time']
            
            # b-value (안전한 방식)
            start_date_1y = current_month_end - pd.DateOffset(years=1)
            mask_1y = (grid_df_original.index >= start_date_1y) & (grid_df_original.index <= current_month_end)
            past_1y_quakes = grid_df_original.loc[mask_1y, 'magnitude']
            b_val = calculate_b_value(past_1y_quakes)

            # days_since_last_quake (안전한 방식)
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
    print("최종 데이터 정리 중...")
    final_df = final_df.dropna(subset=['label_mag', 'b_value']).copy()
    final_df['label'] = (final_df['label_mag'] >= 4.0).astype(int)

    print("\n✅ 데이터 생성 완료!")
    print("\n--- 최종 데이터 정답(Label) 분포 확인 ---")
    print(final_df['label'].value_counts())

    # ==================================================================
    # 2. 최종 모델 훈련 및 평가
    # ==================================================================
    print("\n### 2. 최종 AI 모델 훈련 및 평가 시작 ###")
    features = ['count', 'mean', 'count_3m_avg', 'count_6m_avg', 'count_12m_avg', 'b_value', 'days_since_last_quake']
    print(f"\n사용할 힌트: {features}")

    split_date = '2019-01-01'
    train_df = final_df[final_df['time'] < split_date]
    test_df = final_df[final_df['time'] >= split_date]

    X_train, y_train = train_df[features], train_df['label']
    X_test, y_test = test_df[features], test_df['label']
    
    print(f"\n학습 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")
    print("\n테스트 데이터의 정답 분포:")
    print(y_test.value_counts())

    if 1 not in y_test.value_counts() or len(y_test.value_counts()) < 2:
        print("\n❌ 에러: 테스트 데이터에 '위험(1)' 샘플이 없거나 클래스가 하나뿐입니다. 모델 훈련을 중단합니다.")
        return

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n[최종 진화 모델 분류 리포트]")
    print(classification_report(y_test, y_pred, target_names=['안전(0)', '위험(1)'], digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted: Safe', 'Predicted: Danger'], yticklabels=['Actual: Safe', 'Actual: Danger'])
    plt.title('Final Evolved Model - Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

if __name__ == '__main__':
    run_the_whole_thing()