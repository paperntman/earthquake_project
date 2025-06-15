# 파일 이름: final_challenge.py
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# tqdm을 pandas와 함께 사용하기 위해 등록
tqdm.pandas()

def run_final_analysis():
    # ==================================================================
    # 1. 최종 데이터 생성 (가장 안정적인 방법)
    # ==================================================================
    print("### 1. 최종 데이터 생성 시작 (안정화 버전) ###")
    try:
        df = pd.read_csv("japan_earthquake_data_raw.csv", parse_dates=['time'])
    except FileNotFoundError:
        print("앗, 'japan_earthquake_data_raw.csv' 파일이 없습니다. 다운로드 코드를 먼저 실행해주세요.")
        return

    df['grid_id'] = (
        pd.cut(df['latitude'], bins=np.arange(24, 47.1, 0.5), labels=False, include_lowest=True).astype(str) + '_' +
        pd.cut(df['longitude'], bins=np.arange(122, 148.1, 0.5), labels=False, include_lowest=True).astype(str)
    )
    df.dropna(subset=['grid_id'], inplace=True)
    
    print("월별 데이터 집계 중...")
    # 1. 월별로 기본 데이터 집계
    monthly_agg = df.groupby('grid_id').resample('ME', on='time')['magnitude'].agg(['count', 'mean', 'max']).fillna(0)
    monthly_agg.reset_index(inplace=True)

    print("라벨(정답) 생성 중...")
    # 2. 그룹별로 shift를 적용해 안전하게 라벨 생성
    # 이렇게 하면 각 grid_id 안에서만 shift가 적용돼.
    monthly_agg['label_mag'] = monthly_agg.groupby('grid_id')['max'].shift(-1)
    
    print("이동 평균 힌트(Feature) 생성 중...")
    # 3. 그룹별로 rolling을 적용해 안전하게 힌트 생성
    windows = [3, 6, 12]
    # groupby().progress_apply()를 사용해 tqdm 진행도 표시
    for w in windows:
        monthly_agg[f'count_{w}m_avg'] = monthly_agg.groupby('grid_id')['count'].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
        
    # 4. 최종 데이터 정리
    final_df = monthly_agg.dropna(subset=['label_mag']).copy()
    final_df['label'] = (final_df['label_mag'] >= 4.0).astype(int)
    
    print("\n✅ 데이터 생성 완료!")
    print("\n--- 최종 데이터 정답(Label) 분포 확인 ---")
    print(final_df['label'].value_counts())

    # ==================================================================
    # 2. 최종 모델 훈련 및 평가
    # ==================================================================
    print("\n### 2. 최종 AI 모델 훈련 및 평가 시작 ###")
    
    features = ['count', 'mean', 'count_3m_avg', 'count_6m_avg', 'count_12m_avg']
    print(f"\n사용할 힌트: {features}")

    split_date = '2019-01-01'
    train_df = final_df[final_df['time'] < split_date]
    test_df = final_df[final_df['time'] >= split_date]

    X_train, y_train = train_df[features], train_df['label']
    X_test, y_test = test_df[features], test_df['label']

    print(f"\n학습 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")
    print("\n테스트 데이터의 정답 분포:")
    print(y_test.value_counts())
    
    # 테스트 데이터에 '위험' 샘플이 있는지 최종 확인
    if 1 not in y_test.value_counts():
        print("\n❌ 에러: 테스트 데이터에 '위험(1)' 샘플이 없습니다. 모델 훈련을 중단합니다.")
        return

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n[최종 분류 리포트]")
    print(classification_report(y_test, y_pred, target_names=['안전(0)', '위험(1)'], digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted: Safe', 'Predicted: Danger'], yticklabels=['Actual: Safe', 'Actual: Danger'])
    plt.title('Final Model - Confusion Matrix')
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
    run_final_analysis()