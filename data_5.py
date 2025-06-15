import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- 코드 시작 ---
print("AI 학습용 데이터를 불러옵니다...")
try:
    df = pd.read_csv("grid_features_and_labels.csv", parse_dates=['time'])
except FileNotFoundError:
    print("앗, 'grid_features_and_labels.csv' 파일을 찾을 수 없어. 이전 단계 코드를 확인해봐!")
    exit()

# 지진이 한 번도 없었던 '완전 비활성' 격자-시간 데이터는 학습에 큰 도움이 안 될 수 있으므로 제거
# (선택 사항이지만, 이렇게 하면 학습이 더 효율적일 수 있어)
df_active = df[df['quake_count'] > 0].copy()
# 만약 active 데이터가 너무 적으면 원본 데이터를 사용
if len(df_active) < 1000:
    print("활성 데이터가 너무 적어 전체 데이터를 사용합니다.")
    df_active = df.copy()
else:
    print(f"전체 {len(df)}개 데이터 중, 지진이 한 번이라도 있었던 {len(df_active)}개의 활성 데이터를 학습에 사용합니다.")


# 1. 데이터 준비: 힌트(X)와 정답(y) 분리
# time, grid_id는 힌트가 아니므로 제외
features = ['quake_count', 'mean_magnitude', 'total_energy_log']
X = df_active[features]
y = df_active['label']

# 2. 시간 순서대로 데이터 분리
# 2019년을 기준으로 학습(train) 데이터와 테스트(test) 데이터로 분리
split_date = '2019-01-01'
train_df = df_active[df_active['time'] < split_date]
test_df = df_active[df_active['time'] >= split_date]

X_train = train_df[features]
y_train = train_df['label']
X_test = test_df[features]
y_test = test_df['label']

print(f"\n학습 데이터: {len(X_train)}개 (기간: ~ {split_date})")
print(f"테스트 데이터: {len(X_test)}개 (기간: {split_date} ~)")
print("\n테스트 데이터의 정답 분포:")
print(y_test.value_counts())


# 3. AI 모델 생성 및 학습
print("\nRandom Forest 모델을 생성하고 학습을 시작합니다...")
# class_weight='balanced' 옵션으로 데이터 불균형 문제에 대응!
# n_estimators는 나무의 개수, random_state는 결과를 동일하게 만들기 위한 시드
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)

model.fit(X_train, y_train)
print("학습 완료!")


# 4. 모델 평가
print("\n테스트 데이터로 모델의 성능을 평가합니다...")
y_pred = model.predict(X_test)

# --- 평가 결과 출력 ---
# 정확도
accuracy = accuracy_score(y_test, y_pred)
print(f"\n전체 정확도 (Accuracy): {accuracy:.4f}")

# 분류 리포트 (정밀도, 재현율 등)
# label 1의 recall(재현율)이 우리가 얼마나 '위험'을 잘 잡아냈는지 보여주는 핵심 지표!
print("\n[분류 리포트]")
print(classification_report(y_test, y_pred, target_names=['안전(0)', '위험(1)']))

# 혼동 행렬 (Confusion Matrix)
print("\n[혼동 행렬]")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['예측:안전', '예측:위험'], yticklabels=['실제:안전', '실제:위험'])
plt.title('혼동 행렬 (Confusion Matrix)')
plt.ylabel('실제 값')
plt.xlabel('모델 예측 값')
plt.show()