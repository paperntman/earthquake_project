import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- 코드 시작 ---
print("AI 학습용 일본 데이터를 불러옵니다...")
try:
    # --- 파일 이름만 바꿔주면 돼! ---
    df = pd.read_csv("grid_features_and_labels_japan.csv", parse_dates=['time'])
except FileNotFoundError:
    print("앗, 'grid_features_and_labels_japan.csv' 파일을 찾을 수 없어. 이전 단계 코드를 확인해봐!")
    exit()

# 지진이 한 번이라도 있었던 데이터만 사용하는 것이 여전히 유효해
df_active = df[df['quake_count'] > 0].copy()
if len(df_active) < 1000:
    print("활성 데이터가 너무 적어 전체 데이터를 사용합니다.")
    df_active = df.copy()
else:
    print(f"전체 {len(df)}개 데이터 중, 지진이 한 번이라도 있었던 {len(df_active)}개의 활성 데이터를 학습에 사용합니다.")


# 1. 데이터 준비
features = ['quake_count', 'mean_magnitude', 'total_energy_log']
X = df_active[features]
y = df_active['label']

# 2. 시간 순서대로 데이터 분리 (2019년 기준)
split_date = '2019-01-01'
train_df = df_active[df_active['time'] < split_date]
test_df = df_active[df_active['time'] >= split_date]

X_train = train_df[features]
y_train = train_df['label']
X_test = test_df[features]
y_test = test_df['label']

print(f"\n학습 데이터: {len(X_train)}개 (기간: 2000 ~ {split_date})")
print(f"테스트 데이터: {len(X_test)}개 (기간: {split_date} ~)")
print("\n테스트 데이터의 정답 분포:")
print(y_test.value_counts())


# 3. AI 모델 생성 및 학습
print("\nRandom Forest 모델을 생성하고 일본 데이터로 학습을 시작합니다...")
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("학습 완료!")


# 4. 모델 평가
print("\n테스트 데이터로 모델의 성능을 평가합니다...")
y_pred = model.predict(X_test)

# --- 평가 결과 출력 ---
print(f"\n전체 정확도 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")

# 이번에는 '위험(1)'의 recall 값이 0.00이 아니길 기대해보자!
print("\n[분류 리포트]")
print(classification_report(y_test, y_pred, target_names=['안전(0)', '위험(1)']))

print("\n[혼동 행렬]")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
# 폰트가 깨질 경우를 대비해 영문으로 설정
plt.rcParams['font.family'] = 'sans-serif'
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted: Safe', 'Predicted: Danger'], yticklabels=['Actual: Safe', 'Actual: Danger'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()