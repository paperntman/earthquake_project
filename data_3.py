import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트가 깨지지 않도록 설정 (만약 깨지면 맞는 폰트 이름으로 바꿔줘야 해)
# 윈도우 사용자: 'Malgun Gothic'
# 맥 사용자: 'AppleGothic'
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    print("경고: 'Malgun Gothic' 폰트를 찾을 수 없어. 그래프의 한글이 깨질 수 있습니다.")
    print("맥 사용자는 'AppleGothic'을, 다른 환경에서는 맞는 폰트를 설정해주세요.")
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호도 깨지지 않게


# 전처리 완료된 데이터 불러오기
try:
    df = pd.read_csv("korea_earthquake_data_preprocessed.csv")
    df['time'] = pd.to_datetime(df['time']) # CSV로 저장하면 time 컬럼이 다시 object로 바뀌어서 datetime으로 변환
    print("전처리된 데이터를 성공적으로 불러왔어! 이제 시각화를 시작할게.\n")
except FileNotFoundError:
    print("앗, 'korea_earthquake_data_preprocessed.csv' 파일을 찾을 수 없어. 이전 단계 코드를 확인해봐!")
    exit()


# 시각화를 위해 그림판(figure)을 준비하자. 여러 그래프를 한 번에 그릴 거야.
fig, axes = plt.subplots(2, 2, figsize=(18, 12)) # 2x2 격자에 총 4개의 그래프를 그릴 준비
fig.suptitle('한반도 지진 데이터 탐색 (EDA)', fontsize=20, y=1.02)


# --- 1. 연도별 지진 발생 횟수 ---
sns.countplot(ax=axes[0, 0], x=df['time'].dt.year)
axes[0, 0].set_title('연도별 지진 발생 횟수')
axes[0, 0].set_xlabel('연도')
axes[0, 0].set_ylabel('발생 횟수')
axes[0, 0].tick_params(axis='x', rotation=90) # x축 라벨이 길어서 90도 회전


# --- 2. 지진 발생 위치 (위도, 경도) ---
# hue='magnitude' 옵션으로 규모에 따라 점의 색깔을 다르게 표시
sns.scatterplot(ax=axes[0, 1], data=df, x='longitude', y='latitude', 
                hue='magnitude', size='magnitude', palette='viridis', alpha=0.6)
axes[0, 1].set_title('지진 발생 위치 분포도 (규모별 색상)')
axes[0, 1].set_xlabel('경도 (Longitude)')
axes[0, 1].set_ylabel('위도 (Latitude)')
axes[0, 1].legend(title='규모')


# --- 3. 지진 규모(magnitude) 분포 ---
sns.histplot(ax=axes[1, 0], data=df, x='magnitude', bins=30, kde=True)
axes[1, 0].set_title('지진 규모 분포')
axes[1, 0].set_xlabel('규모 (Magnitude)')
axes[1, 0].set_ylabel('횟수')


# --- 4. 지진 깊이(depth) 분포 ---
sns.histplot(ax=axes[1, 1], data=df, x='depth', bins=30, kde=True)
axes[1, 1].set_title('지진 깊이 분포')
axes[1, 1].set_xlabel('깊이 (km)')
axes[1, 1].set_ylabel('횟수')


# 그래프들끼리 겹치지 않게 레이아웃 조정
plt.tight_layout(rect=[0, 0.03, 1, 0.98])

# 그래프 보여주기
plt.show()

# 규모 5.0 이상 지진 정보 출력
print("\n--- 규모 5.0 이상 주요 지진 ---")
print(df[df['magnitude'] >= 5.0].sort_values(by='time', ascending=False))