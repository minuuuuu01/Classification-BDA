import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import xgboost as xgb
import glob
import os
import re
import matplotlib.pyplot as plt

# 1. 데이터 로드
train_path = r"..\\datasets\\train.csv"
test_path  = r"..\\datasets\\test.csv"

train_df = pd.read_csv(train_path, encoding="utf-8-sig")
test_df  = pd.read_csv(test_path,  encoding="utf-8-sig")

# 2. 데이터 병합
train_len = len(train_df)
df_all = pd.concat([train_df, test_df], axis=0, ignore_index=True)

print(f"전체 데이터 shape: {df_all.shape} (Train: {train_len})")

# ==========================================
# 3. 전처리 (통합 수행)
# ==========================================

# (1) 불필요 컬럼 제거
drop_cols = ['ID', 'generation', 'nationality', 'school1', 
             'interested_company', 'incumbents_lecture_scale_reason']
df_all.drop(columns=drop_cols, inplace=True, errors='ignore')

# (2) 결측치 50% 이상 제거 (타겟 보호)
na_ratio = df_all.isna().sum() / len(df_all)
high_na_cols = na_ratio[na_ratio >= 0.5].index
high_na_cols = [c for c in high_na_cols if c != 'completed'] # 타겟 보호
df_all.drop(columns=high_na_cols, inplace=True)

print(f"결측 50% 이상 제거: {len(high_na_cols)}개 컬럼")

# (3) 결측치 채우기 및 파생변수
median_sem = df_all['completed_semester'].median()
df_all['completed_semester'] = df_all['completed_semester'].fillna(median_sem).astype(int)
df_all.loc[df_all['completed_semester'] > 100, 'completed_semester'] = median_sem

df_all['major1_1'] = df_all['major1_1'].fillna('Unknown')
df_all['major1_2'] = df_all['major1_2'].fillna('없음')
df_all['major_field'] = df_all['major_field'].fillna('Unknown')

def create_major_type(row):
    m1 = pd.notna(row.get('major1_1'))
    m2 = pd.notna(row.get('major1_2')) and row['major1_2'] != '없음'
    return '복수 전공' if m1 and m2 else '단일 전공'
df_all['major_type'] = df_all.apply(create_major_type, axis=1)

# (4) 인코딩

# [수동 레이블 인코딩]
binary_map = {
    're_registration': {'예': 1, '아니요': 0},
    'project_type': {'팀': 1, '개인': 0},
    'major_data': {'TRUE': 1, 'FALSE': 0},
    'incumbents_level': {'시니어 (10년차 ~)': 1, '주니어 (0~3년차)': 0},
    'major_type': {'복수 전공': 1, '단일 전공': 0}
}
for col, mapping in binary_map.items():
    if col in df_all.columns:
        df_all[col] = df_all[col].map(mapping).fillna(0)

# [멀티-핫 인코딩]
multi_hot_cols = ['certificate_acquisition', 'desired_certificate', 'onedayclass_topic',
                  'desired_job_except_data', 'expected_domain', 'desired_job', 'major_field']

for col in multi_hot_cols:
    if col in df_all.columns:
        dummies = df_all[col].str.get_dummies(sep=',')
        dummies.columns = [f"{col}_{c.strip()}" for c in dummies.columns]
        df_all = pd.concat([df_all, dummies], axis=1)
        df_all.drop(columns=[col], inplace=True)

# [원-핫 인코딩]
obj_cols = df_all.select_dtypes(include=['object']).columns
obj_cols = [c for c in obj_cols if c != 'completed']
df_all = pd.get_dummies(df_all, columns=obj_cols, dummy_na=True)

# [중복 컬럼 제거]
df_all = df_all.loc[:, ~df_all.columns.duplicated()]

# [특수문자 제거]
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
df_all.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df_all.columns]

print(f"최종 전처리 후 총 컬럼 수: {df_all.shape[1]}")

# ==========================================
# 4. 데이터 다시 분리
# ==========================================
train_final = df_all.iloc[:train_len].copy()
test_final  = df_all.iloc[train_len:].copy()

y = train_final['completed'].astype(int)
X = train_final.drop(columns=['completed'])
X_test = test_final.drop(columns=['completed'])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"최종 학습 데이터: {X_train.shape}")
print(f"최종 검증 데이터: {X_valid.shape}")

# ==========================================
# 5. 모델 학습 (XGBoost 튜닝 버전 - 최신 문법 적용)
# ==========================================

# 클래스 불균형 비율 계산
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

# [수정] early_stopping_rounds를 여기서 선언합니다.
model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    early_stopping_rounds=30,  # <--- [여기로 이동!]
    
    # [과적합 방지 파라미터]
    max_depth=3,
    learning_rate=0.05,
    n_estimators=2000,
    colsample_bytree=0.5,
    subsample=0.8,
    reg_alpha=1,
    reg_lambda=1,
    scale_pos_weight=ratio
)

print("\n[모델 학습 시작]")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=50  # 로그 출력 주기
)


# ==========================================
# 6. 예측 및 저장
# ==========================================
# best_iteration을 사용하여 예측 (Early Stopping된 최적 지점)
print(f"\nBest Iteration: {model.best_iteration}")
preds = model.predict(X_test, iteration_range=(0, model.best_iteration + 1))

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'completed': preds
})

submit_files = glob.glob("submit*.csv")
max_num = 0
for file in submit_files:
    basename = os.path.basename(file)
    name_part = os.path.splitext(basename)[0]
    if name_part.startswith("submit") and name_part[6:].isdigit():
        num = int(name_part[6:])
        if num > max_num:
            max_num = num

save_filename = f"submit{max_num + 1}.csv"
submission.to_csv(save_filename, index=False)
print(f"\n저장 완료: {save_filename}")

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform

# 1. 한글 폰트 설정
if platform.system() == 'Windows':
    # 윈도우인 경우 '맑은 고딕' 사용
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
elif platform.system() == 'Darwin':
    # 맥(Mac)인 경우 'AppleGothic' 사용
    rc('font', family='AppleGothic')
else:
    # 리눅스/코랩 등은 별도 설치 필요 (나눔고딕 등)
    print("Linux/Colab 환경입니다. 한글 폰트가 설치되어 있어야 합니다.")
    # rc('font', family='NanumGothic')

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# ... (폰트 설정 코드는 그대로 유지) ...

# 2. 시각화 실행 (수정됨)
try:
    print("\n[Feature Importance 시각화]")
    
    # [수정 1] figure와 ax를 동시에 생성 (이러면 창이 2개 안 뜹니다)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # [수정 2] ax=ax 옵션을 추가해서, 위에서 만든 창에다가 그리라고 명령함
    xgb.plot_importance(model, 
                        ax=ax,  # <--- 여기가 핵심!
                        max_num_features=20, 
                        height=0.5, 
                        title="상위 20개 피처 중요도 (Feature Importance)",
                        xlabel="F-Score (기여도)")
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"시각화 오류: {e}")

