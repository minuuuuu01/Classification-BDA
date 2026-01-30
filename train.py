import pandas as pd
import numpy as np

# =========================
# 0) 데이터 로드
# =========================
df = pd.read_csv(r"..\datasets\train.csv", encoding="utf-8-sig")

# =========================
# 1) 50% 이상 결측치 피처 제거 (major1_2는 유지)
# =========================
threshold = 0.5
missing_ratio = df.isnull().mean()

cols_to_drop = [
    col for col in missing_ratio[missing_ratio >= threshold].index
    if col != "major1_2"
]

print(f"제거할 피처 ({len(cols_to_drop)}개): {cols_to_drop}")
print(f"제거 전 shape: {df.shape}")

df_clean = df.drop(columns=cols_to_drop)
print(f"50% 이상 결측 제거 후 shape: {df_clean.shape}")

# =========================
# 1.5) 추가 피처 제거: nationality, ID, generation
# =========================
extra_drop_cols = ["nationality", "ID", "generation"]
df_clean = df_clean.drop(columns=extra_drop_cols, errors="ignore")  # 없으면 무시
print(f"추가 제거 컬럼: {extra_drop_cols}")
print(f"추가 제거 후 shape: {df_clean.shape}")

# =========================
# 1.6) school1, class1: 정수 -> 문자(범주형 취급)
# =========================
# (인코딩은 나중 단계에서 진행. 지금은 '코드/라벨'임을 명확히 하기 위한 변환)
for col in ["school1", "class1"]:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(str)  # 정수형 코드를 문자열로 변환 [web:437]
print("school1/class1 dtype -> str 변환 완료")

# =========================
# 2) completed_semester 이상치/결측 처리 + 정수 변환
# =========================
print("\n=== 2. completed_semester 이상치/결측 처리 ===")

# 2-1) 이상치(20241.00 2개, 2020.02 1개) -> NaN (0은 유지)
outlier_values = [20241.00, 2020.02]
outlier_mask = df_clean["completed_semester"].isin(outlier_values)
print(f"completed_semester 이상치 개수: {outlier_mask.sum()}개")

df_clean.loc[outlier_mask, "completed_semester"] = np.nan

# 2-2) 결측(기존 결측 + 이상치 치환분) -> 중앙값으로 대체
median_semester = df_clean["completed_semester"].median()  # NaN 자동 제외
df_clean["completed_semester"] = df_clean["completed_semester"].fillna(median_semester)
print(f"completed_semester 결측/이상치 -> 중앙값 {median_semester}로 대체 완료")

# 2-3) 정수형으로 변환
df_clean["completed_semester"] = df_clean["completed_semester"].round().astype(int)
print(f"completed_semester dtype 변환 완료: {df_clean['completed_semester'].dtype}")

# =========================
# 3) 전공 관련 4개 처리 (요청사항 반영)
# =========================

# 3-1) major1_2: (빈문자 포함) 결측 판단을 위해 먼저 정리 + "원래 있었는지" 플래그 저장
if "major1_2" in df_clean.columns:
    df_clean["major1_2"] = df_clean["major1_2"].replace(r"^\s*$", np.nan, regex=True)
    has_major2_original = df_clean["major1_2"].notna()  # '없음' 채우기 전에 저장
else:
    has_major2_original = pd.Series(False, index=df_clean.index)

# 3-2) major1_1: Unknown으로 대체 (빈문자 포함)
df_clean["major1_1"] = df_clean["major1_1"].replace(r"^\s*$", np.nan, regex=True)
df_clean["major1_1"] = df_clean["major1_1"].fillna("Unknown")

# 3-3) major_field: Unknown으로 대체 (빈문자 포함)
df_clean["major_field"] = df_clean["major_field"].replace(r"^\s*$", np.nan, regex=True)
df_clean["major_field"] = df_clean["major_field"].fillna("Unknown")

# 3-4) major1_2: '없음'으로 대체
df_clean["major1_2"] = df_clean["major1_2"].fillna("없음")

# 3-5) major type 생성
df_clean["major type"] = np.where(
    has_major2_original,
    "복수 전공 ( 다중전공, 이중전공 포함 )",
    "단일 전공"
)

print("major1_1 결측 -> Unknown 대체 완료")
print("major_field 결측 -> Unknown 대체 완료")
print("major1_2 결측 -> '없음' 대체 완료")
print(f"major type 생성 완료 (남은 결측: {df_clean['major type'].isnull().sum()}개)")

# =========================
# 4) 확인
# =========================
print("\n=== 처리 후 결측 확인(관심 컬럼) ===")
print(df_clean[["major1_1", "major1_2", "major type", "major_field"]].isnull().sum())

print("\n=== 전체 남은 결측치(0 초과만) ===")
left_na = df_clean.isnull().sum()
print(left_na[left_na > 0].sort_values(ascending=False))

print(f"\n최종 shape: {df_clean.shape}")
