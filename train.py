import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# =========================
# 유틸: 결측치 요약(원하면 상위 n개만)
# =========================
def na_summary(df: pd.DataFrame, top_n: int = 10):
    na = df.isnull().sum()
    na = na[na > 0].sort_values(ascending=False)
    return len(na), (na.head(top_n) if len(na) > 0 else na)


def print_one_line(tag: str, shape_in, shape_out, drop50_cnt: int, extra_removed, converted, na_cols_cnt: int):
    print(
        f"[{tag}] shape {shape_in} -> {shape_out} | "
        f"drop50%={drop50_cnt} | extra_drop={len(extra_removed)} | "
        f"to_str={converted} | na_cols={na_cols_cnt}"
    )


# =========================
# 0) 전처리 파라미터 fit (train split에서만)
# =========================
def fit_preprocess_params(train_df: pd.DataFrame, threshold: float = 0.5):
    # (1) 50% 이상 결측 컬럼 드롭 목록 (train 기준으로만 결정)
    missing_ratio = train_df.isnull().mean()
    cols_to_drop = [
        col for col in missing_ratio[missing_ratio >= threshold].index
        if col != "major1_2"
    ]

    # (2) completed_semester 중앙값 (train에서만 계산, 이상치 NaN 처리 후)
    tmp = train_df.drop(columns=cols_to_drop)
    tmp = tmp.drop(columns=["nationality", "ID", "generation"], errors="ignore")

    outlier_values = [20241.00, 2020.02]
    tmp.loc[tmp["completed_semester"].isin(outlier_values), "completed_semester"] = np.nan
    median_semester = tmp["completed_semester"].median()

    return cols_to_drop, median_semester


# =========================
# 1) 전처리 transform (train/valid/test 공통 적용)
# =========================
def transform_preprocess(df: pd.DataFrame, cols_to_drop, median_semester: float, tag: str, show_na_top: bool = False):
    shape_in = df.shape

    # (1) 50% 이상 결측 컬럼 드롭
    df_clean = df.drop(columns=cols_to_drop)
    drop50_cnt = len(cols_to_drop)

    # (2) 추가 제거: nationality, ID, generation
    extra_drop_cols = ["nationality", "ID", "generation"]
    extra_removed = [c for c in extra_drop_cols if c in df_clean.columns]
    df_clean = df_clean.drop(columns=extra_drop_cols, errors="ignore")

    # (3) school1, class1: 정수 -> 문자
    converted = []
    for col in ["school1", "class1"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)
            converted.append(col)

    # (4) completed_semester: 이상치->NaN, 결측/이상치->중앙값(train 기준), int 변환 (0은 유지)
    outlier_values = [20241.00, 2020.02]
    df_clean.loc[df_clean["completed_semester"].isin(outlier_values), "completed_semester"] = np.nan
    df_clean["completed_semester"] = df_clean["completed_semester"].fillna(median_semester)
    df_clean["completed_semester"] = df_clean["completed_semester"].round().astype(int)

    # (5) 전공 처리 + major type
    if "major1_2" in df_clean.columns:
        df_clean["major1_2"] = df_clean["major1_2"].replace(r"^\s*$", np.nan, regex=True)
        has_major2_original = df_clean["major1_2"].notna()
    else:
        has_major2_original = pd.Series(False, index=df_clean.index)

    df_clean["major1_1"] = df_clean["major1_1"].replace(r"^\s*$", np.nan, regex=True).fillna("Unknown")
    df_clean["major_field"] = df_clean["major_field"].replace(r"^\s*$", np.nan, regex=True).fillna("Unknown")
    df_clean["major1_2"] = df_clean["major1_2"].fillna("없음")

    df_clean["major type"] = np.where(
        has_major2_original,
        "복수 전공 ( 다중전공, 이중전공 포함 )",
        "단일 전공"
    )

    # =========================
    # (6) 이진 범주형 레이블 인코딩 (0/1)
    # =========================
    # major type: 단일=0, 복수=1
    if "major type" in df_clean.columns:
        major_type_map = {
            "단일 전공": 0,
            "복수 전공 ( 다중전공, 이중전공 포함 )": 1
        }
        df_clean["major type"] = df_clean["major type"].map(major_type_map).astype(int)

    # re_registration: 아니요=0, 예=1
    if "re_registration" in df_clean.columns:
        re_reg_map = {"아니요": 0, "예": 1}
        df_clean["re_registration"] = df_clean["re_registration"].map(re_reg_map).astype(int)

    # project_type: 개인=0, 팀=1
    if "project_type" in df_clean.columns:
        project_type_map = {"개인": 0, "팀": 1}
        df_clean["project_type"] = df_clean["project_type"].map(project_type_map).astype(int)

    # major_data: False=0, True=1 (bool/string 섞여도 안전하게)
    if "major_data" in df_clean.columns:
        s = df_clean["major_data"].astype(str).str.strip().str.upper()
        major_data_map = {"FALSE": 0, "TRUE": 1}
        df_clean["major_data"] = s.map(major_data_map).astype(int)

    # 핵심 요약 1줄 출력
    na_cols_cnt, na_top = na_summary(df_clean, top_n=10)
    print_one_line(
        tag=tag,
        shape_in=shape_in,
        shape_out=df_clean.shape,
        drop50_cnt=drop50_cnt,
        extra_removed=extra_removed,
        converted=converted,
        na_cols_cnt=na_cols_cnt
    )

    # 원하면(보통 TRAIN_SPLIT만) 결측 상위만 추가 출력
    if show_na_top and na_cols_cnt > 0:
        print(na_top.to_string())

    return df_clean


# =========================
# 2) 실행: train/valid split -> fit -> transform
# =========================
train_path = r"..\datasets\train.csv"
test_path  = r"..\datasets\test.csv"

train_df = pd.read_csv(train_path, encoding="utf-8-sig")
test_df  = pd.read_csv(test_path,  encoding="utf-8-sig")

# (A) 검증 세트 분할
y_raw = train_df["completed"]
train_part, valid_part = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=y_raw
)
print(f"[Split] train_part={train_part.shape}, valid_part={valid_part.shape}")

# (B) 전처리 파라미터는 train_part에서만 fit
cols_to_drop, median_semester = fit_preprocess_params(train_part, threshold=0.5)
print(f"[FIT] drop50% cols={len(cols_to_drop)}")

# (C) 동일 transform 적용 (결측 상위 출력은 TRAIN_SPLIT에서만)
train_clean = transform_preprocess(train_part, cols_to_drop, median_semester, tag="TRAIN_SPLIT", show_na_top=True)
valid_clean = transform_preprocess(valid_part, cols_to_drop, median_semester, tag="VALID_SPLIT", show_na_top=False)
test_clean  = transform_preprocess(test_df,      cols_to_drop, median_semester, tag="TEST",        show_na_top=False)

# (D) X/y 분리 + 컬럼 정렬(인코딩 전 단계)
y_train = train_clean["completed"]
y_valid = valid_clean["completed"]

X_train = train_clean.drop(columns=["completed"])
X_valid = valid_clean.drop(columns=["completed"])

if "completed" in test_clean.columns:
    test_clean = test_clean.drop(columns=["completed"])

common_cols = [c for c in X_train.columns if c in X_valid.columns and c in test_clean.columns]
X_train = X_train[common_cols]
X_valid = X_valid[common_cols]
X_test  = test_clean[common_cols]

print(f"[Final] X_train={X_train.shape}, y_train={y_train.shape} | X_valid={X_valid.shape}, y_valid={y_valid.shape} | X_test={X_test.shape}")
