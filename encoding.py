import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


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
# (NEW) 이진 인코딩 안전장치용 유틸
# =========================
def _safe_mode_int(series: pd.Series, default: int = 0) -> int:
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    if len(s) == 0:
        return int(default)
    # mode()가 여러 개면 첫 번째 사용
    return int(s.mode().iloc[0])


def _encode_major_data_to_int(series: pd.Series) -> pd.Series:
    # bool이면 바로 0/1
    if series.dtype == bool:
        return series.astype(int)

    s = series.astype(str).str.strip().str.upper()
    major_data_map = {"FALSE": 0, "TRUE": 1}
    return s.map(major_data_map)


def _binary_encode_with_fallback(df: pd.DataFrame, col: str, mapped: pd.Series, fallback_int: int, tag: str):
    """
    mapped에 NaN이 있으면 fallback_int로 채운 뒤 int로 변환.
    """
    if col not in df.columns:
        return

    unknown_mask = mapped.isna()
    if unknown_mask.any():
        cnt = int(unknown_mask.sum())
        # 너무 시끄러우면 print를 끄고 싶을 수 있는데, 일단은 디버깅을 위해 남김
        print(f"[WARN][{tag}] '{col}' has {cnt} unmapped/NA values -> filled with fallback={fallback_int}")

    df[col] = mapped.fillna(fallback_int).astype(int)


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
    tmp = train_df.drop(columns=cols_to_drop, errors="ignore")
    tmp = tmp.drop(columns=["nationality", "ID", "generation"], errors="ignore")

    outlier_values = [20241.00, 2020.02]
    if "completed_semester" in tmp.columns:
        tmp.loc[tmp["completed_semester"].isin(outlier_values), "completed_semester"] = np.nan
        median_semester = tmp["completed_semester"].median()
    else:
        median_semester = 0

    # (3) (NEW) 이진 컬럼 fallback(= NaN 방지용 대체값) 계산: train에서만 산출
    #     transform에서 map 결과 NaN이면 이 값으로 채워서 astype(int) 에러를 막는다.
    #     - major type: 우리가 생성하므로 원칙상 NaN 없어야 하지만, 안전을 위해 fallback 계산
    #     - re_registration / project_type / major_data: 원본 값이 이상하면 NaN이 나올 수 있음
    binary_fallbacks = {}

    # major type fallback (0/1) — train 기준
    # major1_2가 "실제로 존재했는지"로 생성 (네 로직과 동일)
    if "major1_2" in train_df.columns:
        m2 = train_df["major1_2"].replace(r"^\s*$", np.nan, regex=True)
        has_major2_original = m2.notna()
    else:
        has_major2_original = pd.Series(False, index=train_df.index)

    major_type_int = np.where(has_major2_original, 1, 0)
    binary_fallbacks["major type"] = _safe_mode_int(pd.Series(major_type_int), default=0)

    # re_registration fallback (0/1)
    if "re_registration" in train_df.columns:
        s = train_df["re_registration"].astype(str).str.strip()
        mapped = s.map({"아니요": 0, "예": 1})
        binary_fallbacks["re_registration"] = _safe_mode_int(mapped, default=0)
    else:
        binary_fallbacks["re_registration"] = 0

    # project_type fallback (0/1)
    if "project_type" in train_df.columns:
        s = train_df["project_type"].astype(str).str.strip()
        mapped = s.map({"개인": 0, "팀": 1})
        binary_fallbacks["project_type"] = _safe_mode_int(mapped, default=0)
    else:
        binary_fallbacks["project_type"] = 0

    # major_data fallback (0/1)
    if "major_data" in train_df.columns:
        mapped = _encode_major_data_to_int(train_df["major_data"])
        binary_fallbacks["major_data"] = _safe_mode_int(mapped, default=0)
    else:
        binary_fallbacks["major_data"] = 0

    return cols_to_drop, median_semester, binary_fallbacks


# =========================
# 1) 전처리 transform (train/valid/test 공통 적용)
# =========================
def transform_preprocess(
    df: pd.DataFrame,
    cols_to_drop,
    median_semester: float,
    binary_fallbacks: dict,
    tag: str,
    show_na_top: bool = False
):
    shape_in = df.shape

    # (1) 50% 이상 결측 컬럼 드롭
    df_clean = df.drop(columns=cols_to_drop, errors="ignore")
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

    # (4) completed_semester: 이상치->NaN, 결측/이상치->중앙값(train 기준), int 변환
    if "completed_semester" in df_clean.columns:
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

    if "major1_1" in df_clean.columns:
        df_clean["major1_1"] = df_clean["major1_1"].replace(r"^\s*$", np.nan, regex=True).fillna("Unknown")

    if "major_field" in df_clean.columns:
        df_clean["major_field"] = df_clean["major_field"].replace(r"^\s*$", np.nan, regex=True).fillna("Unknown")

    if "major1_2" in df_clean.columns:
        df_clean["major1_2"] = df_clean["major1_2"].fillna("없음")

    df_clean["major type"] = np.where(
        has_major2_original,
        "복수 전공 ( 다중전공, 이중전공 포함 )",
        "단일 전공"
    )

    # =========================
    # (6) 이진 범주형 레이블 인코딩 (0/1) + (NEW) 안전장치
    # =========================
    # major type: 단일=0, 복수=1
    if "major type" in df_clean.columns:
        s = df_clean["major type"].astype(str).str.strip()
        mapped = s.map({"단일 전공": 0, "복수 전공 ( 다중전공, 이중전공 포함 )": 1})
        _binary_encode_with_fallback(
            df_clean, "major type", mapped,
            fallback_int=int(binary_fallbacks.get("major type", 0)),
            tag=tag
        )

    # re_registration: 아니요=0, 예=1
    if "re_registration" in df_clean.columns:
        s = df_clean["re_registration"].astype(str).str.strip()
        mapped = s.map({"아니요": 0, "예": 1})
        _binary_encode_with_fallback(
            df_clean, "re_registration", mapped,
            fallback_int=int(binary_fallbacks.get("re_registration", 0)),
            tag=tag
        )

    # project_type: 개인=0, 팀=1
    if "project_type" in df_clean.columns:
        s = df_clean["project_type"].astype(str).str.strip()
        mapped = s.map({"개인": 0, "팀": 1})
        _binary_encode_with_fallback(
            df_clean, "project_type", mapped,
            fallback_int=int(binary_fallbacks.get("project_type", 0)),
            tag=tag
        )

    # major_data: False=0, True=1 (bool/string 섞여도 안전하게) + 안전장치
    if "major_data" in df_clean.columns:
        mapped = _encode_major_data_to_int(df_clean["major_data"])
        _binary_encode_with_fallback(
            df_clean, "major_data", mapped,
            fallback_int=int(binary_fallbacks.get("major_data", 0)),
            tag=tag
        )

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

    if show_na_top and na_cols_cnt > 0:
        print(na_top.to_string())

    return df_clean


# =========================
# 1.5) 멀티셀렉트 처리 (정규화 + 희귀→기타 + unknown→기타 + 멀티핫)
# =========================
MULTI_SELECT_COLS = [
    "certificate_acquisition",
    "desired_certificate",
    "desired_job_except_data",
    "expected_domain",
    "onedayclass_topic",
]

RARE_THRESHOLD = 3
RARE_TO_ETC_COLS = {
    "certificate_acquisition": RARE_THRESHOLD,
    "desired_certificate": RARE_THRESHOLD,
    "onedayclass_topic": RARE_THRESHOLD,
}

ETC_TOKEN = "기타"

KEEP_NONE_AS_LABEL = True
NONE_TOKENS = {"없음"}


def resolve_multiselect_columns(df: pd.DataFrame, wanted_cols):
    resolved = []
    col_map = {}
    for c in wanted_cols:
        if c in df.columns:
            resolved.append(c)
            col_map[c] = c
            continue
        alt = c.replace("_", "")
        if alt in df.columns:
            resolved.append(alt)
            col_map[alt] = c
            continue
    return resolved, col_map


def _normalize_separators(s: str) -> str:
    for sep in ["|", "/", ";", "\n"]:
        s = s.replace(sep, ",")
    s = re.sub(r",\s*,+", ",", s)
    return s.strip()


def split_commas_outside_parentheses(text: str):
    parts, buf, depth = [], [], 0
    for ch in text:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    last = "".join(buf).strip()
    if last:
        parts.append(last)
    return [p.strip() for p in parts if p.strip()]


def split_by_code_prefix(text: str, code_regex: str):
    matches = list(re.finditer(code_regex, text))
    if not matches:
        t = text.strip()
        return [t] if t else []

    out = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        chunk = re.sub(r"^[,\s]+", "", chunk).strip()
        if chunk:
            out.append(chunk)
    return out


def label_normalize(label: str, colname: str) -> str:
    s = str(label).strip()
    s = re.sub(r"\s+", " ", s)
    if colname in {"desired_job_except_data", "expected_domain"}:
        s = re.sub(r"\s*,\s*$", "", s).strip()
    return s


def parse_multiselect_cell(value, colname: str):
    if pd.isna(value):
        return []
    s = str(value).strip()
    if s == "":
        return []

    s = _normalize_separators(s)

    if colname == "expected_domain":
        tokens = split_by_code_prefix(s, r"[A-U]\.\s*")
    elif colname == "desired_job_except_data":
        tokens = split_by_code_prefix(s, r"[A-I]\.\s*")
    else:
        tokens = split_commas_outside_parentheses(s)

    out = []
    seen = set()
    for t in tokens:
        t = label_normalize(t, colname)
        if t == "":
            continue
        if (not KEEP_NONE_AS_LABEL) and (t in NONE_TOKENS):
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_rare_to_etc_map(train_tokens_list, threshold: int):
    all_tokens = [t for lst in train_tokens_list for t in lst]
    vc = pd.Series(all_tokens).value_counts()
    rare_labels = set(vc[vc <= threshold].index.tolist())
    rare_labels.discard(ETC_TOKEN)
    return {lab: ETC_TOKEN for lab in rare_labels}


def apply_mapping_A(tokens, rare_map: dict, allowed_set: set):
    out = []
    seen = set()
    for t in tokens:
        t2 = rare_map.get(t, t)
        if t2 not in allowed_set:
            t2 = ETC_TOKEN
        if t2 not in seen:
            seen.add(t2)
            out.append(t2)
    return out


def fit_transform_multiselect_A(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame,
                                multi_cols, show_summary: bool = True):
    resolved_cols, col_map = resolve_multiselect_columns(train_df, multi_cols)
    if len(resolved_cols) == 0:
        print("[MultiSelect] No multi-select columns found. Skip.")
        return train_df, valid_df, test_df

    train_out = train_df.copy()
    valid_out = valid_df.copy()
    test_out = test_df.copy()

    for actual_col in resolved_cols:
        std_col = col_map.get(actual_col, actual_col)

        tr_tokens = train_out[actual_col].map(lambda x: parse_multiselect_cell(x, std_col)).tolist()

        if actual_col in valid_out.columns:
            va_tokens = valid_out[actual_col].map(lambda x: parse_multiselect_cell(x, std_col)).tolist()
        else:
            va_tokens = [[] for _ in range(len(valid_out))]

        if actual_col in test_out.columns:
            te_tokens = test_out[actual_col].map(lambda x: parse_multiselect_cell(x, std_col)).tolist()
        else:
            te_tokens = [[] for _ in range(len(test_out))]

        threshold = RARE_TO_ETC_COLS.get(std_col, None)
        rare_map = build_rare_to_etc_map(tr_tokens, threshold) if threshold is not None else {}

        tr_tokens_mapped = [[rare_map.get(t, t) for t in lst] for lst in tr_tokens]
        allowed_set = set([t for lst in tr_tokens_mapped for t in lst])
        allowed_set.add(ETC_TOKEN)

        tr_final = [apply_mapping_A(lst, rare_map, allowed_set) for lst in tr_tokens]
        va_final = [apply_mapping_A(lst, rare_map, allowed_set) for lst in va_tokens]
        te_final = [apply_mapping_A(lst, rare_map, allowed_set) for lst in te_tokens]

        classes = sorted(list(allowed_set))
        mlb = MultiLabelBinarizer(classes=classes, sparse_output=False)
        X_tr = mlb.fit_transform(tr_final)
        X_va = mlb.transform(va_final)
        X_te = mlb.transform(te_final)

        new_cols = [f"{std_col}__{cls}" for cls in mlb.classes_]

        train_out = pd.concat(
            [train_out.drop(columns=[actual_col]), pd.DataFrame(X_tr, columns=new_cols, index=train_out.index)],
            axis=1
        )
        valid_out = pd.concat(
            [valid_out.drop(columns=[actual_col]), pd.DataFrame(X_va, columns=new_cols, index=valid_out.index)],
            axis=1
        )
        test_out = pd.concat(
            [test_out.drop(columns=[actual_col]), pd.DataFrame(X_te, columns=new_cols, index=test_out.index)],
            axis=1
        )

        if show_summary:
            rare_cnt = len(set(rare_map.keys()))
            print(f"[MultiSelect-A] {std_col}: +{len(new_cols)} cols | rare<= {threshold if threshold is not None else 'N/A'} mapped={rare_cnt} | classes={len(classes)}")

    return train_out, valid_out, test_out


# =========================
# 2) 실행: train/valid split -> fit -> transform -> 멀티셀렉트 처리(A) -> X/y
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
cols_to_drop, median_semester, binary_fallbacks = fit_preprocess_params(train_part, threshold=0.5)
print(f"[FIT] drop50% cols={len(cols_to_drop)}")
print(f"[FIT] binary_fallbacks={binary_fallbacks}")

# (C) 동일 transform 적용
train_clean = transform_preprocess(train_part, cols_to_drop, median_semester, binary_fallbacks, tag="TRAIN_SPLIT", show_na_top=True)
valid_clean = transform_preprocess(valid_part, cols_to_drop, median_semester, binary_fallbacks, tag="VALID_SPLIT", show_na_top=False)
test_clean  = transform_preprocess(test_df,     cols_to_drop, median_semester, binary_fallbacks, tag="TEST",        show_na_top=False)

# (C-2) 멀티셀렉트 5개 컬럼: 정규화 + 희귀->기타 + unknown->기타(A) + 멀티핫
train_enc, valid_enc, test_enc = fit_transform_multiselect_A(
    train_clean, valid_clean, test_clean,
    multi_cols=MULTI_SELECT_COLS,
    show_summary=True
)

# (D) X/y 분리 + 컬럼 정렬(인코딩 후 단계)
y_train = train_enc["completed"]
y_valid = valid_enc["completed"]

X_train = train_enc.drop(columns=["completed"])
X_valid = valid_enc.drop(columns=["completed"])

if "completed" in test_enc.columns:
    test_enc = test_enc.drop(columns=["completed"])

common_cols = [c for c in X_train.columns if c in X_valid.columns and c in test_enc.columns]
X_train = X_train[common_cols]
X_valid = X_valid[common_cols]
X_test  = test_enc[common_cols]

print(f"[Final] X_train={X_train.shape}, y_train={y_train.shape} | X_valid={X_valid.shape}, y_valid={y_valid.shape} | X_test={X_test.shape}")
