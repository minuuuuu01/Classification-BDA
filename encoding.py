import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


# =========================
# 유틸
# =========================
def na_summary(df: pd.DataFrame, top_n: int = 10):
    na = df.isnull().sum()
    na = na[na > 0].sort_values(ascending=False)
    return len(na), (na.head(top_n) if len(na) > 0 else na)


def print_one_line(tag: str, shape_in, shape_out, drop50_cnt: int, extra_removed, converted, na_cols_cnt: int):
    return


# =========================
# (A) 원-핫 인코딩할 일반 명목형 12개 + class1
# =========================
NOMINAL_12 = [
    "major1_1",
    "major1_2",
    "job",
    "inflow_route",
    "whyBDA",
    "what_to_gain",
    "hope_for_group",
    "desired_career_path",
    "incumbents_lecture",
    "incumbents_company_level",
    "incumbents_lecture_type",
    "incumbents_lecture_scale",
]
NOMINAL_12_PLUS_CLASS1 = NOMINAL_12 + ["class1"]

ETC_TOKEN = "기타"

RARE_EQ_1_COLS = {
    "inflow_route",
    "what_to_gain",
    "incumbents_lecture",
    "incumbents_company_level",
    "incumbents_lecture_scale",
    # class1은 10개라 희귀 통합 규칙 적용하지 않음
}


def _normalize_nominal_value(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def fit_nominal_ohe_params(train_df: pd.DataFrame, cols, rare_eq_1_cols, etc_token: str = "기타"):
    params = {}
    for col in cols:
        if col not in train_df.columns:
            continue

        s = train_df[col].map(_normalize_nominal_value)

        vc = s.value_counts(dropna=False)
        rare_set = set()
        if col in rare_eq_1_cols:
            rare_set = set(vc[vc == 1].index.tolist())
            rare_set = {v for v in rare_set if pd.notna(v)}  # NaN-safe

        s2 = s.copy()
        if len(rare_set) > 0:
            s2 = s2.apply(lambda v: etc_token if v in rare_set else v)

        cats = sorted([c for c in s2.dropna().unique().tolist()])
        if etc_token not in cats:
            cats.append(etc_token)
            cats = sorted(cats)

        params[col] = {"rare_set": rare_set, "categories": cats}
    return params


def transform_nominal_ohe(df: pd.DataFrame, cols, params, etc_token: str = "기타"):
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        if col not in params:
            out = out.drop(columns=[col])
            continue

        rare_set = params[col]["rare_set"]
        categories = params[col]["categories"]
        allowed = set(categories)

        s = out[col].map(_normalize_nominal_value)

        if len(rare_set) > 0:
            s = s.apply(lambda v: etc_token if v in rare_set else v)

        s = s.apply(lambda v: etc_token if (pd.notna(v) and v not in allowed) else v)

        s_cat = pd.Categorical(s, categories=categories)
        dummies = pd.get_dummies(s_cat, prefix=f"{col}", prefix_sep="__", dtype=int)

        out = pd.concat([out.drop(columns=[col]), dummies.set_index(out.index)], axis=1)
    return out


# =========================
# 이진 인코딩 안전장치용 유틸
# =========================
def _safe_mode_int(series: pd.Series, default: int = 0) -> int:
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    if len(s) == 0:
        return int(default)
    return int(s.mode().iloc[0])


def _encode_major_data_to_int(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    s = series.astype(str).str.strip().str.upper()
    major_data_map = {"FALSE": 0, "TRUE": 1}
    return s.map(major_data_map)


def _binary_encode_with_fallback(df: pd.DataFrame, col: str, mapped: pd.Series, fallback_int: int, tag: str):
    if col not in df.columns:
        return
    df[col] = mapped.fillna(fallback_int).astype(int)


# =========================
# desired_job 통합 로직 + 테스트 유틸
# 1) 알파벳(A~J)으로 시작하지 않는 라벨 -> '기타'
# 2) 같은 알파벳 그룹은 train에서 최빈 라벨로 통합
# (FIX) "G. PM, 서비스 기획자" 같은 케이스에서 콤마 뒤 설명을 잘라내서
#       '서비스 기획자', 'UX 디자이너' 같은 추가 라벨이 생기지 않도록 함.
# =========================
ALLOWED_JOB_ALPHAS = list("ABCDEFGHIJ")
ALLOWED_JOB_ALPHA_SET = set(ALLOWED_JOB_ALPHAS)


def _normalize_separators_job(s: str) -> str:
    for sep in ["|", "/", ";", "\n"]:
        s = s.replace(sep, ",")
    s = re.sub(r",\s*,+", ",", s)
    return s.strip()


def _split_by_code_prefix_job(text: str):
    # (FIX) A~J로 제한
    matches = list(re.finditer(r"[A-J]\.\s*", text))
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


def _label_normalize_job(label: str) -> str:
    s = str(label).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*,\s*$", "", s).strip()

    # (FIX 핵심) 코드로 시작하는 라벨은 첫 콤마 앞까지만 라벨로 인정
    # ex) "G. PM, 서비스 기획자" -> "G. PM"
    if re.match(r"^[A-Z]\.\s*", s):
        s = s.split(",")[0].strip()

    return s


def parse_desired_job_cell(value):
    if pd.isna(value):
        return []
    s = str(value).strip()
    if s == "":
        return []
    s = _normalize_separators_job(s)

    tokens = _split_by_code_prefix_job(s)
    if len(tokens) == 0:
        # prefix로 못 나누면 콤마 기반 fallback
        tokens = [t.strip() for t in s.split(",") if t.strip()]

    out, seen = [], set()
    for t in tokens:
        t = _label_normalize_job(t)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _alpha_prefix(label: str):
    m = re.match(r"^([A-Z])\.\s*", str(label).strip())
    return m.group(1) if m else None


def _compute_multilabel_counts(series: pd.Series, parser_fn):
    counts = {}
    for v in series.tolist():
        labels = parser_fn(v)
        for lab in set(labels):  # 한 샘플 안에서는 중복 라벨 1회만 카운트
            counts[lab] = counts.get(lab, 0) + 1
    return pd.Series(counts).sort_values(ascending=False)


def fit_desired_job_collapse_map(train_df: pd.DataFrame, col: str = "desired_job", etc_token: str = "기타"):
    if col not in train_df.columns:
        return {}, {"available": False}

    label_counts = _compute_multilabel_counts(train_df[col], parse_desired_job_cell)

    labels_by_alpha = {}
    non_alpha_labels = []
    out_of_range_alpha_labels = []  # (FIX) A~J 밖 코드 라벨 기록용

    for lab, cnt in label_counts.items():
        a = _alpha_prefix(lab)
        if a is None:
            non_alpha_labels.append((lab, int(cnt)))
        else:
            if a not in ALLOWED_JOB_ALPHA_SET:
                out_of_range_alpha_labels.append((lab, int(cnt)))
            else:
                labels_by_alpha.setdefault(a, []).append((lab, int(cnt)))

    rep_by_alpha = {}
    alpha_stats_rows = []
    for a, pairs in labels_by_alpha.items():
        pairs_sorted = sorted(pairs, key=lambda x: (-x[1], x[0]))  # count desc, label asc
        rep_label, rep_cnt = pairs_sorted[0]
        rep_by_alpha[a] = rep_label
        alpha_stats_rows.append({
            "alpha": a,
            "rep_label": rep_label,
            "rep_count": rep_cnt,
            "n_variants": len(pairs_sorted),
            "variants_top5": ", ".join([p[0] for p in pairs_sorted[:5]])
        })

    # 최종 매핑
    label_map = {}
    for lab in label_counts.index.tolist():
        a = _alpha_prefix(lab)
        if a is None:
            label_map[lab] = etc_token
        else:
            # (FIX) A~J 밖이면 기타로 강제
            if a not in ALLOWED_JOB_ALPHA_SET:
                label_map[lab] = etc_token
            else:
                label_map[lab] = rep_by_alpha.get(a, lab)

    stats = {
        "available": True,
        "raw_label_counts": label_counts,
        "alpha_stats": pd.DataFrame(alpha_stats_rows).sort_values("alpha") if len(alpha_stats_rows) else pd.DataFrame(),
        "non_alpha_top": pd.DataFrame(sorted(non_alpha_labels, key=lambda x: -x[1])[:30], columns=["label", "count"]),
        "out_of_range_alpha_top": pd.DataFrame(sorted(out_of_range_alpha_labels, key=lambda x: -x[1])[:30], columns=["label", "count"]),
        "rep_by_alpha": rep_by_alpha,
    }
    return label_map, stats


def transform_desired_job_column(df: pd.DataFrame, label_map: dict, col: str = "desired_job", etc_token: str = "기타"):
    if col not in df.columns or len(label_map) == 0:
        return df

    out = df.copy()

    def _transform_cell(v):
        if pd.isna(v):
            return np.nan
        labels = parse_desired_job_cell(v)
        if len(labels) == 0:
            return np.nan

        mapped = []
        seen = set()
        for lab in labels:
            lab2 = label_map.get(lab, etc_token if _alpha_prefix(lab) is None else etc_token)
            # 위 default는 안전장치: 맵에 없으면 기타(원-핫 고정용)
            if lab2 not in seen:
                seen.add(lab2)
                mapped.append(lab2)

        if len(mapped) == 0:
            return np.nan
        return ", ".join(mapped)

    out[col] = out[col].map(_transform_cell)
    return out


def debug_print_desired_job_checks(train_raw: pd.DataFrame, train_clean: pd.DataFrame, desired_job_map: dict, fit_stats: dict,
                                  col: str = "desired_job", expected_n_categories: int = 11, max_examples: int = 20):
    print("\n" + "=" * 80)
    print("[TEST] desired_job 통합 검증 시작")

    if col not in train_raw.columns or col not in train_clean.columns:
        print(" - desired_job 컬럼이 없어 테스트 스킵")
        print("=" * 80)
        return

    if fit_stats.get("available", False):
        alpha_stats_df = fit_stats.get("alpha_stats", pd.DataFrame())
        non_alpha_top = fit_stats.get("non_alpha_top", pd.DataFrame())
        out_of_range_alpha_top = fit_stats.get("out_of_range_alpha_top", pd.DataFrame())
        raw_label_counts = fit_stats.get("raw_label_counts", pd.Series(dtype=int))

        print("\n[TEST-1] raw 라벨(파싱 후) 개수 및 상위 20개")
        print(f" - n_labels_raw(parsed) = {len(raw_label_counts)}")
        print(raw_label_counts.head(20).to_string())

        if len(alpha_stats_df) > 0:
            print("\n[TEST-2] 알파벳별(A~J) 대표 라벨(최빈) & 변형 수")
            view = alpha_stats_df[["alpha", "rep_label", "rep_count", "n_variants", "variants_top5"]]
            print(view.to_string(index=False))

        if non_alpha_top is not None and len(non_alpha_top) > 0:
            print("\n[TEST-3] 비-알파벳 라벨 TOP (기타로 흡수 대상)")
            print(non_alpha_top.to_string(index=False))

        if out_of_range_alpha_top is not None and len(out_of_range_alpha_top) > 0:
            print("\n[TEST-3b] A~J 밖 코드 라벨 TOP (기타로 강제)")
            print(out_of_range_alpha_top.to_string(index=False))

    before = _compute_multilabel_counts(train_raw[col], parse_desired_job_cell)
    after = _compute_multilabel_counts(train_clean[col], parse_desired_job_cell)

    print("\n[TEST-4] BEFORE: 라벨별 #samples (top 20)")
    print(before.head(20).to_string())

    print("\n[TEST-5] AFTER: 라벨별 #samples (전체)")
    print(after.to_string())

    n_after = len(after)
    print("\n[TEST-6] AFTER 카테고리 개수 체크")
    print(f" - n_categories_after = {n_after} (expected={expected_n_categories})")
    if n_after != expected_n_categories:
        print(" - [WARN] expected와 다릅니다. (라벨에 콤마 꼬리/범위 밖 코드가 남았는지 확인)")

    raw_s = train_raw[col].astype(str)
    clean_s = train_clean[col].astype(str)
    diff_mask = raw_s.notna() & clean_s.notna() & (raw_s != clean_s)
    diff_idx = train_raw.index[diff_mask].tolist()[:max_examples]

    print("\n[TEST-7] raw -> mapped 변경 예시 (최대 {}개)".format(max_examples))
    if len(diff_idx) == 0:
        print(" - 변경된 샘플이 없습니다(혹은 NA/동일).")
    else:
        for i in diff_idx:
            print(f" - RAW   : {train_raw.loc[i, col]}")
            print(f"   MAPPED: {train_clean.loc[i, col]}")

    print("\n[TEST-8] label_map 요약")
    print(f" - label_map_size = {len(desired_job_map)}")
    if len(desired_job_map) > 0:
        mapped_targets = sorted(set(desired_job_map.values()))
        print(f" - mapped_target_labels = {mapped_targets}")

    print("[TEST] desired_job 통합 검증 종료")
    print("=" * 80)


# =========================
# major_field 통합 로직 + 테스트 유틸
# 1) 지정 4개 -> '이공계'
# 2) 지정 6개 -> '인문계'
# 3) '예체능','의약학','Unknown' 유지
# =========================
MAJOR_FIELD_TO_STEM = {
    "IT (컴퓨터 공학 포함)",
    "공학 (컴퓨터 공학 제외)",
    "자연과학",
    "자연고학",  # 오타 케이스도 포함
}
MAJOR_FIELD_TO_HUM = {
    "경영학",
    "사회과학",
    "인문학",
    "경제통상학",
    "교육학",
    "법학",
}
MAJOR_FIELD_KEEP = {"예체능", "의약학", "Unknown"}

MAJOR_FIELD_STEM_LABEL = "이공계"
MAJOR_FIELD_HUM_LABEL = "인문계"


def _normalize_separators_major(s: str) -> str:
    for sep in ["|", "/", ";", "\n"]:
        s = s.replace(sep, ",")
    s = re.sub(r",\s*,+", ",", s)
    return s.strip()


def _label_normalize_major(label: str) -> str:
    s = str(label).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*,\s*$", "", s).strip()
    return s


def parse_major_field_cell(value):
    if pd.isna(value):
        return []
    s = str(value).strip()
    if s == "":
        return []
    s = _normalize_separators_major(s)

    tokens = [t.strip() for t in s.split(",") if t.strip()]
    out, seen = [], set()
    for t in tokens:
        t = _label_normalize_major(t)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def map_major_field_label(label: str) -> str:
    lab = _label_normalize_major(label)
    if lab in MAJOR_FIELD_TO_STEM:
        return MAJOR_FIELD_STEM_LABEL
    if lab in MAJOR_FIELD_TO_HUM:
        return MAJOR_FIELD_HUM_LABEL
    if lab in MAJOR_FIELD_KEEP:
        return lab
    # 요구사항에 없는 라벨은 그대로 유지(테스트에서 발견 가능)
    return lab


def transform_major_field_column(df: pd.DataFrame, col: str = "major_field"):
    if col not in df.columns:
        return df

    out = df.copy()

    def _transform_cell(v):
        if pd.isna(v):
            return np.nan
        labels = parse_major_field_cell(v)
        if len(labels) == 0:
            return np.nan

        mapped = []
        seen = set()
        for lab in labels:
            lab2 = map_major_field_label(lab)
            if lab2 and lab2 not in seen:
                seen.add(lab2)
                mapped.append(lab2)

        if len(mapped) == 0:
            return np.nan
        return ", ".join(mapped)

    out[col] = out[col].map(_transform_cell)
    return out


def debug_print_major_field_checks(train_raw: pd.DataFrame, train_clean: pd.DataFrame, col: str = "major_field", max_examples: int = 20):
    print("\n" + "=" * 80)
    print("[TEST] major_field 통합 검증 시작")

    if col not in train_raw.columns or col not in train_clean.columns:
        print(" - major_field 컬럼이 없어 테스트 스킵")
        print("=" * 80)
        return

    before = _compute_multilabel_counts(train_raw[col], parse_major_field_cell)
    after = _compute_multilabel_counts(train_clean[col], parse_major_field_cell)

    print("\n[TEST-M1] BEFORE: 라벨별 #samples (전체)")
    print(before.to_string())

    print("\n[TEST-M2] AFTER: 라벨별 #samples (전체)")
    print(after.to_string())

    allowed = {MAJOR_FIELD_STEM_LABEL, MAJOR_FIELD_HUM_LABEL} | MAJOR_FIELD_KEEP
    unexpected = [lab for lab in after.index.tolist() if lab not in allowed]
    print("\n[TEST-M3] AFTER에서 예상 밖 라벨")
    if len(unexpected) == 0:
        print(" - 없음")
    else:
        print(" - " + " | ".join(unexpected))

    raw_s = train_raw[col].astype(str)
    clean_s = train_clean[col].astype(str)
    diff_mask = raw_s.notna() & clean_s.notna() & (raw_s != clean_s)
    diff_idx = train_raw.index[diff_mask].tolist()[:max_examples]

    print("\n[TEST-M4] raw -> mapped 변경 예시 (최대 {}개)".format(max_examples))
    if len(diff_idx) == 0:
        print(" - 변경된 샘플이 없습니다(혹은 NA/동일).")
    else:
        for i in diff_idx:
            print(f" - RAW   : {train_raw.loc[i, col]}")
            print(f"   MAPPED: {train_clean.loc[i, col]}")

    print("[TEST] major_field 통합 검증 종료")
    print("=" * 80)


# =========================
# desired_job + major_field 원-핫(멀티라벨용) 유틸
# =========================
def _safe_parse_tokens_from_comma_cell(value):
    # "A, B, C" 형태(또는 NaN) -> ["A","B","C"]
    if pd.isna(value):
        return []
    s = str(value).strip()
    if s == "":
        return []
    parts = [p.strip() for p in re.split(r"\s*,\s*", s) if p.strip()]
    out, seen = [], set()
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def fit_transform_multilabel_ohe_fixed(train_df: pd.DataFrame,
                                      valid_df: pd.DataFrame,
                                      test_df: pd.DataFrame,
                                      col: str,
                                      parser_fn,
                                      classes: list,
                                      expected_n: int):
    if col not in train_df.columns:
        return train_df, valid_df, test_df

    if len(classes) != expected_n:
        raise ValueError(f"[OHE] {col} classes len={len(classes)} expected={expected_n}. classes={classes}")

    mlb = MultiLabelBinarizer(classes=classes, sparse_output=False)
    X_tr = mlb.fit_transform(train_df[col].map(parser_fn).tolist())
    X_va = mlb.transform(valid_df[col].map(parser_fn).tolist()) if col in valid_df.columns else np.zeros((len(valid_df), len(classes)), dtype=int)
    X_te = mlb.transform(test_df[col].map(parser_fn).tolist())  if col in test_df.columns  else np.zeros((len(test_df),  len(classes)), dtype=int)

    new_cols = [f"{col}__{cls}" for cls in mlb.classes_]

    train_out = pd.concat([train_df.drop(columns=[col]), pd.DataFrame(X_tr, columns=new_cols, index=train_df.index)], axis=1)
    valid_out = pd.concat([valid_df.drop(columns=[col]), pd.DataFrame(X_va, columns=new_cols, index=valid_df.index)], axis=1)
    test_out  = pd.concat([test_df.drop(columns=[col]),  pd.DataFrame(X_te, columns=new_cols, index=test_df.index)], axis=1)
    return train_out, valid_out, test_out


def assert_ohe_columns_exist(df: pd.DataFrame, col: str, classes: list):
    need = [f"{col}__{c}" for c in classes]
    missing = [c for c in need if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"[OHE-ASSERT] missing columns for {col}: {missing}")


# =========================
# 0) 전처리 파라미터 fit (train split에서만)
# =========================
def fit_preprocess_params(train_df: pd.DataFrame, threshold: float = 0.5):
    missing_ratio = train_df.isnull().mean()
    cols_to_drop = [
        col for col in missing_ratio[missing_ratio >= threshold].index
        if col != "major1_2"
    ]

    tmp = train_df.drop(columns=cols_to_drop, errors="ignore")
    tmp = tmp.drop(columns=["nationality", "ID", "generation"], errors="ignore")

    outlier_values = [20241.00, 2020.02]
    if "completed_semester" in tmp.columns:
        tmp.loc[tmp["completed_semester"].isin(outlier_values), "completed_semester"] = np.nan
        median_semester = tmp["completed_semester"].median()
    else:
        median_semester = 0

    binary_fallbacks = {}

    if "major1_2" in train_df.columns:
        m2 = train_df["major1_2"].replace(r"^\s*$", np.nan, regex=True)
        has_major2_original = m2.notna()
    else:
        has_major2_original = pd.Series(False, index=train_df.index)

    major_type_int = np.where(has_major2_original, 1, 0)
    binary_fallbacks["major type"] = _safe_mode_int(pd.Series(major_type_int), default=0)

    if "re_registration" in train_df.columns:
        s = train_df["re_registration"].astype(str).str.strip()
        mapped = s.map({"아니요": 0, "예": 1})
        binary_fallbacks["re_registration"] = _safe_mode_int(mapped, default=0)
    else:
        binary_fallbacks["re_registration"] = 0

    if "project_type" in train_df.columns:
        s = train_df["project_type"].astype(str).str.strip()
        mapped = s.map({"개인": 0, "팀": 1})
        binary_fallbacks["project_type"] = _safe_mode_int(mapped, default=0)
    else:
        binary_fallbacks["project_type"] = 0

    if "major_data" in train_df.columns:
        mapped = _encode_major_data_to_int(train_df["major_data"])
        binary_fallbacks["major_data"] = _safe_mode_int(mapped, default=0)
    else:
        binary_fallbacks["major_data"] = 0

    if "incumbents_level" in train_df.columns:
        s = train_df["incumbents_level"].astype(str).str.strip()
        mapped = s.map({"주니어 (0~3년차)": 0, "시니어 (10년차 ~)": 1})
        binary_fallbacks["incumbents_level"] = _safe_mode_int(mapped, default=0)
    else:
        binary_fallbacks["incumbents_level"] = 0

    desired_job_map, desired_job_fit_stats = fit_desired_job_collapse_map(train_df, col="desired_job", etc_token=ETC_TOKEN)

    return cols_to_drop, median_semester, binary_fallbacks, desired_job_map, desired_job_fit_stats


# =========================
# 1) 전처리 transform (train/valid/test 공통 적용)
# =========================
def transform_preprocess(
    df: pd.DataFrame,
    cols_to_drop,
    median_semester: float,
    binary_fallbacks: dict,
    desired_job_map: dict,
    tag: str,
    show_na_top: bool = False
):
    df_clean = df.drop(columns=cols_to_drop, errors="ignore")
    df_clean = df_clean.drop(columns=["nationality", "ID", "generation"], errors="ignore")

    df_clean = df_clean.drop(
        columns=["school1", "interested_company", "incumbents_lecture_scale_reason"],
        errors="ignore"
    )

    if "completed_semester" in df_clean.columns:
        outlier_values = [20241.00, 2020.02]
        df_clean.loc[df_clean["completed_semester"].isin(outlier_values), "completed_semester"] = np.nan
        df_clean["completed_semester"] = df_clean["completed_semester"].fillna(median_semester)
        df_clean["completed_semester"] = df_clean["completed_semester"].round().astype(int)

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

    if "major type" in df_clean.columns:
        s = df_clean["major type"].astype(str).str.strip()
        mapped = s.map({"단일 전공": 0, "복수 전공 ( 다중전공, 이중전공 포함 )": 1})
        _binary_encode_with_fallback(df_clean, "major type", mapped, int(binary_fallbacks.get("major type", 0)), tag)

    if "re_registration" in df_clean.columns:
        s = df_clean["re_registration"].astype(str).str.strip()
        mapped = s.map({"아니요": 0, "예": 1})
        _binary_encode_with_fallback(df_clean, "re_registration", mapped, int(binary_fallbacks.get("re_registration", 0)), tag)

    if "project_type" in df_clean.columns:
        s = df_clean["project_type"].astype(str).str.strip()
        mapped = s.map({"개인": 0, "팀": 1})
        _binary_encode_with_fallback(df_clean, "project_type", mapped, int(binary_fallbacks.get("project_type", 0)), tag)

    if "major_data" in df_clean.columns:
        mapped = _encode_major_data_to_int(df_clean["major_data"])
        _binary_encode_with_fallback(df_clean, "major_data", mapped, int(binary_fallbacks.get("major_data", 0)), tag)

    if "incumbents_level" in df_clean.columns:
        s = df_clean["incumbents_level"].astype(str).str.strip()
        mapped = s.map({"주니어 (0~3년차)": 0, "시니어 (10년차 ~)": 1})
        _binary_encode_with_fallback(
            df_clean,
            "incumbents_level",
            mapped,
            int(binary_fallbacks.get("incumbents_level", 0)),
            tag
        )

    # desired_job 통합 적용
    df_clean = transform_desired_job_column(df_clean, desired_job_map, col="desired_job", etc_token=ETC_TOKEN)

    # major_field 통합 적용
    df_clean = transform_major_field_column(df_clean, col="major_field")

    return df_clean


# =========================
# 1.5) 멀티셀렉트 처리 (기존 유지)
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
        return train_df, valid_df, test_df

    train_out = train_df.copy()
    valid_out = valid_df.copy()
    test_out = test_df.copy()

    for actual_col in resolved_cols:
        std_col = col_map.get(actual_col, actual_col)

        tr_tokens = train_out[actual_col].map(lambda x: parse_multiselect_cell(x, std_col)).tolist()
        va_tokens = valid_out[actual_col].map(lambda x: parse_multiselect_cell(x, std_col)).tolist() if actual_col in valid_out.columns else [[] for _ in range(len(valid_out))]
        te_tokens = test_out[actual_col].map(lambda x: parse_multiselect_cell(x, std_col)).tolist() if actual_col in test_out.columns else [[] for _ in range(len(test_out))]

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

        train_out = pd.concat([train_out.drop(columns=[actual_col]), pd.DataFrame(X_tr, columns=new_cols, index=train_out.index)], axis=1)
        valid_out = pd.concat([valid_out.drop(columns=[actual_col]), pd.DataFrame(X_va, columns=new_cols, index=valid_out.index)], axis=1)
        test_out  = pd.concat([test_out.drop(columns=[actual_col]),  pd.DataFrame(X_te, columns=new_cols, index=test_out.index)], axis=1)

    return train_out, valid_out, test_out


# =========================
# 2) 실행
# =========================
train_path = r"..\datasets\train.csv"
test_path  = r"..\datasets\test.csv"

train_df = pd.read_csv(train_path, encoding="utf-8-sig")
test_df  = pd.read_csv(test_path,  encoding="utf-8-sig")

y_raw = train_df["completed"]
train_part, valid_part = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=y_raw
)

cols_to_drop, median_semester, binary_fallbacks, desired_job_map, desired_job_fit_stats = fit_preprocess_params(train_part, threshold=0.5)

# raw(통합 전) 보존: 테스트 비교용
train_part_raw_for_test = train_part.copy()

train_clean = transform_preprocess(train_part, cols_to_drop, median_semester, binary_fallbacks, desired_job_map, tag="TRAIN_SPLIT", show_na_top=False)
valid_clean = transform_preprocess(valid_part, cols_to_drop, median_semester, binary_fallbacks, desired_job_map, tag="VALID_SPLIT", show_na_top=False)
test_clean  = transform_preprocess(test_df,     cols_to_drop, median_semester, binary_fallbacks, desired_job_map, tag="TEST",        show_na_top=False)


# =========================
# (TEST) desired_job / major_field 통합 검증
# =========================
debug_print_desired_job_checks(
    train_raw=train_part_raw_for_test,
    train_clean=train_clean,
    desired_job_map=desired_job_map,
    fit_stats=desired_job_fit_stats,
    col="desired_job",
    expected_n_categories=11,
    max_examples=20
)

debug_print_major_field_checks(
    train_raw=train_part_raw_for_test,
    train_clean=train_clean,
    col="major_field",
    max_examples=20
)


# =========================
# 기존 인코딩 파이프라인(명목형 OHE + 나머지 멀티셀렉트)
# =========================
nominal_params = fit_nominal_ohe_params(train_clean, NOMINAL_12_PLUS_CLASS1, RARE_EQ_1_COLS, etc_token=ETC_TOKEN)
train_nom = transform_nominal_ohe(train_clean, NOMINAL_12_PLUS_CLASS1, nominal_params, etc_token=ETC_TOKEN)
valid_nom = transform_nominal_ohe(valid_clean, NOMINAL_12_PLUS_CLASS1, nominal_params, etc_token=ETC_TOKEN)
test_nom  = transform_nominal_ohe(test_clean,  NOMINAL_12_PLUS_CLASS1, nominal_params, etc_token=ETC_TOKEN)

train_enc, valid_enc, test_enc = fit_transform_multiselect_A(
    train_nom, valid_nom, test_nom,
    multi_cols=MULTI_SELECT_COLS,
    show_summary=False
)


# =========================
# (NEW) desired_job / major_field 원-핫 인코딩(통합 완료된 컬럼을 대상으로)
#  - desired_job: 통합 후 11개(= A~J 대표라벨 + 기타) "고정"
#  - major_field: 통합 후 5개(= 고정 리스트)
# =========================
MAJOR_FIELD_CLASSES = ["이공계", "인문계", "예체능", "의약학", "Unknown"]

rep_by_alpha = desired_job_fit_stats.get("rep_by_alpha", {}) if isinstance(desired_job_fit_stats, dict) else {}

missing_alphas = [a for a in ALLOWED_JOB_ALPHAS if a not in rep_by_alpha]
if len(missing_alphas) > 0:
    raise ValueError(f"desired_job rep_by_alpha missing alphas: {missing_alphas}. (train split에 해당 알파벳이 없거나 파싱이 깨짐)")

DESIRED_JOB_CLASSES = [rep_by_alpha[a] for a in ALLOWED_JOB_ALPHAS] + [ETC_TOKEN]
if len(DESIRED_JOB_CLASSES) != 11:
    raise ValueError(f"desired_job fixed classes should be 11, got {len(DESIRED_JOB_CLASSES)}: {DESIRED_JOB_CLASSES}")

# (선택) major_field 라벨 체크(원하면 assert로 바꿔도 됨)
if "major_field" in train_enc.columns:
    mf_counts = _compute_multilabel_counts(train_enc["major_field"], _safe_parse_tokens_from_comma_cell)
    extra_mf = sorted(list(set(mf_counts.index.tolist()) - set(MAJOR_FIELD_CLASSES)))
    if len(extra_mf) > 0:
        print("[WARN] major_field has unexpected labels:", extra_mf)

# OHE 적용
train_final, valid_final, test_final = fit_transform_multilabel_ohe_fixed(
    train_enc, valid_enc, test_enc,
    col="desired_job",
    parser_fn=_safe_parse_tokens_from_comma_cell,
    classes=DESIRED_JOB_CLASSES,
    expected_n=11
)

train_final, valid_final, test_final = fit_transform_multilabel_ohe_fixed(
    train_final, valid_final, test_final,
    col="major_field",
    parser_fn=_safe_parse_tokens_from_comma_cell,
    classes=MAJOR_FIELD_CLASSES,
    expected_n=5
)

# (TEST) 원-핫 컬럼 검증
assert_ohe_columns_exist(train_final, "desired_job", DESIRED_JOB_CLASSES)
assert_ohe_columns_exist(train_final, "major_field", MAJOR_FIELD_CLASSES)
print("[OHE-TEST] desired_job one-hot =", len(DESIRED_JOB_CLASSES))
print("[OHE-TEST] major_field one-hot =", len(MAJOR_FIELD_CLASSES))


# =========================
# 최종 학습/검증/테스트 행렬 구성
# =========================
y_train = train_final["completed"]
y_valid = valid_final["completed"]

X_train = train_final.drop(columns=["completed"])
X_valid = valid_final.drop(columns=["completed"])

if "completed" in test_final.columns:
    test_final = test_final.drop(columns=["completed"])

common_cols = [c for c in X_train.columns if c in X_valid.columns and c in test_final.columns]
X_train = X_train[common_cols]
X_valid = X_valid[common_cols]
X_test  = test_final[common_cols]
