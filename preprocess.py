import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


# =============================================================================
# (A) 일반 명목형(단일 선택) 원-핫 인코딩 대상 컬럼 목록
# -----------------------------------------------------------------------------
# - NOMINAL_12: 단일 값(카테고리)인 컬럼들
# - class1도 같이 인코딩하고 싶어서 PLUS_CLASS1로 확장
# =============================================================================
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

# "기타"는 (1) 희귀 라벨 묶기, (2) 허용되지 않은 라벨이 들어왔을 때 fallback 용도로 사용
ETC_TOKEN = "기타"

# 컬럼별로 "train에서 딱 1번만 나온 라벨"을 기타로 묶을지 여부
# - 어떤 컬럼은 라벨이 너무 다양해서 1회 등장 라벨이 노이즈일 수 있음(희귀값)
RARE_EQ_1_COLS = {
    "inflow_route",
    "what_to_gain",
    "incumbents_lecture",
    "incumbents_company_level",
    "incumbents_lecture_scale",
    # class1은 10개라 희귀 통합 규칙 적용하지 않음
}


def _normalize_nominal_value(x):
    """
    명목형 값 정규화(데이터 입력이 사람 손으로 들어간 설문일 때 흔히 필요)
    - 앞뒤 공백 제거
    - 중간에 여러 칸 공백 -> 한 칸 공백으로 통일
    - NaN은 그대로 NaN 유지
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def fit_nominal_ohe_params(train_df: pd.DataFrame, cols, rare_eq_1_cols, etc_token: str = "기타"):
    """
    (fit 단계) 명목형 원-핫 인코딩 규칙을 train 데이터로부터 만든다.
    반환 params[col] = {
        "rare_set": {train에서 1번만 등장한 라벨 집합(선택 컬럼만)},
        "categories": [train 기준으로 확정된 카테고리 목록 + '기타']
    }

    왜 카테고리 목록을 train에서 확정하나?
    - valid/test에서 새로운 라벨이 나오면 차원이 흔들릴 수 있는데,
      여기서는 그 경우를 '기타'로 흡수해서 컬럼 구조를 고정하려고 함.
    """
    params = {}
    for col in cols:
        if col not in train_df.columns:
            continue

        # 1) 값 정규화
        s = train_df[col].map(_normalize_nominal_value)

        # 2) 희귀 라벨(= 1회 등장) 처리 대상 컬럼이면 rare_set을 만든다
        vc = s.value_counts(dropna=False)
        rare_set = set()
        if col in rare_eq_1_cols:
            rare_set = set(vc[vc == 1].index.tolist())
            rare_set = {v for v in rare_set if pd.notna(v)}  # NaN-safe

        # 3) rare_set에 해당하는 라벨은 '기타'로 통합한 버전(s2)으로 카테고리 목록 생성
        s2 = s.copy()
        if len(rare_set) > 0:
            s2 = s2.apply(lambda v: etc_token if v in rare_set else v)

        # 4) train에서 관측된 카테고리를 확정
        cats = sorted([c for c in s2.dropna().unique().tolist()])
        if etc_token not in cats:
            cats.append(etc_token)
            cats = sorted(cats)

        params[col] = {"rare_set": rare_set, "categories": cats}
    return params


def transform_nominal_ohe(df: pd.DataFrame, cols, params, etc_token: str = "기타"):
    """
    (transform 단계) fit에서 만든 params를 사용해 원-핫 인코딩을 수행한다.

    핵심 안전장치:
    - train에 없던 새로운 라벨이 valid/test에 나오면 -> '기타'로 강제 변환
    - params에 없는 컬럼은 drop (학습/추론에서 컬럼 불일치를 줄이기 위함)
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        if col not in params:
            # fit 시점에 컬럼이 없었거나 제외된 경우, 추후 혼란 방지를 위해 컬럼 제거
            out = out.drop(columns=[col])
            continue

        rare_set = params[col]["rare_set"]
        categories = params[col]["categories"]
        allowed = set(categories)

        # 1) 값 정규화
        s = out[col].map(_normalize_nominal_value)

        # 2) 희귀 라벨은 기타로
        if len(rare_set) > 0:
            s = s.apply(lambda v: etc_token if v in rare_set else v)

        # 3) train에서 허용되지 않은 라벨은 기타로(= valid/test에서 새 라벨 방지)
        s = s.apply(lambda v: etc_token if (pd.notna(v) and v not in allowed) else v)

        # 4) 카테고리를 고정한 상태에서 get_dummies 수행(컬럼 구조 고정)
        s_cat = pd.Categorical(s, categories=categories)
        dummies = pd.get_dummies(s_cat, prefix=f"{col}", prefix_sep="__", dtype=int)

        # 5) 원래 컬럼을 제거하고, 원-핫 컬럼을 붙임
        out = pd.concat([out.drop(columns=[col]), dummies.set_index(out.index)], axis=1)
    return out


# =============================================================================
# 이진 인코딩(0/1) 안전장치 유틸
# -----------------------------------------------------------------------------
# 목적:
# - 설문 데이터는 값이 비어있거나 예상치 못한 표현이 섞일 수 있음
# - 이때 NaN이 남아 모델이 깨지는 것을 막기 위해 "대체값(fallback)"을 준비
# =============================================================================
def _safe_mode_int(series: pd.Series, default: int = 0) -> int:
    """
    숫자로 바꿀 수 있는 값만 남기고(mode=최빈값)를 반환한다.
    - 데이터가 전부 NaN이면 default 반환
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    if len(s) == 0:
        return int(default)
    return int(s.mode().iloc[0])


def _encode_major_data_to_int(series: pd.Series) -> pd.Series:
    """
    major_data 컬럼을 0/1로 변환.
    - bool이면 바로 int 변환
    - 문자열이면 TRUE/FALSE로 매핑
    """
    if series.dtype == bool:
        return series.astype(int)
    s = series.astype(str).str.strip().str.upper()
    major_data_map = {"FALSE": 0, "TRUE": 1}
    return s.map(major_data_map)


def _binary_encode_with_fallback(df: pd.DataFrame, col: str, mapped: pd.Series, fallback_int: int, tag: str):
    """
    이진 인코딩 적용 + 결측/이상값을 fallback 값으로 채움.
    - tag는 원래는 로그 찍을 때 쓰려던 흔적(현재는 사용하지 않음)
    """
    if col not in df.columns:
        return
    df[col] = mapped.fillna(fallback_int).astype(int)


# =============================================================================
# desired_job 통합 로직 + 테스트 유틸
# -----------------------------------------------------------------------------
# 문제 상황(왜 이 로직이 필요한가):
# - desired_job은 "멀티 선택" + 입력 형태가 제각각(콤마, 줄바꿈, 슬래시 등)일 수 있음
# - 라벨이 "A. ~", "B. ~" 처럼 코드/그룹이 있는 구조인데, 사람 입력이라 변형이 많음
#
# 목표:
# 1) A~J로 시작하지 않는 라벨은 모두 '기타'로 흡수(정해진 범위 밖은 통제)
# 2) 같은 알파벳 그룹(A/B/...)은 train에서 가장 많이 나온 대표 라벨(최빈 라벨)로 통합
# 3) "G. PM, 서비스 기획자" 같이 콤마 뒤 설명이 붙어도 "G. PM"만 남게 정규화(라벨 폭발 방지)
# =============================================================================
ALLOWED_JOB_ALPHAS = list("ABCDEFGHIJ")
ALLOWED_JOB_ALPHA_SET = set(ALLOWED_JOB_ALPHAS)


def _normalize_separators_job(s: str) -> str:
    """
    desired_job 셀의 구분자 통일:
    - |, /, ;, 줄바꿈 등을 모두 콤마로 통일
    - 연속 콤마를 정리
    """
    for sep in ["|", "/", ";", "\n"]:
        s = s.replace(sep, ",")
    s = re.sub(r",\s*,+", ",", s)
    return s.strip()


def _split_by_code_prefix_job(text: str):
    """
    A-J. 로 시작하는 패턴을 기준으로 토큰을 쪼갠다.
    - 예: "A. 데이터분석, B. 개발" -> ["A. 데이터분석", "B. 개발"]
    - A~J가 전혀 없으면 통째로 하나의 토큰으로 취급(나중에 기타로 가거나 fallback 처리됨)
    """
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
    """
    라벨 표준화:
    - 공백 정리
    - 마지막 콤마 제거
    - (핵심) "G. PM, 서비스 기획자" -> "G. PM" (콤마 뒤 설명 제거)
    """
    s = str(label).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*,\s*$", "", s).strip()

    # (FIX 핵심) 코드로 시작하는 라벨은 첫 콤마 앞까지만 라벨로 인정
    # ex) "G. PM, 서비스 기획자" -> "G. PM"
    if re.match(r"^[A-Z]\.\s*", s):
        s = s.split(",")[0].strip()

    return s


def parse_desired_job_cell(value):
    """
    desired_job의 한 셀을 "라벨 리스트"로 변환하는 파서(parser).
    - 입력이 NaN/빈 문자열이면 빈 리스트
    - 구분자를 통일하고,
    - A~J 코드 기준으로 쪼개고,
    - 중복 라벨은 제거
    """
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
    """
    라벨에서 맨 앞 알파벳 코드(A/B/C...)만 뽑아낸다.
    - 예: "G. PM" -> "G"
    - 코드가 없으면 None
    """
    m = re.match(r"^([A-Z])\.\s*", str(label).strip())
    return m.group(1) if m else None


def _compute_multilabel_counts(series: pd.Series, parser_fn):
    """
    멀티라벨 컬럼의 라벨별 등장 샘플 수를 센다.
    - 한 샘플 안에서 라벨이 중복되어도 1번만 카운트(= 존재 여부만 체크)
    """
    counts = {}
    for v in series.tolist():
        labels = parser_fn(v)
        for lab in set(labels):  # 한 샘플 안에서는 중복 라벨 1회만 카운트
            counts[lab] = counts.get(lab, 0) + 1
    return pd.Series(counts).sort_values(ascending=False)


def fit_desired_job_collapse_map(train_df: pd.DataFrame, col: str = "desired_job", etc_token: str = "기타"):
    """
    (fit 단계) desired_job 라벨을 "통합(collapse)하기 위한 매핑표(label_map)"를 만든다.

    통합 규칙:
    - A~J 코드가 없는 라벨: 전부 기타
    - A~J 밖 코드 라벨: 전부 기타
    - A~J 코드 라벨: 같은 알파벳 그룹 내에서 train에서 가장 많이 나온 대표 라벨로 통합

    반환:
    - label_map: 원래 라벨 -> 통합 라벨
    - stats: 디버깅/리포팅용 통계(상위 라벨, 알파벳별 대표 등)
    """
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

    # 알파벳별 대표 라벨(= 최빈)
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

    # 최종 매핑 생성
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
    """
    (transform 단계) desired_job 컬럼을 "통합된 라벨 문자열"로 변환한다.
    - 한 셀 안에 여러 라벨이 있으면 통합 후 중복 제거하여 "A대표, B대표, 기타" 형태로 합침
    - 결과는 이후 멀티라벨 원-핫의 입력으로 사용됨(콤마로 join된 문자열)
    """
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
            # 안전장치: 맵에 없으면 기타로 보냄(원-핫 컬럼 고정이 목적)
            lab2 = label_map.get(lab, etc_token if _alpha_prefix(lab) is None else etc_token)
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
    """
    디버그/검증용 출력:
    - 통합 전/후 라벨 수와 분포가 기대대로인지 확인
    - A~J 대표 라벨이 잘 뽑혔는지 확인
    - 예시 몇 개를 뽑아 raw -> mapped가 어떻게 바뀌었는지 보여줌

    왜 필요한가?
    - 문자열 파싱/정규화는 "조용히 망가지는" 경우가 많음
      (예: 콤마 꼬리 때문에 라벨이 예상보다 늘어나거나, 범위 밖 코드가 섞이거나)
    """
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


# =============================================================================
# major_field 통합 로직 + 테스트 유틸
# -----------------------------------------------------------------------------
# 목표:
# - 전공 분야 라벨이 너무 세분화되어 있으면 모델이 희귀값에 취약해질 수 있음
# - 그래서 지정된 항목들을 큰 범주(이공계/인문계)로 묶고,
#   일부는 그대로 유지(예체능/의약학/Unknown)
# =============================================================================
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
    """major_field의 구분자(멀티선택 형태)를 콤마로 통일"""
    for sep in ["|", "/", ";", "\n"]:
        s = s.replace(sep, ",")
    s = re.sub(r",\s*,+", ",", s)
    return s.strip()


def _label_normalize_major(label: str) -> str:
    """major_field 라벨 공백/꼬리 콤마 제거 등 기본 정규화"""
    s = str(label).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*,\s*$", "", s).strip()
    return s


def parse_major_field_cell(value):
    """
    major_field의 한 셀을 라벨 리스트로 파싱.
    - 입력이 NaN/빈 문자열이면 빈 리스트
    - 콤마 기반으로 쪼개고 중복 제거
    """
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
    """
    major_field 라벨을 큰 범주로 매핑.
    - 지정 목록이면 이공계/인문계로 통합
    - keep 목록이면 그대로 유지
    - 그 외는 "요구사항 밖 라벨"이므로 그대로 두고(= 테스트에서 잡히게)
    """
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
    """
    major_field 컬럼에 통합 규칙을 적용한다.
    - 멀티라벨이므로, 라벨 리스트를 변환하고 "A, B" 형태 문자열로 다시 합친다.
    """
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
    """
    major_field 통합 검증:
    - BEFORE/AFTER 라벨 분포 출력
    - AFTER에 예상 밖 라벨이 남아있는지 확인(= 매핑 누락/오타/신규 라벨)
    """
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


# =============================================================================
# desired_job + major_field 원-핫(멀티라벨용) 유틸
# -----------------------------------------------------------------------------
# - 앞 단계에서 desired_job/major_field는 "A, B, 기타" 같은 콤마 문자열로 정리됨
# - 이를 MultiLabelBinarizer로 원-핫(멀티라벨) 인코딩한다.
# - classes를 고정하고 expected_n을 검사하여 컬럼 수가 흔들리지 않게 한다.
# =============================================================================
def _safe_parse_tokens_from_comma_cell(value):
    """
    "A, B, C" 형태(또는 NaN) -> ["A","B","C"]
    - 문자열을 콤마 기준으로 안전하게 분리
    - 중복 제거
    """
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
    """
    멀티라벨 원-핫 인코딩(컬럼 구조 고정 버전).

    왜 classes를 고정하나?
    - 멀티라벨은 데이터에 따라 라벨 집합이 바뀌기 쉬움
    - 학습/검증/테스트의 컬럼 수(차원)가 달라지면 모델이 입력을 못 받음
    - 그래서 classes를 "사전에 확정"하고, 그 외는 기타로 흡수하거나 앞 단계에서 통제해야 함
    """
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
    """
    모델 입력에서 "반드시 있어야 하는 원-핫 컬럼"이 다 존재하는지 검사.
    - 데이터나 전처리 변경으로 컬럼이 누락되면 여기서 즉시 에러로 잡아냄(조기 실패)
    """
    need = [f"{col}__{c}" for c in classes]
    missing = [c for c in need if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"[OHE-ASSERT] missing columns for {col}: {missing}")


# =============================================================================
# 0) 전처리 파라미터 fit (train split에서만)
# -----------------------------------------------------------------------------
# 여기서는 "데이터를 보고 결정해야 하는 전처리 규칙"만 계산한다.
# 예: 결측치 비율로 drop할 컬럼, 이상치 처리 후 중앙값, 이진 컬럼 fallback(최빈값), desired_job 매핑
# =============================================================================
def fit_preprocess_params(train_df: pd.DataFrame, threshold: float = 0.5):
    """
    반환값:
    - cols_to_drop: 결측치가 threshold 이상인 컬럼 목록(major1_2는 예외)
    - median_semester: completed_semester 중앙값(이상치 제거 후)
    - binary_fallbacks: 이진 인코딩에 사용할 최빈값 fallback
    - desired_job_map: desired_job 라벨 통합 매핑
    - desired_job_fit_stats: desired_job 통합 과정 통계(검증용)
    """
    # 1) 결측치 비율이 너무 큰 컬럼은 제거(정보가 거의 없다고 판단)
    missing_ratio = train_df.isnull().mean()
    cols_to_drop = [
        col for col in missing_ratio[missing_ratio >= threshold].index
        if col != "major1_2"
    ]

    # 2) drop 대상 제외 + 명시적으로 제거할 컬럼 제거
    tmp = train_df.drop(columns=cols_to_drop, errors="ignore")
    tmp = tmp.drop(columns=["nationality", "ID", "generation"], errors="ignore")

    # 3) completed_semester 이상치(코딩 실수로 보이는 값) 제거 후 중앙값 계산
    outlier_values = [20241.00, 2020.02]
    if "completed_semester" in tmp.columns:
        tmp.loc[tmp["completed_semester"].isin(outlier_values), "completed_semester"] = np.nan
        median_semester = tmp["completed_semester"].median()
    else:
        median_semester = 0

    # 4) 이진 변수들은 결측/이상값이 있을 수 있으므로 fallback(최빈값) 준비
    binary_fallbacks = {}

    # major type은 major1_2(복수전공 여부)로부터 파생되는 값
    if "major1_2" in train_df.columns:
        m2 = train_df["major1_2"].replace(r"^\s*$", np.nan, regex=True)
        has_major2_original = m2.notna()
    else:
        has_major2_original = pd.Series(False, index=train_df.index)

    major_type_int = np.where(has_major2_original, 1, 0)
    binary_fallbacks["major type"] = _safe_mode_int(pd.Series(major_type_int), default=0)

    # 재수강 여부(re_registration): "예/아니요" -> 1/0
    if "re_registration" in train_df.columns:
        s = train_df["re_registration"].astype(str).str.strip()
        mapped = s.map({"아니요": 0, "예": 1})
        binary_fallbacks["re_registration"] = _safe_mode_int(mapped, default=0)
    else:
        binary_fallbacks["re_registration"] = 0

    # 프로젝트 형태(project_type): "개인/팀" -> 0/1
    if "project_type" in train_df.columns:
        s = train_df["project_type"].astype(str).str.strip()
        mapped = s.map({"개인": 0, "팀": 1})
        binary_fallbacks["project_type"] = _safe_mode_int(mapped, default=0)
    else:
        binary_fallbacks["project_type"] = 0

    # major_data: TRUE/FALSE 또는 bool -> 1/0
    if "major_data" in train_df.columns:
        mapped = _encode_major_data_to_int(train_df["major_data"])
        binary_fallbacks["major_data"] = _safe_mode_int(mapped, default=0)
    else:
        binary_fallbacks["major_data"] = 0

    # incumbents_level: 주니어/시니어 -> 0/1
    if "incumbents_level" in train_df.columns:
        s = train_df["incumbents_level"].astype(str).str.strip()
        mapped = s.map({"주니어 (0~3년차)": 0, "시니어 (10년차 ~)": 1})
        binary_fallbacks["incumbents_level"] = _safe_mode_int(mapped, default=0)
    else:
        binary_fallbacks["incumbents_level"] = 0

    # desired_job 통합 맵은 train에서만 학습(대표 라벨 선정)
    desired_job_map, desired_job_fit_stats = fit_desired_job_collapse_map(train_df, col="desired_job", etc_token=ETC_TOKEN)

    return cols_to_drop, median_semester, binary_fallbacks, desired_job_map, desired_job_fit_stats


# =============================================================================
# 1) 전처리 transform (train/valid/test 공통 적용)
# -----------------------------------------------------------------------------
# fit에서 만든 규칙을 동일하게 적용한다.
# - 여기서의 목표는 "정리된 df_clean"을 만드는 것
# - 이후 인코딩(OHE/멀티라벨) 단계가 이어짐
# =============================================================================
def transform_preprocess(
    df: pd.DataFrame,
    cols_to_drop,
    median_semester: float,
    binary_fallbacks: dict,
    desired_job_map: dict,
    tag: str,
    show_na_top: bool = False
):
    """
    입력 df -> df_clean으로 변환한다.

    주요 작업:
    - 결측치가 많은 컬럼 제거 + 불필요 컬럼 제거
    - completed_semester 이상치 제거 후 median으로 채우기
    - major 관련 컬럼 정리(Unknown/없음 처리, major type 파생)
    - 여러 이진 변수들을 0/1로 안전하게 인코딩(fallback 포함)
    - desired_job / major_field 통합 적용
    """
    # 1) 결측치 비율 기반 drop
    df_clean = df.drop(columns=cols_to_drop, errors="ignore")

    # 2) 명시적으로 필요 없다고 판단되는 컬럼 제거(개인정보/식별자/세대 등)
    df_clean = df_clean.drop(columns=["nationality", "ID", "generation"], errors="ignore")

    # 3) 추가로 제거할 컬럼(프로젝트/모델링 목적에서 사용하지 않는 변수로 판단)
    df_clean = df_clean.drop(
        columns=["school1", "interested_company", "incumbents_lecture_scale_reason"],
        errors="ignore"
    )

    # 4) completed_semester: 이상치 제거 후 중앙값으로 채움(정수화)
    if "completed_semester" in df_clean.columns:
        outlier_values = [20241.00, 2020.02]
        df_clean.loc[df_clean["completed_semester"].isin(outlier_values), "completed_semester"] = np.nan
        df_clean["completed_semester"] = df_clean["completed_semester"].fillna(median_semester)
        df_clean["completed_semester"] = df_clean["completed_semester"].round().astype(int)

    # 5) major1_2가 빈 문자열이면 NaN으로 보고, "원래 존재했는지"를 기록해 major type 파생에 사용
    if "major1_2" in df_clean.columns:
        df_clean["major1_2"] = df_clean["major1_2"].replace(r"^\s*$", np.nan, regex=True)
        has_major2_original = df_clean["major1_2"].notna()
    else:
        has_major2_original = pd.Series(False, index=df_clean.index)

    # 6) major1_1 / major_field는 빈 값일 때 Unknown으로 채움(모델 입력에서 결측 제거)
    if "major1_1" in df_clean.columns:
        df_clean["major1_1"] = df_clean["major1_1"].replace(r"^\s*$", np.nan, regex=True).fillna("Unknown")

    if "major_field" in df_clean.columns:
        df_clean["major_field"] = df_clean["major_field"].replace(r"^\s*$", np.nan, regex=True).fillna("Unknown")

    # 7) major1_2는 결측이면 "없음"으로 채움(단일 전공의 의미)
    if "major1_2" in df_clean.columns:
        df_clean["major1_2"] = df_clean["major1_2"].fillna("없음")

    # 8) major type 파생: 복수전공 여부(원래 major1_2가 있었는지로 판단)
    df_clean["major type"] = np.where(
        has_major2_original,
        "복수 전공 ( 다중전공, 이중전공 포함 )",
        "단일 전공"
    )

    # 9) 파생/이진 컬럼들을 0/1로 변환(결측은 train에서 학습한 fallback 값으로 대체)
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

    # 10) desired_job / major_field는 문자열 파싱 후 "통합 규칙" 적용
    df_clean = transform_desired_job_column(df_clean, desired_job_map, col="desired_job", etc_token=ETC_TOKEN)
    df_clean = transform_major_field_column(df_clean, col="major_field")

    return df_clean


# =============================================================================
# 1.5) 멀티셀렉트(복수 선택) 처리
# -----------------------------------------------------------------------------
# MULTI_SELECT_COLS의 특징:
# - 한 셀에 "여러 값"이 들어있고 구분자가 복잡함(콤마/줄바꿈/괄호 등)
# - 희귀값은 모델에 노이즈가 될 수 있어 '기타'로 묶고,
# - train에서 등장한 라벨들로 allowed_set을 만들고 valid/test는 그 밖을 '기타'로 처리한다.
# =============================================================================
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

# "없음"을 하나의 라벨로 유지할지 여부(설문에서 '없음'도 중요한 정보일 수 있음)
KEEP_NONE_AS_LABEL = True
NONE_TOKENS = {"없음"}


def resolve_multiselect_columns(df: pd.DataFrame, wanted_cols):
    """
    데이터 컬럼명이 약간 다를 수 있는 상황 대응:
    - 'expected_domain' vs 'expecteddomain' 같이 underscore 유무가 다른 경우를 흡수
    - 실제 존재하는 컬럼명(resolved_cols)과 표준 컬럼명(std_col) 매핑(col_map)을 반환
    """
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
    """멀티셀렉트 문자열의 구분자 통일: | / ; 줄바꿈 등을 콤마로 통일"""
    for sep in ["|", "/", ";", "\n"]:
        s = s.replace(sep, ",")
    s = re.sub(r",\s*,+", ",", s)
    return s.strip()


def split_commas_outside_parentheses(text: str):
    """
    콤마로 분리하되, 괄호(...) 안에 있는 콤마는 분리 기준에서 제외.
    - 예: "A(1,2), B" -> ["A(1,2)", "B"]
    """
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
    """
    특정 코드 패턴(예: A-U.)을 기준으로 토큰을 나눔.
    - expected_domain / desired_job_except_data처럼 코드가 붙는 설문 문항에서 사용
    """
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
    """라벨 공백 정리 + 일부 컬럼은 꼬리 콤마 제거"""
    s = str(label).strip()
    s = re.sub(r"\s+", " ", s)
    if colname in {"desired_job_except_data", "expected_domain"}:
        s = re.sub(r"\s*,\s*$", "", s).strip()
    return s


def parse_multiselect_cell(value, colname: str):
    """
    멀티셀렉트 한 셀 -> 토큰 리스트로 파싱.

    컬럼별 파싱 전략이 다른 이유:
    - expected_domain / desired_job_except_data는 "A. ..." 같은 코드가 있어 코드 기준 split이 더 안전
    - 그 외는 괄호를 고려한 콤마 split이 더 안전
    """
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
    """
    train에서 등장 횟수가 threshold 이하인 라벨을 '기타'로 매핑하는 dict 생성.
    - 목표: 너무 희귀한 라벨로 인해 원-핫 차원이 과도하게 커지거나 과적합되는 것 방지
    """
    all_tokens = [t for lst in train_tokens_list for t in lst]
    vc = pd.Series(all_tokens).value_counts()
    rare_labels = set(vc[vc <= threshold].index.tolist())
    rare_labels.discard(ETC_TOKEN)
    return {lab: ETC_TOKEN for lab in rare_labels}


def apply_mapping_A(tokens, rare_map: dict, allowed_set: set):
    """
    라벨 리스트에 대해:
    - rare_map으로 희귀 라벨은 기타로 바꾸고
    - allowed_set(= train에서 허용된 라벨 집합) 밖이면 기타로 바꾸고
    - 중복은 제거
    """
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
    """
    (fit+transform) 멀티셀렉트 컬럼들을 원-핫으로 확장한다.

    흐름:
    1) train에서 토큰을 파싱해서 희귀 라벨 맵(rare_map) 생성(컬럼별 threshold 적용)
    2) train에서 등장한 라벨들로 allowed_set 구성(= 컬럼 구조의 기준)
    3) valid/test에는 allowed_set 밖 라벨이 나오면 기타로 흡수
    4) MultiLabelBinarizer로 원-핫 확장(컬럼명은 'col__label')
    """
    resolved_cols, col_map = resolve_multiselect_columns(train_df, multi_cols)
    if len(resolved_cols) == 0:
        return train_df, valid_df, test_df

    train_out = train_df.copy()
    valid_out = valid_df.copy()
    test_out = test_df.copy()

    for actual_col in resolved_cols:
        std_col = col_map.get(actual_col, actual_col)

        # 1) 파싱: 각 행을 토큰 리스트로 변환
        tr_tokens = train_out[actual_col].map(lambda x: parse_multiselect_cell(x, std_col)).tolist()
        va_tokens = valid_out[actual_col].map(lambda x: parse_multiselect_cell(x, std_col)).tolist() if actual_col in valid_out.columns else [[] for _ in range(len(valid_out))]
        te_tokens = test_out[actual_col].map(lambda x: parse_multiselect_cell(x, std_col)).tolist() if actual_col in test_out.columns else [[] for _ in range(len(test_out))]

        # 2) 희귀값 기타 처리 맵 생성(컬럼별로만 적용)
        threshold = RARE_TO_ETC_COLS.get(std_col, None)
        rare_map = build_rare_to_etc_map(tr_tokens, threshold) if threshold is not None else {}

        # 3) train에서 희귀 처리까지 반영한 토큰들을 기반으로 allowed_set 구성
        tr_tokens_mapped = [[rare_map.get(t, t) for t in lst] for lst in tr_tokens]
        allowed_set = set([t for lst in tr_tokens_mapped for t in lst])
        allowed_set.add(ETC_TOKEN)

        # 4) 최종 토큰 리스트 생성(allowed_set 밖은 기타)
        tr_final = [apply_mapping_A(lst, rare_map, allowed_set) for lst in tr_tokens]
        va_final = [apply_mapping_A(lst, rare_map, allowed_set) for lst in va_tokens]
        te_final = [apply_mapping_A(lst, rare_map, allowed_set) for lst in te_tokens]

        # 5) 원-핫 인코딩
        classes = sorted(list(allowed_set))
        mlb = MultiLabelBinarizer(classes=classes, sparse_output=False)
        X_tr = mlb.fit_transform(tr_final)
        X_va = mlb.transform(va_final)
        X_te = mlb.transform(te_final)

        new_cols = [f"{std_col}__{cls}" for cls in mlb.classes_]

        # 6) 원래 멀티셀렉트 컬럼 제거 후, 원-핫 컬럼 추가
        train_out = pd.concat([train_out.drop(columns=[actual_col]), pd.DataFrame(X_tr, columns=new_cols, index=train_out.index)], axis=1)
        valid_out = pd.concat([valid_out.drop(columns=[actual_col]), pd.DataFrame(X_va, columns=new_cols, index=valid_out.index)], axis=1)
        test_out  = pd.concat([test_out.drop(columns=[actual_col]),  pd.DataFrame(X_te, columns=new_cols, index=test_out.index)], axis=1)

    return train_out, valid_out, test_out


# =============================================================================
# 2) 실행(파이프라인 구간) - "실제로 무엇이 언제 실행되는지"가 한눈에 보이는 부분
# =============================================================================
train_path = r"..\datasets\train.csv"
test_path  = r"..\datasets\test.csv"

# (STEP 1) 데이터 로드
train_df = pd.read_csv(train_path, encoding="utf-8-sig")
test_df  = pd.read_csv(test_path,  encoding="utf-8-sig")

# (STEP 2) train/valid 분할
# - stratify=y_raw: 타깃(completed)의 비율을 train/valid에 비슷하게 맞추기 위한 옵션
y_raw = train_df["completed"]
train_part, valid_part = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=y_raw
)

# (STEP 3) fit 단계(훈련 데이터에서만 전처리 규칙을 학습)
cols_to_drop, median_semester, binary_fallbacks, desired_job_map, desired_job_fit_stats = fit_preprocess_params(train_part, threshold=0.5)

# raw(통합 전) 보존: 통합 로직이 제대로 작동했는지 비교/검증하기 위한 목적
train_part_raw_for_test = train_part.copy()

# (STEP 4) transform 단계(train/valid/test에 동일 전처리 적용)
train_clean = transform_preprocess(train_part, cols_to_drop, median_semester, binary_fallbacks, desired_job_map, tag="TRAIN_SPLIT", show_na_top=False)
valid_clean = transform_preprocess(valid_part, cols_to_drop, median_semester, binary_fallbacks, desired_job_map, tag="VALID_SPLIT", show_na_top=False)
test_clean  = transform_preprocess(test_df,     cols_to_drop, median_semester, binary_fallbacks, desired_job_map, tag="TEST",        show_na_top=False)

# =============================================================================
# (TEST) 문자열 통합 로직 검증(원인 추적이 어려운 영역이라 출력 검증이 중요)
# =============================================================================
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

# =============================================================================
# 기존 인코딩 파이프라인(명목형 OHE + 나머지 멀티셀렉트)
# -----------------------------------------------------------------------------
# - 명목형(단일 선택) 컬럼들은 get_dummies 기반 OHE
# - 멀티셀렉트 컬럼들은 MultiLabelBinarizer 기반 OHE
# =============================================================================
nominal_params = fit_nominal_ohe_params(train_clean, NOMINAL_12_PLUS_CLASS1, RARE_EQ_1_COLS, etc_token=ETC_TOKEN)
train_nom = transform_nominal_ohe(train_clean, NOMINAL_12_PLUS_CLASS1, nominal_params, etc_token=ETC_TOKEN)
valid_nom = transform_nominal_ohe(valid_clean, NOMINAL_12_PLUS_CLASS1, nominal_params, etc_token=ETC_TOKEN)
test_nom  = transform_nominal_ohe(test_clean,  NOMINAL_12_PLUS_CLASS1, nominal_params, etc_token=ETC_TOKEN)

train_enc, valid_enc, test_enc = fit_transform_multiselect_A(
    train_nom, valid_nom, test_nom,
    multi_cols=MULTI_SELECT_COLS,
    show_summary=False
)

# =============================================================================
# (NEW) desired_job / major_field 멀티라벨 원-핫 인코딩(통합 완료된 컬럼 대상)
# -----------------------------------------------------------------------------
# - desired_job: 통합 후 "항상 11개"로 고정(A~J 대표라벨 + 기타)
# - major_field: 통합 후 "항상 5개"로 고정
#
# 이렇게 고정하는 이유:
# - 데이터 split이나 샘플 구성에 따라 라벨이 빠지면(예: train_part에 특정 알파벳이 안 나오면)
#   원-핫 컬럼 수가 변할 수 있는데, 그걸 방지하기 위한 강제 체크
# =============================================================================
MAJOR_FIELD_CLASSES = ["이공계", "인문계", "예체능", "의약학", "Unknown"]

rep_by_alpha = desired_job_fit_stats.get("rep_by_alpha", {}) if isinstance(desired_job_fit_stats, dict) else {}

# train split에 A~J가 다 등장하지 않으면, "고정 11개" 설계가 깨지므로 즉시 에러로 알림
missing_alphas = [a for a in ALLOWED_JOB_ALPHAS if a not in rep_by_alpha]
if len(missing_alphas) > 0:
    raise ValueError(f"desired_job rep_by_alpha missing alphas: {missing_alphas}. (train split에 해당 알파벳이 없거나 파싱이 깨짐)")

DESIRED_JOB_CLASSES = [rep_by_alpha[a] for a in ALLOWED_JOB_ALPHAS] + [ETC_TOKEN]
if len(DESIRED_JOB_CLASSES) != 11:
    raise ValueError(f"desired_job fixed classes should be 11, got {len(DESIRED_JOB_CLASSES)}: {DESIRED_JOB_CLASSES}")

# (선택) major_field 라벨 체크: 통합 규칙 밖 라벨이 남았는지 경고
if "major_field" in train_enc.columns:
    mf_counts = _compute_multilabel_counts(train_enc["major_field"], _safe_parse_tokens_from_comma_cell)
    extra_mf = sorted(list(set(mf_counts.index.tolist()) - set(MAJOR_FIELD_CLASSES)))
    if len(extra_mf) > 0:
        print("[WARN] major_field has unexpected labels:", extra_mf)

# OHE 적용(멀티라벨)
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

# (TEST) 원-핫 컬럼 검증: 컬럼이 하나라도 빠지면 즉시 실패
assert_ohe_columns_exist(train_final, "desired_job", DESIRED_JOB_CLASSES)
assert_ohe_columns_exist(train_final, "major_field", MAJOR_FIELD_CLASSES)
print("[OHE-TEST] desired_job one-hot =", len(DESIRED_JOB_CLASSES))
print("[OHE-TEST] major_field one-hot =", len(MAJOR_FIELD_CLASSES))

# =============================================================================
# 최종 학습/검증/테스트 행렬 구성
# -----------------------------------------------------------------------------
# - y: 타깃(completed)
# - X: 입력 피처(전처리/인코딩 완료된 컬럼들)
# - common_cols: train/valid/test에 공통으로 존재하는 컬럼만 남겨 차원 불일치 방지
# =============================================================================
y_train = train_final["completed"]
y_valid = valid_final["completed"]

X_train = train_final.drop(columns=["completed"])
X_valid = valid_final.drop(columns=["completed"])

# test에는 completed가 없거나(대회 데이터), 있어도 학습 입력에서는 제거
if "completed" in test_final.columns:
    test_final = test_final.drop(columns=["completed"])

# train/valid/test의 공통 컬럼만 유지(안전한 입력 행렬 구성)
common_cols = [c for c in X_train.columns if c in X_valid.columns and c in test_final.columns]
X_train = X_train[common_cols]
X_valid = X_valid[common_cols]
X_test  = test_final[common_cols]
