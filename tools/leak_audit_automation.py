#!/usr/bin/env python
"""
T4: leak audit automation
=========================

SKB / TYB padock_idx 級 leak 事故 永久防止 framework。
学習前に features list を audit、 POST-RACE / LIVE-only datatype を機械的に detect し、
高 corr_target feature には audit flag を立てる。

Usage:
    # V15 cache 全 features を audit (read-only)
    python tools/leak_audit_automation.py --v15-cache

    # 任意 features list (1 行 1 feature) を audit
    python tools/leak_audit_automation.py --features path/to/features.txt

    # v15.2 候補 17 features 事前 audit
    python tools/leak_audit_automation.py --candidates path/to/features_v152.txt

Exit codes:
    0 = PASS (leak 0 件、 高 corr suspect 0 件)
    1 = FAIL (leak / suspect 検出)

★ V15 production 完全不変 ★ — read-only analysis のみ、 .pkl.gz / predict_core / app.py に一切 触れない。

Authors: T4 sub-task (2026-05-17)
References:
    - docs/SUBTASK_18_V152_FE_COMPLETE_DESIGN_2026_05_16.md (Sub-task 18)
    - docs/TYB_LEAK_AUDIT_2026_05_16.md (Sub-task 5-1 / 6 / 11)
    - CLAUDE.md Session #38 (SKB POST-RACE LEAK)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V15_CACHE_PATH = os.path.join(BASE_DIR, "data", "_v15_optuna_df_cache.pkl.gz")

# -------------------------------------------------------------------
# 1. JRDB 29 datatype release timing mapping
#    Sub-task 5-1 / 11 / 18 結果を完全反映
# -------------------------------------------------------------------
# safety:
#   'safe'        = 学習・live 共に投入可
#   'live_only'   = live 直前 fetch のみ可 (学習 retro は OK)
#   'prev_only'   = current race は LEAK、 前走 reference のみ可
#   'leak'        = POST-RACE、 学習禁止 (V15.1 SKB / TYB padock_idx 等)

JRDB_DATATYPE_RELEASE: dict[str, dict] = {
    # ----- SAFE: 学習・live 共に投入可 -----
    "kyi":   {"timing": "morning_06",        "safety": "safe",      "desc": "当日朝 06:00 確定 (基本 race meta + 馬指数)"},
    "bac":   {"timing": "previous_thursday", "safety": "safe",      "desc": "前週木曜 (番組公表)"},
    "cha":   {"timing": "minus_3_days",      "safety": "safe",      "desc": "中央木曜追切後 3-4 日前 確定"},
    "kab":   {"timing": "morning_06",        "safety": "safe",      "desc": "当日朝 06:00 (kaisai-level)"},
    "paci":  {"timing": "morning_06",        "safety": "safe",      "desc": "前日夜更新 + 当日朝 06:00 sync"},
    "kta":   {"timing": "morning_06",        "safety": "safe",      "desc": "場別調教師、 当日朝 06:00"},
    "jo":    {"timing": "morning_06",        "safety": "safe",      "desc": "JO ファイル (CID/LS 指数)"},
    "joa":   {"timing": "previous_thursday", "safety": "safe",      "desc": "race meta v2"},

    # ----- master (静的、 年次更新) -----
    "ukc":   {"timing": "static_master", "safety": "safe", "desc": "sire/bms/breeder master"},
    "ksa":   {"timing": "static_master", "safety": "safe", "desc": "騎手 master"},
    "kz":    {"timing": "static_master", "safety": "safe", "desc": "騎手 v2 master"},
    "csa":   {"timing": "static_master", "safety": "safe", "desc": "調教師 master"},
    "cz":    {"timing": "static_master", "safety": "safe", "desc": "調教師 v2 master"},
    "kkb":   {"timing": "weekly",        "safety": "safe", "desc": "週次集計"},
    "kza":   {"timing": "weekly",        "safety": "safe", "desc": "週次集計"},

    # ----- live_only: 直前 -15 min (朝予測には不可) -----
    "cyb":     {"timing": "live_minus_15",  "safety": "live_only", "desc": "直前 (CYB 取消等)"},
    "cyb_v2":  {"timing": "live_minus_15",  "safety": "live_only", "desc": "直前 v2"},

    # ----- prev_only: current race は LEAK、 前走 reference のみ OK -----
    # 当日 publish される race-level 集計だが publish タイミングが race 後
    "sed":   {"timing": "post_race",         "safety": "prev_only", "desc": "成績、 前走集計に利用"},
    "sr":    {"timing": "post_race",         "safety": "prev_only", "desc": "ラップ、 前走集計に利用"},
    "srb":   {"timing": "post_race",         "safety": "prev_only", "desc": "track bias、 前走集計に利用"},
    "ze":    {"timing": "post_race",         "safety": "prev_only", "desc": "過去 5 走集計に利用"},
    "zk":    {"timing": "post_race",         "safety": "prev_only", "desc": "前走着差等"},
    "kka":   {"timing": "post_race",         "safety": "prev_only", "desc": "旧 schema、 集計値"},
    "kka_v2":     {"timing": "post_race", "safety": "prev_only", "desc": "breeder 集計 (生涯)、 static として利用可"},
    "kka_features":{"timing": "post_race","safety": "prev_only", "desc": "未確認、 後発"},
    "kaa":   {"timing": "post_race",         "safety": "prev_only", "desc": "kaisai 旧 schema"},

    # ----- LEAK: POST-RACE 完全除外 -----
    "skb":  {"timing": "post_race",        "safety": "leak", "desc": "★ POST-RACE LEAK 確定 (V15.1 SKB、 Session #38)"},
    "tyb":  {"timing": "post_race_17:00",  "safety": "leak", "desc": "★ TYB ZIP 17:00 JST publish (Sub-task 11 確定)"},
    "hjc":  {"timing": "post_race",        "safety": "leak", "desc": "★ 払戻系"},
    "oz":   {"timing": "post_race",        "safety": "leak", "desc": "★ 配当系"},
    "ou":   {"timing": "post_race",        "safety": "leak", "desc": "★ 賠率系 OU"},
    "ot":   {"timing": "post_race",        "safety": "leak", "desc": "★ 賠率系 OT"},
    "ov":   {"timing": "post_race",        "safety": "leak", "desc": "★ 賠率系 OV"},
    "ow":   {"timing": "post_race",        "safety": "leak", "desc": "★ 賠率系 OW"},
}

# -------------------------------------------------------------------
# 2. feature prefix → datatype マッピング
# -------------------------------------------------------------------
# V15 cache 名規則の経験則:
#   - jrdb_<datatype>_<col>     : 直接 prefix
#   - jrdb_<col>                : kyi 由来の単独 (idm / class_code 等)
#   - paci_*                    : paci 由来
#   - oz_*                      : oz (配当系) 由来
#   - prev_*                    : 前走 (sed / ze 等) 由来 → prev_only 扱い
#   - その他 (weight_carry / age / ... ): non-jrdb の自前 FE

KYI_INHERITED_NAMES = {
    # jrdb_<name> で 直接 kyi 由来と判定する名 (Sub-task 18 表 §3-1 / §1 より)
    "idm", "training_idx", "stable_idx", "composite_idx", "upset_idx",
    "ten_idx_pred", "pace_idx_pred", "agari_idx_pred", "position_idx_pred",
    "class_code", "rise_code", "heavy_apt", "hoof_code", "ranch_rank",
    "stable_rank", "training_arrow", "stable_eval", "running_style",
    "dist_apt",
}

# jrdb_<name> の name → datatype 上書き map (Sub-task 18 §1 表より)
JRDB_NAME_OVERRIDES: dict[str, str] = {
    # cha (追切) 直接 prefix なしで取り込まれた既存名
    "oikiri_idx": "cha",
    "ten_time_idx": "cha",
    "shimai_time_idx": "cha",
    # jo (CID/LS 指数)
    "cid_idx": "jo",
    "ls_idx": "jo",
}

# oz_* prefix は ★ JRDB oz (払戻系) ではなく ★ V15 自前 FE (build_odds_base_retro.py)
# = jrdb_kyi 基準オッズ (前日朝オッズ) 派生。 release timing = morning_06 で safe。
# → datatype = "kyi" 扱い、 「oz」 JRDB datatype とは無関係
OZ_FE_NAMES = {
    "oz_tansho_base_log", "oz_fukusho_base_log", "oz_base_pop_rank",
}

# 自前 FE prefix (non-jrdb)、 LEAK 検査対象外 (datatype = "fe_internal")
NON_JRDB_PREFIXES = {
    "weight_carry", "age", "distance", "course_enc", "surface_enc", "sex_enc",
    "num_horses_val", "horse_num", "bracket", "jockey_wr_calc", "jockey_course_wr_calc",
    "trainer_top3_calc", "avg_finish_3r", "best_finish_3r", "finish_trend",
    "top3_count_3r", "avg_last3f_3r", "dist_change", "dist_change_abs",
    "rest_days", "rest_category", "sire_enc", "bms_enc", "dist_cat", "age_sex",
    "season", "age_season", "horse_num_ratio", "bracket_pos", "carry_diff",
    "age_group", "surface_dist_enc", "course_surface", "location_enc", "is_nar",
    "training_time_filled", "has_training", "training_per_dist",
    "jockey_surface_wr", "horse_career_races", "horse_career_wr",
    "horse_career_top3r", "sire_surface_wr", "sire_dist_wr", "bms_surface_wr",
    "wood_best_4f_filled", "has_wood_training",
    "prev_agari_relative", "wood_count_2w", "sakaro_best_4f_filled",
    "sakaro_best_3f_filled", "has_sakaro_training", "total_training_count",
    "horse_dist_top3r", "horse_surface_top3r", "frame_course_dist_wr",
    "index_max_filled", "index_run1_filled", "index_avg5_filled",
    "time_1f_last_filled", "training_intensity_enc",
    "sire_shinba_top3r", "pci",
    "stable_comment_score",
    "odds_change_rate", "pop_rank_change", "odds_sharp_drop",
    "weight_ma5", "weight_trend", "weight_peak_diff",
    "jockey_horse_rides", "jockey_horse_wr", "jockey_horse_top3r",
    "jockey_change", "jockey_change_to_top",
    "transport_distance_km", "is_long_transport",
    "course_renovated", "post_renovation_flag", "gaisha_rank",
    # prev_* race feature engineered from sed
    "prev_race_first3f", "prev_race_last3f", "prev_race_pace_diff",
    # prev_<finish/last3f/pass4/prize/odds_log/...>: sed 由来の前走 ref → prev_only
}

# 高 corr threshold
HIGH_CORR_THRESHOLD = 0.40

# 既知の safe high-corr (kyi/paci 由来で release_timing 安全と既に確認済)
# 注: corr が高くても release_timing が 'safe' なら audit-OK
# (paci_jockey_*, paci_ninki_*, jrdb_idm 等)
KNOWN_SAFE_HIGH_CORR = {
    # paci 系 high-corr は jockey/odds-based pre-race indicator (V15 採用済)
    "paci_jockey_exp_wr", "paci_jockey_exp_3rd", "paci_ninki_idx",
    "paci_jockey_mark", "paci_sogo_mark", "paci_idm_mark",
    "paci_train_mark", "paci_manken_idx",
    # jrdb_training_idx, jrdb_stable_idx: kyi 由来 morning_06
    "jrdb_training_idx", "jrdb_stable_idx", "jrdb_idm",
    "jrdb_composite_idx", "jrdb_cid_idx",
    # oz_*: pre-race odds (snapshot 経由)、 V15 採用済 (prev_odds_log と類似)
    "oz_tansho_base_log", "oz_fukusho_base_log", "oz_base_pop_rank",
}

# v15.2 候補の事前 corr_target (Sub-task 18 §3 実測値、 ndb_sed.csv finish<=3)
# 候補が V15 cache に存在しない場合に audit 用に使用
EXTERNAL_CORR_PRIOR: dict[str, float] = {
    # paci 由来
    "paci_gekiso_idx": +0.200,
    "paci_ls_idx_rank": -0.275,
    "paci_gekiso_rank": -0.255,
    "paci_info_idx": +0.4139,    # ★ corr +0.41 = leak suspect、 audit 必須 (Sub-task 18 §3-2 注記)
    "paci_gekiso_race_rank": +0.20,   # 派生 (rank)、 base 同等
    "paci_lsidx_race_rank": -0.275,
    # cha 由来
    "cha_chukan_time_idx": +0.030,
    "cha_chukan_idx_race_zscore": +0.030,
    "cha_oikiri_idx_trend": +0.067,   # base oikiri_idx
    "cha_shimai_time_3r_mean": -0.083,
    "cha_oikiri_rank": +0.011,
    "cha_awase_result": -0.058,
    # kta 由来
    "kta_ichi_idx_pred": +0.1446,
    "kta_ichi_pred_race_rank": +0.1446,
    "kta_pace_idx_pred": +0.095,
    # kka_v2 breeder
    "breeder_dist_1": +0.1429,
    "breeder_dist_1_race_rank": +0.1429,
    "breeder_track_1": +0.105,
    # kab interaction
    "kab_turf_baba_x_bracket": -0.004,
    "kab_straight_sa_x_horse_num_ratio": -0.002,
    "kab_renzoku_day": -0.003,
    # srb 前走
    "srb_bias_straight": +0.05,  # 推定 (Sub-task 18 §3-6)
}


# -------------------------------------------------------------------
# 3. 結果 dataclass
# -------------------------------------------------------------------
@dataclass
class FeatureAuditResult:
    feature: str
    datatype: str
    timing: str
    safety: str
    corr_target: Optional[float] = None
    n_valid: int = 0
    corr_flag: str = "unknown"        # ok / HIGH_CORR_LEAK_SUSPECT / unknown
    verdict: str = "UNKNOWN"          # OK / LEAK / LIVE_ONLY / PREV_ONLY / SUSPECT / UNKNOWN
    note: str = ""


def identify_datatype(name: str) -> tuple[str, str]:
    """feature name から (datatype, source_label) を判定。

    source_label は debug 用、 datatype は JRDB_DATATYPE_RELEASE の key、
    自前 FE は datatype = 'fe_internal'。
    """
    n = name.lower().strip()

    # 自前 FE
    if n in NON_JRDB_PREFIXES:
        return "fe_internal", "fe_internal"

    # paci_*
    if n.startswith("paci_"):
        return "paci", "paci_prefixed"

    # v15.2 候補命名規則 (cha_/kta_/kab_/srb_/breeder_/sr_):
    # → 元 datatype を prefix から推定
    if n.startswith("cha_"):
        return "cha", "cha_prefixed_v152"
    if n.startswith("kta_"):
        return "kta", "kta_prefixed_v152"
    if n.startswith("kab_"):
        return "kab", "kab_prefixed_v152"
    if n.startswith("srb_"):
        return "srb", "srb_prefixed_v152"
    if n.startswith("breeder_"):
        return "kka_v2", "breeder_kka_v2_v152"
    if n.startswith("sr_") and not n.startswith("sr_") in NON_JRDB_PREFIXES:
        return "sr", "sr_prefixed_v152"

    # oz_* は ★ JRDB oz (払戻) ではなく ★ jrdb_kyi 基準オッズ FE (build_odds_base_retro)
    # → kyi datatype 扱い、 morning_06 safe
    if n in OZ_FE_NAMES or n.startswith("oz_"):
        return "kyi", "oz_fe_from_kyi_base_odds"

    # prev_* (sed/ze の前走 ref FE) → prev_only として明示
    if n.startswith("prev_") or n.startswith("prev2_") or n.startswith("prev3_"):
        return "sed", "prev_ref_sed"

    # jrdb_<datatype>_*  例: jrdb_kta_idm, jrdb_ze_idm_avg
    if n.startswith("jrdb_"):
        rest = n[len("jrdb_"):]
        # 既知の kyi 直系 (jrdb_idm, jrdb_class_code, jrdb_training_idx 等)
        if rest in KYI_INHERITED_NAMES:
            return "kyi", "jrdb_kyi_inherited"
        # name 上書き map (cha 由来 / jo 由来など)
        if rest in JRDB_NAME_OVERRIDES:
            dt = JRDB_NAME_OVERRIDES[rest]
            return dt, f"jrdb_{dt}_name_override"
        # jrdb_prev_*  → sed (前走 ref)
        if rest.startswith("prev_"):
            return "sed", "jrdb_prev_ref"
        # jrdb_<datatype>_*  pattern
        parts = rest.split("_", 1)
        prefix = parts[0]
        if prefix in JRDB_DATATYPE_RELEASE:
            return prefix, f"jrdb_{prefix}_prefixed"
        # tb_* (track-bias) → srb (前走 ref)
        if prefix == "tb":
            return "srb", "jrdb_tb_srb_prev_ref"
        # dam_*, bms_*, ranch_* → kka_v2 系 (静的 breeder)
        if prefix in ("dam", "bms", "ranch"):
            return "kka_v2", f"jrdb_{prefix}_kka_v2_static"
        # その他 jrdb_ で始まる未知 → unknown
        return "unknown_jrdb", "jrdb_unknown"

    # その他 = 自前 FE
    return "fe_internal", "fe_internal_implicit"


def audit_feature(name: str, cache_df: Optional[pd.DataFrame] = None,
                  target_col: str = "finish",
                  top_n_for_target: int = 3) -> FeatureAuditResult:
    """1 feature の leak audit。

    Args:
        name: feature 名
        cache_df: optional、 corr_target 計算用 DataFrame (target_col 含む)
        target_col: target カラム名 (default 'finish')
        top_n_for_target: target 閾値 (finish <= top_n)

    Returns:
        FeatureAuditResult
    """
    datatype, source = identify_datatype(name)
    info = JRDB_DATATYPE_RELEASE.get(
        datatype,
        {"timing": "n/a", "safety": "fe_internal" if datatype == "fe_internal" else "unknown",
         "desc": source}
    )
    timing = info["timing"]
    safety = info["safety"]

    result = FeatureAuditResult(
        feature=name,
        datatype=datatype,
        timing=timing,
        safety=safety,
        note=info.get("desc", ""),
    )

    # verdict (datatype-based)
    if safety == "leak":
        result.verdict = "LEAK"
    elif safety == "live_only":
        result.verdict = "LIVE_ONLY"
    elif safety == "prev_only":
        # name 自体が prev_/jrdb_prev_ で始まる → 前走 ref で安全
        n = name.lower()
        if n.startswith("prev_") or n.startswith("prev2_") or n.startswith("prev3_") or n.startswith("jrdb_prev_"):
            result.verdict = "OK"
            result.note = "前走 ref (sed/ze) → 安全"
        elif name.lower().startswith("jrdb_tb_"):
            result.verdict = "OK"
            result.note = "前走 srb track bias ref → 安全"
        elif name.lower().startswith("jrdb_ze_"):
            result.verdict = "OK"
            result.note = "過去 5 走集計 (ze) → 安全"
        elif "rensho" in name.lower() or datatype == "kka_v2":
            result.verdict = "OK"
            result.note = "kka_v2 静的 breeder 集計 → 安全"
        else:
            result.verdict = "PREV_ONLY"
            result.note = f"{datatype} = current race は LEAK、 前走 ref として要確認"
    elif safety in ("safe", "fe_internal"):
        result.verdict = "OK"
    else:
        result.verdict = "UNKNOWN"

    # External prior corr (Sub-task 18 §3 実測値、 cache に存在しない候補用)
    used_external = False
    if name in EXTERNAL_CORR_PRIOR and (cache_df is None or name not in cache_df.columns):
        result.corr_target = EXTERNAL_CORR_PRIOR[name]
        result.n_valid = 0  # 外部 prior、 実測 row 数なし
        used_external = True
        c = result.corr_target
        if abs(c) > HIGH_CORR_THRESHOLD:
            if name in KNOWN_SAFE_HIGH_CORR:
                result.corr_flag = "ok_known_safe"
            else:
                result.corr_flag = "HIGH_CORR_LEAK_SUSPECT"
                if result.verdict == "OK":
                    result.verdict = "SUSPECT"
        else:
            result.corr_flag = "ok_external_prior"
        result.note += f" [external prior corr={c:+.4f} (Sub-task 18 実測)]"

    # corr_target audit (cache 由来)
    if not used_external and cache_df is not None and name in cache_df.columns and target_col in cache_df.columns:
        try:
            x = pd.to_numeric(cache_df[name], errors="coerce")
            tgt = (pd.to_numeric(cache_df[target_col], errors="coerce") <= top_n_for_target).astype(int)
            mask = x.notna() & cache_df[target_col].notna()
            n_valid = int(mask.sum())
            result.n_valid = n_valid
            if n_valid > 100:
                xv = x[mask].astype(float).values
                tv = tgt[mask].astype(float).values
                if xv.std() > 0 and tv.std() > 0:
                    c = float(np.corrcoef(xv, tv)[0, 1])
                    result.corr_target = c
                    if abs(c) > HIGH_CORR_THRESHOLD:
                        if name in KNOWN_SAFE_HIGH_CORR:
                            result.corr_flag = "ok_known_safe"
                        else:
                            result.corr_flag = "HIGH_CORR_LEAK_SUSPECT"
                            # verdict を昇格 (OK → SUSPECT)、 LEAK はそのまま
                            if result.verdict == "OK":
                                result.verdict = "SUSPECT"
                    else:
                        result.corr_flag = "ok"
                else:
                    result.corr_flag = "ok"  # std=0 は constant、 別途検査
        except Exception as e:
            result.note += f" [corr error: {e}]"

    return result


def audit_features(features: list[str], cache_df: Optional[pd.DataFrame] = None,
                   target_col: str = "finish") -> list[FeatureAuditResult]:
    """features list 全件 audit。"""
    return [audit_feature(f, cache_df, target_col) for f in features]


# -------------------------------------------------------------------
# 4. CLI
# -------------------------------------------------------------------
def _load_v15_cache() -> tuple[pd.DataFrame, list[str]]:
    if not os.path.exists(V15_CACHE_PATH):
        raise FileNotFoundError(f"V15 cache not found: {V15_CACHE_PATH}")
    with gzip.open(V15_CACHE_PATH, "rb") as f:
        obj = pickle.load(f)
    return obj["df"], obj["features"]


def _read_features_file(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        feats = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    return feats


def summarize(results: list[FeatureAuditResult]) -> dict:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    return {
        "total": len(results),
        "by_verdict": counts,
        "leaks": [r.feature for r in results if r.verdict == "LEAK"],
        "live_only": [r.feature for r in results if r.verdict == "LIVE_ONLY"],
        "prev_only": [r.feature for r in results if r.verdict == "PREV_ONLY"],
        "suspects": [r.feature for r in results if r.corr_flag == "HIGH_CORR_LEAK_SUSPECT"],
    }


def main() -> int:
    # Force UTF-8 stdout on Windows cp932 console
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="T4 leak audit automation")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--v15-cache", action="store_true", help="V15 cache 全 145 features を audit")
    src.add_argument("--features", help="features list file (1 feature per line)")
    src.add_argument("--candidates", help="v15.2 候補 features list file")
    parser.add_argument("--output", help="JSON 出力 path (任意)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cache_df: Optional[pd.DataFrame] = None
    if args.v15_cache:
        cache_df, features = _load_v15_cache()
        label = "V15 cache (145 features)"
    else:
        path = args.features or args.candidates
        features = _read_features_file(path)
        label = f"file: {path}"
        # Cache を load して corr 取得を試みる
        try:
            cache_df, _ = _load_v15_cache()
        except FileNotFoundError:
            cache_df = None

    print(f"=== T4 leak audit ({label}) ===", flush=True)
    print(f"features count: {len(features)}", flush=True)
    results = audit_features(features, cache_df=cache_df)
    summary = summarize(results)

    # 詳細出力
    if args.verbose or len(features) <= 50:
        print()
        print(f"{'feature':<35} {'datatype':<12} {'safety':<10} {'corr':>8} {'flag':<25} verdict")
        print("-" * 110)
        for r in sorted(results, key=lambda x: (x.verdict != "LEAK",
                                                  x.corr_flag != "HIGH_CORR_LEAK_SUSPECT",
                                                  -(abs(x.corr_target) if x.corr_target is not None else 0))):
            corr_s = f"{r.corr_target:+.4f}" if r.corr_target is not None else "   nan"
            print(f"{r.feature:<35.35} {r.datatype:<12} {r.safety:<10} {corr_s:>8} {r.corr_flag:<25} {r.verdict}")

    print()
    print("=== summary ===")
    print(f"  total: {summary['total']}")
    for v, n in sorted(summary["by_verdict"].items()):
        print(f"  {v}: {n}")
    if summary["leaks"]:
        print(f"  ★ LEAK features ★: {summary['leaks']}")
    if summary["suspects"]:
        print(f"  ⚠ HIGH_CORR_LEAK_SUSPECT: {summary['suspects']}")

    if args.output:
        out = {
            "label": label,
            "summary": summary,
            "results": [asdict(r) for r in results],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        print(f"  JSON output: {args.output}")

    # exit code
    if summary["leaks"]:
        print("RESULT: FAIL (LEAK detected)")
        return 1
    if summary["suspects"]:
        print("RESULT: FAIL (HIGH_CORR_LEAK_SUSPECT detected)")
        return 1
    if summary["by_verdict"].get("UNKNOWN", 0) > 0:
        print("RESULT: WARN (UNKNOWN datatype, audit manually)")
        return 1
    print("RESULT: PASS (no leak / suspect)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
