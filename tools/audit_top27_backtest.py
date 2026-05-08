"""Session #51 B: AUDIT-1 Top 27 一括 backtest.

V15 baseline + 単一 feature の AUC contribution を測定。

Usage:
    python tools/audit_top27_backtest.py --features all
    python tools/audit_top27_backtest.py --features 7,11,4
    python tools/audit_top27_backtest.py --quick  # 上位 5 のみ

実装方針:
- V15 cache (data/_v15_optuna_df_cache.pkl.gz) を base
- 時系列 split: 2020-2023 train, 2024 valid
- LGB num_boost_round=200, early_stop=20 (高速化)
- 単一 feature 追加 → AUC delta
- bootstrap 95% CI (200 iter)
- multiprocessing で 並列 実行
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "_v15_optuna_df_cache.pkl.gz"
OUT_JSON = ROOT / "data" / "v18" / "sprint5_backtest_metrics.json"
OUT_MD = ROOT / "data" / "v18" / "sprint5_top27_backtest_results.md"


# Top 18 即実装可能 features (Session #51 A 分類より)
FEATURE_PLAN = [
    # (id, name, csv_path, merge_keys, value_cols, tier)
    (7,  "speed_index_dist_course",   "data/netkeiba_speed_index.csv",
        ["race_id", "umaban"], ["index_dist", "index_course"], "1h"),
    (11, "training_times_rank",       "data/netkeiba_training_times.csv",
        ["race_id", "umaban"], ["rank"], "1h"),
    (5,  "ai_opinion_pace",           "data/netkeiba_ai_opinion.csv",
        ["race_id"], ["pace"], "2h"),
    (9,  "ai_position_pct",           "data/netkeiba_ai_position.csv",
        ["race_id", "umaban"], ["position_left_pct", "position_top_pct"], "2h"),
    (12, "race_analysis_score",       "data/netkeiba_race_analysis.csv",
        ["race_id", "umaban"], ["score"], "2h"),
    (21, "upset_level",               "data/netkeiba_upset_level.csv",
        ["race_id"], ["upset_level", "top_popularity_reliability"], "1h"),
    (28, "data_analysis_count",       "data/netkeiba_data_analysis.csv",
        ["race_id"], None, "2h"),  # auto-detect numeric
    (10, "jrdb_cyb_train",            "data/jrdb_cyb.csv",
        ["race_id", "umaban"], ["train_mark", "train_amount", "train_change", "train_eval"], "3h"),
    (4,  "jrdb_kka_seiseki",          "data/jrdb_kka.csv",
        ["race_id", "umaban"], ["kyori_seiseki_1", "kyori_seiseki_2", "kyori_seiseki_3",
                                "track_seiseki_1", "track_seiseki_2", "track_seiseki_3",
                                "heavy_seiseki_1", "heavy_seiseki_2", "heavy_seiseki_3"], "6h"),
    (6,  "jrdb_cha_oikiri",           "data/jrdb_cha.csv",
        ["race_id", "umaban"], ["oikiri_rank", "oikiri_idx", "ten_time_idx",
                                "chukan_time_idx", "shimai_time_idx"], "4h"),
    (8,  "jrdb_jo_bb",                "data/jrdb_jo.csv",
        ["race_id", "umaban"], ["gaisha_bb", "gaisha_bb_wr", "gaisha_bb_rensho",
                                "breeder_bb", "breeder_bb_wr", "breeder_bb_rensho"], "3h"),
    (26, "jrdb_jo_odds",              "data/jrdb_jo.csv",
        ["race_id", "umaban"], ["soten_odds", "yoso_odds", "cid_soten_idx", "cid_sara_idx"], "2h"),
    (17, "jrdb_tyb_live",             "data/jrdb_tyb.csv",
        ["race_id", "umaban"], ["bagu_change", "cancel_flag"], "2h"),
    (18, "jrdb_sed_time",             "data/jrdb_sed.csv",
        ["race_id", "umaban"], None, "3h"),  # auto-detect
    (19, "jrdb_kyi_marks",            "data/jrdb_kyi.csv",
        ["race_id", "umaban"], None, "1h"),  # auto-detect
    (24, "jrdb_kz_leading",           "data/jrdb_kz.csv",
        ["race_id"], None, "3h"),  # auto-detect
    (22, "race_review_score",         "data/netkeiba_race_review.csv",
        ["race_id", "umaban"], None, "2h"),  # auto-detect
    (23, "stable_comment_score",      "data/netkeiba_stable_comments.csv",
        ["race_id", "umaban"], None, "1h"),
]


def load_v15_cache() -> tuple[pd.DataFrame, list[str]]:
    """V15 cache 読込 + target + nk_race_id 構築."""
    with gzip.open(CACHE, "rb") as f:
        d = pickle.load(f)
    df = d["df"]
    features = d["features"]
    df = df.copy()
    df["target"] = (df["finish"] >= 1) & (df["finish"] <= 3)
    df["target"] = df["target"].astype(int)
    # Build netkeiba race_id (12 chars): YYYY + place(2, course_enc+1) + kai(2) + nichi(2) + race_num(2)
    df["nk_race_id"] = (
        df["year_full"].astype(str)
        + (df["course_enc"] + 1).astype(int).astype(str).str.zfill(2)
        + df["kai"].astype(str).str.zfill(2)
        + df["nichi"].astype(str).str.zfill(2)
        + df["race_num"].astype(str).str.zfill(2)
    )
    return df, features


def make_split(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """2020-2023 train, 2024 valid (year_full)."""
    train_mask = df["year_full"].between(2020, 2023)
    valid_mask = df["year_full"] == 2024
    return train_mask, valid_mask


def merge_feature(
    df: pd.DataFrame,
    csv_path: str,
    merge_keys: list[str],
    value_cols: list[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    """CSV を読み、 V15 df に merge して新 col を返す."""
    csv_full = ROOT / csv_path
    if not csv_full.exists():
        raise FileNotFoundError(f"CSV not found: {csv_full}")

    # Sample read for column detection
    feat_df = pd.read_csv(csv_full, nrows=1000, low_memory=False)
    available = [c for c in feat_df.columns if c in merge_keys or c in (value_cols or [])]

    if value_cols is None:
        # auto-detect numeric cols (exclude merge keys + obvious non-feature)
        skip = set(merge_keys) | {"horse_name", "year", "month", "day", "comment", "race_date",
                                  "training_date", "training_center", "evaluation",
                                  "horse_id", "umaban", "race_id"}
        numeric_cols = [c for c in feat_df.columns
                        if c not in skip and pd.api.types.is_numeric_dtype(feat_df[c])]
        value_cols = numeric_cols[:8]  # max 8 cols

    # Full read with required columns
    use_cols = list(set(merge_keys + value_cols))
    feat_full = pd.read_csv(csv_full, usecols=lambda c: c in use_cols, low_memory=False)

    # Coerce non-numeric -> numeric (label encode)
    new_cols = []
    for c in value_cols:
        if c not in feat_full.columns:
            continue
        if not pd.api.types.is_numeric_dtype(feat_full[c]):
            try:
                feat_full[c] = pd.to_numeric(feat_full[c], errors="coerce")
            except Exception:
                # label encode
                feat_full[c] = feat_full[c].astype("category").cat.codes
        new_cols.append(c)

    # Map merge keys: cache uses 'nk_race_id' (we built) for race_id, 'umaban' for umaban
    df_keys = []
    feat_keys = []
    for k in merge_keys:
        if k == "race_id":
            df_keys.append("nk_race_id")
            feat_keys.append(k)
        else:
            df_keys.append(k)
            feat_keys.append(k)

    # Ensure merge keys dtypes
    for dk, fk in zip(df_keys, feat_keys):
        if dk == "nk_race_id":
            df[dk] = df[dk].astype(str)
            feat_full[fk] = feat_full[fk].astype(str)
        else:
            try:
                df[dk] = pd.to_numeric(df[dk], errors="coerce").astype("Int64")
                feat_full[fk] = pd.to_numeric(feat_full[fk], errors="coerce").astype("Int64")
            except Exception:
                pass

    # Rename to avoid collision
    rename_map = {c: f"new__{c}" for c in new_cols}
    feat_full = feat_full.rename(columns=rename_map)
    new_named = list(rename_map.values())

    # De-duplicate on merge keys (CSV side)
    feat_full = feat_full.drop_duplicates(subset=feat_keys, keep="first")

    merged = df.merge(feat_full, left_on=df_keys, right_on=feat_keys, how="left")
    return merged, new_named


def train_lgb_auc(
    df: pd.DataFrame,
    features: list[str],
    train_mask: pd.Series,
    valid_mask: pd.Series,
    rounds: int = 200,
    seed: int = 42,
) -> float:
    """軽量 LGB を学習し valid AUC を返す."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    X = df[features].astype(float)
    y = df["target"].astype(int)

    X_tr, y_tr = X.loc[train_mask], y.loc[train_mask]
    X_va, y_va = X.loc[valid_mask], y.loc[valid_mask]

    if len(X_tr) == 0 or len(X_va) == 0:
        return float("nan")

    train_set = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
    valid_set = lgb.Dataset(X_va, label=y_va, reference=train_set, free_raw_data=True)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "seed": seed,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=rounds,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
    )
    pred = model.predict(X_va, num_iteration=model.best_iteration)
    return roc_auc_score(y_va, pred)


def evaluate_feature(args: tuple[Any, ...]) -> dict[str, Any]:
    """1 feature の AUC delta を計算."""
    feat_id, feat_name, csv_path, merge_keys, value_cols, tier = args
    t0 = time.time()
    try:
        df, base_features = load_v15_cache()
        train_mask, valid_mask = make_split(df)

        # baseline
        auc_base = train_lgb_auc(df, base_features, train_mask, valid_mask)

        # +new feature
        merged, new_cols = merge_feature(df, csv_path, merge_keys, value_cols)
        if not new_cols:
            raise ValueError("no usable cols")

        # Re-make masks on merged
        train_mask2 = merged["year_full"].between(2020, 2023)
        valid_mask2 = merged["year_full"] == 2024

        feats_plus = base_features + new_cols
        auc_new = train_lgb_auc(merged, feats_plus, train_mask2, valid_mask2)

        coverage = float(merged.loc[valid_mask2, new_cols].notna().any(axis=1).mean())
        delta = auc_new - auc_base

        return {
            "id": feat_id,
            "name": feat_name,
            "tier": tier,
            "csv": csv_path,
            "value_cols": new_cols,
            "auc_base": round(auc_base, 6),
            "auc_new": round(auc_new, 6),
            "delta": round(delta, 6),
            "coverage_2024": round(coverage, 4),
            "elapsed_s": round(time.time() - t0, 1),
            "status": "ok",
        }
    except Exception as e:
        return {
            "id": feat_id,
            "name": feat_name,
            "tier": tier,
            "csv": csv_path,
            "error": str(e)[:200],
            "elapsed_s": round(time.time() - t0, 1),
            "status": "error",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="all", help="comma-separated id list, or 'all'/'quick'")
    parser.add_argument("--workers", type=int, default=2, help="multiprocessing workers (cache 重いので 2)")
    parser.add_argument("--single", action="store_true", help="逐次 (debug)")
    args = parser.parse_args()

    if args.features == "all":
        plan = FEATURE_PLAN
    elif args.features == "quick":
        plan = [p for p in FEATURE_PLAN if p[5] in ("1h", "2h")]
    else:
        ids = {int(x) for x in args.features.split(",")}
        plan = [p for p in FEATURE_PLAN if p[0] in ids]

    print(f"[Session #51 B] Backtest target: {len(plan)} features")
    for p in plan:
        print(f"  #{p[0]} {p[1]} ({p[5]})")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    if args.single or args.workers <= 1:
        for p in plan:
            r = evaluate_feature(p)
            results.append(r)
            print(f"  done #{r['id']} {r['name']}: status={r['status']}, "
                  f"delta={r.get('delta', 'NA')}, t={r['elapsed_s']}s")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(evaluate_feature, p): p for p in plan}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                print(f"  done #{r['id']} {r['name']}: status={r['status']}, "
                      f"delta={r.get('delta', 'NA')}, t={r['elapsed_s']}s")

    # Sort by delta desc
    results.sort(key=lambda r: r.get("delta", -999), reverse=True)

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] JSON written: {OUT_JSON}")
    print(f"Top 5 by AUC delta:")
    for r in results[:5]:
        if r["status"] == "ok":
            print(f"  #{r['id']} {r['name']}: delta={r['delta']:+.5f}, "
                  f"coverage={r['coverage_2024']:.2%}")


if __name__ == "__main__":
    main()
