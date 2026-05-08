"""Session #51 C: AUDIT-1 Top 27 combo search (V15 + 2-3 features).

Greedy forward selection ベース、 過適合 risk 監視。

Usage:
    python tools/audit_combo_search.py --max-k 2  # 2-feature combo
    python tools/audit_combo_search.py --max-k 3  # 3-feature combo (重)
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "_v15_optuna_df_cache.pkl.gz"
OUT_JSON = ROOT / "data" / "v18" / "sprint5_combo_metrics.json"

# B 結果から LEAK を除外、 delta ≥ -0.0005 候補 (top 6)
# 期待: combo で +0.001 以上の 相乗効果あれば V15.5 / V15.6 候補
COMBO_CANDIDATES = [
    # (id, name, csv, merge_keys, value_cols)
    (8,  "jrdb_jo_bb",        "data/jrdb_jo.csv",
        ["race_id", "umaban"], ["gaisha_bb", "gaisha_bb_wr", "breeder_bb", "breeder_bb_wr"]),
    (6,  "jrdb_cha_oikiri",   "data/jrdb_cha.csv",
        ["race_id", "umaban"], ["oikiri_rank", "oikiri_idx", "ten_time_idx",
                                "chukan_time_idx", "shimai_time_idx"]),
    (17, "jrdb_tyb_live",     "data/jrdb_tyb.csv",
        ["race_id", "umaban"], ["bagu_change", "cancel_flag"]),
    (10, "jrdb_cyb_train",    "data/jrdb_cyb.csv",
        ["race_id", "umaban"], ["train_mark", "train_amount", "train_change", "train_eval"]),
    (12, "race_analysis_score", "data/netkeiba_race_analysis.csv",
        ["race_id", "umaban"], ["score"]),
    (7,  "speed_index_dist_course", "data/netkeiba_speed_index.csv",
        ["race_id", "umaban"], ["index_dist", "index_course"]),
]


def load_v15_cache() -> tuple[pd.DataFrame, list[str]]:
    with gzip.open(CACHE, "rb") as f:
        d = pickle.load(f)
    df = d["df"].copy()
    features = d["features"]
    df["target"] = ((df["finish"] >= 1) & (df["finish"] <= 3)).astype(int)
    df["nk_race_id"] = (
        df["year_full"].astype(str)
        + (df["course_enc"] + 1).astype(int).astype(str).str.zfill(2)
        + df["kai"].astype(str).str.zfill(2)
        + df["nichi"].astype(str).str.zfill(2)
        + df["race_num"].astype(str).str.zfill(2)
    )
    return df, features


def merge_one(
    df: pd.DataFrame,
    csv_path: str,
    merge_keys: list[str],
    value_cols: list[str],
    suffix: str,
) -> tuple[pd.DataFrame, list[str]]:
    csv_full = ROOT / csv_path
    use_cols = list(set(merge_keys + value_cols))
    feat = pd.read_csv(csv_full, usecols=lambda c: c in use_cols, low_memory=False)
    new_cols = []
    for c in value_cols:
        if c not in feat.columns:
            continue
        if not pd.api.types.is_numeric_dtype(feat[c]):
            feat[c] = pd.to_numeric(feat[c], errors="coerce")
        new_cols.append(c)
    df_keys, feat_keys = [], []
    for k in merge_keys:
        if k == "race_id":
            df_keys.append("nk_race_id")
            feat_keys.append(k)
        else:
            df_keys.append(k)
            feat_keys.append(k)
    for dk, fk in zip(df_keys, feat_keys):
        if dk == "nk_race_id":
            df[dk] = df[dk].astype(str)
            feat[fk] = feat[fk].astype(str)
        else:
            try:
                df[dk] = pd.to_numeric(df[dk], errors="coerce").astype("Int64")
                feat[fk] = pd.to_numeric(feat[fk], errors="coerce").astype("Int64")
            except Exception:
                pass
    rename = {c: f"new_{suffix}__{c}" for c in new_cols}
    feat = feat.rename(columns=rename)
    new_named = list(rename.values())
    feat = feat.drop_duplicates(subset=feat_keys, keep="first")
    merged = df.merge(feat, left_on=df_keys, right_on=feat_keys, how="left",
                      suffixes=("", f"_{suffix}"))
    return merged, new_named


def train_lgb_auc(
    df: pd.DataFrame, features: list[str],
    train_mask: pd.Series, valid_mask: pd.Series,
    rounds: int = 200, seed: int = 42,
) -> float:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    X = df[features].astype(float)
    y = df["target"].astype(int)
    train_set = lgb.Dataset(X.loc[train_mask], y.loc[train_mask], free_raw_data=True)
    valid_set = lgb.Dataset(X.loc[valid_mask], y.loc[valid_mask],
                            reference=train_set, free_raw_data=True)
    params = {
        "objective": "binary", "metric": "auc",
        "boosting_type": "gbdt", "num_leaves": 63,
        "learning_rate": 0.05, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5,
        "min_child_samples": 50, "reg_alpha": 0.1, "reg_lambda": 0.1,
        "verbose": -1, "seed": seed,
    }
    model = lgb.train(params, train_set, num_boost_round=rounds,
                      valid_sets=[valid_set],
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
    pred = model.predict(X.loc[valid_mask], num_iteration=model.best_iteration)
    return roc_auc_score(y.loc[valid_mask], pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-k", type=int, default=2, help="combo size (2 or 3)")
    args = parser.parse_args()

    print(f"[Session #51 C] Combo search: V15 + {args.max_k} features from {len(COMBO_CANDIDATES)} candidates")

    df, base_features = load_v15_cache()

    # baseline
    train_mask = df["year_full"].between(2020, 2023)
    valid_mask = df["year_full"] == 2024
    t0 = time.time()
    auc_base = train_lgb_auc(df, base_features, train_mask, valid_mask)
    print(f"  V15 baseline AUC: {auc_base:.5f} (t={time.time()-t0:.1f}s)")

    # 2-feature combos: 6C2 = 15
    results = []
    pairs = list(itertools.combinations(COMBO_CANDIDATES, args.max_k))
    print(f"  Testing {len(pairs)} combos...")

    for combo in pairs:
        ids = [c[0] for c in combo]
        names = [c[1] for c in combo]
        t0 = time.time()
        try:
            merged = df.copy()
            all_new = []
            for c_id, name, csv, mk, vc in combo:
                merged, new = merge_one(merged, csv, mk, vc, suffix=str(c_id))
                all_new.extend(new)
            tr = merged["year_full"].between(2020, 2023)
            va = merged["year_full"] == 2024
            auc_combo = train_lgb_auc(merged, base_features + all_new, tr, va)
            delta = auc_combo - auc_base
            results.append({
                "ids": ids,
                "names": names,
                "k": args.max_k,
                "auc_base": round(auc_base, 6),
                "auc_combo": round(auc_combo, 6),
                "delta": round(delta, 6),
                "elapsed_s": round(time.time() - t0, 1),
                "status": "ok",
            })
            print(f"    combo {ids}: delta={delta:+.5f}, t={time.time()-t0:.1f}s")
        except Exception as e:
            results.append({
                "ids": ids, "names": names, "k": args.max_k,
                "error": str(e)[:200], "status": "error",
                "elapsed_s": round(time.time() - t0, 1),
            })
            print(f"    combo {ids}: ERROR {e}")

    results.sort(key=lambda r: r.get("delta", -999), reverse=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] JSON written: {OUT_JSON}")
    print(f"Top 5:")
    for r in results[:5]:
        if r["status"] == "ok":
            print(f"  {r['ids']}: delta={r['delta']:+.5f}")


if __name__ == "__main__":
    main()
