"""V20 interaction 深掘り (Session #57 D).

C で 2-way interaction が AUC 飽和 (-2bp) と判明。
本 D で:
  1. 3-way interaction (jockey × course × dist_cat 等)
  2. shrinkage tuning (alpha-scale 0.5 / 1.0 / 2.0)
  3. クラス別 (新馬戦 / 重賞 / 平場) AUC

input:
  data/_v15_train_df_cache.pkl       — V15 train cache
  keiba_model_v15_central_live.pkl.gz — V15 features list

output:
  data/v18/session_57_interaction_deep.json — 全実験 metrics
  data/v18/session_57_interaction_deep.md   — 結果 doc

Usage:
  python tools/v20_interaction_deep.py
"""
import sys
import os
import io
import json
import time
import pickle
import gzip
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import lightgbm as lgb
from sklearn.metrics import roc_auc_score


THREE_WAY_SPEC = [
    # name, keys, alpha
    ("int3_jky_crs_dist", ["jockey_id", "course_enc", "dist_cat"], 15),
    ("int3_jky_crs_baba", ["jockey_id", "course_enc", "condition_enc"], 15),
    ("int3_sire_crs_dist", ["sire_enc", "course_enc", "dist_cat"], 30),
]


SHRINKAGE_SPEC = [
    # alpha-scale variants of 2-way (sub-set, the 5 best from C)
    ("int_sire_baba_top3r",      ["sire_enc", "condition_enc"],     20),
    ("int_trainer_course_top3r", ["trainer_id", "course_enc"],       10),
    ("int_sire_course_top3r",    ["sire_enc", "course_enc"],         30),
    ("int_jockey_class_top3r",   ["jockey_id", "class_code"],         5),
    ("int_jockey_trainer_top3r", ["jockey_id", "trainer_id"],         5),
]


LGB_PARAMS = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "lambda_l1": 0.1, "lambda_l2": 0.1, "verbose": -1, "seed": 42,
}


def expand_top3(df: pd.DataFrame, keys: list, alpha: float, prior: float) -> np.ndarray:
    """Compute expanding top3 rate for given keys with Bayesian smoothing."""
    sub = df[keys + ["is_top3"]].copy()
    for k in keys:
        if sub[k].dtype.kind not in ("i", "u", "f"):
            sub[k] = sub[k].astype("category").cat.codes.astype("int32")
        else:
            sub[k] = pd.to_numeric(sub[k], errors="coerce").fillna(-1).astype("int64")
    sub["is_top3"] = sub["is_top3"].astype("int8")
    cum_sum = sub.groupby(keys, sort=False)["is_top3"].cumsum() - sub["is_top3"]
    cum_cnt = sub.groupby(keys, sort=False).cumcount()
    return ((cum_sum + alpha * prior) / (cum_cnt + alpha)).astype("float32").values


def train_eval(name, X_tr, y_tr, X_te, y_te, num_boost=500):
    booster = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(X_tr, y_tr),
        num_boost_round=num_boost,
        valid_sets=[lgb.Dataset(X_te, y_te)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    p_te = booster.predict(X_te)
    auc = roc_auc_score(y_te, p_te)
    print(f"  [{name}] AUC={auc:.4f}, best_iter={booster.best_iteration}, n_features={X_tr.shape[1]}")
    return booster, p_te, auc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/_v15_train_df_cache.pkl")
    ap.add_argument("--v15-model", default="keiba_model_v15_central_live.pkl.gz")
    ap.add_argument("--num-boost", type=int, default=500)
    args = ap.parse_args()

    print("=" * 60)
    print("V20 interaction 深掘り (Session #57 D)")
    print(f"start: {datetime.now()}")
    print("=" * 60)

    # === load ===
    t0 = time.time()
    print("\nLoading V15 cache...")
    with open(args.cache, "rb") as f:
        d = pickle.load(f)
    df = d["df"]
    print(f"  {len(df):,} rows, {time.time()-t0:.1f}s")

    if "is_top3" not in df.columns:
        df["is_top3"] = (df["finish"] <= 3).astype(int)
    df["_y"] = 2000 + df["year"]
    df["dist_cat"] = pd.cut(df["distance"], bins=[0, 1300, 1600, 2000, 2400, 9999],
                            labels=False, right=False).fillna(2).astype(int)

    print("Loading V15 features list...")
    with gzip.open(args.v15_model, "rb") as f:
        m15 = pickle.load(f)
    v15_features = list(m15["features"])
    v15_avail = [c for c in v15_features if c in df.columns]
    print(f"  V15 features available: {len(v15_avail)}")

    # === sort by date ===
    print("Sorting by date_num...")
    df = df.sort_values(["date_num", "race_id"], kind="mergesort").reset_index(drop=True)
    prior = float(df["is_top3"].mean())
    print(f"global prior: {prior:.4f}")

    # === fold ===
    train_mask = (df["_y"] >= 2015) & (df["_y"] <= 2024)
    test_mask = df["_y"] == 2025
    y_tr = df.loc[train_mask, "is_top3"].astype(int).values
    y_te = df.loc[test_mask, "is_top3"].astype(int).values
    print(f"Train: {int(train_mask.sum()):,}, Test: {int(test_mask.sum()):,}")

    results = {
        "session": "Session #57 D",
        "ts": datetime.now().isoformat(),
        "experiments": [],
    }

    # === BASELINE ===
    print("\n## EXP 0: BASELINE (V15 only) ##")
    X_tr_b = df.loc[train_mask, v15_avail]
    X_te_b = df.loc[test_mask, v15_avail]
    _, p_base, auc_base = train_eval("baseline", X_tr_b, y_tr, X_te_b, y_te, args.num_boost)
    results["experiments"].append({"name": "baseline_V15", "auc": float(auc_base), "n_features": len(v15_avail)})

    # === EXP 1: 3-way interaction ===
    print("\n## EXP 1: 3-way interaction ##")
    three_way_feats = []
    for name, keys, alpha in THREE_WAY_SPEC:
        ts = time.time()
        df[name] = expand_top3(df, keys, alpha, prior)
        std = float(df[name].std())
        print(f"  computed [{name}] alpha={alpha}, std={std:.4f}, {time.time()-ts:.1f}s")
        three_way_feats.append(name)
    full_3w = v15_avail + three_way_feats
    X_tr_3 = df.loc[train_mask, full_3w]
    X_te_3 = df.loc[test_mask, full_3w]
    booster_3w, p_3w, auc_3w = train_eval("3-way", X_tr_3, y_tr, X_te_3, y_te, args.num_boost)
    delta_3w = auc_3w - auc_base
    print(f"  Δ AUC vs baseline: {delta_3w:+.4f} ({delta_3w*10000:+.1f} bp)")

    # 3-way feature importance
    imp = booster_3w.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({"feature": full_3w, "gain": imp}).sort_values("gain", ascending=False).reset_index(drop=True)
    rank_3w = []
    for n in three_way_feats:
        r = imp_df.index[imp_df["feature"] == n].tolist()
        if r:
            rank_3w.append({
                "feature": n,
                "rank": int(r[0] + 1),
                "gain": float(imp_df.loc[r[0], "gain"]),
            })
    results["experiments"].append({
        "name": "3-way",
        "auc": float(auc_3w),
        "delta_auc": float(delta_3w),
        "n_features": len(full_3w),
        "ranks": rank_3w,
    })

    # === EXP 2: shrinkage tuning (alpha-scale 0.5 / 1.0 / 2.0) ===
    print("\n## EXP 2: shrinkage tuning (best 5 of 2-way) ##")
    for scale in [0.5, 1.0, 2.0]:
        sf_feats = []
        for name, keys, alpha in SHRINKAGE_SPEC:
            colname = f"{name}_a{scale}"
            df[colname] = expand_top3(df, keys, alpha * scale, prior)
            sf_feats.append(colname)
        full_s = v15_avail + sf_feats
        X_tr_s = df.loc[train_mask, full_s]
        X_te_s = df.loc[test_mask, full_s]
        _, _, auc_s = train_eval(f"shrinkage_x{scale}", X_tr_s, y_tr, X_te_s, y_te, args.num_boost)
        delta_s = auc_s - auc_base
        results["experiments"].append({
            "name": f"shrinkage_x{scale}",
            "auc": float(auc_s),
            "delta_auc": float(delta_s),
            "n_features": len(full_s),
        })
        print(f"  Δ AUC: {delta_s:+.4f} ({delta_s*10000:+.1f} bp)")

    # === EXP 3: class 別 AUC (3-way) ===
    print("\n## EXP 3: class 別 AUC (3-way model) ##")
    test_idx = df.index[test_mask].values
    test_y = df.loc[test_mask, "is_top3"].astype(int).values
    test_class = df.loc[test_mask, "class_code"].values

    # baseline preds were already computed (p_base from EXP 0)
    test_class_buckets = {
        "新馬": [1, 2],
        "未勝利": [3, 4, 5],
        "一勝/2勝": [6, 7, 8, 9],
        "オープン/重賞": [10, 11, 12, 13, 14, 15],
    }
    by_class = []
    for label, codes in test_class_buckets.items():
        m = np.isin(test_class, codes)
        n = int(m.sum())
        if n < 100 or test_y[m].sum() == 0:
            continue
        a_b = roc_auc_score(test_y[m], p_base[m])
        a_3 = roc_auc_score(test_y[m], p_3w[m])
        delta = a_3 - a_b
        by_class.append({
            "label": label, "n": n,
            "auc_baseline": float(a_b),
            "auc_3way": float(a_3),
            "delta": float(delta),
        })
        print(f"  [{label}] n={n}, baseline={a_b:.4f}, 3-way={a_3:.4f}, Δ={delta*10000:+.1f}bp")
    results["class_breakdown"] = by_class

    # === save ===
    out_json = "data/v18/session_57_interaction_deep.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_json}")
    print(f"Total: {(time.time()-t0)/60:.1f}min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
