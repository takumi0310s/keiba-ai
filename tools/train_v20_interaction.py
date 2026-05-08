"""V20 + interaction LGB 学習 (Session #57 C).

V20 base (V15 145 features) + interaction 10 features = 155 features の LGB single fold 学習。
V20 alone (interaction なし) と比較で AUC contribution 計測。

input:
  data/_v15_train_df_cache.pkl       — V15 train cache
  data/v20/interaction_features.csv  — Session #57 B output
  keiba_model_v15_central_live.pkl.gz — V15 features list

output:
  data/v20/models/v20_interaction_v1.pkl                — interaction 込 model
  data/v20/models/v20_baseline.pkl                       — interaction なし model
  data/v18/session_57_v20_interaction_training.json     — metrics + feature importance
  data/v18/session_57_v20_interaction_training.md       — 結果 doc

V15 production 完全不変 (新規 model file、 別 dir)。

Usage:
  python tools/train_v20_interaction.py
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
from sklearn.metrics import roc_auc_score, log_loss


INTERACTION_COLS = [
    "int_horse_jockey_top3r",
    "int_jockey_course_top3r",
    "int_jockey_distcat_top3r",
    "int_jockey_baba_top3r",
    "int_jockey_class_top3r",
    "int_trainer_course_top3r",
    "int_sire_course_top3r",
    "int_sire_distcat_top3r",
    "int_sire_baba_top3r",
    "int_jockey_trainer_top3r",
]


LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": 42,
}


def train_eval(name: str, X_tr, y_tr, X_te, y_te, num_boost: int = 500):
    print(f"\n=== {name}: features={X_tr.shape[1]}, train={len(X_tr):,}, test={len(X_te):,} ===")
    ts = time.time()
    booster = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(X_tr, y_tr),
        num_boost_round=num_boost,
        valid_sets=[lgb.Dataset(X_te, y_te)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    p_te = booster.predict(X_te)
    auc = roc_auc_score(y_te, p_te)
    ll = log_loss(y_te, np.clip(p_te, 1e-7, 1 - 1e-7))
    print(f"  AUC={auc:.4f}, logloss={ll:.4f}, best_iter={booster.best_iteration}, "
          f"time={(time.time()-ts)/60:.1f}min")
    return booster, p_te, auc, ll


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/_v15_train_df_cache.pkl")
    ap.add_argument("--inter", default="data/v20/interaction_features.csv")
    ap.add_argument("--v15-model", default="keiba_model_v15_central_live.pkl.gz")
    ap.add_argument("--out-dir", default="data/v20/models")
    ap.add_argument("--target", default="is_top3", choices=["is_top3", "is_win"])
    ap.add_argument("--num-boost", type=int, default=500)
    args = ap.parse_args()

    print("=" * 60)
    print("V20 + interaction LGB 学習 (Session #57 C)")
    print(f"target: {args.target}, num_boost: {args.num_boost}")
    print(f"start: {datetime.now()}")
    print("=" * 60)

    os.makedirs(args.out_dir, exist_ok=True)

    # === load V15 cache ===
    t0 = time.time()
    print("\nLoading V15 cache...")
    with open(args.cache, "rb") as f:
        d = pickle.load(f)
    df = d["df"]
    print(f"  {len(df):,} rows × {df.shape[1]} cols, {time.time()-t0:.1f}s")

    # ensure target
    if "is_top3" not in df.columns:
        df["is_top3"] = (df["finish"] <= 3).astype(int)
    if "is_win" not in df.columns:
        df["is_win"] = (df["finish"] == 1).astype(int)
    df["_y"] = 2000 + df["year"]

    # === load V15 features list ===
    print("Loading V15 model for feature list...")
    with gzip.open(args.v15_model, "rb") as f:
        m15 = pickle.load(f)
    v15_features = list(m15["features"])
    v15_avail = [c for c in v15_features if c in df.columns]
    print(f"  V15 features: {len(v15_features)}, available in cache: {len(v15_avail)}")

    # === load interaction CSV ===
    print(f"Loading interaction features: {args.inter}")
    inter_df = pd.read_csv(args.inter, dtype={"race_id": str, "horse_id": str})
    print(f"  {len(inter_df):,} rows, cols={list(inter_df.columns)}")

    # === merge ===
    df["race_id"] = df["race_id"].astype(str)
    df["horse_id"] = df["horse_id"].astype(str)
    n_before = len(df)
    df = df.merge(inter_df, on=["race_id", "horse_id"], how="left")
    matched = df[INTERACTION_COLS[0]].notna().sum()
    print(f"  merge: matched {matched:,}/{n_before:,} = {matched/n_before*100:.1f}%")
    # fill missing with global prior
    prior = float(df["is_top3"].mean())
    for col in INTERACTION_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(prior)

    # === fold split: train 2015-2024, test 2025 ===
    train_mask = (df["_y"] >= 2015) & (df["_y"] <= 2024)
    test_mask = df["_y"] == 2025
    print(f"\nTrain: {int(train_mask.sum()):,}, Test: {int(test_mask.sum()):,}")

    y_tr = df.loc[train_mask, args.target].astype(int).values
    y_te = df.loc[test_mask, args.target].astype(int).values

    # === baseline (V15 features only) ===
    base_feats = v15_avail
    X_tr_b = df.loc[train_mask, base_feats]
    X_te_b = df.loc[test_mask, base_feats]
    booster_base, p_base, auc_base, ll_base = train_eval(
        "BASELINE (V15 only)", X_tr_b, y_tr, X_te_b, y_te, args.num_boost
    )

    # === interaction (V15 + 10 interaction) ===
    full_feats = v15_avail + INTERACTION_COLS
    X_tr_i = df.loc[train_mask, full_feats]
    X_te_i = df.loc[test_mask, full_feats]
    booster_int, p_int, auc_int, ll_int = train_eval(
        "INTERACTION (V15 + 10 int)", X_tr_i, y_tr, X_te_i, y_te, args.num_boost
    )

    delta_auc = auc_int - auc_base
    print(f"\n=== Δ AUC: {delta_auc:+.4f} ({delta_auc*10000:+.1f} bp) ===")

    # === feature importance (interaction model) ===
    imp_gain = booster_int.feature_importance(importance_type="gain")
    imp_split = booster_int.feature_importance(importance_type="split")
    feat_imp = pd.DataFrame({
        "feature": full_feats,
        "gain": imp_gain,
        "split": imp_split,
    }).sort_values("gain", ascending=False).reset_index(drop=True)

    print("\n=== Top 30 features (gain) ===")
    print(feat_imp.head(30).to_string(index=False))

    print("\n=== Interaction features rank & gain ===")
    inter_rank = feat_imp[feat_imp["feature"].isin(INTERACTION_COLS)].copy()
    inter_rank["rank"] = inter_rank.index + 1
    print(inter_rank[["rank", "feature", "gain", "split"]].to_string(index=False))

    # === save ===
    base_path = os.path.join(args.out_dir, "v20_baseline.pkl")
    int_path = os.path.join(args.out_dir, "v20_interaction_v1.pkl")
    with open(base_path, "wb") as f:
        pickle.dump({
            "model": booster_base, "features": base_feats,
            "target": args.target, "auc": auc_base, "logloss": ll_base,
        }, f)
    with open(int_path, "wb") as f:
        pickle.dump({
            "model": booster_int, "features": full_feats,
            "target": args.target, "auc": auc_int, "logloss": ll_int,
        }, f)
    print(f"\nSaved: {base_path}, {int_path}")

    # === metrics JSON ===
    metrics = {
        "session": "Session #57 C",
        "ts": datetime.now().isoformat(),
        "target": args.target,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "baseline": {
            "auc": float(auc_base),
            "logloss": float(ll_base),
            "n_features": len(base_feats),
            "best_iter": int(booster_base.best_iteration or 0),
        },
        "interaction": {
            "auc": float(auc_int),
            "logloss": float(ll_int),
            "n_features": len(full_feats),
            "best_iter": int(booster_int.best_iteration or 0),
        },
        "delta_auc": float(delta_auc),
        "interaction_importance": [
            {
                "feature": row["feature"],
                "rank": int(row["rank"]),
                "gain": float(row["gain"]),
                "split": int(row["split"]),
            }
            for _, row in inter_rank.iterrows()
        ],
        "top30_features": [
            {"feature": r["feature"], "gain": float(r["gain"]), "split": int(r["split"])}
            for _, r in feat_imp.head(30).iterrows()
        ],
    }
    metrics_path = "data/v18/session_57_v20_interaction_training.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics: {metrics_path}")

    print(f"\nTotal: {(time.time()-t0)/60:.1f}min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
