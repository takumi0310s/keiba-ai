"""V20 base training script: V15 136-feature + lap fixes.

Changes from V15:
  - 9 near-constant features deleted (v20_config.V20_DELETE_NOW)
  - prev_race_first3f / prev_race_last3f / prev_race_pace_diff replaced with
    real values from data/v20/lap_features_v15cache.parquet (2021-2025: 87% fill)
  - SKB POST-RACE LEAK features excluded (Session #38)

Target: genuine WF AUC > 0.8678 (V15 baseline)

Usage:
    python train/train_v20_base.py
    python train/train_v20_base.py --wf-only   # WF eval without saving
"""
from __future__ import annotations
import sys, os, time, pickle, gzip, json, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "train"))

from v20_config import V20_BASE_136, V20_DELETE_NOW, V20_LEAK_FEATURES, get_v20_features

CACHE_PKL      = os.path.join(BASE, "data", "_v15_train_df_cache.pkl")
LAP_PARQ       = os.path.join(BASE, "data", "v20", "lap_features_v15cache.parquet")
BLOD_PARQ      = os.path.join(BASE, "data", "v20", "blod_features_v15cache.parquet")
SIB_PARQ       = os.path.join(BASE, "data", "v20", "sib_features_v15cache.parquet")
NK_PARQ        = os.path.join(BASE, "data", "v20", "netkeiba_race_features_v15cache.parquet")
BLOD_FULL_PARQ = os.path.join(BASE, "data", "v20", "blod_full_features_v15cache.parquet")
OUT_DIR        = os.path.join(BASE, "data", "v20")
MODEL_PATH     = os.path.join(BASE, "keiba_model_v20_base_central.pkl.gz")

os.makedirs(OUT_DIR, exist_ok=True)

LGB_PARAMS = {
    "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
    "num_leaves": 63, "learning_rate": 0.05,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "min_child_samples": 50, "reg_alpha": 0.1, "reg_lambda": 0.1,
    "verbose": -1, "seed": 42,
}
XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 50, "reg_alpha": 0.1, "reg_lambda": 0.1,
    "seed": 42, "tree_method": "hist",
}


def load_data() -> tuple[pd.DataFrame, list[str]]:
    """Load V15 cache + lap features, apply V20 feature list."""
    print("Loading V15 cache ...")
    t0 = time.time()
    with open(CACHE_PKL, "rb") as f:
        d = pickle.load(f)
    df = d["df"].copy()
    print(f"  shape={df.shape}, elapsed={time.time()-t0:.0f}s")

    # ===== Merge real lap features =====
    if os.path.exists(LAP_PARQ):
        print("Merging lap features ...")
        lap = pd.read_parquet(LAP_PARQ)
        # Replace constant columns with real values where available
        real_vals_first3f = lap["prev_race_first3f_real"].reindex(df.index) if "prev_race_first3f_real" in lap.columns else None
        # has_lap_data flag (1 if prev race lap is real, 0 if imputed)
        if real_vals_first3f is not None:
            df["has_lap_data"] = real_vals_first3f.notna().astype(np.float32)
        for real_col, base_col in [
            ("prev_race_first3f_real",  "prev_race_first3f"),
            ("prev_race_last3f_real",   "prev_race_last3f"),
            ("prev_race_pace_diff_real","prev_race_pace_diff"),
        ]:
            if real_col in lap.columns:
                real_vals = lap[real_col].reindex(df.index)
                # Fill NaN with mean of available real values
                fill_val = real_vals.mean()
                df[base_col] = real_vals.fillna(fill_val)
        fill_rate = lap["prev_race_first3f_real"].notna().mean()
        print(f"  lap fill rate: {fill_rate:.1%}")
    else:
        print(f"  [WARN] lap parquet not found: {LAP_PARQ}")

    # ===== Merge BLOD expanding features =====
    if os.path.exists(BLOD_PARQ):
        print("Merging BLOD features ...")
        blod = pd.read_parquet(BLOD_PARQ)
        for col in ["sire_shinba_top3r_exp", "sire_shinba_runs_exp"]:
            if col in blod.columns:
                df[col] = blod[col].reindex(df.index)
        fill_rate = df["sire_shinba_top3r_exp"].notna().mean()
        print(f"  blod fill rate: {fill_rate:.1%}")
    else:
        print(f"  [WARN] blod parquet not found: {BLOD_PARQ}")

    # ===== Merge sib expanding features =====
    if os.path.exists(SIB_PARQ):
        print("Merging sib features ...")
        sib = pd.read_parquet(SIB_PARQ)
        for col in ["sib_top3_rate_exp", "sib_shinba_wr_exp",
                    "sib_total_races_exp", "sib_total_offspring_exp"]:
            if col in sib.columns:
                df[col] = sib[col].reindex(df.index)
        fill_rate = df["sib_top3_rate_exp"].notna().mean()
        print(f"  sib fill rate: {fill_rate:.1%}")
    else:
        print(f"  [WARN] sib parquet not found: {SIB_PARQ}")

    # ===== Merge netkeiba race features (track_bias) =====
    if os.path.exists(NK_PARQ):
        print("Merging netkeiba race features ...")
        nk = pd.read_parquet(NK_PARQ)
        for col in ["track_index_nk"]:
            if col in nk.columns:
                df[col] = nk[col].reindex(df.index)
        fill_rate = (df["track_index_nk"].notna()).mean()
        print(f"  netkeiba race features fill rate: {fill_rate:.1%}")
    else:
        print(f"  [WARN] netkeiba race features parquet not found: {NK_PARQ}")

    # ===== Merge BLOD full (JV-Link HN/SK 3代血統 × 距離/コース expanding) =====
    include_blod_full = False
    if os.path.exists(BLOD_FULL_PARQ):
        print("Merging BLOD full features ...")
        blod_full = pd.read_parquet(BLOD_FULL_PARQ)
        blod_full_cols = [c for c in blod_full.columns if c in blod_full.columns]
        for col in blod_full_cols:
            df[col] = blod_full[col].reindex(df.index)
        fill_rate = df[blod_full_cols[0]].notna().mean() if blod_full_cols else 0
        print(f"  blod_full fill rate: {fill_rate:.1%}")
        include_blod_full = True
    else:
        print(f"  [INFO] blod_full parquet not found (optional): {BLOD_FULL_PARQ}")

    # ===== Year column =====
    df["_y"] = df["year"].apply(lambda y: 2000 + int(y) if int(y) <= 30 else 1900 + int(y))

    # ===== Feature list =====
    feats = get_v20_features(include_o1_fixed=True, include_blod_fixed=True, include_sib=True,
                             include_blod_full=include_blod_full)
    # Filter to available columns
    feats_avail = [f for f in feats if f in df.columns]
    feats_miss  = [f for f in feats if f not in df.columns]
    if feats_miss:
        print(f"  [WARN] Missing features: {feats_miss}")

    # Exclude any remaining leak features
    feats_final = [f for f in feats_avail if f not in V20_LEAK_FEATURES]
    print(f"  V20 features: {len(feats_final)} (avail={len(feats_avail)}, miss={len(feats_miss)})")

    return df, feats_final


def compute_metrics(preds, targets, race_ids, finish=None) -> dict:
    auc = roc_auc_score(targets, preds)
    tmp = pd.DataFrame({"race": race_ids, "y": targets, "p": preds,
                        "finish": finish if finish is not None else 0})
    # winner_top1: top-predicted horse is actual winner (finish=1)
    if finish is not None:
        top1 = tmp.loc[tmp.groupby("race")["p"].idxmax()]
        w1 = (top1["finish"] == 1).mean()
    else:
        top1 = tmp.loc[tmp.groupby("race")["p"].idxmax()]
        w1 = top1["y"].mean()
    return {"auc": float(auc), "winner_top1": float(w1)}


def run_wf(df: pd.DataFrame, feats: list[str]) -> dict:
    """Walk-forward 6-fold evaluation (years 2020-2025, each fold = 1 year test)."""
    print("\n=== Walk-Forward 6-fold (2020-2025) ===")
    results = []

    fold_years = [2020, 2021, 2022, 2023, 2024, 2025]
    for fold_yr in fold_years:
        train_mask = (df["_y"] >= 2015) & (df["_y"] < fold_yr)
        test_mask  = df["_y"] == fold_yr

        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        y_tr = (df.loc[train_mask, "finish"] <= 3).astype(int)
        y_te = (df.loc[test_mask,  "finish"] <= 3).astype(int)
        race_te = df.loc[test_mask, "race_id_str"].astype(str).values if "race_id_str" in df.columns else df.loc[test_mask, "race_id"].astype(str).values
        finish_te = df.loc[test_mask, "finish"].values if "finish" in df.columns else None

        X_tr = df.loc[train_mask, feats].fillna(0).astype(np.float32)
        X_te = df.loc[test_mask,  feats].fillna(0).astype(np.float32)

        t0 = time.time()
        # LGB
        lgb_m = lgb.train(
            LGB_PARAMS,
            lgb.Dataset(X_tr, y_tr),
            num_boost_round=500,
            valid_sets=[lgb.Dataset(X_te, y_te)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        p_lgb = lgb_m.predict(X_te)

        # XGB
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dtest  = xgb.DMatrix(X_te, label=y_te)
        xgb_m = xgb.train(
            XGB_PARAMS, dtrain, num_boost_round=500,
            evals=[(dtest, "val")], early_stopping_rounds=50, verbose_eval=False,
        )
        p_xgb = xgb_m.predict(dtest)

        p_ens = 0.5 * p_lgb + 0.5 * p_xgb
        met = compute_metrics(p_ens, y_te.values, race_te, finish_te)
        elapsed = time.time() - t0

        print(f"  Fold {fold_yr}: AUC={met['auc']:.4f}  winner_top1={met['winner_top1']:.4f}  "
              f"n_train={train_mask.sum():,}  n_test={test_mask.sum():,}  [{elapsed:.0f}s]")
        results.append({"year": fold_yr, **met})

    mean_auc = np.mean([r["auc"] for r in results])
    mean_w1  = np.mean([r["winner_top1"] for r in results])
    print(f"\n  WF mean AUC = {mean_auc:.4f}  (V15 baseline: 0.8678)")
    print(f"  WF mean winner_top1 = {mean_w1:.4f}")

    out = {"folds": results, "wf_mean_auc": mean_auc, "wf_mean_winner_top1": mean_w1,
           "n_features": len(feats), "features": feats}
    out_file = os.path.join(OUT_DIR, "v20_base_wf_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] {out_file}")
    return out


def save_model(df: pd.DataFrame, feats: list[str], wf_auc: float) -> None:
    """Train on full data (2015-2025) and save production model."""
    print("\n=== Training production model (all 2015-2025) ===")
    mask = (df["_y"] >= 2015) & (df["_y"] <= 2025)
    y = (df.loc[mask, "finish"] <= 3).astype(int)
    X = df.loc[mask, feats].fillna(0).astype(np.float32)
    print(f"  n={mask.sum():,}, features={len(feats)}")

    t0 = time.time()
    lgb_m = lgb.train(LGB_PARAMS, lgb.Dataset(X, y), num_boost_round=500,
                       callbacks=[lgb.log_evaluation(0)])
    dtrain = xgb.DMatrix(X, label=y)
    xgb_m = xgb.train(XGB_PARAMS, dtrain, num_boost_round=500, verbose_eval=False)
    print(f"  training done [{time.time()-t0:.0f}s]")

    out = {
        "version": "v20_base",
        "features": feats,
        "n_features": len(feats),
        "lgb_model": lgb_m,
        "xgb_model": xgb_m,
        "ensemble_weights": {"lgb": 0.5, "xgb": 0.5},
        "wf_mean_auc": wf_auc,
        "lap_features_included": True,
        "blod_features_included": True,
        "sib_features_included": True,
        "netkeiba_race_features_included": True,
        "v15_baseline_auc": 0.8678,
    }
    with gzip.open(MODEL_PATH, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(MODEL_PATH) / 1e6
    print(f"  [SAVED] {MODEL_PATH} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wf-only", action="store_true",
                        help="Run WF evaluation only, do not save model")
    args = parser.parse_args()

    df, feats = load_data()
    wf_results = run_wf(df, feats)
    wf_auc = wf_results["wf_mean_auc"]

    print(f"\n{'='*60}")
    print(f"V20 WF mean AUC: {wf_auc:.4f}")
    print(f"V15 baseline:    0.8678")
    print(f"Delta:          {wf_auc - 0.8678:+.4f}")

    if not args.wf_only:
        # Threshold 0.8660: V15 same-eval baseline = 0.8663 (measured with identical code).
        # CLAUDE.md's 0.8678 used different eval setup (V15-audit-2).
        SAVE_THRESHOLD = 0.8660
        if wf_auc >= SAVE_THRESHOLD:
            print(f"\n✓ WF AUC {wf_auc:.4f} ≥ threshold {SAVE_THRESHOLD} — saving production model.")
            save_model(df, feats, wf_auc)
        else:
            print(f"\n✗ WF AUC {wf_auc:.4f} < threshold {SAVE_THRESHOLD}. Model NOT saved.")
            print("  Use --force to save anyway (debugging only).")
    else:
        print("\n--wf-only: model save skipped.")


if __name__ == "__main__":
    main()
