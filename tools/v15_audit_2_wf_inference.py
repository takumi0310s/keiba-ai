"""V15-audit-2: READ-ONLY WF AUC inference for V15 stored model.

Loads keiba_model_v15_central.pkl.gz (LGB + XGB) and computes per-fold
validation AUC against data/_v15_optuna_df_cache.pkl.gz.

CRITICAL: this script must NEVER retrain or modify any model file.
Only model.predict() is called. The v15.2 training process (PID 23528)
must not be interrupted.
"""
from __future__ import annotations

import gzip
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data" / "_v15_optuna_df_cache.pkl.gz"
MODEL_PATH = ROOT / "keiba_model_v15_central.pkl.gz"
OUTPUT_JSON = ROOT / "data" / "v15_2" / "v15_audit_2_wf_inference_20260517.json"


def main() -> None:
    t0 = time.time()
    print(f"[load] cache: {CACHE_PATH}")
    with gzip.open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    df: pd.DataFrame = cache["df"]
    print(f"[load] df shape: {df.shape}")

    print(f"[load] model: {MODEL_PATH}")
    with gzip.open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    lgb_booster = model["model"]
    xgb_booster = model["xgb_model"]
    features = model["features"]
    weights = model["ensemble_weights"]
    stored_auc = model["auc"]
    trained_at = model["trained_at"]
    print(f"[load] model version={model['version']} stored_auc={stored_auc} trained_at={trained_at}")
    print(f"[load] ensemble weights: {weights}")
    print(f"[load] features count: {len(features)}")

    # Build matrix once
    print("[prep] building feature matrix")
    X_full = df[features].astype("float32").values
    y_full = df["target"].astype("int8").values
    year_full = df["year_full"].astype(int).values

    fold_years = [2020, 2021, 2022, 2023, 2024, 2025]
    fold_results = []
    for year in fold_years:
        fold_t0 = time.time()
        valid_mask = year_full == year
        train_mask = year_full < year
        valid_n = int(valid_mask.sum())
        train_n = int(train_mask.sum())
        print(f"[fold] year={year} train_n={train_n} valid_n={valid_n}")
        if valid_n == 0:
            continue
        X_valid = X_full[valid_mask]
        y_valid = y_full[valid_mask]

        lgb_pred = lgb_booster.predict(X_valid, num_iteration=lgb_booster.best_iteration)
        dvalid = xgb.DMatrix(X_valid, feature_names=features)
        xgb_pred = xgb_booster.predict(dvalid)
        # ensemble per stored weights (LGB+XGB; mlp weight = 0 in v15)
        w_lgb = float(weights.get("lgb", 0.5))
        w_xgb = float(weights.get("xgb", 0.5))
        ens_pred = w_lgb * lgb_pred + w_xgb * xgb_pred

        auc_lgb = float(roc_auc_score(y_valid, lgb_pred))
        auc_xgb = float(roc_auc_score(y_valid, xgb_pred))
        auc_ens = float(roc_auc_score(y_valid, ens_pred))
        # equal-weight LGB+XGB (sanity)
        auc_ens_eq = float(roc_auc_score(y_valid, 0.5 * lgb_pred + 0.5 * xgb_pred))

        elapsed = time.time() - fold_t0
        print(
            f"[fold] year={year} LGB={auc_lgb:.6f} XGB={auc_xgb:.6f} "
            f"ENS(stored_w)={auc_ens:.6f} ENS(eq)={auc_ens_eq:.6f} took {elapsed:.1f}s"
        )
        fold_results.append({
            "year": year,
            "train_n": train_n,
            "valid_n": valid_n,
            "lgb_auc": auc_lgb,
            "xgb_auc": auc_xgb,
            "lgbxgb_auc_stored_weights": auc_ens,
            "lgbxgb_auc_equal_weights": auc_ens_eq,
            "elapsed_sec": round(elapsed, 1),
        })

    def mean(key: str, subset=None):
        rows = fold_results if subset is None else [r for r in fold_results if r["year"] in subset]
        return float(np.mean([r[key] for r in rows]))

    summary = {
        "stored_auc_field_in_pkl": stored_auc,
        "ensemble_weights": weights,
        "trained_at": trained_at,
        "n_features": len(features),
        "fold_results": fold_results,
        "mean_6fold_2020_2025": {
            "lgb": mean("lgb_auc"),
            "xgb": mean("xgb_auc"),
            "lgbxgb_stored_weights": mean("lgbxgb_auc_stored_weights"),
            "lgbxgb_equal_weights": mean("lgbxgb_auc_equal_weights"),
        },
        "mean_5fold_2021_2025": {
            "lgb": mean("lgb_auc", subset=[2021, 2022, 2023, 2024, 2025]),
            "xgb": mean("xgb_auc", subset=[2021, 2022, 2023, 2024, 2025]),
            "lgbxgb_stored_weights": mean("lgbxgb_auc_stored_weights", subset=[2021, 2022, 2023, 2024, 2025]),
            "lgbxgb_equal_weights": mean("lgbxgb_auc_equal_weights", subset=[2021, 2022, 2023, 2024, 2025]),
        },
        "single_2025": [r for r in fold_results if r["year"] == 2025][0],
        "note": (
            "Read-only inference. The single V15 .pkl.gz was trained on ALL years <= 2025 (NOT a true WF). "
            "Therefore for years 2020..2024 the validation rows were INCLUDED in training, making fold AUCs "
            "leaky upper bounds (not genuine WF generalization). For the 2025 fold the validation rows were "
            "also included in training (model trained on all data). See v15_2/v15_baseline_lgb_xgb_*.json for "
            "the genuine WF retrain baseline (lightweight). This file proves the stored .pkl is a SINGLE-RUN "
            "all-data model and quantifies the in-sample inference AUC."
        ),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[done] wrote {OUTPUT_JSON}")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("fold_results",)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
