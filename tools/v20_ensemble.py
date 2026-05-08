"""Session #56 D: V20 4-model ensemble.

LGB + XGB + FT-Transformer + IntraRace Attention の 4-model grid ensemble.
weight optimization (validation で grid search)。

Usage:
    python tools/v20_ensemble.py
"""

from __future__ import annotations

import gzip
import json
import pickle
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "_v15_optuna_df_cache.pkl.gz"
FT_PRED = ROOT / "data" / "v20" / "models" / "v20_ft_pred.npz"
IR_PRED = ROOT / "data" / "v20" / "models" / "v20_ir_pred.npz"
ENS_OUT = ROOT / "data" / "v20" / "models" / "v20_ensemble_v1.pkl"
METRICS_OUT = ROOT / "data" / "v18" / "session_56_ensemble_metrics.json"


def train_lgb_xgb(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Train LGB+XGB on V15 cache, return valid predictions."""
    import lightgbm as lgb
    import xgboost as xgb

    train_mask = df["year_full"].between(2020, 2023)
    valid_mask = df["year_full"] == 2024

    X = df[features].apply(lambda c: pd.to_numeric(c, errors="coerce")).fillna(0).astype(float).values
    y = df["target"].astype(int).values
    X_tr, X_va = X[train_mask.values], X[valid_mask.values]
    y_tr, y_va = y[train_mask.values], y[valid_mask.values]

    print(f"  train: {len(X_tr)}, valid: {len(X_va)}")

    # LGB
    print("  Training LGB...")
    t0 = time.time()
    params_lgb = {
        "objective": "binary", "metric": "auc",
        "boosting_type": "gbdt", "num_leaves": 63,
        "learning_rate": 0.05, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5,
        "min_child_samples": 50, "reg_alpha": 0.1, "reg_lambda": 0.1,
        "verbose": -1, "seed": 42,
    }
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    lgb_model = lgb.train(
        params_lgb, dtrain, num_boost_round=500,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    lgb_pred = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
    lgb_auc = roc_auc_score(y_va, lgb_pred)
    print(f"  LGB AUC: {lgb_auc:.5f} (t={time.time()-t0:.1f}s)")

    # XGB
    print("  Training XGB...")
    t0 = time.time()
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    xgb_params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 50, "reg_alpha": 0.1, "reg_lambda": 0.1,
        "seed": 42, "tree_method": "hist", "verbosity": 0,
    }
    xgb_model = xgb.train(
        xgb_params, dtr, num_boost_round=500,
        evals=[(dva, "valid")],
        early_stopping_rounds=30, verbose_eval=0,
    )
    xgb_pred = xgb_model.predict(dva)
    xgb_auc = roc_auc_score(y_va, xgb_pred)
    print(f"  XGB AUC: {xgb_auc:.5f} (t={time.time()-t0:.1f}s)")

    return lgb_pred, xgb_pred, lgb_auc, xgb_auc, y_va


def grid_search_weights(preds: dict[str, np.ndarray], y: np.ndarray) -> dict:
    """Grid search 4-model weights (5×5×5×5 = 625 combos)."""
    print("\n  Grid search weights...")
    grid = np.arange(0.0, 1.01, 0.25)  # 0, 0.25, 0.5, 0.75, 1.0

    best_auc = 0.0
    best_w = None
    for w_l, w_x, w_f, w_i in product(grid, grid, grid, grid):
        s = w_l + w_x + w_f + w_i
        if s <= 0:
            continue
        ens = (w_l * preds["lgb"] + w_x * preds["xgb"] +
               w_f * preds["ft"] + w_i * preds["ir"]) / s
        auc = roc_auc_score(y, ens)
        if auc > best_auc:
            best_auc = auc
            best_w = (w_l / s, w_x / s, w_f / s, w_i / s)

    return {
        "best_auc": best_auc,
        "weights": {"lgb": best_w[0], "xgb": best_w[1], "ft": best_w[2], "ir": best_w[3]},
    }


def fine_tune_weights(preds: dict[str, np.ndarray], y: np.ndarray, base_w: tuple,
                       grid_step: float = 0.05, span: float = 0.15) -> dict:
    """Fine grid search around best initial weights."""
    print("\n  Fine grid search...")
    best_auc = 0.0
    best_w = None
    grids = []
    for b in base_w:
        lo = max(0.0, b - span)
        hi = min(1.0, b + span)
        grids.append(np.arange(lo, hi + 1e-9, grid_step))

    for w_l, w_x, w_f, w_i in product(*grids):
        s = w_l + w_x + w_f + w_i
        if s <= 0:
            continue
        ens = (w_l * preds["lgb"] + w_x * preds["xgb"] +
               w_f * preds["ft"] + w_i * preds["ir"]) / s
        auc = roc_auc_score(y, ens)
        if auc > best_auc:
            best_auc = auc
            best_w = (w_l / s, w_x / s, w_f / s, w_i / s)

    return {
        "best_auc": best_auc,
        "weights": {"lgb": best_w[0], "xgb": best_w[1], "ft": best_w[2], "ir": best_w[3]},
    }


def main() -> None:
    print("[Session #56 D] 4-model ensemble", flush=True)

    # Load V15 cache
    print(f"Loading {CACHE}", flush=True)
    with gzip.open(CACHE, "rb") as f:
        d = pickle.load(f)
    df = d["df"].copy()
    features = d["features"]
    df["target"] = ((df["finish"] >= 1) & (df["finish"] <= 3)).astype(int)
    print(f"  rows: {len(df)}, features: {len(features)}", flush=True)

    # Train LGB + XGB
    lgb_pred, xgb_pred, lgb_auc, xgb_auc, y_va = train_lgb_xgb(df, features)

    # Load FT/IR predictions
    print("\nLoading FT/IR predictions...")
    ft_data = np.load(FT_PRED)
    ir_data = np.load(IR_PRED)
    ft_pred = ft_data["valid_probs"]
    ir_pred = ir_data["valid_probs"]
    ft_targets = ft_data["valid_targets"]
    ir_targets = ir_data["valid_targets"]

    # Sanity check: targets must align
    assert len(ft_pred) == len(y_va) == len(ir_pred), \
        f"length mismatch: ft={len(ft_pred)}, lgb={len(y_va)}, ir={len(ir_pred)}"
    assert np.array_equal(y_va, ft_targets), "FT targets misaligned"
    assert np.array_equal(y_va, ir_targets), "IR targets misaligned"

    ft_auc = roc_auc_score(y_va, ft_pred)
    ir_auc = roc_auc_score(y_va, ir_pred)
    print(f"  FT  AUC: {ft_auc:.5f}")
    print(f"  IR  AUC: {ir_auc:.5f}")

    # Equal weight ensemble
    ens_eq = (lgb_pred + xgb_pred + ft_pred + ir_pred) / 4
    eq_auc = roc_auc_score(y_va, ens_eq)
    print(f"\n  Equal weight ensemble AUC: {eq_auc:.5f}")

    # Coarse grid search
    preds = {"lgb": lgb_pred, "xgb": xgb_pred, "ft": ft_pred, "ir": ir_pred}
    coarse = grid_search_weights(preds, y_va)
    print(f"  Coarse grid best AUC: {coarse['best_auc']:.5f}")
    print(f"  Coarse weights: {coarse['weights']}")

    # Fine grid search around coarse best
    base_w = tuple(coarse["weights"].values())
    fine = fine_tune_weights(preds, y_va, base_w)
    print(f"\n  Fine grid best AUC: {fine['best_auc']:.5f}")
    print(f"  Fine weights: {fine['weights']}")

    # 3-model variants (LGB+XGB+IR、 LGB+IR+FT 等)
    print("\n  3-model variants:")
    variants = {
        "LGB+IR (no XGB/FT)": {"lgb": 0.5, "ir": 0.5},
        "LGB+XGB+IR (no FT)": {"lgb": 0.33, "xgb": 0.33, "ir": 0.34},
        "IR + LGB+XGB avg": {"lgb": 0.25, "xgb": 0.25, "ir": 0.5},
    }
    variant_results = {}
    for name, w in variants.items():
        s = sum(w.values())
        ens = sum(w_v * preds[k] for k, w_v in w.items()) / s
        auc = roc_auc_score(y_va, ens)
        variant_results[name] = {"auc": auc, "weights": w}
        print(f"    {name}: AUC {auc:.5f}")

    # Final
    best_auc = fine["best_auc"]
    best_w = fine["weights"]
    print(f"\n{'='*60}")
    print(f"  Final ensemble AUC: {best_auc:.5f}")
    print(f"  vs LGB-only baseline: {best_auc - lgb_auc:+.5f}")
    print(f"  Best weights: {best_w}")
    print(f"{'='*60}")

    # Save
    payload = {
        "version": "v20_ensemble_v1",
        "weights": best_w,
        "individual_aucs": {
            "lgb": lgb_auc, "xgb": xgb_auc,
            "ft": float(ft_auc), "ir": float(ir_auc),
        },
        "ensemble_auc": best_auc,
        "delta_vs_lgb": best_auc - lgb_auc,
        "equal_weight_auc": eq_auc,
        "coarse_grid_auc": coarse["best_auc"],
        "variant_results": variant_results,
    }
    ENS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(ENS_OUT, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n[OK] Ensemble payload: {ENS_OUT}")

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                           encoding="utf-8")
    print(f"[OK] Metrics: {METRICS_OUT}")


if __name__ == "__main__":
    main()
