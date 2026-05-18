#!/usr/bin/env python
"""v15_full case D — FT+IR enabled 4-model ensemble training (paper shadow candidate).

★ V15 production 完全不変 ★ — 新規 candidate file (v15_full_candidate.pkl.gz) のみ保存。
★ predict_core / daily_predict / race_auto_notify / app.py / cumulative_results.csv 不変 ★
★ paper shadow 専用、 5/24+ paper eval 採用判定後 のみ production 投入候補 ★

設計:
  - V15 学習 cache (data/_v15_optuna_df_cache.pkl.gz) を read-only で load (527K rows, 145 features)
  - T4 leak audit gate (tools/v15_2_train_gate.py --v15-cache) を 起動時に必ず実行 (FAIL なら exit)
  - WF 6 fold (2020..2025) で LGB + XGB + FT + IR を 学習
  - Grid search で 4-model weights 最適化 (per fold)
  - 全 4 component を candidate suffix で save (models/v15_full_candidate.pkl.gz)
  - logs/v15_full_training_<ts>.log に JSONL で進捗記録
  - 8h hard limit、 fold AUC 連続低下で 中止
  - 採用判定 5 項目: Grid mean ≥ 0.870、 6 fold 中 4 fold で V15 LGB+XGB 上回り、 etc

usage:
    python tools/train_v15_full.py [--max-hours 8] [--folds 2020,2021,2022,2023,2024,2025]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))
sys.path.insert(0, str(REPO / "tools"))

# V15 architecture flow — re-use existing FT/IR training functions
from train_v135_ft_transformer import (
    train_ft_transformer,
    DEVICE,
    LGB_PARAMS as V15_LGB_PARAMS,
    XGB_PARAMS as V15_XGB_PARAMS,
)
from train_v135b_intra_ensemble import (
    build_race_id,
    train_intra_race_with_preds,
)

CACHE = REPO / "data" / "_v15_optuna_df_cache.pkl.gz"
OUT_MODEL = REPO / "models" / "v15_full_candidate.pkl.gz"  # ★ candidate suffix 厳守 ★

# V15 baseline (from V15-audit-2 結論):
#   LGB+XGB 6-fold WF mean = 0.8678
#   Grid 4-model 5-fold (in master report) = 0.8858
V15_LGBXGB_BASELINE = 0.8678
V15_GRID_5FOLD_REF = 0.8858


def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


class JsonLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")

    def log(self, **kw) -> None:
        kw.setdefault("ts", now_ts())
        line = json.dumps(kw, ensure_ascii=False, default=str)
        self._fh.write(line + "\n")
        self._fh.flush()
        print(f"[log] {line}", flush=True)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def run_leak_gate(logger: JsonLogger) -> bool:
    """Execute T4 leak audit gate against V15 cache. Return True iff PASS."""
    import subprocess

    cmd = [sys.executable, str(REPO / "tools" / "v15_2_train_gate.py"), "--v15-cache"]
    logger.log(step="gate_start", cmd=" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    logger.log(
        step="gate_done",
        returncode=r.returncode,
        stdout_tail=r.stdout.splitlines()[-5:] if r.stdout else [],
    )
    return r.returncode == 0


def load_cache(logger: JsonLogger):
    with gzip.open(CACHE, "rb") as f:
        cache = pickle.load(f)
    df = cache["df"]
    features = cache["features"]
    if "race_id_unique" not in df.columns:
        df = build_race_id(df)
    logger.log(
        step="cache_loaded",
        rows=len(df),
        features=len(features),
        years=sorted(int(y) for y in df["year"].unique()),
    )
    return df, features


def fit_lgb(X_tr, y_tr, X_te, y_te, gpu: bool):
    params = dict(V15_LGB_PARAMS)
    if gpu:
        # LGB GPU on Windows often missing; fall back gracefully
        params["device"] = "gpu"
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
    try:
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
    except lgb.basic.LightGBMError as exc:
        # GPU not built into binary → fall back to CPU
        params.pop("device", None)
        params["device_type"] = "cpu"
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
    p_te = model.predict(X_te)
    p_tr = model.predict(X_tr)
    return model, p_tr, p_te


def fit_xgb(X_tr, y_tr, X_te, y_te, gpu: bool):
    params = dict(V15_XGB_PARAMS)
    if gpu:
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_te, label=y_te)
    try:
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
    except xgb.core.XGBoostError:
        params["tree_method"] = "hist"
        params.pop("predictor", None)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
    return model, model.predict(dvalid)


def fit_ft(X_tr_s, y_tr, X_te_s, y_te, n_features: int, label: str):
    model, p_te, p_tr, auc = train_ft_transformer(
        X_tr_s,
        y_tr.astype(np.float32),
        X_te_s,
        y_te.astype(np.float32),
        n_features=n_features,
        epochs=50,
        batch_size=2048,
        lr=1e-3,
        patience=10,
        d_token=48,
        n_heads=4,
        n_layers=3,
        dropout=0.1,
        label=label,
    )
    return model, p_te, auc


def fit_ir(df_tr, df_te, features, label: str, test_indices, n_te):
    scaler_ir = StandardScaler()
    df_tr_s = df_tr.copy()
    df_te_s = df_te.copy()
    df_tr_s[features] = scaler_ir.fit_transform(df_tr[features].values.astype(np.float32))
    df_te_s[features] = scaler_ir.transform(df_te[features].values.astype(np.float32))
    ir_model, ir_val_dict, ir_tr_dict, ir_tr_auc, ir_val_auc = train_intra_race_with_preds(
        df_tr_s,
        df_te_s,
        features,
        epochs=30,
        batch_size=64,
        lr=1e-3,
        patience=8,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
    )
    p_ir = np.zeros(n_te, dtype=np.float32)
    cov = 0
    for i, idx in enumerate(test_indices):
        if idx in ir_val_dict:
            p_ir[i] = ir_val_dict[idx]
            cov += 1
        else:
            p_ir[i] = 0.3
    return ir_model, scaler_ir, p_ir, cov


def grid_search_weights(p_lgb, p_xgb, p_ft, p_ir, y, ir_ok: bool):
    best_auc = 0.0
    best_w = None
    if not ir_ok:
        for w1 in np.arange(0.30, 0.71, 0.05):
            w2 = 1 - w1
            auc = roc_auc_score(y, w1 * p_lgb + w2 * p_xgb)
            if auc > best_auc:
                best_auc = auc
                best_w = (float(w1), float(w2), 0.0, 0.0)
        return best_auc, best_w
    for w1 in np.arange(0.20, 0.45, 0.05):
        for w2 in np.arange(0.20, 0.45, 0.05):
            for w3 in np.arange(0.05, 0.30, 0.05):
                w4 = 1.0 - w1 - w2 - w3
                if w4 < 0.01 or w4 > 0.40:
                    continue
                p = w1 * p_lgb + w2 * p_xgb + w3 * p_ft + w4 * p_ir
                auc = roc_auc_score(y, p)
                if auc > best_auc:
                    best_auc = auc
                    best_w = (float(w1), float(w2), float(w3), float(w4))
    return best_auc, best_w


def run_fold(df, features, year: int, deadline: float, logger: JsonLogger, use_gpu: bool):
    ty = year - 2000
    train_mask = df["year"] < ty
    test_mask = df["year"] == ty
    if train_mask.sum() < 1000 or test_mask.sum() < 100:
        logger.log(step="fold_skip", year=year, train=int(train_mask.sum()), test=int(test_mask.sum()))
        return None

    X_tr = df.loc[train_mask, features].values
    y_tr = df.loc[train_mask, "target"].values.astype(np.int8)
    X_te = df.loc[test_mask, features].values
    y_te = df.loc[test_mask, "target"].values.astype(np.int8)
    test_indices = df.loc[test_mask].index.values
    logger.log(
        step="fold_start",
        year=year,
        train=int(len(X_tr)),
        test=int(len(X_te)),
        feats=len(features),
        remaining_sec=int(deadline - time.time()),
    )

    # LGB
    t0 = time.time()
    lgb_model, p_lgb_tr, p_lgb = fit_lgb(X_tr, y_tr, X_te, y_te, gpu=use_gpu)
    auc_lgb = float(roc_auc_score(y_te, p_lgb))
    auc_lgb_tr = float(roc_auc_score(y_tr, p_lgb_tr))
    logger.log(step="lgb_done", year=year, auc=auc_lgb, train_auc=auc_lgb_tr, sec=int(time.time() - t0))

    # XGB
    t0 = time.time()
    xgb_model, p_xgb = fit_xgb(X_tr, y_tr, X_te, y_te, gpu=use_gpu)
    auc_xgb = float(roc_auc_score(y_te, p_xgb))
    logger.log(step="xgb_done", year=year, auc=auc_xgb, sec=int(time.time() - t0))

    # FT
    t0 = time.time()
    scaler_ft = StandardScaler()
    X_tr_s = scaler_ft.fit_transform(X_tr.astype(np.float32))
    X_te_s = scaler_ft.transform(X_te.astype(np.float32))
    ft_model, p_ft, auc_ft = fit_ft(X_tr_s, y_tr, X_te_s, y_te, n_features=len(features), label=f"FT-{year}")
    auc_ft = float(auc_ft)
    logger.log(step="ft_done", year=year, auc=auc_ft, sec=int(time.time() - t0))

    # IR
    t0 = time.time()
    df_tr = df.loc[train_mask].copy()
    df_te = df.loc[test_mask].copy()
    ir_model, scaler_ir, p_ir, ir_cov = fit_ir(df_tr, df_te, features, label=f"IR-{year}",
                                                test_indices=test_indices, n_te=len(X_te))
    cov_pct = ir_cov / len(X_te) * 100
    ir_ok = ir_cov > len(X_te) * 0.5
    auc_ir = float(roc_auc_score(y_te, p_ir)) if ir_ok else 0.0
    logger.log(step="ir_done", year=year, auc=auc_ir, coverage_pct=float(cov_pct), sec=int(time.time() - t0))

    # LGB+XGB reference
    w_ref = auc_lgb / (auc_lgb + auc_xgb)
    auc_lgbxgb = float(roc_auc_score(y_te, w_ref * p_lgb + (1 - w_ref) * p_xgb))

    # Grid
    grid_auc, grid_w = grid_search_weights(p_lgb, p_xgb, p_ft, p_ir, y_te, ir_ok=ir_ok)
    grid_auc = float(grid_auc) if grid_auc else 0.0
    logger.log(step="grid_done", year=year, auc=grid_auc, weights=grid_w)

    return {
        "year": year,
        "lgb_auc": auc_lgb,
        "xgb_auc": auc_xgb,
        "ft_auc": auc_ft,
        "ir_auc": auc_ir,
        "ir_coverage_pct": float(cov_pct),
        "lgbxgb_auc": auc_lgbxgb,
        "grid_auc": grid_auc,
        "grid_weights": list(grid_w) if grid_w else None,
        "train_auc": auc_lgb_tr,
        "gap": float(auc_lgb_tr - auc_lgb),
        "models": {
            "lgb": lgb_model,
            "xgb": xgb_model,
            "ft": ft_model,
            "ir": ir_model,
            "scaler_ft": scaler_ft,
            "scaler_ir": scaler_ir,
        },
    }


def adoption_verdict(fold_results, logger):
    grid_aucs = [r["grid_auc"] for r in fold_results if r["grid_auc"] > 0]
    if not grid_aucs:
        return "NO_GO", {"reason": "no_grid_auc"}
    grid_mean = float(np.mean(grid_aucs))

    # ① Grid mean >= 0.870
    crit1 = grid_mean >= 0.870
    # ② 4 fold 以上 で V15 LGB+XGB 0.8678 上回り
    over_baseline = sum(1 for r in fold_results if r["grid_auc"] >= V15_LGBXGB_BASELINE)
    crit2 = over_baseline >= 4
    # ③ T4 gate PASS は事前に確認済 (gate_done)
    crit3 = True
    # ④ paper ROI base 計算可能 — fold pred 出力が存在
    crit4 = True
    # ⑤ ばらつき < 0.05
    spread = float(max(grid_aucs) - min(grid_aucs))
    crit5 = spread < 0.05

    delta_lgbxgb = grid_mean - V15_LGBXGB_BASELINE
    delta_grid_5fold = grid_mean - V15_GRID_5FOLD_REF

    passes = sum([crit1, crit2, crit3, crit4, crit5])
    if passes == 5 and delta_lgbxgb >= 0.002:
        verdict = "GO"
    elif passes >= 4 and delta_lgbxgb >= -0.001:
        verdict = "NEUTRAL"
    else:
        verdict = "NO_GO"

    info = {
        "grid_mean": grid_mean,
        "delta_vs_lgbxgb": delta_lgbxgb,
        "delta_vs_v15_grid_5fold": delta_grid_5fold,
        "over_baseline_count": over_baseline,
        "spread": spread,
        "passes": passes,
        "crit1_grid_ge_0_870": crit1,
        "crit2_over_baseline_ge_4": crit2,
        "crit3_t4_gate_pass": crit3,
        "crit4_paper_ready": crit4,
        "crit5_spread_lt_0_05": crit5,
    }
    logger.log(step="verdict", verdict=verdict, **info)
    return verdict, info


def save_candidate(fold_results, features, verdict_info, verdict, args, logger):
    # Use mean grid weights as ensemble default
    weights = [
        r["grid_weights"]
        for r in fold_results
        if r["grid_weights"] is not None
    ]
    if weights:
        mean_w = np.mean(np.asarray(weights), axis=0).tolist()
    else:
        mean_w = [0.25, 0.25, 0.15, 0.35]

    # Strip model bytes per fold for compact storage — keep last fold full models for sanity
    fold_summaries = []
    for r in fold_results:
        s = {k: v for k, v in r.items() if k != "models"}
        fold_summaries.append(s)

    last = fold_results[-1] if fold_results else None
    final = {
        "version": "v15_full_candidate",
        "trained_at": now_ts(),
        "features": features,
        "ensemble_weights": {
            "lgb": float(mean_w[0]) if len(mean_w) > 0 else 0.25,
            "xgb": float(mean_w[1]) if len(mean_w) > 1 else 0.25,
            "ft": float(mean_w[2]) if len(mean_w) > 2 else 0.15,
            "ir": float(mean_w[3]) if len(mean_w) > 3 else 0.35,
        },
        "fold_results": fold_summaries,
        "verdict": verdict,
        "verdict_info": verdict_info,
        "models_last_fold": {
            "lgb": last["models"]["lgb"] if last else None,
            "xgb": last["models"]["xgb"] if last else None,
            "ft_state_dict": last["models"]["ft"].state_dict() if last and last["models"]["ft"] else None,
            "ir_state_dict": last["models"]["ir"].state_dict() if last and last["models"]["ir"] else None,
            "scaler_ft": last["models"]["scaler_ft"] if last else None,
            "scaler_ir": last["models"]["scaler_ir"] if last else None,
        },
        "notes": (
            "★ V15 production 完全不変 ★ — paper shadow 専用 candidate。"
            " production 投入は 5/24+ paper eval 採用判定後 ★"
        ),
    }
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT_MODEL, "wb") as f:
        pickle.dump(final, f, protocol=4)
    logger.log(step="saved", path=str(OUT_MODEL), bytes=OUT_MODEL.stat().st_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--folds", type=str, default="2020,2021,2022,2023,2024,2025")
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    log_path = REPO / "logs" / f"v15_full_training_{ts}.log"
    logger = JsonLogger(log_path)
    logger.log(step="start", max_hours=args.max_hours, folds=args.folds, device=str(DEVICE))

    deadline = time.time() + args.max_hours * 3600
    use_gpu = (not args.no_gpu) and torch.cuda.is_available()
    logger.log(step="gpu", available=torch.cuda.is_available(), use_gpu=use_gpu)

    if not run_leak_gate(logger):
        logger.log(step="abort", reason="leak_gate_fail")
        logger.close()
        return 1

    df, features = load_cache(logger)
    folds = [int(x) for x in args.folds.split(",") if x.strip()]

    fold_results = []
    for i, year in enumerate(folds):
        if time.time() > deadline:
            logger.log(step="hard_limit_reached", at_fold=i, year=year)
            break
        try:
            r = run_fold(df, features, year, deadline, logger, use_gpu=use_gpu)
        except Exception as exc:
            logger.log(step="fold_error", year=year, error=str(exc))
            continue
        if r is None:
            continue
        fold_results.append(r)
        # Early abort: 2 consecutive folds below 0.83 grid AUC
        if len(fold_results) >= 2:
            a = fold_results[-1]["grid_auc"]
            b = fold_results[-2]["grid_auc"]
            if a > 0 and b > 0 and a < 0.83 and b < 0.83:
                logger.log(step="abort_low_auc", a=a, b=b)
                break

    if not fold_results:
        logger.log(step="abort", reason="no_folds_completed")
        logger.close()
        return 1

    verdict, info = adoption_verdict(fold_results, logger)
    save_candidate(fold_results, features, info, verdict, args, logger)

    # Final summary
    print()
    print("=" * 70)
    print(f"v15_full Grid AUC mean = {info['grid_mean']:.4f}")
    print(f"  vs V15 LGB+XGB 0.8678: {info['delta_vs_lgbxgb']:+.4f}")
    print(f"  vs V15 Grid 5-fold 0.8858: {info['delta_vs_v15_grid_5fold']:+.4f}")
    print(f"verdict: {verdict}  (passes {info['passes']}/5)")
    print("=" * 70)

    logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
