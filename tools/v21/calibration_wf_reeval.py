#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sub-task 15: Calibration WF reevaluation (read-only audit).

V15 production を一切 touch せず、 4 calibration methods を WF 6-fold で完全比較。

Methods (4):
  - Platt (sklearn LogisticRegression, C=1e9)
  - Isotonic (sklearn IsotonicRegression)
  - Beta calibration (scipy curve_fit, 3-parameter beta)
  - Temperature scaling (torch, GPU 活用)

Output: data/v21/calibration_wf/{wf_oof_predictions.parquet, metrics.json, report.md}

Design choices (★ honest ★):
  - V15 production 4-model (LGB+XGB+FT+IR) の full retraining は 数時間 で重い
    → LGB+XGB 2-model WF を 6-fold で実行 (production の score 比例性 を保つ)
    → これは production AUC 0.8939 のうち LGB+XGB 部分 (≈ 0.873-0.876)
    → calibration の相対比較には十分 (★ AUC 維持 verify は LGB+XGB level で行う ★)
  - WF 6-fold = years [2020, 2021, 2022, 2023, 2024, 2025]
  - train = year < test_year, test = year == test_year

NEVER touched:
  - keiba_model_v15_central*.pkl.gz
  - predict_core.py, daily_predict.py, race_auto_notify.py, app.py
  - calibrator_v15_pilot.pkl, calibrator_v15_pilot_v2.pkl
  - cumulative_results.csv

OK:
  - read _v15_optuna_df_cache.pkl.gz
  - GPU 活用 (torch.cuda)
  - write data/v21/calibration_wf/*

Usage:
    python tools/v21/calibration_wf_reeval.py
    python tools/v21/calibration_wf_reeval.py --quick   # 2 fold only (2024,2025)
"""
import argparse
import json
import os
import pickle
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'v21', 'calibration_wf')
os.makedirs(OUT_DIR, exist_ok=True)

WF_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


# =====================================================
# Metrics
# =====================================================

def brier_score(y_true, y_pred):
    return float(np.mean((np.asarray(y_pred) - np.asarray(y_true, dtype=float)) ** 2))


def ece(y_true, y_pred, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred)
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(y_pred)
    total = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_pred >= bins[i]) & (y_pred <= bins[i+1])
        else:
            mask = (y_pred >= bins[i]) & (y_pred < bins[i+1])
        m = mask.sum()
        if m == 0:
            continue
        total += (m / n) * abs(y_true[mask].mean() - y_pred[mask].mean())
    return float(total)


def auc_score(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_pred))


def logloss(y_true, y_pred, eps=1e-12):
    p = np.clip(y_pred, eps, 1 - eps)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# =====================================================
# Calibrators
# =====================================================

def fit_platt(y_true, y_pred):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1e9, solver='lbfgs', max_iter=1000)
    clf.fit(np.asarray(y_pred).reshape(-1, 1), y_true)
    return clf


def apply_platt(clf, y_pred):
    return clf.predict_proba(np.asarray(y_pred).reshape(-1, 1))[:, 1]


def fit_isotonic(y_true, y_pred):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(y_pred, y_true)
    return ir


def apply_isotonic(ir, y_pred):
    return ir.transform(y_pred)


def fit_beta(y_true, y_pred):
    """3-param Beta calibration: p_cal = 1 / (1 + 1/(exp(a)*p^b/(1-p)^c)).

    Simplified Kull et al. 2017 formulation:
      logit(p_cal) = a + b * log(p) - c * log(1-p)
    """
    from sklearn.linear_model import LogisticRegression
    eps = 1e-12
    p = np.clip(np.asarray(y_pred), eps, 1 - eps)
    X = np.column_stack([np.log(p), -np.log(1 - p)])
    clf = LogisticRegression(C=1e9, solver='lbfgs', max_iter=1000)
    clf.fit(X, y_true)
    return clf


def apply_beta(clf, y_pred):
    eps = 1e-12
    p = np.clip(np.asarray(y_pred), eps, 1 - eps)
    X = np.column_stack([np.log(p), -np.log(1 - p)])
    return clf.predict_proba(X)[:, 1]


def fit_temperature(y_true, y_pred, device='cuda', max_iter=200):
    """Temperature scaling: p_cal = sigmoid(logit(p) / T). 1-param, torch GPU."""
    import torch
    import torch.nn as nn
    eps = 1e-12
    p = np.clip(np.asarray(y_pred, dtype=np.float32), eps, 1 - eps)
    logit = np.log(p / (1 - p))
    y = np.asarray(y_true, dtype=np.float32)

    dev = torch.device(device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    logit_t = torch.from_numpy(logit).to(dev)
    y_t = torch.from_numpy(y).to(dev)
    T = nn.Parameter(torch.ones(1, device=dev))

    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = bce(logit_t / T, y_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return {'T': float(T.detach().cpu().item()), 'device': str(dev)}


def apply_temperature(temp, y_pred):
    eps = 1e-12
    p = np.clip(np.asarray(y_pred), eps, 1 - eps)
    logit = np.log(p / (1 - p))
    T = temp['T']
    z = logit / T
    return 1.0 / (1.0 + np.exp(-z))


# =====================================================
# WF training: LGB + XGB only (production core)
# =====================================================

LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'seed': 42,
}

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'seed': 42,
    'tree_method': 'hist',
    'verbosity': 0,
}


def run_wf_fold(df, features, test_year, target='target'):
    """Run one WF fold (LGB+XGB), return OOF preds for test_year."""
    import lightgbm as lgb
    import xgboost as xgb

    ty = test_year - 2000
    train_mask = df['year'] < ty
    test_mask = df['year'] == ty

    X_tr = df.loc[train_mask, features].values
    y_tr = df.loc[train_mask, target].values
    X_te = df.loc[test_mask, features].values
    y_te = df.loc[test_mask, target].values

    print(f"  [fold {test_year}] train={len(X_tr)}, test={len(X_te)}, feats={len(features)}")
    t0 = time.time()

    # LightGBM
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
    m_lgb = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                      valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    p_lgb = m_lgb.predict(X_te)
    auc_lgb = auc_score(y_te, p_lgb)
    t_lgb = time.time() - t0

    # XGBoost
    t1 = time.time()
    dxtr = xgb.DMatrix(X_tr, label=y_tr)
    dxte = xgb.DMatrix(X_te, label=y_te)
    m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=1000,
                      evals=[(dxte, 'valid')],
                      early_stopping_rounds=50, verbose_eval=False)
    p_xgb = m_xgb.predict(dxte)
    auc_xgb = auc_score(y_te, p_xgb)
    t_xgb = time.time() - t1

    # Ensemble (AUC-weighted, mirroring production fallback when FT/IR unavailable)
    w_lgb = auc_lgb / (auc_lgb + auc_xgb)
    p_ens = w_lgb * p_lgb + (1 - w_lgb) * p_xgb
    auc_ens = auc_score(y_te, p_ens)

    print(f"    LGB AUC={auc_lgb:.4f} ({t_lgb:.1f}s), "
          f"XGB AUC={auc_xgb:.4f} ({t_xgb:.1f}s), "
          f"ENS AUC={auc_ens:.4f} (w_lgb={w_lgb:.3f})")

    return {
        'year': test_year,
        'n_train': int(len(X_tr)),
        'n_test': int(len(X_te)),
        'auc_lgb': auc_lgb,
        'auc_xgb': auc_xgb,
        'auc_ens': auc_ens,
        'w_lgb': float(w_lgb),
        't_lgb_sec': t_lgb,
        't_xgb_sec': t_xgb,
    }, p_ens, y_te, test_mask


# =====================================================
# Calibration evaluation
# =====================================================

def eval_one_method(method_name, fit_fn, apply_fn, y_train, p_train, y_test, p_test, extra_args=None):
    extra_args = extra_args or {}
    t0 = time.time()
    cal = fit_fn(y_train, p_train, **extra_args) if extra_args else fit_fn(y_train, p_train)
    fit_time = time.time() - t0

    t1 = time.time()
    p_train_cal = apply_fn(cal, p_train)
    p_test_cal = apply_fn(cal, p_test)
    apply_time = time.time() - t1

    metrics = {
        'method': method_name,
        'fit_time_sec': fit_time,
        'apply_time_sec': apply_time,
        'train': {
            'auc': auc_score(y_train, p_train_cal),
            'brier': brier_score(y_train, p_train_cal),
            'ece': ece(y_train, p_train_cal),
            'logloss': logloss(y_train, p_train_cal),
        },
        'test': {
            'auc': auc_score(y_test, p_test_cal),
            'brier': brier_score(y_test, p_test_cal),
            'ece': ece(y_test, p_test_cal),
            'logloss': logloss(y_test, p_test_cal),
        },
    }
    return metrics, p_test_cal


def eval_baseline(y_test, p_test):
    return {
        'method': 'baseline',
        'test': {
            'auc': auc_score(y_test, p_test),
            'brier': brier_score(y_test, p_test),
            'ece': ece(y_test, p_test),
            'logloss': logloss(y_test, p_test),
        },
    }


# =====================================================
# Main
# =====================================================

def main():
    import gzip

    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='Use only 2 folds (2024, 2025)')
    ap.add_argument('--years', nargs='+', type=int, default=None)
    ap.add_argument('--cache', default=os.path.join(BASE_DIR, 'data', '_v15_optuna_df_cache.pkl.gz'))
    args = ap.parse_args()

    if args.years:
        years = args.years
    elif args.quick:
        years = [2024, 2025]
    else:
        years = WF_YEARS

    print("=" * 70)
    print("  Sub-task 15: Calibration WF Re-evaluation")
    print(f"  WF years: {years}")
    print(f"  Output: {OUT_DIR}")
    print("=" * 70)

    # Load V15 cache
    print(f"\n[1/4] Loading V15 cache: {args.cache}")
    t0 = time.time()
    with gzip.open(args.cache, 'rb') as f:
        cache = pickle.load(f)
    df = cache['df']
    features = cache['features']
    print(f"  loaded in {time.time()-t0:.1f}s: shape={df.shape}, features={len(features)}")
    print(f"  target: {df['target'].value_counts().to_dict()}")

    # Run WF folds
    print(f"\n[2/4] Running WF folds (LGB+XGB)...")
    fold_results = []
    oof_records = []  # for combined OOF eval
    for ty in years:
        res, p_ens, y_te, test_mask = run_wf_fold(df, features, ty)
        fold_results.append(res)
        # store
        rid = df.loc[test_mask, 'race_id'].values if 'race_id' in df.columns else np.arange(len(p_ens))
        for i in range(len(p_ens)):
            oof_records.append({
                'year': ty,
                'race_id': int(rid[i]) if not isinstance(rid[i], str) else rid[i],
                'p_raw': float(p_ens[i]),
                'y_true': int(y_te[i]),
            })

    oof_df = pd.DataFrame(oof_records)
    oof_path = os.path.join(OUT_DIR, 'wf_oof_predictions.parquet')
    try:
        oof_df.to_parquet(oof_path, index=False)
    except Exception:
        oof_path = oof_path.replace('.parquet', '.csv')
        oof_df.to_csv(oof_path, index=False)
    print(f"\n  OOF predictions saved: {oof_path} ({len(oof_df)} rows)")

    # Per-fold calibration eval: train calibrator on year-1, eval on year
    # Strategy: train calibrator on past year predictions, apply to current year.
    # For first year, use 80/20 within-year split (proper-CV-like).
    print(f"\n[3/4] Calibration evaluation (4 methods x N folds)...")

    all_metrics = []
    # Build fold pairs: train on year[i-1] OOF, test on year[i] OOF
    # For i=0: skip (no prior year). Use within-year split.

    fold_summaries = []
    for i, ty in enumerate(years):
        cur = oof_df[oof_df['year'] == ty]
        y_te = cur['y_true'].values.astype(int)
        p_te = cur['p_raw'].values.astype(float)

        # Calibration train data: prior year OOF preferred
        if i > 0:
            prior_year = years[i-1]
            prev = oof_df[oof_df['year'] == prior_year]
            y_tr = prev['y_true'].values.astype(int)
            p_tr = prev['p_raw'].values.astype(float)
            cal_train_source = f'year={prior_year}'
        else:
            # First year: within-year 80/20 random split
            np.random.seed(42)
            idx = np.random.permutation(len(p_te))
            split = int(len(p_te) * 0.8)
            train_idx, test_idx = idx[:split], idx[split:]
            y_tr = y_te[train_idx]
            p_tr = p_te[train_idx]
            y_te = y_te[test_idx]
            p_te = p_te[test_idx]
            cal_train_source = f'year={ty} (80/20 within-year)'

        baseline = eval_baseline(y_te, p_te)
        print(f"\n  fold test_year={ty} (cal_train={cal_train_source})")
        print(f"    baseline: AUC={baseline['test']['auc']:.4f}, "
              f"Brier={baseline['test']['brier']:.4f}, "
              f"ECE={baseline['test']['ece']:.4f}, "
              f"LL={baseline['test']['logloss']:.4f}")

        methods = [
            ('platt', fit_platt, apply_platt),
            ('isotonic', fit_isotonic, apply_isotonic),
            ('beta', fit_beta, apply_beta),
            ('temperature', fit_temperature, apply_temperature),
        ]

        fold_methods = {'baseline': baseline['test']}
        for name, fit_fn, apply_fn in methods:
            m, _ = eval_one_method(name, fit_fn, apply_fn, y_tr, p_tr, y_te, p_te)
            print(f"    {name:<12}: AUC={m['test']['auc']:.4f}, "
                  f"Brier={m['test']['brier']:.4f}, "
                  f"ECE={m['test']['ece']:.4f}, "
                  f"LL={m['test']['logloss']:.4f} "
                  f"(fit {m['fit_time_sec']:.2f}s)")
            fold_methods[name] = m['test']
            m['test_year'] = ty
            m['cal_train_source'] = cal_train_source
            all_metrics.append(m)
        fold_summaries.append({
            'test_year': ty,
            'cal_train_source': cal_train_source,
            'methods': fold_methods,
        })

    # Aggregate across folds
    print(f"\n[4/4] Aggregating WF results...")
    method_names = ['baseline', 'platt', 'isotonic', 'beta', 'temperature']
    agg = {}
    for m in method_names:
        aucs, briers, eces, lls = [], [], [], []
        for fs in fold_summaries:
            d = fs['methods'][m]
            aucs.append(d['auc'])
            briers.append(d['brier'])
            eces.append(d['ece'])
            lls.append(d['logloss'])
        agg[m] = {
            'mean_auc': float(np.mean(aucs)),
            'mean_brier': float(np.mean(briers)),
            'mean_ece': float(np.mean(eces)),
            'mean_logloss': float(np.mean(lls)),
            'min_auc': float(np.min(aucs)),
            'max_auc': float(np.max(aucs)),
            'n_folds': len(aucs),
        }

    # AUC retention check (vs baseline)
    auc_base = agg['baseline']['mean_auc']
    for m in method_names:
        agg[m]['delta_auc_vs_baseline'] = agg[m]['mean_auc'] - auc_base
        agg[m]['delta_brier_vs_baseline'] = agg[m]['mean_brier'] - agg['baseline']['mean_brier']
        agg[m]['delta_ece_vs_baseline'] = agg[m]['mean_ece'] - agg['baseline']['mean_ece']

    # Save metrics JSON
    out = {
        'meta': {
            'task': 'Sub-task 15: Calibration WF re-evaluation',
            'wf_years': years,
            'cache': args.cache,
            'cache_shape': list(df.shape),
            'n_features': len(features),
            'model_used': 'LGB+XGB AUC-weighted ensemble (production core; FT/IR omitted for cost)',
            'note': 'V15 production 4-model AUC ~0.8939. LGB+XGB-only baseline AUC is documented for relative calibration ranking.',
            'generated_at': datetime.now().isoformat(),
        },
        'fold_results': fold_results,
        'fold_summaries': fold_summaries,
        'aggregate': agg,
    }
    out_json = os.path.join(OUT_DIR, 'metrics.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  metrics saved: {out_json}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"  WF Calibration Summary ({len(years)}-fold)")
    print("=" * 70)
    print(f"  {'method':<12} {'AUC':>7} {'ΔAUC':>7} {'Brier':>8} {'ΔBrier':>9} {'ECE':>7} {'ΔECE':>8}")
    for m in method_names:
        a = agg[m]
        print(f"  {m:<12} {a['mean_auc']:7.4f} {a['delta_auc_vs_baseline']:+7.4f} "
              f"{a['mean_brier']:8.4f} {a['delta_brier_vs_baseline']:+9.4f} "
              f"{a['mean_ece']:7.4f} {a['delta_ece_vs_baseline']:+8.4f}")

    print("\n[OK] Sub-task 15 WF calibration eval complete.")
    print(f"  next: python tools/v21/calibration_wf_paper_sim.py")
    return out


if __name__ == '__main__':
    main()
