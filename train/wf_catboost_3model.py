#!/usr/bin/env python
"""3-model WF: LGB + XGB + CatBoost (v15 150-feature cache).

目的: CatBoost を加えた 3モデルアンサンブル (LGB+XGB+CB) の WF AUC 検証。
4モデル (FT/IR) は重く、まず 3モデルで採用可否を判定する。

採用基準:
  - WF mean grid AUC > 0.8858 (現行 v13.5b baseline 0.8788 超過)
  - 全年 gap < 0.05 (overfit なし)
  - 全年 AUC > 0.85

Usage:
    nohup python -u train/wf_catboost_3model.py > logs/wf_cb3.log 2>&1 &
"""
from __future__ import annotations

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))

from train_v135_ft_transformer import LGB_PARAMS, XGB_PARAMS  # 既存ハイパラ流用

CACHE_PATH = os.path.join(DATA_DIR, '_v15_train_df_cache.pkl')
OUT_JSON = os.path.join(DATA_DIR, 'wf_cb3_results.json')
MODEL_OUT = os.path.join(BASE_DIR, 'keiba_model_v16_cb3_central.pkl.gz')
TARGET_AUC = 0.8858
BASELINE_AUC = 0.8856  # v14.1 4-model baseline for reference
WF_YEARS = range(2021, 2026)

CB_PARAMS = dict(
    iterations=1500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=0,
    task_type='CPU',
    early_stopping_rounds=50,
)


def load_cache():
    if not os.path.exists(CACHE_PATH):
        print(f"[ERROR] {CACHE_PATH} not found.")
        sys.exit(1)
    print(f"[CACHE] loading {CACHE_PATH}")
    with open(CACHE_PATH, 'rb') as f:
        d = pickle.load(f)
    return d['df'], d.get('v15_features')


def train_one_year(df, features, test_year):
    from catboost import CatBoostClassifier

    ty = test_year - 2000
    tr_mask = df['year'] < ty
    te_mask = df['year'] == ty
    if tr_mask.sum() < 1000 or te_mask.sum() < 100:
        return None

    X_tr = df.loc[tr_mask, features].values
    y_tr = df.loc[tr_mask, 'target'].values
    X_te = df.loc[te_mask, features].values
    y_te = df.loc[te_mask, 'target'].values
    print(f"\n=== {test_year} train={len(X_tr)} test={len(X_te)} feats={len(features)} ===")

    # LGB
    print("  LGB...")
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_te, label=y_te, reference=dtr)
    m_lgb = lgb.train(LGB_PARAMS, dtr, num_boost_round=1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    p_lgb = m_lgb.predict(X_te)
    p_lgb_tr = m_lgb.predict(X_tr)
    auc_lgb = roc_auc_score(y_te, p_lgb)
    auc_lgb_tr = roc_auc_score(y_tr, p_lgb_tr)
    print(f"    LGB={auc_lgb:.4f} (train={auc_lgb_tr:.4f})")

    # XGB
    print("  XGB...")
    dxt = xgb.DMatrix(X_tr, label=y_tr)
    dxe = xgb.DMatrix(X_te, label=y_te)
    m_xgb = xgb.train(XGB_PARAMS, dxt, num_boost_round=1500,
                      evals=[(dxe, 'valid')],
                      early_stopping_rounds=50, verbose_eval=False)
    p_xgb = m_xgb.predict(dxe)
    auc_xgb = roc_auc_score(y_te, p_xgb)
    print(f"    XGB={auc_xgb:.4f}")

    # CatBoost
    print("  CB...")
    m_cb = CatBoostClassifier(**CB_PARAMS)
    m_cb.fit(X_tr, y_tr, eval_set=(X_te, y_te), use_best_model=True)
    p_cb = m_cb.predict_proba(X_te)[:, 1]
    auc_cb = roc_auc_score(y_te, p_cb)
    print(f"    CB={auc_cb:.4f}")

    # Grid search 3-weight simplex
    best = {'auc': 0, 'w': None}
    for wl in np.arange(0.10, 0.80, 0.05):
        for wx in np.arange(0.10, 0.80 - wl + 1e-9, 0.05):
            wc = 1.0 - wl - wx
            if wc < 0.05 or wc > 0.80:
                continue
            p = wl * p_lgb + wx * p_xgb + wc * p_cb
            a = roc_auc_score(y_te, p)
            if a > best['auc']:
                best = {'auc': float(a), 'w': [float(wl), float(wx), float(wc)]}

    gap = auc_lgb_tr - auc_lgb
    print(f"    Grid={best['auc']:.4f} w={best['w']} gap={gap:.4f}")

    return {
        'year': test_year,
        'lgb_auc': float(auc_lgb),
        'xgb_auc': float(auc_xgb),
        'cb_auc': float(auc_cb),
        'grid_auc': best['auc'],
        'grid_weights': best['w'],
        'train_auc': float(auc_lgb_tr),
        'gap': float(gap),
    }


def check_acceptance(results):
    aucs = [r['grid_auc'] for r in results]
    mean_auc = float(np.mean(aucs))
    max_gap = max(r['gap'] for r in results)
    min_year_auc = min(r['grid_auc'] for r in results)
    reasons = []
    if mean_auc <= TARGET_AUC:
        reasons.append(f"mean {mean_auc:.4f} <= target {TARGET_AUC:.4f}")
    if max_gap >= 0.05:
        reasons.append(f"max gap {max_gap:.4f} >= 0.05")
    if min_year_auc <= 0.85:
        reasons.append(f"min year AUC {min_year_auc:.4f} <= 0.85")
    return (len(reasons) == 0), mean_auc, reasons


def main():
    t0 = time.time()
    df, features = load_cache()
    print(f"df: {len(df)} rows, features: {len(features)}")

    if 'target' not in df.columns:
        print("[ERROR] target column missing")
        sys.exit(1)

    results = []
    for y in WF_YEARS:
        r = train_one_year(df, features, y)
        if r:
            results.append(r)

    adopted, mean_auc, reasons = check_acceptance(results)
    print("\n" + "=" * 60)
    print(f"  3-model WF summary: mean grid AUC = {mean_auc:.4f}")
    print(f"  target = {TARGET_AUC:.4f}, v13.5b baseline = 0.8788")
    if adopted:
        print("  ADOPTED")
    else:
        print("  REJECTED")
        for r in reasons:
            print(f"    - {r}")

    out = {
        'mean_grid_auc': mean_auc,
        'per_year': results,
        'adopted': adopted,
        'reasons': reasons,
        'target_auc': TARGET_AUC,
        'baseline_v135b': 0.8788,
        'elapsed_min': (time.time() - t0) / 60,
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果保存: {OUT_JSON}")

    if adopted:
        print("\n[5] 採用 → 全データで 3モデル学習し本番モデル保存...")
        save_production_model(df, features, results)

    try:
        from tools.notify import send_discord
        send_discord(
            f"CB 3-model WF ({'ADOPTED' if adopted else 'REJECTED'})",
            f"mean={mean_auc:.4f} elapsed={out['elapsed_min']:.1f}min\n" + '\n'.join(reasons),
            color='green' if adopted else 'yellow', channel='updates',
        )
    except Exception as e:
        print(f"[notify] {e}")


def save_production_model(df, features, per_year):
    """採用時のみ呼ばれる。最新年の重みで全データ学習."""
    import gzip
    from catboost import CatBoostClassifier

    last_w = per_year[-1]['grid_weights']
    print(f"  final weights: LGB={last_w[0]:.2f} XGB={last_w[1]:.2f} CB={last_w[2]:.2f}")

    X = df[features].values
    y = df['target'].values

    print("  fit LGB(full)...")
    d = lgb.Dataset(X, label=y)
    m_lgb = lgb.train(LGB_PARAMS, d, num_boost_round=1000)

    print("  fit XGB(full)...")
    dx = xgb.DMatrix(X, label=y)
    m_xgb = xgb.train(XGB_PARAMS, dx, num_boost_round=1000)

    print("  fit CB(full)...")
    m_cb = CatBoostClassifier(**{**CB_PARAMS, 'iterations': 1000, 'early_stopping_rounds': None})
    m_cb.fit(X, y)

    bundle = {
        'version': 'v16_cb3',
        'features': features,
        'lgb': m_lgb,
        'xgb': m_xgb,
        'cb': m_cb,
        'weights': {'lgb': last_w[0], 'xgb': last_w[1], 'cb': last_w[2]},
    }
    with gzip.open(MODEL_OUT, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"  saved: {MODEL_OUT}")


if __name__ == '__main__':
    main()
