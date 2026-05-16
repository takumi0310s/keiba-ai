"""V15 teacher → V22 student distillation で V15 越え 試行.

Distillation flow:
1. V15 (teacher) で 全 race の soft label (prob) 生成
2. V22 student を hard label (target=top3) + soft label (V15 prob) で学習
   loss = α * BCE(student, target) + (1-α) * MSE(student, V15_prob)
3. WF 6-fold で V22 student の AUC を 測定
4. V15 alone と 比較

理論:
- V15 が 強い patterns を soft label として transfer
- student は V22 features の 追加 information を活用
- 期待: V15 alone より +0.005-0.010 AUC

V15 投資保護 完全: V15 .pkl.gz / predict_core / app.py 完全不変。
V22 distilled は別 file。

usage:
    python train/train_v22_distillation.py --quick   # fold 25 only
    python train/train_v22_distillation.py           # full WF
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'train'))

from train_v22_4ensemble import LGB_PARAMS, XGB_PARAMS
from train_v22_enhanced import load_v22_enhanced_data
from train_v22_enhanced_top100 import load_top_n_features

V15_MODEL = BASE / 'keiba_model_v15_central_live.pkl.gz'
V15_CACHE = BASE / 'data' / '_v15_optuna_df_cache.pkl.gz'
MODEL_DIR = BASE / 'models' / 'v22_distilled'
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_v15_predictions(df, cache_features):
    """V15 LGB+XGB ensemble で 全 row predict (in-sample 注意)."""
    print('Loading V15 model ...')
    with gzip.open(V15_MODEL, 'rb') as f:
        v15 = pickle.load(f)
    v15_lgb = v15['model']
    v15_xgb = v15['xgb_model']
    ens_w = v15.get('ensemble_weights', {'lgb': 0.5, 'xgb': 0.5})

    print(f'V15 predict on {len(df):,} rows ...')
    X = df[cache_features].fillna(0).astype(np.float32).values
    p_lgb = v15_lgb.predict(X)
    dmat = xgb.DMatrix(X)
    p_xgb = v15_xgb.predict(dmat)
    p = ens_w['lgb'] * p_lgb + ens_w['xgb'] * p_xgb
    return p


def train_lgb_distilled(df_tr, df_va, features, target_col='target', soft_label_col='v15_pred',
                        alpha=0.5):
    """LGB with distillation: hard target + soft label を combine."""
    X_tr = df_tr[features].astype(np.float32).values
    y_tr = df_tr[target_col].values.astype(np.float32)
    soft_tr = df_tr[soft_label_col].values.astype(np.float32)
    X_va = df_va[features].astype(np.float32).values
    y_va = df_va[target_col].values

    # distilled target: alpha * hard + (1-alpha) * soft
    combined_target = alpha * y_tr + (1 - alpha) * soft_tr

    # LGB regression-style (binary だが combined_target は continuous)
    params = dict(LGB_PARAMS)
    params['objective'] = 'regression'  # MSE
    params['metric'] = 'mse'

    train_set = lgb.Dataset(X_tr, label=combined_target)
    val_set = lgb.Dataset(X_va, label=df_va[soft_label_col].values * (1-alpha) + y_va.astype(np.float32) * alpha,
                          reference=train_set)
    model = lgb.train(params, train_set, num_boost_round=1000,
                       valid_sets=[val_set], valid_names=['val'],
                       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    p_va = model.predict(X_va)
    auc = roc_auc_score(y_va, p_va)
    return model, p_va, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--alpha', type=float, default=0.5,
                    help='hard label weight (1-alpha = soft label)')
    args = ap.parse_args()

    print('=' * 60)
    print(f'V22 Distillation (V15 teacher → V22 student、 alpha={args.alpha})')
    print('=' * 60)
    t0 = time.time()

    # V22 enhanced data load
    df, all_features = load_v22_enhanced_data()
    top_100 = load_top_n_features(100)
    available = [f for f in top_100 if f in df.columns]
    print(f'features (top 100 V22): {len(available)}')

    # V15 cache features
    with gzip.open(V15_CACHE, 'rb') as f:
        cache = pickle.load(f)
    cache_features = cache['features']

    # V15 predictions for all rows (soft labels)
    df['v15_pred'] = get_v15_predictions(df, cache_features)
    print(f'V15 pred mean: {df["v15_pred"].mean():.4f}, std: {df["v15_pred"].std():.4f}')

    # WF
    folds = [(25, 25)] if args.quick else [(20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)]
    results = []

    for y_lo, y_hi in folds:
        print(f'\n=== fold {y_lo}-{y_hi} ===')
        train_mask = df['year'] < y_lo
        test_mask = (df['year'] >= y_lo) & (df['year'] <= y_hi)
        df_tr = df[train_mask].copy()
        df_te = df[test_mask].copy()
        print(f'  train: {len(df_tr):,}, test: {len(df_te):,}')

        if len(df_tr) < 1000 or len(df_te) < 100:
            continue

        # Baseline LGB (no distill)
        train_set = lgb.Dataset(df_tr[available].astype(np.float32).values,
                                 label=df_tr['target'].values)
        val_set = lgb.Dataset(df_te[available].astype(np.float32).values,
                               label=df_te['target'].values, reference=train_set)
        lgb_baseline = lgb.train(LGB_PARAMS, train_set, num_boost_round=1000,
                                  valid_sets=[val_set], valid_names=['val'],
                                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        p_baseline = lgb_baseline.predict(df_te[available].astype(np.float32).values)
        auc_baseline = roc_auc_score(df_te['target'].values, p_baseline)
        print(f'  baseline V22 LGB: {auc_baseline:.4f}')

        # Distilled LGB
        _, p_distilled, auc_distilled = train_lgb_distilled(
            df_tr, df_te, available, target_col='target',
            soft_label_col='v15_pred', alpha=args.alpha)
        print(f'  distilled (α={args.alpha}): {auc_distilled:.4f}')

        # V15 alone on this fold
        p_v15_fold = df_te['v15_pred'].values
        auc_v15 = roc_auc_score(df_te['target'].values, p_v15_fold)
        print(f'  V15 alone (in-sample on test): {auc_v15:.4f}')

        # Ensemble: distilled + V15
        for w in [0.0, 0.3, 0.5, 0.7]:
            p_ens = w * p_distilled + (1-w) * p_v15_fold
            auc_ens = roc_auc_score(df_te['target'].values, p_ens)
            print(f'  ens (distilled w={w:.1f} + V15 w={1-w:.1f}): {auc_ens:.4f}')

        results.append({
            'fold': f'{y_lo}-{y_hi}',
            'baseline': float(auc_baseline),
            'distilled': float(auc_distilled),
            'v15_alone': float(auc_v15),
            'delta_distilled': float(auc_distilled - auc_baseline),
        })

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    if results:
        mean_baseline = np.mean([r['baseline'] for r in results])
        mean_distilled = np.mean([r['distilled'] for r in results])
        mean_v15 = np.mean([r['v15_alone'] for r in results])
        print(f'mean baseline V22 LGB:  {mean_baseline:.4f}')
        print(f'mean distilled (α={args.alpha}): {mean_distilled:.4f}')
        print(f'mean V15 (in-sample test): {mean_v15:.4f}')
        print(f'mean delta distilled - baseline: {mean_distilled - mean_baseline:+.4f}')

    out_path = MODEL_DIR / f'distill_alpha{args.alpha}_{datetime.now():%Y%m%d_%H%M%S}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'alpha': args.alpha,
            'mode': 'quick' if args.quick else 'full',
            'results': results,
            'elapsed_s': time.time() - t0,
        }, f, ensure_ascii=False, indent=2)
    print(f'\nsaved: {out_path}')


if __name__ == '__main__':
    main()
