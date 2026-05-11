#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V21 LGB+XGB WF training (V20 base + new high-signal features + video infra).

V15 production を 完全 不変 で V21 候補 を 並行 学習。
v21_training_data_full.csv (101 tabular + 33 video=sparse) を 使用。
video features は 0.03% coverage と sparse、 LGB は NaN tolerant なので 含めて学習し
将来 coverage 拡大時に 自動 活性化。

【V15 投資保護】 V15 .pkl.gz / predict_core / daily_predict 完全 不変。
本 model は models/v21/ 出力、 production 投入は 手動判断。

Usage:
    python train/train_v21_lgb_xgb.py            # WF 6-fold
    python train/train_v21_lgb_xgb.py --quick    # 単 fold (2025 only)
    python train/train_v21_lgb_xgb.py --no-video # video features 除外 (cleaner baseline)
"""
import argparse
import gzip
import json
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
V21_DATA = os.path.join(DATA_DIR, 'v21_training_data_full.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'v21')
os.makedirs(MODEL_DIR, exist_ok=True)

EXCLUDE_COLS = {
    # target
    'finish', 'finish2', 'top3',
    # ID
    'horse_name', 'jockey', 'trainer', 'jockey_id', 'trainer_id',
    'race_name', 'race_id', 'horse_id', 'horse_id_str', 'horse_id_int', 'horse_id_v21',
    # ★ post-race result LEAKS (V20 data includes these — STRICT exclusion) ★
    'prize', 'abnormal_code', 'time_margin', 'run_time',
    'agari_3f', 'pass1', 'pass2', 'pass3', 'pass4',
    'order_diff', 'finish_time', 'last3f_official',
    'popularity',   # finalized popularity = post-betting close
    # day-of leaks (Pattern A strict)
    'odds_log', 'horse_weight', 'condition_enc',
    'weight_change', 'weight_change_abs', 'weight_cat',
    'weight_cat_dist', 'cond_surface',
    # SKB post-race leaks (Session #38)
    'skb_kishi_code_1', 'skb_kishi_code_2', 'skb_kishi_code_3',
    'skb_baba_code_1', 'skb_baba_code_2', 'skb_baba_code_3',
    'skb_kyaku_code_1', 'skb_kyaku_code_2', 'skb_kyaku_code_3',
    'skb_turf_hoof',
}

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
}


def build_feature_list(df, include_video=True):
    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if c.startswith('video_') and not include_video:
            continue
        if df[c].dtype == 'object':
            continue
        cols.append(c)
    return cols


def encode_object_cols(df, fit=False, encoders=None):
    """object 列を simple label encode (train fit / val transform)."""
    if encoders is None:
        encoders = {}
    for c in df.columns:
        if df[c].dtype != 'object':
            continue
        if c in {'horse_name', 'jockey', 'trainer', 'race_name', 'horse_id', 'race_id'}:
            continue
        if fit:
            vals = df[c].fillna('NA').astype(str)
            uniq = pd.unique(vals)
            encoders[c] = {v: i for i, v in enumerate(uniq)}
        if c in encoders:
            df[c] = df[c].fillna('NA').astype(str).map(encoders[c]).fillna(-1).astype('int32')
    return df, encoders


def train_one_fold(train_df, val_df, features, target='top3'):
    """LGB + XGB 学習 + val AUC 返却."""
    X_tr = train_df[features]
    y_tr = train_df[target]
    X_va = val_df[features]
    y_va = val_df[target]

    # LGB
    dtr = lgb.Dataset(X_tr, y_tr, free_raw_data=False)
    dva = lgb.Dataset(X_va, y_va, free_raw_data=False)
    lgb_m = lgb.train(
        LGB_PARAMS, dtr,
        num_boost_round=1000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    p_lgb = lgb_m.predict(X_va, num_iteration=lgb_m.best_iteration)

    # XGB
    dtr_x = xgb.DMatrix(X_tr, label=y_tr)
    dva_x = xgb.DMatrix(X_va, label=y_va)
    xgb_m = xgb.train(
        XGB_PARAMS, dtr_x,
        num_boost_round=1000,
        evals=[(dva_x, 'va')],
        early_stopping_rounds=50,
        verbose_eval=0,
    )
    p_xgb = xgb_m.predict(dva_x, iteration_range=(0, xgb_m.best_iteration + 1))

    auc_lgb = roc_auc_score(y_va, p_lgb)
    auc_xgb = roc_auc_score(y_va, p_xgb)
    # AUC-weighted ensemble
    w_lgb = auc_lgb / (auc_lgb + auc_xgb)
    w_xgb = 1 - w_lgb
    p_ens = w_lgb * p_lgb + w_xgb * p_xgb
    auc_ens = roc_auc_score(y_va, p_ens)

    return {
        'auc_lgb': auc_lgb,
        'auc_xgb': auc_xgb,
        'auc_ens': auc_ens,
        'w_lgb': w_lgb,
        'w_xgb': w_xgb,
        'lgb_model': lgb_m,
        'xgb_model': xgb_m,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='2025 only single fold')
    ap.add_argument('--no-video', action='store_true', help='exclude video features')
    args = ap.parse_args()

    print(f'[INFO] V21 LGB+XGB training start ({datetime.now().isoformat()})')
    print(f'[INFO] V15 production: 完全不変')

    print(f'[INFO] loading {V21_DATA}...')
    df = pd.read_csv(V21_DATA, low_memory=False)
    print(f'  shape: {df.shape}')

    df = df.dropna(subset=['top3', 'year'])
    df['top3'] = df['top3'].astype(int)
    df['year'] = df['year'].astype(int)

    include_video = not args.no_video
    features = build_feature_list(df, include_video=include_video)
    print(f'[INFO] features: {len(features)} (video: {"included" if include_video else "excluded"})')

    df, _ = encode_object_cols(df, fit=True)
    features = [f for f in features if f in df.columns and df[f].dtype.kind in 'iufb']
    print(f'[INFO] numeric features after encoding: {len(features)}')

    # year col in v21 data is 2-digit (20=2020...)
    yr_max = int(df['year'].max())
    if yr_max < 100:
        folds = [(25, 25)] if args.quick else [(20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)]
    else:
        folds = [(2025, 2025)] if args.quick else [
            (2020, 2020), (2021, 2021), (2022, 2022),
            (2023, 2023), (2024, 2024), (2025, 2025),
        ]

    results = []
    final_models = None

    for y_va_lo, y_va_hi in folds:
        train_mask = df['year'] < y_va_lo
        val_mask = (df['year'] >= y_va_lo) & (df['year'] <= y_va_hi)
        train_df = df[train_mask]
        val_df = df[val_mask]
        if len(train_df) < 1000 or len(val_df) < 100:
            print(f'[SKIP] fold {y_va_lo}: train={len(train_df)} val={len(val_df)}')
            continue
        print(f'\n=== Fold val={y_va_lo} (train={len(train_df)}, val={len(val_df)}) ===')
        t0 = time.time()
        r = train_one_fold(train_df, val_df, features)
        dt = time.time() - t0
        print(f'  LGB={r["auc_lgb"]:.4f} XGB={r["auc_xgb"]:.4f} ENS={r["auc_ens"]:.4f} '
              f'(w_lgb={r["w_lgb"]:.2f}) [{dt:.0f}s]')
        results.append({
            'val_year': y_va_lo,
            'auc_lgb': r['auc_lgb'],
            'auc_xgb': r['auc_xgb'],
            'auc_ens': r['auc_ens'],
            'w_lgb': r['w_lgb'],
            'n_train': len(train_df),
            'n_val': len(val_df),
        })
        final_models = r

    print(f'\n=== V21 WF summary ===')
    if results:
        aucs = [r['auc_ens'] for r in results]
        print(f'  mean ENS AUC: {np.mean(aucs):.4f}')
        print(f'  per-year ENS: ' + ', '.join(f'{r["val_year"]}={r["auc_ens"]:.4f}' for r in results))

    # Save model
    if final_models:
        model_pkg = {
            'version': 'v21',
            'date': datetime.now().isoformat(),
            'features': features,
            'lgb_model': final_models['lgb_model'],
            'xgb_model': final_models['xgb_model'],
            'w_lgb': final_models['w_lgb'],
            'w_xgb': final_models['w_xgb'],
            'wf_results': results,
            'include_video': include_video,
        }
        suffix = '_no_video' if not include_video else ''
        out_path = os.path.join(BASE_DIR, f'keiba_model_v21_central{suffix}.pkl.gz')
        with gzip.open(out_path, 'wb') as f:
            pickle.dump(model_pkg, f)
        print(f'[OK] model saved: {out_path}')

        # WF summary JSON
        summary_path = os.path.join(MODEL_DIR, f'wf_summary{suffix}.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'date': datetime.now().isoformat(),
                'include_video': include_video,
                'n_features': len(features),
                'wf_results': results,
                'mean_auc_ens': float(np.mean([r['auc_ens'] for r in results])) if results else None,
            }, f, indent=2, ensure_ascii=False)
        print(f'[OK] summary: {summary_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
