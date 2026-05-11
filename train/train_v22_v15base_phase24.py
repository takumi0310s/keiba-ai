#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V22 = V15 cache (145 features) + Phase 24/26 新 features 28 件 WF 学習.

V15 (AUC 0.8939) を 真に 超える 候補 model。 V21 (V20 base ベース) は features 数 100
で V15 (145) より 劣る = 構造的 不利。 V22 は V15 全 145 features を 維持しつつ、
Phase 24/26 で 発見した jockey_trainer_combo (+21.3pt) / corner_position_delta (+10.2pt) /
exp 化 features 等 を 追加。

【V15 投資保護】 V15 .pkl.gz 完全不変。 V22 は別 file 出力、 production 投入 手動判断。

Usage:
    python train/train_v22_v15base_phase24.py --quick   # 2025 fold のみ
    python train/train_v22_v15base_phase24.py           # 6-fold WF
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
V15_CACHE = os.path.join(DATA_DIR, '_v15_optuna_df_cache.pkl.gz')
V20_DATA = os.path.join(DATA_DIR, 'v20_training_data_full.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'v22')
os.makedirs(MODEL_DIR, exist_ok=True)

# Phase 24/26 で 追加 する 新 features (V20 base に 存在、 V15 cache に 不存在)
PHASE24_26_FEATURES = [
    # Phase 24 expanding rate features
    'jockey_change_top3_rate_exp',
    'trainer_change_top3_rate_exp',
    'class_up_top3_rate_exp',
    'class_down_top3_rate_exp',
    'horse_long_layoff_top3_rate_exp',
    'horse_shorten_top3_rate_exp',
    'horse_extend_top3_rate_exp',
    'horse_surface_change_top3_rate_exp',
    'sire_overall_top3_rate_exp',
    'sire_class_down_top3_rate_exp',
    'sire_no_class_down_top3_rate_exp',
    'sire_class_down_boost_exp',
    # Phase 26 high-signal
    'jockey_trainer_combo_top3_exp',
    'corner_position_delta',
    # category flags
    'fresh_horse', 'long_layoff', 'very_long_layoff',
    'class_down', 'surface_change',
    # remark scores
    'rmk_delay', 'rmk_trouble', 'rmk_yore', 'rmk_fukure', 'rmk_contact',
    'rmk_demote', 'rmk_late_pace', 'rmk_fast_pace', 'rmk_any',
    # distance change
    'distance_change', 'distance_change_abs',
    'turf_to_dirt', 'dirt_to_turf',
]

LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_child_samples': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'verbose': -1, 'seed': 42,
}
XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'max_depth': 6, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'seed': 42, 'tree_method': 'hist',
}


def load_v15_cache():
    print(f'[INFO] loading V15 cache: {V15_CACHE}')
    with gzip.open(V15_CACHE, 'rb') as f:
        d = pickle.load(f)
    df = d['df']
    features = d['features']
    print(f'  V15 cache shape: {df.shape}, features: {len(features)}')
    return df, features


def load_v20_phase24_features():
    print(f'[INFO] loading V20 Phase 24/26 features: {V20_DATA}')
    cols_to_load = ['horse_id', 'year', 'month', 'day', 'race_num', 'umaban'] + PHASE24_26_FEATURES
    df = pd.read_csv(V20_DATA, usecols=lambda c: c in cols_to_load,
                     dtype={'horse_id': str}, low_memory=False)
    print(f'  V20 P24/26 shape: {df.shape}')
    return df


def merge_features(df15, df20):
    """V15 cache + V20 Phase 24/26 features merge (race-row level join)."""
    df15['horse_id_str'] = df15['horse_id'].astype(str)
    df20['horse_id_str'] = df20['horse_id'].astype(str)

    # composite key: year + month + day + race_num + horse_id + umaban
    key_cols = ['year', 'month', 'day', 'race_num', 'horse_id_str']
    if 'umaban' in df15.columns and 'umaban' in df20.columns:
        key_cols.append('umaban')

    # Force same dtype on merge keys (year/month/day/race_num → Int64, horse_id_str → str)
    for c in ['year', 'month', 'day', 'race_num', 'umaban']:
        if c in df15.columns:
            df15[c] = pd.to_numeric(df15[c], errors='coerce').astype('Int64')
        if c in df20.columns:
            df20[c] = pd.to_numeric(df20[c], errors='coerce').astype('Int64')
    df15['horse_id_str'] = df15['horse_id_str'].astype(str)
    df20['horse_id_str'] = df20['horse_id_str'].astype(str)

    add_cols = [c for c in PHASE24_26_FEATURES if c in df20.columns]
    df20_sub = df20[key_cols + add_cols].drop_duplicates(subset=key_cols, keep='last')

    print(f'[INFO] merging on {key_cols}, adding {len(add_cols)} features')
    merged = df15.merge(df20_sub, on=key_cols, how='left')
    print(f'  V22 merged shape: {merged.shape}')
    cov = merged[add_cols[0]].notna().mean() if add_cols else 0
    print(f'  coverage ({add_cols[0]}): {cov*100:.1f}%')
    return merged, add_cols


def train_fold(train_df, val_df, features, target='top3'):
    X_tr = train_df[features]
    y_tr = train_df[target]
    X_va = val_df[features]
    y_va = val_df[target]

    dtr = lgb.Dataset(X_tr, y_tr, free_raw_data=False)
    dva = lgb.Dataset(X_va, y_va, free_raw_data=False)
    lgb_m = lgb.train(LGB_PARAMS, dtr, num_boost_round=1000,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    p_lgb = lgb_m.predict(X_va, num_iteration=lgb_m.best_iteration)

    dtr_x = xgb.DMatrix(X_tr, label=y_tr)
    dva_x = xgb.DMatrix(X_va, label=y_va)
    xgb_m = xgb.train(XGB_PARAMS, dtr_x, num_boost_round=1000,
                      evals=[(dva_x, 'va')],
                      early_stopping_rounds=50, verbose_eval=0)
    p_xgb = xgb_m.predict(dva_x, iteration_range=(0, xgb_m.best_iteration + 1))

    auc_lgb = roc_auc_score(y_va, p_lgb)
    auc_xgb = roc_auc_score(y_va, p_xgb)
    w_lgb = auc_lgb / (auc_lgb + auc_xgb)
    p_ens = w_lgb * p_lgb + (1 - w_lgb) * p_xgb
    auc_ens = roc_auc_score(y_va, p_ens)
    return {
        'auc_lgb': auc_lgb, 'auc_xgb': auc_xgb, 'auc_ens': auc_ens,
        'w_lgb': w_lgb, 'lgb_model': lgb_m, 'xgb_model': xgb_m,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    print(f'[INFO] V22 = V15 base + Phase 24/26 features ({datetime.now().isoformat()})')
    print('[INFO] V15 production 完全不変')

    df15, v15_features = load_v15_cache()

    if 'top3' not in df15.columns:
        if 'finish' in df15.columns:
            df15['top3'] = (df15['finish'] <= 3).astype(int)
        else:
            print('[ERROR] no target col')
            return 1

    df20 = load_v20_phase24_features()
    df22, new_features = merge_features(df15, df20)

    all_features = list(v15_features) + new_features
    all_features = [f for f in all_features if f in df22.columns]
    # only numeric
    numeric_features = []
    for f in all_features:
        if df22[f].dtype.kind in 'iufb':
            numeric_features.append(f)
    print(f'[INFO] V22 numeric features: {len(numeric_features)} (V15: {len(v15_features)} + new: {len(new_features)})')

    df22 = df22.dropna(subset=['top3', 'year'])
    df22['year'] = df22['year'].astype(int)

    yr_max = int(df22['year'].max())
    if yr_max < 100:
        folds = [(25, 25)] if args.quick else [
            (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)]
    else:
        folds = [(2025, 2025)] if args.quick else [
            (2020, 2020), (2021, 2021), (2022, 2022),
            (2023, 2023), (2024, 2024), (2025, 2025)]

    results = []
    final_models = None
    for y_lo, y_hi in folds:
        train_df = df22[df22['year'] < y_lo]
        val_df = df22[(df22['year'] >= y_lo) & (df22['year'] <= y_hi)]
        if len(train_df) < 1000 or len(val_df) < 100:
            print(f'[SKIP] fold {y_lo}: train={len(train_df)} val={len(val_df)}')
            continue
        print(f'\n=== Fold val={y_lo} (train={len(train_df)}, val={len(val_df)}) ===')
        t0 = time.time()
        r = train_fold(train_df, val_df, numeric_features)
        dt = time.time() - t0
        print(f'  LGB={r["auc_lgb"]:.4f} XGB={r["auc_xgb"]:.4f} ENS={r["auc_ens"]:.4f} '
              f'(w_lgb={r["w_lgb"]:.2f}) [{dt:.0f}s]')
        results.append({
            'val_year': y_lo,
            'auc_lgb': r['auc_lgb'], 'auc_xgb': r['auc_xgb'],
            'auc_ens': r['auc_ens'], 'w_lgb': r['w_lgb'],
            'n_train': len(train_df), 'n_val': len(val_df),
        })
        final_models = r

    print(f'\n=== V22 WF summary ===')
    if results:
        aucs = [r['auc_ens'] for r in results]
        mean = np.mean(aucs)
        print(f'  mean ENS AUC: {mean:.4f}')
        print(f'  V15 baseline: 0.8939')
        print(f'  delta: {mean - 0.8939:+.4f}')
        print(f'  per-year: ' + ', '.join(f'{r["val_year"]}={r["auc_ens"]:.4f}' for r in results))

    if final_models:
        out_path = os.path.join(BASE_DIR, 'keiba_model_v22_central.pkl.gz')
        with gzip.open(out_path, 'wb') as f:
            pickle.dump({
                'version': 'v22',
                'date': datetime.now().isoformat(),
                'features': numeric_features,
                'lgb_model': final_models['lgb_model'],
                'xgb_model': final_models['xgb_model'],
                'w_lgb': final_models['w_lgb'],
                'wf_results': results,
            }, f)
        print(f'[OK] model saved: {out_path}')

        with open(os.path.join(MODEL_DIR, 'wf_summary.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'date': datetime.now().isoformat(),
                'n_features': len(numeric_features),
                'wf_results': results,
                'mean_auc_ens': float(np.mean([r['auc_ens'] for r in results])) if results else None,
                'v15_baseline_auc': 0.8939,
            }, f, indent=2, ensure_ascii=False)

    return 0


if __name__ == '__main__':
    sys.exit(main())
