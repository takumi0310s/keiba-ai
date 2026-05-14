"""LGB importance から top N features を 選別.

V22 base or enhanced から LGB を 1 fold 学習 → importance gain でソート → top N output。
V22 enhanced (282 features) を top 100 程度に絞ることで:
- CUDA OOM 回避 (FT 復活)
- IR 安定化 (noise 削減)
- V15 越え 試行

usage:
    python train/select_top_features.py [--top-n 100] [--fold-year 24]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'train'))

V15_CACHE = BASE / 'data' / '_v15_optuna_df_cache.pkl.gz'
V20_DATA = BASE / 'data' / 'v20_training_data_full.csv'
MERGED_FEATURES = BASE / 'data' / 'features_merged_all.csv'
OUT_DIR = BASE / 'data'

LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_child_samples': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'verbose': -1, 'seed': 42,
}

PHASE24_26_FEATURES = [
    'jockey_change_top3_rate_exp', 'trainer_change_top3_rate_exp',
    'class_up_top3_rate_exp', 'class_down_top3_rate_exp',
    'horse_long_layoff_top3_rate_exp', 'horse_shorten_top3_rate_exp',
    'horse_extend_top3_rate_exp', 'horse_surface_change_top3_rate_exp',
    'sire_overall_top3_rate_exp', 'sire_class_down_top3_rate_exp',
    'sire_no_class_down_top3_rate_exp', 'sire_class_down_boost_exp',
    'jockey_trainer_combo_top3_exp', 'corner_position_delta',
    'fresh_horse', 'long_layoff', 'very_long_layoff',
    'class_down', 'surface_change',
    'rmk_delay', 'rmk_trouble', 'rmk_yore', 'rmk_fukure', 'rmk_contact',
    'rmk_demote', 'rmk_late_pace', 'rmk_fast_pace', 'rmk_any',
    'distance_change', 'distance_change_abs',
    'turf_to_dirt', 'dirt_to_turf',
]


def load_data():
    print(f'[INFO] loading V15 cache + V20 P24/26 + features_merged_all')
    with gzip.open(V15_CACHE, 'rb') as f:
        d = pickle.load(f)
    df15 = d['df']
    v15_features = d['features']
    print(f'  V15: {df15.shape}, features: {len(v15_features)}')

    if 'top3' not in df15.columns and 'finish' in df15.columns:
        df15['top3'] = (df15['finish'] <= 3).astype(int)
    if 'target' not in df15.columns:
        df15['target'] = df15['top3']
    df15['horse_id_str'] = df15['horse_id'].astype(str)
    for c in ['year', 'month', 'day', 'race_num', 'umaban']:
        df15[c] = pd.to_numeric(df15[c], errors='coerce').astype('Int64')

    df20 = pd.read_csv(V20_DATA, usecols=lambda c: c in ['horse_id', 'year', 'month',
                                                          'day', 'race_num', 'umaban'] + PHASE24_26_FEATURES,
                       dtype={'horse_id': str}, low_memory=False)
    df20['horse_id_str'] = df20['horse_id'].astype(str)
    for c in ['year', 'month', 'day', 'race_num', 'umaban']:
        df20[c] = pd.to_numeric(df20[c], errors='coerce').astype('Int64')
    key = ['year', 'month', 'day', 'race_num', 'horse_id_str', 'umaban']
    add_p24 = [c for c in PHASE24_26_FEATURES if c in df20.columns]
    df20_sub = df20[key + add_p24].drop_duplicates(subset=key, keep='last')
    df = df15.merge(df20_sub, on=key, how='left')

    df_extra = pd.read_csv(MERGED_FEATURES, encoding='utf-8-sig', low_memory=False)
    df_extra['race_id'] = df_extra['race_id'].astype(str)
    df_extra['umaban'] = pd.to_numeric(df_extra['umaban'], errors='coerce').astype('Int64')
    df_extra = df_extra.drop_duplicates(['race_id', 'umaban'], keep='last')
    extra_cols = [c for c in df_extra.columns if c not in ('race_id', 'horse_id', 'umaban')]

    df['race_id'] = df['race_id'].astype(str)
    df = df.merge(df_extra[['race_id', 'umaban'] + extra_cols],
                  on=['race_id', 'umaban'], how='left')
    print(f'  merged: {df.shape}')

    all_features = list(v15_features) + add_p24 + extra_cols
    numeric = [f for f in all_features if f in df.columns and df[f].dtype.kind in 'iufb']
    df = df.dropna(subset=['target', 'year'])
    df['year'] = df['year'].astype(int)
    df[numeric] = df[numeric].fillna(0)

    return df, numeric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top-n', type=int, default=100)
    ap.add_argument('--fold-year', type=int, default=24,
                    help='train year < this, test year == this')
    ap.add_argument('--output', default=str(OUT_DIR / 'top_features_v22enh.json'))
    args = ap.parse_args()

    df, features = load_data()
    print(f'[INFO] total features: {len(features)}')

    train_mask = df['year'] < args.fold_year
    test_mask = df['year'] == args.fold_year
    df_tr = df[train_mask]
    df_te = df[test_mask]
    print(f'[INFO] train: {len(df_tr):,}, test: {len(df_te):,}')

    X_tr = df_tr[features].astype(np.float32).values
    y_tr = df_tr['target'].values
    X_te = df_te[features].astype(np.float32).values
    y_te = df_te['target'].values

    print('[INFO] training LGB...')
    t0 = time.time()
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_te, label=y_te, reference=train_set)
    model = lgb.train(LGB_PARAMS, train_set, num_boost_round=1000,
                       valid_sets=[val_set], valid_names=['val'],
                       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    p = model.predict(X_te)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_te, p)
    print(f'[INFO] LGB AUC: {auc:.4f} [{time.time()-t0:.0f}s]')

    importance = model.feature_importance(importance_type='gain')
    pairs = sorted(zip(features, importance), key=lambda x: -x[1])
    top_n = pairs[:args.top_n]

    out = {
        'top_n': args.top_n,
        'fold_year': args.fold_year,
        'lgb_auc': float(auc),
        'total_features': len(features),
        'top_features': [{'name': n, 'gain': float(g)} for n, g in top_n],
        'all_features_with_zero_gain': [n for n, g in pairs if g == 0],
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n[INFO] saved {args.output}')

    print(f'\n=== TOP {min(20, args.top_n)} ===')
    for i, (n, g) in enumerate(top_n[:20]):
        print(f'  {i+1:>3}. {n}: gain={g:.0f}')

    n_zero = sum(1 for _, g in pairs if g == 0)
    print(f'\n[stats] {n_zero}/{len(features)} features have zero gain (no use)')


if __name__ == '__main__':
    main()
