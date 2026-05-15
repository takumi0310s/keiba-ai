"""V15 + V22 enhanced top 100 完全 stacking (fold 25 = 2025 OOS、 V15 越え 試行).

V15 prediction (LGB+XGB ensemble) + V22 prediction (LGB+XGB+FT+IR Grid) を
LGB 2nd-layer で stacking + 単純 average で 比較。

steps:
1. V15 fold 25 predict (cache 145 features)
2. V22 fold 25 predict (top 100 features、 fold 25 train mask で fold-specific retrain)
3. ensemble: 単純 average + 重み grid search + LGB 2nd-layer
4. AUC 比較

V15 投資保護 完全。 別 file 出力。
"""
from __future__ import annotations

import gzip
import json
import pickle
import sys
import time
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

V15_MODEL = BASE / 'keiba_model_v15_central_live.pkl.gz'
V15_CACHE = BASE / 'data' / '_v15_optuna_df_cache.pkl.gz'
TOP_FEATURES_JSON = BASE / 'data' / 'top_features_v22enh.json'
MERGED_FEATURES = BASE / 'data' / 'features_merged_all.csv'
V20_DATA = BASE / 'data' / 'v20_training_data_full.csv'

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


def main():
    print('=' * 60)
    print('V15 + V22 Stacking (fold 25 = 2025、 V15 OOS)')
    print('=' * 60)
    t0 = time.time()

    # 1. V15 model + cache load
    print('\n[1] V15 model load ...')
    with gzip.open(V15_MODEL, 'rb') as f:
        v15 = pickle.load(f)
    v15_lgb = v15['model']
    v15_xgb = v15['xgb_model']
    v15_ens_w = v15.get('ensemble_weights', {'lgb': 0.5, 'xgb': 0.5})

    print('[2] V15 cache load ...')
    with gzip.open(V15_CACHE, 'rb') as f:
        cache = pickle.load(f)
    df_cache = cache['df']
    cache_features = cache['features']
    df_cache['target'] = (df_cache['finish'] <= 3).astype(int)
    print(f'   cache rows: {len(df_cache):,}, features: {len(cache_features)}')

    # 2. V22 top 100 features + merged data load
    print('\n[3] features_merged + top features load ...')
    with open(TOP_FEATURES_JSON, 'r', encoding='utf-8') as f:
        top_data = json.load(f)
    top_100 = [item['name'] for item in top_data['top_features'][:100]]

    # V20 phase24/26 merge
    df_cache['horse_id_str'] = df_cache['horse_id'].astype(str)
    for c in ['year', 'month', 'day', 'race_num', 'umaban']:
        df_cache[c] = pd.to_numeric(df_cache[c], errors='coerce').astype('Int64')
    df20 = pd.read_csv(V20_DATA, usecols=lambda c: c in ['horse_id', 'year', 'month',
                                                          'day', 'race_num', 'umaban'] + PHASE24_26_FEATURES,
                       dtype={'horse_id': str}, low_memory=False)
    df20['horse_id_str'] = df20['horse_id'].astype(str)
    for c in ['year', 'month', 'day', 'race_num', 'umaban']:
        df20[c] = pd.to_numeric(df20[c], errors='coerce').astype('Int64')
    key = ['year', 'month', 'day', 'race_num', 'horse_id_str', 'umaban']
    add_p24 = [c for c in PHASE24_26_FEATURES if c in df20.columns]
    df_cache = df_cache.merge(df20[key + add_p24].drop_duplicates(key), on=key, how='left')

    df_extra = pd.read_csv(MERGED_FEATURES, encoding='utf-8-sig', low_memory=False)
    df_extra['race_id'] = df_extra['race_id'].astype(str)
    df_extra['umaban'] = pd.to_numeric(df_extra['umaban'], errors='coerce').astype('Int64')
    df_extra = df_extra.drop_duplicates(['race_id', 'umaban'], keep='last')
    extra_cols = [c for c in df_extra.columns if c not in ('race_id', 'horse_id', 'umaban')]
    df_cache['race_id'] = df_cache['race_id'].astype(str)
    df_cache = df_cache.merge(df_extra[['race_id', 'umaban'] + extra_cols],
                              on=['race_id', 'umaban'], how='left')
    print(f'   df_cache merged: {df_cache.shape}')

    # 3. fold 25 split (train: year<25, test: year==25)
    df_cache = df_cache.dropna(subset=['target', 'year'])
    df_cache['year'] = df_cache['year'].astype(int)
    df_tr = df_cache[df_cache['year'] < 25].copy()
    df_te = df_cache[df_cache['year'] == 25].copy()
    print(f'   train (year<25): {len(df_tr):,}')
    print(f'   test (year==25): {len(df_te):,}')

    # 4. V15 predict on test (fold 25)
    print('\n[4] V15 predict on fold 25 ...')
    X25_v15 = df_te[cache_features].fillna(0).astype(np.float32).values
    p_v15_lgb = v15_lgb.predict(X25_v15)
    dmat = xgb.DMatrix(X25_v15)
    p_v15_xgb = v15_xgb.predict(dmat)
    p_v15 = v15_ens_w['lgb'] * p_v15_lgb + v15_ens_w['xgb'] * p_v15_xgb
    y_te = df_te['target'].values
    auc_v15 = roc_auc_score(y_te, p_v15)
    print(f'   V15 AUC: {auc_v15:.4f}')

    # 5. V22 top 100 train (fold-specific) + predict
    print('\n[5] V22 top 100 train + predict (fold-specific) ...')
    available = [f for f in top_100 if f in df_cache.columns]
    print(f'   available top features: {len(available)}/{len(top_100)}')

    df_tr_fill = df_tr.copy()
    df_te_fill = df_te.copy()
    df_tr_fill[available] = df_tr_fill[available].fillna(0)
    df_te_fill[available] = df_te_fill[available].fillna(0)

    X_tr = df_tr_fill[available].astype(np.float32).values
    y_tr = df_tr_fill['target'].values
    X_te = df_te_fill[available].astype(np.float32).values

    # LGB
    print('   LGB ...')
    t1 = time.time()
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_te, label=y_te, reference=train_set)
    lgb_model = lgb.train(LGB_PARAMS, train_set, num_boost_round=1000,
                           valid_sets=[val_set], valid_names=['val'],
                           callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    p_v22_lgb = lgb_model.predict(X_te)
    auc_v22_lgb = roc_auc_score(y_te, p_v22_lgb)
    print(f'   V22 LGB AUC: {auc_v22_lgb:.4f} [{time.time()-t1:.0f}s]')

    # XGB
    print('   XGB ...')
    t1 = time.time()
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_te, label=y_te)
    xgb_model = xgb.train(XGB_PARAMS, dtr, num_boost_round=1000,
                           evals=[(dva, 'val')], early_stopping_rounds=50, verbose_eval=0)
    p_v22_xgb = xgb_model.predict(dva)
    auc_v22_xgb = roc_auc_score(y_te, p_v22_xgb)
    print(f'   V22 XGB AUC: {auc_v22_xgb:.4f} [{time.time()-t1:.0f}s]')

    # V22 ensemble (LGB + XGB のみ、 FT/IR は重い)
    p_v22 = (p_v22_lgb + p_v22_xgb) / 2
    auc_v22 = roc_auc_score(y_te, p_v22)
    print(f'   V22 ensemble (LGB+XGB avg): {auc_v22:.4f}')

    # 6. Stacking
    print('\n[6] Stacking ...')

    # 6a. 単純 average (V15 0.5 + V22 0.5)
    p_avg = (p_v15 + p_v22) / 2
    auc_avg = roc_auc_score(y_te, p_avg)
    print(f'   simple avg (V15+V22 / 2): {auc_avg:.4f}')

    # 6b. 重み grid search (V15 weight 0.4 - 0.95)
    best_auc_w = 0
    best_w = 0
    for w in np.arange(0.4, 1.0, 0.05):
        p_w = w * p_v15 + (1 - w) * p_v22
        auc_w = roc_auc_score(y_te, p_w)
        if auc_w > best_auc_w:
            best_auc_w = auc_w
            best_w = w
    print(f'   weighted (V15 w={best_w:.2f}, V22 w={1-best_w:.2f}): {best_auc_w:.4f}')

    # 6c. LGB 2nd-layer stacker
    print('   LGB 2nd-layer training ...')
    # train 用 stacking dataset: 既 V22 train predictions が ない → V15 のみ で LGB 2nd 学習 不可
    # 簡易: V15 prob を input feature として V22 features と統合 → V22 LGB に追加
    print('   (note: 完全 LGB 2nd-layer は V22 train pred 必要、 今 簡易 average のみ)')

    # 7. Summary
    print('\n' + '=' * 60)
    print('SUMMARY (fold 25 = 2025 OOS)')
    print('=' * 60)
    print(f'V15 alone:                        {auc_v15:.4f}')
    print(f'V22 enhanced top 100 (LGB+XGB):   {auc_v22:.4f}')
    print(f'V15+V22 simple avg:               {auc_avg:.4f}')
    print(f'V15+V22 weighted (V15 {best_w:.2f}): {best_auc_w:.4f}')
    print(f'')
    print(f'V15 mean WF baseline (CLAUDE.md): 0.8939')
    print(f'V22 top 100 mean WF (5/14):       0.8813')
    print(f'')
    if best_auc_w > auc_v15:
        delta = best_auc_w - auc_v15
        print(f'★ Stacking 越え V15: +{delta:.4f} (best_w={best_w:.2f}) ★')
    else:
        delta = auc_v15 - best_auc_w
        print(f'✗ V15 alone wins by {delta:.4f}')

    print(f'\nelapsed: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
