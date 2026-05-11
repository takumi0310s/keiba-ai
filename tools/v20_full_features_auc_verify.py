#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V20 全 features で LGB AUC 最終 verify.

v20_training_data_full.csv (190K rows × 101 cols) を 入力に、
year 24 train / 25 test の WF で LGB の AUC を 計測。

【V15 投資保護】 検証のみ、 V15 model 不変

Usage:
    python tools/v20_full_features_auc_verify.py
"""
import argparse
import os
import sys

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    import pandas as pd
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    path = os.path.join(BASE_DIR, 'data', 'v20_training_data_full.csv')
    print(f'[INFO] loading: {path}')
    df = pd.read_csv(path, encoding='utf-8', low_memory=False)
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    print(f'[INFO] shape: {df.shape}')

    # categorical encode
    for c in ['surface', 'condition', 'class_code', 'father', 'bms']:
        if c in df.columns:
            df[c] = df[c].astype('category').cat.codes

    # 数値 only + label
    target = 'top3'
    # V20 LIVE 想定: popularity / horse_weight / tansho_odds は事前確定 (LIVE 入力)
    drop_cols = {'race_id', 'horse_id', 'horse_name', 'jockey', 'trainer',
                 'owner', 'breeder', 'finish', 'finish2', 'abnormal_code',
                 'time_margin', 'run_time', 'run_time_x10', 'empty',
                 'pass1', 'pass2', 'pass3', 'pass4', 'agari_3f',  # POST-RACE LEAK
                 'birthday', 'mark1', 'mark2', 'training_4f',
                 'distance_change_cat',  # category cat
                 'top3', 'prize',  # POST-RACE
                 'umaban', 'horse_num',
                 'race_date', 'prev_race_date', '_year_full', '_idx',
                 # popularity / horse_weight は LIVE で 事前確定なので 残す
                 # tansho_odds は data 不在のため 自動 drop
                 }
    feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype in ('int64', 'float64', 'int32', 'float32', 'int8', 'int16')]
    print(f'[INFO] feature count: {len(feature_cols)}')

    # year filter
    train_df = df[df['year'] < 25]
    test_df = df[df['year'] == 25]
    print(f'[INFO] train: {len(train_df):,}, test: {len(test_df):,}')

    if len(test_df) < 1000:
        print('[WARN] insufficient test data')
        return 1

    X_tr = train_df[feature_cols].fillna(-1)
    X_te = test_df[feature_cols].fillna(-1)
    y_tr = train_df[target]
    y_te = test_df[target]

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'min_child_samples': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
    }

    print('\n[Training LGB with V20 full features...]')
    model = lgb.train(params, lgb.Dataset(X_tr, y_tr),
                       num_boost_round=500,
                       valid_sets=[lgb.Dataset(X_te, y_te)],
                       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
    pred = model.predict(X_te)
    auc = roc_auc_score(y_te, pred)
    print(f'\n[V20 FULL LGB AUC]: {auc:.4f}')
    print(f'[V15 baseline (per CLAUDE.md): 0.8939]')
    print(f'[Delta]: {auc - 0.8939:+.4f}')

    # Top 20 features
    print('\n[Top 20 feature importance (gain)]')
    imps = model.feature_importance(importance_type='gain')
    fnames = model.feature_name()
    ranked = sorted(zip(fnames, imps), key=lambda x: -x[1])[:20]
    for name, imp in ranked:
        # mark new features
        new_prefixes = ('class_', 'jockey_recent', 'trainer_recent', 'horse_recent',
                        'pace_career', 'pace_recent', 'sire_class', 'rmk_',
                        'rest_days', 'fresh_', 'long_', 'very_long_',
                        'distance_change', 'surface_change', 'turf_to_', 'dirt_to_',
                        'horse_shorten', 'horse_extend', 'jockey_change', 'trainer_change')
        is_new = any(name.startswith(p) for p in new_prefixes)
        marker = ' ★' if is_new else '  '
        print(f'  {imp:>12,.0f}  {marker} {name}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
