"""V21 = V20 (text features) + paddock video features 統合 trainer.

V20 base (V15 cache + Phase 24/26 + features_merged) + paddock_video features を
LGB+XGB ensemble で 学習。

★ V15 投資保護 完全 ★:
V15 .pkl.gz / predict_core / app.py 完全不変。
V21 は別 file (keiba_model_v21_central.pkl.gz、 別 dir)。

★ data 前提 ★:
- paddock features 既生成 (tools/paddock_yolo_inference.py 実行 済)
- 蓄積 1000+ races (現状 237 dirs、 不足)
- 5/24+ ~ 7-8月 累積後 学習 候補

usage:
    python train/train_v21_video.py --quick   # quick test
    python train/train_v21_video.py           # full WF
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
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'train'))

from train_v22_4ensemble import LGB_PARAMS, XGB_PARAMS

PADDOCK_FEATURES = BASE / 'data' / 'features_paddock_video.csv'
V15_CACHE = BASE / 'data' / '_v15_optuna_df_cache.pkl.gz'
MODEL_DIR = BASE / 'models' / 'v21'
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    if not PADDOCK_FEATURES.exists():
        print(f'[ERROR] paddock features 未生成: {PADDOCK_FEATURES}')
        print('実行 順:')
        print('  1. python tools/paddock_yolo_inference.py')
        print('  2. python train/features_paddock_video.py')
        sys.exit(1)

    print('=== V21 trainer (V20 base + paddock video features) ===')

    # paddock features load
    df_paddock = pd.read_csv(PADDOCK_FEATURES, low_memory=False)
    df_paddock['race_id'] = df_paddock['race_id'].astype(str)
    df_paddock['horse_id'] = df_paddock['horse_id'].astype(str)
    print(f'paddock features: {len(df_paddock)} rows × {len(df_paddock.columns)} cols')

    if len(df_paddock) < 1000:
        print(f'★ 注意: paddock features {len(df_paddock)} rows 少、 V21 学習 困難 ★')
        print('★ 推奨: 5/16+ 数 ヶ月 蓄積後 retry ★')

    # V15 cache load
    print('V15 cache load ...')
    import lightgbm as lgb
    import xgboost as xgb

    with gzip.open(V15_CACHE, 'rb') as f:
        cache = pickle.load(f)
    df_cache = cache['df']
    cache_features = cache['features']
    df_cache['target'] = (df_cache['finish'] <= 3).astype(int)
    df_cache['horse_id_str'] = df_cache['horse_id'].astype(str)
    df_cache['race_id'] = df_cache['race_id'].astype(str)
    print(f'V15 cache: {df_cache.shape}, features: {len(cache_features)}')

    # merge paddock features
    paddock_feature_cols = [c for c in df_paddock.columns
                            if c not in ('race_id', 'horse_id', 'race_date', 'horse_name')]
    df = df_cache.merge(df_paddock[['race_id', 'horse_id'] + paddock_feature_cols],
                        left_on=['race_id', 'horse_id_str'],
                        right_on=['race_id', 'horse_id'],
                        how='left', suffixes=('', '_paddock'))
    matched = df[paddock_feature_cols[0]].notna().sum() if paddock_feature_cols else 0
    print(f'paddock features matched: {matched} / {len(df)} rows')

    if matched < 100:
        print('★ paddock features match 少、 V21 学習 結果 limited ★')

    # 全 features (V15 + paddock)
    df[paddock_feature_cols] = df[paddock_feature_cols].fillna(0)
    all_features = cache_features + paddock_feature_cols
    print(f'total V21 features: {len(all_features)}')

    # WF (quick mode)
    if args.quick:
        train_mask = df['year'] < 25
        test_mask = df['year'] == 25
        df_tr = df[train_mask]
        df_te = df[test_mask]
        print(f'train: {len(df_tr):,}, test: {len(df_te):,}')

        if len(df_tr) > 100 and len(df_te) > 100:
            X_tr = df_tr[all_features].astype(np.float32).values
            y_tr = df_tr['target'].values
            X_te = df_te[all_features].astype(np.float32).values
            y_te = df_te['target'].values

            # LGB
            print('LGB train ...')
            train_set = lgb.Dataset(X_tr, label=y_tr)
            val_set = lgb.Dataset(X_te, label=y_te, reference=train_set)
            lgb_model = lgb.train(LGB_PARAMS, train_set, num_boost_round=500,
                                   valid_sets=[val_set], valid_names=['val'],
                                   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            p_lgb = lgb_model.predict(X_te)
            auc_lgb = roc_auc_score(y_te, p_lgb)
            print(f'V21 LGB AUC: {auc_lgb:.4f}')

            # V15 baseline
            print(f'V15 baseline: 0.8939')
            print(f'V20-PLUS top 100: 0.8811')
            print(f'V21 (V20+paddock): {auc_lgb:.4f}')
            if auc_lgb > 0.8939:
                print(f'★ V21 越え V15: +{auc_lgb-0.8939:.4f} ★')
            else:
                print(f'V21 - V15: {auc_lgb-0.8939:+.4f}')

    print('\n★ paddock data 蓄積 1000+ 後 完全 retrain 推奨 ★')


if __name__ == '__main__':
    main()
