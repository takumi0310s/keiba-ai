#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""competitor_gap_features_v2.csv の LGB importance + Q5-Q1 delta 測定 (verify).

【V15 投資保護】 model 学習 はせず、 単体 features 検証 only。
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
_MAIN_REPO = r'C:\Users\takum\keiba-ai'
DATA_DIR = (
    os.path.join(_MAIN_REPO, 'data')
    if not os.path.exists(os.path.join(BASE_DIR, 'data', 'jra_races_full.csv'))
    and os.path.exists(os.path.join(_MAIN_REPO, 'data', 'jra_races_full.csv'))
    else os.path.join(BASE_DIR, 'data')
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-csv',
                    default=os.path.join(DATA_DIR, 'competitor_gap_features_v2.csv'))
    ap.add_argument('--year-from', type=int, default=18)
    ap.add_argument('--year-to', type=int, default=25)
    args = ap.parse_args()

    import lightgbm as lgb
    import pandas as pd
    import numpy as np

    print(f'[INFO] loading features: {args.features_csv}')
    feats_df = pd.read_csv(args.features_csv)
    feats_df['race_id'] = feats_df['race_id'].astype(str)
    feats_df['horse_id'] = feats_df['horse_id'].astype(str)
    feat_cols = [c for c in feats_df.columns if c not in ('race_id', 'horse_id')]
    print(f'[INFO] features: {feat_cols}')
    print(f'[INFO] feats shape: {feats_df.shape}')

    print('[INFO] loading jra_races_full for target...')
    src = pd.read_csv(os.path.join(DATA_DIR, 'jra_races_full.csv'),
                       encoding='utf-8', low_memory=False,
                       usecols=['race_id', 'horse_id', 'finish', 'year'])
    src = src[(src['year'] >= args.year_from) & (src['year'] <= args.year_to)]
    src['race_id'] = src['race_id'].astype(str)
    src['horse_id'] = src['horse_id'].astype(str)
    src = src[src['finish'] > 0]
    src['top3'] = (src['finish'] <= 3).astype(int)

    df = feats_df.merge(src[['race_id', 'horse_id', 'top3', 'year']],
                          on=['race_id', 'horse_id'], how='inner')
    print(f'[INFO] joined: {df.shape}')

    # Q5-Q1 delta
    print('\n[Q5-Q1 top3 rate delta]')
    for f in feat_cols:
        v = df.dropna(subset=[f])
        if len(v) < 100 or v[f].std() == 0:
            print(f'  {f:30s}: skipped')
            continue
        try:
            q = pd.qcut(v[f].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        except Exception:
            print(f'  {f:30s}: qcut failed')
            continue
        v = v.copy()
        v['_q'] = q
        grp = v.groupby('_q', observed=True)['top3'].mean()
        delta = grp.loc[5] - grp.loc[1]
        print(f'  {f:30s}: N={len(v):>8,}  Q1={grp.loc[1]:.4f}  Q5={grp.loc[5]:.4f}  delta={delta:+.4f}')

    # LGB importance (single fit、 quick check)
    print('\n[LGB single-fit importance]')
    df_lgb = df.dropna(subset=feat_cols, how='all').copy()
    X = df_lgb[feat_cols].fillna(df_lgb[feat_cols].median())
    y = df_lgb['top3']
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                 random_state=42, verbose=-1)
    model.fit(X, y)
    imp = sorted(zip(feat_cols, model.feature_importances_), key=lambda x: -x[1])
    for f, i in imp:
        print(f'  {f:30s}: {i}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
