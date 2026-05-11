#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sire × class_down boost を 馬 ごとの explicit feature 化.

V20 / V21 学習 で directly 使える "sire_class_down_top3_rate_exp" feature を 生成。
expanding window で LEAK-free、 馬の父馬の 過去 降級時 top3 rate を 当該レース予測 features に。

【V15 投資保護】 derivative 計算のみ、 V15 model 不変

Usage:
    python tools/build_sire_class_down_features.py
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
    base = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    df = pd.read_csv(base, encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'horse_id', 'finish', 'year',
                               'father', 'bms'])
    df = df[df['year'] >= 22]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)

    # event merge
    evt = pd.read_csv(os.path.join(BASE_DIR, 'data', 'event_effect_features.csv'),
                      encoding='utf-8')
    evt['race_id'] = evt['race_id'].astype(str)
    evt['horse_id'] = evt['horse_id'].astype(str)
    df = df.merge(evt[['race_id', 'horse_id', 'class_down']].drop_duplicates(['race_id', 'horse_id']),
                   on=['race_id', 'horse_id'], how='left')
    df = df.dropna(subset=['father', 'class_down'])
    df = df.sort_values(['race_id'])
    print(f'[INFO] {len(df):,} rows')

    # 父馬別 expanding stat: 降級時 top3 rate (当該 race 除外)
    print('[INFO] computing sire × class_down expanding top3 rate...')
    df['_idx'] = range(len(df))
    df = df.sort_values(['father', '_idx']).reset_index(drop=True)

    # 父馬 × class_down=1 の sub-set で expanding mean of top3
    # = 父馬の過去 (当該 race 含まない) 降級時 top3 rate
    cd1_sub = df[df['class_down'] == 1].copy()
    cd1_sub = cd1_sub.sort_values(['father', '_idx']).reset_index(drop=True)
    gb = cd1_sub.groupby('father')
    cd1_sub['sire_class_down_top3_rate_exp'] = (
        gb['top3'].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    )

    # 父馬 × class_down=0 の sub-set 同様
    cd0_sub = df[df['class_down'] == 0].copy()
    cd0_sub = cd0_sub.sort_values(['father', '_idx']).reset_index(drop=True)
    gb0 = cd0_sub.groupby('father')
    cd0_sub['sire_no_class_down_top3_rate_exp'] = (
        gb0['top3'].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    )

    # merge back
    df = df.merge(cd1_sub[['race_id', 'horse_id', 'sire_class_down_top3_rate_exp']],
                   on=['race_id', 'horse_id'], how='left')
    df = df.merge(cd0_sub[['race_id', 'horse_id', 'sire_no_class_down_top3_rate_exp']],
                   on=['race_id', 'horse_id'], how='left')

    # 父馬 全体 expanding top3 rate (sire 平均)
    df_sorted = df.sort_values(['father', '_idx']).reset_index(drop=True)
    gb_all = df_sorted.groupby('father')
    df_sorted['sire_overall_top3_rate_exp'] = (
        gb_all['top3'].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    )
    df = df.merge(
        df_sorted[['race_id', 'horse_id', 'sire_overall_top3_rate_exp']].drop_duplicates(['race_id', 'horse_id']),
        on=['race_id', 'horse_id'], how='left'
    )

    # boost (sire_class_down - sire_no_class_down)
    df['sire_class_down_boost_exp'] = (
        df['sire_class_down_top3_rate_exp'].fillna(0.25) -
        df['sire_no_class_down_top3_rate_exp'].fillna(0.20)
    )

    out_cols = ['race_id', 'horse_id', 'father',
                'sire_overall_top3_rate_exp',
                'sire_class_down_top3_rate_exp',
                'sire_no_class_down_top3_rate_exp',
                'sire_class_down_boost_exp']
    out = df[out_cols].copy()
    out_path = os.path.join(BASE_DIR, 'data', 'sire_class_down_features.csv')
    out.to_csv(out_path, index=False)
    print(f'[OK] saved: {out_path}')
    print(f'[OK] shape: {out.shape}')
    print('\n[stats]')
    for c in out_cols[3:]:
        non_null = out[c].notna().sum()
        print(f'  {c}: {non_null:,} non-null, mean={out[c].mean():.3f}, std={out[c].std():.3f}')

    # signal verify: boost が高い 馬の 実 top3 rate
    df_v = df.dropna(subset=['sire_class_down_boost_exp'])
    df_v['boost_q'] = pd.qcut(df_v['sire_class_down_boost_exp'].rank(method='first'),
                                 5, labels=['Q1_低', 'Q2', 'Q3', 'Q4', 'Q5_高'])
    print('\n[signal verify: sire_class_down_boost_exp quintile × top3 rate]')
    print(df_v.groupby('boost_q', observed=True)['top3'].mean().round(3))

    return 0


if __name__ == '__main__':
    sys.exit(main())
