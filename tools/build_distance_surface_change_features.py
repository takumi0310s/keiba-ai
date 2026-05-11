#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""距離変更 / 馬場変更 effect features (V20/V21 candidates).

馬の前走から 距離変更 / 芝→ダート / ダート→芝 等の変化が 当該レースの 着順に
どう影響するか + expanding window 化 で features 提供。

【V15 投資保護】 derivative 計算のみ、 V15 model 不変

Usage:
    python tools/build_distance_surface_change_features.py
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
                               'distance', 'surface'])
    df = df[df['year'] >= 22]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')

    df = df.sort_values(['horse_id', 'race_id']).reset_index(drop=True)
    gb = df.groupby('horse_id')

    # Prev race info per horse
    df['prev_distance'] = gb['distance'].shift(1)
    df['prev_surface'] = gb['surface'].shift(1)

    # Distance change features
    df['distance_change'] = df['distance'] - df['prev_distance']
    df['distance_change_abs'] = df['distance_change'].abs()

    # 距離 変更 categories
    df['dist_change_cat'] = pd.cut(df['distance_change'],
        bins=[-3000, -400, -200, -100, 100, 200, 400, 3000],
        labels=['短縮大', '短縮中', '短縮小', '同距離', '延長小', '延長中', '延長大']
    )

    # Surface change
    df['surface_change'] = ((df['surface'] != df['prev_surface']) &
                              df['prev_surface'].notna()).astype(int)
    df['turf_to_dirt'] = ((df['surface'] == 'ダ') &
                           (df['prev_surface'] == '芝')).astype(int)
    df['dirt_to_turf'] = ((df['surface'] == '芝') &
                           (df['prev_surface'] == 'ダ')).astype(int)

    print(f'[INFO] {len(df):,} rows、 baseline top3: {df["top3"].mean():.3f}')

    # === Signal verify ===
    print('\n=== 距離変更 別 top3 rate ===')
    for cat, sub in df.groupby('dist_change_cat', observed=True):
        if len(sub) < 100:
            continue
        tr = sub['top3'].mean()
        n = len(sub)
        delta = tr - df['top3'].mean()
        marker = ' ★' if abs(delta) > 0.04 else ''
        print(f'  {str(cat):<12} n={n:>6,}  top3={tr:.3f}  Δ={delta:+.3f}{marker}')

    print('\n=== 馬場変更 別 top3 rate ===')
    for col in ['surface_change', 'turf_to_dirt', 'dirt_to_turf']:
        cd1 = df[df[col] == 1]
        cd0 = df[df[col] == 0]
        if len(cd1) < 100:
            continue
        delta = cd1['top3'].mean() - cd0['top3'].mean()
        marker = ' ★' if abs(delta) > 0.04 else ''
        print(f'  {col:<22} n={len(cd1):>6,}  top3 when 1: {cd1["top3"].mean():.3f} vs 0: {cd0["top3"].mean():.3f}  Δ={delta:+.3f}{marker}')

    # === expanding features (LEAK-free): 馬の past 距離変更時 top3 rate ===
    print('\n[INFO] computing expanding "horse_distance_change_top3_rate"...')
    df['_idx'] = range(len(df))

    # 短縮時の expanding rate
    short_sub = df[df['dist_change_cat'].isin(['短縮中', '短縮大'])].copy()
    short_sub = short_sub.sort_values(['horse_id', '_idx']).reset_index(drop=True)
    short_gb = short_sub.groupby('horse_id')
    short_sub['horse_shorten_top3_rate_exp'] = (
        short_gb['top3'].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    )
    df = df.merge(short_sub[['race_id', 'horse_id', 'horse_shorten_top3_rate_exp']],
                   on=['race_id', 'horse_id'], how='left')

    # 延長時の expanding rate
    extend_sub = df[df['dist_change_cat'].isin(['延長中', '延長大'])].copy()
    extend_sub = extend_sub.sort_values(['horse_id', '_idx']).reset_index(drop=True)
    extend_gb = extend_sub.groupby('horse_id')
    extend_sub['horse_extend_top3_rate_exp'] = (
        extend_gb['top3'].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    )
    df = df.merge(extend_sub[['race_id', 'horse_id', 'horse_extend_top3_rate_exp']],
                   on=['race_id', 'horse_id'], how='left')

    # 馬場変更 expanding
    sc_sub = df[df['surface_change'] == 1].copy()
    sc_sub = sc_sub.sort_values(['horse_id', '_idx']).reset_index(drop=True)
    sc_gb = sc_sub.groupby('horse_id')
    sc_sub['horse_surface_change_top3_rate_exp'] = (
        sc_gb['top3'].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    )
    df = df.merge(sc_sub[['race_id', 'horse_id', 'horse_surface_change_top3_rate_exp']],
                   on=['race_id', 'horse_id'], how='left')

    out_cols = ['race_id', 'horse_id', 'distance_change', 'distance_change_abs',
                'surface_change', 'turf_to_dirt', 'dirt_to_turf',
                'horse_shorten_top3_rate_exp', 'horse_extend_top3_rate_exp',
                'horse_surface_change_top3_rate_exp']
    out = df[out_cols].copy()
    out_path = os.path.join(BASE_DIR, 'data', 'distance_surface_change_features.csv')
    out.to_csv(out_path, index=False)
    print(f'\n[OK] saved: {out_path}')
    print(f'[OK] shape: {out.shape}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
