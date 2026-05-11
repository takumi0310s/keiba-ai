#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""長期 layoff (休養) features (V20/V21 candidates).

馬の前走から の 経過日数 (rest_days、 V15 既存) を 細分化 + layoff return 効果 features。

【V15 投資保護】 V15 既存 rest_days / rest_category とは 別 axis (細分化 + interaction)

Usage:
    python tools/build_layoff_features.py
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
                      usecols=['race_id', 'horse_id', 'finish', 'year', 'month', 'day',
                               'distance', 'class_code'])
    df = df[df['year'] >= 22]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)

    # 日付 計算
    df['_year_full'] = 2000 + df['year']
    df['race_date'] = pd.to_datetime(
        df['_year_full'].astype(str) + '-' +
        df['month'].astype(str).str.zfill(2) + '-' +
        df['day'].astype(str).str.zfill(2),
        errors='coerce'
    )
    df = df.dropna(subset=['race_date'])
    df = df.sort_values(['horse_id', 'race_date']).reset_index(drop=True)

    gb = df.groupby('horse_id')
    df['prev_race_date'] = gb['race_date'].shift(1)
    df['rest_days'] = (df['race_date'] - df['prev_race_date']).dt.days
    df['rest_days'] = df['rest_days'].clip(0, 1500)

    # layoff カテゴリ細分化
    df['layoff_cat'] = pd.cut(df['rest_days'],
        bins=[-1, 7, 14, 21, 35, 60, 90, 150, 240, 365, 1500],
        labels=['1w以内', '2w', '3w', '5w', '2m', '3m', '5m', '8m', '12m', '12m+']
    )

    # binary flags
    df['fresh_horse'] = (df['rest_days'] < 30).astype(int)  # 1ヶ月以内
    df['long_layoff'] = (df['rest_days'] > 90).astype(int)  # 3ヶ月以上
    df['very_long_layoff'] = (df['rest_days'] > 180).astype(int)  # 半年以上
    df['debut_or_layoff'] = df['rest_days'].isna().astype(int)

    print(f'[INFO] {len(df):,} rows、 baseline top3: {df["top3"].mean():.3f}')

    # === Signal verify ===
    print('\n=== layoff カテゴリ 別 top3 rate ===')
    base_tr = df['top3'].mean()
    for cat, sub in df.groupby('layoff_cat', observed=True):
        if len(sub) < 100:
            continue
        tr = sub['top3'].mean()
        delta = tr - base_tr
        marker = ' ★' if abs(delta) > 0.04 else ''
        print(f'  {str(cat):<8} n={len(sub):>6,}  top3={tr:.3f}  Δ={delta:+.3f}{marker}')

    print('\n=== binary flags ===')
    for col in ['fresh_horse', 'long_layoff', 'very_long_layoff']:
        cd1 = df[df[col] == 1]
        cd0 = df[df[col] == 0]
        delta = cd1['top3'].mean() - cd0['top3'].mean()
        marker = ' ★' if abs(delta) > 0.04 else ''
        print(f'  {col:<22} n={len(cd1):>6,}  top3 when 1: {cd1["top3"].mean():.3f} vs 0: {cd0["top3"].mean():.3f}  Δ={delta:+.3f}{marker}')

    # === expanding: 馬の long layoff 後 top3 rate ===
    print('\n[INFO] computing horse_long_layoff_return_top3_rate_exp...')
    df['_idx'] = range(len(df))
    ll_sub = df[df['long_layoff'] == 1].copy()
    ll_sub = ll_sub.sort_values(['horse_id', '_idx']).reset_index(drop=True)
    ll_gb = ll_sub.groupby('horse_id')
    ll_sub['horse_long_layoff_top3_rate_exp'] = (
        ll_gb['top3'].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    )
    df = df.merge(ll_sub[['race_id', 'horse_id', 'horse_long_layoff_top3_rate_exp']],
                   on=['race_id', 'horse_id'], how='left')

    out_cols = ['race_id', 'horse_id', 'rest_days', 'fresh_horse',
                'long_layoff', 'very_long_layoff',
                'horse_long_layoff_top3_rate_exp']
    out = df[out_cols].copy()
    out_path = os.path.join(BASE_DIR, 'data', 'layoff_features.csv')
    out.to_csv(out_path, index=False)
    print(f'\n[OK] saved: {out_path}')
    print(f'[OK] shape: {out.shape}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
