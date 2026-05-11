#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V20 training data FULL builder: 今夜 marathon 全 features を merge した 1 CSV 生成.

v21_training_data_builder.py の拡張版。 hot_streak / layoff / distance_surface /
sire_class_down / remarks / events / pace_expanding を 全 merge し、 V20 学習に
直 投入可能な data を 生成。

【V15 投資保護】 既存 csv の merge のみ、 V15 model 不変

Usage:
    python tools/v20_training_data_full_builder.py
    python tools/v20_training_data_full_builder.py --year-from 2020 --year-to 2026
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
    ap = argparse.ArgumentParser(description='V20 training data FULL builder')
    ap.add_argument('--year-from', dest='year_from', type=int, default=2020)
    ap.add_argument('--year-to', dest='year_to', type=int, default=2026)
    ap.add_argument('--out', default=os.path.join(BASE_DIR, 'data', 'v20_training_data_full.csv'))
    args = ap.parse_args()

    import pandas as pd

    base_path = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    print(f'[INFO] loading: {base_path}')
    df = pd.read_csv(base_path, encoding='utf-8', low_memory=False)
    yf = args.year_from - 2000 if args.year_from >= 2000 else args.year_from
    yt = args.year_to - 2000 if args.year_to >= 2000 else args.year_to
    df = df[(df['year'] >= yf) & (df['year'] <= yt)]
    df['race_id'] = df['race_id'].astype(str)
    # horse_id を int 化 (".0" suffix 除去 + 統一 dtype)
    df['horse_id'] = pd.to_numeric(df['horse_id'], errors='coerce').astype('Int64').astype(str)
    df['horse_id'] = df['horse_id'].str.replace('<NA>', '', regex=False)
    print(f'[INFO] base after year filter: {df.shape}')

    # All feature CSVs to merge
    feature_sources = [
        ('event_effect_features.csv', ['race_id', 'horse_id']),
        ('pace_features_expanding.csv', ['race_id', 'horse_id']),
        ('hot_streak_features.csv', ['race_id', 'horse_id']),
        ('layoff_features.csv', ['race_id', 'horse_id']),
        ('distance_surface_change_features.csv', ['race_id', 'horse_id']),
        ('sire_class_down_features.csv', ['race_id', 'horse_id']),
        ('competitor_gap_features.csv', ['race_id', 'horse_id']),
    ]

    total_new_cols = 0
    for fname, keys in feature_sources:
        path = os.path.join(BASE_DIR, 'data', fname)
        if not os.path.exists(path):
            print(f'  [SKIP] {fname} (not found)')
            continue
        sub = pd.read_csv(path, encoding='utf-8')
        # horse_id dtype 統一 (int → str)
        if 'horse_id' in sub.columns:
            sub['horse_id'] = pd.to_numeric(sub['horse_id'], errors='coerce').astype('Int64').astype(str)
            sub['horse_id'] = sub['horse_id'].str.replace('<NA>', '', regex=False)
        if 'race_id' in sub.columns:
            sub['race_id'] = sub['race_id'].astype(str)
        original_cols = set(sub.columns)
        # merge
        before = df.shape[1]
        df = df.merge(sub, on=keys, how='left', suffixes=('', '_dup'))
        # drop duplicate cols
        df = df.drop(columns=[c for c in df.columns if c.endswith('_dup')])
        added = df.shape[1] - before
        total_new_cols += added
        print(f'  [merge] {fname}: +{added} cols')

    # remarks by umaban
    rmk_path = os.path.join(BASE_DIR, 'data', 'race_review_features.csv')
    if os.path.exists(rmk_path) and 'umaban' in df.columns:
        rmk = pd.read_csv(rmk_path, encoding='utf-8',
                           dtype={'race_id': str})
        rmk_cols = ['race_id', 'umaban'] + [c for c in rmk.columns if c.startswith('rmk_')]
        rmk = rmk[rmk_cols].drop_duplicates(['race_id', 'umaban'])
        df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce').astype('Int64')
        rmk['umaban'] = pd.to_numeric(rmk['umaban'], errors='coerce').astype('Int64')
        before = df.shape[1]
        df = df.merge(rmk, on=['race_id', 'umaban'], how='left')
        added = df.shape[1] - before
        total_new_cols += added
        print(f'  [merge] race_review_features.csv (by umaban): +{added} cols')

    print(f'\n[OK] total new features merged: {total_new_cols}')
    print(f'[OK] final shape: {df.shape}')

    df.to_csv(args.out, index=False)
    print(f'[OK] saved: {args.out}')

    # coverage stats
    print('\n[non-null coverage (new features)]')
    new_cols = [c for c in df.columns if any(c.startswith(p) for p in [
        'class_', 'jockey_change', 'trainer_change', 'jockey_recent', 'trainer_recent',
        'horse_recent', 'pace_career', 'pace_recent', 'sire_', 'rmk_', 'rest_days',
        'fresh_horse', 'long_layoff', 'distance_change', 'surface_change',
        'turf_to_dirt', 'dirt_to_turf', 'horse_shorten', 'horse_extend',
    ])]
    for c in new_cols[:20]:
        rate = df[c].notna().mean()
        print(f'  {c:<40} {rate*100:.1f}%')
    return 0


if __name__ == '__main__':
    sys.exit(main())
