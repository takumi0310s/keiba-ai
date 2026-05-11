#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_pace_features.py の race_id bug 修正版.

jra_races_full.csv の race_id は umaban 込み horse-unique。
真の race grouping = (year, month, day, course, race_num) で group。

【V15 投資保護】 既存 pace_features.csv は 不変、 別 file 出力。

Usage:
    python tools/build_pace_features_FIXED.py
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
    ap = argparse.ArgumentParser(description='Fixed pace features (race_id bug fix)')
    ap.add_argument('--year-from', type=int, default=2020)
    ap.add_argument('--year-to', type=int, default=2025)
    ap.add_argument('--out', default=os.path.join(BASE_DIR, 'data', 'pace_features_FIXED.csv'))
    args = ap.parse_args()

    import pandas as pd
    print('[INFO] loading...')
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'jra_races_full.csv'),
                      encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'horse_id', 'umaban', 'finish', 'year',
                               'month', 'day', 'course', 'race_num',
                               'num_horses', 'pass1', 'pass2', 'pass3', 'pass4',
                               'agari_3f'])
    yf = args.year_from - 2000
    yt = args.year_to - 2000
    df = df[(df['year'] >= yf) & (df['year'] <= yt)]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['race_id'] = df['race_id'].astype(str)

    # 真の race key (race-unique)
    df['true_race_key'] = (df['year'].astype(str) + '_' +
                            df['month'].astype(str).str.zfill(2) + '_' +
                            df['day'].astype(str).str.zfill(2) + '_' +
                            df['course'].astype(str) + '_' +
                            df['race_num'].astype(str).str.zfill(2))
    print(f'[INFO] rows: {len(df):,}, unique races: {df["true_race_key"].nunique():,}')
    print(f'  avg horses/race: {len(df) / df["true_race_key"].nunique():.1f}')

    for c in ['pass1', 'pass2', 'pass3', 'pass4', 'agari_3f']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # race-level aggregates (正しい race key で)
    print('[INFO] race-level pace stats (FIXED grouping)...')
    grouped = df.groupby('true_race_key').agg(
        pace_avg_pass1=('pass1', 'mean'),
        pace_std_pass1=('pass1', 'std'),
        pace_avg_pass4=('pass4', 'mean'),
        pace_std_pass4=('pass4', 'std'),
        pace_avg_agari=('agari_3f', 'mean'),
        pace_std_agari=('agari_3f', 'std'),
    ).reset_index()
    df = df.merge(grouped, on='true_race_key', how='left')

    # relative
    df['agari_3f_relative'] = df['agari_3f'] - df['pace_avg_agari']
    df['pass4_relative'] = df['pass4'] - df['pace_avg_pass4']
    df['pos_change_1to4'] = df['pass1'] - df['pass4']
    df['pos_change_4tofin'] = df['pass4'] - df['finish']
    df['final_burst'] = df['pos_change_4tofin']

    # Save
    keep_cols = ['race_id', 'horse_id', 'umaban', 'finish', 'true_race_key',
                 'pace_avg_pass1', 'pace_std_pass1',
                 'pace_avg_pass4', 'pace_std_pass4',
                 'pace_avg_agari', 'pace_std_agari',
                 'pos_change_1to4', 'pos_change_4tofin',
                 'agari_3f_relative', 'pass4_relative', 'final_burst']
    out = df[keep_cols].copy()
    out.to_csv(args.out, index=False)
    print(f'\n[OK] saved: {args.out}')
    print(f'[OK] shape: {out.shape}')

    # Verify (race-avg がちゃんと race ごとに 異なる か)
    print('\n[verify] race-level aggregation 動作確認:')
    sample_race = out['true_race_key'].iloc[0]
    sample_df = out[out['true_race_key'] == sample_race]
    print(f'  sample race: {sample_race}')
    print(f'  horses: {len(sample_df)}')
    print(f'  pace_avg_pass1 (should be same for all): {sample_df["pace_avg_pass1"].nunique() == 1}')
    print(f'  agari_3f_relative (should differ per horse): {sample_df["agari_3f_relative"].nunique() > 1}')

    print('\n[stats by 着順]')
    print(out.groupby(out['finish'].astype(int))[
        ['pos_change_1to4', 'final_burst', 'agari_3f_relative']
    ].mean().head(10))
    return 0


if __name__ == '__main__':
    sys.exit(main())
