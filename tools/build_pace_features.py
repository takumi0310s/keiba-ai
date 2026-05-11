#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Race-level pace features 算出 (V20/V21 candidates).

既存 pass1-4 (通過順位) + agari_3f (上がり 3F) + run_time から、 race-level pace 統計 +
horse-level position change features を 算出。 V15 既存 prev_race_first3f 等 とは別 axis。

【追加 features (per-race × per-horse、 12 種)】
1. pace_avg_pass1: 1 角 通過順位 race avg (混戦度の proxy)
2. pace_std_pass1: 1 角 std
3. pos_change_1to4: 自分の 1 角 - 4 角 (前進 / 後退)
4. pos_relative_4corner: 4 角時 num_horses 相対 位置 (0-1)
5. agari_3f_relative: 上がり 3F race avg からの差
6. final_burst: 4 角 - finish の差 (位差 → 何頭交わしたか)
7. early_pace_diff: 1 角 - 自分の最終 finish (前残り or 差し)
8. avg_finish_other_horses_in_race: 他馬 finish 平均 (race quality proxy)

【V15 投資保護】 derivative features 算出のみ、 V15 model 不変

Usage:
    python tools/build_pace_features.py
    python tools/build_pace_features.py --year-from 2020 --year-to 2025
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
    ap = argparse.ArgumentParser(description='Race-level pace features')
    ap.add_argument('--year-from', dest='year_from', type=int, default=2020)
    ap.add_argument('--year-to', dest='year_to', type=int, default=2025)
    ap.add_argument('--out', default=os.path.join(BASE_DIR, 'data', 'pace_features.csv'))
    args = ap.parse_args()

    import pandas as pd

    base = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    print(f'[INFO] loading: {base}')
    df = pd.read_csv(base, encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'horse_id', 'umaban', 'finish', 'year',
                               'num_horses', 'pass1', 'pass2', 'pass3', 'pass4', 'agari_3f'])
    print(f'[INFO] base: {df.shape}')

    # year filter
    yf_2digit = args.year_from - 2000
    yt_2digit = args.year_to - 2000
    df = df[(df['year'] >= yf_2digit) & (df['year'] <= yt_2digit)]
    print(f'[INFO] after year filter: {df.shape}')

    df = df.dropna(subset=['finish', 'num_horses'])
    df = df[df['finish'] > 0]

    for col in ['pass1', 'pass2', 'pass3', 'pass4', 'agari_3f']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # per-horse derivative
    df['pos_change_1to4'] = df['pass1'] - df['pass4']  # >0 = 前進
    df['pos_change_4tofin'] = df['pass4'] - df['finish']  # >0 = final 直線で 前進
    df['pos_relative_4corner'] = (df['pass4'] / df['num_horses']).clip(0, 1)
    df['pos_relative_1corner'] = (df['pass1'] / df['num_horses']).clip(0, 1)

    # race-level aggregates (per race avg / std)
    print('[INFO] computing race-level pace stats...')
    grouped = df.groupby('race_id').agg(
        pace_avg_pass1=('pass1', 'mean'),
        pace_std_pass1=('pass1', 'std'),
        pace_avg_pass4=('pass4', 'mean'),
        pace_std_pass4=('pass4', 'std'),
        pace_avg_agari=('agari_3f', 'mean'),
        pace_std_agari=('agari_3f', 'std'),
    ).reset_index()
    df = df.merge(grouped, on='race_id', how='left')

    # relative (vs race avg)
    df['agari_3f_relative'] = df['agari_3f'] - df['pace_avg_agari']
    df['pass4_relative'] = df['pass4'] - df['pace_avg_pass4']

    # final_burst (final 直線での 順位移動)
    df['final_burst'] = df['pos_change_4tofin']

    # early_pace_diff (1 角 - finish、 大きい = 差し / 大きく上がった)
    df['early_pace_diff'] = df['pass1'] - df['finish']

    # output
    keep_cols = ['race_id', 'horse_id', 'umaban', 'finish',
                 'pace_avg_pass1', 'pace_std_pass1',
                 'pace_avg_pass4', 'pace_std_pass4',
                 'pace_avg_agari', 'pace_std_agari',
                 'pos_change_1to4', 'pos_change_4tofin',
                 'pos_relative_4corner', 'pos_relative_1corner',
                 'agari_3f_relative', 'pass4_relative',
                 'final_burst', 'early_pace_diff']
    out = df[keep_cols].copy()
    out.to_csv(args.out, index=False)
    print(f'[OK] saved: {args.out}')
    print(f'[OK] shape: {out.shape}, columns: {list(out.columns)[5:]}')

    # 統計
    print('\n[stats by 着順]')
    print(out.groupby(out['finish'].astype(int))[
        ['pos_change_1to4', 'final_burst', 'agari_3f_relative']
    ].mean().head(10))

    return 0


if __name__ == '__main__':
    sys.exit(main())
