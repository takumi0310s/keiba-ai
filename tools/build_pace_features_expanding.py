#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pace features の expanding window 版 (LEAK-free、 V20/V21 直接使用可).

build_pace_features.py の POST-RACE 値を 馬 ごと に 過去 K 走 で 集計し、
当該レース予測の features として安全に使える形に変換。 V15 既存 V12 features と
overlap しない 新 axis を 提供。

【V15 投資保護】 V12 既存 prev_race_first3f / agari_relative 等は 不変、 補完 features 追加

【追加 features (horse_id 単位、 各 race の "当該レース 直前" 時点で 計算)】
- pace_career_burst_mean: 過去 全走 final_burst 平均
- pace_career_burst_recent5: 過去 5 走 平均
- pace_career_change_1to4_mean: 過去 全走 pos_change_1to4 平均
- pace_career_relative_4cor_mean: 過去 全走 pos_relative_4corner 平均
- pace_career_agari_relative_mean: 過去 全走 agari_3f race-relative 平均

【WF 時系列保証】 各 race について、 過去 race の値 のみ (当該 race 除外) を集計

Usage:
    python tools/build_pace_features_expanding.py
    python tools/build_pace_features_expanding.py --year-from 2020 --year-to 2025
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
    ap = argparse.ArgumentParser(description='Pace features expanding window (LEAK-free)')
    ap.add_argument('--year-from', dest='year_from', type=int, default=2020)
    ap.add_argument('--year-to', dest='year_to', type=int, default=2025)
    ap.add_argument('--recent-k', dest='recent_k', type=int, default=5)
    ap.add_argument('--in', dest='in_path',
                    default=os.path.join(BASE_DIR, 'data', 'pace_features.csv'))
    ap.add_argument('--out', default=os.path.join(BASE_DIR, 'data', 'pace_features_expanding.csv'))
    args = ap.parse_args()

    import pandas as pd
    print(f'[INFO] loading: {args.in_path}')
    df = pd.read_csv(args.in_path, encoding='utf-8')
    print(f'[INFO] base: {df.shape}')

    # 必須 cols
    must = ['race_id', 'horse_id', 'final_burst', 'pos_change_1to4',
            'pos_relative_4corner', 'agari_3f_relative']
    missing = [c for c in must if c not in df.columns]
    if missing:
        print(f'[ERROR] missing: {missing}')
        return 1

    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)
    # sort by horse_id × race_id (race_id ≈ time order for same horse)
    df = df.sort_values(['horse_id', 'race_id']).reset_index(drop=True)

    print(f'[INFO] computing expanding stats per horse (recent_k={args.recent_k})...')
    # shift で 当該 race 除外
    gb = df.groupby('horse_id')

    # career mean (expanding, 当該除外)
    for src, dst in [
        ('final_burst', 'pace_career_burst_mean'),
        ('pos_change_1to4', 'pace_career_change_1to4_mean'),
        ('pos_relative_4corner', 'pace_career_relative_4cor_mean'),
        ('agari_3f_relative', 'pace_career_agari_relative_mean'),
    ]:
        # expanding mean、 shift で当該 race 除外
        df[dst] = gb[src].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)

    # recent K mean
    for src, dst in [
        ('final_burst', f'pace_recent{args.recent_k}_burst_mean'),
        ('pos_change_1to4', f'pace_recent{args.recent_k}_change_mean'),
    ]:
        df[dst] = gb[src].apply(
            lambda s: s.shift(1).rolling(args.recent_k, min_periods=1).mean()
        ).reset_index(level=0, drop=True)

    # 出力 columns
    out_cols = ['race_id', 'horse_id'] + [c for c in df.columns if c.startswith('pace_career_') or c.startswith('pace_recent')]
    out = df[out_cols].copy()
    out.to_csv(args.out, index=False)
    print(f'[OK] saved: {args.out}')
    print(f'[OK] shape: {out.shape}')

    # 統計 + signal preview
    print('\n[stats (non-null counts)]')
    for c in out_cols[2:]:
        non_null = out[c].notna().sum()
        if non_null > 0:
            print(f'  {c}: {non_null:,} non-null, mean={out[c].mean():.3f}, std={out[c].std():.3f}')

    # signal: career_burst_mean ↔ next race outcome (LEAK-free check)
    # 馬の career burst が高い (差し力) ほど 次走 top3 率高いか
    if 'finish' in df.columns:
        finish = pd.to_numeric(df['finish'], errors='coerce')
        valid = df[finish.notna() & df['pace_career_burst_mean'].notna()].copy()
        valid['top3'] = (finish[valid.index] <= 3).astype(int)
        valid['burst_bin'] = pd.qcut(valid['pace_career_burst_mean'].rank(method='first'),
                                       5, labels=['Q1低', 'Q2', 'Q3', 'Q4', 'Q5高'])
        print('\n[career_burst quintile × top3 rate]')
        print(valid.groupby('burst_bin', observed=True)['top3'].mean().round(4))

    return 0


if __name__ == '__main__':
    sys.exit(main())
