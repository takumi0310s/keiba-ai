#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""3 strong signals (class_down × pace_career_burst × pace_career_change) interaction 分析.

単独で +10-13pt の signal が 組合せで さらに boost されるか？
V20 では 最終的に 単独 features として使われるが、 interaction effect 確認で
V20 学習の重み付け 参考 になる。

【V15 投資保護】 分析のみ、 V15 model 不変

Usage:
    python tools/signal_interaction_analysis.py
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
    ap = argparse.ArgumentParser(description='Signal interaction analysis')
    args = ap.parse_args()

    import pandas as pd

    # 既存 csv 読み込み + merge
    base = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    df = pd.read_csv(base, encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'horse_id', 'finish', 'year'])
    df = df[df['year'] >= 22]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)

    evt = pd.read_csv(os.path.join(BASE_DIR, 'data', 'event_effect_features.csv'),
                      encoding='utf-8')
    evt['race_id'] = evt['race_id'].astype(str)
    evt['horse_id'] = evt['horse_id'].astype(str)
    evt = evt[['race_id', 'horse_id', 'class_down', 'jockey_change',
                'trainer_change']].drop_duplicates(['race_id', 'horse_id'])
    df = df.merge(evt, on=['race_id', 'horse_id'], how='left')

    pace = pd.read_csv(os.path.join(BASE_DIR, 'data', 'pace_features_expanding.csv'),
                       encoding='utf-8')
    pace['race_id'] = pace['race_id'].astype(str)
    pace['horse_id'] = pace['horse_id'].astype(str)
    df = df.merge(pace, on=['race_id', 'horse_id'], how='left')

    df = df.dropna(subset=['pace_career_burst_mean', 'class_down'])
    print(f'[INFO] sample: {len(df):,} rows')

    # quintile bin for continuous signals
    df['burst_bin'] = pd.qcut(df['pace_career_burst_mean'].rank(method='first'),
                                5, labels=['Q1_低', 'Q2', 'Q3', 'Q4', 'Q5_高'])
    df['change_bin'] = pd.qcut(df['pace_career_change_1to4_mean'].rank(method='first'),
                                 5, labels=['Q1_低', 'Q2', 'Q3', 'Q4', 'Q5_高'])

    # =====================================================
    # 1. class_down × pace_career_burst_mean (quintile)
    # =====================================================
    print('\n=== 1. class_down × pace_career_burst (quintile) ===')
    print('(top3 rate)')
    pivot = df.pivot_table(values='top3', index='burst_bin',
                              columns='class_down', aggfunc='mean', observed=True)
    print(pivot.round(3))
    # baseline (no class_down=0, Q3) vs interaction (class_down=1, Q5)
    base_rate = df['top3'].mean()
    int_rate = df[(df['class_down'] == 1) & (df['burst_bin'] == 'Q5_高')]['top3'].mean()
    print(f'  全体 baseline: {base_rate:.3f}')
    print(f'  class_down=1 & burst Q5_高: {int_rate:.3f} (Δ vs baseline: {int_rate-base_rate:+.3f})')

    # =====================================================
    # 2. class_down × trainer_change (binary × binary)
    # =====================================================
    print('\n=== 2. class_down × trainer_change ===')
    pivot2 = df.pivot_table(values='top3', index='trainer_change',
                              columns='class_down', aggfunc='mean')
    print(pivot2.round(3))

    # =====================================================
    # 3. burst × change_1to4 (両方 strong continuous)
    # =====================================================
    print('\n=== 3. pace_career_burst × change_1to4 (quintile × quintile) ===')
    pivot3 = df.pivot_table(values='top3', index='burst_bin',
                              columns='change_bin', aggfunc='mean', observed=True)
    print(pivot3.round(3))
    max_cell = pivot3.values.max()
    min_cell = pivot3.values.min()
    print(f'\n  最高 (両方 Q5_高): {max_cell:.3f}')
    print(f'  最低 (両方 Q1_低): {min_cell:.3f}')
    print(f'  範囲: {max_cell - min_cell:+.3f}')

    # =====================================================
    # 4. 3-way: class_down × burst × jockey_change
    # =====================================================
    print('\n=== 4. 3-way: class_down × burst Q5_高 × jockey_change ===')
    for jc in [0, 1]:
        for cd in [0, 1]:
            sub = df[(df['jockey_change'] == jc) & (df['class_down'] == cd) &
                       (df['burst_bin'] == 'Q5_高')]
            if len(sub) < 30:
                continue
            print(f'  jockey_change={jc}, class_down={cd}, burst=Q5: n={len(sub):,}, top3={sub["top3"].mean():.3f}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
