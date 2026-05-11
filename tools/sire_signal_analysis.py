#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""父馬 (sire) / 母父 (bms) 系統別 signal 強度 + class_down interaction.

血統 systems で 降級 + 黄金 pattern に hit しやすい系統を 発見。
V15 既存 sire_enc / bms_enc は ID encoding のみ、 系統 特化 signal は新規 axis。

【V15 投資保護】 分析のみ、 V15 model 不変

Usage:
    python tools/sire_signal_analysis.py
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
                               'father', 'bms', 'distance', 'surface'])
    df = df[df['year'] >= 22]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)

    # event_effect_features merge
    evt = pd.read_csv(os.path.join(BASE_DIR, 'data', 'event_effect_features.csv'),
                      encoding='utf-8')
    evt['race_id'] = evt['race_id'].astype(str)
    evt['horse_id'] = evt['horse_id'].astype(str)
    df = df.merge(evt[['race_id', 'horse_id', 'class_down', 'jockey_change']].drop_duplicates(['race_id', 'horse_id']),
                   on=['race_id', 'horse_id'], how='left')

    pace = pd.read_csv(os.path.join(BASE_DIR, 'data', 'pace_features_expanding.csv'),
                       encoding='utf-8')
    pace['race_id'] = pace['race_id'].astype(str)
    pace['horse_id'] = pace['horse_id'].astype(str)
    df = df.merge(pace[['race_id', 'horse_id', 'pace_career_burst_mean']],
                   on=['race_id', 'horse_id'], how='left')

    df = df.dropna(subset=['father', 'pace_career_burst_mean'])
    print(f'[INFO] sample: {len(df):,} rows')

    # 父馬 TOP 30 系統 signal
    sire_top = df['father'].value_counts().head(30)
    print('\n=== 父馬 TOP 30 系統 全体 top3 rate ===')
    overall = df['top3'].mean()
    print(f'  全体 baseline: {overall:.3f}')
    sire_stats = []
    for sire, n in sire_top.items():
        sub = df[df['father'] == sire]
        tr = sub['top3'].mean()
        delta = tr - overall
        sire_stats.append({'sire': sire, 'n': n, 'top3': tr, 'delta': delta})

    sire_stats.sort(key=lambda x: -x['delta'])
    print('  父馬 top3 rate (vs baseline):')
    for s in sire_stats[:15]:
        marker = ' ★' if abs(s['delta']) > 0.04 else ''
        print(f'    {s["sire"][:25]:<25}  n={s["n"]:>5,}  top3={s["top3"]:.3f}  Δ={s["delta"]:+.3f}{marker}')

    # 父馬 × class_down interaction
    print('\n=== 父馬 × class_down (降級時 top3 boost が大きい系統) ===')
    interactions = []
    for sire in sire_top.index[:30]:
        sub = df[df['father'] == sire]
        cd1 = sub[sub['class_down'] == 1]
        cd0 = sub[sub['class_down'] == 0]
        if len(cd1) < 50 or len(cd0) < 50:
            continue
        boost = cd1['top3'].mean() - cd0['top3'].mean()
        interactions.append({
            'sire': sire,
            'n_cd1': len(cd1),
            'top3_cd1': cd1['top3'].mean(),
            'top3_cd0': cd0['top3'].mean(),
            'boost': boost,
        })

    interactions.sort(key=lambda x: -x['boost'])
    print('  父馬で 降級時 boost 大きい系統 (top 10):')
    for i in interactions[:10]:
        print(f'    {i["sire"][:25]:<25}  cd1: {i["top3_cd1"]:.3f}, cd0: {i["top3_cd0"]:.3f}, boost={i["boost"]:+.3f}')

    print('\n  父馬で 降級時 boost 小さい系統 (bottom 5):')
    for i in interactions[-5:]:
        print(f'    {i["sire"][:25]:<25}  cd1: {i["top3_cd1"]:.3f}, cd0: {i["top3_cd0"]:.3f}, boost={i["boost"]:+.3f}')

    # 黄金 pattern 父馬別
    print('\n=== 黄金 pattern (降級+同騎手+差し力Q5) を出しやすい父馬 ===')
    df['burst_bin'] = pd.qcut(df['pace_career_burst_mean'].rank(method='first'),
                                5, labels=[1, 2, 3, 4, 5])
    golden = df[(df['class_down'] == 1) & (df['jockey_change'] == 0) &
                 (df['burst_bin'] == 5)]
    sire_golden = golden['father'].value_counts().head(15)
    for sire, n in sire_golden.items():
        if n < 5:
            break
        sub = golden[golden['father'] == sire]
        tr = sub['top3'].mean()
        print(f'    {sire[:25]:<25}  n={n:>3}  top3={tr:.3f}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
