#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""自動 golden pattern miner: 全 features combinations を 試行し、 top3 rate 最高の patterns 列挙.

手動で探した 黄金 pattern (降級+同騎手+差し力Q5 = 43.8%) を 系統的に網羅探索。
2-way / 3-way combinations を 全部 grid search し、 n >= 100 で top3 rate >= 35% の pattern 列挙。

【V15 投資保護】 分析のみ、 V15 model 不変

Usage:
    python tools/golden_pattern_miner.py
    python tools/golden_pattern_miner.py --min-rate 0.40  # top3 rate 40%+
"""
import argparse
import os
import sys
from itertools import combinations

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description='Golden pattern miner')
    ap.add_argument('--min-rate', dest='min_rate', type=float, default=0.35)
    ap.add_argument('--min-n', dest='min_n', type=int, default=100)
    args = ap.parse_args()

    import pandas as pd

    base = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    df = pd.read_csv(base, encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'horse_id', 'finish', 'year'])
    df = df[df['year'] >= 22]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)

    # merge all features
    for fname, prefix in [
        ('event_effect_features.csv', None),
        ('pace_features_expanding.csv', None),
        ('hot_streak_features.csv', None),
        ('layoff_features.csv', None),
        ('distance_surface_change_features.csv', None),
    ]:
        path = os.path.join(BASE_DIR, 'data', fname)
        if not os.path.exists(path):
            continue
        sub = pd.read_csv(path, encoding='utf-8')
        sub['race_id'] = sub['race_id'].astype(str)
        sub['horse_id'] = sub['horse_id'].astype(str)
        df = df.merge(sub, on=['race_id', 'horse_id'], how='left', suffixes=('', '_dup'))
        # drop _dup cols
        df = df.drop(columns=[c for c in df.columns if c.endswith('_dup')])

    print(f'[INFO] merged: {df.shape}')
    baseline = df['top3'].mean()
    print(f'[INFO] baseline top3: {baseline:.3f}')

    # quintile bin for continuous strong signals
    cont_features = [
        'horse_recent5_top3', 'jockey_recent30_top3', 'trainer_recent30_top3',
        'pace_career_burst_mean', 'pace_career_change_1to4_mean',
    ]
    for f in cont_features:
        if f in df.columns:
            try:
                df[f + '_q5'] = (pd.qcut(df[f].rank(method='first'),
                                           5, labels=[1, 2, 3, 4, 5]) == 5).astype(int)
            except Exception:
                pass

    # binary features (1 = positive condition for top3)
    binary_features = ['class_down', 'fresh_horse',
                       'horse_recent5_top3_q5', 'jockey_recent30_top3_q5',
                       'trainer_recent30_top3_q5',
                       'pace_career_burst_mean_q5', 'pace_career_change_1to4_mean_q5']

    # negative-direction binary (0 = positive)
    negative_features = ['jockey_change', 'trainer_change',
                         'surface_change', 'long_layoff', 'very_long_layoff',
                         'turf_to_dirt', 'dirt_to_turf']

    # Convert negatives to "positive" indicator (0 → 1)
    for col in negative_features:
        if col in df.columns:
            df[col + '_neg'] = (df[col] == 0).astype(int)

    all_positive = binary_features + [c + '_neg' for c in negative_features if c in df.columns]
    valid_features = [c for c in all_positive if c in df.columns]
    print(f'[INFO] candidate features: {len(valid_features)}')

    # =========================================
    # 2-way combinations
    # =========================================
    print('\n=== 2-way patterns (top3 rate >= 0.35 or so) ===')
    results_2way = []
    for f1, f2 in combinations(valid_features, 2):
        sub = df[(df[f1] == 1) & (df[f2] == 1)]
        if len(sub) < args.min_n:
            continue
        rate = sub['top3'].mean()
        if rate >= args.min_rate:
            results_2way.append({
                'pattern': (f1, f2),
                'n': len(sub),
                'rate': rate,
                'delta': rate - baseline,
            })
    results_2way.sort(key=lambda x: -x['rate'])
    for r in results_2way[:15]:
        print(f'  {r["rate"]:.3f}  n={r["n"]:>5,}  Δ={r["delta"]:+.3f}  {r["pattern"]}')

    # =========================================
    # 3-way combinations
    # =========================================
    print('\n=== 3-way patterns (top3 rate >= 0.40) ===')
    results_3way = []
    # 3-way は爆発するので strong features に限定
    strong = ['class_down', 'horse_recent5_top3_q5', 'jockey_recent30_top3_q5',
              'trainer_recent30_top3_q5', 'pace_career_burst_mean_q5',
              'pace_career_change_1to4_mean_q5',
              'jockey_change_neg', 'trainer_change_neg', 'surface_change_neg']
    strong = [c for c in strong if c in df.columns]
    for f1, f2, f3 in combinations(strong, 3):
        sub = df[(df[f1] == 1) & (df[f2] == 1) & (df[f3] == 1)]
        if len(sub) < args.min_n:
            continue
        rate = sub['top3'].mean()
        if rate >= 0.40:
            results_3way.append({
                'pattern': (f1, f2, f3),
                'n': len(sub),
                'rate': rate,
                'delta': rate - baseline,
            })
    results_3way.sort(key=lambda x: -x['rate'])
    for r in results_3way[:15]:
        p = ', '.join(r['pattern'])
        print(f'  {r["rate"]:.3f}  n={r["n"]:>5,}  Δ={r["delta"]:+.3f}  {p}')

    # =========================================
    # 4-way (only top-rated 3-ways + 1 extra)
    # =========================================
    if results_3way:
        print('\n=== 4-way patterns (top3 rate >= 0.45) ===')
        top_3way = results_3way[:5]
        results_4way = []
        for top in top_3way:
            f1, f2, f3 = top['pattern']
            for f4 in strong:
                if f4 in (f1, f2, f3):
                    continue
                sub = df[(df[f1] == 1) & (df[f2] == 1) & (df[f3] == 1) & (df[f4] == 1)]
                if len(sub) < 30:
                    continue
                rate = sub['top3'].mean()
                if rate >= 0.45:
                    results_4way.append({
                        'pattern': (f1, f2, f3, f4),
                        'n': len(sub),
                        'rate': rate,
                        'delta': rate - baseline,
                    })
        # dedupe by sorted tuple
        seen = set()
        unique = []
        for r in sorted(results_4way, key=lambda x: -x['rate']):
            key = tuple(sorted(r['pattern']))
            if key in seen:
                continue
            seen.add(key)
            unique.append(r)
        for r in unique[:15]:
            p = ', '.join(r['pattern'])
            print(f'  {r["rate"]:.3f}  n={r["n"]:>4,}  Δ={r["delta"]:+.3f}  {p}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
