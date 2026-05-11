#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""5-way pattern miner: 5 つの features 組合せで top3 rate 最高 patterns 列挙.

4-way Jackpot (top3 64.8%) を ベースに、 5 番目 feature を 追加して さらに 高い rate
を 探索。

【V15 投資保護】 分析のみ、 V15 model 不変

Usage:
    python tools/pattern_miner_5way.py
    python tools/pattern_miner_5way.py --min-rate 0.70 --min-n 50
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
    ap = argparse.ArgumentParser(description='5-way pattern miner')
    ap.add_argument('--min-rate', dest='min_rate', type=float, default=0.65)
    ap.add_argument('--min-n', dest='min_n', type=int, default=30)
    args = ap.parse_args()

    import pandas as pd

    base = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    df = pd.read_csv(base, encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'horse_id', 'finish', 'year',
                               'popularity', 'num_horses', 'distance', 'condition',
                               'class_code', 'surface', 'age'])
    df = df[df['year'] >= 22]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)

    # merge all features
    feature_files = [
        'event_effect_features.csv',
        'pace_features_expanding.csv',
        'hot_streak_features.csv',
        'layoff_features.csv',
        'distance_surface_change_features.csv',
        'sire_class_down_features.csv',
    ]
    for fname in feature_files:
        path = os.path.join(BASE_DIR, 'data', fname)
        if not os.path.exists(path):
            continue
        sub = pd.read_csv(path, encoding='utf-8',
                           dtype={'race_id': str, 'horse_id': str})
        df = df.merge(sub, on=['race_id', 'horse_id'], how='left', suffixes=('', '_d'))
        df = df.drop(columns=[c for c in df.columns if c.endswith('_d')])

    df = df.dropna(subset=['horse_recent5_top3', 'jockey_recent30_top3',
                             'class_down', 'jockey_change', 'popularity'])
    baseline = df['top3'].mean()
    print(f'[INFO] {len(df):,} rows、 baseline: {baseline:.3f}')

    # === Base Jackpot 4-way 既知 ===
    # class_down=1, horse_q5=1, jockey_q5=1, jockey_change=0
    df['horse_q5'] = (pd.qcut(df['horse_recent5_top3'].rank(method='first'),
                                 5, labels=[1, 2, 3, 4, 5]) == 5).astype(int)
    df['jockey_q5'] = (pd.qcut(df['jockey_recent30_top3'].rank(method='first'),
                                  5, labels=[1, 2, 3, 4, 5]) == 5).astype(int)
    if 'pace_career_burst_mean' in df.columns:
        df['burst_q5'] = (pd.qcut(df['pace_career_burst_mean'].rank(method='first'),
                                     5, labels=[1, 2, 3, 4, 5]) == 5).astype(int)
    if 'pace_career_change_1to4_mean' in df.columns:
        df['change_q5'] = (pd.qcut(df['pace_career_change_1to4_mean'].rank(method='first'),
                                      5, labels=[1, 2, 3, 4, 5]) == 5).astype(int)

    # popularity による pre-filter (1人気は trivial、 4-7 人気は穴 = ROI 美味しい)
    df['pop_2_5'] = ((df['popularity'] >= 2) & (df['popularity'] <= 5)).astype(int)
    df['pop_6plus'] = (df['popularity'] >= 6).astype(int)
    df['pop_1to3'] = ((df['popularity'] >= 1) & (df['popularity'] <= 3)).astype(int)

    # 全 binary candidates
    binaries = ['class_down', 'horse_q5', 'jockey_q5', 'burst_q5', 'change_q5',
                'jockey_change', 'trainer_change', 'surface_change',
                'turf_to_dirt', 'dirt_to_turf', 'long_layoff', 'very_long_layoff',
                'fresh_horse',
                'pop_2_5', 'pop_6plus', 'pop_1to3',
                'class_up', 'class_change']

    # negative-flip features
    for col in ['jockey_change', 'trainer_change', 'surface_change',
                'turf_to_dirt', 'dirt_to_turf', 'long_layoff', 'very_long_layoff']:
        if col in df.columns:
            df[col + '_neg'] = (df[col] == 0).astype(int)
            binaries.append(col + '_neg')

    binaries = [c for c in binaries if c in df.columns]
    print(f'[INFO] candidate binaries: {len(binaries)}')

    # Base Jackpot 4-way: 1 番目 feature を 既知の class_down 限定
    # 5 番目 feature を 全候補から
    base_features = ['class_down', 'horse_q5', 'jockey_q5', 'jockey_change_neg']

    base_filter = df.copy()
    for f in base_features:
        if f in base_filter.columns:
            base_filter = base_filter[base_filter[f] == 1]
    print(f'[INFO] Base 4-way Jackpot n={len(base_filter):,}, top3={base_filter["top3"].mean():.3f}')

    # 5 番目 feature 追加で top3 が上がる pattern
    print(f'\n=== 5-way patterns (base 4-way + 1 feature、 top3 rate >= {args.min_rate}) ===')
    results_5way = []
    for f5 in binaries:
        if f5 in base_features:
            continue
        sub = base_filter[base_filter[f5] == 1]
        if len(sub) < args.min_n:
            continue
        rate = sub['top3'].mean()
        if rate >= args.min_rate:
            results_5way.append({
                'feature': f5, 'n': len(sub), 'rate': rate, 'delta_vs_base4': rate - base_filter['top3'].mean()
            })
    results_5way.sort(key=lambda x: -x['rate'])
    for r in results_5way[:20]:
        print(f'  {r["rate"]:.3f}  n={r["n"]:>4}  Δ vs base4: {r["delta_vs_base4"]:+.3f}  +{r["feature"]}')

    # 別 base 4-way: jockey_q5 + horse_q5 + class_down + 同厩舎
    print(f'\n=== alternative 4-way: class_down + horse_q5 + jockey_q5 + trainer_change=0 ===')
    alt = df[(df['class_down'] == 1) & (df['horse_q5'] == 1) &
              (df['jockey_q5'] == 1) & (df['trainer_change'] == 0)]
    print(f'  n={len(alt):,}, top3={alt["top3"].mean():.3f}')

    # 5-way 拡張
    for f5 in binaries:
        if f5 in ['class_down', 'horse_q5', 'jockey_q5']:
            continue
        sub = alt[alt[f5] == 1]
        if len(sub) < args.min_n:
            continue
        rate = sub['top3'].mean()
        if rate >= args.min_rate:
            print(f'  +{f5}: n={len(sub):>4}, rate={rate:.3f}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
