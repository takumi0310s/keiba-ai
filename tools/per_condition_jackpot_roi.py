#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""6 条件別 Jackpot pattern ROI 詳細 検証 (どの条件で 最も effective か).

V15 戦略⑦ は条件別 (A/B/C/D/E/X) 最適化済。 Jackpot pattern が どの条件で
特に強いか / 弱いか を 検証し、 V20 投入時の bet sizing 戦略 を細分化。

【V15 投資保護】 分析のみ、 V15 model 不変

Usage:
    python tools/per_condition_jackpot_roi.py
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

POP_ODDS_WIN = {1: 3.0, 2: 5.5, 3: 8.0, 4: 11.0, 5: 14.5,
                6: 19.0, 7: 24.0, 8: 29.0, 9: 35.0, 10: 42.0,
                11: 50.0, 12: 60.0, 13: 70.0, 14: 80.0, 15: 95.0,
                16: 110.0, 17: 130.0, 18: 150.0}


def classify_condition(row):
    nh = row.get('num_horses', 0)
    dist = row.get('distance', 0)
    cond_str = str(row.get('condition', ''))
    heavy = cond_str in ['重', '不良']
    if nh <= 7: return 'E'
    if dist <= 1400: return 'D'
    if 8 <= nh <= 14 and dist >= 1600 and not heavy: return 'A'
    if 8 <= nh <= 14 and dist >= 1600 and heavy: return 'B'
    if nh >= 15 and dist >= 1600 and not heavy: return 'C'
    return 'X'


def main():
    import pandas as pd

    base = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    df = pd.read_csv(base, encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'horse_id', 'finish', 'year',
                               'popularity', 'num_horses', 'distance', 'condition'])
    df = df[df['year'] >= 22]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['win'] = (df['finish'] == 1).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)

    for fname in ['event_effect_features.csv', 'hot_streak_features.csv']:
        path = os.path.join(BASE_DIR, 'data', fname)
        if not os.path.exists(path):
            continue
        sub = pd.read_csv(path, encoding='utf-8',
                           dtype={'race_id': str, 'horse_id': str})
        df = df.merge(sub, on=['race_id', 'horse_id'], how='left', suffixes=('', '_d'))
        df = df.drop(columns=[c for c in df.columns if c.endswith('_d')])

    df = df.dropna(subset=['horse_recent5_top3', 'jockey_recent30_top3',
                             'class_down', 'jockey_change', 'popularity'])

    # 条件分類
    df['cond_cat'] = df.apply(classify_condition, axis=1)

    # Q5 bins
    df['horse_q5'] = (pd.qcut(df['horse_recent5_top3'].rank(method='first'),
                                 5, labels=[1, 2, 3, 4, 5]) == 5).astype(int)
    df['jockey_q5'] = (pd.qcut(df['jockey_recent30_top3'].rank(method='first'),
                                  5, labels=[1, 2, 3, 4, 5]) == 5).astype(int)

    # Jackpot 該当
    df['is_jackpot'] = (
        (df['class_down'] == 1) & (df['horse_q5'] == 1) &
        (df['jockey_q5'] == 1) & (df['jockey_change'] == 0)
    ).astype(int)

    df['win_odds_est'] = df['popularity'].astype(int).map(POP_ODDS_WIN).fillna(50.0)
    df['win_pnl'] = df.apply(
        lambda r: 100 * (r['win_odds_est'] - 1) if r['win'] else -100, axis=1
    )

    print('=== 条件別 Jackpot pattern ROI ===\n')
    print(f'{"cond":<5} {"n":>6} {"jpot_n":>7} {"jpot%":>7} {"jpot_top3":>10} {"jpot_win":>10} {"jpot_ROI":>9} {"base_top3":>10} {"base_ROI":>9}')
    print('-' * 90)
    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        sub = df[df['cond_cat'] == cond]
        if len(sub) < 100:
            continue
        jpot = sub[sub['is_jackpot'] == 1]
        if len(jpot) < 5:
            print(f'  {cond:<5} {len(sub):>6,} {len(jpot):>7} (n少なすぎ)')
            continue
        jpot_inv = len(jpot) * 100
        base_inv = len(sub) * 100
        jpot_roi = (jpot['win_pnl'].sum() / jpot_inv + 1) * 100
        base_roi = (sub['win_pnl'].sum() / base_inv + 1) * 100
        print(f'  {cond:<5} {len(sub):>6,} {len(jpot):>7} {len(jpot)/len(sub)*100:>6.2f}% '
              f'{jpot["top3"].mean():>9.3f} {jpot["win"].mean():>10.3f} '
              f'{jpot_roi:>8.1f}% {sub["top3"].mean():>9.3f} {base_roi:>8.1f}%')

    print('\n=== 条件別 + 人気別 Jackpot ROI ===')
    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        cond_sub = df[(df['cond_cat'] == cond) & (df['is_jackpot'] == 1)]
        if len(cond_sub) < 10:
            continue
        print(f'\n--- 条件 {cond} ({len(cond_sub)} 件) ---')
        for low, high, label in [(1, 1, '1人気'), (2, 3, '2-3人気'),
                                    (4, 7, '4-7人気'), (8, 99, '8人気+')]:
            ssub = cond_sub[(cond_sub['popularity'] >= low) & (cond_sub['popularity'] <= high)]
            if len(ssub) < 3:
                continue
            inv = len(ssub) * 100
            pnl = ssub['win_pnl'].sum()
            roi = (pnl / inv + 1) * 100
            print(f'  {label:<10} n={len(ssub):>3}  win={ssub["win"].mean()*100:>5.1f}%  '
                  f'top3={ssub["top3"].mean()*100:>5.1f}%  ROI={roi:>6.1f}%')

    return 0


if __name__ == '__main__':
    sys.exit(main())
