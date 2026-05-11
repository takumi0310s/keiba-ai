#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""月別 / 季節別 Jackpot pattern 安定性 検証.

季節 / 月 によって signal が 変わるか。 G1 シーズン (春 / 秋) や 夏 ローカル で
pattern 効きやすさが 違う可能性。

【V15 投資保護】 分析のみ、 V15 model 不変
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


def main():
    import pandas as pd

    base = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    df = pd.read_csv(base, encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'horse_id', 'finish', 'year', 'month',
                               'popularity'])
    df = df[df['year'] >= 22]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['win'] = (df['finish'] == 1).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)

    for fname in ['event_effect_features.csv', 'hot_streak_features.csv']:
        path = os.path.join(BASE_DIR, 'data', fname)
        sub = pd.read_csv(path, encoding='utf-8', dtype={'race_id': str, 'horse_id': str})
        df = df.merge(sub, on=['race_id', 'horse_id'], how='left', suffixes=('', '_d'))
        df = df.drop(columns=[c for c in df.columns if c.endswith('_d')])

    df = df.dropna(subset=['horse_recent5_top3', 'jockey_recent30_top3',
                             'class_down', 'jockey_change', 'popularity'])
    df['horse_q5'] = (pd.qcut(df['horse_recent5_top3'].rank(method='first'),
                                 5, labels=[1, 2, 3, 4, 5]) == 5).astype(int)
    df['jockey_q5'] = (pd.qcut(df['jockey_recent30_top3'].rank(method='first'),
                                  5, labels=[1, 2, 3, 4, 5]) == 5).astype(int)
    df['is_jackpot'] = (
        (df['class_down'] == 1) & (df['horse_q5'] == 1) &
        (df['jockey_q5'] == 1) & (df['jockey_change'] == 0)
    ).astype(int)
    df['win_odds_est'] = df['popularity'].astype(int).map(POP_ODDS_WIN).fillna(50.0)
    df['win_pnl'] = df.apply(lambda r: 100 * (r['win_odds_est'] - 1) if r['win'] else -100, axis=1)

    print('=== 月別 Jackpot pattern stability ===\n')
    print(f'{"month":<5} {"n_all":>7} {"jpot_n":>7} {"jpot_top3":>10} {"jpot_win":>9} {"jpot_ROI":>9} {"base_ROI":>9}')
    print('-' * 70)
    for m in range(1, 13):
        sub = df[df['month'] == m]
        if len(sub) < 500:
            continue
        jpot = sub[sub['is_jackpot'] == 1]
        if len(jpot) < 5:
            continue
        jpot_inv = len(jpot) * 100
        base_inv = len(sub) * 100
        jpot_roi = (jpot['win_pnl'].sum() / jpot_inv + 1) * 100
        base_roi = (sub['win_pnl'].sum() / base_inv + 1) * 100
        print(f'  {m:<5} {len(sub):>7,} {len(jpot):>7,} '
              f'{jpot["top3"].mean():>10.3f} {jpot["win"].mean():>9.3f} '
              f'{jpot_roi:>8.1f}% {base_roi:>8.1f}%')

    # 季節 (春 3-5、 夏 6-8、 秋 9-11、 冬 12-2)
    def season(m):
        if m in [3, 4, 5]: return '春'
        if m in [6, 7, 8]: return '夏'
        if m in [9, 10, 11]: return '秋'
        return '冬'

    df['season'] = df['month'].apply(season)
    print('\n=== 季節別 Jackpot ROI ===')
    for s, sub in df.groupby('season'):
        if len(sub) < 1000:
            continue
        jpot = sub[sub['is_jackpot'] == 1]
        if len(jpot) < 20:
            continue
        jpot_inv = len(jpot) * 100
        base_inv = len(sub) * 100
        jpot_roi = (jpot['win_pnl'].sum() / jpot_inv + 1) * 100
        base_roi = (sub['win_pnl'].sum() / base_inv + 1) * 100
        print(f'  {s} (n={len(sub):,}、 jpot {len(jpot)}): jpot ROI {jpot_roi:.1f}% vs base {base_roi:.1f}%')

    return 0


if __name__ == '__main__':
    sys.exit(main())
