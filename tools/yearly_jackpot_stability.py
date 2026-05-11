#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""4-way Jackpot pattern の 年度別 安定性 検証.

2022-2025 各年で Jackpot top3 rate / win rate が 一貫しているかチェック。
single year の noise でないことを 確認。

【V15 投資保護】 分析のみ、 V15 model 不変

Usage:
    python tools/yearly_jackpot_stability.py
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
                      usecols=['race_id', 'horse_id', 'finish', 'year', 'popularity'])
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

    df['win_odds'] = df['popularity'].astype(int).map(POP_ODDS_WIN).fillna(50.0)
    df['win_pnl'] = df.apply(lambda r: 100 * (r['win_odds'] - 1) if r['win'] else -100, axis=1)

    print('=== 年度別 4-way Jackpot pattern 安定性 ===\n')
    print(f'{"year":<5} {"all_n":>8} {"jpot_n":>7} {"jpot_top3":>10} {"jpot_win":>9} {"jpot_ROI":>9} {"base_top3":>10} {"base_ROI":>9}')
    print('-' * 80)

    for yy in [22, 23, 24, 25]:
        sub = df[df['year'] == yy]
        if len(sub) < 100:
            continue
        jpot = sub[sub['is_jackpot'] == 1]
        if len(jpot) < 5:
            continue
        jpot_inv = len(jpot) * 100
        base_inv = len(sub) * 100
        jpot_roi = (jpot['win_pnl'].sum() / jpot_inv + 1) * 100
        base_roi = (sub['win_pnl'].sum() / base_inv + 1) * 100
        print(f'  {yy:<5} {len(sub):>8,} {len(jpot):>7,} '
              f'{jpot["top3"].mean():>10.3f} {jpot["win"].mean():>9.3f} '
              f'{jpot_roi:>8.1f}% {sub["top3"].mean():>10.3f} {base_roi:>8.1f}%')

    print('\n[判定]')
    print('  各年で jpot ROI > base ROI なら 安定 signal')
    print('  jpot top3 rate が 60%+ で 一貫していれば 信頼性 高')

    # 5-way (+pop_1to3) でも 同様検証
    print('\n=== 年度別 5-way (Jackpot + pop_1to3) ===')
    print(f'{"year":<5} {"jpot5_n":>7} {"top3":>7} {"win":>7} {"ROI":>7}')
    for yy in [22, 23, 24, 25]:
        sub = df[df['year'] == yy]
        p5 = sub[(sub['is_jackpot'] == 1) & (sub['popularity'] >= 1) & (sub['popularity'] <= 3)]
        if len(p5) < 5:
            continue
        inv = len(p5) * 100
        roi = (p5['win_pnl'].sum() / inv + 1) * 100
        print(f'  {yy:<5} {len(p5):>7,} {p5["top3"].mean():>6.1%} '
              f'{p5["win"].mean():>6.1%} {roi:>6.1f}%')

    return 0


if __name__ == '__main__':
    sys.exit(main())
