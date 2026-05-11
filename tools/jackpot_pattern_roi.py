#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""4-way Jackpot pattern (top3 64.8%) の 単勝 + 複勝 ROI simulation.

class_down + horse_recent5_Q5 + jockey_recent30_Q5 + jockey_change=0 (n=596) を
popularity 推定 odds + 実 finish で ROI 計算。

【V15 投資保護】 分析のみ、 V15 model 不変

Usage:
    python tools/jackpot_pattern_roi.py
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
# 複勝 オッズ (経験的に 単勝 オッズ × 0.20-0.35、 ここでは 0.25)
POP_ODDS_PLACE = {k: max(1.1, v * 0.25) for k, v in POP_ODDS_WIN.items()}


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
        if not os.path.exists(path):
            continue
        sub = pd.read_csv(path, encoding='utf-8')
        sub['race_id'] = sub['race_id'].astype(str)
        sub['horse_id'] = sub['horse_id'].astype(str)
        df = df.merge(sub, on=['race_id', 'horse_id'], how='left', suffixes=('', '_d'))
        df = df.drop(columns=[c for c in df.columns if c.endswith('_d')])

    df = df.dropna(subset=['horse_recent5_top3', 'jockey_recent30_top3',
                             'class_down', 'jockey_change', 'popularity'])

    df['horse_q5'] = pd.qcut(df['horse_recent5_top3'].rank(method='first'),
                               5, labels=[1, 2, 3, 4, 5])
    df['jockey_q5'] = pd.qcut(df['jockey_recent30_top3'].rank(method='first'),
                                5, labels=[1, 2, 3, 4, 5])

    jackpot = df[
        (df['class_down'] == 1) &
        (df['horse_q5'] == 5) &
        (df['jockey_q5'] == 5) &
        (df['jockey_change'] == 0)
    ]
    print(f'[INFO] Jackpot pattern n={len(jackpot):,}')
    print(f'  top3 rate: {jackpot["top3"].mean():.3f}')
    print(f'  win rate: {jackpot["win"].mean():.3f}')

    # 単勝 ROI
    jackpot = jackpot.copy()
    jackpot['win_odds_est'] = jackpot['popularity'].astype(int).map(POP_ODDS_WIN).fillna(50.0)
    jackpot['place_odds_est'] = jackpot['popularity'].astype(int).map(POP_ODDS_PLACE).fillna(12.5)

    jackpot['win_pnl'] = jackpot.apply(
        lambda r: 100 * (r['win_odds_est'] - 1) if r['win'] else -100, axis=1)
    jackpot['place_pnl'] = jackpot.apply(
        lambda r: 100 * (r['place_odds_est'] - 1) if r['top3'] else -100, axis=1)

    print(f'\n=== 単勝 ROI ===')
    total_inv = len(jackpot) * 100
    total_pnl = jackpot['win_pnl'].sum()
    print(f'  N: {len(jackpot):,}')
    print(f'  invest: {total_inv:,} 円')
    print(f'  PnL: {total_pnl:+,.0f} 円')
    print(f'  ROI: {(total_pnl/total_inv + 1)*100:.1f}%')

    print(f'\n=== 複勝 ROI ===')
    total_pnl_place = jackpot['place_pnl'].sum()
    print(f'  PnL: {total_pnl_place:+,.0f} 円')
    print(f'  ROI: {(total_pnl_place/total_inv + 1)*100:.1f}%')

    # popularity 別
    print(f'\n=== 人気 別 単勝 ROI ===')
    for low, high in [(1, 1), (2, 3), (4, 7), (8, 99)]:
        sub = jackpot[(jackpot['popularity'] >= low) & (jackpot['popularity'] <= high)]
        if len(sub) < 5:
            continue
        sub_inv = len(sub) * 100
        sub_pnl = sub['win_pnl'].sum()
        sub_place = sub['place_pnl'].sum()
        print(f'  人気 {low}-{high}: n={len(sub):>3}, win={sub["win"].mean()*100:.1f}%, '
              f'top3={sub["top3"].mean()*100:.1f}%, '
              f'tan ROI={(sub_pnl/sub_inv + 1)*100:.1f}%, '
              f'pla ROI={(sub_place/sub_inv + 1)*100:.1f}%')

    return 0


if __name__ == '__main__':
    sys.exit(main())
