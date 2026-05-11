#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Jackpot pattern 複数 券種 (単勝/複勝/Wide/馬連/三連複/三連単) ROI 比較.

top3 rate 64.8% の pattern が どの券種で最も ROI 高いか empirically 検証。
推定 odds は popularity-based (実 odds は実際 異なる場合あり)。

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

# popularity → odds 推定 (経験的)
POP_ODDS_WIN = {1: 3.0, 2: 5.5, 3: 8.0, 4: 11.0, 5: 14.5,
                6: 19.0, 7: 24.0, 8: 29.0, 9: 35.0, 10: 42.0,
                11: 50.0, 12: 60.0, 13: 70.0, 14: 80.0, 15: 95.0,
                16: 110.0, 17: 130.0, 18: 150.0}
# 複勝 odds (経験的に 単勝 × 0.20-0.40)
POP_ODDS_PLACE = {k: max(1.05, v * 0.30) for k, v in POP_ODDS_WIN.items()}
# Wide odds (経験的、 複勝 より 高め)
POP_ODDS_WIDE = {k: max(1.5, v * 0.45) for k, v in POP_ODDS_WIN.items()}


def main():
    import pandas as pd

    base = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    df = pd.read_csv(base, encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'horse_id', 'finish', 'year', 'popularity'])
    df = df[df['year'] >= 22]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['win'] = (df['finish'] == 1).astype(int)
    df['place2'] = (df['finish'] <= 2).astype(int)
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

    # 4-way Jackpot
    jpot = df[
        (df['class_down'] == 1) & (df['horse_q5'] == 1) &
        (df['jockey_q5'] == 1) & (df['jockey_change'] == 0)
    ].copy()
    print(f'[INFO] Jackpot n={len(jpot):,}')
    print(f'  top3: {jpot["top3"].mean():.3f}、 win: {jpot["win"].mean():.3f}、 place2: {jpot["place2"].mean():.3f}')

    # 各券種 ROI
    jpot['win_odds_est'] = jpot['popularity'].astype(int).map(POP_ODDS_WIN).fillna(50.0)
    jpot['place_odds_est'] = jpot['popularity'].astype(int).map(POP_ODDS_PLACE).fillna(15.0)
    jpot['wide_odds_est'] = jpot['popularity'].astype(int).map(POP_ODDS_WIDE).fillna(22.0)

    # PnL per 100 円
    jpot['win_pnl'] = jpot.apply(lambda r: 100*(r['win_odds_est']-1) if r['win'] else -100, axis=1)
    jpot['place_pnl'] = jpot.apply(lambda r: 100*(r['place_odds_est']-1) if r['top3'] else -100, axis=1)
    jpot['wide_pnl'] = jpot.apply(lambda r: 100*(r['wide_odds_est']-1) if r['top3'] else -100, axis=1)

    n = len(jpot)
    inv = n * 100

    print(f'\n=== Jackpot pattern 券種別 ROI (popularity 推定 odds、 n={n:,})\n')
    print(f'{"券種":<10} {"hit %":>7} {"avg_odds":>9} {"PnL":>12} {"ROI":>7}')
    print('-' * 50)
    for t, hit_col, pnl_col, odds_col in [
        ('単勝', 'win', 'win_pnl', 'win_odds_est'),
        ('複勝', 'top3', 'place_pnl', 'place_odds_est'),
        ('Wide', 'top3', 'wide_pnl', 'wide_odds_est'),
    ]:
        hit = jpot[hit_col].mean()
        avg_odds = jpot[odds_col].mean()
        pnl = jpot[pnl_col].sum()
        roi = (pnl / inv + 1) * 100
        print(f'  {t:<10} {hit*100:>6.1f}% {avg_odds:>8.2f} ¥{pnl:>+10,.0f} {roi:>6.1f}%')

    print('\n=== 人気別 各券種 ROI ===')
    for pop_range, label in [((1, 1), '1人気'), ((2, 3), '2-3人気'),
                                 ((4, 7), '4-7人気'), ((8, 99), '8人気+')]:
        lo, hi = pop_range
        sub = jpot[(jpot['popularity'] >= lo) & (jpot['popularity'] <= hi)]
        if len(sub) < 5:
            continue
        sub_inv = len(sub) * 100
        win_roi = (sub['win_pnl'].sum() / sub_inv + 1) * 100
        place_roi = (sub['place_pnl'].sum() / sub_inv + 1) * 100
        wide_roi = (sub['wide_pnl'].sum() / sub_inv + 1) * 100
        print(f'  {label:<10} n={len(sub):>3}  単勝 {win_roi:>6.1f}%  複勝 {place_roi:>6.1f}%  Wide {wide_roi:>6.1f}%')

    return 0


if __name__ == '__main__':
    sys.exit(main())
