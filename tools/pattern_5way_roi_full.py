#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""5-way pattern (4-way Jackpot + pop_1to3) 詳細 ROI 検証 + 投資 sizing.

実運用に直接 使える pattern (n=456) の 単勝 / 複勝 ROI + bet sizing 提案。

【V15 投資保護】 分析のみ、 V15 model 不変

Usage:
    python tools/pattern_5way_roi_full.py
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
POP_ODDS_PLACE = {k: max(1.05, min(v * 0.25, v - 0.1)) for k, v in POP_ODDS_WIN.items()}


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
        if not os.path.exists(path):
            continue
        sub = pd.read_csv(path, encoding='utf-8',
                           dtype={'race_id': str, 'horse_id': str})
        df = df.merge(sub, on=['race_id', 'horse_id'], how='left', suffixes=('', '_d'))
        df = df.drop(columns=[c for c in df.columns if c.endswith('_d')])

    df = df.dropna(subset=['horse_recent5_top3', 'jockey_recent30_top3',
                             'class_down', 'jockey_change', 'popularity'])
    df['horse_q5'] = (pd.qcut(df['horse_recent5_top3'].rank(method='first'),
                                 5, labels=[1, 2, 3, 4, 5]) == 5).astype(int)
    df['jockey_q5'] = (pd.qcut(df['jockey_recent30_top3'].rank(method='first'),
                                  5, labels=[1, 2, 3, 4, 5]) == 5).astype(int)

    # 5-way pattern
    p5 = df[(df['class_down'] == 1) & (df['horse_q5'] == 1) &
             (df['jockey_q5'] == 1) & (df['jockey_change'] == 0) &
             (df['popularity'] >= 1) & (df['popularity'] <= 3)]
    print(f'[INFO] 5-way pattern (Jackpot + pop_1to3) n={len(p5):,}')
    print(f'  top3: {p5["top3"].mean()*100:.1f}%')
    print(f'  win: {p5["win"].mean()*100:.1f}%')
    print(f'  place2: {p5["place2"].mean()*100:.1f}%')

    # ROI
    p5 = p5.copy()
    p5['win_odds'] = p5['popularity'].astype(int).map(POP_ODDS_WIN).fillna(50.0)
    p5['place_odds'] = p5['popularity'].astype(int).map(POP_ODDS_PLACE).fillna(12.5)

    p5['win_pnl'] = p5.apply(lambda r: 100 * (r['win_odds'] - 1) if r['win'] else -100, axis=1)
    p5['place_pnl'] = p5.apply(lambda r: 100 * (r['place_odds'] - 1) if r['top3'] else -100, axis=1)

    n = len(p5)
    inv = n * 100
    win_roi = (p5['win_pnl'].sum() / inv + 1) * 100
    place_roi = (p5['place_pnl'].sum() / inv + 1) * 100

    print(f'\n=== ROI (popularity 推定 odds) ===')
    print(f'  N: {n}、 investment 100円/race')
    print(f'  単勝 PnL: {p5["win_pnl"].sum():+,.0f}、 ROI: {win_roi:.1f}%')
    print(f'  複勝 PnL: {p5["place_pnl"].sum():+,.0f}、 ROI: {place_roi:.1f}%')

    print(f'\n=== 人気別 詳細 ===')
    for pop in [1, 2, 3]:
        sub = p5[p5['popularity'] == pop]
        if len(sub) < 5:
            continue
        sub_inv = len(sub) * 100
        wp = sub['win_pnl'].sum()
        pp = sub['place_pnl'].sum()
        print(f'  {pop} 人気: n={len(sub):>3}、 win={sub["win"].mean()*100:>5.1f}%、 '
              f'top3={sub["top3"].mean()*100:>5.1f}%、 '
              f'単勝 ROI={(wp/sub_inv+1)*100:>6.1f}%、 複勝 ROI={(pp/sub_inv+1)*100:>6.1f}%')

    # Kelly bet sizing 推奨 (bankroll 30,000)
    bankroll = 30000
    print(f'\n=== 推奨 Kelly bet sizing (bankroll {bankroll:,} 円、 fraction 0.25x) ===')
    for pop in [1, 2, 3]:
        sub = p5[p5['popularity'] == pop]
        if len(sub) < 5:
            continue
        win_rate = sub['win'].mean()
        odds = POP_ODDS_WIN.get(pop, 8.0)
        b = odds - 1
        kelly_f = max(0, (b * win_rate - (1 - win_rate)) / b) * 0.25
        cap = bankroll * 0.05
        bet = min(bankroll * kelly_f, cap)
        bet = int(bet // 100 * 100)
        print(f'  {pop} 人気: win={win_rate:.3f}、 odds={odds:.1f}、 Kelly f={kelly_f:.3f}、 bet={bet:,} 円')

    # 月利 試算
    print(f'\n=== 月利 試算 (年間 推定) ===')
    year_n = len(p5) / 4  # 4 年 data → 1 年
    month_n = year_n / 12
    avg_win_bet = 1500  # Kelly 推奨 cap
    avg_win_roi_pct = (win_roi - 100) / 100  # 純 ROI 比率
    monthly_pnl = month_n * avg_win_bet * avg_win_roi_pct
    print(f'  年間想定 5-way 該当 race: {year_n:.0f} 件')
    print(f'  月間 想定 該当: {month_n:.1f} 件')
    print(f'  推定 月利 (1500円/race、 ROI {win_roi:.0f}%): +¥{monthly_pnl:,.0f}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
