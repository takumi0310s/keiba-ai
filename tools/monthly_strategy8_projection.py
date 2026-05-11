#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""月別 戦略⑧ ROI projection (年間 12 ヶ月分の expected monthly PnL).

過去 4 年 data から 月別 Jackpot 出現率 + ROI を 計測し、 1 年間運用での
expected monthly PnL を 月別に projection。 user 投資 plan の 参考。

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
                               'popularity', 'num_horses', 'distance', 'condition',
                               'course'])
    df = df[df['year'] >= 22]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['win'] = (df['finish'] == 1).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)

    for fname in ['event_effect_features.csv', 'hot_streak_features.csv']:
        path = os.path.join(BASE_DIR, 'data', fname)
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
    df['is_jackpot'] = (
        (df['class_down'] == 1) & (df['horse_q5'] == 1) &
        (df['jockey_q5'] == 1) & (df['jockey_change'] == 0)
    ).astype(int)

    # bet sizing: Jackpot は 条件別 (戦略⑧ 設計通り)
    def jackpot_bet(row):
        cond = row.get('condition')
        nh = row.get('num_horses', 0)
        dist = row.get('distance', 0)
        course = row.get('course', '')
        heavy = cond in ['重', '不良']
        if nh <= 7: return 0  # E は除外
        if course == '06_特別': return 0  # 06_特別 除外
        if 1200 <= dist <= 1400: return 700  # D 控えめ
        if course == '京都': return 1500  # 京都 復活
        if nh >= 15 and heavy: return 1500  # X 復活
        if 8 <= nh <= 14 and heavy: return 1000  # B 復活
        return 1500  # A / C default

    df['jpot_bet'] = df.apply(jackpot_bet, axis=1)
    df['win_odds_est'] = df['popularity'].astype(int).map(POP_ODDS_WIN).fillna(50.0)
    df['jpot_pnl'] = df.apply(
        lambda r: r['jpot_bet'] * (r['win_odds_est'] - 1) if r['win'] else -r['jpot_bet']
        if r['is_jackpot'] == 1 and r['jpot_bet'] > 0 else 0, axis=1
    )
    df.loc[df['is_jackpot'] != 1, 'jpot_pnl'] = 0

    # 月別 集計 (4 年 cumulative → 12 ヶ月 平均 として projection)
    print('=== 月別 戦略⑧ Jackpot 期待 月利 projection ===\n')
    print(f'{"month":<5} {"jpot_n_4y":>10} {"jpot_n/mo":>10} {"win%":>7} {"投資/mo":>10} {"PnL/mo":>10} {"ROI":>7}')
    print('-' * 75)

    total_yearly = 0
    for m in range(1, 13):
        sub = df[df['month'] == m]
        jpot = sub[sub['is_jackpot'] == 1]
        if len(jpot) < 1:
            continue
        # / 4 年 で 1 年あたり、 さらに /12 = 月平均ではなく 該当月の年間 平均
        jpot_n_per_month = len(jpot) / 4  # 4 年分の月別 → 1 年あたり 該当月
        win_rate = jpot['win'].mean()
        inv_per_month = jpot['jpot_bet'].sum() / 4
        pnl_per_month = jpot['jpot_pnl'].sum() / 4
        roi = (pnl_per_month / max(1, inv_per_month) + 1) * 100
        total_yearly += pnl_per_month
        print(f'  {m:<5} {len(jpot):>10,} {jpot_n_per_month:>10.1f} '
              f'{win_rate*100:>6.1f}% ¥{inv_per_month:>8,.0f} '
              f'¥{pnl_per_month:>+8,.0f} {roi:>6.1f}%')

    print(f'\n=== TOTAL (Jackpot のみ、 4 年平均) ===')
    print(f'  年間 PnL: ¥{total_yearly:+,.0f}')
    print(f'  月平均 PnL: ¥{total_yearly/12:+,.0f}')

    # V15 baseline と合算
    print(f'\n=== 戦略⑧ TOTAL (V15 baseline + Jackpot) ===')
    v15_monthly = 28000  # CLAUDE.md 想定 月利
    jpot_monthly = total_yearly / 12
    total_monthly = v15_monthly + jpot_monthly
    print(f'  V15 戦略⑦ baseline: ¥{v15_monthly:+,}/月')
    print(f'  Jackpot 増分: ¥{jpot_monthly:+,.0f}/月')
    print(f'  戦略⑧ TOTAL: ¥{total_monthly:+,.0f}/月')
    print(f'  年間 想定: ¥{total_monthly*12:+,.0f}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
