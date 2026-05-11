#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""戦略⑧ vs V15 戦略⑦ 4 年 backtest (2022-2025).

戦略⑦ (現行 V15) と 戦略⑧ (V15 + Jackpot bet 追加) の 4 年累積 ROI 比較。
popularity 推定 odds なので 実 odds より conservative だが、 V15 単独 vs 戦略⑧
の **差** が 明確に出る。

【V15 投資保護】 backtest のみ、 V15 model 不変

Usage:
    python tools/strategy8_vs_v15_4year_backtest.py
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

# popularity → odds 推定
POP_ODDS_WIN = {1: 3.0, 2: 5.5, 3: 8.0, 4: 11.0, 5: 14.5,
                6: 19.0, 7: 24.0, 8: 29.0, 9: 35.0, 10: 42.0,
                11: 50.0, 12: 60.0, 13: 70.0, 14: 80.0, 15: 95.0,
                16: 110.0, 17: 130.0, 18: 150.0}

# 戦略⑦ exclusion
S7_EXCLUDE_CONDITIONS = {'E', 'B'}  # course/cond_cat
S7_EXCLUDE_COURSES = {'京都', '06_特別'}

# 戦略⑧ Jackpot bet
JACKPOT_BET_CONDITION = {'A': 1500, 'C': 1500, 'X': 1500, 'B': 1000, 'D': 700, 'E': 0}
JACKPOT_BET_KYOTO = 1500  # 京都 復活


def classify_condition(num_horses, distance, condition):
    """V15 条件分類."""
    heavy = condition in ['重', '不良']
    if num_horses <= 7:
        return 'E'
    if distance <= 1400:
        return 'D'
    if 8 <= num_horses <= 14 and distance >= 1600 and not heavy:
        return 'A'
    if 8 <= num_horses <= 14 and distance >= 1600 and heavy:
        return 'B'
    if num_horses >= 15 and distance >= 1600 and not heavy:
        return 'C'
    return 'X'


def main():
    import pandas as pd

    base = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    df = pd.read_csv(base, encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'horse_id', 'finish', 'year',
                               'popularity', 'num_horses', 'distance',
                               'condition', 'course'])
    df = df[df['year'] >= 22]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['win'] = (df['finish'] == 1).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)
    df['cond_cat'] = df.apply(lambda r: classify_condition(r['num_horses'],
                                                              r['distance'],
                                                              r['condition']), axis=1)
    print(f'[INFO] base: {len(df):,}')

    # features merge
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

    # ===========================
    # 戦略⑦ (V15 現行): race ごとに top1 馬 で trio bet
    # 簡略化のため: 単馬 単勝 風 PnL で 比較 (1 race 1 馬、 700円固定)
    # 注意: 実際の V15 は trio bet で payout 高め
    # ===========================
    print('\n=== 戦略⑦ (V15 現行 simulation) ===')
    s7 = df.copy()
    s7['s7_eligible'] = ~(
        s7['cond_cat'].isin(S7_EXCLUDE_CONDITIONS) |
        s7['course'].isin(S7_EXCLUDE_COURSES)
    )
    # race 単位で top1 horse_id (popularity=1 を 簡易 代用、 実 V15 は scores から選ぶ)
    s7_bet = s7[s7['s7_eligible'] & (s7['popularity'] == 1)].copy()
    s7_bet['win_pnl'] = s7_bet.apply(
        lambda r: 700 * (r['win_odds_est'] - 1) / 1.0 if r['win'] else -700, axis=1
    )
    n_s7 = len(s7_bet)
    inv_s7 = n_s7 * 700
    pnl_s7 = s7_bet['win_pnl'].sum()
    roi_s7 = (pnl_s7 / inv_s7 + 1) * 100 if inv_s7 > 0 else 0
    print(f'  N race (simulated): {n_s7:,}')
    print(f'  投資: ¥{inv_s7:,}')
    print(f'  PnL: ¥{pnl_s7:+,.0f}')
    print(f'  ROI: {roi_s7:.1f}%')

    # ===========================
    # 戦略⑧: 戦略⑦ + Jackpot 該当 horse に 単勝 追加 bet
    # ===========================
    print('\n=== 戦略⑧ (戦略⑦ + Jackpot 単勝 追加) ===')
    s8_bet_v15 = s7_bet.copy()  # V15 portion 同じ

    # Jackpot 該当 馬 (戦略⑦ 除外 race の Jackpot 復活分 も含む)
    jackpot_df = df[df['is_jackpot'] == 1].copy()
    # Jackpot bet 金額
    def jackpot_bet_amount(row):
        cond = row['cond_cat']
        course = row['course']
        if cond in S7_EXCLUDE_CONDITIONS or course in S7_EXCLUDE_COURSES:
            # 戦略⑦ 除外中: 京都 / B / X 復活
            if cond == 'B': return 1000
            if course == '京都': return 1500
            if cond == 'X': return 1500
            # E / 06_特別 / その他 は 復活せず
            return 0
        # 戦略⑦ 有効: 通常 Jackpot 追加
        return JACKPOT_BET_CONDITION.get(cond, 0)

    jackpot_df['jpot_bet'] = jackpot_df.apply(jackpot_bet_amount, axis=1)
    jackpot_df['jpot_pnl'] = jackpot_df.apply(
        lambda r: r['jpot_bet'] * (r['win_odds_est'] - 1) if r['win'] else -r['jpot_bet'], axis=1
    )
    jackpot_active = jackpot_df[jackpot_df['jpot_bet'] > 0]
    inv_jpot = jackpot_active['jpot_bet'].sum()
    pnl_jpot = jackpot_active['jpot_pnl'].sum()
    roi_jpot = (pnl_jpot / inv_jpot + 1) * 100 if inv_jpot > 0 else 0

    print(f'  Jackpot bets: {len(jackpot_active):,}')
    print(f'  Jackpot 投資: ¥{inv_jpot:,.0f}')
    print(f'  Jackpot PnL: ¥{pnl_jpot:+,.0f}')
    print(f'  Jackpot ROI: {roi_jpot:.1f}%')

    # 合計 戦略⑧
    total_inv_s8 = inv_s7 + inv_jpot
    total_pnl_s8 = pnl_s7 + pnl_jpot
    roi_s8 = (total_pnl_s8 / total_inv_s8 + 1) * 100 if total_inv_s8 > 0 else 0
    print(f'\n=== 戦略⑧ TOTAL ===')
    print(f'  投資: ¥{total_inv_s8:,.0f}')
    print(f'  PnL: ¥{total_pnl_s8:+,.0f}')
    print(f'  ROI: {roi_s8:.1f}%')

    # 比較
    print(f'\n=== 比較 SUMMARY (4 年 cumulative) ===')
    print(f'  戦略⑦ (V15)     : ROI {roi_s7:.1f}%、 PnL ¥{pnl_s7:+,.0f}')
    print(f'  戦略⑧ (V15+JP)  : ROI {roi_s8:.1f}%、 PnL ¥{total_pnl_s8:+,.0f}')
    print(f'  差分            : +¥{total_pnl_s8 - pnl_s7:,.0f}')
    print(f'  月利 想定       : V15 +¥{pnl_s7/48:,.0f}/月、 戦略⑧ +¥{total_pnl_s8/48:,.0f}/月')

    return 0


if __name__ == '__main__':
    sys.exit(main())
