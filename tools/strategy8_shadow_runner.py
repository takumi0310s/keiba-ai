#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""戦略⑧ shadow runner: V15 戦略⑦ + Jackpot pattern を combine、 別 channel 通知 推奨.

V15 production は完全 不変、 daily_predictions/{date}.csv を読み取って
- V15 戦略⑦ recommendations
- Jackpot 該当馬 alert
を一括 表示。 5/17 試験用 + 5/24+ integration 候補。

【V15 投資保護】 V15 production 一切 unchanged、 shadow output のみ

Usage:
    python tools/strategy8_shadow_runner.py 20260517
    python tools/strategy8_shadow_runner.py 20260510  # 過去 verify
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 戦略⑦ 除外
STRATEGY_7_EXCLUDE = {'06_特別', '京都', 'E', 'B'}

# 戦略⑧ Jackpot 該当時 復活 (V15 で除外中の race も)
JACKPOT_REVIVE_CONDITIONS = {'X', '京都', 'B'}
JACKPOT_REVIVE_BET = {
    'A': 1500, 'C': 1500, 'X': 1500, '京都': 1500,
    'B': 1000, 'D': 700,
    'E': 0,  # E は base 高くて 不要
    '06_特別': 0,  # 06_特別 は 復活せず
}


def main():
    ap = argparse.ArgumentParser(description='戦略⑧ shadow runner')
    ap.add_argument('date', help='YYYYMMDD')
    ap.add_argument('--bankroll', type=float, default=30000)
    args = ap.parse_args()

    import pandas as pd

    # 1. daily_predictions read
    daily_path = os.path.join(BASE_DIR, 'data', 'daily_predictions', f'{args.date}.csv')
    if not os.path.exists(daily_path):
        print(f'[ERROR] {daily_path} 未生成')
        return 1
    pred_df = pd.read_csv(daily_path, encoding='utf-8-sig')
    pred_df['race_id'] = pred_df['race_id'].astype(str)
    print(f'[INFO] {len(pred_df)} races for {args.date}')

    # 2. features merge for Jackpot detect
    hs_path = os.path.join(BASE_DIR, 'data', 'hot_streak_features.csv')
    ev_path = os.path.join(BASE_DIR, 'data', 'event_effect_features.csv')

    if not all(os.path.exists(p) for p in [hs_path, ev_path]):
        print('[WARN] features 未生成、 Jackpot detect 不可')
        hs = None
        ev = None
    else:
        hs = pd.read_csv(hs_path, encoding='utf-8',
                          dtype={'race_id': str, 'horse_id': str})
        ev = pd.read_csv(ev_path, encoding='utf-8',
                          dtype={'race_id': str, 'horse_id': str})

    # 3. 各 race を 戦略⑦ + Jackpot で 評価
    output_lines = []
    output_lines.append(f'=== 戦略⑧ shadow recommendations: {args.date} ===\n')

    v15_total_bet = 0
    jackpot_total_bet = 0
    n_v15_bet = 0
    n_jackpot = 0

    for _, p in pred_df.iterrows():
        race_id = p['race_id']
        cond = str(p.get('condition', '?'))
        course = str(p.get('course', '?'))
        race_num = p.get('race_num', '?')
        race_name = str(p.get('race_name', '?'))

        # 戦略⑦ 判定
        is_excluded = (cond in STRATEGY_7_EXCLUDE or course in STRATEGY_7_EXCLUDE)
        v15_bet = 0
        if not is_excluded:
            # 条件 E 馬連 700、 他 trio 700
            v15_bet = 700
            n_v15_bet += 1
            v15_total_bet += v15_bet

        # Jackpot detect
        jackpot_horses = []
        if hs is not None and ev is not None:
            hs_race = hs[hs['race_id'] == race_id]
            ev_race = ev[ev['race_id'] == race_id]
            if not hs_race.empty and not ev_race.empty:
                merged = hs_race.merge(ev_race, on=['race_id', 'horse_id'],
                                          how='inner', suffixes=('', '_d'))
                for _, h in merged.iterrows():
                    if (h.get('class_down', 0) == 1 and
                        h.get('horse_recent5_top3', 0) >= 0.6 and
                        h.get('jockey_recent30_top3', 0) >= 0.30 and
                        h.get('jockey_change', 1) == 0):
                        jackpot_horses.append(h['horse_id'])

        # 戦略⑧ 推奨
        line = f'{race_id} {course} {race_num}R {race_name[:20]}'
        line += f'  cond={cond}'
        if v15_bet > 0:
            line += f'  V15-bet={v15_bet}'
        elif is_excluded:
            line += f'  V15-SKIP({cond if cond in STRATEGY_7_EXCLUDE else course})'

        if jackpot_horses:
            j_bet = JACKPOT_REVIVE_BET.get(cond, JACKPOT_REVIVE_BET.get(course, 0))
            if j_bet > 0:
                n_jackpot += 1
                jackpot_total_bet += j_bet
                line += f'  🎰JACKPOT(+¥{j_bet}/horse): {len(jackpot_horses)} horses {jackpot_horses[:3]}'
            else:
                line += f'  🎰Jackpot 該当 but 戦略⑧除外'

        output_lines.append(line)

    # Summary
    output_lines.append('\n=== SUMMARY ===')
    output_lines.append(f'  V15 戦略⑦ bet count: {n_v15_bet} race / 投資 ¥{v15_total_bet:,}')
    output_lines.append(f'  戦略⑧ Jackpot 追加 bet: {n_jackpot} race / 投資 ¥{jackpot_total_bet:,}')
    output_lines.append(f'  合計 投資 (戦略⑧): ¥{v15_total_bet + jackpot_total_bet:,}')
    output_lines.append(f'  bankroll: ¥{args.bankroll:,.0f}')

    # 出力
    out_dir = os.path.join(BASE_DIR, 'data', 'strategy8_shadow')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.date}.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f'[OK] saved: {out_path}')
    print('\n'.join(output_lines[-15:]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
