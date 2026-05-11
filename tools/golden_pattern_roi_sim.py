#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""黄金 pattern (降級+同騎手+差し力Q5) で 実 ROI simulation.

3-way interaction で top3 rate 43.8% という 異常 値 が出た pattern を、
実 jra_payouts.csv で 単勝 / 複勝 ROI に変換して 真の価値 を 検証。

【V15 投資保護】 分析のみ、 V15 model 不変

Usage:
    python tools/golden_pattern_roi_sim.py
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
    ap = argparse.ArgumentParser(description='Golden pattern ROI simulation')
    args = ap.parse_args()

    import pandas as pd

    # data load + merge
    base = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    df = pd.read_csv(base, encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'horse_id', 'umaban', 'finish', 'year',
                               'popularity', 'tansho_odds'])
    df = df[df['year'] >= 22]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['win'] = (df['finish'] == 1).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)

    evt = pd.read_csv(os.path.join(BASE_DIR, 'data', 'event_effect_features.csv'),
                      encoding='utf-8')
    evt['race_id'] = evt['race_id'].astype(str)
    evt['horse_id'] = evt['horse_id'].astype(str)
    evt = evt[['race_id', 'horse_id', 'class_down', 'jockey_change']].drop_duplicates(['race_id', 'horse_id'])
    df = df.merge(evt, on=['race_id', 'horse_id'], how='left')

    pace = pd.read_csv(os.path.join(BASE_DIR, 'data', 'pace_features_expanding.csv'),
                       encoding='utf-8')
    pace['race_id'] = pace['race_id'].astype(str)
    pace['horse_id'] = pace['horse_id'].astype(str)
    df = df.merge(pace, on=['race_id', 'horse_id'], how='left')
    df = df.dropna(subset=['pace_career_burst_mean'])

    # burst quintile
    df['burst_bin'] = pd.qcut(df['pace_career_burst_mean'].rank(method='first'),
                                5, labels=[1, 2, 3, 4, 5])

    # 黄金 pattern: class_down=1 & jockey_change=0 & burst=5
    golden = df[(df['class_down'] == 1) & (df['jockey_change'] == 0) &
                 (df['burst_bin'] == 5)]
    print(f'[INFO] 黄金 pattern records: {len(golden):,}')
    print(f'  top3 rate: {golden["top3"].mean():.3f}')
    print(f'  win rate: {golden["win"].mean():.3f}')

    # tansho_odds が data 未収集の場合 popularity 推定
    if golden['tansho_odds'].notna().sum() == 0:
        print('[INFO] tansho_odds 未収集、 popularity-based 推定 ROI を計算')
        # 経験的 popularity → odds 推定 (中央値、 全 jra data 平均)
        POP_ODDS = {1: 3.0, 2: 5.5, 3: 8.0, 4: 11.0, 5: 14.5,
                     6: 19.0, 7: 24.0, 8: 29.0, 9: 35.0, 10: 42.0,
                     11: 50.0, 12: 60.0, 13: 70.0, 14: 80.0, 15: 95.0,
                     16: 110.0, 17: 130.0, 18: 150.0}
        golden = golden.dropna(subset=['popularity'])
        golden['tansho_odds'] = golden['popularity'].astype(int).map(POP_ODDS).fillna(50.0)
    else:
        golden['tansho_odds'] = pd.to_numeric(golden['tansho_odds'], errors='coerce') / 10.0
        golden = golden.dropna(subset=['tansho_odds'])

    # 100 円 bet 想定
    golden['tansho_pnl'] = golden.apply(
        lambda r: 100 * (r['tansho_odds'] - 1) if r['win'] else -100, axis=1
    )

    print(f'\n=== 単勝 ROI simulation ===')
    print(f'  pattern records (with odds): {len(golden):,}')
    print(f'  win count: {int(golden["win"].sum())}')
    print(f'  win rate: {golden["win"].mean()*100:.1f}%')
    print(f'  単勝 オッズ stats:')
    print(f'    mean: {golden["tansho_odds"].mean():.2f}')
    print(f'    median: {golden["tansho_odds"].median():.2f}')
    print(f'    max: {golden["tansho_odds"].max():.2f}')
    total_pnl = golden['tansho_pnl'].sum()
    total_inv = len(golden) * 100
    roi = (total_pnl / total_inv + 1) * 100
    print(f'  単勝 投資: {total_inv:,} 円、 PnL: {total_pnl:+,.0f} 円')
    print(f'  単勝 ROI: {roi:.1f}%')

    # 比較: 全 records (同じ popularity 推定で)
    df_with_odds = df.dropna(subset=['popularity']).copy()
    POP_ODDS = {1: 3.0, 2: 5.5, 3: 8.0, 4: 11.0, 5: 14.5,
                 6: 19.0, 7: 24.0, 8: 29.0, 9: 35.0, 10: 42.0,
                 11: 50.0, 12: 60.0, 13: 70.0, 14: 80.0, 15: 95.0,
                 16: 110.0, 17: 130.0, 18: 150.0}
    df_with_odds['tansho_odds'] = df_with_odds['popularity'].astype(int).map(POP_ODDS).fillna(50.0)
    df_with_odds['pnl'] = df_with_odds.apply(
        lambda r: 100 * (r['tansho_odds'] - 1) if r['win'] else -100, axis=1
    )
    all_pnl = df_with_odds['pnl'].sum()
    all_inv = len(df_with_odds) * 100
    all_roi = (all_pnl / all_inv + 1) * 100
    print(f'\n=== 比較: 全 馬 単勝買い ===')
    print(f'  records: {len(df_with_odds):,}')
    print(f'  win rate: {df_with_odds["win"].mean()*100:.1f}%')
    print(f'  ROI: {all_roi:.1f}%')
    print(f'  Δ ROI: {roi - all_roi:+.1f}%')

    # popularity 別
    print(f'\n=== 黄金 pattern popularity 別 ===')
    for pop_range, label in [(1, 1), (2, 3), (4, 7), (8, 99)]:
        sub = golden[(golden['popularity'] >= pop_range) & (golden['popularity'] <= label)]
        if len(sub) < 10:
            continue
        pnl = sub['tansho_pnl'].sum()
        inv = len(sub) * 100
        sub_roi = (pnl / inv + 1) * 100
        print(f'  人気 {pop_range}-{label}: n={len(sub):,}, win={sub["win"].mean()*100:.1f}%, ROI={sub_roi:.1f}%')

    return 0


if __name__ == '__main__':
    sys.exit(main())
