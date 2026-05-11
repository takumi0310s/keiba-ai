#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B6: Closing odds drift model PoC.

5min interval オッズ 変動から 「真の市場確率 (closing)」 と V15 model 確率の gap を検出し、
EV opportunity を抽出する。 後場で オッズ が伸びる馬 = 市場 underestimate の sign。

【アプローチ】
- 5 分前 オッズ / 直前 オッズ / 締切 オッズ の比較
- drift_factor = closing_odds / opening_odds
  - >1: 後場で人気下落 (closing > opening) → 市場 信頼度低下
  - <1: 後場で人気上昇 → 市場 信頼度上昇
- V15 model 確率 と closing_market_prob の 比較 → EV opportunity detection

【入力】 odds_base_*.csv の 5min snapshot 履歴
【出力】 data/closing_drift_features.csv (race_id × umaban × drift features)

【V15 投資保護】 V15 model / production 一切不変、 補助 features のみ

Usage:
    python tools/closing_odds_drift.py 20260510
    python tools/closing_odds_drift.py demo
"""
import argparse
import csv
import glob
import os
import sys
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def market_implied_prob(odds, market_take=0.20):
    """odds から market implied probability (control by take rate)."""
    if odds <= 1.0:
        return 0.0
    return (1 - market_take) / odds


def cmd_demo(args):
    """Synthetic data で drift detection PoC."""
    import random
    random.seed(42)
    print('=== closing odds drift model demo ===\n')
    print('シナリオ: 12 頭立て、 V15 model 確率 + opening/closing odds の対比\n')

    n = 12
    # V15 model 確率
    v15_probs = sorted([random.uniform(0.05, 0.25) for _ in range(n)], reverse=True)
    total = sum(v15_probs); v15_probs = [p/total for p in v15_probs]

    print(f'{"馬番":>4} | {"V15確率":>8} | {"始値odds":>8} | {"締切odds":>8} | {"drift":>6} | {"市場確率":>8} | {"EV":>6}')
    print('-' * 80)
    insights = []
    for i, p in enumerate(v15_probs):
        umaban = i + 1
        # opening: V15 prob から控除率込み逆算 + noise
        opening_odds = (1 - 0.20) / p * random.uniform(0.85, 1.15)
        # closing: ある程度 真の確率に近づく + 馬によっては大きく ずれる
        if i < 3:  # 人気馬: 締切で 引き締まる
            drift = random.uniform(0.85, 1.05)
        elif i < 6:  # 中位: 振動
            drift = random.uniform(0.92, 1.12)
        else:  # 穴: 締切で 大きく 振れる
            drift = random.uniform(0.75, 1.35)
        closing_odds = opening_odds * drift
        market_p = market_implied_prob(closing_odds)
        ev = p * closing_odds - 1
        print(f'{umaban:>4} | {p:>8.3f} | {opening_odds:>8.2f} | {closing_odds:>8.2f} | '
              f'{drift:>6.2f} | {market_p:>8.3f} | {ev:>+6.2f}')

        # gap detection: V15 確率 > 市場 確率 → 期待値 +
        gap = p - market_p
        if gap > 0.03 and ev > 0:
            insights.append((umaban, p, closing_odds, gap, ev))

    print('\n[EV opportunity (V15 > market 確率 + EV>0)]')
    for ub, p, odds, gap, ev in insights:
        print(f'  馬番 {ub}: V15={p:.3f}, 市場={p-gap:.3f}, gap={gap:+.3f}, EV={ev:+.2f}')

    return 0


def cmd_extract(args):
    """odds_base_*.csv 履歴 から drift features 抽出 (実装の skeleton)."""
    date = args.date
    pattern = os.path.join(BASE_DIR, 'data', f'odds_base_{date}*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'[ERROR] no odds files for {date}: {pattern}')
        return 1

    print(f'[INFO] {len(files)} odds snapshot files for {date}')

    import pandas as pd
    all_snapshots = []
    for fp in files:
        try:
            df = pd.read_csv(fp, encoding='utf-8', low_memory=False)
            df['snapshot_file'] = os.path.basename(fp)
            all_snapshots.append(df)
        except Exception as e:
            print(f'  [WARN] {fp}: {e}')

    if not all_snapshots:
        return 1
    combined = pd.concat(all_snapshots, ignore_index=True)
    print(f'[INFO] combined: {combined.shape}')

    # detect odds 列
    odds_col = None
    for c in ['odds', 'tansho_odds', 'win_odds', 'odds_win']:
        if c in combined.columns:
            odds_col = c
            break
    if not odds_col:
        print(f'[ERROR] no odds column found. cols: {list(combined.columns)[:15]}')
        return 1

    print(f'[INFO] using odds column: {odds_col}')

    # race_id × 馬番 で 最初 vs 最後 snapshot
    uma_col = None
    for c in ['umaban', 'horse_num', 'horse_number']:
        if c in combined.columns:
            uma_col = c
            break
    if 'race_id' not in combined.columns or uma_col is None:
        print(f'[ERROR] race_id or 馬番 column missing. cols: {list(combined.columns)[:8]}')
        return 1
    if uma_col != 'umaban':
        combined['umaban'] = combined[uma_col]
        print(f'[INFO] mapped {uma_col} → umaban')

    grouped = combined.groupby(['race_id', 'umaban'])
    drift_records = []
    for (race_id, umaban), g in grouped:
        g_sorted = g.sort_values('snapshot_file')
        if len(g_sorted) < 2:
            continue
        opening_odds = float(g_sorted[odds_col].iloc[0]) if g_sorted[odds_col].iloc[0] > 0 else None
        closing_odds = float(g_sorted[odds_col].iloc[-1]) if g_sorted[odds_col].iloc[-1] > 0 else None
        if opening_odds is None or closing_odds is None or opening_odds <= 1.0:
            continue
        drift = closing_odds / opening_odds
        market_p_open = market_implied_prob(opening_odds)
        market_p_close = market_implied_prob(closing_odds)
        drift_records.append({
            'race_id': race_id, 'umaban': umaban,
            'opening_odds': round(opening_odds, 2),
            'closing_odds': round(closing_odds, 2),
            'drift_factor': round(drift, 3),
            'market_prob_open': round(market_p_open, 4),
            'market_prob_close': round(market_p_close, 4),
            'market_prob_delta': round(market_p_close - market_p_open, 4),
        })

    if not drift_records:
        print('[WARN] no valid drift records')
        return 1

    out_path = os.path.join(BASE_DIR, 'data', f'closing_drift_{date}.csv')
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(drift_records[0].keys()))
        w.writeheader()
        w.writerows(drift_records)
    print(f'[OK] {len(drift_records)} drift records saved: {out_path}')
    return 0


def main():
    ap = argparse.ArgumentParser(description='Closing odds drift model (B6)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('demo')
    ext = sub.add_parser('extract', help='odds_base_*.csv から drift 抽出')
    ext.add_argument('date', help='YYYYMMDD')
    args = ap.parse_args()
    if args.cmd == 'demo':
        return cmd_demo(args)
    elif args.cmd == 'extract':
        return cmd_extract(args)
    return 1


if __name__ == '__main__':
    sys.exit(main())
