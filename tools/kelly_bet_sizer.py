#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B1: Kelly criterion bet sizing (fractional Kelly 推奨).

V15 現状: 700 円/race 固定。 Kelly criterion で EV と勝率に応じて最適 bet size 配分。
fractional Kelly (0.25x) で分散抑制、 撤退 line +63K 保護維持。

【Kelly formula】
    f* = (b * p - q) / b
    b = odds - 1 (オッズ - 1)
    p = 勝率 (calibrated 推奨)
    q = 1 - p
    bet = bankroll * f*  (フル Kelly = 数値的最大、 分散大)
    bet = bankroll * 0.25 * f*  (1/4 Kelly = 実用 推奨、 分散 1/16)

【V15 投資保護】 daily_predict.py / app.py 一切触らず、 helper として呼出可能。

Usage:
    # 単一 bet 計算
    python tools/kelly_bet_sizer.py --p 0.65 --odds 3.5 --bankroll 30000 --fraction 0.25

    # CSV 一括 (pred,odds 列必須、 既存 daily_predictions/*.csv 想定)
    python tools/kelly_bet_sizer.py --csv data/daily_predictions/20260510.csv --bankroll 30000 --fraction 0.25 --out data/kelly_sizes.csv

    # demo (V15 戦略⑦ 想定 シミュレーション)
    python tools/kelly_bet_sizer.py demo
"""
import argparse
import os
import sys
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def kelly_fraction(p, odds):
    """Kelly fraction f* を計算. p=勝率, odds=配当倍率 (3.5 等)."""
    if odds <= 1.0 or p <= 0 or p >= 1:
        return 0.0
    b = odds - 1
    q = 1 - p
    f = (b * p - q) / b
    return max(0.0, f)  # 負 = bet しない


def kelly_bet(bankroll, p, odds, fraction=0.25, min_bet=100, max_bet_pct=0.05):
    """Bankroll に対する Kelly bet 額 (整数)."""
    f = kelly_fraction(p, odds) * fraction
    bet = bankroll * f
    bet = min(bet, bankroll * max_bet_pct)  # 単一 bet 上限 5%
    bet = int(round(bet / 100) * 100)        # 100円単位
    if bet < min_bet:
        return 0
    return bet


def cmd_single(args):
    p, odds, bankroll, fraction = args.p, args.odds, args.bankroll, args.fraction
    f_full = kelly_fraction(p, odds)
    f_used = f_full * fraction
    bet = kelly_bet(bankroll, p, odds, fraction=fraction)
    ev = p * odds - 1
    print(f'[INPUT] p={p}, odds={odds}, bankroll={bankroll}, fraction={fraction}x Kelly')
    print(f'[EV] {ev:+.3f} ({"plus" if ev > 0 else "MINUS - BET 0"})')
    print(f'[KELLY] full f*={f_full:.4f}, used f={f_used:.4f}')
    print(f'[BET ] {bet} 円  ({bet/bankroll*100:.2f}% of bankroll)')
    return 0


def cmd_csv(args):
    import pandas as pd
    df = pd.read_csv(args.csv)
    needed = {'pred', 'odds'}
    if not needed.issubset(df.columns):
        # try alternate column names
        if 'top1_score' in df.columns and 'top1_odds' in df.columns:
            df = df.rename(columns={'top1_score': 'pred', 'top1_odds': 'odds'})
        else:
            print(f'[ERROR] CSV must have columns: pred, odds (found: {list(df.columns)[:10]})')
            return 1

    df['kelly_full'] = df.apply(lambda r: kelly_fraction(r['pred'], r['odds']), axis=1)
    df['kelly_bet'] = df.apply(
        lambda r: kelly_bet(args.bankroll, r['pred'], r['odds'], fraction=args.fraction),
        axis=1,
    )
    df['ev'] = df['pred'] * df['odds'] - 1
    df.to_csv(args.out, index=False)
    n_bet = (df['kelly_bet'] > 0).sum()
    total_bet = df['kelly_bet'].sum()
    print(f'[OK] {len(df)} races, {n_bet} bets, total_bet={total_bet} 円')
    print(f'[OK] saved: {args.out}')
    return 0


def cmd_demo(args):
    """V15 戦略⑦ 想定 シミュレーション (条件別)."""
    print('=== V15 戦略⑦ Kelly demo (bankroll=30,000 円, fractional 0.25x) ===\n')
    scenarios = [
        ('A 8-14頭/1600m+/良〜稍重 (trio)',   0.65, 3.55),
        ('B 8-14頭/1600m+/重〜不良 (trio)',   0.61, 3.47),
        ('C 15頭+/1600m+/良〜稍重 (trio)',    0.52, 6.23),
        ('D 1200-1400m (trio)',               0.48, 3.61),
        ('E 7頭以下 (umaren)',                0.73, 1.96),
        ('X 15頭+/重〜不良 (trio)',           0.54, 7.01),
        ('合算 (trio 全体)',                  0.55, 4.28),
        ('低 EV 例 (capped)',                 0.40, 2.00),
        ('マイナス EV 例 (skip)',             0.25, 2.50),
    ]
    bankroll = 30000
    for name, p, odds in scenarios:
        bet = kelly_bet(bankroll, p, odds, fraction=0.25)
        f_full = kelly_fraction(p, odds)
        ev = p * odds - 1
        marker = '★' if bet > 0 else '✗'
        print(f'  {marker} {name:<40} p={p:.2f} odds={odds:.2f} EV={ev:+.2f}  '
              f'f*={f_full:.3f}  bet={bet:>5} 円')
    print('\n比較: 現行 V15 = 700 円 (条件E 700/umaren 2 点 = 350円+350円)')
    print('Kelly 0.25x = EV 高 race は ~1500-3000 円、 低 EV は 0 円 → ROI 平均 + 分散 ↓')
    return 0


def main():
    ap = argparse.ArgumentParser(description='Kelly criterion bet sizer (B1)')
    sub = ap.add_subparsers(dest='cmd')

    single_p = sub.add_parser('single', help='単一 bet 計算')
    single_p.add_argument('--p', type=float, required=True, help='勝率 0-1 (calibrated 推奨)')
    single_p.add_argument('--odds', type=float, required=True, help='配当倍率')
    single_p.add_argument('--bankroll', type=float, default=30000)
    single_p.add_argument('--fraction', type=float, default=0.25, help='Kelly fraction (0.25 = 1/4 Kelly)')

    csv_p = sub.add_parser('csv', help='CSV 一括処理')
    csv_p.add_argument('--csv', required=True)
    csv_p.add_argument('--bankroll', type=float, default=30000)
    csv_p.add_argument('--fraction', type=float, default=0.25)
    csv_p.add_argument('--out', required=True)

    sub.add_parser('demo')

    # for backward compat: --p/--odds 直接指定でも動く
    ap.add_argument('--p', type=float, help='shortcut single (alias)')
    ap.add_argument('--odds', type=float)
    ap.add_argument('--bankroll', type=float, default=30000)
    ap.add_argument('--fraction', type=float, default=0.25)

    args = ap.parse_args()

    if args.cmd == 'single' or (args.p is not None and args.odds is not None):
        if args.cmd != 'single':
            # shortcut mode
            args.cmd = 'single'
        return cmd_single(args)
    elif args.cmd == 'csv':
        return cmd_csv(args)
    elif args.cmd == 'demo':
        return cmd_demo(args)
    else:
        ap.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
