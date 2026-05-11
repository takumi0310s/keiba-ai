#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 23 tool group shadow mode runner.

V15 production を **一切 変更せず**、 race_auto_notify / daily_predict の output を
読み取って Phase 23 tool 一式の "もしも" 結果を log する。

【V15 投資保護】 production 通知 / 投票には 1 mm も触らない。 shadow log のみ。

【入力】 race_auto_notify が出力する race 予測結果 JSON / CSV (推定 format)、 or
manual: race_id + horse list + pred + odds 指定。

【出力】 data/shadow_log/{date}/{race_id}_phase23.json
- Kelly bet size (vs 700 円 固定)
- Pari-mutuel optimal trio (vs V15 7 点固定)
- Calibration adjusted prob (calibrator あれば)
- Drawdown breaker status

【使い方】
    # 自動 hook (5/13+ で race_auto_notify から呼出可能)
    python tools/phase23_shadow_runner.py --auto --date 20260510

    # 手動 1 race
    python tools/phase23_shadow_runner.py --race-id 202608030611 --probs "0.65,0.13,0.10" \
        --odds "3.5,5.2,8.1"

    # cumulative_results.csv 読込で 過去 全 race shadow 評価
    python tools/phase23_shadow_runner.py --backtest --from 20260301
"""
import argparse
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
SHADOW_LOG_BASE = os.path.join(BASE_DIR, 'data', 'shadow_log')
CALIBRATOR_PATH = os.path.join(BASE_DIR, 'data', 'calibrator_v15.pkl')


# 各 Phase 23 tool の関数を import (relative 用)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))


def apply_calibration_safe(pred):
    """calibrator file あれば適用、 無ければ raw 返す."""
    if not os.path.exists(CALIBRATOR_PATH):
        return pred, 'no_calibrator'
    try:
        import pickle
        cal = pickle.load(open(CALIBRATOR_PATH, 'rb'))
        from calibrate_confidence import apply_calibrator
        cal_pred = float(apply_calibrator(cal, [pred], 'isotonic')[0])
        return cal_pred, 'isotonic'
    except Exception as e:
        return pred, f'error: {e}'


def shadow_one_race(race_id, probs, odds, bankroll=30000, kelly_fraction=0.25):
    """1 race の Phase 23 shadow analysis."""
    from kelly_bet_sizer import kelly_bet, kelly_fraction as kf_calc
    from exotic_optimizer import select_optimal_trio, trio_prob, estimate_trio_odds, normalize_probs
    from drawdown_circuit_breaker import check_status

    probs = list(probs)
    odds = list(odds)
    n = len(probs)

    # 1. Kelly per-horse bet (top horse only, simple)
    top_idx = 0
    p_top = probs[0]
    cal_pred, cal_method = apply_calibration_safe(p_top)
    o_top = odds[0] if odds and odds[0] > 0 else 0
    k_bet = kelly_bet(bankroll, cal_pred, o_top, fraction=kelly_fraction) if o_top > 0 else 0
    k_full = kf_calc(cal_pred, o_top) if o_top > 0 else 0

    # 2. Pari-mutuel optimal trio (top 2 axis)
    optimal = select_optimal_trio(probs, n, top_k_axis=2, max_points=7, min_ev=-0.5)

    # 3. Drawdown circuit breaker
    breaker = check_status()

    # 4. shadow vs current 比較
    # Current strategy: 700 円固定、 EV+ なら go
    ev = cal_pred * o_top - 1 if o_top > 0 else 0
    current_bet = 700 if ev > 0 else 700  # 戦略⑦ で除外されてない限り 700
    shadow_diff = k_bet - current_bet

    return {
        'race_id': race_id,
        'top_horse_pred': p_top,
        'top_horse_calibrated_pred': cal_pred,
        'cal_method': cal_method,
        'top_horse_odds': o_top,
        'top_horse_ev': ev,
        'kelly_bet_full_pct': k_full,
        'kelly_bet_size': k_bet,
        'current_bet_size': current_bet,
        'shadow_bet_diff': shadow_diff,
        'optimal_trio_points': [{'triple': c['triple'], 'prob': c['prob'],
                                   'odds_est': c['odds_est'], 'ev': c['ev']}
                                 for c in optimal[:7]],
        'breaker_status': breaker.get('status'),
        'breaker_pnl': breaker.get('cumulative_pnl'),
        'shadow_at': datetime.now().isoformat(),
    }


def cmd_manual(args):
    probs = [float(x) for x in args.probs.split(',')]
    odds = [float(x) for x in args.odds.split(',')]
    result = shadow_one_race(args.race_id, probs, odds,
                              bankroll=args.bankroll, kelly_fraction=args.fraction)

    today = datetime.now().strftime('%Y%m%d')
    out_dir = os.path.join(SHADOW_LOG_BASE, today)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.race_id}_phase23.json')
    json.dump(result, open(out_path, 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print(f'[OK] shadow log: {out_path}')
    print(f'  pred (raw):    {result["top_horse_pred"]:.3f}')
    print(f'  pred (cal):    {result["top_horse_calibrated_pred"]:.3f}  [{result["cal_method"]}]')
    print(f'  odds:          {result["top_horse_odds"]:.2f}')
    print(f'  EV:            {result["top_horse_ev"]:+.3f}')
    print(f'  Kelly bet:     {result["kelly_bet_size"]} 円 (vs current 700 円, diff {result["shadow_bet_diff"]:+})')
    print(f'  Breaker:       {result["breaker_status"]}')
    print(f'  Top 7 optimal trio (EV ranked):')
    for c in result['optimal_trio_points']:
        print(f'    {c["triple"]}  prob={c["prob"]:.4f}  odds={c["odds_est"]:>5.2f}  EV={c["ev"]:+.3f}')
    return 0


def cmd_backtest(args):
    """cumulative_results.csv 全 race を shadow re-evaluate."""
    import pandas as pd
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'cumulative_results.csv'),
                      encoding='utf-8-sig')
    df = df[df['status'] == 'settled'].copy()
    df['top1_score'] = pd.to_numeric(df['top1_score'], errors='coerce')
    df = df.dropna(subset=['top1_score'])

    if args.from_date:
        df['date'] = df['date'].astype(str)
        df = df[df['date'] >= args.from_date]
    print(f'[INFO] {len(df)} races with valid pred')

    total_v15_pnl = 0
    total_shadow_pnl = 0
    results = []
    for _, row in df.iterrows():
        race_id = row['race_id']
        pred = float(row['top1_score'])
        # 過去レコードに実 odds は無い → trio_payout から逆算 (簡易)
        payout = float(row.get('trio_payout', 0) or 0)
        hit = int(row.get('trio_hit', 0) or 0)
        # 簡易 odds estimate: payout / 100 (trio 100円 → payout 円)
        est_odds = payout / 100.0 if payout > 0 else 5.0

        v15_bet = 700
        v15_pnl = (payout - v15_bet) if hit else -v15_bet
        cal_pred, _ = apply_calibration_safe(pred)
        from kelly_bet_sizer import kelly_bet
        k_bet = kelly_bet(args.bankroll, cal_pred, est_odds, fraction=args.fraction)
        shadow_pnl = (est_odds * k_bet - k_bet) if hit and k_bet > 0 else (-k_bet)

        total_v15_pnl += v15_pnl
        total_shadow_pnl += shadow_pnl
        results.append({
            'race_id': race_id,
            'pred': pred, 'cal_pred': cal_pred, 'est_odds': est_odds,
            'v15_bet': v15_bet, 'v15_pnl': v15_pnl,
            'shadow_bet': k_bet, 'shadow_pnl': shadow_pnl, 'hit': hit,
        })

    print(f'\n[BACKTEST RESULT] {len(results)} races')
    print(f'  V15 (700 円固定) PnL:   {total_v15_pnl:+,.0f}')
    print(f'  Shadow (Kelly+Cal) PnL: {total_shadow_pnl:+,.0f}')
    print(f'  差分: {total_shadow_pnl - total_v15_pnl:+,.0f}')
    if results:
        v15_inv = len(results) * 700
        v15_roi = (total_v15_pnl / v15_inv + 1) * 100
        shadow_inv = sum(r['shadow_bet'] for r in results)
        shadow_roi = (total_shadow_pnl / max(1, shadow_inv) + 1) * 100 if shadow_inv > 0 else 0
        print(f'  V15 ROI:    {v15_roi:.1f}% (inv {v15_inv:,})')
        print(f'  Shadow ROI: {shadow_roi:.1f}% (inv {shadow_inv:,})')

    out_dir = os.path.join(SHADOW_LOG_BASE, 'backtest')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'shadow_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    json.dump({'summary': {'v15_pnl': total_v15_pnl, 'shadow_pnl': total_shadow_pnl,
                            'n_races': len(results)},
                'results': results}, open(out_path, 'w', encoding='utf-8'),
                indent=2, ensure_ascii=False)
    print(f'\n[OK] saved: {out_path}')
    return 0


def main():
    ap = argparse.ArgumentParser(description='Phase 23 shadow mode runner')
    ap.add_argument('--race-id', help='manual race')
    ap.add_argument('--probs', help='comma-separated 馬別 pred (top horse first)')
    ap.add_argument('--odds', help='comma-separated 馬別 odds')
    ap.add_argument('--bankroll', type=float, default=30000)
    ap.add_argument('--fraction', type=float, default=0.25)
    ap.add_argument('--backtest', action='store_true', help='全 cumulative_results.csv re-evaluate')
    ap.add_argument('--from', dest='from_date', default=None)
    args = ap.parse_args()

    if args.backtest:
        return cmd_backtest(args)
    elif args.race_id and args.probs and args.odds:
        return cmd_manual(args)
    else:
        ap.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
