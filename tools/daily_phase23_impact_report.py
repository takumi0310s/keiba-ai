#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Daily Phase 23 impact report: 当日 daily_predictions vs Phase 23 shadow run の比較.

朝 (daily_predict 後) または夜 (daily_results 後) に 1 コマンドで Phase 23 適用なら
どう変わるかを report 化。 5/17 開催前 dry-run + 5/17 開催後 verdict 両方に使える。

【入力】 data/daily_predictions/{date}.csv (現行 V15 出力)
        data/daily_results/{date}.csv (あれば、 実 outcome 比較)

【出力】 data/v18/phase23_daily_impact_{date}.md (人間可読 report)

Usage:
    python tools/daily_phase23_impact_report.py 20260510
    python tools/daily_phase23_impact_report.py 20260517  # 当日 (results 無 → projection のみ)
"""
import argparse
import csv
import os
import sys
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 推定 odds (race condition 別)
COND_AVG_ODDS = {
    'A': 3.55, 'B': 3.47, 'C': 6.23, 'D': 3.61, 'E': 1.96, 'X': 7.01,
    '6_06_特別': 4.00,  # 戦略⑦ 除外対象
    'NAR_X': 4.0,
}


def load_predictions(date):
    path = os.path.join(BASE_DIR, 'data', 'daily_predictions', f'{date}.csv')
    if not os.path.exists(path):
        return None
    preds = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            preds.append(row)
    return preds


def load_results(date):
    path = os.path.join(BASE_DIR, 'data', 'daily_results', f'{date}.csv')
    if not os.path.exists(path):
        return None
    results = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            results[row['race_id']] = row
    return results


def kelly_simple(p, odds, bankroll=30000, fraction=0.25, cap_pct=0.05):
    """Kelly bet (簡易、 max 5% cap)."""
    if odds <= 1 or p <= 0 or p >= 1:
        return 0
    b = odds - 1
    f = (b * p - (1 - p)) / b
    if f <= 0:
        return 0
    bet = bankroll * f * fraction
    cap = bankroll * cap_pct
    return int(min(bet, cap) // 100 * 100)


def main():
    ap = argparse.ArgumentParser(description='Daily Phase 23 impact report')
    ap.add_argument('date')
    ap.add_argument('--bankroll', type=float, default=30000)
    args = ap.parse_args()

    preds = load_predictions(args.date)
    if not preds:
        print(f'[ERROR] no daily_predictions for {args.date}')
        return 1

    results = load_results(args.date)
    has_results = results is not None
    print(f'[INFO] {len(preds)} races loaded, results: {"available" if has_results else "N/A"}')

    # filter strategy 7: 除外
    excluded_races = {'06_特別', '京都', 'E', 'B', 'NAR_X'}
    filtered = []
    skipped = []
    for p in preds:
        cond = p.get('condition', '')
        course = p.get('course', '')
        if cond in excluded_races or course in excluded_races:
            skipped.append(p)
        else:
            filtered.append(p)

    print(f'[INFO] 戦略⑦ 適用: {len(filtered)} target, {len(skipped)} excluded')

    # V15 + Phase 23 比較
    v15_total_bet = 0
    v15_total_pnl = 0
    shadow_total_bet = 0
    shadow_total_pnl = 0
    rows_report = []

    for p in filtered:
        race_id = p['race_id']
        cond = p.get('condition', '?')
        course = p.get('course', '?')
        try:
            top1_score = float(p.get('top1_score') or 0)
        except Exception:
            top1_score = 0
        est_odds = COND_AVG_ODDS.get(cond, 4.0)

        # V15 strategy: 700 円固定、 EV+ なら go (条件E は umaren、 他は trio)
        v15_bet = 700
        v15_ev = top1_score * est_odds - 1
        v15_total_bet += v15_bet

        # Shadow: Kelly
        shadow_bet = kelly_simple(top1_score, est_odds, bankroll=args.bankroll)
        shadow_total_bet += shadow_bet

        # outcome
        v15_pnl = 0
        shadow_pnl = 0
        outcome = 'pending'
        if has_results and race_id in results:
            r = results[race_id]
            hit = int(float(r.get('trio_hit', 0) or 0))
            payout = float(r.get('trio_payout', 0) or 0)
            outcome = 'HIT' if hit else 'MISS'
            v15_pnl = (payout - v15_bet) if hit else -v15_bet
            shadow_pnl = (est_odds * shadow_bet - shadow_bet) if hit and shadow_bet > 0 else -shadow_bet
        v15_total_pnl += v15_pnl
        shadow_total_pnl += shadow_pnl

        rows_report.append({
            'race_id': race_id,
            'cond': cond,
            'course': course,
            'top1_score': top1_score,
            'est_odds': est_odds,
            'ev': v15_ev,
            'v15_bet': v15_bet,
            'shadow_bet': shadow_bet,
            'outcome': outcome,
            'v15_pnl': v15_pnl,
            'shadow_pnl': shadow_pnl,
        })

    # md report
    md_path = os.path.join(BASE_DIR, 'data', 'v18',
                            f'phase23_daily_impact_{args.date}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f'# Phase 23 daily impact report: {args.date}\n\n')
        f.write(f'- 全 races: {len(preds)}, 戦略⑦ 適用後: {len(filtered)}\n')
        f.write(f'- results: {"available" if has_results else "未確定 (projection)"}\n')
        f.write(f'- bankroll: ¥{args.bankroll:,.0f}\n\n')

        f.write('## V15 vs Phase 23 Shadow 比較\n\n')
        f.write('| 項目 | V15 production | Phase 23 Shadow | 差分 |\n')
        f.write('|------|---------------|----------------|------|\n')
        f.write(f'| 投資合計 | ¥{v15_total_bet:,} | ¥{shadow_total_bet:,} | {shadow_total_bet - v15_total_bet:+,} |\n')
        if has_results:
            f.write(f'| PnL | ¥{v15_total_pnl:+,} | ¥{shadow_total_pnl:+,} | {shadow_total_pnl - v15_total_pnl:+,} |\n')
            v15_roi = (v15_total_pnl / max(1, v15_total_bet) + 1) * 100
            shadow_roi = (shadow_total_pnl / max(1, shadow_total_bet) + 1) * 100
            f.write(f'| ROI | {v15_roi:.1f}% | {shadow_roi:.1f}% | {shadow_roi - v15_roi:+.1f}% |\n')

        f.write('\n## per-race 詳細\n\n')
        f.write('| race_id | cond | top1_score | EV | V15 bet | Shadow bet | outcome | V15 PnL | Shadow PnL |\n')
        f.write('|---------|------|-----------|-----|---------|------------|---------|---------|------------|\n')
        for r in rows_report[:30]:  # top 30
            f.write(f'| {r["race_id"]} | {r["cond"]} | {r["top1_score"]:.3f} | '
                     f'{r["ev"]:+.2f} | ¥{r["v15_bet"]} | ¥{r["shadow_bet"]} | '
                     f'{r["outcome"]} | {r["v15_pnl"]:+,} | {r["shadow_pnl"]:+,} |\n')

        f.write(f'\n## 戦略⑦ 除外 races ({len(skipped)})\n')
        for s in skipped[:20]:
            f.write(f'- {s["race_id"]} ({s.get("course", "?")}) cond={s.get("condition", "?")}\n')

    print(f'[OK] saved: {md_path}')
    print(f'\n=== Summary ({args.date}) ===')
    print(f'V15 production:   bet=¥{v15_total_bet:,}, PnL={v15_total_pnl:+,}, ROI={(v15_total_pnl/max(1,v15_total_bet)+1)*100:.1f}%')
    print(f'Phase 23 Shadow:  bet=¥{shadow_total_bet:,}, PnL={shadow_total_pnl:+,}, ROI={(shadow_total_pnl/max(1,shadow_total_bet)+1)*100:.1f}%')
    print(f'差分: {shadow_total_pnl - v15_total_pnl:+,} 円')
    return 0


if __name__ == '__main__':
    sys.exit(main())
