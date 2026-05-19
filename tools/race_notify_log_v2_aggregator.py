"""race_notify_log v2 (3 phase) aggregator

phase 1+2+3 揃った race を 集計、 真の race-time formation ROI 計算。
8 strategy paper ROI を並行追跡。

★ daily 20:30 fire 想定 (DailyResultsEvening 20:00 後) ★

Usage:
  python tools/race_notify_log_v2_aggregator.py
  python tools/race_notify_log_v2_aggregator.py --date 20260518
  python tools/race_notify_log_v2_aggregator.py --range 20260518:20260524
  python tools/race_notify_log_v2_aggregator.py --all
  python tools/race_notify_log_v2_aggregator.py --strategy-report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Windows cp932 console で 絵文字 / yen sign 印字 → utf-8 化 (schtask redirect 対策)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'tools'))

from race_notify_log_v2 import _log_root, STRATEGY_KEYS  # noqa: E402

LOG_DIR = _log_root()
OUT_DIR = BASE_DIR / 'data' / 'race_notify_log_v2_summary'

# Investment per bet (default; matches CONDITION_PROFILES standard)
DEFAULT_TRIO_INV = 700
DEFAULT_UMAREN_INV = 700  # 400+300


def _read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _empty_strategy_stats():
    """strategy 別集計 dict の初期値を返す。"""
    return {k: {'n': 0, 'hits': 0, 'inv': 0, 'pay': 0} for k in STRATEGY_KEYS}


def _finalize_strategy_stats(stats: dict) -> dict:
    """strategy 別集計に ROI / hit_rate を追加。"""
    out = {}
    for k, v in stats.items():
        roi = round(v['pay'] / v['inv'] * 100, 2) if v['inv'] > 0 else 0.0
        hr = round(v['hits'] / v['n'] * 100, 2) if v['n'] > 0 else 0.0
        out[k] = {
            **v,
            'roi_pct': roi,
            'hit_rate_pct': hr,
            'pnl': v['pay'] - v['inv'],
        }
    return out


def aggregate_day(date_str: str) -> dict | None:
    """1 日分の 3 phase log を集計。8 strategy paper ROI を含む。"""
    p1_dir = LOG_DIR / date_str / 'phase1'
    p2_dir = LOG_DIR / date_str / 'phase2'
    p3_dir = LOG_DIR / date_str / 'phase3'

    if not p2_dir.exists():
        return {
            'date': date_str,
            'phase1_count': len(list(p1_dir.glob('*.json'))) if p1_dir.exists() else 0,
            'phase2_count': 0,
            'phase3_count': len(list(p3_dir.glob('*.json'))) if p3_dir.exists() else 0,
            'complete_races': 0,
            'voted_races': 0,
            'skipped_races': 0,
            'hits': 0,
            'inv': 0,
            'pay': 0,
            'roi_pct': 0.0,
            'pnl': 0,
            'strategy_stats': _finalize_strategy_stats(_empty_strategy_stats()),
        }

    p1_ids = {f.stem for f in p1_dir.glob('*.json')} if p1_dir.exists() else set()
    p2_ids = {f.stem for f in p2_dir.glob('*.json')} if p2_dir.exists() else set()
    p3_ids = {f.stem for f in p3_dir.glob('*.json')} if p3_dir.exists() else set()

    # complete = phase2 + phase3 (phase1 は morning predict、 phase2 が voting baseline)
    complete = p2_ids & p3_ids

    total_inv = 0
    total_pay = 0
    hits = 0
    voted_races = 0
    skipped_races = 0
    cond_breakdown = {}
    strategy_stats = _empty_strategy_stats()

    for race_id in sorted(complete):
        p2 = _read_json(p2_dir / f'{race_id}.json')
        p3 = _read_json(p3_dir / f'{race_id}.json')
        if not p2 or not p3:
            continue

        if p2.get('strategy_7c_skip') or p2.get('channel') in ('skip', 'error'):
            skipped_races += 1
            # strategy paper も skip race としてカウント (inv/pay 0)
            # → strategy_stats は更新しない (voted base の ROI を追跡)
            continue

        bet_type = p2.get('bet_type') or 'trio'
        cond_key = p2.get('cond_key') or 'unknown'

        inv = DEFAULT_UMAREN_INV if bet_type == 'umaren' else DEFAULT_TRIO_INV
        total_inv += inv
        voted_races += 1

        hit_miss = p3.get('hit_miss', {}) or {}
        real_payouts = p3.get('real_payouts', {}) or {}

        pay = 0
        hit_key = 'umaren_hit' if bet_type == 'umaren' else 'trio_hit'
        pay_key = 'umaren' if bet_type == 'umaren' else 'trio'
        if hit_miss.get(hit_key):
            pay = int(real_payouts.get(pay_key, 0) or 0)
            hits += 1
        total_pay += pay

        # 条件別 breakdown
        cb = cond_breakdown.setdefault(cond_key, {
            'n': 0, 'hits': 0, 'inv': 0, 'pay': 0,
        })
        cb['n'] += 1
        cb['inv'] += inv
        cb['pay'] += pay
        if hit_miss.get(hit_key):
            cb['hits'] += 1

        # 8 strategy paper stats (phase3 の strategy_results を利用)
        sr = p3.get('strategy_results', {})
        for k in STRATEGY_KEYS:
            kdata = sr.get(k, {})
            if not kdata or kdata.get('skipped'):
                continue
            ss = strategy_stats[k]
            ss['n'] += 1
            ss['inv'] += int(kdata.get('investment', 0) or 0)
            ss['pay'] += int(kdata.get('payout', 0) or 0)
            if kdata.get('hit'):
                ss['hits'] += 1

    # 条件別 roi 計算
    for k, v in cond_breakdown.items():
        v['roi_pct'] = round(v['pay'] / v['inv'] * 100, 2) if v['inv'] > 0 else 0.0
        v['hit_rate_pct'] = round(v['hits'] / v['n'] * 100, 2) if v['n'] > 0 else 0.0

    return {
        'date': date_str,
        'phase1_count': len(p1_ids),
        'phase2_count': len(p2_ids),
        'phase3_count': len(p3_ids),
        'complete_races': len(complete),
        'voted_races': voted_races,
        'skipped_races': skipped_races,
        'hits': hits,
        'hit_rate_pct': round(hits / voted_races * 100, 2) if voted_races > 0 else 0.0,
        'inv': total_inv,
        'pay': total_pay,
        'roi_pct': round(total_pay / total_inv * 100, 2) if total_inv > 0 else 0.0,
        'pnl': total_pay - total_inv,
        'cond_breakdown': cond_breakdown,
        'strategy_stats': _finalize_strategy_stats(strategy_stats),
    }


def aggregate_range(start: str, end: str) -> list[dict]:
    """日付範囲集計 (YYYYMMDD:YYYYMMDD)。"""
    s = datetime.strptime(start, '%Y%m%d').date()
    e = datetime.strptime(end, '%Y%m%d').date()
    if s > e:
        s, e = e, s
    results = []
    cur = s
    while cur <= e:
        d = cur.strftime('%Y%m%d')
        if (LOG_DIR / d).exists():
            results.append(aggregate_day(d))
        cur += timedelta(days=1)
    return results


def aggregate_all() -> list[dict]:
    """log_dir 内 全日付集計。"""
    if not LOG_DIR.exists():
        return []
    results = []
    for d in sorted(LOG_DIR.iterdir()):
        if d.is_dir() and len(d.name) == 8 and d.name.isdigit():
            results.append(aggregate_day(d.name))
    return results


def _merge_strategy_stats(daily_list: list[dict]) -> dict:
    """複数日の strategy_stats を合算して返す。"""
    merged = {k: {'n': 0, 'hits': 0, 'inv': 0, 'pay': 0} for k in STRATEGY_KEYS}
    for s in daily_list:
        ss = s.get('strategy_stats', {})
        for k in STRATEGY_KEYS:
            kdata = ss.get(k, {})
            merged[k]['n'] += int(kdata.get('n', 0) or 0)
            merged[k]['hits'] += int(kdata.get('hits', 0) or 0)
            merged[k]['inv'] += int(kdata.get('inv', 0) or 0)
            merged[k]['pay'] += int(kdata.get('pay', 0) or 0)
    out = {}
    for k, v in merged.items():
        roi = round(v['pay'] / v['inv'] * 100, 2) if v['inv'] > 0 else 0.0
        hr = round(v['hits'] / v['n'] * 100, 2) if v['n'] > 0 else 0.0
        out[k] = {**v, 'roi_pct': roi, 'hit_rate_pct': hr, 'pnl': v['pay'] - v['inv']}
    return out


def write_summary(summary, out_path=None) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = OUT_DIR / f'summary_{ts}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return out_path


def print_strategy_report(strategy_stats: dict, header: str = '') -> None:
    """8 strategy 集計を表示。"""
    if header:
        print(f"\n  {header}")
    print(f"  {'strategy':<16} {'N':>5} {'hits':>5} {'hit%':>7} {'inv':>8} {'pay':>8} {'ROI%':>8} {'PnL':>8}")
    print(f"  {'-'*16} {'-'*5} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for k in STRATEGY_KEYS:
        v = strategy_stats.get(k, {})
        n = v.get('n', 0)
        hits = v.get('hits', 0)
        hr = v.get('hit_rate_pct', 0.0)
        inv = v.get('inv', 0)
        pay = v.get('pay', 0)
        roi = v.get('roi_pct', 0.0)
        pnl = v.get('pnl', 0)
        print(f"  {k:<16} {n:>5} {hits:>5} {hr:>6.1f}% {inv:>8} {pay:>8} {roi:>7.2f}% {pnl:>+8}")


def print_summary(summary, show_strategy: bool = True) -> None:
    if isinstance(summary, list):
        total_inv = sum(s.get('inv', 0) for s in summary)
        total_pay = sum(s.get('pay', 0) for s in summary)
        total_voted = sum(s.get('voted_races', 0) for s in summary)
        total_hits = sum(s.get('hits', 0) for s in summary)
        print(f"\n=== race_notify_log v2 aggregate ({len(summary)} days) ===")
        for s in summary:
            print(f"  {s['date']}: voted={s['voted_races']:3d} hits={s['hits']:3d} "
                  f"inv={s['inv']:>6}JPY pay={s['pay']:>6}JPY ROI={s['roi_pct']:>6.2f}% PnL={s['pnl']:+}JPY")
        roi = total_pay / total_inv * 100 if total_inv > 0 else 0.0
        print(f"\n  TOTAL: voted={total_voted} hits={total_hits} "
              f"inv={total_inv}JPY pay={total_pay}JPY ROI={roi:.2f}% PnL={total_pay - total_inv:+}JPY")
        if show_strategy:
            merged = _merge_strategy_stats(summary)
            print_strategy_report(merged, header='8 strategy paper ROI (cumulative)')
    else:
        s = summary
        print(f"\n=== {s['date']} ===")
        print(f"  phase1 / phase2 / phase3 = {s['phase1_count']} / {s['phase2_count']} / {s['phase3_count']}")
        print(f"  complete races = {s['complete_races']} (voted {s['voted_races']}, skipped {s['skipped_races']})")
        print(f"  hits = {s['hits']} ({s.get('hit_rate_pct', 0)}%)")
        print(f"  inv = {s['inv']}JPY  pay = {s['pay']}JPY  ROI = {s['roi_pct']}%  PnL = {s['pnl']:+}JPY")
        if s.get('cond_breakdown'):
            print(f"  cond breakdown:")
            for k, v in sorted(s['cond_breakdown'].items()):
                print(f"    {k}: n={v['n']} hits={v['hits']} inv={v['inv']}JPY pay={v['pay']}JPY "
                      f"ROI={v.get('roi_pct', 0)}% hit_rate={v.get('hit_rate_pct', 0)}%")
        if show_strategy and s.get('strategy_stats'):
            print_strategy_report(s['strategy_stats'], header='8 strategy paper ROI')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='YYYYMMDD')
    ap.add_argument('--range', help='YYYYMMDD:YYYYMMDD')
    ap.add_argument('--all', action='store_true', help='全日付集計')
    ap.add_argument('--out', help='出力 JSON path')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--no-strategy', action='store_true', help='8 strategy 表示を省略')
    ap.add_argument('--strategy-report', action='store_true', help='8 strategy paper ROI のみ表示')
    args = ap.parse_args()

    if args.all:
        result = aggregate_all()
    elif args.range:
        start, end = args.range.split(':')
        result = aggregate_range(start, end)
    elif args.date:
        result = aggregate_day(args.date)
    else:
        result = aggregate_day(datetime.now().strftime('%Y%m%d'))

    out = write_summary(result, args.out)
    if not args.quiet:
        show_strategy = not args.no_strategy
        if args.strategy_report:
            # strategy report のみ
            if isinstance(result, list):
                merged = _merge_strategy_stats(result)
                print_strategy_report(merged, header='8 strategy paper ROI (all days)')
            else:
                ss = result.get('strategy_stats', {})
                print_strategy_report(ss, header=f"8 strategy paper ROI ({result.get('date', '')})")
        else:
            print_summary(result, show_strategy=show_strategy)
        print(f"\n  summary written to: {out}")


if __name__ == '__main__':
    main()
