#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 23 shadow log aggregator: 過去 shadow run の集計 + V15 比較.

5/12 から 5/17 までの shadow log を集計し、 V15 production vs Phase 23 shadow の
パフォーマンス差を analyze。 5/24 V20 投入判定 の根拠 data。

【入力】 data/shadow_log/{date}/*.json (shadow runner output)
【出力】 data/shadow_log/aggregate_{from}_{to}.json + .md (人間可読)

Usage:
    python tools/shadow_log_aggregator.py
    python tools/shadow_log_aggregator.py --from 20260512 --to 20260517
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADOW_DIR = os.path.join(BASE_DIR, 'data', 'shadow_log')


def gather_logs(date_from, date_to):
    """日付範囲の shadow log を gather."""
    all_logs = []
    pattern = os.path.join(SHADOW_DIR, '*', '*.json')
    for fp in glob.glob(pattern):
        date_dir = os.path.basename(os.path.dirname(fp))
        if not date_dir.isdigit():
            continue
        if date_from and date_dir < date_from:
            continue
        if date_to and date_dir > date_to:
            continue
        try:
            data = json.load(open(fp, 'r', encoding='utf-8'))
            data['_date'] = date_dir
            data['_file'] = os.path.basename(fp)
            all_logs.append(data)
        except Exception:
            pass
    return all_logs


def aggregate(logs):
    """V15 vs Shadow 比較 集計."""
    if not logs:
        return {'note': 'no logs'}

    n = len(logs)
    by_breaker_status = defaultdict(int)
    kelly_bets = []
    current_bets = []
    cal_methods = defaultdict(int)
    optimal_trio_count = []

    for log in logs:
        by_breaker_status[log.get('breaker_status', 'UNKNOWN')] += 1
        if log.get('kelly_bet_size'):
            kelly_bets.append(log['kelly_bet_size'])
        if log.get('current_bet_size'):
            current_bets.append(log['current_bet_size'])
        cal_methods[log.get('cal_method', 'unknown')] += 1
        optimal_trio_count.append(len(log.get('optimal_trio_points', [])))

    def stats(vals):
        if not vals:
            return None
        return {
            'n': len(vals),
            'sum': sum(vals),
            'mean': round(sum(vals) / len(vals), 2),
            'min': min(vals),
            'max': max(vals),
        }

    return {
        'n_logs': n,
        'breaker_status_dist': dict(by_breaker_status),
        'kelly_bet_stats': stats(kelly_bets),
        'current_bet_stats': stats(current_bets),
        'shadow_diff_total': sum(kelly_bets) - sum(current_bets),
        'cal_methods_used': dict(cal_methods),
        'optimal_trio_avg_count': round(sum(optimal_trio_count) / max(1, len(optimal_trio_count)), 1),
    }


def main():
    ap = argparse.ArgumentParser(description='Shadow log aggregator')
    ap.add_argument('--from', dest='date_from', default=None)
    ap.add_argument('--to', dest='date_to', default=None)
    args = ap.parse_args()

    logs = gather_logs(args.date_from, args.date_to)
    print(f'[INFO] {len(logs)} shadow logs gathered')

    agg = aggregate(logs)

    print('\n=== Shadow log Aggregate ===')
    print(json.dumps(agg, indent=2, ensure_ascii=False, default=str))

    # save
    today = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(SHADOW_DIR, f'aggregate_{today}.json')
    json.dump({'aggregate': agg, 'logs_count': len(logs),
                'generated_at': datetime.now().isoformat(),
                'date_from': args.date_from, 'date_to': args.date_to},
              open(out_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False, default=str)
    print(f'\n[OK] saved: {out_path}')

    # md (人間可読)
    md_path = out_path.replace('.json', '.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f'# Shadow log Aggregate ({args.date_from or "all"} → {args.date_to or "now"})\n\n')
        f.write(f'- 集計対象 logs: {len(logs)} 件\n')
        f.write(f'- 生成日時: {datetime.now().isoformat()}\n\n')
        f.write('## breaker status 分布\n\n')
        for status, count in agg.get('breaker_status_dist', {}).items():
            f.write(f'- {status}: {count} 件\n')
        f.write('\n## Kelly vs current bet 比較\n\n')
        kbs = agg.get('kelly_bet_stats')
        cbs = agg.get('current_bet_stats')
        if kbs and cbs:
            f.write(f'| 項目 | Kelly (shadow) | 現行 V15 (700円固定) |\n')
            f.write(f'|------|---------------|-------------------|\n')
            f.write(f'| n | {kbs["n"]} | {cbs["n"]} |\n')
            f.write(f'| mean 円 | {kbs["mean"]} | {cbs["mean"]} |\n')
            f.write(f'| sum 円 | {kbs["sum"]:,} | {cbs["sum"]:,} |\n')
            f.write(f'| min-max | {kbs["min"]}-{kbs["max"]} | {cbs["min"]}-{cbs["max"]} |\n')
        f.write(f'\n## 差分: {agg.get("shadow_diff_total", 0):+,} 円\n')
    print(f'[OK] saved: {md_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
