#!/usr/bin/env python
"""スクレイピング状態一覧

全CSV年別カバレッジ + 登録プロセスの生死 + 前回実行時からの差分。
一発でスクレイピング体制の全体像を把握する。

Usage:
    python tools/scrape_status.py                 # テーブル表示のみ
    python tools/scrape_status.py --notify        # Discord にも送信
    python tools/scrape_status.py --json          # JSON出力のみ
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
PID_DIR = os.path.join(LOG_DIR, 'pids')
SNAPSHOT = os.path.join(DATA_DIR, '_scrape_status_snapshot.json')

# 監視対象CSVとrace_id列名
TARGET_CSVS = [
    ('netkeiba_upset_level.csv', 'race_id'),
    ('netkeiba_track_bias.csv', 'race_id'),
    ('netkeiba_race_lap.csv', 'race_id'),
    ('netkeiba_race_laps.csv', 'race_id'),
    ('netkeiba_training_eval.csv', 'race_id'),
    ('netkeiba_master_index.csv', 'race_id'),
    ('netkeiba_ai_position.csv', 'race_id'),
    ('netkeiba_ai_opinion.csv', 'race_id'),
    ('netkeiba_ana_best.csv', 'race_id'),
    ('netkeiba_track_index.csv', 'race_id'),
    ('netkeiba_shinba_eval.csv', 'race_id'),
    ('netkeiba_race_review.csv', 'race_id'),
    ('netkeiba_speed_index.csv', 'race_id'),
    ('netkeiba_stable_comments.csv', 'race_id'),
    ('netkeiba_training_times.csv', 'race_id'),
]

YEARS = ['2020', '2021', '2022', '2023', '2024', '2025', '2026']


def scan_csv_coverage():
    """各CSVの年別行数を集計して返す"""
    import pandas as pd
    rows = []
    for fn, col in TARGET_CSVS:
        p = os.path.join(DATA_DIR, fn)
        entry = {'file': fn, 'exists': os.path.exists(p), 'total': 0, 'years': {y: 0 for y in YEARS}}
        if not entry['exists']:
            rows.append(entry)
            continue
        try:
            df = pd.read_csv(p, encoding='utf-8-sig', usecols=[col], dtype=str, low_memory=False)
            entry['total'] = len(df)
            yr = df[col].astype(str).str[:4].value_counts()
            for y in YEARS:
                entry['years'][y] = int(yr.get(y, 0))
        except Exception as e:
            entry['error'] = str(e)[:120]
        rows.append(entry)
    return rows


def load_snapshot():
    if os.path.exists(SNAPSHOT):
        try:
            with open(SNAPSHOT, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_snapshot(snap):
    with open(SNAPSHOT, 'w', encoding='utf-8') as f:
        json.dump(snap, f, ensure_ascii=False, indent=2, default=str)


def _is_alive(pid: int) -> bool:
    try:
        from process_watchdog import _is_pid_alive
        return _is_pid_alive(pid)
    except Exception:
        return False


def scan_processes():
    """logs/pids/*.json を読み、生死と最新ログを取得"""
    procs = []
    if not os.path.exists(PID_DIR):
        return procs
    for fn in sorted(os.listdir(PID_DIR)):
        if not fn.endswith('.json'):
            continue
        p = os.path.join(PID_DIR, fn)
        try:
            with open(p, 'r', encoding='utf-8') as f:
                e = json.load(f)
        except Exception:
            continue
        pid = int(e.get('pid', 0))
        alive = _is_alive(pid)
        last_line = ''
        log = e.get('log')
        if log and os.path.exists(log):
            try:
                with open(log, 'rb') as lf:
                    lf.seek(0, os.SEEK_END)
                    size = lf.tell()
                    lf.seek(max(0, size - 2048))
                    tail = lf.read().decode('utf-8', errors='replace').splitlines()
                    last_line = tail[-1][:160] if tail else ''
            except Exception:
                pass
        procs.append({
            'name': e.get('name'),
            'pid': pid,
            'alive': alive,
            'started_at': e.get('started_at'),
            'restart_count': e.get('restart_count', 0),
            'last_line': last_line,
        })
    return procs


def render_table(coverage, procs, diff):
    lines = []
    lines.append('=' * 100)
    lines.append(f"  scrape_status  {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append('=' * 100)
    # Coverage table
    header = f"  {'CSV':<35} {'TOTAL':>8}  " + '  '.join(f"{y:>6}" for y in YEARS) + '  DIFF'
    lines.append(header)
    lines.append('  ' + '-' * (len(header) - 2))
    for e in coverage:
        d = diff.get(e['file'], 0)
        dstr = f"+{d}" if d > 0 else str(d)
        yrs = '  '.join(f"{e['years'].get(y,0):>6}" for y in YEARS)
        status = '  ' if e['exists'] else ' X'
        lines.append(f"{status}{e['file']:<35} {e['total']:>8}  {yrs}  {dstr}")
    # Processes
    lines.append('')
    lines.append('  --- Processes ---')
    if not procs:
        lines.append('  (no entries in logs/pids/)')
    else:
        for p in procs:
            lines.append(f"  [{'OK' if p['alive'] else 'DEAD'}] {p['name']:<30} pid={p['pid']:<7} "
                          f"restart={p['restart_count']}")
            if p['last_line']:
                lines.append(f"       last: {p['last_line']}")
    lines.append('=' * 100)
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--notify', action='store_true', help='Discord にも送信')
    ap.add_argument('--json', action='store_true', help='JSON出力のみ')
    ap.add_argument('--save', action='store_true', default=True, help='snapshot更新 (default)')
    args = ap.parse_args()

    coverage = scan_csv_coverage()
    procs = scan_processes()

    # 差分計算
    prev = load_snapshot()
    diff = {}
    for e in coverage:
        prev_total = prev.get('coverage', {}).get(e['file'], {}).get('total', e['total'])
        diff[e['file']] = e['total'] - prev_total

    if args.json:
        print(json.dumps({'coverage': coverage, 'processes': procs, 'diff': diff},
                          ensure_ascii=False, indent=2))
    else:
        text = render_table(coverage, procs, diff)
        print(text)

    # Snapshot更新
    if args.save:
        save_snapshot({
            'timestamp': datetime.now().isoformat(),
            'coverage': {e['file']: {'total': e['total'], 'years': e['years']} for e in coverage},
            'processes': procs,
        })

    if args.notify:
        try:
            from notify import send_discord
            # 簡易サマリのみ（テーブルは長すぎるため）
            total_rows = sum(e['total'] for e in coverage)
            diff_positive = sum(v for v in diff.values() if v > 0)
            alive = sum(1 for p in procs if p['alive'])
            msg = (f"CSVs: {len(coverage)} 全{total_rows:,}行\n"
                   f"直近差分: +{diff_positive:,}行\n"
                   f"Processes: {alive}/{len(procs)} alive")
            send_discord("📊 scrape_status", msg, color='green', channel='updates')
        except Exception as e:
            print(f"[WARN] notify failed: {e}")


if __name__ == '__main__':
    main()
