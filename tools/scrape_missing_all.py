#!/usr/bin/env python
"""欠落データ一括取得バッチ

2026-04-13時点で欠落している全データを直列実行で補填する。

実行順（軽いものから、IPバンリスク分散）:
  1. bulk_scrape_upset       — upset 2025 + 2024足りない分
  2. scrape_master_index     — track_bias/race_lap(単) 2023/2024 埋め
  3. scrape_super_premium    — training_eval 2020-2023（年ごと直列）
  4. scrape_master_course    — race_laps(複) 2020-2026

Usage:
    python -u tools/scrape_missing_all.py                 # 全ステップ実行
    python -u tools/scrape_missing_all.py --only upset    # 個別実行
    python -u tools/scrape_missing_all.py --skip premium  # 一部スキップ
    python -u tools/scrape_missing_all.py --dry-run       # 実行内容表示のみ

IPバン対策で直列実行。全ステップ完了まで数十時間〜。
"""
import os
import sys
import subprocess
import argparse
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

STEPS = [
    ('upset',   ['python', '-u', 'tools/bulk_scrape_upset.py']),
    ('mi2020',  ['python', '-u', 'tools/scrape_master_index.py', '--year', '2020']),
    ('mi2021',  ['python', '-u', 'tools/scrape_master_index.py', '--year', '2021']),
    ('mi2022',  ['python', '-u', 'tools/scrape_master_index.py', '--year', '2022']),
    ('mi2023',  ['python', '-u', 'tools/scrape_master_index.py', '--year', '2023']),
    ('mi2024',  ['python', '-u', 'tools/scrape_master_index.py', '--year', '2024']),
    ('te2020',  ['python', '-u', 'tools/scrape_super_premium.py', '--year', '2020', '--source', 'newspaper']),
    ('te2021',  ['python', '-u', 'tools/scrape_super_premium.py', '--year', '2021', '--source', 'newspaper']),
    ('te2022',  ['python', '-u', 'tools/scrape_super_premium.py', '--year', '2022', '--source', 'newspaper']),
    ('te2023',  ['python', '-u', 'tools/scrape_super_premium.py', '--year', '2023', '--source', 'newspaper']),
    ('laps',    ['python', '-u', 'tools/scrape_master_course.py', '--source', 'race_laps', '--year', '2020-2026']),
]


def run_step(name, cmd, log_path):
    print(f"\n{'='*60}\n[{datetime.now():%H:%M:%S}] STEP: {name}\n  cmd: {' '.join(cmd)}\n  log: {log_path}\n{'='*60}", flush=True)
    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')
    with open(log_path, 'w', encoding='utf-8', errors='replace') as lf:
        proc = subprocess.run(cmd, cwd=BASE_DIR, stdout=lf, stderr=subprocess.STDOUT, env=env)
    print(f"[{datetime.now():%H:%M:%S}] STEP {name} exit={proc.returncode}", flush=True)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='+', help='指定ステップのみ実行')
    ap.add_argument('--skip', nargs='+', default=[], help='指定ステップをスキップ')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    steps = STEPS
    if args.only:
        steps = [(n, c) for n, c in steps if n in args.only]
    if args.skip:
        steps = [(n, c) for n, c in steps if n not in args.skip]

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"scrape_missing_all.py start {ts}")
    print(f"  steps: {[n for n,_ in steps]}")
    if args.dry_run:
        for n, c in steps:
            print(f"  [{n}] {' '.join(c)}")
        return

    results = {}
    t0 = time.time()
    for name, cmd in steps:
        log_path = os.path.join(LOG_DIR, f'missing_{name}_{ts}.log')
        try:
            rc = run_step(name, cmd, log_path)
            results[name] = rc
        except KeyboardInterrupt:
            print(f"\nInterrupted at {name}")
            results[name] = 'INT'
            break
        except Exception as e:
            print(f"EXC at {name}: {e}")
            results[name] = f'ERR:{e}'

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}\nALL DONE elapsed={elapsed:.1f}min\n{'='*60}")
    for n, r in results.items():
        print(f"  {n}: {r}")

    try:
        sys.path.insert(0, BASE_DIR)
        from tools.notify_done import notify_done  # type: ignore
        notify_done('scrape_missing_all', f"{elapsed:.1f}min  {results}")
    except Exception:
        pass


if __name__ == '__main__':
    main()
