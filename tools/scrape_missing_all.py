#!/usr/bin/env python
"""欠落データ一括取得バッチ

2026-04-13時点で欠落している全データを直列実行で補填する。

実行順（軽いものから、IPバンリスク分散）:
  1. bulk_scrape_upset         - upset 2025 + 2024足りない分
  2. scrape_master_index       - master_index/track_bias/race_lap(単) 2020-2024 埋め
  3. scrape_super_premium --source newspaper - training_eval/ai_position/
     ai_opinion/ana_best 2020-2023
  4. scrape_super_premium --source result    - track_index 2020-2023
  5. scrape_shinba_eval --year  - 新馬評価 2020-2023
  6. scrape_master_course race_laps - race_laps(複) 2020-2026

各ステップ間に 5分クールダウン、金曜22:00以降は自動停止。

Usage:
    python -u tools/scrape_missing_all.py                 # 全ステップ実行
    python -u tools/scrape_missing_all.py --only upset    # 個別実行
    python -u tools/scrape_missing_all.py --skip premium  # 一部スキップ
    python -u tools/scrape_missing_all.py --dry-run       # 実行内容表示のみ
    python -u tools/scrape_missing_all.py --no-cooldown   # クールダウン無効

IPバン対策で直列実行。全ステップ完了まで数十〜百時間級。
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

sys.path.insert(0, BASE_DIR)

# クールダウン秒数 (各ステップ間)
COOLDOWN_SEC = 300

STEPS = [
    # Upset
    ('upset',   ['python', '-u', 'tools/bulk_scrape_upset.py']),

    # master_index / track_bias / race_lap(単) - 全年補填
    ('mi2020',  ['python', '-u', 'tools/scrape_master_index.py', '--year', '2020']),
    ('mi2021',  ['python', '-u', 'tools/scrape_master_index.py', '--year', '2021']),
    ('mi2022',  ['python', '-u', 'tools/scrape_master_index.py', '--year', '2022']),
    ('mi2023',  ['python', '-u', 'tools/scrape_master_index.py', '--year', '2023']),
    ('mi2024',  ['python', '-u', 'tools/scrape_master_index.py', '--year', '2024']),

    # newspaper: training_eval / ai_position / ai_opinion / ana_best - 2020-2023
    ('te2020',  ['python', '-u', 'tools/scrape_super_premium.py', '--year', '2020', '--source', 'newspaper']),
    ('te2021',  ['python', '-u', 'tools/scrape_super_premium.py', '--year', '2021', '--source', 'newspaper']),
    ('te2022',  ['python', '-u', 'tools/scrape_super_premium.py', '--year', '2022', '--source', 'newspaper']),
    ('te2023',  ['python', '-u', 'tools/scrape_super_premium.py', '--year', '2023', '--source', 'newspaper']),

    # result: track_index - 2020-2023
    ('ti2020',  ['python', '-u', 'tools/scrape_super_premium.py', '--year', '2020', '--source', 'result']),
    ('ti2021',  ['python', '-u', 'tools/scrape_super_premium.py', '--year', '2021', '--source', 'result']),
    ('ti2022',  ['python', '-u', 'tools/scrape_super_premium.py', '--year', '2022', '--source', 'result']),
    ('ti2023',  ['python', '-u', 'tools/scrape_super_premium.py', '--year', '2023', '--source', 'result']),

    # 新馬評価 - 2020-2023
    ('se2020',  ['python', '-u', 'tools/scrape_shinba_eval.py', '--year', '2020']),
    ('se2021',  ['python', '-u', 'tools/scrape_shinba_eval.py', '--year', '2021']),
    ('se2022',  ['python', '-u', 'tools/scrape_shinba_eval.py', '--year', '2022']),
    ('se2023',  ['python', '-u', 'tools/scrape_shinba_eval.py', '--year', '2023']),

    # race_laps(複) 全年
    ('laps',    ['python', '-u', 'tools/scrape_master_course.py', '--source', 'race_laps', '--year', '2020-2026']),
]


def check_guard():
    """scraper_guard: 金曜22:00 - 月曜06:00 はブロック"""
    try:
        from tools.scraper_guard import is_scraping_allowed
        return is_scraping_allowed()
    except Exception:
        return True


def cooldown(seconds, reason='between steps'):
    """指定秒数クールダウン。途中でガード解除を再確認"""
    if seconds <= 0:
        return
    print(f"[{datetime.now():%H:%M:%S}] cooldown {seconds}s ({reason})", flush=True)
    # 60秒刻みで進捗表示
    remaining = seconds
    while remaining > 0:
        step = min(60, remaining)
        time.sleep(step)
        remaining -= step
        if remaining > 0 and remaining % 60 == 0:
            print(f"    cooldown remaining {remaining}s", flush=True)


def run_step(name, cmd, log_path):
    print(f"\n{'='*60}\n[{datetime.now():%H:%M:%S}] STEP: {name}\n  cmd: {' '.join(cmd)}\n  log: {log_path}\n{'='*60}", flush=True)
    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    with open(log_path, 'w', encoding='utf-8', errors='replace') as lf:
        proc = subprocess.run(cmd, cwd=BASE_DIR, stdout=lf, stderr=subprocess.STDOUT, env=env)
    print(f"[{datetime.now():%H:%M:%S}] STEP {name} exit={proc.returncode}", flush=True)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='+', help='指定ステップのみ実行')
    ap.add_argument('--skip', nargs='+', default=[], help='指定ステップをスキップ')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-cooldown', action='store_true', help='クールダウン無効')
    ap.add_argument('--cooldown', type=int, default=COOLDOWN_SEC,
                    help=f'クールダウン秒数 (default {COOLDOWN_SEC})')
    args = ap.parse_args()

    steps = STEPS
    if args.only:
        steps = [(n, c) for n, c in steps if n in args.only]
    if args.skip:
        steps = [(n, c) for n, c in steps if n not in args.skip]

    cooldown_sec = 0 if args.no_cooldown else args.cooldown

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"scrape_missing_all.py start {ts}")
    print(f"  steps: {[n for n,_ in steps]}")
    print(f"  cooldown: {cooldown_sec}s between steps")
    if args.dry_run:
        for n, c in steps:
            print(f"  [{n}] {' '.join(c)}")
        return

    # 起動時ガードチェック
    if not check_guard():
        print(f"[SCRAPER-GUARD] 金曜22:00〜月曜06:00 のためブロック中、実行中止")
        return

    results = {}
    t0 = time.time()
    for i, (name, cmd) in enumerate(steps):
        # 各ステップ前にガード再確認
        if not check_guard():
            print(f"[SCRAPER-GUARD] ブロック時間帯に突入、残りステップを中止")
            break

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

        # クールダウン (最終ステップ後は不要)
        if i < len(steps) - 1 and cooldown_sec > 0:
            cooldown(cooldown_sec)

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
