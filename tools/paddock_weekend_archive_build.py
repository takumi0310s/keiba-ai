#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paddock 過去 開催 一括 archive build (V21 学習 data 蓄積 用).

5/4-5/10 など 過去 数週末分 paddock 動画 を 全 race × top-N で 一括取得し、 YOLOv8 + gait + body_condition 特徴量まで chain 抽出。 中断 / resume / log を完備。

【規約】 netkeiba 規約 14 条 私的利用範囲、 SCRAPER-GUARD 範囲 rate limit (5s/horse 間隔)
【V15 投資保護】 V15 model / production 一切 不変、 background data 蓄積のみ

Usage:
    # 5/4-5/10 の archive を build (約 6 開催 × 12 race × 3 馬 = 216 動画)
    python tools/paddock_weekend_archive_build.py 20260504 20260510 --top-n 3

    # 単発 (1 開催) build
    python tools/paddock_weekend_archive_build.py 20260411 20260411 --top-n 3

    # dry-run (実 capture せず list のみ)
    python tools/paddock_weekend_archive_build.py 20260504 20260510 --dry-run
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROGRESS_LOG = os.path.join(BASE_DIR, 'data', 'paddock_weekend_progress.json')


def date_range(start_str, end_str):
    """YYYYMMDD 形式の range generator."""
    s = datetime.strptime(start_str, '%Y%m%d')
    e = datetime.strptime(end_str, '%Y%m%d')
    cur = s
    while cur <= e:
        yield cur.strftime('%Y%m%d')
        cur += timedelta(days=1)


def has_daily_predictions(date):
    return os.path.exists(os.path.join(BASE_DIR, 'data', 'daily_predictions', f'{date}.csv'))


def load_progress():
    if not os.path.exists(PROGRESS_LOG):
        return {}
    try:
        return json.load(open(PROGRESS_LOG, 'r', encoding='utf-8'))
    except Exception:
        return {}


def save_progress(p):
    with open(PROGRESS_LOG, 'w', encoding='utf-8') as f:
        json.dump(p, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description='paddock weekend archive build')
    ap.add_argument('start', help='YYYYMMDD 開始日')
    ap.add_argument('end', help='YYYYMMDD 終了日')
    ap.add_argument('--top-n', dest='top_n', type=int, default=3)
    ap.add_argument('--fps', type=int, default=3)
    ap.add_argument('--duration', type=int, default=30)
    ap.add_argument('--max-races-per-day', dest='max_races', type=int, default=None,
                    help='1 日 上限 race (test 用)')
    ap.add_argument('--sleep-between', dest='sleep', type=int, default=5)
    ap.add_argument('--dry-run', dest='dry_run', action='store_true')
    ap.add_argument('--skip-features', dest='skip_features', action='store_true',
                    help='capture のみ、 YOLOv8/gait/body 抽出 skip')
    args = ap.parse_args()

    dates = list(date_range(args.start, args.end))
    print(f'[INFO] target dates: {dates}')

    progress = load_progress()
    if args.start + '_' + args.end not in progress:
        progress[args.start + '_' + args.end] = {'dates': {}, 'started': datetime.now().isoformat()}
    session_key = args.start + '_' + args.end

    total_capture = 0
    total_skip = 0
    total_fail = 0

    for date in dates:
        if not has_daily_predictions(date):
            print(f'\n[{date}] daily_predictions 無 (非開催日)、 SKIP')
            continue

        already_done = progress[session_key]['dates'].get(date, {}).get('done', False)
        if already_done:
            print(f'\n[{date}] 既 done (progress log)、 SKIP')
            continue

        print(f'\n=== {date} archive build 開始 ===')

        # 1. paddock_pipeline.py で capture
        cmd = [sys.executable, os.path.join(BASE_DIR, 'tools', 'paddock_pipeline.py'),
                date, '--top-n', str(args.top_n),
                '--fps', str(args.fps), '--duration', str(args.duration),
                '--sleep-between', str(args.sleep)]
        if args.max_races:
            cmd += ['--max-races', str(args.max_races)]
        if args.dry_run:
            cmd += ['--dry-run']

        print(f'[CAPTURE] cmd: paddock_pipeline.py {date} --top-n {args.top_n}')
        try:
            r = subprocess.run(cmd, timeout=args.duration * args.top_n * 24 * 2)
            cap_rc = r.returncode
        except subprocess.TimeoutExpired:
            cap_rc = -1
            print(f'[CAPTURE] TIMEOUT')

        if cap_rc != 0:
            total_fail += 1
            progress[session_key]['dates'][date] = {'done': False, 'error': f'capture rc={cap_rc}'}
        else:
            # 2. YOLOv8 + gait + body 抽出 (skip-features オフなら)
            if not args.skip_features and not args.dry_run:
                print(f'[FEATURES] video_pipeline_unified.py --features-only')
                cmd_feat = [sys.executable,
                              os.path.join(BASE_DIR, 'tools', 'video_pipeline_unified.py'),
                              date, '--top-n', str(args.top_n),
                              '--features-only']
                if args.max_races:
                    cmd_feat += ['--max-races', str(args.max_races)]
                try:
                    subprocess.run(cmd_feat, timeout=600)
                except subprocess.TimeoutExpired:
                    pass

            progress[session_key]['dates'][date] = {
                'done': True,
                'completed_at': datetime.now().isoformat(),
            }
            total_capture += 1

        save_progress(progress)

    progress[session_key]['ended'] = datetime.now().isoformat()
    save_progress(progress)

    print(f'\n=== SUMMARY ===')
    print(f'  total dates processed: {len(dates)}')
    print(f'  successful: {total_capture}')
    print(f'  failed: {total_fail}')
    print(f'  progress: {PROGRESS_LOG}')

    # V21 training data builder 自動実行 (蓄積後)
    if not args.dry_run and not args.skip_features and total_capture > 0:
        print('\n[BUILD V21 training data]')
        try:
            r = subprocess.run([sys.executable,
                                  os.path.join(BASE_DIR, 'tools', 'v21_training_data_builder.py'),
                                  '--year-from', '2024', '--year-to', '2026'],
                                timeout=300, capture_output=True, text=True,
                                encoding='utf-8', errors='replace')
            print(r.stdout[-300:])
        except Exception as e:
            print(f'  [WARN] {e}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
