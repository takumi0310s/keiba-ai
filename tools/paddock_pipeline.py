#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paddock 動画 capture 自動 pipeline (全レース全頭 / Top N).

daily_predictions/{YYYYMMDD}.csv から race_id list 取得、 各 race の shutuba から
horse_id 抽出、 paddock 動画を bulk capture。 V21 学習 data 蓄積の基盤。

【規約遵守】 netkeiba 第 14 条 私的利用範囲、 frame のみ抽出 (screenshot)、 配布 NG、
rate limit 厳守 (sleep 5 秒/race、 SCRAPER-GUARD 範囲)。

Usage:
    # 5/10 開催の全 race × Top 3 馬 paddock capture
    python tools/paddock_pipeline.py 20260510 --top-n 3 --fps 3 --duration 30

    # 全 馬 (15 頭) capture (時間長、 night-only 推奨)
    python tools/paddock_pipeline.py 20260510 --top-n 18 --duration 20

    # DRY-RUN (taglist のみ、 capture skip)
    python tools/paddock_pipeline.py 20260510 --dry-run
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY_PRED_DIR = os.path.join(BASE_DIR, 'data', 'daily_predictions')
FRAME_DIR = os.path.join(BASE_DIR, 'data', 'paddock_frames')
PIPELINE_LOG = os.path.join(BASE_DIR, 'data', 'paddock_pipeline_log.json')

CAPTURE_SCRIPT = os.path.join(BASE_DIR, 'tools', 'paddock_video_capture.py')
COOKIE_PATH = os.path.join(BASE_DIR, 'data', 'cookies.json')


def fetch_horse_ids_for_race(race_id):
    """shutuba ページから 馬番 → horse_id mapping 取得."""
    import requests
    if not os.path.exists(COOKIE_PATH):
        return {}
    cookies_list = json.load(open(COOKIE_PATH, 'r', encoding='utf-8'))
    cookies = {c['name']: c['value'] for c in cookies_list}
    url = f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}'
    try:
        r = requests.get(url, cookies=cookies, timeout=15,
                          headers={'User-Agent': 'Mozilla/5.0'})
    except Exception as e:
        print(f'  [WARN] fetch error: {e}')
        return {}

    r.encoding = 'euc-jp'
    text = r.text
    # 馬番 + horse_id を 一緒に取れる pattern (HorseInfo class 周辺)
    # netkeiba shutuba: <td class="Umaban*">N</td> ... <td class="HorseInfo"> ... db.netkeiba.com/horse/HORSEID
    horse_ids = {}
    # シンプル: horse_id とその直前の Umaban を組で取る
    pattern = re.compile(
        r'<td[^>]*class=["\'][^"\']*Umaban[^"\']*["\'][^>]*>\s*(\d+)\s*</td>'
        r'[\s\S]{0,2000}?'
        r'/horse/(\d{10})',
    )
    for m in pattern.finditer(text):
        umaban = int(m.group(1))
        h_id = m.group(2)
        if umaban not in horse_ids:
            horse_ids[umaban] = h_id
    return horse_ids


def is_already_captured(race_id, horse_id):
    out_dir = os.path.join(FRAME_DIR, f'{race_id}_{horse_id}')
    manifest = os.path.join(out_dir, 'manifest.json')
    if not os.path.exists(manifest):
        return False
    try:
        m = json.load(open(manifest, 'r', encoding='utf-8'))
        return m.get('summary', {}).get('frames_saved', 0) > 5
    except Exception:
        return False


def load_log():
    if os.path.exists(PIPELINE_LOG):
        try:
            return json.load(open(PIPELINE_LOG, 'r', encoding='utf-8'))
        except Exception:
            return {}
    return {}


def save_log(log):
    with open(PIPELINE_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def capture_one(race_id, horse_id, fps, duration):
    cmd = [
        sys.executable, CAPTURE_SCRIPT, str(horse_id),
        '--race-id', str(race_id),
        '--fps', str(fps),
        '--duration', str(duration),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 90,
                           encoding='utf-8', errors='replace')
        if r.returncode == 0:
            # parse last line for "[OK] frames=N"
            m = re.search(r'frames=(\d+).*errs=(\d+)', r.stdout)
            if m:
                return {'status': 'OK', 'frames': int(m.group(1)), 'errs': int(m.group(2))}
            return {'status': 'OK', 'note': 'no frame info'}
        return {'status': 'FAIL', 'rc': r.returncode, 'stderr': r.stderr[:200]}
    except subprocess.TimeoutExpired:
        return {'status': 'TIMEOUT'}
    except Exception as e:
        return {'status': 'ERROR', 'msg': str(e)}


def main():
    ap = argparse.ArgumentParser(description='paddock 自動 capture pipeline')
    ap.add_argument('date', help='YYYYMMDD (e.g., 20260510)')
    ap.add_argument('--top-n', dest='top_n', type=int, default=3,
                    help='各 race で 取得する 馬数 (default 3)')
    ap.add_argument('--fps', type=int, default=3)
    ap.add_argument('--duration', type=int, default=30)
    ap.add_argument('--max-races', type=int, default=None,
                    help='処理 race 上限 (test 用)')
    ap.add_argument('--sleep-between', type=int, default=5,
                    help='馬 ごとの sleep 秒 (default 5、 rate limit)')
    ap.add_argument('--dry-run', dest='dry_run', action='store_true',
                    help='task list のみ表示、 capture skip')
    args = ap.parse_args()

    csv_path = os.path.join(DAILY_PRED_DIR, f'{args.date}.csv')
    if not os.path.exists(csv_path):
        print(f'[ERROR] not found: {csv_path}')
        return 1

    # 1. race_id 一覧 取得
    races = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('race_id'):
                races.append({
                    'race_id': row['race_id'],
                    'course': row.get('course'),
                    'top1_num': row.get('top1_num'),
                    'top2_num': row.get('top2_num'),
                    'top3_num': row.get('top3_num'),
                })

    if args.max_races:
        races = races[:args.max_races]
    print(f'[INFO] {len(races)} races to process from {args.date}')

    # 2. log 読込 (resume)
    log = load_log()
    if args.date not in log:
        log[args.date] = {}

    # 3. 各 race 処理
    total_capture = 0
    total_skip = 0
    total_fail = 0

    for i, race in enumerate(races, 1):
        race_id = race['race_id']
        course = race.get('course', '?')
        print(f'\n[{i}/{len(races)}] race_id={race_id} ({course})')

        # horse_id 取得
        horse_ids = fetch_horse_ids_for_race(race_id)
        if not horse_ids:
            print('  [SKIP] no horse_ids')
            total_fail += 1
            continue

        # top-N の 馬番 → horse_id
        umabans_to_capture = []
        if args.top_n <= 3 and race.get('top1_num'):
            umabans_to_capture = [int(float(race['top1_num']))]
            if args.top_n >= 2 and race.get('top2_num'):
                umabans_to_capture.append(int(float(race['top2_num'])))
            if args.top_n >= 3 and race.get('top3_num'):
                umabans_to_capture.append(int(float(race['top3_num'])))
        else:
            # 全 馬 (--top-n 18 等)
            umabans_to_capture = sorted(horse_ids.keys())[:args.top_n]

        print(f'  {len(horse_ids)} horses found, target {len(umabans_to_capture)}: {umabans_to_capture}')

        for umaban in umabans_to_capture:
            h_id = horse_ids.get(umaban)
            if not h_id:
                print(f'    umaban {umaban} -> horse_id 不明')
                continue
            key = f'{race_id}_{h_id}'

            if is_already_captured(race_id, h_id):
                print(f'    [SKIP] {key} already captured')
                total_skip += 1
                continue

            if args.dry_run:
                print(f'    [DRY-RUN] would capture {key}')
                total_capture += 1
                continue

            print(f'    [CAPTURE] umaban={umaban}, horse_id={h_id}')
            result = capture_one(race_id, h_id, args.fps, args.duration)
            log[args.date][key] = {**result, 'umaban': umaban,
                                    'captured_at': datetime.now().isoformat()}
            if result['status'] == 'OK':
                total_capture += 1
                print(f'      OK: frames={result.get("frames", "?")}')
            else:
                total_fail += 1
                print(f'      {result["status"]}: {result.get("stderr", result.get("msg", ""))[:120]}')

            save_log(log)
            time.sleep(args.sleep_between)

    print(f'\n[SUMMARY] capture={total_capture}, skip={total_skip}, fail={total_fail}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
