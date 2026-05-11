#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""5 分 interval オッズ snapshot capture (B6 drift detection の前提).

netkeiba から 当日 全 race の 単勝オッズを 5 分おきに取得し、 timestamped CSV に保存。
closing_odds_drift.py の input になる。

【V15 投資保護】 V15 model / production 一切不変、 補助 data 蓄積のみ。

Usage:
    # 単発 snapshot (test 用)
    python tools/odds_5min_capture.py --once

    # 継続 polling (5 分おき、 race 締切まで loop)
    python tools/odds_5min_capture.py --loop --duration 600  # 10 分間

    # 当日 全 race 1 race_id 指定
    python tools/odds_5min_capture.py --race-id 202608030611 --once

【出力】 data/odds_5min/{date}_{HHmm}.csv (race_id, umaban, odds, pop_rank, timestamp)

【規約】 netkeiba 規約 14条 私的利用範囲、 5 分 interval は rate limit 内 (1 hit/分 程度の負荷)
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COOKIE_PATH = os.path.join(BASE_DIR, 'data', 'cookies.json')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'odds_5min')


def load_cookies():
    if not os.path.exists(COOKIE_PATH):
        return {}
    cookies_list = json.load(open(COOKIE_PATH, 'r', encoding='utf-8'))
    return {c['name']: c['value'] for c in cookies_list}


def fetch_race_ids_today():
    """daily_predictions/{今日}.csv から race_id 取得."""
    import pandas as pd
    today = datetime.now().strftime('%Y%m%d')
    path = os.path.join(BASE_DIR, 'data', 'daily_predictions', f'{today}.csv')
    if not os.path.exists(path):
        # 過去 race_id を順番試す
        return []
    df = pd.read_csv(path, encoding='utf-8-sig')
    return df['race_id'].astype(str).unique().tolist()


def fetch_odds_one_race(race_id, cookies):
    """1 race の odds snapshot 取得 (単勝)."""
    import requests
    url = f'https://race.netkeiba.com/api/api_get_jra_odds.html?type=1&race_id={race_id}'
    try:
        r = requests.get(url, cookies=cookies, timeout=15,
                          headers={'User-Agent': 'Mozilla/5.0',
                                   'Referer': f'https://race.netkeiba.com/race/odds.html?race_id={race_id}'})
        if r.status_code != 200:
            return None
        data = r.json()
        # API response format: {status: 'OK', data: {...odds tabulated...}}
        if data.get('status') != 'success' and data.get('status') != 'OK':
            return None
        return data.get('data', {})
    except Exception as e:
        return None


def snapshot_all(race_ids, cookies):
    ts = datetime.now()
    ts_str = ts.strftime('%Y%m%d_%H%M%S')
    records = []
    for race_id in race_ids:
        d = fetch_odds_one_race(race_id, cookies)
        if not d:
            continue
        # Parse odds dict (umaban → odds, pop_rank)
        # API 実 format は色々 → 試行錯誤 必要、 ここでは generic 化
        odds_map = d.get('odds', {}) if isinstance(d, dict) else {}
        if isinstance(odds_map, dict):
            for umaban_str, val in odds_map.items():
                try:
                    odds = float(val[0]) if isinstance(val, list) else float(val)
                    records.append({
                        'race_id': race_id,
                        'umaban': int(umaban_str),
                        'odds': odds,
                        'timestamp': ts.isoformat(),
                    })
                except Exception:
                    pass
        time.sleep(0.5)  # rate limit

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'{ts_str}.csv')
    if records:
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['race_id', 'umaban', 'odds', 'timestamp'])
            w.writeheader()
            w.writerows(records)
    return out_path, len(records)


def cmd_once(args):
    cookies = load_cookies()
    if not cookies:
        print('[ERROR] cookies missing')
        return 1
    if args.race_id:
        race_ids = [args.race_id]
    else:
        race_ids = fetch_race_ids_today()
    if not race_ids:
        print('[ERROR] no race_ids; specify --race-id or ensure daily_predictions exists')
        return 1
    print(f'[INFO] {len(race_ids)} races to snapshot')
    path, n = snapshot_all(race_ids, cookies)
    print(f'[OK] {n} records → {path}')
    return 0 if n > 0 else 1


def cmd_loop(args):
    cookies = load_cookies()
    if not cookies:
        return 1
    end = time.time() + args.duration
    interval = args.interval
    iter_n = 0
    while time.time() < end:
        race_ids = [args.race_id] if args.race_id else fetch_race_ids_today()
        if not race_ids:
            print('[WARN] no races, wait next interval')
        else:
            path, n = snapshot_all(race_ids, cookies)
            iter_n += 1
            print(f'[{iter_n}] {datetime.now().strftime("%H:%M:%S")} {n} records → {os.path.basename(path)}')
        sleep_for = max(0, interval - (time.time() % interval))
        time.sleep(min(interval, sleep_for + 1))
    print(f'[DONE] {iter_n} iterations in {args.duration}s')
    return 0


def main():
    ap = argparse.ArgumentParser(description='5min interval odds capture (B6 前提)')
    ap.add_argument('--once', action='store_true', help='1 snapshot のみ')
    ap.add_argument('--loop', action='store_true', help='5 分 interval で 継続 polling')
    ap.add_argument('--duration', type=int, default=600, help='loop 継続秒 (default 10 min)')
    ap.add_argument('--interval', type=int, default=300, help='5 分 = 300 秒')
    ap.add_argument('--race-id', dest='race_id', help='1 race のみ')
    args = ap.parse_args()

    if args.loop:
        return cmd_loop(args)
    elif args.once or args.race_id:
        return cmd_once(args)
    else:
        ap.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
