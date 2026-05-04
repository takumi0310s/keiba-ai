"""TYB publish タイミング 観測スクリプト.

JRDB Tyb (直前データ) の publish タイミング を時系列で記録。
5/3 14:50 の取得失敗 (HTTP 404) を契機に、本来の publish 時刻不明問題を解決する。

Usage:
  python tools/tyb_publish_monitor.py            # 今日 fetch 試行
  python tools/tyb_publish_monitor.py --date 20260510  # 指定日

動作:
  1. data/tyb_publish_log.csv に試行記録 (date, fetch_time, http_status, size, available)
  2. 新規に 200 OK になった (前回 404 → 今回 200 の遷移) → Discord 通知
  3. 既に取得済 → スキップ
"""
import sys, os, io, argparse, json
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import requests
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

LOG_CSV = 'data/tyb_publish_log.csv'
LOG_HEADER = ['date_iso', 'jrdb_date', 'fetch_time', 'http_status', 'size_bytes', 'first_publish']


def load_env():
    env_path = '.env'
    if not os.path.exists(env_path):
        return None, None
    j_id, j_pw = None, None
    with open(env_path) as f:
        for line in f:
            if line.startswith('JRDB_ID='):
                j_id = line.split('=',1)[1].strip().strip('"').strip("'")
            elif line.startswith('JRDB_PASSWORD='):
                j_pw = line.split('=',1)[1].strip().strip('"').strip("'")
    return j_id, j_pw


def fetch_tyb(jrdb_date):
    """Try to HEAD/GET TYB{jrdb_date}.lzh"""
    url = f'http://www.jrdb.com/member/data/Tyb/TYB{jrdb_date}.lzh'
    j_id, j_pw = load_env()
    if not j_id:
        print('JRDB credentials not found in .env')
        return None, 0
    try:
        # Use HEAD first (lighter)
        resp = requests.head(url, auth=(j_id, j_pw), timeout=10, allow_redirects=True)
        size = int(resp.headers.get('Content-Length', '0') or 0)
        return resp.status_code, size
    except Exception as e:
        print(f'Error: {e}')
        return None, 0


def append_log(row):
    write_header = not os.path.exists(LOG_CSV) or os.path.getsize(LOG_CSV) == 0
    with open(LOG_CSV, 'a', encoding='utf-8') as f:
        if write_header:
            f.write(','.join(LOG_HEADER) + '\n')
        f.write(','.join(str(v) for v in row) + '\n')


def is_first_publish(jrdb_date):
    """Returns True if this is the first time we got 200 for this jrdb_date."""
    if not os.path.exists(LOG_CSV):
        return False
    import pandas as pd
    df = pd.read_csv(LOG_CSV, dtype=str)
    sub = df[df['jrdb_date'] == jrdb_date]
    prev_200 = (pd.to_numeric(sub.get('http_status'), errors='coerce') == 200).any()
    return not prev_200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=None, help='YYYYMMDD (default: today)')
    args = ap.parse_args()

    if args.date:
        date_iso = args.date
    else:
        # default = next race day prediction (today or upcoming Sat/Sun)
        date_iso = datetime.now().strftime('%Y%m%d')
    jrdb_date = date_iso[2:8]

    fetch_time = datetime.now().strftime('%H:%M:%S')
    print(f"=== TYB publish monitor ===")
    print(f"  date: {date_iso} (jrdb={jrdb_date})")
    print(f"  fetch_time: {fetch_time}")

    status, size = fetch_tyb(jrdb_date)
    print(f"  HTTP={status} Size={size}B")

    first = False
    if status == 200 and size > 200:
        if is_first_publish(jrdb_date):
            first = True
            print(f"  ✨ FIRST PUBLISH!")

    append_log([date_iso, jrdb_date, fetch_time, status if status else 'ERR', size, 'yes' if first else 'no'])

    # Discord 通知 (first publish only)
    if first:
        try:
            sys.path.insert(0, 'tools')
            from notify import send_discord
            send_discord(
                title=f'⏰ TYB {date_iso} publish 検出',
                message=f'JRDB Tyb {jrdb_date}.lzh が初公開検出\nfetch_time: {fetch_time}\nsize: {size}B',
                color='green', channel='updates',
            )
            print('  Discord 通知 OK')
        except Exception as e:
            print(f'  Discord error: {e}')

    return 0 if status else 1


if __name__ == '__main__':
    sys.exit(main())
