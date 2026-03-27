#!/usr/bin/env python
"""毎週土曜朝に実行: 当日全レースのプレミアムデータを事前取得

取得データ:
- 調教タイム（oikiri）
- タイム指数（speed index）
- 厩舎コメント

キャッシュ: data/weekly_premium_cache/{date}/ に保存
予測時にキャッシュがあればスクレイピング不要で高速化

Usage:
    python tools/weekly_premium_update.py              # 今日
    python tools/weekly_premium_update.py --date 20260328
"""
import pandas as pd
import numpy as np
import requests
import re
import os
import sys
import io
import json
import time
import argparse
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'weekly_premium_cache')
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def get_race_ids_for_date(date_str):
    """netkeibaの開催情報から当日のrace_idリストを取得"""
    from scrape_training import _make_session
    session = _make_session()
    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)

    race_ids = []
    # netkeiba race list page
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
    try:
        resp = session.get(url, timeout=15)
        resp.encoding = 'EUC-JP'
        soup = BeautifulSoup(resp.text, 'html.parser')

        for a in soup.find_all('a', href=True):
            m = re.search(r'race_id=(\d{12})', a['href'])
            if m:
                race_ids.append(m.group(1))
    except Exception as e:
        print(f"  WARNING: race list取得失敗: {e}")

    return sorted(set(race_ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='')
    args = parser.parse_args()

    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime('%Y%m%d')

    print("=" * 60)
    print(f"  Weekly Premium Update: {date_str}")
    print("=" * 60)

    # Create cache directory
    cache_path = os.path.join(CACHE_DIR, date_str)
    os.makedirs(cache_path, exist_ok=True)

    # Get race IDs
    race_ids = get_race_ids_for_date(date_str)
    print(f"  Races found: {len(race_ids)}")

    if not race_ids:
        # Also try tomorrow (for Saturday morning pre-fetch of Sunday races)
        tomorrow = (datetime.strptime(date_str, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
        race_ids_tmr = get_race_ids_for_date(tomorrow)
        if race_ids_tmr:
            print(f"  Tomorrow ({tomorrow}): {len(race_ids_tmr)} races")
            race_ids.extend(race_ids_tmr)

    if not race_ids:
        print("  No races found for today or tomorrow.")
        return

    # Import scrapers
    from scrape_training import fetch_training_times, fetch_stable_comments, _make_session
    from scrape_speed_index import scrape_speed_index, _load_session

    si_session = _load_session()

    all_data = {}
    for i, race_id in enumerate(race_ids):
        print(f"\r  [{i+1}/{len(race_ids)}] {race_id}", end='', flush=True)

        race_data = {'training': {}, 'speed_index': {}, 'comments': {}}

        # Training times
        try:
            training = fetch_training_times(race_id)
            race_data['training'] = {str(k): v for k, v in training.items()}
        except Exception:
            pass

        # Speed index
        if si_session:
            try:
                si_rows = scrape_speed_index(si_session, race_id)
                for row in si_rows:
                    race_data['speed_index'][str(row[1])] = {
                        'horse_name': row[2],
                        'index_max': row[6], 'index_avg5': row[7],
                        'index_dist': row[8], 'index_course': row[9],
                        'index_run1': row[10], 'index_run2': row[11], 'index_run3': row[12],
                    }
            except Exception:
                pass

        # Comments
        try:
            comments = fetch_stable_comments(race_id)
            race_data['comments'] = {str(k): v for k, v in comments.items()}
        except Exception:
            pass

        all_data[race_id] = race_data
        time.sleep(3)

    # Save cache
    cache_file = os.path.join(cache_path, 'premium_cache.json')
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2, default=str)

    # Summary
    n_training = sum(1 for d in all_data.values() if d['training'])
    n_si = sum(1 for d in all_data.values() if d['speed_index'])
    n_comment = sum(1 for d in all_data.values() if d['comments'])

    print(f"\n\n  Results:")
    print(f"    Training: {n_training}/{len(race_ids)} races")
    print(f"    Speed Index: {n_si}/{len(race_ids)} races")
    print(f"    Comments: {n_comment}/{len(race_ids)} races")
    print(f"    Cache: {cache_file}")
    print("  Done!")

    try:
        sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
        from notify import send_discord
        send_discord("週末Premium更新完了",
                     f"調教: {n_training}R / 指数: {n_si}R / コメント: {n_comment}R",
                     color="green")
    except Exception:
        pass


if __name__ == '__main__':
    main()
