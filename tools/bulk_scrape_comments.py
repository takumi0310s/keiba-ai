#!/usr/bin/env python
"""厩舎コメント一括取得スクリプト（コメントのみ特化版）

2020-2024の未取得レースの厩舎コメントを取得。
既存データをスキップし、コメントが存在する高クラスレースを優先取得。

netkeibaではclass_code 7(新馬)と23(未勝利/1勝)はコメントデータなし。
class 43以上のレースを対象とする。

Usage:
    python tools/bulk_scrape_comments.py                  # 2020-2024 高クラス
    python tools/bulk_scrape_comments.py --year 2023      # 指定年のみ
    python tools/bulk_scrape_comments.py --all-classes     # 全クラス（低ヒット率含む）
"""
import pandas as pd
import numpy as np
import requests
import re
import os
import sys
import io
import time
import argparse
from datetime import datetime
from bs4 import BeautifulSoup

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DELAY_MIN = 2.0
DELAY_MAX = 4.0
RETRY_DELAY = 30
MAX_RETRIES = 3

COMMENT_CSV = os.path.join(DATA_DIR, 'netkeiba_stable_comments.csv')
COMMENT_HEADER = ['race_id', 'race_date', 'umaban', 'horse_name', 'comment', 'score']

# Classes known to have comments on netkeiba
HIGH_CLASSES = ['15', '43', '67', '115', '131', '163', '179', '195']


def _load_session():
    env_path = os.path.join(BASE_DIR, '.env')
    if not os.path.exists(env_path):
        print("ERROR: .env not found")
        return None
    cookie_str = ''
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('NETKEIBA_COOKIE='):
                cookie_str = line.strip().split('=', 1)[1].strip('"').strip("'")
    if not cookie_str:
        print("ERROR: NETKEIBA_COOKIE not set in .env")
        return None
    session = requests.Session()
    session.headers.update(HEADERS)
    for part in cookie_str.split(';'):
        part = part.strip()
        if '=' in part:
            k, v = part.split('=', 1)
            session.cookies.set(k.strip(), v.strip())
    return session


def _get(session, url, retry=0):
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code in (403, 429):
            if retry < MAX_RETRIES:
                wait = RETRY_DELAY * (retry + 1)
                print(f"\n  {resp.status_code} - waiting {wait}s (retry {retry+1}/{MAX_RETRIES})")
                time.sleep(wait)
                return _get(session, url, retry + 1)
            return None
        resp.encoding = 'EUC-JP'
        return resp
    except Exception:
        if retry < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return _get(session, url, retry + 1)
        return None


def _append_csv(path, rows):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', encoding='utf-8-sig', newline='') as f:
        if write_header:
            f.write(','.join(COMMENT_HEADER) + '\n')
        for row in rows:
            f.write(','.join(str(v).replace(',', '；') for v in row) + '\n')


def _target_to_netkeiba(target_rid):
    rid = str(target_rid).zfill(10)
    cc, yy, k, n, rr = rid[0:2], rid[2:4], rid[4:5], rid[5:6], rid[6:8]
    n_map = {'A': '10', 'B': '11', 'C': '12'}
    n_dec = n_map.get(n, n.zfill(2))
    return f"20{yy}{cc}{k.zfill(2)}{n_dec}{rr}"


def _score_comment(text):
    if not text:
        return 0
    pos3 = ['絶好調', '抜群の動き', '破格', '文句なし']
    pos2 = ['好調', '上昇', '好仕上がり', '万全', '充実', '態勢整', '上積み']
    pos1 = ['順調', '落ち着き', 'まずまず', '元気', '状態面問題な', '上向き', 'いい動き',
            '力は出せる', '態勢は整', 'しっかり']
    neg1 = ['平凡', '変わり身', '現状維持', '微妙', '物足りな']
    neg2 = ['不安', '太め', '細め', '久々', '叩き台', '上積み必要', 'ピリッとし']
    neg3 = ['故障', '出来落ち', '復帰戦', '大幅プラス', '大幅マイナス']

    for w in pos3:
        if w in text: return 3
    for w in neg3:
        if w in text: return -3
    score = 0
    for w in pos2:
        if w in text: score = max(score, 2)
    for w in neg2:
        if w in text: score = min(score, -2)
    if score != 0:
        return score
    for w in pos1:
        if w in text: return 1
    for w in neg1:
        if w in text: return -1
    return 0


def scrape_comment(session, race_id):
    url = f"https://race.netkeiba.com/race/comment.html?race_id={race_id}"
    resp = _get(session, url)
    if resp is None:
        return []
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', class_=re.compile(r'Stable_Comment|Comment_Table'))
    if not table:
        return []

    results = []
    for row in table.find_all('tr'):
        tds = row.find_all('td')
        if len(tds) < 4:
            continue
        try:
            umaban = int(tds[1].get_text(strip=True))
        except (ValueError, TypeError):
            continue
        horse_name = tds[2].get_text(strip=True)[:20]
        comment = tds[3].get_text(strip=True)
        score = _score_comment(comment)
        results.append([race_id, '', umaban, horse_name, comment, score])

    return results


def load_existing_race_ids():
    if not os.path.exists(COMMENT_CSV):
        return set()
    try:
        df = pd.read_csv(COMMENT_CSV, encoding='utf-8-sig', usecols=['race_id'], dtype=str)
        return set(df['race_id'].unique())
    except Exception:
        return set()


def build_task_list(years, all_classes=False):
    """Build list of race IDs to scrape, prioritizing high-class races."""
    csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
    races = pd.read_csv(csv_path, encoding='utf-8-sig',
                         dtype=str, low_memory=False,
                         usecols=['race_id', 'year', 'class_code'])
    races['year_int'] = pd.to_numeric(races['year'], errors='coerce')
    races['nk_id'] = races['race_id'].apply(
        lambda x: _target_to_netkeiba(x) if pd.notna(x) else '')

    existing = load_existing_race_ids()

    target_classes = None if all_classes else HIGH_CLASSES
    tasks = []
    for year in years:
        yr2 = year % 100
        yr_races = races[races['year_int'] == yr2]
        if target_classes:
            yr_races = yr_races[yr_races['class_code'].isin(target_classes)]

        total = yr_races['nk_id'].nunique()
        new_ids = sorted(set(yr_races['nk_id'].unique()) - existing)
        tasks.extend(new_ids)
        already = total - len(new_ids)
        print(f"  {year}: {len(new_ids)} new / {total} target ({already} already done)")

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=0, help='0=2020-2024 all')
    parser.add_argument('--all-classes', action='store_true', help='Include low-class races')
    parser.add_argument('--limit', type=int, default=0, help='Max total races')
    args = parser.parse_args()

    years = [args.year] if args.year > 0 else [2020, 2021, 2022, 2023, 2024, 2025]

    print("=" * 60)
    print(f"  Stable Comment Bulk Scraper")
    print(f"  Years: {years}")
    print(f"  Classes: {'ALL' if args.all_classes else 'HIGH (' + ','.join(HIGH_CLASSES) + ')'}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    session = _load_session()
    if session is None:
        return

    existing = load_existing_race_ids()
    print(f"  Existing: {len(existing)} unique race_ids\n")

    tasks = build_task_list(years, args.all_classes)
    if args.limit > 0:
        tasks = tasks[:args.limit]

    total = len(tasks)
    print(f"\n  Total to scrape: {total} races")
    if total == 0:
        print("  Nothing to do!")
        return

    avg_delay = (DELAY_MIN + DELAY_MAX) / 2
    est_min = total * avg_delay / 60
    print(f"  Estimated time: ~{est_min:.0f} minutes ({est_min/60:.1f} hours)\n")

    stats = {'scraped': 0, 'rows': 0, 'empty': 0, 'errors': 0}
    start_time = time.time()

    for i, race_id in enumerate(tasks):
        pct = (i + 1) / total * 100
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
        eta_min = (total - i - 1) / rate if rate > 0 else 0
        print(f"\r  [{i+1}/{total} {pct:.1f}%] {race_id} | "
              f"{rate:.1f}R/min | ETA {eta_min:.0f}min | "
              f"OK:{stats['scraped']} empty:{stats['empty']} err:{stats['errors']}",
              end='', flush=True)

        try:
            rows = scrape_comment(session, race_id)
            if rows:
                _append_csv(COMMENT_CSV, rows)
                stats['scraped'] += 1
                stats['rows'] += len(rows)
            else:
                stats['empty'] += 1
        except Exception as e:
            stats['errors'] += 1
            if stats['errors'] <= 5:
                print(f"\n  ERROR: {e}")

        delay = DELAY_MIN + np.random.random() * (DELAY_MAX - DELAY_MIN)
        time.sleep(delay)

        if (i + 1) % 100 == 0:
            elapsed_min = (time.time() - start_time) / 60
            print(f"\n  === Checkpoint {i+1}/{total} ({elapsed_min:.0f}min) ===")
            print(f"  OK:{stats['scraped']} empty:{stats['empty']} err:{stats['errors']} rows:{stats['rows']}")

    elapsed_min = (time.time() - start_time) / 60
    print(f"\n\n{'=' * 60}")
    print(f"  SCRAPING COMPLETE ({elapsed_min:.1f} min)")
    print(f"{'=' * 60}")
    print(f"  Scraped: {stats['scraped']} races, {stats['rows']} rows")
    print(f"  Empty: {stats['empty']}, Errors: {stats['errors']}")

    if os.path.exists(COMMENT_CSV):
        size = os.path.getsize(COMMENT_CSV)
        total_rows = len(pd.read_csv(COMMENT_CSV, encoding='utf-8-sig'))
        print(f"\n  CSV: {size/1024/1024:.1f}MB, {total_rows} rows")

    # Notify
    try:
        sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
        from notify_done import send_notification
        send_notification("厩舎コメント取得",
                          f"取得: {stats['scraped']}R/{stats['rows']}行\n"
                          f"空: {stats['empty']}, エラー: {stats['errors']}\n"
                          f"所要: {elapsed_min:.0f}分")
    except Exception:
        pass

    return stats


if __name__ == '__main__':
    main()
