#!/usr/bin/env python
"""netkeiba Premium 一括データ取得スクリプト

Cookie認証で以下を取得:
1. 調教タイム（oikiriページ）→ data/netkeiba_training_times.csv
2. 厩舎コメント → data/netkeiba_stable_comments.csv
3. レース傾向（data_listページ）→ data/netkeiba_race_tendency.csv

Usage:
    python tools/scrape_premium_data.py                    # 2025年全レース
    python tools/scrape_premium_data.py --year 2024        # 指定年
    python tools/scrape_premium_data.py --year 2025 --month 1  # 指定月
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
from datetime import datetime
from bs4 import BeautifulSoup

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DELAY_MIN = 3
DELAY_MAX = 5
RETRY_DELAY = 30
MAX_RETRIES = 3

# Output files
TRAINING_CSV = os.path.join(DATA_DIR, 'netkeiba_training_times.csv')
COMMENT_CSV = os.path.join(DATA_DIR, 'netkeiba_stable_comments.csv')
TENDENCY_CSV = os.path.join(DATA_DIR, 'netkeiba_race_tendency.csv')

TRAINING_HEADER = [
    'race_id', 'race_date', 'umaban', 'horse_name', 'course', 'condition',
    'rider', 'time_6f', 'time_5f', 'time_4f', 'time_3f', 'time_1f',
    'intensity', 'rank', 'evaluation', 'training_date',
]
COMMENT_HEADER = ['race_id', 'race_date', 'umaban', 'horse_name', 'comment', 'score']
TENDENCY_HEADER = ['race_id', 'race_date', 'category', 'value']


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
                print(f"  {resp.status_code} - waiting {RETRY_DELAY}s (retry {retry+1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
                return _get(session, url, retry + 1)
            return None
        resp.encoding = 'EUC-JP'
        return resp
    except Exception as e:
        if retry < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return _get(session, url, retry + 1)
        return None


def _append_csv(path, rows, header):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', encoding='utf-8-sig', newline='') as f:
        if write_header:
            f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(v).replace(',', '；') for v in row) + '\n')


def get_race_ids_for_year(year):
    """jra_races_full.csvからnetkeibaフォーマットのレースIDを生成。

    TARGET race_id (10桁): CC(2)+YY(2)+K(1)+N(1)+RR(2)+UU(2)
    netkeiba race_id (12桁): YYYY+CC(2)+K(1)+N(1)+RR(2)+(00)

    変換: CC+YY+K+N+RR → 20YY+CC+K+N+RR
    """
    csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
    df = pd.read_csv(csv_path, encoding='utf-8-sig',
                      usecols=['year', 'race_id', 'month'],
                      dtype=str, low_memory=False)
    df['year_int'] = pd.to_numeric(df['year'], errors='coerce')
    yr2 = year % 100
    df = df[df['year_int'] == yr2]

    # Convert TARGET race_id to netkeiba format
    # TARGET: CC(2)+YY(2)+K(1)+N(1)+RR(2)+UU(2) → 10 digits
    # netkeiba: 20YY+CC(2)+0K(1)+0N(1)+RR(2) → 12 digits
    nk_ids = set()
    for rid in df['race_id'].dropna().unique():
        rid = str(rid).zfill(10)
        cc = rid[0:2]   # course code (01-10)
        yy = rid[2:4]   # year (2 digit)
        k = rid[4:5]    # kai
        n = rid[5:6]    # nichi
        rr = rid[6:8]   # race_num
        # netkeiba: YYYY(4) + CC(2) + KK(2) + NN(2) + RR(2) = 12 digits
        nk_id = f"20{yy}{cc}{k.zfill(2)}{n.zfill(2)}{rr}"
        nk_ids.add(nk_id)

    print(f"  Year {year}: {len(nk_ids)} unique netkeiba race_ids")
    return sorted(nk_ids)


def scrape_oikiri(session, race_id, race_date=''):
    """Scrape training times from oikiri page."""
    url = f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}"
    resp = _get(session, url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', class_=re.compile(r'OikiriTable'))
    if not table:
        return []

    time_lists = soup.find_all('ul', class_='TrainingTimeDataList')
    rows_data = table.find_all('tr', class_=re.compile(r'OikiriDataHead'))

    results = []
    for idx, row in enumerate(rows_data):
        td_umaban = row.select_one('td.Umaban')
        if not td_umaban:
            continue
        try:
            umaban = int(td_umaban.get_text(strip=True))
        except (ValueError, TypeError):
            continue

        # Horse name
        horse_div = row.select_one('div.Horse_Name')
        horse_name = horse_div.get_text(strip=True) if horse_div else ''

        full_text = row.get_text(strip=True)

        # Course
        course = ''
        if re.search(r'[美栗]坂', full_text): course = '坂路'
        elif re.search(r'[美栗][ＷW]', full_text): course = 'CW'
        elif re.search(r'ポリ', full_text): course = 'ポリトラック'

        # Condition (馬場状態)
        cond = ''
        for c in ['良', '稍', '重', '不']:
            if c in full_text:
                cond = c
                break

        # Rider
        rider = ''
        td_rider = row.select_one('td.Jockey') or row.select_one('td.Rider')
        if td_rider:
            rider = td_rider.get_text(strip=True)[:10]

        # Intensity
        intensity = ''
        for pat, name in [('一杯', '一杯'), ('強め', '強め'), ('馬なり', '馬なり'), ('末強め', '末強め')]:
            if pat in full_text:
                intensity = name
                break

        # Rank
        rank = ''
        td_critic = row.select_one('td.Training_Critic')
        evaluation = td_critic.get_text(strip=True) if td_critic else ''
        for td in row.find_all('td'):
            for cls in td.get('class', []):
                if cls.startswith('Rank_'):
                    rank_text = cls.replace('Rank_', '')
                    if rank_text in ('A', 'B', 'C', 'D'):
                        rank = rank_text
                    elif any(x in rank_text for x in ['好調教', '抜群', '絶好']):
                        rank = 'A'
                    elif any(x in rank_text for x in ['上々', '乗込入念', '良化']):
                        rank = 'B'
                    else:
                        rank = 'C'
                    break
            if rank:
                break

        # Training date
        training_date = ''
        m = re.search(r'(\d{4})/(\d{2})/(\d{2})', full_text)
        if m:
            training_date = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"

        # Times from TrainingTimeDataList
        t6f = t5f = t4f = t3f = t1f = 0.0
        if idx < len(time_lists):
            lis = time_lists[idx].find_all('li')
            times = []
            for li in lis:
                t = li.get_text(strip=True)
                m = re.match(r'(\d+\.?\d*)\(', t)
                times.append(float(m.group(1)) if m else 0.0)
            if len(times) >= 5:
                if times[0] > 0:  # CW: 6F,5F,4F,3F,1F
                    t6f, t5f, t4f, t3f, t1f = times[0], times[1], times[2], times[3], times[4]
                else:  # 坂路: -,4F,3F,2F,1F
                    t4f, t3f, t1f = times[1], times[2], times[4]

        results.append([
            race_id, race_date, umaban, horse_name, course, cond,
            rider, t6f, t5f, t4f, t3f, t1f,
            intensity, rank, evaluation, training_date,
        ])

    return results


def scrape_comment(session, race_id, race_date=''):
    """Scrape stable comments."""
    url = f"https://race.netkeiba.com/race/comment.html?race_id={race_id}"
    resp = _get(session, url)
    if resp is None:
        return []
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', class_=re.compile(r'Stable_Comment|Comment_Table'))
    if not table:
        return []

    from scrape_training import _score_comment

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
        results.append([race_id, race_date, umaban, horse_name, comment, score])

    return results


def scrape_tendency(session, race_id, race_date=''):
    """Scrape race tendency data from data_list page."""
    url = f"https://race.netkeiba.com/race/data_list.html?race_id={race_id}"
    resp = _get(session, url)
    if resp is None:
        return []
    soup = BeautifulSoup(resp.text, 'html.parser')

    results = []
    for table in soup.find_all('table', class_='RaceCommon_Table'):
        tds = table.find_all(['th', 'td'])
        if len(tds) >= 2:
            category = tds[0].get_text(strip=True)[:30]
            value = tds[1].get_text(strip=True)[:60]
            results.append([race_id, race_date, category, value])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--month', type=int, default=0, help='0=all months')
    parser.add_argument('--limit', type=int, default=0, help='Max races (0=unlimited)')
    args = parser.parse_args()

    print("=" * 60)
    print(f"  netkeiba Premium Data Scraper")
    print(f"  Year: {args.year}, Month: {args.month or 'all'}")
    print("=" * 60)

    session = _load_session()
    if session is None:
        return

    # Get race IDs
    race_ids = get_race_ids_for_year(args.year)
    if args.month > 0:
        csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
        df = pd.read_csv(csv_path, encoding='utf-8-sig',
                          usecols=['year', 'month', 'race_id'], dtype=str, low_memory=False)
        yr2 = args.year % 100
        df['year_int'] = pd.to_numeric(df['year'], errors='coerce')
        df['month_int'] = pd.to_numeric(df['month'], errors='coerce')
        month_rids = set()
        for rid in df[(df['year_int'] == yr2) & (df['month_int'] == args.month)]['race_id'].dropna().unique():
            rid = str(rid).zfill(10)
            nk_id = f"20{rid[2:4]}{rid[0:2]}{rid[4:5].zfill(2)}{rid[5:6].zfill(2)}{rid[6:8]}"
            month_rids.add(nk_id)
        race_ids = sorted(set(race_ids) & month_rids)
        print(f"  Filtered to month {args.month}: {len(race_ids)} races")

    # Deduplicate: only 12-digit race IDs
    race_ids = sorted(set(rid for rid in race_ids if len(str(rid)) >= 10))

    if args.limit > 0:
        race_ids = race_ids[:args.limit]

    # Check what we already have
    existing_training = set()
    if os.path.exists(TRAINING_CSV):
        try:
            edf = pd.read_csv(TRAINING_CSV, encoding='utf-8-sig', usecols=['race_id'], dtype=str)
            existing_training = set(edf['race_id'].unique())
        except Exception:
            pass
    existing_comment = set()
    if os.path.exists(COMMENT_CSV):
        try:
            edf = pd.read_csv(COMMENT_CSV, encoding='utf-8-sig', usecols=['race_id'], dtype=str)
            existing_comment = set(edf['race_id'].unique())
        except Exception:
            pass

    new_training_ids = [rid for rid in race_ids if str(rid) not in existing_training]
    new_comment_ids = [rid for rid in race_ids if str(rid) not in existing_comment]

    print(f"  Total race IDs: {len(race_ids)}")
    print(f"  New for training: {len(new_training_ids)}")
    print(f"  New for comments: {len(new_comment_ids)}")

    # Scrape all new race IDs
    all_ids = sorted(set(new_training_ids + new_comment_ids))
    total = len(all_ids)
    stats = {'training_races': 0, 'training_rows': 0, 'comment_races': 0, 'comment_rows': 0,
             'tendency_races': 0, 'errors': 0}

    for i, race_id in enumerate(all_ids):
        rid = str(race_id)
        pct = (i + 1) / total * 100
        print(f"\r  [{i+1}/{total} {pct:.0f}%] {rid}", end='', flush=True)

        try:
            # Training
            if rid not in existing_training:
                rows = scrape_oikiri(session, rid)
                if rows:
                    _append_csv(TRAINING_CSV, rows, TRAINING_HEADER)
                    stats['training_races'] += 1
                    stats['training_rows'] += len(rows)

            # Comments
            if rid not in existing_comment:
                rows = scrape_comment(session, rid)
                if rows:
                    _append_csv(COMMENT_CSV, rows, COMMENT_HEADER)
                    stats['comment_races'] += 1
                    stats['comment_rows'] += len(rows)

            # Tendency (only first 100 races to avoid excessive requests)
            if i < 100:
                rows = scrape_tendency(session, rid)
                if rows:
                    _append_csv(TENDENCY_CSV, rows, TENDENCY_HEADER)
                    stats['tendency_races'] += 1

        except Exception as e:
            stats['errors'] += 1

        # Rate limiting
        delay = DELAY_MIN + np.random.random() * (DELAY_MAX - DELAY_MIN)
        time.sleep(delay)

        # Progress every 50 races
        if (i + 1) % 50 == 0:
            print(f"\n  Progress: training={stats['training_races']}R/{stats['training_rows']}rows, "
                  f"comments={stats['comment_races']}R/{stats['comment_rows']}rows, errors={stats['errors']}")

    # Summary
    print(f"\n\n{'=' * 60}")
    print(f"  SCRAPING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Training: {stats['training_races']} races, {stats['training_rows']} rows")
    print(f"  Comments: {stats['comment_races']} races, {stats['comment_rows']} rows")
    print(f"  Tendency: {stats['tendency_races']} races")
    print(f"  Errors: {stats['errors']}")

    for path, name in [(TRAINING_CSV, 'Training'), (COMMENT_CSV, 'Comments'), (TENDENCY_CSV, 'Tendency')]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            try:
                n = len(pd.read_csv(path, encoding='utf-8-sig'))
            except Exception:
                n = '?'
            print(f"  {name}: {path} ({size/1024:.0f}KB, {n} rows)")

    return stats


if __name__ == '__main__':
    main()
