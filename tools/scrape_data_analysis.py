#!/usr/bin/env python
"""netkeiba データ分析43種スクレイパー（レース当日〜5日後限定）

枠番別・距離別・馬場別・脚質別の着度数データを取得。
※ netkeibaの仕様上、レース当日およびレース後5日間のみ公開。

Usage:
    python tools/scrape_data_analysis.py --race_id 202609010901
    python tools/scrape_data_analysis.py --date 20260329
    python tools/scrape_data_analysis.py --today
"""
import pandas as pd
import requests
import re
import os
import sys
import io
import json
import time
import argparse
from datetime import datetime, date
from bs4 import BeautifulSoup

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_CSV = os.path.join(DATA_DIR, 'netkeiba_data_analysis.csv')

HEADERS_HTTP = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DELAY_MIN = 3
DELAY_MAX = 5


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
        print("ERROR: NETKEIBA_COOKIE not set")
        return None
    session = requests.Session()
    session.headers.update(HEADERS_HTTP)
    for part in cookie_str.split(';'):
        part = part.strip()
        if '=' in part:
            k, v = part.split('=', 1)
            session.cookies.set(k.strip(), v.strip())
    return session


def scrape_data_analysis(session, race_id):
    """Scrape data analysis tables from data_list.html.

    Returns list of dicts with analysis data, or empty list if not available.
    """
    url = f"https://race.netkeiba.com/race/data_list.html?race_id={race_id}"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return []
        resp.encoding = 'euc-jp'
    except Exception as e:
        print(f"  Error fetching {race_id}: {e}")
        return []

    soup = BeautifulSoup(resp.text, 'lxml')

    # Check if data is available (premium wall check)
    premium_btn = soup.find('a', class_='Premium_Regist_Btn')
    if premium_btn:
        # Data tables might still be present for premium users
        pass

    results = []

    # Parse pickup data tables (RaceCommon_Table)
    # Each table has 1 row with 2 cells: [category, value]
    for table in soup.find_all('table', class_='RaceCommon_Table'):
        for tr in table.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if len(cells) >= 2:
                results.append({
                    'race_id': race_id,
                    'category': cells[0],
                    'value': cells[1],
                })

    # Also parse race metadata from RaceData divs
    race_data1 = soup.find('div', class_='RaceData01')
    race_data2 = soup.find('div', class_='RaceData02')
    if race_data1:
        results.append({
            'race_id': race_id,
            'category': 'race_info',
            'value': race_data1.get_text(strip=True),
        })
    if race_data2:
        results.append({
            'race_id': race_id,
            'category': 'race_detail',
            'value': race_data2.get_text(strip=True),
        })

    return results


def get_today_race_ids(session):
    """Get race IDs for today's races from netkeiba."""
    today_str = date.today().strftime('%Y%m%d')
    url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={today_str}"
    try:
        resp = session.get(url, timeout=15)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'lxml')
        race_ids = []
        for a in soup.find_all('a', href=True):
            m = re.search(r'race_id=(\d{12})', a['href'])
            if m:
                race_ids.append(m.group(1))
        return sorted(set(race_ids))
    except Exception as e:
        print(f"Error getting today's races: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Scrape netkeiba data analysis tables')
    parser.add_argument('--race_id', help='Single race ID to scrape')
    parser.add_argument('--date', help='Date YYYYMMDD to scrape all races')
    parser.add_argument('--today', action='store_true', help='Scrape today\'s races')
    args = parser.parse_args()

    session = _load_session()
    if not session:
        return

    race_ids = []
    if args.race_id:
        race_ids = [args.race_id]
    elif args.today:
        race_ids = get_today_race_ids(session)
        print(f"Found {len(race_ids)} races for today")
    elif args.date:
        url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={args.date}"
        try:
            resp = session.get(url, timeout=15)
            resp.encoding = 'utf-8'
            soup = BeautifulSoup(resp.text, 'lxml')
            for a in soup.find_all('a', href=True):
                m = re.search(r'race_id=(\d{12})', a['href'])
                if m:
                    race_ids.append(m.group(1))
            race_ids = sorted(set(race_ids))
            print(f"Found {len(race_ids)} races for {args.date}")
        except Exception as e:
            print(f"Error: {e}")
            return
    else:
        parser.print_help()
        return

    if not race_ids:
        print("No race IDs found")
        return

    # Load existing data to skip
    existing = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df = pd.read_csv(OUTPUT_CSV)
            existing = set(df['race_id'].astype(str).unique())
        except Exception:
            pass

    all_results = []
    for i, rid in enumerate(race_ids):
        if str(rid) in existing:
            print(f"  [{i+1}/{len(race_ids)}] {rid} - already scraped, skip")
            continue

        print(f"  [{i+1}/{len(race_ids)}] {rid}", end='')
        rows = scrape_data_analysis(session, rid)
        print(f" → {len(rows)} rows")
        all_results.extend(rows)

        if i < len(race_ids) - 1:
            import random
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    if all_results:
        df_new = pd.DataFrame(all_results)
        if os.path.exists(OUTPUT_CSV):
            df_existing = pd.read_csv(OUTPUT_CSV)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"\nSaved {len(all_results)} new rows to {OUTPUT_CSV}")
        print(f"Total: {len(df_combined)} rows")
    else:
        print("\nNo new data found (data analysis may not be available for these races)")


if __name__ == '__main__':
    main()
