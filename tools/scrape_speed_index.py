#!/usr/bin/env python
"""netkeiba タイム指数スクレイパー (スーパープレミアム限定)

URL: https://race.netkeiba.com/race/speed.html?race_id=XXXX
テーブル: SpeedIndex_Table
指数カラム: sk__max_index, sk__average_index, sk__max_distance_index,
           sk__max_course_index, sk__index1, sk__index2, sk__index3

Usage:
    python tools/scrape_speed_index.py --year 2025 --limit 200
    python tools/scrape_speed_index.py --race_id 202605010811
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
from bs4 import BeautifulSoup

if sys.platform == 'win32' and getattr(sys.stdout, 'encoding', '') != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

HEADERS_HTTP = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DELAY_MIN = 3
DELAY_MAX = 5
RETRY_DELAY = 30
MAX_RETRIES = 3

OUTPUT_CSV = os.path.join(DATA_DIR, 'netkeiba_speed_index.csv')
CSV_HEADER = [
    'race_id', 'umaban', 'horse_name', 'sex_age', 'weight_carry', 'jockey',
    'index_max', 'index_avg5', 'index_dist', 'index_course',
    'index_run1', 'index_run2', 'index_run3',
    'odds', 'popularity',
]


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


def _get(session, url, retry=0):
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code in (400, 403, 429, 500, 502, 503):
            if retry < MAX_RETRIES:
                print(f"  {resp.status_code} - retry {retry+1}")
                time.sleep(RETRY_DELAY)
                return _get(session, url, retry + 1)
            print(f"  FAIL {resp.status_code} after {MAX_RETRIES} retries: {url}")
            return None
        if resp.status_code == 404:
            return None
        resp.encoding = 'EUC-JP'
        return resp
    except Exception:
        if retry < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return _get(session, url, retry + 1)
        return None


def _extract_index(td):
    """Extract numeric index from td cell.
    The real value is in span.Sort_Function_Data_Hidden.
    """
    span = td.find('span', class_='Sort_Function_Data_Hidden')
    if span:
        txt = span.get_text(strip=True)
        try:
            return int(txt)
        except ValueError:
            pass
    # Fallback: parse the visible text
    txt = td.get_text(strip=True)
    # Format: "1111111" → first 4 digits
    m = re.match(r'(\d{3,4})', txt)
    if m:
        return int(m.group(1))
    return 0


def scrape_speed_index(session, race_id):
    """Scrape speed index data for one race."""
    url = f"https://race.netkeiba.com/race/speed.html?race_id={race_id}"
    resp = _get(session, url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', class_=re.compile(r'SpeedIndex'))
    if not table:
        return []

    results = []
    for row in table.find_all('tr'):
        tds = row.find_all('td')
        if len(tds) < 10:
            continue

        # Build a map of class → td
        td_map = {}
        for td in tds:
            for cls in td.get('class', []):
                if cls.startswith('sk__'):
                    td_map[cls] = td

        umaban_td = td_map.get('sk__umaban')
        if not umaban_td:
            continue
        try:
            umaban = int(umaban_td.get_text(strip=True))
        except (ValueError, TypeError):
            continue

        horse_td = td_map.get('sk__horse_name')
        horse_name = ''
        if horse_td:
            span = horse_td.find('span', class_='Sort_Function_Data_Hidden')
            if span:
                horse_name = span.get_text(strip=True)
            else:
                horse_name = horse_td.get_text(strip=True)[:20]

        sex_age = ''
        for td in tds:
            cls = ' '.join(td.get('class', []))
            if 'Txt_C' in cls and not any(x in cls for x in ['sk__', 'Waku']):
                sex_age = td.get_text(strip=True)
                break

        weight_td = td_map.get('sk__load_weight')
        weight_carry = weight_td.get_text(strip=True) if weight_td else ''

        jockey_td = row.find('td', class_='Jockey')
        jockey = jockey_td.get_text(strip=True)[:10] if jockey_td else ''

        idx_max = _extract_index(td_map.get('sk__max_index', BeautifulSoup('<td></td>', 'html.parser').find('td')))
        idx_avg = _extract_index(td_map.get('sk__average_index', BeautifulSoup('<td></td>', 'html.parser').find('td')))
        idx_dist = _extract_index(td_map.get('sk__max_distance_index', BeautifulSoup('<td></td>', 'html.parser').find('td')))
        idx_course = _extract_index(td_map.get('sk__max_course_index', BeautifulSoup('<td></td>', 'html.parser').find('td')))
        idx_r1 = _extract_index(td_map.get('sk__index1', BeautifulSoup('<td></td>', 'html.parser').find('td')))
        idx_r2 = _extract_index(td_map.get('sk__index2', BeautifulSoup('<td></td>', 'html.parser').find('td')))
        idx_r3 = _extract_index(td_map.get('sk__index3', BeautifulSoup('<td></td>', 'html.parser').find('td')))

        odds_td = td_map.get('sk__odds')
        odds = odds_td.get_text(strip=True) if odds_td else ''

        ninki_td = td_map.get('sk__ninki')
        popularity = ninki_td.get_text(strip=True) if ninki_td else ''

        results.append([
            race_id, umaban, horse_name, sex_age, weight_carry, jockey,
            idx_max, idx_avg, idx_dist, idx_course,
            idx_r1, idx_r2, idx_r3,
            odds, popularity,
        ])

    return results


def _append_csv(rows):
    write_header = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    with open(OUTPUT_CSV, 'a', encoding='utf-8-sig', newline='') as f:
        if write_header:
            f.write(','.join(CSV_HEADER) + '\n')
        for row in rows:
            f.write(','.join(str(v).replace(',', '；') for v in row) + '\n')


def _target_to_netkeiba(target_rid):
    """Convert TARGET JV race_id (10-digit) to netkeiba format (12-digit).

    TARGET: CC(2)+YY(2)+K(1)+N(1)+RR(2)+UU(2)
    netkeiba: YYYY(4)+CC(2)+KK(2)+NN(2)+RR(2)
    N can be hex: A=10, B=11, C=12
    """
    rid = str(target_rid).zfill(10)
    cc = rid[0:2]
    yy = rid[2:4]
    k = rid[4:5]
    n = rid[5:6]
    rr = rid[6:8]
    n_map = {'A': '10', 'B': '11', 'C': '12'}
    n_dec = n_map.get(n, n.zfill(2))
    return f"20{yy}{cc}{k.zfill(2)}{n_dec}{rr}"


def get_race_ids(year):
    csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
    df = pd.read_csv(csv_path, encoding='utf-8-sig', usecols=['year', 'race_id'], dtype=str, low_memory=False)
    yr2 = year % 100
    df['year_int'] = pd.to_numeric(df['year'], errors='coerce')
    df = df[df['year_int'] == yr2]
    nk_ids = set()
    for rid in df['race_id'].dropna().unique():
        nk_ids.add(_target_to_netkeiba(rid))
    return sorted(nk_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--race_id', type=str, default='')
    args = parser.parse_args()

    print("=" * 60)
    print(f"  netkeiba Speed Index Scraper")
    print("=" * 60)

    session = _load_session()
    if session is None:
        return

    if args.race_id:
        race_ids = [args.race_id]
    else:
        race_ids = get_race_ids(args.year)
        print(f"  Year {args.year}: {len(race_ids)} races")

    # Skip already scraped
    existing = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            edf = pd.read_csv(OUTPUT_CSV, encoding='utf-8-sig', usecols=['race_id'], dtype=str)
            existing = set(edf['race_id'].unique())
        except Exception:
            pass

    race_ids = [rid for rid in race_ids if str(rid) not in existing]
    if args.limit > 0:
        race_ids = race_ids[:args.limit]
    print(f"  New to scrape: {len(race_ids)}")

    total = len(race_ids)
    stats = {'races': 0, 'rows': 0, 'errors': 0}

    for i, race_id in enumerate(race_ids):
        print(f"\r  [{i+1}/{total} {(i+1)/total*100:.0f}%] {race_id}", end='', flush=True)

        try:
            rows = scrape_speed_index(session, race_id)
            if rows:
                _append_csv(rows)
                stats['races'] += 1
                stats['rows'] += len(rows)
        except Exception:
            stats['errors'] += 1

        delay = DELAY_MIN + np.random.random() * (DELAY_MAX - DELAY_MIN)
        time.sleep(delay)

        if (i + 1) % 50 == 0:
            print(f"\n  Progress: {stats['races']}R / {stats['rows']} rows / {stats['errors']} errors")

    print(f"\n\n{'=' * 60}")
    print(f"  COMPLETE: {stats['races']} races, {stats['rows']} rows, {stats['errors']} errors")
    total_rows = 0
    if os.path.exists(OUTPUT_CSV):
        total_rows = len(pd.read_csv(OUTPUT_CSV, encoding='utf-8-sig'))
        print(f"  Output: {OUTPUT_CSV} ({total_rows} total rows)")
    print("=" * 60)

    try:
        from notify import send_discord
        send_discord("Speed Index取得完了",
                     f"{stats['races']}R / {stats['rows']}行 / エラー{stats['errors']}\n累計: {total_rows}行",
                     color="green" if stats['errors'] == 0 else "yellow")
    except Exception:
        pass


if __name__ == '__main__':
    from tools.scraper_guard import check_scraping_allowed; check_scraping_allowed()
    main()
