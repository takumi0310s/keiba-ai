#!/usr/bin/env python
"""netkeiba マスターコース 3分解指数 + 馬場指数 + トラックバイアス スクレイパー

db.netkeiba.com/race/{race_id}/ の結果テーブルから:
  - タイム指数M (master speed index)
  - スタート指数 (start index)
  - 追走指数 (chase index)
  - 上がり指数 (agari/finish index)
  - 馬場指数 (track condition index)
  - トラックバイアス (inner/outer bias)
  - レースラップ (race-level lap times)

Usage:
    python tools/scrape_master_index.py --year 2025 --limit 100
    python tools/scrape_master_index.py --year 2020 --all_years
    python tools/scrape_master_index.py --race_id 202605010811
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

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
sys.path.insert(0, BASE_DIR)

HEADERS_HTTP = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DELAY_MIN = 15  # 15秒間隔（並行スクレイピング考慮）
DELAY_MAX = 18
RETRY_DELAY = 60
MAX_RETRIES = 3

# Output files
MASTER_INDEX_CSV = os.path.join(DATA_DIR, 'netkeiba_master_index.csv')
TRACK_BIAS_CSV = os.path.join(DATA_DIR, 'netkeiba_track_bias.csv')
RACE_LAP_CSV = os.path.join(DATA_DIR, 'netkeiba_race_lap.csv')

MASTER_INDEX_HEADER = [
    'race_id', 'umaban', 'horse_name', 'finish_order',
    'time_index', 'master_index', 'start_index', 'chase_index', 'agari_index',
]

TRACK_BIAS_HEADER = [
    'race_id', 'track_index', 'track_bias_text', 'track_comment',
]

RACE_LAP_HEADER = [
    'race_id', 'lap_times', 'pace_first_half', 'pace_second_half',
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
        resp = session.get(url, timeout=20)
        if resp.status_code in (400, 403, 429, 500, 502, 503):
            if retry < MAX_RETRIES:
                wait = RETRY_DELAY * (retry + 1)
                print(f"\n  {resp.status_code} - waiting {wait}s (retry {retry+1}/{MAX_RETRIES})")
                time.sleep(wait)
                return _get(session, url, retry + 1)
            print(f"\n  FAIL {resp.status_code} after {MAX_RETRIES} retries: {url}")
            return None
        if resp.status_code == 404:
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


def _safe_int(text):
    """Extract integer from text, return '' if not found."""
    if not text:
        return ''
    text = text.strip()
    m = re.search(r'-?\d+', text)
    return int(m.group()) if m else ''


def _safe_float(text):
    """Extract float from text, return '' if not found."""
    if not text:
        return ''
    text = text.strip()
    m = re.search(r'-?\d+\.?\d*', text)
    return float(m.group()) if m else ''


def scrape_result_page(session, race_id):
    """db.netkeiba.com結果ページから3分解指数+馬場情報+ラップを取得。

    Returns: (master_rows, bias_row, lap_row) or (None, None, None)
    """
    url = f'https://db.netkeiba.com/race/{race_id}/'
    resp = _get(session, url)
    if not resp:
        return None, None, None

    soup = BeautifulSoup(resp.text, 'html.parser')

    # === 1. Main result table: race_table_01 ===
    master_rows = []
    table = soup.find('table', class_='race_table_01')
    if not table:
        return None, None, None

    # Get header to find column indices
    thead = table.find('tr')
    if not thead:
        return None, None, None

    # Parse data rows (skip header row)
    data_rows = table.find_all('tr')
    for tr in data_rows[1:]:
        tds = tr.find_all('td')
        if len(tds) < 14:
            continue

        finish_text = tds[0].get_text(strip=True)
        # Skip non-numeric finish (取消, 除外, 中止)
        if not re.match(r'^\d+$', finish_text):
            continue

        finish_order = int(finish_text)
        umaban_text = tds[2].get_text(strip=True)
        umaban = _safe_int(umaban_text)
        horse_name = tds[3].get_text(strip=True)

        # Columns 9-13: タイム指数, タイム指数M, スタート指数, 追走指数, 上がり指数
        time_index = _safe_int(tds[9].get_text(strip=True)) if len(tds) > 9 else ''
        master_index = _safe_int(tds[10].get_text(strip=True)) if len(tds) > 10 else ''
        start_index = _safe_int(tds[11].get_text(strip=True)) if len(tds) > 11 else ''
        chase_index = _safe_int(tds[12].get_text(strip=True)) if len(tds) > 12 else ''
        agari_index = _safe_int(tds[13].get_text(strip=True)) if len(tds) > 13 else ''

        # Skip if all indices are empty (non-master page)
        if master_index == '' and start_index == '' and chase_index == '' and agari_index == '':
            continue

        master_rows.append([
            race_id, umaban, horse_name, finish_order,
            time_index, master_index, start_index, chase_index, agari_index,
        ])

    # === 2. Track bias / 馬場情報 ===
    track_index = ''
    track_bias_text = ''
    track_comment = ''

    # Find 馬場情報 tables
    for tbl in soup.find_all('table', class_='result_table_02'):
        summary = tbl.get('summary', '')
        text = tbl.get_text()
        if '馬場指数' in text or '馬場' in summary:
            for tr in tbl.find_all('tr'):
                th = tr.find('th')
                td = tr.find('td')
                if not th or not td:
                    continue
                label = th.get_text(strip=True)
                value = td.get_text(strip=True)
                if '馬場指数' in label:
                    track_index = _safe_int(value)
                elif 'トラックバイアス' in label:
                    track_bias_text = value
                elif '馬場コメント' in label:
                    track_comment = value

    # Also check TrackBiasWrap div
    bias_div = soup.find('div', class_='TrackBiasWrap')
    if bias_div and not track_bias_text:
        track_bias_text = bias_div.get_text(strip=True)

    bias_row = [race_id, track_index, track_bias_text, track_comment]

    # === 3. Race lap times ===
    lap_times_str = ''
    pace_first = ''
    pace_second = ''

    for tbl in soup.find_all('table', class_='result_table_02'):
        text = tbl.get_text()
        if 'ラップ' in text:
            for tr in tbl.find_all('tr'):
                th = tr.find('th')
                td = tr.find('td')
                if not th or not td:
                    continue
                label = th.get_text(strip=True)
                value = td.get_text(strip=True)
                if 'ラップ' in label and 'ペース' not in label:
                    lap_times_str = value
                elif 'ペース' in label:
                    # Parse pace from parentheses: "(35.1-36.2)" format
                    m = re.search(r'\((\d+\.\d+)\s*[-−]\s*(\d+\.\d+)\)', value)
                    if m:
                        pace_first = float(m.group(1))
                        pace_second = float(m.group(2))

    lap_row = [race_id, lap_times_str, pace_first, pace_second]

    return master_rows, bias_row, lap_row


def _target_to_netkeiba(target_rid):
    """Convert TARGET JV race_id (10-digit) to netkeiba format (12-digit)."""
    rid = str(target_rid).zfill(10)
    cc = rid[0:2]
    yy = rid[2:4]
    k = rid[4:5]
    n = rid[5:6]
    rr = rid[6:8]
    n_map = {'A': '10', 'B': '11', 'C': '12'}
    n_dec = n_map.get(n, n.zfill(2))
    return f"20{yy}{cc}{k.zfill(2)}{n_dec}{rr}"


def get_race_ids_for_year(year):
    """jra_races_full.csvからnetkeibaフォーマットのレースIDを生成。"""
    csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
    df = pd.read_csv(csv_path, encoding='utf-8-sig',
                     usecols=['year', 'race_id', 'month'],
                     dtype=str, low_memory=False)
    df['year_int'] = pd.to_numeric(df['year'], errors='coerce')
    yr2 = year % 100
    df = df[df['year_int'] == yr2]

    nk_ids = set()
    for rid in df['race_id'].dropna().unique():
        nk_ids.add(_target_to_netkeiba(rid))

    print(f"  Year {year}: {len(nk_ids)} unique netkeiba race_ids")
    return sorted(nk_ids)


def main():
    parser = argparse.ArgumentParser(description='Scrape master 3-component index + track bias')
    parser.add_argument('--year', type=int, default=2025, help='Year to scrape')
    parser.add_argument('--all_years', action='store_true', help='Scrape 2020-2026')
    parser.add_argument('--limit', type=int, default=0, help='Limit races (0=all)')
    parser.add_argument('--race_id', type=str, help='Single race_id to test')
    args = parser.parse_args()

    session = _load_session()
    if not session:
        return

    # Single race test mode
    if args.race_id:
        print(f"Testing race_id: {args.race_id}")
        master_rows, bias_row, lap_row = scrape_result_page(session, args.race_id)
        if master_rows:
            print(f"\n=== Master Index ({len(master_rows)} horses) ===")
            for r in master_rows[:5]:
                print(f"  #{r[1]} {r[2]}: 着{r[3]} TI={r[4]} MI={r[5]} S={r[6]} C={r[7]} A={r[8]}")
        else:
            print("No master index data found")
        if bias_row:
            print(f"\n=== Track Bias ===")
            print(f"  馬場指数: {bias_row[1]}")
            print(f"  バイアス: {bias_row[2]}")
            print(f"  コメント: {bias_row[3]}")
        if lap_row:
            print(f"\n=== Lap Times ===")
            print(f"  ラップ: {lap_row[1]}")
            print(f"  ペース: {lap_row[2]} - {lap_row[3]}")
        return

    # Build race_id list
    years = list(range(2020, 2027)) if args.all_years else [args.year]
    all_ids = []
    for y in years:
        all_ids.extend(get_race_ids_for_year(y))

    if args.limit > 0:
        all_ids = all_ids[:args.limit]

    # Load existing to skip (check ONLY MASTER_INDEX_CSV - bug fix 4/28)
    # Old code: checked all 3 CSVs, but TRACK_BIAS/RACE_LAP had 2020-2025 race_ids
    # while MASTER_INDEX only had 2023-2025, causing 2020-2022 to be wrongly skipped
    existing_ids = set()
    for csv_path in [MASTER_INDEX_CSV]:  # bug fix: master_index only
        if os.path.exists(csv_path):
            try:
                df_exist = pd.read_csv(csv_path, encoding='utf-8-sig',
                                       usecols=['race_id'], dtype=str, low_memory=False)
                existing_ids.update(df_exist['race_id'].unique())
            except Exception:
                pass

    new_ids = [rid for rid in all_ids if rid not in existing_ids]
    print(f"\n  Total race IDs: {len(all_ids)}")
    print(f"  Already scraped: {len(existing_ids)}")
    print(f"  New to scrape: {len(new_ids)}")

    if not new_ids:
        print("  Nothing to scrape!")
        return

    total = len(new_ids)
    stats = {'master_races': 0, 'master_rows': 0, 'bias_races': 0,
             'lap_races': 0, 'errors': 0}
    consecutive_empty = 0

    from tools.scraper_guard import check_scraping_allowed
    for i, race_id in enumerate(new_ids):
        check_scraping_allowed()  # Fri22:00〜Mon06:00は自動停止→再開
        pct = (i + 1) / total * 100
        print(f"\r  [{i+1}/{total} {pct:.0f}%] {race_id}", end='', flush=True)

        try:
            master_rows, bias_row, lap_row = scrape_result_page(session, race_id)

            got_any = False

            if master_rows:
                _append_csv(MASTER_INDEX_CSV, master_rows, MASTER_INDEX_HEADER)
                stats['master_races'] += 1
                stats['master_rows'] += len(master_rows)
                got_any = True

            if bias_row and (bias_row[1] != '' or bias_row[2] != ''):
                _append_csv(TRACK_BIAS_CSV, [bias_row], TRACK_BIAS_HEADER)
                stats['bias_races'] += 1
                got_any = True

            if lap_row and lap_row[1] != '':
                _append_csv(RACE_LAP_CSV, [lap_row], RACE_LAP_HEADER)
                stats['lap_races'] += 1
                got_any = True

            if got_any:
                consecutive_empty = 0
            else:
                consecutive_empty += 1

        except Exception as e:
            stats['errors'] += 1
            consecutive_empty += 1

        # Early abort if server seems down
        if consecutive_empty >= 15 and i >= 15:
            print(f"\n  ABORT: {consecutive_empty} consecutive empty. Server may be blocking.")
            break

        # Rate limiting - 15 second minimum for concurrent scraping safety
        delay = DELAY_MIN + np.random.random() * (DELAY_MAX - DELAY_MIN)
        time.sleep(delay)

        # Progress log every 50
        if (i + 1) % 50 == 0:
            print(f"\n  Progress: master={stats['master_races']}R/{stats['master_rows']}rows, "
                  f"bias={stats['bias_races']}R, lap={stats['lap_races']}R, err={stats['errors']}")

    # Summary
    print(f"\n\n{'=' * 60}")
    print(f"  SCRAPING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Master Index: {stats['master_races']} races, {stats['master_rows']} rows")
    print(f"  Track Bias: {stats['bias_races']} races")
    print(f"  Race Laps: {stats['lap_races']} races")
    print(f"  Errors: {stats['errors']}")

    for path, name in [(MASTER_INDEX_CSV, 'Master Index'),
                       (TRACK_BIAS_CSV, 'Track Bias'),
                       (RACE_LAP_CSV, 'Race Lap')]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            try:
                n = len(pd.read_csv(path, encoding='utf-8-sig'))
            except Exception:
                n = '?'
            print(f"  {name}: {path} ({size/1024:.0f}KB, {n} rows)")

    try:
        from notify import send_discord
        send_discord("Master Index取得完了",
                     f"3分解指数: {stats['master_races']}R/{stats['master_rows']}行\n"
                     f"バイアス: {stats['bias_races']}R\nラップ: {stats['lap_races']}R",
                     color="green" if stats['errors'] == 0 else "yellow")
    except Exception:
        pass


if __name__ == '__main__':
    from tools.scraper_guard import check_scraping_allowed; check_scraping_allowed()
    main()
