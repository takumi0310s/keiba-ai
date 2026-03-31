#!/usr/bin/env python
"""NAR全競馬場スクレイパー v6 (requests + BeautifulSoup)
8-18秒ランダム間隔、403リトライ、進捗保存、再開可能。
"""
import os
import sys
import json
import csv
import time
import random
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from bs4 import BeautifulSoup
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_CSV = os.path.join(DATA_DIR, 'nar_all_races.csv')
PROGRESS_FILE = os.path.join(DATA_DIR, 'nar_scrape_progress.json')

NAR_TRACKS = {
    30: '門別', 35: '盛岡', 36: '水沢', 42: '浦和', 43: '船橋',
    44: '大井', 45: '川崎', 46: '金沢', 47: '笠松', 48: '名古屋',
    50: '園田', 51: '姫路', 54: '高知', 55: '佐賀', 65: '帯広',
}

# 2024→2015 (2025 already done)
YEARS = list(range(2024, 2014, -1))

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
]

CSV_HEADER = [
    'race_id', 'race_name', 'race_date', 'course', 'course_code',
    'distance', 'surface', 'condition', 'weather', 'class_info',
    'num_horses', 'finish', 'bracket', 'horse_num', 'horse_name',
    'sex_age', 'weight_carry', 'jockey_name', 'time', 'margin',
    'pass_order', 'last3f', 'odds', 'pop_rank', 'trainer_name',
    'horse_weight', 'horse_weight_change',
    'tansho_payout', 'fukusho_payout', 'umaren_payout',
    'wide_payout', 'trio_payout', 'tierce_payout',
]


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'completed_dates': [], 'failed_dates': [], 'total_races': 0, 'total_rows': 0}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def append_to_csv(results, csv_path):
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow(r)


_executor = ThreadPoolExecutor(max_workers=2)


def create_session():
    s = requests.Session()
    s.headers.update({
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'Referer': 'https://nar.netkeiba.com/',
        'Connection': 'keep-alive',
    })
    return s


def safe_get(session, url, hard_timeout=30):
    """GET with a hard thread-based timeout."""
    future = _executor.submit(session.get, url, timeout=(10, 20))
    try:
        return future.result(timeout=hard_timeout)
    except FutureTimeout:
        return None
    except Exception:
        return None


def race_sleep():
    """8-18秒ランダム間隔"""
    time.sleep(random.uniform(8, 18))


def nav_sleep():
    """カレンダー/リスト用 3-6秒"""
    time.sleep(random.uniform(3, 6))


def get_calendar_dates(session, year, month):
    url = f'https://nar.netkeiba.com/top/calendar.html?year={year}&month={month}'
    try:
        r = safe_get(session, url)
        if r is None:
            return None  # Distinguish timeout from empty
        if r.status_code == 400:
            return 'blocked'
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, 'html.parser')
        dates = []
        for a in soup.select('table.Calendar_Table td a'):
            href = a.get('href', '')
            m = re.search(r'kaisai_date=(\d{8})', href)
            if m:
                dates.append(m.group(1))
        return sorted(set(dates))
    except Exception as e:
        print(f'  Calendar error {year}/{month}: {e}', flush=True)
        return []


def get_race_ids(session, date_str):
    url = f'https://nar.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}'
    try:
        r = safe_get(session, url)
        if r is None or r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, 'html.parser')
        race_ids = []
        for a in soup.select('a[href*="race_id="]'):
            href = a.get('href', '')
            m = re.search(r'race_id=(\d+)', href)
            if m:
                race_ids.append(m.group(1))
        return sorted(set(race_ids))
    except Exception as e:
        print(f'  RaceList error {date_str}: {e}', flush=True)
        return []


def parse_race(session, race_id, date_hint=''):
    url = f'https://nar.netkeiba.com/race/result.html?race_id={race_id}'
    try:
        r = safe_get(session, url)
        if r is None:
            return None, 'timeout'
        if r.status_code == 403 or r.status_code == 400:
            return None, f'http_{r.status_code}'
        if r.status_code != 200:
            return None, f'http_{r.status_code}'
    except Exception as e:
        return None, str(e)

    try:
        text = r.content.decode('euc-jp', errors='replace')
        soup = BeautifulSoup(text, 'html.parser')

        title = soup.title.string if soup.title else ''
        if '403' in title or 'Error' in title or 'エラー' in title:
            return None, '403'

        rn_el = soup.select_one('.RaceName')
        race_name = rn_el.get_text(strip=True) if rn_el else ''

        rd1_el = soup.select_one('.RaceData01')
        rd1 = rd1_el.get_text(strip=True) if rd1_el else ''

        rd2_el = soup.select_one('.RaceData02')
        rd2 = rd2_el.get_text(' ', strip=True) if rd2_el else ''

        distance, surface, condition, weather = 0, '', '', ''
        dm = re.search(r'([ダ芝障])(\d+)m', rd1)
        if dm:
            surface, distance = dm.group(1), int(dm.group(2))
        cm = re.search(r'馬場:(\S+)', rd1)
        if cm:
            condition = cm.group(1)
        wm = re.search(r'天候:(\S+)', rd1)
        if wm:
            weather = wm.group(1)

        course, course_code = '', race_id[4:6] if len(race_id) >= 6 else ''
        for code, name in NAR_TRACKS.items():
            if name in rd2:
                course, course_code = name, str(code)
                break

        class_info = ''
        clm = re.search(r'サラ系?\S*\s+(\S+)', rd2)
        if clm:
            class_info = clm.group(1)

        race_date = date_hint
        if not race_date:
            tm = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', title)
            if tm:
                race_date = f'{tm.group(1)}{int(tm.group(2)):02d}{int(tm.group(3)):02d}'

        result_table = soup.select_one('.ResultTableWrap table')
        if not result_table:
            return None, 'no_table'

        rows = result_table.select('tbody tr')
        if not rows:
            return None, 'no_rows'

        num_horses = len(rows)

        payouts = dict.fromkeys(['tansho', 'fukusho', 'umaren', 'wide', 'trio', 'tierce'], '')
        for pr in soup.select('.Payout_Detail_Table tr'):
            cells = pr.select('td, th')
            if len(cells) >= 3:
                lbl = cells[0].get_text(strip=True)
                val = cells[2].get_text('|', strip=True).replace(',', '')
                if lbl == '単勝':
                    payouts['tansho'] = val
                elif lbl == '複勝':
                    payouts['fukusho'] = val
                elif lbl == '馬連':
                    payouts['umaren'] = val
                elif lbl == 'ワイド':
                    payouts['wide'] = val
                elif lbl in ('3連複', '三連複'):
                    payouts['trio'] = val
                elif lbl in ('3連単', '三連単'):
                    payouts['tierce'] = val

        results = []
        for row in rows:
            cells = row.select('td')
            if len(cells) < 8:
                continue
            t = [c.get_text(strip=True) for c in cells]
            n = len(t)
            hw, hwc = '', ''
            if n > 13:
                wm2 = re.match(r'(\d+)\(([+-]?\d+)\)', t[13])
                if wm2:
                    hw, hwc = wm2.group(1), wm2.group(2)
                elif t[13].replace(' ', '').isdigit():
                    hw = t[13].replace(' ', '')

            results.append({
                'race_id': race_id, 'race_name': race_name, 'race_date': race_date,
                'course': course, 'course_code': course_code, 'distance': distance,
                'surface': surface, 'condition': condition, 'weather': weather,
                'class_info': class_info, 'num_horses': num_horses,
                'finish': t[0], 'bracket': t[1] if n > 1 else '',
                'horse_num': t[2] if n > 2 else '', 'horse_name': t[3] if n > 3 else '',
                'sex_age': t[4] if n > 4 else '', 'weight_carry': t[5] if n > 5 else '',
                'jockey_name': t[6] if n > 6 else '', 'time': t[7] if n > 7 else '',
                'margin': t[8] if n > 8 else '', 'pass_order': '',
                'last3f': t[11] if n > 11 else '', 'odds': t[10] if n > 10 else '',
                'pop_rank': t[9] if n > 9 else '', 'trainer_name': t[12] if n > 12 else '',
                'horse_weight': hw, 'horse_weight_change': hwc,
                'tansho_payout': payouts['tansho'], 'fukusho_payout': payouts['fukusho'],
                'umaren_payout': payouts['umaren'], 'wide_payout': payouts['wide'],
                'trio_payout': payouts['trio'], 'tierce_payout': payouts['tierce'],
            })

        return results, 'ok'

    except Exception as e:
        return None, str(e)


def main():
    print('=' * 60, flush=True)
    print('  NAR SCRAPER v6 (8-18s intervals, 403 retry)', flush=True)
    print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M")}', flush=True)
    print(f'  Years: {YEARS[0]}-{YEARS[-1]}', flush=True)
    print('=' * 60, flush=True)

    progress = load_progress()
    completed = set(progress['completed_dates'])
    failed_dates = set(progress.get('failed_dates', []))
    print(f'  Resumed: {len(completed)} dates, {progress["total_rows"]} rows', flush=True)

    session = create_session()
    total_new_races = 0
    total_new_rows = 0
    start_time = time.time()
    request_count = 0
    consecutive_403 = 0

    for year in YEARS:
        year_races = 0
        year_rows = 0
        year_start = time.time()

        for month in range(1, 13):
            now = datetime.now()
            if year == now.year and month > now.month:
                break
            if year > now.year:
                break

            # Check connectivity first
            dates = get_calendar_dates(session, year, month)
            nav_sleep()

            if dates == 'blocked':
                print(f'  {year}/{month:02d}: BLOCKED (400). Waiting 120s...', flush=True)
                time.sleep(120)
                session = create_session()
                # Retry once
                dates = get_calendar_dates(session, year, month)
                nav_sleep()
                if dates == 'blocked':
                    print(f'  Still blocked. Aborting.', flush=True)
                    print(f'  Tip: Restart router to change IP, then re-run.', flush=True)
                    # Save and exit
                    elapsed = time.time() - start_time
                    print(f'\n  Scraped before block: {total_new_races}R/{total_new_rows}rows ({elapsed/60:.0f}min)', flush=True)
                    _print_summary()
                    return

            if dates is None or not dates:
                continue

            pending = [d for d in dates if d not in completed]
            if not pending:
                continue

            print(f'  {year}/{month:02d}: {len(pending)} pending ({len(dates)} total)', flush=True)

            for date_str in pending:
                race_ids = get_race_ids(session, date_str)
                nav_sleep()
                request_count += 1

                if not race_ids:
                    completed.add(date_str)
                    progress['completed_dates'] = sorted(completed)
                    save_progress(progress)
                    continue

                date_results = []
                date_errors = 0

                for race_id in race_ids:
                    retries = 0
                    success = False

                    while retries < 3:
                        results, status = parse_race(session, race_id, date_str)
                        request_count += 1

                        if status == 'ok' and results:
                            date_results.extend(results)
                            consecutive_403 = 0
                            success = True
                            break
                        elif status in ('http_403', 'http_400', '403'):
                            retries += 1
                            consecutive_403 += 1
                            if retries < 3:
                                wait = 60 * retries
                                print(f'    {race_id}: {status}, retry {retries}/3 (wait {wait}s)', flush=True)
                                time.sleep(wait)
                                session = create_session()
                            else:
                                date_errors += 1
                                print(f'    {race_id}: SKIP after 3 retries', flush=True)
                        elif 'timeout' in status:
                            date_errors += 1
                            session = create_session()
                            break
                        else:
                            date_errors += 1
                            break

                    # 403が連続10回以上→IPブロック濃厚
                    if consecutive_403 >= 10:
                        print(f'  [10+ consecutive 403s - IP likely blocked]', flush=True)
                        print(f'  Saving progress and stopping.', flush=True)
                        # Save partial results for this date
                        if date_results:
                            append_to_csv(date_results, OUTPUT_CSV)
                        progress['completed_dates'] = sorted(completed)
                        progress['failed_dates'] = sorted(failed_dates)
                        save_progress(progress)
                        _print_summary()
                        return

                    # 8-18秒間隔
                    race_sleep()

                    # UA rotation every 30 requests
                    if request_count % 30 == 0:
                        session = create_session()

                # Save date results
                if date_results:
                    append_to_csv(date_results, OUTPUT_CSV)
                    n_new = len(set(r['race_id'] for r in date_results))
                    year_races += n_new
                    year_rows += len(date_results)
                    total_new_races += n_new
                    total_new_rows += len(date_results)

                completed.add(date_str)
                if date_errors > len(race_ids) // 2:
                    failed_dates.add(date_str)

                progress['completed_dates'] = sorted(completed)
                progress['failed_dates'] = sorted(failed_dates)
                progress['total_races'] = progress.get('total_races', 0) + (
                    len(set(r['race_id'] for r in date_results)) if date_results else 0)
                progress['total_rows'] = progress.get('total_rows', 0) + len(date_results)
                save_progress(progress)

                elapsed = time.time() - start_time
                rate = total_new_races / (elapsed / 3600) if elapsed > 60 else 0
                n_races = len(set(r['race_id'] for r in date_results)) if date_results else 0
                print(f'  {date_str}: {n_races}R/{len(date_results)}rows '
                      f'[cum: {total_new_races}R/{total_new_rows}rows, '
                      f'{rate:.0f}R/h, {elapsed/60:.0f}min]', flush=True)

        if year_rows > 0:
            ye = time.time() - year_start
            print(f'\n  === {year}: {year_races}R, {year_rows}rows ({ye/60:.0f}min) ===\n', flush=True)

    elapsed = time.time() - start_time
    print(f'\n{"="*60}', flush=True)
    print(f'  COMPLETE ({elapsed/3600:.1f}h)', flush=True)
    print(f'  Races: {total_new_races}, Rows: {total_new_rows}', flush=True)
    print(f'  Failed: {len(failed_dates)} dates', flush=True)
    print(f'{"="*60}', flush=True)
    _print_summary()


def _print_summary():
    if os.path.exists(OUTPUT_CSV):
        import pandas as pd
        df = pd.read_csv(OUTPUT_CSV, encoding='utf-8')
        print(f'\n  Total CSV: {len(df)} rows, {df["race_id"].nunique()} races', flush=True)
        print(f'\n  By course:', flush=True)
        for c, grp in df.groupby('course'):
            print(f'    {c}: {grp["race_id"].nunique()} races', flush=True)
        df['year'] = df['race_date'].astype(str).str[:4]
        print(f'\n  By year:', flush=True)
        for yr, grp in sorted(df.groupby('year')):
            print(f'    {yr}: {grp["race_id"].nunique()} races', flush=True)


if __name__ == '__main__':
    main()
