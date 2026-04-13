#!/usr/bin/env python
"""波乱度（upset_level）全年バルク取得スクレイパー

shutuba.htmlから波乱度Lv1-5 + 上位人気信頼度を2020-2026全レース分取得。
Resumable: 進捗を自動保存し、中断後に再開可能。

Usage:
    python tools/bulk_scrape_upset.py                    # 全年取得（resume対応）
    python tools/bulk_scrape_upset.py --year 2024        # 指定年のみ
    python tools/bulk_scrape_upset.py --status            # 進捗確認
    python tools/bulk_scrape_upset.py --reset             # 進捗リセット

リクエスト間隔: 16秒（並行スクレイピング考慮）
IPバン検知: 403/429で10分待機、3回連続失敗で中断
"""

import os
import sys
import re
import json
import time
import csv
import argparse
import random
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')

DELAY_MIN = 16
DELAY_MAX = 20
BAN_WAIT = 600  # 10 min
MAX_BAN_RETRIES = 3
SAVE_INTERVAL = 50  # Save progress every N races

PROGRESS_FILE = os.path.join(DATA_DIR, '_upset_scrape_progress.json')
OUTPUT_FILE = os.path.join(DATA_DIR, 'netkeiba_upset_level.csv')
RACE_LIST_FILE = os.path.join(DATA_DIR, '_nk_race_ids_2020_2026.txt')

COURSE_MAP = {
    '札幌': '01', '函館': '02', '福島': '03', '新潟': '04',
    '東京': '05', '中山': '06', '中京': '07', '京都': '08',
    '阪神': '09', '小倉': '10',
}


def _load_cookie():
    env_path = os.path.join(BASE_DIR, '.env')
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('NETKEIBA_COOKIE='):
                val = line[len('NETKEIBA_COOKIE='):]
                return val.strip('"').strip("'")
    return None


def _make_session():
    import requests
    cookie_str = _load_cookie()
    if not cookie_str:
        return None
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://race.netkeiba.com/',
    })
    for part in cookie_str.split(';'):
        part = part.strip()
        if '=' in part:
            key, val = part.split('=', 1)
            session.cookies.set(key.strip(), val.strip())
    return session


def build_race_id_list(year_filter=None):
    """Build netkeiba-format race IDs from jra_races_full.csv or cached file."""
    # Use cached list if available
    if os.path.exists(RACE_LIST_FILE) and year_filter is None:
        with open(RACE_LIST_FILE, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    import pandas as pd
    df = pd.read_csv(os.path.join(DATA_DIR, 'jra_races_full.csv'),
                     usecols=['race_id', 'year', 'course', 'kai', 'nichi', 'race_num'],
                     dtype=str, low_memory=False)
    df['year_int'] = pd.to_numeric(df['year'], errors='coerce')

    if year_filter:
        yy = year_filter - 2000
        df = df[df['year_int'] == yy].copy()
    else:
        df = df[(df['year_int'] >= 20) & (df['year_int'] <= 26)].copy()

    df['course_code'] = df['course'].str.strip().map(COURSE_MAP)
    df['nk_race_id'] = ('20' + df['year'].str.zfill(2) +
                         df['course_code'] +
                         df['kai'].str.zfill(2) +
                         df['nichi'].str.zfill(2) +
                         df['race_num'].str.zfill(2))
    return sorted(df['nk_race_id'].dropna().unique().tolist())


def load_progress():
    """Load scraping progress."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'completed': [], 'failed': [], 'last_updated': None}


def save_progress(progress):
    """Save scraping progress."""
    progress['last_updated'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False)


def scrape_upset(session, race_id):
    """Scrape upset level from shutuba.html. Returns dict or raises exception."""
    from bs4 import BeautifulSoup

    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = session.get(url, timeout=30)

    # IP ban detection
    if resp.status_code in (403, 429):
        raise ConnectionError(f"HTTP {resp.status_code} - possible IP ban")

    if resp.status_code == 404:
        return {'race_id': race_id, 'upset_level': None, 'top_popularity_reliability': None,
                'note': '404'}

    html = resp.content.decode('utf-8', errors='ignore')

    # Check for premium paywall
    if 'プレミアムサービス案内' in html:
        raise PermissionError("Premium cookie expired")

    result = {'race_id': race_id}

    # 波乱度 (Lv 1-5) from JS
    upset_match = re.search(r"var\s+upsetLevel\s*=\s*parseInt\('(\d+)'\)", html)
    if upset_match:
        result['upset_level'] = int(upset_match.group(1))
    else:
        result['upset_level'] = None

    # 上位人気信頼度
    soup = BeautifulSoup(html, 'html.parser')
    turmoil = soup.find('td', class_='Turmoil_Score')
    if turmoil:
        p = turmoil.find('p')
        if p:
            span = p.find('span')
            if span:
                val = span.get_text(strip=True).replace('%', '')
                try:
                    result['top_popularity_reliability'] = float(val)
                except ValueError:
                    result['top_popularity_reliability'] = None
            else:
                result['top_popularity_reliability'] = None
        else:
            result['top_popularity_reliability'] = None
    else:
        result['top_popularity_reliability'] = None

    return result


def append_to_csv(result, filepath):
    """Append a single result to CSV (creates header if new)."""
    file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
    with open(filepath, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['race_id', 'upset_level', 'top_popularity_reliability'])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'race_id': result['race_id'],
            'upset_level': result.get('upset_level', ''),
            'top_popularity_reliability': result.get('top_popularity_reliability', ''),
        })


def run_bulk_scrape(year_filter=None):
    """Main bulk scraping loop."""
    print("=" * 60)
    print("  波乱度バルク取得")
    print("=" * 60)

    session = _make_session()
    if not session:
        print("ERROR: Cookie not found. Run: python tools/refresh_cookie.py")
        sys.exit(1)

    # Build race list
    print("\n[1] Building race ID list...")
    all_races = build_race_id_list(year_filter)
    print(f"  Total races: {len(all_races)}")

    # Load progress
    progress = load_progress()
    completed_set = set(progress['completed'])
    failed_set = set(progress['failed'])
    print(f"  Already completed: {len(completed_set)}")
    print(f"  Previously failed: {len(failed_set)}")

    # Filter to remaining
    remaining = [r for r in all_races if r not in completed_set]
    print(f"  Remaining: {len(remaining)}")

    if not remaining:
        print("\n  All races already completed!")
        return

    # Estimate time
    est_hours = len(remaining) * DELAY_MIN / 3600
    print(f"  Estimated time: {est_hours:.1f} hours")

    # Start scraping
    print(f"\n[2] Starting bulk scrape...")
    consecutive_bans = 0
    scraped = 0
    errors = 0
    start_time = time.time()

    from tools.scraper_guard import check_scraping_allowed
    for i, race_id in enumerate(remaining):
        check_scraping_allowed()  # Fri22:00〜Mon06:00は自動停止→再開
        try:
            result = scrape_upset(session, race_id)

            if result.get('upset_level') is not None:
                append_to_csv(result, OUTPUT_FILE)
                progress['completed'].append(race_id)
                scraped += 1
                consecutive_bans = 0

                if scraped % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = scraped / elapsed * 3600
                    eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
                    print(f"  [{i+1}/{len(remaining)}] {race_id}: "
                          f"Lv{result['upset_level']} ({result.get('top_popularity_reliability','?')}%) "
                          f"| {scraped} done, {rate:.0f}/h, ETA {eta:.1f}h")
            else:
                # No upset data (old race without the feature?)
                progress['completed'].append(race_id)
                scraped += 1

                if scraped % 50 == 0:
                    print(f"  [{i+1}/{len(remaining)}] {race_id}: no data (skipped)")

        except ConnectionError as e:
            # IP ban
            consecutive_bans += 1
            errors += 1
            print(f"\n  !!! BAN DETECTED ({consecutive_bans}/{MAX_BAN_RETRIES}): {e}")

            if consecutive_bans >= MAX_BAN_RETRIES:
                print(f"  ABORTING: {MAX_BAN_RETRIES} consecutive bans. Resume later.")
                save_progress(progress)
                return

            print(f"  Waiting {BAN_WAIT}s ({BAN_WAIT//60} min)...")
            save_progress(progress)
            time.sleep(BAN_WAIT)

            # Refresh session
            session = _make_session()
            continue

        except PermissionError as e:
            print(f"\n  !!! COOKIE EXPIRED: {e}")
            print("  Run: python tools/refresh_cookie.py")
            save_progress(progress)
            return

        except Exception as e:
            errors += 1
            progress['failed'].append(race_id)
            if errors % 10 == 0:
                print(f"  [{i+1}/{len(remaining)}] {race_id}: ERROR {e}")

        # Save progress periodically
        if (i + 1) % SAVE_INTERVAL == 0:
            save_progress(progress)

        # Random delay
        delay = DELAY_MIN + random.random() * (DELAY_MAX - DELAY_MIN)
        time.sleep(delay)

    # Final save
    save_progress(progress)
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  COMPLETED")
    print(f"{'='*60}")
    print(f"  Scraped: {scraped}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed/3600:.1f} hours")
    print(f"  Output: {OUTPUT_FILE}")

    # Count total CSV rows
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            total_rows = sum(1 for _ in f) - 1  # minus header
        print(f"  Total CSV rows: {total_rows}")


def show_status():
    """Show current scraping progress."""
    progress = load_progress()
    all_races = build_race_id_list()
    completed = len(progress['completed'])
    total = len(all_races)
    remaining = total - completed

    print(f"Total races: {total}")
    print(f"Completed: {completed} ({completed/total*100:.1f}%)")
    print(f"Failed: {len(progress['failed'])}")
    print(f"Remaining: {remaining}")
    print(f"Est. time remaining: {remaining * DELAY_MIN / 3600:.1f} hours")
    print(f"Last updated: {progress.get('last_updated', 'N/A')}")

    if os.path.exists(OUTPUT_FILE):
        size = os.path.getsize(OUTPUT_FILE)
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            rows = sum(1 for _ in f) - 1
        print(f"\nOutput file: {OUTPUT_FILE}")
        print(f"  Size: {size:,} bytes, Rows: {rows}")


def main():
    parser = argparse.ArgumentParser(description='Bulk scrape upset levels')
    parser.add_argument('--year', type=int, default=None, help='Year filter (e.g., 2024)')
    parser.add_argument('--status', action='store_true', help='Show progress')
    parser.add_argument('--reset', action='store_true', help='Reset progress')
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.reset:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            print("Progress reset.")
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
            print("Output file removed.")
        return

    run_bulk_scrape(args.year)


if __name__ == '__main__':
    from tools.scraper_guard import check_scraping_allowed; check_scraping_allowed()
    main()
