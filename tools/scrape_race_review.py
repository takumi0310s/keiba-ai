#!/usr/bin/env python3
"""
db.netkeiba.com レース結果ページから備考(短評/レースレビュー)を取得

- URL: https://db.netkeiba.com/race/{race_id}/
- Encoding: EUC-JP
- Result table class: race_table_01
- Cell [0]=着順, [2]=馬番, [3]=馬名, [21]=備考
- 備考をスコア化 (-3 ~ +3)

対象: 2024-2025 JRA全レース (~6,900 races)
出力: data/netkeiba_race_review.csv
"""

import sys
import os
import time
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_CSV = os.path.join(DATA_DIR, 'netkeiba_race_review.csv')

# ===== Cookie読み込み =====

def load_cookie():
    """Read NETKEIBA_COOKIE from .env file"""
    env_path = os.path.join(BASE_DIR, '.env')
    if not os.path.exists(env_path):
        return ''
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('NETKEIBA_COOKIE='):
                return line[len('NETKEIBA_COOKIE='):]
    return ''

_cookie = load_cookie()
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
if _cookie:
    HEADERS['Cookie'] = _cookie
    print(f"Premium cookie loaded ({len(_cookie)} chars)")
else:
    print("WARNING: No NETKEIBA_COOKIE found. Remarks may be empty without premium.")

REQUEST_INTERVAL = 3  # seconds

# ===== 備考スコア化 =====

NEGATIVE_PATTERNS = [
    ('出遅', -2),
    ('不利', -2),
    ('挟まれ', -2),
    ('挟まる', -2),
    ('躓', -1),    # つまずき
    ('ふらつ', -1),
    ('掛かり', -1),
    ('掛かっ', -1),
    ('外を回', -1),
    ('外回り', -1),
    ('後方', -1),
    ('伸びず', -1),
    ('伸び欠', -1),
    ('一杯', -1),
    ('落馬', -3),
    ('競走中止', -3),
    ('故障', -3),
]

POSITIVE_PATTERNS = [
    ('好位', +1),
    ('好発', +1),
    ('先行', +1),
    ('楽勝', +2),
    ('圧勝', +2),
    ('快勝', +2),
    ('直線伸び', +1),
    ('鮮やか', +2),
    ('余裕', +1),
    ('持ったまま', +2),
    ('抜け出', +1),
    ('好走', +1),
]


def score_remarks(text):
    """備考テキストをスコア化 (-3 ~ +3)"""
    if not text:
        return 0
    score = 0
    for pattern, val in NEGATIVE_PATTERNS:
        if pattern in text:
            score += val
    for pattern, val in POSITIVE_PATTERNS:
        if pattern in text:
            score += val
    return max(-3, min(3, score))


# ===== race_id変換 =====

def jv_to_netkeiba(jv_rid, nichi_num):
    """TARGET JV race_id -> netkeiba race_id (12 digits)
    JV format: CCYYKNRRHH (10 chars, nichi may be hex)
    netkeiba format: YYYYCCKKNNNRR
    nichi_num: numeric nichi value from CSV column
    """
    s = str(jv_rid).zfill(10)
    course = s[0:2]
    year = '20' + s[2:4]
    kai = s[4:5]
    race_num = s[6:8]
    return f'{year}{course}{int(kai):02d}{int(nichi_num):02d}{race_num}'


def get_race_ids(years=(24, 25)):
    """jra_races_full.csvから指定年のユニークレースIDリストを取得"""
    csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
    df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False,
                     usecols=['year', 'month', 'day', 'kai', 'nichi',
                              'race_num', 'race_id'])

    df = df[df['year'].isin(years)]

    # race_key = first 8 chars of race_id (without umaban)
    df['race_key'] = df['race_id'].astype(str).str[:8]
    df_unique = df.drop_duplicates('race_key')

    races = []
    seen = set()
    for _, r in df_unique.iterrows():
        nk_id = jv_to_netkeiba(r['race_id'], r['nichi'])
        # nk_id is 12 digits but netkeiba expects without umaban part
        # Actually jv_to_netkeiba uses full race_id including HH (umaban)
        # We need race-level ID: YYYYCCKKNNNRR (without horse)
        # race_num from JV is already in the output, nichi_num is day
        # The function already strips umaban via s[6:8] = race_num
        if nk_id not in seen:
            seen.add(nk_id)
            race_date = f"20{int(r['year']):02d}-{int(r['month']):02d}-{int(r['day']):02d}"
            races.append({
                'nk_race_id': nk_id,
                'race_date': race_date,
                'year': int(r['year']),
            })

    # Sort by race_id (chronological)
    races.sort(key=lambda x: x['nk_race_id'])

    # Filter out future races (no results yet)
    today = datetime.now().strftime('%Y-%m-%d')
    races = [r for r in races if r['race_date'] <= today]

    return races


# ===== スクレイピング =====

def scrape_race_review(race_id):
    """1レースの備考データを取得
    Returns: list of dicts or None on error
    """
    url = f'https://db.netkeiba.com/race/{race_id}/'
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
    except requests.RequestException as e:
        print(f"  Request error: {e}")
        return None

    if resp.status_code == 403:
        return 'FORBIDDEN'
    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code}")
        return None

    soup = BeautifulSoup(resp.content, 'html.parser', from_encoding='euc-jp')
    table = soup.find('table', class_='race_table_01')
    if not table:
        return None

    rows = table.find_all('tr')
    if len(rows) < 2:
        return None

    results = []
    for row in rows[1:]:  # skip header
        cells = row.find_all('td')
        if len(cells) < 22:
            continue

        finish_text = cells[0].get_text(strip=True)
        umaban_text = cells[2].get_text(strip=True)
        horse_name = cells[3].get_text(strip=True)
        remarks = cells[21].get_text(strip=True)

        # Parse finish (may be non-numeric for 中止, 除外, etc.)
        try:
            finish = int(finish_text)
        except (ValueError, TypeError):
            finish = 0  # 中止/除外/取消

        try:
            umaban = int(umaban_text)
        except (ValueError, TypeError):
            continue  # skip invalid rows

        review_score = score_remarks(remarks)

        results.append({
            'race_id': race_id,
            'umaban': umaban,
            'horse_name': horse_name,
            'finish': finish,
            'remarks': remarks,
            'review_score': review_score,
        })

    return results


# ===== メイン =====

def main():
    print("=" * 60, flush=True)
    print("netkeiba Race Review Scraper (備考/短評)", flush=True)
    print("=" * 60, flush=True)

    # Get race IDs
    print("\nLoading race IDs from jra_races_full.csv...", flush=True)
    races = get_race_ids(years=(24, 25))
    print(f"Total unique races (2024-2025): {len(races)}")

    # Load existing data for resume support
    done_ids = set()
    existing_rows = []
    if os.path.exists(OUTPUT_CSV):
        try:
            existing = pd.read_csv(OUTPUT_CSV, encoding='utf-8')
            done_ids = set(existing['race_id'].astype(str).unique())
            existing_rows = existing.to_dict('records')
            print(f"Existing data: {len(done_ids)} races, {len(existing_rows)} rows")
        except Exception as e:
            print(f"Warning: Could not read existing CSV: {e}")

    # Filter out already scraped
    remaining = [r for r in races if r['nk_race_id'] not in done_ids]
    print(f"Remaining to scrape: {len(remaining)} races")

    if not remaining:
        print("All races already scraped!")
        return

    # Scrape
    new_rows = []
    consecutive_403 = 0
    scraped = 0
    empty_remarks = 0
    has_remarks = 0
    start_time = time.time()

    for i, race in enumerate(remaining):
        race_id = race['nk_race_id']

        result = scrape_race_review(race_id)

        if result == 'FORBIDDEN':
            consecutive_403 += 1
            print(f"  [{i+1}/{len(remaining)}] {race_id} -> 403 FORBIDDEN (#{consecutive_403})")
            if consecutive_403 >= 10:
                print("\n*** 10 consecutive 403s - stopping ***")
                break
            time.sleep(REQUEST_INTERVAL)
            continue
        elif result is None:
            print(f"  [{i+1}/{len(remaining)}] {race_id} -> ERROR/no table")
            consecutive_403 = 0
            time.sleep(REQUEST_INTERVAL)
            continue
        else:
            consecutive_403 = 0

        new_rows.extend(result)
        scraped += 1

        # Count remarks
        race_has_remarks = any(r['remarks'] for r in result)
        if race_has_remarks:
            has_remarks += 1
        else:
            empty_remarks += 1

        # Progress
        elapsed = time.time() - start_time
        rate = scraped / elapsed * 3600 if elapsed > 0 else 0
        remaining_est = (len(remaining) - i - 1) / (rate / 3600) if rate > 0 else 0
        print(f"  [{i+1}/{len(remaining)}] {race_id} ({race['race_date']}) "
              f"-> {len(result)} horses, "
              f"remarks={'YES' if race_has_remarks else 'no'} "
              f"[{rate:.0f}/hr, ETA {remaining_est/60:.0f}min]",
              flush=True)

        # Save every 100 races
        if scraped % 100 == 0:
            save_csv(existing_rows + new_rows)
            print(f"  --- Saved ({len(existing_rows) + len(new_rows)} total rows) ---")

        time.sleep(REQUEST_INTERVAL)

    # Final save
    if new_rows:
        save_csv(existing_rows + new_rows)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Scraped: {scraped} races ({len(new_rows)} horse rows)")
    print(f"With remarks: {has_remarks}, Empty remarks: {empty_remarks}")
    print(f"Time: {elapsed/60:.1f} min ({elapsed/scraped:.1f}s per race)" if scraped else "")
    total_rows = len(existing_rows) + len(new_rows)
    print(f"Total in CSV: {total_rows} rows")
    print(f"Output: {OUTPUT_CSV}")

    # Score distribution
    if new_rows:
        scores = [r['review_score'] for r in new_rows if r['remarks']]
        if scores:
            print(f"\nReview scores (non-empty remarks only):")
            for s in range(-3, 4):
                cnt = scores.count(s)
                if cnt:
                    print(f"  Score {s:+d}: {cnt} ({cnt/len(scores)*100:.1f}%)")


def save_csv(rows):
    """Save rows to CSV"""
    df = pd.DataFrame(rows)
    cols = ['race_id', 'umaban', 'horse_name', 'finish', 'remarks', 'review_score']
    for c in cols:
        if c not in df.columns:
            df[c] = ''
    df = df[cols]
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')


if __name__ == '__main__':
    main()
