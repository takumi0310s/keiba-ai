#!/usr/bin/env python3
"""厩舎コメント一括取得スクレイパー (comment.html)

対象: 新馬戦 + 特別レース (comment.htmlはこれらのみ公開)
URL: https://race.netkeiba.com/race/comment.html?race_id=XXXXX
保存: data/netkeiba_race_analysis.csv

Usage:
    python tools/scrape_stable_comment.py                # 2024-2026
    python tools/scrape_stable_comment.py --year 2025    # 指定年
    python tools/scrape_stable_comment.py --limit 50     # テスト
"""
import sys
import io
import os
import re
import time
import random
import argparse
import json
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_CSV = os.path.join(DATA_DIR, 'netkeiba_race_analysis.csv')
PROGRESS_KEY = 'stable_comment_v2'

DELAY_MIN = 3
DELAY_MAX = 5
RETRY_DELAY = 30
MAX_RETRIES = 3

# コメントスコアリング辞書
POSITIVE_WORDS = [
    '好調', '上昇', '仕上', '充実', '好気配', '順調', '態勢', '整っ',
    '良く', 'いい', '抜群', '上々', '絶好', '万全', '期待', '楽しみ',
    '力を出', '好走', '自信', '手応え', '手ごたえ', '好内容',
    '素直', '前向き', '元気', '活気', '動き良', '良い動き',
]
NEGATIVE_WORDS = [
    '不安', '課題', '心配', '微妙', '難し', '厳し', '余裕',
    '太め', '物足り', '時間', 'まだ', '初めて', '未知数',
    '成長途上', '力不足', '合わ', '苦手', '重い', 'もう一つ',
    'もうひとつ', 'ピリッと', '反応', '行きたがる',
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
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
    for part in cookie_str.split(';'):
        part = part.strip()
        if '=' in part:
            k, v = part.split('=', 1)
            session.cookies.set(k.strip(), v.strip())
    return session


def _target_to_netkeiba(target_rid):
    rid = str(target_rid).zfill(10)
    cc, yy, k, n, rr = rid[0:2], rid[2:4], rid[4:5], rid[5:6], rid[6:8]
    n_map = {'A': '10', 'B': '11', 'C': '12'}
    n_dec = n_map.get(n, n.zfill(2))
    return f"20{yy}{cc}{k.zfill(2)}{n_dec}{rr}"


def get_special_maiden_race_ids(year):
    """jra_races_full.csvから特別レース + 新馬戦のrace_idを取得"""
    csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
    df = pd.read_csv(csv_path, usecols=['year', 'race_id', 'class_code', 'race_name'],
                     dtype=str, low_memory=False)
    yr2 = str(year % 100)
    df = df[df['year'] == yr2]

    nk_ids = set()
    for _, row in df.iterrows():
        rid = str(row['race_id']).strip()
        class_code = str(row.get('class_code', '')).strip()
        race_name = str(row.get('race_name', '')).strip()

        # 特別レース: race_numが11 or 12、またはclass_codeが特別系
        race_num = rid[6:8] if len(rid) >= 8 else ''
        is_special = race_num in ('11', '12')
        # 新馬: class_code = '15' or race_name contains '新馬'
        is_maiden = class_code == '15' or '新馬' in race_name

        if is_special or is_maiden:
            nk_id = _target_to_netkeiba(rid)
            nk_ids.add(nk_id)

    return sorted(nk_ids)


def score_comment(text):
    """コメントをスコアリング (-3 ~ +3)"""
    if not text or len(text) < 3:
        return 0
    score = 0
    for w in POSITIVE_WORDS:
        if w in text:
            score += 1
    for w in NEGATIVE_WORDS:
        if w in text:
            score -= 1
    return max(-3, min(3, score))


def scrape_comment_page(session, race_id):
    """comment.htmlから厩舎コメントをスクレイプ"""
    url = f"https://race.netkeiba.com/race/comment.html?race_id={race_id}"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 404:
            return []
        if resp.status_code in (403, 429):
            time.sleep(RETRY_DELAY)
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                return []
    except Exception:
        return []

    text = resp.content.decode('euc-jp', errors='replace')
    soup = BeautifulSoup(text, 'lxml')

    results = []
    # Find the comment table (枠, 馬番, 馬名, コメント, 評価)
    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        if len(rows) < 2:
            continue

        header = [c.get_text(strip=True) for c in rows[0].find_all(['td', 'th'])]
        if '馬名' not in str(header) and 'コメント' not in str(header):
            continue

        for row in rows[1:]:
            cells = [c.get_text(strip=True) for c in row.find_all(['td', 'th'])]
            if len(cells) < 4:
                continue

            # Expected: [枠, 馬番, 馬名, コメント, 評価?]
            waku = cells[0] if len(cells) > 0 else ''
            umaban = cells[1] if len(cells) > 1 else ''
            horse_name = cells[2] if len(cells) > 2 else ''
            comment = cells[3] if len(cells) > 3 else ''
            evaluation = cells[4] if len(cells) > 4 else ''

            if not horse_name or not comment or len(comment) < 5:
                continue

            try:
                umaban_int = int(umaban)
            except (ValueError, TypeError):
                continue

            sc = score_comment(comment)
            results.append({
                'race_id': race_id,
                'umaban': umaban_int,
                'horse_name': horse_name,
                'comment': comment.replace(',', '；').replace('\n', ' '),
                'score': sc,
                'evaluation': evaluation,
            })

    return results


def _load_progress():
    path = os.path.join(DATA_DIR, 'scrape_progress.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _save_progress(progress):
    path = os.path.join(DATA_DIR, 'scrape_progress.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=None)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    session = _load_session()
    if not session:
        return

    years = [args.year] if args.year else [2024, 2025, 2026]

    print("=" * 60)
    print("  厩舎コメント一括取得 (comment.html)")
    print(f"  Years: {years}")
    print("=" * 60)

    # Collect race IDs
    all_race_ids = []
    for y in years:
        ids = get_special_maiden_race_ids(y)
        print(f"  {y}: {len(ids)} special/maiden races")
        all_race_ids.extend(ids)

    # Load existing data to skip
    existing_ids = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df_existing = pd.read_csv(OUTPUT_CSV)
            existing_ids = set(df_existing['race_id'].astype(str).unique())
            print(f"  Already scraped: {len(existing_ids)} races")
        except Exception:
            pass

    to_scrape = [rid for rid in all_race_ids if rid not in existing_ids]
    if args.limit:
        to_scrape = to_scrape[:args.limit]

    print(f"  To scrape: {len(to_scrape)} races")

    if not to_scrape:
        print("  Nothing to scrape!")
        return

    total_rows = 0
    total_races_with_data = 0
    errors = 0

    for i, rid in enumerate(to_scrape):
        rows = scrape_comment_page(session, rid)
        if rows:
            total_races_with_data += 1
            total_rows += len(rows)
            # Append to CSV
            df_new = pd.DataFrame(rows)
            write_header = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
            df_new.to_csv(OUTPUT_CSV, mode='a', index=False, header=write_header, encoding='utf-8-sig')

        pct = (i + 1) / len(to_scrape) * 100
        marker = f"+{len(rows)}" if rows else "empty"
        print(f"  [{i+1}/{len(to_scrape)} {pct:.0f}%] {rid} → {marker}", end='')

        if (i + 1) % 20 == 0:
            print(f"  (cumul: {total_rows} rows, {total_races_with_data} races)")
        else:
            print()

        if i < len(to_scrape) - 1:
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"{'='*60}")
    print(f"  Scraped: {len(to_scrape)} races")
    print(f"  With data: {total_races_with_data} races")
    print(f"  Total rows: {total_rows}")
    if os.path.exists(OUTPUT_CSV):
        df_all = pd.read_csv(OUTPUT_CSV)
        print(f"  CSV total: {len(df_all)} rows, {df_all['race_id'].nunique()} races")
    print(f"  Output: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
