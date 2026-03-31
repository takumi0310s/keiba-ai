#!/usr/bin/env python3
"""2025年netkeiba調教タイム一括取得（調教のみ、コメントスキップ）

既存scrape_premium_data.pyベースだが調教タイムのみに特化して高速化。
delay=2秒、コメント/傾向スキップで約110分見込み。

Usage:
    python tools/scrape_training_bulk_2025.py
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
TRAINING_CSV = os.path.join(DATA_DIR, 'netkeiba_training_times.csv')
PROGRESS_FILE = os.path.join(DATA_DIR, 'scrape_training_progress.json')

TRAINING_HEADER = [
    'race_id', 'race_date', 'umaban', 'horse_name', 'course', 'condition',
    'rider', 'time_6f', 'time_5f', 'time_4f', 'time_3f', 'time_1f',
    'intensity', 'rank', 'evaluation', 'training_date',
]

DELAY = 2.0  # seconds between requests


def _load_session():
    env_path = os.path.join(BASE_DIR, '.env')
    cookie_str = ''
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('NETKEIBA_COOKIE='):
                cookie_str = line.strip().split('=', 1)[1].strip('"').strip("'")
    session = requests.Session()
    session.headers.update(HEADERS)
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


def get_race_ids_2025():
    csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
    df = pd.read_csv(csv_path, encoding='utf-8-sig',
                      usecols=['year', 'race_id'], dtype=str, low_memory=False)
    df['year_int'] = pd.to_numeric(df['year'], errors='coerce')
    df = df[df['year_int'] == 25]
    nk_ids = set()
    for rid in df['race_id'].dropna().unique():
        nk_ids.add(_target_to_netkeiba(rid))
    return sorted(nk_ids)


def scrape_oikiri(session, race_id):
    """Scrape training times using the refactored scrape_training module."""
    from scrape_training import fetch_training_times, _CACHE
    _CACHE.pop(race_id, None)  # Clear cache to force fresh fetch

    data = fetch_training_times(race_id, is_nar=False)
    if not data:
        return []

    results = []
    for umaban, d in sorted(data.items()):
        results.append([
            race_id, '', umaban, '',
            d.get('course', ''), '',
            '', 0.0, 0.0,
            d.get('time_4f', 0.0),
            d.get('time_3f', 0.0),
            d.get('time_1f', 0.0),
            d.get('intensity', ''),
            d.get('rank', ''),
            d.get('evaluation', ''),
            d.get('date', ''),
        ])
    return results


def _append_csv(rows):
    write_header = not os.path.exists(TRAINING_CSV) or os.path.getsize(TRAINING_CSV) == 0
    with open(TRAINING_CSV, 'a', encoding='utf-8-sig', newline='') as f:
        if write_header:
            f.write(','.join(TRAINING_HEADER) + '\n')
        for row in rows:
            f.write(','.join(str(v).replace(',', '；') for v in row) + '\n')


def _save_progress(done_ids):
    import json
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({'done': list(done_ids), 'timestamp': datetime.now().isoformat()}, f)


def _load_progress():
    import json
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            data = json.load(f)
        return set(data.get('done', []))
    return set()


def main():
    start = datetime.now()
    print("=" * 60)
    print("  2025 Training Times Bulk Scraper")
    print("=" * 60)

    session = _load_session()

    # Get all 2025 race IDs
    all_ids = get_race_ids_2025()
    print(f"  Total 2025 race IDs: {len(all_ids)}")

    # Already scraped (from CSV)
    existing = set()
    if os.path.exists(TRAINING_CSV):
        try:
            edf = pd.read_csv(TRAINING_CSV, encoding='utf-8-sig', usecols=['race_id'], dtype=str)
            existing = set(edf['race_id'].unique())
        except Exception:
            pass

    # Also load progress file (for resume after interruption)
    progress_done = _load_progress()
    existing = existing | progress_done

    new_ids = [rid for rid in all_ids if str(rid) not in existing]
    print(f"  Already done: {len(existing)}")
    print(f"  Remaining: {len(new_ids)}")
    print(f"  Estimated: {len(new_ids) * DELAY / 60:.0f} min")

    if not new_ids:
        print("  Nothing to do!")
        return

    # Scrape
    stats = {'races': 0, 'rows': 0, 'empty': 0, 'errors': 0}
    done_ids = set(existing)

    for i, race_id in enumerate(new_ids):
        rid = str(race_id)
        pct = (i + 1) / len(new_ids) * 100
        elapsed = (datetime.now() - start).total_seconds() / 60
        eta = elapsed / max(i, 1) * (len(new_ids) - i) if i > 0 else 0

        if (i + 1) % 20 == 0 or i == 0:
            print(f"\r  [{i+1}/{len(new_ids)} {pct:.0f}%] {rid} "
                  f"({stats['races']}R/{stats['rows']}rows, "
                  f"empty={stats['empty']}, err={stats['errors']}, "
                  f"ETA={eta:.0f}min)", end='', flush=True)

        try:
            rows = scrape_oikiri(session, rid)
            if rows:
                _append_csv(rows)
                stats['races'] += 1
                stats['rows'] += len(rows)
            else:
                stats['empty'] += 1
            done_ids.add(rid)
        except Exception as e:
            stats['errors'] += 1
            if stats['errors'] % 10 == 0:
                print(f"\n  WARNING: {stats['errors']} errors so far. Last: {e}")

        time.sleep(DELAY)

        # Save progress every 100 races
        if (i + 1) % 100 == 0:
            _save_progress(done_ids)
            print(f"\n  Checkpoint: {stats['races']}R/{stats['rows']}rows saved")

    _save_progress(done_ids)

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\n\n{'=' * 60}")
    print(f"  COMPLETE in {elapsed:.1f} min")
    print(f"{'=' * 60}")
    print(f"  Training: {stats['races']} races, {stats['rows']} rows")
    print(f"  Empty: {stats['empty']} (no training data)")
    print(f"  Errors: {stats['errors']}")

    if os.path.exists(TRAINING_CSV):
        df = pd.read_csv(TRAINING_CSV, encoding='utf-8-sig')
        print(f"  CSV total: {len(df)} rows")


if __name__ == '__main__':
    main()
