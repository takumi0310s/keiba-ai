#!/usr/bin/env python3
"""netkeiba厩舎コメント一括取得（複数年対応）

scrape_training.pyのfetch_stable_commentsを使い、commentページから取得。
中断→再開対応。

Usage:
    python tools/scrape_comments_bulk.py --years 2021 2022 2023
"""
import pandas as pd
import os
import sys
import io
import time
import json
import argparse
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')

COMMENT_CSV = os.path.join(DATA_DIR, 'netkeiba_stable_comments.csv')
PROGRESS_FILE = os.path.join(DATA_DIR, 'scrape_comments_bulk_progress.json')
DELAY = 2.0

COMMENT_HEADER = ['race_id', 'race_date', 'umaban', 'horse_name', 'comment', 'score']


def _target_to_netkeiba(target_rid):
    rid = str(target_rid).zfill(10)
    cc, yy, k, n, rr = rid[0:2], rid[2:4], rid[4:5], rid[5:6], rid[6:8]
    n_map = {'A': '10', 'B': '11', 'C': '12'}
    return f"20{yy}{cc}{k.zfill(2)}{n_map.get(n, n.zfill(2))}{rr}"


def get_race_ids(year):
    csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
    df = pd.read_csv(csv_path, encoding='utf-8-sig',
                      usecols=['year', 'race_id'], dtype=str, low_memory=False)
    df['year_int'] = pd.to_numeric(df['year'], errors='coerce')
    yr2 = year % 100
    nk_ids = set()
    for rid in df[df['year_int'] == yr2]['race_id'].dropna().unique():
        nk_ids.add(_target_to_netkeiba(rid))
    return sorted(nk_ids)


def scrape_one(race_id):
    from scrape_training import fetch_stable_comments, _COMMENT_CACHE
    _COMMENT_CACHE.pop(race_id, None)
    data = fetch_stable_comments(race_id, is_nar=False)
    if not data:
        return []
    from scrape_training import _score_comment
    results = []
    for umaban, d in sorted(data.items()):
        results.append([
            race_id, '', umaban, '',
            d.get('comment', ''), d.get('score', 0),
        ])
    return results


def _append_csv(rows):
    write_header = not os.path.exists(COMMENT_CSV) or os.path.getsize(COMMENT_CSV) == 0
    with open(COMMENT_CSV, 'a', encoding='utf-8-sig', newline='') as f:
        if write_header:
            f.write(','.join(COMMENT_HEADER) + '\n')
        for row in rows:
            f.write(','.join(str(v).replace(',', '；') for v in row) + '\n')


def _save_progress(done_ids):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({'done': sorted(done_ids), 'ts': datetime.now().isoformat()}, f)


def _load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return set(json.load(f).get('done', []))
    return set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', nargs='+', type=int, required=True)
    args = parser.parse_args()

    start = datetime.now()
    print("=" * 60)
    print(f"  Stable Comments Bulk Scraper: {args.years}")
    print("=" * 60)

    all_ids = []
    for y in args.years:
        ids = get_race_ids(y)
        all_ids.extend(ids)
        print(f"  {y}: {len(ids)} races")
    all_ids = sorted(set(all_ids))
    print(f"  Total: {len(all_ids)} unique race IDs")

    existing = set()
    if os.path.exists(COMMENT_CSV):
        try:
            edf = pd.read_csv(COMMENT_CSV, encoding='utf-8-sig',
                               usecols=['race_id'], dtype=str, low_memory=False)
            existing = set(edf['race_id'].unique())
        except Exception:
            pass
    existing |= _load_progress()

    new_ids = [rid for rid in all_ids if str(rid) not in existing]
    print(f"  Already done: {len(existing)}")
    print(f"  Remaining: {len(new_ids)}")
    print(f"  Estimated: {len(new_ids) * DELAY / 60:.0f} min")

    if not new_ids:
        print("  Nothing to do!")
        return

    stats = {'races': 0, 'rows': 0, 'empty': 0, 'errors': 0}
    done = set(existing)

    for i, rid in enumerate(new_ids):
        try:
            rows = scrape_one(rid)
            if rows:
                _append_csv(rows)
                stats['races'] += 1
                stats['rows'] += len(rows)
            else:
                stats['empty'] += 1
            done.add(rid)
        except Exception:
            stats['errors'] += 1

        if (i + 1) % 20 == 0:
            elapsed = (datetime.now() - start).total_seconds() / 60
            eta = elapsed / (i + 1) * (len(new_ids) - i - 1)
            print(f"\r  [{i+1}/{len(new_ids)} {(i+1)/len(new_ids)*100:.0f}%] "
                  f"{stats['races']}R/{stats['rows']}rows "
                  f"empty={stats['empty']} err={stats['errors']} "
                  f"ETA={eta:.0f}min", end='', flush=True)

        if (i + 1) % 200 == 0:
            _save_progress(done)
            print(f"\n  Checkpoint saved ({len(done)} done)")

        time.sleep(DELAY)

    _save_progress(done)
    elapsed = (datetime.now() - start).total_seconds() / 60

    print(f"\n\n{'=' * 60}")
    print(f"  COMPLETE in {elapsed:.1f} min")
    print(f"  Comments: {stats['races']}R / {stats['rows']} rows")
    print(f"  Empty: {stats['empty']}, Errors: {stats['errors']}")
    if os.path.exists(COMMENT_CSV):
        total = len(pd.read_csv(COMMENT_CSV, encoding='utf-8-sig', low_memory=False))
        print(f"  CSV total: {total:,} rows")
    print("=" * 60)


if __name__ == '__main__':
    from tools.scraper_guard import check_scraping_allowed; check_scraping_allowed()
    main()
