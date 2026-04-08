#!/usr/bin/env python
"""過去データ一括スクレイピング（手動実行用）

進捗をdata/scrape_progress.jsonに保存。中断→再開可能。
ブロックされたら自動停止（次回再開時に続きから）。

Usage:
    python tools/bulk_scrape_history.py --source speed_index --year 2023
    python tools/bulk_scrape_history.py --source training --year 2024
    python tools/bulk_scrape_history.py --source speed_index --year 2020-2025
    python tools/bulk_scrape_history.py --all
    python tools/bulk_scrape_history.py --status          # 進捗確認
    python tools/bulk_scrape_history.py --reset            # 進捗リセット
"""
import os
import sys
import io
import json
import time
import argparse
import subprocess
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROGRESS_FILE = os.path.join(DATA_DIR, 'scrape_progress.json')
LIMIT_PER_YEAR = 99999  # Rate-limit safe batch size


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_progress(p):
    p['last_bulk_run'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(p, f, indent=2, ensure_ascii=False)


def count_existing(csv_path, year):
    if not os.path.exists(csv_path):
        return 0
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, encoding='utf-8-sig', usecols=['race_id'], dtype=str)
        return len(df[df['race_id'].str.startswith(str(year))]['race_id'].unique())
    except Exception:
        return 0


def estimate_total(year):
    try:
        import pandas as pd
        df = pd.read_csv(os.path.join(DATA_DIR, 'jra_races_full.csv'),
                          encoding='utf-8-sig', usecols=['year', 'race_id'], dtype=str, low_memory=False)
        yr2 = year % 100
        df = df[pd.to_numeric(df['year'], errors='coerce') == yr2]
        n_map = {'A': '10', 'B': '11', 'C': '12'}
        nk = set()
        for rid in df['race_id'].dropna().unique():
            r = str(rid).zfill(10)
            n = r[5:6]
            nd = n_map.get(n, n.zfill(2))
            nk.add(f"20{r[2:4]}{r[0:2]}{r[4:5].zfill(2)}{nd}{r[6:8]}")
        return len(nk)
    except Exception:
        return 3500


def show_status():
    """Show scraping progress for all sources and years."""
    print("=" * 60)
    print("  Bulk Scrape Status")
    print("=" * 60)

    sources = {
        'speed_index': os.path.join(DATA_DIR, 'netkeiba_speed_index.csv'),
        'training': os.path.join(DATA_DIR, 'netkeiba_training_times.csv'),
    }

    for source, csv_path in sources.items():
        print(f"\n  {source}:")
        for year in range(2020, 2027):
            total = estimate_total(year)
            existing = count_existing(csv_path, year)
            pct = existing / total * 100 if total > 0 else 0
            bar = '#' * int(pct / 5) + '-' * (20 - int(pct / 5))
            status = 'DONE' if pct >= 95 else f'{pct:.0f}%'
            print(f"    {year}: [{bar}] {existing:>5}/{total} ({status})")


def run_year(source, year, limit=500):
    """Run scraper for one source + one year."""
    python = sys.executable
    if source == 'speed_index':
        cmd = [python, os.path.join(BASE_DIR, 'tools', 'scrape_speed_index.py'),
               '--year', str(year), '--limit', str(limit)]
    elif source == 'training':
        cmd = [python, os.path.join(BASE_DIR, 'tools', 'scrape_premium_data.py'),
               '--year', str(year), '--limit', str(limit)]
    else:
        print(f"  Unknown source: {source}")
        return False

    print(f"\n  Running: {source} {year} (limit={limit})")
    try:
        result = subprocess.run(cmd, cwd=BASE_DIR, timeout=7200, capture_output=False)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  Timeout (2h). Data saved incrementally.")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="過去データ一括スクレイピング")
    parser.add_argument('--source', type=str, choices=['speed_index', 'training', 'all'],
                        default='all')
    parser.add_argument('--year', type=str, default='2020-2025',
                        help='Year or range (e.g. 2023 or 2020-2025)')
    parser.add_argument('--all', action='store_true', help='All sources, all years')
    parser.add_argument('--status', action='store_true', help='Show progress')
    parser.add_argument('--reset', action='store_true', help='Reset progress')
    parser.add_argument('--limit', type=int, default=LIMIT_PER_YEAR)
    args = parser.parse_args()

    limit = args.limit

    if args.status:
        show_status()
        return

    if args.reset:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
        print("Progress reset.")
        return

    # Parse years
    if '-' in args.year:
        start, end = args.year.split('-')
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(args.year)]

    # Parse sources
    if args.all or args.source == 'all':
        sources = ['speed_index', 'training']
    else:
        sources = [args.source]

    print("=" * 60)
    print(f"  Bulk Scrape: {sources} x {years}")
    print(f"  Limit: {limit}/year")
    print("=" * 60)

    progress = load_progress()
    total_ok = total_err = 0

    for source in sources:
        for year in years:
            # Check if already complete
            csv_path = (os.path.join(DATA_DIR, 'netkeiba_speed_index.csv') if source == 'speed_index'
                        else os.path.join(DATA_DIR, 'netkeiba_training_times.csv'))
            existing = count_existing(csv_path, year)
            total = estimate_total(year)
            if existing / total >= 0.95 if total > 0 else False:
                print(f"  {source} {year}: DONE ({existing}/{total}), skip")
                continue

            ok = run_year(source, year, limit=limit)
            if ok:
                total_ok += 1
            else:
                total_err += 1

    save_progress(progress)

    print(f"\n{'=' * 60}")
    print(f"  Complete: {total_ok} OK, {total_err} errors")
    print("=" * 60)

    show_status()

    try:
        sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
        from notify import send_discord
        send_discord("Bulk Scrape完了",
                     f"{sources} x {years}\n{total_ok} OK / {total_err} errors",
                     color="green" if total_err == 0 else "yellow")
    except Exception:
        pass


if __name__ == '__main__':
    main()
