#!/usr/bin/env python
"""毎日自動実行のプレミアムデータ分散取得スクリプト

各データソースを年ごとに順次取得し、進捗をdata/scrape_progress.jsonに保存。
次回実行時は前回の続きから自動再開。

取得対象:
  - speed_index: 2020-2026年のタイム指数（各年500レース/日）
  - newspaper: 2020-2026年のAI展開予測/調教評価/AI見解
  - result: 2020-2026年の馬場指数

Usage:
    python tools/daily_premium_scrape.py              # 全ソース実行
    python tools/daily_premium_scrape.py --dry-run    # 実行計画のみ表示
    python tools/daily_premium_scrape.py --limit 100  # 各ソース100件に制限
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
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# 取得対象年（古い方から順に）
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
DAILY_LIMIT = 500  # 1ソース1日あたりの最大レース数


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'speed_index': {'completed_years': [], 'current_year': None},
        'training_times': {'completed_years': [], 'current_year': None},
        'history': [],
    }


def save_progress(progress):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def count_existing(csv_path, year):
    """Count how many races for a given year are already in the CSV."""
    if not os.path.exists(csv_path):
        return 0
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, encoding='utf-8-sig', usecols=['race_id'], dtype=str)
        year_str = str(year)
        return len(df[df['race_id'].str.startswith(year_str)]['race_id'].unique())
    except Exception:
        return 0


def estimate_total(year):
    """Estimate total races for a given year from jra_races_full.csv.

    TARGET race_id is 10 digits: CC(2)+YY(2)+K(1)+N(1)+RR(2)+UU(2).
    Race-level ID is the first 8 digits (without UU/umaban).
    Convert to netkeiba format to count unique races.
    """
    try:
        import pandas as pd
        csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
        df = pd.read_csv(csv_path, encoding='utf-8-sig',
                          usecols=['year', 'race_id'], dtype=str, low_memory=False)
        yr2 = year % 100
        df['year_int'] = pd.to_numeric(df['year'], errors='coerce')
        df = df[df['year_int'] == yr2]
        # Convert to netkeiba race-level ID (strip UU/umaban)
        # N can be hex: A=10, B=11, C=12
        n_map = {'A': '10', 'B': '11', 'C': '12'}
        nk_ids = set()
        for rid in df['race_id'].dropna().unique():
            rid = str(rid).zfill(10)
            n = rid[5:6]
            n_dec = n_map.get(n, n.zfill(2))
            nk_id = f"20{rid[2:4]}{rid[0:2]}{rid[4:5].zfill(2)}{n_dec}{rid[6:8]}"
            nk_ids.add(nk_id)
        return len(nk_ids)
    except Exception:
        return 3500  # default estimate


def get_next_year(source, progress):
    """Determine the next year to scrape for a source."""
    completed = progress[source]['completed_years']
    for year in YEARS:
        if year not in completed:
            return year
    return None


def run_scraper(source, year, limit, dry_run=False):
    """Run the appropriate scraper for the given source and year."""
    python = sys.executable

    if source == 'speed_index':
        cmd = [python, os.path.join(BASE_DIR, 'tools', 'scrape_speed_index.py'),
               '--year', str(year), '--limit', str(limit)]
    elif source == 'training_times':
        cmd = [python, os.path.join(BASE_DIR, 'tools', 'scrape_premium_data.py'),
               '--year', str(year), '--limit', str(limit)]
    else:
        print(f"  Unknown source: {source}")
        return False

    print(f"\n{'='*60}")
    print(f"  Running: {source} year={year} limit={limit}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return True

    try:
        result = subprocess.run(cmd, cwd=BASE_DIR, timeout=3600,
                                capture_output=False)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  WARNING: Scraper timed out after 1 hour")
        return True  # partial success, data is saved incrementally
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def check_year_complete(source, year):
    """Check if a year is fully scraped."""
    total = estimate_total(year)

    if source == 'speed_index':
        existing = count_existing(os.path.join(DATA_DIR, 'netkeiba_speed_index.csv'), year)
    elif source == 'training_times':
        existing = count_existing(os.path.join(DATA_DIR, 'netkeiba_training_times.csv'), year)
    else:
        return False

    # Consider complete if we have at least 95% of races
    # (some races may not have data, e.g. cancelled races)
    completeness = existing / total if total > 0 else 0
    print(f"  {source} {year}: {existing}/{total} races ({completeness:.0%})")
    return completeness >= 0.95


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Show plan without executing')
    parser.add_argument('--limit', type=int, default=DAILY_LIMIT, help='Max races per source')
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    now = datetime.now()
    print("=" * 60)
    print(f"  Daily Premium Scrape - {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Limit: {args.limit} races per source")
    print("=" * 60)

    progress = load_progress()

    # Process each source
    sources = ['speed_index', 'training_times']
    run_summary = []

    for source in sources:
        year = get_next_year(source, progress)
        if year is None:
            print(f"\n  {source}: All years completed!")
            run_summary.append(f"{source}: all done")
            continue

        # Check if current year is already complete
        if check_year_complete(source, year):
            print(f"  {source} {year}: Already complete, marking done")
            if year not in progress[source]['completed_years']:
                progress[source]['completed_years'].append(year)
            progress[source]['current_year'] = None
            save_progress(progress)
            # Move to next year
            year = get_next_year(source, progress)
            if year is None:
                run_summary.append(f"{source}: all done")
                continue

        progress[source]['current_year'] = year
        save_progress(progress)

        # Run scraper
        success = run_scraper(source, year, args.limit, args.dry_run)

        if not args.dry_run:
            # Check if year is now complete
            if check_year_complete(source, year):
                if year not in progress[source]['completed_years']:
                    progress[source]['completed_years'].append(year)
                progress[source]['current_year'] = None
                print(f"  {source} {year}: COMPLETED!")

        run_summary.append(f"{source}: year={year} {'OK' if success else 'FAIL'}")
        save_progress(progress)

    # Record history
    progress['history'].append({
        'date': now.strftime('%Y-%m-%d %H:%M'),
        'limit': args.limit,
        'results': run_summary,
    })
    # Keep only last 30 entries
    progress['history'] = progress['history'][-30:]
    save_progress(progress)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  DAILY SCRAPE SUMMARY")
    print(f"{'='*60}")
    for s in run_summary:
        print(f"  {s}")
    print(f"\n  Progress saved to: {PROGRESS_FILE}")

    # Show overall status
    print(f"\n  Overall Status:")
    for source in sources:
        completed = progress[source]['completed_years']
        current = progress[source]['current_year']
        remaining = [y for y in YEARS if y not in completed and y != current]
        print(f"    {source}: done={completed}, current={current}, remaining={remaining}")

    print(f"{'='*60}")


if __name__ == '__main__':
    main()
