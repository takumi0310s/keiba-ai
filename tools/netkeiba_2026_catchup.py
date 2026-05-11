#!/usr/bin/env python3
"""netkeiba 2026 csv 不足分 catchup wrapper

QA Audit (Task #10, 2026/05/11) で確認した 2026 完全未取得 csv 群:
  - netkeiba_race_review.csv      (2026: 0)
  - netkeiba_master_index.csv     (2026: 0)  --> bias / race_lap も同梱
  - netkeiba_track_bias.csv       (2026: 0)
  - netkeiba_race_lap.csv         (2026: 0)
  - netkeiba_training_eval.csv    (2026: 0)
  - netkeiba_ai_position.csv      (2026: 0)
  - netkeiba_ai_opinion.csv       (2026: 0)
  - netkeiba_ana_best.csv         (2026: 0)
  - netkeiba_track_index.csv      (2026: 0)
  - netkeiba_shinba_eval.csv      (2026: 0)
  - netkeiba_training_times.csv   (2026: 0)
  - netkeiba_upset_level.csv      (2026: 144 のみ)

本 script は 既存 scrape_*.py を 順次 subprocess で実行する wrapper。
- resume: 各 sub-scraper が 既存 CSV を見て自動 skip するため、 重複取得しない
- dry-run: 実行せず、 取得対象 race_id 件数 を サマリ表示
- rate limit: 各 scraper 内部の DELAY を尊重 (SCRAPER-GUARD 範囲)

V15 投資保護:
  - predict_core / daily_predict / モデル には触らない
  - 既存 scraper を 呼ぶだけ (新規 scraper logic は追加しない)
  - SCRAPER-GUARD 範囲 (Fri22:00-Mon06:00 は自動停止)

Usage:
  python tools/netkeiba_2026_catchup.py --dry-run             # 件数のみ
  python tools/netkeiba_2026_catchup.py                       # 全 csv 取得
  python tools/netkeiba_2026_catchup.py --types review,master # 指定 csv のみ
  python tools/netkeiba_2026_catchup.py --limit 50            # 各 csv 上限 50R
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Iterable

import pandas as pd

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TOOLS_DIR = os.path.join(BASE_DIR, 'tools')
PYTHON_EXE = sys.executable


# =============================================================
# CATCHUP_JOBS: csv_name → (sub-scraper path, argv builder)
#   - argv builder(year, month, limit) -> list[str]
#   - 既存 sub-scraper は 自身の resume / skip ロジックを持つので、
#     本 wrapper は subprocess.run するだけ
# =============================================================

def _argv_master(year: int, month: int, limit: int) -> list[str]:
    """scrape_master_index.py: master_index + track_bias + race_lap を同時取得"""
    a = [PYTHON_EXE, os.path.join(TOOLS_DIR, 'scrape_master_index.py'),
         '--year', str(year)]
    if limit:
        a += ['--limit', str(limit)]
    return a


def _argv_review(year: int, month: int, limit: int) -> list[str]:
    """scrape_race_review.py: race_review"""
    yr2 = year % 100
    a = [PYTHON_EXE, os.path.join(TOOLS_DIR, 'scrape_race_review.py'),
         '--years', str(yr2)]
    if limit:
        a += ['--limit', str(limit)]
    return a


def _argv_super_premium(year: int, month: int, limit: int) -> list[str]:
    """scrape_super_premium.py: ai_position / ai_opinion / ana_best /
    track_index / training_eval を同時取得 (source=both)"""
    a = [PYTHON_EXE, os.path.join(TOOLS_DIR, 'scrape_super_premium.py'),
         '--year', str(year), '--source', 'both']
    if month:
        a += ['--month', str(month)]
    if limit:
        a += ['--limit', str(limit)]
    return a


def _argv_shinba(year: int, month: int, limit: int) -> list[str]:
    """scrape_shinba_eval.py: shinba_eval (新馬戦のみ)"""
    a = [PYTHON_EXE, os.path.join(TOOLS_DIR, 'scrape_shinba_eval.py'),
         '--year', str(year)]
    return a


def _argv_premium(year: int, month: int, limit: int) -> list[str]:
    """scrape_premium_data.py: training_times (調教タイム)"""
    a = [PYTHON_EXE, os.path.join(TOOLS_DIR, 'scrape_premium_data.py'),
         '--year', str(year)]
    if month:
        a += ['--month', str(month)]
    if limit:
        a += ['--limit', str(limit)]
    return a


def _argv_upset(year: int, month: int, limit: int) -> list[str]:
    """bulk_scrape_upset.py: upset_level"""
    a = [PYTHON_EXE, os.path.join(TOOLS_DIR, 'bulk_scrape_upset.py'),
         '--year', str(year)]
    return a


# 1 job = 1 sub-scraper. 1 scraper が複数 csv を埋める場合もある (super_premium 等)
CATCHUP_JOBS = {
    'master': {
        'script': 'scrape_master_index.py',
        'csvs': [
            'netkeiba_master_index.csv',
            'netkeiba_track_bias.csv',
            'netkeiba_race_lap.csv',
        ],
        'argv': _argv_master,
    },
    'review': {
        'script': 'scrape_race_review.py',
        'csvs': ['netkeiba_race_review.csv'],
        'argv': _argv_review,
    },
    'super_premium': {
        'script': 'scrape_super_premium.py',
        'csvs': [
            'netkeiba_ai_position.csv',
            'netkeiba_ai_opinion.csv',
            'netkeiba_ana_best.csv',
            'netkeiba_track_index.csv',
            'netkeiba_training_eval.csv',
        ],
        'argv': _argv_super_premium,
    },
    'shinba': {
        'script': 'scrape_shinba_eval.py',
        'csvs': ['netkeiba_shinba_eval.csv'],
        'argv': _argv_shinba,
    },
    'training_times': {
        'script': 'scrape_premium_data.py',
        'csvs': ['netkeiba_training_times.csv'],
        'argv': _argv_premium,
    },
    'upset': {
        'script': 'bulk_scrape_upset.py',
        'csvs': ['netkeiba_upset_level.csv'],
        'argv': _argv_upset,
    },
}


# =============================================================
# Race-id list (target year)
# =============================================================

def _target_to_netkeiba(target_rid: str) -> str:
    """TARGET JV race_id (10桁) → netkeiba 12桁。 nichi hex (A=10/B=11/C=12)対応"""
    rid = str(target_rid).zfill(10)
    cc = rid[0:2]
    yy = rid[2:4]
    k = rid[4:5]
    n = rid[5:6]
    rr = rid[6:8]
    n_map = {'A': '10', 'B': '11', 'C': '12'}
    n_dec = n_map.get(n, n.zfill(2))
    return f"20{yy}{cc}{k.zfill(2)}{n_dec}{rr}"


def _all_target_race_ids(year: int) -> set[str]:
    """jra_races_full.csv から指定年の netkeiba race_id を抽出"""
    csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
    if not os.path.exists(csv_path):
        print(f'[WARN] {csv_path} not found - cannot count target race_ids')
        return set()
    df = pd.read_csv(
        csv_path, encoding='utf-8-sig',
        usecols=['year', 'race_id'], dtype=str, low_memory=False,
    )
    df['year_int'] = pd.to_numeric(df['year'], errors='coerce')
    yr2 = year % 100
    df = df[df['year_int'] == yr2]
    nk_ids: set[str] = set()
    for rid in df['race_id'].dropna().unique():
        nk_ids.add(_target_to_netkeiba(rid))
    return nk_ids


def _csv_race_ids(csv_path: str) -> set[str]:
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig',
                         usecols=['race_id'], dtype=str, low_memory=False)
        return set(df['race_id'].dropna().astype(str).unique())
    except Exception:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8',
                             usecols=['race_id'], dtype=str, low_memory=False)
            return set(df['race_id'].dropna().astype(str).unique())
        except Exception:
            return set()


# =============================================================
# Dry-run summary
# =============================================================

def dry_run_report(year: int, jobs: Iterable[str]) -> dict[str, dict]:
    """各 csv で 不足 race_id 件数 を計算 (取得は行わない)"""
    all_ids = _all_target_race_ids(year)
    print(f'\n[dry-run] year={year}  total target races: {len(all_ids)}')
    print('=' * 60)
    out: dict[str, dict] = {}
    for job in jobs:
        spec = CATCHUP_JOBS[job]
        print(f'\n[job: {job}] script={spec["script"]}')
        job_out = {'script': spec['script'], 'csvs': {}}
        for csv_name in spec['csvs']:
            csv_path = os.path.join(DATA_DIR, csv_name)
            done = _csv_race_ids(csv_path)
            done_in_year = done & all_ids
            missing = all_ids - done
            print(f'  {csv_name}:')
            print(f'    exists: {os.path.exists(csv_path)}')
            print(f'    {year} done : {len(done_in_year):>5}')
            print(f'    {year} missing: {len(missing):>5}')
            job_out['csvs'][csv_name] = {
                'done_in_year': len(done_in_year),
                'missing_in_year': len(missing),
            }
        out[job] = job_out
    return out


# =============================================================
# Execution
# =============================================================

def run_job(job_key: str, year: int, month: int, limit: int) -> int:
    spec = CATCHUP_JOBS[job_key]
    argv = spec['argv'](year, month, limit)
    print(f'\n[run job: {job_key}] {" ".join(argv)}')
    print('-' * 60)
    t0 = time.time()
    try:
        ret = subprocess.run(argv, cwd=BASE_DIR)
        rc = ret.returncode
    except Exception as e:
        print(f'  [ERROR] subprocess failed: {e}')
        return 1
    dt = time.time() - t0
    print(f'-> rc={rc}  elapsed={dt:.1f}s')
    return rc


# =============================================================
# main
# =============================================================

def main() -> None:
    ap = argparse.ArgumentParser(description='netkeiba 2026 csv catchup wrapper')
    ap.add_argument('--year', type=int, default=2026,
                    help='target year (default: 2026)')
    ap.add_argument('--month', type=int, default=0,
                    help='month filter (0=all, only used by super_premium / training_times)')
    ap.add_argument('--limit', type=int, default=0,
                    help='per-scraper race cap (0=unlimited)')
    ap.add_argument('--types', type=str, default='',
                    help='comma-separated job keys; default=all of: '
                         + ', '.join(CATCHUP_JOBS.keys()))
    ap.add_argument('--dry-run', action='store_true',
                    help='only print missing race_id counts, no scraping')
    ap.add_argument('--list-jobs', action='store_true',
                    help='list available job keys and exit')
    args = ap.parse_args()

    if args.list_jobs:
        print('Available jobs:')
        for k, v in CATCHUP_JOBS.items():
            print(f'  {k}: script={v["script"]}, csvs={", ".join(v["csvs"])}')
        return

    if args.types:
        jobs = [j.strip() for j in args.types.split(',') if j.strip()]
        for j in jobs:
            if j not in CATCHUP_JOBS:
                print(f'[ERROR] unknown job: {j}')
                print(f'  available: {", ".join(CATCHUP_JOBS.keys())}')
                sys.exit(2)
    else:
        jobs = list(CATCHUP_JOBS.keys())

    print('=' * 60)
    print(f'netkeiba {args.year} catchup wrapper')
    print(f'  jobs : {", ".join(jobs)}')
    print(f'  dry  : {args.dry_run}')
    print(f'  limit: {args.limit}')
    print(f'  month: {args.month}')
    print(f'  now  : {datetime.now().isoformat(timespec="seconds")}')
    print('=' * 60)

    report = dry_run_report(args.year, jobs)

    if args.dry_run:
        # サマリ
        print('\n' + '=' * 60)
        print('[dry-run summary]')
        for j, spec in report.items():
            total_missing = sum(c['missing_in_year'] for c in spec['csvs'].values())
            print(f'  {j}: missing_total={total_missing:>5} '
                  f'(across {len(spec["csvs"])} csv)')
        print('=' * 60)
        return

    # SCRAPER-GUARD 確認 (operational caller ではないので、 ガード時間帯は wait)
    try:
        from tools.scraper_guard import is_scraping_allowed  # type: ignore
        if not is_scraping_allowed(caller='netkeiba_2026_catchup'):
            print('[GUARD] 現在 SCRAPER-GUARD 範囲 (Fri22-Mon06)。 即終了します。')
            print('  (各 sub-scraper も自前 guard を持っているため、 ループに入る前に停止)')
            sys.exit(0)
    except ImportError:
        pass

    # 順次実行
    print('\n' + '=' * 60)
    print('[execute jobs]')
    results: dict[str, int] = {}
    t_all0 = time.time()
    for j in jobs:
        rc = run_job(j, args.year, args.month, args.limit)
        results[j] = rc
    dt_all = time.time() - t_all0

    print('\n' + '=' * 60)
    print('[summary]')
    for j, rc in results.items():
        flag = 'OK ' if rc == 0 else 'NG '
        print(f'  {flag} {j} : rc={rc}')
    print(f'  total elapsed: {dt_all/60:.1f} min')
    print('=' * 60)


if __name__ == '__main__':
    main()
