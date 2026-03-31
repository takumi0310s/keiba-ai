#!/usr/bin/env python
"""JRDB一括データダウンロード＆解凍スクリプト

2020-2026年のJRDBデータをZIP形式でダウンロードし、解凍する。

データ種類:
  - Paci: 前日データ（IDM、展開予想指数等）
  - Sed:  成績データ（馬場差、ペース指数等）
  - Tyb:  直前データ（パドック指数、オッズ指数等）
  - Kyi:  競走馬データ
  - Bac:  レース番組データ

Usage:
    python tools/download_jrdb.py
    python tools/download_jrdb.py --types Paci Sed
    python tools/download_jrdb.py --years 2024 2025
"""
import os
import sys
import io
import time
import zipfile
import argparse
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
from bs4 import BeautifulSoup

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'jrdb', 'raw')
EXTRACT_DIR = os.path.join(BASE_DIR, 'data', 'jrdb', 'extracted')

# JRDB URL base (HTTP: JRDB's SSL cert has hostname mismatch, Basic Auth works over HTTP)
JRDB_BASE = 'http://www.jrdb.com/member'

# Data types with yearly archive support
YEARLY_TYPES = ['Sed', 'Tyb', 'Kyi', 'Bac', 'Hjc', 'Cyb', 'Skb', 'Kab', 'Ukc', 'Kka']
# Data types requiring individual file download
INDIVIDUAL_TYPES = ['Paci']

ALL_TYPES = YEARLY_TYPES + INDIVIDUAL_TYPES
YEARS = list(range(2020, 2027))  # 2020-2026

import warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')


def load_credentials():
    env_path = os.path.join(BASE_DIR, '.env')
    jrdb_id = jrdb_pw = ''
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('JRDB_ID='):
                jrdb_id = line.split('=', 1)[1].strip('"').strip("'")
            elif line.startswith('JRDB_PASSWORD='):
                jrdb_pw = line.split('=', 1)[1].strip('"').strip("'")
    return jrdb_id, jrdb_pw


def download_file(session, url, dest_path, auth):
    """Download a file with retry logic."""
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        if size > 100:  # Skip if already downloaded (>100 bytes = not error page)
            return True, 'skip'

    for attempt in range(3):
        try:
            resp = session.get(url, auth=auth, timeout=60, verify=True, stream=True)
            if resp.status_code == 404:
                return False, '404'
            if resp.status_code == 401:
                return False, '401'
            if resp.status_code != 200:
                if attempt < 2:
                    time.sleep(5)
                    continue
                return False, f'{resp.status_code}'

            content_len = int(resp.headers.get('content-length', 0))
            if content_len < 100:
                return False, 'empty'

            with open(dest_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True, f'{os.path.getsize(dest_path):,}B'
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
                continue
            return False, str(e)

    return False, 'max_retries'


def extract_zip(zip_path, extract_to):
    """Extract a ZIP file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
            return len(zf.namelist())
    except zipfile.BadZipFile:
        return -1
    except Exception:
        return -1


def get_paci_dates(session, auth, year):
    """Get all PACI file dates for a given year from the index page."""
    url = f'{JRDB_BASE}/datazip/Paci/index.html'
    resp = session.get(url, auth=auth, timeout=15, verify=True)
    resp.encoding = 'shift_jis'
    soup = BeautifulSoup(resp.text, 'html.parser')

    dates = []
    yy = str(year)[2:]  # 2020 -> "20"
    for a in soup.find_all('a', href=True):
        text = a.get_text(strip=True)
        href = a['href']
        if f'PACI{yy}' in text and '.zip' in text:
            dates.append((text, href))
    return dates


def download_yearly_archives(session, auth, data_types, years):
    """Download yearly ZIP archives for types that support them."""
    results = {}

    for dtype in data_types:
        if dtype not in YEARLY_TYPES:
            continue

        dtype_upper = dtype.upper()
        type_dir = os.path.join(RAW_DIR, dtype)
        extract_type_dir = os.path.join(EXTRACT_DIR, dtype)
        os.makedirs(type_dir, exist_ok=True)
        os.makedirs(extract_type_dir, exist_ok=True)

        print(f'\n{"="*50}')
        print(f'  {dtype_upper} - Yearly Archives')
        print(f'{"="*50}')

        type_results = {'downloaded': 0, 'skipped': 0, 'failed': 0, 'extracted': 0}

        for year in years:
            if year == 2026:
                # 2026: download individual files
                print(f'  {year}: individual files...')
                idx_url = f'{JRDB_BASE}/datazip/{dtype}/index.html'
                resp = session.get(idx_url, auth=auth, timeout=15, verify=True)
                resp.encoding = 'shift_jis'
                soup = BeautifulSoup(resp.text, 'html.parser')

                count = 0
                for a in soup.find_all('a', href=True):
                    text = a.get_text(strip=True)
                    href = a['href']
                    if f'{dtype_upper}26' in text and '.zip' in text:
                        file_url = f'{JRDB_BASE}/datazip/{dtype}/{href}'
                        dest = os.path.join(type_dir, text)
                        ok, msg = download_file(session, file_url, dest, auth)
                        if ok:
                            if msg == 'skip':
                                type_results['skipped'] += 1
                            else:
                                type_results['downloaded'] += 1
                                n = extract_zip(dest, extract_type_dir)
                                if n > 0:
                                    type_results['extracted'] += n
                                count += 1
                        else:
                            type_results['failed'] += 1
                        time.sleep(0.5)

                print(f'    → {count} files downloaded')
            else:
                # 2020-2025: yearly archive
                filename = f'{dtype_upper}_{year}.zip'
                file_url = f'{JRDB_BASE}/datazip/{dtype}/{filename}'
                dest = os.path.join(type_dir, filename)

                print(f'  {year}: {filename}...', end=' ', flush=True)
                ok, msg = download_file(session, file_url, dest, auth)

                if ok:
                    if msg == 'skip':
                        print(f'already exists')
                        type_results['skipped'] += 1
                        # Still count extracted files
                        if os.path.exists(dest):
                            try:
                                with zipfile.ZipFile(dest, 'r') as zf:
                                    type_results['extracted'] += len(zf.namelist())
                            except:
                                pass
                    else:
                        print(f'OK ({msg})')
                        type_results['downloaded'] += 1
                        n = extract_zip(dest, extract_type_dir)
                        if n > 0:
                            type_results['extracted'] += n
                            print(f'    → extracted {n} files')
                        else:
                            print(f'    → extraction failed')
                else:
                    print(f'FAILED ({msg})')
                    type_results['failed'] += 1

                time.sleep(1)

        results[dtype] = type_results
        print(f'  Summary: DL={type_results["downloaded"]}, Skip={type_results["skipped"]}, '
              f'Fail={type_results["failed"]}, Files={type_results["extracted"]}')

    return results


def download_individual_files(session, auth, data_types, years):
    """Download individual date ZIP files for types without yearly archives."""
    results = {}

    for dtype in data_types:
        if dtype not in INDIVIDUAL_TYPES:
            continue

        dtype_upper = dtype.upper()
        type_dir = os.path.join(RAW_DIR, dtype)
        extract_type_dir = os.path.join(EXTRACT_DIR, dtype)
        os.makedirs(type_dir, exist_ok=True)
        os.makedirs(extract_type_dir, exist_ok=True)

        print(f'\n{"="*50}')
        print(f'  {dtype_upper} - Individual Files')
        print(f'{"="*50}')

        type_results = {'downloaded': 0, 'skipped': 0, 'failed': 0, 'extracted': 0}

        for year in years:
            dates = get_paci_dates(session, auth, year)
            print(f'  {year}: {len(dates)} files found')

            for i, (filename, href) in enumerate(dates):
                file_url = f'{JRDB_BASE}/datazip/Paci/{href}'
                dest = os.path.join(type_dir, filename)
                ok, msg = download_file(session, file_url, dest, auth)

                if ok:
                    if msg == 'skip':
                        type_results['skipped'] += 1
                    else:
                        type_results['downloaded'] += 1
                        n = extract_zip(dest, extract_type_dir)
                        if n > 0:
                            type_results['extracted'] += n

                        if (i + 1) % 20 == 0:
                            print(f'    [{i+1}/{len(dates)}] downloaded...')
                else:
                    type_results['failed'] += 1

                time.sleep(0.3)

            print(f'    → {year} done')

        results[dtype] = type_results
        print(f'  Summary: DL={type_results["downloaded"]}, Skip={type_results["skipped"]}, '
              f'Fail={type_results["failed"]}, Files={type_results["extracted"]}')

    return results


def print_report(all_results):
    """Print final download report."""
    print(f'\n{"="*60}')
    print(f'  JRDB DATA DOWNLOAD REPORT')
    print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*60}')

    total_dl = total_skip = total_fail = total_files = 0

    for dtype, r in sorted(all_results.items()):
        print(f'\n  {dtype.upper()}:')
        print(f'    Downloaded: {r["downloaded"]}')
        print(f'    Skipped:    {r["skipped"]}')
        print(f'    Failed:     {r["failed"]}')
        print(f'    Extracted:  {r["extracted"]} files')
        total_dl += r['downloaded']
        total_skip += r['skipped']
        total_fail += r['failed']
        total_files += r['extracted']

    print(f'\n  {"─"*40}')
    print(f'  TOTAL: {total_dl} downloaded, {total_skip} skipped, {total_fail} failed')
    print(f'  TOTAL extracted files: {total_files}')

    # Show disk usage
    raw_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(RAW_DIR)
        for f in fns
    )
    ext_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(EXTRACT_DIR)
        for f in fns
    )
    print(f'  Disk: raw={raw_size/1024/1024:.1f}MB, extracted={ext_size/1024/1024:.1f}MB')
    print(f'{"="*60}')


def main():
    parser = argparse.ArgumentParser(description='JRDB data downloader')
    parser.add_argument('--types', nargs='+', default=ALL_TYPES,
                        help=f'Data types to download (default: {ALL_TYPES})')
    parser.add_argument('--years', nargs='+', type=int, default=YEARS,
                        help=f'Years to download (default: {YEARS})')
    args = parser.parse_args()

    jrdb_id, jrdb_pw = load_credentials()
    if not jrdb_id or not jrdb_pw:
        print('ERROR: JRDB_ID / JRDB_PASSWORD not found in .env')
        sys.exit(1)

    auth = HTTPBasicAuth(jrdb_id, jrdb_pw)
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    # Test auth
    print(f'JRDB Login: ID={jrdb_id[:4]}****')
    resp = session.get(f'{JRDB_BASE}/data/index.html', auth=auth, timeout=15, verify=True)
    if resp.status_code != 200:
        print(f'ERROR: Login failed (status={resp.status_code})')
        sys.exit(1)
    print(f'Login OK')

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    print(f'\nTypes: {args.types}')
    print(f'Years: {args.years}')

    all_results = {}

    # Phase 1: Yearly archives (Sed, Tyb, Kyi, Bac)
    yearly_types = [t for t in args.types if t in YEARLY_TYPES]
    if yearly_types:
        results = download_yearly_archives(session, auth, yearly_types, args.years)
        all_results.update(results)

    # Phase 2: Individual files (Paci)
    individual_types = [t for t in args.types if t in INDIVIDUAL_TYPES]
    if individual_types:
        results = download_individual_files(session, auth, individual_types, args.years)
        all_results.update(results)

    print_report(all_results)


if __name__ == '__main__':
    main()
