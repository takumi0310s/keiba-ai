#!/usr/bin/env python3
"""JRDB追加データ Batch2 ダウンロード＆パース
対象: SRB(成績レース), CHA(追切), OZ/OW/OT/OV(オッズ), HJC(払戻), KAA(開催), KTA(登録馬)

Usage:
    python tools/download_parse_jrdb_batch2.py
    python tools/download_parse_jrdb_batch2.py --skip-download
    python tools/download_parse_jrdb_batch2.py --types srb hjc
"""
import os
import sys
import io
import time
import zipfile
import subprocess
import argparse
import glob
import requests
import pandas as pd
from requests.auth import HTTPBasicAuth
from datetime import datetime
from bs4 import BeautifulSoup

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'jrdb', 'raw')
EXTRACT_DIR = os.path.join(BASE_DIR, 'data', 'jrdb', 'extracted')
DATA_DIR = os.path.join(BASE_DIR, 'data')
JRDB_BASE = 'http://www.jrdb.com/member'
YEARS = list(range(2020, 2027))
SEVEN_ZIP = r'C:\Program Files\7-Zip\7z.exe'

ALL_TYPES = ['srb', 'cha', 'oz', 'ow', 'ot', 'ov', 'hjc', 'kaa', 'kta']


# =============================================================================
# Shared helpers
# =============================================================================
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
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100:
        return True, 'skip'
    for attempt in range(3):
        try:
            resp = session.get(url, auth=auth, timeout=60, verify=True, stream=True)
            if resp.status_code in (404, 401):
                return False, str(resp.status_code)
            if resp.status_code != 200:
                if attempt < 2:
                    time.sleep(5)
                    continue
                return False, str(resp.status_code)
            clen = int(resp.headers.get('content-length', 0))
            if clen < 100:
                return False, 'empty'
            with open(dest_path, 'wb') as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return True, f'{os.path.getsize(dest_path):,}B'
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
                continue
            return False, str(e)
    return False, 'max_retries'


def extract_zip(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
            return len(zf.namelist())
    except Exception:
        return -1


def extract_lzh(lzh_path, extract_to):
    """Extract LZH file using 7-Zip."""
    try:
        result = subprocess.run(
            [SEVEN_ZIP, 'x', '-y', f'-o{extract_to}', lzh_path],
            capture_output=True, timeout=60
        )
        return result.returncode == 0
    except Exception:
        return False


def _hex_day(c):
    try:
        return int(c, 16)
    except (ValueError, TypeError):
        return 0


def _safe_float(s):
    s = s.strip()
    if not s or s == '-':
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_int(s):
    s = s.strip()
    if not s or s == '-':
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _field(line_bytes, start, length):
    s = start - 1
    return line_bytes[s:s+length].decode('cp932', errors='replace')


def _build_race_id(basho, year, kai, nichi, race_num):
    yr2 = year.strip()
    yr4 = f'20{yr2}' if len(yr2) == 2 else yr2
    return f'{yr4}{basho.strip()}{int(kai):02d}{_hex_day(nichi):02d}{int(race_num):02d}'


def _build_kaisai_key(basho, year, kai, nichi):
    yr2 = year.strip()
    yr4 = f'20{yr2}' if len(yr2) == 2 else yr2
    return f'{yr4}{basho.strip()}{int(kai):02d}{_hex_day(nichi):02d}'


def download_yearly_zips(session, auth, datazip_name, raw_dir, ext_dir, prefix, years):
    """Download yearly ZIP archives + 2026 individual files."""
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ext_dir, exist_ok=True)
    stats = {'downloaded': 0, 'skipped': 0, 'failed': 0, 'extracted': 0}

    for year in years:
        if year == 2026:
            # Get individual file list from index
            idx_url = f'{JRDB_BASE}/datazip/{datazip_name}/index.html'
            resp = session.get(idx_url, auth=auth, timeout=15, verify=True)
            resp.encoding = 'shift_jis'
            soup = BeautifulSoup(resp.text, 'html.parser')
            yy = '26'
            for a in soup.find_all('a', href=True):
                text = a.get_text(strip=True)
                href = a['href']
                if f'{prefix}{yy}' in text and '.zip' in text:
                    file_url = f'{JRDB_BASE}/datazip/{datazip_name}/{href}'
                    dest = os.path.join(raw_dir, os.path.basename(href))
                    ok, msg = download_file(session, file_url, dest, auth)
                    if ok:
                        if msg == 'skip':
                            stats['skipped'] += 1
                        else:
                            stats['downloaded'] += 1
                            n = extract_zip(dest, ext_dir)
                            if n > 0:
                                stats['extracted'] += n
                    else:
                        stats['failed'] += 1
                    time.sleep(0.3)
        else:
            filename = f'{prefix}_{year}.zip'
            file_url = f'{JRDB_BASE}/datazip/{datazip_name}/{filename}'
            dest = os.path.join(raw_dir, filename)
            print(f'  {year}: {filename}...', end=' ', flush=True)
            ok, msg = download_file(session, file_url, dest, auth)
            if ok:
                if msg == 'skip':
                    print('exists')
                    stats['skipped'] += 1
                else:
                    print(f'OK ({msg})')
                    stats['downloaded'] += 1
                    n = extract_zip(dest, ext_dir)
                    if n > 0:
                        stats['extracted'] += n
                        print(f'    → extracted {n} files')
            else:
                print(f'FAILED ({msg})')
                stats['failed'] += 1
            time.sleep(1)
    return stats


def download_yearly_lzh(session, auth, data_name, raw_dir, ext_dir, prefix, years):
    """Download yearly LZH archives + individual files, extract with 7-Zip."""
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ext_dir, exist_ok=True)
    stats = {'downloaded': 0, 'skipped': 0, 'failed': 0, 'extracted': 0}

    for year in years:
        filename = f'{prefix}_{year}.lzh'
        file_url = f'{JRDB_BASE}/data/{data_name}/{filename}'
        dest = os.path.join(raw_dir, filename)
        print(f'  {year}: {filename}...', end=' ', flush=True)
        ok, msg = download_file(session, file_url, dest, auth)
        if ok:
            if msg == 'skip':
                print('exists')
                stats['skipped'] += 1
            else:
                print(f'OK ({msg})')
                stats['downloaded'] += 1
            # Extract with 7-Zip
            if extract_lzh(dest, ext_dir):
                stats['extracted'] += 1
            else:
                print(f'    → LZH extraction failed')
        else:
            # Try individual files for this year
            if year >= 2025:
                print(f'no yearly, trying individual...')
                idx_url = f'{JRDB_BASE}/data/{data_name}/index.html'
                resp = session.get(idx_url, auth=auth, timeout=15, verify=True)
                text = resp.content.decode('cp932', errors='replace')
                soup = BeautifulSoup(text, 'html.parser')
                yy = str(year)[2:]
                count = 0
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    t = a.get_text(strip=True)
                    if f'{prefix}{yy}' in t and '.lzh' in href:
                        furl = f'{JRDB_BASE}/data/{data_name}/{href}'
                        fdest = os.path.join(raw_dir, os.path.basename(href))
                        fok, fmsg = download_file(session, furl, fdest, auth)
                        if fok:
                            if fmsg != 'skip':
                                stats['downloaded'] += 1
                                count += 1
                            else:
                                stats['skipped'] += 1
                            extract_lzh(fdest, ext_dir)
                        else:
                            stats['failed'] += 1
                        time.sleep(0.3)
                print(f'    → {count} individual files downloaded')
            else:
                print(f'FAILED ({msg})')
                stats['failed'] += 1
        time.sleep(0.5)
    return stats


def download_individual_zips(session, auth, datazip_name, raw_dir, ext_dir, prefix, years):
    """Download individual ZIP files (no yearly archives available)."""
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ext_dir, exist_ok=True)
    stats = {'downloaded': 0, 'skipped': 0, 'failed': 0, 'extracted': 0}

    idx_url = f'{JRDB_BASE}/datazip/{datazip_name}/index.html'
    resp = session.get(idx_url, auth=auth, timeout=15, verify=True)
    if resp.status_code != 200:
        # Fallback: get from main data page
        idx_url = f'{JRDB_BASE}/data/index.html'
        resp = session.get(idx_url, auth=auth, timeout=15, verify=True)

    text = resp.content.decode('cp932', errors='replace')
    soup = BeautifulSoup(text, 'html.parser')

    target_years_yy = [str(y)[2:] for y in years]
    for a in soup.find_all('a', href=True):
        href = a['href']
        text_val = a.get_text(strip=True)
        if prefix not in text_val or '.zip' not in href:
            continue
        # Check year
        matched = False
        for yy in target_years_yy:
            if f'{prefix}{yy}' in text_val:
                matched = True
                break
        if not matched:
            continue

        if 'datazip' in href:
            file_url = f'{JRDB_BASE}/{href.lstrip("../")}'
        else:
            file_url = f'{JRDB_BASE}/datazip/{datazip_name}/{href}'
        dest = os.path.join(raw_dir, os.path.basename(href))
        ok, msg = download_file(session, file_url, dest, auth)
        if ok:
            if msg == 'skip':
                stats['skipped'] += 1
            else:
                stats['downloaded'] += 1
                n = extract_zip(dest, ext_dir)
                if n > 0:
                    stats['extracted'] += n
        else:
            stats['failed'] += 1
        time.sleep(0.3)

    return stats


def parse_files(file_pattern, parse_func, min_line_len, dedup_cols=None):
    files = sorted(glob.glob(file_pattern))
    print(f'  {len(files)} files found')
    rows = []
    errors = 0
    for fpath in files:
        try:
            with open(fpath, 'rb') as f:
                for line_bytes in f:
                    line_bytes = line_bytes.rstrip(b'\r\n')
                    if len(line_bytes) < min_line_len:
                        continue
                    try:
                        rows.append(parse_func(line_bytes))
                    except Exception:
                        errors += 1
        except Exception:
            errors += 1
    print(f'  Parsed: {len(rows):,} rows ({errors} errors)')
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if dedup_cols:
        before = len(df)
        df = df.drop_duplicates(subset=dedup_cols, keep='last')
        if before != len(df):
            print(f'  Deduplicated: {before:,} → {len(df):,}')
    return df


# =============================================================================
# SRB parser (成績レースデータ, record=850 bytes)
# =============================================================================
def parse_srb_line(line_bytes):
    row = {}
    # レースキー
    row['basho_code'] = _field(line_bytes, 1, 2)
    row['year'] = _field(line_bytes, 3, 2)
    row['kai'] = _field(line_bytes, 5, 1)
    row['nichi'] = _field(line_bytes, 6, 1)
    row['race_num'] = _field(line_bytes, 7, 2)

    row['race_id'] = _build_race_id(
        row['basho_code'], row['year'], row['kai'], row['nichi'], row['race_num']
    )

    # ハロンタイム (18区間 x 3バイト, 0.1秒単位)
    furlong_times = []
    for i in range(18):
        pos = 9 + i * 3
        val = _field(line_bytes, pos, 3)
        ft = _safe_float(val)
        if ft is not None:
            ft = ft / 10.0
        furlong_times.append(ft)
    row['furlong_times'] = ','.join(str(f) if f is not None else '' for f in furlong_times)

    # コーナー通過順位 (各64バイト テキスト)
    row['corner1_order'] = _field(line_bytes, 63, 64).strip()
    row['corner2_order'] = _field(line_bytes, 127, 64).strip()
    row['corner3_order'] = _field(line_bytes, 191, 64).strip()
    row['corner4_order'] = _field(line_bytes, 255, 64).strip()

    # ペースアップ位置
    row['pace_up_pos'] = _safe_int(_field(line_bytes, 319, 2))

    # トラックバイアス (内/中/外)
    row['bias_1corner'] = _field(line_bytes, 321, 3).strip()
    row['bias_2corner'] = _field(line_bytes, 324, 3).strip()
    row['bias_backstr'] = _field(line_bytes, 327, 3).strip()
    row['bias_3corner'] = _field(line_bytes, 330, 3).strip()
    row['bias_4corner'] = _field(line_bytes, 333, 5).strip()
    row['bias_straight'] = _field(line_bytes, 338, 5).strip()

    # レースコメント (500バイト)
    row['race_comment'] = _field(line_bytes, 343, 500).strip()

    for k in ['basho_code', 'year', 'kai', 'nichi', 'race_num']:
        del row[k]
    return row


# =============================================================================
# CHA parser (追切本データ, record=62 bytes)
# =============================================================================
def parse_cha_line(line_bytes):
    row = {}
    row['basho_code'] = _field(line_bytes, 1, 2)
    row['year'] = _field(line_bytes, 3, 2)
    row['kai'] = _field(line_bytes, 5, 1)
    row['nichi'] = _field(line_bytes, 6, 1)
    row['race_num'] = _field(line_bytes, 7, 2)
    row['umaban'] = _safe_int(_field(line_bytes, 9, 2))

    row['race_id'] = _build_race_id(
        row['basho_code'], row['year'], row['kai'], row['nichi'], row['race_num']
    )

    row['youbi'] = _field(line_bytes, 11, 2).strip()
    row['oikiri_date'] = _field(line_bytes, 13, 8).strip()
    row['oikiri_count'] = _safe_int(_field(line_bytes, 21, 1))
    row['oikiri_course'] = _field(line_bytes, 22, 2).strip()
    row['oikiri_shurui'] = _safe_int(_field(line_bytes, 24, 1))  # 1:併せ 2:単走 3:馬なり
    row['oikiri_aite'] = _safe_int(_field(line_bytes, 25, 2))    # 相手コード
    row['oikiri_rank'] = _safe_int(_field(line_bytes, 27, 1))    # 1:先着 2:同入 3:遅れ等
    row['oikiri_naiyou'] = _safe_int(_field(line_bytes, 28, 1))  # 追切内容

    # タイム (テン・中間・終い、各3バイト)
    row['ten_time'] = _safe_int(_field(line_bytes, 29, 3))
    row['chukan_time'] = _safe_int(_field(line_bytes, 32, 3))
    row['shimai_time'] = _safe_int(_field(line_bytes, 35, 3))
    row['ten_time_idx'] = _safe_int(_field(line_bytes, 38, 3))
    row['chukan_time_idx'] = _safe_int(_field(line_bytes, 41, 3))
    row['shimai_time_idx'] = _safe_int(_field(line_bytes, 44, 3))
    row['oikiri_idx'] = _safe_int(_field(line_bytes, 47, 3))

    # 併せ馬データ
    row['awase_result'] = _field(line_bytes, 50, 1).strip()
    row['awase_shurui'] = _safe_int(_field(line_bytes, 51, 1))
    row['awase_nenrei'] = _safe_int(_field(line_bytes, 52, 2))
    row['awase_class'] = _field(line_bytes, 54, 2).strip()

    for k in ['basho_code', 'year', 'kai', 'nichi', 'race_num']:
        del row[k]
    return row


# =============================================================================
# OZ parser (基準オッズ, record=955 bytes)
# 単勝18頭 x 5B + 複勝18 x 5B + 馬連153 x 5B = 955
# =============================================================================
def parse_oz_line(line_bytes):
    row = {}
    row['basho_code'] = _field(line_bytes, 1, 2)
    row['year'] = _field(line_bytes, 3, 2)
    row['kai'] = _field(line_bytes, 5, 1)
    row['nichi'] = _field(line_bytes, 6, 1)
    row['race_num'] = _field(line_bytes, 7, 2)
    row['race_id'] = _build_race_id(
        row['basho_code'], row['year'], row['kai'], row['nichi'], row['race_num']
    )
    row['num_horses'] = _safe_int(_field(line_bytes, 9, 2))

    # 単勝オッズ (18頭 x 5バイト)
    for i in range(18):
        val = _safe_float(_field(line_bytes, 11 + i * 5, 5))
        row[f'tansho_{i+1:02d}'] = val

    # 複勝オッズ (18頭 x 5バイト)
    for i in range(18):
        val = _safe_float(_field(line_bytes, 101 + i * 5, 5))
        row[f'fukusho_{i+1:02d}'] = val

    # 馬連オッズ (153組 x 5バイト) → 上位のみ保存
    umaren_odds = []
    for i in range(153):
        val = _safe_float(_field(line_bytes, 191 + i * 5, 5))
        umaren_odds.append(val)
    row['umaren_min'] = min((v for v in umaren_odds if v and v > 0), default=None)
    row['umaren_max'] = max((v for v in umaren_odds if v and v > 0), default=None)

    for k in ['basho_code', 'year', 'kai', 'nichi', 'race_num']:
        del row[k]
    return row


# =============================================================================
# OW parser (ワイドオッズ, record=778 bytes)
# ワイド153組 x 5B
# =============================================================================
def parse_ow_line(line_bytes):
    row = {}
    row['basho_code'] = _field(line_bytes, 1, 2)
    row['year'] = _field(line_bytes, 3, 2)
    row['kai'] = _field(line_bytes, 5, 1)
    row['nichi'] = _field(line_bytes, 6, 1)
    row['race_num'] = _field(line_bytes, 7, 2)
    row['race_id'] = _build_race_id(
        row['basho_code'], row['year'], row['kai'], row['nichi'], row['race_num']
    )
    row['num_horses'] = _safe_int(_field(line_bytes, 9, 2))

    # ワイドオッズ (153組 x 5バイト)
    wide_odds = []
    for i in range(153):
        val = _safe_float(_field(line_bytes, 11 + i * 5, 5))
        wide_odds.append(val)
    row['wide_min'] = min((v for v in wide_odds if v and v > 0), default=None)
    row['wide_max'] = max((v for v in wide_odds if v and v > 0), default=None)
    row['wide_median'] = None
    valid = sorted(v for v in wide_odds if v and v > 0)
    if valid:
        row['wide_median'] = valid[len(valid) // 2]

    for k in ['basho_code', 'year', 'kai', 'nichi', 'race_num']:
        del row[k]
    return row


# =============================================================================
# OT parser (3連複オッズ, record=4910 bytes)
# 816組 x 6B
# =============================================================================
def parse_ot_line(line_bytes):
    row = {}
    row['basho_code'] = _field(line_bytes, 1, 2)
    row['year'] = _field(line_bytes, 3, 2)
    row['kai'] = _field(line_bytes, 5, 1)
    row['nichi'] = _field(line_bytes, 6, 1)
    row['race_num'] = _field(line_bytes, 7, 2)
    row['race_id'] = _build_race_id(
        row['basho_code'], row['year'], row['kai'], row['nichi'], row['race_num']
    )
    row['num_horses'] = _safe_int(_field(line_bytes, 9, 2))

    # 3連複オッズ summary (816組 x 6B = huge, store summary only)
    trio_odds = []
    for i in range(816):
        val = _safe_float(_field(line_bytes, 11 + i * 6, 6))
        if val and val > 0 and val < 9999:
            trio_odds.append(val)
    row['trio_count'] = len(trio_odds)
    row['trio_min'] = min(trio_odds) if trio_odds else None
    row['trio_max'] = max(trio_odds) if trio_odds else None
    if trio_odds:
        trio_odds.sort()
        row['trio_median'] = trio_odds[len(trio_odds) // 2]
        row['trio_p10'] = trio_odds[len(trio_odds) // 10] if len(trio_odds) >= 10 else trio_odds[0]
    else:
        row['trio_median'] = None
        row['trio_p10'] = None

    for k in ['basho_code', 'year', 'kai', 'nichi', 'race_num']:
        del row[k]
    return row


# =============================================================================
# OV parser (3連単オッズ, record=34286 bytes)
# 4896組 x 7B, 0.1倍単位
# =============================================================================
def parse_ov_line(line_bytes):
    row = {}
    row['basho_code'] = _field(line_bytes, 1, 2)
    row['year'] = _field(line_bytes, 3, 2)
    row['kai'] = _field(line_bytes, 5, 1)
    row['nichi'] = _field(line_bytes, 6, 1)
    row['race_num'] = _field(line_bytes, 7, 2)
    row['race_id'] = _build_race_id(
        row['basho_code'], row['year'], row['kai'], row['nichi'], row['race_num']
    )
    row['num_horses'] = _safe_int(_field(line_bytes, 9, 2))

    # 3連単オッズ summary (4896組 x 7B, 0.1倍単位)
    tierce_odds = []
    for i in range(4896):
        val = _safe_int(_field(line_bytes, 11 + i * 7, 7))
        if val and val > 0 and val < 9999999:
            tierce_odds.append(val / 10.0)
    row['tierce_count'] = len(tierce_odds)
    row['tierce_min'] = min(tierce_odds) if tierce_odds else None
    row['tierce_max'] = max(tierce_odds) if tierce_odds else None
    if tierce_odds:
        tierce_odds.sort()
        row['tierce_median'] = tierce_odds[len(tierce_odds) // 2]
        row['tierce_p10'] = tierce_odds[len(tierce_odds) // 10] if len(tierce_odds) >= 10 else tierce_odds[0]
    else:
        row['tierce_median'] = None
        row['tierce_p10'] = None

    for k in ['basho_code', 'year', 'kai', 'nichi', 'race_num']:
        del row[k]
    return row


# =============================================================================
# HJC parser (払戻情報, record=442 bytes)
# =============================================================================
def parse_hjc_line(line_bytes):
    row = {}
    row['basho_code'] = _field(line_bytes, 1, 2)
    row['year'] = _field(line_bytes, 3, 2)
    row['kai'] = _field(line_bytes, 5, 1)
    row['nichi'] = _field(line_bytes, 6, 1)
    row['race_num'] = _field(line_bytes, 7, 2)
    row['race_id'] = _build_race_id(
        row['basho_code'], row['year'], row['kai'], row['nichi'], row['race_num']
    )

    # 払戻データ: 各券種 x 3組 (馬番/組番 + 払戻金)
    # HJC spec: 各券種3通り x (馬番4B + 払戻9B) = 39B per type
    # Format by byte inspection:
    # pos 9: 単勝 3x(num4 + pay9) = 39B → pos 9-47
    # pos 48: 複勝 3x(num4 + pay9) = 39B → pos 48-86
    # pos 87: 枠連 3x(num4 + pay9) = 39B → pos 87-125
    # pos 126: 馬連 3x(num4 + pay9) = 39B → pos 126-164
    # pos 165: ワイド 3x(num4 + pay9) = 39B → pos 165-203
    # pos 204: 馬単 3x(num4 + pay9) = 39B → pos 204-242
    # pos 243: 3連複 3x(num4 + pay9) = 39B → pos 243-281
    # pos 282: 3連単 3x(num6 + pay9) = 45B → pos 282-326
    bet_types = [
        ('tansho', 9, 4, 9, 3),
        ('fukusho', 48, 4, 9, 3),
        ('wakuren', 87, 4, 9, 3),
        ('umaren', 126, 4, 9, 3),
        ('wide', 165, 4, 9, 3),
        ('umatan', 204, 4, 9, 3),
        ('trio', 243, 4, 9, 3),
        ('tierce', 282, 6, 9, 3),
    ]
    for btype, start, num_len, pay_len, count in bet_types:
        for j in range(count):
            offset = start + j * (num_len + pay_len)
            nums_str = _field(line_bytes, offset, num_len).strip()
            pay_str = _field(line_bytes, offset + num_len, pay_len).strip()
            row[f'{btype}_num{j+1}'] = nums_str if nums_str else None
            row[f'{btype}_pay{j+1}'] = _safe_int(pay_str)

    for k in ['basho_code', 'year', 'kai', 'nichi', 'race_num']:
        del row[k]
    return row


# =============================================================================
# KAA parser (開催データ, record=70 bytes)
# =============================================================================
def parse_kaa_line(line_bytes):
    row = {}
    row['basho_code'] = _field(line_bytes, 1, 2)
    row['year'] = _field(line_bytes, 3, 2)
    row['kai'] = _field(line_bytes, 5, 1)
    row['nichi'] = _field(line_bytes, 6, 1)

    row['kaisai_key'] = _build_kaisai_key(
        row['basho_code'], row['year'], row['kai'], row['nichi']
    )

    row['yyyymmdd'] = _field(line_bytes, 7, 8).strip()
    row['kaisai_ku'] = _safe_int(_field(line_bytes, 15, 1))  # 1:関東 2:関西 3:ローカル
    row['youbi'] = _field(line_bytes, 16, 2).strip()
    row['course_name'] = _field(line_bytes, 18, 4).strip()
    row['weather_code'] = _field(line_bytes, 22, 1).strip()
    row['turf_baba_code'] = _field(line_bytes, 23, 2).strip()
    row['turf_baba_inner'] = _safe_int(_field(line_bytes, 25, 1))
    row['turf_baba_mid'] = _safe_int(_field(line_bytes, 26, 1))
    row['turf_baba_outer'] = _safe_int(_field(line_bytes, 27, 1))
    row['turf_baba_sa'] = _safe_float(_field(line_bytes, 28, 3))
    row['straight_sa_most'] = _safe_float(_field(line_bytes, 31, 2))
    row['straight_sa_inner'] = _safe_float(_field(line_bytes, 33, 2))
    row['straight_sa_mid'] = _safe_float(_field(line_bytes, 35, 2))
    row['straight_sa_outer'] = _safe_float(_field(line_bytes, 37, 2))
    row['straight_sa_far'] = _safe_float(_field(line_bytes, 39, 2))
    row['dirt_baba_code'] = _field(line_bytes, 41, 2).strip()
    row['dirt_baba_inner'] = _safe_int(_field(line_bytes, 43, 1))
    row['dirt_baba_mid'] = _safe_int(_field(line_bytes, 44, 1))
    row['dirt_baba_outer'] = _safe_int(_field(line_bytes, 45, 1))
    row['dirt_baba_sa'] = _safe_float(_field(line_bytes, 46, 3))
    row['data_ku'] = _field(line_bytes, 49, 1).strip()
    # V3 extended fields (pos 50+, may not exist in shorter records)
    if len(line_bytes) >= 63:
        row['renzoku_day'] = _safe_int(_field(line_bytes, 50, 2))
        row['turf_type'] = _field(line_bytes, 52, 1).strip()
        row['grass_height'] = _safe_float(_field(line_bytes, 53, 4))
        row['tennatsu'] = _field(line_bytes, 57, 1).strip()
        row['touketsu'] = _field(line_bytes, 58, 1).strip()
        row['rain_mid'] = _safe_float(_field(line_bytes, 59, 5))
    else:
        row['renzoku_day'] = None
        row['turf_type'] = None
        row['grass_height'] = None
        row['tennatsu'] = None
        row['touketsu'] = None
        row['rain_mid'] = None

    for k in ['basho_code', 'year', 'kai', 'nichi']:
        del row[k]
    return row


# =============================================================================
# KTA parser (登録馬データ, record=386 bytes)
# =============================================================================
def parse_kta_line(line_bytes):
    row = {}
    row['basho_code'] = _field(line_bytes, 1, 2)
    row['year'] = _field(line_bytes, 3, 2)
    row['kai'] = _field(line_bytes, 5, 1)
    row['nichi'] = _field(line_bytes, 6, 1)
    row['race_num'] = _field(line_bytes, 7, 2)
    row['race_id'] = _build_race_id(
        row['basho_code'], row['year'], row['kai'], row['nichi'], row['race_num']
    )

    row['horse_key'] = _field(line_bytes, 9, 40).strip()
    row['blood_num'] = _field(line_bytes, 49, 8).strip()
    row['horse_name'] = _field(line_bytes, 57, 36).strip()
    row['sex_code'] = _field(line_bytes, 93, 1).strip()
    row['keito_code'] = _field(line_bytes, 94, 2).strip()
    row['blinker'] = _field(line_bytes, 96, 1).strip()
    row['jockey_name'] = _field(line_bytes, 97, 12).strip()
    row['weight_carry'] = _safe_float(_field(line_bytes, 109, 3))
    if row['weight_carry'] is not None:
        row['weight_carry'] = row['weight_carry'] / 10.0
    row['apprentice'] = _safe_int(_field(line_bytes, 112, 1))
    row['trainer_name'] = _field(line_bytes, 113, 12).strip()
    row['trainer_area'] = _field(line_bytes, 125, 4).strip()
    row['idm'] = _safe_float(_field(line_bytes, 129, 5))
    row['josho'] = _safe_int(_field(line_bytes, 134, 1))
    row['rotation'] = _safe_int(_field(line_bytes, 135, 3))
    row['kyakushitsu'] = _safe_int(_field(line_bytes, 138, 1))
    row['condition_class'] = _safe_int(_field(line_bytes, 139, 1))
    row['condition_class2'] = _safe_int(_field(line_bytes, 140, 1))
    row['turf_apt'] = _field(line_bytes, 141, 1).strip()
    row['dirt_apt'] = _field(line_bytes, 142, 1).strip()
    row['heavy_apt'] = _safe_int(_field(line_bytes, 143, 1))
    row['class_code'] = _safe_int(_field(line_bytes, 144, 2))
    row['hair_color'] = _safe_int(_field(line_bytes, 148, 2))

    # 前走成績リンクキー
    row['prev1_sed_key'] = _field(line_bytes, 150, 16).strip()

    # 騎手/調教師コード
    row['jockey_code'] = _field(line_bytes, 270, 5).strip()
    row['trainer_code'] = _field(line_bytes, 275, 5).strip()

    # 賞金
    row['prize_total'] = _safe_int(_field(line_bytes, 280, 6))
    row['prize_shutoku'] = _safe_int(_field(line_bytes, 286, 5))
    row['shutoku_class'] = _safe_int(_field(line_bytes, 291, 1))

    # 展開予想指数
    row['ten_idx_pred'] = _safe_float(_field(line_bytes, 342, 5))
    row['pace_idx_pred'] = _safe_float(_field(line_bytes, 347, 5))
    row['agari_idx_pred'] = _safe_float(_field(line_bytes, 352, 5))
    row['ichi_idx_pred'] = _safe_float(_field(line_bytes, 357, 5))

    row['data_ku'] = _field(line_bytes, 377, 1).strip()

    for k in ['basho_code', 'year', 'kai', 'nichi', 'race_num']:
        del row[k]
    return row


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='JRDB batch2 download & parse')
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--types', nargs='+', default=ALL_TYPES)
    args = parser.parse_args()
    types = [t.lower() for t in args.types]

    start_time = datetime.now()
    print('=' * 60)
    print('  JRDB Batch2 Download & Parse')
    print(f'  Types: {types}')
    print('=' * 60)

    # =================================================================
    # Phase 1: Download
    # =================================================================
    if not args.skip_download:
        jrdb_id, jrdb_pw = load_credentials()
        if not jrdb_id or not jrdb_pw:
            print('ERROR: JRDB credentials not found in .env')
            sys.exit(1)

        auth = HTTPBasicAuth(jrdb_id, jrdb_pw)
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        print(f'\nJRDB Login: ID={jrdb_id[:4]}****')
        resp = session.get(f'{JRDB_BASE}/data/index.html', auth=auth, timeout=15, verify=True)
        if resp.status_code != 200:
            print(f'ERROR: Login failed ({resp.status_code})')
            sys.exit(1)
        print('Login OK\n')

        # SRB is inside Sed archives (already extracted)
        if 'srb' in types:
            srb_count = len(glob.glob(os.path.join(EXTRACT_DIR, 'Sed', 'SRB*.txt')))
            print(f'SRB: already in Sed archives ({srb_count} files)')

        # CHA
        if 'cha' in types:
            print('\n--- CHA (追切本データ) ---')
            download_yearly_zips(session, auth, 'Cha', os.path.join(RAW_DIR, 'Cha'),
                                 os.path.join(EXTRACT_DIR, 'Cha'), 'CHA', YEARS)

        # OZ
        if 'oz' in types:
            print('\n--- OZ (基準オッズ) ---')
            download_yearly_zips(session, auth, 'Oz', os.path.join(RAW_DIR, 'Oz'),
                                 os.path.join(EXTRACT_DIR, 'Oz'), 'OZ', YEARS)

        # OW
        if 'ow' in types:
            print('\n--- OW (ワイドオッズ) ---')
            download_yearly_zips(session, auth, 'Ow', os.path.join(RAW_DIR, 'Ow'),
                                 os.path.join(EXTRACT_DIR, 'Ow'), 'OW', YEARS)

        # OT
        if 'ot' in types:
            print('\n--- OT (3連複オッズ) ---')
            download_yearly_zips(session, auth, 'Ot', os.path.join(RAW_DIR, 'Ot'),
                                 os.path.join(EXTRACT_DIR, 'Ot'), 'OT', YEARS)

        # OV
        if 'ov' in types:
            print('\n--- OV (3連単オッズ) ---')
            download_yearly_zips(session, auth, 'Ov', os.path.join(RAW_DIR, 'Ov'),
                                 os.path.join(EXTRACT_DIR, 'Ov'), 'OV', YEARS)

        # HJC already downloaded
        if 'hjc' in types:
            hjc_count = len(glob.glob(os.path.join(EXTRACT_DIR, 'Hjc', 'HJC*.txt')))
            print(f'\nHJC: already extracted ({hjc_count} files)')

        # KAA (LZH yearly)
        if 'kaa' in types:
            print('\n--- KAA (開催データ) ---')
            download_yearly_lzh(session, auth, 'Kaa', os.path.join(RAW_DIR, 'Kaa'),
                                os.path.join(EXTRACT_DIR, 'Kaa'), 'KAA', YEARS)

        # KTA (individual ZIPs from weekly data page)
        if 'kta' in types:
            print('\n--- KTA (登録馬データ) ---')
            kta_raw = os.path.join(RAW_DIR, 'Kta')
            kta_ext = os.path.join(EXTRACT_DIR, 'Kta')
            os.makedirs(kta_raw, exist_ok=True)
            os.makedirs(kta_ext, exist_ok=True)

            # KTA has individual zips: ../datazip/Kta/YYYY/KTAYYMMDD.zip
            # Get file list from main data page and all year-specific pages
            idx_url = f'{JRDB_BASE}/data/index.html'
            resp = session.get(idx_url, auth=auth, timeout=15, verify=True)
            text = resp.content.decode('cp932', errors='replace')
            soup = BeautifulSoup(text, 'html.parser')
            all_kta_zips = []

            for a in soup.find_all('a', href=True):
                h = a['href']
                t = a.get_text(strip=True)
                if 'KTA' in t and '.zip' in h:
                    all_kta_zips.append(h)

            # Also scrape individual year pages on data/Kta
            kta_idx_url = f'{JRDB_BASE}/data/Kta/index.html'
            r2 = session.get(kta_idx_url, auth=auth, timeout=15, verify=True)
            if r2.status_code == 200:
                t2 = r2.content.decode('cp932', errors='replace')
                s2 = BeautifulSoup(t2, 'html.parser')
                for a in s2.find_all('a', href=True):
                    h = a['href']
                    t = a.get_text(strip=True)
                    if 'KTA' in t and '.lzh' in h:
                        # Check if year is in range
                        for yy in [str(y)[2:] for y in YEARS]:
                            if f'KTA{yy}' in t:
                                all_kta_zips.append(h)
                                break

            # Download individual KTA zip files from datazip
            # Get the index from the data page links
            kta_download_count = 0
            seen = set()
            for href in all_kta_zips:
                basename = os.path.basename(href)
                if basename in seen:
                    continue
                seen.add(basename)

                if '.zip' in href:
                    if 'datazip' in href:
                        file_url = f'{JRDB_BASE}/{href.lstrip("../")}'
                    else:
                        file_url = f'{JRDB_BASE}/datazip/Kta/{href}'
                    dest = os.path.join(kta_raw, basename)
                    ok, msg = download_file(session, file_url, dest, auth)
                    if ok:
                        if msg != 'skip':
                            kta_download_count += 1
                        extract_zip(dest, kta_ext)
                    time.sleep(0.2)
                elif '.lzh' in href:
                    file_url = f'{JRDB_BASE}/data/Kta/{href}'
                    dest = os.path.join(kta_raw, basename)
                    ok, msg = download_file(session, file_url, dest, auth)
                    if ok:
                        if msg != 'skip':
                            kta_download_count += 1
                        extract_lzh(dest, kta_ext)
                    time.sleep(0.2)

            print(f'  Downloaded {kta_download_count} new KTA files')

    # =================================================================
    # Phase 2: Parse
    # =================================================================
    print('\n' + '=' * 60)
    print('  Phase 2: Parse → CSV')
    print('=' * 60)

    results = {}

    # --- SRB ---
    if 'srb' in types:
        print('\n--- SRB (成績レースデータ) ---')
        df = parse_files(os.path.join(EXTRACT_DIR, 'Sed', 'SRB*.txt'), parse_srb_line, 800,
                         dedup_cols=['race_id'])
        if df is not None:
            out = os.path.join(DATA_DIR, 'jrdb_srb.csv')
            df.to_csv(out, index=False, encoding='utf-8-sig')
            print(f'  Saved: {out} ({len(df):,} rows, {len(df.columns)} cols)')
            results['srb'] = len(df)

    # --- CHA ---
    if 'cha' in types:
        print('\n--- CHA (追切本データ) ---')
        df = parse_files(os.path.join(EXTRACT_DIR, 'Cha', 'CHA*.txt'), parse_cha_line, 55,
                         dedup_cols=['race_id', 'umaban', 'oikiri_date'])
        if df is not None:
            out = os.path.join(DATA_DIR, 'jrdb_cha.csv')
            df.to_csv(out, index=False, encoding='utf-8-sig')
            print(f'  Saved: {out} ({len(df):,} rows, {len(df.columns)} cols)')
            results['cha'] = len(df)

    # --- OZ ---
    if 'oz' in types:
        print('\n--- OZ (基準オッズ: 単複・馬連) ---')
        df = parse_files(os.path.join(EXTRACT_DIR, 'Oz', 'OZ*.txt'), parse_oz_line, 900,
                         dedup_cols=['race_id'])
        if df is not None:
            out = os.path.join(DATA_DIR, 'jrdb_oz.csv')
            df.to_csv(out, index=False, encoding='utf-8-sig')
            print(f'  Saved: {out} ({len(df):,} rows, {len(df.columns)} cols)')
            results['oz'] = len(df)

    # --- OW ---
    if 'ow' in types:
        print('\n--- OW (ワイドオッズ) ---')
        df = parse_files(os.path.join(EXTRACT_DIR, 'Ow', 'OW*.txt'), parse_ow_line, 750,
                         dedup_cols=['race_id'])
        if df is not None:
            out = os.path.join(DATA_DIR, 'jrdb_ow.csv')
            df.to_csv(out, index=False, encoding='utf-8-sig')
            print(f'  Saved: {out} ({len(df):,} rows, {len(df.columns)} cols)')
            results['ow'] = len(df)

    # --- OT ---
    if 'ot' in types:
        print('\n--- OT (3連複オッズ) ---')
        df = parse_files(os.path.join(EXTRACT_DIR, 'Ot', 'OT*.txt'), parse_ot_line, 4900,
                         dedup_cols=['race_id'])
        if df is not None:
            out = os.path.join(DATA_DIR, 'jrdb_ot.csv')
            df.to_csv(out, index=False, encoding='utf-8-sig')
            print(f'  Saved: {out} ({len(df):,} rows, {len(df.columns)} cols)')
            results['ot'] = len(df)

    # --- OV ---
    if 'ov' in types:
        print('\n--- OV (3連単オッズ) ---')
        df = parse_files(os.path.join(EXTRACT_DIR, 'Ov', 'OV*.txt'), parse_ov_line, 34000,
                         dedup_cols=['race_id'])
        if df is not None:
            out = os.path.join(DATA_DIR, 'jrdb_ov.csv')
            df.to_csv(out, index=False, encoding='utf-8-sig')
            print(f'  Saved: {out} ({len(df):,} rows, {len(df.columns)} cols)')
            results['ov'] = len(df)

    # --- HJC ---
    if 'hjc' in types:
        print('\n--- HJC (払戻情報) ---')
        df = parse_files(os.path.join(EXTRACT_DIR, 'Hjc', 'HJC*.txt'), parse_hjc_line, 300,
                         dedup_cols=['race_id'])
        if df is not None:
            out = os.path.join(DATA_DIR, 'jrdb_hjc.csv')
            df.to_csv(out, index=False, encoding='utf-8-sig')
            print(f'  Saved: {out} ({len(df):,} rows, {len(df.columns)} cols)')
            results['hjc'] = len(df)

    # --- KAA ---
    if 'kaa' in types:
        print('\n--- KAA (開催データ) ---')
        df = parse_files(os.path.join(EXTRACT_DIR, 'Kaa', 'KAA*.txt'), parse_kaa_line, 48,
                         dedup_cols=['kaisai_key', 'yyyymmdd', 'data_ku'])
        if df is not None:
            out = os.path.join(DATA_DIR, 'jrdb_kaa.csv')
            df.to_csv(out, index=False, encoding='utf-8-sig')
            print(f'  Saved: {out} ({len(df):,} rows, {len(df.columns)} cols)')
            results['kaa'] = len(df)

    # --- KTA ---
    if 'kta' in types:
        print('\n--- KTA (登録馬データ) ---')
        df = parse_files(os.path.join(EXTRACT_DIR, 'Kta', 'KTA*.txt'), parse_kta_line, 340,
                         dedup_cols=['race_id', 'blood_num'])
        if df is not None:
            out = os.path.join(DATA_DIR, 'jrdb_kta.csv')
            df.to_csv(out, index=False, encoding='utf-8-sig')
            print(f'  Saved: {out} ({len(df):,} rows, {len(df.columns)} cols)')
            results['kta'] = len(df)

    # =================================================================
    # Phase 3: Overlap check with existing data
    # =================================================================
    print('\n' + '=' * 60)
    print('  Phase 3: Overlap Check')
    print('=' * 60)

    overlap_checks = [
        ('jrdb_srb.csv', 'jrdb_sed.csv', 'race_id', 'SRB vs SED'),
        ('jrdb_hjc.csv', 'jrdb_sed.csv', 'race_id', 'HJC vs SED'),
        ('jrdb_oz.csv', 'jrdb_sed.csv', 'race_id', 'OZ vs SED'),
        ('jrdb_cha.csv', 'jrdb_cyb.csv', 'race_id', 'CHA vs CYB'),
        ('jrdb_kaa.csv', 'jrdb_kab.csv', 'kaisai_key', 'KAA vs KAB'),
    ]
    for new_file, ref_file, key_col, label in overlap_checks:
        new_path = os.path.join(DATA_DIR, new_file)
        ref_path = os.path.join(DATA_DIR, ref_file)
        if os.path.exists(new_path) and os.path.exists(ref_path):
            try:
                df_new = pd.read_csv(new_path, usecols=[key_col], dtype=str)
                df_ref = pd.read_csv(ref_path, usecols=[key_col], dtype=str)
                new_ids = set(df_new[key_col].dropna())
                ref_ids = set(df_ref[key_col].dropna())
                overlap = len(new_ids & ref_ids)
                print(f'  {label}: new={len(new_ids):,}, ref={len(ref_ids):,}, '
                      f'overlap={overlap:,} ({overlap/max(len(new_ids),1)*100:.1f}%)')
            except Exception as e:
                print(f'  {label}: check failed ({e})')

    # =================================================================
    # Summary
    # =================================================================
    elapsed = (datetime.now() - start_time).total_seconds()
    print('\n' + '=' * 60)
    print(f'  Done in {elapsed:.1f}s')
    print('=' * 60)
    for fname in ['jrdb_srb.csv', 'jrdb_cha.csv', 'jrdb_oz.csv', 'jrdb_ow.csv',
                   'jrdb_ot.csv', 'jrdb_ov.csv', 'jrdb_hjc.csv', 'jrdb_kaa.csv', 'jrdb_kta.csv']:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / 1024 / 1024
            print(f'  {fname}: {size:.1f} MB')


if __name__ == '__main__':
    main()
