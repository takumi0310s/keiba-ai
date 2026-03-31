#!/usr/bin/env python3
"""JRDB追加データ ダウンロード＆パース
対象: KZ(騎手マスタ全), KS(騎手マスタ週), CZ(調教師マスタ全), CS(調教師マスタ週),
      JO(情報データ), KKA(競走馬拡張), UKC(馬基本データ)

Usage:
    python tools/download_parse_jrdb_extra.py
    python tools/download_parse_jrdb_extra.py --skip-download   # パースのみ
    python tools/download_parse_jrdb_extra.py --types jo kka    # 指定タイプのみ
"""
import os
import sys
import io
import time
import zipfile
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
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        if size > 100:
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
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
            return len(zf.namelist())
    except Exception:
        return -1


# =============================================================================
# Download functions
# =============================================================================

def download_yearly_type(session, auth, zip_dir, extract_dir, type_upper, years):
    """Download yearly archives (JOA_YYYY.zip) + 2026 individual files."""
    os.makedirs(zip_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    stats = {'downloaded': 0, 'skipped': 0, 'failed': 0, 'extracted': 0}

    for year in years:
        if year == 2026:
            # Individual files for 2026
            idx_url = f'{JRDB_BASE}/datazip/{os.path.basename(zip_dir)}/index.html'
            resp = session.get(idx_url, auth=auth, timeout=15, verify=True)
            resp.encoding = 'shift_jis'
            soup = BeautifulSoup(resp.text, 'html.parser')
            yy = '26'
            for a in soup.find_all('a', href=True):
                text = a.get_text(strip=True)
                href = a['href']
                if f'{type_upper}{yy}' in text and '.zip' in text:
                    file_url = f'{JRDB_BASE}/datazip/{os.path.basename(zip_dir)}/{href}'
                    dest = os.path.join(zip_dir, os.path.basename(href))
                    ok, msg = download_file(session, file_url, dest, auth)
                    if ok:
                        if msg == 'skip':
                            stats['skipped'] += 1
                        else:
                            stats['downloaded'] += 1
                            n = extract_zip(dest, extract_dir)
                            if n > 0:
                                stats['extracted'] += n
                    else:
                        stats['failed'] += 1
                    time.sleep(0.3)
        else:
            filename = f'{type_upper}_{year}.zip'
            file_url = f'{JRDB_BASE}/datazip/{os.path.basename(zip_dir)}/{filename}'
            dest = os.path.join(zip_dir, filename)
            print(f'  {year}: {filename}...', end=' ', flush=True)
            ok, msg = download_file(session, file_url, dest, auth)
            if ok:
                if msg == 'skip':
                    print('exists')
                    stats['skipped'] += 1
                else:
                    print(f'OK ({msg})')
                    stats['downloaded'] += 1
                    n = extract_zip(dest, extract_dir)
                    if n > 0:
                        stats['extracted'] += n
                        print(f'    → extracted {n} files')
            else:
                print(f'FAILED ({msg})')
                stats['failed'] += 1
            time.sleep(1)
    return stats


def download_master_file(session, auth, zip_dir, extract_dir, type_upper):
    """Download latest master file (KZA/CZA - all jockeys/trainers)."""
    os.makedirs(zip_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    # Find latest file from data index page
    idx_url = f'{JRDB_BASE}/data/index.html'
    resp = session.get(idx_url, auth=auth, timeout=15, verify=True)
    text = resp.content.decode('cp932', errors='replace')
    soup = BeautifulSoup(text, 'html.parser')

    target_href = None
    for a in soup.find_all('a', href=True):
        href = a['href']
        if type_upper in href and 'datazip' in href and '.zip' in href:
            target_href = href
            break

    if not target_href:
        print(f'  {type_upper}: not found on index page')
        return {'downloaded': 0, 'skipped': 0, 'failed': 1, 'extracted': 0}

    file_url = f'{JRDB_BASE}/{target_href.lstrip("../")}'
    filename = os.path.basename(target_href)
    dest = os.path.join(zip_dir, filename)
    print(f'  Latest: {filename}...', end=' ', flush=True)

    ok, msg = download_file(session, file_url, dest, auth)
    if ok:
        if msg == 'skip':
            print('exists')
        else:
            print(f'OK ({msg})')
        n = extract_zip(dest, extract_dir)
        if n > 0:
            print(f'    → extracted {n} files')
        return {'downloaded': 0 if msg == 'skip' else 1, 'skipped': 1 if msg == 'skip' else 0,
                'failed': 0, 'extracted': max(n, 0)}
    else:
        print(f'FAILED ({msg})')
        return {'downloaded': 0, 'skipped': 0, 'failed': 1, 'extracted': 0}


# =============================================================================
# Parser helpers (shared with parse_jrdb.py)
# =============================================================================

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


# =============================================================================
# UKC parser (馬基本データ, record=290 bytes excl CRLF)
# Spec: レコード長292BYTE (290 data + 2 CRLF)
# =============================================================================
UKC_COLUMNS = [
    ('blood_num',          1,   8),   # 血統登録番号
    ('horse_name',         9,  36),   # 馬名 (全角18文字)
    ('sex_code',          45,   1),   # 性別 1:牡 2:牝 3:セン
    ('hair_color_code',   46,   2),   # 毛色コード
    ('keito_code',        48,   2),   # 馬記号コード
    ('father_name',       50,  36),   # 父馬名
    ('mother_name',       86,  36),   # 母馬名
    ('bms_name',         122,  36),   # 母父馬名
    ('birthday',         158,   8),   # 生年月日 YYYYMMDD
    ('father_birth_year', 166,  4),   # 父馬生年 YYYY
    ('mother_birth_year', 170,  4),   # 母馬生年 YYYY
    ('bms_birth_year',   174,   4),   # 母父馬生年 YYYY
    ('owner_name',       178,  40),   # 馬主名 (全角20文字)
    ('owner_code',       218,   2),   # 馬主会コード
    ('breeder_name',     220,  40),   # 生産者名 (全角20文字)
    ('birthplace',       260,   8),   # 産地名 (全角4文字)
    ('register_del_flag', 268,  1),   # 登録抹消フラグ 0:現役 1:抹消
    ('data_date',        269,   8),   # データ年月日 YYYYMMDD
    ('father_code',      277,   4),   # 父馬系統コード
    ('bms_code',         281,   4),   # 母父馬系統コード
]

UKC_STR_COLS = [
    'horse_name', 'father_name', 'mother_name', 'bms_name',
    'owner_name', 'breeder_name', 'birthplace', 'blood_num',
    'birthday', 'data_date',
]


def parse_ukc_line(line_bytes):
    row = {}
    for name, start, length in UKC_COLUMNS:
        row[name] = _field(line_bytes, start, length)
    for col in UKC_STR_COLS:
        row[col] = row[col].strip()
    for col in ['sex_code', 'hair_color_code', 'keito_code', 'owner_code',
                'register_del_flag', 'father_code', 'bms_code']:
        row[col] = row[col].strip()
    row['father_birth_year'] = _safe_int(row['father_birth_year'])
    row['mother_birth_year'] = _safe_int(row['mother_birth_year'])
    row['bms_birth_year'] = _safe_int(row['bms_birth_year'])
    return row


# =============================================================================
# KKA parser (競走馬拡張データ, record=322 bytes excl CRLF)
# Spec: レコード長324BYTE (322 data + 2 CRLF)
# =============================================================================
# KKA has race key + umaban + many 12-byte (ZZ9*4 = 4 x 3-digit) level fields
KKA_COLUMNS = [
    # レースキー
    ('basho_code',   1,  2),
    ('year',         3,  2),
    ('kai',          5,  1),
    ('nichi',        6,  1),
    ('race_num',     7,  2),
    ('umaban',       9,  2),
    # レベル (各12 bytes = 4区分 x 3桁: 1着 2着 3着 着外)
    ('jra_seiseki',     11, 12),   # JRA成績
    ('koryu_seiseki',   23, 12),   # 交流成績
    ('other_seiseki',   35, 12),   # その他成績
    # 産駒産地別成績レベル集計
    ('turf_dirt_2着',   47, 12),   # 芝ダ2着成績
    ('turf_dirt_2着_dist', 59, 12),
    ('track_seiseki',   71, 12),   # トラック回り成績
    ('rotation_sei',    83, 12),   # ローテーション
    ('kyori_seiseki',   95, 12),   # 距離成績
    ('saka_seiseki',   107, 12),   # 坂成績
    ('heavy_seiseki',  119, 12),   # 重馬場成績
    ('rest_seiseki',   131, 12),   # 休み明け成績
    ('class_seiseki',  143, 12),   # クラス成績
    ('speed_seiseki',  155, 12),   # Sペース成績
    ('slow_seiseki',   167, 12),   # Nペース成績
    ('mid_seiseki',    179, 12),   # Tペース成績
    ('season_seiseki', 191, 12),   # 季節成績
    ('waku_seiseki',   203, 12),   # 枠成績
    # 産駒産地別成績
    ('breeder_dist',   215, 12),   # 距離成績(産駒)
    ('breeder_track',  227, 12),   # トラック(産駒)
    ('breeder_adjust', 239, 12),   # 調整(産駒)
    ('breeder_baba',   251, 12),   # 馬場(産駒)
    ('breeder_blanker',263, 12),   # ブランカー(産駒)
    ('breeder_surface',275, 12),   # 芝ダ(産駒)
    # その他
    ('dam_rensho_max',   287, 3),  # 母産駒最連勝率 (%)
    ('dam_rensho_min',   290, 3),  # 母産駒最低連勝率
    ('dam_rensho_avg',   293, 4),  # 母産駒平均連勝距離
    ('bms_rensho_max',   297, 3),  # 母父産駒最連勝率
    ('bms_rensho_min',   300, 3),  # 母父産駒最低連勝率
    ('bms_rensho_avg',   303, 4),  # 母父産駒平均連勝距離
]


def _parse_level_12(val_str):
    """12バイトのレベルデータ(ZZ9*4)をパース → (1着, 2着, 3着, 着外)"""
    v = val_str.strip()
    if len(v) < 12:
        return None, None, None, None
    try:
        w = _safe_int(v[0:3])
        p2 = _safe_int(v[3:6])
        p3 = _safe_int(v[6:9])
        out = _safe_int(v[9:12])
        return w, p2, p3, out
    except Exception:
        return None, None, None, None


def parse_kka_line(line_bytes):
    row = {}
    for name, start, length in KKA_COLUMNS:
        row[name] = _field(line_bytes, start, length)

    row['race_id'] = _build_race_id(
        row['basho_code'], row['year'], row['kai'], row['nichi'], row['race_num']
    )
    row['umaban'] = _safe_int(row['umaban'])

    # Parse level fields into flattened columns
    level_fields = [
        'jra_seiseki', 'koryu_seiseki', 'other_seiseki',
        'turf_dirt_2着', 'turf_dirt_2着_dist', 'track_seiseki',
        'rotation_sei', 'kyori_seiseki', 'saka_seiseki',
        'heavy_seiseki', 'rest_seiseki', 'class_seiseki',
        'speed_seiseki', 'slow_seiseki', 'mid_seiseki',
        'season_seiseki', 'waku_seiseki',
        'breeder_dist', 'breeder_track', 'breeder_adjust',
        'breeder_baba', 'breeder_blanker', 'breeder_surface',
    ]
    for f in level_fields:
        w, p2, p3, out = _parse_level_12(row[f])
        row[f'{f}_1'] = w
        row[f'{f}_2'] = p2
        row[f'{f}_3'] = p3
        row[f'{f}_out'] = out
        del row[f]

    # Other numeric fields
    row['dam_rensho_max'] = _safe_int(row['dam_rensho_max'])
    row['dam_rensho_min'] = _safe_int(row['dam_rensho_min'])
    row['dam_rensho_avg'] = _safe_int(row['dam_rensho_avg'])
    row['bms_rensho_max'] = _safe_int(row['bms_rensho_max'])
    row['bms_rensho_min'] = _safe_int(row['bms_rensho_min'])
    row['bms_rensho_avg'] = _safe_int(row['bms_rensho_avg'])

    for k in ['basho_code', 'year', 'kai', 'nichi', 'race_num']:
        del row[k]

    return row


# =============================================================================
# KZ/KS parser (騎手データ, record=270 bytes excl CRLF)
# Spec: レコード長272BYTE (270 data + 2 CRLF)
# KZA=全騎手, KSA=今週出走分 — 同一フォーマット
# =============================================================================
KS_COLUMNS = [
    ('jockey_code',        1,  5),   # 騎手コード
    ('register_del_flag',  6,  1),   # 登録抹消フラグ 1:現役 0:抹消
    ('register_del_date',  7,  8),   # 登録抹消年月日 YYYYMMDD
    ('jockey_name',       15, 12),   # 騎手名 (全角6文字)
    ('jockey_kana',       27, 30),   # 騎手カナ (全角15文字)
    ('jockey_name_short',  57, 6),   # 騎手名略称 (全角3文字)
    ('area_code',          63, 1),   # 所属コード 1:関東 2:関西 3:他
    ('area_name',          64, 4),   # 所属地名 (全角2文字)
    ('birthday',           68, 8),   # 生年月日 YYYYMMDD
    ('first_ride_year',    76, 4),   # 初騎乗年 YYYY
    ('apprentice_class',   80, 1),   # 見習い区分 1:1K減 2:2K 3:3K
    ('stable_code',        81, 5),   # 厩舎(所属)コード
    ('jockey_comment',     86, 40),  # 騎手コメント (全角20文字)
    ('comment_date',      126, 8),   # コメント入力年月日
    # 本年成績
    ('year_leading',      134, 3),   # 本年リーディング
    ('year_turf_results', 137, 12),  # 本年芝成績 (1-2-3-着外 3*4)
    ('year_dirt_results', 149, 12),  # 本年ダ成績
    ('year_turf_wr',      161, 3),   # 本年特別勝数
    ('year_special_w',    164, 3),   # 本年重賞勝数
    # 昨年成績
    ('last_leading',      167, 3),   # 昨年リーディング
    ('last_turf_results', 170, 12),  # 昨年芝成績
    ('last_dirt_results', 182, 12),  # 昨年ダ成績
    ('last_turf_wr',      194, 3),   # 昨年特別勝数
    ('last_special_w',    197, 3),   # 昨年重賞勝数
    # 通算成績
    ('total_turf_results',200, 20),  # 通算芝成績 (1-2-3-着外 5*4)
    ('total_dirt_results',220, 20),  # 通算ダ成績
    ('data_date',         240, 8),   # データ年月日
]


def _parse_results_12(val_str):
    """12バイト成績(ZZ9*4) → (1着, 2着, 3着, 着外)"""
    v = val_str.strip()
    if len(v) < 12:
        return None, None, None, None
    return _safe_int(v[0:3]), _safe_int(v[3:6]), _safe_int(v[6:9]), _safe_int(v[9:12])


def _parse_results_20(val_str):
    """20バイト成績(ZZZZ9*4) → (1着, 2着, 3着, 着外)"""
    v = val_str.strip()
    if len(v) < 20:
        return None, None, None, None
    return _safe_int(v[0:5]), _safe_int(v[5:10]), _safe_int(v[10:15]), _safe_int(v[15:20])


def parse_ks_line(line_bytes):
    row = {}
    for name, start, length in KS_COLUMNS:
        row[name] = _field(line_bytes, start, length)

    # Trim strings
    for col in ['jockey_code', 'jockey_name', 'jockey_kana', 'jockey_name_short',
                'area_code', 'area_name', 'birthday', 'register_del_date',
                'register_del_flag', 'stable_code', 'jockey_comment',
                'comment_date', 'data_date', 'apprentice_class']:
        row[col] = row[col].strip()

    # Numeric
    row['first_ride_year'] = _safe_int(row['first_ride_year'])
    row['year_leading'] = _safe_int(row['year_leading'])
    row['last_leading'] = _safe_int(row['last_leading'])
    row['year_special_w'] = _safe_int(row['year_special_w'])
    row['last_special_w'] = _safe_int(row['last_special_w'])
    row['year_turf_wr'] = _safe_int(row['year_turf_wr'])
    row['last_turf_wr'] = _safe_int(row['last_turf_wr'])

    # Parse results
    for prefix, col in [('year_turf', 'year_turf_results'), ('year_dirt', 'year_dirt_results'),
                         ('last_turf', 'last_turf_results'), ('last_dirt', 'last_dirt_results')]:
        w, p2, p3, out = _parse_results_12(row[col])
        row[f'{prefix}_1st'] = w
        row[f'{prefix}_2nd'] = p2
        row[f'{prefix}_3rd'] = p3
        row[f'{prefix}_out'] = out
        del row[col]

    for prefix, col in [('total_turf', 'total_turf_results'), ('total_dirt', 'total_dirt_results')]:
        w, p2, p3, out = _parse_results_20(row[col])
        row[f'{prefix}_1st'] = w
        row[f'{prefix}_2nd'] = p2
        row[f'{prefix}_3rd'] = p3
        row[f'{prefix}_out'] = out
        del row[col]

    return row


# =============================================================================
# CZ/CS parser (調教師データ, record=270 bytes excl CRLF)
# Spec: レコード長272BYTE (270 data + 2 CRLF)
# CZA=全調教師, CSA=今週出走分 — 同一フォーマット
# =============================================================================
CS_COLUMNS = [
    ('trainer_code',       1,  5),   # 調教師コード
    ('register_del_flag',  6,  1),   # 登録抹消フラグ 1:現役 0:抹消
    ('register_del_date',  7,  8),   # 登録抹消年月日 YYYYMMDD
    ('trainer_name',      15, 12),   # 調教師名 (全角6文字)
    ('trainer_kana',      27, 30),   # 調教師カナ (全角15文字)
    ('trainer_name_short', 57, 6),   # 調教師名略称 (全角3文字)
    ('area_code',          63, 1),   # 所属コード 1:関東 2:関西 3:他
    ('area_name',          64, 4),   # 所属地名 (全角2文字)
    ('birthday',           68, 8),   # 生年月日 YYYYMMDD
    ('first_train_year',   76, 4),   # 初出走年 YYYY
    ('trainer_comment',    80, 40),  # 調教師コメント (全角20文字)
    ('comment_date',      120, 8),   # コメント入力年月日
    # 本年成績
    ('year_leading',      128, 3),   # 本年リーディング
    ('year_turf_results', 131, 12),  # 本年芝成績 (3*4)
    ('year_dirt_results', 143, 12),  # 本年ダ成績
    ('year_turf_wr',      155, 3),   # 本年特別勝数
    ('year_special_w',    158, 3),   # 本年重賞勝数
    # 昨年成績
    ('last_leading',      161, 3),   # 昨年リーディング
    ('last_turf_results', 164, 12),  # 昨年芝成績
    ('last_dirt_results', 176, 12),  # 昨年ダ成績
    ('last_turf_wr',      188, 3),   # 昨年特別勝数
    ('last_special_w',    191, 3),   # 昨年重賞勝数
    # 通算成績
    ('total_turf_results',194, 20),  # 通算芝成績 (5*4)
    ('total_dirt_results',214, 20),  # 通算ダ成績
    ('data_date',         234, 8),   # データ年月日
]


def parse_cs_line(line_bytes):
    row = {}
    for name, start, length in CS_COLUMNS:
        row[name] = _field(line_bytes, start, length)

    for col in ['trainer_code', 'trainer_name', 'trainer_kana', 'trainer_name_short',
                'area_code', 'area_name', 'birthday', 'register_del_date',
                'register_del_flag', 'trainer_comment', 'comment_date', 'data_date']:
        row[col] = row[col].strip()

    row['first_train_year'] = _safe_int(row['first_train_year'])
    row['year_leading'] = _safe_int(row['year_leading'])
    row['last_leading'] = _safe_int(row['last_leading'])
    row['year_special_w'] = _safe_int(row['year_special_w'])
    row['last_special_w'] = _safe_int(row['last_special_w'])
    row['year_turf_wr'] = _safe_int(row['year_turf_wr'])
    row['last_turf_wr'] = _safe_int(row['last_turf_wr'])

    for prefix, col in [('year_turf', 'year_turf_results'), ('year_dirt', 'year_dirt_results'),
                         ('last_turf', 'last_turf_results'), ('last_dirt', 'last_dirt_results')]:
        w, p2, p3, out = _parse_results_12(row[col])
        row[f'{prefix}_1st'] = w
        row[f'{prefix}_2nd'] = p2
        row[f'{prefix}_3rd'] = p3
        row[f'{prefix}_out'] = out
        del row[col]

    for prefix, col in [('total_turf', 'total_turf_results'), ('total_dirt', 'total_dirt_results')]:
        w, p2, p3, out = _parse_results_20(row[col])
        row[f'{prefix}_1st'] = w
        row[f'{prefix}_2nd'] = p2
        row[f'{prefix}_3rd'] = p3
        row[f'{prefix}_out'] = out
        del row[col]

    return row


# =============================================================================
# JO parser (情報データ, record=114 bytes excl CRLF)
# Spec: レコード長116BYTE (114 data + 2 CRLF)
# =============================================================================
JO_COLUMNS = [
    # レースキー
    ('basho_code',         1,  2),
    ('year',               3,  2),
    ('kai',                5,  1),
    ('nichi',              6,  1),
    ('race_num',           7,  2),
    ('umaban',             9,  2),
    ('blood_num',         11,  8),   # 血統登録番号
    ('horse_name',        19, 36),   # 馬名 (全角18文字)
    # 情報指数
    ('soten_odds',        55,  5),   # 総合オッズ ZZ9.9
    ('yoso_odds',         60,  5),   # 予想オッズ ZZ9.9
    ('cid_soten_idx',     65,  5),   # CID総合素点 ZZ9.9
    ('cid_sara_idx',      70,  5),   # CIDさら素点 ZZ9.9
    ('cid_idx',           75,  5),   # CID素点 ZZ9.9
    ('cid',               80,  3),   # CID ZZ9
    ('ls_idx',            83,  5),   # LS指数 ZZ9.9
    ('ls_eval',           88,  1),   # LS評価 A/B/C
    ('em',                89,  1),   # EM 1:該当, それ以外スペース
    # 第2版追加 (外厩/坂路BB)
    ('gaisha_bb',         90,  1),   # 外厩BB区分 (バスターコール参照)
    ('gaisha_bb_wr',      91,  5),   # 外厩BB単勝回収率 ZZZZ9
    ('gaisha_bb_rensho',  96,  5),   # 外厩BB連勝率 ZZZZ9
    ('breeder_bb',       101,  1),   # 坂路BB区分
    ('breeder_bb_wr',    102,  5),   # 坂路BB単勝回収率 ZZZZ9
    ('breeder_bb_rensho',107,  5),   # 坂路BB連勝率 ZZZZ9
]


def parse_jo_line(line_bytes):
    row = {}
    for name, start, length in JO_COLUMNS:
        row[name] = _field(line_bytes, start, length)

    row['race_id'] = _build_race_id(
        row['basho_code'], row['year'], row['kai'], row['nichi'], row['race_num']
    )
    row['umaban'] = _safe_int(row['umaban'])

    for col in ['blood_num', 'horse_name', 'ls_eval', 'em',
                'gaisha_bb', 'breeder_bb']:
        row[col] = row[col].strip()

    for col in ['soten_odds', 'yoso_odds', 'cid_soten_idx', 'cid_sara_idx',
                'cid_idx', 'ls_idx']:
        row[col] = _safe_float(row[col])
    row['cid'] = _safe_int(row['cid'])
    row['gaisha_bb_wr'] = _safe_int(row['gaisha_bb_wr'])
    row['gaisha_bb_rensho'] = _safe_int(row['gaisha_bb_rensho'])
    row['breeder_bb_wr'] = _safe_int(row['breeder_bb_wr'])
    row['breeder_bb_rensho'] = _safe_int(row['breeder_bb_rensho'])

    for k in ['basho_code', 'year', 'kai', 'nichi', 'race_num']:
        del row[k]

    return row


# =============================================================================
# Generic parse function
# =============================================================================
def parse_files(file_pattern, parse_func, min_line_len, dedup_cols=None):
    """Parse fixed-width files matching pattern, return DataFrame."""
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
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='JRDB extra data download & parse')
    parser.add_argument('--skip-download', action='store_true', help='Skip download, parse only')
    parser.add_argument('--types', nargs='+', default=['kz', 'cs', 'jo', 'kka', 'ukc'],
                        help='Types to process (default: all)')
    args = parser.parse_args()

    types = [t.lower() for t in args.types]
    start_time = datetime.now()
    print('=' * 60)
    print('  JRDB Extra Data Download & Parse')
    print(f'  Types: {types}')
    print('=' * 60)

    # =================================================================
    # Phase 1: Download
    # =================================================================
    if not args.skip_download:
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
        print(f'\nJRDB Login: ID={jrdb_id[:4]}****')
        resp = session.get(f'{JRDB_BASE}/data/index.html', auth=auth, timeout=15, verify=True)
        if resp.status_code != 200:
            print(f'ERROR: Login failed (status={resp.status_code})')
            sys.exit(1)
        print('Login OK\n')

        # --- KZ (全騎手マスタ) + KS (週次) ---
        if 'kz' in types:
            print('\n--- KZ/KS (騎手マスタ) ---')
            ks_raw = os.path.join(RAW_DIR, 'Ks')
            ks_ext = os.path.join(EXTRACT_DIR, 'Ks')

            # KZA latest master
            print('  [KZA] Downloading latest master...')
            download_master_file(session, auth, ks_raw, ks_ext, 'KZA')

            # KSA yearly archives (2020-2026)
            print('  [KSA] Downloading yearly archives...')
            download_yearly_type(session, auth, ks_raw, ks_ext, 'KSA', YEARS)

        # --- CZ (全調教師マスタ) + CS (週次) ---
        if 'cs' in types:
            print('\n--- CZ/CS (調教師マスタ) ---')
            cs_raw = os.path.join(RAW_DIR, 'Cs')
            cs_ext = os.path.join(EXTRACT_DIR, 'Cs')

            print('  [CZA] Downloading latest master...')
            download_master_file(session, auth, cs_raw, cs_ext, 'CZA')

            print('  [CSA] Downloading yearly archives...')
            download_yearly_type(session, auth, cs_raw, cs_ext, 'CSA', YEARS)

        # --- JO (情報データ) ---
        if 'jo' in types:
            print('\n--- JO (情報データ) ---')
            jo_raw = os.path.join(RAW_DIR, 'Jo')
            jo_ext = os.path.join(EXTRACT_DIR, 'Jo')
            download_yearly_type(session, auth, jo_raw, jo_ext, 'JOA', YEARS)

        # KKA and UKC are already downloaded via download_jrdb.py
        # But handle case where they're not
        if 'kka' in types:
            kka_ext = os.path.join(EXTRACT_DIR, 'Kka')
            if not os.path.exists(kka_ext) or len(os.listdir(kka_ext)) == 0:
                print('\n--- KKA (競走馬拡張) ---')
                kka_raw = os.path.join(RAW_DIR, 'Kka')
                download_yearly_type(session, auth, kka_raw, kka_ext, 'KKA', YEARS)
            else:
                print(f'\nKKA: already extracted ({len(os.listdir(kka_ext))} files)')

        if 'ukc' in types:
            ukc_ext = os.path.join(EXTRACT_DIR, 'Ukc')
            if not os.path.exists(ukc_ext) or len(os.listdir(ukc_ext)) == 0:
                print('\n--- UKC (馬基本) ---')
                ukc_raw = os.path.join(RAW_DIR, 'Ukc')
                download_yearly_type(session, auth, ukc_raw, ukc_ext, 'UKC', YEARS)
            else:
                print(f'\nUKC: already extracted ({len(os.listdir(ukc_ext))} files)')

    # =================================================================
    # Phase 2: Parse
    # =================================================================
    print('\n' + '=' * 60)
    print('  Phase 2: Parse → CSV')
    print('=' * 60)

    # --- UKC (馬基本データ) ---
    if 'ukc' in types:
        print('\n--- UKC (馬基本データ) ---')
        ukc_pattern = os.path.join(EXTRACT_DIR, 'Ukc', 'UKC*.txt')
        df_ukc = parse_files(ukc_pattern, parse_ukc_line, 260, dedup_cols=['blood_num'])
        if df_ukc is not None:
            out_path = os.path.join(DATA_DIR, 'jrdb_ukc.csv')
            df_ukc.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f'  Saved: {out_path} ({len(df_ukc):,} rows, {len(df_ukc.columns)} cols)')

    # --- KKA (競走馬拡張) ---
    if 'kka' in types:
        print('\n--- KKA (競走馬拡張データ) ---')
        kka_pattern = os.path.join(EXTRACT_DIR, 'Kka', 'KKA*.txt')
        df_kka = parse_files(kka_pattern, parse_kka_line, 280, dedup_cols=['race_id', 'umaban'])
        if df_kka is not None:
            # Select useful columns (drop the many level detail cols, keep summaries)
            key_cols = ['race_id', 'umaban',
                        'jra_seiseki_1', 'jra_seiseki_2', 'jra_seiseki_3', 'jra_seiseki_out',
                        'koryu_seiseki_1', 'koryu_seiseki_2', 'koryu_seiseki_3', 'koryu_seiseki_out',
                        'kyori_seiseki_1', 'kyori_seiseki_2', 'kyori_seiseki_3', 'kyori_seiseki_out',
                        'track_seiseki_1', 'track_seiseki_2', 'track_seiseki_3', 'track_seiseki_out',
                        'heavy_seiseki_1', 'heavy_seiseki_2', 'heavy_seiseki_3', 'heavy_seiseki_out',
                        'rest_seiseki_1', 'rest_seiseki_2', 'rest_seiseki_3', 'rest_seiseki_out',
                        'class_seiseki_1', 'class_seiseki_2', 'class_seiseki_3', 'class_seiseki_out',
                        'season_seiseki_1', 'season_seiseki_2', 'season_seiseki_3', 'season_seiseki_out',
                        'waku_seiseki_1', 'waku_seiseki_2', 'waku_seiseki_3', 'waku_seiseki_out',
                        'saka_seiseki_1', 'saka_seiseki_2', 'saka_seiseki_3', 'saka_seiseki_out',
                        'speed_seiseki_1', 'speed_seiseki_2', 'speed_seiseki_3', 'speed_seiseki_out',
                        'dam_rensho_max', 'dam_rensho_min', 'dam_rensho_avg',
                        'bms_rensho_max', 'bms_rensho_min', 'bms_rensho_avg']
            avail = [c for c in key_cols if c in df_kka.columns]
            df_out = df_kka[avail].copy()
            out_path = os.path.join(DATA_DIR, 'jrdb_kka.csv')
            df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f'  Saved: {out_path} ({len(df_out):,} rows, {len(avail)} cols)')

            # Year coverage
            df_kka['_year'] = df_kka['race_id'].str[:4]
            print('  Year coverage:')
            for yr, cnt in df_kka.groupby('_year').size().items():
                print(f'    {yr}: {cnt:,}')

    # --- KZ (騎手マスタ: KZA=全 + KSA=週次、マージ) ---
    if 'kz' in types:
        print('\n--- KZ (騎手マスタ) ---')
        ks_dir = os.path.join(EXTRACT_DIR, 'Ks')
        # Parse all KZA and KSA files (same format)
        all_pattern = os.path.join(ks_dir, 'K[ZS]A*.txt')
        df_kz = parse_files(all_pattern, parse_ks_line, 240, dedup_cols=['jockey_code'])
        if df_kz is not None:
            out_path = os.path.join(DATA_DIR, 'jrdb_kz.csv')
            df_kz.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f'  Saved: {out_path} ({len(df_kz):,} rows, {len(df_kz.columns)} cols)')
            active = df_kz[df_kz['register_del_flag'] == '1']
            print(f'  Active jockeys: {len(active):,}')

    # --- CZ (調教師マスタ: CZA=全 + CSA=週次、マージ) ---
    if 'cs' in types:
        print('\n--- CZ (調教師マスタ) ---')
        cs_dir = os.path.join(EXTRACT_DIR, 'Cs')
        all_pattern = os.path.join(cs_dir, 'C[ZS]A*.txt')
        df_cz = parse_files(all_pattern, parse_cs_line, 230, dedup_cols=['trainer_code'])
        if df_cz is not None:
            out_path = os.path.join(DATA_DIR, 'jrdb_cz.csv')
            df_cz.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f'  Saved: {out_path} ({len(df_cz):,} rows, {len(df_cz.columns)} cols)')
            active = df_cz[df_cz['register_del_flag'] == '1']
            print(f'  Active trainers: {len(active):,}')

    # --- JO (情報データ) ---
    if 'jo' in types:
        print('\n--- JO (情報データ) ---')
        jo_pattern = os.path.join(EXTRACT_DIR, 'Jo', 'JOA*.txt')
        df_jo = parse_files(jo_pattern, parse_jo_line, 90, dedup_cols=['race_id', 'umaban'])
        if df_jo is not None:
            out_path = os.path.join(DATA_DIR, 'jrdb_jo.csv')
            df_jo.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f'  Saved: {out_path} ({len(df_jo):,} rows, {len(df_jo.columns)} cols)')

            df_jo['_year'] = df_jo['race_id'].str[:4]
            print('  Year coverage:')
            for yr, cnt in df_jo.groupby('_year').size().items():
                print(f'    {yr}: {cnt:,}')

            # Stats on gaisha (外厩) data
            gaisha_filled = df_jo['gaisha_bb'].notna() & (df_jo['gaisha_bb'] != '')
            print(f'  外厩BB filled: {gaisha_filled.sum():,}/{len(df_jo):,} ({gaisha_filled.mean()*100:.1f}%)')

    # =================================================================
    # Summary
    # =================================================================
    elapsed = (datetime.now() - start_time).total_seconds()
    print('\n' + '=' * 60)
    print(f'  Done in {elapsed:.1f}s')
    print('=' * 60)

    # Check output files
    for fname in ['jrdb_ukc.csv', 'jrdb_kka.csv', 'jrdb_kz.csv', 'jrdb_cz.csv', 'jrdb_jo.csv']:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / 1024 / 1024
            print(f'  {fname}: {size:.1f} MB')


if __name__ == '__main__':
    main()
