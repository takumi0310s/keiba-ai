#!/usr/bin/env python3
"""JRDB KTA / MZA / MSA scraper (火曜先行 出走確定 検知用)

Task #12 (2026/05/11):
  - **KTA** (登録馬データ、 火曜 20:00 頃 更新): 今週の出走登録馬の事前 IDM/騎手/枠順
  - **MZA** (全抹消馬、 火曜 19:00): 全抹消馬の snapshot
  - **MSA** (抹消馬差分、 火曜 19:00): 直近 抹消馬 差分

KTA は既存 tools/download_parse_jrdb_batch2.py に embed されているが、
本 script は KTA + MZA + MSA を standalone で取得する独立ツール。
火曜 schtask 登録 用途。

V15 投資保護:
  - predict_core / daily_predict / model には一切触れない
  - 既存 jrdb_kta.csv は 上書きしない (append + dedup)
  - 新規 csv (jrdb_mza.csv / jrdb_msa.csv) を 追加するのみ

Usage:
  python tools/scrape_jrdb_kta_mza.py --dry-run             # 何取得するか確認
  python tools/scrape_jrdb_kta_mza.py                       # 直近 4 週分
  python tools/scrape_jrdb_kta_mza.py --types KTA           # KTA のみ
  python tools/scrape_jrdb_kta_mza.py --date 20260512       # 指定日
  python tools/scrape_jrdb_kta_mza.py --range 20260101 20260512

Reference:
  http://www.jrdb.com/program/jrdb_data_doc.html
  KTA: 386-byte fixed-length record (登録馬)
  MZA: 全抹消馬 snapshot (~50-byte/record, 競走馬名 + 血統登録番号 + 抹消日)
  MSA: 抹消差分 (1 週分の MZA 差分)
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd
from requests.auth import HTTPBasicAuth

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

# dotenv
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'jrdb_raw')

# JRDB 認証
JRDB_ID = os.environ.get('JRDB_ID', '')
JRDB_PASSWORD = os.environ.get('JRDB_PASSWORD', '')
JRDB_BASE = 'http://www.jrdb.com/member/data'

# 7-Zip path candidates (LZH 解凍用)
SEVEN_ZIP_CANDIDATES = [
    r'C:\Program Files\7-Zip\7z.exe',
    r'C:\Program Files (x86)\7-Zip\7z.exe',
    '7z',
]

# JRDB ファイル URL pattern (yymmdd 6 桁)
JRDB_URLS = {
    'KTA': f'{JRDB_BASE}/Kta/KTA{{ymd}}.lzh',
    'MZA': f'{JRDB_BASE}/Mza/MZA{{ymd}}.lzh',
    'MSA': f'{JRDB_BASE}/Msa/MSA{{ymd}}.lzh',
}

CSV_PATHS = {
    'KTA': os.path.join(DATA_DIR, 'jrdb_kta.csv'),
    'MZA': os.path.join(DATA_DIR, 'jrdb_mza.csv'),
    'MSA': os.path.join(DATA_DIR, 'jrdb_msa.csv'),
}


# ===================================================
# helpers
# ===================================================

def _safe_int(s: str):
    s = s.strip()
    if not s or s == '-':
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _safe_float(s: str):
    s = s.strip()
    if not s or s == '-':
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _field(line_bytes: bytes, start: int, length: int) -> str:
    s = start - 1
    return line_bytes[s:s + length].decode('cp932', errors='replace')


def _hex_day(c: str) -> int:
    try:
        return int(c.strip(), 16)
    except (ValueError, TypeError):
        return 0


def _build_race_id(basho: str, year: str, kai: str, nichi: str, race_num: str) -> str:
    yr2 = year.strip()
    yr4 = f'20{yr2}' if len(yr2) == 2 else yr2
    return f'{yr4}{basho.strip()}{int(kai):02d}{_hex_day(nichi):02d}{int(race_num):02d}'


# ===================================================
# Tuesdays (KTA/MZA/MSA 火曜更新)
# ===================================================

def _tuesdays(start: datetime, end: datetime) -> list[datetime]:
    out = []
    d = start
    while d <= end:
        if d.weekday() == 1:  # Tue
            out.append(d)
        d += timedelta(days=1)
    return out


# ===================================================
# 7-Zip / extract
# ===================================================

def _find_7zip() -> str | None:
    for p in SEVEN_ZIP_CANDIDATES:
        if p == '7z':
            try:
                subprocess.run([p, '--help'], capture_output=True, timeout=5)
                return p
            except FileNotFoundError:
                continue
            except Exception:
                continue
        if os.path.exists(p):
            return p
    return None


def extract_lzh_to_dir(lzh_bytes: bytes, out_dir: str) -> list[str]:
    """LZH バイト列 → out_dir に展開、 展開ファイル path list を返す"""
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix='.lzh', delete=False, dir=out_dir)
    tmp.write(lzh_bytes)
    tmp.close()

    extracted: list[str] = []
    sz = _find_7zip()
    if sz:
        try:
            subprocess.run([sz, 'x', '-y', f'-o{out_dir}', tmp.name],
                           capture_output=True, check=True, timeout=60)
            # 展開ファイル: tmp.name 自体ではなく out_dir 内の .txt / .dat
            for f in os.listdir(out_dir):
                fp = os.path.join(out_dir, f)
                if fp == tmp.name:
                    continue
                if os.path.isfile(fp) and not f.lower().endswith('.lzh'):
                    extracted.append(fp)
        except subprocess.CalledProcessError:
            pass
        except FileNotFoundError:
            pass

    # fallback: lhafile
    if not extracted:
        try:
            import lhafile  # type: ignore
            lha = lhafile.Lhafile(io.BytesIO(lzh_bytes))
            for info in lha.infolist():
                fname = info.filename
                data = lha.read(fname)
                out_path = os.path.join(out_dir, os.path.basename(fname))
                with open(out_path, 'wb') as fh:
                    fh.write(data)
                extracted.append(out_path)
        except ImportError:
            pass
        except Exception:
            pass

    # cleanup
    try:
        os.unlink(tmp.name)
    except OSError:
        pass

    return extracted


# ===================================================
# Download
# ===================================================

def download_lzh(file_type: str, ymd: str, force: bool = False) -> bytes | None:
    """JRDB から 1 ファイル DL。 cache 済みならそれを返す。

    Args:
        file_type: 'KTA' / 'MZA' / 'MSA'
        ymd: YYMMDD (6 桁)
    """
    if not JRDB_ID or not JRDB_PASSWORD:
        print('[ERROR] JRDB_ID / JRDB_PASSWORD が .env にありません')
        return None

    cache_dir = os.path.join(RAW_DIR, file_type.lower())
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{file_type}{ymd}.lzh')

    if os.path.exists(cache_path) and not force:
        with open(cache_path, 'rb') as f:
            return f.read()

    url = JRDB_URLS[file_type].format(ymd=ymd)
    try:
        resp = requests.get(url, auth=HTTPBasicAuth(JRDB_ID, JRDB_PASSWORD), timeout=30)
    except requests.RequestException as e:
        print(f'  [ERROR] {url} - {e}')
        return None

    if resp.status_code == 401:
        print('[ERROR] JRDB 認証失敗')
        return None
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        print(f'  [WARN] {url} -> HTTP {resp.status_code}')
        return None
    if len(resp.content) < 100:
        return None

    with open(cache_path, 'wb') as f:
        f.write(resp.content)
    return resp.content


# ===================================================
# Parsers
# ===================================================

def parse_kta_line(line_bytes: bytes) -> dict:
    """KTA 登録馬データ (record=386 byte)

    既存 tools/download_parse_jrdb_batch2.py の parse_kta_line と同等 schema。
    主要 column: race_id, blood_num, horse_name, jockey_name, weight_carry,
    idm, ten_idx_pred, pace_idx_pred, agari_idx_pred, ichi_idx_pred
    """
    row: dict = {}
    row['basho_code'] = _field(line_bytes, 1, 2)
    row['year']       = _field(line_bytes, 3, 2)
    row['kai']        = _field(line_bytes, 5, 1)
    row['nichi']      = _field(line_bytes, 6, 1)
    row['race_num']   = _field(line_bytes, 7, 2)
    row['race_id']    = _build_race_id(row['basho_code'], row['year'],
                                        row['kai'], row['nichi'], row['race_num'])
    row['horse_key']   = _field(line_bytes, 9, 40).strip()
    row['blood_num']   = _field(line_bytes, 49, 8).strip()
    row['horse_name']  = _field(line_bytes, 57, 36).strip()
    row['sex_code']    = _field(line_bytes, 93, 1).strip()
    row['keito_code']  = _field(line_bytes, 94, 2).strip()
    row['blinker']     = _field(line_bytes, 96, 1).strip()
    row['jockey_name'] = _field(line_bytes, 97, 12).strip()

    wc = _safe_float(_field(line_bytes, 109, 3))
    row['weight_carry'] = wc / 10.0 if wc is not None else None
    row['apprentice']   = _safe_int(_field(line_bytes, 112, 1))
    row['trainer_name'] = _field(line_bytes, 113, 12).strip()
    row['trainer_area'] = _field(line_bytes, 125, 4).strip()
    row['idm']          = _safe_float(_field(line_bytes, 129, 5))
    row['josho']        = _safe_int(_field(line_bytes, 134, 1))
    row['rotation']     = _safe_int(_field(line_bytes, 135, 3))
    row['kyakushitsu']  = _safe_int(_field(line_bytes, 138, 1))
    row['condition_class']  = _safe_int(_field(line_bytes, 139, 1))
    row['condition_class2'] = _safe_int(_field(line_bytes, 140, 1))
    row['turf_apt']  = _field(line_bytes, 141, 1).strip()
    row['dirt_apt']  = _field(line_bytes, 142, 1).strip()
    row['heavy_apt'] = _safe_int(_field(line_bytes, 143, 1))
    row['class_code']    = _safe_int(_field(line_bytes, 144, 2))
    row['hair_color']    = _safe_int(_field(line_bytes, 148, 2))
    row['prev1_sed_key'] = _field(line_bytes, 150, 16).strip()
    row['jockey_code']   = _field(line_bytes, 270, 5).strip()
    row['trainer_code']  = _field(line_bytes, 275, 5).strip()
    row['prize_total']   = _safe_int(_field(line_bytes, 280, 6))
    row['prize_shutoku'] = _safe_int(_field(line_bytes, 286, 5))
    row['shutoku_class'] = _safe_int(_field(line_bytes, 291, 1))
    row['ten_idx_pred']   = _safe_float(_field(line_bytes, 342, 5))
    row['pace_idx_pred']  = _safe_float(_field(line_bytes, 347, 5))
    row['agari_idx_pred'] = _safe_float(_field(line_bytes, 352, 5))
    row['ichi_idx_pred']  = _safe_float(_field(line_bytes, 357, 5))
    row['data_ku'] = _field(line_bytes, 377, 1).strip()

    for k in ('basho_code', 'year', 'kai', 'nichi', 'race_num'):
        row.pop(k, None)
    return row


def parse_mza_line(line_bytes: bytes) -> dict:
    """MZA / MSA 抹消馬データ (record≧50 byte 想定)

    JRDB 抹消馬データ レイアウト (公開フォーマット仕様より):
      1-8   血統登録番号 (8 byte)
      9-44  競走馬名     (36 byte)
      45-52 抹消年月日   (YYYYMMDD, 8 byte)
      53-54 抹消事由コード (2 byte)  [一部 file 形式で省略あり → optional]

    MSA は MZA と同 layout の差分 file。
    """
    row: dict = {}
    row['blood_num']   = _field(line_bytes, 1, 8).strip()
    row['horse_name']  = _field(line_bytes, 9, 36).strip()
    row['delete_date'] = _field(line_bytes, 45, 8).strip()
    # 余裕があれば理由コード
    if len(line_bytes) >= 54:
        row['delete_reason'] = _field(line_bytes, 53, 2).strip()
    else:
        row['delete_reason'] = ''
    return row


def parse_file(file_type: str, fpath: str, min_line_len: int) -> list[dict]:
    parser = {'KTA': parse_kta_line,
              'MZA': parse_mza_line,
              'MSA': parse_mza_line}[file_type]
    rows: list[dict] = []
    errors = 0
    try:
        with open(fpath, 'rb') as f:
            for line in f:
                line = line.rstrip(b'\r\n')
                if len(line) < min_line_len:
                    continue
                try:
                    rows.append(parser(line))
                except Exception:
                    errors += 1
    except OSError:
        errors += 1
    if errors:
        print(f'    [WARN] {os.path.basename(fpath)}: parse errors={errors}')
    return rows


# ===================================================
# Save
# ===================================================

def save_csv(file_type: str, new_rows: list[dict], append: bool = True) -> int:
    """CSV へ append + dedup。 新規行数を返す。"""
    if not new_rows:
        return 0
    path = CSV_PATHS[file_type]
    df_new = pd.DataFrame(new_rows)

    if append and os.path.exists(path):
        try:
            df_old = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
        except Exception:
            df_old = pd.read_csv(path, encoding='utf-8', dtype=str)
        df_new_str = df_new.astype(str)
        merged = pd.concat([df_old, df_new_str], ignore_index=True)
        # dedup keys
        if file_type == 'KTA':
            keys = ['race_id', 'blood_num']
        else:
            keys = ['blood_num', 'delete_date']
        keys = [k for k in keys if k in merged.columns]
        if keys:
            merged = merged.drop_duplicates(subset=keys, keep='last')
        merged.to_csv(path, index=False, encoding='utf-8-sig')
        print(f'  Saved (append): {path} -> {len(merged):,} total rows '
              f'(+{len(merged) - len(df_old):,} new)')
        return len(merged) - len(df_old)

    df_new.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'  Saved (new): {path} -> {len(df_new):,} rows')
    return len(df_new)


# ===================================================
# Per-date pipeline
# ===================================================

MIN_LINE_LEN = {'KTA': 340, 'MZA': 44, 'MSA': 44}


def fetch_and_parse(file_type: str, ymd: str, force: bool = False) -> list[dict]:
    """1 日分 DL → 展開 → parse → row list"""
    lzh = download_lzh(file_type, ymd, force=force)
    if not lzh:
        return []

    work_dir = tempfile.mkdtemp(prefix=f'jrdb_{file_type.lower()}_')
    try:
        extracted = extract_lzh_to_dir(lzh, work_dir)
        if not extracted:
            return []
        rows: list[dict] = []
        for fpath in extracted:
            if file_type.upper() not in os.path.basename(fpath).upper():
                # 関係ない添付 file は skip
                continue
            rows.extend(parse_file(file_type, fpath, MIN_LINE_LEN[file_type]))
        return rows
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ===================================================
# Dry-run
# ===================================================

def dry_run(file_types: list[str], dates: list[datetime]) -> None:
    print('\n[dry-run]')
    print(f'  types : {", ".join(file_types)}')
    print(f'  dates : {len(dates)} (Tuesdays) '
          f'[{dates[0].strftime("%Y-%m-%d") if dates else "-"}'
          f' ... {dates[-1].strftime("%Y-%m-%d") if dates else "-"}]')
    print(f'  expected URLs:')
    for d in dates:
        ymd = d.strftime('%y%m%d')
        for t in file_types:
            print(f'    {JRDB_URLS[t].format(ymd=ymd)}')
    print(f'  outputs:')
    for t in file_types:
        p = CSV_PATHS[t]
        exists = os.path.exists(p)
        size = os.path.getsize(p) if exists else 0
        print(f'    {p}  exists={exists}  size={size:,} B')


# ===================================================
# main
# ===================================================

def main() -> None:
    ap = argparse.ArgumentParser(description='JRDB KTA / MZA / MSA scraper')
    ap.add_argument('--date', type=str, default='',
                    help='single date YYYYMMDD')
    ap.add_argument('--range', nargs=2, type=str, metavar=('START', 'END'),
                    help='date range YYYYMMDD YYYYMMDD (Tuesdays in range are scraped)')
    ap.add_argument('--weeks', type=int, default=4,
                    help='直近 N 週 (火曜のみ; default 4)')
    ap.add_argument('--types', type=str, default='KTA,MZA,MSA',
                    help='comma-separated: KTA,MZA,MSA')
    ap.add_argument('--force', action='store_true',
                    help='LZH cache を無視して再 DL')
    ap.add_argument('--dry-run', action='store_true',
                    help='URL list のみ表示')
    args = ap.parse_args()

    file_types = [t.strip().upper() for t in args.types.split(',') if t.strip()]
    for t in file_types:
        if t not in JRDB_URLS:
            print(f'[ERROR] unknown type: {t}')
            sys.exit(2)

    # 取得対象日付 (火曜のみ)
    if args.range:
        start = datetime.strptime(args.range[0], '%Y%m%d')
        end   = datetime.strptime(args.range[1], '%Y%m%d')
        dates = _tuesdays(start, end)
    elif args.date:
        d = datetime.strptime(args.date, '%Y%m%d')
        dates = [d]
    else:
        end = datetime.now()
        start = end - timedelta(weeks=args.weeks)
        dates = _tuesdays(start, end)

    print('=' * 60)
    print('JRDB KTA / MZA / MSA scraper')
    print(f'  types: {", ".join(file_types)}')
    print(f'  dates: {len(dates)} dates '
          f'({dates[0].strftime("%Y-%m-%d") if dates else "-"} '
          f'... {dates[-1].strftime("%Y-%m-%d") if dates else "-"})')
    print(f'  now  : {datetime.now().isoformat(timespec="seconds")}')
    print('=' * 60)

    if args.dry_run:
        dry_run(file_types, dates)
        return

    if not JRDB_ID or not JRDB_PASSWORD:
        print('[ERROR] JRDB_ID / JRDB_PASSWORD が .env にありません')
        sys.exit(1)

    # SCRAPER-GUARD (KTA/MZA/MSA は火曜更新なので通常時間帯)
    try:
        from tools.scraper_guard import is_scraping_allowed  # type: ignore
        if not is_scraping_allowed(caller='scrape_jrdb_kta_mza'):
            print('[GUARD] SCRAPER-GUARD 時間帯につき終了します')
            sys.exit(0)
    except ImportError:
        pass

    os.makedirs(RAW_DIR, exist_ok=True)

    stats: dict[str, dict] = {t: {'dates': 0, 'rows': 0, 'new': 0, 'errors': 0}
                              for t in file_types}

    for t in file_types:
        print(f'\n=== {t} ===')
        all_rows: list[dict] = []
        for d in dates:
            ymd = d.strftime('%y%m%d')
            print(f'  [{d.strftime("%Y-%m-%d")}] {t}{ymd}.lzh', end=' ', flush=True)
            try:
                rows = fetch_and_parse(t, ymd, force=args.force)
            except Exception as e:
                print(f'-> ERROR {e}')
                stats[t]['errors'] += 1
                continue
            if rows:
                all_rows.extend(rows)
                stats[t]['dates'] += 1
                stats[t]['rows'] += len(rows)
                print(f'-> {len(rows):,} rows')
            else:
                print(f'-> 0 rows / 404')
            time.sleep(1)  # rate limit
        # save
        new_added = save_csv(t, all_rows, append=True)
        stats[t]['new'] = new_added

    print('\n' + '=' * 60)
    print('[summary]')
    for t, s in stats.items():
        print(f'  {t}: dates={s["dates"]}/{len(dates)}  '
              f'rows={s["rows"]:,}  new_added={s["new"]:,}  errors={s["errors"]}')
    print('=' * 60)


if __name__ == '__main__':
    main()
