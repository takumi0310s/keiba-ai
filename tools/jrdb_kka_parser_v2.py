"""JRDB KKA (競走馬拡張) parser v2 — Session #53 fix.

Root cause of seiseki_* 0% NaN: existing `_parse_level_12` in
`tools/download_parse_jrdb_extra.py` calls `val_str.strip()` BEFORE the length
check, which collapses the leading whitespace of right-aligned 3-byte fields and
causes every `ZZ9*4` block to be rejected as too short.

This v2 keeps each 3-byte slice intact and lets `_safe_int` strip individually.

Read-only operation:
    - Reads:  data/jrdb/extracted/Paci/KKA*.txt
    - Writes: data/jrdb_kka_v2.csv (NEW file, original jrdb_kka.csv untouched)

Usage:
    python tools/jrdb_kka_parser_v2.py            # parse all
    python tools/jrdb_kka_parser_v2.py --sample   # parse first file only (smoke)
    python tools/jrdb_kka_parser_v2.py --max 30   # parse first N files
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Optional

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_GLOB = os.path.join(REPO, 'data', 'jrdb', 'extracted', 'Paci', 'KKA*.txt')
OUT_CSV = os.path.join(REPO, 'data', 'jrdb_kka_v2.csv')

# Spec (1-indexed start, length in bytes). Mirrors download_parse_jrdb_extra.py
# but with the strip bug removed in `_parse_level_12_v2`.
KKA_COLUMNS = [
    ('basho_code',   1,  2),
    ('year',         3,  2),
    ('kai',          5,  1),
    ('nichi',        6,  1),
    ('race_num',     7,  2),
    ('umaban',       9,  2),
    # ZZ9*4 (12 bytes = 4 x 3-digit, right-aligned with leading spaces)
    ('jra_seiseki',     11, 12),
    ('koryu_seiseki',   23, 12),
    ('other_seiseki',   35, 12),
    ('turf_dirt_2',     47, 12),
    ('turf_dirt_2_dist', 59, 12),
    ('track_seiseki',   71, 12),
    ('rotation_sei',    83, 12),
    ('kyori_seiseki',   95, 12),
    ('saka_seiseki',   107, 12),
    ('heavy_seiseki',  119, 12),
    ('rest_seiseki',   131, 12),
    ('class_seiseki',  143, 12),
    ('speed_seiseki',  155, 12),
    ('slow_seiseki',   167, 12),
    ('mid_seiseki',    179, 12),
    ('season_seiseki', 191, 12),
    ('waku_seiseki',   203, 12),
    ('breeder_dist',   215, 12),
    ('breeder_track',  227, 12),
    ('breeder_adjust', 239, 12),
    ('breeder_baba',   251, 12),
    ('breeder_blanker', 263, 12),
    ('breeder_surface', 275, 12),
    # 3- and 4-byte single numerics
    ('dam_rensho_max',   287, 3),
    ('dam_rensho_min',   290, 3),
    ('dam_rensho_avg',   293, 4),
    ('bms_rensho_max',   297, 3),
    ('bms_rensho_min',   300, 3),
    ('bms_rensho_avg',   303, 4),
]

LEVEL_FIELDS = [
    'jra_seiseki', 'koryu_seiseki', 'other_seiseki',
    'turf_dirt_2', 'turf_dirt_2_dist', 'track_seiseki',
    'rotation_sei', 'kyori_seiseki', 'saka_seiseki',
    'heavy_seiseki', 'rest_seiseki', 'class_seiseki',
    'speed_seiseki', 'slow_seiseki', 'mid_seiseki',
    'season_seiseki', 'waku_seiseki',
    'breeder_dist', 'breeder_track', 'breeder_adjust',
    'breeder_baba', 'breeder_blanker', 'breeder_surface',
]


def _safe_int(s: str) -> Optional[int]:
    s = s.strip()
    if not s or s == '-':
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _field(line_bytes: bytes, start: int, length: int) -> str:
    s = start - 1
    return line_bytes[s:s + length].decode('cp932', errors='replace')


def _hex_day(nichi_str: str) -> int:
    """JRDB の日 field は 1-9 / A-F の 16 進。 数値化する。"""
    s = nichi_str.strip()
    if not s:
        return 0
    try:
        return int(s, 16)
    except ValueError:
        return 0


def _build_race_id(basho: str, year: str, kai: str, nichi: str, race_num: str) -> str:
    yr2 = year.strip()
    yr4 = f'20{yr2}' if len(yr2) == 2 else yr2
    return f'{yr4}{basho.strip()}{int(kai):02d}{_hex_day(nichi):02d}{int(race_num):02d}'


def _parse_level_12_v2(val_str: str):
    """12 byte の ZZ9*4 を 3-byte ずつ切って int 化 (strip は _safe_int 内部)。"""
    if len(val_str) < 12:
        return None, None, None, None
    return (
        _safe_int(val_str[0:3]),
        _safe_int(val_str[3:6]),
        _safe_int(val_str[6:9]),
        _safe_int(val_str[9:12]),
    )


def parse_kka_line_v2(line_bytes: bytes) -> dict:
    row = {}
    for name, start, length in KKA_COLUMNS:
        row[name] = _field(line_bytes, start, length)

    row['race_id'] = _build_race_id(
        row['basho_code'], row['year'], row['kai'], row['nichi'], row['race_num']
    )
    row['umaban'] = _safe_int(row['umaban'])

    for f in LEVEL_FIELDS:
        w, p2, p3, out = _parse_level_12_v2(row[f])
        row[f'{f}_1'] = w
        row[f'{f}_2'] = p2
        row[f'{f}_3'] = p3
        row[f'{f}_out'] = out
        del row[f]

    for k in ['dam_rensho_max', 'dam_rensho_min', 'dam_rensho_avg',
              'bms_rensho_max', 'bms_rensho_min', 'bms_rensho_avg']:
        row[k] = _safe_int(row[k])

    for k in ['basho_code', 'year', 'kai', 'nichi', 'race_num']:
        del row[k]

    return row


def parse_files_v2(file_list, min_line_len: int = 280) -> pd.DataFrame:
    rows = []
    errors = 0
    for fpath in file_list:
        try:
            with open(fpath, 'rb') as f:
                for line_bytes in f:
                    line_bytes = line_bytes.rstrip(b'\r\n')
                    if len(line_bytes) < min_line_len:
                        continue
                    try:
                        rows.append(parse_kka_line_v2(line_bytes))
                    except Exception:
                        errors += 1
        except Exception:
            errors += 1
    print(f'  Parsed: {len(rows):,} rows ({errors} errors)')
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates(subset=['race_id', 'umaban'], keep='last')
    if before != len(df):
        print(f'  Deduplicated: {before:,} -> {len(df):,}')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', action='store_true', help='Parse only first file (smoke test)')
    ap.add_argument('--max', type=int, default=0, help='Parse only first N files (0=all)')
    ap.add_argument('--out', default=OUT_CSV)
    args = ap.parse_args()

    files = sorted(glob.glob(SRC_GLOB))
    print(f'KKA files found: {len(files)}')
    if args.sample:
        files = files[:1]
    elif args.max > 0:
        files = files[:args.max]
    print(f'Parsing: {len(files)} files')

    df = parse_files_v2(files)
    if df.empty:
        print('No rows parsed.')
        return

    seiseki_cols = [c for c in df.columns if c.endswith(('_1', '_2', '_3', '_out'))]
    overall_non_null = df[seiseki_cols].notna().any(axis=1).mean() * 100
    jra_non_null = df['jra_seiseki_1'].notna().mean() * 100
    kyori_non_null = df['kyori_seiseki_1'].notna().mean() * 100
    track_non_null = df['track_seiseki_1'].notna().mean() * 100

    print('--- Coverage (% non-null) ---')
    print(f'  any seiseki_*: {overall_non_null:.1f}%')
    print(f'  jra_seiseki_1: {jra_non_null:.1f}%')
    print(f'  kyori_seiseki_1: {kyori_non_null:.1f}%')
    print(f'  track_seiseki_1: {track_non_null:.1f}%')

    df['_year'] = df['race_id'].str[:4]
    print('--- Year coverage (rows) ---')
    for yr, cnt in df.groupby('_year').size().items():
        print(f'    {yr}: {cnt:,}')
    df = df.drop(columns=['_year'])

    df.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f'Saved: {args.out} ({len(df):,} rows, {len(df.columns)} cols)')


if __name__ == '__main__':
    main()
