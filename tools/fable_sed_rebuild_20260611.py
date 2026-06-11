#!/usr/bin/env python3
"""jrdb_sed.csv 完全再構築 (2026-06-11 Fable sweep 復旧)。

経緯: scrape_jrdb.save_csv の dedup キー (jra_race_id) が bulk 版 jrdb_sed.csv の
スキーマ (race_id) と不一致 → 旧 548,780 行が NaN キー同士の重複と見なされ 1 行に
潰された (バックフィル実行時に発覚)。

復旧: data/jrdb/extracted/Sed の全 txt (歴史) + data/jrdb_raw/sed の lzh (5/10-6/7 含む)
を parse_jrdb.parse_sed_line で全量パースし直し、bulk 版と同一スキーマで再構築する。
書き込み前に現状を .bak_20260611_destroyed として退避。
"""
from __future__ import annotations
import os, sys, glob, shutil
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'tools'))
from parse_jrdb import parse_sed_line  # bulk 版と同一パーサ
from scrape_jrdb import extract_lzh

EXTRACTED = os.path.join(BASE, 'data', 'jrdb', 'extracted', 'Sed')
RAW_LZH = os.path.join(BASE, 'data', 'jrdb_raw', 'sed')
OUT = os.path.join(BASE, 'data', 'jrdb_sed.csv')

SED_KEY_COLS = [
    'race_id', 'umaban', 'horse_name', 'finish', 'abnormal',
    'distance', 'surface_code', 'num_horses',
    'time_sec', 'weight_carry', 'odds_final', 'popularity',
    'idm', 'soten', 'baba_sa', 'ten_idx', 'agari_idx',
    'pace_idx', 'race_pace_idx', 'pace', 'deokure', 'ichi_dori',
    'furi', 'mae_furi', 'naka_furi', 'ato_furi',
    'race_idx', 'course_dori', 'josho_code',
    'batai_code', 'kehai_code', 'race_pace', 'uma_pace',
    'first_3f', 'last_3f', 'corner1', 'corner2', 'corner3', 'corner4',
    'horse_weight', 'weight_diff', 'race_style',
    'jockey_code', 'trainer_code', 'blood_num', 'yyyymmdd',
    'race_name', 'grade', 'class_code',
]


def parse_lines(data: bytes, rows: list, errs: list):
    for line in data.split(b'\n'):
        line = line.rstrip(b'\r')
        if len(line) < 340:
            continue
        try:
            rows.append(parse_sed_line(line))
        except Exception:
            errs.append(1)


def main():
    rows, errs = [], []
    # 1) 歴史 extracted txt
    txts = sorted(glob.glob(os.path.join(EXTRACTED, 'SED*.txt')))
    print(f'extracted txt: {len(txts)}')
    for i, p in enumerate(txts, 1):
        with open(p, 'rb') as f:
            parse_lines(f.read(), rows, errs)
        if i % 500 == 0:
            print(f'  {i}/{len(txts)} rows={len(rows)}')
    n_hist = len(rows)
    print(f'歴史分: {n_hist:,} rows ({len(errs)} errors)')

    # 2) lzh (extracted に txt が無い日付分のみ)
    have = {os.path.basename(p).upper() for p in txts}
    lzhs = sorted(glob.glob(os.path.join(RAW_LZH, 'SED*.lzh')))
    n_lzh_used = 0
    for p in lzhs:
        txt_name = os.path.basename(p).upper().replace('.LZH', '.TXT')
        if txt_name in have:
            continue
        try:
            for fname, data in extract_lzh(open(p, 'rb').read()).items():
                if fname.upper().startswith('SED'):
                    parse_lines(data, rows, errs)
            n_lzh_used += 1
        except Exception as e:
            print(f'  [WARN] {p}: {e}')
    print(f'lzh 追加: {n_lzh_used} files → 計 {len(rows):,} rows')

    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates(subset=['race_id', 'umaban'], keep='last')
    print(f'dedup: {before:,} → {len(df):,}')
    avail = [c for c in SED_KEY_COLS if c in df.columns]
    df = df[avail]
    if 'yyyymmdd' in df.columns:
        print('yyyymmdd max:', pd.to_numeric(df['yyyymmdd'], errors='coerce').max())

    assert len(df) > 500000, f'再構築行数が少なすぎる: {len(df)}'
    bak = OUT + '.bak_20260611_destroyed'
    if os.path.exists(OUT) and not os.path.exists(bak):
        shutil.move(OUT, bak)
    df.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f'-> {OUT} ({len(df):,} rows, {len(avail)} cols) 再構築完了')


if __name__ == '__main__':
    main()
