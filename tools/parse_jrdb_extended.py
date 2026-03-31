#!/usr/bin/env python3
"""JRDBの拡張データ(SRB/ZED/ZKB)をパースしてCSV化する。

SRB: 成績レースデータ（レース単位: ハロンタイム・トラックバイアス・レースコメント）
ZED: 前走データ（馬単位: SED形式の過去走成績）
ZKB: 前走拡張データ（馬単位: コメント・馬場適性コード等）

Usage:
    python tools/parse_jrdb_extended.py
    python tools/parse_jrdb_extended.py --types srb zed zkb
"""
import os
import sys
import io
import glob
import argparse
import pandas as pd
from datetime import datetime

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_DIR = os.path.join(BASE_DIR, 'data', 'jrdb', 'extracted')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Import shared utilities from parse_jrdb
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
from parse_jrdb import _hex_day, _safe_float, _safe_int, _field, _build_race_id, \
    SED_COLUMNS, SED_INT_COLS, SED_FLOAT_COLS, parse_sed_line


# =============================================================================
# SRB パーサー (成績レースデータ、レコード長852バイト、レース単位)
# =============================================================================
# Spec: data/jrdb_srb_spec.txt
SRB_RECORD_LEN = 852
SRB_MIN_LEN = 340  # minimum usable bytes


def parse_srb_line(line_bytes):
    """SRBの1行(bytes)をパースして辞書を返す。"""
    row = {}

    # レースキー (pos 1-8)
    basho_code = _field(line_bytes, 1, 2)
    year = _field(line_bytes, 3, 2)
    kai = _field(line_bytes, 5, 1)
    nichi = _field(line_bytes, 6, 1)
    race_num = _field(line_bytes, 7, 2)

    row['race_id'] = _build_race_id(basho_code, year, kai, nichi, race_num)

    # ハロンタイム 18本 × 3バイト (pos 9-62, 0.1秒単位)
    harlon_times = []
    for j in range(18):
        start = 9 + j * 3  # 1-based
        val = _safe_int(_field(line_bytes, start, 3))
        if val is not None and val > 0:
            harlon_times.append(val / 10.0)  # 0.1秒→秒
        else:
            harlon_times.append(None)
    row['harlon_count'] = sum(1 for h in harlon_times if h is not None)
    # Store individual harlon times
    for j in range(18):
        row[f'harlon_{j+1}'] = harlon_times[j]

    # コーナー位置取り (pos 63-318, 4×64バイト)
    row['corner1_pos'] = _field(line_bytes, 63, 64).strip()
    row['corner2_pos'] = _field(line_bytes, 127, 64).strip()
    row['corner3_pos'] = _field(line_bytes, 191, 64).strip()
    row['corner4_pos'] = _field(line_bytes, 255, 64).strip()

    # ペースアップ位置 (pos 319-320)
    row['pace_up_pos'] = _safe_int(_field(line_bytes, 319, 2))

    # トラックバイアス (pos 321-342)
    # 1角 (3: inner,mid,outer), 2角, 向正, 3角: each 3 bytes
    # 4角, 直線: each 5 bytes (最内,内,中,外,大外)
    tb_1c = _field(line_bytes, 321, 3)
    tb_2c = _field(line_bytes, 324, 3)
    tb_str = _field(line_bytes, 327, 3)  # 向正面
    tb_3c = _field(line_bytes, 330, 3)
    tb_4c = _field(line_bytes, 333, 5)
    tb_fin = _field(line_bytes, 338, 5)  # 直線

    # Parse each byte as track bias code (1-5 for turf, 1-3 for dirt)
    def _parse_tb(s):
        return [_safe_int(c) for c in s]

    row['tb_1corner'] = tb_1c.strip()
    row['tb_2corner'] = tb_2c.strip()
    row['tb_backstr'] = tb_str.strip()
    row['tb_3corner'] = tb_3c.strip()
    row['tb_4corner'] = tb_4c.strip()
    row['tb_homestr'] = tb_fin.strip()

    # レースコメント (pos 343-842, 500バイト, CP932)
    if len(line_bytes) >= 842:
        row['race_comment'] = _field(line_bytes, 343, 500).strip()
    else:
        row['race_comment'] = ''

    return row


# =============================================================================
# ZKB パーサー (前走拡張データ、レコード長304バイト)
# =============================================================================
# ZKB = 前走版のSKB (成績拡張データ)
# Same format as SKB: 騎手コード, 馬場コード, 脚質コード, コメント群, 馬場適性
ZKB_RECORD_LEN = 304
ZKB_MIN_LEN = 270


def parse_zkb_line(line_bytes):
    """ZKBの1行(bytes)をパースして辞書を返す。"""
    row = {}

    # レースキー (pos 1-8) - 当該レースのキー
    basho_code = _field(line_bytes, 1, 2)
    year = _field(line_bytes, 3, 2)
    kai = _field(line_bytes, 5, 1)
    nichi = _field(line_bytes, 6, 1)
    race_num = _field(line_bytes, 7, 2)
    row['race_id'] = _build_race_id(basho_code, year, kai, nichi, race_num)

    # 馬番 (pos 9-10)
    row['umaban'] = _safe_int(_field(line_bytes, 9, 2))

    # 競走成績キー
    row['blood_num'] = _field(line_bytes, 11, 8).strip()
    row['prev_date'] = _field(line_bytes, 19, 8).strip()  # 前走日 YYYYMMDD

    # 騎手コード 6項目 × 3バイト (pos 27-44)
    # Spec: 騎手コード表参照
    kishi_codes = []
    for j in range(6):
        start = 27 + j * 3
        kishi_codes.append(_safe_int(_field(line_bytes, start, 3)))
    row['kishi_code_1'] = kishi_codes[0]
    row['kishi_code_2'] = kishi_codes[1]
    row['kishi_code_3'] = kishi_codes[2]
    row['kishi_code_4'] = kishi_codes[3]
    row['kishi_code_5'] = kishi_codes[4]
    row['kishi_code_6'] = kishi_codes[5]

    # 馬場コード 8項目 × 3バイト (pos 45-68)
    baba_codes = []
    for j in range(8):
        start = 45 + j * 3
        baba_codes.append(_safe_int(_field(line_bytes, start, 3)))
    row['baba_code_1'] = baba_codes[0]
    row['baba_code_2'] = baba_codes[1]
    row['baba_code_3'] = baba_codes[2]
    row['baba_code_4'] = baba_codes[3]
    row['baba_code_5'] = baba_codes[4]
    row['baba_code_6'] = baba_codes[5]
    row['baba_code_7'] = baba_codes[6]
    row['baba_code_8'] = baba_codes[7]

    # 脚質コード 5ポジション × 3×3バイト (pos 69-113)
    kyaku_codes = []
    for j in range(5):
        pos_codes = []
        for k in range(3):
            start = 69 + j * 9 + k * 3
            pos_codes.append(_safe_int(_field(line_bytes, start, 3)))
        kyaku_codes.append(pos_codes)
    row['kyaku_left'] = kyaku_codes[0][0]
    row['kyaku_left_front'] = kyaku_codes[1][0]
    row['kyaku_right_front'] = kyaku_codes[2][0]
    row['kyaku_left_rear'] = kyaku_codes[3][0]
    row['kyaku_right_rear'] = kyaku_codes[4][0]

    # コメント群 (各40バイト CP932)
    row['padock_comment'] = _field(line_bytes, 114, 40).strip()
    row['kyaku_comment'] = _field(line_bytes, 154, 40).strip()
    row['baba_comment'] = _field(line_bytes, 194, 40).strip()
    row['race_comment'] = _field(line_bytes, 234, 40).strip()

    # 入力用データ (第2版追加)
    if len(line_bytes) >= 291:
        row['turf_hoof'] = _safe_int(_field(line_bytes, 274, 3))
        row['run_stage'] = _safe_int(_field(line_bytes, 277, 3))
        row['anshin'] = _safe_int(_field(line_bytes, 280, 3))
        row['heavy_apt'] = _safe_int(_field(line_bytes, 283, 3))
        row['aisho'] = _safe_int(_field(line_bytes, 286, 3))
        row['niwa'] = _safe_int(_field(line_bytes, 289, 3))

    return row


# =============================================================================
# メイン処理
# =============================================================================
def parse_srb(verbose=True):
    """SRBファイルをパースしてDataFrameを返す。"""
    if verbose:
        print('\n--- SRB (成績レースデータ: ハロンタイム・バイアス・コメント) ---')

    srb_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, 'Sed', 'SRB*.txt')))
    if verbose:
        print(f'  {len(srb_files)} files found')

    rows = []
    errors = 0

    for fpath in srb_files:
        try:
            with open(fpath, 'rb') as f:
                for line_bytes in f:
                    line_bytes = line_bytes.rstrip(b'\r\n')
                    if len(line_bytes) < SRB_MIN_LEN:
                        continue
                    try:
                        rows.append(parse_srb_line(line_bytes))
                    except Exception:
                        errors += 1
        except Exception:
            errors += 1

    if verbose:
        print(f'  Parsed: {len(rows):,} rows ({errors} errors)')

    if not rows:
        if verbose:
            print('  No SRB data found!')
        return None

    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates(subset=['race_id'], keep='last')
    if verbose:
        print(f'  Deduplicated: {before:,} → {len(df):,}')

    return df


def parse_zed(verbose=True):
    """ZEDファイルをパースしてDataFrameを返す。SED形式を流用。"""
    if verbose:
        print('\n--- ZED (前走データ: SED形式の過去走成績) ---')

    zed_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, 'Paci', 'ZED*.txt')))
    if verbose:
        print(f'  {len(zed_files)} files found')

    rows = []
    errors = 0

    for fpath in zed_files:
        try:
            with open(fpath, 'rb') as f:
                for line_bytes in f:
                    line_bytes = line_bytes.rstrip(b'\r\n')
                    if len(line_bytes) < 340:
                        continue
                    try:
                        row = parse_sed_line(line_bytes)
                        rows.append(row)
                    except Exception:
                        errors += 1
        except Exception:
            errors += 1

    if verbose:
        print(f'  Parsed: {len(rows):,} rows ({errors} errors)')

    if not rows:
        if verbose:
            print('  No ZED data found!')
        return None

    df = pd.DataFrame(rows)
    before = len(df)
    # ZED has multiple rows per horse (past 5 races) → deduplicate by race_id + umaban + date
    if 'yyyymmdd' in df.columns:
        df = df.drop_duplicates(subset=['race_id', 'umaban', 'yyyymmdd'], keep='last')
    else:
        df = df.drop_duplicates(subset=['race_id', 'umaban'], keep='last')
    if verbose:
        print(f'  Deduplicated: {before:,} → {len(df):,}')

    return df


def parse_zkb(verbose=True):
    """ZKBファイルをパースしてDataFrameを返す。"""
    if verbose:
        print('\n--- ZKB (前走拡張データ: コメント・馬場適性コード) ---')

    zkb_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, 'Paci', 'ZKB*.txt')))
    if verbose:
        print(f'  {len(zkb_files)} files found')

    rows = []
    errors = 0

    for fpath in zkb_files:
        try:
            with open(fpath, 'rb') as f:
                for line_bytes in f:
                    line_bytes = line_bytes.rstrip(b'\r\n')
                    if len(line_bytes) < ZKB_MIN_LEN:
                        continue
                    try:
                        rows.append(parse_zkb_line(line_bytes))
                    except Exception:
                        errors += 1
        except Exception:
            errors += 1

    if verbose:
        print(f'  Parsed: {len(rows):,} rows ({errors} errors)')

    if not rows:
        if verbose:
            print('  No ZKB data found!')
        return None

    df = pd.DataFrame(rows)
    before = len(df)
    # ZKB: race_id(当該レース) + umaban + prev_date で一意
    if 'prev_date' in df.columns:
        df = df.drop_duplicates(subset=['race_id', 'umaban', 'prev_date'], keep='last')
    else:
        df = df.drop_duplicates(subset=['race_id', 'umaban'], keep='last')
    if verbose:
        print(f'  Deduplicated: {before:,} → {len(df):,}')

    return df


def save_csv(df, filename, key_cols=None, verbose=True):
    """DataFrameをCSVに保存。key_colsで出力カラムをフィルタ。"""
    if df is None or len(df) == 0:
        return

    if key_cols:
        avail = [c for c in key_cols if c in df.columns]
        df_out = df[avail].copy()
    else:
        df_out = df.copy()

    out_path = os.path.join(DATA_DIR, filename)
    df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
    if verbose:
        print(f'  Saved: {out_path} ({len(df_out):,} rows, {len(df_out.columns)} cols)')

        # Year coverage
        if 'race_id' in df_out.columns:
            df_out['_year'] = df_out['race_id'].str[:4]
            print(f'  Year coverage:')
            for yr, cnt in df_out.groupby('_year').size().items():
                print(f'    {yr}: {cnt:,} rows')


def main():
    parser = argparse.ArgumentParser(description='Parse JRDB SRB/ZED/ZKB data')
    parser.add_argument('--types', nargs='+', default=['srb', 'zed', 'zkb'],
                        choices=['srb', 'zed', 'zkb'],
                        help='Data types to parse (default: all)')
    args = parser.parse_args()

    start = datetime.now()
    print('=' * 60)
    print('  JRDB Extended Data Parser → CSV')
    print(f'  Types: {args.types}')
    print('=' * 60)

    # === SRB ===
    if 'srb' in args.types:
        df_srb = parse_srb()
        if df_srb is not None:
            srb_cols = [
                'race_id', 'harlon_count',
                *[f'harlon_{j}' for j in range(1, 19)],
                'corner1_pos', 'corner2_pos', 'corner3_pos', 'corner4_pos',
                'pace_up_pos',
                'tb_1corner', 'tb_2corner', 'tb_backstr', 'tb_3corner',
                'tb_4corner', 'tb_homestr',
                'race_comment',
            ]
            save_csv(df_srb, 'jrdb_sr.csv', srb_cols)

    # === ZED ===
    if 'zed' in args.types:
        df_zed = parse_zed()
        if df_zed is not None:
            zed_cols = [
                'race_id', 'umaban', 'horse_name', 'blood_num', 'yyyymmdd',
                'finish', 'abnormal', 'distance', 'surface_code', 'num_horses',
                'time_sec', 'weight_carry', 'odds_final', 'popularity',
                'idm', 'soten', 'baba_sa', 'ten_idx', 'agari_idx',
                'pace_idx', 'race_pace_idx', 'pace', 'deokure', 'ichi_dori',
                'furi', 'mae_furi', 'naka_furi', 'ato_furi',
                'race_idx', 'course_dori', 'josho_code',
                'batai_code', 'kehai_code', 'race_pace', 'uma_pace',
                'first_3f', 'last_3f', 'corner1', 'corner2', 'corner3', 'corner4',
                'horse_weight', 'weight_diff', 'race_style',
                'jockey_code', 'trainer_code',
                'race_name', 'grade', 'class_code',
            ]
            save_csv(df_zed, 'jrdb_ze.csv', zed_cols)

    # === ZKB ===
    if 'zkb' in args.types:
        df_zkb = parse_zkb()
        if df_zkb is not None:
            zkb_cols = [
                'race_id', 'umaban', 'blood_num', 'prev_date',
                'kishi_code_1', 'kishi_code_2', 'kishi_code_3',
                'kishi_code_4', 'kishi_code_5', 'kishi_code_6',
                'baba_code_1', 'baba_code_2', 'baba_code_3', 'baba_code_4',
                'baba_code_5', 'baba_code_6', 'baba_code_7', 'baba_code_8',
                'kyaku_left', 'kyaku_left_front', 'kyaku_right_front',
                'kyaku_left_rear', 'kyaku_right_rear',
                'padock_comment', 'kyaku_comment', 'baba_comment', 'race_comment',
                'turf_hoof', 'run_stage', 'anshin', 'heavy_apt', 'aisho', 'niwa',
            ]
            save_csv(df_zkb, 'jrdb_zk.csv', zkb_cols)

    elapsed = (datetime.now() - start).total_seconds()
    print(f'\n{"="*60}')
    print(f'  Done in {elapsed:.1f}s')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
