#!/usr/bin/env python3
"""JRDBの未パースデータ(BAC/SKB/CSA/KSA/OU)をパースしてCSV化する。

BAC: レース番組データ（レース単位: 距離・条件・出馬情報）
SKB: 成績拡張データ（馬単位: コメント・適性コード、ZKBと同一形式）
CSA: 調教師マスタ（週次更新: 成績・コメント）
KSA: 騎手マスタ（週次更新: 成績・コメント）
OU:  馬連オッズ（レース単位: 全組合せ馬連オッズ）

Usage:
    python tools/parse_jrdb_missing.py
"""
import os
import sys
import io
import glob
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

sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
from parse_jrdb import _hex_day, _safe_float, _safe_int, _field, _build_race_id


# =============================================================================
# BAC パーサー (レース番組データ、レコード長184バイト、レース単位)
# =============================================================================
def parse_bac_line(line_bytes):
    row = {}
    basho = _field(line_bytes, 1, 2)
    year = _field(line_bytes, 3, 2)
    kai = _field(line_bytes, 5, 1)
    nichi = _field(line_bytes, 6, 1)
    race_num = _field(line_bytes, 7, 2)
    row['race_id'] = _build_race_id(basho, year, kai, nichi, race_num)

    row['race_date'] = _field(line_bytes, 9, 8).strip()     # YYYYMMDD
    row['start_time'] = _field(line_bytes, 17, 4).strip()   # HHMM
    row['distance'] = _safe_int(_field(line_bytes, 21, 4))
    row['surface_code'] = _field(line_bytes, 25, 1).strip()  # 1:芝 2:ダ 3:障
    row['turn'] = _field(line_bytes, 26, 1).strip()          # 1:右 2:左 3:直
    row['inner_outer'] = _field(line_bytes, 27, 1).strip()
    row['baba_code'] = _field(line_bytes, 28, 2).strip()
    row['race_type'] = _field(line_bytes, 30, 2).strip()
    row['race_cond'] = _field(line_bytes, 32, 2).strip()
    row['race_symbol'] = _field(line_bytes, 34, 3).strip()
    row['weight_type'] = _field(line_bytes, 37, 1).strip()
    row['grade'] = _field(line_bytes, 38, 1).strip()
    row['race_name'] = _field(line_bytes, 39, 50).strip()
    row['num_horses'] = _safe_int(_field(line_bytes, 89, 2))
    row['race_name_short'] = _field(line_bytes, 91, 8).strip()
    row['race_cond_name'] = _field(line_bytes, 99, 30).strip()

    # 残りフィールド (賞金等)
    if len(line_bytes) >= 160:
        row['prize_1st'] = _safe_int(_field(line_bytes, 129, 5))
        row['prize_2nd'] = _safe_int(_field(line_bytes, 134, 5))
        row['prize_3rd'] = _safe_int(_field(line_bytes, 139, 5))
        row['prize_4th'] = _safe_int(_field(line_bytes, 144, 5))
        row['prize_5th'] = _safe_int(_field(line_bytes, 149, 5))
    return row


# =============================================================================
# SKB パーサー (成績拡張データ、304バイト、ZKBと同一形式)
# =============================================================================
def parse_skb_line(line_bytes):
    row = {}
    basho = _field(line_bytes, 1, 2)
    year = _field(line_bytes, 3, 2)
    kai = _field(line_bytes, 5, 1)
    nichi = _field(line_bytes, 6, 1)
    race_num = _field(line_bytes, 7, 2)
    row['race_id'] = _build_race_id(basho, year, kai, nichi, race_num)
    row['umaban'] = _safe_int(_field(line_bytes, 9, 2))
    row['blood_num'] = _field(line_bytes, 11, 8).strip()
    row['yyyymmdd'] = _field(line_bytes, 19, 8).strip()

    # 騎手コード 6x3
    for j in range(6):
        row[f'kishi_code_{j+1}'] = _safe_int(_field(line_bytes, 27 + j*3, 3))
    # 馬場コード 8x3
    for j in range(8):
        row[f'baba_code_{j+1}'] = _safe_int(_field(line_bytes, 45 + j*3, 3))
    # 脚質コード
    for j in range(5):
        row[f'kyaku_code_{j+1}'] = _safe_int(_field(line_bytes, 69 + j*9, 3))

    # コメント
    row['padock_comment'] = _field(line_bytes, 114, 40).strip()
    row['kyaku_comment'] = _field(line_bytes, 154, 40).strip()
    row['baba_comment'] = _field(line_bytes, 194, 40).strip()
    row['race_comment'] = _field(line_bytes, 234, 40).strip()

    # 適性コード
    if len(line_bytes) >= 291:
        row['turf_hoof'] = _safe_int(_field(line_bytes, 274, 3))
        row['run_stage'] = _safe_int(_field(line_bytes, 277, 3))
        row['anshin'] = _safe_int(_field(line_bytes, 280, 3))
        row['heavy_apt'] = _safe_int(_field(line_bytes, 283, 3))
        row['aisho'] = _safe_int(_field(line_bytes, 286, 3))
        row['niwa'] = _safe_int(_field(line_bytes, 289, 3))
    return row


# =============================================================================
# CSA パーサー (調教師マスタ、272バイト)
# =============================================================================
def parse_csa_line(line_bytes):
    row = {}
    row['trainer_code'] = _field(line_bytes, 1, 5).strip()
    row['del_flag'] = _safe_int(_field(line_bytes, 6, 1))
    row['del_date'] = _field(line_bytes, 7, 8).strip()
    row['trainer_name'] = _field(line_bytes, 15, 12).strip()
    row['trainer_kana'] = _field(line_bytes, 27, 30).strip()
    row['trainer_short'] = _field(line_bytes, 57, 6).strip()
    row['area_code'] = _safe_int(_field(line_bytes, 63, 1))
    row['area_name'] = _field(line_bytes, 64, 4).strip()
    row['birthday'] = _field(line_bytes, 68, 8).strip()
    row['first_year'] = _safe_int(_field(line_bytes, 76, 4))
    row['trainer_comment'] = _field(line_bytes, 80, 40).strip()
    row['comment_date'] = _field(line_bytes, 120, 8).strip()
    # 本年成績
    row['year_leading'] = _safe_int(_field(line_bytes, 128, 3))
    row['year_flat_1'] = _safe_int(_field(line_bytes, 131, 3))
    row['year_flat_2'] = _safe_int(_field(line_bytes, 134, 3))
    row['year_flat_3'] = _safe_int(_field(line_bytes, 137, 3))
    row['year_flat_out'] = _safe_int(_field(line_bytes, 140, 3))
    row['year_obs_1'] = _safe_int(_field(line_bytes, 143, 3))
    row['year_obs_2'] = _safe_int(_field(line_bytes, 146, 3))
    row['year_obs_3'] = _safe_int(_field(line_bytes, 149, 3))
    row['year_obs_out'] = _safe_int(_field(line_bytes, 152, 3))
    row['year_special_win'] = _safe_int(_field(line_bytes, 155, 3))
    row['year_graded_win'] = _safe_int(_field(line_bytes, 158, 3))
    # 昨年成績
    row['last_leading'] = _safe_int(_field(line_bytes, 161, 3))
    row['last_flat_1'] = _safe_int(_field(line_bytes, 164, 3))
    row['last_flat_2'] = _safe_int(_field(line_bytes, 167, 3))
    row['last_flat_3'] = _safe_int(_field(line_bytes, 170, 3))
    row['last_flat_out'] = _safe_int(_field(line_bytes, 173, 3))
    # 通算成績
    row['total_flat_1'] = _safe_int(_field(line_bytes, 194, 5))
    row['total_flat_2'] = _safe_int(_field(line_bytes, 199, 5))
    row['total_flat_3'] = _safe_int(_field(line_bytes, 204, 5))
    row['total_flat_out'] = _safe_int(_field(line_bytes, 209, 5))
    row['data_date'] = _field(line_bytes, 234, 8).strip()
    return row


# =============================================================================
# KSA パーサー (騎手マスタ、272バイト)
# =============================================================================
def parse_ksa_line(line_bytes):
    row = {}
    row['jockey_code'] = _field(line_bytes, 1, 5).strip()
    row['del_flag'] = _safe_int(_field(line_bytes, 6, 1))
    row['del_date'] = _field(line_bytes, 7, 8).strip()
    row['jockey_name'] = _field(line_bytes, 15, 12).strip()
    row['jockey_kana'] = _field(line_bytes, 27, 30).strip()
    row['jockey_short'] = _field(line_bytes, 57, 6).strip()
    row['area_code'] = _safe_int(_field(line_bytes, 63, 1))
    row['area_name'] = _field(line_bytes, 64, 4).strip()
    row['birthday'] = _field(line_bytes, 68, 8).strip()
    row['first_year'] = _safe_int(_field(line_bytes, 76, 4))
    row['apprentice'] = _safe_int(_field(line_bytes, 80, 1))
    row['stable_code'] = _field(line_bytes, 81, 5).strip()
    row['jockey_comment'] = _field(line_bytes, 86, 40).strip()
    row['comment_date'] = _field(line_bytes, 126, 8).strip()
    # 本年成績
    row['year_leading'] = _safe_int(_field(line_bytes, 134, 3))
    row['year_flat_1'] = _safe_int(_field(line_bytes, 137, 3))
    row['year_flat_2'] = _safe_int(_field(line_bytes, 140, 3))
    row['year_flat_3'] = _safe_int(_field(line_bytes, 143, 3))
    row['year_flat_out'] = _safe_int(_field(line_bytes, 146, 3))
    row['year_obs_1'] = _safe_int(_field(line_bytes, 149, 3))
    row['year_obs_2'] = _safe_int(_field(line_bytes, 152, 3))
    row['year_obs_3'] = _safe_int(_field(line_bytes, 155, 3))
    row['year_obs_out'] = _safe_int(_field(line_bytes, 158, 3))
    row['year_special_win'] = _safe_int(_field(line_bytes, 161, 3))
    row['year_graded_win'] = _safe_int(_field(line_bytes, 164, 3))
    # 昨年成績
    row['last_leading'] = _safe_int(_field(line_bytes, 167, 3))
    row['last_flat_1'] = _safe_int(_field(line_bytes, 170, 3))
    row['last_flat_2'] = _safe_int(_field(line_bytes, 173, 3))
    row['last_flat_3'] = _safe_int(_field(line_bytes, 176, 3))
    row['last_flat_out'] = _safe_int(_field(line_bytes, 179, 3))
    # 通算成績
    row['total_flat_1'] = _safe_int(_field(line_bytes, 200, 5))
    row['total_flat_2'] = _safe_int(_field(line_bytes, 205, 5))
    row['total_flat_3'] = _safe_int(_field(line_bytes, 210, 5))
    row['total_flat_out'] = _safe_int(_field(line_bytes, 215, 5))
    row['data_date'] = _field(line_bytes, 240, 8).strip()
    return row


# =============================================================================
# OU パーサー (馬連オッズ、レコード長1856バイト、レース単位)
# 馬連153組 x 12B(オッズ6B + 人気順6B) = 1836 + header
# =============================================================================
def parse_ou_line(line_bytes):
    row = {}
    basho = _field(line_bytes, 1, 2)
    year = _field(line_bytes, 3, 2)
    kai = _field(line_bytes, 5, 1)
    nichi = _field(line_bytes, 6, 1)
    race_num = _field(line_bytes, 7, 2)
    row['race_id'] = _build_race_id(basho, year, kai, nichi, race_num)
    row['num_horses'] = _safe_int(_field(line_bytes, 9, 2))

    # 馬連オッズ (up to 153組)
    umaren_odds = []
    for i in range(153):
        start = 11 + i * 12
        if start + 6 > len(line_bytes):
            break
        val = _safe_float(_field(line_bytes, start, 6))
        if val and val > 0 and val < 99999:
            umaren_odds.append(val)

    row['umaren_count'] = len(umaren_odds)
    row['umaren_min'] = min(umaren_odds) if umaren_odds else None
    row['umaren_max'] = max(umaren_odds) if umaren_odds else None
    if umaren_odds:
        umaren_odds.sort()
        row['umaren_median'] = umaren_odds[len(umaren_odds) // 2]
        row['umaren_p10'] = umaren_odds[len(umaren_odds) // 10] if len(umaren_odds) >= 10 else umaren_odds[0]
    else:
        row['umaren_median'] = None
        row['umaren_p10'] = None
    return row


# =============================================================================
# メイン処理
# =============================================================================
def _parse_files(file_list, parse_func, min_len, label):
    rows = []
    errors = 0
    for fpath in file_list:
        try:
            with open(fpath, 'rb') as f:
                for line_bytes in f:
                    line_bytes = line_bytes.rstrip(b'\r\n')
                    if len(line_bytes) < min_len:
                        continue
                    try:
                        rows.append(parse_func(line_bytes))
                    except Exception:
                        errors += 1
        except Exception:
            errors += 1
    print(f'  Parsed: {len(rows):,} rows ({errors} errors)')
    return rows


def main():
    start = datetime.now()
    print('=' * 60)
    print('  JRDB Missing Data Parser → CSV')
    print('=' * 60)

    # === BAC ===
    print('\n--- BAC (レース番組データ) ---')
    bac_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, 'Bac', 'BAC*.txt')))
    bac_files += sorted(glob.glob(os.path.join(EXTRACT_DIR, 'Paci', 'BAC*.txt')))
    print(f'  {len(bac_files)} files found')
    bac_rows = _parse_files(bac_files, parse_bac_line, 90, 'BAC')
    if bac_rows:
        df = pd.DataFrame(bac_rows).drop_duplicates(subset=['race_id'], keep='last')
        print(f'  Deduplicated: → {len(df):,}')
        out = os.path.join(DATA_DIR, 'jrdb_bac.csv')
        df.to_csv(out, index=False, encoding='utf-8-sig')
        print(f'  Saved: {out} ({len(df):,} rows, {len(df.columns)} cols)')
        df['_y'] = df['race_id'].str[:4]
        for y, c in df.groupby('_y').size().items():
            print(f'    {y}: {c:,}')

    # === SKB ===
    print('\n--- SKB (成績拡張データ) ---')
    skb_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, 'Skb', 'SKB*.txt')))
    print(f'  {len(skb_files)} files found')
    skb_rows = _parse_files(skb_files, parse_skb_line, 270, 'SKB')
    if skb_rows:
        df = pd.DataFrame(skb_rows).drop_duplicates(subset=['race_id', 'umaban'], keep='last')
        print(f'  Deduplicated: → {len(df):,}')
        out = os.path.join(DATA_DIR, 'jrdb_skb.csv')
        df.to_csv(out, index=False, encoding='utf-8-sig')
        print(f'  Saved: {out} ({len(df):,} rows, {len(df.columns)} cols)')
        df['_y'] = df['race_id'].str[:4]
        for y, c in df.groupby('_y').size().items():
            print(f'    {y}: {c:,}')

    # === CSA ===
    print('\n--- CSA (調教師マスタ) ---')
    csa_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, 'Cs', 'CSA*.txt')))
    print(f'  {len(csa_files)} files found')
    csa_rows = _parse_files(csa_files, parse_csa_line, 230, 'CSA')
    if csa_rows:
        df = pd.DataFrame(csa_rows).drop_duplicates(subset=['trainer_code'], keep='last')
        print(f'  Deduplicated: → {len(df):,}')
        out = os.path.join(DATA_DIR, 'jrdb_csa.csv')
        df.to_csv(out, index=False, encoding='utf-8-sig')
        print(f'  Saved: {out} ({len(df):,} rows, {len(df.columns)} cols)')

    # === KSA ===
    print('\n--- KSA (騎手マスタ) ---')
    ksa_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, 'Ks', 'KSA*.txt')))
    print(f'  {len(ksa_files)} files found')
    ksa_rows = _parse_files(ksa_files, parse_ksa_line, 230, 'KSA')
    if ksa_rows:
        df = pd.DataFrame(ksa_rows).drop_duplicates(subset=['jockey_code'], keep='last')
        print(f'  Deduplicated: → {len(df):,}')
        out = os.path.join(DATA_DIR, 'jrdb_ksa.csv')
        df.to_csv(out, index=False, encoding='utf-8-sig')
        print(f'  Saved: {out} ({len(df):,} rows, {len(df.columns)} cols)')

    # === OU ===
    print('\n--- OU (馬連オッズ) ---')
    ou_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, 'Paci', 'OU*.txt')))
    print(f'  {len(ou_files)} files found')
    ou_rows = _parse_files(ou_files, parse_ou_line, 100, 'OU')
    if ou_rows:
        df = pd.DataFrame(ou_rows).drop_duplicates(subset=['race_id'], keep='last')
        print(f'  Deduplicated: → {len(df):,}')
        out = os.path.join(DATA_DIR, 'jrdb_ou.csv')
        df.to_csv(out, index=False, encoding='utf-8-sig')
        print(f'  Saved: {out} ({len(df):,} rows, {len(df.columns)} cols)')
        df['_y'] = df['race_id'].str[:4]
        for y, c in df.groupby('_y').size().items():
            print(f'    {y}: {c:,}')

    elapsed = (datetime.now() - start).total_seconds()
    print(f'\n{"="*60}')
    print(f'  Done in {elapsed:.1f}s')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
