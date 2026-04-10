#!/usr/bin/env python3
"""JRDBの固定長テキスト(Sed/Tyb)をパースしてCSV化する。

仕様:
  Sed: http://www.jrdb.com/program/Sed/sed_doc.txt (レコード長376B)
  Tyb: http://www.jrdb.com/program/Tyb/tyb_doc.txt (レコード長128B)

Usage:
    python tools/parse_jrdb.py
"""
import os
import sys
import io
import glob
import pandas as pd
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_DIR = os.path.join(BASE_DIR, 'data', 'jrdb', 'extracted')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# JRDB「日」フィールド: 16進数1桁 → 10進数
def _hex_day(c):
    """1文字の16進数 → int (1-18)"""
    try:
        return int(c, 16)
    except (ValueError, TypeError):
        return 0


def _safe_float(s):
    """文字列を安全にfloat変換。空白/非数値は None。"""
    s = s.strip()
    if not s or s == '-':
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_int(s):
    """文字列を安全にint変換。"""
    s = s.strip()
    if not s or s == '-':
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _build_race_id(basho, year, kai, nichi, race_num):
    """JRDBレースキー → netkeiba形式 race_id (12桁)
    basho(2桁) + 20XX(4桁) + kai(2桁) + nichi(2桁) + race_num(2桁)
    """
    yr2 = year.strip()
    yr4 = f'20{yr2}' if len(yr2) == 2 else yr2
    return f'{yr4}{basho.strip()}{int(kai):02d}{_hex_day(nichi):02d}{int(race_num):02d}'


# =============================================================================
# Sed パーサー (成績データ、レコード長376バイト)
# =============================================================================
# カラム定義: (name, start_pos_1based, length)
# 相対位置は1始まり → Pythonスライスは start-1 : start-1+length
SED_COLUMNS = [
    # レースキー
    ('basho_code',       1,  2),
    ('year',             3,  2),
    ('kai',              5,  1),
    ('nichi',            6,  1),
    ('race_num',         7,  2),
    # 馬番
    ('umaban',           9,  2),
    # 競走成績キー
    ('blood_num',       11,  8),
    ('yyyymmdd',        19,  8),
    # 馬名
    ('horse_name',      27, 36),
    # レース条件
    ('distance',        63,  4),
    ('surface_code',    67,  1),   # 1:芝, 2:ダート, 3:障害
    ('turn',            68,  1),   # 1:右, 2:左, 3:直
    ('inner_outer',     69,  1),
    ('baba_code',       70,  2),
    ('race_type',       72,  2),
    ('race_cond',       74,  2),
    ('race_symbol',     76,  3),
    ('weight_type',     79,  1),
    ('grade',           80,  1),
    ('race_name',       81, 50),
    ('num_horses',     131,  2),
    ('race_name_short',133,  8),
    # 馬成績
    ('finish',         141,  2),
    ('abnormal',       143,  1),
    ('time_raw',       144,  4),   # 分秒(0.1秒)
    ('weight_carry',   148,  3),   # 0.1kg単位
    ('jockey_name',    151, 12),
    ('trainer_name',   163, 12),
    ('odds_final',     175,  6),
    ('popularity',     181,  2),
    # JRDBデータ
    ('idm',            183,  3),
    ('soten',          186,  3),
    ('baba_sa',        189,  3),
    ('pace',           192,  3),
    ('deokure',        195,  3),
    ('ichi_dori',      198,  3),
    ('furi',           201,  3),
    ('mae_furi',       204,  3),
    ('naka_furi',      207,  3),
    ('ato_furi',       210,  3),
    ('race_idx',       213,  3),
    ('course_dori',    216,  1),   # 1:最内〜5:大外
    ('josho_code',     217,  1),   # 1:AA〜5:?
    ('class_code',     218,  2),
    ('batai_code',     220,  1),
    ('kehai_code',     221,  1),
    ('race_pace',      222,  1),   # H/M/S
    ('uma_pace',       223,  1),   # H/M/S
    ('ten_idx',        224,  5),
    ('agari_idx',      229,  5),
    ('pace_idx',       234,  5),
    ('race_pace_idx',  239,  5),
    ('winner_name',    244, 12),
    ('winner_diff',    256,  3),   # 0.1秒
    ('first_3f',       259,  3),   # 0.1秒
    ('last_3f',        262,  3),   # 0.1秒
    ('remarks',        265, 24),
    # 第2版追加
    ('fukusho_odds',   291,  6),
    ('odds_10am',      297,  6),
    ('fukusho_10am',   303,  6),
    ('corner1',        309,  2),
    ('corner2',        311,  2),
    ('corner3',        313,  2),
    ('corner4',        315,  2),
    ('first_3f_diff',  317,  3),   # 0.1秒
    ('last_3f_diff',   320,  3),   # 0.1秒
    ('jockey_code',    323,  5),
    ('trainer_code',   328,  5),
    # 第3版追加
    ('horse_weight',   333,  3),
    ('weight_diff',    336,  3),
    ('weather_code',   339,  1),
    ('course_type',    340,  1),
    ('race_style',     341,  1),
    # 第4版追加
    ('tansho_payout',  342,  7),
    ('fukusho_payout', 349,  7),
    ('prize_main',     356,  5),   # 万円
    ('prize_shutoku',  361,  5),   # 万円
    ('race_pace_flow', 366,  2),
    ('uma_pace_flow',  368,  2),
    ('corner4_course', 370,  1),
    ('start_time',     371,  4),   # HHMM
]

# 数値変換するカラム
SED_INT_COLS = [
    'umaban', 'distance', 'num_horses', 'finish', 'abnormal',
    'popularity', 'corner1', 'corner2', 'corner3', 'corner4',
    'class_code', 'josho_code', 'batai_code', 'kehai_code',
    'course_dori', 'corner4_course', 'grade',
]
SED_FLOAT_COLS = [
    'idm', 'soten', 'baba_sa', 'pace', 'deokure', 'ichi_dori',
    'furi', 'mae_furi', 'naka_furi', 'ato_furi', 'race_idx',
    'ten_idx', 'agari_idx', 'pace_idx', 'race_pace_idx',
    'odds_final', 'fukusho_odds', 'odds_10am', 'fukusho_10am',
    'winner_diff', 'first_3f', 'last_3f',
    'first_3f_diff', 'last_3f_diff',
    'weight_carry', 'horse_weight',
    'tansho_payout', 'fukusho_payout', 'prize_main', 'prize_shutoku',
    'race_pace_flow', 'uma_pace_flow',
]


def _field(line_bytes, start, length):
    """バイト列から固定長フィールドを抽出してcp932デコード。"""
    s = start - 1  # 1-based → 0-based
    return line_bytes[s:s+length].decode('cp932', errors='replace')


def parse_sed_line(line_bytes):
    """Sedの1行(bytes)をパースして辞書を返す。"""
    row = {}
    for name, start, length in SED_COLUMNS:
        row[name] = _field(line_bytes, start, length)

    # race_id構築
    row['race_id'] = _build_race_id(
        row['basho_code'], row['year'], row['kai'], row['nichi'], row['race_num']
    )

    # 文字列トリム
    for col in ['horse_name', 'race_name', 'race_name_short', 'jockey_name',
                'trainer_name', 'winner_name', 'remarks', 'race_pace', 'uma_pace',
                'blood_num', 'yyyymmdd', 'jockey_code', 'trainer_code',
                'race_style', 'weather_code', 'course_type', 'start_time',
                'race_cond', 'race_symbol', 'surface_code', 'turn', 'inner_outer',
                'baba_code', 'weight_type']:
        if col in row:
            row[col] = row[col].strip()

    # 数値変換
    for col in SED_INT_COLS:
        row[col] = _safe_int(row[col])
    for col in SED_FLOAT_COLS:
        row[col] = _safe_float(row[col])

    # タイム変換 (分秒 → 秒)
    t = row.get('time_raw', '').strip()
    if t and len(t) == 4 and t.isdigit():
        mins = int(t[0])
        secs = int(t[1:]) / 10.0
        row['time_sec'] = mins * 60 + secs
    else:
        row['time_sec'] = None
    del row['time_raw']

    # 斤量 0.1kg→kg
    if row.get('weight_carry') is not None:
        row['weight_carry'] = row['weight_carry'] / 10.0

    # 0.1秒単位 → 秒
    for col in ['first_3f', 'last_3f', 'winner_diff', 'first_3f_diff', 'last_3f_diff']:
        if row.get(col) is not None:
            row[col] = row[col] / 10.0

    # 馬体重増減
    wd = row.get('weight_diff', '')
    if isinstance(wd, str):
        wd = wd.strip()
        if wd:
            try:
                row['weight_diff'] = int(wd)
            except ValueError:
                row['weight_diff'] = None
        else:
            row['weight_diff'] = None

    # 不要キー削除
    for k in ['basho_code', 'year', 'kai', 'nichi', 'race_num']:
        del row[k]

    return row


# =============================================================================
# Tyb パーサー (直前データ、レコード長128バイト)
# =============================================================================
TYB_COLUMNS = [
    ('basho_code',       1,  2),
    ('year',             3,  2),
    ('kai',              5,  1),
    ('nichi',            6,  1),
    ('race_num',         7,  2),
    ('umaban',           9,  2),
    ('idm',             11,  5),
    ('jockey_idx',      16,  5),
    ('info_idx',        21,  5),
    ('odds_idx',        26,  5),
    ('padock_idx',      31,  5),
    ('reserve1',        36,  5),
    ('sogo_idx',        41,  5),
    ('bagu_change',     46,  1),
    ('ashimoto',        47,  1),   # 0:平行線,1:良化,2:疑問,3:悪化
    ('cancel_flag',     48,  1),
    ('jockey_code',     49,  5),
    ('jockey_name',     54, 12),
    ('weight_carry',    66,  3),   # 0.1kg
    ('minarai',         69,  1),
    ('baba_code',       70,  2),
    ('weather_code',    72,  1),
    ('tansho_odds',     73,  6),
    ('fukusho_odds',    79,  6),
    ('odds_time',       85,  4),   # HHMM
    ('horse_weight',    89,  3),
    ('weight_diff',     92,  3),
    ('odds_mark',       95,  1),
    ('padock_mark',     96,  1),
    ('sogo_mark',       97,  1),
    ('batai_code',      98,  1),
    ('kehai_code',      99,  1),
    ('start_time',     100,  4),   # HHMM
]

TYB_FLOAT_COLS = [
    'idm', 'jockey_idx', 'info_idx', 'odds_idx', 'padock_idx',
    'reserve1', 'sogo_idx', 'tansho_odds', 'fukusho_odds', 'weight_carry',
]
TYB_INT_COLS = [
    'umaban', 'bagu_change', 'ashimoto', 'cancel_flag', 'horse_weight',
]


def parse_tyb_line(line_bytes):
    """Tybの1行(bytes)をパースして辞書を返す。"""
    row = {}
    for name, start, length in TYB_COLUMNS:
        row[name] = _field(line_bytes, start, length)

    row['race_id'] = _build_race_id(
        row['basho_code'], row['year'], row['kai'], row['nichi'], row['race_num']
    )

    # 文字列トリム
    for col in ['jockey_name', 'jockey_code', 'odds_mark', 'padock_mark',
                'sogo_mark', 'batai_code', 'kehai_code', 'odds_time',
                'baba_code', 'weather_code', 'minarai', 'start_time']:
        if col in row:
            row[col] = row[col].strip()

    # 数値変換
    for col in TYB_INT_COLS:
        row[col] = _safe_int(row[col])
    for col in TYB_FLOAT_COLS:
        row[col] = _safe_float(row[col])

    # 斤量 0.1kg→kg
    if row.get('weight_carry') is not None:
        row['weight_carry'] = row['weight_carry'] / 10.0

    # 馬体重増減
    wd = row.get('weight_diff', '')
    if isinstance(wd, str):
        wd = wd.strip()
        if wd:
            try:
                row['weight_diff'] = int(wd)
            except ValueError:
                row['weight_diff'] = None
        else:
            row['weight_diff'] = None

    for k in ['basho_code', 'year', 'kai', 'nichi', 'race_num', 'reserve1']:
        del row[k]

    return row


# =============================================================================
# KYI パーサー (競走馬データ=Paci前日データ、レコード長1024バイト)
# =============================================================================
KYI_COLUMNS = [
    ('basho_code',       1,  2),
    ('year',             3,  2),
    ('kai',              5,  1),
    ('nichi',            6,  1),
    ('race_num',         7,  2),
    ('umaban',           9,  2),
    ('blood_num',       11,  8),
    ('horse_name',      19, 36),
    ('idm',             55,  5),
    ('jockey_idx',      60,  5),
    ('info_idx',        65,  5),
    ('sogo_idx',        85,  5),
    ('kyakushitsu',     90,  1),   # 脚質 1:逃 2:先 3:差 4:追
    ('dist_aptitude',   91,  1),   # 距離適性
    ('josho',           92,  1),   # 上昇度
    ('rotation',        93,  3),   # ローテーション
    ('base_odds',       96,  5),
    ('base_pop',       101,  2),
    ('base_fuku_odds', 103,  5),
    ('base_fuku_pop',  108,  2),
    ('ninki_idx',      140,  5),   # 人気指数
    ('train_idx',      145,  5),   # 調教指数
    ('stable_idx',     150,  5),   # 厩舎指数
    ('train_arrow',    155,  1),   # 調教矢印コード
    ('stable_eval',    156,  1),   # 厩舎評価コード
    ('jockey_exp_wr',  157,  4),   # 騎手期待連対率
    ('gekiso_idx',     161,  3),   # 激走指数
    ('hizume_code',    164,  2),   # 蹄コード
    ('heavy_apt',      166,  1),   # 重適正コード
    ('class_code',     167,  2),   # クラスコード
    ('blinker',        171,  1),
    ('jockey_name',    172, 12),
    ('weight_carry',   184,  3),   # 0.1kg
    ('minarai',        187,  1),
    ('trainer_name',   188, 12),
    ('trainer_area',   200,  4),
    ('wakuban',        324,  1),
    ('sogo_mark',      327,  1),   # 総合印
    ('idm_mark',       328,  1),
    ('info_mark',      329,  1),
    ('jockey_mark',    330,  1),
    ('stable_mark',    331,  1),
    ('train_mark',     332,  1),
    ('gekiso_mark',    333,  1),
    ('turf_apt',       334,  1),   # 芝適性 1:◎ 2:○ 3:△
    ('dirt_apt',       335,  1),   # ダ適性
    ('jockey_code',    336,  5),
    ('trainer_code',   341,  5),
    ('prize_total',    347,  6),   # 獲得賞金(万円)
    ('prize_shutoku',  353,  5),   # 収得賞金(万円)
    ('cond_class',     358,  1),   # 条件クラス
    ('ten_idx_pred',   359,  5),   # テン指数予想
    ('pace_idx_pred',  364,  5),   # ペース指数予想
    ('agari_idx_pred', 369,  5),   # 上がり指数予想
    ('ichi_idx_pred',  374,  5),   # 位置指数予想
    ('pace_pred',      379,  1),   # ペース予想 H/M/S
    ('dochu_rank',     380,  2),   # 道中順位
    ('dochu_diff',     382,  2),   # 道中差
    ('dochu_io',       384,  1),   # 道中内外
    ('last3f_rank',    385,  2),   # 後3F順位
    ('last3f_diff',    387,  2),   # 後3F差
    ('last3f_io',      389,  1),   # 後3F内外
    ('goal_rank',      390,  2),   # ゴール順位
    ('goal_diff',      392,  2),   # ゴール差
    ('goal_io',        394,  1),   # ゴール内外
    ('tenkai_symbol',  395,  1),   # 展開記号
    ('dist_aptitude2', 396,  1),
    ('confirm_weight', 397,  3),   # 枠確定馬体重
    ('confirm_wdiff',  400,  3),   # 枠確定馬体重増減
    ('cancel_flag',    403,  1),
    ('sex_code',       404,  1),   # 1:牡 2:牝 3:セン
    ('owner_name',     405, 40),
    ('gekiso_rank',    449,  2),
    ('ls_idx_rank',    451,  2),   # LS指数順位
    ('ten_idx_rank',   453,  2),
    ('pace_idx_rank',  455,  2),
    ('agari_idx_rank', 457,  2),
    ('ichi_idx_rank',  459,  2),
    ('jockey_exp_win', 461,  4),   # 騎手期待単勝率
    ('jockey_exp_3rd', 465,  4),   # 騎手期待3着内率
    ('transport',      469,  1),   # 輸送区分
    ('manken_idx',     535,  3),   # 万券指数
    ('manken_mark',    538,  1),
    ('kokyuu_flag',    539,  1),   # 降級フラグ
    ('gekiso_type',    540,  2),   # 激走タイプ
    ('rest_reason',    542,  2),   # 休養理由
    ('entry_race_num', 560,  2),   # 入厩何走目 (例: 2=入厩後2走目)
    ('entry_date',     562,  8),   # 入厩年月日 YYYYMMDD
    ('entry_days_ago', 570,  3),   # 入厩何日前 (レース日から遡った日数)
    ('ranch_name',     573, 50),   # 放牧先名称
    ('gaisha_rank',    623,  1),   # 放牧先ランク A-E
    ('stable_rank',    624,  1),   # 厩舎ランク 1-9
]

KYI_FLOAT_COLS = [
    'idm', 'jockey_idx', 'info_idx', 'sogo_idx',
    'base_odds', 'base_fuku_odds', 'ninki_idx',
    'train_idx', 'stable_idx', 'jockey_exp_wr',
    'ten_idx_pred', 'pace_idx_pred', 'agari_idx_pred', 'ichi_idx_pred',
    'jockey_exp_win', 'jockey_exp_3rd', 'weight_carry',
]
KYI_INT_COLS = [
    'umaban', 'kyakushitsu', 'josho', 'rotation',
    'base_pop', 'base_fuku_pop', 'gekiso_idx',
    'class_code', 'wakuban', 'dochu_rank', 'dochu_diff', 'dochu_io',
    'last3f_rank', 'last3f_diff', 'last3f_io',
    'goal_rank', 'goal_diff', 'goal_io',
    'confirm_weight', 'cancel_flag', 'sex_code',
    'gekiso_rank', 'ls_idx_rank', 'ten_idx_rank',
    'pace_idx_rank', 'agari_idx_rank', 'ichi_idx_rank',
    'manken_idx', 'kokyuu_flag', 'prize_total', 'prize_shutoku',
    'entry_race_num', 'entry_days_ago',
]


def parse_kyi_line(line_bytes):
    """KYIの1行(bytes)をパースして辞書を返す。"""
    row = {}
    for name, start, length in KYI_COLUMNS:
        row[name] = _field(line_bytes, start, length)

    row['race_id'] = _build_race_id(
        row['basho_code'], row['year'], row['kai'], row['nichi'], row['race_num']
    )

    # 文字列トリム
    for col in ['horse_name', 'jockey_name', 'trainer_name', 'trainer_area',
                'blood_num', 'owner_name', 'pace_pred', 'tenkai_symbol',
                'jockey_code', 'trainer_code', 'turf_apt', 'dirt_apt',
                'blinker', 'minarai', 'hizume_code', 'heavy_apt',
                'dist_aptitude', 'dist_aptitude2', 'train_arrow', 'stable_eval',
                'sogo_mark', 'idm_mark', 'info_mark', 'jockey_mark',
                'stable_mark', 'train_mark', 'gekiso_mark', 'cond_class',
                'transport', 'manken_mark', 'gekiso_type', 'rest_reason',
                'entry_date', 'ranch_name', 'gaisha_rank', 'stable_rank']:
        if col in row:
            row[col] = row[col].strip()

    for col in KYI_INT_COLS:
        row[col] = _safe_int(row[col])
    for col in KYI_FLOAT_COLS:
        row[col] = _safe_float(row[col])

    # 斤量 0.1kg→kg
    if row.get('weight_carry') is not None:
        row['weight_carry'] = row['weight_carry'] / 10.0

    # 枠確定馬体重増減
    wd = row.get('confirm_wdiff', '')
    if isinstance(wd, str):
        wd = wd.strip()
        if wd:
            try:
                row['confirm_wdiff'] = int(wd)
            except ValueError:
                row['confirm_wdiff'] = None
        else:
            row['confirm_wdiff'] = None

    for k in ['basho_code', 'year', 'kai', 'nichi', 'race_num']:
        del row[k]

    return row


# =============================================================================
# メイン処理
# =============================================================================
def main():
    start = datetime.now()
    print('=' * 60)
    print('  JRDB Fixed-Width Parser → CSV')
    print('=' * 60)

    # === Sed ===
    print('\n--- Sed (成績データ) ---')
    # SED files: SED*.txt or RCA*.csv (both are fixed-width despite .csv extension)
    sed_files_txt = glob.glob(os.path.join(EXTRACT_DIR, 'Sed', 'SED*.txt'))
    sed_files_csv = glob.glob(os.path.join(EXTRACT_DIR, 'Sed', 'RCA*.csv'))
    # RCA files are actually SED format (renamed in archives)
    # Check which format we have
    if sed_files_csv:
        sample = sed_files_csv[0]
        with open(sample, 'rb') as f:
            first = f.readline()
        if b',' in first[:50]:
            print(f'  RCA files appear to be CSV format (comma-separated)')
            rca_is_csv = True
        else:
            print(f'  RCA files are fixed-width (SED format)')
            rca_is_csv = False
    else:
        rca_is_csv = False

    sed_rows = []
    sed_errors = 0

    def _process_sed_files(file_list):
        nonlocal sed_errors
        for fpath in sorted(file_list):
            try:
                with open(fpath, 'rb') as f:
                    for line_bytes in f:
                        line_bytes = line_bytes.rstrip(b'\r\n')
                        if len(line_bytes) < 340:
                            continue
                        try:
                            sed_rows.append(parse_sed_line(line_bytes))
                        except Exception:
                            sed_errors += 1
            except Exception:
                sed_errors += 1

    _process_sed_files(sed_files_txt)
    if sed_files_csv and not rca_is_csv:
        _process_sed_files(sed_files_csv)

    print(f'  Parsed: {len(sed_rows):,} rows ({sed_errors} errors)')

    if sed_rows:
        df_sed = pd.DataFrame(sed_rows)
        # Deduplicate
        before = len(df_sed)
        df_sed = df_sed.drop_duplicates(subset=['race_id', 'umaban'], keep='last')
        print(f'  Deduplicated: {before:,} → {len(df_sed):,}')

        # Select key columns for output
        sed_key_cols = [
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
        avail = [c for c in sed_key_cols if c in df_sed.columns]
        df_out = df_sed[avail].copy()

        out_path = os.path.join(DATA_DIR, 'jrdb_sed.csv')
        df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f'  Saved: {out_path} ({len(df_out):,} rows, {len(avail)} cols)')

        # Year coverage
        df_sed['_year'] = df_sed['race_id'].str[:4]
        print(f'  Year coverage:')
        for yr, cnt in df_sed.groupby('_year').size().items():
            print(f'    {yr}: {cnt:,} rows')
    else:
        print('  No Sed data found!')

    # === Tyb ===
    print('\n--- Tyb (直前データ) ---')
    tyb_rows = []
    tyb_errors = 0
    tyb_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, 'Tyb', 'TYB*.txt')))
    print(f'  {len(tyb_files)} files found')

    for fpath in tyb_files:
        try:
            with open(fpath, 'rb') as f:
                for line_bytes in f:
                    line_bytes = line_bytes.rstrip(b'\r\n')
                    if len(line_bytes) < 95:
                        continue
                    try:
                        tyb_rows.append(parse_tyb_line(line_bytes))
                    except Exception:
                        tyb_errors += 1
        except Exception:
            tyb_errors += 1

    print(f'  Parsed: {len(tyb_rows):,} rows ({tyb_errors} errors)')

    if tyb_rows:
        df_tyb = pd.DataFrame(tyb_rows)
        before = len(df_tyb)
        df_tyb = df_tyb.drop_duplicates(subset=['race_id', 'umaban'], keep='last')
        print(f'  Deduplicated: {before:,} → {len(df_tyb):,}')

        tyb_key_cols = [
            'race_id', 'umaban',
            'idm', 'jockey_idx', 'info_idx', 'odds_idx', 'padock_idx', 'sogo_idx',
            'bagu_change', 'ashimoto', 'cancel_flag',
            'tansho_odds', 'fukusho_odds',
            'horse_weight', 'weight_diff', 'weight_carry',
            'odds_mark', 'padock_mark', 'sogo_mark',
            'batai_code', 'kehai_code',
            'jockey_code', 'jockey_name',
            'baba_code', 'weather_code', 'start_time',
        ]
        avail = [c for c in tyb_key_cols if c in df_tyb.columns]
        df_out = df_tyb[avail].copy()

        out_path = os.path.join(DATA_DIR, 'jrdb_tyb.csv')
        df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f'  Saved: {out_path} ({len(df_out):,} rows, {len(avail)} cols)')

        df_tyb['_year'] = df_tyb['race_id'].str[:4]
        print(f'  Year coverage:')
        for yr, cnt in df_tyb.groupby('_year').size().items():
            print(f'    {yr}: {cnt:,} rows')
    else:
        print('  No Tyb data found!')

    # === KYI (Paci前日データ) ===
    print('\n--- KYI (Paci前日データ) ---')
    kyi_rows = []
    kyi_errors = 0
    # KYI files are inside the Paci extracted directory
    kyi_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, 'Paci', 'KYI*.txt')))
    print(f'  {len(kyi_files)} files found')

    for fpath in kyi_files:
        try:
            with open(fpath, 'rb') as f:
                for line_bytes in f:
                    line_bytes = line_bytes.rstrip(b'\r\n')
                    if len(line_bytes) < 400:
                        continue
                    try:
                        kyi_rows.append(parse_kyi_line(line_bytes))
                    except Exception:
                        kyi_errors += 1
        except Exception:
            kyi_errors += 1

    print(f'  Parsed: {len(kyi_rows):,} rows ({kyi_errors} errors)')

    if kyi_rows:
        df_kyi = pd.DataFrame(kyi_rows)
        before = len(df_kyi)
        df_kyi = df_kyi.drop_duplicates(subset=['race_id', 'umaban'], keep='last')
        print(f'  Deduplicated: {before:,} → {len(df_kyi):,}')

        kyi_key_cols = [
            'race_id', 'umaban', 'horse_name', 'blood_num',
            'idm', 'jockey_idx', 'info_idx', 'sogo_idx',
            'kyakushitsu', 'dist_aptitude', 'josho', 'rotation',
            'base_odds', 'base_pop', 'ninki_idx',
            'train_idx', 'stable_idx', 'train_arrow', 'stable_eval',
            'jockey_exp_wr', 'gekiso_idx',
            'hizume_code', 'heavy_apt', 'class_code',
            'ten_idx_pred', 'pace_idx_pred', 'agari_idx_pred', 'ichi_idx_pred',
            'pace_pred', 'dochu_rank', 'goal_rank', 'goal_diff',
            'sogo_mark', 'idm_mark', 'info_mark', 'jockey_mark',
            'stable_mark', 'train_mark', 'gekiso_mark',
            'turf_apt', 'dirt_apt',
            'jockey_code', 'trainer_code',
            'jockey_name', 'trainer_name', 'weight_carry',
            'sex_code', 'wakuban', 'cancel_flag',
            'ls_idx_rank', 'gekiso_rank',
            'jockey_exp_win', 'jockey_exp_3rd',
            'manken_idx', 'kokyuu_flag', 'gekiso_type',
            'gaisha_rank', 'stable_rank',
            'confirm_weight', 'confirm_wdiff',
            'prize_total', 'prize_shutoku', 'cond_class',
        ]
        avail = [c for c in kyi_key_cols if c in df_kyi.columns]
        df_out = df_kyi[avail].copy()

        out_path = os.path.join(DATA_DIR, 'jrdb_paci.csv')
        df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f'  Saved: {out_path} ({len(df_out):,} rows, {len(avail)} cols)')

        df_kyi['_year'] = df_kyi['race_id'].str[:4]
        print(f'  Year coverage:')
        for yr, cnt in df_kyi.groupby('_year').size().items():
            print(f'    {yr}: {cnt:,} rows')
    else:
        print('  No KYI data found!')

    # === KAB (開催データ: トラックバイアス・馬場差) ===
    print('\n--- KAB (開催データ: 馬場差・バイアス) ---')
    kab_rows = []
    kab_errors = 0
    kab_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, 'Paci', 'KAB*.txt')))
    print(f'  {len(kab_files)} files found')

    KAB_COLUMNS = [
        ('basho_code',        1,  2),
        ('year',              3,  2),
        ('kai',               5,  1),
        ('nichi',             6,  1),
        ('yyyymmdd',          7,  8),
        ('kaisai_ku',        15,  1),  # 1:関東 2:関西 3:ローカル
        ('youbi',            16,  2),
        ('course_name',      18,  4),
        ('weather_code',     22,  1),
        ('turf_baba_code',   23,  2),
        ('turf_baba_inner',  25,  1),  # 1:絶好 2:良 3:稍荒 4:荒
        ('turf_baba_mid',    26,  1),
        ('turf_baba_outer',  27,  1),
        ('turf_baba_sa',     28,  3),  # 芝馬場差
        ('straight_sa_most', 31,  2),  # 直線馬場差最内
        ('straight_sa_inner',33,  2),
        ('straight_sa_mid',  35,  2),
        ('straight_sa_outer',37,  2),
        ('straight_sa_far',  39,  2),  # 大外
        ('dirt_baba_code',   41,  2),
        ('dirt_baba_inner',  43,  1),
        ('dirt_baba_mid',    44,  1),
        ('dirt_baba_outer',  45,  1),
        ('dirt_baba_sa',     46,  3),  # ダート馬場差
        ('data_ku',          49,  1),
        ('renzoku_day',      50,  2),
        ('turf_type',        52,  1),  # 1:野芝 2:洋芝 3:混生
        ('grass_height',     53,  4),
        ('tennatsu',         57,  1),
        ('touketsu',         58,  1),
        ('rain_mid',         59,  5),  # 中間降水量(mm)
    ]

    for fpath in kab_files:
        try:
            with open(fpath, 'rb') as f:
                for line_bytes in f:
                    line_bytes = line_bytes.rstrip(b'\r\n')
                    if len(line_bytes) < 60:
                        continue
                    try:
                        row = {}
                        for name, s, l in KAB_COLUMNS:
                            row[name] = _field(line_bytes, s, l)
                        # Build a 開催キー (not race-level, but 場+日 level)
                        row['kaisai_key'] = _build_race_id(
                            row['basho_code'], row['year'],
                            row['kai'], row['nichi'], '00'
                        )[:10]  # Without race_num → 10 chars
                        # Trim strings
                        for c in ['course_name', 'youbi', 'yyyymmdd', 'turf_type',
                                  'weather_code', 'data_ku', 'tennatsu', 'touketsu']:
                            row[c] = row[c].strip()
                        # Numeric conversions
                        for c in ['turf_baba_inner', 'turf_baba_mid', 'turf_baba_outer',
                                  'dirt_baba_inner', 'dirt_baba_mid', 'dirt_baba_outer',
                                  'renzoku_day', 'kaisai_ku']:
                            row[c] = _safe_int(row[c])
                        for c in ['turf_baba_sa', 'dirt_baba_sa',
                                  'straight_sa_most', 'straight_sa_inner',
                                  'straight_sa_mid', 'straight_sa_outer',
                                  'straight_sa_far', 'rain_mid', 'grass_height']:
                            row[c] = _safe_float(row[c])
                        del row['basho_code'], row['year'], row['kai'], row['nichi']
                        kab_rows.append(row)
                    except Exception:
                        kab_errors += 1
        except Exception:
            kab_errors += 1

    print(f'  Parsed: {len(kab_rows):,} rows ({kab_errors} errors)')

    if kab_rows:
        df_kab = pd.DataFrame(kab_rows)
        df_kab = df_kab.drop_duplicates(subset=['kaisai_key', 'yyyymmdd'], keep='last')
        print(f'  Deduplicated: → {len(df_kab):,}')

        kab_key_cols = [
            'kaisai_key', 'yyyymmdd', 'course_name', 'kaisai_ku',
            'weather_code', 'turf_baba_code', 'turf_baba_sa',
            'turf_baba_inner', 'turf_baba_mid', 'turf_baba_outer',
            'straight_sa_most', 'straight_sa_inner', 'straight_sa_mid',
            'straight_sa_outer', 'straight_sa_far',
            'dirt_baba_code', 'dirt_baba_sa',
            'dirt_baba_inner', 'dirt_baba_mid', 'dirt_baba_outer',
            'turf_type', 'grass_height', 'rain_mid', 'renzoku_day',
            'data_ku',
        ]
        avail = [c for c in kab_key_cols if c in df_kab.columns]
        df_out = df_kab[avail].copy()

        out_path = os.path.join(DATA_DIR, 'jrdb_kab.csv')
        df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f'  Saved: {out_path} ({len(df_out):,} rows, {len(avail)} cols)')

        # Year coverage
        df_kab['_year'] = df_kab['yyyymmdd'].str[:4]
        for yr, cnt in df_kab.groupby('_year').size().items():
            print(f'    {yr}: {cnt:,} rows')
    else:
        print('  No KAB data found!')

    # === CYB (調教分析データ) ===
    cyb_dir = os.path.join(EXTRACT_DIR, 'Paci')
    cyb_files = sorted(glob.glob(os.path.join(cyb_dir, 'CYB*.txt')))
    if cyb_files:
        print(f'\n--- CYB (調教分析データ) ---')
        print(f'  {len(cyb_files)} files found')
        # Read CYB spec from byte structure
        # CYB: record ~94 bytes, horse-level
        cyb_rows = []
        cyb_errors = 0

        CYB_COLUMNS = [
            ('basho_code',     1,  2),
            ('year',           3,  2),
            ('kai',            5,  1),
            ('nichi',          6,  1),
            ('race_num',       7,  2),
            ('umaban',         9,  2),
            ('train_type',    11,  1),  # 調教タイプ
            ('train_course_type', 12, 1),
            ('train_baba',    13,  1),  # 坂路/CW等
            ('train_mark',    14,  1),  # 追切指標
            ('train_amount',  15,  1),  # 仕上げ指標
            ('train_change',  16,  1),  # 変化指標
            ('train_comment', 17, 40),  # 調教コメント
            ('comment_year',  57,  3),  # コメント年
            ('comment_date',  60,  3),  # コメント回日
            ('train_eval',    63,  1),  # 調教評価
            ('train_course',  64,  2),  # 調教コース
        ]

        for fpath in cyb_files:
            try:
                with open(fpath, 'rb') as f:
                    for line_bytes in f:
                        line_bytes = line_bytes.rstrip(b'\r\n')
                        if len(line_bytes) < 60:
                            continue
                        try:
                            row = {}
                            for name, s, l in CYB_COLUMNS:
                                row[name] = _field(line_bytes, s, l)
                            row['race_id'] = _build_race_id(
                                row['basho_code'], row['year'],
                                row['kai'], row['nichi'], row['race_num']
                            )
                            row['umaban'] = _safe_int(row['umaban'])
                            for c in ['train_comment']:
                                row[c] = row[c].strip()
                            for c in ['basho_code', 'year', 'kai', 'nichi', 'race_num']:
                                del row[c]
                            cyb_rows.append(row)
                        except Exception:
                            cyb_errors += 1
            except Exception:
                cyb_errors += 1

        print(f'  Parsed: {len(cyb_rows):,} rows ({cyb_errors} errors)')
        if cyb_rows:
            df_cyb = pd.DataFrame(cyb_rows)
            df_cyb = df_cyb.drop_duplicates(subset=['race_id', 'umaban'], keep='last')
            out_path = os.path.join(DATA_DIR, 'jrdb_cyb.csv')
            df_cyb.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f'  Saved: {out_path} ({len(df_cyb):,} rows)')

    elapsed = (datetime.now() - start).total_seconds()
    print(f'\n{"="*60}')
    print(f'  Done in {elapsed:.1f}s')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
