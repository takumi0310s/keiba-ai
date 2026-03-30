#!/usr/bin/env python3
"""JRDB データ取得・パーサー

JRDBアドバンスコースからLZH形式の固定長テキストをダウンロードし、
CSV化して保存する。

認証: .env の JRDB_ID / JRDB_PASSWORD（HTTP Basic Auth）
形式: CP932固定長テキスト → CSV

対象ファイル:
  KYI (前日データ): IDM, 調教指数, 厩舎指数, 情報指数, 展開予想指数, 激走指数 等
  TYB (直前データ): パドック指数, オッズ指数, 総合指数, 馬体コード, 気配コード
  SED (成績データ): 馬場差, レースペース指数, 馬ペース, 上昇度, IDM(確定), 不利 等

使い方:
  python tools/scrape_jrdb.py                    # 直近1週分を取得
  python tools/scrape_jrdb.py --date 20260329    # 指定日を取得
  python tools/scrape_jrdb.py --range 20260301 20260331  # 期間取得
  python tools/scrape_jrdb.py --type KYI         # KYIのみ取得
"""

import os
import sys
import io
import time
import struct
import zipfile
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# dotenv
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
JRDB_DIR = os.path.join(DATA_DIR, 'jrdb_raw')  # 生ファイル保存

# JRDB認証
JRDB_ID = os.environ.get('JRDB_ID', '')
JRDB_PASSWORD = os.environ.get('JRDB_PASSWORD', '')

# ダウンロードURL（/member/data/ が正しいパス）
JRDB_BASE_URL = 'http://www.jrdb.com/member/data'
JRDB_URLS = {
    'KYI': f'{JRDB_BASE_URL}/Kyi/KYI{{date}}.lzh',
    'SED': f'{JRDB_BASE_URL}/Sed/SED{{date}}.lzh',
    'TYB': f'{JRDB_BASE_URL}/Tyb/TYB{{date}}.lzh',
}

# =====================================================
# JRDB場コード → 競馬場名
# =====================================================
JRDB_COURSE_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
    '05': '東京', '06': '中山', '07': '中京', '08': '京都',
    '09': '阪神', '10': '小倉',
}

# =====================================================
# KYI (前日データ) パーサー定義 - 1024バイト/レコード
# =====================================================
KYI_FIELDS = [
    # (名前, 開始位置(1-indexed), バイト数, 型)
    # --- レースキー ---
    ('場コード',          1,  2, 'str'),
    ('年',                3,  2, 'str'),
    ('回',                5,  1, 'str'),
    ('日',                6,  1, 'hex'),   # hex: 1-9, a=10, b=11, c=12
    ('R',                 7,  2, 'str'),
    ('馬番',              9,  2, 'int'),
    ('血統登録番号',     11,  8, 'str'),
    ('馬名',             19, 36, 'str'),
    # --- JRDB指数 ---
    ('IDM',              55,  5, 'float'),  # INDEX MEMORY（主力指数）
    ('騎手指数',         60,  5, 'float'),
    ('情報指数',         65,  5, 'float'),
    ('予備1',            70,  5, 'float'),
    ('予備2',            75,  5, 'float'),
    ('予備3',            80,  5, 'float'),
    ('総合指数',         85,  5, 'float'),  # IDM+騎手+情報
    # --- 特性 ---
    ('脚質',             90,  1, 'int'),    # 1:逃 2:先行 3:差し 4:追込 5:好位差 6:自在
    ('距離適性',         91,  1, 'int'),    # 1:短 2:中 3:長 5:マイル 6:万能
    ('上昇度',           92,  1, 'int'),    # 1:AA 2:A 3:B 4:C 5:?
    ('ローテーション',   93,  3, 'float'),  # 週数
    # --- オッズ ---
    ('基準オッズ',       96,  5, 'float'),
    ('基準人気順位',    101,  2, 'int'),
    ('基準複勝オッズ',  103,  5, 'float'),
    ('基準複勝人気順位',108,  2, 'int'),
    # --- 指数(重要) ---
    ('人気指数',        140,  5, 'float'),
    ('調教指数',        145,  5, 'float'),  # 調教評価（主力特徴量候補）
    ('厩舎指数',        150,  5, 'float'),  # 厩舎評価（主力特徴量候補）
    # --- 第3版 ---
    ('調教矢印コード',  155,  1, 'int'),    # 1:抜群 2:上昇 3:平行 4:やや下降 5:落ち
    ('厩舎評価コード',  156,  1, 'int'),    # 1:超強気 2:強気 3:現状維持 4:弱気
    ('騎手期待連対率',  157,  4, 'float'),
    ('激走指数',        161,  3, 'int'),    # 激走ポテンシャル
    ('蹄コード',        164,  2, 'int'),    # 蹄の状態
    ('重適性コード',    166,  1, 'int'),    # 1:◎得意 2:○普通 3:△苦手
    ('クラスコード',    167,  2, 'int'),    # 能力クラス(01-62)
    # --- 第5版 ---
    ('総合印',          327,  1, 'int'),    # 1:◎ 2:○ 3:▲ 4:注 5:△
    ('IDM印',           328,  1, 'int'),
    ('情報印',          329,  1, 'int'),
    ('騎手印',          330,  1, 'int'),
    ('厩舎印',          331,  1, 'int'),
    ('調教印',          332,  1, 'int'),
    ('激走印',          333,  1, 'int'),    # 1:激走馬
    ('芝適性コード',    334,  1, 'str'),    # 1:◎ 2:○ 3:△
    ('ダ適性コード',    335,  1, 'str'),    # 1:◎ 2:○ 3:△
    ('騎手コード',      336,  5, 'str'),
    ('調教師コード',    341,  5, 'str'),
    # --- 第6版: 展開予想 ---
    ('テン指数予想',    359,  5, 'float'),  # 前半3F指数予想
    ('ペース指数予想',  364,  5, 'float'),  # ペース指数予想
    ('上がり指数予想',  369,  5, 'float'),  # 上がり3F指数予想
    ('位置指数予想',    374,  5, 'float'),  # 位置指数予想
    ('ペース予想',      379,  1, 'str'),    # H/M/S
    # --- 第7版 ---
    ('取消フラグ',      403,  1, 'int'),    # 1:取消
    ('性別コード',      404,  1, 'int'),    # 1:牡 2:牝 3:セン
    ('激走順位',        449,  2, 'int'),
    ('LS指数順位',      451,  2, 'int'),
    ('テン指数順位',    453,  2, 'int'),
    ('ペース指数順位',  455,  2, 'int'),
    ('上がり指数順位',  457,  2, 'int'),
    ('位置指数順位',    459,  2, 'int'),
    # --- 第8版 ---
    ('騎手期待単勝率',  461,  4, 'float'),
    ('騎手期待3着内率', 465,  4, 'float'),
    ('輸送区分',        469,  1, 'str'),    # 1:滞在 2:通常 3:遠征 4:連闘
    # --- 第10版 ---
    ('降級フラグ',      539,  1, 'int'),
    ('激走タイプ',      540,  2, 'str'),
    # --- 第11版 ---
    ('放牧先ランク',    623,  1, 'str'),    # A-E
    ('厩舎ランク',      624,  1, 'int'),    # 1(top)-9(low)
]

# =====================================================
# TYB (直前データ) パーサー定義 - 128バイト/レコード
# =====================================================
TYB_FIELDS = [
    # --- レースキー ---
    ('場コード',          1,  2, 'str'),
    ('年',                3,  2, 'str'),
    ('回',                5,  1, 'str'),
    ('日',                6,  1, 'hex'),
    ('R',                 7,  2, 'str'),
    ('馬番',              9,  2, 'int'),
    # --- 直前指数 ---
    ('IDM',              11,  5, 'float'),
    ('騎手指数',         16,  5, 'float'),
    ('情報指数',         21,  5, 'float'),
    ('オッズ指数',       26,  5, 'float'),  # 直前オッズ指数（主力特徴量候補）
    ('パドック指数',     31,  5, 'float'),  # パドック指数（主力特徴量候補）
    ('予備1',            36,  5, 'float'),
    ('総合指数',         41,  5, 'float'),  # 直前総合指数
    # --- 直前情報 ---
    ('馬具変更情報',     46,  1, 'int'),    # 0:無 1:変更 2:変更(効果有)
    ('脚元情報',         47,  1, 'int'),    # 0:フラット 1:好転 2:疑問 3:悪化
    ('取消フラグ',       48,  1, 'int'),
    ('馬場状態コード',   70,  2, 'int'),    # 10:良 20:稍重 30:重 40:不良
    ('天候コード',       72,  1, 'int'),    # 1:晴 2:曇 3:小雨 4:雨 5:小雪 6:雪
    ('単勝オッズ',       73,  6, 'float'),
    ('複勝オッズ',       79,  6, 'float'),
    ('馬体重',           89,  3, 'int'),
    ('馬体重増減',       92,  3, 'str'),    # 符号+2桁
    ('オッズ印',         95,  1, 'str'),
    ('パドック印',       96,  1, 'str'),
    ('直前総合印',       97,  1, 'str'),
    ('馬体コード',       98,  1, 'int'),    # 1:太い 2:余裕 3:良い 4:普通 5:細い 6:張り 7:緩い
    ('気配コード',       99,  1, 'int'),    # 1:良 2:平凡 3:不安定 4:イレ込 5:チャカ 6:おとなしい 7:ピカ 8:イレチ
]

# =====================================================
# SED (成績データ) パーサー定義 - 376バイト/レコード
# =====================================================
SED_FIELDS = [
    # --- レースキー ---
    ('場コード',          1,  2, 'str'),
    ('年',                3,  2, 'str'),
    ('回',                5,  1, 'str'),
    ('日',                6,  1, 'hex'),
    ('R',                 7,  2, 'str'),
    ('馬番',              9,  2, 'int'),
    ('血統登録番号',     11,  8, 'str'),
    ('年月日',           19,  8, 'str'),    # YYYYMMDD
    ('馬名',             27, 36, 'str'),
    # --- レース条件 ---
    ('距離',             63,  4, 'int'),
    ('芝ダ障害コード',   67,  1, 'int'),    # 1:芝 2:ダ 3:障害
    ('右左',             68,  1, 'int'),
    ('内外',             69,  1, 'int'),
    ('馬場状態',         70,  2, 'int'),    # 10:良 20:稍重 30:重 40:不良
    ('種別',             72,  2, 'int'),
    ('頭数',            131,  2, 'int'),
    # --- 馬成績 ---
    ('着順',            141,  2, 'int'),
    ('異常区分',        143,  1, 'int'),    # 0:正常 1:取消 2:除外 3:中止 4:失格 5:降着
    ('タイム',          144,  4, 'str'),    # 分+秒(0.1s)
    ('斤量',            148,  3, 'float'),  # 0.1kg単位
    ('確定単勝オッズ',  175,  6, 'float'),
    ('確定単勝人気',    181,  2, 'int'),
    # --- JRDBデータ(主力) ---
    ('IDM',             183,  3, 'float'),  # 確定IDM
    ('素点',            186,  3, 'float'),
    ('馬場差',          189,  3, 'float'),  # トラックバイアス補正値
    ('ペース',          192,  3, 'float'),
    ('出遅',            195,  3, 'float'),  # 出遅れ補正
    ('位置取',          198,  3, 'float'),
    ('不利',            201,  3, 'float'),  # 不利補正（合計）
    ('前不利',          204,  3, 'float'),  # 前半3F不利
    ('中不利',          207,  3, 'float'),
    ('後不利',          210,  3, 'float'),  # 後半3F不利
    ('レース',          213,  3, 'float'),  # レースレベル
    ('コース取り',      216,  1, 'int'),    # 1:最内 2:内 3:中 4:外 5:大外
    ('上昇度コード',    217,  1, 'int'),    # 1:AA 2:A 3:B 4:C 5:?
    ('クラスコード',    218,  2, 'int'),
    ('馬体コード',      220,  1, 'int'),
    ('気配コード',      221,  1, 'int'),
    ('レースペース',    222,  1, 'str'),    # H/M/S
    ('馬ペース',        223,  1, 'str'),    # H/M/S
    ('テン指数',        224,  5, 'float'),  # 確定前半3F指数
    ('上がり指数',      229,  5, 'float'),  # 確定上がり3F指数
    ('ペース指数',      234,  5, 'float'),
    ('レースP指数',     239,  5, 'float'),  # レースペース指数
    ('前3Fタイム',      259,  3, 'float'),  # 0.1s単位
    ('後3Fタイム',      262,  3, 'float'),  # 0.1s単位
    # --- 第2版 ---
    ('コーナー順位1',   309,  2, 'int'),
    ('コーナー順位2',   311,  2, 'int'),
    ('コーナー順位3',   313,  2, 'int'),
    ('コーナー順位4',   315,  2, 'int'),
    # --- 第3版 ---
    ('馬体重',          333,  3, 'int'),
    ('馬体重増減',      336,  3, 'str'),
    ('天候コード',      339,  1, 'int'),
    # --- 第4版 ---
    ('レースペース流れ',366,  2, 'int'),
    ('馬ペース流れ',    368,  2, 'int'),
    ('4角コース取り',   370,  1, 'int'),
]


# =====================================================
# LZH解凍（簡易実装）
# =====================================================

def extract_lzh(lzh_bytes):
    """LZH/ZIP形式のバイトデータからファイルを抽出する。

    解凍手段の優先順位:
    1. ZIP形式として試行（Python標準ライブラリ）
    2. lhafile（pip install lhafile、C拡張が必要）
    3. 7-Zip外部コマンド
    4. Pure Python LZH (-lh0- 無圧縮のみ)
    """
    # Method 1: ZIP（JRDBはzipでも配信している場合がある）
    try:
        zf = zipfile.ZipFile(io.BytesIO(lzh_bytes))
        result = {}
        for name in zf.namelist():
            result[name] = zf.read(name)
        if result:
            return result
    except (zipfile.BadZipFile, Exception):
        pass

    # Method 2: lhafile
    try:
        import lhafile
        lha = lhafile.Lhafile(io.BytesIO(lzh_bytes))
        result = {}
        for info in lha.infolist():
            fname = info.filename
            data = lha.read(fname)
            result[fname] = data
        return result
    except ImportError:
        pass

    # Method 3: 7-Zip外部コマンド
    import tempfile
    import subprocess
    tmp_dir = tempfile.mkdtemp()
    lzh_path = os.path.join(tmp_dir, 'data.lzh')
    with open(lzh_path, 'wb') as f:
        f.write(lzh_bytes)

    seven_zip_paths = [
        r'C:\Program Files\7-Zip\7z.exe',
        r'C:\Program Files (x86)\7-Zip\7z.exe',
        '7z',
    ]
    extracted = False
    for sz_path in seven_zip_paths:
        try:
            subprocess.run(
                [sz_path, 'x', '-y', f'-o{tmp_dir}', lzh_path],
                capture_output=True, check=True
            )
            extracted = True
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    if extracted:
        result = {}
        for f in os.listdir(tmp_dir):
            if f.endswith('.lzh'):
                continue
            fpath = os.path.join(tmp_dir, f)
            with open(fpath, 'rb') as fh:
                result[f] = fh.read()
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return result

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Method 4: Pure Python LZH parser (-lh0- stored only)
    result = _parse_lzh_stored(lzh_bytes)
    if result:
        return result

    raise RuntimeError(
        "LZH解凍に失敗。以下のいずれかを実行してください:\n"
        "  1. 7-Zipをインストール: https://www.7-zip.org/\n"
        "  2. pip install lhafile (C++ビルド環境が必要)\n"
        "  3. JRDBからZIP形式でダウンロード"
    )


def _parse_lzh_stored(lzh_bytes):
    """LZH -lh0- (無圧縮) ファイルのpure Pythonパーサー

    JRDBの一部ファイルは-lh0-(stored)で圧縮されている場合がある。
    -lh5-等の圧縮形式には非対応。
    """
    result = {}
    pos = 0
    data = lzh_bytes

    while pos < len(data) - 2:
        # Level 0/1/2 header parsing
        header_size = data[pos]
        if header_size == 0:
            break

        checksum = data[pos + 1]
        method = data[pos + 2:pos + 7].decode('ascii', errors='replace')

        if method not in ['-lh0-', '-lzs-', '-lz4-', '-lh5-', '-lh6-', '-lh7-']:
            break

        compressed_size = struct.unpack('<I', data[pos + 7:pos + 11])[0]
        original_size = struct.unpack('<I', data[pos + 11:pos + 15])[0]

        # ファイル名の長さと名前
        name_len = data[pos + 21]
        filename = data[pos + 22:pos + 22 + name_len].decode('cp932', errors='replace')

        # ヘッダーの後のデータ位置
        header_total = header_size + 2
        data_start = pos + header_total
        file_data = data[data_start:data_start + compressed_size]

        if method == '-lh0-':
            # 無圧縮
            result[filename] = file_data
        else:
            # 圧縮形式は対応外 - 7-Zipが必要
            print(f"  [WARN] {filename}: {method} compression not supported in pure Python")
            return {}

        pos = data_start + compressed_size

    return result


# =====================================================
# 固定長テキストパーサー
# =====================================================

def _parse_hex_nichi(val):
    """JRDB日コード(hex) → 整数: 1-9はそのまま, a=10, b=11, c=12"""
    val = val.strip().lower()
    if val in ('a', 'b', 'c'):
        return 10 + ord(val) - ord('a')
    try:
        return int(val)
    except ValueError:
        return 0


def _parse_value(raw_bytes, dtype):
    """固定長フィールドの生バイトを指定型に変換"""
    text = raw_bytes.decode('cp932', errors='replace').strip()
    if not text or text == '*' * len(text):
        return None

    if dtype == 'str':
        return text
    elif dtype == 'hex':
        return _parse_hex_nichi(text)
    elif dtype == 'int':
        try:
            return int(text)
        except ValueError:
            return None
    elif dtype == 'float':
        try:
            return float(text)
        except ValueError:
            return None
    return text


def parse_fixed_length(data_bytes, fields, record_len=None):
    """固定長テキストをパースしてDataFrameに変換

    Args:
        data_bytes: CP932エンコードされた固定長テキストのバイト列
        fields: [(名前, 開始位置(1-indexed), バイト数, 型), ...]
        record_len: 1レコードのバイト数（Noneなら最初の改行から自動検出）
    """
    # レコード長の自動検出
    if record_len is None:
        # CR+LFを探す
        crlf_pos = data_bytes.find(b'\r\n')
        if crlf_pos > 0:
            record_len = crlf_pos + 2
        else:
            lf_pos = data_bytes.find(b'\n')
            if lf_pos > 0:
                record_len = lf_pos + 1
            else:
                raise ValueError("改行が見つからない。record_lenを指定してください。")

    records = []
    offset = 0
    while offset + record_len <= len(data_bytes):
        line = data_bytes[offset:offset + record_len]
        record = {}
        for name, start, length, dtype in fields:
            # 1-indexed → 0-indexed
            s = start - 1
            e = s + length
            raw = line[s:e] if e <= len(line) else b''
            record[name] = _parse_value(raw, dtype)
        records.append(record)
        offset += record_len

    return pd.DataFrame(records)


# =====================================================
# JRDB race_key → JRA race_id 変換
# =====================================================

def jrdb_to_jra_race_id(row):
    """JRDBレコードからJRA形式のrace_id(10桁)を生成

    JRDB: 場コード(2) + 年(2) + 回(1) + 日(1,hex) + R(2)
    JRA:  course_code(2) + year(2) + kai(2) + nichi(2) + race_num(2)
    """
    course = str(row.get('場コード', '')).zfill(2)
    year = str(row.get('年', '')).zfill(2)
    kai_raw = row.get('回', '')
    kai = str(kai_raw).zfill(2) if kai_raw is not None else '00'
    nichi_raw = row.get('日', 0)
    if isinstance(nichi_raw, int):
        nichi = str(nichi_raw).zfill(2)
    else:
        nichi = str(_parse_hex_nichi(str(nichi_raw))).zfill(2)
    race_num = str(row.get('R', '')).zfill(2)
    return f"{course}{year}{kai}{nichi}{race_num}"


def jrdb_to_netkeiba_race_id(row):
    """JRDBレコードからnetkeiba形式のrace_id(12桁)を生成

    netkeiba: 20YY + course_code(2) + kai(2) + nichi(2) + race_num(2)
    """
    year_2d = str(row.get('年', '')).zfill(2)
    course = str(row.get('場コード', '')).zfill(2)
    kai_raw = row.get('回', '')
    kai = str(kai_raw).zfill(2) if kai_raw is not None else '00'
    nichi_raw = row.get('日', 0)
    if isinstance(nichi_raw, int):
        nichi = str(nichi_raw).zfill(2)
    else:
        nichi = str(_parse_hex_nichi(str(nichi_raw))).zfill(2)
    race_num = str(row.get('R', '')).zfill(2)
    return f"20{year_2d}{course}{kai}{nichi}{race_num}"


# =====================================================
# ダウンロード
# =====================================================

def download_jrdb_file(file_type, date_str, force=False):
    """JRDBからデータファイルをダウンロード

    Args:
        file_type: 'KYI', 'SED', 'TYB'
        date_str: 'YYMMDD' 形式
        force: Trueなら既存ファイルを上書き

    Returns:
        dict: {ファイル名: バイトデータ} or None
    """
    if not JRDB_ID or not JRDB_PASSWORD:
        print("[ERROR] JRDB_ID/JRDB_PASSWORD が .env に設定されていません")
        return None

    # キャッシュ確認
    cache_dir = os.path.join(JRDB_DIR, file_type.lower())
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{file_type}{date_str}.lzh')

    if os.path.exists(cache_path) and not force:
        print(f"  [CACHE] {cache_path}")
        with open(cache_path, 'rb') as f:
            return extract_lzh(f.read())

    url = JRDB_URLS[file_type].format(date=date_str)
    print(f"  [DL] {url}")

    try:
        resp = requests.get(url, auth=(JRDB_ID, JRDB_PASSWORD), timeout=30)
        if resp.status_code == 401:
            print("[ERROR] 認証失敗。JRDB_ID/JRDB_PASSWORDを確認してください")
            return None
        if resp.status_code == 404:
            print(f"  [SKIP] {file_type}{date_str} - データなし(404)")
            return None
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] ダウンロード失敗: {e}")
        return None

    # キャッシュ保存
    with open(cache_path, 'wb') as f:
        f.write(resp.content)

    return extract_lzh(resp.content)


# =====================================================
# メインパース関数
# =====================================================

def parse_kyi(data_bytes):
    """KYI(前日データ)をパースしてDataFrameを返す"""
    df = parse_fixed_length(data_bytes, KYI_FIELDS)
    df['jra_race_id'] = df.apply(jrdb_to_jra_race_id, axis=1)
    df['nk_race_id'] = df.apply(jrdb_to_netkeiba_race_id, axis=1)
    # 馬名トリム
    if '馬名' in df.columns:
        df['馬名'] = df['馬名'].str.strip()
    return df


def parse_tyb(data_bytes):
    """TYB(直前データ)をパースしてDataFrameを返す"""
    df = parse_fixed_length(data_bytes, TYB_FIELDS)
    df['jra_race_id'] = df.apply(jrdb_to_jra_race_id, axis=1)
    df['nk_race_id'] = df.apply(jrdb_to_netkeiba_race_id, axis=1)
    return df


def parse_sed(data_bytes):
    """SED(成績データ)をパースしてDataFrameを返す"""
    df = parse_fixed_length(data_bytes, SED_FIELDS)
    df['jra_race_id'] = df.apply(jrdb_to_jra_race_id, axis=1)
    df['nk_race_id'] = df.apply(jrdb_to_netkeiba_race_id, axis=1)
    # SED特有の変換
    if '斤量' in df.columns:
        df['斤量'] = df['斤量'].apply(lambda x: x / 10 if pd.notna(x) else None)
    if '前3Fタイム' in df.columns:
        df['前3Fタイム'] = df['前3Fタイム'].apply(lambda x: x / 10 if pd.notna(x) else None)
    if '後3Fタイム' in df.columns:
        df['後3Fタイム'] = df['後3Fタイム'].apply(lambda x: x / 10 if pd.notna(x) else None)
    if '馬名' in df.columns:
        df['馬名'] = df['馬名'].str.strip()
    return df


# =====================================================
# 日付範囲のレースデータ取得
# =====================================================

def get_race_dates(start_date, end_date):
    """開催日リストを生成（土日のみ）"""
    dates = []
    d = start_date
    while d <= end_date:
        if d.weekday() in (5, 6):  # 土日
            dates.append(d)
        d += timedelta(days=1)
    return dates


def fetch_and_parse(file_type, date_str):
    """1日分のデータをダウンロード・パース

    Returns:
        DataFrame or None
    """
    files = download_jrdb_file(file_type, date_str)
    if not files:
        return None

    parser = {'KYI': parse_kyi, 'TYB': parse_tyb, 'SED': parse_sed}[file_type]

    dfs = []
    for fname, data in files.items():
        ext = fname.lower().split('.')[-1]
        if ext not in ('txt', 'dat', ''):
            # KYI260329.txt のようなテキストファイルのみパース
            # ファイル名にfile_typeが含まれるか確認
            if file_type.lower() not in fname.lower():
                continue
        try:
            df = parser(data)
            if len(df) > 0:
                dfs.append(df)
                print(f"    {fname}: {len(df)} records")
        except Exception as e:
            print(f"    {fname}: パース失敗 - {e}")

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def download_range(file_type, start_date, end_date, force=False):
    """指定期間のJRDBデータを取得してCSV保存

    Returns:
        DataFrame: 統合済みデータ
    """
    dates = get_race_dates(start_date, end_date)
    print(f"\n=== {file_type} データ取得 ({len(dates)}日間) ===")

    all_dfs = []
    for d in dates:
        date_str = d.strftime('%y%m%d')
        print(f"\n[{d.strftime('%Y-%m-%d')}] {file_type}{date_str}")
        df = fetch_and_parse(file_type, date_str)
        if df is not None:
            all_dfs.append(df)
        time.sleep(1)  # サーバー負荷軽減

    if not all_dfs:
        print(f"  データなし")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    # 重複除去（race_key + 馬番）
    key_cols = ['jra_race_id', '馬番']
    if all(c in result.columns for c in key_cols):
        before = len(result)
        result = result.drop_duplicates(subset=key_cols, keep='last')
        if before > len(result):
            print(f"  重複除去: {before} → {len(result)}")

    return result


# =====================================================
# CSV保存・読込
# =====================================================

CSV_PATHS = {
    'KYI': os.path.join(DATA_DIR, 'jrdb_kyi.csv'),
    'TYB': os.path.join(DATA_DIR, 'jrdb_tyb.csv'),
    'SED': os.path.join(DATA_DIR, 'jrdb_sed.csv'),
}


def save_csv(df, file_type, append=True):
    """CSVに保存（既存データに追記 or 上書き）"""
    path = CSV_PATHS[file_type]
    if append and os.path.exists(path):
        existing = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
        df_str = df.astype(str)
        combined = pd.concat([existing, df_str], ignore_index=True)
        key_cols = ['jra_race_id', '馬番']
        if all(c in combined.columns for c in key_cols):
            combined = combined.drop_duplicates(subset=key_cols, keep='last')
        combined.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"  保存: {path} ({len(combined)} rows, 追記)")
    else:
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"  保存: {path} ({len(df)} rows)")


def load_csv(file_type):
    """保存済みCSVを読み込み"""
    path = CSV_PATHS[file_type]
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding='utf-8-sig', dtype=str)


# =====================================================
# メイン
# =====================================================

def main():
    parser = argparse.ArgumentParser(description='JRDBデータ取得・パーサー')
    parser.add_argument('--date', type=str, help='取得日 (YYYYMMDD)')
    parser.add_argument('--range', nargs=2, type=str, metavar=('START', 'END'),
                        help='取得期間 (YYYYMMDD YYYYMMDD)')
    parser.add_argument('--type', type=str, choices=['KYI', 'TYB', 'SED', 'ALL'],
                        default='ALL', help='取得データ種別')
    parser.add_argument('--force', action='store_true', help='キャッシュ無視で再取得')
    parser.add_argument('--weeks', type=int, default=1, help='直近N週分を取得（デフォルト1）')
    args = parser.parse_args()

    # 日付範囲決定
    if args.range:
        start = datetime.strptime(args.range[0], '%Y%m%d')
        end = datetime.strptime(args.range[1], '%Y%m%d')
    elif args.date:
        d = datetime.strptime(args.date, '%Y%m%d')
        start = d
        end = d
    else:
        # 直近N週分
        end = datetime.now()
        start = end - timedelta(weeks=args.weeks)

    types = ['KYI', 'TYB', 'SED'] if args.type == 'ALL' else [args.type]

    print(f"JRDB データ取得")
    print(f"  期間: {start.strftime('%Y-%m-%d')} ～ {end.strftime('%Y-%m-%d')}")
    print(f"  種別: {', '.join(types)}")
    print(f"  ID: {JRDB_ID}")

    os.makedirs(JRDB_DIR, exist_ok=True)

    for ft in types:
        df = download_range(ft, start, end, force=args.force)
        if len(df) > 0:
            save_csv(df, ft, append=True)
            print(f"\n  {ft}: {len(df)} records saved")
            # サマリー表示
            if 'IDM' in df.columns:
                valid_idm = pd.to_numeric(df['IDM'], errors='coerce').notna().sum()
                print(f"  IDM有効率: {valid_idm}/{len(df)} ({valid_idm/len(df)*100:.1f}%)")
        else:
            print(f"\n  {ft}: データなし")

    print("\n完了!")


if __name__ == '__main__':
    main()
