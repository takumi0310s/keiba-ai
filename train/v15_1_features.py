"""V15.1 features 拡張 (KKA/SKB/SR 逆輸入).

V15 (150 features) に以下を追加:
- KKA seiseki 系 (16): jra/heavy/class/speed × seiseki_1/2/3/out
- SKB code 系 (10): kishi/baba/kyaku × code_1/2/3 + turf_hoof
- SRB race-level (8): corner3/4_order, pace_up_pos, bias_3corner/4corner/straight, race_comment_len

合計 +34 features → V15.1 = 184 features

usage:
  from train.v15_1_features import merge_v15_1_features, V15_1_NEW_FEATURES
  df_extended = merge_v15_1_features(df, kka_path, skb_path, srb_path)
"""
from __future__ import annotations

import os, sys
import pandas as pd
import numpy as np

BASE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


# ===== race_id 形式変換 =====
# Cache (V15) race_id は JRDB-internal 8 chars + umaban 2 chars = 10 chars
#   race-level part: course(2) + year_short(2) + kai(1) + nichi(1, hex) + race(2)
#   末尾 2 chars = umaban (zero padded)
# KKA/SKB/SR race_id は netkeiba-format 12 chars: 20yy + course(2) + kai(2, dec) + nichi(2, dec) + race(2)
def netkeiba_to_jrdb_internal(rid: str) -> str:
    """KKA/SKB/SR の 12-char race_id を JRDB-internal 8-char に変換 (race-level only)."""
    if not isinstance(rid, str) or len(rid) != 12:
        return ''
    yy = rid[2:4]
    course = rid[4:6]
    try:
        kai_int = int(rid[6:8])
        nichi_int = int(rid[8:10])
    except ValueError:
        return ''
    race = rid[10:12]
    kai_c = str(kai_int) if kai_int <= 9 else format(kai_int, 'X')[-1]
    nichi_c = format(nichi_int, 'X')[-1] if nichi_int <= 15 else 'F'
    return f"{course}{yy}{kai_c}{nichi_c}{race}"


# ===== V15.1 で追加する features =====

V15_1_KKA_FEATURES = [
    # JRA 成績
    'kka_jra_seiseki_1', 'kka_jra_seiseki_2', 'kka_jra_seiseki_3', 'kka_jra_seiseki_out',
    # 重馬場 成績
    'kka_heavy_seiseki_1', 'kka_heavy_seiseki_2', 'kka_heavy_seiseki_3', 'kka_heavy_seiseki_out',
    # クラス 成績
    'kka_class_seiseki_1', 'kka_class_seiseki_2', 'kka_class_seiseki_3', 'kka_class_seiseki_out',
    # スピード 成績
    'kka_speed_seiseki_1', 'kka_speed_seiseki_2', 'kka_speed_seiseki_3', 'kka_speed_seiseki_out',
]

V15_1_SKB_FEATURES = [
    # 騎手相性 (kishi_code 1〜3)
    'skb_kishi_code_1', 'skb_kishi_code_2', 'skb_kishi_code_3',
    # 馬場適性 (baba_code 1〜3)
    'skb_baba_code_1', 'skb_baba_code_2', 'skb_baba_code_3',
    # 脚質 (kyaku_code 1〜3)
    'skb_kyaku_code_1', 'skb_kyaku_code_2', 'skb_kyaku_code_3',
    # フラグ
    'skb_turf_hoof',
]

V15_1_SRB_FEATURES = [
    # コーナー位置 (race-level、broadcast)
    'srb_corner3_order_avg', 'srb_corner4_order_avg',
    # ペース
    'srb_pace_up_pos',
    # トラックバイアス (3coner / 4corner / straight)
    'srb_bias_3corner', 'srb_bias_4corner', 'srb_bias_straight',
    # コメント長 (race_comment 字数 → bias の指標代替)
    'srb_race_comment_len',
    # furlong_times (1F あたり 平均、解析が必要、ここでは長さ)
    'srb_furlong_times_n',
]

V15_1_NEW_FEATURES = V15_1_KKA_FEATURES + V15_1_SKB_FEATURES + V15_1_SRB_FEATURES


# ===== KKA 取込 =====

KKA_SOURCE_COLS = {
    'kka_jra_seiseki_1':    'jra_seiseki_1',
    'kka_jra_seiseki_2':    'jra_seiseki_2',
    'kka_jra_seiseki_3':    'jra_seiseki_3',
    'kka_jra_seiseki_out':  'jra_seiseki_out',
    'kka_heavy_seiseki_1':  'heavy_seiseki_1',
    'kka_heavy_seiseki_2':  'heavy_seiseki_2',
    'kka_heavy_seiseki_3':  'heavy_seiseki_3',
    'kka_heavy_seiseki_out':'heavy_seiseki_out',
    'kka_class_seiseki_1':  'class_seiseki_1',
    'kka_class_seiseki_2':  'class_seiseki_2',
    'kka_class_seiseki_3':  'class_seiseki_3',
    'kka_class_seiseki_out':'class_seiseki_out',
    'kka_speed_seiseki_1':  'speed_seiseki_1',
    'kka_speed_seiseki_2':  'speed_seiseki_2',
    'kka_speed_seiseki_3':  'speed_seiseki_3',
    'kka_speed_seiseki_out':'speed_seiseki_out',
}


def load_kka(path: str = 'data/jrdb_kka.csv') -> pd.DataFrame:
    """KKA 統計を horse 単位 (race_id, umaban) で読み込む。race_id は cache (JRA-format) に変換."""
    use = ['race_id', 'umaban'] + list(KKA_SOURCE_COLS.values())
    df = pd.read_csv(os.path.join(BASE, path), dtype={'race_id': str, 'umaban': str}, usecols=use)
    df = df.rename(columns={v: k for k, v in KKA_SOURCE_COLS.items()})
    df['race_part'] = df['race_id'].apply(netkeiba_to_jrdb_internal)
    df = df.drop(columns=['race_id'])
    df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce').astype('Int64').astype(str)
    df = df[df['race_part'] != ''].drop_duplicates(subset=['race_part','umaban'], keep='last')
    return df


# ===== SKB 取込 =====

SKB_SOURCE_COLS = {
    'skb_kishi_code_1': 'kishi_code_1',
    'skb_kishi_code_2': 'kishi_code_2',
    'skb_kishi_code_3': 'kishi_code_3',
    'skb_baba_code_1':  'baba_code_1',
    'skb_baba_code_2':  'baba_code_2',
    'skb_baba_code_3':  'baba_code_3',
    'skb_kyaku_code_1': 'kyaku_code_1',
    'skb_kyaku_code_2': 'kyaku_code_2',
    'skb_kyaku_code_3': 'kyaku_code_3',
    'skb_turf_hoof':    'turf_hoof',
}


def load_skb(path: str = 'data/jrdb_skb.csv') -> pd.DataFrame:
    use = ['race_id', 'umaban'] + list(SKB_SOURCE_COLS.values())
    df = pd.read_csv(os.path.join(BASE, path), dtype={'race_id': str, 'umaban': str}, usecols=use)
    df = df.rename(columns={v: k for k, v in SKB_SOURCE_COLS.items()})
    # turf_hoof は object → category-encode
    if 'skb_turf_hoof' in df.columns:
        df['skb_turf_hoof'] = pd.Categorical(df['skb_turf_hoof']).codes
    df['race_part'] = df['race_id'].apply(netkeiba_to_jrdb_internal)
    df = df.drop(columns=['race_id'])
    df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce').astype('Int64').astype(str)
    df = df[df['race_part'] != ''].drop_duplicates(subset=['race_part','umaban'], keep='last')
    return df


# ===== SRB 取込 =====

def load_srb(path: str = 'data/jrdb_srb.csv') -> pd.DataFrame:
    """race-level 集計を race_id ごとに 1 行に."""
    df = pd.read_csv(os.path.join(BASE, path), dtype={'race_id': str})

    out = pd.DataFrame({'race_id': df['race_id']})

    # corner3/4_order — '01-02-03' のような string、平均位置 を抽出
    def parse_avg(s):
        try:
            parts = [int(x) for x in str(s).split('-') if x.strip().isdigit()]
            return float(np.mean(parts)) if parts else np.nan
        except Exception:
            return np.nan
    out['srb_corner3_order_avg'] = df['corner3_order'].apply(parse_avg) if 'corner3_order' in df.columns else np.nan
    out['srb_corner4_order_avg'] = df['corner4_order'].apply(parse_avg) if 'corner4_order' in df.columns else np.nan

    out['srb_pace_up_pos'] = pd.to_numeric(df.get('pace_up_pos', np.nan), errors='coerce')
    out['srb_bias_3corner'] = pd.to_numeric(df.get('bias_3corner', np.nan), errors='coerce')
    out['srb_bias_4corner'] = pd.to_numeric(df.get('bias_4corner', np.nan), errors='coerce')
    out['srb_bias_straight'] = pd.to_numeric(df.get('bias_straight', np.nan), errors='coerce')

    out['srb_race_comment_len'] = df['race_comment'].astype(str).str.len() if 'race_comment' in df.columns else 0

    # furlong_times '1F-2F-3F-...' → セグメント数 (race の 距離 / pace 推定の代替)
    def furlong_n(s):
        try:
            return len([x for x in str(s).split('-') if x.strip()])
        except Exception:
            return 0
    out['srb_furlong_times_n'] = df['furlong_times'].apply(furlong_n) if 'furlong_times' in df.columns else 0

    out['race_part'] = out['race_id'].apply(netkeiba_to_jrdb_internal)
    out = out.drop(columns=['race_id'])
    out = out[out['race_part'] != ''].drop_duplicates(subset=['race_part'], keep='last')
    return out


# ===== merge =====

def merge_v15_1_features(df: pd.DataFrame,
                          kka_path: str = 'data/jrdb_kka.csv',
                          skb_path: str = 'data/jrdb_skb.csv',
                          srb_path: str = 'data/jrdb_srb.csv') -> pd.DataFrame:
    """V15 学習データに KKA/SKB/SR features を merge."""
    df_local = df.copy()
    df_local['race_id'] = df_local['race_id'].astype(str)
    # cache race_id は jrdb-internal 8-char + umaban 2-char zero-padded suffix
    df_local['race_part'] = df_local['race_id'].str[:-2]
    if 'umaban' in df_local.columns:
        df_local['umaban'] = pd.to_numeric(df_local['umaban'], errors='coerce').astype('Int64').astype(str)

    kka = load_kka(kka_path)
    skb = load_skb(skb_path)
    srb = load_srb(srb_path)

    n0 = len(df_local)

    df_local = df_local.merge(kka, on=['race_part', 'umaban'], how='left')
    df_local = df_local.merge(skb, on=['race_part', 'umaban'], how='left')
    df_local = df_local.merge(srb, on='race_part', how='left')

    assert len(df_local) == n0, f"merge changed row count {n0} → {len(df_local)}"
    df_local = df_local.drop(columns=['race_part'])

    # numeric 化 + missing → 0 (V15 と同じ方針)
    for col in V15_1_NEW_FEATURES:
        if col in df_local.columns:
            df_local[col] = pd.to_numeric(df_local[col], errors='coerce').fillna(0)
        else:
            df_local[col] = 0

    return df_local


def coverage_report(df_merged: pd.DataFrame) -> dict:
    """各新 feature の coverage (=非0行率) を返す."""
    out = {}
    for f in V15_1_NEW_FEATURES:
        if f in df_merged.columns:
            non_zero = (df_merged[f] != 0).sum()
            out[f] = float(non_zero / len(df_merged))
        else:
            out[f] = 0.0
    return out


if __name__ == '__main__':
    # smoke test
    print('V15.1 NEW FEATURES:')
    print(f'  KKA: {len(V15_1_KKA_FEATURES)}')
    print(f'  SKB: {len(V15_1_SKB_FEATURES)}')
    print(f'  SRB: {len(V15_1_SRB_FEATURES)}')
    print(f'  total: {len(V15_1_NEW_FEATURES)}')
    print()
    # try loading each
    for fn, path in [(load_kka, 'data/jrdb_kka.csv'), (load_skb, 'data/jrdb_skb.csv'), (load_srb, 'data/jrdb_srb.csv')]:
        try:
            df = fn(path)
            print(f'{path}: shape={df.shape}, cols={list(df.columns)[:5]}...')
        except Exception as e:
            print(f'{path}: ERROR {e}')
