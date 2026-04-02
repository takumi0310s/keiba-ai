#!/usr/bin/env python3
"""JRDB特徴量統合モジュール

predict_core.py の build_features() に追加するJRDB特徴量の
生成・マージロジック。

マージキー: jra_race_id(10桁) + 馬番 or nk_race_id(12桁) + 馬番
欠損時はデフォルト値（全体平均 or 中央値）を使用。

使い方（将来的にpredict_core.pyに統合）:
    from tools.jrdb_features import merge_jrdb_features
    df = merge_jrdb_features(df, race_id, horse_nums)

学習時の使い方（train_v13_jrdb.pyから呼出）:
    from tools.jrdb_features import merge_jrdb_train_features
    df = merge_jrdb_train_features(df)  # jra_race_id + umabanで結合
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# =====================================================
# JRDB特徴量候補リスト
# =====================================================

# Pattern A（前日データ: KYI）— リークフリー
JRDB_FEATURES_PRE_RACE = [
    'jrdb_idm',                # IDM（Index Memory、主力指数）
    'jrdb_training_idx',       # 調教指数
    'jrdb_stable_idx',         # 厩舎指数
    'jrdb_info_idx',           # 情報指数
    'jrdb_composite_idx',      # 総合指数
    'jrdb_upset_idx',          # 激走指数
    'jrdb_ten_idx_pred',       # テン指数予想（展開予想）
    'jrdb_pace_idx_pred',      # ペース指数予想
    'jrdb_agari_idx_pred',     # 上がり指数予想
    'jrdb_position_idx_pred',  # 位置指数予想
    'jrdb_class_code',         # クラスコード
    'jrdb_rise_code',          # 上昇度(1:AA 2:A 3:B 4:C)
    'jrdb_heavy_apt',          # 重馬場適性(1:◎ 2:○ 3:△)
    'jrdb_hoof_code',          # 蹄コード
    'jrdb_ranch_rank',         # 外厩ランク(A=1..E=5, 0=不明)
    'jrdb_stable_rank',        # 厩舎ランク(1-9)
    'jrdb_training_arrow',     # 調教矢印(1:抜群..5:落ち)
    'jrdb_stable_eval',        # 厩舎評価(1:超強気..4:弱気)
    'jrdb_running_style',      # 脚質(1:逃..6:自在)
    'jrdb_dist_apt',           # 距離適性(1:短 2:中 3:長 5:マイル 6:万能)
]

# Pattern B（直前データ: TYB）— 当日情報（リーク扱い for Pattern A）
JRDB_FEATURES_LIVE = [
    'jrdb_paddock_idx',        # パドック指数
    'jrdb_odds_idx',           # オッズ指数
    'jrdb_live_composite_idx', # 直前総合指数
    'jrdb_body_code',          # 馬体コード(1:太..7:緩)
    'jrdb_demeanor_code',      # 気配コード(1:良..8:イレチ)
]

# 前走成績特徴量（SED）— 前走データなのでリークなし
JRDB_FEATURES_PREV_RACE = [
    'jrdb_prev_idm',           # 前走確定IDM
    'jrdb_prev_track_bias',    # 前走馬場差
    'jrdb_prev_interference',  # 前走不利補正
    'jrdb_prev_late_start',    # 前走出遅れ
    'jrdb_prev_ten_idx',       # 前走テン指数（確定）
    'jrdb_prev_agari_idx',     # 前走上がり指数（確定）
    'jrdb_prev_pace_idx',      # 前走ペース指数
    'jrdb_prev_rise_code',     # 前走上昇度
]

# 全特徴量（v13候補）
JRDB_ALL_FEATURES = JRDB_FEATURES_PRE_RACE + JRDB_FEATURES_PREV_RACE
JRDB_LIVE_FEATURES = JRDB_FEATURES_LIVE

# デフォルト値（欠損時）
JRDB_DEFAULTS = {
    'jrdb_idm': 50.0,
    'jrdb_training_idx': 50.0,
    'jrdb_stable_idx': 50.0,
    'jrdb_info_idx': 50.0,
    'jrdb_composite_idx': 50.0,
    'jrdb_upset_idx': 30,
    'jrdb_ten_idx_pred': 50.0,
    'jrdb_pace_idx_pred': 50.0,
    'jrdb_agari_idx_pred': 50.0,
    'jrdb_position_idx_pred': 50.0,
    'jrdb_class_code': 30,
    'jrdb_rise_code': 3,       # B（中間）
    'jrdb_heavy_apt': 2,       # ○普通
    'jrdb_hoof_code': 0,
    'jrdb_ranch_rank': 0,      # 不明
    'jrdb_stable_rank': 5,     # 中間
    'jrdb_training_arrow': 3,  # 平行
    'jrdb_stable_eval': 3,     # 現状維持
    'jrdb_running_style': 0,   # 不明
    'jrdb_dist_apt': 0,        # 不明
    'jrdb_paddock_idx': 50.0,
    'jrdb_odds_idx': 50.0,
    'jrdb_live_composite_idx': 50.0,
    'jrdb_body_code': 4,       # 普通
    'jrdb_demeanor_code': 2,   # 平凡
    'jrdb_prev_idm': 50.0,
    'jrdb_prev_track_bias': 0.0,
    'jrdb_prev_interference': 0.0,
    'jrdb_prev_late_start': 0.0,
    'jrdb_prev_ten_idx': 50.0,
    'jrdb_prev_agari_idx': 50.0,
    'jrdb_prev_pace_idx': 50.0,
    'jrdb_prev_rise_code': 3,
    'jrdb_upset_rank': 0,
    'jrdb_ls_rank': 0,
}


# =====================================================
# KYI → 特徴量変換
# =====================================================

def _ranch_rank_to_num(val):
    """外厩ランク(A-E) → 数値(1-5, 0=不明)"""
    mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
    return mapping.get(str(val).strip().upper(), 0)


def extract_kyi_features(kyi_df):
    """KYI DataFrameからモデル特徴量を抽出

    Args:
        kyi_df: parse_kyi()の出力DataFrame

    Returns:
        DataFrame with columns: jra_race_id, 馬番, + JRDB_FEATURES_PRE_RACE
    """
    if kyi_df is None or len(kyi_df) == 0:
        return pd.DataFrame()

    result = pd.DataFrame()
    result['jra_race_id'] = kyi_df['jra_race_id']
    result['nk_race_id'] = kyi_df['nk_race_id']
    result['馬番'] = pd.to_numeric(kyi_df['馬番'], errors='coerce')

    # 数値指数
    result['jrdb_idm'] = pd.to_numeric(kyi_df['IDM'], errors='coerce')
    result['jrdb_training_idx'] = pd.to_numeric(kyi_df['調教指数'], errors='coerce')
    result['jrdb_stable_idx'] = pd.to_numeric(kyi_df['厩舎指数'], errors='coerce')
    result['jrdb_info_idx'] = pd.to_numeric(kyi_df['情報指数'], errors='coerce')
    result['jrdb_composite_idx'] = pd.to_numeric(kyi_df['総合指数'], errors='coerce')
    result['jrdb_upset_idx'] = pd.to_numeric(kyi_df['激走指数'], errors='coerce')

    # 展開予想指数
    result['jrdb_ten_idx_pred'] = pd.to_numeric(kyi_df['テン指数予想'], errors='coerce')
    result['jrdb_pace_idx_pred'] = pd.to_numeric(kyi_df['ペース指数予想'], errors='coerce')
    result['jrdb_agari_idx_pred'] = pd.to_numeric(kyi_df['上がり指数予想'], errors='coerce')
    result['jrdb_position_idx_pred'] = pd.to_numeric(kyi_df['位置指数予想'], errors='coerce')

    # コード系
    result['jrdb_class_code'] = pd.to_numeric(kyi_df['クラスコード'], errors='coerce')
    result['jrdb_rise_code'] = pd.to_numeric(kyi_df['上昇度'], errors='coerce')
    result['jrdb_heavy_apt'] = pd.to_numeric(kyi_df['重適性コード'], errors='coerce')
    result['jrdb_hoof_code'] = pd.to_numeric(kyi_df['蹄コード'], errors='coerce')
    result['jrdb_ranch_rank'] = kyi_df['放牧先ランク'].apply(_ranch_rank_to_num)
    result['jrdb_stable_rank'] = pd.to_numeric(kyi_df['厩舎ランク'], errors='coerce')
    result['jrdb_training_arrow'] = pd.to_numeric(kyi_df['調教矢印コード'], errors='coerce')
    result['jrdb_stable_eval'] = pd.to_numeric(kyi_df['厩舎評価コード'], errors='coerce')
    result['jrdb_running_style'] = pd.to_numeric(kyi_df['脚質'], errors='coerce')
    result['jrdb_dist_apt'] = pd.to_numeric(kyi_df['距離適性'], errors='coerce')

    # 順位系（DARK HORSE SCAN用）
    if '激走順位' in kyi_df.columns:
        result['jrdb_upset_rank'] = pd.to_numeric(kyi_df['激走順位'], errors='coerce')
    if 'LS指数順位' in kyi_df.columns:
        result['jrdb_ls_rank'] = pd.to_numeric(kyi_df['LS指数順位'], errors='coerce')

    return result


# =====================================================
# TYB → 特徴量変換
# =====================================================

def extract_tyb_features(tyb_df):
    """TYB DataFrameからモデル特徴量を抽出"""
    if tyb_df is None or len(tyb_df) == 0:
        return pd.DataFrame()

    result = pd.DataFrame()
    result['jra_race_id'] = tyb_df['jra_race_id']
    result['nk_race_id'] = tyb_df['nk_race_id']
    result['馬番'] = pd.to_numeric(tyb_df['馬番'], errors='coerce')

    result['jrdb_paddock_idx'] = pd.to_numeric(tyb_df['パドック指数'], errors='coerce')
    result['jrdb_odds_idx'] = pd.to_numeric(tyb_df['オッズ指数'], errors='coerce')
    result['jrdb_live_composite_idx'] = pd.to_numeric(tyb_df['総合指数'], errors='coerce')
    result['jrdb_body_code'] = pd.to_numeric(tyb_df['馬体コード'], errors='coerce')
    result['jrdb_demeanor_code'] = pd.to_numeric(tyb_df['気配コード'], errors='coerce')

    return result


# =====================================================
# SED → 前走特徴量変換
# =====================================================

def extract_sed_prev_features(sed_df, target_race_id=None, target_umaban=None):
    """SED DataFrameから前走特徴量を抽出

    学習時: 全データを処理し、前走レースのSEDデータをマージ
    予測時: 指定レース・馬番の前走データを取得
    """
    if sed_df is None or len(sed_df) == 0:
        return pd.DataFrame()

    # English→Japanese column mapping (handle re-exported CSVs)
    _sed_col_map = {
        'race_id': 'jra_race_id', 'umaban': '馬番',
        'idm': 'IDM', 'baba_sa': '馬場差', 'furi': '不利',
        'deokure': '出遅', 'ten_idx': 'テン指数', 'agari_idx': '上がり指数',
        'pace_idx': 'ペース指数', 'josho_code': '上昇度コード',
        'blood_num': '血統登録番号', 'yyyymmdd': '年月日',
    }
    for eng, jpn in _sed_col_map.items():
        if eng in sed_df.columns and jpn not in sed_df.columns:
            sed_df = sed_df.rename(columns={eng: jpn})

    # Ensure required columns exist
    rid_col = 'jra_race_id' if 'jra_race_id' in sed_df.columns else 'race_id'
    nk_col = 'nk_race_id' if 'nk_race_id' in sed_df.columns else rid_col

    result = pd.DataFrame()
    result['jra_race_id'] = sed_df[rid_col]
    result['nk_race_id'] = sed_df[nk_col]
    result['馬番'] = pd.to_numeric(sed_df['馬番'], errors='coerce')

    # 確定指数を前走特徴量として使用（prev_としてリネーム）
    result['jrdb_prev_idm'] = pd.to_numeric(sed_df['IDM'], errors='coerce')
    result['jrdb_prev_track_bias'] = pd.to_numeric(sed_df['馬場差'], errors='coerce')
    result['jrdb_prev_interference'] = pd.to_numeric(sed_df['不利'], errors='coerce')
    result['jrdb_prev_late_start'] = pd.to_numeric(sed_df['出遅'], errors='coerce')
    result['jrdb_prev_ten_idx'] = pd.to_numeric(sed_df['テン指数'], errors='coerce')
    result['jrdb_prev_agari_idx'] = pd.to_numeric(sed_df['上がり指数'], errors='coerce')
    result['jrdb_prev_pace_idx'] = pd.to_numeric(sed_df['ペース指数'], errors='coerce')
    result['jrdb_prev_rise_code'] = pd.to_numeric(sed_df['上昇度コード'], errors='coerce')

    return result


# =====================================================
# 学習用マージ関数
# =====================================================

def _build_nk_race_id_from_jv(df):
    """JVデータからnetkeiba形式race_id(12桁)を構築

    JV race_id: course(2) + year(2) + kai(1) + nichi(1) + race_num(2) + umaban(2) = 10桁
    netkeiba:   '20' + year(2) + course(2) + kai(2) + nichi(2) + race_num(2) = 12桁
    """
    jv_rid = df['race_id'].astype(str).str.zfill(10)
    course_code = jv_rid.str[:2]
    year_2d = jv_rid.str[2:4]
    kai = df['kai'].astype(int).apply(lambda x: f'{x:02d}')
    nichi = df['nichi'].astype(int).apply(lambda x: f'{x:02d}')
    race_num = jv_rid.str[6:8]
    return '20' + year_2d + course_code + kai + nichi + race_num


def merge_jrdb_train_features(df):
    """学習データにJRDB特徴量をマージ

    nk_race_id(12桁) + umaban でjrdb_kyi.csv / jrdb_sed.csvと結合。
    JV race_idはcourse+year+kai+nichi+race_num+umabanの10桁で
    JRDBとフォーマットが異なるため、netkeiba形式に統一して結合する。

    Args:
        df: 学習データ（jra_races_full.csvベース）
            必須列: race_id, umaban, kai, nichi

    Returns:
        df: JRDB特徴量を追加したDataFrame
    """
    print("  Merging JRDB features (train)...")

    # JVデータのnk_race_id構築
    df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    df['_uma'] = df['umaban'].astype(int)

    # --- KYI（前日データ） ---
    kyi_path = os.path.join(DATA_DIR, 'jrdb_kyi.csv')
    if os.path.exists(kyi_path):
        kyi_raw = pd.read_csv(kyi_path, encoding='utf-8-sig', dtype=str)
        kyi_feats = extract_kyi_features(kyi_raw)
        if len(kyi_feats) > 0:
            kyi_feats['_nk_rid'] = kyi_feats['nk_race_id'].astype(str).str.zfill(12)
            kyi_feats['_uma'] = kyi_feats['馬番'].astype(int)

            kyi_cols = ['_nk_rid', '_uma'] + [c for c in kyi_feats.columns
                                               if c.startswith('jrdb_')]
            kyi_dedup = kyi_feats[kyi_cols].drop_duplicates(subset=['_nk_rid', '_uma'], keep='last')

            before = len(df)
            df = df.merge(kyi_dedup, on=['_nk_rid', '_uma'], how='left')
            matched = df[[c for c in df.columns if c.startswith('jrdb_')]].notna().any(axis=1).sum()
            print(f"    KYI: {matched}/{before} matched ({matched/before*100:.1f}%)")
        else:
            print("    KYI: empty features")
    else:
        print(f"    KYI: {kyi_path} not found")

    # --- SED（前走成績データ） ---
    sed_path = os.path.join(DATA_DIR, 'jrdb_sed.csv')
    if os.path.exists(sed_path):
        sed_raw = pd.read_csv(sed_path, encoding='utf-8-sig', dtype=str)
        # Handle English column names from re-exported CSV
        _sed_raw_map = {
            'race_id': 'jra_race_id', 'umaban': '馬番',
            'blood_num': '血統登録番号', 'yyyymmdd': '年月日',
        }
        for eng, jpn in _sed_raw_map.items():
            if eng in sed_raw.columns and jpn not in sed_raw.columns:
                sed_raw = sed_raw.rename(columns={eng: jpn})
        # race_id(12桁nk形式)をnk_race_idとしても保持
        if 'nk_race_id' not in sed_raw.columns and 'jra_race_id' in sed_raw.columns:
            sed_raw['nk_race_id'] = sed_raw['jra_race_id']
        sed_feats = extract_sed_prev_features(sed_raw)
        if len(sed_feats) > 0:
            # 前走データとして結合するため、blood_registration_idでの結合が理想
            # だが現状はrace_id+umabanで直接結合（当該レースのSED = 成績）
            # → 学習時は「前走のSED」をlag特徴量として使う必要がある
            # ここでは簡易版: SEDデータを血統登録番号+年月日でソートし、
            # 各馬の前走データを取得
            _merge_sed_as_prev(df, sed_feats, sed_raw)
        else:
            print("    SED: empty features")
    else:
        print(f"    SED: {sed_path} not found")

    # デフォルト値で欠損を埋める
    for feat, default in JRDB_DEFAULTS.items():
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(default)
        else:
            df[feat] = default

    # 一時列の削除
    df.drop(columns=['_nk_rid', '_uma'], inplace=True, errors='ignore')

    # 有効率表示
    jrdb_cols = [c for c in df.columns if c.startswith('jrdb_')]
    if jrdb_cols:
        valid_rates = {c: (df[c] != JRDB_DEFAULTS.get(c, 0)).mean() for c in jrdb_cols[:5]}
        for c, r in valid_rates.items():
            print(f"    {c}: {r*100:.1f}% non-default")

    return df


def _merge_sed_as_prev(df, sed_feats, sed_raw):
    """SED成績データを前走特徴量としてマージ

    各馬の血統登録番号をキーに、当該レースの直前のSEDレコードを前走データとして結合。
    """
    print("    SED: merging as prev-race features...")

    if '血統登録番号' not in sed_raw.columns or '年月日' not in sed_raw.columns:
        print("    SED: 血統登録番号 or 年月日 column missing")
        return

    # SEDデータを血統登録番号+年月日でソート
    sed_merged = sed_raw.copy()
    sed_merged['_date'] = pd.to_numeric(sed_merged['年月日'], errors='coerce')
    sed_merged['_blood_id'] = sed_merged['血統登録番号'].astype(str).str.strip()

    for feat in sed_feats.columns:
        if feat.startswith('jrdb_prev_'):
            sed_merged[feat] = sed_feats[feat].values

    sed_merged = sed_merged.sort_values(['_blood_id', '_date'])

    # 各馬の直前レースのデータを取得（shift(1)で1行前=前走）
    prev_cols = [c for c in sed_merged.columns if c.startswith('jrdb_prev_')]
    sed_merged[prev_cols] = sed_merged.groupby('_blood_id')[prev_cols].shift(1)

    # nk_race_id + 馬番で元dfにマージ
    sed_merged['_nk_rid'] = sed_merged['nk_race_id'].astype(str).str.zfill(12) if 'nk_race_id' in sed_merged.columns else ''
    sed_merged['_uma'] = pd.to_numeric(sed_merged['馬番'], errors='coerce')

    # df側のnk_race_idは既に_build_nk_race_id_from_jvで構築済み
    if '_nk_rid' not in df.columns:
        df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    if '_uma' not in df.columns:
        df['_uma'] = df['umaban'].astype(int)

    merge_cols = ['_nk_rid', '_uma'] + prev_cols
    sed_dedup = sed_merged[merge_cols].drop_duplicates(subset=['_nk_rid', '_uma'], keep='last')

    before_cols = set(df.columns)
    df_merged = df.merge(sed_dedup, on=['_nk_rid', '_uma'], how='left', suffixes=('', '_sed'))

    # マージ結果を元dfに反映
    for c in prev_cols:
        if c in df_merged.columns:
            if c not in before_cols:
                df[c] = df_merged[c].values
            else:
                # 既存値がNaNの行のみ更新
                mask = df[c].isna() & df_merged[c].notna()
                df.loc[mask, c] = df_merged.loc[mask, c].values

    matched = df[prev_cols[0]].notna().sum() if prev_cols else 0
    print(f"    SED prev: {matched}/{len(df)} matched ({matched/len(df)*100:.1f}%)")


# =====================================================
# 予測用マージ関数（predict_core.py統合用）
# =====================================================

def merge_jrdb_predict_features(horses_df, race_id_nk):
    """予測時にJRDB特徴量をマージ

    CSVファイルから該当レースのKYI/TYBデータを取得してマージ。
    CSVにない場合はデフォルト値を使用。

    Args:
        horses_df: 予測対象馬のDataFrame（馬番列が必要）
        race_id_nk: netkeiba形式のrace_id(12桁)

    Returns:
        horses_df: JRDB特徴量を追加したDataFrame
    """
    # KYIからの特徴量
    kyi_path = os.path.join(DATA_DIR, 'jrdb_kyi.csv')
    if os.path.exists(kyi_path):
        try:
            kyi = pd.read_csv(kyi_path, encoding='utf-8-sig', dtype=str)
            kyi_race = kyi[kyi['nk_race_id'].astype(str) == str(race_id_nk)]
            if len(kyi_race) > 0:
                kyi_feats = extract_kyi_features(kyi_race)
                kyi_feats['_uma'] = kyi_feats['馬番'].astype(int)
                horses_df['_uma'] = horses_df['horse_num'].astype(int) if 'horse_num' in horses_df.columns else horses_df.index + 1

                jrdb_cols = [c for c in kyi_feats.columns if c.startswith('jrdb_')]
                kyi_merge = kyi_feats[['_uma'] + jrdb_cols].drop_duplicates(subset='_uma', keep='last')
                horses_df = horses_df.merge(kyi_merge, on='_uma', how='left')
                horses_df.drop(columns=['_uma'], inplace=True, errors='ignore')
        except Exception as e:
            print(f"[WARN] JRDB KYI merge failed: {e}")

    # TYBからの特徴量
    tyb_path = os.path.join(DATA_DIR, 'jrdb_tyb.csv')
    if os.path.exists(tyb_path):
        try:
            tyb = pd.read_csv(tyb_path, encoding='utf-8-sig', dtype=str)
            tyb_race = tyb[tyb['nk_race_id'].astype(str) == str(race_id_nk)]
            if len(tyb_race) > 0:
                tyb_feats = extract_tyb_features(tyb_race)
                tyb_feats['_uma'] = tyb_feats['馬番'].astype(int)
                if '_uma' not in horses_df.columns:
                    horses_df['_uma'] = horses_df['horse_num'].astype(int) if 'horse_num' in horses_df.columns else horses_df.index + 1

                jrdb_cols = [c for c in tyb_feats.columns if c.startswith('jrdb_')]
                tyb_merge = tyb_feats[['_uma'] + jrdb_cols].drop_duplicates(subset='_uma', keep='last')
                horses_df = horses_df.merge(tyb_merge, on='_uma', how='left')
                horses_df.drop(columns=['_uma'], inplace=True, errors='ignore')
        except Exception as e:
            print(f"[WARN] JRDB TYB merge failed: {e}")

    # デフォルト値で埋め
    for feat, default in JRDB_DEFAULTS.items():
        if feat in horses_df.columns:
            horses_df[feat] = pd.to_numeric(horses_df[feat], errors='coerce').fillna(default)
        else:
            horses_df[feat] = default

    return horses_df


# =====================================================
# ユーティリティ
# =====================================================

def get_jrdb_coverage():
    """JRDBデータのカバレッジを確認"""
    info = {}
    for ft, path in [('KYI', os.path.join(DATA_DIR, 'jrdb_kyi.csv')),
                      ('TYB', os.path.join(DATA_DIR, 'jrdb_tyb.csv')),
                      ('SED', os.path.join(DATA_DIR, 'jrdb_sed.csv'))]:
        if os.path.exists(path):
            df = pd.read_csv(path, encoding='utf-8-sig', dtype=str, nrows=0)
            n = sum(1 for _ in open(path, encoding='utf-8-sig')) - 1
            info[ft] = {'path': path, 'rows': n, 'columns': list(df.columns)}
        else:
            info[ft] = {'path': path, 'rows': 0, 'columns': []}
    return info


if __name__ == '__main__':
    # カバレッジ確認
    print("JRDB データカバレッジ:")
    for ft, info in get_jrdb_coverage().items():
        print(f"  {ft}: {info['rows']} rows")
        if info['columns']:
            print(f"    Columns: {info['columns'][:10]}...")

    print(f"\n特徴量候補:")
    print(f"  Pattern A (前日): {len(JRDB_FEATURES_PRE_RACE)} features")
    print(f"  前走成績: {len(JRDB_FEATURES_PREV_RACE)} features")
    print(f"  Pattern B (直前): {len(JRDB_FEATURES_LIVE)} features")
    print(f"  合計: {len(JRDB_ALL_FEATURES) + len(JRDB_LIVE_FEATURES)} features")
