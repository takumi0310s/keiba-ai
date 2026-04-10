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
    'jrdb_entry_days_ago',     # 入厩何日前(レース日から遡った日数)
    'jrdb_entry_race_num',     # 入厩何走目
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
    'jrdb_entry_days_ago': 0,  # 入厩情報なし
    'jrdb_entry_race_num': 0,  # 入厩情報なし
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

# PACI Tier A デフォルト値（merge_jrdb_train_featuresとは別管理）
PACI_TIER_A_DEFAULTS = {
    'paci_manken_idx': 36.0,       # 万券指数（穴馬評価）
    'paci_goal_rank': 8.0,         # ゴール順位予想
    'paci_dochu_rank': 8.0,        # 道中順位予想
    'paci_goal_diff': 12.0,        # ゴール差予想
    'paci_jockey_exp_wr': 14.5,    # 騎手期待勝率
    'paci_jockey_exp_3rd': 21.9,   # 騎手期待3着率
    'paci_ninki_idx': 159.0,       # 人気指数
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

    # 入厩情報
    if '入厩何日前' in kyi_df.columns:
        result['jrdb_entry_days_ago'] = pd.to_numeric(kyi_df['入厩何日前'], errors='coerce')
    if '入厩何走目' in kyi_df.columns:
        result['jrdb_entry_race_num'] = pd.to_numeric(kyi_df['入厩何走目'], errors='coerce')

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

    # English→Japanese column mapping (handle re-exported CSVs)
    _tyb_col_map = {
        'race_id': 'nk_race_id',
        'umaban': '馬番',
        'padock_idx': 'パドック指数',
        'odds_idx': 'オッズ指数',
        'sogo_idx': '総合指数',
        'batai_code': '馬体コード',
        'kehai_code': '気配コード',
    }
    for eng, jpn in _tyb_col_map.items():
        if eng in tyb_df.columns and jpn not in tyb_df.columns:
            tyb_df = tyb_df.rename(columns={eng: jpn})

    result = pd.DataFrame()
    if 'jra_race_id' in tyb_df.columns:
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
    _kyi_matched = False
    _kyi_fallback_blood_map = None  # horse name → blood_num (for ZE/SED fallback)
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
                _kyi_matched = True
            else:
                # KYI当日データなし → 馬名ベースで過去KYIから最新データを取得
                _name_col = '馬名' if '馬名' in horses_df.columns else None
                _uma_col_h = 'horse_num' if 'horse_num' in horses_df.columns else '馬番'
                if _name_col and '馬名' in kyi.columns:
                    horses_df['_uma'] = horses_df[_uma_col_h].astype(int)
                    _fb_rows = []
                    _fb_blood = {}  # _uma → blood_num
                    for _, _h in horses_df.iterrows():
                        _hname = str(_h.get(_name_col, ''))
                        _uma_val = int(_h['_uma'])
                        _past = kyi[kyi['馬名'] == _hname].sort_values('nk_race_id', ascending=False)
                        if len(_past) > 0:
                            _latest = _past.iloc[0:1]
                            _feats = extract_kyi_features(_latest)
                            _feats['_uma'] = _uma_val
                            _fb_rows.append(_feats)
                            _bn = str(_past.iloc[0].get('血統登録番号', ''))
                            if _bn:
                                _fb_blood[_uma_val] = _bn
                    if _fb_rows:
                        _fb_df = pd.concat(_fb_rows, ignore_index=True)
                        jrdb_cols = [c for c in _fb_df.columns if c.startswith('jrdb_')]
                        _fb_merge = _fb_df[['_uma'] + jrdb_cols].drop_duplicates(subset='_uma', keep='last')
                        horses_df = horses_df.merge(_fb_merge, on='_uma', how='left')
                        _kyi_matched = True
                        print(f"[JRDB] KYI当日データなし → 馬名フォールバックで{len(_fb_rows)}/{len(horses_df)}馬取得")
                    if _fb_blood:
                        _kyi_fallback_blood_map = pd.DataFrame([
                            {'_uma_str': str(u), 'blood_num': b} for u, b in _fb_blood.items()
                        ])
                    horses_df.drop(columns=['_uma'], inplace=True, errors='ignore')
        except Exception as e:
            print(f"[WARN] JRDB KYI merge failed: {e}")

    # TYBからの特徴量
    tyb_path = os.path.join(DATA_DIR, 'jrdb_tyb.csv')
    if os.path.exists(tyb_path):
        try:
            tyb = pd.read_csv(tyb_path, encoding='utf-8-sig', dtype=str)
            # English→Japanese column mapping for TYB CSV
            if 'race_id' in tyb.columns and 'nk_race_id' not in tyb.columns:
                tyb = tyb.rename(columns={'race_id': 'nk_race_id'})
            if 'umaban' in tyb.columns and '馬番' not in tyb.columns:
                tyb = tyb.rename(columns={'umaban': '馬番'})
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

    # 拡張JRDB特徴量（CHA/JO/KTA/ZE/SR/KKA）
    # _uma列を準備
    if '_uma' not in horses_df.columns:
        horses_df['_uma'] = horses_df['horse_num'].astype(int) if 'horse_num' in horses_df.columns else horses_df.index + 1

    _rid_str = str(race_id_nk).zfill(12)

    # CHA: 追切指数
    _cha_path = os.path.join(DATA_DIR, 'jrdb_cha.csv')
    if os.path.exists(_cha_path):
        try:
            _cha = pd.read_csv(_cha_path, encoding='utf-8-sig', dtype=str)
            _cha_race = _cha[_cha['race_id'].astype(str).str.zfill(12) == _rid_str]
            if len(_cha_race) > 0:
                _cr = pd.DataFrame()
                _cr['_uma'] = pd.to_numeric(_cha_race['umaban'], errors='coerce')
                _cr['jrdb_oikiri_idx'] = pd.to_numeric(_cha_race['oikiri_idx'], errors='coerce')
                _cr['jrdb_ten_time_idx'] = pd.to_numeric(_cha_race['ten_time_idx'], errors='coerce')
                _cr['jrdb_shimai_time_idx'] = pd.to_numeric(_cha_race['shimai_time_idx'], errors='coerce')
                _cr = _cr.drop_duplicates(subset='_uma', keep='last')
                horses_df = horses_df.merge(_cr, on='_uma', how='left', suffixes=('', '_cha'))
        except Exception as e:
            print(f"[WARN] JRDB CHA merge failed: {e}")

    # JO: CID/LS指数（当日データ優先、なければ馬名フォールバック）
    _jo_path = os.path.join(DATA_DIR, 'jrdb_jo.csv')
    if os.path.exists(_jo_path):
        try:
            _jo = pd.read_csv(_jo_path, encoding='utf-8-sig', dtype=str)
            _jo_race = _jo[_jo['race_id'].astype(str).str.zfill(12) == _rid_str]
            if len(_jo_race) > 0:
                _jr = pd.DataFrame()
                _jr['_uma'] = pd.to_numeric(_jo_race['umaban'], errors='coerce')
                _jr['jrdb_cid_idx'] = pd.to_numeric(_jo_race['cid_idx'], errors='coerce')
                _jr['jrdb_ls_idx'] = pd.to_numeric(_jo_race['ls_idx'], errors='coerce')
                _jr = _jr.drop_duplicates(subset='_uma', keep='last')
                horses_df = horses_df.merge(_jr, on='_uma', how='left', suffixes=('', '_jo'))
            elif 'horse_name' in _jo.columns or '馬名' in _jo.columns:
                # 当日データなし → 馬名から過去JO最新値を取得
                _hn_col_jo = 'horse_name' if 'horse_name' in _jo.columns else '馬名'
                _name_col_h = '馬名' if '馬名' in horses_df.columns else None
                _uma_col_h = 'horse_num' if 'horse_num' in horses_df.columns else '馬番'
                if _name_col_h:
                    _jo_fb_rows = []
                    for _, _h in horses_df.iterrows():
                        _hname = str(_h.get(_name_col_h, ''))
                        _uma_val = int(_h[_uma_col_h])
                        _jo_past = _jo[_jo[_hn_col_jo] == _hname].sort_values('race_id', ascending=False)
                        if len(_jo_past) > 0:
                            _r = _jo_past.iloc[0]
                            _cid = pd.to_numeric(_r.get('cid_idx', 0), errors='coerce')
                            _ls = pd.to_numeric(_r.get('ls_idx', 0), errors='coerce')
                            _jo_fb_rows.append({'_uma': _uma_val, 'jrdb_cid_idx': _cid, 'jrdb_ls_idx': _ls})
                    if _jo_fb_rows:
                        _jo_fb = pd.DataFrame(_jo_fb_rows).drop_duplicates(subset='_uma', keep='last')
                        horses_df = horses_df.merge(_jo_fb, on='_uma', how='left', suffixes=('', '_jo'))
                        _n_jo = (_jo_fb['jrdb_cid_idx'].notna() & (_jo_fb['jrdb_cid_idx'] != 0)).sum()
                        if _n_jo > 0:
                            print(f"[JRDB] JO当日データなし → 馬名フォールバックで{_n_jo}馬取得")
        except Exception as e:
            print(f"[WARN] JRDB JO merge failed: {e}")

    # blood_num → 馬番マッピング（KYIから取得、PACIフォールバック、馬名フォールバック）
    _blood_map = None
    _kyi_path = os.path.join(DATA_DIR, 'jrdb_kyi.csv')
    _paci_path = os.path.join(DATA_DIR, 'jrdb_paci.csv')
    for _bp in [_kyi_path, _paci_path]:
        if _blood_map is not None:
            break
        if os.path.exists(_bp):
            try:
                _uma_col = '馬番' if _bp == _kyi_path else 'umaban'
                _rid_col = 'nk_race_id' if _bp == _kyi_path else 'race_id'
                _bn_col = '血統登録番号' if _bp == _kyi_path else 'blood_num'
                _bdf = pd.read_csv(_bp, encoding='utf-8-sig', dtype=str,
                                   usecols=[_rid_col, _uma_col, _bn_col])
                _bdf_race = _bdf[_bdf[_rid_col].astype(str).str.zfill(12) == _rid_str]
                if len(_bdf_race) > 0:
                    _blood_map = _bdf_race[[_uma_col, _bn_col]].drop_duplicates(subset=_bn_col, keep='last')
                    _blood_map = _blood_map.rename(columns={_uma_col: '_uma_str', _bn_col: 'blood_num'})
            except Exception:
                pass
    # KYI馬名フォールバックからのblood_map
    if _blood_map is None and _kyi_fallback_blood_map is not None:
        _blood_map = _kyi_fallback_blood_map
        print(f"[JRDB] blood_map: 馬名フォールバックから{len(_blood_map)}馬取得")

    # KTA: IDM予想/展開予想（blood_num経由、馬名フォールバック付き）
    _kta_path = os.path.join(DATA_DIR, 'jrdb_kta.csv')
    if os.path.exists(_kta_path):
        try:
            _kta = pd.read_csv(_kta_path, encoding='utf-8-sig', dtype=str)
            _kta_race = _kta[_kta['race_id'].astype(str).str.zfill(12) == _rid_str]
            if len(_kta_race) > 0:
                if _blood_map is not None:
                    _kta_m = _kta_race.merge(_blood_map, on='blood_num', how='left')
                    _kr = pd.DataFrame()
                    _kr['_uma'] = pd.to_numeric(_kta_m['_uma_str'], errors='coerce')
                elif 'horse_name' in _kta_race.columns and '馬名' in horses_df.columns:
                    # blood_mapなし → horse_nameでmerge
                    _uma_col_h = 'horse_num' if 'horse_num' in horses_df.columns else '馬番'
                    _name_map = horses_df[['馬名', _uma_col_h]].copy()
                    _name_map['_uma'] = _name_map[_uma_col_h].astype(int)
                    _kta_m = _kta_race.merge(_name_map[['馬名', '_uma']], left_on='horse_name', right_on='馬名', how='inner')
                    _kr = pd.DataFrame()
                    _kr['_uma'] = _kta_m['_uma']
                else:
                    _kta_m = pd.DataFrame()
                    _kr = pd.DataFrame()
                if len(_kta_m) > 0:
                    _kr['jrdb_kta_idm'] = pd.to_numeric(_kta_m['idm'], errors='coerce').values
                    _kr['jrdb_kta_ten_pred'] = pd.to_numeric(_kta_m['ten_idx_pred'], errors='coerce').values
                    _kr['jrdb_kta_agari_pred'] = pd.to_numeric(_kta_m['agari_idx_pred'], errors='coerce').values
                    _kr = _kr.dropna(subset=['_uma']).drop_duplicates(subset='_uma', keep='last')
                    # IDMが全0の場合、過去KTAの最新非ゼロIDMで補完
                    if (_kr['jrdb_kta_idm'].fillna(0) == 0).all() and 'horse_name' in _kta.columns:
                        _name_col_h = '馬名' if '馬名' in horses_df.columns else None
                        _uma_col_h = 'horse_num' if 'horse_num' in horses_df.columns else '馬番'
                        if _name_col_h:
                            for _ki in range(len(_kr)):
                                _uma_v = int(_kr.iloc[_ki]['_uma'])
                                _hmask = horses_df[_uma_col_h].astype(int) == _uma_v
                                if _hmask.any():
                                    _hname = str(horses_df.loc[_hmask, _name_col_h].iloc[0])
                                    _past = _kta[_kta['horse_name'] == _hname].sort_values('race_id', ascending=False)
                                    _past_idm = pd.to_numeric(_past['idm'], errors='coerce')
                                    _nz = _past_idm[_past_idm > 0]
                                    if len(_nz) > 0:
                                        _kr.iloc[_ki, _kr.columns.get_loc('jrdb_kta_idm')] = float(_nz.iloc[0])
                    horses_df = horses_df.merge(_kr, on='_uma', how='left', suffixes=('', '_kta'))
        except Exception as e:
            print(f"[WARN] JRDB KTA merge failed: {e}")

    # ZE: 過去5走集計（blood_num経由）
    _ze_path = os.path.join(DATA_DIR, 'jrdb_ze.csv')
    if os.path.exists(_ze_path) and _blood_map is not None:
        try:
            _blood_nums = _blood_map['blood_num'].unique().tolist()
            _ze = pd.read_csv(_ze_path, encoding='utf-8-sig', dtype=str)
            _ze_filt = _ze[_ze['blood_num'].isin(_blood_nums)]
            if len(_ze_filt) > 0:
                _ze_filt = _ze_filt.copy()
                _ze_filt['_idm'] = pd.to_numeric(_ze_filt['idm'], errors='coerce')
                _ze_filt['_ten'] = pd.to_numeric(_ze_filt['ten_idx'], errors='coerce')
                _ze_filt['_agari'] = pd.to_numeric(_ze_filt['agari_idx'], errors='coerce')
                _ze_filt['_furi'] = (pd.to_numeric(_ze_filt['furi'], errors='coerce').fillna(0) > 0).astype(int)
                _agg = _ze_filt.groupby('blood_num').agg(
                    jrdb_ze_idm_avg=('_idm', 'mean'),
                    jrdb_ze_ten_avg=('_ten', 'mean'),
                    jrdb_ze_agari_avg=('_agari', 'mean'),
                    jrdb_ze_furi_count=('_furi', 'sum'),
                ).reset_index()
                _bm_ze = _blood_map.merge(_agg, on='blood_num', how='left')
                _zr = pd.DataFrame()
                _zr['_uma'] = pd.to_numeric(_bm_ze['_uma_str'], errors='coerce')
                for c in ['jrdb_ze_idm_avg', 'jrdb_ze_ten_avg', 'jrdb_ze_agari_avg', 'jrdb_ze_furi_count']:
                    _zr[c] = _bm_ze[c].values
                _zr = _zr.dropna(subset=['_uma']).drop_duplicates(subset='_uma', keep='last')
                horses_df = horses_df.merge(_zr, on='_uma', how='left', suffixes=('', '_ze'))
        except Exception as e:
            print(f"[WARN] JRDB ZE merge failed: {e}")

    # SED: 前走データ（馬場差/不利/出遅）— blood_num経由で最新行を取得
    _sed_path = os.path.join(DATA_DIR, 'jrdb_sed.csv')
    _sed_feats_needed = ['jrdb_prev_track_bias', 'jrdb_prev_interference', 'jrdb_prev_late_start']
    _sed_feats_zero = all(
        c not in horses_df.columns or (horses_df.get(c, pd.Series([0])) == 0).all()
        for c in _sed_feats_needed
    )
    if os.path.exists(_sed_path) and _blood_map is not None and _sed_feats_zero:
        try:
            _blood_nums = _blood_map['blood_num'].unique().tolist()
            _sed_cols = ['blood_num', 'race_id', '馬���差', '不利', '出遅']
            # English column fallback
            _sed_raw = pd.read_csv(_sed_path, encoding='utf-8-sig', dtype=str)
            _sed_col_map = {}
            for _jc, _ec in [('馬場差', 'baba_sa'), ('不利', 'furi'), ('出遅', 'deokure')]:
                if _jc in _sed_raw.columns:
                    _sed_col_map[_jc] = _jc
                elif _ec in _sed_raw.columns:
                    _sed_col_map[_jc] = _ec
            if 'blood_num' in _sed_raw.columns and len(_sed_col_map) == 3:
                _sed_filt = _sed_raw[_sed_raw['blood_num'].isin(_blood_nums)].copy()
                if len(_sed_filt) > 0:
                    # 各馬の最新SED行を取得（= 前走データ）
                    _sed_filt = _sed_filt.sort_values('race_id', ascending=False)
                    _sed_latest = _sed_filt.drop_duplicates(subset='blood_num', keep='first')
                    _sr_df = _sed_latest.merge(_blood_map, on='blood_num', how='inner')
                    _sr_result = pd.DataFrame()
                    _sr_result['_uma'] = pd.to_numeric(_sr_df['_uma_str'], errors='coerce')
                    _sr_result['jrdb_prev_track_bias'] = pd.to_numeric(_sr_df[_sed_col_map['馬場差']], errors='coerce')
                    _sr_result['jrdb_prev_interference'] = pd.to_numeric(_sr_df[_sed_col_map['不利']], errors='coerce')
                    _sr_result['jrdb_prev_late_start'] = pd.to_numeric(_sr_df[_sed_col_map['出遅']], errors='coerce')
                    _sr_result = _sr_result.dropna(subset=['_uma']).drop_duplicates(subset='_uma', keep='last')
                    if '_uma' not in horses_df.columns:
                        horses_df['_uma'] = horses_df['horse_num'].astype(int) if 'horse_num' in horses_df.columns else horses_df.index + 1
                    horses_df = horses_df.merge(_sr_result, on='_uma', how='left', suffixes=('', '_sed_fb'))
                    # Prefer new values over existing zeros
                    for _sc in _sed_feats_needed:
                        _sc_fb = f'{_sc}_sed_fb'
                        if _sc_fb in horses_df.columns:
                            horses_df[_sc] = horses_df[_sc].fillna(0)
                            _mask = horses_df[_sc] == 0
                            horses_df.loc[_mask, _sc] = horses_df.loc[_mask, _sc_fb]
                            horses_df.drop(columns=[_sc_fb], inplace=True, errors='ignore')
                    _n_filled = sum(1 for c in _sed_feats_needed if c in horses_df.columns and (horses_df[c] != 0).any())
                    if _n_filled > 0:
                        print(f"[JRDB] SED前走データ: blood_numフォールバックで{_n_filled}/3特徴量取得")
        except Exception as e:
            print(f"[WARN] JRDB SED prev fallback failed: {e}")

    # SR: トラックバイアス（当該レースのバイアス）
    _sr_path = os.path.join(DATA_DIR, 'jrdb_sr.csv')
    if os.path.exists(_sr_path):
        try:
            _sr = pd.read_csv(_sr_path, encoding='utf-8-sig', dtype=str)
            _sr_race = _sr[_sr['race_id'].astype(str).str.zfill(12) == _rid_str]
            if len(_sr_race) > 0:
                _sr_row = _sr_race.iloc[-1]
                _tb = str(_sr_row.get('tb_homestr', ''))
                _inner = int(_tb[0]) if _tb and len(_tb) >= 1 and _tb[0].isdigit() else 2
                horses_df['jrdb_tb_homestr_inner'] = _inner
        except Exception as e:
            print(f"[WARN] JRDB SR merge failed: {e}")

    # SKB: パドック観察データ（構造化コード）
    _skb_path = os.path.join(DATA_DIR, 'jrdb_skb.csv')
    if os.path.exists(_skb_path):
        try:
            _skb = pd.read_csv(_skb_path, encoding='utf-8-sig', dtype=str)
            _skb_race = _skb[_skb['race_id'].astype(str).str.zfill(12) == _rid_str]
            if len(_skb_race) > 0:
                _skr = pd.DataFrame()
                _skr['_uma'] = pd.to_numeric(_skb_race['umaban'], errors='coerce')
                _skr['jrdb_heavy_apt_skb'] = pd.to_numeric(_skb_race.get('heavy_apt', 0), errors='coerce')
                _skr['jrdb_anshin'] = pd.to_numeric(_skb_race.get('anshin', 0), errors='coerce')
                _skr['jrdb_run_stage'] = pd.to_numeric(_skb_race.get('run_stage', 0), errors='coerce')
                _skr = _skr.drop_duplicates(subset='_uma', keep='last')
                horses_df = horses_df.merge(_skr, on='_uma', how='left', suffixes=('', '_skb'))
        except Exception as e:
            print(f"[WARN] JRDB SKB merge failed: {e}")

    # JOA: 開催条件（馬場バイアス詳細）— レース単位
    _joa_path = os.path.join(DATA_DIR, 'jrdb_joa.csv')
    if os.path.exists(_joa_path):
        try:
            _joa = pd.read_csv(_joa_path, encoding='utf-8-sig', dtype=str)
            _joa_nk = _joa[_joa.get('nk_race_id', pd.Series(dtype=str)).astype(str) == _rid_str]
            if len(_joa_nk) > 0:
                _jr = _joa_nk.iloc[-1]
                horses_df['jrdb_turf_baba_code'] = pd.to_numeric(_jr.get('turf_baba_code', 0), errors='coerce') or 0
                horses_df['jrdb_dirt_baba_code'] = pd.to_numeric(_jr.get('dirt_baba_code', 0), errors='coerce') or 0
        except Exception as e:
            print(f"[WARN] JRDB JOA merge failed: {e}")

    # KKA: 母/BMS連勝指数
    _kka_path = os.path.join(DATA_DIR, 'jrdb_kka.csv')
    if os.path.exists(_kka_path):
        try:
            _kka = pd.read_csv(_kka_path, encoding='utf-8-sig', dtype=str)
            _kka_race = _kka[_kka['race_id'].astype(str).str.zfill(12) == _rid_str]
            if len(_kka_race) > 0:
                _kkr = pd.DataFrame()
                _kkr['_uma'] = pd.to_numeric(_kka_race['umaban'], errors='coerce')
                _kkr['jrdb_dam_rensho_avg'] = pd.to_numeric(_kka_race['dam_rensho_avg'], errors='coerce')
                _kkr['jrdb_bms_rensho_avg'] = pd.to_numeric(_kka_race['bms_rensho_avg'], errors='coerce')
                _kkr = _kkr.drop_duplicates(subset='_uma', keep='last')
                horses_df = horses_df.merge(_kkr, on='_uma', how='left', suffixes=('', '_kka'))
        except Exception as e:
            print(f"[WARN] JRDB KKA merge failed: {e}")

    # PACI: Tier A 展開予想指数（manken/goal_rank/dochu_rank/goal_diff/jockey_exp/ninki）
    _paci_path = os.path.join(DATA_DIR, 'jrdb_paci.csv')
    if os.path.exists(_paci_path):
        try:
            _paci = pd.read_csv(_paci_path, encoding='utf-8-sig', dtype=str)
            _paci_race = _paci[_paci['race_id'].astype(str).str.zfill(12) == _rid_str]
            if len(_paci_race) > 0:
                _pr = pd.DataFrame()
                _pr['_uma'] = pd.to_numeric(_paci_race['umaban'], errors='coerce')
                _pr['paci_manken_idx'] = pd.to_numeric(_paci_race['manken_idx'], errors='coerce')
                _pr['paci_goal_rank'] = pd.to_numeric(_paci_race['goal_rank'], errors='coerce')
                _pr['paci_dochu_rank'] = pd.to_numeric(_paci_race['dochu_rank'], errors='coerce')
                _pr['paci_goal_diff'] = pd.to_numeric(_paci_race['goal_diff'], errors='coerce')
                _pr['paci_jockey_exp_wr'] = pd.to_numeric(_paci_race['jockey_exp_wr'], errors='coerce')
                _pr['paci_jockey_exp_3rd'] = pd.to_numeric(_paci_race['jockey_exp_3rd'], errors='coerce')
                _pr['paci_ninki_idx'] = pd.to_numeric(_paci_race['ninki_idx'], errors='coerce')
                _pr = _pr.drop_duplicates(subset='_uma', keep='last')
                horses_df = horses_df.merge(_pr, on='_uma', how='left', suffixes=('', '_paci'))
        except Exception as e:
            print(f"[WARN] JRDB PACI merge failed: {e}")

    # OZ: 基準オッズ特徴量
    _oz_path = os.path.join(DATA_DIR, 'jrdb_oz.csv')
    if os.path.exists(_oz_path):
        try:
            _oz = pd.read_csv(_oz_path, encoding='utf-8-sig', dtype={'race_id': str})
            _oz_race = _oz[_oz['race_id'].astype(str).str.zfill(12) == _rid_str]
            if len(_oz_race) > 0:
                _oz_row = _oz_race.iloc[0]
                _ozr = pd.DataFrame()
                _uma_list = []
                _base_t_list = []
                _base_f_list = []
                for i in range(1, 19):
                    t_val = pd.to_numeric(_oz_row.get(f'tansho_{i:02d}'), errors='coerce')
                    f_val = pd.to_numeric(_oz_row.get(f'fukusho_{i:02d}'), errors='coerce')
                    if pd.notna(t_val) and t_val > 0:
                        _uma_list.append(i)
                        _base_t_list.append(t_val)
                        _base_f_list.append(f_val if pd.notna(f_val) and f_val > 0 else np.nan)
                if _uma_list:
                    _ozr = pd.DataFrame({
                        '_uma': _uma_list,
                        'oz_tansho_base_log': np.log1p(np.array(_base_t_list).clip(1.0)),
                        'oz_fukusho_base_log': np.log1p(np.array([f if not np.isnan(f) else 2.0 for f in _base_f_list]).clip(1.0)),
                    })
                    # Compute base popularity rank (lower odds = higher rank)
                    _ozr['oz_base_pop_rank'] = pd.Series(_base_t_list).rank(method='min', ascending=True).astype(int).values
                    _ozr = _ozr.drop_duplicates(subset='_uma', keep='last')
                    horses_df = horses_df.merge(_ozr, on='_uma', how='left', suffixes=('', '_oz'))
        except Exception as e:
            print(f"[WARN] JRDB OZ merge failed: {e}")

    # KYI: 基準オッズ特徴量（OZがない場合のフォールバック）
    if 'oz_tansho_base_log' not in horses_df.columns or horses_df['oz_tansho_base_log'].isna().all():
        try:
            # Re-read KYI for 基準オッズ columns
            if os.path.exists(kyi_path):
                _kyi2 = pd.read_csv(kyi_path, encoding='utf-8-sig', dtype=str)
                _kyi2_race = _kyi2[_kyi2['nk_race_id'].astype(str) == str(race_id_nk)]
                if len(_kyi2_race) > 0 and '基準オッズ' in _kyi2_race.columns:
                    _kr2 = pd.DataFrame()
                    _kr2['_uma'] = pd.to_numeric(_kyi2_race['馬番'], errors='coerce')
                    _base_odds = pd.to_numeric(_kyi2_race['基準オッズ'], errors='coerce')
                    _base_fuku = pd.to_numeric(_kyi2_race.get('基準複勝オッズ', pd.Series(dtype=float)), errors='coerce')
                    _base_pop = pd.to_numeric(_kyi2_race.get('基準人気順位', pd.Series(dtype=float)), errors='coerce')
                    _kr2['oz_tansho_base_log'] = np.log1p(_base_odds.clip(lower=1.0).fillna(10.0))
                    _kr2['oz_fukusho_base_log'] = np.log1p(_base_fuku.clip(lower=1.0).fillna(2.0))
                    _kr2['oz_base_pop_rank'] = _base_pop.fillna(8).astype(int)
                    _kr2 = _kr2.dropna(subset=['_uma']).drop_duplicates(subset='_uma', keep='last')
                    if '_uma' not in horses_df.columns:
                        horses_df['_uma'] = horses_df['horse_num'].astype(int) if 'horse_num' in horses_df.columns else horses_df.index + 1
                    horses_df = horses_df.merge(_kr2, on='_uma', how='left', suffixes=('', '_kyi2'))
                    # Prefer non-null values from KYI
                    for _c in ['oz_tansho_base_log', 'oz_fukusho_base_log', 'oz_base_pop_rank']:
                        _c2 = f'{_c}_kyi2'
                        if _c2 in horses_df.columns:
                            horses_df[_c] = horses_df[_c].fillna(horses_df[_c2])
                            horses_df.drop(columns=[_c2], inplace=True, errors='ignore')
        except Exception as e:
            print(f"[WARN] JRDB KYI oz fallback failed: {e}")

    # odds_change_rate / pop_rank_change / odds_sharp_drop
    # These require realtime odds vs base odds comparison
    # At prediction time, if realtime odds are available in horses_df, compute them
    try:
        if 'oz_tansho_base_log' in horses_df.columns:
            _base_odds_raw = np.expm1(horses_df['oz_tansho_base_log'].fillna(2.3))
            # Check if realtime odds are available (from 単勝オッズ column)
            _rt_odds_col = None
            for _oc in ['単勝オッズ', 'odds_log']:
                if _oc in horses_df.columns:
                    _rt_odds_col = _oc
                    break
            if _rt_odds_col and (horses_df.get(_rt_odds_col, pd.Series([0])) > 0).any():
                if _rt_odds_col == 'odds_log':
                    _rt_odds = np.expm1(horses_df[_rt_odds_col].fillna(0))
                else:
                    _rt_odds = horses_df[_rt_odds_col].fillna(0)
                _valid = (_base_odds_raw > 0) & (_rt_odds > 0)
                _change = pd.Series(0.0, index=horses_df.index)
                _change[_valid] = ((_base_odds_raw[_valid] - _rt_odds[_valid]) / _base_odds_raw[_valid]).clip(-2.0, 2.0)
                horses_df['odds_change_rate'] = _change
                # pop_rank_change
                _base_rank = horses_df.get('oz_base_pop_rank', pd.Series([8] * len(horses_df))).fillna(8)
                _rt_rank = _rt_odds.rank(method='min', ascending=True)
                horses_df['pop_rank_change'] = (_base_rank - _rt_rank).fillna(0).astype(int)
                # odds_sharp_drop: realtime <= base * 0.8
                _sharp = pd.Series(0, index=horses_df.index)
                _sharp[_valid] = (_rt_odds[_valid] <= _base_odds_raw[_valid] * 0.8).astype(int)
                horses_df['odds_sharp_drop'] = _sharp
    except Exception as e:
        print(f"[WARN] Odds change features failed: {e}")

    horses_df.drop(columns=['_uma'], inplace=True, errors='ignore')

    # デフォルト値で埋め（拡張特徴量含む）
    _ext_defaults = {
        'jrdb_oikiri_idx': 53.0, 'jrdb_ten_time_idx': 14.5, 'jrdb_shimai_time_idx': 21.0,
        'jrdb_cid_idx': 0.0, 'jrdb_ls_idx': 0.0,
        'jrdb_kta_idm': 13.0, 'jrdb_kta_ten_pred': -14.0, 'jrdb_kta_agari_pred': -11.0,
        'jrdb_ze_idm_avg': 37.0, 'jrdb_ze_ten_avg': -15.0, 'jrdb_ze_agari_avg': -12.0,
        'jrdb_ze_furi_count': 0.0, 'jrdb_tb_homestr_inner': 2.0,
        'jrdb_dam_rensho_avg': 1600.0, 'jrdb_bms_rensho_avg': 1600.0,
        'jrdb_heavy_apt_skb': 0, 'jrdb_anshin': 0, 'jrdb_run_stage': 0,
        'jrdb_turf_baba_code': 0, 'jrdb_dirt_baba_code': 0,
        'oz_tansho_base_log': 2.3, 'oz_fukusho_base_log': 0.7,
        'oz_base_pop_rank': 8, 'odds_change_rate': 0.0,
        'pop_rank_change': 0, 'odds_sharp_drop': 0,
    }
    _all_defaults = {**JRDB_DEFAULTS, **_ext_defaults, **PACI_TIER_A_DEFAULTS}
    for feat, default in _all_defaults.items():
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
