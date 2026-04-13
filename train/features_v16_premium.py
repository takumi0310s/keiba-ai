#!/usr/bin/env python
"""v16 Premium Feature Module — 未使用netkeibaプレミアムCSVから5候補抽出

特徴量:
  1. upset_level (race-level broadcast)
     - upset_level_val (1-5) / top_popularity_reliability (0-100)
  2. prev_review_score (horse lag-1)
     - netkeiba_race_review.csv から前走のreview_scoreを取得
  3. training_eval_rank (per horse same-race)
     - netkeiba_training_eval.csv training_rank A=4,B=3,C=2,D=1
  4. prev_master_index (horse lag-1)
     - netkeiba_master_index.csv から前走のmaster_index(=time_index_m)
  5. prev_track_index (horse lag-1)
     - netkeiba_track_bias.csv から前走のtrack_index(馬場指数)

Lag-1 特徴量は (horse_id, date_num) 昇順ソート + groupby('horse_id').shift(1) で
同一馬の「前走値」を取得する。Leak-free.
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')


V16_PREMIUM_FEATURE_NAMES = [
    'upset_level_val',
    'top_popularity_reliability',
    'prev_review_score',
    'training_eval_rank',
    'prev_master_index',
    'prev_track_index_val',
]

V16_PREMIUM_DEFAULTS = {
    'upset_level_val': 0,
    'top_popularity_reliability': 50.0,
    'prev_review_score': 0,
    'training_eval_rank': 0,
    'prev_master_index': 0.0,
    'prev_track_index_val': 0.0,
}

RANK_MAP = {'A': 4, 'B': 3, 'C': 2, 'D': 1}


def get_v16_premium_features():
    return list(V16_PREMIUM_FEATURE_NAMES)


def _ensure_nk_rid(df):
    """df にキー列 _nk_rid (12桁) / _uma を用意"""
    if '_nk_rid' not in df.columns:
        from jrdb_features import _build_nk_race_id_from_jv
        df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    if '_uma' not in df.columns:
        df['_uma'] = pd.to_numeric(df['umaban'], errors='coerce')
    return df


def merge_upset_level(df):
    """race-level: upset_level_val, top_popularity_reliability"""
    path = os.path.join(DATA_DIR, 'netkeiba_upset_level.csv')
    if not os.path.exists(path):
        for f in ['upset_level_val', 'top_popularity_reliability']:
            df[f] = V16_PREMIUM_DEFAULTS[f]
        return df

    us = pd.read_csv(path, encoding='utf-8-sig', dtype=str,
                     usecols=['race_id', 'upset_level', 'top_popularity_reliability'])
    us = us.rename(columns={'race_id': '_nk_rid',
                             'upset_level': 'upset_level_val',
                             'top_popularity_reliability': 'top_popularity_reliability'})
    us['_nk_rid'] = us['_nk_rid'].astype(str).str.zfill(12)
    us['upset_level_val'] = pd.to_numeric(us['upset_level_val'], errors='coerce')
    us['top_popularity_reliability'] = pd.to_numeric(us['top_popularity_reliability'], errors='coerce')
    us = us.drop_duplicates(subset='_nk_rid', keep='last')

    df = _ensure_nk_rid(df)
    before = len(df)
    df = df.merge(us, on='_nk_rid', how='left')
    df['upset_level_val'] = df['upset_level_val'].fillna(V16_PREMIUM_DEFAULTS['upset_level_val'])
    df['top_popularity_reliability'] = df['top_popularity_reliability'].fillna(V16_PREMIUM_DEFAULTS['top_popularity_reliability'])
    matched = (df['upset_level_val'] > 0).sum()
    print(f"    upset_level merged: {matched}/{before} ({matched/before*100:.1f}%)")
    return df


def merge_training_eval_rank(df):
    """per-horse per-race: training_eval_rank (A-D → 4-1)"""
    path = os.path.join(DATA_DIR, 'netkeiba_training_eval.csv')
    if not os.path.exists(path):
        df['training_eval_rank'] = V16_PREMIUM_DEFAULTS['training_eval_rank']
        return df

    te = pd.read_csv(path, encoding='utf-8-sig', dtype=str,
                     usecols=['race_id', 'umaban', 'training_rank'])
    te = te.rename(columns={'race_id': '_nk_rid', 'umaban': '_uma'})
    te['_nk_rid'] = te['_nk_rid'].astype(str).str.zfill(12)
    te['_uma'] = pd.to_numeric(te['_uma'], errors='coerce')
    te['training_eval_rank'] = te['training_rank'].str.upper().map(RANK_MAP).fillna(0).astype(int)
    te = te[['_nk_rid', '_uma', 'training_eval_rank']].drop_duplicates(subset=['_nk_rid', '_uma'], keep='last')

    df = _ensure_nk_rid(df)
    before = len(df)
    df = df.merge(te, on=['_nk_rid', '_uma'], how='left', suffixes=('', '_te'))
    df['training_eval_rank'] = df['training_eval_rank'].fillna(V16_PREMIUM_DEFAULTS['training_eval_rank'])
    matched = (df['training_eval_rank'] > 0).sum()
    print(f"    training_eval_rank merged: {matched}/{before} ({matched/before*100:.1f}%)")
    return df


def merge_prev_review_score(df):
    """horse lag-1: prev_review_score — 前走のreview_scoreを取得"""
    path = os.path.join(DATA_DIR, 'netkeiba_race_review.csv')
    if not os.path.exists(path):
        df['prev_review_score'] = V16_PREMIUM_DEFAULTS['prev_review_score']
        return df

    rv = pd.read_csv(path, encoding='utf-8-sig', dtype=str,
                     usecols=['race_id', 'umaban', 'horse_name', 'review_score'])
    rv['_nk_rid'] = rv['race_id'].astype(str).str.zfill(12)
    rv['_uma'] = pd.to_numeric(rv['umaban'], errors='coerce')
    rv['review_score'] = pd.to_numeric(rv['review_score'], errors='coerce').fillna(0)
    rv = rv[['_nk_rid', '_uma', 'horse_name', 'review_score']].drop_duplicates(subset=['_nk_rid', '_uma'], keep='last')

    df = _ensure_nk_rid(df)
    before = len(df)

    # 現走のreview_scoreを一旦マージ → horse_id + date昇順 → groupby shift(1) で前走値
    df = df.merge(rv[['_nk_rid', '_uma', 'review_score']].rename(columns={'review_score': '_review_current'}),
                  on=['_nk_rid', '_uma'], how='left')
    df['_review_current'] = df['_review_current'].fillna(0)

    # Lag-1: 同一馬の前走値
    if 'horse_id' in df.columns and 'date_num' in df.columns:
        df = df.sort_values(['horse_id', 'date_num']).reset_index()
        df['prev_review_score'] = df.groupby('horse_id')['_review_current'].shift(1).fillna(0)
        df = df.sort_values('index').drop(columns='index').reset_index(drop=True)
    else:
        df['prev_review_score'] = V16_PREMIUM_DEFAULTS['prev_review_score']

    df.drop(columns=['_review_current'], inplace=True, errors='ignore')
    matched = (df['prev_review_score'] != 0).sum()
    print(f"    prev_review_score (lag-1): {matched}/{before} non-zero ({matched/before*100:.1f}%)")
    return df


def merge_prev_master_index(df):
    """horse lag-1: prev_master_index — 前走の masterコース3分解指数の master_index 列"""
    path = os.path.join(DATA_DIR, 'netkeiba_master_index.csv')
    if not os.path.exists(path):
        df['prev_master_index'] = V16_PREMIUM_DEFAULTS['prev_master_index']
        return df

    mi = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
    if 'master_index' not in mi.columns:
        df['prev_master_index'] = V16_PREMIUM_DEFAULTS['prev_master_index']
        return df
    mi['_nk_rid'] = mi['race_id'].astype(str).str.zfill(12)
    mi['_uma'] = pd.to_numeric(mi['umaban'], errors='coerce')
    mi['master_index'] = pd.to_numeric(mi['master_index'], errors='coerce').fillna(0)
    mi = mi[['_nk_rid', '_uma', 'master_index']].drop_duplicates(subset=['_nk_rid', '_uma'], keep='last')

    df = _ensure_nk_rid(df)
    before = len(df)
    df = df.merge(mi.rename(columns={'master_index': '_mi_current'}),
                  on=['_nk_rid', '_uma'], how='left')
    df['_mi_current'] = df['_mi_current'].fillna(0)

    if 'horse_id' in df.columns and 'date_num' in df.columns:
        df = df.sort_values(['horse_id', 'date_num']).reset_index()
        df['prev_master_index'] = df.groupby('horse_id')['_mi_current'].shift(1).fillna(0)
        df = df.sort_values('index').drop(columns='index').reset_index(drop=True)
    else:
        df['prev_master_index'] = V16_PREMIUM_DEFAULTS['prev_master_index']

    df.drop(columns=['_mi_current'], inplace=True, errors='ignore')
    matched = (df['prev_master_index'] != 0).sum()
    print(f"    prev_master_index (lag-1): {matched}/{before} non-zero ({matched/before*100:.1f}%)")
    return df


def merge_prev_track_index(df):
    """race lag-1 per horse: prev_track_index_val — 前走の馬場指数"""
    path = os.path.join(DATA_DIR, 'netkeiba_track_bias.csv')
    if not os.path.exists(path):
        df['prev_track_index_val'] = V16_PREMIUM_DEFAULTS['prev_track_index_val']
        return df

    tb = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
    if 'track_index' not in tb.columns:
        df['prev_track_index_val'] = V16_PREMIUM_DEFAULTS['prev_track_index_val']
        return df
    tb['_nk_rid'] = tb['race_id'].astype(str).str.zfill(12)
    tb['track_index'] = pd.to_numeric(tb['track_index'], errors='coerce').fillna(0)
    tb = tb[['_nk_rid', 'track_index']].drop_duplicates(subset='_nk_rid', keep='last')

    df = _ensure_nk_rid(df)
    before = len(df)
    df = df.merge(tb.rename(columns={'track_index': '_ti_current'}), on='_nk_rid', how='left')
    df['_ti_current'] = df['_ti_current'].fillna(0)

    if 'horse_id' in df.columns and 'date_num' in df.columns:
        df = df.sort_values(['horse_id', 'date_num']).reset_index()
        df['prev_track_index_val'] = df.groupby('horse_id')['_ti_current'].shift(1).fillna(0)
        df = df.sort_values('index').drop(columns='index').reset_index(drop=True)
    else:
        df['prev_track_index_val'] = V16_PREMIUM_DEFAULTS['prev_track_index_val']

    df.drop(columns=['_ti_current'], inplace=True, errors='ignore')
    matched = (df['prev_track_index_val'] != 0).sum()
    print(f"    prev_track_index_val (lag-1): {matched}/{before} non-zero ({matched/before*100:.1f}%)")
    return df


def compute_all_v16_premium_features(df):
    """5新特徴量を一括マージ"""
    print("  [v16 premium] Merging 5 candidate features...")
    df = merge_upset_level(df)
    df = merge_training_eval_rank(df)
    df = merge_prev_review_score(df)
    df = merge_prev_master_index(df)
    df = merge_prev_track_index(df)
    # Clean up key columns
    df.drop(columns=['_nk_rid', '_uma'], inplace=True, errors='ignore')
    return df
