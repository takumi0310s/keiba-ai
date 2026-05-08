"""V15.5 features wrapper: V15 (145 features) + Sprint 4 ★★★ 13 features 統合

★★★ 1: SRB bias 6 (jrdb_srb.csv)
★★★ 2: master_index 5 expanding-prev (netkeiba_master_index.csv)
★★★ 3: JO cid_idx / ls_idx 2 (jrdb_jo.csv)

絶対遵守:
- 既存 V15 logic は import / call only (改変禁止)
- predict_core / daily_predict / app.py / 既存 train code 不変
- read-only data load
"""
import gzip
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sprint4_feature1 import v15_rid_to_nk, load_srb
from sprint4_feature2 import build_expanding_mi, INDEX_COLS, NEW_COLS as MI_NEW_COLS
from sprint4_feature3 import NEW_COLS as JO_NEW_COLS

BASE = Path(__file__).resolve().parent.parent
V15_CACHE = BASE / 'data' / '_v15_optuna_df_cache.pkl.gz'
JO_CSV = BASE / 'data' / 'jrdb_jo.csv'

SRB_NEW_COLS = ['srb_bias_1c', 'srb_bias_2c', 'srb_bias_bs',
                'srb_bias_3c', 'srb_bias_4c', 'srb_bias_st']
ALL_NEW_COLS = SRB_NEW_COLS + MI_NEW_COLS + JO_NEW_COLS  # 6+5+2 = 13


def load_v15_cache():
    with gzip.open(V15_CACHE, 'rb') as f:
        obj = pickle.load(f)
    return obj['df'], obj['features']


def build_v15_5(df_v15=None, v15_features=None):
    """Build V15.5 dataframe with all 13 ★★★ features merged.

    Returns:
        df_aug: V15 cache + 13 new features
        features_v15_5: V15 features + 13 new
    """
    if df_v15 is None or v15_features is None:
        df_v15, v15_features = load_v15_cache()

    df = df_v15.copy()
    df['nk_race_id'] = df['race_id'].apply(v15_rid_to_nk)

    # SRB merge
    srb = load_srb()
    df = df.merge(srb, left_on='nk_race_id', right_on='race_id',
                  how='left', suffixes=('', '_srb'))
    df = df.drop(columns=[c for c in df.columns if c.endswith('_srb')])

    # MI expanding-prev
    mi_features, _ = build_expanding_mi(df_v15)
    df_aug = df.set_index(['race_id', 'umaban'])
    df_aug = df_aug.join(mi_features, how='left')
    df_aug = df_aug.reset_index()

    # JO merge
    df_aug['umaban'] = df_aug['umaban'].astype(int)
    jo = pd.read_csv(JO_CSV, dtype={'race_id': str}, encoding='utf-8-sig')
    jo = jo[['race_id', 'umaban', 'cid_idx', 'ls_idx']].copy()
    jo['umaban'] = pd.to_numeric(jo['umaban'], errors='coerce').astype('Int64')
    jo['cid_idx'] = pd.to_numeric(jo['cid_idx'], errors='coerce')
    jo['ls_idx'] = pd.to_numeric(jo['ls_idx'], errors='coerce')
    jo = jo.rename(columns={'cid_idx': 'jo_cid_idx', 'ls_idx': 'jo_ls_idx'})
    jo = jo.dropna(subset=['umaban'])
    jo['umaban'] = jo['umaban'].astype(int)
    df_aug = df_aug.merge(jo, left_on=['nk_race_id', 'umaban'],
                          right_on=['race_id', 'umaban'],
                          how='left', suffixes=('', '_jo'))
    df_aug = df_aug.drop(columns=['race_id_jo'], errors='ignore')

    features_v15_5 = list(v15_features) + ALL_NEW_COLS
    return df_aug, features_v15_5


if __name__ == '__main__':
    df_aug, features_v15_5 = build_v15_5()
    print(f'V15.5 shape: {df_aug.shape}, features: {len(features_v15_5)}')
    print(f'New cols ({len(ALL_NEW_COLS)}):')
    for c in ALL_NEW_COLS:
        cov = df_aug[c].notna().mean()
        print(f'  {c}: coverage {cov*100:.1f}%')
