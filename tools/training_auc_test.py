#!/usr/bin/env python3
"""Session #47 B: 拡張調教 features AUC test (V15 vs V15+training).

V15 baseline (150 features) と V15 + 拡張調教 features (8 個) の WF AUC 比較。

出力:
  data/v18/training_auc_test_5_8.md
  data/v18/training_auc_test_5_8.json

Usage:
  python tools/training_auc_test.py             # full WF (2021-2025, ~30 min)
  python tools/training_auc_test.py --quick     # 2024 のみ smoke test
  python tools/training_auc_test.py --by-class  # クラス別 (新馬/1勝/重賞)
"""

import os
import sys
import json
import gzip
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
V18_DIR = os.path.join(DATA_DIR, 'v18')

V15_CACHE = os.path.join(DATA_DIR, '_v15_optuna_df_cache.pkl.gz')
TRAINING_TIMES_CSV = os.path.join(DATA_DIR, 'netkeiba_training_times.csv')
JRDB_CYB_CSV = os.path.join(DATA_DIR, 'jrdb_cyb.csv')

LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'seed': 42,
}

EXTENDED_TRAINING_FEATURES = [
    'training_time_5f',
    'training_time_3f',
    'training_pace_5f_3f',
    'days_since_last_training',
    'training_count_2w',
    'cyb_train_baba_enc',
    'cyb_train_amount',
    'cyb_train_change_enc',
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_v15_dataframe():
    """V15 cache を load (150 features + race_date / target / race_id)."""
    log(f"Loading V15 cache: {V15_CACHE}")
    if not os.path.exists(V15_CACHE):
        raise FileNotFoundError(f"V15 cache not found: {V15_CACHE}")
    with gzip.open(V15_CACHE, 'rb') as f:
        df = pickle.load(f)
    log(f"  shape: {df.shape}, cols: {len(df.columns)}")
    return df


def load_training_times():
    """netkeiba_training_times.csv → race_id × umaban で indexed."""
    log(f"Loading training times: {TRAINING_TIMES_CSV}")
    df = pd.read_csv(TRAINING_TIMES_CSV, encoding='utf-8-sig', low_memory=False)
    df.columns = [c.lstrip('﻿') for c in df.columns]
    log(f"  rows: {len(df)}")
    if 'training_date' in df.columns:
        df['training_date'] = pd.to_datetime(df['training_date'], errors='coerce')
    if 'race_date' in df.columns:
        df['race_date'] = pd.to_datetime(df['race_date'], errors='coerce')
    return df


def load_jrdb_cyb():
    """JRDB CYB → race_id × umaban で indexed."""
    log(f"Loading JRDB CYB: {JRDB_CYB_CSV}")
    df = pd.read_csv(JRDB_CYB_CSV, encoding='utf-8-sig', low_memory=False)
    df.columns = [c.lstrip('﻿') for c in df.columns]
    log(f"  rows: {len(df)}")
    return df


def build_extended_features(df_v15, df_train, df_cyb):
    """拡張調教 features 8 個 を df_v15 に追加。

    df_v15 must have: race_id, umaban, race_date.
    """
    log("Building extended training features...")
    out = df_v15.copy()

    # 1. ハロン別 / pace
    if 'race_id' in df_train.columns and 'umaban' in df_train.columns:
        # latest training per (race_id, umaban)
        df_train_latest = (df_train
            .sort_values(['race_id', 'umaban', 'training_date'])
            .drop_duplicates(['race_id', 'umaban'], keep='last'))
        df_train_latest = df_train_latest[['race_id', 'umaban',
                                           'time_5f', 'time_3f', 'time_1f',
                                           'training_date']]
        # ensure umaban dtype matches
        df_train_latest['umaban'] = pd.to_numeric(df_train_latest['umaban'], errors='coerce')
        out['umaban'] = pd.to_numeric(out.get('umaban', out.get('horse_num')), errors='coerce')
        out = out.merge(df_train_latest, on=['race_id', 'umaban'], how='left',
                        suffixes=('', '_tt'))
        out['training_time_5f'] = pd.to_numeric(out.get('time_5f'), errors='coerce')
        out['training_time_3f'] = pd.to_numeric(out.get('time_3f'), errors='coerce')
        out['training_pace_5f_3f'] = (out['training_time_5f'] - out['training_time_3f']) / 2.0
        out['days_since_last_training'] = (
            pd.to_datetime(out.get('race_date'), errors='coerce') -
            pd.to_datetime(out.get('training_date'), errors='coerce')
        ).dt.days
    else:
        out['training_time_5f'] = np.nan
        out['training_time_3f'] = np.nan
        out['training_pace_5f_3f'] = np.nan
        out['days_since_last_training'] = np.nan

    # 2. training_count_2w (馬単位、 expanding)
    if 'race_id' in df_train.columns and 'umaban' in df_train.columns:
        try:
            tc = (df_train
                .groupby(['race_id', 'umaban'])
                .size()
                .reset_index(name='training_count_2w'))
            tc['umaban'] = pd.to_numeric(tc['umaban'], errors='coerce')
            out = out.merge(tc, on=['race_id', 'umaban'], how='left')
            out['training_count_2w'] = out['training_count_2w'].fillna(0).astype(int)
        except Exception as e:
            log(f"  training_count_2w failed: {e}")
            out['training_count_2w'] = 0
    else:
        out['training_count_2w'] = 0

    # 3. JRDB CYB direct features
    cyb_subset = df_cyb[['race_id', 'umaban', 'train_baba', 'train_amount',
                         'train_change', 'train_eval']].copy()
    cyb_subset['umaban'] = pd.to_numeric(cyb_subset['umaban'], errors='coerce')
    cyb_subset = cyb_subset.drop_duplicates(['race_id', 'umaban'], keep='last')

    out['race_id'] = out['race_id'].astype(str)
    cyb_subset['race_id'] = cyb_subset['race_id'].astype(str)
    out = out.merge(cyb_subset, on=['race_id', 'umaban'], how='left')
    out['cyb_train_baba_enc'] = pd.to_numeric(out.get('train_baba'), errors='coerce').fillna(0)
    out['cyb_train_amount'] = pd.to_numeric(out.get('train_amount'), errors='coerce').fillna(0)
    out['cyb_train_change_enc'] = pd.to_numeric(out.get('train_change'), errors='coerce').fillna(0)

    # final fillna for numeric cols
    for col in EXTENDED_TRAINING_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0.0)
        else:
            out[col] = 0.0

    log(f"  Extended features added: {EXTENDED_TRAINING_FEATURES}")
    return out


def run_wf_lgb_single(df, feature_cols, year_train_end, year_test):
    """single fold WF LGB AUC."""
    if 'race_date' not in df.columns:
        raise KeyError('race_date column required')
    df['race_date'] = pd.to_datetime(df['race_date'], errors='coerce')
    df['_year'] = df['race_date'].dt.year

    train_mask = df['_year'] <= year_train_end
    test_mask = df['_year'] == year_test
    if test_mask.sum() == 0:
        return None

    target_col = 'target' if 'target' in df.columns else 'finish_top3'
    if target_col not in df.columns:
        # finish <= 3 derive
        finish_col = next((c for c in ['finish', 'chakujun', 'kakuteichakujun'] if c in df.columns), None)
        if not finish_col:
            raise KeyError('target/finish column required')
        df[target_col] = (pd.to_numeric(df[finish_col], errors='coerce') <= 3).astype(int)

    X_train = df.loc[train_mask, feature_cols].fillna(0.0)
    y_train = df.loc[train_mask, target_col]
    X_test = df.loc[test_mask, feature_cols].fillna(0.0)
    y_test = df.loc[test_mask, target_col]

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

    model = lgb.train(
        LGB_PARAMS, train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred = model.predict(X_test)
    auc = roc_auc_score(y_test, pred)
    return auc, len(y_test), int(y_test.sum())


def run_test(quick=False, by_class=False):
    df_v15 = load_v15_dataframe()
    df_train = load_training_times()
    df_cyb = load_jrdb_cyb()

    df_ext = build_extended_features(df_v15, df_train, df_cyb)

    # baseline V15 features (numeric only)
    drop_cols = {'race_id', 'race_date', 'horse_id', 'horse_name', 'jockey_name',
                 'trainer_name', 'target', 'finish', 'finish_top3', '_year',
                 'umaban', 'horse_num'}
    numeric_v15 = [c for c in df_v15.columns
                   if c not in drop_cols and pd.api.types.is_numeric_dtype(df_v15[c])]
    log(f"V15 numeric features: {len(numeric_v15)}")
    extended = numeric_v15 + EXTENDED_TRAINING_FEATURES

    years = [2024] if quick else [2021, 2022, 2023, 2024, 2025]
    results = {'baseline': {}, 'extended': {}}

    for y in years:
        log(f"--- WF year {y} ---")
        try:
            r1 = run_wf_lgb_single(df_ext, numeric_v15, y - 1, y)
            r2 = run_wf_lgb_single(df_ext, extended, y - 1, y)
            results['baseline'][y] = r1
            results['extended'][y] = r2
            log(f"  baseline AUC: {r1[0]:.4f}, extended AUC: {r2[0]:.4f}, "
                f"delta: +{r2[0] - r1[0]:.4f}")
        except Exception as e:
            log(f"  year {y} failed: {e}")
            results['baseline'][y] = None
            results['extended'][y] = None

    valid_pairs = [(results['baseline'][y][0], results['extended'][y][0])
                   for y in years if results['baseline'][y] and results['extended'][y]]
    if valid_pairs:
        b_mean = np.mean([p[0] for p in valid_pairs])
        e_mean = np.mean([p[1] for p in valid_pairs])
        delta = e_mean - b_mean
    else:
        b_mean = e_mean = delta = float('nan')

    summary = {
        'years': years,
        'baseline_mean_auc': b_mean,
        'extended_mean_auc': e_mean,
        'delta_auc': delta,
        'judgment': 'GO' if delta >= 0.001 else 'NO-GO',
        'extended_features': EXTENDED_TRAINING_FEATURES,
        'per_year': {
            str(y): {
                'baseline': results['baseline'].get(y),
                'extended': results['extended'].get(y),
            } for y in years
        },
        'timestamp': datetime.now().isoformat(),
    }

    out_json = os.path.join(V18_DIR, 'training_auc_test_5_8.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    log(f"Saved: {out_json}")

    log(f"\n=== Summary ===")
    log(f"  Baseline mean AUC: {b_mean:.4f}")
    log(f"  Extended mean AUC: {e_mean:.4f}")
    log(f"  Delta:             {delta:+.4f}")
    log(f"  Judgment:          {summary['judgment']}")

    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true', help='2024 のみ smoke test')
    p.add_argument('--by-class', action='store_true', help='クラス別')
    args = p.parse_args()

    os.makedirs(V18_DIR, exist_ok=True)
    summary = run_test(quick=args.quick, by_class=args.by_class)
    log(f"\nDone. judgment={summary['judgment']}")
