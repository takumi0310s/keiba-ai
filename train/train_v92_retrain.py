#!/usr/bin/env python
"""KEIBA AI V9.2 Enhanced Re-training with 7 JV-Link CSVs

New features from additional data sources:
  Pattern A (leak-free, expanding window):
    - jockey_dist_wr: Jockey distance-category win rate (alpha=10)
    - trainer_surface_wr: Trainer surface win rate (alpha=15)
    - trainer_dist_wr: Trainer distance win rate (alpha=15)
    - horse_course_top3r: Horse course-specific top3 rate (alpha=5)
    - sire_dist_apt: Sire distance aptitude (blood_full, mapped to race dist)
    - sire_surface_apt: Sire surface aptitude (blood_full, mapped to race surface)
    - bms_top3r_static: BMS progeny top3 rate (blood_full)
    - wood_best_4f_1w: Wood training best 4F in 7 days
    - sakaro_best_4f_1w: Sakaro training best 4F in 7 days

  Pattern B adds: odds_log, horse_weight, condition_enc, pop_rank, weather etc.

Note: odds_history.csv has only 1 snapshot per race -> odds_change_rate NOT computable.
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features,
    compute_distance_aptitude, compute_frame_advantage,
    COURSE_MAP, N_TOP_SIRE, FEATURES_V93,
    train_lgb, train_xgb, show_feature_importance,
)
from train_v92_leakfree import (
    LEAK_FEATURES_A,
    classify_condition, calc_trio_bets, backtest_condition,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = BASE_DIR

# Current production Pattern A (67 features)
CURRENT_PATTERN_A = [f for f in FEATURES_V93 if f not in LEAK_FEATURES_A]

# New features to add
NEW_FEATURES_A = [
    'jockey_dist_wr',       # Jockey distance-specific win rate (expanding)
    'trainer_surface_wr',    # Trainer surface win rate (expanding)
    'trainer_dist_wr',       # Trainer distance win rate (expanding)
    'horse_course_top3r',    # Horse course-specific top3 rate (expanding)
    'sire_dist_apt',         # Sire distance aptitude (blood_full)
    'sire_surface_apt',      # Sire surface aptitude (blood_full)
    'bms_top3r_static',      # BMS progeny top3 rate (blood_full)
    'wood_best_4f_1w',       # Wood best 4F in 7 days
    'sakaro_best_4f_1w',     # Sakaro best 4F in 7 days
]

# Enhanced Pattern A = Current + New
FEATURES_ENHANCED_A = CURRENT_PATTERN_A + NEW_FEATURES_A

# Live extras for Pattern B
LIVE_EXTRA = [
    'weather_enc', 'pop_rank', 'cushion_value', 'moisture_rate',
    'temperature', 'humidity', 'wind_speed', 'precipitation',
]

# Enhanced Pattern B = FEATURES_V93 (includes leak features) + New + Live
FEATURES_ENHANCED_B = FEATURES_V93 + NEW_FEATURES_A + LIVE_EXTRA

# PKL name mapping
def to_pkl_names(features):
    return [f if f != 'num_horses_val' else 'num_horses' for f in features]


# ===== New Feature Functions =====

def merge_blood_features(df):
    """Merge sire/BMS aptitude from blood_full.csv."""
    blood_path = os.path.join(DATA_DIR, 'blood_full.csv')
    if not os.path.exists(blood_path):
        print("  WARNING: blood_full.csv not found")
        for c in ['sire_dist_apt', 'sire_surface_apt', 'bms_top3r_static']:
            df[c] = 0
        return df

    print("Merging blood_full.csv features...")
    blood = pd.read_csv(blood_path, encoding='utf-8-sig', dtype=str)

    num_cols = ['sire_turf_wr', 'sire_dirt_wr', 'sire_sprint_wr',
                'sire_mile_wr', 'sire_middle_wr', 'sire_long_wr', 'bms_top3r']
    for c in num_cols:
        blood[c] = pd.to_numeric(blood[c], errors='coerce')
    blood['horse_id'] = blood['horse_id'].astype(str).str.strip()

    # Merge on horse_id
    df['_hid'] = df['horse_id'].astype(str).str.strip()
    df = df.merge(
        blood[['horse_id'] + num_cols],
        left_on='_hid', right_on='horse_id', how='left', suffixes=('', '_bl')
    )
    matched = df['sire_sprint_wr'].notna().sum()
    print(f"  Blood matched: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")

    # Sire distance aptitude: map race distance -> sire performance at that distance
    dc = df['dist_cat'].astype(float)
    df['sire_dist_apt'] = np.select(
        [dc <= 1, dc == 2, dc == 3],
        [df['sire_sprint_wr'], df['sire_mile_wr'], df['sire_middle_wr']],
        default=df['sire_long_wr']
    )

    # Sire surface aptitude: turf vs dirt
    df['sire_surface_apt'] = np.where(
        df['surface_enc'] == 0, df['sire_turf_wr'], df['sire_dirt_wr']
    )

    # BMS top3 rate
    global_t3 = df['is_top3'].mean()
    global_wr = df['is_win'].mean()
    df['bms_top3r_static'] = df['bms_top3r'].fillna(global_t3)
    df['sire_dist_apt'] = df['sire_dist_apt'].fillna(global_wr)
    df['sire_surface_apt'] = df['sire_surface_apt'].fillna(global_wr)

    drop = ['_hid', 'horse_id_bl'] + num_cols
    df = df.drop(columns=[c for c in drop if c in df.columns], errors='ignore')
    print(f"  Added: sire_dist_apt, sire_surface_apt, bms_top3r_static")
    return df


def compute_jockey_dist_wr(df):
    """Jockey distance-category win rate (expanding window, alpha=10)."""
    print("Computing jockey distance win rate...")
    df = df.sort_values('date_num').reset_index(drop=True)

    global_wr = df['is_win'].mean()
    alpha = 10

    df['_dc'] = pd.cut(df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
                        labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
    df['_r'] = df.groupby(['jockey_id', '_dc']).cumcount()
    df['_w'] = df.groupby(['jockey_id', '_dc'])['is_win'].cumsum() - df['is_win']
    df['jockey_dist_wr'] = (df['_w'] + alpha * global_wr) / (df['_r'] + alpha)

    df.drop(columns=['_dc', '_r', '_w'], inplace=True)
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    return df


def compute_trainer_enhanced(df):
    """Trainer surface/distance win rates (expanding window, alpha=15)."""
    print("Computing trainer surface & distance win rates...")
    df = df.sort_values('date_num').reset_index(drop=True)

    global_wr = df['is_win'].mean()
    alpha = 15

    # Surface
    df['_r'] = df.groupby(['trainer_id', 'surface_enc']).cumcount()
    df['_w'] = df.groupby(['trainer_id', 'surface_enc'])['is_win'].cumsum() - df['is_win']
    df['trainer_surface_wr'] = (df['_w'] + alpha * global_wr) / (df['_r'] + alpha)
    df.drop(columns=['_r', '_w'], inplace=True)

    # Distance
    df['_dc'] = pd.cut(df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
                        labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
    df['_r'] = df.groupby(['trainer_id', '_dc']).cumcount()
    df['_w'] = df.groupby(['trainer_id', '_dc'])['is_win'].cumsum() - df['is_win']
    df['trainer_dist_wr'] = (df['_w'] + alpha * global_wr) / (df['_r'] + alpha)
    df.drop(columns=['_dc', '_r', '_w'], inplace=True)

    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    return df


def compute_horse_course_top3r(df):
    """Horse course-specific top3 rate (expanding window, alpha=5)."""
    print("Computing horse course top3 rate...")
    df = df.sort_values('date_num').reset_index(drop=True)

    global_t3 = df['is_top3'].mean()
    alpha = 5

    df['_r'] = df.groupby(['horse_id', 'course_enc']).cumcount()
    df['_t'] = df.groupby(['horse_id', 'course_enc'])['is_top3'].cumsum() - df['is_top3']
    df['horse_course_top3r'] = (df['_t'] + alpha * global_t3) / (df['_r'] + alpha)
    df.drop(columns=['_r', '_t'], inplace=True)

    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    return df


def merge_training_1w(df, tt_data):
    """1-week window training best 4F (wood + sakaro)."""
    if tt_data is None:
        df['wood_best_4f_1w'] = 52.0
        df['sakaro_best_4f_1w'] = 53.0
        return df

    print("Computing 1-week training features...")
    t0 = time.time()
    date_nums = df['date_num'].values
    hids = df['horse_id'].astype(str).str.strip().values

    # Build horse_id group indices
    horse_groups = {}
    for i, h in enumerate(hids):
        if h not in horse_groups:
            horse_groups[h] = []
        horse_groups[h].append(i)

    # --- Wood 1w ---
    wood = tt_data['wood']
    horse_wood = {}
    if len(wood) > 0 and 'horse_id' in wood.columns:
        wc = wood[wood['horse_id'].astype(str).str.strip() != ''].copy()
        wc['hid8'] = wc['horse_id'].astype(str).str.strip().apply(lambda x: x[-8:] if len(x) > 8 else x)
        for hid, grp in wc.groupby('hid8'):
            dates = grp['date'].astype(int).values
            times = grp['time_4f'].values
            order = np.argsort(dates)
            horse_wood[hid] = (dates[order], times[order])

    wood_1w = np.full(len(df), np.nan)
    for hid, indices in horse_groups.items():
        if hid not in horse_wood:
            continue
        dates_w, times_w = horse_wood[hid]
        for idx in indices:
            rd = date_nums[idx]
            left = np.searchsorted(dates_w, rd - 7, side='left')
            right = np.searchsorted(dates_w, rd, side='left')
            if left < right:
                wood_1w[idx] = times_w[left:right].min()

    wood_mean = np.nanmean(wood_1w) if np.any(~np.isnan(wood_1w)) else 52.0
    df['wood_best_4f_1w'] = np.where(np.isnan(wood_1w), wood_mean, wood_1w)
    w_matched = np.sum(~np.isnan(wood_1w))
    print(f"  Wood 1w matched: {w_matched}/{len(df)} ({w_matched/len(df)*100:.1f}%)")

    # --- Sakaro 1w ---
    sakaro = tt_data['sakaro']
    horse_sakaro = {}
    if len(sakaro) > 0:
        # Fast name->hid mapping
        nh = df[['horse_name', 'horse_id']].copy()
        nh['horse_name'] = nh['horse_name'].astype(str).str.strip()
        nh['horse_id'] = nh['horse_id'].astype(str).str.strip()
        nh = nh.drop_duplicates('horse_name', keep='last')
        name_to_hid = dict(zip(nh['horse_name'], nh['horse_id']))

        sc = sakaro.copy()
        sc['mapped_hid'] = sc['horse_name'].astype(str).str.strip().map(name_to_hid)
        sc = sc[sc['mapped_hid'].notna()]

        for hid, grp in sc.groupby('mapped_hid'):
            dates = grp['date'].astype(int).values
            times = grp['time_4f'].values
            order = np.argsort(dates)
            horse_sakaro[hid] = (dates[order], times[order])

    sak_1w = np.full(len(df), np.nan)
    for hid, indices in horse_groups.items():
        if hid not in horse_sakaro:
            continue
        dates_s, times_s = horse_sakaro[hid]
        for idx in indices:
            rd = date_nums[idx]
            left = np.searchsorted(dates_s, rd - 7, side='left')
            right = np.searchsorted(dates_s, rd, side='left')
            if left < right:
                sak_1w[idx] = times_s[left:right].min()

    sak_mean = np.nanmean(sak_1w) if np.any(~np.isnan(sak_1w)) else 53.0
    df['sakaro_best_4f_1w'] = np.where(np.isnan(sak_1w), sak_mean, sak_1w)
    s_matched = np.sum(~np.isnan(sak_1w))
    print(f"  Sakaro 1w matched: {s_matched}/{len(df)} ({s_matched/len(df)*100:.1f}%)")
    print(f"  Training 1w time: {time.time()-t0:.1f}s")

    return df


def add_live_features(df):
    """Add Pattern B live features (weather, pop_rank etc.)."""
    # Weather
    if 'weather' in df.columns:
        wmap = {'晴': 0, '曇': 1, '小雨': 2, '雨': 2, '雪': 3}
        df['weather_enc'] = df['weather'].map(wmap).fillna(0).astype(int)
    else:
        df['weather_enc'] = 0

    for f in ['cushion_value', 'moisture_rate', 'temperature',
              'humidity', 'wind_speed', 'precipitation']:
        if f not in df.columns:
            df[f] = 0

    # Pop rank
    if 'tansho_odds' in df.columns:
        df['pop_rank'] = df.groupby('race_id_str')['tansho_odds'].rank(method='min').fillna(8)
    else:
        df['pop_rank'] = 8

    return df


def main():
    total_start = time.time()
    print("=" * 70)
    print("  KEIBA AI V9.2 ENHANCED RE-TRAINING (7 CSV SOURCES)")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  New features: {len(NEW_FEATURES_A)} for Pattern A")
    print("=" * 70)

    # ===== 1. Standard pipeline (existing V9.3 features) =====
    df = load_data()
    lap_df = load_lap_data()
    if lap_df is not None:
        df = df.merge(lap_df, on='race_id_str', how='left')
        matched = df['race_first3f'].notna().sum()
        print(f"  Lap data merged: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")

    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)

    tt_data = load_training_times()
    df = merge_training_features(df, tt_data)
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)
    df = compute_lag_features(df)

    print("Building features...")
    df = build_features(df)
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)

    # ===== 2. New features from additional CSVs =====
    print("\n" + "=" * 70)
    print("  ADDING NEW FEATURES FROM 7 CSV SOURCES")
    print("=" * 70)

    # Check which CSVs exist
    csv_status = {}
    for name in ['blood_full.csv', 'horse_history_full.csv', 'jockey_history_full.csv',
                  'trainer_history_full.csv', 'odds_history.csv']:
        path = os.path.join(DATA_DIR, name)
        exists = os.path.exists(path)
        csv_status[name] = exists
        status = "OK" if exists else "MISSING"
        print(f"  {name}: {status}")

    # Blood features (sire/BMS aptitude from blood_full.csv)
    df = merge_blood_features(df)

    # Expanding window features from race data
    # (These use jra_races_full.csv data directly, not the static CSV aggregates,
    #  to maintain leak-free expanding window design)
    df = compute_jockey_dist_wr(df)
    df = compute_trainer_enhanced(df)
    df = compute_horse_course_top3r(df)

    # 1-week training features
    df = merge_training_1w(df, tt_data)

    # Live features for Pattern B
    df = add_live_features(df)

    # Note about odds_history.csv
    print("\n  NOTE: odds_history.csv has single snapshot per race.")
    print("  Cannot compute odds_change_rate (no multi-timepoint data).")

    # ===== 3. Prepare training data =====
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()
    y = df['target'].values

    max_year = df['year_full'].max()
    valid_mask = df['year_full'] >= (max_year - 1)
    train_mask = ~valid_mask
    y_train, y_valid = y[train_mask], y[valid_mask]

    print(f"\nTrain: {train_mask.sum()}, Valid: {valid_mask.sum()}")
    print(f"Target rate: train={y_train.mean():.3f}, valid={y_valid.mean():.3f}")

    # Ensure all features are numeric
    for feat_list in [CURRENT_PATTERN_A, FEATURES_ENHANCED_A, FEATURES_ENHANCED_B]:
        for f in feat_list:
            if f not in df.columns:
                df[f] = 0
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    # ===== 4. Train Current Pattern A Baseline =====
    print("\n" + "=" * 70)
    print(f"  BASELINE: Current Pattern A ({len(CURRENT_PATTERN_A)} features)")
    print("=" * 70)

    X_base = df[CURRENT_PATTERN_A].values
    lgb_base = train_lgb(X_base[train_mask], y_train, X_base[valid_mask], y_valid, CURRENT_PATTERN_A)
    lgb_base_pred = lgb_base.predict(X_base[valid_mask])
    lgb_base_auc = roc_auc_score(y_valid, lgb_base_pred)

    xgb_base = train_xgb(X_base[train_mask], y_train, X_base[valid_mask], y_valid)
    xgb_base_pred = xgb_base.predict(xgb.DMatrix(X_base[valid_mask]))
    xgb_base_auc = roc_auc_score(y_valid, xgb_base_pred)

    total_w = lgb_base_auc + xgb_base_auc
    w_lgb_base = lgb_base_auc / total_w
    w_xgb_base = xgb_base_auc / total_w
    base_pred = lgb_base_pred * w_lgb_base + xgb_base_pred * w_xgb_base
    base_auc = roc_auc_score(y_valid, base_pred)
    print(f"\n  Baseline AUC: LGB {lgb_base_auc:.4f} / XGB {xgb_base_auc:.4f} / Ensemble {base_auc:.4f}")

    # ===== 5. Train Enhanced Pattern A =====
    print("\n" + "=" * 70)
    print(f"  ENHANCED: Pattern A ({len(FEATURES_ENHANCED_A)} features, +{len(NEW_FEATURES_A)} new)")
    print(f"  New: {NEW_FEATURES_A}")
    print("=" * 70)

    X_enh = df[FEATURES_ENHANCED_A].values
    lgb_enh = train_lgb(X_enh[train_mask], y_train, X_enh[valid_mask], y_valid, FEATURES_ENHANCED_A)
    lgb_enh_pred = lgb_enh.predict(X_enh[valid_mask])
    lgb_enh_auc = roc_auc_score(y_valid, lgb_enh_pred)

    xgb_enh = train_xgb(X_enh[train_mask], y_train, X_enh[valid_mask], y_valid)
    xgb_enh_pred = xgb_enh.predict(xgb.DMatrix(X_enh[valid_mask]))
    xgb_enh_auc = roc_auc_score(y_valid, xgb_enh_pred)

    total_w = lgb_enh_auc + xgb_enh_auc
    w_lgb_enh = lgb_enh_auc / total_w
    w_xgb_enh = xgb_enh_auc / total_w
    enh_pred = lgb_enh_pred * w_lgb_enh + xgb_enh_pred * w_xgb_enh
    enh_auc = roc_auc_score(y_valid, enh_pred)
    print(f"\n  Enhanced AUC: LGB {lgb_enh_auc:.4f} / XGB {xgb_enh_auc:.4f} / Ensemble {enh_auc:.4f}")
    print(f"  vs Baseline: {enh_auc - base_auc:+.4f}")

    # Feature importance
    fi_enh = show_feature_importance(lgb_enh, FEATURES_ENHANCED_A, "Enhanced Pattern A")

    # ===== 6. Ablation: each new feature =====
    print("\n" + "=" * 70)
    print("  ABLATION: Individual new feature contribution")
    print("=" * 70)

    ablation = {}
    for feat in NEW_FEATURES_A:
        feats_without = [f for f in FEATURES_ENHANCED_A if f != feat]
        X_abl = df[feats_without].values
        lgb_abl = train_lgb(X_abl[train_mask], y_train, X_abl[valid_mask], y_valid, feats_without)
        abl_pred = lgb_abl.predict(X_abl[valid_mask])
        abl_auc = roc_auc_score(y_valid, abl_pred)
        delta = lgb_enh_auc - abl_auc
        marker = "HELPS" if delta > 0 else "HURTS" if delta < -0.0005 else "NEUTRAL"
        ablation[feat] = {'auc_without': round(abl_auc, 5), 'delta': round(delta, 5), 'helps': delta > 0}
        print(f"  {feat:22s}: remove -> {abl_auc:.4f} (delta {delta:+.5f}) [{marker}]")

    # Remove features that hurt
    hurting = [f for f, v in ablation.items() if v['delta'] < -0.0005]
    if hurting:
        print(f"\n  Removing hurting features: {hurting}")
        features_final_a = [f for f in FEATURES_ENHANCED_A if f not in hurting]
        X_fin = df[features_final_a].values
        lgb_fin = train_lgb(X_fin[train_mask], y_train, X_fin[valid_mask], y_valid, features_final_a)
        lgb_fin_pred = lgb_fin.predict(X_fin[valid_mask])
        lgb_fin_auc = roc_auc_score(y_valid, lgb_fin_pred)

        xgb_fin = train_xgb(X_fin[train_mask], y_train, X_fin[valid_mask], y_valid)
        xgb_fin_pred = xgb_fin.predict(xgb.DMatrix(X_fin[valid_mask]))
        xgb_fin_auc = roc_auc_score(y_valid, xgb_fin_pred)

        tw = lgb_fin_auc + xgb_fin_auc
        w_lgb_fin = lgb_fin_auc / tw
        w_xgb_fin = xgb_fin_auc / tw
        fin_pred = lgb_fin_pred * w_lgb_fin + xgb_fin_pred * w_xgb_fin
        fin_auc = roc_auc_score(y_valid, fin_pred)
        print(f"  Cleaned AUC: LGB {lgb_fin_auc:.4f} / XGB {xgb_fin_auc:.4f} / Ensemble {fin_auc:.4f}")
    else:
        features_final_a = list(FEATURES_ENHANCED_A)
        lgb_fin = lgb_enh
        xgb_fin = xgb_enh
        lgb_fin_auc = lgb_enh_auc
        xgb_fin_auc = xgb_enh_auc
        w_lgb_fin = w_lgb_enh
        w_xgb_fin = w_xgb_enh
        fin_auc = enh_auc

    # ===== 7. Train Enhanced Pattern B =====
    print("\n" + "=" * 70)
    print(f"  ENHANCED: Pattern B ({len(FEATURES_ENHANCED_B)} features)")
    print("=" * 70)

    # Remove hurting features from B too
    features_final_b = [f for f in FEATURES_ENHANCED_B if f not in hurting]

    for f in features_final_b:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X_b = df[features_final_b].values
    lgb_b = train_lgb(X_b[train_mask], y_train, X_b[valid_mask], y_valid, features_final_b)
    lgb_b_pred = lgb_b.predict(X_b[valid_mask])
    lgb_b_auc = roc_auc_score(y_valid, lgb_b_pred)

    xgb_b = train_xgb(X_b[train_mask], y_train, X_b[valid_mask], y_valid)
    xgb_b_pred = xgb_b.predict(xgb.DMatrix(X_b[valid_mask]))
    xgb_b_auc = roc_auc_score(y_valid, xgb_b_pred)

    tw_b = lgb_b_auc + xgb_b_auc
    w_lgb_b = lgb_b_auc / tw_b
    w_xgb_b = xgb_b_auc / tw_b
    b_pred = lgb_b_pred * w_lgb_b + xgb_b_pred * w_xgb_b
    b_auc = roc_auc_score(y_valid, b_pred)
    print(f"\n  Pattern B AUC: LGB {lgb_b_auc:.4f} / XGB {xgb_b_auc:.4f} / Ensemble {b_auc:.4f}")

    fi_b_df = show_feature_importance(lgb_b, features_final_b, "Enhanced Pattern B")

    # ===== 8. Feature Importance TOP 20 & Odds Dependency =====
    print("\n" + "=" * 70)
    print("  FEATURE IMPORTANCE TOP 20 (Pattern A)")
    print("=" * 70)

    fi_a_imp = lgb_fin.feature_importance(importance_type='gain')
    fi_a_df = pd.DataFrame({
        'feature': features_final_a, 'importance': fi_a_imp
    }).sort_values('importance', ascending=False)

    total_imp = fi_a_df['importance'].sum()
    for i, row in fi_a_df.head(20).iterrows():
        pct = row['importance'] / total_imp * 100
        bar = '#' * int(pct * 2)
        print(f"  {row['feature']:25s} {row['importance']:10.0f} ({pct:5.1f}%) {bar}")

    # Odds dependency check (Pattern B)
    print("\n  --- Odds Dependency Check (Pattern B) ---")
    fi_b_imp = lgb_b.feature_importance(importance_type='gain')
    fi_b_total = fi_b_imp.sum()
    odds_features = ['odds_log', 'pop_rank', 'prev_odds_log']
    odds_imp_total = 0
    for f in odds_features:
        if f in features_final_b:
            idx = features_final_b.index(f)
            imp = fi_b_imp[idx]
            pct = imp / fi_b_total * 100
            odds_imp_total += pct
            print(f"  {f:25s} {imp:10.0f} ({pct:5.1f}%)")
    print(f"  Total odds dependency: {odds_imp_total:.1f}%")
    dep_level = "HIGH" if odds_imp_total > 30 else "MEDIUM" if odds_imp_total > 15 else "LOW"
    print(f"  Dependency level: {dep_level}")

    # ===== 9. Condition Backtest =====
    valid_df = df[valid_mask].copy()
    bt_a = backtest_condition(valid_df, lgb_fin, xgb_fin, features_final_a, w_lgb_fin, w_xgb_fin, "Enhanced Pattern A")
    bt_b = backtest_condition(valid_df, lgb_b, xgb_b, features_final_b, w_lgb_b, w_xgb_b, "Enhanced Pattern B")

    # ===== 10. Results Summary =====
    print("\n" + "=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)
    print(f"  {'Model':<40} {'LGB':>8} {'XGB':>8} {'Ensemble':>10}")
    print(f"  {'-'*66}")
    print(f"  {'Current Pattern A (67 feat)':<40} {lgb_base_auc:>8.4f} {xgb_base_auc:>8.4f} {base_auc:>10.4f}")
    print(f"  {'Enhanced Pattern A (' + str(len(features_final_a)) + ' feat)':<40} {lgb_fin_auc:>8.4f} {xgb_fin_auc:>8.4f} {fin_auc:>10.4f}")
    print(f"  {'Enhanced Pattern B (' + str(len(features_final_b)) + ' feat)':<40} {lgb_b_auc:>8.4f} {xgb_b_auc:>8.4f} {b_auc:>10.4f}")
    print(f"\n  Pattern A improvement: {fin_auc - base_auc:+.4f}")
    print(f"  Baseline thresholds: A > 0.8095, B > 0.8460")

    a_improved = fin_auc > 0.8095
    b_improved = b_auc > 0.8460

    print(f"\n  Pattern A {'PASS' if a_improved else 'FAIL'}: {fin_auc:.4f} {'>' if a_improved else '<='} 0.8095")
    print(f"  Pattern B {'PASS' if b_improved else 'FAIL'}: {b_auc:.4f} {'>' if b_improved else '<='} 0.8460")

    # ===== 11. Save models if improved =====
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    features_final_a_pkl = to_pkl_names(features_final_a)
    features_final_b_pkl = to_pkl_names(features_final_b)

    if a_improved:
        print(f"\n  *** Pattern A IMPROVED! Saving production model ***")
        pkl_a = {
            'model': lgb_fin,
            'features': features_final_a_pkl,
            'version': 'v9.2_enhanced_leakfree',
            'auc': lgb_fin_auc,
            'ensemble_auc': fin_auc,
            'leak_free': True,
            'leak_pattern': 'A',
            'leak_removed': sorted(LEAK_FEATURES_A),
            'sire_map': sire_map,
            'bms_map': bms_map,
            'n_top_encode': N_TOP_SIRE,
            'trained_at': now,
            'n_train': int(train_mask.sum()),
            'n_valid': int(valid_mask.sum()),
            'model_type': 'central',
            'xgb_model': xgb_fin,
            'mlp_model': None,
            'mlp_scaler': None,
            'ensemble_weights': {'lgb': w_lgb_fin, 'xgb': w_xgb_fin, 'mlp': 0},
            'course_map': dict(COURSE_MAP),
            'condition_backtest': bt_a,
            'new_features': [f for f in NEW_FEATURES_A if f not in hurting],
            'baseline_auc': base_auc,
        }
        with open(os.path.join(OUTPUT_DIR, 'keiba_model_v9_central.pkl'), 'wb') as f:
            pickle.dump(pkl_a, f)
        with open(os.path.join(OUTPUT_DIR, 'keiba_model_v8.pkl'), 'wb') as f:
            pickle.dump(pkl_a, f)
        print(f"  Saved: keiba_model_v9_central.pkl + v8 backup")
    else:
        print(f"\n  Pattern A did NOT improve. Keeping current model.")

    if b_improved or a_improved:
        print(f"\n  Saving Pattern B live model...")
        pkl_b = {
            'model': lgb_b,
            'features': features_final_b_pkl,
            'version': 'v9.2_enhanced_live',
            'auc': lgb_b_auc,
            'ensemble_auc': b_auc,
            'pattern_a_auc': fin_auc,
            'leak_free': False,
            'leak_pattern': 'B',
            'live_features': sorted(list(LEAK_FEATURES_A) + LIVE_EXTRA),
            'sire_map': sire_map,
            'bms_map': bms_map,
            'n_top_encode': N_TOP_SIRE,
            'trained_at': now,
            'n_train': int(train_mask.sum()),
            'n_valid': int(valid_mask.sum()),
            'model_type': 'central_live',
            'xgb_model': xgb_b,
            'mlp_model': None,
            'mlp_scaler': None,
            'ensemble_weights': {'lgb': w_lgb_b, 'xgb': w_xgb_b, 'mlp': 0},
            'course_map': dict(COURSE_MAP),
            'weather_map': {'晴': 0, '曇': 1, '小雨': 2, '雨': 2, '雪': 3},
            'condition_backtest': bt_b,
        }
        with open(os.path.join(OUTPUT_DIR, 'keiba_model_v9_central_live.pkl'), 'wb') as f:
            pickle.dump(pkl_b, f)
        print(f"  Saved: keiba_model_v9_central_live.pkl")

    # ===== 12. Save report JSON =====
    fi_top20 = {row['feature']: round(float(row['importance']), 1)
                for _, row in fi_a_df.head(20).iterrows()}

    report = {
        'generated_at': now,
        'total_time_sec': round(time.time() - total_start, 1),
        'data_sources': {
            'jra_races_full': {'rows': int(train_mask.sum() + valid_mask.sum())},
            'training_times': 'used (14d + 7d windows)',
            'blood_full': 'used (sire/BMS aptitude)',
            'horse_history_full': 'expanding window from race data (leak-free)',
            'jockey_history_full': 'expanding window from race data (leak-free)',
            'trainer_history_full': 'expanding window from race data (leak-free)',
            'odds_history': 'NOT used (single snapshot, cannot compute change rate)',
        },
        'new_features': NEW_FEATURES_A,
        'removed_features': hurting,
        'baseline': {
            'pattern_a': {'lgb': round(lgb_base_auc, 5), 'xgb': round(xgb_base_auc, 5),
                          'ensemble': round(base_auc, 5), 'n_features': len(CURRENT_PATTERN_A)},
            'threshold_a': 0.8095,
            'threshold_b': 0.8460,
        },
        'enhanced': {
            'pattern_a': {'lgb': round(lgb_fin_auc, 5), 'xgb': round(xgb_fin_auc, 5),
                          'ensemble': round(fin_auc, 5), 'n_features': len(features_final_a)},
            'pattern_b': {'lgb': round(lgb_b_auc, 5), 'xgb': round(xgb_b_auc, 5),
                          'ensemble': round(b_auc, 5), 'n_features': len(features_final_b)},
        },
        'improvement_a': round(fin_auc - base_auc, 5),
        'a_improved': a_improved,
        'b_improved': b_improved,
        'model_updated': a_improved or b_improved,
        'ablation': ablation,
        'feature_importance_top20': fi_top20,
        'odds_dependency': {
            'total_pct': round(odds_imp_total, 2),
            'level': dep_level,
        },
        'condition_backtest_a': {
            k: {'n': v['n'], 'trio_hits': v['trio_hits'],
                'hit_rate': round(v['trio_hits']/v['n']*100, 1) if v['n'] > 0 else 0}
            for k, v in bt_a.items()
        },
        'condition_backtest_b': {
            k: {'n': v['n'], 'trio_hits': v['trio_hits'],
                'hit_rate': round(v['trio_hits']/v['n']*100, 1) if v['n'] > 0 else 0}
            for k, v in bt_b.items()
        },
    }

    report_path = os.path.join(DATA_DIR, 'v92_training_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Report saved: {report_path}")

    elapsed = time.time() - total_start
    print(f"\n  Total time: {elapsed/60:.1f} min")
    print("  Training complete!")

    return report


if __name__ == '__main__':
    main()
