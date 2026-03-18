#!/usr/bin/env python
"""Feature Engineering Experiment: Reduce Odds Dependency

Goal: Add ability-based features to reduce dependency on odds_log/pop_rank.
Compare 3 patterns:
  - Pattern B  (current):  83 features with odds
  - Pattern B2 (new):      no odds + new ability features
  - Pattern B3 (new):      with odds + new ability features

New features:
  1. time_deviation      - 走破タイム偏差値 (z-score within distance/surface group)
  2. avg_finish_5r       - 過去5走の着順平均
  3. avg_margin_3r       - 過去3走の着差平均
  4. trainer_dist_wr     - 調教師×距離別勝率 (expanding, alpha=20)
  5. sire_dist_top3r     - 父系×距離別複勝率 (expanding, alpha=50)
  6. horse_dist_wr       - 同距離での勝率 (expanding, alpha=5)
  7. weight_ma_dev       - 馬体重移動平均からの乖離
  8. jockey_dist_wr      - 騎手×距離別勝率 (expanding, alpha=10)
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features,
    compute_distance_aptitude, compute_frame_advantage,
    COURSE_MAP, N_TOP_SIRE, FEATURES_V93,
    train_lgb, train_xgb, show_feature_importance,
)
from train_v92_leakfree import LEAK_FEATURES_A
from train_v93_pattern_b import (
    LIVE_EXTRA_FEATURES, FEATURES_PATTERN_B, FEATURES_PATTERN_B_PKL,
    add_weather_feature, add_pop_rank,
)

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Odds-related features to remove for B2
ODDS_FEATURES = {'odds_log', 'pop_rank'}


def compute_new_features(df):
    """Compute new ability-based features."""
    print("\n  Computing new ability-based features...")

    # Sort for expanding window calculations
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    grp = df.groupby('horse_id')

    # === 1. 走破タイム偏差値 (time deviation) ===
    # Parse run_time
    df['run_time_sec'] = pd.to_numeric(df.get('run_time_x10', pd.Series(dtype=float)),
                                        errors='coerce')
    # run_time_x10 is in 1/10 seconds, convert to seconds
    df['run_time_sec'] = df['run_time_sec'] / 10.0
    # Filter invalid times
    df.loc[df['run_time_sec'] <= 30, 'run_time_sec'] = np.nan
    df.loc[df['run_time_sec'] > 300, 'run_time_sec'] = np.nan

    # Z-score within distance/surface group (expanding window)
    # For each row, compute mean/std of run_time for same distance/surface up to that point
    df = df.sort_values('date_num').reset_index(drop=True)
    dist_surf_key = df['distance'].astype(str) + '_' + df['surface_enc'].astype(str)
    df['_ds_key'] = dist_surf_key

    ds_cum_sum = df.groupby('_ds_key')['run_time_sec'].cumsum() - df['run_time_sec'].fillna(0)
    ds_cum_count = df.groupby('_ds_key')['run_time_sec'].apply(
        lambda s: s.notna().cumsum() - s.notna().astype(int)
    ).reset_index(level=0, drop=True)
    ds_cum_sq = df.groupby('_ds_key').apply(
        lambda g: (g['run_time_sec'] ** 2).cumsum() - (g['run_time_sec'] ** 2).fillna(0)
    ).reset_index(level=0, drop=True)

    ds_mean = ds_cum_sum / ds_cum_count.clip(1)
    ds_var = (ds_cum_sq / ds_cum_count.clip(1)) - ds_mean ** 2
    ds_std = np.sqrt(ds_var.clip(0)).replace(0, 1)

    # Shift to use previous race's time for deviation
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    grp = df.groupby('horse_id')
    prev_run_time = grp['run_time_sec'].shift(1)
    prev_ds_mean = grp[ds_mean.reindex(df.index)].shift(1) if False else np.nan

    # Simpler approach: use previous race's deviation from distance/surface mean
    # First compute per-horse previous run_time z-score
    df['_prev_run_time'] = grp['run_time_sec'].shift(1)

    # For the time deviation, compute z-score of previous run time
    # relative to expanding mean of that distance/surface group
    # Use a simpler but effective approach: prev_run_time relative to global dist/surf mean
    df = df.sort_values('date_num').reset_index(drop=True)
    dist_surf_groups = df.groupby('_ds_key')
    expanding_means = {}
    expanding_stds = {}
    for key, g in dist_surf_groups:
        vals = g['run_time_sec'].values
        cum_n = np.zeros(len(vals))
        cum_sum = np.zeros(len(vals))
        cum_sq = np.zeros(len(vals))
        for i in range(len(vals)):
            if i > 0:
                cum_n[i] = cum_n[i-1]
                cum_sum[i] = cum_sum[i-1]
                cum_sq[i] = cum_sq[i-1]
            if not np.isnan(vals[i]):
                cum_n[i] += 1
                cum_sum[i] += vals[i]
                cum_sq[i] += vals[i] ** 2
        mean = np.where(cum_n > 0, cum_sum / cum_n, 0)
        var = np.where(cum_n > 1, cum_sq / cum_n - mean ** 2, 1)
        std = np.sqrt(np.clip(var, 0.01, None))
        for idx, m, s in zip(g.index, mean, std):
            expanding_means[idx] = m
            expanding_stds[idx] = s

    df['_ds_exp_mean'] = pd.Series(expanding_means)
    df['_ds_exp_std'] = pd.Series(expanding_stds)

    # Now compute time_deviation using previous race's run_time
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    grp = df.groupby('horse_id')

    # We want: (prev_run_time - mean_for_that_dist_surf) / std_for_that_dist_surf
    # But we need the mean/std AT the previous race, not at the current race
    # Simpler: shift the z-score itself
    df['_run_time_zscore'] = np.where(
        df['run_time_sec'].notna() & (df['_ds_exp_std'] > 0),
        (df['run_time_sec'] - df['_ds_exp_mean']) / df['_ds_exp_std'],
        0
    )
    # Negate so that faster = higher score (lower time = better)
    df['time_deviation'] = -grp['_run_time_zscore'].shift(1).fillna(0)
    time_dev_valid = (df['time_deviation'] != 0).sum()
    print(f"    time_deviation: {time_dev_valid}/{len(df)} valid ({time_dev_valid/len(df)*100:.1f}%)")

    # === 2. 過去5走の着順平均 ===
    df['prev4_finish'] = grp['finish'].shift(4).fillna(5)
    df['prev5_finish'] = grp['finish'].shift(5).fillna(5)
    finish_5_cols = ['prev_finish', 'prev2_finish', 'prev3_finish', 'prev4_finish', 'prev5_finish']
    df['avg_finish_5r'] = df[finish_5_cols].mean(axis=1)
    print(f"    avg_finish_5r: computed")

    # === 3. 過去3走の着差平均 ===
    df['time_margin_val'] = pd.to_numeric(df.get('time_margin', pd.Series(dtype=float)),
                                           errors='coerce').fillna(0)
    df['prev_margin'] = grp['time_margin_val'].shift(1).fillna(0)
    df['prev2_margin'] = grp['time_margin_val'].shift(2).fillna(0)
    df['prev3_margin'] = grp['time_margin_val'].shift(3).fillna(0)
    df['avg_margin_3r'] = df[['prev_margin', 'prev2_margin', 'prev3_margin']].mean(axis=1)
    print(f"    avg_margin_3r: computed")

    # === 4. 調教師×距離別勝率 ===
    df = df.sort_values('date_num').reset_index(drop=True)
    df['_dist_cat_new'] = pd.cut(df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
                                  labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
    global_wr = df['is_win'].mean()
    global_t3 = df['is_top3'].mean()

    tr_dist_cum_races = df.groupby(['trainer_id', '_dist_cat_new']).cumcount()
    tr_dist_cum_wins = df.groupby(['trainer_id', '_dist_cat_new'])['is_win'].cumsum() - df['is_win']
    alpha_tr = 20
    df['trainer_dist_wr'] = (tr_dist_cum_wins + alpha_tr * global_wr) / (tr_dist_cum_races + alpha_tr)
    print(f"    trainer_dist_wr: computed")

    # === 5. 父系×距離別複勝率 ===
    sire_dist_cum_races = df.groupby(['father', '_dist_cat_new']).cumcount()
    sire_dist_cum_top3 = df.groupby(['father', '_dist_cat_new'])['is_top3'].cumsum() - df['is_top3']
    alpha_sire = 50
    df['sire_dist_top3r'] = (sire_dist_cum_top3 + alpha_sire * global_t3) / (sire_dist_cum_races + alpha_sire)
    print(f"    sire_dist_top3r: computed")

    # === 6. 同距離での勝率 ===
    df = df.sort_values('date_num').reset_index(drop=True)
    hd_cum_races = df.groupby(['horse_id', '_dist_cat_new']).cumcount()
    hd_cum_wins = df.groupby(['horse_id', '_dist_cat_new'])['is_win'].cumsum() - df['is_win']
    alpha_hd = 5
    df['horse_dist_wr'] = (hd_cum_wins + alpha_hd * global_wr) / (hd_cum_races + alpha_hd)
    print(f"    horse_dist_wr: computed")

    # === 7. 馬体重の移動平均からの乖離 ===
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    grp = df.groupby('horse_id')
    # Compute expanding mean of horse_weight (excluding current)
    hw_cum_sum = grp['horse_weight'].cumsum() - df['horse_weight']
    hw_cum_count = grp.cumcount()  # 0-indexed = count before current
    hw_exp_mean = hw_cum_sum / hw_cum_count.clip(1)
    # For first race, use current weight (deviation = 0)
    df['weight_ma_dev'] = np.where(
        hw_cum_count > 0,
        df['horse_weight'] - hw_exp_mean,
        0
    )
    print(f"    weight_ma_dev: computed")

    # === 8. 騎手×距離別勝率 ===
    df = df.sort_values('date_num').reset_index(drop=True)
    jd_cum_races = df.groupby(['jockey_id', '_dist_cat_new']).cumcount()
    jd_cum_wins = df.groupby(['jockey_id', '_dist_cat_new'])['is_win'].cumsum() - df['is_win']
    alpha_jd = 10
    df['jockey_dist_wr'] = (jd_cum_wins + alpha_jd * global_wr) / (jd_cum_races + alpha_jd)
    print(f"    jockey_dist_wr: computed")

    # Cleanup temp columns
    drop_cols = [c for c in df.columns if c.startswith('_')]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Re-sort
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)

    new_feature_names = [
        'time_deviation',
        'avg_finish_5r',
        'avg_margin_3r',
        'trainer_dist_wr',
        'sire_dist_top3r',
        'horse_dist_wr',
        'weight_ma_dev',
        'jockey_dist_wr',
    ]
    print(f"  New features added: {len(new_feature_names)}")
    return df, new_feature_names


def main():
    print("=" * 70)
    print("  FEATURE ENGINEERING EXPERIMENT: Reduce Odds Dependency")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # === Data Loading (same pipeline as Pattern B) ===
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

    # Pattern B extras
    df = add_weather_feature(df)
    df = add_pop_rank(df)

    # === New ability-based features ===
    df, new_features = compute_new_features(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()
    y = df['target'].values

    # Time-based split
    max_year = df['year_full'].max()
    valid_mask = df['year_full'] >= (max_year - 1)
    train_mask = ~valid_mask
    y_train, y_valid = y[train_mask], y[valid_mask]

    print(f"\nTrain: {train_mask.sum()}, Valid: {valid_mask.sum()}")
    print(f"Target rate: train={y_train.mean():.3f}, valid={y_valid.mean():.3f}")

    # === Step 1: Identify odds-related features ===
    print("\n" + "=" * 70)
    print("  STEP 1: Odds-Related Features in Pattern B")
    print("=" * 70)
    print(f"  Total Pattern B features: {len(FEATURES_PATTERN_B)}")
    print(f"  Odds-related features:")
    for f in sorted(ODDS_FEATURES):
        print(f"    - {f}")
    print(f"  Note: prev_odds_log is HISTORICAL (前走オッズ), NOT current-race odds")

    # === Feature lists ===
    # Current Pattern B
    features_b = list(FEATURES_PATTERN_B)

    # B2: No odds + new features
    features_b2 = [f for f in FEATURES_PATTERN_B if f not in ODDS_FEATURES] + new_features

    # B3: With odds + new features
    features_b3 = list(FEATURES_PATTERN_B) + new_features

    print(f"\n  Feature counts:")
    print(f"    Pattern B  (current):         {len(features_b)}")
    print(f"    Pattern B2 (no odds + new):   {len(features_b2)}")
    print(f"    Pattern B3 (with odds + new): {len(features_b3)}")

    # Ensure all features are numeric
    all_features = set(features_b + features_b2 + features_b3)
    for f in all_features:
        if f not in df.columns:
            print(f"  WARNING: feature '{f}' not found, filling with 0")
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    import xgboost as xgb

    results = {}

    # === Train Pattern B (current baseline) ===
    print("\n" + "=" * 70)
    print("  PATTERN B (Current - with odds)")
    print(f"  Features: {len(features_b)}")
    print("=" * 70)

    X_b = df[features_b].values
    lgb_b = train_lgb(X_b[train_mask], y_train, X_b[valid_mask], y_valid, features_b)
    lgb_b_pred = lgb_b.predict(X_b[valid_mask])
    lgb_b_auc = roc_auc_score(y_valid, lgb_b_pred)

    xgb_b = train_xgb(X_b[train_mask], y_train, X_b[valid_mask], y_valid)
    xgb_b_pred = xgb_b.predict(xgb.DMatrix(X_b[valid_mask]))
    xgb_b_auc = roc_auc_score(y_valid, xgb_b_pred)

    total_b = lgb_b_auc + xgb_b_auc
    w_lgb_b = lgb_b_auc / total_b
    w_xgb_b = xgb_b_auc / total_b
    b_pred = lgb_b_pred * w_lgb_b + xgb_b_pred * w_xgb_b
    b_auc = roc_auc_score(y_valid, b_pred)
    print(f"\n  Pattern B AUC: LGB {lgb_b_auc:.4f} / XGB {xgb_b_auc:.4f} / Ensemble {b_auc:.4f}")
    fi_b_df = show_feature_importance(lgb_b, features_b, "Pattern B (Current)")
    results['pattern_b'] = {
        'lgb': round(lgb_b_auc, 4), 'xgb': round(xgb_b_auc, 4),
        'ensemble': round(b_auc, 4), 'n_features': len(features_b),
    }

    # Odds feature importance in current model
    fi_b_dict = dict(zip(fi_b_df['feature'], fi_b_df['importance']))
    odds_importance = {}
    for f in ODDS_FEATURES:
        if f in fi_b_dict:
            odds_importance[f] = float(fi_b_dict[f])
    print(f"\n  Odds feature importance in Pattern B:")
    for f, imp in sorted(odds_importance.items(), key=lambda x: -x[1]):
        total_imp = fi_b_df['importance'].sum()
        pct = imp / total_imp * 100
        print(f"    {f}: {imp:,.0f} ({pct:.1f}% of total)")

    # === Train Pattern B2 (no odds + new features) ===
    print("\n" + "=" * 70)
    print("  PATTERN B2 (No odds + new ability features)")
    print(f"  Features: {len(features_b2)}")
    print(f"  Removed: {sorted(ODDS_FEATURES)}")
    print(f"  Added: {new_features}")
    print("=" * 70)

    X_b2 = df[features_b2].values
    lgb_b2 = train_lgb(X_b2[train_mask], y_train, X_b2[valid_mask], y_valid, features_b2)
    lgb_b2_pred = lgb_b2.predict(X_b2[valid_mask])
    lgb_b2_auc = roc_auc_score(y_valid, lgb_b2_pred)

    xgb_b2 = train_xgb(X_b2[train_mask], y_train, X_b2[valid_mask], y_valid)
    xgb_b2_pred = xgb_b2.predict(xgb.DMatrix(X_b2[valid_mask]))
    xgb_b2_auc = roc_auc_score(y_valid, xgb_b2_pred)

    total_b2 = lgb_b2_auc + xgb_b2_auc
    w_lgb_b2 = lgb_b2_auc / total_b2
    w_xgb_b2 = xgb_b2_auc / total_b2
    b2_pred = lgb_b2_pred * w_lgb_b2 + xgb_b2_pred * w_xgb_b2
    b2_auc = roc_auc_score(y_valid, b2_pred)
    print(f"\n  Pattern B2 AUC: LGB {lgb_b2_auc:.4f} / XGB {xgb_b2_auc:.4f} / Ensemble {b2_auc:.4f}")
    fi_b2_df = show_feature_importance(lgb_b2, features_b2, "Pattern B2 (No Odds + New)")
    results['pattern_b2'] = {
        'lgb': round(lgb_b2_auc, 4), 'xgb': round(xgb_b2_auc, 4),
        'ensemble': round(b2_auc, 4), 'n_features': len(features_b2),
    }

    # New feature importance in B2
    fi_b2_dict = dict(zip(fi_b2_df['feature'], fi_b2_df['importance']))
    print(f"\n  New feature importance in Pattern B2:")
    for f in new_features:
        imp = fi_b2_dict.get(f, 0)
        total_imp_b2 = fi_b2_df['importance'].sum()
        pct = imp / total_imp_b2 * 100 if total_imp_b2 > 0 else 0
        print(f"    {f}: {imp:,.0f} ({pct:.1f}%)")

    # === Train Pattern B3 (with odds + new features) ===
    print("\n" + "=" * 70)
    print("  PATTERN B3 (With odds + new ability features)")
    print(f"  Features: {len(features_b3)}")
    print(f"  Added: {new_features}")
    print("=" * 70)

    X_b3 = df[features_b3].values
    lgb_b3 = train_lgb(X_b3[train_mask], y_train, X_b3[valid_mask], y_valid, features_b3)
    lgb_b3_pred = lgb_b3.predict(X_b3[valid_mask])
    lgb_b3_auc = roc_auc_score(y_valid, lgb_b3_pred)

    xgb_b3 = train_xgb(X_b3[train_mask], y_train, X_b3[valid_mask], y_valid)
    xgb_b3_pred = xgb_b3.predict(xgb.DMatrix(X_b3[valid_mask]))
    xgb_b3_auc = roc_auc_score(y_valid, xgb_b3_pred)

    total_b3 = lgb_b3_auc + xgb_b3_auc
    w_lgb_b3 = lgb_b3_auc / total_b3
    w_xgb_b3 = xgb_b3_auc / total_b3
    b3_pred = lgb_b3_pred * w_lgb_b3 + xgb_b3_pred * w_xgb_b3
    b3_auc = roc_auc_score(y_valid, b3_pred)
    print(f"\n  Pattern B3 AUC: LGB {lgb_b3_auc:.4f} / XGB {xgb_b3_auc:.4f} / Ensemble {b3_auc:.4f}")
    fi_b3_df = show_feature_importance(lgb_b3, features_b3, "Pattern B3 (Odds + New)")
    results['pattern_b3'] = {
        'lgb': round(lgb_b3_auc, 4), 'xgb': round(xgb_b3_auc, 4),
        'ensemble': round(b3_auc, 4), 'n_features': len(features_b3),
    }

    # Odds importance in B3 vs B
    fi_b3_dict = dict(zip(fi_b3_df['feature'], fi_b3_df['importance']))
    total_imp_b3 = fi_b3_df['importance'].sum()
    print(f"\n  Odds feature importance in Pattern B3:")
    for f in sorted(ODDS_FEATURES):
        imp_b = fi_b_dict.get(f, 0)
        imp_b3 = fi_b3_dict.get(f, 0)
        pct_b = imp_b / fi_b_df['importance'].sum() * 100
        pct_b3 = imp_b3 / total_imp_b3 * 100 if total_imp_b3 > 0 else 0
        print(f"    {f}: B={pct_b:.1f}% → B3={pct_b3:.1f}% (change: {pct_b3 - pct_b:+.1f}%)")

    # === Odds Dependency Analysis ===
    print("\n" + "=" * 70)
    print("  ODDS DEPENDENCY ANALYSIS")
    print("=" * 70)
    auc_drop_current = b_auc - b2_auc  # How much AUC drops without odds (old features)
    # For fair comparison, also compute B-without-odds (no new features)
    features_b_no_odds = [f for f in features_b if f not in ODDS_FEATURES]
    X_bno = df[features_b_no_odds].values
    lgb_bno = train_lgb(X_bno[train_mask], y_train, X_bno[valid_mask], y_valid, features_b_no_odds)
    lgb_bno_pred = lgb_bno.predict(X_bno[valid_mask])
    lgb_bno_auc = roc_auc_score(y_valid, lgb_bno_pred)
    xgb_bno = train_xgb(X_bno[train_mask], y_train, X_bno[valid_mask], y_valid)
    xgb_bno_pred = xgb_bno.predict(xgb.DMatrix(X_bno[valid_mask]))
    xgb_bno_auc = roc_auc_score(y_valid, xgb_bno_pred)
    total_bno = lgb_bno_auc + xgb_bno_auc
    bno_pred = lgb_bno_pred * (lgb_bno_auc / total_bno) + xgb_bno_pred * (xgb_bno_auc / total_bno)
    bno_auc = roc_auc_score(y_valid, bno_pred)

    results['pattern_b_no_odds'] = {
        'lgb': round(lgb_bno_auc, 4), 'xgb': round(xgb_bno_auc, 4),
        'ensemble': round(bno_auc, 4), 'n_features': len(features_b_no_odds),
    }

    auc_drop_old = b_auc - bno_auc  # Current odds dependency
    auc_drop_new = b3_auc - b2_auc  # New odds dependency (with new features)

    print(f"\n  AUC Comparison:")
    print(f"  {'Model':<40} {'Ensemble AUC':>12}")
    print(f"  {'-' * 52}")
    print(f"  {'Pattern B  (current, with odds)':<40} {b_auc:>12.4f}")
    print(f"  {'Pattern B  (current, NO odds)':<40} {bno_auc:>12.4f}")
    print(f"  {'Pattern B2 (new features, NO odds)':<40} {b2_auc:>12.4f}")
    print(f"  {'Pattern B3 (new features, WITH odds)':<40} {b3_auc:>12.4f}")

    print(f"\n  Odds Dependency:")
    print(f"  Current model AUC drop without odds:     {auc_drop_old:+.4f} ({auc_drop_old / b_auc * 100:.1f}%)")
    print(f"  New model AUC drop without odds:         {auc_drop_new:+.4f} ({auc_drop_new / b3_auc * 100:.1f}%)")
    dependency_improvement = abs(auc_drop_old) - abs(auc_drop_new)
    print(f"  Dependency reduction:                    {dependency_improvement:+.4f}")

    # === Judgements ===
    print("\n" + "=" * 70)
    print("  JUDGEMENT")
    print("=" * 70)

    b2_target = 0.82
    b2_pass = b2_auc >= b2_target
    print(f"  B2 AUC >= {b2_target}: {'PASS' if b2_pass else 'FAIL'} (B2={b2_auc:.4f})")
    if b2_pass:
        print(f"    → オッズなしでもAUC {b2_target}以上を達成。オッズ依存度改善。")
    else:
        print(f"    → オッズなしではAUC {b2_target}未達。更なる特徴量強化が必要。")

    b3_improvement = b3_auc > b_auc
    print(f"  B3 > B:    {'PASS' if b3_improvement else 'FAIL'} (B3={b3_auc:.4f} vs B={b_auc:.4f}, diff={b3_auc - b_auc:+.4f})")
    if b3_improvement:
        print(f"    → 新特徴量により現行Pattern Bを上回った。本番モデル更新推奨。")
    else:
        print(f"    → 現行Pattern Bが最良。新特徴量の追加効果なし。")

    # === Save Results ===
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = {
        'generated_at': now,
        'objective': 'Reduce odds dependency by adding ability-based features',
        'odds_features_identified': sorted(ODDS_FEATURES),
        'new_features_added': new_features,
        'new_feature_descriptions': {
            'time_deviation': '走破タイム偏差値 (z-score within distance/surface group, prev race)',
            'avg_finish_5r': '過去5走の着順平均',
            'avg_margin_3r': '過去3走の着差平均',
            'trainer_dist_wr': '調教師×距離別勝率 (expanding, alpha=20)',
            'sire_dist_top3r': '父系×距離別複勝率 (expanding, alpha=50)',
            'horse_dist_wr': '同距離での勝率 (expanding, alpha=5)',
            'weight_ma_dev': '馬体重の移動平均からの乖離',
            'jockey_dist_wr': '騎手×距離別勝率 (expanding, alpha=10)',
        },
        'results': results,
        'comparison': {
            'pattern_b_ensemble_auc': round(b_auc, 4),
            'pattern_b_no_odds_ensemble_auc': round(bno_auc, 4),
            'pattern_b2_ensemble_auc': round(b2_auc, 4),
            'pattern_b3_ensemble_auc': round(b3_auc, 4),
        },
        'odds_dependency': {
            'current_auc_drop_without_odds': round(auc_drop_old, 4),
            'current_dependency_pct': round(auc_drop_old / b_auc * 100, 2),
            'new_auc_drop_without_odds': round(auc_drop_new, 4),
            'new_dependency_pct': round(auc_drop_new / b3_auc * 100, 2),
            'dependency_reduction': round(dependency_improvement, 4),
        },
        'odds_feature_importance': {
            'pattern_b': {f: round(fi_b_dict.get(f, 0), 1) for f in ODDS_FEATURES},
            'pattern_b3': {f: round(fi_b3_dict.get(f, 0), 1) for f in ODDS_FEATURES},
        },
        'new_feature_importance': {
            'pattern_b2': {f: round(fi_b2_dict.get(f, 0), 1) for f in new_features},
            'pattern_b3': {f: round(fi_b3_dict.get(f, 0), 1) for f in new_features},
        },
        'judgement': {
            'b2_auc_target': b2_target,
            'b2_pass': b2_pass,
            'b3_beats_b': b3_improvement,
            'recommendation': (
                'UPDATE_MODEL' if b3_improvement else
                ('ODDS_DEPENDENCY_IMPROVED' if b2_pass else 'NO_IMPROVEMENT')
            ),
        },
        'feature_lists': {
            'pattern_b': features_b,
            'pattern_b2': features_b2,
            'pattern_b3': features_b3,
        },
    }

    report_path = os.path.join(DATA_DIR, 'feature_engineering_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {report_path}")

    print("\n  Experiment complete!")
    return report


if __name__ == '__main__':
    main()
