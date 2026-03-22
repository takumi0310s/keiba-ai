#!/usr/bin/env python
"""KEIBA AI v10.1 Training Script

New features over v10 (71 Pattern A / 87 Pattern B):
1. Growth curve: age_peak_dist (age vs optimal performance curve by distance category)
2. Jockey switch: jockey_switch_wr_diff (current vs previous jockey win rate difference)
3. Pace simulation: pace_sim_advantage (Monte Carlo pace scenario advantage score)

All features are pre-day safe (Pattern A leak-free).
"""
import pandas as pd
import numpy as np
import pickle
import gzip
import os
import sys
import json
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
    compute_running_style_features,
    COURSE_MAP, N_TOP_SIRE,
    train_lgb, train_xgb,
)
from train_v92_leakfree import (
    LEAK_FEATURES_A, classify_condition, calc_trio_bets, backtest_condition,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
OUTPUT_DIR = BASE_DIR

# Current v10 Pattern A features (71)
FEATURES_V10_A = [
    'weight_carry', 'age', 'distance', 'course_enc', 'surface_enc', 'sex_enc',
    'num_horses', 'horse_num', 'bracket', 'jockey_wr_calc', 'jockey_course_wr_calc',
    'trainer_top3_calc', 'prev_finish', 'prev_last3f', 'prev_pass4', 'prev_prize',
    'prev2_finish', 'prev3_finish', 'avg_finish_3r', 'best_finish_3r', 'finish_trend',
    'top3_count_3r', 'avg_last3f_3r', 'prev2_last3f', 'dist_change', 'dist_change_abs',
    'rest_days', 'rest_category', 'sire_enc', 'bms_enc', 'dist_cat', 'age_sex',
    'season', 'age_season', 'horse_num_ratio', 'bracket_pos', 'carry_diff', 'age_group',
    'surface_dist_enc', 'course_surface', 'location_enc', 'is_nar', 'prev_odds_log',
    'training_time_filled', 'has_training', 'training_per_dist', 'jockey_surface_wr',
    'horse_career_races', 'horse_career_wr', 'horse_career_top3r',
    'sire_surface_wr', 'sire_dist_wr', 'bms_surface_wr',
    'wood_best_4f_filled', 'has_wood_training',
    'prev_race_first3f', 'prev_race_last3f', 'prev_race_pace_diff', 'prev_agari_relative',
    'wood_count_2w', 'sakaro_best_4f_filled', 'sakaro_best_3f_filled',
    'has_sakaro_training', 'total_training_count',
    'horse_dist_top3r', 'horse_surface_top3r', 'frame_course_dist_wr',
    'running_style', 'predicted_pace', 'pace_advantage', 'style_vs_pace',
]

# New v10.1 features (3)
NEW_FEATURES = [
    'age_peak_dist',          # Growth curve: distance-adjusted age peak score
    'jockey_switch_wr_diff',  # Jockey switch: WR difference (current - prev)
    'pace_sim_advantage',     # Pace simulation: Monte Carlo advantage score
]

FEATURES_V101_A = FEATURES_V10_A + NEW_FEATURES

# Pattern B adds live features
LIVE_FEATURES = [
    'horse_weight', 'weight_change', 'weight_change_abs', 'weight_cat',
    'weight_cat_dist', 'condition_enc', 'cond_surface',
    'odds_log', 'pop_rank',
    'weather_enc', 'cushion_value', 'moisture_rate',
    'temperature', 'humidity', 'wind_speed', 'precipitation',
]
FEATURES_V101_B = FEATURES_V101_A + LIVE_FEATURES


# ===== NEW FEATURE 1: Growth Curve (age_peak_dist) =====
def compute_growth_curve(df):
    """Compute age-based performance peak score adjusted by distance category.

    Peak age varies by distance:
    - Sprint (dist_cat 0-1): peaks at age 4-5
    - Mile/Middle (dist_cat 2-3): peaks at age 4-5
    - Long (dist_cat 4): peaks at age 5-6

    Uses a Gaussian-like decay from peak, combined with career experience.
    All info is pre-day (age, distance are known before race).
    """
    print("  Computing growth curve features...")

    age = df['age'].values.astype(float)
    dist_cat = df['dist_cat'].values.astype(float)
    career = df['horse_career_races'].values.astype(float)

    # Peak age by distance category
    # Sprint/Mile: peak at 4.5, Long: peak at 5.5
    peak_age = np.where(dist_cat >= 4, 5.5, 4.5)

    # Gaussian-like score: exp(-0.5 * ((age - peak) / sigma)^2)
    # sigma = 1.5 years (gentle decay)
    sigma = 1.5
    age_score = np.exp(-0.5 * ((age - peak_age) / sigma) ** 2)

    # Career experience bonus: log(1 + career_races) / log(1 + 30)
    # Caps at ~30 races (experienced horse)
    exp_bonus = np.log1p(career) / np.log1p(30)
    exp_bonus = np.clip(exp_bonus, 0, 1)

    # Combined: 70% age curve + 30% experience
    df['age_peak_dist'] = (0.7 * age_score + 0.3 * exp_bonus).round(4)

    return df


# ===== NEW FEATURE 2: Jockey Switch Effect =====
def compute_jockey_switch(df):
    """Compute jockey switch win rate differential.

    For each horse, compare current jockey's overall WR with previous
    race's jockey WR. Positive = upgrade, negative = downgrade.
    Uses only pre-race info (jockey assignment known before race).

    Uses expanding window jockey WR already computed (jockey_wr_calc).
    For previous jockey, we use the jockey_wr_calc from the horse's previous race.
    """
    print("  Computing jockey switch features...")

    # Sort by horse, then date
    df = df.sort_values(['horse_id', 'date_num']).copy()

    # Get previous jockey WR for same horse
    df['_prev_jockey_wr'] = df.groupby('horse_id')['jockey_wr_calc'].shift(1)

    # Difference: current jockey WR - previous jockey WR
    # Positive = better jockey, negative = worse jockey
    df['jockey_switch_wr_diff'] = (df['jockey_wr_calc'] - df['_prev_jockey_wr']).fillna(0).round(4)

    # Clean up
    df.drop(columns=['_prev_jockey_wr'], inplace=True)

    return df


# ===== NEW FEATURE 3: Pace Simulation Advantage =====
def compute_pace_sim_advantage(df):
    """Monte Carlo pace scenario simulation.

    For each race, simulate 1000 pace scenarios based on the distribution
    of running styles in the field. Calculate how often each horse's
    running style is advantageous given the pace.

    Pace advantage rules (from running_style × predicted_pace mapping):
    - 逃げ(1) + slow(0) = +2, middle(1) = +1, high(2) = -1
    - 先行(2) + slow(0) = +1, middle(1) = +1, high(2) = 0
    - 差し(3) + slow(0) = -1, middle(1) = 0, high(2) = +1
    - 追込(4) + slow(0) = -2, middle(1) = -1, high(2) = +2

    Monte Carlo: for each race, sample pace from running style distribution
    with noise, then compute average advantage across simulations.
    """
    print("  Computing pace simulation advantage...")

    # Advantage matrix: running_style(1-4) x pace(0-2)
    ADV = {
        (1, 0): 2.0, (1, 1): 1.0, (1, 2): -1.0,
        (2, 0): 1.0, (2, 1): 1.0, (2, 2):  0.0,
        (3, 0):-1.0, (3, 1): 0.0, (3, 2):  1.0,
        (4, 0):-2.0, (4, 1):-1.0, (4, 2):  2.0,
    }

    # Need race_id to group by race
    race_id_col = 'race_id' if 'race_id' in df.columns else 'race_id_str'

    n_sim = 1000
    rng = np.random.RandomState(42)

    # Pre-compute pace distribution per race
    results = np.zeros(len(df))

    for race_id, group in df.groupby(race_id_col):
        idx = group.index
        styles = group['running_style'].values.astype(float)

        # Count front runners (style 1-2) vs closers (style 3-4)
        n_front = np.sum(styles <= 2)
        n_total = len(styles)

        if n_total == 0:
            continue

        front_ratio = n_front / n_total

        # Base pace from front runner ratio
        # More front runners -> higher pace
        # front_ratio > 0.5 -> high pace, < 0.3 -> slow pace
        base_pace_prob = np.array([
            max(0, 0.4 - front_ratio),       # P(slow)
            0.3,                               # P(middle)
            max(0, front_ratio - 0.2),         # P(high)
        ])
        base_pace_prob = base_pace_prob / base_pace_prob.sum()

        # Monte Carlo: sample pace scenarios with noise
        pace_samples = rng.choice([0, 1, 2], size=n_sim, p=base_pace_prob)

        # For each horse in the race, compute average advantage
        for i, style in enumerate(styles):
            s = int(np.clip(style, 1, 4))
            advantages = np.array([ADV.get((s, p), 0.0) for p in pace_samples])
            results[idx[i]] = advantages.mean()

    df['pace_sim_advantage'] = np.round(results, 4)
    return df


# ===== Walk-Forward Backtest =====
def walk_forward_backtest(df, features, label=""):
    """Walk-forward AUC evaluation (2020-2025, yearly folds)."""
    print(f"\n  Walk-Forward Backtest: {label}")
    print(f"  Features: {len(features)}")

    df_wf = df[df['year_full'] >= 2020].copy()
    years = sorted(df_wf['year_full'].unique())

    aucs = []
    for test_year in years:
        train = df_wf[df_wf['year_full'] < test_year]
        test = df_wf[df_wf['year_full'] == test_year]
        if len(train) < 1000 or len(test) < 100:
            continue

        X_tr = train[features].values.astype(np.float32)
        y_tr = train['target'].values
        X_te = test[features].values.astype(np.float32)
        y_te = test['target'].values

        dtrain = lgb.Dataset(X_tr, y_tr)
        dvalid = lgb.Dataset(X_te, y_te)

        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'num_leaves': 63, 'learning_rate': 0.05,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
            'min_child_samples': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'verbose': -1, 'seed': 42,
        }

        model = lgb.train(
            params, dtrain, num_boost_round=1000,
            valid_sets=[dvalid], callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        pred = model.predict(X_te)
        auc = roc_auc_score(y_te, pred)
        aucs.append((test_year, auc))
        print(f"    {test_year}: AUC={auc:.4f} (train={len(train):,}, test={len(test):,})")

    if aucs:
        mean_auc = np.mean([a for _, a in aucs])
        print(f"    Mean WF AUC: {mean_auc:.4f}")
        return aucs, mean_auc
    return [], 0.0


# ===== Full Training =====
def train_full_model(df, features, label=""):
    """Train LGB + XGB ensemble on full time-split."""
    print(f"\n  Training full model: {label} ({len(features)} features)")

    max_year = df['year_full'].max()
    valid_mask = df['year_full'] >= (max_year - 1)
    train_mask = ~valid_mask

    X_train = df.loc[train_mask, features].values.astype(np.float32)
    y_train = df.loc[train_mask, 'target'].values
    X_valid = df.loc[valid_mask, features].values.astype(np.float32)
    y_valid = df.loc[valid_mask, 'target'].values

    print(f"    Train: {len(y_train):,}, Valid: {len(y_valid):,}")

    # LGB
    lgb_model = train_lgb(X_train, y_train, X_valid, y_valid, features)
    lgb_pred = lgb_model.predict(X_valid)
    lgb_auc = roc_auc_score(y_valid, lgb_pred)
    print(f"    LGB AUC: {lgb_auc:.4f}")

    # XGB
    xgb_model = train_xgb(X_train, y_train, X_valid, y_valid)
    xgb_pred = xgb_model.predict(xgb.DMatrix(X_valid))
    xgb_auc = roc_auc_score(y_valid, xgb_pred)
    print(f"    XGB AUC: {xgb_auc:.4f}")

    # Ensemble
    w_lgb = lgb_auc / (lgb_auc + xgb_auc)
    w_xgb = xgb_auc / (lgb_auc + xgb_auc)
    ens_pred = w_lgb * lgb_pred + w_xgb * xgb_pred
    ens_auc = roc_auc_score(y_valid, ens_pred)
    print(f"    Ensemble AUC: {ens_auc:.4f} (LGB={w_lgb:.3f}, XGB={w_xgb:.3f})")

    return lgb_model, xgb_model, ens_auc, w_lgb, w_xgb


def main():
    print("=" * 70)
    print("  KEIBA AI v10.1 TRAINING")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  New features: {NEW_FEATURES}")
    print("=" * 70)

    # === Data Loading ===
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

    print("Building base features...")
    df = build_features(df)
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)
    df = compute_running_style_features(df)

    # === NEW FEATURES ===
    print("\n=== Computing v10.1 new features ===")
    df = compute_growth_curve(df)
    df = compute_jockey_switch(df)
    df = compute_pace_sim_advantage(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()

    # Ensure all features are numeric
    for f in FEATURES_V101_A:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
        else:
            print(f"  WARNING: Feature {f} not found, filling with 0")
            df[f] = 0

    # Column name fix: num_horses_val -> num_horses
    if 'num_horses' not in df.columns and 'num_horses_val' in df.columns:
        df['num_horses'] = df['num_horses_val']

    print(f"\n  Final dataset: {len(df):,} rows, {len(FEATURES_V101_A)} Pattern A features")

    # === Walk-Forward Comparison ===
    print("\n" + "=" * 70)
    print("  WALK-FORWARD COMPARISON")
    print("=" * 70)

    # Baseline: v10 features
    v10_available = [f for f in FEATURES_V10_A if f in df.columns]
    wf_v10, mean_v10 = walk_forward_backtest(df, v10_available, "v10 Baseline")

    # New: v10.1 features
    v101_available = [f for f in FEATURES_V101_A if f in df.columns]
    wf_v101, mean_v101 = walk_forward_backtest(df, v101_available, "v10.1 (3 new features)")

    # Individual feature contribution
    print("\n  --- Feature Ablation ---")
    for feat in NEW_FEATURES:
        ablated = [f for f in v101_available if f != feat]
        _, mean_abl = walk_forward_backtest(df, ablated, f"v10.1 minus {feat}")
        delta = mean_v101 - mean_abl
        print(f"    {feat}: delta={delta:+.4f} ({'USEFUL' if delta > 0 else 'HARMFUL'})")

    # === Decision ===
    print("\n" + "=" * 70)
    print("  DECISION")
    print("=" * 70)
    improvement = mean_v101 - mean_v10
    print(f"  v10  WF AUC: {mean_v10:.4f}")
    print(f"  v10.1 WF AUC: {mean_v101:.4f}")
    print(f"  Improvement: {improvement:+.4f}")

    # Check all years >= 0.78
    all_years_ok = all(auc >= 0.78 for _, auc in wf_v101)
    print(f"  All years >= 0.78: {'YES' if all_years_ok else 'NO'}")

    if improvement <= 0 or not all_years_ok:
        print("\n  *** v10.1 does NOT improve over v10. ABORTING. ***")
        return False

    print(f"\n  *** v10.1 improves by {improvement:+.4f}! Training full models... ***")

    # === Train Pattern A ===
    print("\n" + "=" * 70)
    print("  TRAINING PATTERN A (v10.1)")
    print("=" * 70)

    lgb_a, xgb_a, auc_a, w_lgb_a, w_xgb_a = train_full_model(df, v101_available, "Pattern A")

    # === Condition-based Backtest ===
    print("\n  --- Condition-based Backtest (Pattern A) ---")
    max_year = df['year_full'].max()
    valid_mask = df['year_full'] >= (max_year - 1)
    X_valid_a = df.loc[valid_mask, v101_available].values.astype(np.float32)
    pred_lgb_a = lgb_a.predict(X_valid_a)
    pred_xgb_a = xgb_a.predict(xgb.DMatrix(X_valid_a))
    pred_a = w_lgb_a * pred_lgb_a + w_xgb_a * pred_xgb_a
    df.loc[valid_mask, 'pred'] = pred_a

    df_valid = df[valid_mask].copy()
    all_roi_ok = True
    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        sub = df_valid.copy()
        sub['race_condition'] = sub.apply(
            lambda r: classify_condition(
                {'surface': '芝' if r['surface_enc'] == 0 else 'ダ',
                 'condition': {0: '良', 1: '稍', 2: '重', 3: '不'}.get(int(r.get('condition_enc', 0)), '良')},
                int(r['num_horses_val']), False
            ), axis=1
        )
        cond_df = sub[sub['race_condition'] == cond]
        if len(cond_df) < 10:
            continue
        roi_info = backtest_condition(cond_df, cond)
        if roi_info and roi_info.get('roi', 0) < 100:
            print(f"    WARNING: Condition {cond} ROI={roi_info['roi']:.1f}% < 100%")

    # === Train Pattern B ===
    print("\n" + "=" * 70)
    print("  TRAINING PATTERN B (v10.1)")
    print("=" * 70)

    # Add Pattern B features (defaults for training, real values at prediction time)
    df['weather_enc'] = 0
    df['pop_rank'] = df.groupby('race_id_str')['tansho_odds'].rank(method='min').fillna(8)
    df['cushion_value'] = 0
    df['moisture_rate'] = 0
    df['temperature'] = 0
    df['humidity'] = 0
    df['wind_speed'] = 0
    df['precipitation'] = 0

    # Ensure live features from build_features exist
    for f in LIVE_FEATURES:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    v101_b_available = [f for f in FEATURES_V101_B if f in df.columns]
    lgb_b, xgb_b, auc_b, w_lgb_b, w_xgb_b = train_full_model(df, v101_b_available, "Pattern B")

    # === Save Models ===
    print("\n" + "=" * 70)
    print("  SAVING MODELS")
    print("=" * 70)

    # Pattern A
    model_a_data = {
        'model': lgb_a,
        'xgb_model': xgb_a,
        'features': v101_available,
        'version': 'v10.1',
        'auc': auc_a,
        'ensemble_auc': auc_a,
        'leak_free': True,
        'leak_pattern': 'A',
        'leak_removed': list(LEAK_FEATURES_A),
        'sire_map': sire_map,
        'bms_map': bms_map,
        'n_top_encode': N_TOP_SIRE,
        'trained_at': datetime.now().isoformat(),
        'model_type': 'lgb_xgb_ensemble',
        'ensemble_weights': {'lgb': w_lgb_a, 'xgb': w_xgb_a},
        'course_map': COURSE_MAP,
        'new_features': NEW_FEATURES,
        'wf_aucs': {str(y): a for y, a in wf_v101},
        'wf_mean_auc': mean_v101,
        'baseline_wf_auc': mean_v10,
    }

    path_a = os.path.join(OUTPUT_DIR, 'keiba_model_v9_central.pkl.gz')
    with gzip.open(path_a, 'wb') as f:
        pickle.dump(model_a_data, f)
    print(f"  Pattern A saved: {path_a} (AUC={auc_a:.4f})")

    # Pattern B
    model_b_data = {
        'model': lgb_b,
        'xgb_model': xgb_b,
        'features': v101_b_available,
        'version': 'v10.1_live',
        'auc': auc_b,
        'ensemble_auc': auc_b,
        'pattern_a_auc': auc_a,
        'leak_free': False,
        'leak_pattern': 'B',
        'live_features': LIVE_FEATURES,
        'sire_map': sire_map,
        'bms_map': bms_map,
        'n_top_encode': N_TOP_SIRE,
        'trained_at': datetime.now().isoformat(),
        'n_train': int((~valid_mask).sum()),
        'n_valid': int(valid_mask.sum()),
        'model_type': 'lgb_xgb_ensemble',
        'ensemble_weights': {'lgb': w_lgb_b, 'xgb': w_xgb_b},
        'course_map': COURSE_MAP,
        'weather_map': {'晴': 0, '曇': 1, '小雨': 2, '雨': 2, '雪': 3},
        'new_features': NEW_FEATURES,
    }

    path_b = os.path.join(OUTPUT_DIR, 'keiba_model_v9_central_live.pkl.gz')
    with gzip.open(path_b, 'wb') as f:
        pickle.dump(model_b_data, f)
    print(f"  Pattern B saved: {path_b} (AUC={auc_b:.4f})")

    # === Feature Lookups Update ===
    print("\n  Updating feature_lookups.pkl...")
    lookups_path = os.path.join(BASE_DIR, 'data', 'feature_lookups.pkl')
    if os.path.exists(lookups_path):
        with open(lookups_path, 'rb') as f:
            lookups = pickle.load(f)
    else:
        lookups = {}

    # Update sire/bms maps
    lookups['sire_map'] = sire_map
    lookups['bms_map'] = bms_map

    with open(lookups_path, 'wb') as f:
        pickle.dump(lookups, f)
    print(f"  feature_lookups.pkl updated")

    # === Summary ===
    print("\n" + "=" * 70)
    print("  v10.1 TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Pattern A: {len(v101_available)} features, AUC={auc_a:.4f}")
    print(f"  Pattern B: {len(v101_b_available)} features, AUC={auc_b:.4f}")
    print(f"  WF AUC improvement: {improvement:+.4f} ({mean_v10:.4f} -> {mean_v101:.4f})")
    print(f"  New features: {NEW_FEATURES}")

    # Save results JSON
    results = {
        'version': 'v10.1',
        'date': datetime.now().isoformat(),
        'pattern_a_auc': auc_a,
        'pattern_b_auc': auc_b,
        'wf_aucs': {str(y): a for y, a in wf_v101},
        'wf_mean_auc': mean_v101,
        'baseline_wf_auc': mean_v10,
        'improvement': improvement,
        'new_features': NEW_FEATURES,
        'n_features_a': len(v101_available),
        'n_features_b': len(v101_b_available),
    }
    with open(os.path.join(BASE_DIR, 'data', 'v101_training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return True


if __name__ == '__main__':
    success = main()
    if success:
        print("\n  SUCCESS! v10.1 models saved.")
    else:
        print("\n  FAILED. v10.1 did not improve over v10. No models saved.")
    sys.exit(0 if success else 1)
