#!/usr/bin/env python
"""KEIBA AI v9.2 Leak-Free Training
Pattern A: Strict leak-free (pre-day info only) - removes odds_log, horse_weight, condition_enc
Pattern B: Pre-race info OK - removes only odds_log (final confirmed odds)
Compares AUC for both patterns against original V9.2.
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Reuse data loading and feature computation from v92
sys.path.insert(0, os.path.dirname(__file__))
from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features, COURSE_MAP, N_TOP_SIRE,
    FEATURES_V92, FEATURES_V93, FEATURES_V92_PKL,
    train_lgb, train_xgb, show_feature_importance,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
OUTPUT_DIR = BASE_DIR

# === LEAK ANALYSIS ===
# odds_log: FINAL confirmed odds (確定オッズ) → CRITICAL LEAK (1.2M gain importance)
# prev_odds_log: Previous race's final odds → CLEAN (historical, publicly known)
# horse_weight: Announced 70min before race → Race-day info (not pre-day)
# condition_enc: Track condition, morning announcement → Race-day info
# weight_change/weight_change_abs: Derived from horse_weight → Race-day
# weight_cat: Derived from horse_weight → Race-day
# weight_cat_dist: Derived from weight_cat → Race-day
# cond_surface: Derived from condition_enc → Race-day

# Pattern A: Strict leak-free (pre-day info only)
LEAK_FEATURES_A = {
    'odds_log',          # FINAL confirmed odds - critical leak
    'horse_weight',      # Announced 70min before race
    'condition_enc',     # Morning of race day
    'weight_change',     # Derived from horse_weight
    'weight_change_abs', # Derived from horse_weight
    'weight_cat',        # Derived from horse_weight
    'weight_cat_dist',   # Derived from weight_cat
    'cond_surface',      # Derived from condition_enc
}

# Pattern B: Pre-race info OK (remove only truly leaked features)
LEAK_FEATURES_B = {
    'odds_log',          # FINAL confirmed odds - only this is truly leaked
}

# Use V9.3 features (67 Pattern A features = V9.3 all - leak features)
FEATURES_PATTERN_A = [f for f in FEATURES_V93 if f not in LEAK_FEATURES_A]
FEATURES_PATTERN_B = [f for f in FEATURES_V93 if f not in LEAK_FEATURES_B]
FEATURES_PATTERN_A_PKL = [f if f != 'num_horses_val' else 'num_horses' for f in FEATURES_PATTERN_A]
FEATURES_PATTERN_B_PKL = [f if f != 'num_horses_val' else 'num_horses' for f in FEATURES_PATTERN_B]


def classify_condition(num_horses, distance, condition_enc):
    """Classify race condition for betting strategy."""
    heavy = condition_enc >= 2  # 重 or 不
    if num_horses <= 7:
        return 'E'
    if distance <= 1400:
        return 'D'
    if 8 <= num_horses <= 14 and distance >= 1600 and not heavy:
        return 'A'
    if 8 <= num_horses <= 14 and distance >= 1600 and heavy:
        return 'B'
    if num_horses >= 15 and distance >= 1600 and not heavy:
        return 'C'
    return 'X'


def calc_trio_bets(ranking):
    """Calculate trio betting combinations (TOP1 axis)."""
    if len(ranking) < 3:
        return []
    nums = ranking[:6] if len(ranking) >= 6 else ranking
    n1 = nums[0]
    second = nums[1:3]
    third = nums[1:min(6, len(nums))]
    bets = sorted(set(
        tuple(sorted({n1, s, t}))
        for s in second for t in third
        if len(set({n1, s, t})) == 3
    ))
    return bets


def backtest_condition(df, model_lgb, model_xgb, feature_cols, w_lgb, w_xgb, label=""):
    """Run condition-based backtest on validation data."""
    print(f"\n  --- {label} Condition Backtest ---")

    X = df[feature_cols].values
    scores = model_lgb.predict(X)
    if model_xgb:
        import xgboost as xgb
        xgb_scores = model_xgb.predict(xgb.DMatrix(X))
        scores = scores * w_lgb + xgb_scores * w_xgb

    df = df.copy()
    df['score'] = scores

    condition_results = {}
    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        condition_results[cond] = {'n': 0, 'trio_hits': 0, 'trio_total': 0}

    for rid, race_df in df.groupby('race_id_str'):
        race_df = race_df.sort_values('score', ascending=False)
        if len(race_df) < 3:
            continue

        num_horses = int(race_df['num_horses_val'].iloc[0])
        distance = int(race_df['distance'].iloc[0])
        cond_enc = int(race_df['condition_enc'].iloc[0]) if 'condition_enc' in race_df.columns else 0
        cond_key = classify_condition(num_horses, distance, cond_enc)

        ranking = race_df['umaban'].astype(int).tolist()
        actual_top3 = set(race_df[race_df['finish'] <= 3]['umaban'].astype(int).tolist())

        trio_bets = calc_trio_bets(ranking)
        trio_hit = any(set(combo) == actual_top3 for combo in trio_bets) if len(actual_top3) == 3 else False

        cr = condition_results[cond_key]
        cr['n'] += 1
        cr['trio_total'] += len(trio_bets) * 100
        if trio_hit:
            cr['trio_hits'] += 1

    print(f"  {'COND':<4} {'N':>5} {'TRIO_HIT':>8} {'HIT_RATE':>8}")
    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        cr = condition_results[cond]
        n = cr['n']
        if n == 0:
            print(f"  {cond:<4} {'N/A':>5}")
            continue
        hit_rate = cr['trio_hits'] / n * 100
        print(f"  {cond:<4} {n:>5} {cr['trio_hits']:>8} {hit_rate:>7.1f}%")

    return condition_results


def main():
    print("=" * 70)
    print("  KEIBA AI v9.2 LEAK-FREE TRAINING")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # === Data Loading (same as v9.2) ===
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

    results = {}

    # === Original V9.2 (with leak) for reference ===
    print("\n" + "=" * 70)
    print("  ORIGINAL V9.2 (with odds_log leak) - Reference")
    print("=" * 70)

    for f in FEATURES_V92:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X_orig = df[FEATURES_V92].values
    lgb_orig = train_lgb(X_orig[train_mask], y_train, X_orig[valid_mask], y_valid, FEATURES_V92)
    lgb_orig_pred = lgb_orig.predict(X_orig[valid_mask])
    lgb_orig_auc = roc_auc_score(y_valid, lgb_orig_pred)

    import xgboost as xgb
    xgb_orig = train_xgb(X_orig[train_mask], y_train, X_orig[valid_mask], y_valid)
    xgb_orig_pred = xgb_orig.predict(xgb.DMatrix(X_orig[valid_mask]))
    xgb_orig_auc = roc_auc_score(y_valid, xgb_orig_pred)

    total = lgb_orig_auc + xgb_orig_auc
    w_lgb_orig = lgb_orig_auc / total
    w_xgb_orig = xgb_orig_auc / total
    orig_pred = lgb_orig_pred * w_lgb_orig + xgb_orig_pred * w_xgb_orig
    orig_auc = roc_auc_score(y_valid, orig_pred)
    print(f"\n  Original V9.2 AUC: LGB {lgb_orig_auc:.4f} / XGB {xgb_orig_auc:.4f} / Ensemble {orig_auc:.4f}")
    fi_orig = show_feature_importance(lgb_orig, FEATURES_V92, "Original V9.2 (with leak)")
    results['original'] = {'lgb': lgb_orig_auc, 'xgb': xgb_orig_auc, 'ensemble': orig_auc}

    # === Pattern A: Strict leak-free ===
    print("\n" + "=" * 70)
    print("  PATTERN A: Strict Leak-Free (pre-day info only)")
    print(f"  Removed: {sorted(LEAK_FEATURES_A)}")
    print(f"  Features: {len(FEATURES_PATTERN_A)} (was {len(FEATURES_V92)})")
    print("=" * 70)

    for f in FEATURES_PATTERN_A:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X_a = df[FEATURES_PATTERN_A].values
    lgb_a = train_lgb(X_a[train_mask], y_train, X_a[valid_mask], y_valid, FEATURES_PATTERN_A)
    lgb_a_pred = lgb_a.predict(X_a[valid_mask])
    lgb_a_auc = roc_auc_score(y_valid, lgb_a_pred)

    xgb_a = train_xgb(X_a[train_mask], y_train, X_a[valid_mask], y_valid)
    xgb_a_pred = xgb_a.predict(xgb.DMatrix(X_a[valid_mask]))
    xgb_a_auc = roc_auc_score(y_valid, xgb_a_pred)

    total_a = lgb_a_auc + xgb_a_auc
    w_lgb_a = lgb_a_auc / total_a
    w_xgb_a = xgb_a_auc / total_a
    a_pred = lgb_a_pred * w_lgb_a + xgb_a_pred * w_xgb_a
    a_auc = roc_auc_score(y_valid, a_pred)
    print(f"\n  Pattern A AUC: LGB {lgb_a_auc:.4f} / XGB {xgb_a_auc:.4f} / Ensemble {a_auc:.4f}")
    fi_a = show_feature_importance(lgb_a, FEATURES_PATTERN_A, "Pattern A (Strict Leak-Free)")
    results['pattern_a'] = {'lgb': lgb_a_auc, 'xgb': xgb_a_auc, 'ensemble': a_auc}

    # Condition backtest Pattern A
    valid_df = df[valid_mask].copy()
    bt_a = backtest_condition(valid_df, lgb_a, xgb_a, FEATURES_PATTERN_A, w_lgb_a, w_xgb_a, "Pattern A")

    # === Pattern B: Pre-race info OK ===
    print("\n" + "=" * 70)
    print("  PATTERN B: Pre-race Info OK (remove only final odds)")
    print(f"  Removed: {sorted(LEAK_FEATURES_B)}")
    print(f"  Features: {len(FEATURES_PATTERN_B)} (was {len(FEATURES_V92)})")
    print("=" * 70)

    for f in FEATURES_PATTERN_B:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X_b = df[FEATURES_PATTERN_B].values
    lgb_b = train_lgb(X_b[train_mask], y_train, X_b[valid_mask], y_valid, FEATURES_PATTERN_B)
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
    fi_b = show_feature_importance(lgb_b, FEATURES_PATTERN_B, "Pattern B (Pre-race OK)")
    results['pattern_b'] = {'lgb': lgb_b_auc, 'xgb': xgb_b_auc, 'ensemble': b_auc}

    # Condition backtest Pattern B
    bt_b = backtest_condition(valid_df, lgb_b, xgb_b, FEATURES_PATTERN_B, w_lgb_b, w_xgb_b, "Pattern B")

    # === Summary ===
    print("\n" + "=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)
    print(f"  {'Pattern':<30} {'LGB AUC':>10} {'XGB AUC':>10} {'Ensemble':>10}")
    print(f"  {'-' * 60}")
    print(f"  {'Original V9.2 (with leak)':<30} {results['original']['lgb']:>10.4f} {results['original']['xgb']:>10.4f} {results['original']['ensemble']:>10.4f}")
    print(f"  {'Pattern A (strict leak-free)':<30} {results['pattern_a']['lgb']:>10.4f} {results['pattern_a']['xgb']:>10.4f} {results['pattern_a']['ensemble']:>10.4f}")
    print(f"  {'Pattern B (pre-race OK)':<30} {results['pattern_b']['lgb']:>10.4f} {results['pattern_b']['xgb']:>10.4f} {results['pattern_b']['ensemble']:>10.4f}")
    print(f"\n  AUC drop from leak removal:")
    print(f"  Pattern A: {results['pattern_a']['ensemble'] - results['original']['ensemble']:+.4f}")
    print(f"  Pattern B: {results['pattern_b']['ensemble'] - results['original']['ensemble']:+.4f}")

    # === Save Pattern A as production (conservative) ===
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save Pattern A
    pkl_a = {
        'model': lgb_a,
        'features': FEATURES_PATTERN_A_PKL,
        'version': 'v9.2a_leakfree',
        'auc': lgb_a_auc,
        'ensemble_auc': a_auc,
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
        'xgb_model': xgb_a,
        'mlp_model': None,
        'mlp_scaler': None,
        'ensemble_weights': {'lgb': w_lgb_a, 'xgb': w_xgb_a, 'mlp': 0},
        'course_map': dict(COURSE_MAP),
        'condition_backtest': bt_a,
    }

    central_path = os.path.join(OUTPUT_DIR, 'keiba_model_v9_central.pkl')
    with open(central_path, 'wb') as f:
        pickle.dump(pkl_a, f)
    print(f"\n  Saved Pattern A (production): {central_path}")

    # Also save as v8 backup
    v8_path = os.path.join(OUTPUT_DIR, 'keiba_model_v8.pkl')
    v8_pkl = dict(pkl_a)
    v8_pkl['auc'] = a_auc
    with open(v8_path, 'wb') as f:
        pickle.dump(v8_pkl, f)
    print(f"  Saved Pattern A (v8 backup): {v8_path}")

    # Save Pattern B for reference
    pkl_b = {
        'model': lgb_b,
        'features': FEATURES_PATTERN_B_PKL,
        'version': 'v9.2b_prerace',
        'auc': lgb_b_auc,
        'ensemble_auc': b_auc,
        'leak_free': True,
        'leak_pattern': 'B',
        'leak_removed': sorted(LEAK_FEATURES_B),
        'sire_map': sire_map,
        'bms_map': bms_map,
        'n_top_encode': N_TOP_SIRE,
        'trained_at': now,
        'n_train': int(train_mask.sum()),
        'n_valid': int(valid_mask.sum()),
        'model_type': 'central',
        'xgb_model': xgb_b,
        'mlp_model': None,
        'mlp_scaler': None,
        'ensemble_weights': {'lgb': w_lgb_b, 'xgb': w_xgb_b, 'mlp': 0},
        'course_map': dict(COURSE_MAP),
        'condition_backtest': bt_b,
    }

    b_path = os.path.join(OUTPUT_DIR, 'keiba_model_v92b_central.pkl')
    with open(b_path, 'wb') as f:
        pickle.dump(pkl_b, f)
    print(f"  Saved Pattern B (reference): {b_path}")

    # Save comparison results
    comparison = {
        'generated_at': now,
        'original_v92': results['original'],
        'pattern_a': {
            **results['pattern_a'],
            'features_removed': sorted(LEAK_FEATURES_A),
            'n_features': len(FEATURES_PATTERN_A),
            'condition_backtest': {k: {'n': v['n'], 'trio_hits': v['trio_hits'],
                                        'hit_rate': v['trio_hits']/v['n']*100 if v['n']>0 else 0}
                                   for k, v in bt_a.items()},
        },
        'pattern_b': {
            **results['pattern_b'],
            'features_removed': sorted(LEAK_FEATURES_B),
            'n_features': len(FEATURES_PATTERN_B),
            'condition_backtest': {k: {'n': v['n'], 'trio_hits': v['trio_hits'],
                                        'hit_rate': v['trio_hits']/v['n']*100 if v['n']>0 else 0}
                                   for k, v in bt_b.items()},
        },
    }

    comp_path = os.path.join(OUTPUT_DIR, 'leak_comparison_central.json')
    with open(comp_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"  Saved comparison: {comp_path}")

    print("\n  Central leak-free training complete!")
    return results


if __name__ == '__main__':
    main()
