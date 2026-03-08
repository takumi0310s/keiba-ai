#!/usr/bin/env python
"""KEIBA AI v9.3 Leak-Free Training (Central)
Adds new features to v9.2a baseline:
- Pace features (prev race first3f/last3f/pace_diff, agari relative)
- Sakaro training time features
- Distance aptitude (expanding window)
- Horse surface aptitude (expanding window)
- Frame advantage by course×distance (expanding window)
All features are pre-day (Pattern A leak-free).
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

sys.path.insert(0, os.path.dirname(__file__))
from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features,
    compute_distance_aptitude, compute_frame_advantage,
    COURSE_MAP, N_TOP_SIRE,
    FEATURES_V92, FEATURES_V93, FEATURES_V93_PKL,
    train_lgb, train_xgb, show_feature_importance,
)
from train_v92_leakfree import (
    LEAK_FEATURES_A, FEATURES_PATTERN_A, FEATURES_PATTERN_A_PKL,
    classify_condition, calc_trio_bets, backtest_condition,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
OUTPUT_DIR = BASE_DIR

# Pattern A leak-free features for V9.3
# All new V9.3 features are pre-day safe (no race-day info)
FEATURES_V93_PATTERN_A = [f for f in FEATURES_V93 if f not in LEAK_FEATURES_A]
FEATURES_V93_PATTERN_A_PKL = [f if f != 'num_horses_val' else 'num_horses' for f in FEATURES_V93_PATTERN_A]

# Identify which features are new vs baseline
NEW_FEATURE_NAMES = [f for f in FEATURES_V93_PATTERN_A if f not in FEATURES_PATTERN_A]


def main():
    print("=" * 70)
    print("  KEIBA AI v9.3 LEAK-FREE TRAINING (CENTRAL)")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  New features: {NEW_FEATURE_NAMES}")
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

    print("Building features...")
    df = build_features(df)

    # These need bracket/build_features output
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)

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

    # === V9.2a Baseline (Pattern A) ===
    print("\n" + "=" * 70)
    print("  BASELINE: V9.2a Pattern A (current production)")
    print(f"  Features: {len(FEATURES_PATTERN_A)}")
    print("=" * 70)

    for f in FEATURES_PATTERN_A:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X_base = df[FEATURES_PATTERN_A].values
    lgb_base = train_lgb(X_base[train_mask], y_train, X_base[valid_mask], y_valid, FEATURES_PATTERN_A)
    lgb_base_pred = lgb_base.predict(X_base[valid_mask])
    lgb_base_auc = roc_auc_score(y_valid, lgb_base_pred)

    import xgboost as xgb
    xgb_base = train_xgb(X_base[train_mask], y_train, X_base[valid_mask], y_valid)
    xgb_base_pred = xgb_base.predict(xgb.DMatrix(X_base[valid_mask]))
    xgb_base_auc = roc_auc_score(y_valid, xgb_base_pred)

    total_base = lgb_base_auc + xgb_base_auc
    w_lgb_base = lgb_base_auc / total_base
    w_xgb_base = xgb_base_auc / total_base
    base_pred = lgb_base_pred * w_lgb_base + xgb_base_pred * w_xgb_base
    base_auc = roc_auc_score(y_valid, base_pred)
    print(f"\n  V9.2a AUC: LGB {lgb_base_auc:.4f} / XGB {xgb_base_auc:.4f} / Ensemble {base_auc:.4f}")

    # === V9.3 Pattern A (all new features) ===
    print("\n" + "=" * 70)
    print("  V9.3 Pattern A (+ new features)")
    print(f"  Features: {len(FEATURES_V93_PATTERN_A)} (+{len(NEW_FEATURE_NAMES)} new)")
    print(f"  New: {NEW_FEATURE_NAMES}")
    print("=" * 70)

    for f in FEATURES_V93_PATTERN_A:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X_v93 = df[FEATURES_V93_PATTERN_A].values
    lgb_v93 = train_lgb(X_v93[train_mask], y_train, X_v93[valid_mask], y_valid, FEATURES_V93_PATTERN_A)
    lgb_v93_pred = lgb_v93.predict(X_v93[valid_mask])
    lgb_v93_auc = roc_auc_score(y_valid, lgb_v93_pred)

    xgb_v93 = train_xgb(X_v93[train_mask], y_train, X_v93[valid_mask], y_valid)
    xgb_v93_pred = xgb_v93.predict(xgb.DMatrix(X_v93[valid_mask]))
    xgb_v93_auc = roc_auc_score(y_valid, xgb_v93_pred)

    total_v93 = lgb_v93_auc + xgb_v93_auc
    w_lgb_v93 = lgb_v93_auc / total_v93
    w_xgb_v93 = xgb_v93_auc / total_v93
    v93_pred = lgb_v93_pred * w_lgb_v93 + xgb_v93_pred * w_xgb_v93
    v93_auc = roc_auc_score(y_valid, v93_pred)
    print(f"\n  V9.3 AUC: LGB {lgb_v93_auc:.4f} / XGB {xgb_v93_auc:.4f} / Ensemble {v93_auc:.4f}")

    fi_v93 = show_feature_importance(lgb_v93, FEATURES_V93_PATTERN_A, "V9.3 Pattern A")

    # === Ablation: test each new feature group ===
    print("\n" + "=" * 70)
    print("  ABLATION: Contribution of each new feature")
    print("=" * 70)

    feature_groups = {
        'pace': ['prev_race_first3f', 'prev_race_last3f', 'prev_race_pace_diff', 'prev_agari_relative'],
        'sakaro': ['sakaro_best_4f_filled', 'sakaro_best_3f_filled', 'has_sakaro_training', 'total_training_count', 'wood_count_2w'],
        'dist_aptitude': ['horse_dist_top3r', 'horse_surface_top3r'],
        'frame_adv': ['frame_course_dist_wr'],
    }

    ablation_results = {}
    for group_name, group_features in feature_groups.items():
        # Remove this group from V9.3 features
        features_without = [f for f in FEATURES_V93_PATTERN_A if f not in group_features]
        for f in features_without:
            if f not in df.columns:
                df[f] = 0
        X_abl = df[features_without].values
        lgb_abl = train_lgb(X_abl[train_mask], y_train, X_abl[valid_mask], y_valid, features_without)
        abl_pred = lgb_abl.predict(X_abl[valid_mask])
        abl_auc = roc_auc_score(y_valid, abl_pred)

        delta = lgb_v93_auc - abl_auc
        ablation_results[group_name] = {
            'features': group_features,
            'auc_without': abl_auc,
            'delta': delta,
            'improves': delta > 0,
        }
        marker = "HELPS" if delta > 0 else "HURTS" if delta < -0.0005 else "NEUTRAL"
        print(f"  {group_name:15s}: remove -> AUC {abl_auc:.4f} (delta {delta:+.4f}) [{marker}]")

    # === Determine best feature set ===
    # Remove feature groups that hurt performance
    features_final = list(FEATURES_V93_PATTERN_A)
    removed_groups = []
    for group_name, result in ablation_results.items():
        if result['delta'] < -0.0005:  # This group hurts
            for f in result['features']:
                if f in features_final:
                    features_final.remove(f)
            removed_groups.append(group_name)

    if removed_groups:
        print(f"\n  Removed hurting groups: {removed_groups}")
        # Retrain with cleaned features
        X_final = df[features_final].values
        lgb_final = train_lgb(X_final[train_mask], y_train, X_final[valid_mask], y_valid, features_final)
        lgb_final_pred = lgb_final.predict(X_final[valid_mask])
        lgb_final_auc = roc_auc_score(y_valid, lgb_final_pred)

        xgb_final = train_xgb(X_final[train_mask], y_train, X_final[valid_mask], y_valid)
        xgb_final_pred = xgb_final.predict(xgb.DMatrix(X_final[valid_mask]))
        xgb_final_auc = roc_auc_score(y_valid, xgb_final_pred)

        total_f = lgb_final_auc + xgb_final_auc
        w_lgb_f = lgb_final_auc / total_f
        w_xgb_f = xgb_final_auc / total_f
        final_pred = lgb_final_pred * w_lgb_f + xgb_final_pred * w_xgb_f
        final_auc = roc_auc_score(y_valid, final_pred)
        print(f"  Final (cleaned) AUC: LGB {lgb_final_auc:.4f} / XGB {xgb_final_auc:.4f} / Ensemble {final_auc:.4f}")
    else:
        lgb_final = lgb_v93
        xgb_final = xgb_v93
        lgb_final_auc = lgb_v93_auc
        xgb_final_auc = xgb_v93_auc
        w_lgb_f = w_lgb_v93
        w_xgb_f = w_xgb_v93
        final_auc = v93_auc
        features_final = list(FEATURES_V93_PATTERN_A)

    features_final_pkl = [f if f != 'num_horses_val' else 'num_horses' for f in features_final]

    # === Summary ===
    print("\n" + "=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)
    print(f"  {'Model':<30} {'LGB AUC':>10} {'XGB AUC':>10} {'Ensemble':>10}")
    print(f"  {'-' * 60}")
    print(f"  {'V9.2a (current production)':<30} {lgb_base_auc:>10.4f} {xgb_base_auc:>10.4f} {base_auc:>10.4f}")
    print(f"  {'V9.3 (all new features)':<30} {lgb_v93_auc:>10.4f} {xgb_v93_auc:>10.4f} {v93_auc:>10.4f}")
    if removed_groups:
        print(f"  {'V9.3 (cleaned)':<30} {lgb_final_auc:>10.4f} {xgb_final_auc:>10.4f} {final_auc:>10.4f}")
    print(f"\n  Improvement: {final_auc - base_auc:+.4f}")

    # === Condition Backtest ===
    valid_df = df[valid_mask].copy()
    bt_final = backtest_condition(valid_df, lgb_final, xgb_final, features_final, w_lgb_f, w_xgb_f, "V9.3 Final")

    # === Save if improved ===
    improved = final_auc > base_auc
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if improved:
        print(f"\n  *** V9.3 IMPROVED! ({final_auc:.4f} > {base_auc:.4f}) ***")
        print(f"  Updating production model...")

        pkl = {
            'model': lgb_final,
            'features': features_final_pkl,
            'version': 'v9.3_leakfree',
            'auc': lgb_final_auc,
            'ensemble_auc': final_auc,
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
            'xgb_model': xgb_final,
            'mlp_model': None,
            'mlp_scaler': None,
            'ensemble_weights': {'lgb': w_lgb_f, 'xgb': w_xgb_f, 'mlp': 0},
            'course_map': dict(COURSE_MAP),
            'condition_backtest': bt_final,
            'new_features_added': NEW_FEATURE_NAMES,
            'ablation_results': ablation_results,
            'removed_groups': removed_groups,
            'baseline_auc': base_auc,
        }

        central_path = os.path.join(OUTPUT_DIR, 'keiba_model_v9_central.pkl')
        with open(central_path, 'wb') as f:
            pickle.dump(pkl, f)
        print(f"  Saved: {central_path}")

        v8_path = os.path.join(OUTPUT_DIR, 'keiba_model_v8.pkl')
        v8_pkl = dict(pkl)
        v8_pkl['auc'] = final_auc
        with open(v8_path, 'wb') as f:
            pickle.dump(v8_pkl, f)
        print(f"  Saved backup: {v8_path}")
    else:
        print(f"\n  V9.3 did NOT improve ({final_auc:.4f} <= {base_auc:.4f})")
        print(f"  Keeping current V9.2a production model.")

    # Save results JSON
    results = {
        'generated_at': now,
        'baseline_v92a': {'lgb': lgb_base_auc, 'xgb': xgb_base_auc, 'ensemble': base_auc},
        'v93_all': {'lgb': lgb_v93_auc, 'xgb': xgb_v93_auc, 'ensemble': v93_auc},
        'v93_final': {'lgb': lgb_final_auc, 'xgb': xgb_final_auc, 'ensemble': final_auc,
                      'n_features': len(features_final)},
        'improved': improved,
        'improvement': final_auc - base_auc,
        'ablation': {k: {'delta': v['delta'], 'improves': v['improves'], 'features': v['features']}
                     for k, v in ablation_results.items()},
        'removed_groups': removed_groups,
        'new_features': NEW_FEATURE_NAMES,
    }

    comp_path = os.path.join(OUTPUT_DIR, 'v93_training_results_central.json')
    with open(comp_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved results: {comp_path}")

    print("\n  Central v9.3 training complete!")
    return results


if __name__ == '__main__':
    main()
