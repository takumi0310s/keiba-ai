#!/usr/bin/env python
"""Phase 5 Supplement: Feature ablation using Pattern B model (87 features).

Loads the production Pattern B model (v10_live), zero-fills feature groups,
re-predicts on test data (2023-2025), and computes real-payout ROI.
No retraining. Model weights unchanged.
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import pickle
import gzip
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, BASE_DIR)

from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features, COURSE_MAP, N_TOP_SIRE,
)
from backtest_central_leakfree import (
    classify_condition, calc_trio_bets, encode_sires_fold, train_lgb_fold,
)
from calc_actual_roi import (
    load_payouts, get_actual_payouts, calc_actual_returns,
)

TEST_YEARS = [2023, 2024, 2025]


def load_pattern_b_model():
    path = os.path.join(BASE_DIR, 'keiba_model_v9_central_live.pkl.gz')
    with gzip.open(path, 'rb') as f:
        pkl = pickle.load(f)
    print(f"  Model: {pkl['version']}, {len(pkl['features'])} features, pattern={pkl['leak_pattern']}")
    return pkl


def prepare_data(model_features):
    """Prepare data with ALL features needed for Pattern B."""
    print("  Loading data...")
    df = load_data()
    lap_df = load_lap_data()
    if lap_df is not None:
        df = df.merge(lap_df, on='race_id_str', how='left')
    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)
    tt_data = load_training_times()
    df = merge_training_features(df, tt_data)
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)
    df = compute_lag_features(df)
    df = build_features(df)

    # Pattern B needs additional features that Pattern A doesn't:
    # odds_log, horse_weight, condition_enc, weight_change, weight_change_abs,
    # weight_cat, weight_cat_dist, cond_surface, pop_rank, weather_enc,
    # cushion_value, moisture_rate, temperature, humidity, wind_speed, precipitation

    # Map model feature names back to df column names
    feat_map = {'num_horses': 'num_horses_val'}
    actual_features = []
    for f in model_features:
        col = feat_map.get(f, f)
        actual_features.append(col)
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()
    print(f"  Data: {len(df)} rows")
    return df, actual_features


def predict_ensemble(pkl, X):
    """Predict using Pattern B ensemble (LGB + XGB)."""
    lgb_model = pkl['model']
    xgb_model = pkl.get('xgb_model')
    weights = pkl.get('ensemble_weights', {})
    w_lgb = weights.get('lgb', 0.5)
    w_xgb = weights.get('xgb', 0.5)

    lgb_pred = lgb_model.predict(X)
    if xgb_model is not None:
        xgb_pred = xgb_model.predict(xgb.DMatrix(X))
        pred = lgb_pred * w_lgb + xgb_pred * w_xgb
    else:
        pred = lgb_pred
    return pred


def run_ablation(df, actual_features, pkl, payout_lookup, zero_indices, label):
    """Run backtest with specified feature indices zeroed out."""
    all_results = []

    for test_year in TEST_YEARS:
        test_mask = df['year_full'] == test_year
        test_df = df[test_mask].copy()
        if len(test_df) < 100:
            continue

        X = test_df[actual_features].values.copy()
        # Zero-fill specified indices
        for idx in zero_indices:
            X[:, idx] = 0

        test_df['pred'] = predict_ensemble(pkl, X)

        for rid in test_df['race_id_str'].unique():
            race_df = test_df[test_df['race_id_str'] == rid].copy()
            if len(race_df) < 5:
                continue
            race_sorted = race_df.sort_values('finish')
            actual_top3 = {}
            for _, r in race_sorted.head(3).iterrows():
                actual_top3[int(r['finish'])] = int(r['umaban'])
            if len(actual_top3) < 3:
                continue
            race_df = race_df.sort_values('pred', ascending=False)
            ranking = race_df['umaban'].astype(int).tolist()
            trio_bets = calc_trio_bets(ranking)
            payout_info = get_actual_payouts(payout_lookup, rid)
            if payout_info is None:
                continue
            (trio_hit, trio_return, _, _, _, _) = calc_actual_returns(
                payout_info, trio_bets, [], [])
            all_results.append({'trio_hit': trio_hit, 'trio_return': trio_return})

    n = len(all_results)
    if n == 0:
        return 0, 0, 0
    hits = sum(1 for r in all_results if r['trio_hit'])
    total_ret = sum(r['trio_return'] for r in all_results)
    inv = n * 700
    roi = round(total_ret / inv * 100, 1) if inv > 0 else 0
    hr = round(hits / n * 100, 1)
    return roi, hr, n


def main():
    t0 = time.time()
    print("=" * 70)
    print("  PHASE 5 SUPPLEMENT: PATTERN B FEATURE ABLATION")
    print("=" * 70)

    pkl = load_pattern_b_model()
    model_features = pkl['features']  # 87 features with model-expected names

    df, actual_features = prepare_data(model_features)
    payout_lookup = load_payouts()

    # Baseline: no zeroing
    print("\n  Baseline (Pattern B, all features)...")
    baseline_roi, baseline_hr, baseline_n = run_ablation(
        df, actual_features, pkl, payout_lookup, [], "Baseline")
    print(f"  Baseline: ROI={baseline_roi:.1f}%, Hit={baseline_hr:.1f}%, N={baseline_n}")

    # Define feature groups by index in model_features
    feat_to_idx = {f: i for i, f in enumerate(model_features)}

    groups = {
        'odds': ['odds_log', 'pop_rank', 'prev_odds_log'],
        'odds_current_only': ['odds_log', 'pop_rank'],
        'blood': ['sire_enc', 'bms_enc', 'sire_surface_wr', 'sire_dist_wr', 'bms_surface_wr'],
        'jockey': ['jockey_wr_calc', 'jockey_course_wr_calc', 'jockey_surface_wr'],
        'past_performance': [
            'prev_finish', 'prev2_finish', 'prev3_finish', 'prev_last3f', 'prev2_last3f',
            'prev_pass4', 'prev_prize', 'avg_finish_3r', 'best_finish_3r', 'finish_trend',
            'top3_count_3r', 'avg_last3f_3r', 'horse_career_races', 'horse_career_wr',
            'horse_career_top3r', 'horse_dist_top3r', 'horse_surface_top3r',
        ],
        'pace': [
            'prev_race_first3f', 'prev_race_last3f', 'prev_race_pace_diff',
            'prev_agari_relative', 'running_style', 'predicted_pace',
            'pace_advantage', 'style_vs_pace',
        ],
        'training': [
            'training_time_filled', 'has_training', 'training_per_dist',
            'wood_best_4f_filled', 'has_wood_training', 'wood_count_2w',
            'sakaro_best_4f_filled', 'sakaro_best_3f_filled', 'has_sakaro_training',
            'total_training_count',
        ],
        'weight': ['horse_weight', 'weight_change', 'weight_change_abs',
                    'weight_cat', 'weight_cat_dist'],
        'track_weather': ['condition_enc', 'cond_surface', 'weather_enc',
                          'cushion_value', 'moisture_rate', 'temperature',
                          'humidity', 'wind_speed', 'precipitation'],
    }

    results = {
        'model': pkl['version'],
        'n_features': len(model_features),
        'pattern': 'B',
        'baseline': {'roi': baseline_roi, 'hit_rate': baseline_hr, 'races': baseline_n},
        'ablations': {},
    }

    print(f"\n  {'Group':25s} {'Features':>4} {'ROI':>8} {'Change':>8} {'Change%':>8} {'Hit%':>6}")
    print(f"  {'-'*65}")
    print(f"  {'BASELINE':25s} {'--':>4} {baseline_roi:>7.1f}% {'--':>8} {'--':>8} {baseline_hr:>5.1f}%")

    for group_name, feat_names in groups.items():
        indices = [feat_to_idx[f] for f in feat_names if f in feat_to_idx]
        if not indices:
            print(f"  {group_name:25s}: no matching features")
            continue

        roi, hr, n = run_ablation(df, actual_features, pkl, payout_lookup, indices, group_name)
        change = roi - baseline_roi
        change_pct = change / baseline_roi * 100 if baseline_roi > 0 else 0

        results['ablations'][group_name] = {
            'roi': roi,
            'hit_rate': hr,
            'roi_change': round(change, 1),
            'roi_change_pct': round(change_pct, 1),
            'features_zeroed': [f for f in feat_names if f in feat_to_idx],
            'n_features': len(indices),
        }
        print(f"  {group_name:25s} {len(indices):>4} {roi:>7.1f}% {change:>+7.1f}% {change_pct:>+7.1f}% {hr:>5.1f}%")

    # Most important
    if results['ablations']:
        most_imp = min(results['ablations'].items(), key=lambda x: x[1]['roi'])
        results['most_important_group'] = most_imp[0]
        print(f"\n  Most important group: {most_imp[0]} (ROI drops to {most_imp[1]['roi']:.1f}%)")

    # Save to phase5 report
    report_path = os.path.join(BASE_DIR, 'data', 'phase5_market_resilience.json')
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    report['feature_ablation_pattern_b'] = results
    report['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Saved to: {report_path}")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()
