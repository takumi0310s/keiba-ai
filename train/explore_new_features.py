#!/usr/bin/env python
"""Feature Exploration: Test 10 candidate derivative features one by one.
Walk-forward AUC evaluation (Pattern A, leak-free).
Features that don't improve AUC >= 0.8095 baseline are rejected.
Actual ROI is recalculated for accepted features using jra_payouts.csv.
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features,
    compute_distance_aptitude, compute_frame_advantage,
    COURSE_MAP, N_TOP_SIRE,
)
from train_v92_leakfree import FEATURES_PATTERN_A, LEAK_FEATURES_A

from backtest_central_leakfree import (
    classify_condition, get_axes, calc_trio_bets, calc_umaren_bets,
    calc_wide_bets, check_bets, estimate_payouts, encode_sires_fold,
    train_lgb_fold,
)

# Import actual ROI functions from calc_actual_roi
from calc_actual_roi import (
    load_payouts, calc_actual_returns,
)

TEST_YEARS = list(range(2020, 2026))
BASELINE_AUC = 0.8095  # Current production ensemble AUC
BASELINE_WF_AUC = 0.8017  # Current WF average AUC


def prepare_data():
    """Load and prepare all data (shared across all feature tests)."""
    print("=" * 70)
    print("  FEATURE EXPLORATION - Data Preparation")
    print("=" * 70)

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

    print("Building features...")
    df = build_features(df)
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)

    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()

    # Ensure baseline features are numeric
    for f in FEATURES_PATTERN_A:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"  Data ready: {len(df)} rows, {df['race_id_str'].nunique()} races")
    return df


def compute_candidate_features(df):
    """Compute all 10 candidate derivative features."""
    print("\n" + "=" * 70)
    print("  Computing 10 Candidate Features")
    print("=" * 70)

    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    grp = df.groupby('horse_id')

    # 1. avg_finish_last3: 過去3走の平均着順 (already exists as avg_finish_3r, but recompute cleanly)
    # Already in baseline as avg_finish_3r - skip, use a weighted version instead
    # Use exponentially weighted: more recent = higher weight
    df['avg_finish_weighted_3r'] = (
        df['prev_finish'] * 0.5 + df['prev2_finish'] * 0.3 + df['prev3_finish'] * 0.2
    )
    print("  [1] avg_finish_weighted_3r (weighted avg of last 3 finishes)")

    # 2. jockey_dist_wr: 騎手×距離勝率 (expanding window, leak-free)
    print("  [2] jockey_dist_wr (jockey × distance win rate)")
    df = df.sort_values('date_num').reset_index(drop=True)
    global_wr = df['is_win'].mean()
    alpha_jd = 20

    df['dist_cat_jd'] = pd.cut(df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
                                labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
    df['jd_cum_races'] = df.groupby(['jockey_id', 'dist_cat_jd']).cumcount()
    df['jd_cum_wins'] = df.groupby(['jockey_id', 'dist_cat_jd'])['is_win'].cumsum() - df['is_win']
    df['jockey_dist_wr'] = (
        (df['jd_cum_wins'] + alpha_jd * global_wr) /
        (df['jd_cum_races'] + alpha_jd)
    )
    df = df.drop(columns=['dist_cat_jd', 'jd_cum_races', 'jd_cum_wins'])

    # 3. rest_days_log: 前走からの間隔日数 (log transform for better distribution)
    # rest_days already exists, but log-transformed version
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    df['rest_days_log'] = np.log1p(df['rest_days'].clip(1, 365))
    print("  [3] rest_days_log (log-transformed rest days)")

    # 4. prev_agari_3f: 前走上がり3Fタイム (already as prev_last3f, but add relative to distance)
    # prev_last3f exists. Create distance-adjusted version.
    df['prev_last3f_per_dist'] = df['prev_last3f'] / (df['prev_distance'].clip(800, 3600) / 1000)
    print("  [4] prev_last3f_per_dist (prev last3f adjusted by distance)")

    # 5. horse_course_top3r: 同コース過去成績 (expanding window)
    print("  [5] horse_course_top3r (horse × course top3 rate)")
    df = df.sort_values('date_num').reset_index(drop=True)
    global_t3 = df['is_top3'].mean()
    alpha_hc = 3

    df['hc_cum_races'] = df.groupby(['horse_id', 'course_enc']).cumcount()
    df['hc_cum_top3'] = df.groupby(['horse_id', 'course_enc'])['is_top3'].cumsum() - df['is_top3']
    df['horse_course_top3r'] = (
        (df['hc_cum_top3'] + alpha_hc * global_t3) /
        (df['hc_cum_races'] + alpha_hc)
    )
    df = df.drop(columns=['hc_cum_races', 'hc_cum_top3'])

    # 6. horse_exact_dist_top3r: 同距離過去成績 (expanding window, exact distance)
    print("  [6] horse_exact_dist_top3r (horse × exact distance top3 rate)")
    df['hed_cum_races'] = df.groupby(['horse_id', 'distance']).cumcount()
    df['hed_cum_top3'] = df.groupby(['horse_id', 'distance'])['is_top3'].cumsum() - df['is_top3']
    df['horse_exact_dist_top3r'] = (
        (df['hed_cum_top3'] + alpha_hc * global_t3) /
        (df['hed_cum_races'] + alpha_hc)
    )
    df = df.drop(columns=['hed_cum_races', 'hed_cum_top3'])

    # 7. prev_pop_finish_gap: 前走人気と着順の差 (popularity - finish)
    print("  [7] prev_pop_finish_gap (prev popularity - prev finish)")
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    grp = df.groupby('horse_id')
    df['popularity_num'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(8)
    df['prev_popularity'] = grp['popularity_num'].shift(1).fillna(8)
    df['prev_pop_finish_gap'] = df['prev_popularity'] - df['prev_finish']
    # Positive = beat expectations, negative = disappointed

    # 8. weight_carry_change: 斤量変化 (from previous race)
    print("  [8] weight_carry_change (weight carry change from prev race)")
    df['prev_weight_carry'] = grp['weight_carry'].shift(1).fillna(df['weight_carry'])
    df['weight_carry_change'] = df['weight_carry'] - df['prev_weight_carry']

    # 9. class_change: クラス昇降 (class code change from prev race)
    print("  [9] class_change (class level change from prev race)")
    df['class_code_num'] = pd.to_numeric(df['class_code'], errors='coerce').fillna(1)
    df['prev_class'] = grp['class_code_num'].shift(1).fillna(df['class_code_num'])
    df['class_change'] = df['class_code_num'] - df['prev_class']
    # Positive = promoted, negative = demoted

    # 10. season_dist_interaction: 季節×距離の交互作用
    print("  [10] season_dist_interaction (season × distance interaction)")
    df['season_dist'] = df['season'] * 10 + df['dist_cat']

    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)

    print("  All 10 candidate features computed.")
    return df


CANDIDATE_FEATURES = [
    ('avg_finish_weighted_3r', '過去3走加重平均着順'),
    ('jockey_dist_wr', '騎手×距離勝率'),
    ('rest_days_log', '前走間隔(log)'),
    ('prev_last3f_per_dist', '前走上がり3F/距離'),
    ('horse_course_top3r', '同コース過去成績'),
    ('horse_exact_dist_top3r', '同距離過去成績'),
    ('prev_pop_finish_gap', '前走人気-着順差'),
    ('weight_carry_change', '斤量変化'),
    ('class_change', 'クラス昇降'),
    ('season_dist', '季節×距離交互作用'),
]


def walk_forward_auc(df, features, quiet=False):
    """Run walk-forward backtest and return avg AUC + per-year AUCs."""
    fold_aucs = {}

    for test_year in TEST_YEARS:
        train_mask = (df['year_full'] >= 2010) & (df['year_full'] < test_year)
        test_mask = df['year_full'] == test_year
        n_test = test_mask.sum()
        if n_test < 100:
            continue

        df_fold = df.copy()
        df_fold = encode_sires_fold(df_fold, train_mask)
        for f in features:
            if f not in df_fold.columns:
                df_fold[f] = 0
            df_fold[f] = pd.to_numeric(df_fold[f], errors='coerce').fillna(0)

        train_df = df_fold[train_mask]
        y_train = train_df['target'].values
        dates = train_df['date_num']
        valid_cutoff = dates.quantile(0.85)
        tr_idx = dates < valid_cutoff
        va_idx = dates >= valid_cutoff

        X_tr = train_df.loc[tr_idx, features].values
        y_tr = y_train[tr_idx.values]
        X_va = train_df.loc[va_idx, features].values
        y_va = y_train[va_idx.values]

        model = train_lgb_fold(X_tr, y_tr, X_va, y_va, features)

        test_df_fold = df_fold[test_mask].copy()
        X_test = test_df_fold[features].values
        test_df_fold['pred'] = model.predict(X_test)
        test_auc = roc_auc_score(test_df_fold['target'].values, test_df_fold['pred'])
        fold_aucs[test_year] = test_auc

        if not quiet:
            print(f"    {test_year}: AUC {test_auc:.4f}")

    avg_auc = np.mean(list(fold_aucs.values()))
    return avg_auc, fold_aucs


def walk_forward_with_roi(df, features, payout_lookup):
    """Full WF backtest with actual ROI calculation."""
    fold_aucs = {}
    all_results = []
    matched_count = 0
    unmatched_count = 0

    for test_year in TEST_YEARS:
        train_mask = (df['year_full'] >= 2010) & (df['year_full'] < test_year)
        test_mask = df['year_full'] == test_year
        n_test = test_mask.sum()
        if n_test < 100:
            continue

        df_fold = df.copy()
        df_fold = encode_sires_fold(df_fold, train_mask)
        for f in features:
            if f not in df_fold.columns:
                df_fold[f] = 0
            df_fold[f] = pd.to_numeric(df_fold[f], errors='coerce').fillna(0)

        train_df = df_fold[train_mask]
        y_train = train_df['target'].values
        dates = train_df['date_num']
        valid_cutoff = dates.quantile(0.85)
        tr_idx = dates < valid_cutoff
        va_idx = dates >= valid_cutoff

        X_tr = train_df.loc[tr_idx, features].values
        y_tr = y_train[tr_idx.values]
        X_va = train_df.loc[va_idx, features].values
        y_va = y_train[va_idx.values]

        model = train_lgb_fold(X_tr, y_tr, X_va, y_va, features)

        test_df_fold = df_fold[test_mask].copy()
        X_test = test_df_fold[features].values
        test_df_fold['pred'] = model.predict(X_test)
        test_auc = roc_auc_score(test_df_fold['target'].values, test_df_fold['pred'])
        fold_aucs[test_year] = test_auc
        print(f"    {test_year}: AUC {test_auc:.4f}")

        # Evaluate each race
        for rid in test_df_fold['race_id_str'].unique():
            race_df = test_df_fold[test_df_fold['race_id_str'] == rid].copy()
            if len(race_df) < 5:
                continue

            row0 = race_df.iloc[0]
            axes = get_axes(row0)

            race_sorted = race_df.sort_values('finish')
            actual_top3 = {}
            for _, r in race_sorted.head(3).iterrows():
                actual_top3[int(r['finish'])] = int(r['umaban'])
            if len(actual_top3) < 3:
                continue

            race_df = race_df.sort_values('pred', ascending=False)
            ranking = race_df['umaban'].astype(int).tolist()

            trio_bets = calc_trio_bets(ranking)
            umaren_bets = calc_umaren_bets(ranking)
            wide_bets = calc_wide_bets(ranking)

            payout_info = payout_lookup.get(rid)
            (actual_trio_hit, actual_trio_return,
             actual_umaren_hit, actual_umaren_return,
             actual_wide_hits, actual_wide_return) = calc_actual_returns(
                payout_info, trio_bets, umaren_bets, wide_bets)

            if payout_info is not None:
                matched_count += 1
            else:
                unmatched_count += 1

            all_results.append({
                'race_id': rid, 'year': test_year,
                'axes': axes,
                'has_payout': payout_info is not None,
                'actual_trio_hit': actual_trio_hit,
                'actual_trio_return': actual_trio_return,
                'actual_umaren_hit': actual_umaren_hit,
                'actual_umaren_return': actual_umaren_return,
                'actual_wide_hits': actual_wide_hits,
                'actual_wide_return': actual_wide_return,
            })

    avg_auc = np.mean(list(fold_aucs.values()))

    # Calculate condition ROI
    matched_results = [r for r in all_results if r['has_payout']]
    cond_roi = {}
    for cond_key in ['A', 'B', 'C', 'D', 'E', 'X']:
        cond_races = [r for r in matched_results if r['axes']['cond_key'] == cond_key]
        n = len(cond_races)
        if n < 10:
            cond_roi[cond_key] = {'n': n, 'trio_roi': 0, 'hit_rate': 0}
            continue
        inv = n * 700
        trio_pay = sum(r['actual_trio_return'] for r in cond_races)
        hits = sum(1 for r in cond_races if r['actual_trio_hit'])
        cond_roi[cond_key] = {
            'n': n,
            'trio_roi': round(trio_pay / inv * 100, 1),
            'hit_rate': round(hits / n * 100, 1),
        }

    match_rate = matched_count / max(1, matched_count + unmatched_count) * 100
    return avg_auc, fold_aucs, cond_roi, match_rate


def main():
    t_start = time.time()

    # Prepare data
    df = prepare_data()

    # Compute all candidate features
    df = compute_candidate_features(df)

    # === Phase 1: Baseline WF AUC ===
    print("\n" + "=" * 70)
    print("  PHASE 1: Baseline Walk-Forward AUC")
    print(f"  Features: {len(FEATURES_PATTERN_A)} (Pattern A)")
    print("=" * 70)

    baseline_features = list(FEATURES_PATTERN_A)
    baseline_avg, baseline_folds = walk_forward_auc(df, baseline_features)
    print(f"\n  Baseline WF AUC: {baseline_avg:.4f}")
    print(f"  Year AUCs: {', '.join(f'{y}={a:.4f}' for y, a in baseline_folds.items())}")
    all_above_078 = all(a >= 0.78 for a in baseline_folds.values())
    print(f"  All years >= 0.78: {all_above_078}")

    # === Phase 2: Test each candidate feature ===
    print("\n" + "=" * 70)
    print("  PHASE 2: Test Each Candidate Feature (Add 1 at a time)")
    print("=" * 70)

    accepted_features = []
    rejected_features = []
    current_features = list(baseline_features)
    current_avg = baseline_avg

    for feat_name, feat_desc in CANDIDATE_FEATURES:
        print(f"\n  --- Testing: {feat_name} ({feat_desc}) ---")

        # Ensure feature exists and is numeric
        if feat_name not in df.columns:
            print(f"    SKIP: {feat_name} not in DataFrame")
            rejected_features.append((feat_name, feat_desc, 0, 'missing'))
            continue

        df[feat_name] = pd.to_numeric(df[feat_name], errors='coerce').fillna(0)

        test_features = current_features + [feat_name]
        test_avg, test_folds = walk_forward_auc(df, test_features)
        delta = test_avg - current_avg
        all_ok = all(a >= 0.78 for a in test_folds.values())

        print(f"    WF AUC: {test_avg:.4f} (delta: {delta:+.4f})")
        print(f"    All years >= 0.78: {all_ok}")

        if test_avg > current_avg and all_ok:
            print(f"    >>> ACCEPTED: {feat_name} (AUC {current_avg:.4f} → {test_avg:.4f})")
            accepted_features.append((feat_name, feat_desc, delta))
            current_features.append(feat_name)
            current_avg = test_avg
        else:
            reason = 'auc_drop' if test_avg <= current_avg else 'year_below_078'
            print(f"    >>> REJECTED: {feat_name} ({reason})")
            rejected_features.append((feat_name, feat_desc, delta, reason))

    # === Phase 3: Summary of feature selection ===
    print("\n" + "=" * 70)
    print("  PHASE 3: Feature Selection Summary")
    print("=" * 70)
    print(f"\n  Baseline WF AUC: {baseline_avg:.4f}")
    print(f"  Final WF AUC:    {current_avg:.4f}")
    print(f"  Improvement:     {current_avg - baseline_avg:+.4f}")
    print(f"\n  Accepted features ({len(accepted_features)}):")
    for name, desc, delta in accepted_features:
        print(f"    + {name:30s} ({desc}) delta={delta:+.4f}")
    print(f"\n  Rejected features ({len(rejected_features)}):")
    for name, desc, delta, reason in rejected_features:
        print(f"    - {name:30s} ({desc}) delta={delta:+.4f} [{reason}]")

    # === Phase 4: Actual ROI with accepted features ===
    if len(accepted_features) > 0:
        print("\n" + "=" * 70)
        print("  PHASE 4: Actual ROI Calculation (Final Feature Set)")
        print(f"  Features: {len(current_features)}")
        print("=" * 70)

        payout_lookup = load_payouts()
        final_avg, final_folds, cond_roi, match_rate = walk_forward_with_roi(
            df, current_features, payout_lookup)

        print(f"\n  Final WF AUC: {final_avg:.4f}")
        print(f"  Payout match rate: {match_rate:.1f}%")
        print(f"\n  {'Cond':<6} {'N':>5} {'Hit%':>7} {'trio実ROI':>10}")
        print(f"  {'-' * 35}")
        all_above_100 = True
        for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
            if cond in cond_roi:
                info = cond_roi[cond]
                status = '○' if info['trio_roi'] >= 100 else '×'
                if info['trio_roi'] < 100 and info['n'] >= 10:
                    all_above_100 = False
                print(f"  {cond:<6} {info['n']:>5} {info['hit_rate']:>6.1f}% {info['trio_roi']:>9.1f}% {status}")

        print(f"\n  All conditions ROI >= 100%: {all_above_100}")
    else:
        print("\n  No features accepted. Skipping ROI calculation.")
        final_avg = baseline_avg
        final_folds = baseline_folds if 'baseline_folds' in dir() else {}
        cond_roi = {}
        all_above_100 = True
        current_features = baseline_features

    # Also run baseline ROI for comparison
    print("\n" + "=" * 70)
    print("  PHASE 5: Baseline ROI (for comparison)")
    print("=" * 70)
    payout_lookup = load_payouts()
    base_roi_avg, base_roi_folds, base_cond_roi, base_match = walk_forward_with_roi(
        df, baseline_features, payout_lookup)

    print(f"\n  Baseline WF AUC: {base_roi_avg:.4f}")
    print(f"  {'Cond':<6} {'N':>5} {'Hit%':>7} {'trio実ROI':>10}")
    print(f"  {'-' * 35}")
    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        if cond in base_cond_roi:
            info = base_cond_roi[cond]
            status = '○' if info['trio_roi'] >= 100 else '×'
            print(f"  {cond:<6} {info['n']:>5} {info['hit_rate']:>6.1f}% {info['trio_roi']:>9.1f}% {status}")

    # === Save results ===
    elapsed = time.time() - t_start
    results = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': round(elapsed),
        'baseline': {
            'n_features': len(baseline_features),
            'wf_auc': round(baseline_avg, 4),
            'fold_aucs': {str(k): round(v, 4) for k, v in (base_roi_folds if base_roi_folds else baseline_folds).items()},
            'condition_roi': base_cond_roi,
        },
        'final': {
            'n_features': len(current_features),
            'features_added': [name for name, _, _ in accepted_features],
            'wf_auc': round(current_avg, 4),
            'fold_aucs': {str(k): round(v, 4) for k, v in (final_folds if 'final_folds' in dir() else {}).items()},
            'condition_roi': cond_roi if cond_roi else {},
        },
        'accepted': [
            {'name': name, 'description': desc, 'delta_auc': round(delta, 4)}
            for name, desc, delta in accepted_features
        ],
        'rejected': [
            {'name': name, 'description': desc, 'delta_auc': round(delta, 4), 'reason': reason}
            for name, desc, delta, reason in rejected_features
        ],
        'improvement': round(current_avg - baseline_avg, 4),
    }

    out_path = os.path.join(BASE_DIR, 'data', 'feature_exploration_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  Elapsed: {elapsed/60:.1f} min")
    print(f"  Baseline: {len(baseline_features)} features, WF AUC {baseline_avg:.4f}")
    print(f"  Final:    {len(current_features)} features, WF AUC {current_avg:.4f}")
    print(f"  Delta:    {current_avg - baseline_avg:+.4f}")
    print(f"  Accepted: {len(accepted_features)} / {len(CANDIDATE_FEATURES)}")
    if accepted_features:
        for name, desc, delta in accepted_features:
            print(f"    + {name} ({desc}) +{delta:.4f}")
    print("=" * 70)

    return results


if __name__ == '__main__':
    main()
