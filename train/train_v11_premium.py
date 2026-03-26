#!/usr/bin/env python
"""V11 Premium Features Training: Add speed index + training 1F to model

Tests 4 patterns:
1. Baseline (current 87 features)
2. +index_max, +index_run1 (89 features)
3. +time_1f_last (88 features)
4. All three (90 features)

Walk-forward validation + actual ROI comparison.
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
from datetime import datetime

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
    train_lgb, train_xgb,
)
from train_v92_leakfree import FEATURES_PATTERN_A, LEAK_FEATURES_A
from backtest_central_leakfree import (
    classify_condition, calc_trio_bets, encode_sires_fold, train_lgb_fold,
)

DATA_DIR = os.path.join(BASE_DIR, 'data')

# New premium features
NEW_FEATURES = ['index_max', 'index_run1', 'time_1f_last']

report = {
    'generated_at': None,
    'data_coverage': {},
    'patterns': {},
    'walk_forward': {},
    'roi_comparison': {},
    'adoption': {},
}


def save_report():
    report['generated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    path = os.path.join(DATA_DIR, 'v11_premium_features_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Report saved: {path}")


def jra_to_nk(rid):
    """Convert TARGET race_id (10 digit) to netkeiba format (12 digit)."""
    rid = str(rid).zfill(10)
    return f"20{rid[2:4]}{rid[0:2]}{rid[4:5].zfill(2)}{rid[5:6].zfill(2)}{rid[6:8]}"


def load_premium_data():
    """Load and prepare premium data for merging."""
    print("  Loading premium data...")

    # Speed index
    si_path = os.path.join(DATA_DIR, 'netkeiba_speed_index.csv')
    si = pd.read_csv(si_path, encoding='utf-8-sig', dtype=str)
    si['umaban'] = pd.to_numeric(si['umaban'], errors='coerce')
    si['index_max'] = pd.to_numeric(si['index_max'], errors='coerce').fillna(0)
    si['index_run1'] = pd.to_numeric(si['index_run1'], errors='coerce').fillna(0)
    # Replace 1000 (no data marker) with 0
    si.loc[si['index_max'] <= 1000, 'index_max'] = 0
    si.loc[si['index_run1'] <= 1000, 'index_run1'] = 0
    print(f"    Speed index: {len(si)} rows, {si['race_id'].nunique()} races")

    # Training times (for time_1f_last)
    tt_path = os.path.join(DATA_DIR, 'netkeiba_training_times.csv')
    tt = pd.read_csv(tt_path, encoding='utf-8-sig', dtype=str)
    tt['umaban'] = pd.to_numeric(tt['umaban'], errors='coerce')
    tt['time_1f'] = pd.to_numeric(tt['time_1f'], errors='coerce').fillna(0)
    # Keep only valid 1F times
    tt = tt[tt['time_1f'] > 0].copy()
    tt = tt.rename(columns={'time_1f': 'time_1f_last'})
    # Keep best (fastest) 1F per race+horse
    tt_best = tt.groupby(['race_id', 'umaban'])['time_1f_last'].min().reset_index()
    print(f"    Training 1F: {len(tt_best)} rows, {tt_best['race_id'].nunique()} races")

    return si[['race_id', 'umaban', 'index_max', 'index_run1']], tt_best


def merge_premium_features(df, si_df, tt_df):
    """Merge premium features into the main dataframe."""
    # Create netkeiba race_id for matching
    df['nk_race_id'] = df['race_id'].apply(lambda x: jra_to_nk(str(x).split('_')[0] if '_' in str(x) else x))
    df['umaban_int'] = pd.to_numeric(df['umaban'], errors='coerce')

    # Merge speed index
    merged = df.merge(
        si_df, left_on=['nk_race_id', 'umaban_int'], right_on=['race_id', 'umaban'],
        how='left', suffixes=('', '_si'))

    # Merge training 1F
    merged = merged.merge(
        tt_df, left_on=['nk_race_id', 'umaban_int'], right_on=['race_id', 'umaban'],
        how='left', suffixes=('', '_tt'))

    # Fill NaN with median (computed from available data only)
    for col in NEW_FEATURES:
        if col in merged.columns:
            valid = merged[col][merged[col] > 0]
            median_val = valid.median() if len(valid) > 0 else 0
            merged[col] = merged[col].fillna(0)
            merged.loc[merged[col] == 0, col] = median_val
            coverage = (merged[col] != median_val).mean() * 100
            print(f"    {col}: coverage={coverage:.1f}%, median_fill={median_val:.1f}")
        else:
            merged[col] = 0
            print(f"    {col}: NOT FOUND in merge, all zeros")

    # Clean up
    for col in ['race_id_si', 'umaban_si', 'race_id_tt', 'umaban_tt', 'nk_race_id', 'umaban_int']:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)

    return merged


def run_walk_forward(df, features, test_years, label=""):
    """Walk-forward validation with actual ROI."""
    print(f"\n  WF: {label} ({len(features)} features)")
    fold_aucs = {}
    all_results = []

    # Load payouts if available
    payout_lookup = None
    try:
        from calc_actual_roi import load_payouts, get_actual_payouts, calc_actual_returns
        payout_lookup = load_payouts()
    except Exception:
        print("    WARNING: jra_payouts.csv not available")

    for test_year in test_years:
        yr2 = test_year % 100 if test_year > 100 else test_year
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
        vc = dates.quantile(0.85)

        model = train_lgb_fold(
            train_df.loc[dates < vc, features].values, y_train[(dates < vc).values],
            train_df.loc[dates >= vc, features].values, y_train[(dates >= vc).values],
            features)

        test_df = df_fold[test_mask].copy()
        test_df['pred'] = model.predict(test_df[features].values)
        auc = roc_auc_score(test_df['target'].values, test_df['pred'])
        fold_aucs[test_year] = round(auc, 4)

        # ROI calculation
        if payout_lookup:
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
                cond_key = classify_condition(race_df.iloc[0])
                all_results.append({
                    'year': test_year, 'cond_key': cond_key,
                    'trio_hit': trio_hit, 'trio_return': trio_return,
                })

        print(f"    {test_year}: AUC={auc:.4f} (n={n_test:,})")

    avg_auc = round(np.mean(list(fold_aucs.values())), 4) if fold_aucs else 0

    # ROI summary
    roi_by_cond = {}
    if all_results:
        for ck in ['A', 'B', 'C', 'D', 'E', 'X']:
            cr = [r for r in all_results if r['cond_key'] == ck]
            if not cr:
                continue
            n = len(cr)
            hits = sum(1 for r in cr if r['trio_hit'])
            ret = sum(r['trio_return'] for r in cr)
            inv = n * 700
            roi_by_cond[ck] = {
                'n': n, 'hits': hits,
                'roi': round(ret / inv * 100, 1) if inv > 0 else 0,
                'hit_rate': round(hits / n * 100, 1) if n > 0 else 0,
            }

    return {
        'fold_aucs': fold_aucs,
        'avg_auc': avg_auc,
        'all_above_078': all(a >= 0.78 for a in fold_aucs.values()),
        'n_features': len(features),
        'roi_by_condition': roi_by_cond,
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V11 PREMIUM FEATURES TRAINING")
    print("=" * 70)

    # Load and prepare base data
    print("\n[1] Loading base data...")
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
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()

    features_base = list(FEATURES_PATTERN_A)
    for f in features_base:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"  Base data: {len(df)} rows")

    # Load premium data
    print("\n[2] Loading premium data...")
    si_df, tt_df = load_premium_data()

    # Merge
    print("\n[3] Merging premium features...")
    df = merge_premium_features(df, si_df, tt_df)

    # Check coverage by year
    print("\n[4] Coverage analysis...")
    for yr in range(2020, 2026):
        yr_mask = df['year_full'] == yr
        yr_df = df[yr_mask]
        n = len(yr_df)
        if n == 0:
            continue
        si_cov = (yr_df['index_max'] != yr_df['index_max'].median()).mean() * 100
        t1f_cov = (yr_df['time_1f_last'] != yr_df['time_1f_last'].median()).mean() * 100
        print(f"    {yr}: n={n:>6,}, index_max={si_cov:.1f}%, time_1f={t1f_cov:.1f}%")

    report['data_coverage'] = {
        'speed_index_races': int(si_df['race_id'].nunique()),
        'training_1f_races': int(tt_df['race_id'].nunique()),
        'total_rows': len(df),
        'note': 'Premium data only available for 2025 (scraped subset). Other years use median fill.',
    }
    save_report()

    # Define feature patterns
    patterns = {
        'baseline_87': features_base,
        'plus_index_89': features_base + ['index_max', 'index_run1'],
        'plus_1f_88': features_base + ['time_1f_last'],
        'all_90': features_base + ['index_max', 'index_run1', 'time_1f_last'],
    }

    # Walk-forward validation
    test_years = [2021, 2022, 2023, 2024, 2025]
    print(f"\n[5] Walk-forward validation (test years: {test_years})...")

    for name, feats in patterns.items():
        result = run_walk_forward(df, feats, test_years, label=name)
        report['walk_forward'][name] = result
        save_report()

    # Summary comparison
    print(f"\n{'=' * 70}")
    print(f"  RESULTS COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Pattern':20s} {'Features':>4} {'Avg AUC':>8} {'2025 AUC':>9} {'All>0.78':>8}")
    print(f"  {'-'*55}")

    baseline_auc = report['walk_forward'].get('baseline_87', {}).get('avg_auc', 0)
    baseline_2025 = report['walk_forward'].get('baseline_87', {}).get('fold_aucs', {}).get(2025, 0)

    for name in patterns:
        wf = report['walk_forward'].get(name, {})
        n_feat = wf.get('n_features', 0)
        avg = wf.get('avg_auc', 0)
        auc_2025 = wf.get('fold_aucs', {}).get(2025, 0)
        all_ok = wf.get('all_above_078', False)
        delta = avg - baseline_auc
        ok_str = 'YES' if all_ok else 'NO'
        print(f"  {name:20s} {n_feat:>4} {avg:>8.4f} {auc_2025:>9.4f} {ok_str:>8} ({delta:+.4f})")

    # ROI comparison (2025 only since that's where we have premium data)
    print(f"\n  ROI Comparison (conditions with data):")
    for name in patterns:
        wf = report['walk_forward'].get(name, {})
        roi = wf.get('roi_by_condition', {})
        if roi:
            total_ret = sum(c.get('roi', 0) * c.get('n', 0) for c in roi.values())
            total_n = sum(c.get('n', 0) for c in roi.values())
            avg_roi = total_ret / total_n if total_n > 0 else 0
            print(f"    {name:20s}: weighted avg ROI={avg_roi:.1f}%")

    # Adoption decision
    best_name = 'baseline_87'
    best_auc = baseline_auc
    for name in patterns:
        if name == 'baseline_87':
            continue
        wf = report['walk_forward'].get(name, {})
        avg = wf.get('avg_auc', 0)
        all_ok = wf.get('all_above_078', False)
        if avg > best_auc and all_ok:
            best_auc = avg
            best_name = name

    adopted = best_name != 'baseline_87'
    reason = f"{best_name} AUC {best_auc:.4f} > baseline {baseline_auc:.4f}" if adopted else "No pattern improved over baseline"

    report['adoption'] = {
        'adopted': adopted,
        'best_pattern': best_name,
        'best_auc': best_auc,
        'baseline_auc': baseline_auc,
        'reason': reason,
        'note': 'Premium features only cover 2025 data. Full 2020-2025 scraping needed for definitive conclusion.',
    }
    save_report()

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed/60:.1f} minutes")
    print(f"  Adoption: {'YES' if adopted else 'NO'} - {reason}")
    print(f"  Report: data/v11_premium_features_report.json")


if __name__ == '__main__':
    main()
