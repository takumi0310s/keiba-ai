#!/usr/bin/env python
"""KEIBA AI v11 Training: Speed Index Features
Tests 4 patterns of speed index integration with walk-forward backtest.

Pattern 1: baseline_87 (current 87 features, verification)
Pattern 2: +index_max+index_run1 (89 features)
Pattern 3: +index_all (92 features: max, run1, avg5, dist, course)
Pattern 4: +PCI (93 features: pattern3 + pace_change_index from lap data)

All patterns evaluated with:
- Walk-forward AUC (2020-2025, LGB only)
- Per-year AUC > 0.78 check
- Train-test AUC gap < 0.05 check
- Actual ROI via jra_payouts.csv (for patterns that pass AUC)
"""
import pandas as pd
import numpy as np
import pickle
import gzip
import os
import sys
import json
import time
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
    compute_running_style_features,
    COURSE_MAP, N_TOP_SIRE,
    train_lgb, train_xgb,
)
from train_v92_leakfree import (
    LEAK_FEATURES_A, FEATURES_PATTERN_A,
    classify_condition, calc_trio_bets,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')
TEST_YEARS = list(range(2020, 2026))

# Current baseline features (Pattern A, 67 features from v9.3)
# After pace features added in v9.3, FEATURES_PATTERN_A should be ~71
# But CLAUDE.md says 67. Let's use whatever FEATURES_PATTERN_A is.

# Speed index features to add
SI_FEATURES_2 = ['si_index_max', 'si_index_run1']
SI_FEATURES_ALL = ['si_index_max', 'si_index_run1', 'si_index_avg5',
                   'si_index_dist', 'si_index_course']
PCI_FEATURE = ['pci']  # Pace Change Index from existing lap data


def target_to_netkeiba(rid_str):
    """Convert TARGET race_id_str (8 chars) to netkeiba race_id (12 digits)."""
    cc = rid_str[:2]
    yy = int(rid_str[2:4])
    yyyy = 2000 + yy
    k = int(rid_str[4], 16)
    n = int(rid_str[5], 16)
    rr = rid_str[6:8]
    return f'{yyyy}{cc}{k:02d}{n:02d}{rr}'


def load_speed_index():
    """Load and prepare speed index data for merge."""
    si_path = os.path.join(DATA_DIR, 'netkeiba_speed_index.csv')
    si = pd.read_csv(si_path)
    print(f"  Speed index: {len(si)} rows")

    # Rename to avoid collision with existing columns
    si = si.rename(columns={
        'index_max': 'si_index_max',
        'index_run1': 'si_index_run1',
        'index_avg5': 'si_index_avg5',
        'index_dist': 'si_index_dist',
        'index_course': 'si_index_course',
    })

    si['nk_race_id'] = si['race_id'].astype(str)
    si['umaban'] = si['umaban'].astype(int)

    # Keep only needed columns
    si = si[['nk_race_id', 'umaban'] + SI_FEATURES_ALL].copy()

    # Convert to numeric
    for col in SI_FEATURES_ALL:
        si[col] = pd.to_numeric(si[col], errors='coerce')

    return si


def merge_speed_index(df, si):
    """Merge speed index data into main dataframe."""
    # Build netkeiba race_id from TARGET race_id
    df['nk_race_id'] = df['race_id_str'].apply(target_to_netkeiba)
    df['umaban_int'] = df['umaban'].astype(int)

    df = df.merge(si, left_on=['nk_race_id', 'umaban_int'],
                  right_on=['nk_race_id', 'umaban'],
                  how='left', suffixes=('', '_si'))

    # Drop merge key duplicates
    if 'umaban_si' in df.columns:
        df = df.drop(columns=['umaban_si'])

    matched = df['si_index_max'].notna().sum()
    print(f"  Speed index merged: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")

    return df


def compute_pci(df):
    """Compute Pace Change Index from existing lap data.
    PCI = (first_3f / last_3f) * 100
    Uses previous race's pace data (shift-safe, no leak).
    """
    if 'prev_race_first3f' in df.columns and 'prev_race_last3f' in df.columns:
        valid = (df['prev_race_first3f'] > 0) & (df['prev_race_last3f'] > 0)
        df['pci'] = np.where(valid,
                             (df['prev_race_first3f'] / df['prev_race_last3f']) * 100,
                             0)
        pci_valid = (df['pci'] > 0).sum()
        print(f"  PCI computed: {pci_valid}/{len(df)} ({pci_valid/len(df)*100:.1f}%)")
    else:
        df['pci'] = 0
        print("  PCI: prev_race_first3f/last3f not found, set to 0")
    return df


def encode_sires_fold(df, train_mask, n_top=N_TOP_SIRE):
    """Encode sires using only training data."""
    train_df = df[train_mask]
    sire_counts = train_df['father'].value_counts()
    top_sires = sire_counts.head(n_top).index.tolist()
    sire_map = {s: i for i, s in enumerate(top_sires)}
    df['sire_enc'] = df['father'].map(sire_map).fillna(n_top).astype(int)
    bms_counts = train_df['bms'].value_counts()
    top_bms = bms_counts.head(n_top).index.tolist()
    bms_map = {s: i for i, s in enumerate(top_bms)}
    df['bms_enc'] = df['bms'].map(bms_map).fillna(n_top).astype(int)
    return df


def train_lgb_fold(X_train, y_train, X_valid, y_valid, features):
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
        'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
    }
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=features)
    dvalid = lgb.Dataset(X_valid, label=y_valid, feature_name=features, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    return model


def walk_forward_auc(df, features, label=""):
    """Run walk-forward and return avg AUC, per-year AUCs, and train-test gaps."""
    fold_aucs = {}
    train_aucs = {}

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

        # Train AUC (on validation portion of training data)
        va_pred = model.predict(X_va)
        va_auc = roc_auc_score(y_va, va_pred)
        train_aucs[test_year] = va_auc

        # Test AUC
        test_df = df_fold[test_mask]
        X_test = test_df[features].values
        y_test = test_df['target'].values
        test_pred = model.predict(X_test)
        test_auc = roc_auc_score(y_test, test_pred)
        fold_aucs[test_year] = test_auc

    avg_auc = np.mean(list(fold_aucs.values()))
    return avg_auc, fold_aucs, train_aucs


def walk_forward_with_roi(df, features, payout_lookup, label=""):
    """Full WF backtest with actual ROI via jra_payouts.csv."""
    all_results = []
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

        test_df = df_fold[test_mask].copy()
        X_test = test_df[features].values
        test_df['pred'] = model.predict(X_test)
        test_auc = roc_auc_score(test_df['target'].values, test_df['pred'])
        fold_aucs[test_year] = test_auc
        print(f"    {test_year}: AUC {test_auc:.4f}")

        for rid in test_df['race_id_str'].unique():
            race_df = test_df[test_df['race_id_str'] == rid].copy()
            if len(race_df) < 5:
                continue

            row0 = race_df.iloc[0]
            num_horses = int(row0['num_horses_val'])
            distance = int(row0['distance'])
            cond_enc = int(row0.get('condition_enc', 0))
            cond_key = classify_condition(num_horses, distance, cond_enc)

            race_df = race_df.sort_values('pred', ascending=False)
            ranking = race_df['umaban'].astype(int).tolist()

            # Actual top 3
            race_sorted = race_df.sort_values('finish')
            actual_top3 = {}
            for _, r in race_sorted.head(3).iterrows():
                actual_top3[int(r['finish'])] = int(r['umaban'])
            if len(actual_top3) < 3:
                continue

            trio_bets = calc_trio_bets(ranking)
            top3_set = set(actual_top3.values())
            trio_hit = any(set(c) == top3_set for c in trio_bets) if len(top3_set) == 3 else False

            # Umaren bets (for condition E)
            umaren_bets = []
            if len(ranking) >= 3:
                umaren_bets = [tuple(sorted([ranking[0], ranking[1]])),
                               tuple(sorted([ranking[0], ranking[2]]))]
            top2_set = {actual_top3.get(1), actual_top3.get(2)}
            umaren_hit = any(set(b) == top2_set for b in umaren_bets)

            # Lookup actual payout
            nk_rid = row0.get('nk_race_id', '')
            trio_payout = 0
            umaren_payout = 0
            if nk_rid in payout_lookup:
                p = payout_lookup[nk_rid]
                if trio_hit and 'trio_payout' in p:
                    trio_payout = p['trio_payout']
                if umaren_hit and 'umaren_payout' in p:
                    umaren_payout = p['umaren_payout']

            all_results.append({
                'race_id': rid,
                'year': test_year,
                'cond_key': cond_key,
                'distance': distance,
                'trio_hit': trio_hit,
                'trio_payout': trio_payout,
                'umaren_hit': umaren_hit,
                'umaren_payout': umaren_payout,
            })

    return all_results, fold_aucs


def load_payout_lookup():
    """Load JRA payout data as lookup dict."""
    payout_path = os.path.join(DATA_DIR, 'jra_payouts.csv')
    if not os.path.exists(payout_path):
        print("  WARNING: jra_payouts.csv not found")
        return {}

    payouts = pd.read_csv(payout_path)
    lookup = {}

    for _, row in payouts.iterrows():
        # Build netkeiba-style race_id from payout data
        try:
            date = str(int(row['race_date']))
            course = str(row['course']).strip()
            kai = int(row['kai'])
            nichi = int(row['nichi'])
            race_num = int(row['race_num'])

            course_map_nk = {
                '札幌': '01', '函館': '02', '福島': '03', '新潟': '04',
                '東京': '05', '中山': '06', '中京': '07', '京都': '08',
                '阪神': '09', '小倉': '10',
            }
            cc = course_map_nk.get(course, '')
            if not cc:
                continue

            year = date[:4]
            nk_rid = f"{year}{cc}{kai:02d}{nichi:02d}{race_num:02d}"

            trio_pay = 0
            umaren_pay = 0
            if pd.notna(row.get('trio_payout')):
                trio_pay = int(row['trio_payout'])
            if pd.notna(row.get('umaren_payout')):
                umaren_pay = int(row['umaren_payout'])

            lookup[nk_rid] = {
                'trio_payout': trio_pay,
                'umaren_payout': umaren_pay,
            }
        except (ValueError, KeyError):
            continue

    print(f"  Payout lookup: {len(lookup)} races")
    return lookup


def calc_condition_roi(results):
    """Calculate ROI by condition from backtest results."""
    cond_stats = {}
    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        cond_races = [r for r in results if r['cond_key'] == cond]
        n = len(cond_races)
        if n == 0:
            cond_stats[cond] = {'n': 0, 'roi': 0, 'hit_rate': 0}
            continue

        if cond == 'E':
            hits = sum(1 for r in cond_races if r['umaren_hit'])
            total_payout = sum(r['umaren_payout'] for r in cond_races)
            investment = n * 700
        else:
            hits = sum(1 for r in cond_races if r['trio_hit'])
            total_payout = sum(r['trio_payout'] for r in cond_races)
            investment = n * 700

        roi = total_payout / investment * 100 if investment > 0 else 0
        hit_rate = hits / n * 100

        cond_stats[cond] = {
            'n': n,
            'hits': hits,
            'hit_rate': round(hit_rate, 1),
            'investment': investment,
            'payout': total_payout,
            'roi': round(roi, 1),
        }

    return cond_stats


def main():
    t_start = time.time()
    print("=" * 70)
    print("  KEIBA AI v11 SPEED INDEX TRAINING")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # === Load Data ===
    print("\n--- Loading data ---")
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
    df = compute_running_style_features(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()

    # === Merge speed index ===
    print("\n--- Loading speed index ---")
    si = load_speed_index()
    df = merge_speed_index(df, si)

    # Fill NaN speed index with mean (per year to avoid leak)
    for col in SI_FEATURES_ALL:
        if col in df.columns:
            global_mean = df[col].mean()
            df[col] = df[col].fillna(0)  # 0 = missing indicator (model can learn)
            print(f"  {col}: mean={global_mean:.1f}, zero_rate={(df[col]==0).mean()*100:.1f}%")

    # === Compute PCI ===
    df = compute_pci(df)

    # Ensure all features numeric
    all_possible_features = list(FEATURES_PATTERN_A) + SI_FEATURES_ALL + PCI_FEATURE
    for f in all_possible_features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"\nData ready: {len(df)} rows, {df['race_id_str'].nunique()} races")
    print(f"Baseline features: {len(FEATURES_PATTERN_A)}")

    # === Define 4 patterns ===
    patterns = {
        'baseline': list(FEATURES_PATTERN_A),
        '+si_2': list(FEATURES_PATTERN_A) + SI_FEATURES_2,
        '+si_all': list(FEATURES_PATTERN_A) + SI_FEATURES_ALL,
        '+si_all+pci': list(FEATURES_PATTERN_A) + SI_FEATURES_ALL + PCI_FEATURE,
    }

    # === Walk-Forward AUC for all patterns ===
    results = {}
    for pattern_name, features in patterns.items():
        print(f"\n{'='*70}")
        print(f"  PATTERN: {pattern_name} ({len(features)} features)")
        print(f"{'='*70}")

        t0 = time.time()
        avg_auc, fold_aucs, train_aucs = walk_forward_auc(df, features, pattern_name)
        elapsed = time.time() - t0

        # Check criteria
        all_above_078 = all(a > 0.78 for a in fold_aucs.values())
        max_gap = max(train_aucs[y] - fold_aucs[y] for y in fold_aucs if y in train_aucs)
        no_overfit = max_gap < 0.05

        print(f"\n  Results:")
        for yr in sorted(fold_aucs.keys()):
            gap = train_aucs.get(yr, 0) - fold_aucs[yr]
            ok = "OK" if fold_aucs[yr] > 0.78 else "FAIL"
            print(f"    {yr}: test={fold_aucs[yr]:.4f}, train={train_aucs.get(yr,0):.4f}, gap={gap:.4f} [{ok}]")
        print(f"  Average WF AUC: {avg_auc:.4f}")
        print(f"  All years > 0.78: {all_above_078}")
        print(f"  Max train-test gap: {max_gap:.4f} (overfit: {'NO' if no_overfit else 'YES'})")
        print(f"  Time: {elapsed:.0f}s")

        results[pattern_name] = {
            'features': features,
            'n_features': len(features),
            'avg_auc': round(avg_auc, 5),
            'fold_aucs': {str(k): round(v, 5) for k, v in fold_aucs.items()},
            'train_aucs': {str(k): round(v, 5) for k, v in train_aucs.items()},
            'all_above_078': all_above_078,
            'max_gap': round(max_gap, 5),
            'no_overfit': no_overfit,
            'passed_auc': avg_auc > 0.8001 and all_above_078 and no_overfit,
        }

    # === Summary ===
    print(f"\n{'='*70}")
    print("  WF AUC COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Pattern':<20} {'N_feat':>6} {'WF AUC':>8} {'>0.78':>6} {'Gap':>6} {'Pass':>5}")
    print(f"  {'-'*55}")
    baseline_wf = 0.8017  # From CLAUDE.md
    for pname, res in results.items():
        marker = "PASS" if res['passed_auc'] else "FAIL"
        delta = res['avg_auc'] - baseline_wf
        print(f"  {pname:<20} {res['n_features']:>6} {res['avg_auc']:>8.5f} "
              f"{'Y' if res['all_above_078'] else 'N':>6} {res['max_gap']:>6.4f} {marker:>5} ({delta:+.5f})")

    # === ROI for patterns that passed AUC ===
    passed_patterns = {k: v for k, v in results.items() if v['passed_auc']}
    print(f"\n  Patterns passing AUC: {list(passed_patterns.keys())}")

    payout_lookup = load_payout_lookup()

    for pname in passed_patterns:
        features = results[pname]['features']
        print(f"\n{'='*70}")
        print(f"  ROI BACKTEST: {pname}")
        print(f"{'='*70}")

        bt_results, bt_aucs = walk_forward_with_roi(df, features, payout_lookup, pname)
        cond_roi = calc_condition_roi(bt_results)

        print(f"\n  Condition ROI:")
        print(f"  {'Cond':<5} {'N':>6} {'Hits':>6} {'HitR':>7} {'ROI':>8}")
        all_above_100 = True
        for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
            c = cond_roi[cond]
            if c['n'] == 0:
                print(f"  {cond:<5} {'N/A':>6}")
                continue
            print(f"  {cond:<5} {c['n']:>6} {c.get('hits',0):>6} {c['hit_rate']:>6.1f}% {c['roi']:>7.1f}%")
            if c['roi'] < 100 and c['n'] >= 30:
                all_above_100 = False

        results[pname]['condition_roi'] = cond_roi
        results[pname]['all_roi_above_100'] = all_above_100
        results[pname]['passed_roi'] = all_above_100

    # === Final Decision ===
    print(f"\n{'='*70}")
    print("  FINAL DECISION")
    print(f"{'='*70}")

    # Baseline ROIs from CLAUDE.md
    baseline_roi = {'A': 205.3, 'B': 236.9, 'C': 285.6, 'D': 136.0, 'E': 118.0, 'X': 330.5}

    best_pattern = None
    best_auc = baseline_wf

    for pname, res in results.items():
        if pname == 'baseline':
            continue
        if not res.get('passed_auc', False):
            print(f"  {pname}: REJECTED (AUC criteria not met)")
            continue
        if not res.get('passed_roi', False):
            print(f"  {pname}: REJECTED (ROI below 100% in some condition)")
            continue

        # Check if ROI >= baseline for all conditions
        roi_ok = True
        cond_roi = res.get('condition_roi', {})
        for cond, base_val in baseline_roi.items():
            if cond_roi.get(cond, {}).get('n', 0) < 30:
                continue  # Skip small sample conditions
            if cond_roi[cond]['roi'] < base_val * 0.95:  # Allow 5% tolerance
                roi_ok = False
                print(f"  {pname}: ROI degraded in {cond} ({cond_roi[cond]['roi']:.1f}% < {base_val:.1f}%)")

        if res['avg_auc'] > best_auc:
            best_pattern = pname
            best_auc = res['avg_auc']
            if not roi_ok:
                print(f"  {pname}: AUC improved but ROI degraded in some conditions")

    if best_pattern:
        res = results[best_pattern]
        print(f"\n  *** BEST: {best_pattern} ***")
        print(f"  WF AUC: {res['avg_auc']:.5f} (baseline: {baseline_wf:.4f}, +{res['avg_auc']-baseline_wf:.5f})")
        print(f"  Features: {res['n_features']}")

        # Save v11 model (full retrain on all data except last 2 years for validation)
        print(f"\n  Training final v11 model...")
        features = res['features']
        features_pkl = [f if f != 'num_horses_val' else 'num_horses' for f in features]

        max_year = df['year_full'].max()
        valid_mask = df['year_full'] >= (max_year - 1)
        train_mask = ~valid_mask
        y = df['target'].values
        y_train, y_valid = y[train_mask], y[valid_mask]

        for f in features:
            if f not in df.columns:
                df[f] = 0
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

        X_train = df.loc[train_mask, features].values
        X_valid = df.loc[valid_mask, features].values

        lgb_model = train_lgb(X_train, y_train, X_valid, y_valid, features)
        lgb_auc = roc_auc_score(y_valid, lgb_model.predict(X_valid))

        import xgboost as xgb
        xgb_model = train_xgb(X_train, y_train, X_valid, y_valid)
        xgb_auc = roc_auc_score(y_valid, xgb_model.predict(xgb.DMatrix(X_valid)))

        total = lgb_auc + xgb_auc
        w_lgb = lgb_auc / total
        w_xgb = xgb_auc / total
        ens_pred = lgb_model.predict(X_valid) * w_lgb + xgb_model.predict(xgb.DMatrix(X_valid)) * w_xgb
        ens_auc = roc_auc_score(y_valid, ens_pred)

        print(f"  Final model AUC: LGB {lgb_auc:.4f} / XGB {xgb_auc:.4f} / Ensemble {ens_auc:.4f}")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pkl = {
            'model': lgb_model,
            'features': features_pkl,
            'version': 'v11_speed_index',
            'auc': lgb_auc,
            'ensemble_auc': ens_auc,
            'wf_auc': res['avg_auc'],
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
            'xgb_model': xgb_model,
            'mlp_model': None,
            'mlp_scaler': None,
            'ensemble_weights': {'lgb': w_lgb, 'xgb': w_xgb, 'mlp': 0},
            'course_map': dict(COURSE_MAP),
            'si_features': [f for f in features if f.startswith('si_') or f == 'pci'],
            'pattern_name': best_pattern,
            'fold_aucs': res['fold_aucs'],
            'condition_roi': res.get('condition_roi', {}),
        }

        # Save as v11
        v11_path = os.path.join(BASE_DIR, 'keiba_model_v11_central.pkl')
        with open(v11_path, 'wb') as f:
            pickle.dump(pkl, f)
        print(f"  Saved: {v11_path}")

        # Also save compressed
        v11_gz_path = v11_path + '.gz'
        with gzip.open(v11_gz_path, 'wb') as f:
            pickle.dump(pkl, f)
        print(f"  Saved: {v11_gz_path}")

        # Feature importance
        fi = pd.DataFrame({
            'feature': features,
            'importance': lgb_model.feature_importance('gain'),
        }).sort_values('importance', ascending=False)
        print(f"\n  Top 15 features:")
        for _, row in fi.head(15).iterrows():
            marker = " ★NEW" if row['feature'] in SI_FEATURES_ALL + PCI_FEATURE else ""
            print(f"    {row['feature']:<30} {row['importance']:>12,.0f}{marker}")

    else:
        print(f"\n  No pattern improved over baseline WF AUC {baseline_wf:.4f}")
        print(f"  v11 NOT adopted.")

    # Save results JSON
    results_out = {}
    for pname, res in results.items():
        results_out[pname] = {k: v for k, v in res.items() if k != 'features'}

    out_path = os.path.join(DATA_DIR, 'v11_training_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results_out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")

    elapsed_total = time.time() - t_start
    print(f"\n  Total time: {elapsed_total/60:.1f} min")
    print(f"  v11 training complete!")

    return results


if __name__ == '__main__':
    main()
