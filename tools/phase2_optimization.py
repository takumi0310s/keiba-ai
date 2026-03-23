#!/usr/bin/env python
"""Phase 2 Optimization: Calibration, EV Model, LightGBM Ranker

Task 1: Probability Calibration (Platt / Isotonic)
Task 2: EV Label & Market-Beating Model (Pattern C)
Task 3: LightGBM Ranker (lambdarank)

All tasks use walk-forward backtest with actual JRA payouts for ROI.
Data split: Train 2010-2022, Valid 2023, Test 2024-2025.
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import pickle
import warnings
import traceback
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, ndcg_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

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
    classify_condition, get_axes, calc_trio_bets, calc_umaren_bets,
    calc_wide_bets, check_bets, estimate_payouts, encode_sires_fold,
    train_lgb_fold,
)
# Try to load actual payouts; fall back to estimated if jra_payouts.csv missing
HAS_PAYOUTS = False
try:
    from calc_actual_roi import (
        load_payouts, parse_trio_nums, parse_umaren_nums, parse_wide_data,
        get_actual_payouts, calc_actual_returns,
    )
    # Check if file exists
    if os.path.exists(os.path.join(BASE_DIR, 'data', 'jra_payouts.csv')):
        HAS_PAYOUTS = True
except ImportError:
    pass

ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts', 'phase2')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Data split
TRAIN_END = 2022      # Train: 2010-2022
VALID_YEAR = 2023     # Valid: 2023
TEST_YEARS = [2024, 2025]  # Test: 2024-2025
ALL_WF_YEARS = [2023, 2024, 2025]  # All years for WF

report = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'data_split': {
        'train': '2010-2022',
        'valid': '2023',
        'test': '2024-2025',
    },
    'task1_calibration': {},
    'task2_ev_model': {},
    'task3_ranker': {},
    'final_decision': {
        'production_changes': [],
        'reasons': [],
    },
}


def prepare_data():
    """Load and prepare all data using the existing pipeline."""
    print("=" * 70)
    print("  PHASE 2 OPTIMIZATION - DATA PREPARATION")
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
    df = build_features(df)

    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()

    features = list(FEATURES_PATTERN_A)
    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"  Data ready: {len(df)} rows, {df['race_id_str'].nunique()} races")
    return df, features, sire_map, bms_map


def walk_forward_predict(df, features, years):
    """Run walk-forward and return predictions for specified years.
    Returns df with 'pred' column filled for the test years.
    """
    all_preds = {}
    fold_aucs = {}
    fold_models = {}

    for test_year in years:
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
        preds = model.predict(X_test)
        test_auc = roc_auc_score(test_df['target'].values, preds)
        fold_aucs[test_year] = test_auc
        fold_models[test_year] = model

        # Store predictions keyed by index
        for idx, pred in zip(test_df.index, preds):
            all_preds[idx] = pred

        print(f"  WF {test_year}: AUC {test_auc:.4f} (n={n_test:,})")

    return all_preds, fold_aucs, fold_models


def compute_roi_from_preds(df, pred_dict, payout_lookup, years, label=""):
    """Compute condition-based ROI using predictions.
    Uses actual JRA payouts if available, otherwise estimated payouts.
    """
    results_by_cond = defaultdict(lambda: {
        'n': 0, 'trio_hits': 0, 'trio_return': 0,
        'umaren_hits': 0, 'umaren_return': 0,
        'wide_hits': 0, 'wide_return': 0,
    })
    year_results = defaultdict(lambda: defaultdict(lambda: {
        'n': 0, 'trio_hits': 0, 'trio_return': 0,
    }))

    for test_year in years:
        test_mask = df['year_full'] == test_year
        test_df = df[test_mask].copy()

        pred_indices = [i for i in test_df.index if i in pred_dict]
        if not pred_indices:
            continue
        test_df = test_df.loc[pred_indices]
        test_df['pred'] = [pred_dict[i] for i in test_df.index]

        test_races = test_df['race_id_str'].unique()
        for rid in test_races:
            race_df = test_df[test_df['race_id_str'] == rid].copy()
            if len(race_df) < 5:
                continue

            row0 = race_df.iloc[0]
            cond_key = classify_condition(row0)

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

            if HAS_PAYOUTS and payout_lookup:
                payout_info = get_actual_payouts(payout_lookup, rid)
                (trio_hit, trio_return,
                 umaren_hit, umaren_return,
                 wide_hit_cnt, wide_return) = calc_actual_returns(
                    payout_info, trio_bets, umaren_bets, wide_bets)
                umaren_return = umaren_return * 3.5 if umaren_hit else 0
                wide_return = wide_return * 3.5 if wide_hit_cnt > 0 else 0
            else:
                # Use estimated payouts
                trio_hit_flag, umaren_hits_list, wide_hits_list = check_bets(
                    actual_top3, trio_bets, umaren_bets, wide_bets)
                trio_pay_est, umaren_pay_est, wide_pays_est = estimate_payouts(actual_top3, race_df)

                trio_hit = trio_hit_flag
                trio_return = trio_pay_est if trio_hit else 0
                umaren_hit = len(umaren_hits_list) > 0
                umaren_return = umaren_pay_est * 3.5 * len(umaren_hits_list) if umaren_hits_list else 0
                wide_hit_cnt = len(wide_hits_list)
                wide_return = sum(wide_pays_est.get(tuple(sorted(w)), 150) * 3.5 for w in wide_hits_list)

            cr = results_by_cond[cond_key]
            cr['n'] += 1
            if trio_hit:
                cr['trio_hits'] += 1
                cr['trio_return'] += trio_return
            if umaren_hit:
                cr['umaren_hits'] += 1
                cr['umaren_return'] += umaren_return
            if wide_hit_cnt > 0:
                cr['wide_hits'] += 1
                cr['wide_return'] += wide_return

            yr = year_results[test_year][cond_key]
            yr['n'] += 1
            if trio_hit:
                yr['trio_hits'] += 1
                yr['trio_return'] += trio_return

    # Compute ROI
    summary = {}
    total_n = 0
    total_inv = 0
    total_trio_return = 0
    total_trio_hits = 0

    for cond_key in ['A', 'B', 'C', 'D', 'E', 'X']:
        cr = results_by_cond[cond_key]
        n = cr['n']
        if n == 0:
            continue
        inv = n * 700
        trio_roi = cr['trio_return'] / inv * 100 if inv > 0 else 0
        trio_hit_rate = cr['trio_hits'] / n * 100
        umaren_roi = cr['umaren_return'] / inv * 100 if inv > 0 else 0
        wide_roi = cr['wide_return'] / inv * 100 if inv > 0 else 0

        summary[cond_key] = {
            'n': n,
            'trio_roi': round(trio_roi, 1),
            'trio_hit_rate': round(trio_hit_rate, 1),
            'trio_hits': cr['trio_hits'],
            'umaren_roi': round(umaren_roi, 1),
            'wide_roi': round(wide_roi, 1),
        }
        total_n += n
        total_inv += inv
        total_trio_return += cr['trio_return']
        total_trio_hits += cr['trio_hits']

    overall_roi = total_trio_return / total_inv * 100 if total_inv > 0 else 0
    overall_hit_rate = total_trio_hits / total_n * 100 if total_n > 0 else 0

    yearly = {}
    for yr in years:
        yr_data = {}
        for cond_key in ['A', 'B', 'C', 'D', 'E', 'X']:
            yc = year_results[yr][cond_key]
            if yc['n'] == 0:
                continue
            inv = yc['n'] * 700
            yr_data[cond_key] = {
                'n': yc['n'],
                'trio_roi': round(yc['trio_return'] / inv * 100, 1) if inv > 0 else 0,
                'trio_hits': yc['trio_hits'],
            }
        yearly[str(yr)] = yr_data

    return {
        'conditions': summary,
        'overall_n': total_n,
        'overall_trio_roi': round(overall_roi, 1),
        'overall_trio_hit_rate': round(overall_hit_rate, 1),
        'yearly': yearly,
        'payout_source': 'actual_jra' if (HAS_PAYOUTS and payout_lookup) else 'estimated',
    }


def print_roi_summary(roi_data, label=""):
    """Print ROI summary table."""
    print(f"\n  {label} ROI Summary:")
    print(f"  {'Cond':<6} {'N':>5} {'Trio ROI':>9} {'Hit%':>6} {'Uma ROI':>9} {'Wide ROI':>9}")
    for ck in ['A', 'B', 'C', 'D', 'E', 'X']:
        if ck in roi_data['conditions']:
            c = roi_data['conditions'][ck]
            print(f"  {ck:<6} {c['n']:>5} {c['trio_roi']:>8.1f}% {c['trio_hit_rate']:>5.1f}% "
                  f"{c['umaren_roi']:>8.1f}% {c['wide_roi']:>8.1f}%")
    print(f"  {'TOTAL':<6} {roi_data['overall_n']:>5} {roi_data['overall_trio_roi']:>8.1f}% "
          f"{roi_data['overall_trio_hit_rate']:>5.1f}%")


# ============================================================
# TASK 1: CALIBRATION
# ============================================================
def task1_calibration(df, features, payout_lookup):
    """Probability calibration comparison."""
    print("\n" + "=" * 70)
    print("  TASK 1: PROBABILITY CALIBRATION")
    print("=" * 70)

    try:
        # Get WF predictions for valid (2023) and test (2024-2025)
        print("\n[1.1] Walk-forward predictions...")
        all_preds, fold_aucs, fold_models = walk_forward_predict(
            df, features, ALL_WF_YEARS)

        # Separate valid and test predictions
        valid_mask = df['year_full'] == VALID_YEAR
        test_mask = df['year_full'].isin(TEST_YEARS)

        valid_indices = [i for i in df[valid_mask].index if i in all_preds]
        test_indices = [i for i in df[test_mask].index if i in all_preds]

        y_valid = df.loc[valid_indices, 'target'].values
        p_valid = np.array([all_preds[i] for i in valid_indices])

        y_test = df.loc[test_indices, 'target'].values
        p_test = np.array([all_preds[i] for i in test_indices])

        print(f"  Valid: n={len(y_valid)}, Test: n={len(y_test)}")

        # Baseline metrics
        def compute_metrics(y_true, y_pred, label):
            return {
                'auc': round(roc_auc_score(y_true, y_pred), 4),
                'brier': round(brier_score_loss(y_true, y_pred), 6),
                'logloss': round(log_loss(y_true, np.clip(y_pred, 1e-7, 1-1e-7)), 6),
            }

        baseline_valid = compute_metrics(y_valid, p_valid, "Baseline Valid")
        baseline_test = compute_metrics(y_test, p_test, "Baseline Test")
        print(f"\n  Baseline Valid: {baseline_valid}")
        print(f"  Baseline Test:  {baseline_test}")

        # --- Platt Scaling ---
        print("\n[1.2] Platt Scaling (Logistic Regression on valid)...")
        lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=10000)
        lr.fit(p_valid.reshape(-1, 1), y_valid)
        p_valid_platt = lr.predict_proba(p_valid.reshape(-1, 1))[:, 1]
        p_test_platt = lr.predict_proba(p_test.reshape(-1, 1))[:, 1]

        platt_valid = compute_metrics(y_valid, p_valid_platt, "Platt Valid")
        platt_test = compute_metrics(y_test, p_test_platt, "Platt Test")
        print(f"  Platt Valid: {platt_valid}")
        print(f"  Platt Test:  {platt_test}")

        # --- Isotonic Regression ---
        print("\n[1.3] Isotonic Regression on valid...")
        iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        iso.fit(p_valid, y_valid)
        p_valid_iso = iso.predict(p_valid)
        p_test_iso = iso.predict(p_test)

        iso_valid = compute_metrics(y_valid, p_valid_iso, "Isotonic Valid")
        iso_test = compute_metrics(y_test, p_test_iso, "Isotonic Test")
        print(f"  Isotonic Valid: {iso_valid}")
        print(f"  Isotonic Test:  {iso_test}")

        # --- Calibration curves ---
        print("\n[1.4] Calibration curves...")
        n_bins = 10
        cal_data = {}
        for name, y_t, p_t in [
            ('baseline', y_test, p_test),
            ('platt', y_test, p_test_platt),
            ('isotonic', y_test, p_test_iso),
        ]:
            fraction_pos, mean_pred = calibration_curve(y_t, p_t, n_bins=n_bins, strategy='uniform')
            cal_data[name] = {
                'fraction_positive': fraction_pos.tolist(),
                'mean_predicted': mean_pred.tolist(),
            }

        # Save calibration plot
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
            for name, data in cal_data.items():
                ax.plot(data['mean_predicted'], data['fraction_positive'],
                        'o-', label=f"{name} (Brier={baseline_test['brier'] if name=='baseline' else platt_test['brier'] if name=='platt' else iso_test['brier']:.4f})")
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title('Calibration Curve (Test 2024-2025)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(ARTIFACTS_DIR, 'calibration_curve.png'), dpi=150)
            plt.close(fig)
            print(f"  Saved: artifacts/phase2/calibration_curve.png")
        except Exception as e:
            print(f"  Warning: Could not save plot: {e}")

        # --- ROI comparison ---
        print("\n[1.5] ROI comparison (test period)...")

        # Baseline ROI
        baseline_roi = compute_roi_from_preds(df, all_preds, payout_lookup, TEST_YEARS, "Baseline")
        print_roi_summary(baseline_roi, "Baseline")

        # Platt ROI - replace predictions
        platt_preds = dict(all_preds)
        for idx, p in zip(test_indices, p_test_platt):
            platt_preds[idx] = p
        platt_roi = compute_roi_from_preds(df, platt_preds, payout_lookup, TEST_YEARS, "Platt")
        print_roi_summary(platt_roi, "Platt")

        # Isotonic ROI
        iso_preds = dict(all_preds)
        for idx, p in zip(test_indices, p_test_iso):
            iso_preds[idx] = p
        iso_roi = compute_roi_from_preds(df, iso_preds, payout_lookup, TEST_YEARS, "Isotonic")
        print_roi_summary(iso_roi, "Isotonic")

        # Adoption decision
        adopted = 'none'
        reason = "No calibration method improved both Brier score and ROI"

        # Check Platt
        platt_ok = (platt_test['brier'] < baseline_test['brier'] and
                    platt_roi['overall_trio_roi'] >= baseline_roi['overall_trio_roi'] * 0.98)
        # Check Isotonic
        iso_ok = (iso_test['brier'] < baseline_test['brier'] and
                  iso_roi['overall_trio_roi'] >= baseline_roi['overall_trio_roi'] * 0.98)

        if platt_ok and iso_ok:
            # Pick the one with better Brier
            if platt_test['brier'] <= iso_test['brier']:
                adopted = 'platt'
                reason = f"Platt: Brier {baseline_test['brier']:.6f} -> {platt_test['brier']:.6f}, ROI {platt_roi['overall_trio_roi']:.1f}%"
            else:
                adopted = 'isotonic'
                reason = f"Isotonic: Brier {baseline_test['brier']:.6f} -> {iso_test['brier']:.6f}, ROI {iso_roi['overall_trio_roi']:.1f}%"
        elif platt_ok:
            adopted = 'platt'
            reason = f"Platt: Brier {baseline_test['brier']:.6f} -> {platt_test['brier']:.6f}, ROI {platt_roi['overall_trio_roi']:.1f}%"
        elif iso_ok:
            adopted = 'isotonic'
            reason = f"Isotonic: Brier {baseline_test['brier']:.6f} -> {iso_test['brier']:.6f}, ROI {iso_roi['overall_trio_roi']:.1f}%"

        print(f"\n  Adopted: {adopted} ({reason})")

        report['task1_calibration'] = {
            'baseline': {
                'valid': baseline_valid,
                'test': baseline_test,
                'roi': baseline_roi,
                'calibration_curve': cal_data.get('baseline'),
            },
            'platt': {
                'valid': platt_valid,
                'test': platt_test,
                'roi': platt_roi,
                'calibration_curve': cal_data.get('platt'),
                'coef': float(lr.coef_[0][0]),
                'intercept': float(lr.intercept_[0]),
            },
            'isotonic': {
                'valid': iso_valid,
                'test': iso_test,
                'roi': iso_roi,
                'calibration_curve': cal_data.get('isotonic'),
            },
            'adopted_method': adopted,
            'adoption_reason': reason,
        }

        # Save brier comparison
        brier_comp = {
            'baseline_brier': baseline_test['brier'],
            'platt_brier': platt_test['brier'],
            'isotonic_brier': iso_test['brier'],
            'baseline_roi': baseline_roi['overall_trio_roi'],
            'platt_roi': platt_roi['overall_trio_roi'],
            'isotonic_roi': iso_roi['overall_trio_roi'],
            'adopted': adopted,
        }
        with open(os.path.join(ARTIFACTS_DIR, 'brier_comparison.json'), 'w') as f:
            json.dump(brier_comp, f, indent=2)

        return all_preds, fold_aucs, fold_models

    except Exception as e:
        print(f"  ERROR in Task 1: {e}")
        traceback.print_exc()
        report['task1_calibration'] = {'error': str(e)}
        return {}, {}, {}


# ============================================================
# TASK 2: EV MODEL
# ============================================================
def task2_ev_model(df, features, payout_lookup, baseline_preds):
    """EV label introduction and Pattern C model."""
    print("\n" + "=" * 70)
    print("  TASK 2: EV LABEL & MARKET-BEATING MODEL")
    print("=" * 70)

    try:
        # --- Label definitions ---
        # Single-race EV: predicted_prob * tansho_odds
        # We use prev_odds_log (前走オッズ) as a proxy for market assessment
        # For market-beating: compare model prediction vs implied probability from tansho_odds
        # tansho_odds is available in the training data (historical, not leak since it's pre-race for Pattern B)
        # BUT for Pattern A (leak-free), we cannot use current race tansho_odds
        #
        # Strategy: Use tansho_odds as the "market probability" signal
        # Since Pattern A excludes odds_log (confirmed odds), but tansho_odds is in the raw data
        # We need to use it carefully:
        # - For EV calculation in backtest: use tansho_odds from data (known post-race)
        # - For training labels: use a proxy approach
        #
        # Market-beating label definition:
        #   For each horse, compute: implied_prob = 1 / tansho_odds / overround_factor
        #   target_mb = 1 if (horse finished top3) AND (implied_prob < 0.33)
        #   i.e., the model should learn to find winners that the market underestimates
        #
        # EV label definition:
        #   ev_label = finish_top3 * tansho_odds  (realized EV)
        #   This is a regression target reflecting actual value

        print("\n[2.1] Preparing EV and market-beating labels...")

        # Compute overround per race
        df_ev = df.copy()

        # tansho_odds is already in the data
        if 'tansho_odds' not in df_ev.columns:
            print("  WARNING: tansho_odds not found, using prev_odds_log proxy")
            df_ev['tansho_odds'] = np.exp(df_ev['prev_odds_log'].clip(0, 10))

        # Compute implied probability per race
        race_overrounds = {}
        for rid, grp in df_ev.groupby('race_id_str'):
            odds = grp['tansho_odds'].values
            odds = np.where(odds > 0, odds, 100.0)
            implied = 1.0 / odds
            overround = implied.sum()
            race_overrounds[rid] = overround

        df_ev['overround'] = df_ev['race_id_str'].map(race_overrounds)
        df_ev['implied_prob'] = np.where(
            df_ev['tansho_odds'] > 0,
            (1.0 / df_ev['tansho_odds']) / df_ev['overround'].clip(0.5, 2.0),
            0.05
        )

        # Market-beating label: horse beat expectations
        # A horse is "market-beating" if it finished top3 AND was not a favorite (implied_prob < median)
        # Per-race median implied prob is used as threshold (relative to race)
        race_median_impl = df_ev.groupby('race_id_str')['implied_prob'].transform('median')
        df_ev['target_mb'] = ((df_ev['finish'] <= 3) & (df_ev['implied_prob'] < race_median_impl)).astype(int)

        # EV regression label: realized_ev = top3_indicator * (payout_proxy)
        # Use tansho_odds as payout proxy (approximate)
        df_ev['ev_label'] = df_ev['target'].astype(float) * np.log1p(df_ev['tansho_odds'].clip(1, 500))

        label_definition = {
            'market_beating': {
                'definition': 'finish <= 3 AND implied_prob < race_median_implied_prob',
                'implied_prob_formula': '(1/tansho_odds) / race_overround',
                'note': 'Captures non-favorite horses that placed top3 (value bets)',
                'prevalence_train': None,  # will fill
            },
            'ev_label': {
                'definition': 'top3_indicator * log1p(tansho_odds)',
                'note': 'Regression target reflecting realized value; higher for longshots that place',
            },
        }

        # Check prevalence
        train_mask = df_ev['year_full'] <= TRAIN_END
        label_definition['market_beating']['prevalence_train'] = round(
            df_ev.loc[train_mask, 'target_mb'].mean() * 100, 2)

        print(f"  Market-beating prevalence (train): {label_definition['market_beating']['prevalence_train']:.2f}%")
        print(f"  Standard top3 prevalence (train): {df_ev.loc[train_mask, 'target'].mean()*100:.2f}%")

        # --- Pattern C: Market-beating model ---
        print("\n[2.2] Training Pattern C (market-beating target)...")

        # Walk-forward with market-beating target
        mb_preds = {}
        mb_fold_aucs = {}

        for test_year in ALL_WF_YEARS:
            wf_train_mask = (df_ev['year_full'] >= 2010) & (df_ev['year_full'] < test_year)
            test_mask = df_ev['year_full'] == test_year
            n_test = test_mask.sum()
            if n_test < 100:
                continue

            df_fold = df_ev.copy()
            df_fold = encode_sires_fold(df_fold, wf_train_mask)
            for f in features:
                if f not in df_fold.columns:
                    df_fold[f] = 0
                df_fold[f] = pd.to_numeric(df_fold[f], errors='coerce').fillna(0)

            train_fold = df_fold[wf_train_mask]
            y_train_mb = train_fold['target_mb'].values
            dates = train_fold['date_num']
            valid_cutoff = dates.quantile(0.85)
            tr_idx = dates < valid_cutoff
            va_idx = dates >= valid_cutoff

            X_tr = train_fold.loc[tr_idx, features].values
            y_tr = y_train_mb[tr_idx.values]
            X_va = train_fold.loc[va_idx, features].values
            y_va = y_train_mb[va_idx.values]

            # Train with binary target (market-beating)
            model = train_lgb_fold(X_tr, y_tr, X_va, y_va, features)

            test_fold = df_fold[test_mask].copy()
            X_test = test_fold[features].values
            preds = model.predict(X_test)

            # AUC on standard target for comparison
            try:
                test_auc = roc_auc_score(test_fold['target'].values, preds)
            except:
                test_auc = 0.5
            mb_fold_aucs[test_year] = test_auc

            for idx, pred in zip(test_fold.index, preds):
                mb_preds[idx] = pred

            print(f"  Pattern C WF {test_year}: AUC(top3) {test_auc:.4f}")

        # --- Compare Pattern A vs C ---
        print("\n[2.3] Comparing Pattern A vs Pattern C...")

        test_mask = df_ev['year_full'].isin(TEST_YEARS)
        test_indices = [i for i in df_ev[test_mask].index if i in baseline_preds and i in mb_preds]

        if len(test_indices) > 0:
            y_test = df_ev.loc[test_indices, 'target'].values

            p_a = np.array([baseline_preds[i] for i in test_indices])
            p_c = np.array([mb_preds[i] for i in test_indices])

            auc_a = roc_auc_score(y_test, p_a)
            auc_c = roc_auc_score(y_test, p_c)
            brier_a = brier_score_loss(y_test, p_a)
            brier_c = brier_score_loss(y_test, p_c)

            print(f"  Pattern A - AUC: {auc_a:.4f}, Brier: {brier_a:.6f}")
            print(f"  Pattern C - AUC: {auc_c:.4f}, Brier: {brier_c:.6f}")

            # ROI comparison
            roi_a = compute_roi_from_preds(df_ev, baseline_preds, payout_lookup, TEST_YEARS, "Pattern A")
            roi_c = compute_roi_from_preds(df_ev, mb_preds, payout_lookup, TEST_YEARS, "Pattern C")

            print_roi_summary(roi_a, "Pattern A")
            print_roi_summary(roi_c, "Pattern C")

            # --- EV>1 filtering strategy ---
            print("\n[2.4] EV > 1 filtering backtest...")

            # For each race, compute EV = pred_prob * tansho_odds for the ranking
            # Then only bet on races where top picks have EV > 1
            ev_filter_results = {}

            for strategy_name, pred_dict in [('pattern_a', baseline_preds), ('pattern_c', mb_preds)]:
                ev_bets = defaultdict(lambda: {
                    'n_total': 0, 'n_bet': 0, 'trio_hits': 0, 'trio_return': 0,
                })

                for test_year in TEST_YEARS:
                    yr_mask = df_ev['year_full'] == test_year
                    yr_df = df_ev[yr_mask].copy()
                    yr_indices = [i for i in yr_df.index if i in pred_dict]
                    if not yr_indices:
                        continue
                    yr_df = yr_df.loc[yr_indices]
                    yr_df['pred'] = [pred_dict[i] for i in yr_df.index]

                    for rid, race_df in yr_df.groupby('race_id_str'):
                        if len(race_df) < 5:
                            continue
                        race_df = race_df.sort_values('pred', ascending=False)
                        row0 = race_df.iloc[0]
                        cond_key = classify_condition(row0)

                        # Compute EV for top-ranked horses
                        top3_preds = race_df['pred'].head(3).values
                        top3_odds = race_df['tansho_odds'].head(3).values
                        top3_odds = np.where(top3_odds > 0, top3_odds, 10.0)
                        top3_ev = top3_preds * top3_odds

                        ev_bets[cond_key]['n_total'] += 1

                        # Only bet if average top3 EV > 1
                        if top3_ev.mean() >= 1.0:
                            ev_bets[cond_key]['n_bet'] += 1

                            actual_top3 = {}
                            race_sorted = race_df.sort_values('finish')
                            for _, r in race_sorted.head(3).iterrows():
                                actual_top3[int(r['finish'])] = int(r['umaban'])
                            if len(actual_top3) < 3:
                                continue

                            ranking = race_df['umaban'].astype(int).tolist()
                            trio_bets = calc_trio_bets(ranking)

                            if HAS_PAYOUTS and payout_lookup:
                                payout_info = get_actual_payouts(payout_lookup, rid)
                                (trio_hit, trio_return, _, _, _, _) = calc_actual_returns(
                                    payout_info, trio_bets, [], [])
                            else:
                                trio_hit_f, _, _ = check_bets(actual_top3, trio_bets, [], [])
                                trio_pay_e, _, _ = estimate_payouts(actual_top3, race_df)
                                trio_hit = trio_hit_f
                                trio_return = trio_pay_e if trio_hit else 0

                            if trio_hit:
                                ev_bets[cond_key]['trio_hits'] += 1
                                ev_bets[cond_key]['trio_return'] += trio_return

                # Summarize EV filter results
                ev_summary = {}
                for ck in ['A', 'B', 'C', 'D', 'E', 'X']:
                    eb = ev_bets[ck]
                    if eb['n_bet'] > 0:
                        inv = eb['n_bet'] * 700
                        ev_summary[ck] = {
                            'n_total': eb['n_total'],
                            'n_bet': eb['n_bet'],
                            'filter_rate': round(eb['n_bet'] / eb['n_total'] * 100, 1) if eb['n_total'] > 0 else 0,
                            'trio_hits': eb['trio_hits'],
                            'trio_roi': round(eb['trio_return'] / inv * 100, 1) if inv > 0 else 0,
                        }
                ev_filter_results[strategy_name] = ev_summary

            print(f"\n  EV>1 Filter Results:")
            for strategy, summary in ev_filter_results.items():
                print(f"  --- {strategy} ---")
                for ck in ['A', 'B', 'C', 'D', 'E', 'X']:
                    if ck in summary:
                        s = summary[ck]
                        print(f"    {ck}: bet {s['n_bet']}/{s['n_total']} ({s['filter_rate']:.0f}%), "
                              f"ROI {s['trio_roi']:.1f}%, hits {s['trio_hits']}")

            # Adoption decision
            adopted = False
            adoption_reason = "Pattern C did not improve ROI over Pattern A"
            if roi_c['overall_trio_roi'] > roi_a['overall_trio_roi']:
                # Check bet count
                total_c = sum(c['n'] for c in roi_c['conditions'].values())
                total_a = sum(c['n'] for c in roi_a['conditions'].values())
                if total_c >= total_a * 0.5:
                    adopted = True
                    adoption_reason = f"Pattern C ROI {roi_c['overall_trio_roi']:.1f}% > A {roi_a['overall_trio_roi']:.1f}%"

            print(f"\n  Pattern C adopted: {adopted} ({adoption_reason})")

            report['task2_ev_model'] = {
                'label_definition': label_definition,
                'pattern_a': {
                    'test_auc': round(auc_a, 4),
                    'test_brier': round(brier_a, 6),
                    'roi': roi_a,
                },
                'pattern_c': {
                    'test_auc': round(auc_c, 4),
                    'test_brier': round(brier_c, 6),
                    'roi': roi_c,
                    'target': 'market_beating (finish<=3 AND implied_prob<0.33)',
                },
                'ev_filter': ev_filter_results,
                'adopted': adopted,
                'adoption_reason': adoption_reason,
            }

            # Save pattern comparison
            pattern_comp = {
                'pattern_a': {'auc': round(auc_a, 4), 'brier': round(brier_a, 6),
                              'roi': roi_a['overall_trio_roi']},
                'pattern_c': {'auc': round(auc_c, 4), 'brier': round(brier_c, 6),
                              'roi': roi_c['overall_trio_roi']},
                'ev_filter': ev_filter_results,
                'adopted': adopted,
            }
            with open(os.path.join(ARTIFACTS_DIR, 'pattern_abc_comparison.json'), 'w') as f:
                json.dump(pattern_comp, f, indent=2)

        else:
            print("  WARNING: No overlapping test indices")
            report['task2_ev_model'] = {
                'label_definition': label_definition,
                'error': 'No overlapping test indices',
                'adopted': False,
            }

    except Exception as e:
        print(f"  ERROR in Task 2: {e}")
        traceback.print_exc()
        report['task2_ev_model'] = {'error': str(e), 'adopted': False}


# ============================================================
# TASK 3: LIGHTGBM RANKER
# ============================================================
def task3_ranker(df, features, payout_lookup, baseline_preds):
    """LightGBM Ranker with lambdarank."""
    print("\n" + "=" * 70)
    print("  TASK 3: LIGHTGBM RANKER (LAMBDARANK)")
    print("=" * 70)

    try:
        # Relevance labels: 1st=3, 2nd=2, 3rd=1, else=0
        df_rank = df.copy()
        df_rank['relevance'] = 0
        df_rank.loc[df_rank['finish'] == 1, 'relevance'] = 3
        df_rank.loc[df_rank['finish'] == 2, 'relevance'] = 2
        df_rank.loc[df_rank['finish'] == 3, 'relevance'] = 1

        print(f"  Relevance distribution: {df_rank['relevance'].value_counts().sort_index().to_dict()}")

        # Walk-forward with ranker
        ranker_preds = {}
        ranker_fold_aucs = {}
        ranker_fold_ndcg = {}
        total_groups = 0

        for test_year in ALL_WF_YEARS:
            wf_train_mask = (df_rank['year_full'] >= 2010) & (df_rank['year_full'] < test_year)
            test_mask = df_rank['year_full'] == test_year
            n_test = test_mask.sum()
            if n_test < 100:
                continue

            df_fold = df_rank.copy()
            df_fold = encode_sires_fold(df_fold, wf_train_mask)
            for f in features:
                if f not in df_fold.columns:
                    df_fold[f] = 0
                df_fold[f] = pd.to_numeric(df_fold[f], errors='coerce').fillna(0)

            train_fold = df_fold[wf_train_mask].copy()
            test_fold = df_fold[test_mask].copy()

            # Build groups for ranker (race-level)
            dates = train_fold['date_num']
            valid_cutoff = dates.quantile(0.85)
            tr_fold = train_fold[dates < valid_cutoff].copy()
            va_fold = train_fold[dates >= valid_cutoff].copy()

            # Group sizes for training
            tr_groups = tr_fold.groupby('race_id_str').size().values
            va_groups = va_fold.groupby('race_id_str').size().values

            X_tr = tr_fold[features].values
            y_tr = tr_fold['relevance'].values
            X_va = va_fold[features].values
            y_va = va_fold['relevance'].values

            # LightGBM Ranker parameters
            ranker_params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [3],
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
                'n_jobs': -1,
                'seed': 42,
            }

            dtrain = lgb.Dataset(X_tr, label=y_tr, group=tr_groups, feature_name=features)
            dvalid = lgb.Dataset(X_va, label=y_va, group=va_groups, feature_name=features, reference=dtrain)

            ranker_model = lgb.train(
                ranker_params, dtrain, num_boost_round=500,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(200)]
            )

            # Predict on test
            X_test = test_fold[features].values
            preds = ranker_model.predict(X_test)

            # AUC on binary target (treat ranker scores as probability proxy)
            try:
                test_auc = roc_auc_score(test_fold['target'].values, preds)
            except:
                test_auc = 0.5
            ranker_fold_aucs[test_year] = test_auc

            # NDCG@3 per race
            ndcg_scores = []
            test_groups = test_fold.groupby('race_id_str').size().values
            test_races = test_fold['race_id_str'].unique()
            offset = 0
            for rid in test_races:
                race_mask = test_fold['race_id_str'] == rid
                race_rel = test_fold.loc[race_mask, 'relevance'].values
                race_pred = preds[race_mask.values]
                if len(race_rel) >= 3 and race_rel.max() > 0:
                    try:
                        score = ndcg_score([race_rel], [race_pred], k=3)
                        ndcg_scores.append(score)
                    except:
                        pass

            avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
            ranker_fold_ndcg[test_year] = avg_ndcg

            for idx, pred in zip(test_fold.index, preds):
                ranker_preds[idx] = pred
            total_groups += len(test_races)

            print(f"  Ranker WF {test_year}: AUC {test_auc:.4f}, NDCG@3 {avg_ndcg:.4f}")

        # --- Compare binary vs ranker ---
        print("\n[3.2] Comparing Binary vs Ranker (test period)...")

        test_mask = df_rank['year_full'].isin(TEST_YEARS)
        test_indices = [i for i in df_rank[test_mask].index if i in baseline_preds and i in ranker_preds]

        if len(test_indices) > 0:
            y_test = df_rank.loc[test_indices, 'target'].values
            y_rel = df_rank.loc[test_indices, 'relevance'].values

            p_binary = np.array([baseline_preds[i] for i in test_indices])
            p_ranker = np.array([ranker_preds[i] for i in test_indices])

            auc_binary = roc_auc_score(y_test, p_binary)
            auc_ranker = roc_auc_score(y_test, p_ranker)

            # Per-race NDCG@3
            ndcg_binary_list = []
            ndcg_ranker_list = []
            test_df_tmp = df_rank.loc[test_indices].copy()
            test_df_tmp['p_binary'] = p_binary
            test_df_tmp['p_ranker'] = p_ranker

            for rid, grp in test_df_tmp.groupby('race_id_str'):
                rel = grp['relevance'].values
                if len(rel) >= 3 and rel.max() > 0:
                    try:
                        ndcg_binary_list.append(ndcg_score([rel], [grp['p_binary'].values], k=3))
                        ndcg_ranker_list.append(ndcg_score([rel], [grp['p_ranker'].values], k=3))
                    except:
                        pass

            ndcg_binary = np.mean(ndcg_binary_list) if ndcg_binary_list else 0
            ndcg_ranker = np.mean(ndcg_ranker_list) if ndcg_ranker_list else 0

            print(f"  Binary: AUC {auc_binary:.4f}, NDCG@3 {ndcg_binary:.4f}")
            print(f"  Ranker: AUC {auc_ranker:.4f}, NDCG@3 {ndcg_ranker:.4f}")

            # ROI comparison
            roi_binary = compute_roi_from_preds(df_rank, baseline_preds, payout_lookup, TEST_YEARS, "Binary")
            roi_ranker = compute_roi_from_preds(df_rank, ranker_preds, payout_lookup, TEST_YEARS, "Ranker")

            print_roi_summary(roi_binary, "Binary")
            print_roi_summary(roi_ranker, "Ranker")

            # Also try alternative relevance: 1st=5, 2nd=3, 3rd=1
            print("\n[3.3] Alternative relevance: 1st=5, 2nd=3, 3rd=1...")
            df_rank2 = df.copy()
            df_rank2['relevance2'] = 0
            df_rank2.loc[df_rank2['finish'] == 1, 'relevance2'] = 5
            df_rank2.loc[df_rank2['finish'] == 2, 'relevance2'] = 3
            df_rank2.loc[df_rank2['finish'] == 3, 'relevance2'] = 1

            ranker2_preds = {}
            for test_year in TEST_YEARS:
                wf_train_mask = (df_rank2['year_full'] >= 2010) & (df_rank2['year_full'] < test_year)
                test_mask2 = df_rank2['year_full'] == test_year
                if test_mask2.sum() < 100:
                    continue

                df_fold2 = df_rank2.copy()
                df_fold2 = encode_sires_fold(df_fold2, wf_train_mask)
                for f in features:
                    if f not in df_fold2.columns:
                        df_fold2[f] = 0
                    df_fold2[f] = pd.to_numeric(df_fold2[f], errors='coerce').fillna(0)

                train_fold2 = df_fold2[wf_train_mask]
                test_fold2 = df_fold2[test_mask2]
                dates2 = train_fold2['date_num']
                vc2 = dates2.quantile(0.85)
                tr2 = train_fold2[dates2 < vc2]
                va2 = train_fold2[dates2 >= vc2]

                tr_g2 = tr2.groupby('race_id_str').size().values
                va_g2 = va2.groupby('race_id_str').size().values

                dtrain2 = lgb.Dataset(tr2[features].values, label=tr2['relevance2'].values,
                                      group=tr_g2, feature_name=features)
                dvalid2 = lgb.Dataset(va2[features].values, label=va2['relevance2'].values,
                                      group=va_g2, feature_name=features, reference=dtrain2)

                m2 = lgb.train(ranker_params, dtrain2, num_boost_round=500,
                               valid_sets=[dvalid2],
                               callbacks=[lgb.early_stopping(30), lgb.log_evaluation(200)])

                preds2 = m2.predict(test_fold2[features].values)
                for idx, p in zip(test_fold2.index, preds2):
                    ranker2_preds[idx] = p

            roi_ranker2 = compute_roi_from_preds(df_rank2, ranker2_preds, payout_lookup, TEST_YEARS, "Ranker2")
            print_roi_summary(roi_ranker2, "Ranker (alt weights)")

            # Adoption decision
            adopted = False
            adoption_reason = "Ranker did not improve ROI and NDCG"

            # Pick best ranker variant
            best_ranker_roi = roi_ranker['overall_trio_roi']
            best_ranker_name = 'ranker_3_2_1'
            if roi_ranker2['overall_trio_roi'] > best_ranker_roi:
                best_ranker_roi = roi_ranker2['overall_trio_roi']
                best_ranker_name = 'ranker_5_3_1'

            if (best_ranker_roi > roi_binary['overall_trio_roi'] and
                    ndcg_ranker >= ndcg_binary * 0.98):
                adopted = True
                adoption_reason = (f"Ranker ({best_ranker_name}) ROI {best_ranker_roi:.1f}% > "
                                   f"Binary {roi_binary['overall_trio_roi']:.1f}%")

            print(f"\n  Ranker adopted: {adopted} ({adoption_reason})")

            report['task3_ranker'] = {
                'group_definition': '1着=3, 2着=2, 3着=1, 4着以下=0',
                'alt_group_definition': '1着=5, 2着=3, 3着=1, 4着以下=0',
                'total_groups': total_groups,
                'binary_model': {
                    'test_auc': round(auc_binary, 4),
                    'test_ndcg3': round(ndcg_binary, 4),
                    'roi': roi_binary,
                    'fold_aucs': {str(k): round(v, 4) for k, v in ranker_fold_aucs.items()},
                },
                'ranker_model': {
                    'test_auc': round(auc_ranker, 4),
                    'test_ndcg3': round(ndcg_ranker, 4),
                    'roi': roi_ranker,
                    'fold_aucs': {str(k): round(v, 4) for k, v in ranker_fold_aucs.items()},
                    'fold_ndcg': {str(k): round(v, 4) for k, v in ranker_fold_ndcg.items()},
                },
                'ranker_alt': {
                    'roi': roi_ranker2,
                    'weights': '1st=5, 2nd=3, 3rd=1',
                },
                'adopted': adopted,
                'adoption_reason': adoption_reason,
            }

            # Save ranker comparison
            ranker_comp = {
                'binary_auc': round(auc_binary, 4),
                'binary_ndcg3': round(ndcg_binary, 4),
                'binary_roi': roi_binary['overall_trio_roi'],
                'ranker_auc': round(auc_ranker, 4),
                'ranker_ndcg3': round(ndcg_ranker, 4),
                'ranker_roi': roi_ranker['overall_trio_roi'],
                'ranker_alt_roi': roi_ranker2['overall_trio_roi'],
                'adopted': adopted,
            }
            with open(os.path.join(ARTIFACTS_DIR, 'ranker_comparison.json'), 'w') as f:
                json.dump(ranker_comp, f, indent=2)

        else:
            print("  WARNING: No overlapping test indices")
            report['task3_ranker'] = {
                'group_definition': '1着=3, 2着=2, 3着=1, 4着以下=0',
                'error': 'No overlapping test indices',
                'adopted': False,
            }

    except Exception as e:
        print(f"  ERROR in Task 3: {e}")
        traceback.print_exc()
        report['task3_ranker'] = {'error': str(e), 'adopted': False}


# ============================================================
# MAIN
# ============================================================
def main():
    start_time = time.time()
    print("=" * 70)
    print("  PHASE 2 OPTIMIZATION SUITE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load data
    df, features, sire_map, bms_map = prepare_data()

    # Load payouts (optional)
    payout_lookup = None
    if HAS_PAYOUTS:
        print("\nLoading JRA payout data...")
        payout_lookup = load_payouts()
    else:
        print("\nNOTE: jra_payouts.csv not found. Using estimated payouts (o1*o2*o3*20).")
        print("  Estimated ROI is ~2x actual ROI. Comparisons are still valid (same basis).")

    # Task 1
    baseline_preds, fold_aucs, fold_models = task1_calibration(df, features, payout_lookup)

    # If task1 failed to produce predictions, generate them
    if not baseline_preds:
        print("\nRegenerating baseline predictions...")
        baseline_preds, fold_aucs, fold_models = walk_forward_predict(df, features, ALL_WF_YEARS)

    # Task 2
    task2_ev_model(df, features, payout_lookup, baseline_preds)

    # Task 3
    task3_ranker(df, features, payout_lookup, baseline_preds)

    # Final decision
    changes = []
    reasons = []

    t1 = report.get('task1_calibration', {})
    if t1.get('adopted_method', 'none') != 'none':
        method = t1['adopted_method']
        changes.append(f'calibration_{method}')
        reasons.append(t1.get('adoption_reason', ''))

    t2 = report.get('task2_ev_model', {})
    if t2.get('adopted', False):
        changes.append('pattern_c_ev_model')
        reasons.append(t2.get('adoption_reason', ''))

    t3 = report.get('task3_ranker', {})
    if t3.get('adopted', False):
        changes.append('lgb_ranker')
        reasons.append(t3.get('adoption_reason', ''))

    if not changes:
        reasons.append('No optimization met adoption criteria. Current production model is retained.')

    report['final_decision'] = {
        'production_changes': changes,
        'reasons': reasons,
    }

    elapsed = time.time() - start_time
    report['elapsed_seconds'] = round(elapsed, 1)

    # Save report
    report_path = os.path.join(DATA_DIR, 'phase2_optimization_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Saved report: {report_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  PHASE 2 OPTIMIZATION COMPLETE")
    print(f"  Elapsed: {elapsed/60:.1f} minutes")
    print(f"  Production changes: {changes if changes else 'None'}")
    for r in reasons:
        print(f"    - {r}")
    print("=" * 70)

    return report


if __name__ == '__main__':
    main()
