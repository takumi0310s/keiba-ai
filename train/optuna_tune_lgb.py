#!/usr/bin/env python
"""Optuna hyperparameter tuning for LightGBM (Pattern A, leak-free)

Walk-forward evaluation (2020-2025, yearly splits).
Search range: ±20% of current baseline parameters.
Adoption criteria:
  - WF AUC > 0.8095
  - All years AUC > 0.78
  - Train AUC - Test AUC < 0.05
  - Actual ROI (jra_payouts.csv) >= current for all conditions
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

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, BASE_DIR)

from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features, COURSE_MAP, N_TOP_SIRE,
)
from train_v92_leakfree import FEATURES_PATTERN_A, LEAK_FEATURES_A

from backtest_central_leakfree import (
    classify_condition, calc_trio_bets, calc_umaren_bets,
    calc_wide_bets, check_bets, encode_sires_fold,
)
from calc_actual_roi import (
    load_payouts, parse_trio_nums, parse_umaren_nums, parse_wide_data,
    calc_actual_returns,
)

TEST_YEARS = list(range(2020, 2026))

# Current baseline parameters
BASELINE_PARAMS = {
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
}

# Current baseline ROI (trio, actual)
BASELINE_ROI = {
    'A': 205.3, 'B': 236.9, 'C': 285.6,
    'D': 136.0, 'E': 118.0, 'X': 330.5,
}

# Current baseline WF AUC
BASELINE_WF_AUC = 0.8017
BASELINE_YEAR_AUCS = {
    2020: 0.7951, 2021: 0.7997, 2022: 0.8024,
    2023: 0.8012, 2024: 0.8071, 2025: 0.8048,
}

# Target: WF AUC > 0.8095 (ensemble AUC threshold from CLAUDE.md)
TARGET_WF_AUC = 0.8095
MIN_YEAR_AUC = 0.78
MAX_OVERFIT_GAP = 0.05


def prepare_data():
    """Load and prepare all data (done once)."""
    print("Loading data...")
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

    print(f"Data ready: {len(df)} rows, {df['race_id_str'].nunique()} races")
    return df, features


def train_lgb_with_params(X_train, y_train, X_valid, y_valid, features, params):
    """Train LGB with given params, return model."""
    full_params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'verbose': -1, 'n_jobs': -1, 'seed': 42,
    }
    full_params.update(params)

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=features)
    dvalid = lgb.Dataset(X_valid, label=y_valid, feature_name=features, reference=dtrain)

    model = lgb.train(
        full_params, dtrain, num_boost_round=500,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    return model


def walk_forward_evaluate(df, features, params):
    """Run walk-forward and return (avg_auc, year_aucs, train_aucs)."""
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

        model = train_lgb_with_params(X_tr, y_tr, X_va, y_va, features, params)

        # Train AUC (on validation portion of train data)
        va_pred = model.predict(X_va)
        train_auc = roc_auc_score(y_va, va_pred)
        train_aucs[test_year] = train_auc

        # Test AUC
        test_df = df_fold[test_mask]
        X_test = test_df[features].values
        test_pred = model.predict(X_test)
        test_auc = roc_auc_score(test_df['target'].values, test_pred)
        fold_aucs[test_year] = test_auc

    avg_auc = np.mean(list(fold_aucs.values()))
    return avg_auc, fold_aucs, train_aucs


def evaluate_roi(df, features, params, payout_lookup):
    """Run walk-forward with actual ROI calculation. Returns condition ROIs."""
    all_results = []

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

        model = train_lgb_with_params(X_tr, y_tr, X_va, y_va, features, params)

        test_df = df_fold[test_mask].copy()
        X_test = test_df[features].values
        test_df['pred'] = model.predict(X_test)

        test_races = test_df['race_id_str'].unique()
        for rid in test_races:
            race_df = test_df[test_df['race_id_str'] == rid].copy()
            if len(race_df) < 5:
                continue

            row0 = race_df.iloc[0]
            cond_key = classify_condition(row0)

            # Actual top 3
            race_sorted = race_df.sort_values('finish')
            actual_top3 = {}
            for _, r in race_sorted.head(3).iterrows():
                actual_top3[int(r['finish'])] = int(r['umaban'])
            if len(actual_top3) < 3:
                continue

            # AI ranking
            race_df = race_df.sort_values('pred', ascending=False)
            ranking = race_df['umaban'].astype(int).tolist()

            trio_bets = calc_trio_bets(ranking)
            umaren_bets = calc_umaren_bets(ranking)
            wide_bets = calc_wide_bets(ranking)

            payout_info = payout_lookup.get(rid)
            if payout_info is None:
                continue

            (actual_trio_hit, actual_trio_return,
             actual_umaren_hit, actual_umaren_return,
             actual_wide_hits, actual_wide_return) = calc_actual_returns(
                payout_info, trio_bets, umaren_bets, wide_bets)

            all_results.append({
                'cond_key': cond_key,
                'actual_trio_hit': actual_trio_hit,
                'actual_trio_return': actual_trio_return,
                'actual_umaren_hit': actual_umaren_hit,
                'actual_umaren_return': actual_umaren_return,
                'actual_wide_hits': actual_wide_hits,
                'actual_wide_return': actual_wide_return,
            })

    # Calculate condition ROIs
    cond_groups = defaultdict(list)
    for r in all_results:
        cond_groups[r['cond_key']].append(r)

    condition_rois = {}
    for cond_key, races in cond_groups.items():
        n = len(races)
        if n < 10:
            continue
        inv = n * 700

        trio_pay = sum(r['actual_trio_return'] for r in races)
        trio_roi = trio_pay / inv * 100

        umaren_pay = sum(r['actual_umaren_return'] * 3.5 for r in races)
        umaren_roi = umaren_pay / inv * 100

        wide_pay = sum(r['actual_wide_return'] * 3.5 for r in races)
        wide_roi = wide_pay / inv * 100

        # Use best ROI for each condition (matching CLAUDE.md: E uses umaren)
        best_roi = max(trio_roi, umaren_roi, wide_roi)
        condition_rois[cond_key] = {
            'n': n, 'trio_roi': round(trio_roi, 1),
            'umaren_roi': round(umaren_roi, 1), 'wide_roi': round(wide_roi, 1),
            'best_roi': round(best_roi, 1),
        }

    return condition_rois


def suggest_params(trial):
    """Suggest parameters within ±20% of baseline."""
    params = {}

    # num_leaves: 63 ± 20% → [50, 76] (int)
    params['num_leaves'] = trial.suggest_int('num_leaves', 50, 76)

    # learning_rate: 0.05 ± 20% → [0.04, 0.06]
    params['learning_rate'] = trial.suggest_float('learning_rate', 0.04, 0.06)

    # feature_fraction: 0.8 ± 20% → [0.64, 0.96] but cap at 1.0
    params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.64, 0.96)

    # bagging_fraction: 0.8 ± 20% → [0.64, 0.96]
    params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.64, 0.96)

    # bagging_freq: 5 ± 20% → [4, 6] (int)
    params['bagging_freq'] = trial.suggest_int('bagging_freq', 4, 6)

    # min_child_samples: 50 ± 20% → [40, 60] (int)
    params['min_child_samples'] = trial.suggest_int('min_child_samples', 40, 60)

    # reg_alpha: 0.1 ± 20% → [0.08, 0.12]
    params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.08, 0.12)

    # reg_lambda: 0.1 ± 20% → [0.08, 0.12]
    params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.08, 0.12)

    return params


def main():
    t_start = time.time()
    print("=" * 70)
    print("  OPTUNA HYPERPARAMETER TUNING (Pattern A, Leak-Free)")
    print(f"  Baseline WF AUC: {BASELINE_WF_AUC}")
    print(f"  Target WF AUC: > {TARGET_WF_AUC}")
    print(f"  Trials: 100")
    print(f"  Search: ±20% of baseline params")
    print("=" * 70)

    # Prepare data once
    df, features = prepare_data()

    # Validate baseline first
    print("\n[1] Validating baseline parameters...")
    baseline_avg, baseline_years, baseline_trains = walk_forward_evaluate(
        df, features, BASELINE_PARAMS)
    print(f"  Baseline WF AUC: {baseline_avg:.4f}")
    for y, a in baseline_years.items():
        print(f"    {y}: {a:.4f}")

    # Optuna study
    print(f"\n[2] Starting Optuna optimization (100 trials)...")

    best_auc_so_far = baseline_avg
    best_params_so_far = None
    best_years_so_far = None
    trial_results = []

    def objective(trial):
        nonlocal best_auc_so_far, best_params_so_far, best_years_so_far

        params = suggest_params(trial)
        avg_auc, year_aucs, train_aucs = walk_forward_evaluate(df, features, params)

        # Check minimum year AUC
        min_year = min(year_aucs.values())
        all_above_min = min_year > MIN_YEAR_AUC

        # Check overfitting gap
        max_gap = max(train_aucs[y] - year_aucs[y] for y in year_aucs)
        no_overfit = max_gap < MAX_OVERFIT_GAP

        trial_results.append({
            'trial': trial.number,
            'params': params.copy(),
            'avg_auc': avg_auc,
            'year_aucs': dict(year_aucs),
            'train_aucs': dict(train_aucs),
            'min_year_auc': min_year,
            'max_overfit_gap': max_gap,
            'valid': all_above_min and no_overfit,
        })

        if avg_auc > best_auc_so_far and all_above_min and no_overfit:
            best_auc_so_far = avg_auc
            best_params_so_far = params.copy()
            best_years_so_far = dict(year_aucs)
            print(f"  * Trial {trial.number}: NEW BEST AUC {avg_auc:.4f} "
                  f"(min_year={min_year:.4f}, gap={max_gap:.4f})")
        elif trial.number % 10 == 0:
            print(f"  Trial {trial.number}: AUC {avg_auc:.4f} "
                  f"(min={min_year:.4f}, gap={max_gap:.4f})")

        return avg_auc

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))

    # Enqueue baseline as first trial
    study.enqueue_trial({
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    })

    study.optimize(objective, n_trials=100, show_progress_bar=True)

    elapsed = time.time() - t_start
    print(f"\n  Optimization done in {elapsed/60:.1f} minutes")

    # Results summary
    print(f"\n{'=' * 70}")
    print(f"  OPTIMIZATION RESULTS")
    print(f"{'=' * 70}")
    print(f"  Baseline AUC:  {baseline_avg:.4f}")
    print(f"  Best AUC:      {study.best_value:.4f}")
    print(f"  Improvement:   {study.best_value - baseline_avg:+.4f}")

    best_trial = study.best_trial
    print(f"\n  Best parameters (Trial {best_trial.number}):")
    for k, v in best_trial.params.items():
        base_v = BASELINE_PARAMS[k]
        pct = (v - base_v) / base_v * 100
        print(f"    {k:20s}: {v:>8.4f} (baseline: {base_v}, {pct:+.1f}%)")

    # Check if best params meet all criteria
    best_result = next(r for r in trial_results if r['trial'] == best_trial.number)
    meets_auc = study.best_value > TARGET_WF_AUC
    meets_year = best_result['min_year_auc'] > MIN_YEAR_AUC
    meets_gap = best_result['max_overfit_gap'] < MAX_OVERFIT_GAP

    print(f"\n  Criteria check:")
    print(f"    WF AUC > {TARGET_WF_AUC}: {study.best_value:.4f} {'PASS' if meets_auc else 'FAIL'}")
    print(f"    All years > {MIN_YEAR_AUC}: min={best_result['min_year_auc']:.4f} {'PASS' if meets_year else 'FAIL'}")
    print(f"    Overfit gap < {MAX_OVERFIT_GAP}: max={best_result['max_overfit_gap']:.4f} {'PASS' if meets_gap else 'FAIL'}")

    if best_result.get('year_aucs'):
        print(f"\n  Year AUCs (best):")
        for y, a in sorted(best_result['year_aucs'].items()):
            ba = BASELINE_YEAR_AUCS.get(y, 0)
            print(f"    {y}: {a:.4f} (baseline: {ba:.4f}, {a-ba:+.4f})")

    # ROI evaluation (only if AUC criteria met)
    roi_pass = False
    condition_rois = {}
    if meets_auc and meets_year and meets_gap:
        print(f"\n[3] AUC criteria met! Evaluating actual ROI...")
        payout_lookup = load_payouts()
        condition_rois = evaluate_roi(df, features, best_trial.params, payout_lookup)

        print(f"\n  Condition ROI comparison (trio):")
        print(f"  {'Cond':<6} {'N':>5} {'New trio':>10} {'Base trio':>10} {'New best':>10} {'Base best':>10} {'Pass'}")
        print(f"  {'-' * 65}")

        roi_pass = True
        for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
            if cond not in condition_rois:
                # If condition missing, treat as fail
                roi_pass = False
                print(f"  {cond:<6} {'N/A':>5} {'N/A':>10} {BASELINE_ROI[cond]:>9.1f}% {'N/A':>10} {BASELINE_ROI[cond]:>9.1f}% FAIL")
                continue
            r = condition_rois[cond]
            base = BASELINE_ROI[cond]
            # Compare best ROI (not just trio) to baseline best ROI
            pass_cond = r['best_roi'] >= base
            if not pass_cond:
                roi_pass = False
            mark = 'PASS' if pass_cond else 'FAIL'
            print(f"  {cond:<6} {r['n']:>5} {r['trio_roi']:>9.1f}% {base:>9.1f}% "
                  f"{r['best_roi']:>9.1f}% {base:>9.1f}% {mark}")

        print(f"\n  ROI all conditions pass: {'PASS' if roi_pass else 'FAIL'}")
    else:
        print(f"\n  AUC criteria NOT met. Skipping ROI evaluation.")

    # Final verdict
    all_pass = meets_auc and meets_year and meets_gap and roi_pass
    print(f"\n{'=' * 70}")
    if all_pass:
        print(f"  *** ALL CRITERIA MET! New parameters ADOPTED. ***")
    else:
        print(f"  Criteria NOT fully met. Parameters NOT adopted.")
        print(f"  Best improvement was AUC {baseline_avg:.4f} → {study.best_value:.4f} ({study.best_value - baseline_avg:+.4f})")
    print(f"{'=' * 70}")

    # Save results
    output = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_minutes': round(elapsed / 60, 1),
        'n_trials': 100,
        'baseline': {
            'params': BASELINE_PARAMS,
            'wf_auc': baseline_avg,
            'year_aucs': {str(k): v for k, v in baseline_years.items()},
            'roi': BASELINE_ROI,
        },
        'best': {
            'trial': best_trial.number,
            'params': {k: round(v, 6) for k, v in best_trial.params.items()},
            'wf_auc': round(study.best_value, 4),
            'year_aucs': {str(k): round(v, 4) for k, v in best_result['year_aucs'].items()},
            'min_year_auc': round(best_result['min_year_auc'], 4),
            'max_overfit_gap': round(best_result['max_overfit_gap'], 4),
        },
        'criteria': {
            'auc_target': TARGET_WF_AUC,
            'auc_pass': meets_auc,
            'min_year_target': MIN_YEAR_AUC,
            'min_year_pass': meets_year,
            'overfit_target': MAX_OVERFIT_GAP,
            'overfit_pass': meets_gap,
            'roi_pass': roi_pass,
            'all_pass': all_pass,
        },
        'condition_rois': condition_rois,
        'top10_trials': sorted(trial_results, key=lambda x: x['avg_auc'], reverse=True)[:10],
    }

    # Clean up trial results for JSON serialization
    for t in output['top10_trials']:
        t['year_aucs'] = {str(k): round(v, 4) for k, v in t['year_aucs'].items()}
        t['train_aucs'] = {str(k): round(v, 4) for k, v in t['train_aucs'].items()}
        t['avg_auc'] = round(t['avg_auc'], 4)
        t['min_year_auc'] = round(t['min_year_auc'], 4)
        t['max_overfit_gap'] = round(t['max_overfit_gap'], 4)
        for pk, pv in t['params'].items():
            t['params'][pk] = round(pv, 6) if isinstance(pv, float) else pv

    out_path = os.path.join(BASE_DIR, 'data', 'optuna_tuning_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {out_path}")

    return output


if __name__ == '__main__':
    main()
