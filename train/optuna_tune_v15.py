#!/usr/bin/env python3
"""Optuna hyperparameter tuning for LGB & XGB (v15 / v14.1 feature set)

Walk-forward evaluation (2021-2025, yearly splits).
Uses LGB/XGB only (no FT/IR) for speed during search.

Adoption criteria:
  - WF AUC improvement >= +0.0005 over current params
  - Train-Test AUC gap < 0.05 for ALL years

Usage:
    python train/optuna_tune_v15.py                  # Full run (100+100 trials)
    python train/optuna_tune_v15.py --trials 50      # Fewer trials
    python train/optuna_tune_v15.py --lgb-only       # LGB only
    python train/optuna_tune_v15.py --xgb-only       # XGB only
"""

import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# --- Optuna import with graceful fallback ---
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("optuna not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'optuna'])
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score


# =====================================================
# Constants
# =====================================================

WF_YEARS = list(range(2021, 2026))  # 5-fold walk-forward
MIN_IMPROVEMENT = 0.0005
MAX_GAP = 0.05

# Current baseline params (from train_v135_ft_transformer.py / CLAUDE.md)
CURRENT_LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'seed': 42,
}

CURRENT_XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 6,
    'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'seed': 42, 'tree_method': 'hist',
}


# =====================================================
# Data loading
# =====================================================

_DF_CACHE = {}

def load_dataframe():
    """Load v15 or v14.1 dataframe (cached)."""
    if 'df' in _DF_CACHE:
        return _DF_CACHE['df'], _DF_CACHE['features']

    print("=" * 70)
    print("  Loading dataframe...")
    print("=" * 70)

    try:
        from train_v15_master import build_v15_dataframe, get_v15_all_features, fill_v15_defaults
        print("  Using v15 dataframe (v14.1 + v15 new features)")
        df, sire_map, bms_map = build_v15_dataframe()
        features, _ = get_v15_all_features()
        df = fill_v15_defaults(df, features)
        version = 'v15'
    except Exception as e:
        print(f"  v15 not available ({e}), falling back to v14.1")
        try:
            from train_v141_paci_tierA import build_v141_dataframe, get_v141_features, fill_v141_defaults
            df, sire_map, bms_map = build_v141_dataframe()
            features, _ = get_v141_features()
            df = fill_v141_defaults(df, features)
            version = 'v14.1'
        except Exception as e2:
            print(f"  v14.1 not available ({e2}), falling back to v13.5b")
            from train_v135_ft_transformer import build_v134_dataframe, get_v134_features, fill_defaults
            df, sire_map, bms_map = build_v134_dataframe()
            features, _ = get_v134_features()
            df = fill_defaults(df, features)
            version = 'v13.5b'

    # Ensure target
    if 'target' not in df.columns:
        df['target'] = (df['finish'] <= 3).astype(int)

    # Ensure year_full
    if 'year_full' not in df.columns:
        if 'year' in df.columns:
            yr = df['year'].values
            df['year_full'] = np.where(yr < 100, yr + 2000, yr)
        else:
            df['year_full'] = 2020

    # Filter
    df = df[df['num_horses_val'] >= 5].copy()

    # Ensure numeric
    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"  Data ready: {len(df):,} rows, {len(features)} features ({version})")

    _DF_CACHE['df'] = df
    _DF_CACHE['features'] = features
    return df, features


# =====================================================
# Walk-forward evaluation (single model)
# =====================================================

def wf_evaluate_lgb(df, features, params):
    """Walk-forward LGB evaluation. Returns (mean_auc, year_aucs, gaps)."""
    full_params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'verbose': -1, 'n_jobs': -1, 'seed': 42,
    }
    full_params.update(params)

    year_aucs = {}
    gaps = {}

    for test_year in WF_YEARS:
        # year column may be 2-digit (e.g. 21) or 4-digit (e.g. 2021)
        ty2 = test_year - 2000
        if df['year'].max() > 100:
            train_mask = df['year'] < test_year
            test_mask = df['year'] == test_year
        else:
            train_mask = df['year'] < ty2
            test_mask = df['year'] == ty2

        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[test_mask, features].values
        y_te = df.loc[test_mask, 'target'].values

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)

        model = lgb.train(
            full_params, dtrain, num_boost_round=1000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        p_te = model.predict(X_te)
        p_tr = model.predict(X_tr)
        auc_te = roc_auc_score(y_te, p_te)
        auc_tr = roc_auc_score(y_tr, p_tr)

        year_aucs[test_year] = auc_te
        gaps[test_year] = auc_tr - auc_te

    if not year_aucs:
        return 0.0, {}, {}

    mean_auc = np.mean(list(year_aucs.values()))
    return mean_auc, year_aucs, gaps


def wf_evaluate_xgb(df, features, params):
    """Walk-forward XGB evaluation. Returns (mean_auc, year_aucs, gaps)."""
    full_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'seed': 42, 'tree_method': 'hist', 'verbosity': 0,
    }
    full_params.update(params)

    year_aucs = {}
    gaps = {}

    for test_year in WF_YEARS:
        ty2 = test_year - 2000
        if df['year'].max() > 100:
            train_mask = df['year'] < test_year
            test_mask = df['year'] == test_year
        else:
            train_mask = df['year'] < ty2
            test_mask = df['year'] == ty2

        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[test_mask, features].values
        y_te = df.loc[test_mask, 'target'].values

        dxtr = xgb.DMatrix(X_tr, label=y_tr)
        dxte = xgb.DMatrix(X_te, label=y_te)

        model = xgb.train(
            full_params, dxtr, num_boost_round=1000,
            evals=[(dxte, 'valid')],
            early_stopping_rounds=50, verbose_eval=False,
        )

        p_te = model.predict(dxte)
        p_tr = model.predict(dxtr)
        auc_te = roc_auc_score(y_te, p_te)
        auc_tr = roc_auc_score(y_tr, p_tr)

        year_aucs[test_year] = auc_te
        gaps[test_year] = auc_tr - auc_te

    if not year_aucs:
        return 0.0, {}, {}

    mean_auc = np.mean(list(year_aucs.values()))
    return mean_auc, year_aucs, gaps


# =====================================================
# Optuna objectives
# =====================================================

def lgb_objective(trial, df, features):
    """Optuna objective for LGB hyperparameter search."""
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
        'bagging_freq': 5,
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0, log=True),
    }

    mean_auc, year_aucs, gaps = wf_evaluate_lgb(df, features, params)

    # Prune if gap too large in any year
    for yr, gap in gaps.items():
        if gap > MAX_GAP:
            trial.set_user_attr('pruned_reason', f'gap={gap:.4f} in {yr}')
            raise optuna.TrialPruned()

    # Store details for later inspection
    trial.set_user_attr('year_aucs', {str(k): round(v, 6) for k, v in year_aucs.items()})
    trial.set_user_attr('gaps', {str(k): round(v, 4) for k, v in gaps.items()})

    return mean_auc


def xgb_objective(trial, df, features):
    """Optuna objective for XGB hyperparameter search."""
    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 20, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0, log=True),
    }

    mean_auc, year_aucs, gaps = wf_evaluate_xgb(df, features, params)

    # Prune if gap too large
    for yr, gap in gaps.items():
        if gap > MAX_GAP:
            trial.set_user_attr('pruned_reason', f'gap={gap:.4f} in {yr}')
            raise optuna.TrialPruned()

    trial.set_user_attr('year_aucs', {str(k): round(v, 6) for k, v in year_aucs.items()})
    trial.set_user_attr('gaps', {str(k): round(v, 4) for k, v in gaps.items()})

    return mean_auc


# =====================================================
# Tuning runners
# =====================================================

def tune_lgb(df, features, n_trials=100):
    """Run Optuna LGB tuning."""
    print("\n" + "=" * 70)
    print(f"  LGB Hyperparameter Tuning ({n_trials} trials)")
    print("=" * 70)

    # Baseline evaluation
    print("\n  [Baseline] Evaluating current LGB params...")
    baseline_params = {k: v for k, v in CURRENT_LGB_PARAMS.items()
                       if k not in ('objective', 'metric', 'boosting_type', 'verbose', 'seed')}
    baseline_auc, baseline_years, baseline_gaps = wf_evaluate_lgb(df, features, baseline_params)
    print(f"  Baseline WF AUC: {baseline_auc:.6f}")
    for yr in sorted(baseline_years):
        print(f"    {yr}: AUC={baseline_years[yr]:.6f}, gap={baseline_gaps[yr]:.4f}")

    # Optuna study
    study = optuna.create_study(direction='maximize', study_name='lgb_v15')
    t0 = time.time()

    # Enqueue current params as first trial
    study.enqueue_trial({
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'min_child_samples': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    })

    study.optimize(
        lambda trial: lgb_objective(trial, df, features),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    elapsed = (time.time() - t0) / 60
    best = study.best_trial
    best_auc = best.value
    delta = best_auc - baseline_auc
    adopted = delta >= MIN_IMPROVEMENT

    # Check gaps for best trial
    best_gaps = best.user_attrs.get('gaps', {})
    gap_ok = all(float(v) < MAX_GAP for v in best_gaps.values())
    if not gap_ok:
        adopted = False

    print(f"\n  LGB Tuning Complete ({elapsed:.1f} min)")
    print(f"  Best WF AUC: {best_auc:.6f} (delta: {delta:+.6f})")
    print(f"  Best params: {best.params}")
    if best.user_attrs.get('year_aucs'):
        for yr, auc in sorted(best.user_attrs['year_aucs'].items()):
            g = best_gaps.get(yr, '?')
            print(f"    {yr}: AUC={auc}, gap={g}")
    print(f"  Gap check: {'PASS' if gap_ok else 'FAIL'}")
    print(f"  Adoption: {'YES' if adopted else 'NO'} "
          f"(need delta >= {MIN_IMPROVEMENT:+.4f}, got {delta:+.6f})")

    # Top 5 trials
    print(f"\n  Top 5 trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else 0,
                           reverse=True)
    for i, t in enumerate(trials_sorted[:5]):
        print(f"    #{t.number}: AUC={t.value:.6f} params={t.params}")

    result = {
        'best_params': best.params,
        'best_auc': round(best_auc, 6),
        'baseline_auc': round(baseline_auc, 6),
        'delta': round(delta, 6),
        'adopted': adopted,
        'gap_ok': gap_ok,
        'year_aucs': best.user_attrs.get('year_aucs', {}),
        'gaps': best_gaps,
        'n_trials': len(study.trials),
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'elapsed_minutes': round(elapsed, 1),
        'baseline_year_aucs': {str(k): round(v, 6) for k, v in baseline_years.items()},
    }
    return result


def tune_xgb(df, features, n_trials=100):
    """Run Optuna XGB tuning."""
    print("\n" + "=" * 70)
    print(f"  XGB Hyperparameter Tuning ({n_trials} trials)")
    print("=" * 70)

    # Baseline evaluation
    print("\n  [Baseline] Evaluating current XGB params...")
    baseline_params = {k: v for k, v in CURRENT_XGB_PARAMS.items()
                       if k not in ('objective', 'eval_metric', 'seed', 'tree_method')}
    baseline_auc, baseline_years, baseline_gaps = wf_evaluate_xgb(df, features, baseline_params)
    print(f"  Baseline WF AUC: {baseline_auc:.6f}")
    for yr in sorted(baseline_years):
        print(f"    {yr}: AUC={baseline_years[yr]:.6f}, gap={baseline_gaps[yr]:.4f}")

    # Optuna study
    study = optuna.create_study(direction='maximize', study_name='xgb_v15')
    t0 = time.time()

    # Enqueue current params as first trial
    study.enqueue_trial({
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    })

    study.optimize(
        lambda trial: xgb_objective(trial, df, features),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    elapsed = (time.time() - t0) / 60
    best = study.best_trial
    best_auc = best.value
    delta = best_auc - baseline_auc
    adopted = delta >= MIN_IMPROVEMENT

    best_gaps = best.user_attrs.get('gaps', {})
    gap_ok = all(float(v) < MAX_GAP for v in best_gaps.values())
    if not gap_ok:
        adopted = False

    print(f"\n  XGB Tuning Complete ({elapsed:.1f} min)")
    print(f"  Best WF AUC: {best_auc:.6f} (delta: {delta:+.6f})")
    print(f"  Best params: {best.params}")
    if best.user_attrs.get('year_aucs'):
        for yr, auc in sorted(best.user_attrs['year_aucs'].items()):
            g = best_gaps.get(yr, '?')
            print(f"    {yr}: AUC={auc}, gap={g}")
    print(f"  Gap check: {'PASS' if gap_ok else 'FAIL'}")
    print(f"  Adoption: {'YES' if adopted else 'NO'} "
          f"(need delta >= {MIN_IMPROVEMENT:+.4f}, got {delta:+.6f})")

    # Top 5 trials
    print(f"\n  Top 5 trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else 0,
                           reverse=True)
    for i, t in enumerate(trials_sorted[:5]):
        print(f"    #{t.number}: AUC={t.value:.6f} params={t.params}")

    result = {
        'best_params': best.params,
        'best_auc': round(best_auc, 6),
        'baseline_auc': round(baseline_auc, 6),
        'delta': round(delta, 6),
        'adopted': adopted,
        'gap_ok': gap_ok,
        'year_aucs': best.user_attrs.get('year_aucs', {}),
        'gaps': best_gaps,
        'n_trials': len(study.trials),
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'elapsed_minutes': round(elapsed, 1),
        'baseline_year_aucs': {str(k): round(v, 6) for k, v in baseline_years.items()},
    }
    return result


# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser(description='Optuna v15 hyperparameter tuning')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials per model')
    parser.add_argument('--lgb-only', action='store_true', help='Tune LGB only')
    parser.add_argument('--xgb-only', action='store_true', help='Tune XGB only')
    args = parser.parse_args()

    do_lgb = not args.xgb_only
    do_xgb = not args.lgb_only

    total_start = time.time()

    print("=" * 70)
    print("  Optuna Hyperparameter Tuning (v15)")
    print(f"  Trials: {args.trials} per model")
    print(f"  Models: {'LGB' if do_lgb else ''} {'XGB' if do_xgb else ''}")
    print(f"  WF years: {WF_YEARS}")
    print(f"  Min improvement: {MIN_IMPROVEMENT:+.4f}")
    print(f"  Max gap: {MAX_GAP}")
    print("=" * 70)

    # Load data once
    df, features = load_dataframe()

    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'wf_years': WF_YEARS,
        'n_features': len(features),
        'n_rows': len(df),
        'min_improvement': MIN_IMPROVEMENT,
        'max_gap': MAX_GAP,
    }

    # --- LGB ---
    if do_lgb:
        lgb_result = tune_lgb(df, features, n_trials=args.trials)
        results['lgb'] = lgb_result
        results['trials_lgb'] = lgb_result['n_trials']

    # --- XGB ---
    if do_xgb:
        xgb_result = tune_xgb(df, features, n_trials=args.trials)
        results['xgb'] = xgb_result
        results['trials_xgb'] = xgb_result['n_trials']

    total_elapsed = (time.time() - total_start) / 60
    results['elapsed_minutes'] = round(total_elapsed, 1)

    # Save results
    out_path = os.path.join(DATA_DIR, 'optuna_v15_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if do_lgb:
        r = results['lgb']
        status = "ADOPTED" if r['adopted'] else "NOT ADOPTED"
        print(f"  LGB: {r['baseline_auc']:.6f} -> {r['best_auc']:.6f} "
              f"(delta: {r['delta']:+.6f}) [{status}]")
        if r['adopted']:
            print(f"    New params: {r['best_params']}")

    if do_xgb:
        r = results['xgb']
        status = "ADOPTED" if r['adopted'] else "NOT ADOPTED"
        print(f"  XGB: {r['baseline_auc']:.6f} -> {r['best_auc']:.6f} "
              f"(delta: {r['delta']:+.6f}) [{status}]")
        if r['adopted']:
            print(f"    New params: {r['best_params']}")

    print(f"\n  Total time: {total_elapsed:.1f} minutes")
    print("=" * 70)


if __name__ == '__main__':
    main()
