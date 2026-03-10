#!/usr/bin/env python
"""Optuna Hyperparameter Tuning for KEIBA AI LightGBM (Pattern A, Leak-Free)
Walk-forward evaluation: train 2010~(Y-1), test Y, for Y=2020-2025.
Search space: +/-20% of current baseline params.

Optimized: precomputes all fold data (sire encoding, splits) once.
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
PROJECT_DIR = os.path.join(BASE_DIR, '..')

from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features, N_TOP_SIRE,
)
from train_v92_leakfree import FEATURES_PATTERN_A

TEST_YEARS = list(range(2020, 2026))

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
BASELINE_NUM_BOOST_ROUND = 500


def encode_sires_fold(df, train_mask, features, n_top=N_TOP_SIRE):
    """Encode sires using only training data, return feature arrays."""
    train_df = df[train_mask]
    sire_counts = train_df['father'].value_counts()
    top_sires = sire_counts.head(n_top).index.tolist()
    sire_map = {s: i for i, s in enumerate(top_sires)}
    sire_enc = df['father'].map(sire_map).fillna(n_top).astype(int).values

    bms_counts = train_df['bms'].value_counts()
    top_bms = bms_counts.head(n_top).index.tolist()
    bms_map = {s: i for i, s in enumerate(top_bms)}
    bms_enc = df['bms'].map(bms_map).fillna(n_top).astype(int).values

    return sire_enc, bms_enc


def prepare_data():
    """Load and prepare data once."""
    print("Loading and preparing data...")
    t0 = time.time()

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

    elapsed = time.time() - t0
    print(f"Data ready: {len(df)} rows ({elapsed:.0f}s)")
    return df, features


def precompute_folds(df, features):
    """Precompute all fold data: sire encodings, X/y arrays for each year.
    Returns a list of dicts with precomputed numpy arrays."""
    print("Precomputing fold data...")
    t0 = time.time()

    sire_idx = features.index('sire_enc')
    bms_idx = features.index('bms_enc')

    folds = []
    for test_year in TEST_YEARS:
        train_mask = (df['year_full'] >= 2010) & (df['year_full'] < test_year)
        test_mask = df['year_full'] == test_year

        n_test = test_mask.sum()
        if n_test < 100:
            continue

        # Per-fold sire encoding
        sire_enc, bms_enc = encode_sires_fold(df, train_mask, features)

        # Build feature matrix with fold-specific sire encoding
        X_all = df[features].values.copy()
        X_all[:, sire_idx] = sire_enc
        X_all[:, bms_idx] = bms_enc

        # Train/val split within training data (85/15 by date)
        train_df = df[train_mask]
        dates = train_df['date_num']
        valid_cutoff = dates.quantile(0.85)
        tr_idx = train_mask & (df['date_num'] < valid_cutoff)
        va_idx = train_mask & (df['date_num'] >= valid_cutoff)

        y_all = df['target'].values

        fold = {
            'year': test_year,
            'X_tr': X_all[tr_idx.values].astype(np.float32),
            'y_tr': y_all[tr_idx.values],
            'X_va': X_all[va_idx.values].astype(np.float32),
            'y_va': y_all[va_idx.values],
            'X_test': X_all[test_mask.values].astype(np.float32),
            'y_test': y_all[test_mask.values],
        }
        folds.append(fold)

    elapsed = time.time() - t0
    print(f"  {len(folds)} folds precomputed ({elapsed:.0f}s)")
    return folds


def walk_forward_auc(folds, features, lgb_params, num_boost_round):
    """Run walk-forward evaluation using precomputed fold data."""
    year_aucs = {}
    for fold in folds:
        full_params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'verbose': -1, 'n_jobs': -1, 'seed': 42,
        }
        full_params.update(lgb_params)

        dtrain = lgb.Dataset(fold['X_tr'], label=fold['y_tr'], feature_name=features, free_raw_data=False)
        dvalid = lgb.Dataset(fold['X_va'], label=fold['y_va'], feature_name=features, reference=dtrain, free_raw_data=False)

        model = lgb.train(
            full_params, dtrain, num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        preds = model.predict(fold['X_test'])
        test_auc = roc_auc_score(fold['y_test'], preds)
        year_aucs[fold['year']] = test_auc

    avg_auc = np.mean(list(year_aucs.values())) if year_aucs else 0.0
    return avg_auc, year_aucs


def objective(trial, folds, features):
    """Optuna objective: walk-forward AUC."""
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 50, 76),
        'learning_rate': trial.suggest_float('learning_rate', 0.04, 0.06),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.64, 0.96),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.64, 0.96),
        'bagging_freq': trial.suggest_int('bagging_freq', 4, 6),
        'min_child_samples': trial.suggest_int('min_child_samples', 40, 60),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.08, 0.12),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.08, 0.12),
    }
    num_boost_round = trial.suggest_int('num_boost_round', 300, 700)

    avg_auc, _ = walk_forward_auc(folds, features, params, num_boost_round)
    return avg_auc


def main():
    print("=" * 70)
    print("  OPTUNA HYPERPARAMETER TUNING (Pattern A, LGB Only)")
    print(f"  Test years: {TEST_YEARS}")
    print(f"  Baseline params: {BASELINE_PARAMS}")
    print(f"  Trials: 100")
    print("=" * 70)

    # Prepare data once
    df, features = prepare_data()
    print(f"Features ({len(features)}): {features[:5]}...")

    # Precompute all fold data
    folds = precompute_folds(df, features)

    # Free the large dataframe
    del df
    import gc
    gc.collect()

    # Run baseline
    print("\n--- Baseline Evaluation ---")
    t0 = time.time()
    baseline_auc, baseline_year_aucs = walk_forward_auc(
        folds, features, BASELINE_PARAMS, BASELINE_NUM_BOOST_ROUND
    )
    elapsed = time.time() - t0
    print(f"Baseline WF AUC: {baseline_auc:.4f} ({elapsed:.0f}s)")
    print(f"Year AUCs: {', '.join(f'{y}={a:.4f}' for y, a in baseline_year_aucs.items())}")

    # Optuna study
    print("\n--- Starting Optuna Optimization (100 trials) ---")
    trial_count = [0]
    study_start = time.time()

    def callback(study, trial):
        trial_count[0] += 1
        if trial_count[0] % 10 == 0:
            elapsed = time.time() - study_start
            best = study.best_value
            print(f"  Trial {trial_count[0]}/100 | Best WF AUC: {best:.4f} | Elapsed: {elapsed:.0f}s")

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
        'num_boost_round': 500,
    })

    study.optimize(
        lambda trial: objective(trial, folds, features),
        n_trials=100,
        callbacks=[callback],
    )

    total_elapsed = time.time() - study_start

    # Results
    best_trial = study.best_trial
    best_params = best_trial.params
    best_wf_auc = best_trial.value

    # Extract best params
    best_lgb_params = {k: v for k, v in best_params.items() if k != 'num_boost_round'}
    best_num_boost_round = best_params['num_boost_round']

    # Re-run best to get per-year AUCs
    print("\n--- Re-evaluating Best Params ---")
    best_auc_check, best_year_aucs = walk_forward_auc(
        folds, features, best_lgb_params, best_num_boost_round
    )

    improvement = best_wf_auc - baseline_auc
    adopted = best_wf_auc > baseline_auc

    print(f"\n{'=' * 70}")
    print(f"  OPTUNA RESULTS")
    print(f"{'=' * 70}")
    print(f"  Baseline WF AUC:  {baseline_auc:.4f}")
    print(f"  Best WF AUC:      {best_wf_auc:.4f}")
    print(f"  Improvement:      {improvement:+.4f}")
    print(f"  Adopted:          {adopted}")
    print(f"  Total time:       {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"\n  Baseline year AUCs: {baseline_year_aucs}")
    print(f"  Best year AUCs:     {best_year_aucs}")
    print(f"\n  Best params:")
    for k, v in best_params.items():
        baseline_val = BASELINE_PARAMS.get(k, BASELINE_NUM_BOOST_ROUND if k == 'num_boost_round' else '?')
        print(f"    {k:25s}: {v} (baseline: {baseline_val})")

    # Save results
    results = {
        'baseline_params': {**BASELINE_PARAMS, 'num_boost_round': BASELINE_NUM_BOOST_ROUND},
        'baseline_wf_auc': round(baseline_auc, 6),
        'best_params': {k: round(v, 6) if isinstance(v, float) else v for k, v in best_params.items()},
        'best_wf_auc': round(best_wf_auc, 6),
        'improvement': round(improvement, 6),
        'year_aucs_baseline': {str(k): round(v, 6) for k, v in baseline_year_aucs.items()},
        'year_aucs_best': {str(k): round(v, 6) for k, v in best_year_aucs.items()},
        'n_trials': 100,
        'adopted': adopted,
        'elapsed_seconds': round(total_elapsed, 1),
    }

    out_path = os.path.join(PROJECT_DIR, 'data', 'optuna_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")
    print("  Done!")

    return results


if __name__ == '__main__':
    main()
