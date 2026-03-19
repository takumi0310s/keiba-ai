#!/usr/bin/env python
"""KEIBA AI v10 Precision Optimization Script

5つの分析を順次実行し、Pattern A AUC 0.8106 を超える改善を探る:
1. Feature importance TOP30 + bottom 20% 特定
2. Feature pruning (bottom 20% 除去)
3. Correlation analysis (|corr| > 0.95 除去)
4. Optuna hyperparameter optimization (30 trials)
5. Training period comparison (全期間 / 5年 / 3年)

結果: data/v10_optimization_report.json
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb_lib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features,
    compute_distance_aptitude, compute_frame_advantage,
    compute_running_style_features,
    COURSE_MAP, N_TOP_SIRE,
    FEATURES_V93_PACE,
    train_lgb, train_xgb,
)
from train_v92_leakfree import LEAK_FEATURES_A
from train_v93_pattern_b import (
    LIVE_EXTRA_FEATURES, WEATHER_MAP,
    add_weather_feature, add_pop_rank,
)

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Pattern A: leak-free features (same as train_v10_final.py)
FEATURES_A = [f for f in FEATURES_V93_PACE if f not in LEAK_FEATURES_A]

# Current baseline
BASELINE_AUC = 0.8106  # Pattern A ensemble (avg3) from v10_training_report.json


def separator(title, char='=', width=70):
    """Print a formatted separator."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def prepare_data():
    """Load and prepare data (same pipeline as train_v10_final.py)."""
    separator("DATA LOADING")
    t0 = time.time()

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

    print("  Building features...")
    df = build_features(df)
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)
    df = compute_running_style_features(df)

    # Weather/pop_rank for Pattern B compatibility
    df = add_weather_feature(df)
    df = add_pop_rank(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()

    elapsed = time.time() - t0
    print(f"  Dataset: {len(df)} rows, {df['race_id_str'].nunique()} races ({elapsed:.0f}s)")
    return df, sire_map, bms_map


def ensure_features(df, features):
    """Ensure all features exist and are numeric."""
    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    return df


def get_train_valid_split(df, features):
    """Standard time-based split: train on < max_year-1, validate on last 2 years."""
    max_year = df['year_full'].max()
    valid_mask = df['year_full'] >= (max_year - 1)
    train_mask = ~valid_mask

    X_train = df.loc[train_mask, features].values
    y_train = df.loc[train_mask, 'target'].values
    X_valid = df.loc[valid_mask, features].values
    y_valid = df.loc[valid_mask, 'target'].values

    return X_train, y_train, X_valid, y_valid


def train_ensemble(X_train, y_train, X_valid, y_valid, features, verbose=True):
    """Train LGB+XGB ensemble and return AUC."""
    lgb_model = train_lgb(X_train, y_train, X_valid, y_valid, features)
    lgb_pred = lgb_model.predict(X_valid)
    lgb_auc = roc_auc_score(y_valid, lgb_pred)

    xgb_model = train_xgb(X_train, y_train, X_valid, y_valid)
    xgb_pred = xgb_model.predict(xgb_lib.DMatrix(X_valid))
    xgb_auc = roc_auc_score(y_valid, xgb_pred)

    # Weighted average ensemble
    total = lgb_auc + xgb_auc
    w_lgb = lgb_auc / total
    w_xgb = xgb_auc / total
    ens_pred = lgb_pred * w_lgb + xgb_pred * w_xgb
    ens_auc = roc_auc_score(y_valid, ens_pred)

    if verbose:
        print(f"    LGB={lgb_auc:.4f}  XGB={xgb_auc:.4f}  Ensemble={ens_auc:.4f}")

    return {
        'lgb_model': lgb_model,
        'xgb_model': xgb_model,
        'lgb_auc': lgb_auc,
        'xgb_auc': xgb_auc,
        'ens_auc': ens_auc,
        'lgb_pred': lgb_pred,
        'xgb_pred': xgb_pred,
        'ens_pred': ens_pred,
        'weights': {'lgb': w_lgb, 'xgb': w_xgb},
    }


# ============================================================
#  Analysis 1: Feature Importance TOP30
# ============================================================
def analysis_1_feature_importance(df, features):
    """Train quick LGB and display TOP30 feature importance."""
    separator("ANALYSIS 1: Feature Importance TOP30")
    t0 = time.time()

    df = ensure_features(df, features)
    X_train, y_train, X_valid, y_valid = get_train_valid_split(df, features)

    print(f"  Training LGB on {len(features)} features...")
    print(f"  Train: {len(X_train)}, Valid: {len(X_valid)}")

    lgb_model = train_lgb(X_train, y_train, X_valid, y_valid, features)
    lgb_pred = lgb_model.predict(X_valid)
    lgb_auc = roc_auc_score(y_valid, lgb_pred)
    print(f"  LGB AUC: {lgb_auc:.4f}")

    # Feature importance (gain)
    importance = lgb_model.feature_importance(importance_type='gain')
    fi = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)

    print(f"\n  {'Rank':<5} {'Feature':<35} {'Importance':>12}")
    print(f"  {'-'*55}")
    for i, (fname, imp) in enumerate(fi[:30], 1):
        print(f"  {i:<5} {fname:<35} {imp:>12,.1f}")

    # Bottom 20%
    n_bottom = max(1, len(fi) // 5)
    bottom_features = [f for f, _ in fi[-n_bottom:]]
    bottom_importance = {f: round(imp, 1) for f, imp in fi[-n_bottom:]}

    print(f"\n  Bottom 20% ({n_bottom} features) to consider removing:")
    for fname, imp in fi[-n_bottom:]:
        print(f"    {fname:<35} {imp:>12,.1f}")

    elapsed = time.time() - t0
    print(f"\n  Analysis 1 complete ({elapsed:.0f}s)")

    return {
        'lgb_auc': lgb_auc,
        'top30': {f: round(imp, 1) for f, imp in fi[:30]},
        'bottom_20pct': bottom_importance,
        'bottom_features': bottom_features,
        'all_importance': {f: round(imp, 1) for f, imp in fi},
    }


# ============================================================
#  Analysis 2: Feature Pruning
# ============================================================
def analysis_2_feature_pruning(df, features, bottom_features, baseline_lgb_auc):
    """Remove bottom 20% features and retrain ensemble."""
    separator("ANALYSIS 2: Feature Pruning (remove bottom 20%)")
    t0 = time.time()

    pruned_features = [f for f in features if f not in bottom_features]
    print(f"  Original features: {len(features)}")
    print(f"  Removed: {len(bottom_features)}")
    print(f"  Remaining: {len(pruned_features)}")
    print(f"  Removed features: {bottom_features}")

    df = ensure_features(df, pruned_features)
    X_train, y_train, X_valid, y_valid = get_train_valid_split(df, pruned_features)

    print(f"\n  Training LGB+XGB ensemble on pruned features...")
    result = train_ensemble(X_train, y_train, X_valid, y_valid, pruned_features)

    # Also train full feature set ensemble for fair comparison
    print(f"\n  Training LGB+XGB ensemble on full features (baseline)...")
    df = ensure_features(df, features)
    X_train_full, _, X_valid_full, _ = get_train_valid_split(df, features)
    result_full = train_ensemble(X_train_full, y_train, X_valid_full, y_valid, features)

    delta = result['ens_auc'] - result_full['ens_auc']
    print(f"\n  Full features ensemble:   {result_full['ens_auc']:.4f}")
    print(f"  Pruned features ensemble: {result['ens_auc']:.4f}")
    print(f"  Delta: {delta:+.4f}")
    print(f"  {'>>> IMPROVED <<<' if delta > 0 else 'No improvement'}")

    elapsed = time.time() - t0
    print(f"\n  Analysis 2 complete ({elapsed:.0f}s)")

    return {
        'full_ens_auc': result_full['ens_auc'],
        'pruned_ens_auc': result['ens_auc'],
        'delta': delta,
        'n_original': len(features),
        'n_pruned': len(pruned_features),
        'removed_features': bottom_features,
        'pruned_features': pruned_features,
        'improved': delta > 0,
        'pruned_lgb_auc': result['lgb_auc'],
        'pruned_xgb_auc': result['xgb_auc'],
    }


# ============================================================
#  Analysis 3: Correlation Analysis
# ============================================================
def analysis_3_correlation(df, features):
    """Find highly correlated features and remove redundant ones."""
    separator("ANALYSIS 3: Correlation Analysis (|corr| > 0.95)")
    t0 = time.time()

    df = ensure_features(df, features)

    print(f"  Computing correlation matrix for {len(features)} features...")
    corr_matrix = df[features].corr()

    # Find pairs with |corr| > 0.95
    high_corr_pairs = []
    seen = set()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.95:
                pair = (features[i], features[j], round(corr_val, 4))
                high_corr_pairs.append(pair)
                seen.add(features[j])  # Mark the second feature for removal

    print(f"\n  Highly correlated pairs (|corr| > 0.95): {len(high_corr_pairs)}")
    for f1, f2, corr in high_corr_pairs:
        print(f"    {f1:<30} <-> {f2:<30}  r={corr:.4f}")

    if not high_corr_pairs:
        print("  No highly correlated pairs found. Skipping retrain.")
        elapsed = time.time() - t0
        print(f"\n  Analysis 3 complete ({elapsed:.0f}s)")
        return {
            'high_corr_pairs': [],
            'removed_features': [],
            'full_ens_auc': None,
            'decorr_ens_auc': None,
            'delta': 0,
            'improved': False,
        }

    # Remove one from each pair (keep the one that appears first / has higher importance)
    features_to_remove = list(seen)
    decorr_features = [f for f in features if f not in features_to_remove]

    print(f"\n  Features to remove: {features_to_remove}")
    print(f"  Remaining: {len(decorr_features)}")

    # Train ensemble with decorrelated features
    df = ensure_features(df, decorr_features)
    X_train, y_train, X_valid, y_valid = get_train_valid_split(df, decorr_features)

    print(f"\n  Training LGB+XGB on decorrelated features...")
    result_decorr = train_ensemble(X_train, y_train, X_valid, y_valid, decorr_features)

    # Full baseline
    df = ensure_features(df, features)
    X_train_full, _, X_valid_full, _ = get_train_valid_split(df, features)
    result_full = train_ensemble(X_train_full, y_train, X_valid_full, y_valid, features)

    delta = result_decorr['ens_auc'] - result_full['ens_auc']
    print(f"\n  Full features ensemble:        {result_full['ens_auc']:.4f}")
    print(f"  Decorrelated features ensemble: {result_decorr['ens_auc']:.4f}")
    print(f"  Delta: {delta:+.4f}")
    print(f"  {'>>> IMPROVED <<<' if delta > 0 else 'No improvement'}")

    elapsed = time.time() - t0
    print(f"\n  Analysis 3 complete ({elapsed:.0f}s)")

    return {
        'high_corr_pairs': [(f1, f2, c) for f1, f2, c in high_corr_pairs],
        'removed_features': features_to_remove,
        'n_original': len(features),
        'n_decorr': len(decorr_features),
        'full_ens_auc': result_full['ens_auc'],
        'decorr_ens_auc': result_decorr['ens_auc'],
        'delta': delta,
        'improved': delta > 0,
        'decorr_features': decorr_features,
    }


# ============================================================
#  Analysis 4: Optuna Hyperparameter Optimization
# ============================================================
def analysis_4_optuna(df, features, n_trials=30):
    """Optimize LGB hyperparameters with Optuna."""
    separator(f"ANALYSIS 4: Optuna Hyperparameter Optimization ({n_trials} trials)")
    t0 = time.time()

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  ERROR: optuna not installed. pip install optuna")
        return {'error': 'optuna not installed', 'improved': False}

    df = ensure_features(df, features)
    X_train, y_train, X_valid, y_valid = get_train_valid_split(df, features)

    # First get default AUC
    print(f"  Training default LGB for baseline...")
    lgb_default = train_lgb(X_train, y_train, X_valid, y_valid, features)
    default_pred = lgb_default.predict(X_valid)
    default_auc = roc_auc_score(y_valid, default_pred)
    print(f"  Default LGB AUC: {default_auc:.4f}")

    # Optuna objective
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': 5,
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42,
        }

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=features)
        dvalid = lgb.Dataset(X_valid, label=y_valid, feature_name=features, reference=dtrain)

        model = lgb.train(
            params, dtrain, num_boost_round=1000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        pred = model.predict(X_valid)
        auc = roc_auc_score(y_valid, pred)
        return auc

    print(f"\n  Running {n_trials} Optuna trials...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_auc = study.best_value
    delta = best_auc - default_auc

    print(f"\n  Default LGB AUC: {default_auc:.4f}")
    print(f"  Best Optuna AUC: {best_auc:.4f}")
    print(f"  Delta: {delta:+.4f}")
    print(f"\n  Best parameters:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")

    # Train ensemble with best params
    print(f"\n  Training LGB+XGB ensemble with Optuna-optimized LGB...")
    best_lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'bagging_freq': 5, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
    }
    best_lgb_params.update(best_params)

    lgb_opt = train_lgb(X_train, y_train, X_valid, y_valid, features,
                        params_override=best_lgb_params)
    lgb_opt_pred = lgb_opt.predict(X_valid)
    lgb_opt_auc = roc_auc_score(y_valid, lgb_opt_pred)

    xgb_model = train_xgb(X_train, y_train, X_valid, y_valid)
    xgb_pred = xgb_model.predict(xgb_lib.DMatrix(X_valid))
    xgb_auc = roc_auc_score(y_valid, xgb_pred)

    total = lgb_opt_auc + xgb_auc
    ens_pred = lgb_opt_pred * (lgb_opt_auc / total) + xgb_pred * (xgb_auc / total)
    ens_auc = roc_auc_score(y_valid, ens_pred)

    print(f"    Optimized LGB={lgb_opt_auc:.4f}  XGB={xgb_auc:.4f}  Ensemble={ens_auc:.4f}")

    improved = best_auc > default_auc
    print(f"\n  {'>>> IMPROVED <<<' if improved else 'No significant improvement'}")

    elapsed = time.time() - t0
    print(f"\n  Analysis 4 complete ({elapsed:.0f}s)")

    return {
        'default_lgb_auc': default_auc,
        'best_optuna_lgb_auc': best_auc,
        'lgb_delta': delta,
        'best_params': {k: round(v, 6) if isinstance(v, float) else v for k, v in best_params.items()},
        'n_trials': n_trials,
        'optuna_ens_auc': ens_auc,
        'optuna_lgb_auc': lgb_opt_auc,
        'optuna_xgb_auc': xgb_auc,
        'improved': improved,
        'optuna_lgb_model': lgb_opt,
        'optuna_xgb_model': xgb_model,
    }


# ============================================================
#  Analysis 5: Training Period Comparison
# ============================================================
def analysis_5_training_period(df, features):
    """Compare AUC for different training periods."""
    separator("ANALYSIS 5: Training Period Comparison")
    t0 = time.time()

    df = ensure_features(df, features)

    max_year = df['year_full'].max()
    valid_mask = df['year_full'] >= (max_year - 1)

    X_valid = df.loc[valid_mask, features].values
    y_valid = df.loc[valid_mask, 'target'].values

    periods = {
        'full (2010+)': df['year_full'] >= 2010,
        'recent_5y (2018+)': df['year_full'] >= 2018,
        'recent_3y (2020+)': df['year_full'] >= 2020,
    }

    results = {}
    for period_name, period_mask in periods.items():
        train_mask = period_mask & ~valid_mask
        n_train = train_mask.sum()

        if n_train < 1000:
            print(f"  {period_name}: Skipped (only {n_train} training samples)")
            continue

        X_train = df.loc[train_mask, features].values
        y_train = df.loc[train_mask, 'target'].values

        print(f"\n  {period_name}: train={n_train}, valid={len(X_valid)}")

        result = train_ensemble(X_train, y_train, X_valid, y_valid, features)
        results[period_name] = {
            'n_train': int(n_train),
            'lgb_auc': result['lgb_auc'],
            'xgb_auc': result['xgb_auc'],
            'ens_auc': result['ens_auc'],
        }

    # Summary
    print(f"\n  {'Period':<25} {'N_train':>8} {'LGB':>8} {'XGB':>8} {'Ensemble':>10}")
    print(f"  {'-'*62}")
    best_period = None
    best_auc = 0
    for period_name, r in results.items():
        marker = ""
        if r['ens_auc'] > best_auc:
            best_auc = r['ens_auc']
            best_period = period_name
        print(f"  {period_name:<25} {r['n_train']:>8} {r['lgb_auc']:>8.4f} {r['xgb_auc']:>8.4f} {r['ens_auc']:>10.4f}")

    if best_period:
        print(f"\n  Best period: {best_period} (AUC={best_auc:.4f})")

    elapsed = time.time() - t0
    print(f"\n  Analysis 5 complete ({elapsed:.0f}s)")

    return {
        'periods': results,
        'best_period': best_period,
        'best_auc': best_auc,
    }


# ============================================================
#  Main
# ============================================================
def main():
    start_time = time.time()

    print("=" * 70)
    print("  KEIBA AI v10 PRECISION OPTIMIZATION")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Baseline Pattern A ensemble AUC: {BASELINE_AUC}")
    print(f"  Pattern A features: {len(FEATURES_A)}")
    print("=" * 70)

    # === Load Data ===
    df, sire_map, bms_map = prepare_data()

    results = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'baseline_auc': BASELINE_AUC,
        'n_features': len(FEATURES_A),
        'features': FEATURES_A,
    }

    # === Analysis 1: Feature Importance ===
    a1 = analysis_1_feature_importance(df, FEATURES_A)
    results['analysis_1_feature_importance'] = {
        k: v for k, v in a1.items() if k != 'bottom_features'
    }

    # === Analysis 2: Feature Pruning ===
    a2 = analysis_2_feature_pruning(df, FEATURES_A, a1['bottom_features'], a1['lgb_auc'])
    results['analysis_2_feature_pruning'] = {
        k: v for k, v in a2.items()
        if k not in ('pruned_features',)
    }

    # === Analysis 3: Correlation Analysis ===
    a3 = analysis_3_correlation(df, FEATURES_A)
    results['analysis_3_correlation'] = {
        k: v for k, v in a3.items()
        if k not in ('decorr_features',)
    }

    # === Analysis 4: Optuna ===
    a4 = analysis_4_optuna(df, FEATURES_A, n_trials=30)
    results['analysis_4_optuna'] = {
        k: v for k, v in a4.items()
        if k not in ('optuna_lgb_model', 'optuna_xgb_model')
    }

    # === Analysis 5: Training Period ===
    a5 = analysis_5_training_period(df, FEATURES_A)
    results['analysis_5_training_period'] = a5

    # ============================================================
    #  Final Summary & Decision
    # ============================================================
    separator("FINAL SUMMARY", char='#')

    # Collect all AUC candidates
    candidates = {
        'baseline (current)': BASELINE_AUC,
    }
    if a2.get('pruned_ens_auc'):
        candidates['pruned features'] = a2['pruned_ens_auc']
    if a3.get('decorr_ens_auc'):
        candidates['decorrelated features'] = a3['decorr_ens_auc']
    if a4.get('optuna_ens_auc'):
        candidates['optuna optimized'] = a4['optuna_ens_auc']
    if a5.get('best_auc'):
        candidates[f"best period ({a5['best_period']})"] = a5['best_auc']

    print(f"\n  {'Method':<35} {'AUC':>10} {'vs Baseline':>12}")
    print(f"  {'-'*60}")
    for method, auc in sorted(candidates.items(), key=lambda x: -x[1]):
        delta = auc - BASELINE_AUC
        marker = " <<<" if auc == max(candidates.values()) and auc > BASELINE_AUC else ""
        print(f"  {method:<35} {auc:>10.4f} {delta:>+12.4f}{marker}")

    best_method = max(candidates, key=candidates.get)
    best_auc = candidates[best_method]
    improved = best_auc > BASELINE_AUC

    print(f"\n  Best method: {best_method} (AUC={best_auc:.4f})")
    print(f"  Improvement over baseline: {best_auc - BASELINE_AUC:+.4f}")
    print(f"  Decision: {'>>> UPDATE PRODUCTION <<<' if improved else 'KEEP CURRENT (no improvement)'}")

    results['final_summary'] = {
        'candidates': {k: round(v, 6) for k, v in candidates.items()},
        'best_method': best_method,
        'best_auc': best_auc,
        'improvement': best_auc - BASELINE_AUC,
        'improved': improved,
        'production_updated': False,
    }

    # === Save production model if improved ===
    if improved:
        print(f"\n  Saving improved model...")

        # Determine which model to save
        if best_method == 'optuna optimized' and 'optuna_lgb_model' in a4:
            save_lgb = a4['optuna_lgb_model']
            save_xgb = a4['optuna_xgb_model']
            save_features = FEATURES_A
            save_params = a4.get('best_params', {})
        elif best_method == 'pruned features' and a2.get('pruned_features'):
            # Retrain with pruned features for saving
            pruned_feats = a2['pruned_features']
            df = ensure_features(df, pruned_feats)
            X_train, y_train, X_valid, y_valid = get_train_valid_split(df, pruned_feats)
            save_lgb = train_lgb(X_train, y_train, X_valid, y_valid, pruned_feats)
            save_xgb = train_xgb(X_train, y_train, X_valid, y_valid)
            save_features = pruned_feats
            save_params = {}
        elif best_method.startswith('decorrelated') and a3.get('decorr_features'):
            decorr_feats = a3['decorr_features']
            df = ensure_features(df, decorr_feats)
            X_train, y_train, X_valid, y_valid = get_train_valid_split(df, decorr_feats)
            save_lgb = train_lgb(X_train, y_train, X_valid, y_valid, decorr_feats)
            save_xgb = train_xgb(X_train, y_train, X_valid, y_valid)
            save_features = decorr_feats
            save_params = {}
        else:
            # Best period or other - retrain with full features
            df = ensure_features(df, FEATURES_A)
            X_train, y_train, X_valid, y_valid = get_train_valid_split(df, FEATURES_A)
            save_lgb = train_lgb(X_train, y_train, X_valid, y_valid, FEATURES_A)
            save_xgb = train_xgb(X_train, y_train, X_valid, y_valid)
            save_features = FEATURES_A
            save_params = {}

        save_features_pkl = [f if f != 'num_horses_val' else 'num_horses' for f in save_features]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        pkl = {
            'model': save_lgb,
            'xgb_model': save_xgb,
            'features': save_features_pkl,
            'version': 'v10_optimized',
            'auc': best_auc,
            'ensemble_auc': best_auc,
            'leak_free': True,
            'leak_pattern': 'A',
            'leak_removed': sorted(LEAK_FEATURES_A),
            'sire_map': sire_map,
            'bms_map': bms_map,
            'n_top_encode': N_TOP_SIRE,
            'trained_at': now,
            'model_type': 'central',
            'mlp_model': None,
            'mlp_scaler': None,
            'ensemble_weights': {'lgb': 0.5, 'xgb': 0.5, 'mlp': 0},
            'course_map': dict(COURSE_MAP),
            'optimization_method': best_method,
            'optimization_params': save_params,
        }

        central_path = os.path.join(BASE_DIR, 'keiba_model_v9_central.pkl')
        with open(central_path, 'wb') as f:
            pickle.dump(pkl, f)
        print(f"  Saved: {central_path}")

        v8_path = os.path.join(BASE_DIR, 'keiba_model_v8.pkl')
        with open(v8_path, 'wb') as f:
            pickle.dump(pkl, f)
        print(f"  Saved backup: {v8_path}")

        results['final_summary']['production_updated'] = True

    # === Save report ===
    report_path = os.path.join(DATA_DIR, 'v10_optimization_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  V10 OPTIMIZATION COMPLETE ({elapsed:.0f}s / {elapsed/60:.1f}min)")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    main()
