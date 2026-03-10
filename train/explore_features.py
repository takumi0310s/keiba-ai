#!/usr/bin/env python
"""Feature Exploration for KEIBA AI v9.3+ (Central)
Tests 10 candidate derived features via walk-forward AUC evaluation.
All features must be leak-free (available BEFORE race day).

Walk-forward: train 2010~(Y-1), test Y, for Y in 2020-2025.
Baseline: Pattern A features (V9.3, leak-free).
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
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features,
    compute_distance_aptitude, compute_frame_advantage,
    COURSE_MAP, N_TOP_SIRE,
    FEATURES_V93,
)
from train_v92_leakfree import LEAK_FEATURES_A

# Build Pattern A feature list from V9.3
FEATURES_V93_PATTERN_A = [f for f in FEATURES_V93 if f not in LEAK_FEATURES_A]

TEST_YEARS = list(range(2020, 2026))

LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
}


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


def train_lgb_fold(X_tr, y_tr, X_va, y_va, features):
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=features)
    dvalid = lgb.Dataset(X_va, label=y_va, feature_name=features, reference=dtrain)
    model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=500,
                       valid_sets=[dvalid],
                       callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    return model


def walk_forward_auc(df, features, label=""):
    """Run walk-forward evaluation for given feature set."""
    fold_aucs = {}
    for test_year in TEST_YEARS:
        train_mask = (df['year_full'] >= 2010) & (df['year_full'] < test_year)
        test_mask = df['year_full'] == test_year
        n_test = test_mask.sum()
        if n_test < 100:
            continue

        # Re-encode sires per fold
        df_fold = df.copy()
        df_fold = encode_sires_fold(df_fold, train_mask)

        # Rebuild sire cross features after re-encoding
        df_fold['sire_dist'] = df_fold['sire_enc'] * 10 + df_fold['dist_cat']
        df_fold['sire_surface'] = df_fold['sire_enc'] * 10 + df_fold['surface_enc']
        df_fold['bms_dist'] = df_fold['bms_enc'] * 10 + df_fold['dist_cat']

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

        test_df = df_fold[test_mask]
        X_test = test_df[features].values
        y_test = test_df['target'].values
        preds = model.predict(X_test)
        test_auc = roc_auc_score(y_test, preds)
        fold_aucs[test_year] = test_auc

    avg_auc = np.mean(list(fold_aucs.values())) if fold_aucs else 0
    return avg_auc, fold_aucs


# ===== CANDIDATE FEATURE COMPUTATION =====

def compute_avg_finish_5r(df):
    """1. Average finish position over last 5 races (expanding window, pre-race)."""
    print("  Computing avg_finish_5r...")
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    grp = df.groupby('horse_id')
    # Rolling mean of last 5 finishes (shift to exclude current)
    df['prev4_finish'] = grp['finish'].shift(4).fillna(5)
    df['prev5_finish'] = grp['finish'].shift(5).fillna(5)
    finish_cols = ['prev_finish', 'prev2_finish', 'prev3_finish', 'prev4_finish', 'prev5_finish']
    df['avg_finish_5r'] = df[finish_cols].mean(axis=1)
    df = df.drop(columns=['prev4_finish', 'prev5_finish'], errors='ignore')
    return df


def compute_jockey_dist_wr(df):
    """2. Jockey win rate at specific distance category (expanding window)."""
    print("  Computing jockey_dist_wr...")
    df = df.sort_values('date_num').reset_index(drop=True)
    global_wr = df['is_win'].mean()
    alpha = 20

    df['jd_cum_races'] = df.groupby(['jockey_id', 'dist_cat']).cumcount()
    df['jd_cum_wins'] = df.groupby(['jockey_id', 'dist_cat'])['is_win'].cumsum() - df['is_win']
    df['jockey_dist_wr'] = (
        (df['jd_cum_wins'] + alpha * global_wr) /
        (df['jd_cum_races'] + alpha)
    )
    df = df.drop(columns=['jd_cum_races', 'jd_cum_wins'])
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    return df


def compute_days_since_last(df):
    """3. Days since last race (already exists as rest_days, but we add log transform)."""
    print("  Computing days_since_last_race...")
    # rest_days already exists, add log version
    df['days_since_last_race'] = np.log1p(df['rest_days'].clip(1, 365))
    return df


def compute_horse_course_top3r(df):
    """4. Horse's top3 rate at specific course (expanding window)."""
    print("  Computing horse_course_top3r...")
    df = df.sort_values('date_num').reset_index(drop=True)
    global_t3 = df['is_top3'].mean()
    alpha = 5

    df['hc_cum_races'] = df.groupby(['horse_id', 'course_enc']).cumcount()
    df['hc_cum_top3'] = df.groupby(['horse_id', 'course_enc'])['is_top3'].cumsum() - df['is_top3']
    df['horse_course_top3r'] = (
        (df['hc_cum_top3'] + alpha * global_t3) /
        (df['hc_cum_races'] + alpha)
    )
    df = df.drop(columns=['hc_cum_races', 'hc_cum_top3'])
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    return df


def compute_prev_prize_rank(df):
    """5. Previous race prize money rank within that race."""
    print("  Computing prev_prize_rank...")
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    # Rank within race by prize (higher prize = better rank)
    df['prize_rank_in_race'] = df.groupby('race_id_str')['prize'].rank(ascending=False, method='min')
    df['prize_rank_ratio'] = df['prize_rank_in_race'] / df['num_horses_val'].clip(1)
    grp = df.groupby('horse_id')
    df['prev_prize_rank'] = grp['prize_rank_ratio'].shift(1).fillna(0.5)
    df = df.drop(columns=['prize_rank_in_race', 'prize_rank_ratio'], errors='ignore')
    return df


def compute_class_change(df):
    """6. Class change from previous race (upgrade/downgrade/same)."""
    print("  Computing class_change...")
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    df['class_code_num'] = pd.to_numeric(df['class_code'], errors='coerce').fillna(1)
    grp = df.groupby('horse_id')
    df['prev_class'] = grp['class_code_num'].shift(1).fillna(df['class_code_num'])
    df['class_change'] = df['class_code_num'] - df['prev_class']
    df = df.drop(columns=['prev_class'], errors='ignore')
    return df


def compute_horse_season_wr(df):
    """7. Horse's historical win rate in the current season (expanding window)."""
    print("  Computing horse_season_wr...")
    df = df.sort_values('date_num').reset_index(drop=True)
    global_t3 = df['is_top3'].mean()
    alpha = 3

    df['hsea_cum_races'] = df.groupby(['horse_id', 'season']).cumcount()
    df['hsea_cum_top3'] = df.groupby(['horse_id', 'season'])['is_top3'].cumsum() - df['is_top3']
    df['horse_season_wr'] = (
        (df['hsea_cum_top3'] + alpha * global_t3) /
        (df['hsea_cum_races'] + alpha)
    )
    df = df.drop(columns=['hsea_cum_races', 'hsea_cum_top3'])
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    return df


def compute_trainer_jockey_combo_wr(df):
    """8. Trainer x Jockey combination win rate (expanding window)."""
    print("  Computing trainer_jockey_combo_wr...")
    df = df.sort_values('date_num').reset_index(drop=True)
    global_wr = df['is_win'].mean()
    alpha = 10

    df['tj_cum_races'] = df.groupby(['trainer_id', 'jockey_id']).cumcount()
    df['tj_cum_wins'] = df.groupby(['trainer_id', 'jockey_id'])['is_win'].cumsum() - df['is_win']
    df['trainer_jockey_combo_wr'] = (
        (df['tj_cum_wins'] + alpha * global_wr) /
        (df['tj_cum_races'] + alpha)
    )
    df = df.drop(columns=['tj_cum_races', 'tj_cum_wins'])
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    return df


def compute_horse_avg_prize_3r(df):
    """9. Average prize money in last 3 races."""
    print("  Computing horse_avg_prize_3r...")
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    grp = df.groupby('horse_id')
    df['prev_prize_1'] = grp['prize'].shift(1).fillna(0)
    df['prev_prize_2'] = grp['prize'].shift(2).fillna(0)
    df['prev_prize_3'] = grp['prize'].shift(3).fillna(0)
    df['horse_avg_prize_3r'] = df[['prev_prize_1', 'prev_prize_2', 'prev_prize_3']].mean(axis=1)
    # Log transform for better distribution
    df['horse_avg_prize_3r'] = np.log1p(df['horse_avg_prize_3r'])
    df = df.drop(columns=['prev_prize_1', 'prev_prize_2', 'prev_prize_3'], errors='ignore')
    return df


def compute_sire_age_wr(df):
    """10. Sire progeny win rate at horse's age group (expanding window)."""
    print("  Computing sire_age_wr...")
    df = df.sort_values('date_num').reset_index(drop=True)
    global_wr = df['is_win'].mean()
    alpha = 30

    df['sa_cum_races'] = df.groupby(['father', 'age_group']).cumcount()
    df['sa_cum_wins'] = df.groupby(['father', 'age_group'])['is_win'].cumsum() - df['is_win']
    df['sire_age_wr'] = (
        (df['sa_cum_wins'] + alpha * global_wr) /
        (df['sa_cum_races'] + alpha)
    )
    df = df.drop(columns=['sa_cum_races', 'sa_cum_wins'])
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    return df


# Map of candidate features
CANDIDATES = {
    'avg_finish_5r': {
        'compute': compute_avg_finish_5r,
        'columns': ['avg_finish_5r'],
    },
    'jockey_dist_wr': {
        'compute': compute_jockey_dist_wr,
        'columns': ['jockey_dist_wr'],
    },
    'days_since_last_race': {
        'compute': compute_days_since_last,
        'columns': ['days_since_last_race'],
    },
    'horse_course_top3r': {
        'compute': compute_horse_course_top3r,
        'columns': ['horse_course_top3r'],
    },
    'prev_prize_rank': {
        'compute': compute_prev_prize_rank,
        'columns': ['prev_prize_rank'],
    },
    'class_change': {
        'compute': compute_class_change,
        'columns': ['class_change'],
    },
    'horse_season_wr': {
        'compute': compute_horse_season_wr,
        'columns': ['horse_season_wr'],
    },
    'trainer_jockey_combo_wr': {
        'compute': compute_trainer_jockey_combo_wr,
        'columns': ['trainer_jockey_combo_wr'],
    },
    'horse_avg_prize_3r': {
        'compute': compute_horse_avg_prize_3r,
        'columns': ['horse_avg_prize_3r'],
    },
    'sire_age_wr': {
        'compute': compute_sire_age_wr,
        'columns': ['sire_age_wr'],
    },
}


def main():
    t_start = time.time()
    print("=" * 70)
    print("  FEATURE EXPLORATION: Walk-Forward AUC Evaluation")
    print(f"  Baseline features: {len(FEATURES_V93_PATTERN_A)} (V9.3 Pattern A)")
    print(f"  Candidates: {len(CANDIDATES)}")
    print(f"  Test years: {TEST_YEARS}")
    print("=" * 70)

    # === Load and prepare data (same pipeline as production) ===
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

    # V9.3 additional features
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()

    # Ensure baseline features are numeric
    features_base = list(FEATURES_V93_PATTERN_A)
    for f in features_base:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"\nData ready: {len(df)} rows, {df['race_id_str'].nunique()} races")
    print(f"Baseline features: {len(features_base)}")

    # === Compute all candidate features ===
    print("\n--- Computing candidate features ---")
    for name, info in CANDIDATES.items():
        df = info['compute'](df)
        for col in info['columns']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # === Baseline walk-forward ===
    print("\n" + "=" * 70)
    print("  BASELINE Walk-Forward AUC")
    print("=" * 70)
    t0 = time.time()
    baseline_auc, baseline_folds = walk_forward_auc(df, features_base, "baseline")
    print(f"  Baseline WF AUC: {baseline_auc:.4f} ({time.time()-t0:.0f}s)")
    for y, a in baseline_folds.items():
        print(f"    {y}: {a:.4f}")

    # === Test each candidate feature ===
    results = {}
    print("\n" + "=" * 70)
    print("  CANDIDATE FEATURE EVALUATION")
    print("=" * 70)

    for name, info in CANDIDATES.items():
        cols = info['columns']
        features_test = features_base + cols
        print(f"\n  Testing: {name} (cols={cols})")
        t0 = time.time()
        test_auc, test_folds = walk_forward_auc(df, features_test, name)
        delta = test_auc - baseline_auc
        adopted = delta > 0.0001  # At least +0.0001 improvement
        elapsed = time.time() - t0
        marker = "ADOPTED" if adopted else ("MARGINAL" if delta > 0 else "REJECTED")
        print(f"    WF AUC: {test_auc:.4f} (delta {delta:+.5f}) [{marker}] ({elapsed:.0f}s)")
        for y, a in test_folds.items():
            base_y = baseline_folds.get(y, 0)
            print(f"      {y}: {a:.4f} ({a - base_y:+.4f})")

        results[name] = {
            'wf_auc': round(test_auc, 5),
            'delta': round(delta, 5),
            'adopted': adopted,
            'year_aucs': {str(y): round(a, 5) for y, a in test_folds.items()},
        }

    # === Combined evaluation: all adopted features together ===
    adopted_features = [name for name, r in results.items() if r['adopted']]
    print(f"\n" + "=" * 70)
    print(f"  COMBINED EVALUATION: {len(adopted_features)} adopted features")
    print(f"  Adopted: {adopted_features}")
    print("=" * 70)

    if adopted_features:
        combined_cols = []
        for name in adopted_features:
            combined_cols.extend(CANDIDATES[name]['columns'])
        features_combined = features_base + combined_cols
        t0 = time.time()
        combined_auc, combined_folds = walk_forward_auc(df, features_combined, "combined")
        combined_delta = combined_auc - baseline_auc
        print(f"  Combined WF AUC: {combined_auc:.4f} (delta {combined_delta:+.5f}) ({time.time()-t0:.0f}s)")
        for y, a in combined_folds.items():
            base_y = baseline_folds.get(y, 0)
            print(f"    {y}: {a:.4f} ({a - base_y:+.4f})")
    else:
        combined_auc = baseline_auc
        combined_folds = baseline_folds
        print("  No features adopted. Final AUC = baseline.")

    # === Summary ===
    elapsed_total = time.time() - t_start
    print(f"\n" + "=" * 70)
    print(f"  SUMMARY (total time: {elapsed_total/60:.1f} min)")
    print("=" * 70)
    print(f"  Baseline WF AUC: {baseline_auc:.5f}")
    print(f"  {'Feature':<30} {'WF AUC':>10} {'Delta':>10} {'Status'}")
    print(f"  {'-' * 62}")

    sorted_results = sorted(results.items(), key=lambda x: x[1]['delta'], reverse=True)
    for name, r in sorted_results:
        status = "ADOPTED" if r['adopted'] else "rejected"
        print(f"  {name:<30} {r['wf_auc']:>10.5f} {r['delta']:>+10.5f} {status}")

    print(f"\n  Adopted features: {adopted_features}")
    print(f"  Final combined WF AUC: {combined_auc:.5f} (delta {combined_auc - baseline_auc:+.5f})")

    # === Save results ===
    output = {
        'baseline_wf_auc': round(baseline_auc, 5),
        'baseline_year_aucs': {str(y): round(a, 5) for y, a in baseline_folds.items()},
        'candidates': results,
        'best_features': adopted_features,
        'final_wf_auc': round(combined_auc, 5),
        'final_year_aucs': {str(y): round(a, 5) for y, a in combined_folds.items()},
        'elapsed_minutes': round(elapsed_total / 60, 1),
    }

    out_path = os.path.join(BASE_DIR, 'data', 'feature_exploration_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")
    print("  Feature exploration complete!")


if __name__ == '__main__':
    main()
