#!/usr/bin/env python3
"""v13.1 best_class（母最高クラス）追加テスト

v13(99特徴量) + dam_best_class_exp(expanding window) でWFバックテスト。
採用基準: WF AUC > 0.8207 (v13) かつ全年AUC > 0.78 かつ gap < 0.05

dam_best_class_exp:
  各レース時点で、当該レースを除く過去の全レースから、
  同じ母馬の産駒が達成した最高class_codeを取得。
  class_codeが大きいほど上位（7=新馬 < 23=未勝利 < ... < 195=OP）。
  expanding windowでリークフリー。
"""

import os
import sys
import time
import pickle
import gzip
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

from train_v92_central import (
    load_data, encode_categoricals, encode_sires,
    load_training_times, merge_training_features,
    compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance,
    compute_lag_features, build_features,
    compute_distance_aptitude, compute_frame_advantage,
    FEATURES_V91, V92_NEW_FEATURES, V93_NEW_FEATURES,
    COURSE_MAP, N_TOP_SIRE,
)
from train_v92_leakfree import LEAK_FEATURES_A
from train_v12_comprehensive import (
    V12_NEW_FEATURES, LGB_PARAMS, XGB_PARAMS,
    merge_speed_index, merge_training_1f, merge_sire_shinba,
    merge_dam_stats, compute_pci,
)
from jrdb_features import (
    merge_jrdb_train_features,
    JRDB_ALL_FEATURES, JRDB_LIVE_FEATURES, JRDB_DEFAULTS,
)


# v13で採用されたJRDB特徴量（v13_training_results.jsonから取得）
def load_v13_jrdb_selected():
    result_path = os.path.join(DATA_DIR, 'v13_training_results.json')
    if os.path.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('jrdb_selected', [])
    # fallback: 3個除外
    return [f for f in JRDB_ALL_FEATURES
            if f not in ('jrdb_info_idx', 'jrdb_prev_ten_idx', 'jrdb_prev_agari_idx')]


# =====================================================
# dam_best_class expanding window
# =====================================================

def compute_dam_best_class(df):
    """母馬の産駒最高クラスをexpanding windowで計算（リークフリー）

    class_code: 7=新馬, 15=OP, 23=未勝利, 43=1勝, 67=2勝, 131=3勝, etc.
    数値が大きいほど上位クラス。

    各レースiについて、同じ母馬の産駒がレースiより前に達成した最高class_codeを返す。
    当該レース自身は除外（リークフリー）。
    """
    print("  Computing dam_best_class_exp (expanding window, leak-free)...")

    if 'mother' not in df.columns or 'class_code' not in df.columns:
        df['dam_best_class_exp'] = 0
        print("  WARNING: mother or class_code column not found")
        return df

    df = df.sort_values(['year', 'month', 'day', 'race_num']).reset_index(drop=True)
    dam_best = np.zeros(len(df))

    # class_code to numeric
    cc = pd.to_numeric(df['class_code'], errors='coerce').fillna(0).astype(int)

    # Expanding window: track max class_code per mother
    mother_max_class = {}  # mother -> max class_code seen so far
    for i in range(len(df)):
        m = df.iloc[i]['mother']
        if pd.isna(m) or m == '':
            continue

        # Get current best BEFORE this race
        if m in mother_max_class:
            dam_best[i] = mother_max_class[m]

        # Update with this race's class_code (for future rows)
        this_cc = cc.iloc[i]
        if this_cc > 0:
            if m not in mother_max_class:
                mother_max_class[m] = this_cc
            else:
                mother_max_class[m] = max(mother_max_class[m], this_cc)

    df['dam_best_class_exp'] = dam_best
    valid = (dam_best > 0).sum()
    mean_val = dam_best[dam_best > 0].mean() if valid > 0 else 0
    # Fill NaN/0 with global mean
    df.loc[df['dam_best_class_exp'] == 0, 'dam_best_class_exp'] = mean_val
    print(f"  Dam best class (expanding): {valid}/{len(df)} computed ({valid/len(df)*100:.1f}%)")
    print(f"  Mean value: {mean_val:.1f}")

    return df


# =====================================================
# WF backtest (reuse from v12)
# =====================================================

def walk_forward_backtest(df, features, label='target', years=range(2020, 2026)):
    results = []
    for test_year in years:
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue
        X_train = df.loc[train_mask, features].values
        y_train = df.loc[train_mask, label].values
        X_test = df.loc[test_mask, features].values
        y_test = df.loc[test_mask, label].values

        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)
        lgb_model = lgb.train(
            LGB_PARAMS, dtrain, num_boost_round=1000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        lgb_pred = lgb_model.predict(X_test)
        lgb_auc = roc_auc_score(y_test, lgb_pred)

        dxtr = xgb.DMatrix(X_train, label=y_train)
        dxte = xgb.DMatrix(X_test, label=y_test)
        xgb_model = xgb.train(
            XGB_PARAMS, dxtr, num_boost_round=1000,
            evals=[(dxte, 'valid')],
            early_stopping_rounds=50, verbose_eval=False
        )
        xgb_pred = xgb_model.predict(dxte)
        xgb_auc = roc_auc_score(y_test, xgb_pred)

        lgb_train_pred = lgb_model.predict(X_train)
        train_auc = roc_auc_score(y_train, lgb_train_pred)

        results.append({
            'year': test_year,
            'lgb_auc': lgb_auc, 'xgb_auc': xgb_auc,
            'train_auc': train_auc, 'gap': train_auc - lgb_auc,
        })
    return results


def print_wf(results, label=''):
    print(f"\n  WF Results: {label}")
    print(f"  {'='*55}")
    aucs = []
    for r in results:
        aucs.append(r['lgb_auc'])
        ok = 'Y' if r['lgb_auc'] > 0.78 else 'N'
        gap_ok = 'Y' if r['gap'] < 0.05 else 'N'
        print(f"  {r['year']}: LGB={r['lgb_auc']:.4f} XGB={r['xgb_auc']:.4f} "
              f"Train={r['train_auc']:.4f} Gap={r['gap']:.4f} {ok}{gap_ok}")
    mean_auc = np.mean(aucs)
    print(f"  --- Mean LGB AUC: {mean_auc:.4f} ---")
    return mean_auc


# =====================================================
# Main
# =====================================================

def main():
    start_time = time.time()
    print("=" * 60)
    print("  v13.1 dam_best_class Test")
    print(f"  Baseline: v13 WF AUC 0.8207 (99 features)")
    print("=" * 60)

    # Phase 1-4: Same as v13 (load, encode, stats, v12 features)
    print("\n[1/6] Loading data...")
    df = load_data()
    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)

    print("\n[2/6] Merging training data...")
    tt_data = load_training_times()
    df = merge_training_features(df, tt_data)

    print("\n[3/6] Computing historical stats...")
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)
    df = compute_lag_features(df)
    df = build_features(df)
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)

    print("\n[4/6] Merging v12 features...")
    df = merge_speed_index(df)
    df = merge_training_1f(df)
    df = merge_sire_shinba(df)
    df = merge_dam_stats(df)
    df = compute_pci(df)

    print("\n[5/6] Merging JRDB features...")
    df = merge_jrdb_train_features(df)

    # NEW: dam_best_class
    print("\n[6/6] Computing dam_best_class...")
    df = compute_dam_best_class(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses'] >= 5].copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    print(f"\n  Final dataset: {len(df)} rows")

    # Feature lists
    FEATURES_V93_ALL = list(FEATURES_V91) + list(V92_NEW_FEATURES) + list(V93_NEW_FEATURES)
    FEATURES_V12_BASELINE = [f for f in FEATURES_V93_ALL if f not in LEAK_FEATURES_A]
    V12_ADOPTED = [f for f in V12_NEW_FEATURES if f != 'dam_top3r']
    FEATURES_V12 = FEATURES_V12_BASELINE + V12_ADOPTED

    jrdb_selected = load_v13_jrdb_selected()
    FEATURES_V13 = FEATURES_V12 + jrdb_selected
    FEATURES_V131 = FEATURES_V13 + ['dam_best_class_exp']

    # Ensure all features exist
    for f in FEATURES_V131:
        if f not in df.columns:
            print(f"  WARNING: {f} not in df, filling with default")
            df[f] = JRDB_DEFAULTS.get(f, 0)
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(JRDB_DEFAULTS.get(f, 0))

    print(f"\n  v13 features: {len(FEATURES_V13)}")
    print(f"  v13.1 features: {len(FEATURES_V131)} (+1: dam_best_class_exp)")

    # WF: Baseline (v13)
    print(f"\n{'='*60}")
    print(f"  Walk-Forward Backtest")
    print(f"{'='*60}")

    print("\n--- v13 Baseline ---")
    wf_v13 = walk_forward_backtest(df, FEATURES_V13)
    auc_v13 = print_wf(wf_v13, 'v13 Baseline')

    print("\n--- v13.1 +dam_best_class_exp ---")
    wf_v131 = walk_forward_backtest(df, FEATURES_V131)
    auc_v131 = print_wf(wf_v131, 'v13.1 +dam_best_class_exp')

    # Summary
    delta = auc_v131 - auc_v13
    all_ok = all(r['lgb_auc'] > 0.78 for r in wf_v131)
    no_overfit = all(r['gap'] < 0.05 for r in wf_v131)
    adopted = auc_v131 > 0.8207 and all_ok and no_overfit

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  v13 AUC:     {auc_v13:.4f}")
    print(f"  v13.1 AUC:   {auc_v131:.4f}")
    print(f"  Delta:       {delta:+.4f}")
    print(f"  AUC > 0.8207:     {'Y' if auc_v131 > 0.8207 else 'N'}")
    print(f"  All years > 0.78: {'Y' if all_ok else 'N'}")
    print(f"  No overfitting:   {'Y' if no_overfit else 'N'}")
    print(f"\n  VERDICT: {'ADOPTED as v13.1' if adopted else 'NOT ADOPTED'}")

    # Save results
    result_data = {
        'v13_baseline_auc': auc_v13,
        'v131_auc': auc_v131,
        'delta': delta,
        'adopted': adopted,
        'new_feature': 'dam_best_class_exp',
        'v131_features': FEATURES_V131,
        'yearly_results': [{
            'year': r['year'], 'lgb_auc': r['lgb_auc'], 'xgb_auc': r['xgb_auc'],
            'train_auc': r['train_auc'], 'gap': r['gap']
        } for r in wf_v131],
        'timestamp': datetime.now().isoformat(),
    }
    result_path = os.path.join(DATA_DIR, 'v131_training_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved: {result_path}")

    # Train production model if adopted
    if adopted:
        print(f"\n{'='*60}")
        print(f"  TRAINING PRODUCTION MODEL (v13.1)")
        print(f"{'='*60}")
        train_production(df, FEATURES_V131, sire_map, bms_map, wf_v131, jrdb_selected)

    elapsed = (time.time() - start_time) / 60
    print(f"\n  Total elapsed: {elapsed:.1f} min")


def train_production(df, features, sire_map, bms_map, wf_results, jrdb_features):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    max_year = int(df['year'].max())
    valid_mask = df['year'] >= (max_year - 1)
    train_mask = ~valid_mask

    X_train = df.loc[train_mask, features].values
    y_train = df.loc[train_mask, 'target'].values
    X_valid = df.loc[valid_mask, features].values
    y_valid = df.loc[valid_mask, 'target'].values

    print(f"  Train: {train_mask.sum()}, Valid: {valid_mask.sum()}")
    print(f"  Features: {len(features)}")

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
    lgb_model = lgb.train(
        LGB_PARAMS, dtrain, num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    lgb_auc = roc_auc_score(y_valid, lgb_model.predict(X_valid))
    print(f"  LGB AUC: {lgb_auc:.4f}")

    dxtr = xgb.DMatrix(X_train, label=y_train)
    dxte = xgb.DMatrix(X_valid, label=y_valid)
    xgb_model = xgb.train(
        XGB_PARAMS, dxtr, num_boost_round=1000,
        evals=[(dxte, 'valid')],
        early_stopping_rounds=50, verbose_eval=100
    )
    xgb_auc = roc_auc_score(y_valid, xgb_model.predict(dxte))
    print(f"  XGB AUC: {xgb_auc:.4f}")

    total = lgb_auc + xgb_auc
    w_lgb = lgb_auc / total
    w_xgb = xgb_auc / total
    wf_auc = np.mean([r['lgb_auc'] for r in wf_results])

    pattern_a_features = [f for f in features if f not in LEAK_FEATURES_A]
    pkl_a = {
        'model': lgb_model,
        'features': pattern_a_features,
        'version': 'v131_leakfree',
        'auc': lgb_auc, 'wf_auc': wf_auc,
        'leak_free': True, 'leak_pattern': 'A',
        'leak_removed': sorted(LEAK_FEATURES_A),
        'sire_map': sire_map, 'bms_map': bms_map,
        'n_top_encode': N_TOP_SIRE,
        'trained_at': now,
        'model_type': 'central',
        'xgb_model': xgb_model,
        'mlp_model': None, 'mlp_scaler': None,
        'ensemble_weights': {'lgb': w_lgb, 'xgb': w_xgb, 'mlp': 0},
        'course_map': dict(COURSE_MAP),
        'jrdb_features': jrdb_features,
    }

    a_path = os.path.join(BASE_DIR, 'keiba_model_v131_central.pkl.gz')
    with gzip.open(a_path, 'wb') as f:
        pickle.dump(pkl_a, f)
    print(f"  Pattern A saved: {a_path}")

    pattern_b_features = features + JRDB_LIVE_FEATURES
    pkl_b = dict(pkl_a)
    pkl_b['features'] = pattern_b_features
    pkl_b['version'] = 'v131_live'
    pkl_b['leak_free'] = False
    pkl_b['leak_pattern'] = 'B'
    pkl_b['is_live'] = True

    b_path = os.path.join(BASE_DIR, 'keiba_model_v131_central_live.pkl.gz')
    with gzip.open(b_path, 'wb') as f:
        pickle.dump(pkl_b, f)
    print(f"  Pattern B saved: {b_path}")

    print(f"\n  *** v13.1 Production model saved! ***")
    print(f"  Pattern A: {len(pattern_a_features)} features, WF AUC {wf_auc:.4f}")
    print(f"  Pattern B: {len(pattern_b_features)} features")


if __name__ == '__main__':
    main()
