#!/usr/bin/env python3
"""dam_top3r再投入検証 — v12(74特徴量) + dam_top3r(expanding) でWF AUC比較

採用基準:
  - WF AUC > 0.8037 (v12ベースライン)
  - 全年AUC > 0.78
  - 全年gap < 0.05

Usage:
    python train/test_dam_top3r_reinvestigate.py
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

from train_v92_central import (
    load_data, encode_categoricals, encode_sires,
    load_training_times, merge_training_features,
    compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance,
    compute_lag_features, build_features,
    compute_distance_aptitude, compute_frame_advantage,
    FEATURES_V91, V92_NEW_FEATURES, V93_NEW_FEATURES,
)
from train_v92_leakfree import LEAK_FEATURES_A
from train_v12_comprehensive import (
    merge_speed_index, merge_training_1f, merge_sire_shinba,
    merge_dam_stats, compute_pci,
    LGB_PARAMS, V12_NEW_FEATURES,
)

def main():
    start = time.time()
    print("=" * 60)
    print("  dam_top3r Re-investigation (v12 + dam_top3r)")
    print("=" * 60)

    # === Data loading (same as train_v12_comprehensive.py) ===
    print("\n[1/4] Loading data...")
    df = load_data()
    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)
    print(f"  Loaded: {len(df)} rows")

    print("\n[2/4] Computing features...")
    tt_data = load_training_times()
    df = merge_training_features(df, tt_data)
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)
    df = compute_lag_features(df)
    df = build_features(df)
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)

    print("\n[3/4] Merging v12 features...")
    df = merge_speed_index(df)
    df = merge_training_1f(df)
    df = merge_sire_shinba(df)
    df = merge_dam_stats(df)  # This adds dam_top3r
    df = compute_pci(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses'] >= 5].copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)

    # Feature lists
    FEATURES_V93 = list(FEATURES_V91) + list(V92_NEW_FEATURES) + list(V93_NEW_FEATURES)
    FEATURES_BASELINE = [f for f in FEATURES_V93 if f not in LEAK_FEATURES_A]

    # v12 adopted features (without dam_top3r)
    V12_SELECTED = [f for f in V12_NEW_FEATURES if f != 'dam_top3r']
    FEATURES_V12 = FEATURES_BASELINE + V12_SELECTED

    # v12 + dam_top3r
    FEATURES_V12_DAM = FEATURES_V12 + ['dam_top3r']

    for f in FEATURES_V12_DAM:
        if f not in df.columns:
            print(f"  WARNING: {f} not in df, filling 0")
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"\n  v12 features: {len(FEATURES_V12)}")
    print(f"  v12+dam features: {len(FEATURES_V12_DAM)}")

    # === WF Backtest ===
    print("\n[4/4] Walk-forward backtest...")
    results_base = []
    results_dam = []

    for test_year in range(2020, 2026):
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        y_train = df.loc[train_mask, 'target'].values
        y_test = df.loc[test_mask, 'target'].values

        # v12 baseline
        X_tr = df.loc[train_mask, FEATURES_V12].values
        X_te = df.loc[test_mask, FEATURES_V12].values
        dtrain = lgb.Dataset(X_tr, label=y_train)
        dvalid = lgb.Dataset(X_te, label=y_test, reference=dtrain)
        m = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                      valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = m.predict(X_te)
        auc = roc_auc_score(y_test, pred)
        train_pred = m.predict(X_tr)
        train_auc = roc_auc_score(y_train, train_pred)
        results_base.append({'year': test_year, 'auc': auc, 'train_auc': train_auc,
                             'gap': train_auc - auc})

        # v12 + dam_top3r
        X_tr2 = df.loc[train_mask, FEATURES_V12_DAM].values
        X_te2 = df.loc[test_mask, FEATURES_V12_DAM].values
        dtrain2 = lgb.Dataset(X_tr2, label=y_train)
        dvalid2 = lgb.Dataset(X_te2, label=y_test, reference=dtrain2)
        m2 = lgb.train(LGB_PARAMS, dtrain2, num_boost_round=1000,
                       valid_sets=[dvalid2],
                       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred2 = m2.predict(X_te2)
        auc2 = roc_auc_score(y_test, pred2)
        train_pred2 = m2.predict(X_tr2)
        train_auc2 = roc_auc_score(y_train, train_pred2)
        results_dam.append({'year': test_year, 'auc': auc2, 'train_auc': train_auc2,
                            'gap': train_auc2 - auc2})

        diff = auc2 - auc
        print(f"  {test_year}: v12={auc:.4f} v12+dam={auc2:.4f} diff={diff:+.5f} "
              f"gap_base={train_auc-auc:.4f} gap_dam={train_auc2-auc2:.4f}")

    # === Summary ===
    mean_base = np.mean([r['auc'] for r in results_base])
    mean_dam = np.mean([r['auc'] for r in results_dam])
    improvement = mean_dam - mean_base

    all_years_ok = all(r['auc'] > 0.78 for r in results_dam)
    no_overfit = all(r['gap'] < 0.05 for r in results_dam)
    auc_improved = mean_dam > 0.8037

    print(f"\n{'='*60}")
    print(f"  RESULT")
    print(f"{'='*60}")
    print(f"  v12 baseline:  {mean_base:.4f}")
    print(f"  v12 + dam_top3r: {mean_dam:.4f}")
    print(f"  Improvement:    {improvement:+.5f}")
    print(f"  AUC > 0.8037:   {'PASS' if auc_improved else 'FAIL'}")
    print(f"  All years > 0.78: {'PASS' if all_years_ok else 'FAIL'}")
    print(f"  No overfitting:   {'PASS' if no_overfit else 'FAIL'}")

    adopted = auc_improved and all_years_ok and no_overfit
    print(f"\n  VERDICT: {'ADOPTED' if adopted else 'NOT ADOPTED'}")

    # Save
    out = {
        'test': 'dam_top3r_reinvestigation',
        'timestamp': pd.Timestamp.now().isoformat(),
        'baseline_auc': mean_base,
        'dam_auc': mean_dam,
        'improvement': improvement,
        'adopted': adopted,
        'yearly_base': results_base,
        'yearly_dam': results_dam,
    }
    out_path = os.path.join(DATA_DIR, 'dam_top3r_reverify.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    print(f"  Elapsed: {(time.time()-start)/60:.1f} min")


if __name__ == '__main__':
    main()
