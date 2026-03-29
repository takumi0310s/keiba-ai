#!/usr/bin/env python3
"""dam_top3r再検証: v12の74特徴量 vs 75特徴量(+dam_top3r) WF AUC比較

v12で-0.0006で除外されたdam_top3rを再評価。
データが増えた現在のexpanding windowsで再計算し、WFバックテストで検証。

採用基準: WF AUC > 0.8037 (v12 selected AUC)
"""
import os
import sys
import io
import time
import json

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from datetime import datetime

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
    V12_NEW_FEATURES, LGB_PARAMS, XGB_PARAMS,
)


def walk_forward(df, features, years=range(2020, 2026)):
    results = []
    for test_year in years:
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue
        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[test_mask, features].values
        y_te = df.loc[test_mask, 'target'].values

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
        lgb_model = lgb.train(
            LGB_PARAMS, dtrain, num_boost_round=1000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        lgb_pred = lgb_model.predict(X_te)
        lgb_auc = roc_auc_score(y_te, lgb_pred)

        dxtr = xgb.DMatrix(X_tr, label=y_tr)
        dxte = xgb.DMatrix(X_te, label=y_te)
        xgb_model = xgb.train(
            XGB_PARAMS, dxtr, num_boost_round=1000,
            evals=[(dxte, 'valid')],
            early_stopping_rounds=50, verbose_eval=False
        )
        xgb_pred = xgb_model.predict(dxte)
        xgb_auc = roc_auc_score(y_te, xgb_pred)

        lgb_train_pred = lgb_model.predict(X_tr)
        train_auc = roc_auc_score(y_tr, lgb_train_pred)

        results.append({
            'year': test_year, 'lgb_auc': lgb_auc, 'xgb_auc': xgb_auc,
            'train_auc': train_auc, 'gap': train_auc - lgb_auc,
        })
    return results


def main():
    start = time.time()
    print("=" * 60)
    print("  dam_top3r 再検証")
    print("=" * 60)

    # Phase 1: Load
    print("\n[1/4] Loading & encoding...")
    df = load_data()
    print(f"  Loaded: {len(df)} rows")
    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)

    # Phase 2: Features
    print("\n[2/4] Building all features...")
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

    # v12 new features
    print("\n[3/4] v12 features + dam_top3r...")
    df = merge_speed_index(df)
    df = merge_training_1f(df)
    df = merge_sire_shinba(df)
    df = merge_dam_stats(df)  # This recomputes dam_top3r with latest data
    df = compute_pci(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses'] >= 5].copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    print(f"  Final: {len(df)} rows")

    # Feature lists
    FEATURES_V93 = list(FEATURES_V91) + list(V92_NEW_FEATURES) + list(V93_NEW_FEATURES)
    FEATURES_BASELINE = [f for f in FEATURES_V93 if f not in LEAK_FEATURES_A]

    # v12 selected (74 features, WITHOUT dam_top3r)
    V12_SELECTED = [f for f in V12_NEW_FEATURES if f != 'dam_top3r']
    FEATURES_74 = FEATURES_BASELINE + V12_SELECTED

    # v12 + dam_top3r (75 features)
    FEATURES_75 = FEATURES_BASELINE + V12_NEW_FEATURES  # includes dam_top3r

    for f_list in [FEATURES_74, FEATURES_75]:
        for f in f_list:
            if f not in df.columns:
                print(f"  WARNING: {f} missing, filling 0")
                df[f] = 0
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"\n  74 features (current v12): {len(FEATURES_74)}")
    print(f"  75 features (+dam_top3r):  {len(FEATURES_75)}")

    # Phase 4: WF Backtest
    print("\n[4/4] Walk-forward comparison...")

    print("\n--- v12 Current (74 features) ---")
    wf_74 = walk_forward(df, FEATURES_74)
    aucs_74 = [r['lgb_auc'] for r in wf_74]
    mean_74 = np.mean(aucs_74)
    for r in wf_74:
        ok = '✓' if r['lgb_auc'] > 0.78 else '✗'
        print(f"  {r['year']}: LGB={r['lgb_auc']:.4f} XGB={r['xgb_auc']:.4f} Gap={r['gap']:.4f} {ok}")
    print(f"  Mean: {mean_74:.4f}")

    print("\n--- +dam_top3r (75 features) ---")
    wf_75 = walk_forward(df, FEATURES_75)
    aucs_75 = [r['lgb_auc'] for r in wf_75]
    mean_75 = np.mean(aucs_75)
    for r in wf_75:
        ok = '✓' if r['lgb_auc'] > 0.78 else '✗'
        print(f"  {r['year']}: LGB={r['lgb_auc']:.4f} XGB={r['xgb_auc']:.4f} Gap={r['gap']:.4f} {ok}")
    print(f"  Mean: {mean_75:.4f}")

    # Comparison
    diff = mean_75 - mean_74
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  v12 current (74):  {mean_74:.4f}")
    print(f"  +dam_top3r  (75):  {mean_75:.4f}")
    print(f"  Difference:        {diff:+.4f}")
    print(f"  Threshold:         0.8037")

    all_years_ok = all(r['lgb_auc'] > 0.78 for r in wf_75)
    no_overfit = all(r['gap'] < 0.05 for r in wf_75)
    above_threshold = mean_75 > 0.8037

    adopted = above_threshold and all_years_ok and no_overfit
    print(f"\n  AUC > 0.8037:    {'✓' if above_threshold else '✗'}")
    print(f"  All years > 0.78: {'✓' if all_years_ok else '✗'}")
    print(f"  No overfitting:   {'✓' if no_overfit else '✗'}")
    print(f"  VERDICT: {'✓ ADOPT dam_top3r' if adopted else '✗ KEEP CURRENT (dam_top3r not adopted)'}")

    # Year-by-year diff
    print(f"\n  Year-by-year comparison:")
    for r74, r75 in zip(wf_74, wf_75):
        d = r75['lgb_auc'] - r74['lgb_auc']
        print(f"  {r74['year']}: {r74['lgb_auc']:.4f} → {r75['lgb_auc']:.4f} ({d:+.4f})")

    # Save results
    result = {
        'test': 'dam_top3r_reverify',
        'date': datetime.now().isoformat(),
        'v12_current_auc': mean_74,
        'with_dam_top3r_auc': mean_75,
        'diff': diff,
        'threshold': 0.8037,
        'adopted': adopted,
        'yearly_74': [{'year': r['year'], 'lgb_auc': r['lgb_auc'], 'gap': r['gap']} for r in wf_74],
        'yearly_75': [{'year': r['year'], 'lgb_auc': r['lgb_auc'], 'gap': r['gap']} for r in wf_75],
        'data_rows': len(df),
    }
    out_path = os.path.join(DATA_DIR, 'dam_top3r_reverify.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    print(f"  Elapsed: {time.time()-start:.0f}s")


if __name__ == '__main__':
    main()
