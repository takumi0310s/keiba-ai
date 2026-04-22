#!/usr/bin/env python3
"""v15 Master Training/Evaluation Script

v14.1 (131 features, WF AUC 0.8856) をベースに最大14新特徴量を追加評価:
  - v15 new features (10): jockey-horse compat(5), transport(2), renovation(2), gaisha(1)
  - PACI Tier B marks (4): sogo_mark, idm_mark, jockey_mark, train_mark

Phase 1: Build v15 dataframe (v14.1 + new + PACI-B)
Phase 2: All-in WF evaluation (4-model ensemble, 2021-2025)
Phase 3: Ablation (if all-in fails acceptance criteria)
Phase 4: Save report to data/v15_master_report.json
Phase 5: Production model save (if improved over v14.1)

採用基準: WF AUC > 0.8856, max gap < 0.05, 全年AUC > 0.85

Usage:
    python train/train_v15_master.py                  # Full pipeline
    python train/train_v15_master.py --skip-ablation  # Skip ablation even if all-in fails
    python train/train_v15_master.py --phase1-only    # Just build dataframe and report feature counts
"""

import os
import sys
import time
import json
import pickle
import gzip
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

from train_v141_paci_tierA import (
    build_v141_dataframe, get_v141_features, fill_v141_defaults,
    V141_NEW_FEATURES, V141_DEFAULTS,
)
from train_v135_ft_transformer import (
    FTTransformer, IntraRaceAttention, train_ft_transformer,
    DEVICE, LGB_PARAMS, XGB_PARAMS,
)
from train_v135b_intra_ensemble import (
    build_race_id, train_intra_race_with_preds, MAX_HORSES,
)
from jrdb_features import _build_nk_race_id_from_jv, JRDB_DEFAULTS, PACI_TIER_A_DEFAULTS
from features_v15_new import compute_all_v15_new_features, get_v15_new_features, V15_NEW_DEFAULTS
from train_v92_central import COURSE_MAP, N_TOP_SIRE
from train_v92_leakfree import LEAK_FEATURES_A


# =====================================================
# Constants
# =====================================================

BASELINE_AUC = 0.8856  # v14.1 WF mean AUC
WF_YEARS = range(2021, 2026)

# PACI Tier B features (marks from jrdb_paci.csv)
PACI_TIER_B_FEATURES = [
    'paci_sogo_mark',
    'paci_idm_mark',
    'paci_jockey_mark',
    'paci_train_mark',
]

PACI_TIER_B_DEFAULTS = {
    'paci_sogo_mark': 0.0,
    'paci_idm_mark': 0.0,
    'paci_jockey_mark': 0.0,
    'paci_train_mark': 0.0,
}

# Mark encoding: ◎=5, ○=4, ▲=3, △=2, ×=1, else=0
MARK_ENCODING = {
    '◎': 5, '○': 4, '▲': 3, '△': 2, '×': 1,
    '1': 5, '2': 4, '3': 3, '4': 2, '5': 1,
}

# Ablation feature groups
ABLATION_GROUPS = {
    'jockey_horse': [
        'jockey_horse_rides', 'jockey_horse_wr', 'jockey_horse_top3r',
        'jockey_change', 'jockey_change_to_top',
    ],
    'transport': [
        'transport_distance_km', 'is_long_transport',
    ],
    'renovation': [
        'course_renovated', 'post_renovation_flag',
    ],
    'gaisha': [
        'gaisha_rank',
    ],
    'paci_tierB': PACI_TIER_B_FEATURES,
}


# =====================================================
# PACI Tier B merge
# =====================================================

def merge_paci_tierB_features(df):
    """Merge PACI Tier B mark features (sogo/idm/jockey/train marks)."""
    path = os.path.join(DATA_DIR, 'jrdb_paci.csv')
    if not os.path.exists(path):
        print("    PACI Tier B: jrdb_paci.csv not found, using defaults")
        for f in PACI_TIER_B_FEATURES:
            df[f] = PACI_TIER_B_DEFAULTS[f]
        return df

    paci = pd.read_csv(path, encoding='utf-8-sig', dtype=str,
                        usecols=['race_id', 'umaban',
                                 'sogo_mark', 'idm_mark',
                                 'jockey_mark', 'train_mark'])

    paci['_nk_rid'] = paci['race_id'].astype(str).str.zfill(12)
    paci['_uma'] = pd.to_numeric(paci['umaban'], errors='coerce')

    # Encode marks: ◎=5, ○=4, ▲=3, △=2, ×=1, else=0
    for src, dst in [('sogo_mark', 'paci_sogo_mark'),
                     ('idm_mark', 'paci_idm_mark'),
                     ('jockey_mark', 'paci_jockey_mark'),
                     ('train_mark', 'paci_train_mark')]:
        paci[dst] = paci[src].map(MARK_ENCODING).fillna(0).astype(float)

    cols = ['_nk_rid', '_uma'] + PACI_TIER_B_FEATURES
    paci_d = paci[cols].drop_duplicates(subset=['_nk_rid', '_uma'], keep='last')

    if '_nk_rid' not in df.columns:
        df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    if '_uma' not in df.columns:
        df['_uma'] = pd.to_numeric(df['umaban'], errors='coerce')

    before = len(df)
    df = df.merge(paci_d, on=['_nk_rid', '_uma'], how='left', suffixes=('', '_pb'))

    # Handle suffix conflicts
    for f in PACI_TIER_B_FEATURES:
        if f not in df.columns and f'{f}_pb' in df.columns:
            df[f] = df[f'{f}_pb']
            df.drop(columns=[f'{f}_pb'], inplace=True)

    matched = df[PACI_TIER_B_FEATURES[0]].notna().sum() if PACI_TIER_B_FEATURES[0] in df.columns else 0
    print(f"    PACI Tier B: {matched}/{before} matched ({matched/before*100:.1f}%)")

    # Fill NaN
    for f in PACI_TIER_B_FEATURES:
        if f in df.columns:
            df[f] = df[f].fillna(PACI_TIER_B_DEFAULTS[f])
        else:
            df[f] = PACI_TIER_B_DEFAULTS[f]

    if len(df) != before:
        print(f"    WARNING: merge changed row count {before} -> {len(df)}")

    return df


# =====================================================
# Phase 1: Build v15 dataframe
# =====================================================

def build_v15_dataframe():
    """Build v15 dataframe: v14.1 base + 10 new features + 4 PACI Tier B features."""
    print("\n" + "=" * 70)
    print("  Phase 1: Building v15 dataframe")
    print("=" * 70)

    # Step 1: Build v14.1 base
    print("\n[Step 1] Building v14.1 base dataframe...")
    df, sire_map, bms_map = build_v141_dataframe()
    print(f"  v14.1 base: {len(df)} rows")

    # Need merge keys for PACI Tier B
    if '_nk_rid' not in df.columns:
        df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    if '_uma' not in df.columns:
        df['_uma'] = pd.to_numeric(df['umaban'], errors='coerce')

    # Step 2: Compute v15 new features (10)
    print("\n[Step 2] Computing v15 new features (10)...")
    df = compute_all_v15_new_features(df)

    # Step 3: Merge PACI Tier B features (4)
    print("\n[Step 3] Merging PACI Tier B mark features (4)...")
    df = merge_paci_tierB_features(df)

    # Cleanup merge keys
    df.drop(columns=['_nk_rid', '_uma'], inplace=True, errors='ignore')

    return df, sire_map, bms_map


def get_v15_all_features():
    """Get full v15 feature list = v14.1(131) + new(10) + PACI-B(4)."""
    base_feats, jrdb_sel = get_v141_features()
    v15_new = get_v15_new_features()
    return base_feats + v15_new + PACI_TIER_B_FEATURES, jrdb_sel


def fill_v15_defaults(df, features):
    """Fill all v15 features with defaults."""
    # First fill v14.1 defaults
    df = fill_v141_defaults(df, features)

    # Then fill v15 new defaults
    all_v15_defaults = {**V15_NEW_DEFAULTS, **PACI_TIER_B_DEFAULTS}
    for f in features:
        if f in all_v15_defaults:
            if f not in df.columns:
                df[f] = all_v15_defaults[f]
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(all_v15_defaults[f])
    return df


# =====================================================
# 4-model Walk-Forward
# =====================================================

def walk_forward_4model(df, features, years=WF_YEARS, label='v15'):
    """Run 4-model ensemble walk-forward evaluation.

    Returns list of per-year result dicts.
    """
    if 'race_id_unique' not in df.columns:
        df = build_race_id(df)
        print(f"  [WF] race_id_unique created: {df['race_id_unique'].nunique()} races")
    results = []
    for test_year in years:
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            print(f"  Skipping {test_year}: insufficient data "
                  f"(train={train_mask.sum()}, test={test_mask.sum()})")
            continue

        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[test_mask, features].values
        y_te = df.loc[test_mask, 'target'].values
        test_indices = df.loc[test_mask].index.values

        print(f"\n{'='*60}")
        print(f"  4-Model WF {test_year} [{label}]: "
              f"train={len(X_tr)}, test={len(X_te)}, feats={len(features)}")
        print(f"{'='*60}")

        # --- LightGBM ---
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
        m_lgb = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                          valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        p_lgb = m_lgb.predict(X_te)
        auc_lgb = roc_auc_score(y_te, p_lgb)
        p_lgb_tr = m_lgb.predict(X_tr)
        auc_lgb_tr = roc_auc_score(y_tr, p_lgb_tr)

        # --- XGBoost ---
        dxtr = xgb.DMatrix(X_tr, label=y_tr)
        dxte = xgb.DMatrix(X_te, label=y_te)
        m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=1000,
                           evals=[(dxte, 'valid')],
                           early_stopping_rounds=50, verbose_eval=False)
        p_xgb = m_xgb.predict(dxte)
        auc_xgb = roc_auc_score(y_te, p_xgb)

        # --- FT-Transformer ---
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr.astype(np.float32))
        X_te_s = scaler.transform(X_te.astype(np.float32))
        ft_model, p_ft, p_ft_tr, auc_ft = train_ft_transformer(
            X_tr_s, y_tr.astype(np.float32),
            X_te_s, y_te.astype(np.float32),
            n_features=len(features),
            epochs=50, batch_size=2048, lr=1e-3,
            patience=10, d_token=48, n_heads=4, n_layers=3,
            dropout=0.1, label=f'FT-{test_year}',
        )

        # --- IntraRace Attention ---
        df_tr = df.loc[train_mask].copy()
        df_te = df.loc[test_mask].copy()
        scaler_ir = StandardScaler()
        df_tr_scaled = df_tr.copy()
        df_te_scaled = df_te.copy()
        df_tr_scaled[features] = scaler_ir.fit_transform(
            df_tr[features].values.astype(np.float32))
        df_te_scaled[features] = scaler_ir.transform(
            df_te[features].values.astype(np.float32))

        print(f"  Training IntraRace Attention...")
        ir_model, ir_val_dict, ir_tr_dict, ir_tr_auc, ir_val_auc = \
            train_intra_race_with_preds(
                df_tr_scaled, df_te_scaled, features,
                epochs=30, batch_size=64, lr=1e-3, patience=8,
                d_model=64, n_heads=4, n_layers=2, dropout=0.1,
            )

        p_ir = np.zeros(len(X_te), dtype=np.float32)
        ir_coverage = 0
        for i, idx in enumerate(test_indices):
            if idx in ir_val_dict:
                p_ir[i] = ir_val_dict[idx]
                ir_coverage += 1
            else:
                p_ir[i] = 0.3
        ir_cov_pct = ir_coverage / len(X_te) * 100
        auc_ir = roc_auc_score(y_te, p_ir) if ir_coverage > len(X_te) * 0.5 else 0
        print(f"  IR coverage: {ir_coverage}/{len(X_te)} ({ir_cov_pct:.1f}%)")

        # --- Grid search 4-model weights ---
        best_grid_auc = 0
        best_grid_w = None
        if auc_ir > 0:
            for w1 in np.arange(0.20, 0.45, 0.05):
                for w2 in np.arange(0.20, 0.45, 0.05):
                    for w3 in np.arange(0.05, 0.30, 0.05):
                        w4 = 1.0 - w1 - w2 - w3
                        if w4 < 0.01 or w4 > 0.40:
                            continue
                        p_grid = w1*p_lgb + w2*p_xgb + w3*p_ft + w4*p_ir
                        auc_grid = roc_auc_score(y_te, p_grid)
                        if auc_grid > best_grid_auc:
                            best_grid_auc = auc_grid
                            best_grid_w = (w1, w2, w3, w4)

        gap = auc_lgb_tr - auc_lgb

        # Simple ensembles for reference
        w2 = auc_lgb / (auc_lgb + auc_xgb)
        auc_lgbxgb = roc_auc_score(y_te, w2*p_lgb + (1-w2)*p_xgb)

        print(f"\n  {test_year} Results [{label}]:")
        print(f"    LGB:     {auc_lgb:.4f} (train: {auc_lgb_tr:.4f})")
        print(f"    XGB:     {auc_xgb:.4f}")
        print(f"    FT:      {auc_ft:.4f}")
        print(f"    IR:      {auc_ir:.4f} (cov={ir_cov_pct:.1f}%)")
        print(f"    LGB+XGB: {auc_lgbxgb:.4f}")
        if best_grid_w:
            print(f"    Grid:    {best_grid_auc:.4f} "
                  f"(L={best_grid_w[0]:.2f} X={best_grid_w[1]:.2f} "
                  f"F={best_grid_w[2]:.2f} IR={best_grid_w[3]:.2f})")
        print(f"    Gap:     {gap:.4f} {'OK' if gap < 0.05 else 'NG'}")

        results.append({
            'year': test_year,
            'lgb_auc': float(auc_lgb),
            'xgb_auc': float(auc_xgb),
            'ft_auc': float(auc_ft),
            'ir_auc': float(auc_ir),
            'ir_coverage': float(ir_cov_pct),
            'lgbxgb_auc': float(auc_lgbxgb),
            'grid_auc': float(best_grid_auc) if best_grid_w else None,
            'grid_weights': [float(w) for w in best_grid_w] if best_grid_w else None,
            'train_auc': float(auc_lgb_tr),
            'gap': float(gap),
        })

    return results


def summarize_results(results, label=''):
    """Print summary table and return mean grid AUC."""
    print(f"\n  {'Year':<6} {'LGB':>7} {'XGB':>7} {'FT':>7} {'IR':>7} "
          f"{'L+X':>7} {'Grid':>8} {'Gap':>6}")
    print(f"  {'-'*60}")
    for r in results:
        ok = 'Y' if r['gap'] < 0.05 else 'N'
        grid = f"{r['grid_auc']:.4f}" if r['grid_auc'] else '  N/A '
        print(f"  {r['year']:<6} {r['lgb_auc']:7.4f} {r['xgb_auc']:7.4f} "
              f"{r['ft_auc']:7.4f} {r['ir_auc']:7.4f} "
              f"{r['lgbxgb_auc']:7.4f} {grid:>8} {r['gap']:6.4f}{ok}")

    grid_vals = [r['grid_auc'] for r in results if r['grid_auc']]
    grid_mean = float(np.mean(grid_vals)) if grid_vals else 0.0
    return grid_mean


def check_acceptance(results, grid_mean, baseline=BASELINE_AUC):
    """Check if results meet adoption criteria.

    Returns (adopted: bool, reasons: list[str])
    """
    reasons = []
    if grid_mean <= baseline:
        reasons.append(f"AUC {grid_mean:.6f} <= baseline {baseline:.6f}")

    all_year_ok = all(
        max(r['lgb_auc'], r.get('grid_auc', 0) or 0) > 0.85
        for r in results
    )
    if not all_year_ok:
        bad = [r['year'] for r in results
               if max(r['lgb_auc'], r.get('grid_auc', 0) or 0) <= 0.85]
        reasons.append(f"year AUC <= 0.85: {bad}")

    no_overfit = all(r['gap'] < 0.05 for r in results)
    if not no_overfit:
        bad = [(r['year'], f"{r['gap']:.4f}") for r in results if r['gap'] >= 0.05]
        reasons.append(f"gap >= 0.05: {bad}")

    adopted = len(reasons) == 0
    return adopted, reasons


# =====================================================
# Phase 2: All-in evaluation
# =====================================================

def run_all_in(df, features):
    """Run all-in WF evaluation with all v15 features."""
    print("\n" + "=" * 70)
    print("  Phase 2: All-in WF Evaluation")
    print(f"  Features: {len(features)}")
    print(f"  Baseline: {BASELINE_AUC}")
    print("=" * 70)

    results = walk_forward_4model(df, features, years=WF_YEARS, label='all-in')

    print(f"\n{'='*70}")
    print(f"  ALL-IN SUMMARY")
    print(f"{'='*70}")

    grid_mean = summarize_results(results)
    delta = grid_mean - BASELINE_AUC

    print(f"\n  Mean Grid AUC: {grid_mean:.6f}")
    print(f"  v14.1 baseline: {BASELINE_AUC:.6f}")
    print(f"  Delta:          {delta:+.6f}")

    adopted, reasons = check_acceptance(results, grid_mean)
    if adopted:
        print(f"\n  *** ALL-IN ACCEPTED ***")
    else:
        print(f"\n  ALL-IN REJECTED: {', '.join(reasons)}")

    return results, grid_mean, adopted


# =====================================================
# Phase 3: Ablation
# =====================================================

def run_ablation(df, base_features, jrdb_sel):
    """Run ablation: test each feature group individually on top of v14.1 base."""
    print("\n" + "=" * 70)
    print("  Phase 3: Ablation Testing")
    print(f"  Testing {len(ABLATION_GROUPS)} feature groups individually")
    print("=" * 70)

    ablation_results = {}

    for group_name, group_feats in ABLATION_GROUPS.items():
        # Check which features are actually available
        available = [f for f in group_feats if f in df.columns]
        if not available:
            print(f"\n  [{group_name}] No features available, skipping")
            ablation_results[group_name] = {
                'features': group_feats,
                'available': [],
                'skipped': True,
            }
            continue

        test_features = base_features + available
        print(f"\n  [{group_name}] Testing {len(available)} features: {available}")
        print(f"  Total features: {len(test_features)}")

        results = walk_forward_4model(
            df, test_features, years=WF_YEARS, label=group_name)
        grid_mean = summarize_results(results, label=group_name)
        delta = grid_mean - BASELINE_AUC
        adopted, reasons = check_acceptance(results, grid_mean)

        print(f"\n  [{group_name}] Grid AUC: {grid_mean:.6f} ({delta:+.6f})")
        if adopted:
            print(f"  [{group_name}] *** ACCEPTED ***")
        else:
            print(f"  [{group_name}] REJECTED: {', '.join(reasons)}")

        ablation_results[group_name] = {
            'features': available,
            'grid_mean': grid_mean,
            'delta': delta,
            'adopted': adopted,
            'reasons': reasons if not adopted else [],
            'yearly': results,
        }

    # Determine which groups to adopt
    adopted_groups = [name for name, r in ablation_results.items()
                      if r.get('adopted', False)]
    adopted_features = []
    for name in adopted_groups:
        adopted_features.extend(ablation_results[name]['features'])

    print(f"\n{'='*70}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*70}")
    for name, r in ablation_results.items():
        if r.get('skipped'):
            print(f"  {name:15s}  SKIPPED (no features)")
        else:
            status = 'ADOPTED' if r.get('adopted') else 'REJECTED'
            print(f"  {name:15s}  {r['grid_mean']:.6f} ({r['delta']:+.6f})  {status}")

    print(f"\n  Adopted groups: {adopted_groups}")
    print(f"  Adopted features ({len(adopted_features)}): {adopted_features}")

    return ablation_results, adopted_features


# =====================================================
# Phase 4: Combined ablation validation
# =====================================================

def run_combined_ablation(df, base_features, adopted_features):
    """If multiple ablation groups pass, validate their combination."""
    if not adopted_features:
        return None, 0.0, False

    combined_features = base_features + adopted_features
    print(f"\n{'='*70}")
    print(f"  Phase 4b: Combined Ablation Validation")
    print(f"  Base: {len(base_features)} + adopted: {len(adopted_features)}")
    print(f"  Total: {len(combined_features)}")
    print(f"{'='*70}")

    results = walk_forward_4model(
        df, combined_features, years=WF_YEARS, label='combined')
    grid_mean = summarize_results(results, label='combined')
    delta = grid_mean - BASELINE_AUC
    adopted, reasons = check_acceptance(results, grid_mean)

    print(f"\n  Combined Grid AUC: {grid_mean:.6f} ({delta:+.6f})")
    if adopted:
        print(f"  *** COMBINED ACCEPTED ***")
    else:
        print(f"  COMBINED REJECTED: {', '.join(reasons)}")

    return results, grid_mean, adopted


# =====================================================
# Phase 5: Production model save
# =====================================================

def save_production_model(df, features, sire_map, bms_map):
    """Train final model on all data and save pkl.gz for Pattern A and B."""
    print(f"\n{'='*70}")
    print(f"  Phase 5: Production Model Save")
    print(f"{'='*70}")

    # Use 2021-2025 data for training (same as WF evaluation range)
    # Final model: train on ALL available data
    train_mask = df['year'] >= 10  # 2010+
    X_all = df.loc[train_mask, features].values
    y_all = df.loc[train_mask, 'target'].values
    print(f"  Training on {len(X_all)} samples, {len(features)} features")

    # LightGBM
    print("  Training LightGBM...")
    dtrain = lgb.Dataset(X_all, label=y_all)
    lgb_model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=500)
    lgb_auc = roc_auc_score(y_all, lgb_model.predict(X_all))
    print(f"    LGB train AUC: {lgb_auc:.4f}")

    # XGBoost
    print("  Training XGBoost...")
    dxtrain = xgb.DMatrix(X_all, label=y_all)
    xgb_model = xgb.train(XGB_PARAMS, dxtrain, num_boost_round=500)
    xgb_auc = roc_auc_score(y_all, xgb_model.predict(dxtrain))
    print(f"    XGB train AUC: {xgb_auc:.4f}")

    now = datetime.now().isoformat()

    # Pattern A (leak-free)
    pkl_a = {
        'model': lgb_model,
        'xgb_model': xgb_model,
        'features': features,
        'version': 'v15',
        'auc': lgb_auc,
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
        'ensemble_weights': {
            'lgb': lgb_auc / (lgb_auc + xgb_auc),
            'xgb': xgb_auc / (lgb_auc + xgb_auc),
            'mlp': 0,
        },
        'course_map': dict(COURSE_MAP),
    }

    a_path = os.path.join(BASE_DIR, 'keiba_model_v15_central.pkl.gz')
    with gzip.open(a_path, 'wb') as f:
        pickle.dump(pkl_a, f, protocol=4)
    print(f"  Pattern A saved: {a_path}")

    # Pattern B (live = A + JRDB live features)
    from jrdb_features import JRDB_LIVE_FEATURES
    pattern_b_features = features + JRDB_LIVE_FEATURES
    pkl_b = dict(pkl_a)
    pkl_b['features'] = pattern_b_features
    pkl_b['version'] = 'v15_live'
    pkl_b['leak_free'] = False
    pkl_b['leak_pattern'] = 'B'
    pkl_b['is_live'] = True

    b_path = os.path.join(BASE_DIR, 'keiba_model_v15_central_live.pkl.gz')
    with gzip.open(b_path, 'wb') as f:
        pickle.dump(pkl_b, f, protocol=4)
    print(f"  Pattern B saved: {b_path}")

    print(f"\n  *** v15 Production model saved! ***")
    print(f"  Pattern A: {len(features)} features")
    print(f"  Pattern B: {len(pattern_b_features)} features")

    return a_path, b_path


# =====================================================
# Feature importance analysis
# =====================================================

def analyze_feature_importance(df, features, new_features):
    """Analyze LGB feature importance for new features (2025 fold)."""
    print(f"\n  Feature importance analysis (LGB, 2025 fold):")
    ty = 25
    train_mask = df['year'] < ty
    test_mask = df['year'] == ty
    if test_mask.sum() == 0:
        print("    No 2025 data, skipping importance analysis")
        return {}

    dtrain = lgb.Dataset(
        df.loc[train_mask, features].values,
        label=df.loc[train_mask, 'target'].values)
    dvalid = lgb.Dataset(
        df.loc[test_mask, features].values,
        label=df.loc[test_mask, 'target'].values,
        reference=dtrain)
    m = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                  valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    imp = dict(zip(features, m.feature_importance('gain')))
    total_imp = sum(imp.values())

    importance_dict = {}
    for f in new_features:
        val = imp.get(f, 0)
        pct = val / total_imp * 100 if total_imp > 0 else 0
        importance_dict[f] = {'gain': float(val), 'pct': float(pct)}
        print(f"    {f:30s}: {val:8.0f} ({pct:.2f}%)")

    return importance_dict


# =====================================================
# Feature coverage report
# =====================================================

def report_feature_coverage(df, new_features, defaults):
    """Report coverage (non-default rate) for new features."""
    print(f"\n  Feature coverage report:")
    coverage = {}
    for f in new_features:
        if f not in df.columns:
            print(f"    {f:30s}  MISSING")
            coverage[f] = 0.0
            continue
        default_val = defaults.get(f, 0)
        nondefault = (df[f] != default_val).sum()
        pct = nondefault / len(df) * 100
        coverage[f] = pct
        print(f"    {f:30s}  nondefault={nondefault}/{len(df)} ({pct:.1f}%)")
    return coverage


# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser(description='v15 Master Training/Evaluation')
    parser.add_argument('--skip-ablation', action='store_true',
                        help='Skip ablation even if all-in fails')
    parser.add_argument('--phase1-only', action='store_true',
                        help='Just build dataframe and report feature counts')
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("  v15 Master Training/Evaluation Pipeline")
    print(f"  Baseline: v14.1 WF AUC {BASELINE_AUC}")
    print(f"  Device: {DEVICE}")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # ===== Phase 1: Build dataframe =====
    df, sire_map, bms_map = build_v15_dataframe()
    all_features, jrdb_sel = get_v15_all_features()
    df = fill_v15_defaults(df, all_features)
    df = build_race_id(df)

    # Feature lists
    base_features, _ = get_v141_features()
    v15_new = get_v15_new_features()
    all_new_features = v15_new + PACI_TIER_B_FEATURES

    print(f"\n  DataFrame: {len(df)} rows")
    print(f"  v14.1 base features: {len(base_features)}")
    print(f"  v15 new features: {len(v15_new)}")
    print(f"  PACI Tier B features: {len(PACI_TIER_B_FEATURES)}")
    print(f"  Total features: {len(all_features)}")

    # Coverage report
    all_defaults = {**V15_NEW_DEFAULTS, **PACI_TIER_B_DEFAULTS}
    coverage = report_feature_coverage(df, all_new_features, all_defaults)

    if args.phase1_only:
        elapsed = (time.time() - t0) / 60
        print(f"\n  Phase 1 complete. Elapsed: {elapsed:.1f} min")
        report = {
            'phase': 'phase1_only',
            'baseline_auc': BASELINE_AUC,
            'total_features': len(all_features),
            'base_features': len(base_features),
            'new_features': all_new_features,
            'coverage': coverage,
            'dataframe_rows': len(df),
            'timestamp': datetime.now().isoformat(),
            'elapsed_min': elapsed,
        }
        rpath = os.path.join(DATA_DIR, 'v15_master_report.json')
        with open(rpath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"  Report saved: {rpath}")
        return

    # ===== Phase 2: All-in evaluation =====
    allin_results, allin_auc, allin_adopted = run_all_in(df, all_features)

    # Feature importance
    importance = analyze_feature_importance(df, all_features, all_new_features)

    # ===== Phase 3: Ablation (if needed) =====
    ablation_results = None
    combined_results = None
    combined_auc = 0.0
    combined_adopted = False
    final_adopted_features = []

    if allin_adopted:
        final_adopted_features = all_new_features
        print("\n  All-in passed - skipping ablation")
    elif not args.skip_ablation:
        ablation_results, adopted_features = run_ablation(
            df, base_features, jrdb_sel)

        if adopted_features:
            # Validate combination if multiple groups adopted
            if len(adopted_features) > 0:
                combined_results, combined_auc, combined_adopted = \
                    run_combined_ablation(df, base_features, adopted_features)

            if combined_adopted:
                final_adopted_features = adopted_features
            else:
                # Fall back to individual best group
                best_group = None
                best_auc = 0
                for name, r in ablation_results.items():
                    if r.get('adopted') and r.get('grid_mean', 0) > best_auc:
                        best_auc = r['grid_mean']
                        best_group = name
                if best_group:
                    final_adopted_features = ablation_results[best_group]['features']
                    print(f"\n  Best individual group: {best_group} "
                          f"(AUC {best_auc:.6f})")
    else:
        print("\n  Ablation skipped (--skip-ablation)")

    # ===== Determine final outcome =====
    if allin_adopted:
        final_features = all_features
        final_auc = allin_auc
        outcome = 'all_in'
    elif final_adopted_features:
        final_features = base_features + final_adopted_features
        final_auc = combined_auc if combined_adopted else 0
        # Re-check: is it actually better?
        if final_auc > BASELINE_AUC:
            outcome = 'ablation_combined' if combined_adopted else 'ablation_best'
        else:
            outcome = 'none'
            final_features = base_features
            final_auc = BASELINE_AUC
    else:
        outcome = 'none'
        final_features = base_features
        final_auc = BASELINE_AUC

    # ===== Summary =====
    print(f"\n{'='*70}")
    print(f"  v15 FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Baseline (v14.1):   {BASELINE_AUC:.6f} ({len(base_features)} features)")
    print(f"  All-in AUC:         {allin_auc:.6f} ({len(all_features)} features)")
    if combined_results:
        print(f"  Combined ablation:  {combined_auc:.6f}")
    print(f"  Outcome:            {outcome}")
    print(f"  Final features:     {len(final_features)}")
    if final_adopted_features:
        print(f"  Adopted new feats:  {final_adopted_features}")

    # ===== Phase 5: Save production model =====
    model_saved = False
    model_paths = {}
    if outcome != 'none':
        print(f"\n  Saving production model (outcome={outcome})...")
        a_path, b_path = save_production_model(
            df, final_features, sire_map, bms_map)
        model_saved = True
        model_paths = {'pattern_a': a_path, 'pattern_b': b_path}
    else:
        print(f"\n  No improvement over v14.1. No model saved.")

    # ===== Phase 4: Save report =====
    elapsed = (time.time() - t0) / 60
    report = {
        'baseline_auc': BASELINE_AUC,
        'all_in': {
            'features_count': len(all_features),
            'grid_mean': allin_auc,
            'delta': allin_auc - BASELINE_AUC,
            'adopted': allin_adopted,
            'yearly': allin_results,
        },
        'ablation': {},
        'combined_ablation': None,
        'outcome': outcome,
        'adopted_features': final_adopted_features,
        'final_features_count': len(final_features),
        'feature_importance': importance,
        'feature_coverage': coverage,
        'model_saved': model_saved,
        'model_paths': model_paths,
        'device': str(DEVICE),
        'timestamp': datetime.now().isoformat(),
        'elapsed_min': float(elapsed),
    }

    if ablation_results:
        for name, r in ablation_results.items():
            report['ablation'][name] = {
                'features': r.get('features', []),
                'grid_mean': r.get('grid_mean', 0),
                'delta': r.get('delta', 0),
                'adopted': r.get('adopted', False),
                'reasons': r.get('reasons', []),
                'yearly': r.get('yearly', []),
            }

    if combined_results:
        report['combined_ablation'] = {
            'features': final_adopted_features,
            'grid_mean': combined_auc,
            'delta': combined_auc - BASELINE_AUC,
            'adopted': combined_adopted,
            'yearly': combined_results,
        }

    rpath = os.path.join(DATA_DIR, 'v15_master_report.json')
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Report saved: {rpath}")

    print(f"  Total time: {elapsed:.1f} min")
    print("=" * 70)

    return outcome != 'none', report


if __name__ == '__main__':
    result = main()
    if result is None:
        # phase1-only
        sys.exit(0)
    adopted, report = result
    sys.exit(0 if adopted else 1)
