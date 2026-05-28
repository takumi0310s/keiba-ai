#!/usr/bin/env python3
"""V16a Ability-Enhanced Candidate Training Script

V16 (オッズ除外) に 4グループの新能力特徴量を追加:
  1. jockey × trainer combo top3 rate
  2. 馬体重 slope / std (直近5走)
  3. 正規化パフォーマンス (ELO proxy)
  4. 枠番バイアス

V15 / V16 production 完全不変。出力:
  models/v16a_candidate.pkl.gz
  data/v16a_lookup_tables.pkl

Usage:
    python train/train_v16a_ability.py
    python train/train_v16a_ability.py --wf-only
"""

import os
import sys
import time
import json
import gzip
import pickle
import argparse
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

from train_v16_ability import ODDS_FEATURES_REMOVE, LGB_PARAMS, XGB_PARAMS
from features_v16a import (
    compute_all_v16a_features, V16A_FEATURE_NAMES, V16A_DEFAULTS,
    build_lookup_tables, save_lookup_tables,
)

WF_YEARS = range(2021, 2026)
V15_BASELINE_WF = 0.8678
V16_BASELINE_WF = 0.8677


def load_cache():
    path = os.path.join(DATA_DIR, '_v15_optuna_df_cache.pkl.gz')
    print(f"Loading V15 cache: {path}")
    t0 = time.time()
    with gzip.open(path, 'rb') as f:
        obj = pickle.load(f)
    df = obj['df']
    v15_features = obj['features']
    print(f"  df: {len(df)} rows, {len(df.columns)} cols  ({time.time()-t0:.1f}s)")
    print(f"  V15 features: {len(v15_features)}")
    return df, v15_features


def get_v16a_features(v15_features: list) -> tuple[list, list]:
    """V15 → V16a: オッズ除外 + V16a新特徴量追加。"""
    removed = [f for f in ODDS_FEATURES_REMOVE if f in v15_features]
    v16_base = [f for f in v15_features if f not in ODDS_FEATURES_REMOVE]
    v16a = v16_base + [f for f in V16A_FEATURE_NAMES if f not in v16_base]
    print(f"\n  Feature composition:")
    print(f"    V15:       {len(v15_features)} feats")
    print(f"    - removed: {len(removed)} (odds: {removed})")
    print(f"    + added:   {len(V16A_FEATURE_NAMES)} (v16a new)")
    print(f"    = V16a:    {len(v16a)} feats")
    return v16a, removed


def walk_forward_lgb_xgb(df, features, years=WF_YEARS):
    results = []
    for test_year in years:
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask  = df['year'] == ty
        n_tr = train_mask.sum()
        n_te = test_mask.sum()
        if n_tr < 1000 or n_te < 100:
            print(f"  Skip {test_year}: train={n_tr}, test={n_te}")
            continue

        X_tr = df.loc[train_mask, features].fillna(0).values
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[test_mask,  features].fillna(0).values
        y_te = df.loc[test_mask,  'target'].values
        print(f"\n  WF {test_year}: train={n_tr}, test={n_te}, feats={len(features)}")

        dt = lgb.Dataset(X_tr, label=y_tr)
        dv = lgb.Dataset(X_te, label=y_te, reference=dt)
        m_lgb = lgb.train(LGB_PARAMS, dt, num_boost_round=1000,
                          valid_sets=[dv],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
        p_lgb = m_lgb.predict(X_te)
        auc_lgb = roc_auc_score(y_te, p_lgb)

        dxtr = xgb.DMatrix(X_tr, label=y_tr)
        dxte = xgb.DMatrix(X_te, label=y_te)
        m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=1000,
                           evals=[(dxte, 'valid')],
                           early_stopping_rounds=50, verbose_eval=False)
        p_xgb = m_xgb.predict(dxte)
        auc_xgb = roc_auc_score(y_te, p_xgb)

        p_ens = 0.5 * p_lgb + 0.5 * p_xgb
        auc_ens = roc_auc_score(y_te, p_ens)
        print(f"    LGB={auc_lgb:.4f}  XGB={auc_xgb:.4f}  ENS={auc_ens:.4f}")
        results.append({
            'year': test_year, 'auc_lgb': auc_lgb, 'auc_xgb': auc_xgb,
            'auc_ensemble': auc_ens, 'n_train': int(n_tr), 'n_test': int(n_te),
        })
    return results


def train_final_model(df, features):
    print("\n  Training final model (all data 2020-2025)...")
    mask = (df['year'] >= 20) & (df['year'] <= 25)
    X = df.loc[mask, features].fillna(0).values
    y = df.loc[mask, 'target'].values
    print(f"    Train rows: {len(X)}, features: {len(features)}")

    dt = lgb.Dataset(X, label=y)
    m_lgb = lgb.train({**LGB_PARAMS, 'num_leaves': 63}, dt, num_boost_round=500)
    p_lgb = m_lgb.predict(X)
    auc_lgb = roc_auc_score(y, p_lgb)
    print(f"    LGB in-sample AUC: {auc_lgb:.4f}")

    dxtr = xgb.DMatrix(X, label=y)
    m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=500,
                       evals=[(dxtr, 'train')], verbose_eval=False)
    return m_lgb, m_xgb, auc_lgb


def save_candidate(m_lgb, m_xgb, features, wf_results, auc_lgb):
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    out_path = os.path.join(BASE_DIR, 'models', 'v16a_candidate.pkl.gz')

    wf_mean = float(np.mean([r['auc_ensemble'] for r in wf_results])) if wf_results else 0.0

    pkl = {
        'version': 'v16a_candidate',
        'description': 'V16 ability-only + jockey×trainer/weight_slope/perf_score/gate_bias',
        'model': m_lgb,
        'xgb_model': m_xgb,
        'ensemble_weights': {'lgb': 0.5, 'xgb': 0.5, 'mlp': 0},
        'features': features,
        'auc': auc_lgb,
        'wf_auc_mean': wf_mean,
        'wf_results': wf_results,
        'n_features': len(features),
        'new_features': V16A_FEATURE_NAMES,
        'new_feature_defaults': V16A_DEFAULTS,
        'leak_free': True,
        'leak_pattern': 'A',
        'is_live': False,
        'is_candidate': True,
        'trained_at': datetime.now().isoformat(),
        'v15_baseline_wf_auc': V15_BASELINE_WF,
        'v16_baseline_wf_auc': V16_BASELINE_WF,
    }

    with gzip.open(out_path, 'wb') as f:
        pickle.dump(pkl, f)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n  Saved: {out_path} ({size_mb:.1f} MB)")
    print(f"  WF mean AUC: {wf_mean:.4f}  (V16 baseline: {V16_BASELINE_WF})")
    verdict = "GO [OK]" if wf_mean >= V16_BASELINE_WF else "NO-GO [--]"
    print(f"  Verdict vs V16: {verdict}")
    verdict_v15 = "GO [OK]" if wf_mean >= V15_BASELINE_WF else "NO-GO [--]"
    print(f"  Verdict vs V15: {verdict_v15}")
    return out_path, wf_mean


def save_wf_results(wf_results, wf_mean):
    os.makedirs(os.path.join(DATA_DIR, 'v20'), exist_ok=True)
    out = os.path.join(DATA_DIR, 'v20', 'v16a_wf_results.json')
    record = {
        'model': 'v16a_candidate',
        'wf_mean': wf_mean,
        'wf_results': wf_results,
        'v15_baseline': V15_BASELINE_WF,
        'v16_baseline': V16_BASELINE_WF,
        'delta_vs_v15': round(wf_mean - V15_BASELINE_WF, 4),
        'delta_vs_v16': round(wf_mean - V16_BASELINE_WF, 4),
        'new_features': V16A_FEATURE_NAMES,
        'verdict': 'GO' if wf_mean >= V16_BASELINE_WF else 'NO-GO',
        'run_at': datetime.now().isoformat(),
    }
    with open(out, 'w') as f:
        json.dump(record, f, indent=2)
    print(f"  WF results → {out}")
    return record


def main():
    parser = argparse.ArgumentParser(description="V16a ability-enhanced candidate training")
    parser.add_argument('--wf-only', action='store_true',
                        help='Walk-forward AUC only, skip final model save')
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 60)
    print("V16a Ability-Enhanced Candidate Training")
    print("=" * 60)

    # ── 1. Load cache ──────────────────────────────────────────
    df, v15_features = load_cache()

    # ── 2. Compute V16a features ───────────────────────────────
    print("\n[STEP 1] Computing V16a new features...")
    df = compute_all_v16a_features(df)

    # ── 3. Fill defaults for any missing new features ──────────
    for feat, default in V16A_DEFAULTS.items():
        if feat not in df.columns:
            df[feat] = default
        else:
            df[feat] = df[feat].fillna(default)

    # ── 4. Feature list ────────────────────────────────────────
    v16a_features, removed = get_v16a_features(v15_features)

    # Validate all features exist in df
    missing = [f for f in v16a_features if f not in df.columns]
    if missing:
        print(f"[WARN] Missing features in df: {missing}")
        for f in missing:
            df[f] = V16A_DEFAULTS.get(f, 0.0)

    # ── 5. Walk-forward validation ─────────────────────────────
    print("\n[STEP 2] Walk-forward validation (2021-2025)...")
    wf_results = walk_forward_lgb_xgb(df, v16a_features)

    wf_mean = float(np.mean([r['auc_ensemble'] for r in wf_results])) if wf_results else 0.0
    print(f"\n  WF mean ENS AUC: {wf_mean:.4f}")
    print(f"  vs V15: {wf_mean - V15_BASELINE_WF:+.4f}")
    print(f"  vs V16: {wf_mean - V16_BASELINE_WF:+.4f}")

    record = save_wf_results(wf_results, wf_mean)

    if args.wf_only:
        print("\n[--wf-only] Skipping final model save.")
        return

    # ── 6. Final model training ────────────────────────────────
    print("\n[STEP 3] Final model training (2020-2025)...")
    m_lgb, m_xgb, auc_lgb = train_final_model(df, v16a_features)

    # ── 7. Save candidate model ────────────────────────────────
    print("\n[STEP 4] Saving candidate model...")
    save_candidate(m_lgb, m_xgb, v16a_features, wf_results, auc_lgb)

    # ── 8. Build and save lookup tables ───────────────────────
    print("\n[STEP 5] Building lookup tables for inference...")
    tables = build_lookup_tables(df)
    save_lookup_tables(tables)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} min")
    print(f"  WF AUC: {wf_mean:.4f}  ({'GO' if wf_mean >= V16_BASELINE_WF else 'NO-GO'})")
    print(f"  Model:  models/v16a_candidate.pkl.gz")
    print(f"  Lookup: data/v16a_lookup_tables.pkl")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
