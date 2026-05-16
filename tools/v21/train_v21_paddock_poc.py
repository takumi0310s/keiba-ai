"""Phase A — V21 paddock PoC trainer.

Compares V15 baseline vs V15 + 12 paddock features on 6-fold WF
(2020-2025), LGB+XGB only.

Constraints / honesty:
  - paddock features cover only 33 races / 85 horses, all of 2026
  - V15 cache (_v15_optuna_df_cache.pkl.gz) is 2015-2025
  - (race_id, horse_id) exact-match overlap = 0
  - PoC purpose: validate merge plumbing and produce baseline numbers.

V15 production untouched. Writes:
  - data/v21/phase_a_poc_result.json
  - data/v21/phase_a_poc_result.md
"""

from __future__ import annotations

import os
import sys
import gzip
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
V21_DIR = DATA_DIR / 'v21'
CACHE_PATH = DATA_DIR / '_v15_optuna_df_cache.pkl.gz'
PADDOCK_CSV = V21_DIR / 'paddock_12_features.csv'

OUT_JSON = V21_DIR / 'phase_a_poc_result.json'
OUT_MD = V21_DIR / 'phase_a_poc_result.md'

PADDOCK_FEATURES = [
    'pad_body_condition_score', 'pad_body_condition_std',
    'pad_coat_gloss', 'pad_coat_contrast',
    'pad_body_aspect_mean', 'pad_body_aspect_var', 'pad_body_compactness',
    'pad_gait_motion_speed', 'pad_gait_motion_std',
    'pad_gait_aspect_range', 'pad_gait_area_change', 'pad_yolo_conf_mean',
]

# Conservative defaults (matches V15 style: do not impute beyond LGB native NaN)
LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'seed': 42,
}

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'seed': 42,
    'tree_method': 'hist',
}

WF_YEARS = range(2020, 2026)


def load_cache():
    print(f"[1] loading cache: {CACHE_PATH}")
    with gzip.open(CACHE_PATH, 'rb') as f:
        d = pickle.load(f)
    df = d['df']
    features = d['features']
    print(f"  cache shape: {df.shape}")
    print(f"  features: {len(features)}")
    print(f"  year range: {sorted(df['year'].unique())}")
    return df, features


def merge_paddock(df: pd.DataFrame):
    """Merge paddock 12 features into cache.

    Cache uses TFJV horse_id (8-digit) and TFJV race_id (10-char).
    Paddock CSV uses netkeiba 12-digit race_id and 10-digit horse_id.

    Mapping (from MEMORY/horse_id_mapper.md):
      netkeiba horse_id (10) = '20' + TFJV horse_id (8)
      → reverse: TFJV horse_id = netkeiba_id[2:]

    For race_id: cache has no 2026 entries; paddock all 2026.
    Match on (race_id_nk, horse_id_nk) where:
      We construct an attempt netkeiba-style id from cache, but cache's
      race_id formats are TFJV-internal. We only do a defensive merge:
        horse_id match (tfjv) + race_id same as nk-style if applicable.

    Strategy: attempt exact (race_id_str, horse_id) merge after
    normalising both sides to netkeiba-style 12-digit race_id. If 0
    matches, report honestly.
    """
    print(f"\n[2] loading paddock CSV: {PADDOCK_CSV}")
    if not PADDOCK_CSV.exists():
        print(f"  [ERROR] paddock csv missing")
        return df, 0, 0

    pad = pd.read_csv(PADDOCK_CSV, dtype={'race_id': str, 'horse_id': str})
    print(f"  paddock rows: {len(pad)}")
    print(f"  paddock distinct races: {pad['race_id'].nunique()}")
    print(f"  paddock distinct horses: {pad['horse_id'].nunique()}")

    # Map paddock netkeiba horse_id -> TFJV (drop first 2 chars)
    pad['_horse_tfjv'] = pad['horse_id'].str[2:]

    # Cache horse_id is already TFJV 8-digit
    # Try direct (race_id, horse_id) merge: race_id in cache is TFJV format,
    # paddock race_id is netkeiba 12-digit. They differ -> direct merge
    # produces 0 matches. We attempt anyway and report.
    print(f"\n[3] attempt merge on (race_id, horse_id [tfjv])")
    # Convert paddock race_id to a candidate cache race_id by stripping
    # leading '20' if cache 8-digit, else keep — there is no canonical
    # forward mapping documented for race_id, so direct equality is the
    # safest probe.
    df = df.copy()
    df['horse_id'] = df['horse_id'].astype(str)
    pad_subset = pad[['race_id', '_horse_tfjv'] + PADDOCK_FEATURES].rename(
        columns={'_horse_tfjv': 'horse_id'}
    )
    pad_subset = pad_subset.drop_duplicates(subset=['race_id', 'horse_id'], keep='last')

    df = df.merge(pad_subset, left_on=['race_id', 'horse_id'],
                  right_on=['race_id', 'horse_id'], how='left')

    n_matched = df[PADDOCK_FEATURES[0]].notna().sum()
    pct = n_matched / len(df) * 100
    print(f"  matched rows (race_id+horse_id): {n_matched}/{len(df)} ({pct:.4f}%)")

    # If 0 matches: paddock features will be all-NaN in cache. LGB will
    # treat them as constant (no signal). XGB likewise. PoC = baseline.
    if n_matched == 0:
        print(f"  [INFO] 0% overlap — paddock features all-NaN in 2015-2025 cache")
        print(f"  PoC will produce baseline-equivalent results.")

    return df, int(n_matched), float(pct)


def wf_eval(df: pd.DataFrame, features: list, label: str, years=WF_YEARS):
    """Walk-forward 6-fold (2020-2025) with LGB + XGB.

    Returns list of per-year result dicts.
    """
    print(f"\n=== WF eval [{label}] (LGB+XGB, n_features={len(features)}) ===")
    results = []
    for test_year in years:
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            print(f"  skip {test_year}: train={train_mask.sum()}, test={test_mask.sum()}")
            continue
        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[test_mask, features].values
        y_te = df.loc[test_mask, 'target'].values
        print(f"  [{label}] fold {test_year}: train={len(X_tr)}, test={len(X_te)}")

        # LGB
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
        m_lgb = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                          valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        p_lgb = m_lgb.predict(X_te)
        auc_lgb = roc_auc_score(y_te, p_lgb)

        # XGB
        dxtr = xgb.DMatrix(X_tr, label=y_tr)
        dxte = xgb.DMatrix(X_te, label=y_te)
        m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=1000,
                          evals=[(dxte, 'valid')],
                          early_stopping_rounds=50, verbose_eval=False)
        p_xgb = m_xgb.predict(dxte)
        auc_xgb = roc_auc_score(y_te, p_xgb)

        w = auc_lgb / (auc_lgb + auc_xgb)
        auc_ens = roc_auc_score(y_te, w * p_lgb + (1 - w) * p_xgb)

        print(f"    LGB={auc_lgb:.4f} XGB={auc_xgb:.4f} ENS={auc_ens:.4f}")
        results.append({
            'year': test_year,
            'lgb_auc': float(auc_lgb),
            'xgb_auc': float(auc_xgb),
            'ens_auc': float(auc_ens),
            'n_train': int(len(X_tr)),
            'n_test': int(len(X_te)),
        })
    return results


def summarize(results):
    if not results:
        return {'mean_lgb': None, 'mean_xgb': None, 'mean_ens': None}
    return {
        'mean_lgb': float(np.mean([r['lgb_auc'] for r in results])),
        'mean_xgb': float(np.mean([r['xgb_auc'] for r in results])),
        'mean_ens': float(np.mean([r['ens_auc'] for r in results])),
    }


def main():
    print("=" * 70)
    print("  Phase A - V21 paddock PoC (V15 base + 12 paddock features)")
    print("  baseline: V15 features only | candidate: V15 + 12 paddock")
    print("=" * 70)
    t0 = time.time()

    df, base_features = load_cache()
    # Filter to only features that actually exist in the cache
    base_features = [f for f in base_features if f in df.columns]
    print(f"  usable base features: {len(base_features)}")

    df_merged, n_matched, pct_matched = merge_paddock(df)

    # Baseline: V15 features only
    print("\n" + "=" * 70)
    print("  Baseline: V15 features only")
    print("=" * 70)
    res_base = wf_eval(df_merged, base_features, label='V15-baseline')
    summary_base = summarize(res_base)

    # Candidate: V15 + paddock
    print("\n" + "=" * 70)
    print("  Candidate: V15 + paddock 12 features")
    print("=" * 70)
    feat_v21 = base_features + PADDOCK_FEATURES
    res_v21 = wf_eval(df_merged, feat_v21, label='V21-paddock')
    summary_v21 = summarize(res_v21)

    # Deltas
    delta = {}
    if summary_base['mean_lgb'] is not None and summary_v21['mean_lgb'] is not None:
        delta = {
            'd_lgb': summary_v21['mean_lgb'] - summary_base['mean_lgb'],
            'd_xgb': summary_v21['mean_xgb'] - summary_base['mean_xgb'],
            'd_ens': summary_v21['mean_ens'] - summary_base['mean_ens'],
        }

    elapsed_min = (time.time() - t0) / 60.0
    result = {
        'phase': 'A',
        'timestamp': datetime.now().isoformat(),
        'cache_path': str(CACHE_PATH),
        'paddock_csv': str(PADDOCK_CSV),
        'cache_rows': int(len(df_merged)),
        'paddock_rows': 89,
        'paddock_distinct_races': 33,
        'paddock_distinct_horses': 85,
        'merge_matched_rows': n_matched,
        'merge_matched_pct': pct_matched,
        'n_base_features': len(base_features),
        'n_paddock_features': len(PADDOCK_FEATURES),
        'paddock_features': PADDOCK_FEATURES,
        'baseline': {'per_year': res_base, 'summary': summary_base},
        'v21_paddock': {'per_year': res_v21, 'summary': summary_v21},
        'delta': delta,
        'elapsed_min': elapsed_min,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[OK] wrote: {OUT_JSON}")

    # Generate honest MD report
    write_md_report(result)
    print(f"[OK] wrote: {OUT_MD}")
    print(f"\n=== done in {elapsed_min:.1f} min ===")
    return 0


def write_md_report(r: dict):
    sb = r['baseline']['summary']
    sv = r['v21_paddock']['summary']
    d = r.get('delta', {})

    if sb['mean_ens'] is None or sv['mean_ens'] is None:
        verdict = "BLOCKED (WF eval failed to produce results)"
    elif r['merge_matched_rows'] == 0:
        verdict = ("partial-success-coverage-insufficient: 0% overlap between paddock "
                   "video (2026) and V15 cache (2015-2025). Cannot evaluate signal. "
                   "5/31 coverage target needed.")
    else:
        d_ens = d.get('d_ens', 0)
        if d_ens >= 0.0001:
            verdict = f"PoC success: ENS delta = +{d_ens:.6f} (>= +0.0001 threshold)"
        elif d_ens > 0:
            verdict = f"PoC marginal: ENS delta = +{d_ens:.6f} (positive but < +0.0001)"
        else:
            verdict = f"PoC inconclusive: ENS delta = {d_ens:.6f} (coverage too low for signal)"

    lines = []
    lines.append("# Phase A - V21 Paddock PoC Result")
    lines.append("")
    lines.append(f"Generated: {r['timestamp']}")
    lines.append(f"Elapsed: {r['elapsed_min']:.1f} min")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- V15 cache rows: {r['cache_rows']}")
    lines.append(f"- Paddock entries (data/video_ai_features/): {r['paddock_rows']} rows")
    lines.append(f"  - distinct races: {r['paddock_distinct_races']}")
    lines.append(f"  - distinct horses: {r['paddock_distinct_horses']}")
    lines.append(f"- Merge matched (race_id+horse_id): {r['merge_matched_rows']} rows "
                 f"({r['merge_matched_pct']:.4f}% of cache)")
    lines.append("")
    lines.append("## Features")
    lines.append("")
    lines.append(f"- V15 baseline: {r['n_base_features']} features")
    lines.append(f"- V21 candidate: V15 + {r['n_paddock_features']} paddock features")
    lines.append("")
    lines.append("## WF results (6-fold, 2020-2025, LGB+XGB ensemble)")
    lines.append("")
    lines.append("| Year | V15 LGB | V15 XGB | V15 ENS | V21 LGB | V21 XGB | V21 ENS |")
    lines.append("|------|---------|---------|---------|---------|---------|---------|")
    base_by_year = {x['year']: x for x in r['baseline']['per_year']}
    v21_by_year = {x['year']: x for x in r['v21_paddock']['per_year']}
    years = sorted(set(list(base_by_year.keys()) + list(v21_by_year.keys())))
    for y in years:
        b = base_by_year.get(y, {})
        v = v21_by_year.get(y, {})
        lines.append(
            f"| {y} | {b.get('lgb_auc', float('nan')):.4f} "
            f"| {b.get('xgb_auc', float('nan')):.4f} "
            f"| {b.get('ens_auc', float('nan')):.4f} "
            f"| {v.get('lgb_auc', float('nan')):.4f} "
            f"| {v.get('xgb_auc', float('nan')):.4f} "
            f"| {v.get('ens_auc', float('nan')):.4f} |"
        )
    lines.append(f"| mean | {sb['mean_lgb']:.4f} | {sb['mean_xgb']:.4f} | {sb['mean_ens']:.4f} "
                 f"| {sv['mean_lgb']:.4f} | {sv['mean_xgb']:.4f} | {sv['mean_ens']:.4f} |")
    lines.append("")
    lines.append("## Delta (V21 - V15, ENS)")
    lines.append("")
    if d:
        lines.append(f"- LGB: {d['d_lgb']:+.6f}")
        lines.append(f"- XGB: {d['d_xgb']:+.6f}")
        lines.append(f"- ENS: {d['d_ens']:+.6f}")
    else:
        lines.append("- N/A (eval did not complete)")
    lines.append("")
    lines.append("## Honest verdict")
    lines.append("")
    lines.append(verdict)
    lines.append("")
    lines.append("## Coverage caveats")
    lines.append("")
    lines.append(
        "- V15 cache contains 2015-2025 only; paddock video entries are all "
        "2026 races. Direct (race_id, horse_id) merge therefore matches 0 rows."
    )
    lines.append(
        "- 12 paddock features are all NaN in the merged cache. LGB native NaN "
        "splits will not find usable signal; XGB likewise."
    )
    lines.append(
        "- Any V21 vs V15 delta observed should be treated as noise from "
        "training non-determinism, NOT as a video signal."
    )
    lines.append("")
    lines.append("## 5/31 coverage plan")
    lines.append("")
    lines.append("Target: 1,000-2,000 races with paddock features by 5/31.")
    lines.append("- Current: 33 races")
    lines.append("- Need: ~65 races/day (= ~130-180 paddock videos/day) for 15 days")
    lines.append("- Pipeline: JRA RV / netkeiba paddock fetch -> frame extract -> "
                 "yolov8/gait/body_condition -> data/video_ai_features/")
    lines.append("")
    lines.append("Phase B (post-5/31) will re-run this PoC with merged paddock features on the 2025-end / 2026-start "
                 "test fold, expecting >= 1-2% coverage of the fold for meaningful signal.")
    lines.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    sys.exit(main())
