"""V24 candidate: V15 (145 feats) - zombie7 - delete_candidate8 = 130 feats。

目的:
  - LGB/XGB両方でgain=0のゾンビ特徴量を削除
  - fill<20%かつgain低のdelete_candidate削除
  - 後でprev_race_*_real版を入れたときLGBが新規信号として認識できる土台作り

V15 production 完全不変。candidate専用。
"""
from __future__ import annotations
import gzip
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# T4 leak audit
from train.t4_leak_audit import run_leak_audit

# ============================================================
# 削除特徴量 (union of zombie + delete_candidate)
# ============================================================
ZOMBIE_GAIN0 = [
    'prev_race_last3f', 'prev_race_first3f', 'prev_race_pace_diff',
    'prev_odds_log', 'pci', 'sire_shinba_top3r', 'has_training',
    'odds_sharp_drop',
]
DELETE_CANDIDATE = [
    'is_nar', 'gaisha_rank', 'course_renovated',
    'jrdb_prev_interference', 'stable_comment_score',
    'has_wood_training', 'jrdb_ls_idx',
]
REMOVE_FEATURES = sorted(set(ZOMBIE_GAIN0 + DELETE_CANDIDATE))

# WF settings
WF_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
GO_THRESHOLD = 0.8678

LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'seed': 42,
}
XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 6,
    'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'seed': 42, 'tree_method': 'hist', 'verbosity': 0,
}


def load_v15_features() -> list[str]:
    model_path = REPO / 'models' / 'v15_full_candidate.pkl.gz'
    with gzip.open(model_path, 'rb') as f:
        m = pickle.load(f)
    return m['features']


def build_v24_features(v15_feats: list[str]) -> list[str]:
    remove_set = set(REMOVE_FEATURES)
    v24 = [f for f in v15_feats if f not in remove_set]
    removed_found = [f for f in REMOVE_FEATURES if f in v15_feats]
    print(f"V15 feats: {len(v15_feats)} → remove: {len(removed_found)} → V24: {len(v24)}")
    print(f"Removed: {removed_found}")
    return v24


def wf_fold(df: pd.DataFrame, features: list[str], test_year: int, year_col: str = 'year'):
    train_df = df[df[year_col] < test_year].copy()
    test_df = df[df[year_col] == test_year].copy()
    if len(train_df) < 1000 or len(test_df) < 100:
        return None

    X_tr = train_df[features].values.astype(np.float32)
    y_tr = (train_df['finish'] <= 3).astype(int).values
    X_te = test_df[features].values.astype(np.float32)
    y_te = (test_df['finish'] <= 3).astype(int).values

    # LGB
    d_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=features)
    d_val = lgb.Dataset(X_te, label=y_te, reference=d_tr)
    cb = lgb.train(
        LGB_PARAMS, d_tr,
        num_boost_round=1000,
        valid_sets=[d_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    p_lgb = cb.predict(X_te)

    # XGB
    dm_tr = xgb.DMatrix(X_tr, label=y_tr, feature_names=features)
    dm_te = xgb.DMatrix(X_te, label=y_te, feature_names=features)
    bst = xgb.train(
        XGB_PARAMS, dm_tr,
        num_boost_round=1000,
        evals=[(dm_te, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    p_xgb = bst.predict(dm_te)

    score = 0.5 * p_lgb + 0.5 * p_xgb
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_te, score))
    print(f"  fold {test_year}: AUC={auc:.6f}  (n_train={len(train_df)}, n_test={len(test_df)})")
    return auc, cb, bst


def main():
    print("=" * 60)
    print("V24 CANDIDATE TRAINING")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading training cache...")
    cache_path = REPO / 'data' / '_v15_train_df_cache.pkl'
    raw = pd.read_pickle(cache_path)
    if isinstance(raw, dict):
        df = raw['df']
    else:
        df = raw
    print(f"  Rows: {len(df):,} / Cols: {df.shape[1]}")

    # 2. Build feature set
    print("\n[2] Building V24 feature set...")
    v15_feats = load_v15_features()
    v24_feats = build_v24_features(v15_feats)
    missing = [f for f in v24_feats if f not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} features missing from df: {missing}")
        v24_feats = [f for f in v24_feats if f in df.columns]
    print(f"  Final V24 feature count: {len(v24_feats)}")

    # 3. T4 leak audit
    print("\n[3] T4 Leak Audit...")
    run_leak_audit(df, v24_feats, mode='pattern_a', fail_on_error=True)

    # 4. Prepare df
    if 'year_full' in df.columns:
        df = df.copy()
        df['_wf_year'] = df['year_full'].astype(int)
    elif 'date_num' in df.columns:
        df = df.copy()
        df['_wf_year'] = (df['date_num'].astype(str).str[:4]).astype(int)
    elif 'date' in df.columns:
        df = df.copy()
        df['_wf_year'] = pd.to_datetime(df['date'].astype(str), errors='coerce').dt.year
    else:
        # year col is 2-digit (15 = 2015)
        df = df.copy()
        df['_wf_year'] = df['year'].astype(int) + 2000
    df = df.dropna(subset=['finish', '_wf_year'])
    df['_wf_year'] = df['_wf_year'].astype(int)
    df[v24_feats] = df[v24_feats].fillna(0.0)
    print(f"  Training rows after clean: {len(df):,}")
    print(f"  Year range: {df['_wf_year'].min()} - {df['_wf_year'].max()}")

    # 5. WF
    print("\n[4] Walk-Forward (2020-2025)...")
    fold_aucs = []
    last_lgb = None
    last_xgb = None
    for yr in WF_YEARS:
        result = wf_fold(df, v24_feats, yr, year_col='_wf_year')
        if result is None:
            print(f"  fold {yr}: SKIP (insufficient data)")
            continue
        auc, cb, bst = result
        fold_aucs.append({'year': yr, 'auc': auc})
        last_lgb = cb
        last_xgb = bst

    wf_mean = float(np.mean([r['auc'] for r in fold_aucs]))
    verdict = "GO" if wf_mean >= GO_THRESHOLD else "NO-GO"
    delta = wf_mean - GO_THRESHOLD

    print(f"\n{'='*60}")
    print(f"V24 WF mean AUC: {wf_mean:.6f}")
    print(f"V15 baseline:    {GO_THRESHOLD:.4f}")
    print(f"delta:           {delta:+.6f}")
    print(f"Verdict:         {verdict}")
    print(f"{'='*60}")

    # 6. Save results
    results = {
        'v15_genuine_wf': GO_THRESHOLD,
        'v24_wf': wf_mean,
        'v24_vs_v15': delta,
        'v24_n_feats': len(v24_feats),
        'v15_n_feats': len(v15_feats),
        'removed': REMOVE_FEATURES,
        'n_removed': len([f for f in REMOVE_FEATURES if f in v15_feats]),
        'folds': fold_aucs,
        'verdict': verdict,
        'go_threshold': GO_THRESHOLD,
        'v24_features': v24_feats,
    }
    out_json = REPO / 'data' / 'v20' / 'v24_clean_wf_results.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {out_json}")

    # 7. Save model only if GO
    if verdict == "GO" and last_lgb is not None:
        model_out = REPO / 'models' / 'v24_clean_candidate.pkl.gz'
        payload = {
            'version': 'v24_clean',
            'features': v24_feats,
            'lgb_model': last_lgb,
            'xgb_model': last_xgb,
            'wf_mean_auc': wf_mean,
            'verdict': verdict,
            'removed_from_v15': REMOVE_FEATURES,
        }
        with gzip.open(model_out, 'wb') as f:
            pickle.dump(payload, f)
        print(f"Model saved: {model_out}")
    else:
        print(f"Model NOT saved ({verdict}). results JSON only.")

    print(f"\nV24 clean完了、削除={len([f for f in REMOVE_FEATURES if f in v15_feats])}件"
          f" (145→{len(v24_feats)} feats)、WF={wf_mean:.4f}"
          f" (vs V15 {GO_THRESHOLD})、verdict={verdict}、V15不変")


if __name__ == '__main__':
    main()
