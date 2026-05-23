"""
train/train_v21_candidate.py
V21 candidate = V15 145 features + TYB 10 fields
Save as models/v21_candidate.pkl.gz
V15 production completely unchanged.

Usage:
    python train/train_v21_candidate.py
"""
import gzip
import json
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import lightgbm as lgb
import xgboost as xgb

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, 'data')
MODEL_OUT = os.path.join(REPO, 'models', 'v21_candidate.pkl.gz')
V15_MODEL_PATH = os.path.join(REPO, 'keiba_model_v15_central.pkl.gz')
MERGED_DATA_PATH = os.path.join(DATA_DIR, 'v21_tyb_merged.pkl.gz')

# TYB fields (confirmed present in v21_tyb_merged.pkl.gz)
TYB_FIELDS = [
    'tyb_tansho_odds', 'tyb_fukusho_odds', 'tyb_odds_idx',
    'tyb_jockey_idx', 'tyb_padock_idx', 'tyb_info_idx',
    'tyb_padock_mark', 'tyb_ashimoto', 'tyb_sogo_idx', 'tyb_bagu_change',
]

# LGB params (identical to V15 / CLAUDE.md)
LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'seed': 42,
}

# XGB params (identical to V15 / CLAUDE.md)
XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'min_child_weight': 50,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'seed': 42, 'tree_method': 'hist',
}

V15_BASELINE = 0.8678  # genuine WF LGB+XGB (V15-audit-2)


def load_v15_features():
    print(f'[V15] Loading features from {V15_MODEL_PATH}')
    with gzip.open(V15_MODEL_PATH, 'rb') as f:
        v15 = pickle.load(f)
    feats = v15.get('feature_names', v15.get('features', []))
    auc = v15.get('auc')
    assert abs(auc - 0.8939485520467574) < 1e-6, f'V15 auc mismatch: {auc}'
    assert len(feats) == 145, f'V15 features mismatch: {len(feats)}'
    print(f'  V15 features: {len(feats)}, auc={auc}')
    return feats, v15


def load_merged_data():
    print(f'[DATA] Loading {MERGED_DATA_PATH}')
    with gzip.open(MERGED_DATA_PATH, 'rb') as f:
        result = pickle.load(f)
    if isinstance(result, dict):
        df = result.get('df')
    else:
        df = result
    print(f'  shape: {df.shape}')
    return df


def train_one_fold(X_tr, y_tr, X_va, y_va, features):
    """Train LGB+XGB on one fold, return models + AUC."""
    # LGB
    dtr = lgb.Dataset(X_tr, y_tr, free_raw_data=False)
    dva = lgb.Dataset(X_va, y_va, free_raw_data=False)
    lgb_m = lgb.train(
        LGB_PARAMS, dtr,
        num_boost_round=500,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    p_lgb = lgb_m.predict(X_va, num_iteration=lgb_m.best_iteration)

    # XGB
    dtr_x = xgb.DMatrix(X_tr, label=y_tr)
    dva_x = xgb.DMatrix(X_va, label=y_va)
    xgb_m = xgb.train(
        XGB_PARAMS, dtr_x,
        num_boost_round=500,
        evals=[(dva_x, 'va')],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    p_xgb = xgb_m.predict(dva_x, iteration_range=(0, xgb_m.best_iteration + 1))

    auc_lgb = roc_auc_score(y_va, p_lgb)
    auc_xgb = roc_auc_score(y_va, p_xgb)
    w_lgb = auc_lgb / (auc_lgb + auc_xgb)
    w_xgb = 1 - w_lgb
    p_ens = w_lgb * p_lgb + w_xgb * p_xgb
    auc_ens = roc_auc_score(y_va, p_ens)

    return {
        'auc_lgb': auc_lgb,
        'auc_xgb': auc_xgb,
        'auc_ens': auc_ens,
        'w_lgb': w_lgb,
        'w_xgb': w_xgb,
        'lgb_model': lgb_m,
        'xgb_model': xgb_m,
    }


def main():
    print(f'=== V21 Candidate Training  {datetime.now().strftime("%Y-%m-%d %H:%M")} ===')
    print(f'    V15 production: COMPLETELY UNCHANGED')
    print()

    # [1] Load V15 features (and verify V15 model unchanged)
    v15_features, v15_obj = load_v15_features()

    # [2] Load merged data
    df = load_merged_data()

    # [3] Build V21 feature list: V15 145 + TYB 10
    v15_in_df = [f for f in v15_features if f in df.columns]
    v15_missing = [f for f in v15_features if f not in df.columns]
    tyb_in_df = [f for f in TYB_FIELDS if f in df.columns]
    tyb_missing = [f for f in TYB_FIELDS if f not in df.columns]

    print(f'[FEATURES] V15 in df: {len(v15_in_df)}/{len(v15_features)}')
    if v15_missing:
        print(f'  V15 missing from df: {v15_missing}')
    print(f'[FEATURES] TYB in df: {len(tyb_in_df)}/{len(TYB_FIELDS)}')
    if tyb_missing:
        print(f'  TYB missing: {tyb_missing}')

    v21_features = v15_in_df + tyb_in_df
    print(f'[FEATURES] V21 total: {len(v21_features)} ({len(v15_in_df)} V15 + {len(tyb_in_df)} TYB)')

    # [4] Prepare data
    # Ensure year_full
    if 'year_full' not in df.columns:
        df['year_full'] = df['year'].apply(lambda y: 2000 + int(y) if int(y) < 100 else int(y))

    # Ensure target
    target = 'target'
    if target not in df.columns:
        df[target] = (df['finish'] <= 3).astype(int)

    # Numeric coerce for all features
    for c in v21_features:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df = df.dropna(subset=[target, 'year_full'])
    df[target] = df[target].astype(int)
    df['year_full'] = df['year_full'].astype(int)

    print(f'[DATA] rows after dropna: {len(df):,}')
    print(f'[DATA] year range: {df["year_full"].min()} - {df["year_full"].max()}')
    print(f'[DATA] target rate: {df[target].mean():.3f}')

    # [5] LEAK gate: check corr of TYB fields with target
    print(f'\n[LEAK] Correlation of TYB fields with target:')
    max_corr = 0.0
    leak_fail = False
    for col in tyb_in_df:
        c = df[col].corr(df[target])
        flag = ' *** LEAK FAIL ***' if abs(c) > 0.5 else ' OK'
        print(f'  {col}: corr={c:.4f}{flag}')
        if abs(c) > max_corr:
            max_corr = abs(c)
        if abs(c) > 0.5:
            leak_fail = True

    if leak_fail:
        print('[LEAK] FAIL: at least one TYB field has |corr| > 0.5. Aborting.')
        sys.exit(1)
    print(f'[LEAK] PASS: max |corr| = {max_corr:.4f} < 0.5')

    # [6] 5-fold walk-forward CV (2021-2025)
    print(f'\n[WF] 5-fold walk-forward CV (test years: 2021-2025)...')
    fold_results = []
    final_r = None

    for test_year in [2021, 2022, 2023, 2024, 2025]:
        train_mask = df['year_full'] < test_year
        val_mask = df['year_full'] == test_year
        n_tr = train_mask.sum()
        n_va = val_mask.sum()
        if n_tr < 1000 or n_va < 100:
            print(f'  fold {test_year}: SKIP (train={n_tr}, val={n_va})')
            continue

        X_tr = df.loc[train_mask, v21_features].values
        y_tr = df.loc[train_mask, target].values
        X_va = df.loc[val_mask, v21_features].values
        y_va = df.loc[val_mask, target].values

        t0 = time.time()
        r = train_one_fold(X_tr, y_tr, X_va, y_va, v21_features)
        dt = time.time() - t0
        print(f'  fold {test_year}: LGB={r["auc_lgb"]:.4f} XGB={r["auc_xgb"]:.4f} '
              f'ENS={r["auc_ens"]:.4f} w_lgb={r["w_lgb"]:.3f} [{dt:.0f}s]')
        fold_results.append({
            'test_year': test_year,
            'auc_lgb': r['auc_lgb'],
            'auc_xgb': r['auc_xgb'],
            'auc_ens': r['auc_ens'],
            'w_lgb': r['w_lgb'],
            'n_train': int(n_tr),
            'n_val': int(n_va),
        })
        final_r = r

    if not fold_results:
        print('[ERROR] No valid folds. Aborting.')
        sys.exit(1)

    mean_auc = float(np.mean([r['auc_ens'] for r in fold_results]))
    delta = mean_auc - V15_BASELINE
    print(f'\n[WF] V21 mean ENS AUC: {mean_auc:.4f}')
    print(f'[WF] vs V15 baseline {V15_BASELINE}: delta={delta:+.4f}')

    # [7] Train final model on all data
    print(f'\n[FINAL] Training final model on all data ({len(df):,} rows)...')
    X_all = df[v21_features].values
    y_all = df[target].values

    # LGB final
    dtr_all = lgb.Dataset(X_all, y_all, free_raw_data=False)
    lgb_final = lgb.train(
        LGB_PARAMS, dtr_all,
        num_boost_round=final_r['lgb_model'].best_iteration,
        callbacks=[lgb.log_evaluation(0)],
    )

    # XGB final
    dtr_x_all = xgb.DMatrix(X_all, label=y_all)
    xgb_final = xgb.train(
        XGB_PARAMS, dtr_x_all,
        num_boost_round=final_r['xgb_model'].best_iteration + 1,
        verbose_eval=False,
    )

    # Use mean weights from last fold
    w_lgb_final = final_r['w_lgb']
    w_xgb_final = final_r['w_xgb']

    print(f'[FINAL] lgb rounds={lgb_final.num_trees()}, xgb rounds={final_r["xgb_model"].best_iteration + 1}')
    print(f'[FINAL] w_lgb={w_lgb_final:.4f}, w_xgb={w_xgb_final:.4f}')

    # [8] Save model
    model_pkg = {
        'version': 'v21_candidate',
        'date': datetime.now().isoformat(),
        'feature_names': v21_features,
        'features': v21_features,  # both keys for compat
        'n_v15_features': len(v15_in_df),
        'n_tyb_features': len(tyb_in_df),
        'tyb_fields': tyb_in_df,
        'lgb_model': lgb_final,
        'model': lgb_final,  # alias for compat with V15 structure
        'xgb_model': xgb_final,
        'w_lgb': w_lgb_final,
        'w_xgb': w_xgb_final,
        'ensemble_weights': {'lgb': w_lgb_final, 'xgb': w_xgb_final, 'mlp': 0},
        'auc': mean_auc,
        'wf_auc': mean_auc,
        'wf_fold_results': fold_results,
        'v15_baseline': V15_BASELINE,
        'delta_vs_v15': delta,
        'leak_gate': {'max_corr': float(max_corr), 'verdict': 'PASS'},
        'verdict': 'paper_only',  # NO-GO for production per ablation
        'model_type': 'lgb+xgb',
        'mlp_model': None,
    }

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with gzip.open(MODEL_OUT, 'wb') as f:
        pickle.dump(model_pkg, f)
    print(f'\n[SAVED] {MODEL_OUT}')
    sz = os.path.getsize(MODEL_OUT)
    print(f'[SAVED] size: {sz/1024/1024:.1f} MB')

    # [9] Verify V15 unchanged
    print(f'\n[CHECK] Verifying V15 unchanged...')
    with gzip.open(V15_MODEL_PATH, 'rb') as f:
        v15_check = pickle.load(f)
    v15_feats_check = v15_check.get('feature_names', v15_check.get('features', []))
    assert len(v15_feats_check) == 145, f'V15 features changed! {len(v15_feats_check)}'
    assert abs(v15_check.get('auc', 0) - 0.8939485520467574) < 1e-6, 'V15 auc changed!'
    print(f'[CHECK] V15 unchanged: PASS (145 features, auc=0.8939485520467574)')

    # [10] Summary
    print(f'\n=== V21 CANDIDATE SUMMARY ===')
    print(f'  features:    {len(v21_features)} (V15 {len(v15_in_df)} + TYB {len(tyb_in_df)})')
    print(f'  WF AUC:      {mean_auc:.4f}')
    print(f'  vs V15:      {delta:+.4f}')
    print(f'  leak gate:   PASS (max |corr| = {max_corr:.4f})')
    print(f'  verdict:     paper_only (all TYB fields showed negative delta in ablation)')
    print(f'  saved:       {MODEL_OUT}')
    print(f'  V15 status:  UNCHANGED')

    return 0


if __name__ == '__main__':
    sys.exit(main())
