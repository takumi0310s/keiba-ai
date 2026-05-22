"""
V21-2 TYB Add-One Ablation Test
Baseline: V15 genuine WF AUC = 0.8678 (6-fold LGB+XGB)
"""
import pandas as pd
import numpy as np
import pickle
import gzip
import json
import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb_lib
import warnings
warnings.filterwarnings('ignore')

print("Loading data...", flush=True)
df = pd.read_pickle('data/v21_tyb_merged.pkl.gz')
print(f"Shape: {df.shape}", flush=True)

with gzip.open('keiba_model_v15_central.pkl.gz', 'rb') as f:
    v15 = pickle.load(f)
v15_features = v15['features']
print(f"V15 features: {len(v15_features)}", flush=True)

tyb_cols = [c for c in df.columns if c.startswith('tyb_')]
print(f"TYB cols: {tyb_cols}", flush=True)

# Target
target = 'target'
print(f"Target: {target}, pos rate={df[target].mean():.3f}", flush=True)

# V15 features all in df (confirmed 145/145)
v15_feats = [f for f in v15_features if f in df.columns]
print(f"V15 feats in df: {len(v15_feats)}", flush=True)

# V15 LGB params from CLAUDE.md
params = {
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

BASELINE = 0.8678
NUM_BOOST = 300  # reduced for speed

# ── Step 3: Add-one ablation (5-fold CV) ──────────────────────────────────────
print("\n=== Step 3: Add-one ablation (5-fold CV) ===", flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for tyb_col in tyb_cols:
    feats = v15_feats + [tyb_col]
    X = df[feats].fillna(0).values
    y = df[target].values

    aucs = []
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        ds_tr = lgb_lib.Dataset(X_tr, y_tr, feature_name=feats)
        ds_val = lgb_lib.Dataset(X_val, y_val, feature_name=feats, reference=ds_tr)

        model = lgb_lib.train(
            params, ds_tr, num_boost_round=NUM_BOOST,
            valid_sets=[ds_val],
            callbacks=[
                lgb_lib.early_stopping(50, verbose=False),
                lgb_lib.log_evaluation(-1)
            ]
        )
        pred = model.predict(X_val)
        aucs.append(roc_auc_score(y_val, pred))

    mean_auc = float(np.mean(aucs))
    delta = mean_auc - BASELINE
    results[tyb_col] = {
        'mean_auc': mean_auc,
        'delta': delta,
        'fold_aucs': [round(a, 6) for a in aucs]
    }
    print(f"  {tyb_col}: {mean_auc:.4f} (delta={delta:+.4f}) folds={[f'{a:.4f}' for a in aucs]}", flush=True)

print("\nAblation results:", flush=True)
for k, v in sorted(results.items(), key=lambda x: -x[1]['delta']):
    print(f"  {k}: {v['mean_auc']:.4f} (delta={v['delta']:+.4f})", flush=True)

# GO candidates: delta > +0.001
go_candidates = {k: v for k, v in results.items() if v['delta'] > 0.001}
print(f"\nGO candidates (delta > +0.001): {list(go_candidates.keys())}", flush=True)

# ── Step 4: Full V21 WF (walk-forward) with GO candidates ─────────────────────
full_v21_auc = None
wf_fold_aucs = []

if go_candidates:
    print("\n=== Step 4: Full V21 WF (6-fold walk-forward) ===", flush=True)
    go_feats = list(go_candidates.keys())
    full_feats = v15_feats + go_feats
    print(f"Features: {len(full_feats)} (V15 {len(v15_feats)} + TYB {len(go_feats)})", flush=True)

    # year column
    if 'year' in df.columns:
        years = df['year'].values
    else:
        # extract from race_date
        years = pd.to_datetime(df['race_date']).dt.year.values

    unique_years = sorted(df['year'].unique()) if 'year' in df.columns else sorted(np.unique(years))
    print(f"Years in df: {unique_years}", flush=True)

    # 6-fold WF: fold i trains on years < eval_year, evals on eval_year
    # Use 2020-2025
    eval_years = [y for y in unique_years if 2020 <= y <= 2025]
    print(f"Eval years: {eval_years}", flush=True)

    X_all = df[full_feats].fillna(0).values
    y_all = df[target].values

    for eval_year in eval_years:
        train_mask = years < eval_year
        val_mask = years == eval_year
        n_train = train_mask.sum()
        n_val = val_mask.sum()
        if n_train < 1000 or n_val < 100:
            print(f"  Year {eval_year}: skip (train={n_train}, val={n_val})", flush=True)
            continue

        X_tr = X_all[train_mask]
        y_tr = y_all[train_mask]
        X_val = X_all[val_mask]
        y_val = y_all[val_mask]

        ds_tr = lgb_lib.Dataset(X_tr, y_tr)
        ds_val_d = lgb_lib.Dataset(X_val, y_val, reference=ds_tr)

        model = lgb_lib.train(
            params, ds_tr, num_boost_round=NUM_BOOST,
            valid_sets=[ds_val_d],
            callbacks=[
                lgb_lib.early_stopping(50, verbose=False),
                lgb_lib.log_evaluation(-1)
            ]
        )
        pred = model.predict(X_val)
        fold_auc = float(roc_auc_score(y_val, pred))
        wf_fold_aucs.append(fold_auc)
        print(f"  WF fold {eval_year}: AUC={fold_auc:.4f} (train={n_train}, val={n_val})", flush=True)

    if wf_fold_aucs:
        full_v21_auc = float(np.mean(wf_fold_aucs))
        print(f"\nFull V21 WF mean AUC: {full_v21_auc:.4f} (baseline={BASELINE}, threshold=0.880)", flush=True)
    else:
        print("No valid WF folds.", flush=True)
        full_v21_auc = None
else:
    print("\nNo GO candidates — skipping WF step.", flush=True)
    full_v21_auc = None

# ── Step 5: VIF check for tyb_odds_idx ────────────────────────────────────────
print("\n=== Step 5: Correlation check for tyb_odds_idx ===", flush=True)
odds_related = [f for f in v15_feats if 'odds' in f.lower() or 'pop' in f.lower() or 'rank' in f.lower()]
print(f"Odds-related V15 features: {odds_related}", flush=True)
if 'tyb_odds_idx' in df.columns and odds_related:
    corrs = df[['tyb_odds_idx'] + odds_related].corr()['tyb_odds_idx'].drop('tyb_odds_idx')
    print(f"Correlations with tyb_odds_idx:\n{corrs.to_string()}", flush=True)

# ── Step 6: Save verdict ────────────────────────────────────────────────────────
verdict_val = None
if full_v21_auc is not None:
    verdict_val = 'GO' if full_v21_auc >= 0.880 else 'NO-GO'
elif not go_candidates:
    verdict_val = 'NO-GO (no GO candidates)'

verdict = {
    'ablation': results,
    'go_candidates': list(go_candidates.keys()),
    'full_v21_wf_fold_aucs': wf_fold_aucs,
    'full_v21_wf_auc': full_v21_auc,
    'verdict': verdict_val,
    'baseline_auc': BASELINE,
    'threshold': 0.880,
    'num_boost_round': NUM_BOOST,
}

with open('data/v21_verdict.json', 'w') as f:
    json.dump(verdict, f, indent=2)
print(f"\nVERDICT: {verdict_val}", flush=True)
print("Saved: data/v21_verdict.json", flush=True)
print("Done.", flush=True)
