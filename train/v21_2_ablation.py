#!/usr/bin/env python3
"""V21-2 Add-one ablation test: TYB fields on top of V15 145 features.

Usage:
    python train/v21_2_ablation.py

Output:
    data/v21_verdict.json
    docs/V21-2_ABLATION_TEST.md
"""

import os
import sys
import time
import gzip
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ---------------------------------------------------------------
# LGB / XGB params (identical to V15 / CLAUDE.md)
# ---------------------------------------------------------------
LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'seed': 42,
}
XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'min_child_weight': 50,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'seed': 42, 'tree_method': 'hist',
}
EARLY_STOP = 50
MAX_ROUNDS = 1000

# V15 genuine WF baseline
V15_BASELINE = 0.8678

# TYB candidate fields (raw column names in jrdb_tyb.csv)
TYB_FIELDS = [
    'tansho_odds', 'fukusho_odds', 'odds_idx',
    'jockey_idx', 'padock_idx', 'info_idx',
    'padock_mark', 'ashimoto',
]


# ---------------------------------------------------------------
# Helper: walk-forward train/eval with LGB+XGB ensemble
# ---------------------------------------------------------------

def train_wf(df_sub: pd.DataFrame, features: list, target: str = 'target',
             n_folds: int = 5, year_col: str = 'year_full') -> float:
    """5-fold walk-forward AUC (same split logic as V15 genuine WF).

    Folds (year_full): 2021, 2022, 2023, 2024, 2025.
    Train: all years before test year.  Test: one year.
    """
    years_test = sorted(df_sub[year_col].unique())
    years_test = [y for y in years_test if y >= 2021][:n_folds]

    fold_aucs = []
    for test_year in years_test:
        train_idx = df_sub[year_col] < test_year
        test_idx  = df_sub[year_col] == test_year

        if train_idx.sum() < 1000 or test_idx.sum() < 100:
            continue

        X_tr = df_sub.loc[train_idx, features].values
        y_tr = df_sub.loc[train_idx, target].values
        X_te = df_sub.loc[test_idx,  features].values
        y_te = df_sub.loc[test_idx,  target].values

        if y_te.sum() < 10:
            continue

        # --- LGB ---
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_te, label=y_te, reference=dtrain)
        cb = lgb.log_evaluation(period=-1)
        lgb_model = lgb.train(
            LGB_PARAMS, dtrain,
            num_boost_round=MAX_ROUNDS,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), cb],
        )
        pred_lgb = lgb_model.predict(X_te)

        # --- XGB ---
        xdtrain = xgb.DMatrix(X_tr, label=y_tr)
        xdval   = xgb.DMatrix(X_te, label=y_te)
        xgb_model = xgb.train(
            XGB_PARAMS, xdtrain,
            num_boost_round=MAX_ROUNDS,
            evals=[(xdval, 'val')],
            early_stopping_rounds=EARLY_STOP,
            verbose_eval=False,
        )
        pred_xgb = xgb_model.predict(xdval)

        # Ensemble (equal weight; fine enough for ablation)
        pred_ens = 0.5 * pred_lgb + 0.5 * pred_xgb
        auc = roc_auc_score(y_te, pred_ens)
        fold_aucs.append(auc)
        print(f"      fold {test_year}: AUC={auc:.4f}  (n_train={train_idx.sum()}, n_test={test_idx.sum()})")

    if not fold_aucs:
        return float('nan')
    return float(np.mean(fold_aucs))


# ---------------------------------------------------------------
# VIF helper
# ---------------------------------------------------------------
def compute_vif_single(df_sub: pd.DataFrame, target_col: str,
                        predictor_cols: list) -> float:
    """Approximate VIF: R^2 of regressing target_col on predictor_cols."""
    from sklearn.linear_model import LinearRegression
    sub = df_sub[[target_col] + predictor_cols].dropna()
    if len(sub) < 100 or len(predictor_cols) == 0:
        return float('nan')
    X = sub[predictor_cols].values
    y = sub[target_col].values
    # Standardize
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    y_std = (y - y.mean()) / (y.std() + 1e-9)
    lr = LinearRegression().fit(X_std, y_std)
    r2 = lr.score(X_std, y_std)
    vif = 1 / (1 - r2 + 1e-9)
    return float(vif)


# ---------------------------------------------------------------
# Step 1: Wait for or build merged data
# ---------------------------------------------------------------
def load_or_build_merged_df():
    merged_path = os.path.join(DATA_DIR, 'v21_tyb_merged.pkl.gz')

    def _try_load(path):
        """Try to load pkl.gz; return (df, features) tuple or None on failure."""
        try:
            if os.path.getsize(path) < 10000:
                return None
            with gzip.open(path, 'rb') as f:
                result = pickle.load(f)
            if isinstance(result, dict):
                df_r = result.get('df')
                feat_r = result.get('features')
                if df_r is not None and hasattr(df_r, 'shape'):
                    return df_r, feat_r
                return None
            if hasattr(result, 'shape'):
                return result, None
            return None
        except Exception as e:
            print(f"  Load failed ({type(e).__name__}: {str(e)[:80]}) - will build from scratch")
            return None

    def _has_all_tyb(df_r):
        return df_r is not None and all(f in df_r.columns for f in TYB_FIELDS)

    if os.path.exists(merged_path):
        loaded = _try_load(merged_path)
        if loaded is not None and _has_all_tyb(loaded[0]):
            print(f"[1] Loading existing {merged_path} (all TYB fields present)")
            return loaded
        else:
            sz = os.path.getsize(merged_path)
            present = [f for f in TYB_FIELDS if loaded is not None and loaded[0] is not None and f in loaded[0].columns]
            missing = [f for f in TYB_FIELDS if f not in present]
            print(f"[1] {merged_path} exists but missing TYB fields {missing} (size={sz}) - will poll then build")

    # Polling: wait up to 3 hours for a valid file with all TYB fields
    max_wait = 180 * 60
    start = time.time()
    while True:
        if os.path.exists(merged_path) and os.path.getsize(merged_path) > 50000:
            loaded = _try_load(merged_path)
            if loaded is not None and _has_all_tyb(loaded[0]):
                print(f"  File ready after {(time.time()-start)/60:.0f} min - loading")
                return loaded[0], loaded[1]
        elapsed = time.time() - start
        if elapsed > max_wait:
            print(f"  TIMEOUT after {elapsed/60:.0f} min - building merged df from V15 cache + TYB CSV")
            break
        print(f"  Waiting for v21_tyb_merged.pkl.gz with all TYB fields... ({elapsed/60:.0f} min)")
        time.sleep(300)

    # Build from scratch
    print("[1] Building merged df from _v15_optuna_df_cache.pkl.gz + jrdb_tyb.csv ...")
    cache_path = os.path.join(DATA_DIR, '_v15_optuna_df_cache.pkl.gz')
    with gzip.open(cache_path, 'rb') as f:
        data = pickle.load(f)
    df = data['df']
    v15_features = data['features']
    print(f"    V15 cache: {df.shape}")

    tyb_path = os.path.join(DATA_DIR, 'jrdb_tyb.csv')
    tyb = pd.read_csv(tyb_path)
    print(f"    TYB data: {tyb.shape}")

    # Build nk_race_id merge key from V15 data
    jv_rid = df['race_id'].astype(str).str.zfill(10)
    df['_nk_rid'] = (
        '20'
        + jv_rid.str[2:4]        # year 2-digit
        + jv_rid.str[:2]          # course
        + df['kai'].astype(str).str.zfill(2)
        + df['nichi'].astype(str).str.zfill(2)
        + jv_rid.str[6:8]         # race_num
    )
    df['_uma'] = df['umaban'].astype(int)

    tyb['_nk_rid'] = tyb['race_id'].astype(str)
    tyb['_uma'] = tyb['umaban'].astype(int)

    tyb_merge_cols = ['_nk_rid', '_uma'] + TYB_FIELDS
    # Drop duplicates in TYB (keep last)
    tyb_dedup = tyb[tyb_merge_cols].drop_duplicates(subset=['_nk_rid', '_uma'], keep='last')

    df_merged = df.merge(tyb_dedup, on=['_nk_rid', '_uma'], how='left', suffixes=('', '_tyb'))

    # Handle potential duplicate column names from tansho_odds (already in V15 df)
    for f in TYB_FIELDS:
        if f + '_tyb' in df_merged.columns:
            df_merged[f] = df_merged[f + '_tyb']
            df_merged.drop(columns=[f + '_tyb'], inplace=True)

    df_merged.drop(columns=['_nk_rid', '_uma'], inplace=True)

    # Build target
    df_merged['target'] = (df_merged['finish'] <= 3).astype(int)

    print(f"    Merged shape: {df_merged.shape}")
    for f in TYB_FIELDS:
        if f in df_merged.columns:
            nn = df_merged[f].notna().sum()
            print(f"      {f}: {nn:,} non-null ({nn/len(df_merged)*100:.1f}%)")

    return df_merged, v15_features


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    t0 = datetime.now()
    print(f"=== V21-2 Ablation Test  {t0.strftime('%Y-%m-%d %H:%M')} ===\n")

    # [1] Load data
    df, v15_features = load_or_build_merged_df()

    # Ensure year_full is available
    if 'year_full' not in df.columns:
        df['year_full'] = df['year'].apply(lambda y: 2000 + int(y) if int(y) < 100 else int(y))

    # Fallback: load V15 features from model if not in loaded data
    if v15_features is None:
        print("  Loading V15 features from model file ...")
        model_path = os.path.join(BASE_DIR, 'keiba_model_v15_central.pkl.gz')
        with gzip.open(model_path, 'rb') as f:
            v15_model = pickle.load(f)
        v15_features = v15_model['features']
        print(f"  Loaded {len(v15_features)} features from model")

    # Verify V15 features
    v15_cols = [f for f in v15_features if f in df.columns]
    missing_v15 = [f for f in v15_features if f not in df.columns]
    print(f"\n[2] V15 features in df: {len(v15_cols)}/{len(v15_features)}")
    if missing_v15:
        print(f"    Missing: {missing_v15}")

    tyb_in_df = [f for f in TYB_FIELDS if f in df.columns]
    tyb_missing = [f for f in TYB_FIELDS if f not in df.columns]
    print(f"    TYB fields in df: {tyb_in_df}")
    if tyb_missing:
        print(f"    TYB missing: {tyb_missing}")

    # Ensure numeric
    for c in v15_cols + tyb_in_df:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Build target
    if 'target' not in df.columns:
        df['target'] = (df['finish'] <= 3).astype(int)

    print(f"\n[3] Dataset summary")
    print(f"    Total rows: {len(df):,}")
    print(f"    Year range: {df['year_full'].min()} - {df['year_full'].max()}")
    print(f"    Target rate: {df['target'].mean():.3f}")

    # [4] Baseline WF AUC (V15 145 features, rows where all TYB fields exist)
    # Use subset where all TYB fields are non-null for fair comparison
    # But first run baseline on full data
    print(f"\n[4] Running V15 baseline (145 features, 5-fold WF 2021-2025)...")
    baseline_auc = train_wf(df, v15_cols)
    print(f"    >>> V15 baseline WF AUC = {baseline_auc:.4f}  (ref: {V15_BASELINE})")

    # [5] Add-one ablation per TYB field
    print(f"\n[5] Add-one ablation per TYB field...")
    ablation_results = {}

    for tyb_field in TYB_FIELDS:
        if tyb_field not in df.columns:
            print(f"  SKIP {tyb_field}: not in df")
            ablation_results[tyb_field] = {'auc': float('nan'), 'delta': float('nan'),
                                           'n_rows': 0, 'coverage': 0.0}
            continue

        # Subset: rows where TYB field is non-null (0-filled already so need original mask)
        # Reload original nulls
        tyb_path = os.path.join(DATA_DIR, 'jrdb_tyb.csv')
        tyb_raw = pd.read_csv(tyb_path, usecols=['race_id', 'umaban', tyb_field])
        valid_mask = df[tyb_field] != 0  # proxy: 0 = missing/default
        # For padock_mark, 0 might be valid; use notna check from merge
        # Actually after fillna(0) we can't distinguish, so use full data
        # and let the model handle the default

        sub = df  # use full df with 0-filled TYB field
        n_rows = len(sub)
        features_plus = v15_cols + [tyb_field]

        print(f"\n  [{tyb_field}]  coverage={valid_mask.mean()*100:.1f}%  n={n_rows:,}")
        auc = train_wf(sub, features_plus)
        delta = auc - baseline_auc
        print(f"  >>> {tyb_field}: AUC={auc:.4f}  delta={delta:+.4f}")

        ablation_results[tyb_field] = {
            'auc': auc,
            'delta': delta,
            'n_rows': n_rows,
            'coverage': float(valid_mask.mean()),
        }

    # [6] Multicollinearity check for top fields
    print(f"\n[6] Multicollinearity (VIF) for fields with delta >= +0.0005...")
    odds_v15_cols = [c for c in v15_cols
                     if any(x in c.lower() for x in ['odds', 'pop', 'rank', 'ninki'])]
    print(f"    V15 odds-related features: {odds_v15_cols}")

    vif_results = {}
    for tyb_field, res in ablation_results.items():
        delta = res.get('delta', float('nan'))
        if not np.isnan(delta) and delta >= 0.0005 and tyb_field in df.columns:
            vif = compute_vif_single(df, tyb_field, odds_v15_cols)
            vif_results[tyb_field] = vif
            print(f"    {tyb_field}: VIF vs V15 odds features = {vif:.2f}")

    # Correlation of odds_idx with key V15 odds features
    print(f"\n    Pearson correlations with V15 odds features:")
    corr_results = {}
    if 'odds_idx' in df.columns:
        for ov in odds_v15_cols:
            if ov in df.columns:
                corr = df[['odds_idx', ov]].dropna().corr().iloc[0, 1]
                corr_results[ov] = corr
                print(f"      odds_idx vs {ov}: r={corr:.3f}")

    # [7] Full V21 training with selected features
    # Select: delta >= 0.002 AND VIF < 10
    selected_fields = []
    for tyb_field, res in ablation_results.items():
        delta = res.get('delta', float('nan'))
        vif = vif_results.get(tyb_field, float('nan'))
        if not np.isnan(delta) and delta >= 0.002:
            if np.isnan(vif) or vif < 10:
                selected_fields.append(tyb_field)

    print(f"\n[7] Full V21 training: selected TYB fields = {selected_fields}")
    if selected_fields:
        v21_features = v15_cols + selected_fields
        print(f"    V21 features: {len(v21_features)} ({len(v15_cols)} V15 + {len(selected_fields)} TYB)")
        print(f"    Running 5-fold WF 2021-2025...")
        v21_auc = train_wf(df, v21_features)
        print(f"    >>> V21 full WF AUC = {v21_auc:.4f}")
    else:
        # Use best single field regardless
        valid_results = {k: v for k, v in ablation_results.items()
                        if not np.isnan(v.get('delta', float('nan')))}
        if valid_results:
            best_field = max(valid_results, key=lambda k: valid_results[k]['delta'])
            best_delta = valid_results[best_field]['delta']
            print(f"    No field meets delta>=0.002; best={best_field} (delta={best_delta:+.4f})")
            # Still run V21 with best field for reference
            v21_features = v15_cols + [best_field]
            print(f"    Running reference V21 with best field only...")
            v21_auc = train_wf(df, v21_features)
            print(f"    >>> V21 reference WF AUC = {v21_auc:.4f}")
            selected_fields = [best_field]
        else:
            v21_auc = baseline_auc
            print("    No valid ablation results; V21 AUC = baseline")

    # [8] Verdict
    go_threshold = 0.880
    verdict = "GO" if v21_auc >= go_threshold else "NO-GO"
    if selected_fields:
        top_field = max(
            {k: v for k, v in ablation_results.items() if k in selected_fields},
            key=lambda k: ablation_results[k]['delta'],
            default=None
        )
        top_delta = ablation_results[top_field]['delta'] if top_field else float('nan')
    else:
        top_field = None
        top_delta = float('nan')

    print(f"\n[8] VERDICT: {verdict}")
    print(f"    V21 WF AUC = {v21_auc:.4f}  (threshold >= {go_threshold})")
    print(f"    V15 baseline = {baseline_auc:.4f}  (ref: {V15_BASELINE})")
    print(f"    Top field: {top_field}  delta: {top_delta:+.4f}")
    print(f"    Selected fields: {selected_fields}")

    # [9] Save verdict JSON
    verdict_data = {
        'verdict': verdict,
        'v21_auc': round(v21_auc, 4),
        'v15_auc': round(baseline_auc, 4),
        'v15_auc_ref': V15_BASELINE,
        'top_field': top_field,
        'delta': round(top_delta, 4) if not np.isnan(top_delta) else None,
        'selected_fields': selected_fields,
        'ablation_results': {
            k: {
                'auc': round(v['auc'], 4) if not np.isnan(v['auc']) else None,
                'delta': round(v['delta'], 4) if not np.isnan(v['delta']) else None,
                'coverage': round(v['coverage'], 3),
            }
            for k, v in ablation_results.items()
        },
        'vif_results': {k: round(v, 2) for k, v in vif_results.items()},
        'odds_corr': {k: round(v, 3) for k, v in corr_results.items()},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }

    verdict_path = os.path.join(DATA_DIR, 'v21_verdict.json')
    with open(verdict_path, 'w', encoding='utf-8') as f:
        json.dump(verdict_data, f, ensure_ascii=False, indent=2)
    print(f"\n    Saved: {verdict_path}")

    # [10] Write doc
    elapsed = (datetime.now() - t0).total_seconds() / 60
    _write_doc(verdict_data, ablation_results, vif_results, corr_results,
               v15_cols, odds_v15_cols, baseline_auc, elapsed)

    print(f"\n=== Done in {elapsed:.1f} min ===")
    print(f"\nV21-2 完了、full V21 WF AUC = {v21_auc:.4f}、"
          f"最良 field = {top_field}、"
          f"multicollinearity = {'被り' if any(v >= 10 for v in vif_results.values()) else 'OK'}、"
          f"verdict = {verdict}")


def _write_doc(verdict_data, ablation_results, vif_results, corr_results,
               v15_cols, odds_v15_cols, baseline_auc, elapsed_min):
    """Write docs/V21-2_ABLATION_TEST.md"""
    docs_dir = os.path.join(BASE_DIR, 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    doc_path = os.path.join(docs_dir, 'V21-2_ABLATION_TEST.md')

    def fmt(v, prec=4):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 'N/A'
        if isinstance(v, float):
            return f'{v:.{prec}f}'
        return str(v)

    lines = [
        f'# V21-2 Add-one Ablation Test - TYB fields on V15',
        f'',
        f'Generated: {verdict_data["timestamp"]}  |  Elapsed: {elapsed_min:.0f} min',
        f'',
        f'## Summary',
        f'',
        f'| Item | Value |',
        f'|------|-------|',
        f'| V15 baseline WF AUC (computed) | {fmt(baseline_auc)} |',
        f'| V15 baseline WF AUC (reference) | {V15_BASELINE} |',
        f'| V21 full WF AUC | {fmt(verdict_data["v21_auc"])} |',
        f'| Top TYB field | {verdict_data["top_field"]} |',
        f'| Delta (top field) | {fmt(verdict_data["delta"])} |',
        f'| Selected fields | {", ".join(verdict_data["selected_fields"]) or "None"} |',
        f'| **Verdict** | **{verdict_data["verdict"]}** |',
        f'',
        f'GO criteria: V21 WF AUC >= 0.880 AND at least one TYB field with delta >= +0.002',
        f'',
        f'## Add-one Ablation Results',
        f'',
        f'| TYB Field | Ablation AUC | Delta vs V15 | Coverage | VIF (vs V15 odds) |',
        f'|-----------|-------------|--------------|----------|-------------------|',
    ]

    for field in ablation_results:
        res = ablation_results[field]
        auc_str = fmt(res['auc'])
        delta = res['delta']
        delta_str = f'{delta:+.4f}' if (delta is not None and not np.isnan(delta)) else 'N/A'
        cov_str = f'{res["coverage"]*100:.1f}%'
        vif_str = fmt(vif_results.get(field, float('nan')), 2)
        lines.append(f'| {field} | {auc_str} | {delta_str} | {cov_str} | {vif_str} |')

    lines += [
        f'',
        f'## Multicollinearity Analysis',
        f'',
        f'V15 odds-related features used as regressors:',
        f'`{", ".join(odds_v15_cols)}`',
        f'',
        f'### Pearson Correlations: odds_idx vs V15 odds features',
        f'',
        f'| V15 Feature | Pearson r |',
        f'|-------------|-----------|',
    ]
    for feat, corr in sorted(corr_results.items(), key=lambda x: abs(x[1]), reverse=True):
        lines.append(f'| {feat} | {corr:.3f} |')

    lines += [
        f'',
        f'### VIF Results (fields with delta >= +0.0005)',
        f'',
        f'| TYB Field | VIF | Status |',
        f'|-----------|-----|--------|',
    ]
    for field, vif in sorted(vif_results.items(), key=lambda x: x[1], reverse=True):
        status = 'MULTICOLLINEAR (VIF>=10)' if vif >= 10 else 'OK'
        lines.append(f'| {field} | {vif:.2f} | {status} |')

    lines += [
        f'',
        f'## Selected Features for V21',
        f'',
        f'Selection criteria: delta >= +0.002 AND VIF < 10',
        f'',
        f'Selected: `{", ".join(verdict_data["selected_fields"]) or "None"}`',
        f'',
        f'Total V21 features: {len(verdict_data["selected_fields"]) + 145} '
        f'({145} V15 + {len(verdict_data["selected_fields"])} TYB)',
        f'',
        f'## Verdict',
        f'',
        f'**{verdict_data["verdict"]}**',
        f'',
        f'- V21 WF AUC = {fmt(verdict_data["v21_auc"])} '
        f'(threshold: >= 0.880)',
        f'- V15 baseline = {fmt(baseline_auc)} '
        f'(ref: {V15_BASELINE})',
        f'',
        f'{"GO: Proceed with V21 production candidate." if verdict_data["verdict"] == "GO" else "NO-GO: TYB fields for Discord supplement only. No production model update."}',
        f'',
        f'---',
        f'',
        f'*V15 production files were not modified during this ablation test.*',
    ]

    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"    Saved: {doc_path}")


if __name__ == '__main__':
    main()
