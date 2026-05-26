"""V24b: V24 (130 feats) + add-back 5件 (gain>0 delete_candidate)。

V24 で削除した15件のうち、LGB gain > 0 だった 5 件を戻す:
  jrdb_ls_idx       : fill 16%, LGB gain 1053 (最大寄与)
  stable_comment_score: fill 7%,  LGB gain 209
  jrdb_prev_interference: fill 1.8%, LGB gain 198
  course_renovated  : fill 1.3%, LGB gain 92
  has_wood_training : fill 18%, LGB gain 56

期待: WF 0.8667 (V24) → 0.8678 (V15) 同等に回復
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
from train.t4_leak_audit import run_leak_audit

# V24 で削除した 15 件
ZOMBIE_GAIN0 = [
    'prev_race_last3f', 'prev_race_first3f', 'prev_race_pace_diff',
    'prev_odds_log', 'pci', 'sire_shinba_top3r', 'has_training',
    'odds_sharp_drop',
]
DELETE_CANDIDATE_ZERO = ['is_nar', 'gaisha_rank']
# ↑ の 10 件のみ削除維持 (gain=0 確定)

# add-back: gain>0 だった delete_candidate 5 件を戻す
ADDBACK_FEATURES = [
    'jrdb_ls_idx',
    'stable_comment_score',
    'jrdb_prev_interference',
    'course_renovated',
    'has_wood_training',
]

# V24b で最終的に削除するのはこの 10 件のみ
REMOVE_FINAL = sorted(set(ZOMBIE_GAIN0 + DELETE_CANDIDATE_ZERO))

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
    with gzip.open(REPO / 'models' / 'v15_full_candidate.pkl.gz', 'rb') as f:
        m = pickle.load(f)
    return m['features']


def wf_fold(df, features, test_year):
    tr = df[df['_wf_year'] < test_year].copy()
    te = df[df['_wf_year'] == test_year].copy()
    if len(tr) < 1000 or len(te) < 100:
        return None
    X_tr = tr[features].values.astype(np.float32)
    y_tr = (tr['finish'] <= 3).astype(int).values
    X_te = te[features].values.astype(np.float32)
    y_te = (te['finish'] <= 3).astype(int).values

    d_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=features)
    d_val = lgb.Dataset(X_te, label=y_te, reference=d_tr)
    cb = lgb.train(LGB_PARAMS, d_tr, num_boost_round=1000, valid_sets=[d_val],
                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    p_lgb = cb.predict(X_te)

    dm_tr = xgb.DMatrix(X_tr, label=y_tr, feature_names=features)
    dm_te = xgb.DMatrix(X_te, label=y_te, feature_names=features)
    bst = xgb.train(XGB_PARAMS, dm_tr, num_boost_round=1000,
                    evals=[(dm_te, 'val')], early_stopping_rounds=50, verbose_eval=False)
    p_xgb = bst.predict(dm_te)

    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_te, 0.5 * p_lgb + 0.5 * p_xgb))
    print(f"  fold {test_year}: AUC={auc:.6f}  (n_tr={len(tr)}, n_te={len(te)})")
    return auc, cb, bst


def main():
    print("=" * 60)
    print("V24b CANDIDATE: add-back 5 gain>0 features")
    print("=" * 60)

    # Load data
    raw = pd.read_pickle(REPO / 'data' / '_v15_train_df_cache.pkl')
    df = raw['df'] if isinstance(raw, dict) else raw
    print(f"Rows: {len(df):,}")

    # Build feature set: V15 - zero_gain_only
    v15_feats = load_v15_features()
    remove_set = set(REMOVE_FINAL)
    v24b_feats = [f for f in v15_feats if f not in remove_set]
    removed = [f for f in REMOVE_FINAL if f in v15_feats]
    print(f"V15 {len(v15_feats)} → remove {len(removed)} → V24b {len(v24b_feats)} feats")
    print(f"Removed (gain=0 only): {removed}")
    print(f"Add-back (confirmed in feats): {[f for f in ADDBACK_FEATURES if f in v24b_feats]}")

    # Missing check
    missing = [f for f in v24b_feats if f not in df.columns]
    if missing:
        print(f"WARNING: {len(missing)} missing: {missing}")
        v24b_feats = [f for f in v24b_feats if f in df.columns]
    print(f"Final: {len(v24b_feats)} feats")

    # T4 leak audit
    print("\n[T4 Leak Audit]")
    run_leak_audit(df, v24b_feats, mode='pattern_a', fail_on_error=True)

    # Prep
    df = df.copy()
    df['_wf_year'] = df['year_full'].astype(int) if 'year_full' in df.columns else \
        pd.to_datetime(df['date'].astype(str), errors='coerce').dt.year
    df = df.dropna(subset=['finish', '_wf_year'])
    df['_wf_year'] = df['_wf_year'].astype(int)
    df[v24b_feats] = df[v24b_feats].fillna(0.0)
    print(f"Rows after clean: {len(df):,} | years: {df['_wf_year'].min()}-{df['_wf_year'].max()}")

    # WF
    print("\n[Walk-Forward 2020-2025]")
    fold_aucs, last_lgb, last_xgb = [], None, None
    for yr in WF_YEARS:
        r = wf_fold(df, v24b_feats, yr)
        if r is None:
            print(f"  fold {yr}: SKIP")
            continue
        auc, cb, bst = r
        fold_aucs.append({'year': yr, 'auc': auc})
        last_lgb, last_xgb = cb, bst

    wf_mean = float(np.mean([r['auc'] for r in fold_aucs]))
    verdict = "GO" if wf_mean >= GO_THRESHOLD else "NO-GO"
    delta = wf_mean - GO_THRESHOLD

    print(f"\n{'='*60}")
    print(f"V24b WF mean: {wf_mean:.6f}")
    print(f"V24  WF mean: 0.866707 (ref)")
    print(f"V15 baseline: {GO_THRESHOLD:.4f}")
    print(f"delta vs V15: {delta:+.6f}")
    print(f"Verdict:      {verdict}")
    print(f"{'='*60}")

    # Save
    results = {
        'v15_genuine_wf': GO_THRESHOLD,
        'v24_wf': 0.866707,
        'v24b_wf': wf_mean,
        'v24b_vs_v15': delta,
        'v24b_vs_v24': wf_mean - 0.866707,
        'v24b_n_feats': len(v24b_feats),
        'removed_final': REMOVE_FINAL,
        'n_removed': len(removed),
        'addback_features': ADDBACK_FEATURES,
        'folds': fold_aucs,
        'verdict': verdict,
        'go_threshold': GO_THRESHOLD,
        'v24b_features': v24b_feats,
    }
    out = REPO / 'data' / 'v20' / 'v24b_addback_wf_results.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results: {out}")

    if verdict == "GO" and last_lgb is not None:
        mp = REPO / 'models' / 'v24b_clean_candidate.pkl.gz'
        with gzip.open(mp, 'wb') as f:
            pickle.dump({'version': 'v24b_clean', 'features': v24b_feats,
                         'lgb_model': last_lgb, 'xgb_model': last_xgb,
                         'wf_mean_auc': wf_mean, 'verdict': verdict,
                         'removed_from_v15': REMOVE_FINAL}, f)
        print(f"Model saved: {mp}")
    else:
        print(f"Model NOT saved ({verdict}).")

    print(f"\nV24b完了: 削除={len(removed)}件 (145→{len(v24b_feats)} feats)"
          f" | WF={wf_mean:.4f} (vs V15 {GO_THRESHOLD}) | {verdict} | V15不変")


if __name__ == '__main__':
    main()
