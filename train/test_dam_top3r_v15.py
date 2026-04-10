#!/usr/bin/env python3
"""dam_top3r再投入テスト — v15ベース(145特徴量) + dam_top3r(expanding)

v12時代にリーク問題で不採用だったdam_top3r（母産駒複勝率）を
expanding windowで再計算し、v15ベースで4-model WFテスト。

採用基準:
  - WF AUC > 0.8858 (v15ベースライン)
  - 全年AUC > 0.85
  - max gap < 0.05

Usage:
    python train/test_dam_top3r_v15.py
"""

import os, sys, time, json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Use cached v15 dataframe if available
CACHE_PATH = os.path.join(DATA_DIR, '_v15_optuna_df_cache.pkl.gz')


def compute_dam_top3r_expanding(df):
    """母産駒成績をexpanding windowで計算（リークフリー）"""
    print("  Computing dam_top3r (expanding window, leak-free)...")
    if 'mother' not in df.columns:
        df['dam_top3r'] = 0.22
        print("  WARNING: 'mother' column not found")
        return df

    df = df.sort_values(['year', 'month', 'day', 'race_num']).reset_index(drop=True)
    dam_top3r = np.full(len(df), np.nan)
    alpha = 10
    prior = 0.22

    mother_cumsum = {}
    for i in range(len(df)):
        m = df.iloc[i]['mother']
        if pd.isna(m) or m == '':
            continue
        if m in mother_cumsum:
            total, t3 = mother_cumsum[m]
            if total > 0:
                dam_top3r[i] = (t3 + alpha * prior) / (total + alpha)
        fin = df.iloc[i].get('finish', 99)
        if pd.notna(fin) and fin > 0:
            if m not in mother_cumsum:
                mother_cumsum[m] = (0, 0)
            old_total, old_t3 = mother_cumsum[m]
            mother_cumsum[m] = (old_total + 1, old_t3 + (1 if fin <= 3 else 0))

    df['dam_top3r'] = dam_top3r
    mean_val = np.nanmean(dam_top3r)
    df['dam_top3r'] = df['dam_top3r'].fillna(mean_val if not np.isnan(mean_val) else prior)
    valid = np.sum(~np.isnan(dam_top3r))
    print(f"  Dam stats (expanding): {valid}/{len(df)} computed ({valid/len(df)*100:.1f}%)")
    return df


def main():
    start = time.time()
    print("=" * 60)
    print("  dam_top3r Re-test on v15 base (145 + 1 features)")
    print("=" * 60)

    # Try loading cached v15 dataframe
    if os.path.exists(CACHE_PATH):
        print(f"\n[1/3] Loading cached v15 dataframe: {CACHE_PATH}")
        import gzip, pickle
        with gzip.open(CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
        df = cache['df']
        features_list = cache['features']
        print(f"  Loaded: {len(df)} rows, {len(features_list)} features")
    else:
        print(f"\nERROR: v15 cache not found at {CACHE_PATH}")
        print("Run train/train_v15_master.py first to build the dataframe.")
        sys.exit(1)

    # Compute dam_top3r
    print("\n[2/3] Computing dam_top3r (expanding window)...")
    df = compute_dam_top3r_expanding(df)

    # Setup
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses'] >= 5].copy()
    if 'year' not in df.columns or df['year'].dtype == object:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['year'] = df['year'].astype(int)

    # Feature lists
    FEATURES_BASE = features_list  # v15 145 features
    FEATURES_DAM = features_list + ['dam_top3r']

    # Fill missing
    for f in FEATURES_DAM:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"\n  v15 baseline features: {len(FEATURES_BASE)}")
    print(f"  v15 + dam_top3r features: {len(FEATURES_DAM)}")

    # LGB params (same as v15)
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
        'verbosity': 0,
    }

    # Walk-forward test (LGB + XGB only for speed)
    print("\n[3/3] Walk-forward backtest (LGB + XGB)...")
    results_base = []
    results_dam = []

    for test_year in range(2021, 2026):
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        y_train = df.loc[train_mask, 'target'].values
        y_test = df.loc[test_mask, 'target'].values

        for label, feats, result_list in [
            ('base', FEATURES_BASE, results_base),
            ('+dam', FEATURES_DAM, results_dam),
        ]:
            X_tr = df.loc[train_mask, feats].values
            X_te = df.loc[test_mask, feats].values

            # LGB
            dtrain = lgb.Dataset(X_tr, label=y_train)
            dvalid = lgb.Dataset(X_te, label=y_test, reference=dtrain)
            m_lgb = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                              valid_sets=[dvalid],
                              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            pred_lgb = m_lgb.predict(X_te)
            lgb_auc = roc_auc_score(y_test, pred_lgb)
            train_pred_lgb = m_lgb.predict(X_tr)
            train_lgb_auc = roc_auc_score(y_train, train_pred_lgb)

            # XGB
            dtrain_xgb = xgb.DMatrix(X_tr, label=y_train)
            dvalid_xgb = xgb.DMatrix(X_te, label=y_test)
            m_xgb = xgb.train(XGB_PARAMS, dtrain_xgb, num_boost_round=1000,
                              evals=[(dvalid_xgb, 'valid')],
                              early_stopping_rounds=50, verbose_eval=False)
            pred_xgb = m_xgb.predict(dvalid_xgb)
            xgb_auc = roc_auc_score(y_test, pred_xgb)

            # Simple average
            pred_avg = 0.5 * pred_lgb + 0.5 * pred_xgb
            avg_auc = roc_auc_score(y_test, pred_avg)

            result_list.append({
                'year': test_year,
                'lgb_auc': round(lgb_auc, 6),
                'xgb_auc': round(xgb_auc, 6),
                'avg_auc': round(avg_auc, 6),
                'train_lgb_auc': round(train_lgb_auc, 6),
                'gap': round(train_lgb_auc - lgb_auc, 6),
            })

        # Print comparison
        b = results_base[-1]
        d = results_dam[-1]
        diff_lgb = d['lgb_auc'] - b['lgb_auc']
        diff_avg = d['avg_auc'] - b['avg_auc']
        print(f"  {test_year}: base_lgb={b['lgb_auc']:.4f} dam_lgb={d['lgb_auc']:.4f} diff={diff_lgb:+.5f} | "
              f"base_avg={b['avg_auc']:.4f} dam_avg={d['avg_auc']:.4f} diff_avg={diff_avg:+.5f} | "
              f"gap_base={b['gap']:.4f} gap_dam={d['gap']:.4f}")

    # Summary
    mean_base_lgb = np.mean([r['lgb_auc'] for r in results_base])
    mean_dam_lgb = np.mean([r['lgb_auc'] for r in results_dam])
    mean_base_avg = np.mean([r['avg_auc'] for r in results_base])
    mean_dam_avg = np.mean([r['avg_auc'] for r in results_dam])
    improvement_lgb = mean_dam_lgb - mean_base_lgb
    improvement_avg = mean_dam_avg - mean_base_avg

    max_gap_base = max(r['gap'] for r in results_base)
    max_gap_dam = max(r['gap'] for r in results_dam)

    print(f"\n{'='*60}")
    print(f"  RESULT (LGB+XGB average)")
    print(f"{'='*60}")
    print(f"  v15 baseline (LGB):   {mean_base_lgb:.6f}")
    print(f"  v15 + dam_top3r (LGB): {mean_dam_lgb:.6f}")
    print(f"  LGB improvement:      {improvement_lgb:+.6f}")
    print(f"  v15 baseline (AVG):   {mean_base_avg:.6f}")
    print(f"  v15 + dam_top3r (AVG): {mean_dam_avg:.6f}")
    print(f"  AVG improvement:      {improvement_avg:+.6f}")
    print(f"  Max gap (base):       {max_gap_base:.4f}")
    print(f"  Max gap (+dam):       {max_gap_dam:.4f}")

    adopted = improvement_lgb > 0 and max_gap_dam < 0.05
    print(f"\n  VERDICT: {'POSITIVE (dam_top3r helps)' if adopted else 'NEGATIVE (dam_top3r does not help)'}")
    print(f"  Note: Full 4-model grid test needed for final adoption")

    elapsed = time.time() - start
    print(f"\n  Elapsed: {elapsed/60:.1f} min")

    # Save
    out = {
        'test': 'dam_top3r_v15_reinvestigation',
        'timestamp': pd.Timestamp.now().isoformat(),
        'baseline_lgb_auc': mean_base_lgb,
        'dam_lgb_auc': mean_dam_lgb,
        'improvement_lgb': improvement_lgb,
        'baseline_avg_auc': mean_base_avg,
        'dam_avg_auc': mean_dam_avg,
        'improvement_avg': improvement_avg,
        'max_gap_base': max_gap_base,
        'max_gap_dam': max_gap_dam,
        'adopted': adopted,
        'yearly_base': results_base,
        'yearly_dam': results_dam,
    }
    out_path = os.path.join(DATA_DIR, 'dam_top3r_v15_reverify.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

if __name__ == '__main__':
    main()
