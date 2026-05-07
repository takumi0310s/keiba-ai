"""V15.1 ablation: SKB only / SRB only / KKA only / All で比較.

V15.1 +697bp は SRB (race-level corner/bias) 由来の post-race leak の疑い。
切り分け: SKB-only / SRB-only / KKA-only / All で AUC 比較。
SKB-only が clean な改善、 SRB / KKA が0coverage または leak と判明する想定。

usage:
  python train/run_v15_1_ablation.py --max-rows 200000 --wf-years 2025
"""
from __future__ import annotations
import os, sys, argparse, pickle, gzip, json, time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

BASE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE)
sys.path.insert(0, os.path.join(BASE, 'train'))
from v15_1_features import (merge_v15_1_features,
                             V15_1_KKA_FEATURES, V15_1_SKB_FEATURES, V15_1_SRB_FEATURES)

CACHE_PATH = 'data/_v15_optuna_df_cache.pkl.gz'
OUT_DIR = 'data/v15.1'

LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc',
    'boosting_type': 'gbdt', 'num_leaves': 63, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_child_samples': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'verbose': -1, 'seed': 42,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-rows', type=int, default=200000)
    parser.add_argument('--wf-years', nargs='+', type=int, default=[2025])
    args = parser.parse_args()

    print("=" * 60)
    print("V15.1 ablation: SKB / SRB / KKA 切り分け")
    print(f"max_rows: {args.max_rows}, wf: {args.wf_years}")

    print(f"Loading {CACHE_PATH}...")
    d = pickle.load(gzip.open(CACHE_PATH, 'rb'))
    df, base_features = d['df'], d['features']
    df = merge_v15_1_features(df)
    print(f"After merge: shape={df.shape}")

    if 'is_win' not in df.columns:
        df['is_win'] = (pd.to_numeric(df['finish'], errors='coerce') == 1).astype(int)

    if 'year_full' in df.columns:
        df['_y'] = pd.to_numeric(df['year_full'], errors='coerce')
    else:
        yy = pd.to_numeric(df['race_id'].astype(str).str[2:4], errors='coerce')
        df['_y'] = yy + 2000

    feature_sets = {
        'V15_only':  base_features,
        'V15+KKA':   base_features + V15_1_KKA_FEATURES,
        'V15+SKB':   base_features + V15_1_SKB_FEATURES,
        'V15+SRB':   base_features + V15_1_SRB_FEATURES,
        'V15+all':   base_features + V15_1_KKA_FEATURES + V15_1_SKB_FEATURES + V15_1_SRB_FEATURES,
    }

    all_results = {}
    for test_year in args.wf_years:
        train_mask = (df['_y'] >= 2015) & (df['_y'] < test_year)
        test_mask = df['_y'] == test_year
        n_te = int(test_mask.sum())
        if n_te < 1000:
            continue

        train_idx = df.index[train_mask]
        if args.max_rows and len(train_idx) > args.max_rows:
            np.random.seed(42)
            train_idx = np.random.choice(train_idx.values, size=args.max_rows, replace=False)
        test_idx = df.index[test_mask]
        y_tr = df.loc[train_idx, 'is_win'].astype(int)
        y_te = df.loc[test_idx, 'is_win'].astype(int)

        print(f"\n=== test_year={test_year} (n_tr={len(train_idx)}, n_te={n_te}) ===")
        for name, feats in feature_sets.items():
            X_tr = df.loc[train_idx, feats].apply(pd.to_numeric, errors='coerce').fillna(0)
            X_te = df.loc[test_idx, feats].apply(pd.to_numeric, errors='coerce').fillna(0)
            t0 = time.time()
            booster = lgb.train(
                LGB_PARAMS, lgb.Dataset(X_tr.values, y_tr.values),
                num_boost_round=1000,
                valid_sets=[lgb.Dataset(X_te.values, y_te.values)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
            p = booster.predict(X_te.values)
            auc = roc_auc_score(y_te, p)
            print(f"  {name:12s} ({len(feats):3d} feats): AUC={auc:.4f} t={time.time()-t0:.1f}s")
            all_results.setdefault(test_year, {})[name] = {
                'auc': float(auc), 'n_features': len(feats),
            }

    # 差分計算
    print("\n=== 差分 vs V15_only (bp) ===")
    for yr, rs in all_results.items():
        v15_auc = rs['V15_only']['auc']
        print(f"\nyear {yr}:")
        for name, r in rs.items():
            if name == 'V15_only':
                continue
            delta = (r['auc'] - v15_auc) * 10000
            print(f"  {name:12s}: {delta:+7.1f}bp ({r['auc']:.4f})")

    with open(f"{OUT_DIR}/v15_1_ablation_results.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OK] Saved {OUT_DIR}/v15_1_ablation_results.json")


if __name__ == '__main__':
    main()
