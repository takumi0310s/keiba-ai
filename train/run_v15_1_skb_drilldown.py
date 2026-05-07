"""V15.1 SKB の どの code が leak かを drilldown する.

Session #38 A: SKB +675bp の真相究明。
仮説: kishi_code_3 (corr 0.14) が leak の主因。 これを除けば改善幅は < 50bp。
"""
from __future__ import annotations
import os, sys, pickle, gzip, json, time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

BASE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE)
sys.path.insert(0, os.path.join(BASE, 'train'))
from v15_1_features import merge_v15_1_features, V15_1_SKB_FEATURES

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
    print("=" * 60)
    print("V15.1 SKB drilldown ablation (Session #38 A)")
    print("=" * 60)

    print(f"Loading {CACHE_PATH}...")
    d = pickle.load(gzip.open(CACHE_PATH, 'rb'))
    df, base_features = d['df'], d['features']
    df = merge_v15_1_features(df)
    if 'is_win' not in df.columns:
        df['is_win'] = (pd.to_numeric(df['finish'], errors='coerce') == 1).astype(int)
    if 'year_full' in df.columns:
        df['_y'] = pd.to_numeric(df['year_full'], errors='coerce')
    else:
        yy = pd.to_numeric(df['race_id'].astype(str).str[2:4], errors='coerce')
        df['_y'] = yy + 2000

    train_mask = (df['_y'] >= 2015) & (df['_y'] <= 2024)
    test_mask = df['_y'] == 2025
    train_idx = df.index[train_mask]
    np.random.seed(42)
    train_idx = np.random.choice(train_idx.values, size=200000, replace=False)
    test_idx = df.index[test_mask]

    y_tr = df.loc[train_idx, 'is_win'].astype(int)
    y_te = df.loc[test_idx, 'is_win'].astype(int)

    # 各 SKB code を 1 つずつ削った時の AUC
    feature_sets = {
        'V15_only': base_features,
        'V15+all_SKB': base_features + V15_1_SKB_FEATURES,
        'V15+SKB_no_kishi3': base_features + [f for f in V15_1_SKB_FEATURES if f != 'skb_kishi_code_3'],
        'V15+SKB_no_kishi23': base_features + [f for f in V15_1_SKB_FEATURES if f not in ('skb_kishi_code_2', 'skb_kishi_code_3')],
        'V15+SKB_no_kishi': base_features + [f for f in V15_1_SKB_FEATURES if not f.startswith('skb_kishi_code')],
        'V15+kishi_only': base_features + ['skb_kishi_code_1', 'skb_kishi_code_2', 'skb_kishi_code_3'],
        'V15+kishi3_only': base_features + ['skb_kishi_code_3'],
        'V15+kishi1_only': base_features + ['skb_kishi_code_1'],
        'V15+kishi2_only': base_features + ['skb_kishi_code_2'],
    }

    results = {}
    print(f"\nTrain rows: {len(train_idx):,} (subsample), Test rows: {len(test_idx):,}")
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
        elapsed = time.time() - t0
        results[name] = {'auc': float(auc), 'n_features': len(feats), 't_sec': elapsed}
        print(f"  {name:30s} ({len(feats):3d} feats): AUC={auc:.4f} t={elapsed:.1f}s")

    print("\n=== 差分 vs V15_only ===")
    base_auc = results['V15_only']['auc']
    print(f"V15_only baseline AUC: {base_auc:.4f}")
    for name, r in results.items():
        if name == 'V15_only':
            continue
        delta = (r['auc'] - base_auc) * 10000
        print(f"  {name:30s}: {delta:+7.1f}bp")

    print("\n=== 単独 SKB code 寄与 ===")
    print(f"  kishi_code_1 only: {(results['V15+kishi1_only']['auc'] - base_auc)*10000:+7.1f}bp")
    print(f"  kishi_code_2 only: {(results['V15+kishi2_only']['auc'] - base_auc)*10000:+7.1f}bp")
    print(f"  kishi_code_3 only: {(results['V15+kishi3_only']['auc'] - base_auc)*10000:+7.1f}bp")

    with open(f"{OUT_DIR}/v15_1_skb_drilldown.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Saved {OUT_DIR}/v15_1_skb_drilldown.json")


if __name__ == '__main__':
    main()
