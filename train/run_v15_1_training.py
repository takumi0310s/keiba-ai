"""V15.1 学習 wrapper.

V15 cache (data/_v15_optuna_df_cache.pkl.gz) を base に:
1. KKA/SKB/SR から 34 個の新 features merge
2. LightGBM 単体で 学習 (full ensemble は後段)
3. AUC 比較 (V15 0.8939 vs V15.1)
4. WF backtest (年別 AUC)

usage:
  python train/run_v15_1_training.py
  python train/run_v15_1_training.py --quick       # 早い iter (early stop 50, max 200)
  python train/run_v15_1_training.py --max-rows 100000  # 部分 subsample で test

output:
  data/v15.1/v15_1_lgb.txt         (LGB model)
  data/v15.1/v15_1_results.json    (AUC + 評価)
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

from v15_1_features import merge_v15_1_features, V15_1_NEW_FEATURES, coverage_report

CACHE_PATH = 'data/_v15_optuna_df_cache.pkl.gz'
OUT_DIR = 'data/v15.1'
os.makedirs(OUT_DIR, exist_ok=True)

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


def load_cache():
    print(f"Loading cache from {CACHE_PATH}...")
    t0 = time.time()
    d = pickle.load(gzip.open(CACHE_PATH, 'rb'))
    df, feats = d['df'], d['features']
    print(f"  loaded {len(df)} rows / {len(feats)} features in {time.time()-t0:.1f}s")
    return df, feats


def split_by_year(df: pd.DataFrame, train_until: int = 2024):
    """年別 split: train 〜train_until、val (train_until+1〜)."""
    if 'year_full' in df.columns:
        yr = pd.to_numeric(df['year_full'], errors='coerce')
    elif 'year' in df.columns:
        yr = pd.to_numeric(df['year'], errors='coerce')
        # 'year' could be 2-digit (15) or 4-digit
        if yr.max() < 100: yr = yr + 2000
    elif 'race_date' in df.columns:
        yr = pd.to_datetime(df['race_date'], errors='coerce').dt.year
    else:
        # cache race_id (10-char): yy at chars [2:4]
        yy = pd.to_numeric(df['race_id'].astype(str).str[2:4], errors='coerce')
        yr = yy + 2000

    train_mask = yr <= train_until
    val_mask = yr == train_until + 1
    return train_mask, val_mask


def train_and_eval(df, features, label_col, name, max_rows=None):
    # 重要: 先に年別 split を確定 (proper time-based、leak 防止)
    train_mask, val_mask = split_by_year(df, train_until=2024)
    print(f"  full df: train={train_mask.sum()}, val={val_mask.sum()}")

    if max_rows and len(df) > max_rows:
        # stratified by year mask: train から max_rows を sample、val は full
        train_df = df[train_mask].sample(n=min(max_rows, train_mask.sum()), random_state=42)
        val_df = df[val_mask]
        df_combined = pd.concat([train_df, val_df]).reset_index(drop=True)
        print(f"  subsampled train to {len(train_df)} rows, val={len(val_df)}")
        # rebuild masks
        train_mask = pd.Series([True]*len(train_df) + [False]*len(val_df), index=df_combined.index)
        val_mask = pd.Series([False]*len(train_df) + [True]*len(val_df), index=df_combined.index)
        df = df_combined

    train_idx = df.index[train_mask]
    val_idx = df.index[val_mask]
    if len(val_idx) < 1000:
        print("  WARN: val set too small, fallback to random 20%")
        n = len(df)
        idx = np.arange(n)
        np.random.seed(42); np.random.shuffle(idx)
        split = int(n * 0.8)
        train_idx, val_idx = idx[:split], idx[split:]

    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df[label_col].astype(int)

    X_tr, y_tr = X.loc[train_idx].values, y.loc[train_idx].values
    X_va, y_va = X.loc[val_idx].values, y.loc[val_idx].values

    print(f"  X_tr={X_tr.shape}, X_va={X_va.shape}")
    print(f"  y_tr win_rate={y_tr.mean():.4f}, y_va win_rate={y_va.mean():.4f}")

    print(f"  training LGB ({name})...")
    t0 = time.time()
    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=features)
    val_data = lgb.Dataset(X_va, label=y_va, feature_name=features, reference=train_data)

    booster = lgb.train(
        LGB_PARAMS, train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    print(f"  trained in {time.time()-t0:.1f}s, best_iter={booster.best_iteration}")

    p_va = booster.predict(X_va)
    auc = roc_auc_score(y_va, p_va)
    print(f"  {name} val AUC: {auc:.4f}")
    return auc, booster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='早い iter')
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--label', default='is_win', help='is_win or is_top3')
    args = parser.parse_args()

    if args.quick:
        LGB_PARAMS['learning_rate'] = 0.1

    df, base_features = load_cache()
    print(f"Base features (V15): {len(base_features)}")
    print(f"Adding V15.1 features: {len(V15_1_NEW_FEATURES)}")

    df = merge_v15_1_features(df)
    print(f"After merge: shape={df.shape}")

    cov = coverage_report(df)
    print('\nV15.1 new features coverage (non-zero rate):')
    for f, c in cov.items():
        print(f"  {f:35s} {c*100:6.1f}%")

    label_col = args.label
    if label_col not in df.columns:
        print(f"[WARN] {label_col} not in df. trying 'finish'...")
        if 'finish' in df.columns:
            df[label_col] = (pd.to_numeric(df['finish'], errors='coerce') <= (1 if label_col == 'is_win' else 3)).astype(int)
        else:
            print("[ERROR] no usable label column")
            sys.exit(1)

    print('\n=== V15 baseline (145 features) ===')
    auc_v15, _ = train_and_eval(df, base_features, label_col, 'V15', max_rows=args.max_rows)

    print('\n=== V15.1 (145 + 34 = 179 features) ===')
    extended_features = base_features + V15_1_NEW_FEATURES
    auc_v15_1, booster_15_1 = train_and_eval(df, extended_features, label_col, 'V15.1', max_rows=args.max_rows)

    delta = auc_v15_1 - auc_v15
    print(f"\n=== 比較 ===")
    print(f"  V15  AUC: {auc_v15:.4f}")
    print(f"  V15.1 AUC: {auc_v15_1:.4f}")
    print(f"  Δ:        {delta:+.4f}")

    verdict = 'NEUTRAL'
    if delta >= 0.005:
        verdict = 'ADOPT (5/16 投入候補)'
    elif delta >= 0.001:
        verdict = 'MINOR (要 sample 蓄積)'
    elif delta <= -0.005:
        verdict = 'REJECT (V15 維持)'
    else:
        verdict = 'NEUTRAL (V15 維持、新 features 害なし)'
    print(f"  判定: {verdict}")

    # save
    booster_path = os.path.join(OUT_DIR, 'v15_1_lgb.txt')
    try:
        booster_15_1.save_model(booster_path)
        print(f"  saved {booster_path}")
    except Exception as e:
        print(f"  save error: {e}")

    results = {
        'v15_auc': float(auc_v15),
        'v15_1_auc': float(auc_v15_1),
        'delta': float(delta),
        'verdict': verdict,
        'label': label_col,
        'n_features_v15': len(base_features),
        'n_features_v15_1': len(extended_features),
        'n_new_features': len(V15_1_NEW_FEATURES),
        'coverage': cov,
        'lgb_params': LGB_PARAMS,
        'max_rows': args.max_rows,
        'quick': args.quick,
        'best_iteration': int(booster_15_1.best_iteration) if hasattr(booster_15_1, 'best_iteration') else None,
    }
    with open(os.path.join(OUT_DIR, 'v15_1_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"  saved {os.path.join(OUT_DIR, 'v15_1_results.json')}")


if __name__ == '__main__':
    main()
