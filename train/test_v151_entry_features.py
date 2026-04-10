#!/usr/bin/env python3
"""v15.1テスト — v15(145特徴量) + 入厩情報特徴量

新特徴量:
  - jrdb_entry_days_ago: 入厩何日前（レース日から遡った日数、0=不明）
  - jrdb_entry_race_num: 入厩何走目（0=不明）

採用基準:
  - WF AUC > 0.8858 (v15ベースライン)
  - 全年AUC > 0.85
  - max gap < 0.05

Usage:
    python train/test_v151_entry_features.py
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

CACHE_PATH = os.path.join(DATA_DIR, '_v15_optuna_df_cache.pkl.gz')

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


def merge_entry_features(df):
    """Merge jrdb_entry_days_ago and jrdb_entry_race_num from jrdb_kyi.csv

    Uses nk_race_id (netkeiba 12-digit format) + umaban as merge key,
    same as the existing JRDB feature merge pipeline.
    """
    from jrdb_features import _build_nk_race_id_from_jv

    print("  Merging JRDB entry features...")
    kyi_path = os.path.join(DATA_DIR, 'jrdb_kyi.csv')
    if not os.path.exists(kyi_path):
        print("  WARNING: jrdb_kyi.csv not found")
        df['jrdb_entry_days_ago'] = 0
        df['jrdb_entry_race_num'] = 0
        return df

    kyi = pd.read_csv(kyi_path, encoding='utf-8-sig', dtype=str,
                       usecols=['nk_race_id', '馬番', '入厩何日前', '入厩何走目'])

    kyi['_nk_rid'] = kyi['nk_race_id'].astype(str).str.zfill(12)
    kyi['_uma'] = pd.to_numeric(kyi['馬番'], errors='coerce')
    kyi['jrdb_entry_days_ago'] = pd.to_numeric(kyi['入厩何日前'], errors='coerce').fillna(0)
    kyi['jrdb_entry_race_num'] = pd.to_numeric(kyi['入厩何走目'], errors='coerce').fillna(0)

    kyi_d = kyi[['_nk_rid', '_uma', 'jrdb_entry_days_ago', 'jrdb_entry_race_num']].copy()
    kyi_d = kyi_d.drop_duplicates(subset=['_nk_rid', '_uma'], keep='last')

    # Build nk_race_id for df
    df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    df['_uma'] = pd.to_numeric(df['umaban'], errors='coerce')

    before = len(df)
    df = df.merge(kyi_d, on=['_nk_rid', '_uma'], how='left', suffixes=('', '_entry'))

    for col in ['jrdb_entry_days_ago', 'jrdb_entry_race_num']:
        if f'{col}_entry' in df.columns:
            df[col] = df[f'{col}_entry'].combine_first(df.get(col, pd.Series(0.0)))
            df.drop(columns=[f'{col}_entry'], inplace=True)
        df[col] = df[col].fillna(0)

    df.drop(columns=['_nk_rid', '_uma'], inplace=True, errors='ignore')

    matched = (df['jrdb_entry_days_ago'] > 0).sum()
    print(f"  Entry days matched: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")
    matched2 = (df['jrdb_entry_race_num'] > 0).sum()
    print(f"  Entry race# matched: {matched2}/{len(df)} ({matched2/len(df)*100:.1f}%)")

    return df


def main():
    start = time.time()
    print("=" * 60)
    print("  v15.1 Test: v15 + entry features")
    print("=" * 60)

    # Load cached v15 dataframe
    if not os.path.exists(CACHE_PATH):
        print(f"ERROR: v15 cache not found: {CACHE_PATH}")
        sys.exit(1)

    print("\n[1/3] Loading v15 cache...")
    import gzip, pickle
    with gzip.open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    df = cache['df']
    features_v15 = cache['features']
    print(f"  Loaded: {len(df)} rows, {len(features_v15)} features")

    # Merge entry features
    print("\n[2/3] Merging entry features...")
    df = merge_entry_features(df)

    # Setup
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses'] >= 5].copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)

    NEW_FEATURES = ['jrdb_entry_days_ago', 'jrdb_entry_race_num']
    features_v151 = features_v15 + NEW_FEATURES

    for f in features_v151:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"  v15 features: {len(features_v15)}")
    print(f"  v15.1 features: {len(features_v151)}")

    # WF test
    print("\n[3/3] Walk-forward backtest (LGB + XGB)...")
    results_base = []
    results_new = []

    for test_year in range(2021, 2026):
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        y_train = df.loc[train_mask, 'target'].values
        y_test = df.loc[test_mask, 'target'].values

        for label, feats, result_list in [
            ('v15', features_v15, results_base),
            ('v15.1', features_v151, results_new),
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
            train_lgb = roc_auc_score(y_train, m_lgb.predict(X_tr))

            # XGB
            dtrain_xgb = xgb.DMatrix(X_tr, label=y_train)
            dvalid_xgb = xgb.DMatrix(X_te, label=y_test)
            m_xgb = xgb.train(XGB_PARAMS, dtrain_xgb, num_boost_round=1000,
                              evals=[(dvalid_xgb, 'valid')],
                              early_stopping_rounds=50, verbose_eval=False)
            pred_xgb = m_xgb.predict(dvalid_xgb)
            xgb_auc = roc_auc_score(y_test, pred_xgb)

            avg_auc = roc_auc_score(y_test, 0.5 * pred_lgb + 0.5 * pred_xgb)

            result_list.append({
                'year': test_year,
                'lgb_auc': round(lgb_auc, 6),
                'xgb_auc': round(xgb_auc, 6),
                'avg_auc': round(avg_auc, 6),
                'train_auc': round(train_lgb, 6),
                'gap': round(train_lgb - lgb_auc, 6),
            })

            # Feature importance for new features
            if label == 'v15.1' and test_year == 2025:
                imp = m_lgb.feature_importance(importance_type='gain')
                feat_imp = sorted(zip(feats, imp), key=lambda x: -x[1])
                for fname, val in feat_imp:
                    if fname in NEW_FEATURES:
                        rank = [i for i, (fn, _) in enumerate(feat_imp) if fn == fname][0] + 1
                        print(f"  {fname}: importance={val:.1f}, rank={rank}/{len(feats)}")

        b = results_base[-1]
        n = results_new[-1]
        diff = n['lgb_auc'] - b['lgb_auc']
        print(f"  {test_year}: v15={b['lgb_auc']:.4f} v15.1={n['lgb_auc']:.4f} diff={diff:+.5f} | "
              f"avg: v15={b['avg_auc']:.4f} v15.1={n['avg_auc']:.4f} | "
              f"gap: {b['gap']:.4f}/{n['gap']:.4f}")

    # Summary
    mean_base = np.mean([r['lgb_auc'] for r in results_base])
    mean_new = np.mean([r['lgb_auc'] for r in results_new])
    mean_base_avg = np.mean([r['avg_auc'] for r in results_base])
    mean_new_avg = np.mean([r['avg_auc'] for r in results_new])

    print(f"\n{'='*60}")
    print(f"  RESULT")
    print(f"{'='*60}")
    print(f"  v15 LGB mean:   {mean_base:.6f}")
    print(f"  v15.1 LGB mean: {mean_new:.6f}")
    print(f"  LGB diff:       {mean_new - mean_base:+.6f}")
    print(f"  v15 AVG mean:   {mean_base_avg:.6f}")
    print(f"  v15.1 AVG mean: {mean_new_avg:.6f}")
    print(f"  AVG diff:       {mean_new_avg - mean_base_avg:+.6f}")

    adopted = (mean_new - mean_base) > 0
    print(f"\n  VERDICT: {'POSITIVE' if adopted else 'NEGATIVE'}")

    elapsed = time.time() - start
    print(f"  Time: {elapsed/60:.1f} min")

    # Save
    out = {
        'test': 'v15.1_entry_features',
        'timestamp': pd.Timestamp.now().isoformat(),
        'new_features': NEW_FEATURES,
        'baseline_lgb': mean_base,
        'new_lgb': mean_new,
        'improvement_lgb': mean_new - mean_base,
        'baseline_avg': mean_base_avg,
        'new_avg': mean_new_avg,
        'improvement_avg': mean_new_avg - mean_base_avg,
        'adopted': adopted,
        'yearly_base': results_base,
        'yearly_new': results_new,
    }
    out_path = os.path.join(DATA_DIR, 'v151_entry_test.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
