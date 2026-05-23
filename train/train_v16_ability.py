#!/usr/bin/env python3
"""V16 Ability-Based Candidate Training Script

哲学: オッズは結果。能力の実体 (ジョッキー・血統・調教・厩舎) で予測する。
     paci_ninki_idx 等の人気・オッズ系 feature を完全除外し、
     137 の能力系 feature のみで LGB+XGB walk-forward を実行。

V15 production は完全不変。出力先: models/v16_ability_candidate.pkl.gz

Usage:
    python train/train_v16_ability.py
    python train/train_v16_ability.py --wf-only   # WF AUC のみ (model 保存なし)
"""

import os
import sys
import time
import json
import gzip
import pickle
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# === 除外する オッズ・人気系 feature (V15 145→V16 137) ===
ODDS_FEATURES_REMOVE = [
    'paci_ninki_idx',       # JRDB 人気指数 (gain 16.93% — odds-derived)
    'odds_change_rate',     # 当日オッズ変化率
    'odds_sharp_drop',      # オッズ急落フラグ
    'oz_base_pop_rank',     # 基準人気順位
    'oz_fukusho_base_log',  # 基準複勝オッズ log
    'oz_tansho_base_log',   # 基準単勝オッズ log
    'pop_rank_change',      # 人気順位変化
    'prev_odds_log',        # 前走オッズ log
]

# LGB/XGB params (V15 と同一)
LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_child_samples': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'verbose': -1, 'seed': 42,
}
XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'max_depth': 6, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'seed': 42, 'tree_method': 'hist', 'verbosity': 0,
}

WF_YEARS = range(2021, 2026)


def load_cache():
    path = os.path.join(DATA_DIR, '_v15_optuna_df_cache.pkl.gz')
    print(f"Loading cache: {path}")
    with gzip.open(path, 'rb') as f:
        obj = pickle.load(f)
    df = obj['df']
    v15_features = obj['features']
    print(f"  df: {len(df)} rows, {len(df.columns)} cols")
    print(f"  V15 features: {len(v15_features)}")
    return df, v15_features


def get_v16_features(v15_features):
    removed = [f for f in ODDS_FEATURES_REMOVE if f in v15_features]
    v16 = [f for f in v15_features if f not in ODDS_FEATURES_REMOVE]
    print(f"\n  Feature pruning:")
    print(f"    V15: {len(v15_features)} features")
    print(f"    Removed ({len(removed)}): {removed}")
    print(f"    V16: {len(v16)} features (ability-only)")
    return v16, removed


def build_race_id(df):
    if 'race_id_unique' not in df.columns:
        if 'race_id' in df.columns:
            df['race_id_unique'] = df['race_id'].astype(str)
        else:
            df['race_id_unique'] = (
                df['date_num'].astype(str) + '_' +
                df.get('venue', pd.Series(['0']*len(df))).astype(str) + '_' +
                df.get('race_num', pd.Series(['0']*len(df))).astype(str)
            )
    return df


def walk_forward_lgb_xgb(df, features, years=WF_YEARS):
    """LGB+XGB walk-forward (same as V15 genuine WF)."""
    df = build_race_id(df)
    results = []
    for test_year in years:
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        n_tr = train_mask.sum()
        n_te = test_mask.sum()
        if n_tr < 1000 or n_te < 100:
            print(f"  Skip {test_year}: train={n_tr}, test={n_te}")
            continue
        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[test_mask, features].values
        y_te = df.loc[test_mask, 'target'].values
        print(f"\n  WF {test_year}: train={n_tr}, test={n_te}, feats={len(features)}")

        # LGB
        dt = lgb.Dataset(X_tr, label=y_tr)
        dv = lgb.Dataset(X_te, label=y_te, reference=dt)
        m_lgb = lgb.train(LGB_PARAMS, dt, num_boost_round=1000,
                          valid_sets=[dv],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
        p_lgb = m_lgb.predict(X_te)
        auc_lgb = roc_auc_score(y_te, p_lgb)

        # XGB
        dxtr = xgb.DMatrix(X_tr, label=y_tr)
        dxte = xgb.DMatrix(X_te, label=y_te)
        m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=1000,
                           evals=[(dxte, 'valid')],
                           early_stopping_rounds=50, verbose_eval=False)
        p_xgb = m_xgb.predict(dxte)
        auc_xgb = roc_auc_score(y_te, p_xgb)

        # Ensemble (equal weight LGB+XGB)
        ensemble_w = {'lgb': 0.5, 'xgb': 0.5}
        p_ens = ensemble_w['lgb'] * p_lgb + ensemble_w['xgb'] * p_xgb
        auc_ens = roc_auc_score(y_te, p_ens)

        print(f"    LGB={auc_lgb:.4f}  XGB={auc_xgb:.4f}  ENS={auc_ens:.4f}")
        results.append({
            'year': test_year, 'auc_lgb': auc_lgb, 'auc_xgb': auc_xgb,
            'auc_ensemble': auc_ens, 'n_train': int(n_tr), 'n_test': int(n_te),
        })
    return results


def train_final_model(df, features):
    """Full data train (2020-2025) for production candidate."""
    print("\n  Training final model (all data 2020-2025)...")
    mask = (df['year'] >= 20) & (df['year'] <= 25)
    X = df.loc[mask, features].values
    y = df.loc[mask, 'target'].values
    print(f"    Train rows: {len(X)}, features: {len(features)}")

    # LGB (no early stopping for final)
    dt = lgb.Dataset(X, label=y)
    m_lgb = lgb.train({**LGB_PARAMS, 'num_leaves': 63}, dt, num_boost_round=500)
    p_lgb = m_lgb.predict(X)
    auc_lgb = roc_auc_score(y, p_lgb)
    print(f"    LGB in-sample AUC: {auc_lgb:.4f}")

    # XGB
    dxtr = xgb.DMatrix(X, label=y)
    m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=500,
                       evals=[(dxtr, 'train')], verbose_eval=False)
    p_xgb = m_xgb.predict(dxtr)
    auc_xgb = roc_auc_score(y, p_xgb)

    return m_lgb, m_xgb, auc_lgb, auc_xgb


def save_candidate(m_lgb, m_xgb, features, wf_results, removed_features, auc_lgb):
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    out_path = os.path.join(BASE_DIR, 'models', 'v16_ability_candidate.pkl.gz')

    wf_mean = np.mean([r['auc_ensemble'] for r in wf_results]) if wf_results else 0.0

    pkl = {
        'version': 'v16_ability_candidate',
        'description': 'Ability-only features. No odds/popularity. Candidate only.',
        'model': m_lgb,
        'xgb_model': m_xgb,
        'ensemble_weights': {'lgb': 0.5, 'xgb': 0.5, 'mlp': 0},
        'features': features,
        'auc': auc_lgb,
        'wf_auc_mean': wf_mean,
        'wf_results': wf_results,
        'removed_features': removed_features,
        'n_features': len(features),
        'leak_free': True,
        'leak_pattern': 'A',
        'is_live': False,
        'is_candidate': True,
        'trained_at': datetime.now().isoformat(),
        'v15_baseline_wf_auc': 0.8678,
    }
    with gzip.open(out_path, 'wb') as f:
        pickle.dump(pkl, f, protocol=4)
    print(f"\n  Saved: {out_path}")
    print(f"  V16 WF AUC mean (LGB+XGB): {wf_mean:.4f}")
    print(f"  V15 baseline WF AUC:         0.8678")
    print(f"  Delta:                       {wf_mean - 0.8678:+.4f}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wf-only', action='store_true',
                        help='WF AUC only, no model save')
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("  V16 Ability-Based Candidate Training")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 70)

    df, v15_features = load_cache()
    v16_features, removed = get_v16_features(v15_features)

    # Ensure 'target' column
    if 'target' not in df.columns:
        if 'finish' in df.columns:
            df['target'] = (df['finish'] <= 3).astype(int)
        else:
            raise ValueError("No 'target' or 'finish' column in cache df")

    # Fill NaN in features
    for f in v16_features:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
        else:
            df[f] = 0.0

    # Walk-forward evaluation
    print(f"\n  Walk-forward evaluation (LGB+XGB, {list(WF_YEARS)})...")
    wf_results = walk_forward_lgb_xgb(df, v16_features)

    if wf_results:
        wf_mean_lgb = np.mean([r['auc_lgb'] for r in wf_results])
        wf_mean_xgb = np.mean([r['auc_xgb'] for r in wf_results])
        wf_mean_ens = np.mean([r['auc_ensemble'] for r in wf_results])
        print(f"\n  === V16 WF Summary ===")
        print(f"  LGB mean: {wf_mean_lgb:.4f}")
        print(f"  XGB mean: {wf_mean_xgb:.4f}")
        print(f"  ENS mean: {wf_mean_ens:.4f}")
        print(f"  V15 baseline (LGB+XGB genuine WF): 0.8678")
        print(f"  Delta vs V15: {wf_mean_ens - 0.8678:+.4f}")
        for r in wf_results:
            print(f"    {r['year']}: LGB={r['auc_lgb']:.4f}  XGB={r['auc_xgb']:.4f}  ENS={r['auc_ensemble']:.4f}")

    # Save report
    report = {
        'v16_wf_results': wf_results,
        'removed_features': removed,
        'v16_features_count': len(v16_features),
        'v15_features_count': len(v15_features),
        'v15_baseline_wf_auc': 0.8678,
        'trained_at': datetime.now().isoformat(),
    }
    rpath = os.path.join(DATA_DIR, 'v16_ability_wf_report.json')
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Report saved: {rpath}")

    if not args.wf_only:
        m_lgb, m_xgb, auc_lgb, auc_xgb = train_final_model(df, v16_features)
        save_candidate(m_lgb, m_xgb, v16_features, wf_results, removed, auc_lgb)

    elapsed = (time.time() - t0) / 60
    print(f"\n  Elapsed: {elapsed:.1f} min")
    print("=" * 70)
    print("  V16 ability-based candidate complete. V15 production unchanged.")
    print("=" * 70)


if __name__ == '__main__':
    main()
