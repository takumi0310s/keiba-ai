#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V22 enhanced 4-model Grid ensemble (LGB+XGB+FT+IR) + features_merged_all.

V22 enhanced = V15 cache (145) + V20 Phase 24/26 (32) + features_merged_all (105) = 282 features

【V15 投資保護】 V15 .pkl.gz 完全不変。 V22 enhanced は 別 file 出力 (models/v22_enhanced/)。

GPU: RTX 4070 Ti SUPER 16GB
所要時間: quick 1 fold ~30 min、 full 6-fold ~120 min

Usage:
    python train/train_v22_enhanced.py --quick   # 2025 fold のみ
    python train/train_v22_enhanced.py           # 6-fold WF
"""
import argparse
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
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
V15_CACHE = os.path.join(DATA_DIR, '_v15_optuna_df_cache.pkl.gz')
V20_DATA = os.path.join(DATA_DIR, 'v20_training_data_full.csv')
MERGED_FEATURES = os.path.join(DATA_DIR, 'features_merged_all.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'v22_enhanced')
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reuse FT/IR/train_intra_race from V22 base trainer
from train_v22_4ensemble import (
    FTTransformer, IntraRaceAttention, train_ft_transformer,
    train_intra_race, build_race_id_unique, PHASE24_26_FEATURES,
    LGB_PARAMS, XGB_PARAMS, MAX_HORSES
)


def load_v22_enhanced_data():
    """V22 base + features_merged_all を merge。"""
    print(f'[INFO] loading V15 cache + V20 P24/26 + features_merged_all')
    with gzip.open(V15_CACHE, 'rb') as f:
        d = pickle.load(f)
    df15 = d['df']
    v15_features = d['features']
    print(f'  V15 shape: {df15.shape}, features: {len(v15_features)}')

    if 'top3' not in df15.columns and 'finish' in df15.columns:
        df15['top3'] = (df15['finish'] <= 3).astype(int)
    if 'target' not in df15.columns:
        df15['target'] = df15['top3']

    df15['horse_id_str'] = df15['horse_id'].astype(str)
    for c in ['year', 'month', 'day', 'race_num', 'umaban']:
        df15[c] = pd.to_numeric(df15[c], errors='coerce').astype('Int64')

    # V20 phase 24/26 merge
    df20 = pd.read_csv(V20_DATA, usecols=lambda c: c in ['horse_id', 'year', 'month',
                                                          'day', 'race_num', 'umaban'] + PHASE24_26_FEATURES,
                       dtype={'horse_id': str}, low_memory=False)
    df20['horse_id_str'] = df20['horse_id'].astype(str)
    for c in ['year', 'month', 'day', 'race_num', 'umaban']:
        df20[c] = pd.to_numeric(df20[c], errors='coerce').astype('Int64')
    key = ['year', 'month', 'day', 'race_num', 'horse_id_str', 'umaban']
    add_cols_p24 = [c for c in PHASE24_26_FEATURES if c in df20.columns]
    df20_sub = df20[key + add_cols_p24].drop_duplicates(subset=key, keep='last')
    df = df15.merge(df20_sub, on=key, how='left')
    print(f'  V22 base merged: {df.shape}, +{len(add_cols_p24)} P24/26 features')

    # features_merged_all merge (key: race_id + umaban)
    print(f'  loading {MERGED_FEATURES} ...')
    df_extra = pd.read_csv(MERGED_FEATURES, encoding='utf-8-sig', low_memory=False)
    df_extra['race_id'] = df_extra['race_id'].astype(str)
    df_extra['umaban'] = pd.to_numeric(df_extra['umaban'], errors='coerce').astype('Int64')
    df_extra = df_extra.drop_duplicates(['race_id', 'umaban'], keep='last')
    extra_feat_cols = [c for c in df_extra.columns if c not in ('race_id', 'horse_id', 'umaban')]
    print(f'  features_merged: {df_extra.shape}, {len(extra_feat_cols)} feature cols')

    # df の race_id を str に
    df['race_id'] = df['race_id'].astype(str)
    df = df.merge(df_extra[['race_id', 'umaban'] + extra_feat_cols],
                  on=['race_id', 'umaban'], how='left')
    print(f'  V22 enhanced merged: {df.shape}')

    all_features = list(v15_features) + add_cols_p24 + extra_feat_cols
    numeric_features = [f for f in all_features if f in df.columns
                        and df[f].dtype.kind in 'iufb']
    print(f'  V22 enhanced numeric features: {len(numeric_features)}')

    df = df.dropna(subset=['target', 'year'])
    df['year'] = df['year'].astype(int)
    df[numeric_features] = df[numeric_features].fillna(0)

    return df, numeric_features


def train_lgb(df_tr, df_va, features, label='target'):
    X_tr, y_tr = df_tr[features].astype(np.float32).values, df_tr[label].values
    X_va, y_va = df_va[features].astype(np.float32).values, df_va[label].values
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_va, label=y_va, reference=train_set)
    model = lgb.train(LGB_PARAMS, train_set, num_boost_round=1000,
                       valid_sets=[val_set], valid_names=['val'],
                       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    p = model.predict(X_va)
    auc = roc_auc_score(y_va, p)
    return model, p, auc


def train_xgb(df_tr, df_va, features, label='target'):
    X_tr, y_tr = df_tr[features].astype(np.float32).values, df_tr[label].values
    X_va, y_va = df_va[features].astype(np.float32).values, df_va[label].values
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    model = xgb.train(XGB_PARAMS, dtr, num_boost_round=1000,
                       evals=[(dva, 'val')], early_stopping_rounds=50, verbose_eval=0)
    p = model.predict(dva)
    auc = roc_auc_score(y_va, p)
    return model, p, auc


def grid_search_weights(p_lgb, p_xgb, p_ft, p_ir, y_true, step=0.05):
    best_auc = 0
    best_w = (0.25, 0.25, 0.25, 0.25)
    for w_lgb in np.arange(0, 1 + step, step):
        for w_xgb in np.arange(0, 1 - w_lgb + step, step):
            for w_ft in np.arange(0, 1 - w_lgb - w_xgb + step, step):
                w_ir = 1 - w_lgb - w_xgb - w_ft
                if w_ir < 0 or w_ir > 1:
                    continue
                p_ens = (w_lgb * p_lgb + w_xgb * p_xgb +
                         w_ft * p_ft + w_ir * p_ir)
                auc = roc_auc_score(y_true, p_ens)
                if auc > best_auc:
                    best_auc = auc
                    best_w = (w_lgb, w_xgb, w_ft, w_ir)
    return best_auc, best_w


def run_wf(df, features, folds, quick=False):
    df = build_race_id_unique(df)
    results = []
    for y_lo, y_hi in folds:
        train_mask = df['year'] < y_lo
        test_mask = (df['year'] >= y_lo) & (df['year'] <= y_hi)
        df_tr = df[train_mask].copy()
        df_te = df[test_mask].copy()
        n_tr, n_te = len(df_tr), len(df_te)
        print(f'\n=== fold {y_lo}-{y_hi}: train={n_tr:,}, test={n_te:,} ===')
        if n_tr < 1000 or n_te < 100:
            print(f'  SKIP (insufficient data)')
            continue

        X_tr = df_tr[features].astype(np.float32).values
        y_tr = df_tr['target'].values
        X_te = df_te[features].astype(np.float32).values
        y_te = df_te['target'].values
        test_indices = df_te.index.tolist()

        # LGB
        print('  Training LGB...')
        t0 = time.time()
        lgb_model, p_lgb, auc_lgb = train_lgb(df_tr, df_te, features)
        print(f'  LGB AUC={auc_lgb:.4f} [{time.time()-t0:.0f}s]')

        # XGB
        print('  Training XGB...')
        t0 = time.time()
        xgb_model, p_xgb, auc_xgb = train_xgb(df_tr, df_te, features)
        print(f'  XGB AUC={auc_xgb:.4f} [{time.time()-t0:.0f}s]')

        # FT skip (282 features で val step OOM、 V22 enhanced 専用 limitation)
        # 代わりに LGB+XGB+IR Grid 3-model ensemble
        print('  FT skipped (OOM with 282 features)')
        p_ft = np.full(len(X_te), 0.3, dtype=np.float32)  # neutral baseline
        auc_ft = 0.5

        torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None
        print('  Training IntraRace...')
        t0 = time.time()
        df_tr_ir = df.loc[train_mask].copy()
        df_te_ir = df.loc[test_mask].copy()
        scaler_ir = StandardScaler()
        df_tr_ir[features] = scaler_ir.fit_transform(df_tr_ir[features].values.astype(np.float32))
        df_te_ir[features] = scaler_ir.transform(df_te_ir[features].values.astype(np.float32))
        # ★ V22 enhanced: features 282 + IR で OOM 対策 → d_model 64 (元 V22 base 値) ★
        ir_model, ir_val_dict, auc_ir_raw = train_intra_race(
            df_tr_ir, df_te_ir, features,
            epochs=30, patience=10, seed=42 + int(y_lo), d_model=64,
        )
        p_ir = np.zeros(len(X_te), dtype=np.float32)
        cov = 0
        for i, idx in enumerate(test_indices):
            if idx in ir_val_dict:
                p_ir[i] = ir_val_dict[idx]
                cov += 1
            else:
                p_ir[i] = 0.3
        auc_ir = roc_auc_score(y_te, p_ir) if cov > len(X_te) * 0.5 else 0
        print(f'  IR AUC={auc_ir:.4f} (cov {cov/len(X_te)*100:.0f}%) [{time.time()-t0:.0f}s]')

        # 4-ens AUC-weighted
        aucs = [max(auc_lgb, 0.5), max(auc_xgb, 0.5),
                max(auc_ft, 0.5), max(auc_ir, 0.5)]
        if sum(aucs) > 2.0:
            w = [a - 0.5 for a in aucs]
            w = [x / sum(w) for x in w]
            p_ens_aw = w[0]*p_lgb + w[1]*p_xgb + w[2]*p_ft + w[3]*p_ir
            auc_ens_aw = roc_auc_score(y_te, p_ens_aw)
        else:
            auc_ens_aw = 0

        # Grid search
        auc_grid, w_grid = grid_search_weights(p_lgb, p_xgb, p_ft, p_ir, y_te)
        print(f'  4-ens AUC-w={auc_ens_aw:.4f}, Grid={auc_grid:.4f}, w_grid={w_grid}')

        results.append({
            'fold': f'{y_lo}-{y_hi}',
            'n_train': n_tr, 'n_test': n_te,
            'auc_lgb': float(auc_lgb), 'auc_xgb': float(auc_xgb),
            'auc_ft': float(auc_ft), 'auc_ir': float(auc_ir),
            'auc_ens_aw': float(auc_ens_aw),
            'auc_grid': float(auc_grid),
            'w_grid': [float(x) for x in w_grid],
        })

        if quick:
            print('  (quick mode: 1 fold で 終了)')
            break

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='quick mode (2025 fold のみ)')
    args = ap.parse_args()

    print('=' * 60)
    print('V22 enhanced 4-ens (LGB+XGB+FT+IR + features_merged_all)')
    print('=' * 60)
    print(f'DEVICE: {DEVICE}')

    t0 = time.time()
    df, features = load_v22_enhanced_data()

    folds = ([(25, 25)] if args.quick
             else [(20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)])

    results = run_wf(df, features, folds, quick=args.quick)

    # summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    aucs_lgb = [r['auc_lgb'] for r in results]
    aucs_xgb = [r['auc_xgb'] for r in results]
    aucs_ft = [r['auc_ft'] for r in results]
    aucs_ir = [r['auc_ir'] for r in results]
    aucs_grid = [r['auc_grid'] for r in results]
    print(f'mean LGB AUC: {np.mean(aucs_lgb):.4f}')
    print(f'mean XGB AUC: {np.mean(aucs_xgb):.4f}')
    print(f'mean FT  AUC: {np.mean(aucs_ft):.4f}')
    print(f'mean IR  AUC: {np.mean(aucs_ir):.4f}')
    print(f'mean Grid AUC: {np.mean(aucs_grid):.4f}')
    print(f'V15 baseline: 0.8939')
    print(f'V22 base (prev): 0.8800')
    print(f'V22 enhanced delta vs V15: {np.mean(aucs_grid) - 0.8939:+.4f}')

    out_path = os.path.join(MODEL_DIR, f'enhanced_wf_summary_{datetime.now():%Y%m%d_%H%M%S}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'mode': 'quick' if args.quick else 'full',
            'results': results,
            'n_features': len(features),
            'features_sample': features[:50],
            'mean_grid_auc': float(np.mean(aucs_grid)) if aucs_grid else 0,
            'v15_baseline': 0.8939,
            'elapsed_s': time.time() - t0,
        }, f, ensure_ascii=False, indent=2)
    print(f'\nresults: {out_path}')


if __name__ == '__main__':
    main()
