#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V22 enhanced top 100 features retrain (CUDA OOM 回避 + FT 復活).

select_top_features.py で 抽出した top 100 features のみ使用。
LGB+XGB+FT+IR 4-ensemble Grid。

期待: V15 越え 試行 (V22 enhanced 282 features は -0.016 で 未達)。

Usage:
    python train/train_v22_enhanced_top100.py --top-n 100 --quick
    python train/train_v22_enhanced_top100.py --top-n 100
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'v22_enhanced_top100')
os.makedirs(MODEL_DIR, exist_ok=True)

TOP_FEATURES_JSON = os.path.join(DATA_DIR, 'top_features_v22enh.json')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from train_v22_4ensemble import (
    FTTransformer, IntraRaceAttention, train_ft_transformer,
    train_intra_race, build_race_id_unique,
    LGB_PARAMS, XGB_PARAMS, MAX_HORSES
)
from train_v22_enhanced import load_v22_enhanced_data, train_lgb, train_xgb, grid_search_weights


def load_top_n_features(top_n: int) -> list:
    """top_features_v22enh.json から top N features name 取得."""
    with open(TOP_FEATURES_JSON, 'r', encoding='utf-8') as f:
        d = json.load(f)
    top_n = min(top_n, len(d['top_features']))
    return [item['name'] for item in d['top_features'][:top_n]]


def run_wf_top(df, features, folds, quick=False):
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

        print('  Training LGB...')
        t0 = time.time()
        lgb_model, p_lgb, auc_lgb = train_lgb(df_tr, df_te, features)
        print(f'  LGB AUC={auc_lgb:.4f} [{time.time()-t0:.0f}s]')

        print('  Training XGB...')
        t0 = time.time()
        xgb_model, p_xgb, auc_xgb = train_xgb(df_tr, df_te, features)
        print(f'  XGB AUC={auc_xgb:.4f} [{time.time()-t0:.0f}s]')

        # FT: 100 features なら 16GB GPU で動く
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None
        print('  Training FT-Transformer...')
        t0 = time.time()
        try:
            ft_model, p_ft, auc_ft = train_ft_transformer(
                X_tr_s, y_tr.astype(np.float32),
                X_te_s, y_te.astype(np.float32),
                n_features=len(features),
                epochs=30, batch_size=512, lr=1e-3,
                patience=8, d_token=64, n_heads=4, n_layers=3,
                dropout=0.1, label=f'FT-{y_lo}',
            )
            print(f'  FT AUC={auc_ft:.4f} [{time.time()-t0:.0f}s]')
        except torch.OutOfMemoryError:
            print(f'  FT OOM skipped')
            p_ft = np.full(len(X_te), 0.3, dtype=np.float32)
            auc_ft = 0.5

        torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None
        print('  Training IntraRace...')
        t0 = time.time()
        df_tr_ir = df.loc[train_mask].copy()
        df_te_ir = df.loc[test_mask].copy()
        scaler_ir = StandardScaler()
        df_tr_ir[features] = scaler_ir.fit_transform(df_tr_ir[features].values.astype(np.float32))
        df_te_ir[features] = scaler_ir.transform(df_te_ir[features].values.astype(np.float32))
        ir_model, ir_val_dict, auc_ir_raw = train_intra_race(
            df_tr_ir, df_te_ir, features,
            epochs=30, patience=10, seed=42 + int(y_lo), d_model=128,
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

        aucs = [max(auc_lgb, 0.5), max(auc_xgb, 0.5),
                max(auc_ft, 0.5), max(auc_ir, 0.5)]
        if sum(aucs) > 2.0:
            w = [a - 0.5 for a in aucs]
            w = [x / sum(w) for x in w]
            p_ens_aw = w[0]*p_lgb + w[1]*p_xgb + w[2]*p_ft + w[3]*p_ir
            auc_ens_aw = roc_auc_score(y_te, p_ens_aw)
        else:
            auc_ens_aw = 0

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
            break

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top-n', type=int, default=100)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    print('=' * 60)
    print(f'V22 enhanced TOP {args.top_n} features 4-ens')
    print('=' * 60)
    print(f'DEVICE: {DEVICE}')

    t0 = time.time()
    top_features = load_top_n_features(args.top_n)
    print(f'[INFO] top {len(top_features)} features loaded from {TOP_FEATURES_JSON}')

    df, all_features = load_v22_enhanced_data()

    # filter to top features only (drop empty / missing)
    available_top = [f for f in top_features if f in df.columns]
    print(f'[INFO] available top features: {len(available_top)}/{len(top_features)}')

    folds = ([(24, 24)] if args.quick
             else [(20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)])

    results = run_wf_top(df, available_top, folds, quick=args.quick)

    aucs_grid = [r['auc_grid'] for r in results]
    aucs_lgb = [r['auc_lgb'] for r in results]
    aucs_ft = [r['auc_ft'] for r in results]
    aucs_ir = [r['auc_ir'] for r in results]
    print('\n' + '=' * 60)
    print('SUMMARY (TOP 100 features)')
    print('=' * 60)
    print(f'mean LGB: {np.mean(aucs_lgb):.4f}')
    print(f'mean FT:  {np.mean(aucs_ft):.4f}')
    print(f'mean IR:  {np.mean(aucs_ir):.4f}')
    print(f'mean Grid: {np.mean(aucs_grid):.4f}')
    print(f'V15 baseline: 0.8939')
    print(f'V22 base (177): 0.8800')
    print(f'V22 enhanced (282 FT skip): 0.8776')
    print(f'V22 enhanced top100: {np.mean(aucs_grid):.4f}')
    print(f'delta vs V15: {np.mean(aucs_grid) - 0.8939:+.4f}')

    out_path = os.path.join(MODEL_DIR, f'top{args.top_n}_wf_summary_{datetime.now():%Y%m%d_%H%M%S}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'mode': 'quick' if args.quick else 'full',
            'top_n': args.top_n,
            'available_features': len(available_top),
            'results': results,
            'mean_grid_auc': float(np.mean(aucs_grid)) if aucs_grid else 0,
            'v15_baseline': 0.8939,
            'elapsed_s': time.time() - t0,
        }, f, ensure_ascii=False, indent=2)
    print(f'\nsaved: {out_path}')


if __name__ == '__main__':
    main()
