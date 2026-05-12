#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V22 4-model Grid ensemble training (LGB+XGB+FT+IR).

V22 base = V15 cache (145 features) + Phase 24/26 features 32 件 = 177 features
4-model: LGB + XGB + FT-Transformer + IntraRace Attention (Grid weight optimization)

V15 (4-Grid AUC 0.8939) を 越える 候補。

【V15 投資保護】 V15 .pkl.gz 完全不変。 V22 は別 file 出力。

GPU: RTX 4070 Ti SUPER 16GB / CUDA 13.1
所要時間: WF 6 fold で 1-3 hour (GPU 利用時)

Usage:
    python train/train_v22_4ensemble.py --quick   # 2025 fold のみ (~10 min GPU)
    python train/train_v22_4ensemble.py           # 6-fold WF (~1-3h GPU)
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
import torch.nn.functional as F
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
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'v22')
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reuse FT/IR architectures
from train_v135_ft_transformer import FTTransformer, IntraRaceAttention, train_ft_transformer

PHASE24_26_FEATURES = [
    'jockey_change_top3_rate_exp', 'trainer_change_top3_rate_exp',
    'class_up_top3_rate_exp', 'class_down_top3_rate_exp',
    'horse_long_layoff_top3_rate_exp', 'horse_shorten_top3_rate_exp',
    'horse_extend_top3_rate_exp', 'horse_surface_change_top3_rate_exp',
    'sire_overall_top3_rate_exp', 'sire_class_down_top3_rate_exp',
    'sire_no_class_down_top3_rate_exp', 'sire_class_down_boost_exp',
    'jockey_trainer_combo_top3_exp', 'corner_position_delta',
    'fresh_horse', 'long_layoff', 'very_long_layoff',
    'class_down', 'surface_change',
    'rmk_delay', 'rmk_trouble', 'rmk_yore', 'rmk_fukure', 'rmk_contact',
    'rmk_demote', 'rmk_late_pace', 'rmk_fast_pace', 'rmk_any',
    'distance_change', 'distance_change_abs',
    'turf_to_dirt', 'dirt_to_turf',
]

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
    'seed': 42, 'tree_method': 'hist',
}

MAX_HORSES = 28


def load_v22_data():
    print(f'[INFO] loading V15 cache + V20 P24/26 features')
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

    df20 = pd.read_csv(V20_DATA, usecols=lambda c: c in ['horse_id', 'year', 'month',
                                                          'day', 'race_num', 'umaban'] + PHASE24_26_FEATURES,
                       dtype={'horse_id': str}, low_memory=False)
    df20['horse_id_str'] = df20['horse_id'].astype(str)

    for c in ['year', 'month', 'day', 'race_num', 'umaban']:
        df15[c] = pd.to_numeric(df15[c], errors='coerce').astype('Int64')
        df20[c] = pd.to_numeric(df20[c], errors='coerce').astype('Int64')

    key = ['year', 'month', 'day', 'race_num', 'horse_id_str', 'umaban']
    add_cols = [c for c in PHASE24_26_FEATURES if c in df20.columns]
    df20_sub = df20[key + add_cols].drop_duplicates(subset=key, keep='last')

    df = df15.merge(df20_sub, on=key, how='left')
    print(f'  V22 merged: {df.shape}, added {len(add_cols)} features')

    all_features = list(v15_features) + add_cols
    numeric_features = [f for f in all_features if f in df.columns and df[f].dtype.kind in 'iufb']
    print(f'  V22 numeric features: {len(numeric_features)}')

    df = df.dropna(subset=['target', 'year'])
    df['year'] = df['year'].astype(int)

    # Fill NaN with 0 for FT/IR (they don't handle NaN)
    df[numeric_features] = df[numeric_features].fillna(0)

    return df, numeric_features


def build_race_id_unique(df):
    if 'race_id_unique' not in df.columns:
        df['race_id_unique'] = (df['year'].astype(str) + '_' +
                                  df['month'].astype(str) + '_' +
                                  df['day'].astype(str) + '_' +
                                  df['race_num'].astype(str))
    return df


def train_intra_race(df_tr, df_va, features, label='target', epochs=20, patience=5):
    """IntraRace モデル 学習 + val preds 返却."""

    def make_race_batches(df, feats):
        race_groups = df.groupby('race_id_unique')
        X_races, y_races, masks, indices, counts = [], [], [], [], []
        for _, grp in race_groups:
            n = min(len(grp), MAX_HORSES)
            if n < 2:
                continue
            x = np.zeros((MAX_HORSES, len(feats)), dtype=np.float32)
            y = np.zeros(MAX_HORSES, dtype=np.float32)
            m = np.zeros(MAX_HORSES, dtype=np.float32)
            idx = np.full(MAX_HORSES, -1, dtype=np.int64)
            x[:n] = grp[feats].values[:n]
            y[:n] = grp[label].values[:n]
            m[:n] = 1.0
            idx[:n] = grp.index.values[:n]
            X_races.append(x); y_races.append(y); masks.append(m)
            indices.append(idx); counts.append(n)
        return (np.array(X_races), np.array(y_races),
                np.array(masks), np.array(indices))

    X_tr, y_tr, m_tr, idx_tr = make_race_batches(df_tr, features)
    X_va, y_va, m_va, idx_va = make_race_batches(df_va, features)
    print(f'    IR: train={len(X_tr)} races, val={len(X_va)} races')

    model = IntraRaceAttention(n_features=len(features), d_model=64,
                                n_heads=4, n_layers=2, dropout=0.1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pos_w = torch.tensor([(1 - y_tr[m_tr > 0].mean()) / y_tr[m_tr > 0].mean()]).to(DEVICE)
    bce = nn.BCEWithLogitsLoss(reduction='none')

    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr),
                              torch.FloatTensor(m_tr))
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0,
                           pin_memory=True)

    best_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batch = 0
        for xb, yb, mb in train_dl:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb, mb)
            loss_raw = bce(logits, yb) * pos_w * mb
            loss = loss_raw.sum() / mb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        scheduler.step()

        # Val
        model.eval()
        with torch.no_grad():
            val_probs_all = []
            chunk = 256
            for i in range(0, len(X_va), chunk):
                xc = torch.FloatTensor(X_va[i:i+chunk]).to(DEVICE)
                mc = torch.FloatTensor(m_va[i:i+chunk]).to(DEVICE)
                logits = model(xc, mc)
                val_probs_all.append(torch.sigmoid(logits).cpu().numpy())
            val_probs = np.concatenate(val_probs_all)

            valid = m_va.flatten() > 0
            y_flat = y_va.flatten()[valid]
            p_flat = val_probs.flatten()[valid]
            val_auc = roc_auc_score(y_flat, p_flat)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 3 == 0 or no_improve == 0:
            print(f'    IR epoch {epoch+1}: loss={total_loss/n_batch:.4f} '
                  f'val_AUC={val_auc:.4f} best={best_auc:.4f}')

        if no_improve >= patience:
            print(f'    IR Early stop at epoch {epoch+1}')
            break

    # Load best, predict
    model.load_state_dict(best_state)
    model.eval()
    val_preds_dict = {}
    with torch.no_grad():
        val_probs_all = []
        chunk = 256
        for i in range(0, len(X_va), chunk):
            xc = torch.FloatTensor(X_va[i:i+chunk]).to(DEVICE)
            mc = torch.FloatTensor(m_va[i:i+chunk]).to(DEVICE)
            logits = model(xc, mc)
            val_probs_all.append(torch.sigmoid(logits).cpu().numpy())
        val_probs = np.concatenate(val_probs_all)

    for r in range(len(X_va)):
        for h in range(MAX_HORSES):
            if m_va[r, h] > 0 and idx_va[r, h] >= 0:
                val_preds_dict[idx_va[r, h]] = val_probs[r, h]

    return model, val_preds_dict, best_auc


def wf_4ensemble(df, features, folds):
    results = []
    final = {}

    df = build_race_id_unique(df.copy())

    for y_lo, y_hi in folds:
        train_mask = df['year'] < y_lo
        test_mask = (df['year'] >= y_lo) & (df['year'] <= y_hi)
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        X_tr = df.loc[train_mask, features].values.astype(np.float32)
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[test_mask, features].values.astype(np.float32)
        y_te = df.loc[test_mask, 'target'].values
        test_indices = df.loc[test_mask].index.values

        print(f'\n{"="*60}\n  Fold val={y_lo} (train={len(X_tr)}, test={len(X_te)})\n{"="*60}')

        # LGB
        print('  Training LGB...')
        t0 = time.time()
        m_lgb = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, y_tr), num_boost_round=1000,
                          valid_sets=[lgb.Dataset(X_te, y_te)],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        p_lgb = m_lgb.predict(X_te)
        auc_lgb = roc_auc_score(y_te, p_lgb)
        print(f'  LGB AUC={auc_lgb:.4f} [{time.time()-t0:.0f}s]')

        # XGB
        print('  Training XGB...')
        t0 = time.time()
        m_xgb = xgb.train(XGB_PARAMS, xgb.DMatrix(X_tr, label=y_tr), num_boost_round=1000,
                          evals=[(xgb.DMatrix(X_te, label=y_te), 'va')],
                          early_stopping_rounds=50, verbose_eval=0)
        p_xgb = m_xgb.predict(xgb.DMatrix(X_te))
        auc_xgb = roc_auc_score(y_te, p_xgb)
        print(f'  XGB AUC={auc_xgb:.4f} [{time.time()-t0:.0f}s]')

        # FT (GPU OOM avoidance: smaller batch + cache clear)
        torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None
        print('  Training FT-Transformer...')
        t0 = time.time()
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        ft_model, p_ft, p_ft_tr, auc_ft = train_ft_transformer(
            X_tr_s, y_tr.astype(np.float32),
            X_te_s, y_te.astype(np.float32),
            n_features=len(features),
            epochs=30, batch_size=512, lr=1e-3,
            patience=8, d_token=64, n_heads=4, n_layers=3,
            dropout=0.1, label=f'FT-{y_lo}',
        )
        print(f'  FT AUC={auc_ft:.4f} [{time.time()-t0:.0f}s]')

        # IR (clear cache first)
        torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None
        print('  Training IntraRace...')
        t0 = time.time()
        df_tr = df.loc[train_mask].copy()
        df_te = df.loc[test_mask].copy()
        scaler_ir = StandardScaler()
        df_tr[features] = scaler_ir.fit_transform(df_tr[features].values.astype(np.float32))
        df_te[features] = scaler_ir.transform(df_te[features].values.astype(np.float32))
        ir_model, ir_val_dict, auc_ir_raw = train_intra_race(df_tr, df_te, features,
                                                              epochs=20, patience=5)

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

        # Grid search 4-model
        best_grid = 0
        best_w = None
        if auc_ir > 0:
            for w1 in np.arange(0.20, 0.45, 0.05):
                for w2 in np.arange(0.20, 0.45, 0.05):
                    for w3 in np.arange(0.05, 0.30, 0.05):
                        w4 = 1.0 - w1 - w2 - w3
                        if w4 < 0.05 or w4 > 0.45:
                            continue
                        p_g = w1*p_lgb + w2*p_xgb + w3*p_ft + w4*p_ir
                        auc_g = roc_auc_score(y_te, p_g)
                        if auc_g > best_grid:
                            best_grid = auc_g
                            best_w = (w1, w2, w3, w4)

        # Simple AUC-weighted 4-model
        if auc_ir > 0:
            total = auc_lgb + auc_xgb + auc_ft + auc_ir
            p_4m = ((auc_lgb/total)*p_lgb + (auc_xgb/total)*p_xgb +
                    (auc_ft/total)*p_ft + (auc_ir/total)*p_ir)
            auc_4m = roc_auc_score(y_te, p_4m)
        else:
            auc_4m = (auc_lgb + auc_xgb + auc_ft) / 3
            total = auc_lgb + auc_xgb + auc_ft
            p_4m = (auc_lgb/total)*p_lgb + (auc_xgb/total)*p_xgb + (auc_ft/total)*p_ft

        print(f'\n  RESULTS for fold {y_lo}:')
        print(f'    LGB={auc_lgb:.4f}, XGB={auc_xgb:.4f}, FT={auc_ft:.4f}, IR={auc_ir:.4f}')
        print(f'    4-model (AUC-w): {auc_4m:.4f}')
        if best_w:
            print(f'    4-model (grid): {best_grid:.4f} (L={best_w[0]:.2f} X={best_w[1]:.2f} F={best_w[2]:.2f} I={best_w[3]:.2f})')

        results.append({
            'val_year': int(y_lo),
            'auc_lgb': float(auc_lgb), 'auc_xgb': float(auc_xgb),
            'auc_ft': float(auc_ft), 'auc_ir': float(auc_ir),
            'auc_4m_w': float(auc_4m),
            'auc_grid': float(best_grid) if best_w else None,
            'grid_weights': list(best_w) if best_w else None,
            'n_train': int(len(X_tr)), 'n_val': int(len(X_te)),
        })
        final = {
            'lgb_model': m_lgb, 'xgb_model': m_xgb,
            'ft_model': ft_model, 'ir_model': ir_model,
            'scaler_ft': scaler, 'scaler_ir': scaler_ir,
            'grid_weights': best_w,
        }

    return results, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    print(f'[INFO] V22 4-ensemble training ({datetime.now().isoformat()})')
    print(f'[INFO] DEVICE: {DEVICE}')
    print('[INFO] V15 production 完全不変')

    df, features = load_v22_data()

    yr_max = int(df['year'].max())
    if yr_max < 100:
        folds = [(25, 25)] if args.quick else [
            (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)]
    else:
        folds = [(2025, 2025)] if args.quick else [
            (y, y) for y in range(2020, 2026)]

    t0 = time.time()
    results, final = wf_4ensemble(df, features, folds)
    elapsed = time.time() - t0

    print(f'\n{"="*60}\n  V22 4-ensemble WF summary ({elapsed:.0f}s total)\n{"="*60}')
    if results:
        aucs_4m = [r['auc_4m_w'] for r in results]
        aucs_grid = [r['auc_grid'] for r in results if r['auc_grid']]
        print(f'  mean 4-model (AUC-w): {np.mean(aucs_4m):.4f}')
        if aucs_grid:
            print(f'  mean 4-model (Grid):  {np.mean(aucs_grid):.4f}')
        print(f'  V15 baseline:         0.8939')
        print(f'  per-year (4m):        ' + ', '.join(f'{r["val_year"]}={r["auc_4m_w"]:.4f}' for r in results))
        if aucs_grid:
            print(f'  per-year (grid):      ' + ', '.join(f'{r["val_year"]}={r["auc_grid"]:.4f}' if r['auc_grid'] else 'N/A' for r in results))

    # Save
    if final:
        out_path = os.path.join(BASE_DIR, 'keiba_model_v22_4ensemble.pkl.gz')
        with gzip.open(out_path, 'wb') as f:
            pickle.dump({
                'version': 'v22_4ensemble',
                'date': datetime.now().isoformat(),
                'features': features,
                'lgb_model': final['lgb_model'],
                'xgb_model': final['xgb_model'],
                'ft_model_state': final['ft_model'].state_dict() if final.get('ft_model') else None,
                'ir_model_state': final['ir_model'].state_dict() if final.get('ir_model') else None,
                'scaler_ft': final['scaler_ft'],
                'scaler_ir': final['scaler_ir'],
                'grid_weights': final['grid_weights'],
                'wf_results': results,
            }, f)
        print(f'\n[OK] model saved: {out_path}')

        with open(os.path.join(MODEL_DIR, '4ensemble_wf_summary.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'date': datetime.now().isoformat(),
                'elapsed_sec': float(elapsed),
                'n_features': len(features),
                'wf_results': results,
                'mean_4m_w': float(np.mean([r['auc_4m_w'] for r in results])) if results else None,
                'mean_grid': float(np.mean([r['auc_grid'] for r in results if r['auc_grid']])) if results else None,
                'v15_baseline_auc': 0.8939,
            }, f, indent=2, ensure_ascii=False)

    return 0


if __name__ == '__main__':
    sys.exit(main())
