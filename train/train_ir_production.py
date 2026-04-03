#!/usr/bin/env python3
"""IntraRace Attention 本番モデル学習・保存

2021-2025年データで学習し、既存pkl.gzにIRモデルを追加保存する。
predict_core.pyのpredict_race()で推論に使用できる形式。

Usage:
    python train/train_ir_production.py
"""

import os
import sys
import time
import gzip
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

from train_v135_ft_transformer import (
    build_v134_dataframe, get_v134_features, fill_defaults,
    IntraRaceAttention,
)
from train_v135b_intra_ensemble import build_race_id, train_intra_race_with_preds

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_HORSES = 28

# Target ensemble weights
TARGET_WEIGHTS = {'lgb': 0.25, 'xgb': 0.30, 'ft': 0.15, 'ir': 0.30}


def main():
    t0 = time.time()
    print("=" * 70)
    print("  IntraRace Attention - Production Model Training")
    print(f"  Device: {DEVICE}")
    print(f"  Target weights: {TARGET_WEIGHTS}")
    print("=" * 70)

    # =========================================================
    # 1. Build features (same pipeline as v13.5b)
    # =========================================================
    df, sire_map, bms_map = build_v134_dataframe()
    features, jrdb_sel = get_v134_features()
    df = fill_defaults(df, features)
    df = build_race_id(df)
    print(f"  Data: {len(df)} rows, {len(features)} features")
    print(f"  Races: {df['race_id_unique'].nunique()}")

    # =========================================================
    # 2. Train/Val split: 2021-2024 train, 2025 val
    # =========================================================
    df['year_int'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    # Use 2-digit year for JV data
    train_mask = (df['year_int'] >= 21) & (df['year_int'] <= 24)
    val_mask = (df['year_int'] == 25)

    if train_mask.sum() < 1000:
        # Try 4-digit years
        train_mask = (df['year_int'] >= 2021) & (df['year_int'] <= 2024)
        val_mask = (df['year_int'] == 2025)

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    print(f"\n  Train: {len(df_train)} rows ({df_train['race_id_unique'].nunique()} races)")
    print(f"  Val:   {len(df_val)} rows ({df_val['race_id_unique'].nunique()} races)")

    if len(df_train) < 1000 or len(df_val) < 100:
        print("[ERROR] Insufficient data")
        return

    # =========================================================
    # 3. Scale features
    # =========================================================
    scaler_ir = StandardScaler()
    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_train_scaled[features] = scaler_ir.fit_transform(
        df_train[features].values.astype(np.float32))
    df_val_scaled[features] = scaler_ir.transform(
        df_val[features].values.astype(np.float32))

    # =========================================================
    # 4. Train IntraRace Attention
    # =========================================================
    print("\n[2] Training IntraRace Attention...")
    ir_model, val_preds_dict, tr_preds_dict, tr_auc, val_auc = \
        train_intra_race_with_preds(
            df_train_scaled, df_val_scaled, features,
            epochs=30, batch_size=64, lr=1e-3, patience=8,
            d_model=64, n_heads=4, n_layers=2, dropout=0.1,
        )
    gap = tr_auc - val_auc
    print(f"  IR Train AUC: {tr_auc:.4f}")
    print(f"  IR Val AUC:   {val_auc:.4f}")
    print(f"  Gap:          {gap:.4f} ({'OK' if gap < 0.05 else 'WARNING'})")

    # =========================================================
    # 5. Now retrain on ALL data (2021-2025) for production
    # =========================================================
    print("\n[3] Retraining on full data (2021-2025) for production...")
    all_mask = train_mask | val_mask
    df_all = df[all_mask].copy()
    df_all_scaled = df_all.copy()

    # Fit scaler on all data
    scaler_prod = StandardScaler()
    df_all_scaled[features] = scaler_prod.fit_transform(
        df_all[features].values.astype(np.float32))

    # Use last 10% as pseudo-validation for early stopping
    n_all = len(df_all_scaled)
    n_val_pseudo = max(int(n_all * 0.1), 1000)
    df_prod_train = df_all_scaled.iloc[:-n_val_pseudo].copy()
    df_prod_val = df_all_scaled.iloc[-n_val_pseudo:].copy()

    prod_model, _, _, prod_tr_auc, prod_val_auc = \
        train_intra_race_with_preds(
            df_prod_train, df_prod_val, features,
            epochs=30, batch_size=64, lr=1e-3, patience=8,
            d_model=64, n_heads=4, n_layers=2, dropout=0.1,
        )
    print(f"  Production Train AUC: {prod_tr_auc:.4f}")
    print(f"  Production Val AUC:   {prod_val_auc:.4f}")

    # =========================================================
    # 6. Save IR model into existing pkl.gz files
    # =========================================================
    ir_config = {
        'n_features': len(features),
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff_mult': 2,
        'dropout': 0.1,
    }
    ir_state = {k: v.cpu() for k, v in prod_model.state_dict().items()}
    ir_scaler_mean = scaler_prod.mean_.astype(np.float32)
    ir_scaler_scale = scaler_prod.scale_.astype(np.float32)

    # Update both Pattern A and Pattern B pkl.gz files
    updated_weights = TARGET_WEIGHTS.copy()

    for pkl_name in ['keiba_model_v135_central_live.pkl.gz',
                     'keiba_model_v135_central.pkl.gz']:
        pkl_path = os.path.join(BASE_DIR, pkl_name)
        if not os.path.exists(pkl_path):
            print(f"  [SKIP] {pkl_name} not found")
            continue

        print(f"\n[4] Updating {pkl_name}...")
        with gzip.open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # Add IR model data
        data['ir_model_state'] = ir_state
        data['ir_model_config'] = ir_config
        data['ir_scaler_mean'] = ir_scaler_mean
        data['ir_scaler_scale'] = ir_scaler_scale

        # Update ensemble weights
        data['ensemble_weights'] = updated_weights
        data['ensemble_type'] = 'LGB+XGB+FT+IR'

        # Save
        with gzip.open(pkl_path, 'wb') as f:
            pickle.dump(data, f, protocol=4)

        # Verify
        with gzip.open(pkl_path, 'rb') as f:
            verify = pickle.load(f)
        assert 'ir_model_state' in verify, "IR model state not saved!"
        assert 'ir_model_config' in verify, "IR model config not saved!"
        print(f"  Saved: {pkl_name}")
        print(f"    Keys: {list(verify.keys())}")
        print(f"    Ensemble weights: {verify['ensemble_weights']}")
        print(f"    IR state params: {len(verify['ir_model_state'])}")

    # =========================================================
    # 7. Save training results
    # =========================================================
    result = {
        'val_auc': float(val_auc),
        'tr_auc': float(tr_auc),
        'gap': float(gap),
        'prod_val_auc': float(prod_val_auc),
        'prod_tr_auc': float(prod_tr_auc),
        'n_train': int(len(df_train)),
        'n_val': int(len(df_val)),
        'n_features': len(features),
        'ensemble_weights': updated_weights,
        'ir_config': ir_config,
        'device': str(DEVICE),
        'timestamp': datetime.now().isoformat(),
    }
    rpath = os.path.join(DATA_DIR, 'ir_production_training.json')
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  Results: {rpath}")

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*70}")
    print(f"  DONE in {elapsed:.1f} min")
    print(f"  IR Val AUC: {val_auc:.4f}, Gap: {gap:.4f}")
    print(f"  Weights: {updated_weights}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
