#!/usr/bin/env python3
"""v14.1 本番モデル学習・保存

v14.1 PACI Tier A 7特徴量を含む131特徴量（Pattern A）でLGB+XGB+FT+IRを
全データ(2015-2025)で学習し、pkl.gzに保存する。

Pattern B = Pattern A(131) + TYB live(5) = 136特徴量
"""

import os
import sys
import time
import pickle
import gzip
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

from train_v141_paci_tierA import (
    build_v141_dataframe, get_v141_features, fill_v141_defaults,
    V141_NEW_FEATURES, V141_DEFAULTS,
)
from train_v135_ft_transformer import (
    FTTransformer, IntraRaceAttention, train_ft_transformer,
    DEVICE, LGB_PARAMS, XGB_PARAMS,
)
from train_v135b_intra_ensemble import (
    build_race_id, train_intra_race_with_preds, MAX_HORSES,
)
from jrdb_features import JRDB_DEFAULTS, JRDB_LIVE_FEATURES, PACI_TIER_A_DEFAULTS


def main():
    t0 = time.time()
    print("=" * 70)
    print("  v14.1 Production Model Training")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # =========================================================
    # 1. Build dataframe
    # =========================================================
    df, sire_map, bms_map = build_v141_dataframe()
    features_a, jrdb_sel = get_v141_features()  # Pattern A: 131 features
    df = fill_v141_defaults(df, features_a)
    df = build_race_id(df)

    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses'] >= 5].copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)

    # Pattern B = Pattern A + TYB live features
    features_b = features_a + list(JRDB_LIVE_FEATURES)

    # Fill live features with defaults for training
    for f in JRDB_LIVE_FEATURES:
        if f not in df.columns:
            df[f] = JRDB_DEFAULTS.get(f, 0)
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(JRDB_DEFAULTS.get(f, 0))

    print(f"\n  Data: {len(df)} rows")
    print(f"  Pattern A features: {len(features_a)}")
    print(f"  Pattern B features: {len(features_b)}")

    # =========================================================
    # 2. Train/Val split (2025 as validation)
    # =========================================================
    val_year = 25
    train_mask = df['year'] < val_year
    val_mask = df['year'] == val_year

    X_tr = df.loc[train_mask, features_b].values.astype(np.float32)
    y_tr = df.loc[train_mask, 'target'].values
    X_va = df.loc[val_mask, features_b].values.astype(np.float32)
    y_va = df.loc[val_mask, 'target'].values

    print(f"\n  Train: {len(X_tr)} (2015-2024)")
    print(f"  Val:   {len(X_va)} (2025)")

    # =========================================================
    # 3. Train LightGBM (Pattern B)
    # =========================================================
    print("\n[1] Training LightGBM...")
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    lgb_model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                          valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    lgb_auc = roc_auc_score(y_va, lgb_model.predict(X_va))
    print(f"  LGB Val AUC: {lgb_auc:.4f}")

    # =========================================================
    # 4. Train XGBoost (Pattern B)
    # =========================================================
    print("\n[2] Training XGBoost...")
    dxtr = xgb.DMatrix(X_tr, label=y_tr)
    dxva = xgb.DMatrix(X_va, label=y_va)
    xgb_model = xgb.train(XGB_PARAMS, dxtr, num_boost_round=1000,
                           evals=[(dxva, 'valid')],
                           early_stopping_rounds=50, verbose_eval=100)
    xgb_auc = roc_auc_score(y_va, xgb_model.predict(dxva))
    print(f"  XGB Val AUC: {xgb_auc:.4f}")

    # =========================================================
    # 5. Train FT-Transformer (Pattern B)
    # =========================================================
    print("\n[3] Training FT-Transformer...")
    scaler_ft = StandardScaler()
    X_tr_s = scaler_ft.fit_transform(X_tr)
    X_va_s = scaler_ft.transform(X_va)

    ft_model, p_ft, _, ft_auc = train_ft_transformer(
        X_tr_s, y_tr.astype(np.float32),
        X_va_s, y_va.astype(np.float32),
        n_features=len(features_b),
        epochs=50, batch_size=2048, lr=1e-3,
        patience=10, d_token=64, n_heads=4, n_layers=3,
        dropout=0.1, label='FT-prod',
    )
    print(f"  FT Val AUC: {ft_auc:.4f}")

    # =========================================================
    # 6. Train IntraRace Attention (Pattern A only)
    # =========================================================
    print("\n[4] Training IntraRace Attention (Pattern A)...")
    df_tr = df.loc[train_mask].copy()
    df_va = df.loc[val_mask].copy()

    scaler_ir = StandardScaler()
    df_tr_scaled = df_tr.copy()
    df_va_scaled = df_va.copy()
    df_tr_scaled[features_a] = scaler_ir.fit_transform(
        df_tr[features_a].values.astype(np.float32))
    df_va_scaled[features_a] = scaler_ir.transform(
        df_va[features_a].values.astype(np.float32))

    ir_model, ir_val_dict, ir_tr_dict, ir_tr_auc, ir_val_auc = \
        train_intra_race_with_preds(
            df_tr_scaled, df_va_scaled, features_a,
            epochs=30, batch_size=64, lr=1e-3, patience=8,
            d_model=64, n_heads=4, n_layers=2, dropout=0.1,
        )
    print(f"  IR Val AUC: {ir_val_auc:.4f}")

    # =========================================================
    # 7. Retrain on ALL data for production
    # =========================================================
    print("\n[5] Retraining all models on full data (2015-2025)...")

    X_all = df[features_b].values.astype(np.float32)
    y_all = df['target'].values

    # Use last 10% as pseudo-val for early stopping
    n = len(X_all)
    n_pseudo = max(int(n * 0.1), 1000)

    # LGB production
    print("  LGB production...")
    dtrain_all = lgb.Dataset(X_all[:-n_pseudo], label=y_all[:-n_pseudo])
    dval_pseudo = lgb.Dataset(X_all[-n_pseudo:], label=y_all[-n_pseudo:], reference=dtrain_all)
    lgb_prod = lgb.train(LGB_PARAMS, dtrain_all, num_boost_round=1000,
                         valid_sets=[dval_pseudo],
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    # XGB production
    print("  XGB production...")
    dxall = xgb.DMatrix(X_all[:-n_pseudo], label=y_all[:-n_pseudo])
    dxpseudo = xgb.DMatrix(X_all[-n_pseudo:], label=y_all[-n_pseudo:])
    xgb_prod = xgb.train(XGB_PARAMS, dxall, num_boost_round=1000,
                          evals=[(dxpseudo, 'valid')],
                          early_stopping_rounds=50, verbose_eval=0)

    # FT production
    print("  FT production...")
    scaler_ft_prod = StandardScaler()
    X_all_s = scaler_ft_prod.fit_transform(X_all)
    ft_prod, _, _, _ = train_ft_transformer(
        X_all_s[:-n_pseudo], y_all[:-n_pseudo].astype(np.float32),
        X_all_s[-n_pseudo:], y_all[-n_pseudo:].astype(np.float32),
        n_features=len(features_b),
        epochs=50, batch_size=2048, lr=1e-3,
        patience=10, d_token=64, n_heads=4, n_layers=3,
        dropout=0.1, label='FT-prod-full',
    )

    # IR production (Pattern A)
    print("  IR production...")
    scaler_ir_prod = StandardScaler()
    df_all = df.copy()
    df_all_scaled = df_all.copy()
    df_all_scaled[features_a] = scaler_ir_prod.fit_transform(
        df_all[features_a].values.astype(np.float32))

    n_ir_pseudo = max(int(len(df_all_scaled) * 0.1), 1000)
    df_ir_train = df_all_scaled.iloc[:-n_ir_pseudo].copy()
    df_ir_val = df_all_scaled.iloc[-n_ir_pseudo:].copy()

    ir_prod, _, _, _, ir_prod_auc = \
        train_intra_race_with_preds(
            df_ir_train, df_ir_val, features_a,
            epochs=30, batch_size=64, lr=1e-3, patience=8,
            d_model=64, n_heads=4, n_layers=2, dropout=0.1,
        )
    print(f"  IR production val AUC: {ir_prod_auc:.4f}")

    # =========================================================
    # 8. Save to pkl.gz
    # =========================================================
    print("\n[6] Saving models...")

    # Ensemble weights (from WF grid search typical)
    ensemble_weights = {'lgb': 0.25, 'xgb': 0.25, 'ft': 0.15, 'ir': 0.35}

    ft_config = {
        'n_features': len(features_b),
        'd_token': 64,
        'n_heads': 4,
        'n_layers': 3,
        'dropout': 0.1,
    }
    ir_config = {
        'n_features': len(features_a),
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff_mult': 2,
        'dropout': 0.1,
    }

    data = {
        # Models
        'model': lgb_prod,
        'xgb_model': xgb_prod,
        'mlp_model': None,
        'mlp_scaler': None,
        'ft_model_state': {k: v.cpu() for k, v in ft_prod.state_dict().items()},
        'ft_model_config': ft_config,
        'ft_scaler_mean': scaler_ft_prod.mean_.astype(np.float32),
        'ft_scaler_scale': scaler_ft_prod.scale_.astype(np.float32),
        'ir_model_state': {k: v.cpu() for k, v in ir_prod.state_dict().items()},
        'ir_model_config': ir_config,
        'ir_scaler_mean': scaler_ir_prod.mean_.astype(np.float32),
        'ir_scaler_scale': scaler_ir_prod.scale_.astype(np.float32),
        # Features & Config
        'features': features_b,
        'version': 'v141_live',
        'auc': float(lgb_auc),
        'wf_auc': 0.8856,
        'leak_free': False,
        'leak_pattern': 'B',
        'leak_removed': [],
        'sire_map': sire_map,
        'bms_map': bms_map,
        'n_top_encode': 100,
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'central',
        'ensemble_weights': ensemble_weights,
        'ensemble_type': 'LGB+XGB+FT+IR',
        'is_live': True,
        'course_map': None,
        'jrdb_features': jrdb_sel,
    }

    # Save Pattern B (live)
    pkl_live = os.path.join(BASE_DIR, 'keiba_model_v135_central_live.pkl.gz')
    with gzip.open(pkl_live, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f"  Saved: {pkl_live}")

    # Verify
    with gzip.open(pkl_live, 'rb') as f:
        verify = pickle.load(f)
    print(f"    Version: {verify['version']}")
    print(f"    Features: {len(verify['features'])}")
    print(f"    WF AUC: {verify['wf_auc']}")
    print(f"    Ensemble: {verify['ensemble_weights']}")
    print(f"    Has FT: {'ft_model_state' in verify}")
    print(f"    Has IR: {'ir_model_state' in verify}")

    # Save Pattern A (leak-free) version
    data_a = data.copy()
    data_a['features'] = features_a
    data_a['version'] = 'v141'
    data_a['leak_free'] = True
    data_a['leak_pattern'] = 'A'
    data_a['is_live'] = False
    data_a['ft_model_config'] = {**ft_config, 'n_features': len(features_a)}
    # Retrain FT for Pattern A
    print("\n  Training FT for Pattern A...")
    scaler_ft_a = StandardScaler()
    X_a = df[features_a].values.astype(np.float32)
    X_a_s = scaler_ft_a.fit_transform(X_a)
    ft_a, _, _, _ = train_ft_transformer(
        X_a_s[:-n_pseudo], y_all[:-n_pseudo].astype(np.float32),
        X_a_s[-n_pseudo:], y_all[-n_pseudo:].astype(np.float32),
        n_features=len(features_a),
        epochs=50, batch_size=2048, lr=1e-3,
        patience=10, d_token=64, n_heads=4, n_layers=3,
        dropout=0.1, label='FT-A-prod',
    )
    data_a['ft_model_state'] = {k: v.cpu() for k, v in ft_a.state_dict().items()}
    data_a['ft_scaler_mean'] = scaler_ft_a.mean_.astype(np.float32)
    data_a['ft_scaler_scale'] = scaler_ft_a.scale_.astype(np.float32)
    # Retrain LGB/XGB for Pattern A
    print("  Training LGB/XGB for Pattern A...")
    X_a_all = df[features_a].values.astype(np.float32)
    dtrain_a = lgb.Dataset(X_a_all[:-n_pseudo], label=y_all[:-n_pseudo])
    dval_a = lgb.Dataset(X_a_all[-n_pseudo:], label=y_all[-n_pseudo:], reference=dtrain_a)
    lgb_a = lgb.train(LGB_PARAMS, dtrain_a, num_boost_round=1000,
                      valid_sets=[dval_a],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    dxa = xgb.DMatrix(X_a_all[:-n_pseudo], label=y_all[:-n_pseudo])
    dxva_a = xgb.DMatrix(X_a_all[-n_pseudo:], label=y_all[-n_pseudo:])
    xgb_a = xgb.train(XGB_PARAMS, dxa, num_boost_round=1000,
                       evals=[(dxva_a, 'valid')],
                       early_stopping_rounds=50, verbose_eval=0)
    data_a['model'] = lgb_a
    data_a['xgb_model'] = xgb_a

    pkl_a = os.path.join(BASE_DIR, 'keiba_model_v135_central.pkl.gz')
    with gzip.open(pkl_a, 'wb') as f:
        pickle.dump(data_a, f, protocol=4)
    print(f"  Saved: {pkl_a}")

    with gzip.open(pkl_a, 'rb') as f:
        verify_a = pickle.load(f)
    print(f"    Version: {verify_a['version']}")
    print(f"    Features: {len(verify_a['features'])}")

    elapsed = (time.time() - t0) / 60
    print(f"\n  Total time: {elapsed:.1f} min")
    print("=" * 70)


if __name__ == '__main__':
    main()
