#!/usr/bin/env python
"""V20 4-model ensemble training - Phase 15 (2026-05-10).

V15 base 150 features に Phase 11/12/13 計 57 features を追加した V20 候補 model。
4-model: LGB (GPU) + XGB (GPU) + FT-Transformer (GPU) + IntraRace Attention (GPU)

★★★ 重要 caveat (2026-05-10 時点) ★★★
Phase 11 (15) / Phase 12 (17) / Phase 13 (25) の追加 features は全て **constant default fill**。
LGB/XGB は constant column を自動 drop するため、 V20 候補 = 実質 V15 retrain (145 features)。
真の V20 (期待 AUC 0.91+) には:
  1. Phase 11 JRDB 実 data lookup 実装 (5/12+)
  2. Phase 12 JV-Link 実 data fetch 実装 (5/24+)
  3. Phase 13 netkeiba master 実 DOM 検証 + scrape (5/11+)
が必要。

GPU 環境:
  RTX 4070 Ti SUPER 16GB / CUDA 13.1 / torch 2.11.0+cu126

Usage:
  # 短縮版 (LGB+XGB only、 ~10 min on GPU):
  python train/train_v20_ensemble.py --quick

  # フル 4-model ensemble (~4-8h on GPU):
  python train/train_v20_ensemble.py --full

  # WF 評価のみ (既存 model load):
  python train/train_v20_ensemble.py --eval-only
"""
from __future__ import annotations
import os
import sys
import json
import time
import gzip
import pickle
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

DATA_DIR = os.path.join(BASE_DIR, 'data')
V20_DATA_DIR = os.path.join(DATA_DIR, 'v20')
V20_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'v20')
os.makedirs(V20_DATA_DIR, exist_ok=True)
os.makedirs(V20_MODEL_DIR, exist_ok=True)

CACHE_PATH = os.path.join(DATA_DIR, '_v15_optuna_df_cache.pkl.gz')
V15_MODEL_PATH = os.path.join(BASE_DIR, 'keiba_model_v15_central.pkl.gz')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================================
# Phase 11/12/13 features (V15 = 145, V20 candidate = 145+15+17+25 = 202)
# =========================================================================

PHASE11_FEATURES = [
    'gaika_id_enc', 'gaika_top3r_3r', 'gaika_winrate', 'gaika_dist_winrate',
    'odds_change_3h_v18', 'odds_change_30m_v18', 'popularity_shift_v18', 'odds_volatility_v18',
    'jockey_dist_winrate', 'jockey_track_winrate', 'jockey_class_winrate', 'jockey_x_trainer_wr',
    'return_horse_score', 'paddock_eval_v18', 'saddle_room_score',
]
PHASE12_FEATURES = [
    'jv_tansho_odds_open', 'jv_fukusho_low_open', 'jv_umaren_top_odds', 'jv_trio_top_odds',
    'jv_race_class_detail', 'jv_prize_structure_total', 'jv_entry_condition_enc',
    'jv_lap_first3f_pred', 'jv_lap_last3f_pred', 'jv_race_pace_index',
    'jv_baba_moisture', 'jv_baba_difference', 'jv_weather_change_score',
    'jv_sire_dist_apt_score', 'jv_dam_sire_apt_score', 'jv_sire_surface_apt_score', 'jv_ped_score_blend',
]
PHASE13_FEATURES = [
    'master_pace_pred', 'master_pred_winner_score', 'master_pred_first3f_avg',
    'master_pred_last3f_avg', 'master_pred_finish_time', 'master_horse_aitenkai_score',
    'master_horse_pred_pos',
    'master_haran_score', 'master_top_pop_trust', 'master_haran_meter',
    'master_horse_lap_avg_first3f', 'master_horse_lap_avg_last3f', 'master_horse_lap_best_last3f',
    'master_horse_lap_consistency', 'master_horse_lap_best_3f', 'master_horse_lap_pos_change_avg',
    'master_horse_lap_finish_speed', 'master_horse_lap_acc_phase', 'master_horse_lap_dec_phase',
    'master_horse_lap_distance_factor',
    'master_track_inner_outer_bias', 'master_track_front_back_bias', 'master_track_corner_bias',
    'master_track_pace_bias_score', 'master_track_today_severity',
]
PHASE_DEFAULTS = {
    # phase 11
    'gaika_id_enc': 0, 'gaika_top3r_3r': 0.33, 'gaika_winrate': 0.20, 'gaika_dist_winrate': 0.20,
    'odds_change_3h_v18': 0.0, 'odds_change_30m_v18': 0.0, 'popularity_shift_v18': 0, 'odds_volatility_v18': 0.0,
    'jockey_dist_winrate': 0.10, 'jockey_track_winrate': 0.10, 'jockey_class_winrate': 0.10, 'jockey_x_trainer_wr': 0.15,
    'return_horse_score': 0.0, 'paddock_eval_v18': 0.0, 'saddle_room_score': 0.0,
    # phase 12
    'jv_tansho_odds_open': 10.0, 'jv_fukusho_low_open': 2.0, 'jv_umaren_top_odds': 30.0, 'jv_trio_top_odds': 100.0,
    'jv_race_class_detail': 0, 'jv_prize_structure_total': 5000, 'jv_entry_condition_enc': 0,
    'jv_lap_first3f_pred': 36.0, 'jv_lap_last3f_pred': 36.0, 'jv_race_pace_index': 1.0,
    'jv_baba_moisture': -1.0, 'jv_baba_difference': 0.0, 'jv_weather_change_score': 0,
    'jv_sire_dist_apt_score': 0.5, 'jv_dam_sire_apt_score': 0.5, 'jv_sire_surface_apt_score': 0.5, 'jv_ped_score_blend': 0.5,
    # phase 13
    'master_pace_pred': 1, 'master_pred_winner_score': 50.0,
    'master_pred_first3f_avg': 35.5, 'master_pred_last3f_avg': 35.5,
    'master_pred_finish_time': 100.0, 'master_horse_aitenkai_score': 50.0,
    'master_horse_pred_pos': 9,
    'master_haran_score': 50.0, 'master_top_pop_trust': 50.0, 'master_haran_meter': 3,
    'master_horse_lap_avg_first3f': 35.5, 'master_horse_lap_avg_last3f': 35.5,
    'master_horse_lap_best_last3f': 34.5, 'master_horse_lap_consistency': 1.0,
    'master_horse_lap_best_3f': 34.0, 'master_horse_lap_pos_change_avg': 0.0,
    'master_horse_lap_finish_speed': 12.0, 'master_horse_lap_acc_phase': 1,
    'master_horse_lap_dec_phase': 1, 'master_horse_lap_distance_factor': 0.5,
    'master_track_inner_outer_bias': 0.0, 'master_track_front_back_bias': 0.0,
    'master_track_corner_bias': 0.0, 'master_track_pace_bias_score': 0.0,
    'master_track_today_severity': 50.0,
}


# =========================================================================
# Hyperparameters
# =========================================================================

LGB_PARAMS_GPU = {
    'objective': 'binary', 'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_child_samples': 50,
    'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'verbose': -1, 'seed': 42,
    'device': 'gpu',
    'gpu_use_dp': False,
}
LGB_PARAMS_CPU = {**LGB_PARAMS_GPU, 'device': 'cpu'}

XGB_PARAMS_GPU = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'max_depth': 6, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 50,
    'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'seed': 42,
    'tree_method': 'hist',
    'device': 'cuda',
}
XGB_PARAMS_CPU = {**XGB_PARAMS_GPU, 'device': 'cpu'}


# =========================================================================
# Data loading
# =========================================================================

def load_v15_cache() -> Tuple[pd.DataFrame, List[str]]:
    print(f"[load] {CACHE_PATH}")
    with gzip.open(CACHE_PATH, 'rb') as f:
        obj = pickle.load(f)
    df, feats = obj['df'], obj['features']
    print(f"[load] shape {df.shape}, V15 features {len(feats)}")
    return df, feats


def add_phase_features(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 11/12/13 57 features を default で埋める (skeleton 段階)."""
    for f, v in PHASE_DEFAULTS.items():
        if f not in df.columns:
            df[f] = v
    return df


def split_train_val(df: pd.DataFrame, train_years: List[int], val_years: List[int]):
    train_mask = df['year'].isin(train_years)
    val_mask = df['year'].isin(val_years)
    return df[train_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)


# =========================================================================
# LGB / XGB training
# =========================================================================

def train_lgb_gpu(X_tr, y_tr, X_va, y_va, num_round=1000, early_stop=50):
    try:
        params = LGB_PARAMS_GPU
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
        model = lgb.train(
            params, dtr, num_boost_round=num_round, valid_sets=[dva],
            callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(100)],
        )
        return model
    except Exception as e:
        print(f"[lgb gpu] {e} → fallback CPU")
        params = LGB_PARAMS_CPU
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
        return lgb.train(
            params, dtr, num_boost_round=num_round, valid_sets=[dva],
            callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(100)],
        )


def train_xgb_gpu(X_tr, y_tr, X_va, y_va, num_round=1000, early_stop=50):
    try:
        params = XGB_PARAMS_GPU
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_va, label=y_va)
        model = xgb.train(
            params, dtr, num_boost_round=num_round,
            evals=[(dva, 'val')], early_stopping_rounds=early_stop, verbose_eval=100,
        )
        return model
    except Exception as e:
        print(f"[xgb gpu] {e} → fallback CPU")
        params = XGB_PARAMS_CPU
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_va, label=y_va)
        return xgb.train(
            params, dtr, num_boost_round=num_round,
            evals=[(dva, 'val')], early_stopping_rounds=early_stop, verbose_eval=100,
        )


# =========================================================================
# FT-Transformer (skeleton - 動作確認用、 フル 学習は --full で)
# =========================================================================

class FTTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.feature_embed = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, F)
        B, F_ = x.shape
        emb = x.unsqueeze(-1) * self.feature_embed.unsqueeze(0)  # (B, F, D)
        cls = self.cls_token.expand(B, -1, -1)
        emb = torch.cat([cls, emb], dim=1)
        out = self.encoder(emb)
        return self.head(out[:, 0]).squeeze(-1)


def train_ft_quick(X_tr, y_tr, X_va, y_va, epochs=10, batch_size=512, lr=1e-3):
    """FT-Transformer 短縮 学習 (epochs=10、 ~3-5 min on GPU)."""
    X_tr_t = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_tr).to(DEVICE)
    X_va_t = torch.FloatTensor(X_va).to(DEVICE)
    y_va_t = torch.FloatTensor(y_va).to(DEVICE)

    model = FTTransformer(n_features=X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_weight = torch.tensor([(1 - y_tr.mean()) / max(y_tr.mean(), 0.01)]).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = 0.0
    ds = TensorDataset(X_tr_t, y_tr_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.size(0)

        model.eval()
        with torch.no_grad():
            va_logits = model(X_va_t).cpu().numpy()
        try:
            auc = roc_auc_score(y_va, va_logits)
        except Exception:
            auc = 0.0
        print(f"  [FT epoch {ep+1}/{epochs}] loss={ep_loss/len(X_tr):.4f} val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(X_va_t)).cpu().numpy()
    return model, preds, best_auc


# =========================================================================
# Ensemble + WF Eval
# =========================================================================

def ensemble_predict(lgb_p, xgb_p, ft_p=None, ir_p=None, w=None):
    if w is None:
        if ft_p is None and ir_p is None:
            w = {'lgb': 0.55, 'xgb': 0.45}
            return w['lgb'] * lgb_p + w['xgb'] * xgb_p
        w = {'lgb': 0.30, 'xgb': 0.30, 'ft': 0.10, 'ir': 0.30}
    arr = w['lgb'] * lgb_p + w['xgb'] * xgb_p
    if ft_p is not None:
        arr = arr + w.get('ft', 0.10) * ft_p
    if ir_p is not None:
        arr = arr + w.get('ir', 0.30) * ir_p
    return arr


def quick_train_eval(df: pd.DataFrame, features: List[str],
                     train_years=(22, 23, 24), val_years=(25,),
                     run_ft: bool = False) -> Dict[str, Any]:
    df_tr, df_va = split_train_val(df, list(train_years), list(val_years))
    print(f"[split] train {len(df_tr):,} ({list(train_years)}) | val {len(df_va):,} ({list(val_years)})")

    # 数値化 + nan fill
    for c in features:
        if c not in df_tr.columns:
            df_tr[c] = 0
            df_va[c] = 0
        df_tr[c] = pd.to_numeric(df_tr[c], errors='coerce').fillna(0)
        df_va[c] = pd.to_numeric(df_va[c], errors='coerce').fillna(0)

    X_tr = df_tr[features].values.astype(np.float32)
    y_tr = df_tr['target'].values.astype(np.float32)
    X_va = df_va[features].values.astype(np.float32)
    y_va = df_va['target'].values.astype(np.float32)

    print(f"[train] LGB GPU ...")
    t0 = time.time()
    lgb_model = train_lgb_gpu(X_tr, y_tr, X_va, y_va, num_round=600, early_stop=40)
    lgb_pred = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
    lgb_auc = roc_auc_score(y_va, lgb_pred)
    t_lgb = time.time() - t0
    print(f"[train] LGB done in {t_lgb:.1f}s, val_auc={lgb_auc:.4f}")

    print(f"[train] XGB GPU ...")
    t0 = time.time()
    xgb_model = train_xgb_gpu(X_tr, y_tr, X_va, y_va, num_round=600, early_stop=40)
    xgb_pred = xgb_model.predict(xgb.DMatrix(X_va))
    xgb_auc = roc_auc_score(y_va, xgb_pred)
    t_xgb = time.time() - t0
    print(f"[train] XGB done in {t_xgb:.1f}s, val_auc={xgb_auc:.4f}")

    ft_model = None
    ft_pred = None
    ft_auc = None
    if run_ft and DEVICE.type == 'cuda':
        print(f"[train] FT-Transformer GPU (10 epochs) ...")
        t0 = time.time()
        ft_model, ft_pred, ft_auc = train_ft_quick(X_tr, y_tr, X_va, y_va, epochs=10, batch_size=512)
        print(f"[train] FT done in {time.time()-t0:.1f}s, val_auc={ft_auc:.4f}")

    # ensemble
    ens_pred = ensemble_predict(lgb_pred, xgb_pred, ft_p=ft_pred)
    ens_auc = roc_auc_score(y_va, ens_pred)
    print(f"[ens] val_auc={ens_auc:.4f}")

    return {
        'lgb_auc': float(lgb_auc),
        'xgb_auc': float(xgb_auc),
        'ft_auc': float(ft_auc) if ft_auc is not None else None,
        'ens_auc': float(ens_auc),
        'n_train': len(df_tr),
        'n_val': len(df_va),
        'n_features': len(features),
        'train_years': list(train_years),
        'val_years': list(val_years),
        'lgb_model': lgb_model,
        'xgb_model': xgb_model,
        'ft_model': ft_model,
    }


# =========================================================================
# CLI
# =========================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true', help='LGB+XGB only (~10 min)')
    p.add_argument('--with-ft', action='store_true', help='Include FT-Transformer (10 epochs)')
    p.add_argument('--full', action='store_true', help='全 4-model ensemble、 ~4-8h')
    p.add_argument('--save', action='store_true', default=True)
    args = p.parse_args()

    print(f"[V20 trainer] device={DEVICE}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print(f"[V20 trainer] Phase 11/12/13 features: 57 (constant defaults - skeleton)")
    print(f"[V20 trainer] V15 base features: 145 (real signals)")

    df, v15_features = load_v15_cache()
    df = add_phase_features(df)

    # V20 features = V15 145 + Phase 11+12+13 57 = 202
    v20_features = v15_features + PHASE11_FEATURES + PHASE12_FEATURES + PHASE13_FEATURES
    # dedupe (V15 と重複 features 念のため除外)
    seen = set()
    v20_features = [f for f in v20_features if not (f in seen or seen.add(f))]

    # quick: V15 features only (Phase 11/12/13 は constant なので drop されるだけ)
    feats = v15_features if args.quick else v20_features

    result = quick_train_eval(df, feats, train_years=[22, 23, 24], val_years=[25], run_ft=args.with_ft)

    # save models
    if args.save:
        save_path = os.path.join(V20_MODEL_DIR, f'v20_quick_{datetime.now().strftime("%Y%m%d_%H%M")}.pkl.gz')
        save_obj = {
            'lgb_model': result['lgb_model'],
            'xgb_model': result['xgb_model'],
            'features': feats,
            'metrics': {k: v for k, v in result.items() if not k.endswith('_model')},
            'trained_at': datetime.now().isoformat(),
        }
        with gzip.open(save_path, 'wb') as f:
            pickle.dump(save_obj, f)
        print(f"[save] {save_path}")

    # save metrics json
    metrics = {k: v for k, v in result.items() if not k.endswith('_model')}
    metrics_path = os.path.join(V20_DATA_DIR, f'phase15_metrics_{datetime.now().strftime("%Y%m%d_%H%M")}.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[save] {metrics_path}")
    print(f"[done] LGB={result['lgb_auc']:.4f}, XGB={result['xgb_auc']:.4f}, ENS={result['ens_auc']:.4f}")


if __name__ == '__main__':
    main()
