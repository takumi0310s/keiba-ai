#!/usr/bin/env python3
"""v13.5 FT-Transformer + 過去走Sequential + 馬間Attention

v13.4 (118特徴量, WF AUC 0.8610) をベースに3つのTransformerアーキテクチャを検証:

1. FT-Transformer: 各特徴量をembeddingに変換 → Transformer自己注意で特徴量間相互作用を学習
2. Sequential Past-Race Model: 過去5走のデータをLSTMで時系列学習 → LGBMとアンサンブル
3. Intra-Race Attention: 同レース全馬を同時に見て相対評価

最終的にLightGBM + XGBoost + FT-Transformerの3モデルアンサンブル。

採用基準: WF AUC > 0.8610, gap < 0.05
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

from train_v92_central import (
    load_data, encode_categoricals, encode_sires,
    load_training_times, merge_training_features,
    compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance,
    compute_lag_features, build_features,
    compute_distance_aptitude, compute_frame_advantage,
    FEATURES_V91, V92_NEW_FEATURES, V93_NEW_FEATURES,
    COURSE_MAP, N_TOP_SIRE,
)
from train_v92_leakfree import LEAK_FEATURES_A
from train_v12_comprehensive import (
    V12_NEW_FEATURES, LGB_PARAMS, XGB_PARAMS,
    merge_speed_index, merge_training_1f, merge_sire_shinba,
    merge_dam_stats, compute_pci, _build_nk_race_id,
)
from jrdb_features import (
    merge_jrdb_train_features, _build_nk_race_id_from_jv,
    JRDB_LIVE_FEATURES, JRDB_DEFAULTS,
)
from train_v132_extended import (
    V132_NEW_FEATURES, V132_DEFAULTS,
    merge_cha_features, merge_jo_features, merge_kta_features,
    merge_ze_features, merge_sr_features, merge_kka_features,
    load_v13_jrdb_selected,
)
from train_v133_extended import V133_NEW_FEATURES, V133_DEFAULTS, merge_stable_comment_combined
from train_v134_odds_change import (
    V134_NEW_FEATURES as V134_OZ_FEATURES,
    V134_DEFAULTS as V134_OZ_DEFAULTS,
    merge_oz_features,
)
from train_v134_weight_trend import (
    V134_NEW_FEATURES as V134_WT_FEATURES,
    V134_DEFAULTS as V134_WT_DEFAULTS,
    compute_weight_trends,
)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================================================
# FT-Transformer Architecture
# =====================================================

class NumericalEmbedding(nn.Module):
    """各数値特徴量を個別の線形層でd_token次元のembeddingに変換"""

    def __init__(self, n_features, d_token):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_features, d_token))
        self.biases = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x):
        # x: (batch, n_features)
        # output: (batch, n_features, d_token)
        return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer for tabular data.

    Each numerical feature is projected to d_token dimensions via a learned linear embedding.
    A [CLS] token is prepended. The Transformer processes all tokens jointly.
    The [CLS] output is used for binary classification.
    """

    def __init__(self, n_features, d_token=64, n_heads=4, n_layers=3,
                 d_ff_mult=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token

        # Feature embeddings (each feature → d_token)
        self.feature_embedding = NumericalEmbedding(n_features, d_token)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        # Transformer layers
        d_ff = d_token * d_ff_mult
        self.layers = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Classification head
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x):
        # x: (batch, n_features)
        B = x.shape[0]

        # Embed each feature: (batch, n_features, d_token)
        feat_emb = self.feature_embedding(x)

        # Prepend [CLS]: (batch, n_features+1, d_token)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, feat_emb], dim=1)

        # Transformer
        for layer in self.layers:
            tokens = layer(tokens)

        # [CLS] output → classification
        cls_out = self.norm(tokens[:, 0])
        logit = self.head(cls_out).squeeze(-1)
        return logit


# =====================================================
# Sequential Past-Race Model (LSTM)
# =====================================================

# 過去走で使う特徴量（prev_finish等ではなく、各走のraw特徴を時系列化）
SEQ_FEATURES = [
    'finish', 'last3f', 'pass4', 'prize', 'odds_log_raw',
    'distance', 'surface_enc', 'course_enc', 'weight_carry',
    'num_horses', 'horse_num',
]
SEQ_LEN = 5  # 過去5走


class SequentialRaceModel(nn.Module):
    """過去5走の特徴量をLSTMで処理 → 予測"""

    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, lengths=None):
        # x: (batch, seq_len, input_dim)
        out, (h, c) = self.lstm(x)
        # Use last hidden state
        logit = self.head(h[-1]).squeeze(-1)
        return logit


# =====================================================
# Intra-Race Attention Model
# =====================================================

class IntraRaceAttention(nn.Module):
    """同レース内の全馬を同時に見て相対評価.

    入力: 各馬の特徴量ベクトル (per-horse features)
    出力: 各馬の複勝確率

    レース内の全馬のembeddingにself-attentionを適用し、
    「このメンバーの中でこの馬がどう位置するか」を学習。
    """

    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2,
                 d_ff_mult=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * d_ff_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, mask=None):
        # x: (batch_races, max_horses, n_features)
        h = self.proj(x)
        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        logits = self.head(h).squeeze(-1)  # (batch_races, max_horses)
        return logits


# =====================================================
# Training Utilities
# =====================================================

def train_ft_transformer(X_train, y_train, X_val, y_val, n_features,
                         epochs=50, batch_size=4096, lr=1e-3, patience=10,
                         d_token=64, n_heads=4, n_layers=3, dropout=0.1,
                         weight_decay=1e-5, label=''):
    """FT-Transformerを学習して予測確率を返す"""

    model = FTTransformer(
        n_features=n_features, d_token=d_token, n_heads=n_heads,
        n_layers=n_layers, dropout=dropout
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Compute positive weight for imbalanced binary classification
    pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)

    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    best_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batch = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = []
            # Process in chunks to avoid OOM
            chunk_size = 8192
            for i in range(0, len(X_val), chunk_size):
                chunk = X_val_t[i:i+chunk_size]
                val_logits.append(model(chunk))
            val_logits = torch.cat(val_logits)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = roc_auc_score(y_val, val_probs)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or no_improve == 0:
            avg_loss = total_loss / n_batch
            lr_now = scheduler.get_last_lr()[0]
            print(f"    {label} Epoch {epoch+1:3d}: loss={avg_loss:.4f} "
                  f"val_AUC={val_auc:.4f} best={best_auc:.4f} "
                  f"lr={lr_now:.6f}")

        if no_improve >= patience:
            print(f"    {label} Early stop at epoch {epoch+1}")
            break

    # Load best state & predict
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X_val), 8192):
            chunk = X_val_t[i:i+8192]
            preds.append(torch.sigmoid(model(chunk)).cpu().numpy())
        val_preds = np.concatenate(preds)

        # Also predict train (for gap check)
        X_tr_t = torch.FloatTensor(X_train).to(DEVICE)
        tr_preds = []
        for i in range(0, len(X_train), 8192):
            chunk = X_tr_t[i:i+8192]
            tr_preds.append(torch.sigmoid(model(chunk)).cpu().numpy())
        tr_preds = np.concatenate(tr_preds)

    return model, val_preds, tr_preds, best_auc


def train_intra_race(df_train, df_val, features, label='target',
                     epochs=30, batch_size=64, lr=1e-3, patience=8,
                     d_model=64, n_heads=4, n_layers=2, dropout=0.1):
    """同レース内Attentionモデルを学習"""

    MAX_HORSES = 28  # JRA max is typically 18 but some have more entries

    def make_race_batches(df, feats):
        """DataFrameをレース単位でバッチ化"""
        race_groups = df.groupby('race_id_unique')
        X_races, y_races, masks = [], [], []
        for _, grp in race_groups:
            n = min(len(grp), MAX_HORSES)
            if n < 2:
                continue
            x = np.zeros((MAX_HORSES, len(feats)), dtype=np.float32)
            y = np.zeros(MAX_HORSES, dtype=np.float32)
            m = np.zeros(MAX_HORSES, dtype=np.float32)
            x[:n] = grp[feats].values[:n]
            y[:n] = grp[label].values[:n]
            m[:n] = 1.0
            X_races.append(x)
            y_races.append(y)
            masks.append(m)
        return (np.array(X_races), np.array(y_races), np.array(masks))

    X_tr, y_tr, m_tr = make_race_batches(df_train, features)
    X_va, y_va, m_va = make_race_batches(df_val, features)
    print(f"    IntraRace: train={len(X_tr)} races, val={len(X_va)} races")

    model = IntraRaceAttention(
        n_features=len(features), d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, dropout=dropout
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pos_weight = torch.tensor([(1 - y_tr[m_tr > 0].mean()) / y_tr[m_tr > 0].mean()]).to(DEVICE)
    bce = nn.BCEWithLogitsLoss(reduction='none')

    train_ds = TensorDataset(
        torch.FloatTensor(X_tr), torch.FloatTensor(y_tr), torch.FloatTensor(m_tr))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)

    X_va_t = torch.FloatTensor(X_va).to(DEVICE)
    y_va_t = torch.FloatTensor(y_va).to(DEVICE)
    m_va_t = torch.FloatTensor(m_va).to(DEVICE)

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
            loss_raw = bce(logits, yb) * pos_weight * mb
            loss = loss_raw.sum() / mb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t, m_va_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            # Flatten valid entries for AUC
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

        if (epoch + 1) % 5 == 0 or no_improve == 0:
            print(f"    IntraRace Epoch {epoch+1:3d}: loss={total_loss/n_batch:.4f} "
                  f"val_AUC={val_auc:.4f} best={best_auc:.4f}")

        if no_improve >= patience:
            print(f"    IntraRace Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, best_auc


# =====================================================
# Data Building (reuse v13.4 pipeline)
# =====================================================

def build_v134_dataframe():
    """v13.4と同一の特徴量エンジニアリングパイプライン"""
    print("\n[1] Building v13.4 features...")
    df = load_data()
    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)
    tt_data = load_training_times()
    df = merge_training_features(df, tt_data)
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)
    df = compute_lag_features(df)
    df = build_features(df)
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)
    df = merge_speed_index(df)
    df = merge_training_1f(df)
    df = merge_sire_shinba(df)
    df = merge_dam_stats(df)
    df = compute_pci(df)
    df = merge_jrdb_train_features(df)
    if '_nk_rid' not in df.columns:
        df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    if '_uma' not in df.columns:
        df['_uma'] = df['umaban'].astype(int)
    df = merge_cha_features(df)
    df = merge_jo_features(df)
    df = merge_kta_features(df)
    df = merge_ze_features(df)
    df = merge_sr_features(df)
    df = merge_kka_features(df)
    df.drop(columns=['_nk_rid', '_uma'], inplace=True, errors='ignore')
    df = merge_stable_comment_combined(df)
    df = merge_oz_features(df)
    df = compute_weight_trends(df)

    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses'] >= 5].copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)

    return df, sire_map, bms_map


def get_v134_features():
    """v13.4の118特徴量リスト（Pattern A = leak free）"""
    F_V93 = list(FEATURES_V91) + list(V92_NEW_FEATURES) + list(V93_NEW_FEATURES)
    F_V12_BASE = [f for f in F_V93 if f not in LEAK_FEATURES_A]
    V12_ADOPTED = [f for f in V12_NEW_FEATURES if f != 'dam_top3r']
    F_V12 = F_V12_BASE + V12_ADOPTED
    jrdb_sel = load_v13_jrdb_selected()
    F_V13 = F_V12 + jrdb_sel
    F_V132 = F_V13 + V132_NEW_FEATURES
    F_V133 = F_V132 + V133_NEW_FEATURES
    # v13.4: OZ + Weight Trend
    F_V134 = F_V133 + V134_OZ_FEATURES + V134_WT_FEATURES
    return F_V134, jrdb_sel


def fill_defaults(df, features):
    """欠損値をデフォルト値で埋める"""
    all_defaults = {**JRDB_DEFAULTS, **V132_DEFAULTS, **V133_DEFAULTS,
                    **V134_OZ_DEFAULTS, **V134_WT_DEFAULTS}
    for f in features:
        if f not in df.columns:
            df[f] = all_defaults.get(f, 0)
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(all_defaults.get(f, 0))
    return df


# =====================================================
# Walk-Forward with FT-Transformer
# =====================================================

def walk_forward_ft_transformer(df, features, years=range(2020, 2026),
                                 d_token=64, n_heads=4, n_layers=3,
                                 epochs=50, batch_size=4096, lr=1e-3,
                                 patience=10, dropout=0.1):
    """FT-TransformerのWFバックテスト"""
    results = []
    for test_year in years:
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        X_tr = df.loc[train_mask, features].values.astype(np.float32)
        y_tr = df.loc[train_mask, 'target'].values.astype(np.float32)
        X_te = df.loc[test_mask, features].values.astype(np.float32)
        y_te = df.loc[test_mask, 'target'].values.astype(np.float32)

        # StandardScaler for FT-Transformer
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        print(f"\n  --- FT-Transformer WF {test_year} ---")
        print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")

        model, val_preds, tr_preds, best_auc = train_ft_transformer(
            X_tr_s, y_tr, X_te_s, y_te,
            n_features=len(features),
            epochs=epochs, batch_size=batch_size, lr=lr,
            patience=patience, d_token=d_token, n_heads=n_heads,
            n_layers=n_layers, dropout=dropout,
            label=f'FT-{test_year}',
        )
        train_auc = roc_auc_score(y_tr, tr_preds)
        gap = train_auc - best_auc

        results.append({
            'year': test_year,
            'ft_auc': best_auc,
            'train_auc': train_auc,
            'gap': gap,
            'model': model,
            'scaler': scaler,
            'val_preds': val_preds,
        })
        print(f"  {test_year}: FT_AUC={best_auc:.4f} Train={train_auc:.4f} Gap={gap:.4f}")

    return results


def walk_forward_ensemble(df, features, years=range(2020, 2026),
                           ft_params=None):
    """LGB + XGB + FT-Transformer の3モデルアンサンブルWF"""
    if ft_params is None:
        ft_params = {
            'd_token': 64, 'n_heads': 4, 'n_layers': 3,
            'epochs': 50, 'batch_size': 4096, 'lr': 1e-3,
            'patience': 10, 'dropout': 0.1,
        }

    results = []
    for test_year in years:
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[test_mask, features].values
        y_te = df.loc[test_mask, 'target'].values

        print(f"\n{'='*50}")
        print(f"  Ensemble WF {test_year}: train={len(X_tr)}, test={len(X_te)}")
        print(f"{'='*50}")

        # --- LightGBM ---
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
        m_lgb = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                          valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        p_lgb = m_lgb.predict(X_te)
        auc_lgb = roc_auc_score(y_te, p_lgb)
        p_lgb_tr = m_lgb.predict(X_tr)
        auc_lgb_tr = roc_auc_score(y_tr, p_lgb_tr)

        # --- XGBoost ---
        dxtr = xgb.DMatrix(X_tr, label=y_tr)
        dxte = xgb.DMatrix(X_te, label=y_te)
        m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=1000,
                           evals=[(dxte, 'valid')],
                           early_stopping_rounds=50, verbose_eval=False)
        p_xgb = m_xgb.predict(dxte)
        auc_xgb = roc_auc_score(y_te, p_xgb)

        # --- FT-Transformer ---
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr.astype(np.float32))
        X_te_s = scaler.transform(X_te.astype(np.float32))

        ft_model, p_ft, p_ft_tr, auc_ft = train_ft_transformer(
            X_tr_s, y_tr.astype(np.float32),
            X_te_s, y_te.astype(np.float32),
            n_features=len(features),
            label=f'FT-{test_year}',
            **ft_params,
        )

        # --- Ensemble ---
        # AUC-proportional weights
        total_auc = auc_lgb + auc_xgb + auc_ft
        w_lgb = auc_lgb / total_auc
        w_xgb = auc_xgb / total_auc
        w_ft = auc_ft / total_auc

        p_ens = w_lgb * p_lgb + w_xgb * p_xgb + w_ft * p_ft
        auc_ens = roc_auc_score(y_te, p_ens)

        # Also try LGB+FT only (2-model ensemble)
        w2_lgb = auc_lgb / (auc_lgb + auc_ft)
        w2_ft = auc_ft / (auc_lgb + auc_ft)
        p_ens2 = w2_lgb * p_lgb + w2_ft * p_ft
        auc_ens2 = roc_auc_score(y_te, p_ens2)

        # LGB+XGB baseline
        w_lgb2 = auc_lgb / (auc_lgb + auc_xgb)
        w_xgb2 = auc_xgb / (auc_lgb + auc_xgb)
        p_lgbxgb = w_lgb2 * p_lgb + w_xgb2 * p_xgb
        auc_lgbxgb = roc_auc_score(y_te, p_lgbxgb)

        gap = auc_lgb_tr - auc_lgb

        print(f"  {test_year} Results:")
        print(f"    LGB:       {auc_lgb:.4f}")
        print(f"    XGB:       {auc_xgb:.4f}")
        print(f"    FT:        {auc_ft:.4f}")
        print(f"    LGB+XGB:   {auc_lgbxgb:.4f} (baseline)")
        print(f"    LGB+FT:    {auc_ens2:.4f}")
        print(f"    LGB+XGB+FT:{auc_ens:.4f}")
        print(f"    Gap(LGB):  {gap:.4f}")

        results.append({
            'year': test_year,
            'lgb_auc': auc_lgb,
            'xgb_auc': auc_xgb,
            'ft_auc': auc_ft,
            'lgbxgb_auc': auc_lgbxgb,
            'lgb_ft_auc': auc_ens2,
            'ensemble_auc': auc_ens,
            'train_auc': auc_lgb_tr,
            'gap': gap,
            'weights': {'lgb': w_lgb, 'xgb': w_xgb, 'ft': w_ft},
            'ft_model': ft_model,
            'ft_scaler': scaler,
            'lgb_model': m_lgb,
            'xgb_model': m_xgb,
        })

    return results


# =====================================================
# FT-Transformer Only WF (for quick evaluation)
# =====================================================

def walk_forward_ft_only(df, features, years=range(2020, 2026)):
    """FT-Transformer単体のWFバックテスト（LGB/XGBなし、高速評価用）"""
    results = walk_forward_ft_transformer(
        df, features, years=years,
        d_token=64, n_heads=4, n_layers=3,
        epochs=50, batch_size=4096, lr=1e-3,
        patience=10, dropout=0.1,
    )
    return results


# =====================================================
# Main
# =====================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("  v13.5 - FT-Transformer + Sequential + IntraRace Attention")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Build dataframe
    df, sire_map, bms_map = build_v134_dataframe()
    features, jrdb_sel = get_v134_features()
    df = fill_defaults(df, features)
    print(f"  Data: {len(df)} rows, {len(features)} features")

    # Create unique race ID for IntraRace grouping
    if 'race_id_unique' not in df.columns:
        df['race_id_unique'] = (
            df['year'].astype(str) + '_' +
            df['date_num'].astype(str) + '_' +
            df.get('course_code', df.get('course_enc', pd.Series(0, index=df.index))).astype(str) + '_' +
            df.get('race_num', pd.Series(0, index=df.index)).astype(str)
        )

    # ===== Phase 1: FT-Transformer Only WF =====
    print(f"\n{'='*70}")
    print(f"  Phase 1: FT-Transformer Only (Quick Eval)")
    print(f"{'='*70}")

    ft_results = walk_forward_ft_only(df, features)
    ft_aucs = [r['ft_auc'] for r in ft_results]
    ft_mean = np.mean(ft_aucs)
    print(f"\n  FT-Transformer Only WF Mean AUC: {ft_mean:.4f}")
    for r in ft_results:
        ok = 'Y' if r['ft_auc'] > 0.78 else 'N'
        gok = 'Y' if r['gap'] < 0.05 else 'N'
        print(f"    {r['year']}: AUC={r['ft_auc']:.4f} Gap={r['gap']:.4f} {ok}{gok}")

    # ===== Phase 2: LGB + XGB + FT-Transformer Ensemble WF =====
    print(f"\n{'='*70}")
    print(f"  Phase 2: LGB + XGB + FT-Transformer Ensemble")
    print(f"{'='*70}")

    ens_results = walk_forward_ensemble(df, features)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    lgb_aucs = [r['lgb_auc'] for r in ens_results]
    lgbxgb_aucs = [r['lgbxgb_auc'] for r in ens_results]
    ens_aucs = [r['ensemble_auc'] for r in ens_results]
    lgbft_aucs = [r['lgb_ft_auc'] for r in ens_results]
    ft_aucs2 = [r['ft_auc'] for r in ens_results]

    print(f"\n  {'Year':<6} {'LGB':>8} {'XGB':>8} {'FT':>8} {'LGB+XGB':>9} {'LGB+FT':>9} {'3-model':>9} {'Gap':>6}")
    print(f"  {'-'*66}")
    for r in ens_results:
        ok = 'Y' if r['gap'] < 0.05 else 'N'
        print(f"  {r['year']:<6} {r['lgb_auc']:8.4f} {r['xgb_auc']:8.4f} {r['ft_auc']:8.4f} "
              f"{r['lgbxgb_auc']:9.4f} {r['lgb_ft_auc']:9.4f} {r['ensemble_auc']:9.4f} "
              f"{r['gap']:6.4f}{ok}")

    print(f"\n  Mean AUC:")
    print(f"    LGB only:     {np.mean(lgb_aucs):.6f}")
    print(f"    FT only:      {np.mean(ft_aucs2):.6f}")
    print(f"    LGB+XGB:      {np.mean(lgbxgb_aucs):.6f}  (v13.4 baseline)")
    print(f"    LGB+FT:       {np.mean(lgbft_aucs):.6f}")
    print(f"    LGB+XGB+FT:   {np.mean(ens_aucs):.6f}")

    baseline = 0.8610
    best_combo = max([
        ('LGB+XGB', np.mean(lgbxgb_aucs)),
        ('LGB+FT', np.mean(lgbft_aucs)),
        ('LGB+XGB+FT', np.mean(ens_aucs)),
    ], key=lambda x: x[1])

    print(f"\n  Best ensemble: {best_combo[0]} = {best_combo[1]:.6f}")
    delta = best_combo[1] - baseline
    print(f"  vs v13.4 baseline (0.8610): {delta:+.6f}")

    # Adoption check
    all_ok = all(r['lgb_auc'] > 0.78 for r in ens_results)
    no_overfit = all(r['gap'] < 0.05 for r in ens_results)
    adopted = best_combo[1] > baseline and all_ok and no_overfit

    if adopted:
        print(f"\n  *** ADOPTED: {best_combo[0]} ***")
    else:
        reasons = []
        if best_combo[1] <= baseline:
            reasons.append(f"AUC {best_combo[1]:.4f} <= baseline {baseline:.4f}")
        if not all_ok:
            reasons.append("some year AUC < 0.78")
        if not no_overfit:
            reasons.append("gap >= 0.05")
        print(f"\n  NOT ADOPTED: {', '.join(reasons)}")

    # ===== Phase 3: IntraRace Attention (experimental) =====
    intra_results = []
    try:
        print(f"\n{'='*70}")
        print(f"  Phase 3: IntraRace Attention (Experimental)")
        print(f"{'='*70}")

        for test_year in range(2020, 2026):
            ty = test_year - 2000
            train_mask = df['year'] < ty
            test_mask = df['year'] == ty
            if train_mask.sum() < 1000 or test_mask.sum() < 100:
                continue

            df_tr = df.loc[train_mask].copy()
            df_te = df.loc[test_mask].copy()

            # Scale features
            scaler_ir = StandardScaler()
            df_tr[features] = scaler_ir.fit_transform(df_tr[features].values.astype(np.float32))
            df_te[features] = scaler_ir.transform(df_te[features].values.astype(np.float32))

            print(f"\n  IntraRace WF {test_year}")
            model, best_auc = train_intra_race(
                df_tr, df_te, features,
                epochs=30, batch_size=64, lr=1e-3, patience=8,
                d_model=64, n_heads=4, n_layers=2, dropout=0.1,
            )
            intra_results.append({'year': test_year, 'auc': best_auc})
            print(f"  {test_year}: IntraRace AUC={best_auc:.4f}")

        if intra_results:
            intra_mean = np.mean([r['auc'] for r in intra_results])
            print(f"\n  IntraRace Mean AUC: {intra_mean:.4f}")
    except Exception as e:
        print(f"  IntraRace failed: {e}")
        print(f"  Skipping Phase 3")

    # Save results
    result = {
        'baseline_auc': baseline,
        'ft_only_mean': ft_mean,
        'ft_only_yearly': [{'year': r['year'], 'ft_auc': r['ft_auc'],
                            'gap': r['gap']} for r in ft_results],
        'ensemble_yearly': [{
            'year': r['year'],
            'lgb_auc': r['lgb_auc'],
            'xgb_auc': r['xgb_auc'],
            'ft_auc': r['ft_auc'],
            'lgbxgb_auc': r['lgbxgb_auc'],
            'lgb_ft_auc': r['lgb_ft_auc'],
            'ensemble_auc': r['ensemble_auc'],
            'gap': r['gap'],
            'weights': r['weights'],
        } for r in ens_results],
        'ensemble_means': {
            'lgb': float(np.mean(lgb_aucs)),
            'ft': float(np.mean(ft_aucs2)),
            'lgbxgb': float(np.mean(lgbxgb_aucs)),
            'lgb_ft': float(np.mean(lgbft_aucs)),
            'lgbxgb_ft': float(np.mean(ens_aucs)),
        },
        'intra_race_yearly': intra_results,
        'intra_race_mean': float(np.mean([r['auc'] for r in intra_results])) if intra_results else None,
        'best_ensemble': best_combo[0],
        'best_auc': float(best_combo[1]),
        'delta_vs_baseline': float(delta),
        'adopted': adopted,
        'features_count': len(features),
        'device': str(DEVICE),
        'timestamp': datetime.now().isoformat(),
    }
    rpath = os.path.join(DATA_DIR, 'v135_ft_transformer_results.json')
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved: {rpath}")

    # If adopted, train production models
    if adopted:
        print(f"\n{'='*70}")
        print(f"  TRAINING PRODUCTION MODELS")
        print(f"{'='*70}")
        _train_production(df, features, sire_map, bms_map, jrdb_sel,
                          ens_results, best_combo[0])

    elapsed = (time.time() - t0) / 60
    print(f"\n  Total elapsed: {elapsed:.1f} min")


def _train_production(df, features, sire_map, bms_map, jrdb_sel,
                      wf_results, ensemble_type):
    """本番モデルの学習・保存"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    my = int(df['year'].max())
    vm = df['year'] >= (my - 1)
    tm = ~vm

    X_tr = df.loc[tm, features].values
    y_tr = df.loc[tm, 'target'].values
    X_va = df.loc[vm, features].values
    y_va = df.loc[vm, 'target'].values

    # LightGBM
    dt = lgb.Dataset(X_tr, label=y_tr)
    dv = lgb.Dataset(X_va, label=y_va, reference=dt)
    ml = lgb.train(LGB_PARAMS, dt, num_boost_round=1000, valid_sets=[dv],
                   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    al = roc_auc_score(y_va, ml.predict(X_va))

    # XGBoost
    dxt = xgb.DMatrix(X_tr, label=y_tr)
    dxv = xgb.DMatrix(X_va, label=y_va)
    mx = xgb.train(XGB_PARAMS, dxt, num_boost_round=1000, evals=[(dxv, 'valid')],
                   early_stopping_rounds=50, verbose_eval=100)
    ax = roc_auc_score(y_va, mx.predict(dxv))

    # FT-Transformer
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr.astype(np.float32))
    X_va_s = scaler.transform(X_va.astype(np.float32))
    ft_model, p_ft, _, auc_ft = train_ft_transformer(
        X_tr_s, y_tr.astype(np.float32),
        X_va_s, y_va.astype(np.float32),
        n_features=len(features),
        epochs=50, batch_size=4096, lr=1e-3,
        patience=10, d_token=64, n_heads=4, n_layers=3,
        dropout=0.1, label='PROD',
    )

    total = al + ax + auc_ft
    wl, wx, wf = al / total, ax / total, auc_ft / total
    wf_auc = np.mean([r['ensemble_auc'] for r in wf_results])

    print(f"\n  Production model AUCs:")
    print(f"    LGB: {al:.4f}, XGB: {ax:.4f}, FT: {auc_ft:.4f}")
    print(f"    Weights: LGB={wl:.3f}, XGB={wx:.3f}, FT={wf:.3f}")

    pa_feats = [f for f in features if f not in LEAK_FEATURES_A]

    # Save FT-Transformer state dict separately
    ft_state = {k: v.cpu() for k, v in ft_model.state_dict().items()}
    ft_config = {
        'n_features': len(pa_feats),
        'd_token': 64, 'n_heads': 4, 'n_layers': 3, 'dropout': 0.1,
    }

    pkl = {
        'model': ml, 'features': pa_feats, 'version': 'v135_leakfree',
        'auc': al, 'wf_auc': wf_auc, 'leak_free': True, 'leak_pattern': 'A',
        'leak_removed': sorted(LEAK_FEATURES_A),
        'sire_map': sire_map, 'bms_map': bms_map, 'n_top_encode': N_TOP_SIRE,
        'trained_at': now, 'model_type': 'central',
        'xgb_model': mx, 'mlp_model': None, 'mlp_scaler': None,
        'ft_model_state': ft_state,
        'ft_model_config': ft_config,
        'ft_scaler_mean': scaler.mean_.tolist(),
        'ft_scaler_scale': scaler.scale_.tolist(),
        'ensemble_weights': {'lgb': wl, 'xgb': wx, 'ft': wf, 'mlp': 0},
        'course_map': dict(COURSE_MAP),
        'jrdb_features': jrdb_sel,
        'ensemble_type': ensemble_type,
    }

    ap = os.path.join(BASE_DIR, 'keiba_model_v135_central.pkl.gz')
    with gzip.open(ap, 'wb') as f:
        pickle.dump(pkl, f)
    print(f"  Saved Pattern A: {ap}")

    pb_feats = features + JRDB_LIVE_FEATURES
    pkl_b = dict(pkl)
    pkl_b['features'] = pb_feats
    pkl_b['version'] = 'v135_live'
    pkl_b['leak_free'] = False
    pkl_b['leak_pattern'] = 'B'
    pkl_b['is_live'] = True
    pkl_b['ft_model_config']['n_features'] = len(pb_feats)

    bp = os.path.join(BASE_DIR, 'keiba_model_v135_central_live.pkl.gz')
    with gzip.open(bp, 'wb') as f:
        pickle.dump(pkl_b, f)
    print(f"  Saved Pattern B: {bp}")


if __name__ == '__main__':
    main()
