"""Session #56 B: V20 FT-Transformer 学習 PoC.

V15 cache を base に FT-Transformer 単体 学習 + AUC 計測。
V13.5b architecture (NumericalEmbedding + [CLS] + Multi-Head Self-Attention) を継承。

Usage:
    python tools/v20_ft_transformer.py --epochs 20
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "_v15_optuna_df_cache.pkl.gz"
MODEL_OUT = ROOT / "data" / "v20" / "models" / "v20_ft_transformer.pkl"
METRICS_OUT = ROOT / "data" / "v18" / "session_56_ft_transformer_metrics.json"
PRED_CACHE = ROOT / "data" / "v20" / "models" / "v20_ft_pred.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === FT-Transformer architecture (V13.5b 継承) ===

class NumericalEmbedding(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.biases = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class MHSA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MHSA(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    def __init__(self, n_features: int, d_token: int = 32, n_heads: int = 4,
                 n_layers: int = 3, d_ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.feature_embedding = NumericalEmbedding(n_features, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        d_ff = d_token * d_ff_mult
        self.layers = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        feat = self.feature_embedding(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, feat], dim=1)
        for layer in self.layers:
            tokens = layer(tokens)
        cls_out = self.norm(tokens[:, 0])
        return self.head(cls_out).squeeze(-1)


def load_data():
    print(f"Loading {CACHE}", flush=True)
    with gzip.open(CACHE, "rb") as f:
        d = pickle.load(f)
    df = d["df"]
    features = d["features"]
    df = df.copy()
    df["target"] = ((df["finish"] >= 1) & (df["finish"] <= 3)).astype(int)
    return df, features


def train(epochs: int, d_token: int, n_heads: int, n_layers: int,
          batch_size: int, lr: float) -> dict:
    df, features = load_data()
    print(f"  rows: {len(df)}, features: {len(features)}", flush=True)

    train_mask = df["year_full"].between(2020, 2023)
    valid_mask = df["year_full"] == 2024

    # Numeric coerce + fillna
    X_full = df[features].apply(lambda c: pd.to_numeric(c, errors="coerce")).fillna(0).values.astype(np.float32)
    y = df["target"].values.astype(np.float32)

    X_tr_raw = X_full[train_mask.values]
    X_va_raw = X_full[valid_mask.values]
    y_tr = y[train_mask.values]
    y_va = y[valid_mask.values]

    # Standardize
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_va = scaler.transform(X_va_raw).astype(np.float32)

    # Clip extreme values
    X_tr = np.clip(X_tr, -5, 5)
    X_va = np.clip(X_va, -5, 5)

    print(f"  train: {len(X_tr)}, valid: {len(X_va)}", flush=True)

    n_features = X_tr.shape[1]
    model = FTTransformer(
        n_features=n_features,
        d_token=d_token,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.15,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pos_weight = torch.tensor([(1 - y_tr.mean()) / y_tr.mean()], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)

    X_va_t = torch.FloatTensor(X_va).to(DEVICE)

    best_auc = 0.0
    best_state = None
    no_improve = 0
    patience = 5
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batch = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
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
            chunk = 8192
            for i in range(0, len(X_va), chunk):
                v = X_va_t[i:i + chunk]
                lg = model(v).cpu().numpy()
                val_logits.append(lg)
            val_logits = np.concatenate(val_logits)
            val_probs = 1.0 / (1.0 + np.exp(-val_logits))
            auc = roc_auc_score(y_va, val_probs)

        elapsed = time.time() - t0
        avg_loss = total_loss / max(n_batch, 1)
        history.append({"epoch": epoch + 1, "loss": avg_loss, "auc": auc, "elapsed_s": elapsed})
        print(f"  epoch {epoch+1:02d}/{epochs}: loss={avg_loss:.4f}, auc={auc:.5f}, t={elapsed:.1f}s", flush=True)

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best AUC={best_auc:.5f})")
                break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final predictions
    model.eval()
    with torch.no_grad():
        val_logits = []
        chunk = 8192
        for i in range(0, len(X_va), chunk):
            v = X_va_t[i:i + chunk]
            val_logits.append(model(v).cpu().numpy())
        val_logits = np.concatenate(val_logits)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

    final_auc = roc_auc_score(y_va, val_probs)
    print(f"\n[OK] Final AUC: {final_auc:.5f}")

    # Save model + predictions for ensemble
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v20_ft_v1",
        "state_dict": best_state,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "features": features,
        "config": {
            "d_token": d_token, "n_heads": n_heads, "n_layers": n_layers,
            "dropout": 0.15, "batch_size": batch_size, "lr": lr, "epochs": epochs,
        },
        "auc": float(final_auc),
        "history": history,
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(payload, f)
    print(f"[OK] Model saved: {MODEL_OUT}")

    # Save validation probs for ensemble
    np.savez_compressed(
        PRED_CACHE,
        valid_probs=val_probs.astype(np.float32),
        valid_targets=y_va.astype(np.int32),
    )
    print(f"[OK] Pred cache saved: {PRED_CACHE}")

    return {
        "auc": float(final_auc),
        "n_train": int(len(X_tr)),
        "n_valid": int(len(X_va)),
        "n_features": int(n_features),
        "n_params": int(n_params),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--d-token", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"[Session #56 B] FT-Transformer training", flush=True)
    print(f"  device: {DEVICE}", flush=True)

    metrics = train(
        epochs=args.epochs, d_token=args.d_token, n_heads=args.n_heads,
        n_layers=args.n_layers, batch_size=args.batch_size, lr=args.lr,
    )

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Metrics: {METRICS_OUT}")


if __name__ == "__main__":
    main()
