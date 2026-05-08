"""Session #56 C: V20 IntraRace Attention 学習 PoC.

V13.5b architecture: 同 race 内 全馬を同時に attention で 相対 評価。
入力: (batch_races, max_horses, n_features) + mask
出力: 各馬の logit (per-horse)

Usage:
    python tools/v20_intra_race_attention.py --epochs 15
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

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "_v15_optuna_df_cache.pkl.gz"
MODEL_OUT = ROOT / "data" / "v20" / "models" / "v20_intra_race.pkl"
METRICS_OUT = ROOT / "data" / "v18" / "session_56_intra_race_metrics.json"
PRED_CACHE = ROOT / "data" / "v20" / "models" / "v20_ir_pred.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_HORSES = 18


class MHSA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            # mask: (B, N), 1 = valid, 0 = pad
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class IntraRaceAttention(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        d_ff = d_model * 2
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        return self.head(h).squeeze(-1)  # (B, N)


def build_race_batches(df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> tuple:
    """Group rows by race_id, pad to MAX_HORSES.

    Returns:
        X_batched: (n_races, MAX_HORSES, n_features)
        y_batched: (n_races, MAX_HORSES)
        mask_batched: (n_races, MAX_HORSES) — 1=valid, 0=pad
        idx_to_orig: list of original row indices for unpacking
    """
    n_features = X.shape[1]
    # race_id_str (8-char) = race-level key (race_id 10-char は horse-race-level)
    race_ids = df["race_id_str"].values
    unique_races = pd.unique(race_ids)

    X_batched = np.zeros((len(unique_races), MAX_HORSES, n_features), dtype=np.float32)
    y_batched = np.zeros((len(unique_races), MAX_HORSES), dtype=np.float32)
    mask_batched = np.zeros((len(unique_races), MAX_HORSES), dtype=np.int8)
    orig_indices = np.full((len(unique_races), MAX_HORSES), -1, dtype=np.int64)

    # Fast group by race_id
    df_idx = df.index.values
    race_to_idx = {}
    for orig_i, rid in enumerate(race_ids):
        race_to_idx.setdefault(rid, []).append(orig_i)

    for ri, rid in enumerate(unique_races):
        idxs = race_to_idx[rid][:MAX_HORSES]
        for hi, oi in enumerate(idxs):
            X_batched[ri, hi] = X[oi]
            y_batched[ri, hi] = y[oi]
            mask_batched[ri, hi] = 1
            orig_indices[ri, hi] = oi

    return X_batched, y_batched, mask_batched, orig_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"[Session #56 C] IntraRace Attention", flush=True)
    print(f"  device: {DEVICE}", flush=True)

    print(f"Loading {CACHE}", flush=True)
    with gzip.open(CACHE, "rb") as f:
        d = pickle.load(f)
    df = d["df"].copy()
    features = d["features"]
    df["target"] = ((df["finish"] >= 1) & (df["finish"] <= 3)).astype(int)

    print(f"  rows: {len(df)}, features: {len(features)}", flush=True)

    train_mask = df["year_full"].between(2020, 2023)
    valid_mask = df["year_full"] == 2024

    X_full = df[features].apply(lambda c: pd.to_numeric(c, errors="coerce")).fillna(0).values.astype(np.float32)
    y = df["target"].values.astype(np.float32)

    X_tr_raw = X_full[train_mask.values]
    y_tr = y[train_mask.values]
    df_tr = df.loc[train_mask].reset_index(drop=True)

    X_va_raw = X_full[valid_mask.values]
    y_va = y[valid_mask.values]
    df_va = df.loc[valid_mask].reset_index(drop=True)

    scaler = StandardScaler()
    X_tr = np.clip(scaler.fit_transform(X_tr_raw), -5, 5).astype(np.float32)
    X_va = np.clip(scaler.transform(X_va_raw), -5, 5).astype(np.float32)

    print(f"  train rows: {len(X_tr)}, valid rows: {len(X_va)}", flush=True)

    print("Building race batches...", flush=True)
    Xb_tr, yb_tr, mb_tr, _ = build_race_batches(df_tr, X_tr, y_tr)
    Xb_va, yb_va, mb_va, idx_va = build_race_batches(df_va, X_va, y_va)
    print(f"  train races: {len(Xb_tr)}, valid races: {len(Xb_va)}", flush=True)

    # Convert to tensors
    Xb_tr_t = torch.FloatTensor(Xb_tr)
    yb_tr_t = torch.FloatTensor(yb_tr)
    mb_tr_t = torch.IntTensor(mb_tr.astype(np.int32))

    Xb_va_t = torch.FloatTensor(Xb_va).to(DEVICE)
    yb_va_t = torch.FloatTensor(yb_va).to(DEVICE)
    mb_va_t = torch.IntTensor(mb_va.astype(np.int32)).to(DEVICE)

    n_features = X_tr.shape[1]
    model = IntraRaceAttention(
        n_features=n_features, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=0.15,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    pos_weight = torch.tensor([(1 - y_tr.mean()) / y_tr.mean()], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    n_train_races = len(Xb_tr)
    history = []
    best_auc = 0.0
    best_state = None
    no_improve = 0
    patience = 5

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        # Shuffle race indices
        perm = np.random.permutation(n_train_races)
        total_loss = 0.0
        n_batch = 0
        for i in range(0, n_train_races, args.batch_size):
            sel = perm[i:i + args.batch_size]
            xb = Xb_tr_t[sel].to(DEVICE, non_blocking=True)
            yb = yb_tr_t[sel].to(DEVICE, non_blocking=True)
            mb = mb_tr_t[sel].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb, mb)
            loss_raw = criterion(logits, yb)
            # apply mask
            loss = (loss_raw * mb.float()).sum() / mb.float().sum().clamp(min=1)
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
            chunk = 256
            for j in range(0, len(Xb_va), chunk):
                lg = model(Xb_va_t[j:j + chunk], mb_va_t[j:j + chunk])
                val_logits.append(lg.cpu().numpy())
            val_logits = np.concatenate(val_logits, axis=0)  # (n_races, MAX_HORSES)
            val_probs_all = 1.0 / (1.0 + np.exp(-val_logits))

            # Flatten to row-level via mask
            mask_arr = mb_va.astype(bool)
            val_probs_flat = val_probs_all[mask_arr]
            y_va_flat = yb_va[mask_arr]
            auc = roc_auc_score(y_va_flat, val_probs_flat)

        elapsed = time.time() - t0
        avg_loss = total_loss / max(n_batch, 1)
        history.append({"epoch": epoch + 1, "loss": avg_loss, "auc": auc, "elapsed_s": elapsed})
        print(f"  epoch {epoch+1:02d}/{args.epochs}: loss={avg_loss:.4f}, auc={auc:.5f}, t={elapsed:.1f}s", flush=True)

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best AUC={best_auc:.5f})")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final pred for ensemble (row-aligned to V15 cache valid set)
    model.eval()
    with torch.no_grad():
        val_logits = []
        chunk = 256
        for j in range(0, len(Xb_va), chunk):
            lg = model(Xb_va_t[j:j + chunk], mb_va_t[j:j + chunk])
            val_logits.append(lg.cpu().numpy())
        val_logits = np.concatenate(val_logits, axis=0)
        val_probs_all = 1.0 / (1.0 + np.exp(-val_logits))

    # Map back to original valid row order
    n_valid_rows = len(X_va)
    pred_aligned = np.zeros(n_valid_rows, dtype=np.float32)
    target_aligned = np.zeros(n_valid_rows, dtype=np.int32)
    for ri in range(len(idx_va)):
        for hi in range(MAX_HORSES):
            oi = idx_va[ri, hi]
            if oi >= 0:
                pred_aligned[oi] = val_probs_all[ri, hi]
                target_aligned[oi] = int(yb_va[ri, hi])

    final_auc = roc_auc_score(target_aligned, pred_aligned)
    print(f"\n[OK] Final aligned AUC: {final_auc:.5f}")

    # Save
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v20_ir_v1",
        "state_dict": best_state,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "features": features,
        "config": {
            "d_model": args.d_model, "n_heads": args.n_heads,
            "n_layers": args.n_layers, "dropout": 0.15,
            "max_horses": MAX_HORSES, "batch_size": args.batch_size,
            "lr": args.lr, "epochs": args.epochs,
        },
        "auc": float(final_auc),
        "history": history,
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(payload, f)
    print(f"[OK] Model saved: {MODEL_OUT}")

    np.savez_compressed(
        PRED_CACHE,
        valid_probs=pred_aligned,
        valid_targets=target_aligned,
    )
    print(f"[OK] Pred cache saved: {PRED_CACHE}")

    metrics = {
        "auc": float(final_auc),
        "n_train_races": int(len(Xb_tr)),
        "n_valid_races": int(len(Xb_va)),
        "n_valid_rows": int(n_valid_rows),
        "n_features": int(n_features),
        "n_params": int(n_params),
        "history": history,
    }
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Metrics: {METRICS_OUT}")


if __name__ == "__main__":
    main()
