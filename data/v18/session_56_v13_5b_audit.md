# Session #56 A: V13.5b ensemble audit

**作成**: 2026-05-09 (Session #56 A)
**目的**: V13.5b の 4-model ensemble 構造を 復元し V20 で 再採用判断

---

## 1. V13.5b architecture (既存 code 確認済)

### 1-1. ファイル

| file | 行数 | 内容 |
|------|----|------|
| `train/train_v135_ft_transformer.py` | 1024 | FT-Transformer + Sequential LSTM + IntraRace Attention 実装 |
| `train/train_v135b_intra_ensemble.py` | 499 | 4-model grid ensemble (LGB+XGB+FT+IR) |

### 1-2. 各 model の architecture

**FT-Transformer (Feature Tokenizer + Transformer)**:
- 各 numerical feature を `d_token=64` 次元の embedding に projection
- [CLS] token 追加 → multi-head self-attention で feature 間 相互作用 学習
- 構成: NumericalEmbedding (n_feat × d_token) + 3-layer Transformer (4-head) + classifier
- パラメータ: d_token=64, n_heads=4, n_layers=3, d_ff=128, dropout=0.1

**IntraRace Attention**:
- 同 race 内 全馬を **同時に** 入力 (batch, max_horses, n_features)
- self-attention で 「このメンバー の中で この馬 が どう位置するか」 学習
- 出力: 各馬の logit (per-horse)
- 構成: Linear (n_feat → d_model) + 2-layer Transformer (4-head) + per-horse head
- パラメータ: d_model=64, n_heads=4, n_layers=2

**Sequential Past-Race Model (LSTM)** (V13.5 で 検証済 → V13.5b で 不採用):
- 過去 5 走 の 11 features を LSTM で 時系列 学習
- 不採用理由: AUC 寄与小 + 学習複雑

### 1-3. V13.5b 結果 (CLAUDE.md 記載)

| year | LGB+XGB | FT-Transformer 追加 | IntraRace 追加 (4-model) |
|------|---------|-------------------|------------------------|
| 平均 | 0.8656 (v13.4) | 0.8659 (v13.5) | **0.8788** (v13.5b) |
| 改善 | -- | +0.0003 | +0.0132 |

→ **IntraRace が 最大貢献** (重み 0.35、 typical Grid 重み: LGB=0.25, XGB=0.25-0.30, FT=0.10-0.15, IR=0.35)

---

## 2. V14 / V15 で 4-model → 2-model に簡素化された理由

V14, V15 で LGB+XGB のみ運用:
- 学習 速度 重視 (FT/IR は GPU で 数十分)
- 運用 simplicity (predict_core で 4-model 推論 重い)
- AUC は LGB+XGB でも 0.8858 で 十分高い

---

## 3. V20 で 4-model 復活 する 妥当性

| 軸 | 評価 |
|---|------|
| AUC 改善 期待 | +0.005-0.015 (V13.5b 経験から) |
| 学習 工数 | 高 (FT は数十分 / IR は 1h+) |
| 推論 速度 | 中 (1 race < 5s なら OK) |
| GPU 必要 | ✓ (CUDA 利用可、 確認済) |
| 投資保護 | ✓ (V15 不変、 dev/v20-ensemble にて 並行) |

→ **V20 で 4-model 復活 採用 判断**

---

## 4. data 状況

- **V20 cache**: 不在 (Session #44 PoC の 時の 一時 data、 保存なし)
- **V15 cache**: 利用可能 (data/_v15_optuna_df_cache.pkl.gz、 145 features、 527,280 rows)

→ 本 Session #56 では **V15 cache を base** に FT-Transformer / IntraRace / 4-model ensemble を 検証。 V20 学習 cache は Sprint 6 (5/22-6/8) で 構築予定。

---

## 5. PoC 設計 (B/C/D)

| 領域 | 内容 | 期待 AUC |
|------|------|---------|
| B (FT-Transformer) | V15 145 features を base に FT 学習、 alone AUC 計測 | 0.85-0.87 |
| C (IntraRace Attention) | V15 145 features を per-race batch で 学習 | 0.85-0.87 |
| D (4-model ensemble) | LGB + XGB + FT + IR、 weight grid optimization | **+0.005-0.015** vs LGB+XGB |

### 学習 設定

- **time-based split**: 2020-2023 train (189K rows)、 2024 valid (47K rows)
- **early stopping**: 10 epochs
- **batch size**: FT 4096、 IR per-race (max 18 horses)
- **device**: CUDA (PyTorch 2.11.0+cu126)

---

## 6. リスク + 緩和

| risk | 緩和 |
|------|------|
| GPU OOM | batch size 縮小、 chunked 推論 |
| 学習 時間長 | n_epochs=20、 early_stopping=5 で 短縮 |
| FT 過学習 | dropout=0.1-0.2、 weight_decay=1e-5 |
| IR 過学習 | per-race batch、 mask で固定長 |
| ensemble 重み 不安定 | grid search 5×5×5 で 安定化 |

---

## 7. 結論

✅ V13.5b 4-model ensemble code 完全復元可能 (`train/train_v135_ft_transformer.py`)
✅ FT-Transformer + IntraRace Attention の architecture 確認済
✅ PyTorch 2.11.0 + CUDA 利用可能
✅ V20 cache 不在 → V15 cache (145 features、 527K rows) で PoC

**次 step (Session #56 B)**: tools/v20_ft_transformer.py 作成、 FT-Transformer alone AUC 計測
