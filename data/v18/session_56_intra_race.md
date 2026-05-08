# Session #56 C: IntraRace Attention 学習結果

**作成**: 2026-05-09 (Session #56 C)
**tool**: tools/v20_intra_race_attention.py
**model**: data/v20/models/v20_intra_race.pkl
**pred cache**: data/v20/models/v20_ir_pred.npz

---

## 1. 学習設定

| 項目 | 値 |
|------|----|
| base data | data/_v15_optuna_df_cache.pkl.gz (V15 cache) |
| n_features | 145 |
| race-level group key | **race_id_str** (8-char、 13,824 unique races/train) |
| n_train_races | 13,824 |
| n_valid_races | 3,454 (year 2024) |
| max_horses | 18 (pad with mask) |
| target | finish ≤ 3 |
| device | CUDA |
| architecture | IntraRaceAttention (V13.5b 継承) |
| d_model | 64 |
| n_heads | 4 |
| n_layers | 2 |
| dropout | 0.15 |
| batch_size | 128 races |
| lr | 1e-3, AdamW + Cosine |
| epochs | 15 |
| n_params | 78,529 |

---

## 2. 重要 修正 ★

初版で `race_id` (10-char) を group key に使用 → 各 race が 1 horse のみ → 効果なし AUC 0.8669。

**修正**: `race_id_str` (8-char、 13,824 race / 1 race ≈ 13.7 horses) を 使用 → 真の IntraRace 学習。

---

## 3. 学習 経過

| epoch | loss | AUC | elapsed |
|-------|------|-----|---------|
| 1 | 0.7482 | 0.8840 | 1.2s |
| 2 | 0.6586 | 0.8931 | 0.9s |
| 3 | 0.6373 | 0.8944 | 0.9s |
| 4 | 0.6283 | 0.8977 | 0.9s |
| 5 | 0.6199 | 0.8977 | 0.9s |
| 6 | 0.6139 | 0.8983 | 0.9s |
| 7 | 0.6102 | 0.8983 | 0.9s |
| 8 | 0.6032 | 0.8988 | 0.9s |
| 9 | 0.5985 | 0.8991 | 0.9s |
| 10 | 0.5945 | 0.8991 | 0.9s |
| 11 | 0.5895 | 0.8992 | 0.9s |
| 12 | 0.5873 | 0.8993 | 0.9s |
| 13 | 0.5833 | 0.8994 | 0.9s |
| 14 | 0.5811 | 0.8994 | 0.9s |
| 15 | 0.5789 | 0.8994 | 0.9s |

**Final AUC: 0.89940** (★ +0.018 vs LGB ★)

---

## 4. 比較

| model | AUC | コメント |
|-------|-----|--------|
| V15 LGB (200 rounds、 Session #51 B) | 0.86812 | LGB 単体 baseline |
| V20 FT-Transformer (Session #56 B) | 0.86644 | FT 単体、 -0.00168 |
| **V20 IntraRace Attention (Session #56 C)** | **0.89940** | **+0.03128 vs LGB** |
| V13.5b 4-model (CLAUDE.md) | 0.8788 | IR 採用済 (重み 0.35) |

→ **IntraRace alone が V13.5b 4-model 全体 を 上回る** (0.8994 > 0.8788)
→ 理由: race-level context (相対 比較) が 強力 信号
→ 145 features (V15、 V13.5b は 124) で 学習、 expanding × cross 系 features の 相互作用 を IR が 効率的活用

---

## 5. 訓練 速度

- 総時間: **約 15 秒** (15 epochs)
- 1 epoch: 0.9-1.2s (race-batched なので 高速)
- GPU memory: 適正範囲内 (batch 128 races × 18 horses × 145 features)

---

## 6. 保存物

| file | 内容 |
|------|------|
| `data/v20/models/v20_intra_race.pkl` | model state_dict + scaler + config |
| `data/v20/models/v20_ir_pred.npz` | 行-aligned valid_probs (47K) + targets (D ensemble 用) |
| `data/v18/session_56_intra_race_metrics.json` | metrics + history |

---

## 7. 結論

✅ IntraRace Attention alone AUC **0.8994** (超強力!)
✅ V15 LGB 比 +0.0313、 V13.5b 4-model 比 +0.0206
✅ 78K parameters の 軽量 model で 15 秒 学習
✅ race-level context (相対 比較) が 大幅貢献
✅ valid 予測 cache 保存済 (D ensemble で 利用)

**重要発見**: race-level batching が IR の 真価を発揮。 V13.5b 当時の 重み 0.35 は 妥当 (alone なら 主軸候補)。

**次 step (Session #56 D)**: 4-model ensemble (LGB + XGB + FT + IR)
