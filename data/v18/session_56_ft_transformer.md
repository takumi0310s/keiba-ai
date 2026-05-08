# Session #56 B: FT-Transformer 学習結果

**作成**: 2026-05-09 (Session #56 B)
**tool**: tools/v20_ft_transformer.py
**model**: data/v20/models/v20_ft_transformer.pkl
**pred cache**: data/v20/models/v20_ft_pred.npz

---

## 1. 学習設定

| 項目 | 値 |
|------|----|
| base data | data/_v15_optuna_df_cache.pkl.gz (V15 cache) |
| n_features | 145 |
| n_train | 189,473 (year 2020-2023) |
| n_valid | 46,752 (year 2024) |
| target | finish ≤ 3 |
| device | CUDA |
| architecture | FTTransformer (V13.5b 継承) |
| d_token | 32 |
| n_heads | 4 |
| n_layers | 3 |
| dropout | 0.15 |
| batch_size | 4,096 |
| lr | 1e-3, AdamW + Cosine |
| epochs | 12 (early_stop patience=5) |
| n_params | 36,097 |

---

## 2. 学習 経過

| epoch | loss | AUC | elapsed |
|-------|------|-----|---------|
| 1 | 0.8858 | 0.8446 | 17.6s |
| 2 | 0.7738 | 0.8579 | 24.9s |
| 3 | 0.7485 | 0.8600 | 27.8s |
| 4 | 0.7416 | 0.8625 | 26.8s |
| 5 | 0.7320 | 0.8608 | 26.7s |
| 6 | 0.7286 | 0.8643 | 28.3s |
| 7 | 0.7257 | 0.8652 | 27.2s |
| 8 | 0.7242 | 0.8649 | 26.2s |
| 9 | 0.7183 | 0.8661 | 29.7s |
| 10 | 0.7164 | 0.8662 | 29.3s |
| 11 | 0.7147 | 0.8664 | 27.0s |
| 12 | 0.7143 | 0.8664 | 27.4s |

**Final AUC: 0.86644**

---

## 3. 比較

| model | AUC | n_features | コメント |
|-------|-----|-----------|--------|
| V15 LGB (200 rounds、 Session #51 B) | 0.86812 | 145 | LGB 単体 baseline |
| **V20 FT-Transformer alone** | **0.86644** | 145 | **FT 単体、 -0.00168** |
| V13.5b 4-model (CLAUDE.md) | 0.8788 | 124 | LGB+XGB+FT+IR ensemble |

→ FT-Transformer 単体 は LGB 単体 比 -0.00168 (近い性能)
→ ensemble 期待: FT は 異なる 信号源、 ensemble で +0.001-0.005 寄与

---

## 4. 訓練 速度

- 総時間: 約 5 分 (12 epochs)
- 1 epoch: 約 27 秒
- GPU memory: 適正範囲内 (batch 4096 で OK)

---

## 5. 保存物

| file | 内容 |
|------|------|
| `data/v20/models/v20_ft_transformer.pkl` | model state_dict + scaler + config |
| `data/v20/models/v20_ft_pred.npz` | valid_probs (47K) + valid_targets (ensemble 用) |
| `data/v18/session_56_ft_transformer_metrics.json` | metrics + history |

---

## 6. 結論

✅ FT-Transformer alone AUC 0.86644 (V15 LGB 0.86812 比 -0.00168)
✅ 36K parameters の 軽量 model で 学習成功
✅ GPU 利用 5 分で 完了
✅ valid 予測 cache 保存済 (D ensemble で 利用)

**次 step (Session #56 C)**: IntraRace Attention 学習
