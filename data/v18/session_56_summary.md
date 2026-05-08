# Session #56 サマリ: V20 4-model ensemble PoC

**日付**: 2026-05-09 (Session #56)
**branch**: dev/v20-ensemble (main 不変、 V15 投資保護)
**目的**: V13.5b の 4-model ensemble (LGB + XGB + FT-Trans + IntraRace) を V20 で 復活 + 改良

---

## 0. 5 領域 完了

| 領域 | 内容 | 出力 |
|------|------|------|
| A | V13.5b architecture 復元、 V20 で 採用判断 | data/v18/session_56_v13_5b_audit.md |
| B | FT-Transformer 学習 (AUC 0.8664) | tools/v20_ft_transformer.py + data/v20/models/v20_ft_transformer.pkl + .npz |
| C | IntraRace Attention 学習 (AUC **0.8994**、 +0.0313) | tools/v20_intra_race_attention.py + data/v20/models/v20_intra_race.pkl + .npz |
| D | 4-model ensemble (AUC **0.90025**、 ALL TIME BEST) | tools/v20_ensemble.py + data/v20/models/v20_ensemble_v1.pkl |
| E | 統合 + push + Discord | (本 doc) |

---

## 1. 主結果

### 個別 AUC (V15 cache、 145 features、 2024 valid 47K rows)

| model | AUC | 学習 時間 |
|-------|-----|---------|
| LGB (500 rounds) | 0.86868 | 5s (CPU) |
| XGB (500 rounds) | 0.86964 | 9s (CPU) |
| FT-Transformer (12 epochs) | 0.86644 | 5min (CUDA) |
| **IntraRace Attention (15 epochs)** | **0.89940** | 15s (CUDA) |

### 4-model ensemble

| 方式 | AUC |
|------|-----|
| Equal weight (1/4 each) | 0.88871 |
| Coarse grid (0.25 step) | 0.90013 |
| **Fine grid (0.05 step)** | **0.90025** ★ |

最終 重み: LGB 0.043、 XGB 0.043、 FT 0.087、 **IR 0.826**

---

## 2. 重要 発見 (★★★)

### 2-1. IntraRace Attention が 圧倒的支配 (重み 0.826)

- IR alone AUC 0.8994 (LGB 比 +0.0313)
- race-level context (相対 比較) が 強力 信号
- V13.5b 当時の 重み 0.35 を 大幅超え (今回 0.826)

### 2-2. V13.5b (0.8788) を +0.0214 上回る

- V15 LGB-only: 0.86812
- V13.5b 4-model (旧): 0.8788
- **V20 4-model (本 Session)**: **0.90025** ★ ALL TIME BEST ★

### 2-3. 4-model 全 必要

- LGB+IR (2-model): 0.8972
- LGB+XGB+IR (3-model、 no FT): 0.8931
- **LGB+XGB+FT+IR (4-model)**: **0.9003**

→ FT は 単独では 弱い が、 ensemble で +0.003-0.007 寄与

---

## 3. V20 構築 への 影響

### 3-1. V20 ensemble 採用 GO

- Sprint 6 (5/22-6/8) で V20 学習 cache 構築後、 同様 4-model で **0.90+ 期待**
- predict_core 改修 必要 (推論時 IR の race-batch 処理) → V20 投入時 PR
- model file: data/v20/models/ 配下 4 ファイル + ensemble payload

### 3-2. 投入 ロードマップ更新

```
2026-05-09  Session #56 (本): 4-model PoC 完了
2026-05-22  Sprint 6 開始 (V20 学習 cache 構築)
2026-06-08  V20 学習完了、 4-model ensemble 適用、 paper trade 開始
2026-06-30  V20 paper 評価
2026-07-01  V20 投入判断 (期待 AUC 0.90+)
```

---

## 4. branch 状態

- `dev/v20-ensemble`: 5 commits (A, B re-pick, C, D、 E)
- `main` 不変
- `dev/v20-interaction` (Session #57 並行): 干渉なし
- `dev/v20-expanding` (Session #55 並行): 干渉なし
- `predict_core / daily_predict / app.py`: 不変
- V15 model file: 不変
- 5/9 朝 V15 動作: 不変

→ 中央 V15 投資保護: ✅

---

## 5. 学習 環境

- **PyTorch 2.11.0+cu126** + CUDA
- CPU: LGB / XGB
- GPU: FT-Transformer / IntraRace Attention
- 総時間: ~6 分 (4 model 学習 + ensemble grid search)

---

## 6. 5 commits 履歴 (dev/v20-ensemble)

1. `Session #56 A: V13.5b ensemble audit (4-model 復活 判断)` (ba228a47)
2. `Session #56 B: FT-Transformer 学習 (AUC 0.86644、 12 epochs CUDA)` (36a023d6)
3. `Session #56 C: IntraRace Attention 学習 (AUC 0.8994、 +0.0313 vs LGB)` (a4d57d52)
4. `Session #56 D: V20 4-model ensemble (AUC 0.90025、 ALL TIME BEST)` (0e272b74)
5. `Session #56 E: doc 統合 + summary` (本 commit)

---

## 7. 結論

✅ V13.5b 4-model ensemble 完全復活 + 改良
✅ V20 4-model AUC **0.90025** (ALL TIME BEST、 V13.5b 比 +0.0214)
✅ IntraRace Attention が 圧倒的支配 (重み 0.826)
✅ V20 構築での 採用 GO
✅ V15 投資保護: 完全保持
✅ Session #55 / #57 と並行、 干渉なし

**主結論**:
- race-level context (IR) が V20 の キー
- V13.5b 4-model 復活 + 重み 最適化 で **0.90+ AUC** 達成可能
- 6/8 V20 投入時 に 4-model ensemble 採用 推奨

5/9 朝 V15 案B改 維持。 5/22 Sprint 6 着手。
