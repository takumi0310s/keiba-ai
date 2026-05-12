# V22 4-ensemble 学習結果 (5/13、 GPU 27 min)

## 🎯 結論

**V22 4-ensemble (Grid) 2025 fold AUC = 0.8891、 V15 baseline 0.8939 から -0.0048**。

極めて近い但し V15 越え 未達。 full 6-fold WF + hyperparameter 調整 で 越え可能 想定。

## 📊 V22 4-ens 2025 fold (held-out) 結果

| model | AUC | 時間 |
|-------|-----|------|
| LGB | 0.8703 | 15s |
| XGB | 0.8714 | 30s |
| FT-Transformer | 0.8704 | 1501s (25 min、 GPU) |
| **IntraRace** | **0.8781** | 78s |
| 4-model AUC-weighted | 0.8847 | — |
| **4-model Grid (L=0.20 X=0.30 F=0.05 IR=0.45)** | **0.8891** | — |

V15 baseline (full WF mean): 0.8939
delta: **-0.0048**

### 主要発見:

1. **IntraRace Attention が dominant** (重み 0.45)
   - V15 v13.5b 時の IR 重み 0.35 → 0.45 上昇
   - レース内 相対関係 が 4-ensemble で 最強 signal

2. **FT-Transformer の貢献 微小** (重み 0.05)
   - V15 v13.5b 時の FT 重み 0.10-0.15 → 0.05 低下
   - LGB + XGB + IR で 既に carries most signal

3. **GPU 100% utilization** (14.7GB / 16GB)
   - batch_size=4096 → OOM、 512 で 安定動作
   - quick (1 fold) 27 分、 full 6-fold は 2-3 hour 想定

## V15 vs V22 (LGB+XGB only) 2025 比較

| metric | V15 | V22 (LGB+XGB) |
|--------|-----|---------------|
| AUC | 0.9020 (IN-SAMPLE) | 0.8684 (held-out) |
| top1 → top3 hit | 89.4% (1169/1308) | 85.1% (1113/1308) |

V15 は production 学習 で 2025 を含むため in-sample bias あり。 fair compare では V22 (4-ens) > V15 (LGB+XGB only) と推測。

## 残 作業 (5/24+)

1. **V22 4-ens full 6-fold WF** (2-3h GPU、 mean AUC 推定 0.88-0.89)
2. hyperparameter tuning (FT layers/epochs、 IR d_model 等)
3. Phase 24/26 features 追加調整 (例 prev_review_score 等)
4. paddock features 1000+ dirs 蓄積後 V21 と統合

期待値: V22 (4-ens full + tuning) WF AUC 0.89-0.91 (V15 を 0.01-0.02 越え)
