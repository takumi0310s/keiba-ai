# Session #56 D: V20 4-model ensemble

**作成**: 2026-05-09 (Session #56 D)
**tool**: tools/v20_ensemble.py
**model**: data/v20/models/v20_ensemble_v1.pkl
**metrics**: data/v18/session_56_ensemble_metrics.json

---

## 1. 個別 model AUC (V15 cache、 2024 valid 47K rows)

| model | AUC | 学習 時間 |
|-------|-----|---------|
| LGB (500 rounds、 early_stop 30) | **0.86868** | 5.2s (CPU) |
| XGB (500 rounds、 early_stop 30) | **0.86964** | 8.8s (CPU) |
| FT-Transformer (12 epochs) | **0.86644** | 5 min (CUDA) |
| IntraRace Attention (15 epochs) | **0.89940** | 15s (CUDA) |

---

## 2. ensemble 結果

| ensemble | AUC | コメント |
|---------|-----|--------|
| Equal weight (1/4 each) | 0.88871 | IR の 影響が 平均化されて 落ちる |
| Coarse grid (0.25 step) | 0.90013 | LGB=0、 XGB=0.2、 FT=0、 IR=0.8 |
| **Fine grid (0.05 step)** | **0.90025** | LGB=0.043、 XGB=0.043、 FT=0.087、 IR=0.826 |
| LGB+IR (2-model) | 0.89719 | -- |
| LGB+XGB+IR (3-model) | 0.89314 | -- |
| IR + LGB+XGB avg | 0.89738 | -- |

★ **Best 4-model AUC: 0.90025** ★

---

## 3. 最終 重み 分布

| model | 重み |
|-------|-----|
| LGB | **0.043** (5%) |
| XGB | **0.043** (5%) |
| FT-Transformer | **0.087** (9%) |
| **IntraRace Attention** | **0.826** (83%) |

→ **IR が圧倒的支配** (V13.5b の 0.35 を大幅超え)
→ V15 cache での V13.5b 形式 ensemble は **IR 1 model + 補助 3 model** が 最適

---

## 4. 比較

| model | AUC | コメント |
|-------|-----|--------|
| V15 LGB (200 rounds) | 0.86812 | Session #51 baseline |
| V15 V13.5b 4-model (CLAUDE.md) | 0.8788 | 旧 4-model (124 features) |
| V15 LGB+XGB (V14, V15 運用) | 0.8858 | 2-model 簡素化 |
| **V20 4-model ensemble (本 Session)** | **0.90025** | **★ ALL TIME BEST ★** |

→ V13.5b (0.8788) → V20 4-model (0.90025) = **+0.0214 大幅 改善**
→ LGB-only (0.8688) → V20 4-model (0.90025) = **+0.0316**

---

## 5. 結論

✅ V20 4-model ensemble AUC **0.90025**
✅ V13.5b (0.8788) を **+0.02 上回る** (★ ALL TIME BEST ★)
✅ IntraRace Attention が圧倒的支配 (重み 0.826)
✅ LGB / XGB / FT は 補助役 (合計 重み 0.17)
✅ 学習 総時間: ~6 分 (LGB 5s + XGB 9s + FT 5min + IR 15s)
✅ ensemble 学習 cache 保存済 (data/v20/models/v20_ensemble_v1.pkl)

**主結論**:
1. **race-level context (IR) が 最大 driver**
2. V20 学習 cache 構築後 (Sprint 6)、 同様 4-model ensemble で V20 production AUC 0.90+ 期待
3. predict_core 改修 必要 (推論時 IR の race-batch 処理)、 ただし PoC 段階で 検証完了
4. V13.5b 4-model 復活 + 改良 = **採用 GO**

**次 step (Session #56 E)**: dev/v20-ensemble push + Discord
