# Phase 14 B: V20 evaluation (PoC + 4-model ensemble plan)

**作成**: 2026-05-10 (Session #90 Phase 14 B)
**前提**: Session #44 D/E で V20 PoC LGB 単 fold 完了 (AUC 0.8752)、 本 Phase は v2 + 4-model 学習 design

---

## 1. V20 PoC v1 (Session #44 E、 5/8) 既存 状態

### 1.1 構成

```
features = V15 (150) + V162 (20) + V17 (18) + sib_w5 (2) = 190 features
target  = is_top3
fold    = train 2015-2024、 test 2025 (single fold)
model   = LGB (early stopping、 num_boost_round=2000)
LEAK 除外 = 30 features
```

### 1.2 結果

| 項目 | V20 PoC v1 | V15 baseline (ensemble) |
|------|------------|-------------------------|
| AUC (BT 2025 OOS) | **0.8752** | 0.8856 |
| top1 → top3 hit | 78.47% | — |
| 学習時間 | 0.5 分 | — |
| n_train / n_test | 479,783 / 47,497 | — |

→ V20 PoC v1 LGB **単体** は V19 sib_w5 LGB 単体 と同等、 ensemble (LGB+XGB+FT+IR) で V15 0.8856 → V20 期待 0.890-0.895 を確認するには 4-model 学習必要

---

## 2. V20 PoC v2 plan (本 Phase 14 で着手不可、 5/24+ で実行)

### 2.1 features 拡張 (190 → 200-220)

| source | 追加候補 features | 期待 AUC delta |
|--------|------------------|---------------|
| TFJV HR (払戻 復活) | tfjv_trio_payout_avg, tfjv_tansho_payout, tfjv_umaren_payout | -0.001 (4/6 停止 解消) |
| TFJV SE 詳細 | tfjv_finish_time_norm, tfjv_agari_3f_norm, tfjv_horse_weight_change_3r | +0.001-0.003 |
| TFJV UM 90 年分 | sib_top3_rate_exp_w5_extended (1936-2025) | +0.001-0.003 |
| TFJV BS/OW/W5 | breeder_top3_5y, owner_top3_3y, w5_appearance_3y | +0.001-0.005 |
| JRDB 拡張 (Phase 11) | ranch / odds_change / return_horse_* / jockey_master | +0.001-0.005 |
| netkeiba マスター (Phase 13) | nk_ai_position / agari / upset / grade | +0.002-0.008 |
| **合計** | **+50-70 features** | **AUC +0.005-0.024** |

→ V20 期待 BT WF AUC: **0.890-0.913** (V15 0.8939 + 0.0-0.020、 4-model ensemble で 0.910-0.925 想定)

### 2.2 4-model ensemble 構成

| Model | role | 期待 学習時間 (CPU) |
|-------|------|--------------------|
| LightGBM | base、 速い | 5-10 分 |
| XGBoost | base、 LGB と相補 | 10-15 分 |
| FT-Transformer | tabular DL | **2-4 時間** (PyTorch、 CPU) |
| IntraRace Attention | レース内相対関係 | **1-2 時間** (custom、 CPU) |
| Grid weight optimization | ensemble 重み | 30-60 分 |

→ **合計 4-8 時間 (CPU)、 GPU あれば 1-2 時間**
→ **Phase 14 (本 session) では実行不能**、 5/24+ Phase 3 後半 で実行

---

## 3. V20 投入 schedule (Session #44 F 確定、 1 ヶ月前倒し)

| 期間 | 内容 |
|------|------|
| 5/8-5/16 | V18 sib_w5 LGB 単 fold ✅ 完了 (Session #43 C) |
| 5/16-5/22 | V20 features 拡張 (TFJV + JRDB + netkeiba 統合) |
| 5/23-5/29 | V20 4-model ensemble 学習 + WF 6-fold 検証 |
| 5/30-6/1 | V20 LIVE retro (5/30-5/31) |
| 6/2-6/7 | V20 paper trading + bug fix |
| **6/8 (日)** | **V20 投入候補 GO/no-go 判定** |

---

## 4. V20 GO 条件 (6/8 判定)

| 条件 | 閾値 | 確認方法 |
|------|------|---------|
| WF AUC | ≥ 0.880 | BT 2020-2025 6-fold |
| LIVE retro winner_top1 | ≥ 30% | 5/30-5/31 |
| shift_factor | ≤ 12x | LIVE / BT |
| paper trade ROI | ≥ 110% | 6/2-6/7 daily |
| LEAK 監査 | PASS | features 全件 corr 確認 |
| NAR AUC | ≥ 0.83 | NAR train data (Session #82 で hybrid plan) |

---

## 5. V15 投資保護

✅ V20 学習 data は 別 dir (data/tfjv/、 data/v20/)
✅ V15 model file 不変
✅ predict_core / daily_predict 完全不変
✅ 6/8 V20 GO 判定後も V15 並行運用 (1 ヶ月)、 安定確認後 archive

---

## 6. 結論

✅ V20 PoC v1 LGB 単 fold 既存 (BT AUC 0.8752、 5/8)
⚠ V20 PoC v2 + 4-model ensemble は 5/24+ Phase 3 後半 で実行
✅ Phase 14 は paper trade engine 整備で 5/24+ ready
⚠ Phase 14 本 session で 4-model 学習は CPU 数時間で完結不能

---

**Phase 14 B 完了** (Opus 4.7)
