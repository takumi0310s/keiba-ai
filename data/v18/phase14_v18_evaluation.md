# Phase 14 A: V18 sib_w5 evaluation

**作成**: 2026-05-10 (Session #90 Phase 14 A)
**前提**: Session #43 C で V18 sib_w5 学習完了 (5/8)、 本 Phase は 既存 model の集約 + paper trade ready 化

---

## 1. V18 sib_w5 既存 状態

### 1.1 model file (5/8 学習済、 不変)

```
data/v18/v18v19_sib_exp_w5/
├── v18_lgb_sib_exp_w5.txt          (LightGBM、 21 MB、 190 features)
├── v19_lgb_sib_exp_w5.txt          (LightGBM、 21 MB、 190 features)
├── v18_sib_exp_w5_oos_2025.csv     (BT 2025 OOS predictions)
├── v19_sib_exp_w5_oos_2025.csv
├── sib_exp_w5_metrics.json
└── sib_exp_w5_retro_5_2_5_3_predictions.csv  (LIVE retro 5/2-5/3)
```

→ Phase 14 では **新規学習せず、 既存 model を paper trade engine から参照**

### 1.2 学習構成 (5/8 確定)

| 項目 | 値 |
|------|----|
| target (V18) | is_win (単勝) |
| target (V19) | is_top3 (複勝) |
| features | 190 (V15 150 + v162 + v17 + sib_w5) |
| sib 旧版 削除 | sib_top3_rate, sib_shinba_wr |
| sib_w5 新規 | sib_top3_rate_exp_w5, sib_shinba_wr_exp_w5 |
| LEAK 除外 | 31 features (TYB+SKB+OLD) |
| fold | train 2015-2024、 test 2025 (single fold) |
| n_train / n_test | 479,783 / 47,497 |
| 学習時間 | 約 1.4 分 |

### 1.3 BT 2025 OOS

| Model | AUC | logloss | winner_top1 |
|-------|-----|---------|-------------|
| V18 sib_w5 | **0.8847** | 0.1864 | 0.4550 |
| V19 sib_w5 | **0.8752** | 0.3483 | — |
| V18 v1 (sib_exp full) | 0.8845 | — | 0.4588 |
| V18 delta (w5 vs v1) | +0.0002 | — | -0.0038 |

### 1.4 LIVE retro 5/2-5/3 (29 races)

| Model | LIVE winner_top1 | vs no_sib | shift_factor |
|-------|-----------------|-----------|--------------|
| OLD (sib 含 ens、 リーク) | 34.48% | +10.34pt | 1.39x |
| NO_SIB | 24.14% | (基準) | 1.90x |
| sib_exp v1 (full expanding) | 31.03% | +6.89pt | 1.48x |
| **sib_w5 (window=5)** | **34.48%** ★ | **+10.34pt** ★ | **1.32x** ★ |

→ **window=5 expanding が OLD と LIVE 完全同等、 リーク 0%、 shift 最良**

---

## 2. V15 投資保護 (絶対遵守)

✅ V18 model file は別 dir (data/v18/v18v19_sib_exp_w5/)
✅ V15 model md5: `842b9a5f305c793ed8fa54a74e06b836` 不変
✅ predict_core / daily_predict / app.py 完全不変
✅ 累計収支 +¥14,140 死守

---

## 3. Phase 14 で V18 を投入しない理由

| # | 理由 |
|---|------|
| 1 | Session #38 で 5/16 V18/V19 投入 NO-GO 確定 (sib_*_exp 未確証時点) |
| 2 | Session #43 C で sib_w5 LIVE 完全回復 (34.48%) を確認も、 LIVE データ 1 日分のみ |
| 3 | 4-model ensemble (LGB+XGB+FT+IR) で本来の V18/V19 構成、 LGB 単体は参考値 |
| 4 | 5/16+ V18 trial 投入候補は paper trade で 1 週間以上 LIVE retro 蓄積後 |

→ **Phase 14 では paper trade engine 整備で V18 → 5/17 (土) 本番試行 ready**

---

## 4. 5/17 (土) V18 paper trade GO 条件

| 条件 | 状態 | 備考 |
|------|------|------|
| ✅ V18 model file 存在 | ✅ 完了 (5/8) | data/v18/v18v19_sib_exp_w5/v18_lgb_sib_exp_w5.txt |
| ✅ paper_trade_engine.py | ✅ 完了 (本 Phase 14 B) | tools/paper_trade_engine.py |
| ⚠ V18 5/10+ shadow predictions | ⚠ 未生成 | 別 session で V18 inference pipeline 整備 |
| ⚠ LIVE retro 5/4-5/16 | ⚠ 部分 (5/2-5/3 のみ) | sib_w5 効果の継続性確認待ち |
| ⚠ 4-model ensemble | ⚠ LGB 単体のみ | XGB+FT+IR は Phase 3 後半 (5/24+) |

→ **5/17 (土) は paper trade のみ (実弾 投入 NO)、 5/24+ Phase 3 後半 で V20 構築合流**

---

## 5. 結論

✅ V18 sib_w5 model 既存 (5/8、 BT AUC 0.8847、 LIVE 34.48%)
✅ Phase 14 は新規学習なし、 paper trade ready 化
✅ V15 production 完全不変
⚠ 5/17 V18 実弾投入 NO、 paper trade 蓄積から判断

---

**Phase 14 A 完了** (Opus 4.7)
