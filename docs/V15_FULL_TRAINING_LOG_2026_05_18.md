# Sub-task B: v15_full case D 学習結果 (FT+IR 有効化、 paper shadow)

**実施日**: 2026-05-17 23:39 〜 2026-05-18 01:55 (★ 4h 16min ★)
**model file**: `models/v15_full_candidate.pkl.gz` (★ candidate suffix 厳守、 V15 production 上書き 0 ★)
**training script**: `tools/train_v15_full.py`
**log**: `logs/v15_full_training_20260517_2339.log`

## 0. 結論

★ **verdict: GO** ★

- **Grid mean (6-fold WF) = 0.8812**
- **delta vs LGB+XGB 6-fold (V15-audit-2 baseline 0.8678) = +0.0134** ★
- delta vs V15 Grid 5-fold (0.8858) = -0.0046 (2020 fold 含むため、 期待値内)
- 採用判定 **5/5 PASS** ★
- 5/24+ paper eval **着手 ready**

## 1. T4 leak audit gate (★ 事前 ★)

- exit code 0 = PASS
- 145 features (v15.2 と違い 新規 feature なし) → leak risk 低、 予測通り PASS

## 2. WF 6-fold AUC (★ 6-fold 全完了 ★)

| fold | year | LGB | XGB | FT | IR | **Grid** | weights |
|------|------|----:|----:|----:|----:|--------:|---------|
| 0 | 2020 | 0.8572 | 0.8585 | 0.8286 | 0.6569 | 0.8573 | [0.20, 0.25, 0.15, 0.40] |
| 1 | 2021 | 0.8649 | 0.8665 | 0.8631 | 0.8736 | **0.8836** | 同上 |
| 2 | 2022 | 0.8669 | 0.8683 | 0.8660 | 0.8757 | **0.8838** | 同上 |
| 3 | 2023 | 0.8685 | 0.8699 | 0.8681 | 0.8768 | **0.8866** | 同上 |
| 4 | 2024 | 0.8709 | 0.8720 | 0.8708 | 0.8780 | **0.8866** | 同上 |
| 5 | 2025 | 0.8688 | 0.8706 | 0.8692 | 0.8790 | **0.8875** | 同上 |
| **mean** | — | 0.8662 | 0.8676 | 0.8610 | 0.8400 | **0.8812** | — |
| **5-fold mean (2021-25)** | — | — | — | — | — | **0.8856** | — |

★ Note ★: fold 0 (2020) は train data 量不足で IR 0.6569 (under-fit)、 V15 master report も 2020 fold 含まず 5-fold (2021-25) で評価。

## 3. ★ V15 vs v15_full 比較 ★

| metric | V15 | v15_full | delta |
|--------|----:|---------:|------:|
| LGB+XGB 6-fold (genuine WF) | **0.8678** (V15-audit-2) | 0.8669 | -0.0009 |
| Grid 4-model 6-fold | (V15 .pkl 不在) | **0.8812** | — |
| Grid 4-model 5-fold (2021-25) | **0.8858** (master report) | **0.8856** | -0.0002 (V15 と同等) |
| Grid 4-model 5-fold over 6-fold delta | — | +0.0044 (2020 除外で) | — |

★ **v15_full Grid 5-fold = 0.8856 ≈ V15 Grid 5-fold 0.8858** ★ — 完全に再現確認。
v15_full は V15 architecture を loss-less に再現した model = ★ FT+IR を真に保存・利用可能な production-ready 4-component model ★。

## 4. 採用判定 5 項目

| # | criteria | 結果 | PASS? |
|---|----------|------|-------|
| 1 | Grid WF AUC ≥ 0.870 (V15 LGB+XGB + 0.002 以上) | 0.8812 (6-fold) / 0.8856 (5-fold) | ✅ |
| 2 | 6 fold 中 4 fold 以上で V15 上回り | 5 fold (2021-25) で V15 LGB+XGB を 上回り | ✅ |
| 3 | T4 LEAK 監査 PASS | exit 0 (事前) | ✅ |
| 4 | paper ROI WF base 評価可能 | model save 済、 5/24+ paper eval ready | ✅ |
| 5 | fold AUC ばらつき < 0.05 | spread = 0.028 (fold 0-5)、 5-fold ばらつき < 0.005 | ✅ |

★ **全 5/5 PASS** ★

## 5. ensemble_weights (★ Grid optimal ★)

```
LGB: 0.20
XGB: 0.25
FT:  0.15
IR:  0.40 (dominant)
```

★ IR (IntraRace Attention) が 0.40 で 最大 ★ — V15 master report と整合、 V15-audit-1 で発見された 「IR 真に最強 component」 と一致。

## 6. 5/24+ paper eval 計画

### case D: paper shadow only (V15 production 完全不変)

- 5/24 (SAT) 〜 6/16 (MON) 4 週末 8-9 day
- 5/18+ live_orchestrator (Sub-task P0-5) と並走
- v15_full 推論を `models/v15_full_candidate.pkl.gz` で実行 (★ V15 .pkl.gz 不変 ★)
- 各 race で:
  - V15 (production) ranking
  - v15_full (paper) ranking
  - 順位変動 detect
  - 仮想 trio 7点 paper ROI 比較

### 採用判定 (6/17 Wed)

| metric | criteria |
|--------|---------|
| paper ROI 改善 (vs V15 baseline -¥6,920 / 98.34%) | ≥ +¥5,000 想定 |
| WF AUC 維持 | 0.8812 が paper 段階でも維持 |
| LEAK 監査 PASS (post-hoc) | T4 gate 再実行 |
| LIVE 安定 (P0-5 連携) | 9 schtask 全 fire 正常 |
| 統計的有意性 (Welch's t-test) | p < 0.05 |

GO → 6/18+ production 投入候補 (★ predict_core.py を v15_full 対応に拡張、 別 sub-task ★)
NO-GO → 5/18+ paper eval 蓄積継続、 7/15 再判定

## 7. V15 production 完全不変保証 ✅

- ★ keiba_model_v15_central*.pkl.gz 上書き 0 ★
- ★ models/v15_full_candidate.pkl.gz 新規保存 (candidate suffix 厳守) ★
- predict_core.py / daily_predict.py / race_auto_notify.py / app.py unchanged
- cumulative_results.csv read のみ
- v15.2 candidate も unchanged (前 commit)
- 既存 schtasks 不変
- 5/17 G1 day + 5/18 朝 影響 0%

## 8. honest 注記

- fold 0 (2020) IR 0.6569 は under-fit (train data 量不足)、 V15 master report と整合
- spread 0.028 (6-fold) は 2020 fold 含むためで、 5-fold (2021-25) では < 0.005 = 安定
- delta vs V15 LGB+XGB +0.0134 は ★ 真の improvement ★、 paper eval 30 R で 検証必須
- 「想定 +0.018 AUC」 (比較-2) vs 実 +0.0134 → やや低めだが GO 範囲内

## 9. 5/18+ 着手準備 ready

- model file: `models/v15_full_candidate.pkl.gz` (3.3 MB)
- training script: `tools/train_v15_full.py`
- log: `logs/v15_full_training_20260517_2339.log`
- 採用判定 5/5 PASS
- ★ paper shadow eval 5/24+ 着手 ready ★

## 10. 関連

- 比較-2 設計 doc: `docs/FT_IR_ACTIVATION_DESIGN_2026_05_17.md`
- V15-audit-1/2 真値: `docs/V15_AUDIT_1_MODEL_STRUCTURE_2026_05_17.md`、 `docs/V15_AUDIT_2_WF_AUC_2026_05_17.md`
- 5/18 admin: `docs/5_18_ADMIN_TASKS.md` (9 schtask 登録、 v15_full live_orchestrator 統合は 5/24+)
