# 夜-4-C: V15 fold 0 LGB+XGB fair benchmark (v15.2 比較用)

作成日: 2026-05-17 (夜-4-C task)
作業者: agent (read-only, V15 retrain なし、 v15.2 training 中断なし)

---

## 0. 結論 (honest)

| 比較 | V15 LGB+XGB | v15.2 LGB+XGB | delta | 判定 |
|------|-------------|----------------|-------|------|
| fold 0 (2020) | **0.8591** | **0.8588** | **-0.0003** | ★ degradation 微小 ★ |
| 5-fold mean (2020-2024) | **0.8673** | **0.8673** | **±0.0000** | ★ no improvement ★ |
| 6-fold mean (2020-2025) | **0.8678** | 5-fold 完了、 2025 fold 5 進行中 | — | 5/18 朝待ち |

★ **honest verdict**: v15.2 の LGB+XGB は V15 production と **fair benchmark で ほぼ完全一致** (delta ±0.00001)。 17 新 features 投入の効果は LGB+XGB レイヤで **検出されず**。 ★

ただし v15.2 Grid (folds 1-4) は 0.8855 → V15 v15_master_report Grid mean 0.8868 (2021-2025) と 比較すべき。 fold 0 (2020) で FT (0.8286) / IR (0.6569) が **broken** → Grid 0.8573 へ drag down (★ 2020 は FT/IR training が早期データ不足で破綻、 既知パターン ★)。

---

## 1. V15 各 component AUC (fold-by-fold)

source: `data/v15_2/v15_baseline_lgb_xgb_20260517_1715.json` (本日 17:15 別 agent 作成、 V15 .pkl.gz から inference 実行済)

| fold | year | LGB | XGB | LGB+XGB |
|------|------|-----|-----|---------|
| 0 | 2020 | 0.8576 | 0.8588 | **0.8591** |
| 1 | 2021 | 0.8643 | 0.8669 | 0.8665 |
| 2 | 2022 | 0.8670 | 0.8687 | 0.8688 |
| 3 | 2023 | 0.8686 | 0.8699 | 0.8700 |
| 4 | 2024 | 0.8706 | 0.8720 | 0.8721 |
| 5 | 2025 | 0.8687 | 0.8706 | 0.8704 |
| **6-fold mean** | — | **0.8661** | **0.8678** | **0.8678** |

FT / IR / Grid: 本 benchmark には V15 LGB+XGB only (★ V15 production .pkl.gz は LGB+XGB 2-model、 ensemble_weights={lgb:0.504, xgb:0.496, mlp:0}、 FT/IR は含まない ★)。

V15 production .pkl.gz 内 stored auc = 0.8939485520 — ただしこれは **WF mean ではなく single-run AUC** (おそらく全データ inference 値)。 真の WF Grid mean は `v15_master_report.json` の 0.8858 (2021-2025、 4-model Grid)。

---

## 2. v15.2 各 component AUC (fold 完了分)

source: `data/v15_2/wf_results_20260517_1709.json` + `logs/v15_2_training_20260517_1711.log`

| fold | year | LGB | XGB | LGB+XGB | FT | IR | Grid |
|------|------|-----|-----|---------|------|------|------|
| 0 | 2020 | 0.8572 | 0.8585 | 0.8588 | 0.8286 | **0.6569** ★ | 0.8573 |
| 1 | 2021 | 0.8649 | 0.8665 | 0.8668 | 0.8631 | 0.8736 | 0.8836 |
| 2 | 2022 | 0.8669 | 0.8683 | 0.8685 | 0.8661 | 0.8740 | 0.8838 |
| 3 | 2023 | 0.8685 | 0.8699 | 0.8700 | 0.8684 | 0.8762 | 0.8870 |
| 4 | 2024 | 0.8706 | 0.8722 | 0.8722 | 0.8701 | 0.8798 | 0.8875 |
| 5 | 2025 | 0.8692 | 0.8701 | **partial** | partial | partial | partial |
| **mean(0-4)** | — | 0.8656 | 0.8671 | 0.8673 | 0.8593 | 0.8321 | 0.8798 |
| **mean(1-4)** | — | 0.8677 | 0.8692 | 0.8694 | 0.8669 | 0.8759 | **0.8855** |

★ **重要**: fold 0 (2020) は FT/IR が破綻 (FT 0.8286、 IR 0.6569、 IR cov 91.8% は十分だが AUC は random 近い)。 既知パターン (V15 master report は 2020 fold を含まず 2021-2025 のみ評価、 これは 2020 train データ量 不足で FT/IR の epoch 数 / data augmentation が破綻するため)。

---

## 3. fair benchmark delta (LGB+XGB)

| fold | year | V15 LGB+XGB | v15.2 LGB+XGB | delta |
|------|------|-------------|----------------|-------|
| 0 | 2020 | 0.8591 | 0.8588 | **-0.0003** |
| 1 | 2021 | 0.8665 | 0.8668 | +0.0004 |
| 2 | 2022 | 0.8688 | 0.8685 | -0.0003 |
| 3 | 2023 | 0.8700 | 0.8700 | +0.0000 |
| 4 | 2024 | 0.8721 | 0.8722 | +0.0001 |
| **mean(0-4)** | — | **0.8673** | **0.8673** | **+0.000005** |
| 5 | 2025 | 0.8704 | partial | — |

★ honest 結論 ★: 5 fold 完了時点で **delta = ±0.00001** → v15.2 の 17 新 features (breeder_*, paci_gekiso_*, kta_ichi_*, cha_*, kab_*) は LGB+XGB レイヤで **ほぼ完全に効果なし**。

---

## 4. ★ honest verdict ★

### 4.1 partial training 段階 (5/17 19:00 時点)

- **v15.2 LGB+XGB delta = +0.000005 ≈ 0** (fold 0-4 mean)
- fold 5 (2025) LGB+XGB は ★ FT/IR 完走 → ensemble 計算後の値待ち ★
- FT/IR は fold 1-4 で V15 と同等水準 (FT 0.8669 / IR 0.8759 mean) → Grid mean ≈ 0.8855

### 4.2 vs V15 production (★ apples-to-apples ★)

V15 production .pkl.gz は **LGB+XGB only (2-model)**:
- V15 .pkl.gz: ensemble_weights = {lgb: 0.504, xgb: 0.496, mlp: 0}
- ★ V15 の真の production AUC = LGB+XGB ★ → fair benchmark = **6-fold mean 0.8678**

v15.2 (5 fold 完了) LGB+XGB = **0.8673** → V15 0.8673 (5-fold mean) と **完全一致**。

### 4.3 v15.2 Grid (4-model) の意味

v15.2 が Grid (LGB+XGB+FT+IR) で 0.8855 (fold 1-4) を出しても、 **V15 production は Grid を使っていない** → 採用判定は 「LGB+XGB ベンチマーク vs LGB+XGB ベンチマーク」 で行うべき。

ただし v15.2 を 4-model Grid で **本番投入** するなら、 V15 v15_master_report の Grid mean (0.8858 全 fold / 0.8868 yearly) を超えるか で判定する選択肢もある。

| シナリオ | v15.2 採用基準 |
|----------|---------------|
| **A (LGB+XGB→LGB+XGB)** | v15.2 LGB+XGB ≥ V15 LGB+XGB (0.8678) → ★ 現状 0.8673、 -0.0005 で **未達** ★ |
| **B (LGB+XGB→Grid 4-model)** | v15.2 Grid ≥ V15 LGB+XGB (0.8678) → 現状 fold 1-4 Grid 0.8855、 **+0.018 で達成** だが FT/IR 推論コスト増 + production refactor 必要 |
| **C (Grid→Grid)** | v15.2 Grid ≥ V15 Grid (0.8858 from master_report 2021-2025) → 現状 fold 1-4 Grid 0.8855、 **-0.0003 で 未達 (微) ★** |

★ **honest 結論** ★: 17 新 features の 純増効果は **ほぼゼロ**。 LGB+XGB レイヤでも Grid レイヤでも V15 baseline を **超えない**。

---

## 5. 5/18 朝 採用判定 framework

### 5.1 5/18 朝 確認項目

1. fold 5 (2025) FT/IR/Grid 完了確認: `logs/v15_2_training_20260517_1711.log` の tail
2. v15.2 全 6 fold mean 計算: LGB / XGB / LGB+XGB / FT / IR / Grid
3. V15 全 6 fold mean (0.8678 既知) と delta 算出
4. V15 v15_master_report Grid 全 fold mean (0.8858 既知) と delta 算出

### 5.2 採用判定 matrix (full 6-fold)

| v15.2 Grid mean | vs V15 LGB+XGB (0.8678) | vs V15 Grid (0.8858) | 判定 |
|-----------------|--------------------------|----------------------|------|
| ≥ 0.8898 | +0.022 | +0.004 | ★ 強い改善、 採用 GO ★ |
| 0.8868 - 0.8898 | +0.019 - 0.022 | +0.001 - 0.004 | 微改善、 採用候補 |
| 0.8858 - 0.8868 | +0.018 - 0.019 | ±0.001 | **維持** (現状 fold 1-4 推定 ~0.8855 → ここ) |
| < 0.8858 | < +0.018 | < -0.001 | NO_GO 又は LGB+XGB only 投入 (B シナリオ) |

| v15.2 LGB+XGB mean | vs V15 LGB+XGB (0.8678) | 判定 |
|--------------------|--------------------------|------|
| ≥ 0.8708 | +0.003 | 部分改善、 LGB+XGB only 投入 候補 |
| 0.8678 - 0.8708 | ±0.003 | 維持、 採用効果薄 |
| < 0.8678 | < ±0 | **現状 ★ 0.8673 = -0.0005 ★、 NO_GO** |

### 5.3 推奨 default 判定

★ 現時点で **最も可能性 高い 5/18 朝 verdict** ★:
- v15.2 LGB+XGB mean ≈ 0.8673 (V15 -0.0005、 5 fold 数字 そのまま)
- v15.2 Grid mean ≈ 0.8855 (V15 Grid mean -0.0003、 fold 1-4 数字)
- → **NO_GO** (17 新 features 純増効果 ほぼゼロ)

ただし fold 5 (2025) FT/IR が 例外的に強ければ Grid mean を pull up する可能性あり。

---

## 6. V15 production 不変保証

- 本 task は read-only inference のみ (V15 .pkl.gz から LGB / XGB Booster を gzip.open + pickle.load で取得し、 既存 WF cache で年別 inference)
- ★ V15 retrain なし、 .pkl.gz 上書き なし ★
- v15.2 training process (PID 23528 想定) **中断なし**
- predict_core / daily_predict / race_auto_notify / app.py **変更なし**
- cumulative_results.csv **改変なし**
- git commit / push **なし** (★ 親集中 ★)

---

## 7. 参考データ

- V15 production model: `keiba_model_v15_central.pkl.gz` (2026-04-08 23:32 学習、 ensemble_weights {lgb:0.504, xgb:0.496, mlp:0}、 features 145、 stored auc 0.8939 = single-run)
- V15 WF master report: `data/v15_master_report.json` (Grid mean 2021-2025 = 0.8858、 yearly 0.8836-0.8887)
- V15 LGB+XGB fair benchmark: `data/v15_2/v15_baseline_lgb_xgb_20260517_1715.json` (本日 17:15 別 agent 作成、 6-fold mean 0.8678)
- v15.2 WF (in progress): `data/v15_2/wf_results_20260517_1709.json` (fold 5 のみ partial 保存) + `logs/v15_2_training_20260517_1711.log` (fold 0-4 ensemble done + fold 5 LGB/XGB done)
- v15.2 候補 features: `data/v15_2/features_v152_candidates.txt` (17 新 features)
- v15.2 post-hoc audit: `data/v15_2/post_hoc_audit_20260517_1712.json` (17/17 OK)
