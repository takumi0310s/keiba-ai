# B-4 v15_full 真の verdict honest fair benchmark (2026-05-18)

audit date: 2026-05-18  
auditor: Sub-task b (read-only)  
V15 production: 完全不変 (keiba_model_v135_central*.pkl.gz 未変更)

---

## 1. ensemble_weights 真値

| key | value |
|-----|-------|
| version | v15_full_candidate |
| trained_at | 2026-05-18T01:55:38 |
| features | 145 (V15 production と同一) |
| lgb weight | 0.2333 |
| xgb weight | 0.2917 |
| ft weight | 0.1250 |
| ir weight | 0.3500 |

注: v15_full は V15 の 145 feature セットで FT+IR を .pkl に保存した再学習版。  
v15.2 の 17 新 feature (breeder_dist_1 等) は **含まない**。

---

## 2. WF AUC fold-wise (6-fold, 2020-2025)

| Year | LGB    | XGB    | FT     | IR     | LGB+XGB | Grid   | Grid > V15 Grid (0.8858) |
|------|--------|--------|--------|--------|---------|--------|--------------------------|
| 2020 | 0.8578 | 0.8588 | 0.8433 | 0.7814 | 0.8592  | 0.8593 | NO                       |
| 2021 | 0.8643 | 0.8669 | 0.8640 | 0.8747 | 0.8665  | 0.8834 | NO                       |
| 2022 | 0.8671 | 0.8687 | 0.8660 | 0.8751 | 0.8688  | 0.8836 | NO                       |
| 2023 | 0.8685 | 0.8699 | 0.8681 | 0.8768 | 0.8700  | 0.8866 | YES                      |
| 2024 | 0.8709 | 0.8720 | 0.8708 | 0.8780 | 0.8722  | 0.8866 | YES                      |
| 2025 | 0.8688 | 0.8706 | 0.8692 | 0.8790 | 0.8704  | 0.8875 | YES                      |
| **6-fold mean** | 0.8662 | 0.8678 | 0.8652 | 0.8608 | 0.8679 | **0.8812** | — |
| **5-fold (2021-25)** | 0.8679 | 0.8696 | 0.8676 | 0.8769 | 0.8696 | **0.8855** | — |

注: 2020 fold で IR AUC = 0.7814 (極端に低い)。2020 年は JVLINK データが薄く、IntraRace Attention の coverage が不足。Grid 重み [0.4, 0.4, 0.1, 0.1] でほぼ無視されているが、6-fold 平均を下げる。

---

## 3. fair benchmark verdict

### V15 baselines (V15-audit-2、2026-05-17 確定)

| baseline | AUC | 出典 |
|----------|-----|------|
| V15 genuine WF LGB+XGB 6-fold | 0.8678 | V15-audit-2 |
| V15 Grid 5-fold mean (2021-2025) | 0.8858 | V15-audit-2 / v15_master_report.json |

### v15_full delta

| comparison | v15_full | V15 baseline | delta |
|------------|---------|--------------|-------|
| LGB+XGB 6-fold (production 真値 vs) | 0.8679 | 0.8678 | **+0.0001** (実質ゼロ) |
| Grid 6-fold vs V15 Grid 5-fold | 0.8812 | 0.8858 | **-0.0046** (BELOW) |
| Grid 5-fold (2021-2025) vs V15 Grid | 0.8855 | 0.8858 | **-0.0003** (ほぼ同等) |

### fold-wise 上回り

| 基準 | v15_full Grid が上回る fold 数 |
|------|-------------------------------|
| > 0.8678 (LGB+XGB baseline) | 5/6 ← **B-4 GO 判定の根拠 (誤った比較)** |
| > 0.8858 (Grid vs Grid) | 3/6 (2023-2025 のみ) |

---

## 4. VERDICT LOGIC FLAW (B-4 GO の問題点)

B-4 の GO 判定は以下の criteria で自動判定:

| criteria | 判定 | 実際の根拠 |
|----------|------|-----------|
| crit1: grid_mean >= 0.870 | TRUE | 0.881 >= 0.870 (trivially true、閾値が低すぎる) |
| crit2: over_baseline_count >= 4 | TRUE | Grid > **0.8678** (LGB+XGB baseline!) で 5/6 |
| crit3: T4 gate pass | TRUE | FT+IR が保存済み |
| crit4: paper_ready | TRUE | cosmetic |
| crit5: spread < 0.05 | TRUE | 0.028 |

**問題**: crit2 が 4-model Grid AUC を 2-model LGB+XGB (0.8678) と比較。  
4-model ensemble が 2-model を上回るのは **当然** であり、improvement の証拠にならない。  

fair comparison は **Grid vs Grid (0.8858)** で行うべき:

| fair criteria | 判定 |
|---------------|------|
| fair_crit2: Grid > V15 Grid (0.8858) folds >= 4 | **FALSE** (3/6) |
| fair_grid_mean > V15 Grid | **FALSE** (-0.0003 at 5-fold, -0.0046 at 6-fold) |

---

## 5. LEAK audit

| 対象 | 結果 | 出典 |
|------|------|------|
| v15_full 145 features 全件 | OK (0 leaks) | data/v15_2/v15_leak_audit_2026_05_17.json |
| post_hoc_audit 17 v15.2 new features | OK (0 leaks) | data/v15_2/post_hoc_audit_20260517_1712.json |
| paci_info_idx | **NOT PRESENT in v15_full** | feature list 確認 |
| RED_IMP_BUT_CONST | 0 件 | T1_features_audit_2026_05_18.json |
| oz_ (morning odds) timing | morning_06 (pre-race OK) | v15_leak_audit |
| paci_ninki_idx timing | morning_06 (pre-race OK) | v15_leak_audit |

注 1: paci_jockey_exp_wr / paci_jockey_exp_3rd / paci_ninki_idx は corr_target 0.44-0.46 と高い。  
LGB gain も 266K-314K と top tier。ただし leak audit で timing=morning_06 (前日夜更新+当日朝 06:00 sync) = pre-race 情報として OK 判定済み。  
注 2: background で言及の「paci.info_idx +0.4139 leak 疑い」は v15_full の feature list に存在しない。この懸念は v15_full には適用されない。

**LEAK GATE: PASS**

---

## 6. paper ROI 補足

paper shadow データ未存在 (5/18 時点)。v15_full は 5/22+ paper shadow 予定。  
cumulative_results.csv は V15 production 実績 (n=663、5/18 時点)。  
v15_full と V15 production は同一 feature を使用するため top1/top3 予測差は FT+IR の有無のみ。  
paper ROI delta は 5/22+ 週末実施後に評価可能。

---

## ★ 最終 verdict ★

**限定 GO (条件付き)**

| 評価軸 | 結果 |
|--------|------|
| LEAK | PASS (0 leaks) |
| Production baseline delta (LGB+XGB) | +0.0001 (実質ゼロ) |
| Grid vs Grid delta (5-fold) | -0.0003 (parity) |
| Grid vs Grid delta (6-fold) | -0.0046 (BELOW V15) |
| False positive (B-4 GO は誤った比較?) | YES、crit2 が 4-model vs 2-model 比較 |
| v15_full の本来の価値 | FT+IR を .pkl 保存 = production 投入可能な full ensemble を初めて実現 |

### 5/22 admin fire させる価値: **条件付きあり**

- paper shadow (週末のみ、投資ゼロ) として fire は OK
- V15 production を即刻置き換える根拠は **なし** (Grid delta -0.0003 at 5-fold では有意改善なし)
- production gate は Grid 5-fold mean >= 0.888 を維持すること
- 2-3 週間の paper shadow で real winner_top1 rate / ROI delta を確認してから go/no-go 最終判定

### 誤判定の原因

B-4 の GO verdict は **4-model Grid AUC > 2-model LGB+XGB (0.8678)** を「improvement」と誤分類。  
正しい比較: **4-model Grid AUC > 4-model Grid baseline (0.8858)**  
この基準では v15_full は parity (-0.0003) であり、真の improvement とは言えない。  
ただし FT+IR saved という architecture の改善は別軸での価値があり、paper shadow は有意義。
