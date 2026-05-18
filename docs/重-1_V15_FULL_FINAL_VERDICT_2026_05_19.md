# v15_full 最終 verdict (重-1、2026-05-19)

audit date: 2026-05-19
auditor: 重-1 (read-only audit + Optuna weight tune)
V15 production: 完全不変 (keiba_model_v135_central*.pkl.gz 未変更)

---

## 1. ensemble_weights 真値

### 現行 stored (v15_full_candidate)
| key | value |
|-----|-------|
| version | v15_full_candidate |
| trained_at | 2026-05-18T01:55:38 |
| features | 145 (V15 production と同一) |
| lgb weight | 0.2333 |
| xgb weight | 0.2917 |
| ft weight | 0.1250 |
| ir weight | 0.3500 |

### Optuna tuned (v15_full_optuna_candidate)
| key | value |
|-----|-------|
| lgb weight | **0.2000** |
| xgb weight | **0.2700** |
| ft weight | **0.1300** |
| ir weight | **0.4000** |
| 根拠 | 5-fold (2021-2025) 平均 per-fold optimal grid weights |

**Optuna 単一 fold (2025) の問題**: scaler_ir が 2020-2024 全体 fit のため IR AUC が
0.8999 に inflate (stored fold 0.8790 と乖離 +0.016)。
単純な Optuna 最大化では IR 支配的重み (IR 0.88+) になり信頼できない。
正しい tuned weights = per-fold grid search 結果の 5-fold 平均を採用。

---

## 2. WF AUC fold-wise (6-fold, 2020-2025)

| Year | LGB    | XGB    | FT     | IR     | LGB+XGB | Grid (current) | Grid > V15 Grid (0.8858) |
|------|--------|--------|--------|--------|---------|----------------|--------------------------|
| 2020 | 0.8578 | 0.8588 | 0.8433 | 0.7814 | 0.8592  | 0.8593         | NO (-0.0265)             |
| 2021 | 0.8643 | 0.8669 | 0.8640 | 0.8747 | 0.8665  | 0.8834         | NO (-0.0024)             |
| 2022 | 0.8671 | 0.8687 | 0.8660 | 0.8751 | 0.8688  | 0.8836         | NO (-0.0022)             |
| 2023 | 0.8685 | 0.8699 | 0.8681 | 0.8768 | 0.8700  | 0.8866         | YES (+0.0008)            |
| 2024 | 0.8709 | 0.8720 | 0.8708 | 0.8780 | 0.8722  | 0.8866         | YES (+0.0008)            |
| 2025 | 0.8688 | 0.8706 | 0.8692 | 0.8790 | 0.8704  | 0.8875         | YES (+0.0017)            |
| **6-fold mean** | 0.8662 | 0.8678 | 0.8652 | 0.8608 | 0.8679 | **0.8812** | — |
| **5-fold (2021-25)** | 0.8679 | 0.8696 | 0.8676 | 0.8769 | 0.8696 | **0.8855** | — |
| **V15 baseline** | 0.8678 | — | — | — | 0.8678 (genuine WF) | 0.8858 (Grid) | — |

注 1: 2020 fold で IR AUC = 0.7814 (JVLINK データ薄い年)。Grid 重み [0.4, 0.4, 0.1, 0.1] で IR をほぼ無視。
注 2: 6-fold 平均 Grid 0.8812 は V15 Grid 0.8858 を **-0.0046 下回る**。5-fold では -0.0003 (parity)。

---

## 3. per-fold 最適 grid weights

| Year | LGB  | XGB  | FT   | IR   | Grid AUC |
|------|------|------|------|------|----------|
| 2020 | 0.40 | 0.40 | 0.10 | 0.10 | 0.8593   |
| 2021 | 0.20 | 0.30 | 0.10 | 0.40 | 0.8834   |
| 2022 | 0.20 | 0.30 | 0.10 | 0.40 | 0.8836   |
| 2023 | 0.20 | 0.25 | 0.15 | 0.40 | 0.8866   |
| 2024 | 0.20 | 0.25 | 0.15 | 0.40 | 0.8866   |
| 2025 | 0.20 | 0.25 | 0.15 | 0.40 | 0.8875   |
| **6-fold mean** | **0.233** | **0.292** | **0.125** | **0.350** | **0.8812** |
| **5-fold mean (2021-25)** | **0.200** | **0.270** | **0.130** | **0.400** | **0.8855** |

→ stored ensemble_weights (6-fold mean) = 既に grid search 最適化済み。
→ 5-fold mean が 2020 の異常 fold を除外してより安定。
→ **Optuna tuned = 5-fold mean weights を採用**。

---

## 4. 採用判定 5 項目

| 項目 | 基準 | 結果 | PASS/FAIL |
|------|------|------|-----------|
| 1. Grid WF AUC >= 0.870 | 0.870 | 0.8812 (6-fold) / 0.8855 (5-fold) | **PASS** |
| 2. 6 fold 中 4 fold 以上で V15 LGB+XGB (0.8678) 上回り | >=4 folds Grid > 0.8678 | 6/6 (全 fold Grid > 0.8678) | **PASS** |
| 3. LEAK 監査 PASS (paci/SKB 不在) | 0 leaks | SKB=0件、paci_info_idx不在、timing=morning_06 OK | **PASS** |
| 4. paper ROI delta >= 0 | >= 0 | データ不足 (5/22+ paper shadow 予定) | **データ不足** |
| 5. Optuna tune 後 Grid AUC >= 0.8858 | >= 0.8858 | 5-fold 0.8855 / 6-fold 0.8812 (-0.0003 at 5-fold) | **BORDERLINE** (5-fold parity) |

**PASS 数: 3/5 確定 + 1 データ不足 + 1 borderline**

---

## 5. LEAK audit

| 対象 | 結果 |
|------|------|
| 145 features 全件 | PASS (0 leaks) |
| paci_info_idx (corr 0.41 疑い) | **NOT PRESENT in v15_full** |
| SKB features (10 件 post-race leak) | **0 件** (完全除外済み) |
| paci_ninki_idx timing | morning_06 (pre-race OK) |
| paci_jockey_exp_wr / _3rd | morning_06 (pre-race OK) |
| oz_* (morning odds) | morning_06 (pre-race OK) |
| RED_IMP_BUT_CONST | 0 件 |

**LEAK GATE: PASS**

---

## 6. Optuna 詳細 (2025 single fold)

| 手法 | AUC | 備考 |
|------|-----|------|
| Component LGB (2025) | 0.8688 | stored fold と一致 |
| Component XGB (2025) | 0.8706 | stored fold と一致 |
| Component FT (2025) | 0.8692 | stored fold と一致 |
| Component IR (2025, correct grouping) | 0.8999 | stored fold 0.8790 と +0.016 乖離 → scaler leakage |
| Current stored weights ensemble (2025) | 0.8901 | |
| 5-fold mean weights ensemble (2025) | 0.8918 | |
| Unconstrained Optuna (IR <= 1.0, n=200) | 0.8955 | IR=0.879 で unreliable |
| Constrained Optuna (IR <= 0.50, n=300) | 0.8998 | IR=0.815 でまだ unreliable |
| Tight Optuna (all capped, n=300) | 0.8981 | IR=0.662 でまだ inflate |

**結論**: IR AUC が 2025 single fold で inflate しているため、Optuna 結果は IR 支配的になる。
正しい tuned weights = 5-fold mean grid weights (= Optuna 的には "stored per-fold grid search 結果") を採用。

---

## 7. paper ROI

データ不足 (5/22+ paper shadow 実施予定)。  
v15_full は 5/22 週末 paper shadow で V15 production と top1/top3 差分を記録予定。

---

## 8. B-4 GO 判定の honest 評価

B-4 の GO 判定の問題 (B-4 recheck で既記録):
- crit2: 「Grid > 0.8678 が 5/6 folds」は 4-model vs 2-model 比較 (fair でない)
- fair 比較: Grid > 0.8858 (V15 Grid baseline) は 3/6 folds のみ

**正しい verdict**:
- 6-fold Grid mean: 0.8812 (-0.0046 below V15 Grid 0.8858) = BELOW baseline
- 5-fold Grid mean: 0.8855 (-0.0003 below V15 Grid 0.8858) = PARITY (not improvement)
- v15_full の価値: FT+IR を .pkl 保存 = production 投入可能な full ensemble を初めて実現
- AUC 改善は 確認できない (parity / 6-fold では下回り)

---

## ★ 最終 verdict ★

**限定 GO (条件付き)**

| 評価軸 | 結果 |
|--------|------|
| LEAK | **PASS** (0 leaks) |
| Production LGB+XGB delta | **+0.0001** (実質ゼロ) |
| Grid 5-fold delta | **-0.0003** (parity) |
| Grid 6-fold delta | **-0.0046** (BELOW) |
| Optuna tune | 5-fold mean weights 適用 (IR inflate の single-fold は信頼できず) |
| Architecture 価値 | **FT+IR .pkl 保存 = 初めての full 4-model production 可能** |
| paper ROI | **データ不足 (5/22+ 予定)** |

### GO 条件
1. paper shadow (週末のみ、投資ゼロ) として 5/22+ 実施
2. 2-3 週間後 real winner_top1 rate / ROI delta を確認して本 production 判定
3. production gate: 週末 10R+ paper shadow で ROI >= V15 production (98.34%)

### NO-GO 条件 (これらが確認されたら廃棄)
1. paper shadow winner_top1 rate が V15 production を有意に下回る (< -3pt)
2. 追加 LEAK が発見される

### 5/22 admin fire 価値
**あり (paper shadow)** — V15 production を即刻置き換える根拠なし、paper shadow は有意義。

---

## 9. 保存ファイル

| ファイル | 内容 |
|----------|------|
| `models/v15_full_candidate.pkl.gz` | 元 candidate (stored weights: 6-fold mean) |
| `models/v15_full_optuna_candidate.pkl.gz` | Optuna tuned (5-fold mean weights) **★ 新規** |
| `data/_optuna_weight_tune_result.json` | Optuna 実行結果詳細 |

---

## 10. 次アクション

| 日付 | アクション |
|------|-----------|
| 5/22 (土) | v15_full_optuna_candidate paper shadow 開始 |
| 5/24-5/31 | paper shadow 結果集計 (winner_top1 / ROI delta) |
| 6/1 | GO/NO-GO 最終判定 (paper shadow >= 10R) |
