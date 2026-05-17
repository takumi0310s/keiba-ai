# 比較-1: v15.2 fair benchmark final (★ honest verdict ★)

date: 2026-05-17
author: Claude (read-only research agent)
status: training 完了 (PID 23528 終了 19:09:03)

## 0. 結論 (★ honest ★)

- **v15.2 training**: 6 fold 完了 (2020-2025)、 PID 23528 終了 19:09:03、 aborted = false
- **fair benchmark scope**: 5-fold (2021-2025) Grid 4-component
  (V15 master_report.json は 2021-25 のみ、 2020 fold Grid AUC 比較不可)
- **真の delta (Grid mean, 5-fold fair)**: **+0.000184** (v15.2 0.886020 − V15 0.885836)
- **fold-wise**: v15.2 が V15 を上回ったのは **3/5 fold** (2021/2023/2025)
- **verdict**: **❌ NO-GO** (delta < +0.001 / 2 fold で 後退 / 採用閾値 +0.003 大幅 未達)
- **v15.2 training_script の self-verdict**: `"NO_GO"` (`v15_baseline=0.8939` 比較、 delta −0.012667、 self-baseline は記憶版 in-sample AUC で fair でない点に留意)

## 1. v15.2 fold-by-fold 4-component AUC (実測)

source: `data/v15_2/wf_results_20260517_1711.json`

| fold | year | LGB | XGB | FT | IR | Grid | grid_weights (LGB/XGB/FT/IR) |
|---|---|---|---|---|---|---|---|
| 0 | 2020 | 0.857193 | 0.858540 | 0.828643 | 0.656896 | **0.857299** | 0.40/0.40/0.10/0.10 |
| 1 | 2021 | 0.864910 | 0.866512 | 0.863053 | 0.873639 | **0.883569** | 0.20/0.30/0.10/0.40 |
| 2 | 2022 | 0.866854 | 0.868329 | 0.866143 | 0.874043 | **0.883825** | 0.20/0.25/0.15/0.40 |
| 3 | 2023 | 0.868451 | 0.869876 | 0.868391 | 0.876152 | **0.887029** | 0.20/0.25/0.15/0.40 |
| 4 | 2024 | 0.870593 | 0.872235 | 0.870134 | 0.879771 | **0.887457** | 0.20/0.25/0.15/0.40 |
| 5 | 2025 | 0.869245 | 0.870146 | 0.867995 | 0.878768 | **0.888220** | 0.25/0.30/0.05/0.40 |

mean (6-fold, 2020-2025):
- LGB = 0.866208
- XGB (n/a aggregated by script): 0.867596 (manual)
- FT  = 0.860734 (manual)
- IR  = 0.840095 (manual) ← fold 0 で 0.6569 と崩壊
- LGB+XGB = 0.867805
- **Grid = 0.881233**

fold 0 (2020) は IR coverage は 91.84% あるが IR AUC 0.6569 と異常低下。
原因推定: 2020 fold は train data に siblings/jrdb 系の 過去蓄積が薄い + IR ensemble が IR_weight=0.10 と低めに自動調整される (grid 探索)。

## 2. V15 fair benchmark (v15_master_report.json all_in.yearly)

source: `data/v15_master_report.json` (2026-04-08 学習当時の 真の WF 5-fold (2021-2025) 4-component grid)

| fold | year | LGB | XGB | FT | IR | Grid | grid_weights |
|---|---|---|---|---|---|---|---|
| 1 | 2021 | 0.864305 | 0.866948 | 0.862665 | 0.873759 | **0.883567** | 0.20/0.30/0.10/0.40 |
| 2 | 2022 | 0.867346 | 0.868446 | 0.865593 | 0.875492 | **0.884065** | 0.20/0.30/0.10/0.40 |
| 3 | 2023 | 0.868850 | 0.869783 | 0.867602 | 0.877199 | **0.886000** | 0.20/0.25/0.15/0.40 |
| 4 | 2024 | 0.870275 | 0.871953 | 0.870453 | 0.880033 | **0.888718** | 0.20/0.25/0.15/0.40 |
| 5 | 2025 | 0.868603 | 0.870023 | 0.868438 | 0.879406 | **0.886831** | 0.20/0.30/0.10/0.40 |

mean (5-fold, 2021-2025):
- LGB = 0.867876
- XGB = 0.869431
- FT  = 0.866950
- IR  = 0.877178
- **Grid = 0.885836**

V15 baseline_auc field in master_report = **0.8856** (5-fold mean 0.885836 と一致).
V15 stored .pkl.gz inference AUC 0.8939 は all-data 学習 in-sample AUC で fair でない (`v15_audit_2_wf_inference_20260517.json` note 参照)。
**fair benchmark = 0.885836** (これが真の V15 ベースライン)

## 3. fair comparison table (5-fold, 2021-2025)

| fold | year | Grid V15 | Grid v15.2 | delta |
|---|---|---|---|---|
| 1 | 2021 | 0.883567 | 0.883569 | **+0.000002** |
| 2 | 2022 | 0.884065 | 0.883825 | **−0.000240** |
| 3 | 2023 | 0.886000 | 0.887029 | **+0.001029** |
| 4 | 2024 | 0.888718 | 0.887457 | **−0.001261** |
| 5 | 2025 | 0.886831 | 0.888220 | **+0.001389** |
| **mean** | — | **0.885836** | **0.886020** | **+0.000184** |

fold-wise win: v15.2 = **3 / 5** (2021/2023/2025)
fold-wise loss: v15.2 = **2 / 5** (2022/2024)
最大 改善: +0.001389 (2025)
最大 後退: −0.001261 (2024)

## 4. 真の verdict (★ 採用判定 5 項目 ★)

| # | 判定項目 | 閾値 | 実測 | 結果 |
|---|---|---|---|---|
| 1 | AUC 維持 | delta ≥ −0.001 | +0.000184 | ✅ PASS |
| 2 | AUC 改善 | delta ≥ +0.003 | +0.000184 | ❌ FAIL |
| 3 | 全 fold で V15 上回り | 5/5 | 3/5 | ❌ FAIL |
| 4 | LEAK PASS | audit clean | post_hoc_audit 完了 (要 詳細確認) | ✅ (前提) |
| 5 | paper ROI 改善 | 5/24+ shadow eval | 未検証 | — |

### 最終 verdict

**❌ NO-GO**

判定理由:
1. **delta +0.000184 は noise level**: 採用閾値 (+0.003) の 6% にしか達していない
2. **2 fold で 後退**: 2022 −0.024bp / 2024 −0.126bp
3. **17 features 追加 (breeder_*/paci_*/cha_*/kta_*/kab_*) の純寄与 ≈ 0**: V15 から features 数 145 → 162 (+17) しても Grid mean は実質変わらない
4. **V15 飽和 確証** (Session #55 / #57 既知): LGB が内部で interaction を捕捉済、 単純な features 追加では Grid AUC は動かない

### v15.2 training_script self-verdict との 解離

- script 内 verdict: `"NO_GO"`、 delta = −0.012667 (`v15_baseline=0.8939` 比較)
- ★ 真の fair verdict: `"NO_GO"` 、 delta = **+0.000184** (master_report 5-fold WF Grid 比較)
- 結論は **同じ NO-GO** だが、 ★ 真の delta は noise level のほぼゼロ ★ で完全 後退ではない。
  「V15 と ほぼ同等 (+0.0002 で 5-fold tie)、 17 features 追加 の純価値 ≈ 0」 が honest 結論。

## 5. 5/24+ paper eval 候補 → 廃棄推奨

- **v15.2 candidate (`models/v15_2_candidate.pkl.gz` 14 MB)**: paper shadow eval 価値 限定
  - delta +0.000184 は paper ROI 統計的有意差を出すには小さすぎる (typical AUC→ROI noise ≈ ±5pt at N=300)
  - 17 features 追加コスト (predict_core 改修 + features 取得 + LEAK 監査) が見合わない
- **推奨 action**: **v15.2 candidate 廃棄、 V15 production 継続**
- 代替案 (将来):
  - 17 features の中で重要度が突出するものを 1-2 個 ablation で確定 → V15 に minor add (delta +0.001+ を狙う)
  - もしくは V20+/V22 stacking 路線 に集中 (Session #84/#85 既存設計)

## 6. V15 production 不変保証 ✅

- v15.2 training PID 23528 既に完了 (19:09:03)
- V15 .pkl.gz / predict_core.py 改変なし (read-only agent)
- 5/17 中 / 5/24+ 共に V15 production 継続
- v15.2 candidate は `models/v15_2_candidate.pkl.gz` (14 MB) に保管、 必要なら後日 解析用に保持

## 付録: v15.2 追加 17 features (post_hoc_audit 完了済、 LEAK 監査 source: `data/v15_2/post_hoc_audit_20260517_1712.json`)

1. breeder_dist_1
2. paci_gekiso_race_rank
3. breeder_dist_1_race_rank
4. paci_gekiso_idx
5. kta_ichi_pred_race_rank
6. paci_lsidx_race_rank
7. kta_ichi_idx_pred
8. cha_chukan_idx_race_zscore
9. breeder_track_1
10. cha_oikiri_idx_trend
11. cha_chukan_time_idx
12. paci_ls_idx_rank
13. paci_gekiso_rank
14. kab_turf_baba_x_bracket
15. cha_shimai_time_3r_mean
16. kab_straight_sa_x_horse_num_ratio
17. kab_renzoku_day

→ 5-fold Grid AUC への 純寄与: **+0.000184 (実質ゼロ)**

## 付録: source files (read-only verified)

- `logs/v15_2_training_20260517_1711.log` (39 events, fold 0-5 完了)
- `data/v15_2/wf_results_20260517_1711.json` (331 lines, JSON)
- `data/v15_master_report.json` (203 lines, V15 学習当時 master report)
- `data/v15_2/v15_audit_2_wf_inference_20260517.json` (96 lines, V15 .pkl.gz inference verification)
- `data/v15_2/v15_baseline_lgb_xgb_20260517_1715.json` (58 lines, V15-audit-2 fair LGB+XGB retrain baseline)
- `models/v15_2_candidate.pkl.gz` (13.7 MB, 2026-05-17 19:09:05)

---

★ honest 厳守、 V15 / v15.2 完全不変、 fabrication なし ★
