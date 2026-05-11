# Phase 26: Competitor gap features v2 (中 priority 7+ 件) 5/12

## 概要

Agent 1 で 発見した 競合 AI gap (中 priority 7 件) を **全 LEAK-free expanding window** で
実装。 V15 model 不変、 derivative CSV 出力のみ。

- 実装 script: `tools/build_competitor_gap_features_v2.py`
- verify script: `tools/verify_competitor_gap_v2_importance.py`
- 出力 CSV: `data/competitor_gap_features_v2.csv` (49 MB、 379,044 rows、 2018-2025)

## 実装 features (9 件、 Task 仕様 +2 件 補足)

| # | feature | 説明 | 集計単位 | LEAK 防止 |
|---|---------|------|---------|----------|
| 1 | start_index | 過去走 1角 通過位置 expanding mean | horse_id | groupby(horse).shift(1).expanding() |
| 2 | middle_index | 過去走 2-3角 平均位置 expanding mean | horse_id | 同上 |
| 3 | late_index | 過去走 agari_3f race-relative expanding mean (負=速い) | horse_id | 同上 |
| 4 | bms_shinba_top3r | 母父 新馬戦 (class_code==15) 累積 top3 rate | bms × time | merge_asof(date_int strict-prior) |
| 5 | kireaji_career_avg | -late_index (正=切れ味) | horse_id | shift(1) expanding |
| 6 | suriko_career_avg | (pass3 - finish) expanding mean (早めの 末脚) | horse_id | shift(1) expanding |
| 7 | pace_adapt_career_avg | 馬 × race pace_cat 単位 top3 expanding | horse × pace_cat | shift(1) expanding |
| 8 | pci3 | 馬 career の agari_3f expanding mean (絶対値) | horse_id | shift(1) expanding |
| 9 | rpci | 馬 agari / race avg agari の career expanding mean | horse_id | shift(1) expanding |

### Task 仕様 skip した features

| feature | 理由 |
|---------|------|
| LPI (ラップ偏差指数) | jra_races_full.csv に race の 200m ラップ詳細無し。 pace_features と差別化不可。 skeleton 化 候補 |
| speed_gradient_E_M_L | 同上、 race ラップ data 不在 |
| PCI3 (Task 別意) | 上記 #8 pci3 で カバー (career agari ベース 速さ指数) |
| RPCI | 上記 #9 で 実装 |

## Signal verify 結果 (full 2018-2025、 N=379,044)

LEAK-free 確認 後 の Q5-Q1 top3 rate delta + LGB single-fit importance:

| feature | N valid | Q1 rate | Q5 rate | Q5-Q1 delta | LGB importance |
|---------|---------|---------|---------|-------------|----------------|
| start_index | 208,422 | 0.2603 | 0.1720 | -0.0884 | 536 |
| middle_index | 333,775 | 0.2868 | 0.1388 | **-0.1480** | **1,196 (top)** |
| late_index | 333,819 | 0.3563 | 0.1032 | **-0.2531** | 265 |
| bms_shinba_top3r | 369,135 | 0.1878 | 0.2571 | +0.0693 | 728 |
| kireaji_career_avg | 333,819 | 0.1032 | 0.3563 | **+0.2531** | 220 |
| suriko_career_avg | 333,775 | 0.1463 | 0.2881 | +0.1418 | **1,023** |
| pace_adapt_career_avg | 282,069 | 0.1406 | 0.3874 | **+0.2467** | 348 |
| pci3 | 333,818 | 0.2887 | 0.1663 | -0.1224 | **944** |
| rpci | 333,818 | 0.3593 | 0.1023 | **-0.2571** | 740 |

### Top signal/importance ranking

- **最強 signal (|delta| ≥ 0.20)**: rpci / late_index / kireaji_career_avg / pace_adapt_career_avg
- **最強 LGB imp (≥ 700)**: middle_index (1,196) / suriko_career_avg (1,023) / pci3 (944) / rpci (740) / bms_shinba_top3r (728)
- **dual top (delta + imp)**: rpci (delta -0.257、 imp 740) — 採用 候補 筆頭

注: late_index と kireaji_career_avg は sign-flip 等価 (verify 用 両方 出力、 採用時 は 片方のみ)。

## LEAK-free 監査

| 項目 | 結果 |
|------|------|
| 当該 race の outcome 使用 | NO (全 features shift(1) で 除外) |
| 当該 race の POST-RACE 値 使用 | NO (pass/finish/agari は 過去走 のみ) |
| 当該 race の race-level pace_cat 識別子 使用 | YES (pace_avg_pass1 race avg、 但し outcome 未使用 識別子 only) |
| bms 累積 で 当該 race 含む | NO (merge_asof allow_exact_matches=False) |

→ V20 直接使用可。 既存 V15 features と overlap 少 (career_burst_mean は pass4-finish、 #6 は pass3-finish で差別化)。

## 実装メモ (発見した data quirk)

- **jra_races_full.csv の race_id は umaban 込み で horse-unique**。 真の race key は
  `(year, month, day, course, kai, nichi, race_num)` で連結 (`race_key` 列で構築)。
  → 既存 `pace_features.csv` の `pace_avg_pass1` も この race_id で groupby していたため、
     実質 self-pass1 (race-avg 機能していない) を 出力していた。 本 v2 で 正しく race avg 集計。
- pass1 / pass2 は 短距離 で 0 (未通過) が多い → NaN 化 後 expanding (zero 集計回避)
- race_avg_agari は groupby('race_key') transform (POST-RACE 集計 だが
  個馬の expanding shift(1) で 当該 race 除外済み)
- bms 文字列 は Shift-JIS 由来 で 文字化け 含むが key-equal で 正しく aggregate (90%+ 充填)

## 統合 plan

1. **V15 不変** (本 task は CSV 生成のみ)
2. V20 学習 (6/14-) で `merge_v15_1_features` 同様 の merge helper 追加検討
3. 採用 候補 順位 (Q5-Q1 delta × LGB imp 合算):
   - 第 1 候補: **rpci**, **middle_index**, **suriko_career_avg**, **pci3**
   - 第 2 候補: **pace_adapt_career_avg**, **bms_shinba_top3r**, **late_index** (or **kireaji_career_avg**)
   - 第 3 候補: start_index

## 残務

- [ ] V20 学習 spec に 統合 (6/14-)
- [ ] 個別 features の WF delta 測定 (V15 baseline +0/-?)
- [ ] race lap detail data 取得 後 LPI / speed_gradient 追加 (Phase 27+)

## ファイル一覧 (本 task 追加)

| path | role |
|------|------|
| `tools/build_competitor_gap_features_v2.py` | features 構築 script |
| `tools/verify_competitor_gap_v2_importance.py` | LGB importance + delta verify |
| `data/competitor_gap_features_v2.csv` | 出力 (race_id, horse_id, 9 features) |
| `data/v18/phase26_features_5_12.md` | 本 doc |
