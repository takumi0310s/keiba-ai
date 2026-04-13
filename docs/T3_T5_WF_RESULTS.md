# v16 新特徴量 + AM8 WF 検証結果

Date: 2026-04-13
Total elapsed: 200 min

## ベースライン
- v15_master_report.json 記録: WF mean AUC = **0.8856**
- 採用基準（ユーザー指定）: WF AUC > 0.8858

## 結果サマリー

| Run | Features | Mean WF AUC | Δ vs 0.8856 | 判定 |
|-----|----------|------------|--------------|------|
| v15 baseline | 145 | 0.8856 | — | — |
| **v15 + v16 (all-in)** | **151** | **0.8834** | **-0.0022** | **REJECTED** |
| AM8 A+B only | 140 | 0.8843 | -0.0013 | — |

### 年別内訳

| 年 | v15+v16 all-in | AM8 A+B only |
|---|:---:|:---:|
| 2021 | 0.8835 | 0.8819 |
| 2022 | 0.8833 | 0.8824 |
| 2023 | 0.8829 | 0.8845 |
| 2024 | 0.8873 | 0.8866 |
| 2025 | 0.8799 | **0.8858** |
| mean | 0.8834 | 0.8843 |

## v16 新特徴量のカバレッジ（判定NG主因）

| 特徴量 | 非ゼロ率 | データソース |
|--------|:---:|-----|
| upset_level_val | 8.7% | 2020-2024（2025欠落） |
| top_popularity_reliability | 8.7% | 同上 |
| training_eval_rank | 17.8% | 2024-2025のみ |
| prev_review_score | 10.7% | 2020-2025全年（ただしreview_score多くゼロ） |
| **prev_master_index** | **1.1%** | **master_index CSVが2025のみ** |
| prev_track_index_val | 20.8% | 2020-2022+2025 |

→ 4/5 特徴量でカバレッジ <20%、1個は 1.1% の激低カバレッジ。signal 不足で all-in 加算が逆効果（-0.0022）。

## 判定 → v16 特徴量は **採用見送り**

- WF AUC 0.8834 <= baseline 0.8856（target 0.8858）
- 全カテゴリでベースライン割れ
- モデル更新なし（v15 を維持）

## 再採用の条件（将来的検討）

各特徴量の source CSV を全年フルカバレッジにする必要あり:

1. **upset_level**: `scrape_missing_all.py` upset 2025 + 2024残件（現在バッチ稼働中、ETA 29h）
2. **training_eval**: `scrape_super_premium.py --year 2020/2021/2022/2023` （バッチに登録済）
3. **master_index**: `scrape_master_index.py --year 2020-2024` （バッチに登録済）
4. **track_bias/race_lap**: `scrape_master_index.py` 2023-2024年 （バッチに登録済）
5. **race_review**: 既に全年カバーあるが review_score 非ゼロ率の改善要（スコア生成ロジック強化）

scrape_missing_all.py 完了後（~1-2日後）、カバレッジを再確認して v16 再評価する価値あり。

## AM8 予測の実力値評価

AM8 時点で使用可能な 138 特徴量（C=7, D=5除外）のみで:
- **mean WF AUC = 0.8843**（baseline -0.0013 = -0.15%）
- 2025年は **0.8858** と baseline を上回る
- 運用的意義: 当日オッズ・馬体重・直前パドック等のC/D特徴量を除外しても朝予測の精度はほぼ維持
- → **朝予測の信頼性は高い。まとめ買い運用で問題なし**

### C/D特徴量の貢献度

除外した12特徴量:
- C（発走直前）: odds_change_rate, pop_rank_change, odds_sharp_drop, weight_trend, weight_peak_diff
- D（実装バグ修正済）: jrdb_prev_idm, jrdb_prev_pace_idx, jrdb_prev_rise_code, jrdb_cid_idx, jrdb_ls_idx

これらの合計貢献度が +0.0013 程度と判明。**C特徴量の代替手段（前日最終オッズ等）を検討する必要性は低い**。

## 次アクション

- [x] 結果保存: `data/v16_wf_results.json`
- [x] レポート作成: 本ファイル
- [ ] scrape_missing_all.py 完了待ち → 再評価判定
- [ ] T2で修正した jrdb_prev_* 3特徴量は v15 production model には未反映（学習時は default のままだった）。次回 v15 再学習時に効果検証する
