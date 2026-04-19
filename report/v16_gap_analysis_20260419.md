# v16 再学習 ギャップ分析

- 作成: 2026-04-19 (Sun) 22:30
- 現行: v15 (150特徴量, WF AUC 0.8858, LGB+XGB+FT+IntraRace)
- v16 学習判定: **❌ 不可 (データ不足)**
- 詳細データ: `report/v16_coverage_20260419.md` 参照

---

## 1. カバレッジ サマリー (2020-2025, horse-level)

| ソース | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 判定 |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| training_eval | 133% | 100% | 100% | 100% | 100% | 100% | ✅ 完備 |
| **master_index** | **0%** | **0%** | **0%** | 6% | 192% | 96% | ❌ 2020-2022 欠落 |
| upset (race-level) | 14% | 4% | 8% | 11% | 18% | 22% | ⚠ 低カバ |
| training_times | 100% | 130% | 100% | 100% | 100% | 100% | ✅ 完備 |
| speed_index | 91% | 92% | 92% | 92% | 92% | 92% | ✅ 均一 |
| stable_comments | 69% | 35% | 38% | 35% | 59% | 34% | ⚠ 中カバ |
| race_review | 100% | 100% | 99% | 94% | 89% | 100% | ✅ 完備 |
| shinba_eval | 0% | 0% | 0% | 0% | 8% | 8% | ❌ 2024-2025 片手のみ |
| JRDB KYI | 96% | 97% | 96% | 97% | 98% | 96% | ✅ 完備 |
| **JRDB SED** | **0%** | **0%** | **0%** | **0%** | **100%** | **98%** | ❌ 2020-2023 欠落 |
| JRDB TYB/CYB/JOA | 0% | 0% | 0% | 0% | 0% | 0% | ❌ 直近週分のみ |

---

## 2. Trigger 判定

| 条件 | 閾値 | 最低カバ | 判定 |
|:---|:---:|:---:|:---:|
| training_eval >= 40% | 40% | **100.0%** | ✅ OK |
| master_index >= 30% | 30% | **0.0%** (2020-2022) | ❌ NG |

→ **v16 学習不可**。ただし master_index を 2024-2025 限定で使う設計ならば一部可能。

---

## 3. 不足データ リストアップ (優先順位付き)

### 🔴 優先度 High (WF 2020-2025 学習に必須)

1. **master_index 2020-2023** (0-6% のみ)
   - `netkeiba_master_index.csv` で 4桁年 prefix から 2020-2022 が 0%
   - 2023 も 6% で不完全
   - 特徴量: master_index / time_index / start_index / chase_index / agari_index
   - 取得元: netkeiba `/race/数字/index.html` 系
   - スクレイパー: `tools/scrape_master_index.py`
   - 推定所要時間: 2020-2023 で ~180,000 レース × 2秒 = 100 時間
   - → 週末外 (Mon 06:00 以降) に走らせ続ける必要あり

2. **JRDB SED 2020-2023** (0%)
   - 過去成績データ (前走着順・上がりなど原資料)
   - JRDBサイトから年度ダウンロード要
   - → `tools/scrape_jrdb.py --type SED --date <past>` で手動バックフィル

### 🟡 優先度 Medium (モデル性能向上に寄与するが必須ではない)

3. **shinba_eval 2020-2023** (0%)
   - 新馬戦評価、サンプル少 (7,998 行)
   - 2024-2025 でも 8% のみ → モデル組込には追加取得必要

4. **upset 全年** (1-22%)
   - 波乱度 (race_id → high/low / reliability)
   - 既に bulk_scrape_upset 完了済だが race-level カバが低い
   - 未取得レース ~85% 存在
   - → 再スクレイピング or 現状ママ (既存特徴量で代替可)

5. **stable_comments 2021-2025** (34-59%)
   - 厩舎コメント (文字列 → スコア化 -3〜+3)
   - 低カバで v12.1 で不採用 → 現状ママで良い

### 🟢 優先度 Low (現状ママで OK)

6. speed_index (91% 均一で良好)
7. race_review (89-100% で十分)
8. training_eval / training_times (100%+ で完備)
9. JRDB KYI (96-98% で優秀)

---

## 4. v16 学習の前提条件 (目標)

v16 を training するために必要な状態:

- [ ] master_index 2020-2022: ≥ 80% カバ
- [ ] JRDB SED 2020-2023: ≥ 80% カバ
- [x] training_eval, training_times, race_review, JRDB KYI: 既存レベル維持
- [ ] 新特徴量の検討 (v15 の 150 → v16 で +10-20 程度を目標)
  - 例: JRDB KYI の IDM変化、騎手指数の時系列、馬番×コース×距離の交互作用

---

## 5. 来週以降のアクションプラン

### Monday 4/20 朝以降
1. **06:00 ガード解除後**: scrape_missing_all を再開して master_index の欠落年を埋める
   ```bash
   python tools/scrape_missing_all.py --years 2020,2021,2022,2023
   ```
2. バックグラウンドで走らせ、日次で進捗確認 (tools/coverage_report.py)

### 平日
3. JRDB SED 過去年バックフィル (手動)
4. カバ率確認して週末運用前に Trigger 閾値突破を狙う

### 来週末 (4/25-26) の本番
5. **v15 で継続運用** (v16 は学習不可のため)
6. 本番は SCRAPER-GUARD 修正済で通常通り発火

### 週末明け (4/27-28)
7. カバ率最終確認
8. Trigger 突破したら v16 学習開始 (LGB+XGB+FT+IntraRace の 4-model)
9. WF AUC > 0.8858 and gap < 0.05 で採用判定

---

## 6. バックグラウンドスクレイピング状態 (2026-04-19 時点)

| ログ | 最終更新 | 状態 |
|------|----------|------|
| bulk_scrape_upset.log | 4/14 23:24 | "All races already completed" (20,733 完了, 8 失敗) — 停止 |
| scrape_master_all_years.log | 4/15 12:55 | "SCRAPING COMPLETE" (2,874 races → 48,794 rows) — 停止 |
| scrape_missing_all.log | 4/16 08:29 | エラー (exit 3221226091) — 停止 |
| scrape_missing_all_restart2.log | 4/18 02:04 | ログ残るがプロセス不明 |

→ **現在バックグラウンドで走っているスクレイピングはない**。
   明日 Mon 06:00 以降に再開の必要がある。

---

## 7. v16 training スケルトン は作成しない

学習可否判定が NG のため、スケルトン (`train_v16.py`) は作成せず。
データ補充完了時点で改めて計画 (`v16_training_plan_YYYYMMDD.md`) を起こす。
