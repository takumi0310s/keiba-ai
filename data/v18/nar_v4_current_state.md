# NAR v4 現状分析 (2026-05-05 / Phase 2.5+)

最新 commit: e5f71cfa

---

## 1. モデル: data/nar/models/keiba_model_nar_v4.pkl

| 項目 | 値 |
|------|----|
| サイズ | 167 KB |
| AUC (LGB+XGB ensemble) | **0.8145** |
| LGB AUC | 0.8142 |
| XGB AUC | 0.8144 |
| ensemble weights | LGB 0.4999, XGB 0.5001 |
| features 数 | **22 (Pattern B、odds_log 含む)** |
| 学習データ | 4,821 races / 49,213 rows (NAR 2020-2024) |
| 騎手 stats | 315 人 |

### 1.1 全 22 features

| # | 名前 | 種別 | 備考 |
|---:|------|------|------|
| 1 | odds_log | numeric | **当日確定オッズ log** ← Pattern B (リーク特性) |
| 2 | num_horses | numeric | 出走頭数 |
| 3 | distance | numeric | レース距離 |
| 4 | surface_enc | categ | 芝=0/ダート=1 (NAR は基本 dirt) |
| 5 | condition_enc | categ | 馬場状態 良/稍/重/不良 |
| 6 | course_enc | categ | 競馬場コード (船橋=43 等) |
| 7 | horse_weight | numeric | 馬体重 |
| 8 | weight_carry | numeric | 斤量 |
| 9 | age | numeric | 馬齢 |
| 10 | sex_enc | categ | 性別 |
| 11 | horse_num | numeric | 馬番 |
| 12 | bracket | categ | 枠番 (NAR 8枠) |
| 13 | horse_num_ratio | numeric | horse_num / num_horses |
| 14 | bracket_pos | categ | 枠位置 (内/中/外) |
| 15 | carry_diff | numeric | 斤量差 (vs レース平均) |
| 16 | dist_cat | categ | 距離カテゴリ |
| 17 | weight_cat | categ | 体重カテゴリ |
| 18 | age_group | categ | 年齢グループ |
| 19 | jockey_wr | numeric | 騎手勝率 (NAR 集計) |
| 20 | jockey_place_rate | numeric | 騎手連対率 (NAR 集計) |
| 21 | pop_rank | numeric | **人気順位** ← Pattern B |
| 22 | is_nar | flag | =1 固定 (中央モデルとの共存用) |

### 1.2 LGB feature_importance (gain TOP10)

| feature | gain |
|---------|-----:|
| odds_log | **83,158** ← dominant |
| pop_rank | **36,276** |
| jockey_place_rate | 1,336 |
| num_horses | 928 |
| horse_weight | 522 |
| weight_carry | 267 |
| age_group | 251 |
| horse_num_ratio | 188 |
| sex_enc | 143 |
| distance | 132 |

→ **odds_log + pop_rank で大半を説明**。市場依存度高い、純粋 Pattern B モデル。

### 1.3 XGB feature_importance (gain TOP10)

f20 (pop_rank) 383, f0 (odds_log) 306, f19 (jockey_place_rate) 33 — 同様に市場依存。

---

## 2. データ: data/nar_all_races.csv

| 項目 | 値 |
|------|----|
| 行数 | **54,159** |
| 範囲 | 2024-01-01 〜 **2025-05-31** (1年 stale) |
| 2026年データ | **0 行** |
| 5/5 柏記念データ | 0 行 (csv 未取込み、ad-hoc CSV のみ) |

### 2.1 競馬場別カバレッジ (TOP 10)

| 場 | rows |
|----|-----:|
| 名古屋 | 6,413 |
| 浦和 | 5,454 |
| 高知 | 5,203 |
| 大井 | 5,164 |
| 川崎 | 5,093 |
| 船橋 | 5,009 |
| 金沢 | 3,638 |
| 笠松 | 3,566 |
| 札幌 | 3,119 |
| 佐賀 | 2,727 |

(15 場全て data あり、概ね均等)

### 2.2 進捗 data/nar_scrape_progress.json

| field | 値 |
|------|----|
| completed_dates | 184 dates (2024-01〜01-31, 2025-01-01〜05-31) |
| failed_dates | 0 |
| total_races | 5,246 |
| total_rows | 54,159 |
| **欠 data** | 2024-02 〜 2024-12 (10 か月)、2025-06 〜 現在 (11 か月) |

→ scrape job の中断 / カバレッジ穴 多数。月次 backfill 必要。

---

## 3. tools/scrape_nar_all.py 動作可否

| 項目 | 状態 |
|------|------|
| YEARS hardcode | `range(2024, 2014, -1)` ← 2026 含まず |
| date 単位指定 | なし (year/month のループ) |
| 進捗保存 | あり (resume 可) |
| 再開 | OK (completed_dates skip) |
| 当日取得 | 未対応 (--date 引数なし) |

**改修必要**: 当日 NAR レース予測には `--date YYYYMMDD` 引数追加が必要。

---

## 4. 中央 (V15) と NAR v4 features 比較

V15 features 150 / NAR v4 features 22.

| 重複 | 本数 | 備考 |
|------|----:|------|
| common (両方) | **14** | age, age_group, bracket, bracket_pos, carry_diff, course_enc, dist_cat, distance, horse_num, horse_num_ratio, is_nar, sex_enc, surface_enc, weight_carry |
| NAR-only | 8 | condition_enc, horse_weight, jockey_place_rate, jockey_wr, num_horses, odds_log, pop_rank, weight_cat |
| V15-only | 136 | bms/sire/JRDB/training/blood/expanding stats 等 V15 固有 (NAR データに存在しない) |

### 4.1 統合可能性

- **共通 14 features** で minimum コア構造あり
- **Pattern B 系 8 features** (odds_log, pop_rank, horse_weight, weight_cat, condition_enc, jockey系) は V15 にも対応列あり → 統合学習の最低 base 確立可能
- **V15-only 136** は JRDB 等 中央専用データ依存、NAR では使えない (NAR データに該当列なし)
- 統合モデル (Phase 3 v20) なら 22 features ベースで JRA+NAR を1つの model で扱う構造が現実的

---

## 5. ad-hoc tool: tools/predict_nar_kashiwa_5_5.py

5/5 柏記念用ハードコード:
- HORSES_CSV = `data/results/20260505_kashiwa_kinen_horses.csv` (固定)
- num_horses = 13 (固定)
- distance = 1600 (固定)
- course_enc = 43 船橋 (固定)
- condition_enc = 0 良 (固定)
- horse_weight = 480 mean fill (data 不在)
- JOCKEY_OVERRIDE: ルメール / 川田 / 横山武史 等 JRA elite を NAR jockey_stats に補完

→ 汎用化必須 (D タスク)。

---

## 6. 結論 (現状サマリ)

| 強み | 弱み |
|------|------|
| AUC 0.8145 (Pattern B) | データ 1年 stale (2025-05-31 まで) |
| 22 features 簡潔 | 月次穴 多数 (2024-02〜12, 2025-06〜) |
| 騎手 stats 315 人 | scrape は YEARS hardcode、当日不可 |
| 5/5 柏記念で動作実績 | predict は ad-hoc 固定 script のみ |
| V15 と 14 features 共有 | pipeline 自動化なし、Discord 通知未対応 |

→ B〜F で systematize が必要。
