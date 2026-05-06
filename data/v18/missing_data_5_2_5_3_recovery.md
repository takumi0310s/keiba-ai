# 5/2-5/3 投資レース feature 取得失敗レポート

生成: 2026-05-06
担当: research agent (Session #19)
ベース commit: bed809ec

## 0. 用語整理

- ユーザー方針「投資 26R」 = 戦略⑦ 適用後の `京都/06_特別/条件B/E` を除外した実効投資想定数。
- 実際の `cumulative_results.csv` にレコードが記録された 5/2-5/3 レースは **67R** (5/2: 33R, 5/3: 34R)。
- 戦略⑦ filter (B/E/京都/特別 除外) 適用後 = **35R** (5/2: 13R, 5/3: 22R)。
  → ユーザーの「26R」は更に細い filter (例 ROI 確認できる Live のみ) と推測。
- 本レポートは**全 67R の features 取得実態**を網羅的に検証。

source 確認 path:
- `data/daily_predictions/20260502.csv` (33 行)
- `data/daily_predictions/20260503.csv` (34 行)
- `data/cumulative_results.csv` rows where date ∈ {20260502, 20260503} (67 行)
- ROI = 31.7% (Investment 46,900 / Profit -32,040)

---

## 1. 投資 67R リスト (要約)

| date | 場 | 投資 R 数 | 内訳条件 |
|------|----|-----------|----------|
| 20260502 (土) | 新潟 | 9R | A=0, B=2, C=0, D=2, E=0, X=5 |
| 20260502 (土) | 東京 | 12R | A=3, B=2, C=0, D=4, E=1, X=2 |
| 20260502 (土) | 京都 | 12R | A=2, B=2, C=1, D=5, E=0, X=2 |
| 20260503 (日) | 新潟 | 10R | A=5, B=0, C=1, D=4, E=0, X=0 |
| 20260503 (日) | 東京 | 12R | A=2, B=0, C=7, D=3, E=0, X=0 |
| 20260503 (日) | 京都 | 12R | A=4, B=0, C=2, D=6, E=0, X=0 |
| **合計** | 3場 | **67R** | A=18, B=7, C=11, D=23, E=1, X=7 |

**結果**:
- 的中: 12R / 67R = 17.9%
- ROI: 31.7%
- 損益: **-32,040 円**
- BT 期待 (BT 4/11-4/26 平均的中率 22%) との乖離 = -4.1pt → distribution_shift_analysis.md 「RANK_SHIFT」と一致。

---

## 2. 取得カバレッジ概況

### 2.1 JRDB 系 (raw .txt 単位、`data/jrdb/extracted/`)

| 種別 | 5/2 records | 5/3 records | カバレッジ判定 |
|------|------------:|------------:|----------------|
| KYI (前日指数) | 492 | **15** ⚠️ | 5/3 は 06:00 時点 1R のみ → 後刻 fix |
| SED (前走成績) | 171 | **404 missing** ⚠️ | 既知 15.2% 問題 |
| SKB (専門家印) | **欠損** ⚠️ | **欠損** ⚠️ | 2015年〜停止状態 |
| TYB (LIVE 騎手調子) | 492 | **404 missing** ⚠️ | 5/3 は当日朝 publish 漏れ |
| UKC (馬基本) | 492 | 512 | OK |
| CHA (追切本) | 492 | 512 | OK |
| HJC (払戻) | 12 | 17 | 結果のみ (予測時不要) |
| BAC (馬場) | 492 | 512 | OK |
| KAB (馬場詳細) | 492 (cached at 5/2 12:31) | 3 ⚠️ | 5/3 は 3 行のみ |
| KKA (拡張馬) | 492 | 512 | OK |
| CYB (調教評価) | 492 | 512 | OK |

### 2.2 merged JRDB CSV (`data/jrdb_*.csv`、predict が読む側)

| CSV | 5/2 関連 行 | 5/3 関連 行 | 備考 |
|-----|------------:|------------:|------|
| jrdb_kyi.csv | 492 (3場合計) | 512 | 後刻 fix で full |
| jrdb_sed.csv | 171 | 0 | 5/3 SED は 404 → 反映なし |
| jrdb_skb.csv | **0** | **0** | 全期間 broken |
| jrdb_paci.csv | 1004 | (4/4 以降 broken) | **4/4 から更新停止** |
| jrdb_cyb.csv | 492 | 512 | OK |

### 2.3 netkeiba プレミアム CSV (`data/netkeiba_*.csv`)

ここが**最大の問題点**。premium_scrape ログは「Training: 36/36」「Speed Index: 36/36」と report しているが、CSV を直接 grep すると:

| CSV | 2026 年 行数 | 5/2-5/3 race rows | 備考 |
|-----|---:|---:|------|
| netkeiba_speed_index.csv | **16** | **0** ⚠️ | 5/2-5/3 全滅、CSV 追記処理 broken |
| netkeiba_training_eval.csv | **0** | **0** ⚠️ | 2026年全滅、CSV 追記処理 broken |
| netkeiba_training_times.csv | **0** | **0** ⚠️ | 同上 |
| netkeiba_stable_comments.csv | 1441 | 12 race_ids | 9/10/11R のみ取得仕様 |
| netkeiba_race_review.csv | (2025止) | **0** ⚠️ | 過去分のみ、新規無い |
| netkeiba_shinba_eval.csv | (2025止) | **0** ⚠️ | 同上 |

**判定**: netkeiba premium 系の **新規 race_id 追記処理が broken**。fetch は成功しているが pandas.to_csv の append 部分でデータが落ちている疑い。最終更新 timestamp:
- speed_index.csv → Apr 29 09:52 (5/2 以前)
- training_eval.csv → May 5 19:04 (時刻はあるが 2026 年新規 0 行)

---

### 2.4 v15 が要求する 150 features の予測時カバレッジ推定

| カテゴリ | features 数 | 5/2 平均 fill rate | 5/3 平均 fill rate |
|----------|---:|---:|---:|
| 基本 (距離/コース/枠等) | 14 | 100% | 100% |
| 過去成績 (prev1-3走 lag) | 10 | 70-80% | 70-80% |
| jockey/trainer expanding | 3 | 100% | 100% |
| 集計/派生 | 16 | 100% | 100% |
| 血統 (sire/bms 系) | 8 | ~95% | ~95% |
| 調教 (木/坂路 4F/3F/1F) | 8 | **0% (全て fallback)** | **0%** |
| netkeiba index_max/avg5/run1 | 4 | **0%** | **0%** |
| training_intensity_enc | 1 | **0%** | **0%** |
| pci/race_pace 派生 | 3 | 0% (SED 依存) | 0% |
| 当日情報 (オッズ/馬体重) | 8 | 95-100% | 95-100% |
| 馬場 (cushion/moisture) | 2 | 80% | 50% (KAB 5/3 3 行のみ) |
| 天候 (気象庁) | 5 | 100% | 100% |
| **JRDB SKB 系** | 6+ | **0%** | **0%** |
| **JRDB SED 系** (前走) | 12+ | 75-85% | **0% (404)** |
| **JRDB TYB 系** (LIVE) | 5+ | 80-95% | **0% (404)** |

→ **5/3 は 5/2 より features 取得が大幅劣化**。SED/TYB の 404 + KAB 3 行のみが直撃。

---

## 3. 取得失敗 原因別

| 原因 | 件数 | 該当 features | 備考 |
|------|------|--------------|------|
| **(d) script bug**: netkeiba premium CSV append 処理 broken | 5 CSV (speed/training_eval/training_times/race_review/shinba_eval) | index_max/avg5/run1, time_1f_last, training_intensity, pci 等 ~10 features | log は success と表示するが CSV 行 0 |
| **(c) JRDB publish タイミング**: TYB260503 / SED260503 が 5/3 06:00 時点で 404 | 2 種 | tyb_homestr/_inner 系 5+, sed 前走 系 12+ | 当日朝 publish される設計 → 6:00 早すぎ |
| **(c) JRDB publish タイミング**: KYI260503 が 5/3 03:00 時点 15 records (1R only) | 1 種 | KYI 全特徴 60+ | 5/3 06:00 再 fetch も 15 records 同 → 09時頃 publish 完了。6:00 fetch では足りない |
| **(d) script bug**: jrdb_paci.csv が 4/4 以降更新停止 | 1 種 | pci, race_pace_diff 等 3 features | 取得経路不明 (CLAUDE.md にも記載) |
| **(d) script bug**: jrdb_skb.csv 全期間 broken (2015-2025 の旧データのみ) | 1 種 | SKB 専門家印 6+ features | parse 経路喪失 |
| **(c) publish タイミング**: KAB260503 が 3 records のみ | 1 種 | cushion_value, moisture_rate (2/3 場欠損) | 馬場詳細は当日朝 publish |
| **(e) 構造上不可**: TYB は当日朝発表で 06:30 morning_top_races 時点では未公開 | - | LIVE 騎手調子 | 仕様 |
| **(e) 構造上不可**: 確定オッズ/馬体重は当日 9:00-12:00 公開 | - | odds_log, horse_weight | predict は 8:00 発火、12:00 再発火で対応中 |
| **(a) Cookie 切れ** | 0 | - | 5/2-5/3 共に Cookie OK 確認済 (1634 文字、`pre_fire_check_*.log`) |
| **(b) Ban / レート制限** | 0 | - | log に 4xx/5xx 無し (404 は publish 未発生で別 cause) |

**Top 3 原因**:
1. netkeiba premium CSV 追記 bug (script bug) — **改善可能、最高優先度**
2. JRDB TYB/SED の 5/3 朝 404 (publish タイミング) — 部分改善可能
3. jrdb_paci.csv / jrdb_skb.csv 更新停止 (legacy bug) — 改善可能

---

## 4. 改善可能なもの (構造上不可 除外) リスト

### 4.1 netkeiba premium CSV 追記 bug 修正

| feature | raw count 不足 | 修正方法 | 工数 | 5/9 朝 OK? |
|---------|---------------|----------|------|:-:|
| `index_max_filled`, `index_avg5_filled`, `index_run1_filled` | 5/2-5/3 全 67R 全馬 (~810 horses) | `tools/scrape_speed_index.py` の to_csv append/dedup ロジック確認、`mode='a', header=False` か検証 | 1.5h | **Y** |
| `time_1f_last_filled`, `training_intensity_enc` | 5/2-5/3 全 67R | `tools/scrape_training_times.py` または `scrape_training.py` の CSV 書込確認 | 1h | **Y** |
| (副次効果として) `wood_best_4f_filled`, `sakaro_best_4f_filled` | premium 不足時 fallback で mean fill されているが CSV から再 fetch 可 | 上記と同じ修正でカバー | (含む) | Y |

### 4.2 JRDB 5/3 朝 404 対策

| feature | raw count 不足 | 修正方法 | 工数 | 5/9 朝 OK? |
|---------|---------------|----------|------|:-:|
| TYB 系 (5+) — `jrdb_tb_homestr_inner` 等 | 5/3 全 34R 全馬 | (1) AM6 fetch を AM 9:00 に後ろ倒し or (2) AM 6/9/11 三段階 retry | 30min | **Y** |
| SED 系 (12+) — 前走詳細 | 5/3 全 34R | (1) 同上、9:00 retry / (2) `db.netkeiba.com` フォールバック (CLAUDE.md 既存) | 1h | **Y** (1 の retry のみ) |
| KYI 残り (5/3 朝 6:00 で 15 records → 9:00 で full) | 5/3 全 34R 全馬 | AM 6 → AM 9 へ retry 追加 | 30min | **Y** |
| KAB (馬場詳細) 5/3 3 行 | 5/3 全 34R | 上記 retry 設計に同梱 | (含む) | Y |

### 4.3 jrdb_paci.csv 4/4 以降 broken

| feature | raw count 不足 | 修正方法 | 工数 | 5/9 朝 OK? |
|---------|---------------|----------|------|:-:|
| `pci`, `prev_race_pace_diff`, `prev_race_first3f`, `prev_race_last3f` | 4/4 以降全レース | jrdb_paci の取得経路再確立 (CLAUDE.md 既知バグ "取得経路不明") | 4h+ (調査含) | **N** → Phase 3 |

### 4.4 jrdb_skb.csv 全期間 broken

| feature | raw count 不足 | 修正方法 | 工数 | 5/9 朝 OK? |
|---------|---------------|----------|------|:-:|
| SKB 専門家印 6+ features (kishi_code_1-6, baba_code, kyaku_code, padock_comment 等) | 全期間 (2015 以降は 旧 ekitachi.com 形式のみ) | parse script 修復 + parse → CSV merge | 3-6h | **N** → Phase 3 |

---

## 5. 5/9 投入で対応すべき項目 (high)

5/7-5/8 で着手可能な順:

| 優先 | 項目 | 工数 | 5/9 朝 期限 | 効果見積 |
|:-:|------|-----:|:-:|----------|
| 1 | **netkeiba premium CSV append bug 修正** (speed_index / training_eval) | 1.5h | OK (5/7 終夜) | features ~7個復活、AUC +0.005-0.01 期待 |
| 2 | **JRDB AM 9:00 retry タスク追加** (TYB/SED/KYI/KAB の 404 救済) | 1h | OK (5/8 朝) | 5/9 当日 SED/TYB の網羅率 80%+ 復活 |
| 3 | **5/9 当日 12:00 再 fetch + 13:00 予測再発火 確認** | 30min (動作確認のみ) | OK (5/8 中) | 当日朝の 404 → 昼までに full 復活 |
| 4 | premium_scrape.py の CSV append 検証 (各日 nightly_sanity に組込) | 1h | OK | 再発防止 |
| 5 | jrdb_skb.csv update 部の "broken" alert を nightly_sanity に追加 | 30min | OK (alert のみ) | 5/9 投入時の特徴量カバレッジ可視化 |

**累計工数 5h** → 5/7 夜 + 5/8 中 で消化可能。

---

## 6. Phase 3 (5/24+) 以降 TODO (med/low)

| 優先 | 項目 | 理由 |
|:-:|------|------|
| med | **jrdb_paci.csv 取得経路再確立** | pci/race_pace 系 3 features が完全死亡、4h+ 調査要 |
| med | **jrdb_skb.csv parse 修復** | SKB 専門家印 6+ features、長期的 ROI に効く可能性 |
| med | **shinba_eval / race_review の 2026年自動取得** | 現在 2025年止。新馬戦/前走不利 features (CLAUDE.md v12.1 不採用組) の再評価機会 |
| low | TYB の publish 時刻観測継続 (`data/tyb_publish_log.csv`) → AM 9 の根拠固める | 5/9 暫定対応のため |
| low | 京都データ蓄積 (5/11 以降 course_renovated 効果再評価) | 戦略⑦ 京都除外解除判断 |

---

## 7. 結論

### 7.1 5/2-5/3 投資 features カバレッジ実態

- **150 features 中 ~30 features が事実上 0% fill**:
  - netkeiba premium 7 features (script bug → CSV 追記 broken)
  - JRDB SKB 6 features (legacy broken)
  - JRDB PACI 3 features (4/4 以降 broken)
  - 5/3 のみ JRDB TYB 5 features + SED 12 features (publish タイミング 404)
- **戦略⑦ filter後 35R で見ても**: training 系/speed_index 系は全 R 全馬 0%。**v15 model は学習時 fill rate ~50% で訓練しているのに、本番で 0% fill** → distribution shift 発生。
- これは `data/v18/distribution_shift_analysis.md` の「RANK_SHIFT」(retro race_max_p factor 27.69x)と整合。

### 7.2 5/9 投入の前に着手すべき高優先度

1. **netkeiba premium CSV append bug 修正** (1.5h) — 5/7 夜  ★最優先
2. **JRDB AM 9:00 retry タスク追加** (1h) — 5/8 朝
3. **5/9 当日 12:00 再 fetch → 13:00 予測再発火 動作確認** (30min) — 5/8 夜

→ 累計 3h、5/9 朝 08:00 までに全て間に合う。

### 7.3 Phase 3 移行までに整備すべき構造改善

- jrdb_paci.csv / jrdb_skb.csv の取得経路復元 (合計 7-10h、5/24+ で着手)
- 「features 取得率」自動 alert を nightly_sanity に組込み (1h、5/9 後で OK)
- v15 訓練データ自体が一部 features 0% で fitted されている可能性も、Phase 3 で重み再学習 (v16 候補) に組込

### 7.4 ユーザー方針との整合

- 「取り返し禁止」「累計 +14,140 円死守」 → 5/9 投入は features カバレッジ ~75% (現状) → ~85% (修正後) で実施
- 5/7-5/8 の 3h 投資で improvement 期待 ROI +5〜10pt → 5/9 朝の最悪損失幅縮小に寄与
- Phase 3 は 5/24 以降に余裕を持って着手、撤退ライン -50,000円 まで余裕 +64,140円 を侵食しない範囲で

---

**source files referenced** (絶対パス):
- `C:\Users\takum\keiba-ai\data\daily_predictions\20260502.csv`
- `C:\Users\takum\keiba-ai\data\daily_predictions\20260503.csv`
- `C:\Users\takum\keiba-ai\data\cumulative_results.csv`
- `C:\Users\takum\keiba-ai\data\jrdb\extracted\Kyi\KYI260502.txt` (492 lines), `KYI260503.txt` (15 lines)
- `C:\Users\takum\keiba-ai\data\jrdb\extracted\Tyb\TYB260502.txt` (492), TYB260503.txt — **存在しない**
- `C:\Users\takum\keiba-ai\data\jrdb\extracted\Sed\SED260502.txt` (171), SED260503.txt — **存在せず**
- `C:\Users\takum\keiba-ai\data\netkeiba_speed_index.csv` (2026年 16行のみ、5/2-5/3 race rows = 0)
- `C:\Users\takum\keiba-ai\data\netkeiba_training_eval.csv` (2026年 0行)
- `C:\Users\takum\keiba-ai\data\netkeiba_stable_comments.csv` (5/2-5/3 race_id = 12個 only, R9-11 のみ)
- `C:\Users\takum\keiba-ai\logs\premium_scrape_20260502.log`, `premium_scrape_20260503.log`
- `C:\Users\takum\keiba-ai\logs\jrdb_kyi_auto_20260502.log`, `jrdb_kyi_auto_20260503.log`
