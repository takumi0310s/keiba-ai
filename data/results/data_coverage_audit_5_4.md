# 5/4 朝 データカバレッジ完全監査

生成: 2026-05-04 07:54 (Opus xhigh, Session#7)

## 結論サマリー

| 分類 | 件数 | 主な対象 |
|------|----:|---------|
| 🔴 致命的欠損 | **2** | netkeiba premium データ (race_analysis, stable_comments) |
| 🟠 高影響 | **6** | netkeiba ai_*, siblings, JRDB Tyb 当日 / Skb 当日, training_times 2025停止 |
| 🟡 中影響 | **4** | JRDB ot/ov/ow/oz 33日 stale, jra_payouts 4/26まで |
| 🟢 軽微 | **多数** | Master 訓練データ (静的、運用に影響少) |

→ **致命的欠損 2件は V17 ULTRA-CLEAN 直接影響、即修復推奨**。
   ただし 5/9 投資判断には影響なし (V15 ベースで運用、V17 系は Phase 2.5)。

## 1. JRDB CSV (26種)

🟢 全 5/3 まで取得: kyi, kka, skb, sed, srb, hjc, ukc, paci, tyb, jo, joa, kab, kta, cha, cyb, bac, cz, kz, kaa, csa, ksa, ou

| CSV | rows | 2026 件 | age | 状態 |
|-----|----:|------:|---:|------|
| jrdb_kyi | 290,980 | (race_id列なし、年月日結合) | 0d | 🟢 |
| jrdb_kka | 547,611 | 16,987 | 0d | 🟢 |
| jrdb_skb | 547,100 | 16,495 | 0d | 🟢 |
| jrdb_sed | 547,773 | 16,154 | 0d | 🟢 |
| jrdb_srb | 39,153 | 1,151 | 0d | 🟢 |
| jrdb_paci | 548,606 | 16,987 | 0d | 🟢 |
| jrdb_tyb | 548,112 | 16,475 | 0d | 🟢 |
| jrdb_jo | 301,718 | 15,658 | 0d | 🟢 |
| jrdb_kab | 3,269 | (kaisai_key) | 0d | 🟢 |
| jrdb_ukc | 60,311 | (馬基本) | 0d | 🟢 |
| jrdb_cha | 301,718 | 15,658 | 0d | 🟢 |
| jrdb_cyb | 548,606 | 16,987 | 0d | 🟢 |
| jrdb_kta | 298,551 | 12,581 | 0d | 🟡 (KTA260405停止、上書き不能) |
| jrdb_kaa | 1,805 | - | 1d | 🟡 (KAA260405 停止) |
| jrdb_bac | 39,161 | 1,159 | 1d | 🟢 |
| **jrdb_ot** | 21,591 | 858 | **33d** | 🔴 三連単オッズ (4/1停止) |
| **jrdb_ov** | 21,592 | 858 | **33d** | 🔴 三連複オッズ |
| **jrdb_ow** | 21,591 | 858 | **33d** | 🔴 ワイドオッズ |
| **jrdb_oz** | 21,591 | 858 | **33d** | 🔴 基準オッズ (重要) |
| jrdb_ou | 39,161 | 1,159 | 1d | 🟢 |

### 評価

- **基本データは全て 5/3 まで最新** (kyi, sed, srb, hjc, paci 等)
- **問題**: ot/ov/ow/oz 4種が 33日 stale → これらは V15 学習に未使用、現状運用影響少だが潜在課題
- KTA, KAA は 4/5 で publish 停止 (Phase 0 で確認済、JRDB site 側問題)

### 影響度

**🟡 中** — V15/V17 学習・予測は影響なし。Phase 2.5 で v15.1 拡張時に oz 等を入れたい場合は再取得必要。

## 2. JRDB Extracted (raw .txt)

5/3 公開状況:

| 種別 | 5/3 ファイル | 状態 |
|------|------------|------|
| Bac, Cha, Cyb, Hjc, Jo (JOA), Kab, Kka, Kyi, Paci, Ukc | ✓ 取得済 | 🟢 |
| Sed, Srb (in Sed/) | ✓ 取得済 | 🟢 |
| Tyb | **TYB260502 まで** (5/3 取得失敗 5/3 14:50で 404) | 🔴 |
| Skb | **SKB260426 まで** (5/3 未公開) | 🟠 |
| Kaa, Kta | 4/5 で停止 (上流の publish 停止) | 🟡 |
| Cs, Ks | 5/2 まで | 🟢 |
| Ot, Ov, Ow, Oz | 5/3 取得済 (ファイル) だが集約 CSV 33日 stale | 🟡 |
| Ou | 5/3 のみ (履歴ない、ファイル数 2) | 🟡 |

## 3. netkeiba CSV

| CSV | rows | age | 状態 | 影響 |
|-----|----:|----:|------|------|
| netkeiba_ana_best | 52,696 | 0d | 🟢 | - |
| netkeiba_data_analysis | 5,105 | 0d | 🟢 | - |
| netkeiba_data_analysis_thisweek | 1,329 | 0d | 🟢 | - |
| netkeiba_race_lap | 32,009 | 0d | 🟢 | - |
| netkeiba_track_bias | 33,199 | 0d | 🟢 | - |
| netkeiba_training_eval | 447,694 | 0d | 🟢 | - |
| netkeiba_upset_level | 36,773 | 0d | 🟢 | - |
| netkeiba_speed_index | 269,947 | **4d** | 🟡 | prev_index_* 計算用 |
| **netkeiba_ai_opinion** | 4,929 | **35d** | 🔴 | - |
| **netkeiba_ai_position** | 67,952 | **35d** | 🔴 | ai_pos_left/top (V17 features) |
| **netkeiba_race_analysis** | 52,764 | **32d** | 🔴 | **ra_score (V17 feature)** |
| **netkeiba_stable_comments** | 128,875 | **23d** | 🔴 | **sc_score (V17 feature)** |
| **netkeiba_siblings** | 17,441 | **35d** | 🔴 | sib_* (V17 features、列なし問題) |
| netkeiba_shinba_eval | 7,998 | 35d | 🔴 | - |
| netkeiba_race_review | 277,466 | 35d | 🔴 | - |
| netkeiba_master_index | 139,673 | **18d** | 🔴 | netkeiba master 全体 |
| netkeiba_track_index | 20,769 | 16d | 🟡 | - |
| netkeiba_training_times | 300,573 | 24d | 🔴 | wood_*, sakaro_* features |

### 影響度

**🔴 高影響**: 
- ra_score, sc_score, ai_pos_*, sib_* — V17 ULTRA-CLEAN で全0/NaN になる features (Phase 1.7 audit と一致)
- これらが 23-35日 stale → Phase 2.5 で再取得必要

## 4. 訓練データ (静的)

| CSV | rows | age | 用途 |
|-----|----:|----:|------|
| training_times.csv | 955,580 | **38d** | 🟡 学習用 (cache 経由なので運用影響少) |
| jra_races_full.csv | 531,619 | 38d | 同上 |
| blood_full.csv | 58,921 | 38d | 同上 |
| odds_history.csv | 778,387 | **54d** | 同上 (2ヶ月前) |
| horse_history_full.csv | 81,986 | 54d | 同上 |
| jockey_history_full.csv | 427 | 35d | (小さい、再生成可) |
| trainer_history_full.csv | 457 | 35d | 同上 |

### 評価

🟡 **中影響** — 既存 V15/V17 model は cache (1.2GB) から学習済みなので運用影響なし。  
**Phase 2.5 で v15.1 / v18.1 再 train 時** には これらを最新化する必要。

### training_times.csv の特殊事情

引き継ぎ書記載の "training_times 2025: 2,551件のみ → スクレイピング途中停止"。  
本ファイルは 955,580 rows、5/3 までの累計データ。以下、年別分布確認:

```bash
# (確認別途) 2015-2025 各年の件数 → 2025 が 2,551 のみなら異常
```

→ 後続で確認、もし真に 2025=2,551 なら **Phase 2.5 高優先度** で続き取得。

## 5. odds_base (前日オッズ)

| ファイル | rows | mtime | 用途 |
|---------|----:|-------|------|
| 20260418-04/26 | 473-513 | 各日朝 | ✓ 既存 |
| 20260427/28 | 12 | 5/2-5/4 | (テスト用、無視) |
| **20260502** | 552 | 5/3 retro | ✓ Session#4 で構築 |
| **20260503** | 580 | 5/3 retro | ✓ Session#4 で構築 |
| 20260509 | (未生成) | - | 5/9 当日 daily_predict 内で生成 |

### 評価

🟢 **5/9 まで OK** — 過去分は retro 構築完了、5/9 当日は daily_predict.py が自動生成。

## 6. JRA公式配当 (jra_payouts)

| 項目 | 値 |
|------|---|
| rows | 12,261 |
| latest_date | 20260426 |
| age | 1d |

→ 5/2-5/3 分は未取得 (まだ公開されていないか、scrape_jra_payouts.py 未実行)。  
   5/2-5/3 結果集計は daily_results CSV ベースで実施済 (差は微小)。

### 評価

🟡 **中影響** — Phase 2.5 で本格 ROI BT に jra_payouts 使うなら 5/2-5/3 取得推奨。

## 7. weekly_premium_cache

```
total: 18 dirs (4/19 ~ 5/3)
mostly small (< 400KB) per day
```

🟢 **適切に運用中**

## 8. daily_predictions / daily_results

| 種類 | files | 最新 |
|------|------:|------|
| daily_predictions | 12 | 20260503.csv |
| daily_results | 14 | 20260503.csv |
| paired (両方ある日) | 12 |  4/11, 4/12, 4/18, 4/19, 4/25, 4/26, 5/2, 5/3 |
| pred only | 4/15以前等 | 軽微 |
| result only | 4/4, 4/5 | 軽微 |

### 評価

🟢 **healthy 4日 + その他で十分**。Phase 1-B / Session#4 で 152R 分析完了。

## 9. v17/v18 train cache

| ファイル | size | age |
|---------|----:|----:|
| data/v17/_v17_train_df_cache.pkl | 1.2 GB | 5d (4/29) |
| data/_v15_optuna_df_cache.pkl.gz | (確認別途) | - |

→ V18/V19 学習で使用済 (5/3 train 後)。再学習不要なら更新不要。

## 修復優先順位

### 🔴 致命的 (即修復推奨、~30分)

1. **netkeiba_race_analysis.csv 再起動** (32日 stale)
   - スクリプト: `tools/scrape_data_analysis.py` (要確認)
   - V17 features ra_score 復活
2. **netkeiba_stable_comments.csv 再起動** (23日 stale)
   - スクリプト: `tools/scrape_comments_bulk.py`
   - V17 features sc_score 復活

### 🟠 高 (1週間以内)

3. **netkeiba_ai_position.csv** (35日)
4. **netkeiba_siblings.csv** (35日、sib_* feature の元)
5. **netkeiba_master_index.csv** (18日、master データ)
6. **netkeiba_speed_index.csv** (4日、prev_index_*)
7. **TYB publish タイミング 連続観測** (5/4-5/10、戦略の生死判定)
8. **jra_payouts.csv 5/2-5/3** 取得 (`python scrape_jra_payouts.py`)

### 🟡 中 (Phase 2.5 後半)

9. **JRDB ot/ov/ow/oz 再取得** (33日 stale)
10. **training_times 2025 続き取得** (引き継ぎ書記載課題)
11. **odds_history.csv 再取得** (54日、Phase 2.5 BT 拡張用)

### 🟢 低 (放置可)

- jrdb_kaa/kta 5/3 (publish停止、JRDB site 待ち)
- 静的訓練データ (cache 経由で問題なし)

## 重大な発見

### 1. netkeiba premium データ全般 stale

23-35日 stale な netkeiba_*.csv 多数。これらは V17 ULTRA-CLEAN で 全0/NaN だった features と一致 (Phase 1.7 audit)。

→ **5/4-5/10 で premium scraping 再起動推奨**。

### 2. JRDB オッズ系 (ot/ov/ow/oz) 33日 stale

V15/V17 model の現状学習に未使用だが、将来的な拡張で必要。

### 3. training_times 2025 件数異常

引き継ぎ書記載の "2,551件のみ" を別途検証必要。

### 4. SCRAPER-GUARD 解除中

5/4 月曜 07:54 → 月曜 06:00 以降で SCRAPER-GUARD 解除。  
→ 即修復作業を実行できる時間帯。

## 後続文書

- `data/v18/phase_2_5_remaining_tasks_5_4.md` ← Phase 2.5 残タスク棚卸し
- `data/v18/today_5_4_candidates.md` ← 今日 5/4 やれること
