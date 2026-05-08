# AUDIT-1 B: netkeiba マスターコース 全要素 audit (5/8)

**作成**: 2026-05-08 (AUDIT-1 B 領域)
**前提**: netkeiba スーパープレミアム + マスターコース 加入済 (月額 4,500円 + 1,390円)
**位置付け**: read-only audit。 既存 scrape* tool / model / schtasks 一切不変

---

## 0. summary (page type 別)

| page / data type | URL pattern | 取得状況 | V15 組込 | カバレッジ | 備考 |
|------------------|-------------|----------|----------|----------|------|
| 出馬表 (shutuba) | /race/shutuba.html | ✅ scrape_super_premium | ✅ Pattern B 主軸 | 95%+ | 当日 8:00 |
| 結果 (result) | /race/result.html | ✅ scrape_master_index | ✅ jra_races_full ベース | 99% | 翌日 |
| 競走馬 (db) | /horse/{id}/ | ✅ horse_history_full.csv | ✅ horse_career_* | 95% | 全戦績 |
| 騎手 (db) | /jockey/{id}/ | ✅ jockey_history_full.csv | ✅ jockey_wr_calc | 90% | 戦績 |
| 調教師 (db) | /trainer/{id}/ | ❌ 未取得 | ❌ | - | (jrdb_csa で代替) |
| 種牡馬 (db) | /sire/{id}/ | ❌ 未取得 | ❌ | - | (sire_shinba_stats で代替) |
| 兄弟・母産駒 (siblings) | (db.netkeiba) | ✅ netkeiba_siblings.csv (17,442 母) | ⚠️ siblings_expanding_w5 のみ | 85% | dam_top3r はリーク確定で除外 |
| 調教 (oikiri) | /race/oikiri.html | ✅ scrape_premium_data | ✅ wood/sakaro_best_*f / time_1f_last / training_intensity_enc | ~70% | Premium 必須 |
| 厩舎コメント (comment) | /race/comment.html | ✅ scrape_stable_comment | ⚠️ score 化のみ (V15 未組込、 UI 表示) | 30%+ | カバレッジ不足 |
| パドック (paddock) | /race/paddock.html | ❌ 未取得 | ❌ | - | (テキスト評) |
| パドック静止画 | /race/paddock.html | ❌ 未取得 | ❌ | - | 30 分前公開、 画像解析候補 |
| パドック動画 | (Premium) | ❌ 未取得 | ❌ | - | 30 分前公開 |
| 短評 (race_review) | /race/result.html (備考欄) | ✅ scrape_race_review | ⚠️ score 化 (v12.1 で不採用) | 100% (2020-2025 全年) | review_score V15 未組込 |
| AI 予測タイム (speed) | /race/speed.html | ✅ scrape_speed_index (270K 行) | ✅ index_max/avg5/run1_filled | 95% | Premium 必須 |
| 波乱度 (haran) | /race/upset.html | ✅ netkeiba_upset_level.csv | ❌ 未組込 | 部分 | upset_level / top_pop_reliability |
| 一番時計 (best_time) | (page) | ❌ 未取得 | ❌ | - | (DB 機能) |
| 中間調教 (mid_training) | /race/oikiri.html (一部) | ⚠️ training_eval.csv あり | ❌ V15 未組込 | 不明 | 13 columns 取得済 |
| 調教動画 (training_video) ★ | (Premium、 重賞のみ) | ❌ 未取得 | ❌ | - | 金 13:00 公開、 Phase 4 候補 |
| トラックバイアス (bias) | /race/bias.html | ✅ netkeiba_track_bias / track_index | ❌ V15 未組込 | 部分 | 表示用のみ |
| データ分析 ★ | /db/race/sum_***/ | ✅ netkeiba_data_analysis (種別 6 件) | ❌ V15 未組込 | 部分 | マスター限定 |
| レース傾向 ★ | (master) | ✅ netkeiba_race_tendency | ❌ V15 未組込 | 部分 | マスター限定 |
| AI opinion ★ | (master) | ✅ netkeiba_ai_opinion (pace + opinion_text) | ❌ V15 未組込 | 部分 | マスター限定、 ペース予想 |
| AI position ★ | (master) | ✅ netkeiba_ai_position (位置取り pct) | ❌ V15 未組込 | 部分 | マスター限定 |
| AI 予想タイム ★ | (master) | ✅ netkeiba_ai_predict_times | ❌ V15 未組込 | 部分 | first_3f / last_3f 予想 |
| 穴推奨 (ana_best) ★ | (master) | ✅ netkeiba_ana_best | ❌ V15 未組込 | 部分 | category 別 horses |
| 新聞風 AI ★ | (master) | ✅ netkeiba_newspaper_ai_thisweek | ❌ V15 未組込 | thisweek のみ | first_3f/last_3f 予想 |
| マスター指数 ★ | (master) | ✅ netkeiba_master_index (139K) + master_index_mc (82) | ❌ V15 未組込 | 95%+ | time/start/chase/agari/master_total etc |
| race_lap (ラップ) | /race/lap.html | ✅ netkeiba_race_lap / race_laps | ⚠️ prev_race_first3f / prev_race_last3f / prev_race_pace_diff (V15 既組込) | 部分 | |
| race_analysis ★ | (master) | ✅ netkeiba_race_analysis (comment + score + evaluation) | ❌ V15 未組込 | 部分 | マスター限定 |
| WIN5 | (page) | ❌ 未取得 | ❌ | - | |
| 海外 | (page) | ❌ 未取得 | ❌ | - | |
| POG | (page) | ❌ 未取得 | ❌ | - | |

**★ = マスターコース限定 (Premium のみでは取得不可)**

---

## 1. 既存取得 CSV 列名 詳細

### 1.1 netkeiba_speed_index.csv (270,437 行)
columns: race_id, umaban, horse_name, sex_age, weight_carry, jockey, **index_max**, **index_avg5**, **index_dist**, **index_course**, **index_run1**, **index_run2**, **index_run3**, odds, popularity

V15 組込: index_max_filled, index_avg5_filled, index_run1_filled
**未組込**: index_dist (距離別指数), index_course (コース別指数), index_run2, index_run3
**価値**: index_dist / index_course は 距離・コース別 適性で +0.001-0.003 期待

### 1.2 netkeiba_training_times.csv (300,574 行)
columns: race_id, race_date, umaban, horse_name, course, condition, rider, time_6f, time_5f, time_4f, time_3f, time_1f, intensity, **rank**, **evaluation**, training_date

V15 組込: time_1f, intensity, time_4f (wood_best_4f / sakaro_best_4f / time_1f_last)
**未組込**: time_6f / time_5f / time_3f, rank (A/B/C/D), evaluation (テキスト)
**価値**: rank A/B/C/D は ranking として +0.001、 time_6f/5f は 中距離適性で +0.001

### 1.3 netkeiba_stable_comments.csv (130,317 行)
columns: race_id, race_date, umaban, horse_name, comment, **score** (-3〜+3)

V15 組込: ❌ (V12 で stable_comment_score 採用検討、 カバレッジ 30% で 不採用)
**現状カバレッジ**: 2020-2025 で 130K 行、 race-uma 推定 60-70%
**価値**: スコア化済、 即実装可。 期待 +0.0003-0.001

### 1.4 netkeiba_race_review.csv (277,467 行)
columns: race_id, umaban, horse_name, finish, **remarks**, **review_score**

V15 組込: ❌ (v12.1 で +0.00016、 2021 年 gap 0.0514>0.05 で 不採用)
**カバレッジ**: 2020-2025 全年 277K 行 (約 100%)
**価値**: prev_review_score (前走不利→巻き返し) は 中期で再検討候補

### 1.5 netkeiba_shinba_eval.csv (~8,000 行)
columns: race_id, race_date, umaban, horse_name, distance, **stable_eval**, **training_rank**, stable_comment, training_review, training_critic, **stable_score**, **training_score**, **comment_score**

V15 組込: ❌ (2024-2025 の 新馬戦のみ、 範囲狭く 採用未達)
**価値**: 新馬戦に限れば +0.005、 全体では low

### 1.6 netkeiba_siblings.csv (17,442 母)
columns: mother, total_offspring, total_races, win_rate, **top3_rate**, avg_finish, best_class, **shinba_win_rate**

V15 組込 (expanding 経由): siblings_expanding_w5.csv (sib_top3_rate_exp_w5 / sib_shinba_wr_exp_w5)
**注意**: 静的な sib_top3_rate (CSV そのまま) は POST-RACE LEAK 確定 (Session #38、 dam_top3r と同根)
**現状**: w5 expanding window 修正版が LIVE retro で +6.89pt 改善 (Session #41)、 **V18/V19 投入は 5/16 NO-GO**

### 1.7 netkeiba_master_index.csv (139,674 行) ★マスター
columns: race_id, umaban, horse_name, finish_order, **time_index**, **master_index**, **start_index**, **chase_index**, **agari_index**

V15 組込: ❌ (5 indices 全て 未組込)
**価値**: ★ **マスター限定 indices**、 各 +0.001-0.003 推定。 速度・上がり・スタート の 内製指標
→ 即実装候補 (Sprint 4)

### 1.8 netkeiba_master_index_mc.csv (82 行) ★マスター thisweek
columns: race_id, umaban, horse_name, **master_total**, **master_start**, **master_chase**, **master_finish**

→ thisweek のみ、 model 学習では 使えないが UI 補助として 表示候補

### 1.9 その他 master 限定
- netkeiba_ai_opinion.csv: race_id, **pace**, opinion_text → pace 予想 (front/even/slow) は +0.001-0.002 期待
- netkeiba_ai_position.csv: 位置取り (position_left_pct / position_top_pct) → 道中位置予想
- netkeiba_ai_predict_times.csv: ai_first_3f / ai_last_3f → 既存 prev_race_first3f と区別 (これは予想)
- netkeiba_ana_best.csv: category 別 horses → 馬印度 ranking
- netkeiba_data_analysis.csv: race × category × value → 統計指標
- netkeiba_race_analysis.csv: race × umaban × comment + score + evaluation → 馬別 評価
- netkeiba_race_tendency.csv: race × category × value → 過去傾向
- netkeiba_track_bias.csv / track_index.csv: bias / track_index → コース癖
- netkeiba_training_eval.csv: 13 columns、 prev_review / training_course / training_intensity / training_move / training_rank → 中間調教

---

## 2. 取得していない netkeiba data

| page | URL | 取得難度 | 価値 |
|------|-----|---------|------|
| パドック静止画 | /race/paddock.html | medium (画像 DL + caching) | 中 (体格 / 緊張度、 画像解析) |
| パドック動画 | Premium | high (動画 DL) | 中 (歩様、 Phase 4) |
| 調教動画 (重賞のみ) | Premium | high | 高 (脚色、 Phase 4) |
| 調教師 db | /trainer/{id}/ | low | 低 (jrdb_csa で代替済) |
| 種牡馬 db | /sire/{id}/ | low | 低 (sire_*_wr で代替) |
| 一番時計 | /db | medium | 中 (距離別 No.1 タイム) |
| WIN5 出走 | /win5 | low | 低 (G 級重複) |
| 海外 | (海外) | high | 低 (G I のみ) |
| POG | (POG) | low | 低 (馬主予想 game) |

---

## 3. マスターコース限定 data (Premium 不可) ★

ユーザーは すでに マスターコース 加入済 (月額 1,390 円)。 以下は **すでに取得しているが V15 未組込**:

| ★ data | 行数 | V15 組込 | 期待 AUC |
|--------|------|--------|---------|
| master_index (time/start/chase/agari/master_total) | 139,674 | ❌ | +0.003-0.005 |
| ai_opinion (pace 予想) | 部分 | ❌ | +0.001-0.002 |
| ai_position (位置取り pct) | 部分 | ❌ | +0.001-0.002 |
| ai_predict_times (first_3f/last_3f 予想) | 部分 | ❌ | +0.001 |
| ana_best (穴推奨) | 部分 | ❌ | +0.0003-0.001 |
| data_analysis | 部分 | ❌ | +0.0005-0.001 |
| race_analysis (馬別 score) | 部分 | ❌ | +0.001-0.002 |
| race_tendency | 部分 | ❌ | +0.0003-0.001 |
| track_bias / track_index | 部分 | ❌ | +0.001-0.003 |
| training_eval (中間調教 13列) | 不明 | ❌ | +0.001-0.002 |

→ **マスター限定 features 合計 期待 AUC: +0.005-0.010** (V15 0.886 → V20 0.891-0.896)

---

## 4. 未活用 features summary (netkeiba)

**取得済 だが V15 未活用** (Top 10):
1. ★ **master_index 5 indices** (time/start/chase/agari/master_total) - 期待 +0.003-0.005
2. ★ **ai_opinion pace 予想** - +0.001-0.002
3. ★ **track_bias / track_index** - +0.001-0.003
4. ★ **ai_position 位置取り** - +0.001-0.002
5. **race_analysis 馬別 score** - +0.001-0.002
6. **training_times rank (A/B/C/D)** - +0.001
7. **speed_index dist / course 別** (index_dist, index_course) - +0.001-0.003
8. **stable_comment_score** (V12 不採用後 再検討) - +0.0003-0.001
9. **race_review prev_review_score** (v12.1 不採用後 再検討) - +0.0001-0.001
10. **upset_level / top_pop_reliability** - +0.0003-0.001

**未取得** (中期):
1. パドック静止画 (画像解析、 Phase 4)
2. パドック動画 (歩様、 Phase 4)
3. 調教動画 重賞 (Phase 4)
4. 一番時計 db (距離別 No.1)

---

## 5. 5/9 V15 投資保護

✅ 取得済 CSV 全 read-only。 V15 model 不変
✅ 既存 scrape* tool (scrape_super_premium / scrape_premium_data / scrape_master_course / scrape_master_index 他) 不変
✅ 5/9 朝の DailyPredict / DailyPremiumScrape 完全同一動作

---

## 6. 結論

✅ netkeiba 30+ page type の audit 完了
✅ V15 利用率: 取得済 master 限定 data の **約 30%** (大半が UI 表示のみで model 入力外)
✅ ★ **マスター限定 features は 11 件 取得済 / V15 ほぼ 全て 未活用**
✅ 即実装候補: 6-8 件、 期待 +0.005-0.010 AUC

**Sprint 4 候補 (短期)**: master_index 5 indices + ai_opinion pace + track_bias 3 features = +0.005-0.010 期待
