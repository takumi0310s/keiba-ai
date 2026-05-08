# AUDIT-1 E2: 未活用 features 全リスト (5/8)

**作成**: 2026-05-08 (AUDIT-1 E 領域、 完全版)
**前提**: A + B + C audit 結果から V15 未組込 features 全網羅

---

## 1. JRA-VAN / TFJV 由来 (約 20 件)

| # | feature 候補 | datatype / field | 期待 | 工数 |
|---|------------|----------------|------|------|
| 1 | youbi (曜日) | RA | low | 1h |
| 2 | direction (内外周り) | RA | low | 1h |
| 3 | post_time (発走時刻) | RA | low | 1h |
| 4 | race_symbol (混合等) | RA | low | 1h |
| 5 | weight_type (馬齢別/別定/H/G) | RA | low | 1h |
| 6 | grade_code | RA | low (jrdb_class_code 経由 で部分カバー) | 1h |
| 7 | owner_code (馬主) | SE / UM | medium | 6h |
| 8 | breeder_code (生産者) | SE / UM | medium | 6h |
| 9 | coat_color (毛色) | UM | low | 2h |
| 10 | corner1-3 通過順 (前走以前) | SE | low-medium | 2h |
| 11 | time_sec (前走 走破タイム) | SE | medium | 2h |
| 12 | training_center (入厩先) | SE | low (放牧先で代替) | 2h |
| 13 | half_brother_id (半弟・全弟) | UM | medium (sib 拡張) | 6h |
| 14 | birth_date (生年月日) | UM | low | 1h |
| 15 | breeder_top3r | BS_DATA | medium-high | 8h |
| 16 | owner_top3r | BN_DATA | medium | 8h |
| 17 | dam_*_ext (90 年) | BR_DATA | medium | 8h |
| 18 | WIN5_appearance_count | W5_DATA | low | 4h |
| 19 | TM_DATA 直 利用 | TM_DATA | unknown | 6h |
| 20 | TK_DATA (特殊 race) | DE_DATA TK | low | 4h |
| 21 | JG (取消) リアルタイム | JG_DATA | live 改善 | 6h |
| 22 | wakuren_payouts (枠連) | HR | low (券種拡張時) | 2h |

---

## 2. netkeiba 由来 (約 30 件)

| # | feature 候補 | source | 期待 | 工数 |
|---|------------|--------|------|------|
| 1 | master_total | netkeiba_master_index.csv | medium-high | 1h |
| 2 | master_start | 同上 | medium-high | 1h |
| 3 | master_chase | 同上 | medium | 1h |
| 4 | master_finish | 同上 | medium | 1h |
| 5 | time_index | 同上 | medium | 1h |
| 6 | master_index (= total?) | 同上 | medium | 1h |
| 7 | start_index | 同上 | medium | 1h |
| 8 | chase_index | 同上 | medium | 1h |
| 9 | agari_index | 同上 | medium-high | 1h |
| 10 | netkeiba ai_opinion pace | netkeiba_ai_opinion.csv | medium | 2h |
| 11 | netkeiba ai_opinion text | 同上 | low (テキスト解析) | 6h |
| 12 | netkeiba ai_position 位置取り (left/top pct) | netkeiba_ai_position.csv | medium | 2h |
| 13 | netkeiba ai_predict_times (first_3f / last_3f) | netkeiba_ai_predict_times.csv | low | 2h |
| 14 | netkeiba ana_best 馬印度 | netkeiba_ana_best.csv | low | 2h |
| 15 | netkeiba data_analysis category × value | netkeiba_data_analysis.csv | low | 2h |
| 16 | netkeiba race_analysis 馬別 score / evaluation | netkeiba_race_analysis.csv | medium | 2h |
| 17 | netkeiba race_tendency category × value | netkeiba_race_tendency.csv | low | 2h |
| 18 | netkeiba track_bias text / track_index | netkeiba_track_bias.csv | medium | 2h |
| 19 | netkeiba upset_level / top_pop_reliability | netkeiba_upset_level.csv | low-medium | 1h |
| 20 | netkeiba newspaper_ai (first_3f / last_3f) | netkeiba_newspaper_ai_thisweek.csv | low | 2h |
| 21 | netkeiba speed_index dist 別 | speed_index.csv (index_dist) | medium | 1h |
| 22 | netkeiba speed_index course 別 | speed_index.csv (index_course) | medium | 1h |
| 23 | netkeiba speed_index run2 / run3 | speed_index.csv (index_run2/3) | low | 1h |
| 24 | netkeiba training_times rank A/B/C/D | training_times.csv | medium | 1h |
| 25 | netkeiba training_times evaluation (text) | training_times.csv | low | 4h |
| 26 | netkeiba training_times time_6f / time_5f / time_3f | 同上 | low-medium | 1h |
| 27 | netkeiba training_eval (中間調教 13 cols) | training_eval.csv | medium | 4h |
| 28 | netkeiba stable_comment_score (再検討) | stable_comments.csv | low (カバレッジ 60% 改善後) | 1h |
| 29 | netkeiba race_review prev_review_score (再検討) | race_review.csv | low | 2h |
| 30 | netkeiba shinba_eval 11 cols | netkeiba_shinba_eval.csv | low (新馬戦のみ) | 4h |
| 31 | netkeiba race_lap pace_first_half / second_half | netkeiba_race_lap.csv | low (race-level) | 1h |
| 32 | netkeiba paddock 静止画 解析 | image | medium-high | 80h+ |
| 33 | netkeiba paddock 動画 解析 | video | medium-high | 100h+ |
| 34 | netkeiba 調教動画 (重賞のみ) | video | high | 80h+ |
| 35 | netkeiba 一番時計 db | scrape 拡張 | medium | 6h |
| 36 | netkeiba 海外 db | scrape 拡張 | low | 6h |

---

## 3. JRDB 由来 (約 60 件)

### 3.1 SRB (取得済 / 完全 未組込) ★最優先

| feature | column | 期待 | 工数 |
|---------|--------|------|------|
| furlong_times (1F ごと タイム) | furlong_times | medium | 3h |
| corner1-4_order | corner1-4_order | medium | 2h |
| pace_up_pos (ペース上げ位置) | pace_up_pos | medium | 2h |
| bias_1corner / 2corner / backstr / 3corner / 4corner / straight (6 件) | bias_*corner | high (★) | 4h |
| race_comment | race_comment | low (テキスト) | 4h |

### 3.2 JO (取得済 / 完全 未組込)

| feature | column | 期待 | 工数 |
|---------|--------|------|------|
| soten_odds | soten_odds | medium | 1h |
| yoso_odds | yoso_odds | medium | 1h |
| cid_soten_idx / cid_sara_idx / cid_idx | cid_* | medium-high | 3h |
| ls_idx / ls_eval | ls_* | medium-high | 2h |
| em (馬印度) | em | low | 1h |
| gaisha_bb / gaisha_bb_wr / gaisha_bb_rensho | gaisha_* | medium | 3h |
| breeder_bb / breeder_bb_wr / breeder_bb_rensho | breeder_* | medium | 3h |

### 3.3 KKA (取得済 / 完全 未組込)

各 group 4 値 (1着/2着/3着/着外) × 12 group = 48 fields

| group | fields | 期待 |
|-------|--------|------|
| jra_seiseki | jra_seiseki_1/2/3/out | low (jra ベース) |
| koryu_seiseki | 同 | low (交流) |
| kyori_seiseki | 同 | medium |
| track_seiseki | 同 | medium |
| heavy_seiseki | 同 | medium |
| rest_seiseki | 同 | low |
| class_seiseki | 同 | medium |
| season_seiseki | 同 | low |
| waku_seiseki | 同 | medium |
| saka_seiseki | 同 | low |
| speed_seiseki | 同 | medium |
| dam_rensho_max/min/avg + bms_rensho_max/min/avg (6 fields) | dam/bms_rensho_* | medium (sib_*_exp と補完) |

### 3.4 CHA (取得済 / 完全 未組込)

| feature | column | 期待 |
|---------|--------|------|
| oikiri_date / count | oikiri_date / count | low |
| oikiri_course / shurui / aite | 同 | low-medium |
| oikiri_rank | oikiri_rank | medium |
| oikiri_idx | oikiri_idx | medium |
| ten_time / chukan_time / shimai_time + idx (6) | *_time / *_time_idx | medium |
| awase_result / shurui / nenrei / class (4) | awase_* | low-medium |

### 3.5 CYB (取得済 / 部分 組込)

| feature | column | 期待 |
|---------|--------|------|
| train_type | | low |
| train_course_type / train_course | | low-medium |
| train_baba | | low |
| train_mark | | medium |
| train_amount | | low-medium |
| train_change | | low-medium |
| train_eval | | medium |

### 3.6 KYI 残 (取得済 / 部分 組込)

| feature | column | 期待 |
|---------|--------|------|
| 基準オッズ / 基準人気順位 | | medium (リーク risk 注意) |
| 基準複勝オッズ / 基準複勝人気順位 | | medium |
| 印 残り 3 件 (情報印 / 厩舎印 / 激走印) | | low |
| 激走タイプ / 激走順位 / LS指数順位 | | low |
| テン/ペース/上がり/位置 順位 (4) | | low |
| 騎手期待単勝率 (paci 採用済 重複) | | (重複) |
| 取消フラグ (リアルタイム) | | live 改善 |

### 3.7 TYB 残 (Pattern B)

| feature | column | 期待 |
|---------|--------|------|
| bagu_change / ashimoto | | medium (Pattern B) |
| cancel_flag | | live |
| start_time | | low |

### 3.8 UKC 残 (取得済 / 部分 組込)

| feature | column | 期待 |
|---------|--------|------|
| hair_color_code | | low |
| keito_code | | medium |
| father_birth_year / mother_birth_year / bms_birth_year | | low |
| owner_code | | medium |
| breeder_name | | medium |
| birthplace | | low |

### 3.9 KZ / CZ / KSA / CSA (騎手 / 調教師 master)

各 30+ fields。 V15 内製 expanding と 重複が多い。
- year_leading / last_leading / total_leading - low-medium
- year/last/total turf_*/dirt_* 着回数 - low (内製と重複)

### 3.10 OZ / OW / OU / OT / OV (オッズ系)

- OZ tansho_01-18 / fukusho_01-18 - リーク risk
- OW wide_min/max/median - 配当期待値計算
- OU umaren_count/min/max/median/p10 - 同
- OT trio_count/min/max/median/p10 - 同
- OV tierce_count/min/max/median/p10 - 同

### 3.11 KAB / KAA / JOA / KTA / BAC (場・トラック / 出走馬)

- KAB 直線馬場差 (4) / 草丈 / 転圧 / 凍結防止剤 / 中間降水量 / 連続日 - low-medium
- KTA condition_class / blinker - low-medium
- BAC race_symbol / weight_type - low (RA 経由 取得可)

---

## 4. 派生 / 合成 features 候補 (V15 で未実装)

| feature | 計算 | 期待 |
|---------|------|------|
| jockey_horse_recent_3 (直近 3 ride 着順) | shift | low |
| trainer_horse_compat (調教師 × horse) | expanding wr | medium |
| sire_jockey_combo | sire × jockey | low |
| trainer_jockey_combo | trainer × jockey | low |
| course_distance_ranking (この距離 で 速い馬) | ranking | low |
| momentum_score (3 走連続上昇) | shift × diff | low |
| pace_match_score (個馬 pace と race pace 一致度) | diff | medium |
| weight_change_3r (馬体重 直近 3 走 trend) | shift | medium (Pattern B) |

---

## 5. 集計

| カテゴリ | features 数 | 期待 AUC 合計 |
|---------|-----------|-------------|
| TFJV 未活用 (RA/SE/UM/BR/BS/OW/W5 等) | 22 | +0.010-0.020 |
| netkeiba 未活用 (master + Premium 系) | 36 | +0.010-0.020 (+ 動画 +0.005-0.010) |
| JRDB 未活用 (SRB/JO/KKA/CHA 等) | 60+ | +0.015-0.030 |
| 派生 / 合成 | 8 | +0.001-0.005 |
| **合計** | **約 130 件** | **+0.036-0.075** |

ただし 重複 / 冗長を 考慮すると 実効 AUC ゲインは **+0.010-0.020** が現実的。
V15 0.886 → V20 0.896-0.906 が 妥当な目標。

---

## 6. 結論

✅ 全網羅 audit 完了 (約 130 features 候補 抽出)
✅ Top 30 ROI ranking は audit_unused_features_top30_5_8.md に集約
✅ 採用判定基準 (リーク check / WF gap / カバレッジ) で 真の +0.010-0.020 ゲインを 段階的に 検証
