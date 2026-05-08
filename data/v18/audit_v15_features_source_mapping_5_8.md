# AUDIT-1 D: V15 features → source mapping (5/8)

**作成**: 2026-05-08 (AUDIT-1 D 領域)
**前提**: V15 = 150 features (v14.1 base 131 + v15 new 10 + PACI Tier B 4 + jrdb_extra 5)
**位置付け**: read-only audit。 model / code 不変

---

## 0. V15 全 features 概要 (~150)

### V15 = v14.1 base + v15 新 + PACI Tier B

```
v13.5b base (124) + v14.1 PACI Tier A (7) = v14.1 base (131)
v14.1 base (131) + v15 new (10) + PACI Tier B (4) + jrdb_extra (5) = ~150
```

主要 source 別 内訳:

| source | features 数 | 主な features |
|--------|-----------|--------------|
| TFJV (jra_races_full.csv) | 約 60 件 | 基本 14 + ラグ 10 + 集計 5 + 派生 11 + V9.2/3 系 20 |
| JRDB KYI (前日) | 22 件 | jrdb_idm / *_idx (5) / 予想指数 (4) / コード (10+) 等 |
| JRDB SED (前走) | 8 件 | jrdb_prev_idm/track_bias/interference/late_start/ten_idx/agari_idx/pace_idx/rise_code |
| JRDB PACI Tier A | 7 件 | paci_manken_idx / goal_rank / dochu_rank / goal_diff / jockey_exp_wr / jockey_exp_3rd / ninki_idx |
| JRDB PACI Tier B | 4 件 | paci_sogo_mark / idm_mark / jockey_mark / train_mark |
| netkeiba speed_index | 4 件 | index_max_filled / avg5_filled / run1_filled / pci |
| netkeiba training_times | 5 件 | wood_best_4f / sakaro_best_4f / sakaro_best_3f / time_1f_last / training_intensity_enc |
| netkeiba_siblings (expanding) | 2 件 | sib_top3_rate_exp_w5 / sib_shinba_wr_exp_w5 |
| netkeiba race_lap | 4 件 | prev_race_first3f / last3f / pace_diff / prev_agari_relative |
| 派生 (V15 new、 14 件) | 14 件 | jockey_horse_* (5) / transport_* (2) / course_renovated (2) / gaisha_rank / paci Tier B (4) |
| TFJV (Pattern B のみ) | 8 件 | odds_log / pop_rank / horse_weight / weight_change / weight_cat / cushion_value / moisture_rate / weather_enc |
| 気象庁 API (Pattern B のみ) | 5 件 | temperature / humidity / wind_speed / precipitation / weather_enc (最終) |
| **合計** | **約 150** | (Pattern A) + 約 13 (Pattern B 追加) |

---

## 1. 主要 50 features → source mapping

| # | feature | source | datatype / field | 計算 logic | カバレッジ |
|---|---------|--------|-----------------|----------|----------|
| 1 | weight_carry | TFJV | SE.斤量 | 直接 | 100% |
| 2 | age | TFJV | SE.年齢 | 直接 | 100% |
| 3 | distance | TFJV | RA.距離 | 直接 | 100% |
| 4 | course_enc | TFJV | RA.course_code | mapping (10コード→0-9) | 100% |
| 5 | surface_enc | TFJV | RA.surface | mapping | 100% |
| 6 | sex_enc | TFJV | SE.性別 | mapping | 100% |
| 7 | num_horses_val | TFJV | RA.出走頭数 | 直接 | 100% |
| 8 | horse_num | TFJV | SE.umaban | 直接 | 100% |
| 9 | bracket | TFJV | SE.wakuban | 直接 | 100% |
| 10 | sire_enc | TFJV | UM.sire (TOP100) | TOP100 enc, 100=other | 99% |
| 11 | bms_enc | TFJV | UM.bms (TOP100) | 同上 | 99% |
| 12 | location_enc | TFJV | SE.所属 | mapping (美/栗/地方/外) | 99% |
| 13 | season | TFJV | RA.month | 春0/夏1/秋2/冬3 | 100% |
| 14 | jockey_wr_calc | TFJV | SE.jockey_id | expanding wr alpha=30 | 100% |
| 15 | jockey_course_wr_calc | TFJV | SE.jockey_id × course | expanding alpha=10 | 100% |
| 16 | jockey_surface_wr | TFJV | SE.jockey_id × surface | expanding alpha=10 | 100% |
| 17 | prev_finish | TFJV | SE.前走 finish | shift(1) | 95% |
| 18 | prev2_finish | TFJV | SE.前々走 finish | shift(2) | 90% |
| 19 | prev3_finish | TFJV | SE.3走前 finish | shift(3) | 85% |
| 20 | prev_last3f | TFJV | SE.前走 last_3f | shift | 95% |
| 21 | prev2_last3f | TFJV | SE.前々走 last_3f | shift | 90% |
| 22 | prev_pass4 | TFJV | SE.前走 pass4 | shift | 95% |
| 23 | prev_prize | TFJV | SE.前走 prize | shift | 95% |
| 24 | prev_odds_log | TFJV | SE.前走 odds_final | log shift | 95% |
| 25 | rest_days | TFJV | SE.前走日 | clip 1-365 | 95% |
| 26 | rest_category | TFJV | (rest_days) | 6カテゴリ (7/15/35/64/181) | 95% |
| 27 | avg_finish_3r | TFJV | SE | 直近3走平均 | 90% |
| 28 | best_finish_3r | TFJV | SE | 直近3走最高 | 90% |
| 29 | top3_count_3r | TFJV | SE | 直近3走 top3 数 | 90% |
| 30 | finish_trend | TFJV | SE | prev3 - prev | 85% |
| 31 | avg_last3f_3r | TFJV | SE | 直近3走 last_3f 平均 | 85% |
| 32 | dist_change | TFJV | RA × 前走 | distance - prev_distance | 95% |
| 33 | dist_cat | TFJV | RA.distance | 5 bin (1000/1400/1700/2000/2200+) | 100% |
| 34 | age_sex | TFJV | SE | age*10+sex_enc | 100% |
| 35 | horse_num_ratio | TFJV | SE × RA | umaban / num_horses | 100% |
| 36 | bracket_pos | TFJV | SE.wakuban | 内 0/中 1/外 2 | 100% |
| 37 | carry_diff | TFJV | SE | weight_carry - mean | 100% |
| 38 | horse_career_races | TFJV | SE.horse_id | expanding count | 100% |
| 39 | horse_career_wr | TFJV | SE.horse_id × is_win | expanding wr alpha=5 | 100% |
| 40 | horse_career_top3r | TFJV | SE.horse_id × is_top3 | expanding top3r alpha=5 | 100% |
| 41 | sire_surface_wr | TFJV | UM.sire × surface | expanding alpha=50 | 95% |
| 42 | sire_dist_wr | TFJV | UM.sire × dist | expanding alpha=50 | 95% |
| 43 | bms_surface_wr | TFJV | UM.bms × surface | expanding alpha=50 | 95% |
| 44 | wood_best_4f_filled | netkeiba | training_times.time_4f | 14 日 best mean fill ~52.0s | ~70% |
| 45 | sakaro_best_4f_filled | netkeiba | training_times | 同上 ~53.0s | ~50% |
| 46 | sakaro_best_3f_filled | netkeiba | training_times | 同上 ~39.0s | ~50% |
| 47 | training_intensity_enc | netkeiba | training_times.intensity | 0/1/2/3 | ~70% |
| 48 | time_1f_last_filled | netkeiba | training_times.time_1f | mean fill ~12.5s | ~70% |
| 49 | index_max_filled | netkeiba | speed_index.index_max | mean fill | ~95% |
| 50 | index_avg5_filled | netkeiba | speed_index.index_avg5 | mean fill | ~95% |

---

## 2. JRDB 主要 30 features → source mapping

| feature | datatype | column |
|---------|---------|--------|
| jrdb_idm | KYI | IDM |
| jrdb_training_idx | KYI | 調教指数 |
| jrdb_stable_idx | KYI | 厩舎指数 |
| jrdb_info_idx | KYI | 情報指数 |
| jrdb_composite_idx | KYI | 総合指数 |
| jrdb_upset_idx | KYI | 激走指数 |
| jrdb_ten_idx_pred | KYI | テン指数予想 |
| jrdb_pace_idx_pred | KYI | ペース指数予想 |
| jrdb_agari_idx_pred | KYI | 上がり指数予想 |
| jrdb_position_idx_pred | KYI | 位置指数予想 |
| jrdb_class_code | KYI | クラスコード |
| jrdb_rise_code | KYI | 上昇度 |
| jrdb_heavy_apt | KYI | 重適性コード |
| jrdb_hoof_code | KYI | 蹄コード |
| jrdb_ranch_rank | KYI | 放牧先ランク |
| jrdb_stable_rank | KYI | 厩舎ランク |
| jrdb_entry_days_ago | KYI | 入厩何日前 |
| jrdb_entry_race_num | KYI | 入厩何走目 |
| jrdb_training_arrow | KYI | 調教矢印コード |
| jrdb_stable_eval | KYI | 厩舎評価コード |
| jrdb_running_style | KYI | 脚質 |
| jrdb_dist_apt | KYI | 距離適性 |
| paci_manken_idx | PACI/KYI | 万券指数 |
| paci_goal_rank | PACI/KYI | ゴール順位予想 |
| paci_dochu_rank | PACI/KYI | 道中順位予想 |
| paci_goal_diff | PACI/KYI | ゴール差予想 |
| paci_jockey_exp_wr | PACI/KYI | 騎手期待勝率 |
| paci_jockey_exp_3rd | PACI/KYI | 騎手期待3着率 |
| paci_ninki_idx | PACI/KYI | 人気指数 |
| paci_sogo_mark | PACI | 総合印 (◎○▲△× → 5-1) |
| paci_idm_mark | PACI | IDM 印 |
| paci_jockey_mark | PACI | 騎手印 |
| paci_train_mark | PACI | 調教印 |
| jrdb_prev_idm | SED | 前走 IDM |
| jrdb_prev_track_bias | SED | 前走 baba_sa |
| jrdb_prev_interference | SED | 前走 furi |
| jrdb_prev_late_start | SED | 前走 deokure |
| jrdb_prev_ten_idx | SED | 前走 ten_idx |
| jrdb_prev_agari_idx | SED | 前走 agari_idx |
| jrdb_prev_pace_idx | SED | 前走 pace_idx |
| jrdb_prev_rise_code | SED | 前走 josho_code |

---

## 3. V15 新 14 features (v15 master + Tier B)

| feature | source | logic |
|---------|--------|------|
| jockey_horse_rides | TFJV | jockey_id × horse_id expanding count |
| jockey_horse_wr | TFJV | 同 expanding wr alpha=3 |
| jockey_horse_top3r | TFJV | 同 expanding top3r alpha=3 |
| jockey_change | TFJV | shift で 騎手交代 detect |
| jockey_change_to_top | TFJV | top20 騎手 への 変更 detect |
| transport_distance_km | TFJV | location_enc × course_enc Haversine |
| is_long_transport | 派生 | > 500 km |
| course_renovated | 派生 | 中京 2012/3 / 京都 2023/4 内 1 年 |
| post_renovation_flag | 派生 | リノベ 後 |
| gaisha_rank | JRDB KYI | 放牧先ランク (A=5..E=1) |
| paci_sogo_mark | JRDB PACI | 印 (◎=5..×=1) |
| paci_idm_mark | 同 | |
| paci_jockey_mark | 同 | |
| paci_train_mark | 同 | |

---

## 4. source 別 features 影響度 (逆引き)

「この source 1 file が 落ちた / 古い と V15 features X 件 が default 値に なる」

| source / file | 影響 features 数 | 影響 % | 失敗時 model AUC 影響 |
|--------------|----------------|------|--------------------|
| jra_races_full.csv | 約 60 (基盤) | 40% | -0.05 (致命的) |
| jrdb_kyi.csv (KYI) | 22 | 15% | -0.005 (大) |
| jrdb_sed.csv (SED) | 8 (前走) | 5% | -0.002 |
| jrdb_paci.csv (PACI) | 11 (Tier A 7 + Tier B 4) | 7% | -0.003 |
| jrdb_tyb.csv (TYB、 Pattern B のみ) | 5 | 3% | Pattern B のみ |
| netkeiba_speed_index.csv | 4 | 3% | -0.001 |
| netkeiba_training_times.csv | 5 | 3% | -0.001 |
| netkeiba_siblings_expanding_w5.csv | 2 | 1% | -0.002 (LIVE retro +6.89pt) |
| netkeiba_race_laps.csv | 4 | 3% | -0.001 |
| TFJV HR (jra_payouts.csv 4/6 停止) | 0 (model 入力外、 ROI 計算用) | 0% | ROI 計算 不可 |

---

## 5. 結論

✅ V15 ~150 features の source 別 mapping 完了
✅ 主要 50 features の 詳細 logic / coverage / source 表 完成
✅ source 影響度 逆引き 完成 → 監視で 重要度 判定可

**最重要 source 5 件** (Top の影響):
1. jra_races_full.csv (60 件、 40%)
2. jrdb_kyi.csv (22 件、 15%)
3. jrdb_paci.csv (11 件、 7%)
4. jrdb_sed.csv (8 件、 5%)
5. netkeiba_training_times.csv + speed_index.csv (9 件、 6%)
