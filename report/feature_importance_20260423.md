# v15 特徴量重要度分析

生成日時: 2026-04-23 18:48
モデル: keiba_model_v15_central_live.pkl.gz
特徴量数: 150

## カテゴリ別集計

| カテゴリ | n | LGB total gain | XGB total gain | LGB平均順位 |
|----------|---|---------------|---------------|------------|
| jockey | 11 | 598636 | 3084 | 73.4 |
| jrdb_kyi (基本) | 34 | 383128 | 755 | 64.4 |
| pace_paci | 8 | 349900 | 1415 | 40.6 |
| training | 11 | 158493 | 234 | 88.8 |
| basic | 37 | 124149 | 927 | 76.9 |
| horse_career | 7 | 50359 | 135 | 41.6 |
| prev_race | 12 | 25574 | 129 | 88.2 |
| odds | 5 | 22340 | 126 | 87.8 |
| pedigree | 6 | 18546 | 53 | 69.0 |
| jrdb_sed (前走) | 6 | 17604 | 72 | 87.5 |
| condition | 4 | 14235 | 50 | 60.8 |
| logistics | 4 | 7266 | 43 | 106.0 |
| jrdb_tyb (当日) | 5 | 0 | 0 | 136.0 |

## TOP30 (LGB+XGB 平均順位ベース)

| Rank | feature | category | lgb_imp | xgb_imp |
|------|---------|----------|---------|---------|
| 1 | `paci_jockey_exp_3rd` | jockey | 314750 | 1467 |
| 2 | `paci_jockey_exp_wr` | jockey | 266347 | 1515 |
| 3 | `paci_ninki_idx` | pace_paci | 295197 | 630 |
| 4 | `jrdb_ze_idm_avg` | jrdb_kyi (基本) | 175971 | 92 |
| 5 | `training_time_filled` | training | 96546 | 72 |
| 6 | `paci_sogo_mark` | pace_paci | 18266 | 583 |
| 7 | `jrdb_idm` | jrdb_kyi (基本) | 35433 | 65 |
| 8 | `pop_rank_change` | odds | 17697 | 72 |
| 9 | `paci_goal_rank` | pace_paci | 17174 | 83 |
| 10 | `jrdb_class_code` | jrdb_kyi (基本) | 29243 | 61 |
| 11 | `training_per_dist` | training | 53711 | 52 |
| 12 | `horse_career_wr` | horse_career | 21692 | 45 |
| 13 | `jrdb_ze_ten_avg` | jrdb_kyi (基本) | 36674 | 37 |
| 14 | `jrdb_training_idx` | jrdb_kyi (基本) | 11941 | 54 |
| 15 | `surface_dist_enc` | basic | 16267 | 50 |
| 16 | `age` | basic | 8245 | 67 |
| 17 | `age_season` | basic | 12364 | 38 |
| 18 | `paci_idm_mark` | pace_paci | 7326 | 59 |
| 19 | `distance` | basic | 10038 | 34 |
| 20 | `surface_enc` | basic | 6970 | 54 |
| 21 | `jrdb_composite_idx` | jrdb_kyi (基本) | 10946 | 31 |
| 22 | `jrdb_prev_idm` | jrdb_sed (前走) | 10749 | 28 |
| 23 | `age_sex` | basic | 6655 | 40 |
| 24 | `jrdb_agari_idx_pred` | jrdb_kyi (基本) | 8375 | 22 |
| 25 | `transport_distance_km` | logistics | 7130 | 23 |
| 26 | `jrdb_ze_agari_avg` | jrdb_kyi (基本) | 11171 | 19 |
| 27 | `course_enc` | basic | 6388 | 27 |
| 28 | `horse_career_top3r` | horse_career | 6889 | 20 |
| 29 | `avg_last3f_3r` | prev_race | 6322 | 20 |
| 30 | `jrdb_kta_idm` | jrdb_kyi (基本) | 3625 | 55 |

## 下位30 (削除候補、LGB+XGB ともに寄与小)

| Rank | feature | category | lgb_imp | xgb_imp |
|------|---------|----------|---------|---------|
| 121 | `jrdb_prev_interference` | jrdb_sed (前走) | 160 | 11 |
| 122 | `jockey_horse_wr` | jockey | 964 | 10 |
| 123 | `jockey_change_to_top` | jockey | 127 | 11 |
| 124 | `jockey_change` | jockey | 154 | 11 |
| 125 | `stable_comment_score` | basic | 201 | 11 |
| 126 | `jrdb_ls_idx` | jrdb_kyi (基本) | 776 | 10 |
| 127 | `bracket` | basic | 686 | 10 |
| 128 | `top3_count_3r` | basic | 158 | 11 |
| 129 | `weight_peak_diff` | condition | 972 | 9 |
| 130 | `jrdb_heavy_apt` | jrdb_kyi (基本) | 578 | 10 |
| 131 | `paci_train_mark` | pace_paci | 337 | 10 |
| 132 | `course_renovated` | logistics | 70 | 10 |
| 133 | `jrdb_stable_eval` | jrdb_kyi (基本) | 211 | 10 |
| 134 | `is_long_transport` | logistics | 65 | 10 |
| 135 | `jrdb_prev_rise_code` | jrdb_sed (前走) | 35 | 0 |
| 136 | `odds_sharp_drop` | odds | 14 | 0 |
| 137 | `pci` | basic | 0 | 0 |
| 138 | `sire_shinba_top3r` | pedigree | 0 | 0 |
| 139 | `is_nar` | basic | 0 | 0 |
| 140 | `prev_odds_log` | odds | 0 | 0 |
| 141 | `has_training` | training | 0 | 0 |
| 142 | `prev_race_first3f` | prev_race | 0 | 0 |
| 143 | `prev_race_last3f` | prev_race | 0 | 0 |
| 144 | `prev_race_pace_diff` | prev_race | 0 | 0 |
| 145 | `gaisha_rank` | logistics | 0 | 0 |
| 146 | `jrdb_paddock_idx` | jrdb_tyb (当日) | 0 | 0 |
| 147 | `jrdb_odds_idx` | jrdb_tyb (当日) | 0 | 0 |
| 148 | `jrdb_live_composite_idx` | jrdb_tyb (当日) | 0 | 0 |
| 149 | `jrdb_body_code` | jrdb_tyb (当日) | 0 | 0 |
| 150 | `jrdb_demeanor_code` | jrdb_tyb (当日) | 0 | 0 |

## 4/23 SED merge 修正特徴量の重要度

| feature | LGB rank | XGB rank | LGB imp | XGB imp |
|---------|----------|----------|---------|---------|
| `jrdb_prev_idm` | 19 | 29 | 10749 | 28 |

## 改善方向性

1. **削除候補**: 下位30の中で複数モデルともに 0 ベースの特徴量は次回再学習で削除検討
2. **強化候補**: TOP30 の中で派生・組み合わせ可能な特徴量は新規生成検討
3. **カテゴリ偏り**: jrdb_* が下位にあれば SED merge カバレッジ修正の効果未反映の可能性
