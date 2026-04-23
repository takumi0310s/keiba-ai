# 特徴量カバレッジレポート (20260419)

対象: 35レース / 476頭

## カテゴリ別平均カバレッジ

| category | mean_cov | min_cov | n_features |
|---|---|---|---|
| jrdb_tyb | 0.0% | 0.0% | 5 |
| jrdb_sed_prev | 48.2% | 0.6% | 8 |
| jrdb_kyi_basic | 81.7% | 1.9% | 24 |
| jrdb_blood | 100.0% | 100.0% | 2 |
| jrdb_kab_sr | 100.0% | 100.0% | 3 |
| jrdb_jo | 100.0% | 100.0% | 2 |
| jrdb_kta | 100.0% | 100.0% | 3 |
| jrdb_cha | 100.0% | 100.0% | 3 |
| jrdb_skb | 100.0% | 100.0% | 3 |
| jrdb_ze | 100.0% | 100.0% | 4 |

## 特徴量別 (carverage<80% を抜粋、悪い順)

| feature | category | non_default | default_used | n | status |
|---|---|---|---|---|---|
| jrdb_paddock_idx | jrdb_tyb | 0.0% | 100.0% | 476 | LOW |
| jrdb_odds_idx | jrdb_tyb | 0.0% | 100.0% | 476 | LOW |
| jrdb_live_composite_idx | jrdb_tyb | 0.0% | 100.0% | 476 | LOW |
| jrdb_body_code | jrdb_tyb | 0.0% | 100.0% | 476 | LOW |
| jrdb_demeanor_code | jrdb_tyb | 0.0% | 100.0% | 476 | LOW |
| jrdb_prev_interference | jrdb_sed_prev | 0.6% | 99.4% | 476 | LOW |
| jrdb_rise_code | jrdb_kyi_basic | 1.9% | 98.1% | 476 | LOW |
| jrdb_prev_rise_code | jrdb_sed_prev | 1.9% | 98.1% | 476 | LOW |
| jrdb_stable_eval | jrdb_kyi_basic | 10.3% | 89.7% | 476 | LOW |
| jrdb_prev_late_start | jrdb_sed_prev | 12.6% | 87.4% | 476 | LOW |
| jrdb_training_arrow | jrdb_kyi_basic | 18.3% | 81.7% | 476 | LOW |
| jrdb_entry_days_ago | jrdb_kyi_basic | 53.6% | 46.4% | 476 | REVIEW |
| jrdb_heavy_apt | jrdb_kyi_basic | 57.1% | 42.9% | 476 | REVIEW |
| jrdb_dist_apt | jrdb_kyi_basic | 71.6% | 28.4% | 476 | REVIEW |
| jrdb_prev_idm | jrdb_sed_prev | 73.1% | 26.9% | 476 | REVIEW |
| jrdb_prev_track_bias | jrdb_sed_prev | 73.5% | 26.5% | 476 | REVIEW |
| jrdb_prev_agari_idx | jrdb_sed_prev | 74.4% | 25.6% | 476 | REVIEW |
| jrdb_prev_pace_idx | jrdb_sed_prev | 74.4% | 25.6% | 476 | REVIEW |
| jrdb_prev_ten_idx | jrdb_sed_prev | 74.8% | 25.2% | 476 | REVIEW |
| jrdb_stable_rank | jrdb_kyi_basic | 75.8% | 24.2% | 476 | REVIEW |

## 全特徴量

| feature | category | non_default | default_used | n | status |
|---|---|---|---|---|---|
| jrdb_paddock_idx | jrdb_tyb | 0.0% | 100.0% | 476 | LOW |
| jrdb_odds_idx | jrdb_tyb | 0.0% | 100.0% | 476 | LOW |
| jrdb_live_composite_idx | jrdb_tyb | 0.0% | 100.0% | 476 | LOW |
| jrdb_body_code | jrdb_tyb | 0.0% | 100.0% | 476 | LOW |
| jrdb_demeanor_code | jrdb_tyb | 0.0% | 100.0% | 476 | LOW |
| jrdb_prev_interference | jrdb_sed_prev | 0.6% | 99.4% | 476 | LOW |
| jrdb_rise_code | jrdb_kyi_basic | 1.9% | 98.1% | 476 | LOW |
| jrdb_prev_rise_code | jrdb_sed_prev | 1.9% | 98.1% | 476 | LOW |
| jrdb_stable_eval | jrdb_kyi_basic | 10.3% | 89.7% | 476 | LOW |
| jrdb_prev_late_start | jrdb_sed_prev | 12.6% | 87.4% | 476 | LOW |
| jrdb_training_arrow | jrdb_kyi_basic | 18.3% | 81.7% | 476 | LOW |
| jrdb_ranch_rank | jrdb_kyi_basic | 92.2% | 7.8% | 476 | OK |
| jrdb_hoof_code | jrdb_kyi_basic | 93.7% | 6.3% | 476 | OK |
| jrdb_class_code | jrdb_kyi_basic | 94.1% | 5.9% | 476 | OK |
| jrdb_running_style | jrdb_kyi_basic | 94.1% | 5.9% | 476 | OK |
| jrdb_idm | jrdb_kyi_basic | 98.9% | 1.1% | 476 | OK |
| jrdb_training_idx | jrdb_kyi_basic | 100.0% | 0.0% | 476 | OK |
| jrdb_stable_idx | jrdb_kyi_basic | 100.0% | 0.0% | 476 | OK |
| jrdb_info_idx | jrdb_kyi_basic | 100.0% | 0.0% | 476 | OK |
| jrdb_composite_idx | jrdb_kyi_basic | 100.0% | 0.0% | 476 | OK |
| jrdb_upset_idx | jrdb_kyi_basic | 100.0% | 0.0% | 476 | OK |
| jrdb_ten_idx_pred | jrdb_kyi_basic | 100.0% | 0.0% | 476 | OK |
| jrdb_pace_idx_pred | jrdb_kyi_basic | 100.0% | 0.0% | 476 | OK |
| jrdb_agari_idx_pred | jrdb_kyi_basic | 100.0% | 0.0% | 476 | OK |
| jrdb_position_idx_pred | jrdb_kyi_basic | 100.0% | 0.0% | 476 | OK |
| jrdb_entry_race_num | jrdb_kyi_basic | 100.0% | 0.0% | 476 | OK |
| jrdb_upset_rank | jrdb_kyi_basic | 100.0% | 0.0% | 476 | OK |
| jrdb_ls_rank | jrdb_kyi_basic | 100.0% | 0.0% | 476 | OK |
| jrdb_oikiri_idx | jrdb_cha | 100.0% | 0.0% | 476 | OK |
| jrdb_ten_time_idx | jrdb_cha | 100.0% | 0.0% | 476 | OK |
| jrdb_shimai_time_idx | jrdb_cha | 100.0% | 0.0% | 476 | OK |
| jrdb_cid_idx | jrdb_jo | 100.0% | 0.0% | 476 | OK |
| jrdb_ls_idx | jrdb_jo | 100.0% | 0.0% | 476 | OK |
| jrdb_ze_idm_avg | jrdb_ze | 100.0% | 0.0% | 476 | OK |
| jrdb_ze_ten_avg | jrdb_ze | 100.0% | 0.0% | 476 | OK |
| jrdb_ze_agari_avg | jrdb_ze | 100.0% | 0.0% | 476 | OK |
| jrdb_ze_furi_count | jrdb_ze | 100.0% | 0.0% | 476 | OK |
| jrdb_turf_baba_code | jrdb_kab_sr | 100.0% | 0.0% | 476 | OK |
| jrdb_dirt_baba_code | jrdb_kab_sr | 100.0% | 0.0% | 476 | OK |
| jrdb_kta_idm | jrdb_kta | 100.0% | 0.0% | 476 | OK |
| jrdb_kta_ten_pred | jrdb_kta | 100.0% | 0.0% | 476 | OK |
| jrdb_kta_agari_pred | jrdb_kta | 100.0% | 0.0% | 476 | OK |
| jrdb_tb_homestr_inner | jrdb_kab_sr | 100.0% | 0.0% | 476 | OK |
| jrdb_dam_rensho_avg | jrdb_blood | 100.0% | 0.0% | 476 | OK |
| jrdb_bms_rensho_avg | jrdb_blood | 100.0% | 0.0% | 476 | OK |
| jrdb_heavy_apt_skb | jrdb_skb | 100.0% | 0.0% | 476 | OK |
| jrdb_anshin | jrdb_skb | 100.0% | 0.0% | 476 | OK |
| jrdb_run_stage | jrdb_skb | 100.0% | 0.0% | 476 | OK |
| jrdb_entry_days_ago | jrdb_kyi_basic | 53.6% | 46.4% | 476 | REVIEW |
| jrdb_heavy_apt | jrdb_kyi_basic | 57.1% | 42.9% | 476 | REVIEW |
| jrdb_dist_apt | jrdb_kyi_basic | 71.6% | 28.4% | 476 | REVIEW |
| jrdb_prev_idm | jrdb_sed_prev | 73.1% | 26.9% | 476 | REVIEW |
| jrdb_prev_track_bias | jrdb_sed_prev | 73.5% | 26.5% | 476 | REVIEW |
| jrdb_prev_agari_idx | jrdb_sed_prev | 74.4% | 25.6% | 476 | REVIEW |
| jrdb_prev_pace_idx | jrdb_sed_prev | 74.4% | 25.6% | 476 | REVIEW |
| jrdb_prev_ten_idx | jrdb_sed_prev | 74.8% | 25.2% | 476 | REVIEW |
| jrdb_stable_rank | jrdb_kyi_basic | 75.8% | 24.2% | 476 | REVIEW |