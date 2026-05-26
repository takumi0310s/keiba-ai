"""V20 feature configuration.

V15 (145 features) → V20 base (136 features) by:
  - DELETE 9 near-constant/broken features (STEP 1, immediate)
  - FIX 4 fixable features via JV-Link O1/BLOD (STEP 2, pending)
  - ADD fixed features back after JV-Link bulk fetch (STEP 3 retraining)

Also applies V20_LEAK_FEATURES (Session #38 confirmed: SKB 10 + Pattern A 8 = 18 total).
"""
from __future__ import annotations

# ===== STEP 1: Delete these 9 from V15 base =====
# All confirmed near-constant in T1_features_audit_2026_05_25.json
# lgb_gain=0 AND xgb_gain=0 (or constant value)
V20_DELETE_NOW = [
    'is_nar',              # always 0 (central only)
    'prev_odds_log',       # constant (broken prev race odds)
    'has_training',        # constant (97% = 1, zero signal)
    'pci',                 # constant 1.02
    'gaisha_rank',         # constant 0 (0% fill rate)
    # fixable via JV-Link O1 — deleted until O1 data available:
    'prev_race_first3f',   # constant 35.8 → O1 first 3f lap
    'prev_race_last3f',    # constant 36.5 → O1 last 3f lap
    'prev_race_pace_diff', # constant 0.0 → O1 pace diff
    # fixable via JV-Link BLOD:
    'sire_shinba_top3r',   # constant 0.22 → BLOD expanding window
]

# ===== STEP 2: These will be RE-ADDED after JV-Link data fetch =====
# Do NOT train V20 final without these — they are the main STEP 2 payload
V20_FIX_LATER = {
    'prev_race_first3f': {
        'source': 'JV-Link O1',
        'description': 'Previous race first 3f lap time (seconds)',
        'expected_importance': '3-5%',
    },
    'prev_race_last3f': {
        'source': 'JV-Link O1',
        'description': 'Previous race last 3f lap time (seconds)',
        'expected_importance': '3-5%',
    },
    'prev_race_pace_diff': {
        'source': 'JV-Link O1',
        'description': 'Pace differential (first3f - last3f normalized)',
        'expected_importance': '2-4%',
    },
    'sire_shinba_top3r': {
        'source': 'JV-Link BLOD',
        'description': '種牡馬 新馬 top3 rate (expanding window)',
        'expected_importance': '1-3%',
    },
}

# ===== Leak exclusion (Session #38 confirmed) =====
# Pattern A: 8 features
PATTERN_A_LEAK = {
    'odds_log', 'horse_weight', 'condition_enc',
    'weight_change', 'weight_change_abs', 'weight_cat', 'weight_cat_dist', 'cond_surface',
}
# SKB POST-RACE LEAK: 10 features (Session #38, V15.1 audit)
SKB_LEAK = {
    'skb_kishi_code_1', 'skb_kishi_code_2', 'skb_kishi_code_3',
    'skb_baba_code_1', 'skb_baba_code_2', 'skb_baba_code_3',
    'skb_kyaku_code_1', 'skb_kyaku_code_2', 'skb_kyaku_code_3',
    'skb_turf_hoof',
}
V20_LEAK_FEATURES = PATTERN_A_LEAK | SKB_LEAK  # 18 total


# ===== V20 base 136-feature list (STEP 1 only, before O1/BLOD fix) =====
# Generated from T1_features_audit_2026_05_25.json (V15 145 features minus 9 deletes)
V20_BASE_136 = [
    'weight_carry', 'age', 'distance', 'course_enc', 'surface_enc', 'sex_enc',
    'num_horses_val', 'horse_num', 'bracket', 'jockey_wr_calc',
    'jockey_course_wr_calc', 'trainer_top3_calc', 'prev_finish', 'prev_last3f',
    'prev_pass4', 'prev_prize', 'prev2_finish', 'prev3_finish', 'avg_finish_3r',
    'best_finish_3r', 'finish_trend', 'top3_count_3r', 'avg_last3f_3r',
    'prev2_last3f', 'dist_change', 'dist_change_abs', 'rest_days', 'rest_category',
    'sire_enc', 'bms_enc', 'dist_cat', 'age_sex', 'season', 'age_season',
    'horse_num_ratio', 'bracket_pos', 'carry_diff', 'age_group', 'surface_dist_enc',
    'course_surface', 'location_enc', 'training_time_filled', 'training_per_dist',
    'jockey_surface_wr', 'horse_career_races', 'horse_career_wr',
    'horse_career_top3r', 'sire_surface_wr', 'sire_dist_wr', 'bms_surface_wr',
    'wood_best_4f_filled', 'has_wood_training', 'prev_agari_relative',
    'wood_count_2w', 'sakaro_best_4f_filled', 'sakaro_best_3f_filled',
    'has_sakaro_training', 'total_training_count', 'horse_dist_top3r',
    'horse_surface_top3r', 'frame_course_dist_wr', 'index_max_filled',
    'index_run1_filled', 'index_avg5_filled', 'time_1f_last_filled',
    'training_intensity_enc', 'jrdb_idm', 'jrdb_training_idx', 'jrdb_stable_idx',
    'jrdb_composite_idx', 'jrdb_upset_idx', 'jrdb_ten_idx_pred',
    'jrdb_pace_idx_pred', 'jrdb_agari_idx_pred', 'jrdb_position_idx_pred',
    'jrdb_class_code', 'jrdb_rise_code', 'jrdb_heavy_apt', 'jrdb_hoof_code',
    'jrdb_ranch_rank', 'jrdb_stable_rank', 'jrdb_training_arrow',
    'jrdb_stable_eval', 'jrdb_running_style', 'jrdb_dist_apt', 'jrdb_prev_idm',
    'jrdb_prev_track_bias', 'jrdb_prev_interference', 'jrdb_prev_late_start',
    'jrdb_prev_pace_idx', 'jrdb_prev_rise_code', 'jrdb_oikiri_idx',
    'jrdb_ten_time_idx', 'jrdb_shimai_time_idx', 'jrdb_cid_idx', 'jrdb_ls_idx',
    'jrdb_kta_idm', 'jrdb_kta_ten_pred', 'jrdb_kta_agari_pred',
    'jrdb_ze_idm_avg', 'jrdb_ze_ten_avg', 'jrdb_ze_agari_avg',
    'jrdb_ze_furi_count', 'jrdb_tb_homestr_inner', 'jrdb_dam_rensho_avg',
    'jrdb_bms_rensho_avg', 'stable_comment_score', 'oz_tansho_base_log',
    'oz_fukusho_base_log', 'oz_base_pop_rank', 'odds_change_rate',
    'pop_rank_change', 'odds_sharp_drop', 'weight_ma5', 'weight_trend',
    'weight_peak_diff', 'paci_manken_idx', 'paci_goal_rank', 'paci_dochu_rank',
    'paci_goal_diff', 'paci_jockey_exp_wr', 'paci_jockey_exp_3rd',
    'paci_ninki_idx', 'jockey_horse_rides', 'jockey_horse_wr',
    'jockey_horse_top3r', 'jockey_change', 'jockey_change_to_top',
    'transport_distance_km', 'is_long_transport', 'course_renovated',
    'post_renovation_flag', 'paci_sogo_mark', 'paci_idm_mark',
    'paci_jockey_mark', 'paci_train_mark',
]

assert len(V20_BASE_136) == 136, f"Expected 136, got {len(V20_BASE_136)}"


def get_v20_features(
    include_o1_fixed: bool = False,
    include_blod_fixed: bool = False,
    include_sib: bool = False,
    include_blod_full: bool = False,
) -> list[str]:
    """Return V20 feature list depending on which data sources are available.

    Args:
        include_o1_fixed:   True after lap features built (netkeiba_race_lap.csv).
        include_blod_fixed: True after sire_shinba_top3r_exp built (build_blod_features.py).
        include_sib:        True after sib expanding features built (build_sib_features.py).
        include_blod_full:  True after JV-Link BLOD HN/SK built (build_blod_full_features.py).
    """
    feats = list(V20_BASE_136)
    if include_o1_fixed:
        feats += ['prev_race_first3f', 'prev_race_last3f', 'prev_race_pace_diff', 'has_lap_data']
    if include_blod_fixed:
        feats += ['sire_shinba_top3r_exp']
    if include_sib:
        feats += ['sib_top3_rate_exp', 'sib_shinba_wr_exp',
                  'sib_total_races_exp', 'sib_total_offspring_exp']
    # Always include netkeiba race-level features
    feats += ['track_index_nk']
    # PRE-RACE JRDB columns (+0.0004 on 2025 fold)
    feats += ['jrdb_prev_ten_idx', 'jrdb_prev_agari_idx']
    if include_blod_full:
        # JV-Link BLOD full: 3代血統 × 距離/コース expanding (build_blod_full_features.py)
        feats += [
            'sire_blod_top3r_exp', 'sire_blod_turf_wr_exp', 'sire_blod_dirt_wr_exp',
            'sire_blod_sprint_wr_exp', 'sire_blod_middle_wr_exp', 'sire_blod_long_wr_exp',
            'bms_blod_top3r_exp', 'bms_blod_turf_wr_exp', 'bms_blod_dirt_wr_exp',
            'bms_blod_sprint_wr_exp', 'bms_blod_middle_wr_exp', 'bms_blod_long_wr_exp',
        ]
    return feats
