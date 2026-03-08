#!/usr/bin/env python
"""KEIBA AI v9.2 Central Training Script
- Uses jra_races_full.csv (with correct column headers)
- FIX: Column mapping bug (father/bms were mapped to owner/breeder in v9/v9.1)
- NEW: Sakaro/Wood training time features
- NEW: Horse career features (expanding window, leak-free)
- NEW: Sire/BMS performance features
- LightGBM + XGBoost ensemble
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# ===== Configuration =====
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RACES_PATH = os.path.join(DATA_DIR, 'jra_races_full.csv')
TRAINING_TIMES_PATH = os.path.join(DATA_DIR, 'training_times.csv')
LAP_PATH = os.path.join(DATA_DIR, 'lap_times.csv')
OUTPUT_DIR = BASE_DIR
N_TOP_SIRE = 100

COURSE_MAP = {
    '札幌': 0, '函館': 1, '福島': 2, '新潟': 3,
    '東京': 4, '中山': 5, '中京': 6, '京都': 7,
    '阪神': 8, '小倉': 9,
}
SURFACE_MAP = {'芝': 0, 'ダ': 1, '障': 2}
COND_MAP = {'良': 0, '稍': 1, '重': 2, '不': 3}


def load_data():
    """Load jra_races_full.csv (with headers, utf-8-sig)."""
    print("Loading data...")
    df = pd.read_csv(RACES_PATH, encoding='utf-8-sig', low_memory=False, dtype=str)
    print(f"  Raw: {len(df)} rows, {df.shape[1]} cols")
    print(f"  Columns: {list(df.columns[:10])}...")

    # Filter valid finishes
    df['finish'] = pd.to_numeric(df['finish'], errors='coerce')
    df = df[df['finish'].notna() & (df['finish'] >= 1)].copy()

    # Numeric conversions
    df['year_full'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int) + 2000
    df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(1).astype(int)
    df['day'] = pd.to_numeric(df['day'], errors='coerce').fillna(1).astype(int)
    df['date_num'] = df['year_full'] * 10000 + df['month'] * 100 + df['day']
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(1600)
    df['num_horses'] = pd.to_numeric(df['num_horses'], errors='coerce').fillna(14).astype(int)
    df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(1).astype(int)
    df['agari_3f'] = pd.to_numeric(df['agari_3f'], errors='coerce')
    df['prize'] = pd.to_numeric(df['prize'], errors='coerce').fillna(0)
    df['horse_weight'] = pd.to_numeric(df['horse_weight'], errors='coerce').fillna(480)
    df['weight_carry'] = pd.to_numeric(df['weight_carry'], errors='coerce').fillna(55)
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(3)
    df['tansho_odds'] = pd.to_numeric(df['tansho_odds'], errors='coerce').fillna(15.0)
    df['training_4f'] = pd.to_numeric(df['training_4f'], errors='coerce').fillna(0)

    # Race ID
    df['race_id_str'] = df['race_id'].astype(str).str.strip().str[:8]

    # Sort by horse_id and date for lag features
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)

    print(f"  Valid: {len(df)} rows, {df['race_id_str'].nunique()} races")
    print(f"  Year range: {df['year_full'].min()}-{df['year_full'].max()}")
    print(f"  Father sample: {df['father'].dropna().head(5).tolist()}")
    print(f"  BMS sample: {df['bms'].dropna().head(5).tolist()}")
    return df


def encode_categoricals(df):
    """Encode categorical columns."""
    # Sex
    sex_map = {}
    for val in df['sex'].dropna().unique():
        s = str(val).strip()
        if '牡' in s:
            sex_map[val] = 0
        elif '牝' in s:
            sex_map[val] = 1
        elif 'セ' in s or '騸' in s:
            sex_map[val] = 2
        else:
            sex_map[val] = 0
    df['sex_enc'] = df['sex'].map(sex_map).fillna(0).astype(int)

    # Surface
    surf_map = {}
    for val in df['surface'].dropna().unique():
        s = str(val).strip()
        if '芝' in s:
            surf_map[val] = 0
        elif 'ダ' in s:
            surf_map[val] = 1
        else:
            surf_map[val] = 2
    df['surface_enc'] = df['surface'].map(surf_map).fillna(0).astype(int)

    # Condition
    cond_map = {}
    for val in df['condition'].dropna().unique():
        s = str(val).strip()
        if '良' in s:
            cond_map[val] = 0
        elif '稍' in s:
            cond_map[val] = 1
        elif '重' in s:
            cond_map[val] = 2
        elif '不' in s:
            cond_map[val] = 3
        else:
            cond_map[val] = 0
    df['condition_enc'] = df['condition'].map(cond_map).fillna(0).astype(int)

    # Course
    df['course_enc'] = df['course'].map(COURSE_MAP).fillna(len(COURSE_MAP)).astype(int)

    # Location
    loc_map = {}
    for val in df['location'].dropna().unique():
        s = str(val).strip()
        if '美' in s:
            loc_map[val] = 0
        elif '栗' in s:
            loc_map[val] = 1
        elif '地' in s:
            loc_map[val] = 2
        elif '外' in s:
            loc_map[val] = 3
        else:
            loc_map[val] = 0
    df['location_enc'] = df['location'].map(loc_map).fillna(0).astype(int)

    return df


def encode_sires(df, n_top=N_TOP_SIRE):
    """Encode top N sire/bms names. NOW USES CORRECT COLUMNS."""
    # father column is now CORRECTLY mapped to actual sire name
    sire_counts = df['father'].value_counts()
    top_sires = sire_counts.head(n_top).index.tolist()
    sire_map = {s: i for i, s in enumerate(top_sires)}
    df['sire_enc'] = df['father'].map(sire_map).fillna(n_top).astype(int)
    print(f"  Top 5 sires: {top_sires[:5]}")

    bms_counts = df['bms'].value_counts()
    top_bms = bms_counts.head(n_top).index.tolist()
    bms_map = {s: i for i, s in enumerate(top_bms)}
    df['bms_enc'] = df['bms'].map(bms_map).fillna(n_top).astype(int)
    print(f"  Top 5 BMS: {top_bms[:5]}")

    return df, sire_map, bms_map


def load_training_times():
    """Load sakaro/wood training times for merge."""
    if not os.path.exists(TRAINING_TIMES_PATH):
        print("  WARNING: training_times.csv not found")
        return None

    print("Loading training times...")
    tt = pd.read_csv(TRAINING_TIMES_PATH, encoding='utf-8-sig', dtype=str)
    print(f"  {len(tt)} training records")

    # Parse numeric fields
    tt['time_4f'] = pd.to_numeric(tt['time_4f'], errors='coerce')
    tt['time_3f'] = pd.to_numeric(tt['time_3f'], errors='coerce')
    tt['date'] = pd.to_numeric(tt['date'], errors='coerce').fillna(0).astype(int)

    # Remove invalid
    tt = tt[tt['time_4f'].notna() & (tt['time_4f'] > 30) & (tt['time_4f'] < 80)]

    # For wood: has horse_id
    wood = tt[tt['training_type'] == 'wood'].copy()
    sakaro = tt[tt['training_type'] == 'sakaro'].copy()

    print(f"  Wood: {len(wood)} valid, Sakaro: {len(sakaro)} valid")
    return {'wood': wood, 'sakaro': sakaro}


def merge_training_features(df, tt_data):
    """Merge best training times before each race."""
    if tt_data is None:
        return df

    print("Merging training time features...")
    wood = tt_data['wood']
    sakaro = tt_data['sakaro']

    # For each race row, find best wood 4F time in 14 days before race
    # Wood has horse_id for reliable joining
    wood_best = {}
    if len(wood) > 0 and 'horse_id' in wood.columns:
        wood_clean = wood[wood['horse_id'].astype(str).str.strip() != ''].copy()
        wood_clean['horse_id'] = wood_clean['horse_id'].astype(str).str.strip()
        for hid, grp in wood_clean.groupby('horse_id'):
            for _, row in grp.iterrows():
                key = (str(hid), int(row['date']))
                if key not in wood_best or row['time_4f'] < wood_best[key]:
                    wood_best[key] = row['time_4f']

    # Create lookup: for each horse_id, sorted list of (date, best_4f)
    horse_wood = {}
    for (hid, date), t4f in wood_best.items():
        if hid not in horse_wood:
            horse_wood[hid] = []
        horse_wood[hid].append((date, t4f))
    for hid in horse_wood:
        horse_wood[hid].sort()

    # For each race, find best wood time in 14 days before
    def get_best_wood(horse_id, race_date):
        hid = str(horse_id).strip()
        if hid not in horse_wood:
            return np.nan
        entries = horse_wood[hid]
        best = np.nan
        for d, t in entries:
            # Training within 14 days before race
            if d >= race_date - 14 and d < race_date:
                if np.isnan(best) or t < best:
                    best = t
        return best

    # Vectorized approach: group by horse_id
    df['wood_best_4f'] = np.nan
    df['wood_count_2w'] = 0

    for hid, grp_idx in df.groupby(df['horse_id'].astype(str).str.strip()).groups.items():
        if hid not in horse_wood:
            continue
        entries = horse_wood[hid]
        if not entries:
            continue
        for idx in grp_idx:
            race_date = df.loc[idx, 'date_num']
            # Convert race_date (YYYYMMDD int) to comparable format
            # Training dates are also YYYYMMDD
            best = np.nan
            count = 0
            for d, t in entries:
                if d >= race_date - 14 and d < race_date:
                    count += 1
                    if np.isnan(best) or t < best:
                        best = t
            df.loc[idx, 'wood_best_4f'] = best
            df.loc[idx, 'wood_count_2w'] = count

    matched = df['wood_best_4f'].notna().sum()
    print(f"  Wood training matched: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")

    # Fill NaN with global mean
    wood_mean = df.loc[df['wood_best_4f'].notna(), 'wood_best_4f'].mean()
    df['wood_best_4f_filled'] = df['wood_best_4f'].fillna(wood_mean if not np.isnan(wood_mean) else 52.0)
    df['has_wood_training'] = df['wood_best_4f'].notna().astype(int)

    # === Sakaro training features ===
    print("  Merging sakaro training features...")
    horse_sakaro = {}
    if len(sakaro) > 0 and 'horse_id' in sakaro.columns:
        sak_clean = sakaro[sakaro['horse_id'].astype(str).str.strip() != ''].copy()
        sak_clean['horse_id'] = sak_clean['horse_id'].astype(str).str.strip()
        for hid, grp in sak_clean.groupby('horse_id'):
            horse_sakaro[hid] = sorted(
                [(int(r['date']), r['time_4f'], r['time_3f']) for _, r in grp.iterrows()],
                key=lambda x: x[0]
            )

    df['sakaro_best_4f'] = np.nan
    df['sakaro_best_3f'] = np.nan
    df['sakaro_count_2w'] = 0

    for hid, grp_idx in df.groupby(df['horse_id'].astype(str).str.strip()).groups.items():
        if hid not in horse_sakaro:
            continue
        entries = horse_sakaro[hid]
        for idx in grp_idx:
            race_date = df.loc[idx, 'date_num']
            best4f, best3f, count = np.nan, np.nan, 0
            for d, t4, t3 in entries:
                if d >= race_date - 14 and d < race_date:
                    count += 1
                    if np.isnan(best4f) or t4 < best4f:
                        best4f = t4
                    if not np.isnan(t3) and (np.isnan(best3f) or t3 < best3f):
                        best3f = t3
            df.loc[idx, 'sakaro_best_4f'] = best4f
            df.loc[idx, 'sakaro_best_3f'] = best3f
            df.loc[idx, 'sakaro_count_2w'] = count

    sak_matched = df['sakaro_best_4f'].notna().sum()
    print(f"  Sakaro training matched: {sak_matched}/{len(df)} ({sak_matched/len(df)*100:.1f}%)")

    sak_mean_4f = df.loc[df['sakaro_best_4f'].notna(), 'sakaro_best_4f'].mean()
    sak_mean_3f = df.loc[df['sakaro_best_3f'].notna(), 'sakaro_best_3f'].mean()
    df['sakaro_best_4f_filled'] = df['sakaro_best_4f'].fillna(sak_mean_4f if not np.isnan(sak_mean_4f) else 53.0)
    df['sakaro_best_3f_filled'] = df['sakaro_best_3f'].fillna(sak_mean_3f if not np.isnan(sak_mean_3f) else 39.0)
    df['has_sakaro_training'] = df['sakaro_best_4f'].notna().astype(int)
    df['total_training_count'] = df['wood_count_2w'] + df['sakaro_count_2w']

    return df


def compute_jockey_wr(df):
    """Compute jockey win rate (leak-free: expanding window)."""
    print("Computing jockey stats (expanding window)...")
    df['is_win'] = (df['finish'] == 1).astype(int)
    df['is_top3'] = (df['finish'] <= 3).astype(int)

    # Sort by date for expanding window
    df = df.sort_values('date_num').reset_index(drop=True)

    # Expanding cumulative stats per jockey
    df['jockey_cum_races'] = df.groupby('jockey_id').cumcount()
    df['jockey_cum_wins'] = df.groupby('jockey_id')['is_win'].cumsum() - df['is_win']
    # Subtract current race to avoid leak
    jockey_cum_races = df['jockey_cum_races']  # already excludes current (0-indexed count)

    global_wr = df['is_win'].mean()
    alpha = 30
    df['jockey_wr_calc'] = (
        (df['jockey_cum_wins'] + alpha * global_wr) /
        (jockey_cum_races + alpha)
    )

    # Jockey course win rate (expanding)
    df['jc_cum_races'] = df.groupby(['jockey_id', 'course_enc']).cumcount()
    df['jc_cum_wins'] = df.groupby(['jockey_id', 'course_enc'])['is_win'].cumsum() - df['is_win']
    df['jockey_course_wr_calc'] = (
        (df['jc_cum_wins'] + 10 * global_wr) /
        (df['jc_cum_races'] + 10)
    )

    # Jockey surface win rate (expanding)
    df['js_cum_races'] = df.groupby(['jockey_id', 'surface_enc']).cumcount()
    df['js_cum_wins'] = df.groupby(['jockey_id', 'surface_enc'])['is_win'].cumsum() - df['is_win']
    df['jockey_surface_wr'] = (
        (df['js_cum_wins'] + 10 * global_wr) /
        (df['js_cum_races'] + 10)
    )

    # Clean up temp columns
    df = df.drop(columns=['jockey_cum_races', 'jockey_cum_wins',
                          'jc_cum_races', 'jc_cum_wins',
                          'js_cum_races', 'js_cum_wins'])

    # Re-sort by horse_id for lag features
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    return df


def compute_trainer_stats(df):
    """Compute trainer top3 rate (expanding window)."""
    print("Computing trainer stats (expanding window)...")
    df = df.sort_values('date_num').reset_index(drop=True)

    global_t3 = df['is_top3'].mean()
    alpha = 20

    df['tr_cum_races'] = df.groupby('trainer_id').cumcount()
    df['tr_cum_top3'] = df.groupby('trainer_id')['is_top3'].cumsum() - df['is_top3']
    df['trainer_top3_calc'] = (
        (df['tr_cum_top3'] + alpha * global_t3) /
        (df['tr_cum_races'] + alpha)
    )

    df = df.drop(columns=['tr_cum_races', 'tr_cum_top3'])
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    return df


def compute_horse_career(df):
    """Compute horse career stats (expanding window, leak-free)."""
    print("Computing horse career features...")
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)

    grp = df.groupby('horse_id')

    # Cumulative stats (excluding current race)
    df['horse_career_races'] = grp.cumcount()  # 0-indexed = races before current
    df['horse_career_wins'] = grp['is_win'].cumsum() - df['is_win']
    df['horse_career_top3'] = grp['is_top3'].cumsum() - df['is_top3']

    alpha = 5
    global_wr = df['is_win'].mean()
    global_t3 = df['is_top3'].mean()

    df['horse_career_wr'] = (
        (df['horse_career_wins'] + alpha * global_wr) /
        (df['horse_career_races'] + alpha)
    )
    df['horse_career_top3r'] = (
        (df['horse_career_top3'] + alpha * global_t3) /
        (df['horse_career_races'] + alpha)
    )

    return df


def compute_sire_performance(df):
    """Compute sire/BMS performance by surface/distance (expanding window)."""
    print("Computing sire performance features...")
    df = df.sort_values('date_num').reset_index(drop=True)

    global_wr = df['is_win'].mean()
    alpha = 50

    # Sire surface win rate
    df['sire_surf_cum_races'] = df.groupby(['father', 'surface_enc']).cumcount()
    df['sire_surf_cum_wins'] = df.groupby(['father', 'surface_enc'])['is_win'].cumsum() - df['is_win']
    df['sire_surface_wr'] = (
        (df['sire_surf_cum_wins'] + alpha * global_wr) /
        (df['sire_surf_cum_races'] + alpha)
    )

    # Sire distance category win rate
    df['dist_cat_temp'] = pd.cut(df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
                                  labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
    df['sire_dist_cum_races'] = df.groupby(['father', 'dist_cat_temp']).cumcount()
    df['sire_dist_cum_wins'] = df.groupby(['father', 'dist_cat_temp'])['is_win'].cumsum() - df['is_win']
    df['sire_dist_wr'] = (
        (df['sire_dist_cum_wins'] + alpha * global_wr) /
        (df['sire_dist_cum_races'] + alpha)
    )

    # BMS surface win rate
    df['bms_surf_cum_races'] = df.groupby(['bms', 'surface_enc']).cumcount()
    df['bms_surf_cum_wins'] = df.groupby(['bms', 'surface_enc'])['is_win'].cumsum() - df['is_win']
    df['bms_surface_wr'] = (
        (df['bms_surf_cum_wins'] + alpha * global_wr) /
        (df['bms_surf_cum_races'] + alpha)
    )

    # Clean up
    drop_cols = [c for c in df.columns if c.endswith('_cum_races') or c.endswith('_cum_wins')]
    drop_cols.append('dist_cat_temp')
    df = df.drop(columns=drop_cols)

    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    return df


def compute_distance_aptitude(df):
    """Compute horse's distance-category aptitude (expanding window, leak-free)."""
    print("Computing distance aptitude features...")
    df = df.sort_values('date_num').reset_index(drop=True)

    df['dist_cat_apt'] = pd.cut(df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
                                 labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
    global_t3 = df['is_top3'].mean()
    alpha = 5

    df['hd_cum_races'] = df.groupby(['horse_id', 'dist_cat_apt']).cumcount()
    df['hd_cum_top3'] = df.groupby(['horse_id', 'dist_cat_apt'])['is_top3'].cumsum() - df['is_top3']
    df['horse_dist_top3r'] = (
        (df['hd_cum_top3'] + alpha * global_t3) /
        (df['hd_cum_races'] + alpha)
    )

    # Horse surface aptitude
    df['hs_cum_races'] = df.groupby(['horse_id', 'surface_enc']).cumcount()
    df['hs_cum_top3'] = df.groupby(['horse_id', 'surface_enc'])['is_top3'].cumsum() - df['is_top3']
    df['horse_surface_top3r'] = (
        (df['hs_cum_top3'] + alpha * global_t3) /
        (df['hs_cum_races'] + alpha)
    )

    drop_cols = [c for c in df.columns if c.startswith(('hd_cum_', 'hs_cum_'))]
    drop_cols.append('dist_cat_apt')
    df = df.drop(columns=drop_cols)
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    return df


def compute_frame_advantage(df):
    """Compute bracket advantage by course×distance (expanding window, leak-free)."""
    print("Computing frame advantage features...")
    df = df.sort_values('date_num').reset_index(drop=True)

    global_wr = df['is_win'].mean()
    alpha = 100

    # Bracket win rate by course × dist_cat
    df['dist_cat_frm'] = pd.cut(df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
                                 labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
    df['frame_key'] = df['course_enc'].astype(str) + '_' + df['dist_cat_frm'].astype(str) + '_' + df['bracket'].astype(str)

    df['frm_cum_races'] = df.groupby('frame_key').cumcount()
    df['frm_cum_wins'] = df.groupby('frame_key')['is_win'].cumsum() - df['is_win']
    df['frame_course_dist_wr'] = (
        (df['frm_cum_wins'] + alpha * global_wr) /
        (df['frm_cum_races'] + alpha)
    )

    drop_cols = ['dist_cat_frm', 'frame_key', 'frm_cum_races', 'frm_cum_wins']
    df = df.drop(columns=drop_cols)
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    return df


def load_lap_data():
    """Load lap_times.csv for race-level pace features."""
    if not os.path.exists(LAP_PATH):
        return None
    print("Loading lap time data...")
    lap = pd.read_csv(LAP_PATH, encoding='cp932')
    lap_df = pd.DataFrame({
        'race_id_str': lap.iloc[:, 0].astype(str).str.strip(),
        'race_first3f': pd.to_numeric(lap.iloc[:, 27], errors='coerce'),
        'race_last3f': pd.to_numeric(lap.iloc[:, 28], errors='coerce'),
        'race_pci': pd.to_numeric(lap.iloc[:, 59], errors='coerce'),
    })
    lap_df['race_pace_diff'] = lap_df['race_last3f'] - lap_df['race_first3f']
    lap_df = lap_df.drop_duplicates(subset='race_id_str', keep='first')
    print(f"  Loaded {len(lap_df)} races with lap data")
    return lap_df


def compute_lag_features(df):
    """Compute lag features per horse."""
    print("Computing lag features...")
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    grp = df.groupby('horse_id')

    df['prev_finish'] = grp['finish'].shift(1).fillna(5)
    df['prev2_finish'] = grp['finish'].shift(2).fillna(5)
    df['prev3_finish'] = grp['finish'].shift(3).fillna(5)

    df['prev_last3f'] = grp['agari_3f'].shift(1).fillna(35.5)
    df['prev2_last3f'] = grp['agari_3f'].shift(2).fillna(35.5)

    df['prev_pass4'] = pd.to_numeric(df['pass4'], errors='coerce').fillna(8)
    df['prev_pass4'] = grp['prev_pass4'].shift(1).fillna(8)
    df['prev_prize'] = grp['prize'].shift(1).fillna(0)

    # Odds
    df['prev_odds'] = grp['tansho_odds'].shift(1).fillna(15.0)
    df['prev_odds_log'] = np.log1p(df['prev_odds'].clip(1, 999))
    df['odds_log'] = np.log1p(df['tansho_odds'].clip(1, 999))

    # Distance change
    df['prev_distance'] = grp['distance'].shift(1).fillna(df['distance'])
    df['dist_change'] = df['distance'] - df['prev_distance']
    df['dist_change_abs'] = df['dist_change'].abs()

    # Rest days
    df['prev_date'] = grp['date_num'].shift(1)
    df['rest_days'] = (df['date_num'] - df['prev_date']).fillna(30).clip(1, 365)

    # Aggregates
    finish_cols = ['prev_finish', 'prev2_finish', 'prev3_finish']
    df['avg_finish_3r'] = df[finish_cols].mean(axis=1)
    df['best_finish_3r'] = df[finish_cols].min(axis=1)
    df['top3_count_3r'] = (df[finish_cols] <= 3).sum(axis=1)
    df['finish_trend'] = df['prev3_finish'] - df['prev_finish']
    df['avg_last3f_3r'] = df[['prev_last3f', 'prev2_last3f']].mean(axis=1)

    # Weight change
    df['prev_horse_weight'] = grp['horse_weight'].shift(1).fillna(df['horse_weight'])
    df['weight_change'] = df['horse_weight'] - df['prev_horse_weight']
    df['weight_change_abs'] = df['weight_change'].abs()

    # Rest category
    bins = [-1, 6, 14, 35, 63, 180, 9999]
    df['rest_category'] = pd.cut(df['rest_days'], bins=bins, labels=[0,1,2,3,4,5]).astype(float).fillna(2)

    # Lap lag features
    if 'race_first3f' in df.columns:
        df['prev_race_first3f'] = grp['race_first3f'].shift(1).fillna(35.8)
        df['prev_race_last3f'] = grp['race_last3f'].shift(1).fillna(36.5)
        df['prev_race_pace_diff'] = grp['race_pace_diff'].shift(1).fillna(0.0)

    # Horse agari relative to race pace (how much better/worse than race avg)
    df['prev_agari_relative'] = df['prev_last3f'] - df['prev_race_last3f']

    print(f"  Lag features computed for {df['horse_id'].nunique()} horses")
    return df


def build_features(df):
    """Build derived features."""
    df['horse_num'] = df['umaban']
    df['num_horses_val'] = df['num_horses']
    df['bracket'] = np.clip(((df['horse_num'] - 1) * 8 // df['num_horses_val'].clip(1)) + 1, 1, 8)

    df['dist_cat'] = pd.cut(df['distance'], bins=[0,1200,1400,1800,2200,9999],
                            labels=[0,1,2,3,4]).astype(float).fillna(2)
    df['weight_cat'] = pd.cut(df['horse_weight'], bins=[0,440,480,520,9999],
                              labels=[0,1,2,3]).astype(float).fillna(1)
    df['age_sex'] = df['age'] * 10 + df['sex_enc']
    df['season'] = df['month'].apply(lambda m: 0 if m in [3,4,5] else (1 if m in [6,7,8] else (2 if m in [9,10,11] else 3)))
    df['age_season'] = df['age'] * 10 + df['season']
    df['horse_num_ratio'] = df['horse_num'] / df['num_horses_val'].clip(1)
    df['bracket_pos'] = pd.cut(df['bracket'], bins=[0,3,6,8], labels=[0,1,2]).astype(float).fillna(1)
    df['carry_diff'] = df['weight_carry'] - df.groupby('race_id_str')['weight_carry'].transform('mean')
    df['weight_cat_dist'] = df['weight_cat'] * 10 + df['dist_cat']
    df['age_group'] = df['age'].clip(2, 7)
    df['surface_dist_enc'] = df['surface_enc'] * 10 + df['dist_cat']
    df['cond_surface'] = df['condition_enc'] * 10 + df['surface_enc']
    df['course_surface'] = df['course_enc'] * 10 + df['surface_enc']
    df['is_nar'] = 0

    # Sire cross features (now using CORRECT sire/bms encoding)
    df['sire_dist'] = df['sire_enc'] * 10 + df['dist_cat']
    df['sire_surface'] = df['sire_enc'] * 10 + df['surface_enc']
    df['bms_dist'] = df['bms_enc'] * 10 + df['dist_cat']

    # Training time (col 51)
    df['has_training'] = (df['training_4f'] > 0).astype(int)
    mean_tt = df.loc[df['training_4f'] > 0, 'training_4f'].mean()
    df['training_time_filled'] = df['training_4f'].replace(0, mean_tt).fillna(mean_tt)
    df['training_per_dist'] = df['training_time_filled'] / (df['distance'] / 200).clip(1)

    return df


# ===== Feature Lists =====
# V9.1 compatible features (for comparison)
FEATURES_V91 = [
    'horse_weight', 'weight_carry', 'age', 'distance', 'course_enc',
    'surface_enc', 'condition_enc', 'sex_enc', 'num_horses_val', 'horse_num',
    'bracket', 'jockey_wr_calc', 'jockey_course_wr_calc', 'trainer_top3_calc',
    'prev_finish', 'prev_last3f', 'prev_pass4', 'prev_prize',
    'prev2_finish', 'prev3_finish', 'avg_finish_3r', 'best_finish_3r',
    'finish_trend', 'top3_count_3r', 'avg_last3f_3r', 'prev2_last3f',
    'dist_change', 'dist_change_abs', 'rest_days', 'rest_category',
    'sire_enc', 'bms_enc', 'dist_cat', 'weight_cat', 'age_sex', 'season',
    'age_season', 'horse_num_ratio', 'bracket_pos', 'carry_diff',
    'weight_cat_dist', 'age_group', 'surface_dist_enc', 'cond_surface',
    'course_surface', 'location_enc', 'is_nar',
    'odds_log', 'prev_odds_log',
    'training_time_filled', 'has_training', 'training_per_dist',
]

# V9.2 NEW features
V92_NEW_FEATURES = [
    'jockey_surface_wr',      # Jockey surface-specific win rate
    'horse_career_races',      # Horse career races count
    'horse_career_wr',         # Horse career win rate
    'horse_career_top3r',      # Horse career top3 rate
    'sire_surface_wr',         # Sire progeny surface win rate
    'sire_dist_wr',            # Sire progeny distance win rate
    'bms_surface_wr',          # BMS progeny surface win rate
    'weight_change',           # Weight change from prev race
    'weight_change_abs',       # Absolute weight change
    'wood_best_4f_filled',     # Best wood training 4F time
    'has_wood_training',       # Has wood training data
]

FEATURES_V92 = FEATURES_V91 + V92_NEW_FEATURES

# V9.3 NEW features (all pre-day, leak-free for Pattern A)
V93_NEW_FEATURES = [
    # Pace features (from lap data, previous race)
    'prev_race_first3f',       # Previous race first 3F time
    'prev_race_last3f',        # Previous race last 3F time
    'prev_race_pace_diff',     # Previous race pace diff
    'prev_agari_relative',     # Horse's prev agari vs race pace
    # Training features (sakaro + wood count)
    'wood_count_2w',           # Wood training count in 2 weeks
    'sakaro_best_4f_filled',   # Best sakaro 4F time
    'sakaro_best_3f_filled',   # Best sakaro 3F time
    'has_sakaro_training',     # Has sakaro training
    'total_training_count',    # Total training sessions
    # Distance aptitude (expanding window)
    'horse_dist_top3r',        # Horse top3 rate at distance category
    'horse_surface_top3r',     # Horse top3 rate on surface type
    # Frame advantage (expanding window)
    'frame_course_dist_wr',    # Bracket win rate by course×distance
]

FEATURES_V93 = FEATURES_V92 + V93_NEW_FEATURES

# PKL feature names (rename num_horses_val → num_horses for compatibility)
FEATURES_V92_PKL = [f if f != 'num_horses_val' else 'num_horses' for f in FEATURES_V92]
FEATURES_V93_PKL = [f if f != 'num_horses_val' else 'num_horses' for f in FEATURES_V93]
FEATURES_V91_PKL = [f if f != 'num_horses_val' else 'num_horses' for f in FEATURES_V91]


def train_lgb(X_train, y_train, X_valid, y_valid, feature_names, params_override=None):
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
        'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1,
        'n_jobs': -1, 'seed': 42,
    }
    if params_override:
        params.update(params_override)

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    dvalid = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_names, reference=dtrain)

    model = lgb.train(
        params, dtrain, num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    return model


def train_xgb(X_train, y_train, X_valid, y_valid):
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'min_child_weight': 50,
        'reg_alpha': 0.1, 'reg_lambda': 0.1, 'seed': 42,
        'tree_method': 'hist', 'verbosity': 0,
    }

    model = xgb.train(
        params, dtrain, num_boost_round=1000,
        evals=[(dvalid, 'valid')],
        early_stopping_rounds=50, verbose_eval=100,
    )
    return model


def show_feature_importance(model, feature_names, title="Feature Importance"):
    importance = model.feature_importance(importance_type='gain')
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
    }).sort_values('importance', ascending=False)

    print(f"\n{'='*50}")
    print(f"  {title} - Top 25")
    print(f"{'='*50}")
    for _, row in fi_df.head(25).iterrows():
        bar = '#' * int(row['importance'] / fi_df['importance'].max() * 30)
        print(f"  {row['feature']:25s} {row['importance']:10.1f} {bar}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        top20 = fi_df.head(20).iloc[::-1]
        ax.barh(top20['feature'], top20['importance'], color='#f0c040')
        ax.set_xlabel('Importance (Gain)')
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_v92.png'), dpi=150)
        plt.close()
    except Exception:
        pass

    return fi_df


def main():
    # Load data
    df = load_data()

    # Load and merge lap data
    lap_df = load_lap_data()
    if lap_df is not None:
        before_len = len(df)
        df = df.merge(lap_df, on='race_id_str', how='left')
        matched = df['race_first3f'].notna().sum()
        print(f"  Lap data merged: {matched}/{before_len} ({matched/before_len*100:.1f}%)")

    # Encode categoricals
    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)

    # Load and merge training times
    tt_data = load_training_times()
    df = merge_training_features(df, tt_data)

    # Compute stats (leak-free expanding window)
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)

    # Compute lag features
    df = compute_lag_features(df)

    # Build features
    print("Building features...")
    df = build_features(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()

    # ===== V9.1 baseline (with FIXED column mapping) =====
    print("\n" + "=" * 60)
    print("  V9.1 Baseline (fixed column mapping)")
    print("=" * 60)

    feature_cols_v91 = FEATURES_V91
    for f in feature_cols_v91:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X_v91 = df[feature_cols_v91].values
    y = df['target'].values

    # Time-based split
    max_year = df['year_full'].max()
    valid_mask = df['year_full'] >= (max_year - 1)
    train_mask = ~valid_mask

    X_v91_train, X_v91_valid = X_v91[train_mask], X_v91[valid_mask]
    y_train, y_valid = y[train_mask], y[valid_mask]

    print(f"Train: {len(X_v91_train)}, Valid: {len(X_v91_valid)}")
    print(f"Target rate: train={y_train.mean():.3f}, valid={y_valid.mean():.3f}")

    lgb_v91 = train_lgb(X_v91_train, y_train, X_v91_valid, y_valid, feature_cols_v91)
    lgb_v91_pred = lgb_v91.predict(X_v91_valid)
    lgb_v91_auc = roc_auc_score(y_valid, lgb_v91_pred)
    print(f"\n  V9.1 LightGBM AUC: {lgb_v91_auc:.4f}")

    xgb_v91 = train_xgb(X_v91_train, y_train, X_v91_valid, y_valid)
    import xgboost as xgb
    xgb_v91_pred = xgb_v91.predict(xgb.DMatrix(X_v91_valid))
    xgb_v91_auc = roc_auc_score(y_valid, xgb_v91_pred)
    print(f"  V9.1 XGBoost AUC: {xgb_v91_auc:.4f}")

    total_v91 = lgb_v91_auc + xgb_v91_auc
    w91_lgb = lgb_v91_auc / total_v91
    w91_xgb = xgb_v91_auc / total_v91
    v91_pred = lgb_v91_pred * w91_lgb + xgb_v91_pred * w91_xgb
    v91_auc = roc_auc_score(y_valid, v91_pred)
    print(f"  V9.1 Ensemble AUC: {v91_auc:.4f}")

    fi_v91 = show_feature_importance(lgb_v91, feature_cols_v91, "V9.1 (Fixed Columns)")

    # ===== V9.2 with new features =====
    print("\n" + "=" * 60)
    print("  V9.2 (new features)")
    print("=" * 60)

    feature_cols_v92 = FEATURES_V92
    for f in feature_cols_v92:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X_v92 = df[feature_cols_v92].values
    X_v92_train, X_v92_valid = X_v92[train_mask], X_v92[valid_mask]

    lgb_v92 = train_lgb(X_v92_train, y_train, X_v92_valid, y_valid, feature_cols_v92)
    lgb_v92_pred = lgb_v92.predict(X_v92_valid)
    lgb_v92_auc = roc_auc_score(y_valid, lgb_v92_pred)
    print(f"\n  V9.2 LightGBM AUC: {lgb_v92_auc:.4f}")

    fi_v92 = show_feature_importance(lgb_v92, feature_cols_v92, "V9.2 Feature Importance")

    xgb_v92 = train_xgb(X_v92_train, y_train, X_v92_valid, y_valid)
    xgb_v92_pred = xgb_v92.predict(xgb.DMatrix(X_v92_valid))
    xgb_v92_auc = roc_auc_score(y_valid, xgb_v92_pred)
    print(f"  V9.2 XGBoost AUC: {xgb_v92_auc:.4f}")

    total_v92 = lgb_v92_auc + xgb_v92_auc
    w92_lgb = lgb_v92_auc / total_v92
    w92_xgb = xgb_v92_auc / total_v92
    v92_pred = lgb_v92_pred * w92_lgb + xgb_v92_pred * w92_xgb
    v92_auc = roc_auc_score(y_valid, v92_pred)
    print(f"  V9.2 Ensemble AUC: {v92_auc:.4f}")

    # New feature importance
    print(f"\n  V9.2 new features importance:")
    for feat in V92_NEW_FEATURES:
        if feat in feature_cols_v92:
            idx = feature_cols_v92.index(feat)
            imp = lgb_v92.feature_importance(importance_type='gain')[idx]
            print(f"    {feat:25s} {imp:10.1f}")

    # ===== Summary =====
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  V9.1 (old, wrong columns): AUC 0.8452 (reference)")
    print(f"  V9.1 (fixed columns):      AUC {v91_auc:.4f}")
    print(f"  V9.2 (enhanced):           AUC {v92_auc:.4f}")

    # Determine best
    best_version = 'v9.2'
    best_auc = v92_auc
    best_lgb_model = lgb_v92
    best_xgb_model = xgb_v92
    best_features = FEATURES_V92_PKL
    best_weights = {'lgb': w92_lgb, 'xgb': w92_xgb, 'mlp': 0}
    best_lgb_auc = lgb_v92_auc

    if v91_auc > v92_auc:
        best_version = 'v9.1-fixed'
        best_auc = v91_auc
        best_lgb_model = lgb_v91
        best_xgb_model = xgb_v91
        best_features = FEATURES_V91_PKL
        best_weights = {'lgb': w91_lgb, 'xgb': w91_xgb, 'mlp': 0}
        best_lgb_auc = lgb_v91_auc

    print(f"\n  >>> Best: {best_version} (Ensemble AUC {best_auc:.4f})")

    # ===== Save =====
    if best_auc > 0.8452:
        print(f"\n  {best_version} ({best_auc:.4f}) > V9.1 old (0.8452)")
        print(f"  Saving as production model!")

        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        central_pkl = {
            'model': best_lgb_model,
            'features': best_features,
            'version': best_version,
            'auc': best_lgb_auc,
            'ensemble_auc': best_auc,
            'leak_free': True,
            'sire_map': sire_map,
            'bms_map': bms_map,
            'n_top_encode': N_TOP_SIRE,
            'trained_at': now,
            'n_train': int(train_mask.sum()),
            'n_valid': int(valid_mask.sum()),
            'model_type': 'central',
            'xgb_model': best_xgb_model,
            'mlp_model': None,
            'mlp_scaler': None,
            'ensemble_weights': best_weights,
            'course_map': dict(COURSE_MAP),
        }

        central_path = os.path.join(OUTPUT_DIR, 'keiba_model_v9_central.pkl')
        with open(central_path, 'wb') as f:
            pickle.dump(central_pkl, f)
        print(f"  Saved: {central_path}")

        # Also save as v8 backup
        v8_path = os.path.join(OUTPUT_DIR, 'keiba_model_v8.pkl')
        v8_pkl = dict(central_pkl)
        v8_pkl['auc'] = best_auc
        with open(v8_path, 'wb') as f:
            pickle.dump(v8_pkl, f)
        print(f"  Saved: {v8_path}")
    else:
        print(f"\n  {best_version} ({best_auc:.4f}) <= V9.1 old (0.8452)")
        print(f"  Keeping current production model.")

        # Save V9.2 model separately for reference
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        v92_pkl = {
            'model': best_lgb_model,
            'features': best_features,
            'version': best_version,
            'auc': best_lgb_auc,
            'ensemble_auc': best_auc,
            'leak_free': True,
            'sire_map': sire_map,
            'bms_map': bms_map,
            'n_top_encode': N_TOP_SIRE,
            'trained_at': now,
            'n_train': int(train_mask.sum()),
            'n_valid': int(valid_mask.sum()),
            'model_type': 'central',
            'xgb_model': best_xgb_model,
            'ensemble_weights': best_weights,
            'course_map': dict(COURSE_MAP),
        }
        v92_path = os.path.join(OUTPUT_DIR, 'keiba_model_v92_central.pkl')
        with open(v92_path, 'wb') as f:
            pickle.dump(v92_pkl, f)
        print(f"  Saved (reference only): {v92_path}")

    print("\n  Training complete!")


if __name__ == '__main__':
    main()
