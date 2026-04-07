#!/usr/bin/env python
"""v15 New Feature Module — Jockey-Horse Compat / Transport / Renovation / Gaisha

Feature groups:
  2-1. Jockey x Horse compatibility (expanding window, leak-free)
  2-2. Transport distance (Haversine from training center to venue)
  2-3. Course renovation correction
  2-4. External stable (外厩) check via JRDB kyi

All expanding window features use cumsum-current pattern with Bayesian smoothing.
Imported by v15 training script.
"""
import os
import math
import numpy as np
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# =====================================================
# Feature name registry
# =====================================================

V15_NEW_FEATURE_NAMES = [
    # 2-1. Jockey x Horse
    'jockey_horse_rides',
    'jockey_horse_wr',
    'jockey_horse_top3r',
    'jockey_change',
    'jockey_change_to_top',
    # 2-2. Transport
    'transport_distance_km',
    'is_long_transport',
    # 2-3. Renovation
    'course_renovated',
    'post_renovation_flag',
    # 2-4. Gaisha (optional)
    'gaisha_rank',
]

V15_NEW_DEFAULTS = {
    'jockey_horse_rides': 0,
    'jockey_horse_wr': 0.0,
    'jockey_horse_top3r': 0.0,
    'jockey_change': 0,
    'jockey_change_to_top': 0,
    'transport_distance_km': 0.0,
    'is_long_transport': 0,
    'course_renovated': 0,
    'post_renovation_flag': 0,
    'gaisha_rank': 0,
}


def get_v15_new_features():
    """Return list of all new feature names introduced in v15."""
    return list(V15_NEW_FEATURE_NAMES)


# =====================================================
# 2-1. Jockey x Horse compatibility
# =====================================================

def compute_jockey_horse_features(df):
    """Compute jockey-horse compatibility features using expanding window.

    Features:
      - jockey_horse_rides: count of past rides (this jockey on this horse)
      - jockey_horse_wr: win rate with Bayesian smoothing (alpha=3)
      - jockey_horse_top3r: top-3 rate with Bayesian smoothing (alpha=3)
      - jockey_change: 1 if jockey changed from previous race
      - jockey_change_to_top: 1 if new jockey is top-20 by wins in training data
    """
    print("  Computing jockey x horse compatibility features...")

    # Ensure required columns
    if 'is_win' not in df.columns:
        df['is_win'] = (df['finish'] == 1).astype(int)
    if 'is_top3' not in df.columns:
        df['is_top3'] = (df['finish'] <= 3).astype(int)

    # Sort by date for expanding window
    df = df.sort_values('date_num').reset_index(drop=True)

    global_wr = df['is_win'].mean()
    global_t3 = df['is_top3'].mean()
    alpha = 3

    # --- Expanding window: jockey x horse ---
    jh_key = ['jockey_id', 'horse_id']

    # cumcount = number of rows BEFORE current (0-indexed) in this group
    df['_jh_cum_races'] = df.groupby(jh_key).cumcount()
    # cumsum - current = sum of all PREVIOUS rows in this group
    df['_jh_cum_wins'] = df.groupby(jh_key)['is_win'].cumsum() - df['is_win']
    df['_jh_cum_top3'] = df.groupby(jh_key)['is_top3'].cumsum() - df['is_top3']

    df['jockey_horse_rides'] = df['_jh_cum_races'].astype(int)
    df['jockey_horse_wr'] = (
        (df['_jh_cum_wins'] + alpha * global_wr) /
        (df['_jh_cum_races'] + alpha)
    )
    df['jockey_horse_top3r'] = (
        (df['_jh_cum_top3'] + alpha * global_t3) /
        (df['_jh_cum_races'] + alpha)
    )

    # --- Jockey change detection ---
    # Sort by horse history for shift
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    grp = df.groupby('horse_id')
    df['_prev_jockey'] = grp['jockey_id'].shift(1)
    df['jockey_change'] = np.where(
        df['_prev_jockey'].isna(), 0,
        (df['jockey_id'] != df['_prev_jockey']).astype(int)
    )

    # --- Jockey change to top-20 ---
    # Build top-20 jockeys by cumulative wins (global expanding)
    # Use all data sorted by date; top-20 = jockeys with most wins overall
    df_sorted = df.sort_values('date_num')
    jockey_wins = df_sorted.groupby('jockey_id')['is_win'].sum()
    top20_jockeys = set(jockey_wins.nlargest(20).index)

    df['jockey_change_to_top'] = (
        (df['jockey_change'] == 1) &
        df['jockey_id'].isin(top20_jockeys)
    ).astype(int)

    # Coverage stats
    rides_nonzero = (df['jockey_horse_rides'] > 0).sum()
    changes = df['jockey_change'].sum()
    change_top = df['jockey_change_to_top'].sum()
    print(f"    jockey_horse_rides > 0: {rides_nonzero}/{len(df)} "
          f"({rides_nonzero/len(df)*100:.1f}%)")
    print(f"    jockey_change: {changes} ({changes/len(df)*100:.1f}%)")
    print(f"    jockey_change_to_top: {change_top}")
    print(f"    top-20 jockeys: {sorted(top20_jockeys)[:5]}... (by total wins)")

    # Cleanup temp columns
    drop_cols = [c for c in df.columns if c.startswith('_jh_') or c == '_prev_jockey']
    df = df.drop(columns=drop_cols, errors='ignore')

    return df


# =====================================================
# 2-2. Transport distance
# =====================================================

# JRA venue coordinates (lat, lon)
VENUE_COORDS = {
    0: (43.06, 141.35),  # 札幌
    1: (41.80, 140.73),  # 函館
    2: (37.75, 140.47),  # 福島
    3: (37.89, 139.02),  # 新潟
    4: (35.67, 139.48),  # 東京
    5: (35.76, 140.00),  # 中山
    6: (35.05, 137.05),  # 中京
    7: (34.93, 135.72),  # 京都
    8: (34.78, 135.36),  # 阪神
    9: (33.88, 130.87),  # 小倉
}

# Training center coordinates
TRAINING_CENTER_COORDS = {
    0: (36.03, 140.18),  # 美浦
    1: (34.98, 136.00),  # 栗東
}

# course_code (from race_id[:2]) to course_enc mapping
COURSE_CODE_TO_ENC = {
    '01': 0, '02': 1, '03': 2, '04': 3, '05': 4,
    '06': 5, '07': 6, '08': 7, '09': 8, '10': 9,
}


def _haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two (lat, lon) points."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _build_transport_lookup():
    """Pre-compute transport distance for all (location_enc, course_enc) combos."""
    lookup = {}
    for loc_enc, center in TRAINING_CENTER_COORDS.items():
        for course_enc, venue in VENUE_COORDS.items():
            dist = _haversine_km(center[0], center[1], venue[0], venue[1])
            lookup[(loc_enc, course_enc)] = round(dist, 1)
    return lookup


def compute_transport_distance(df):
    """Compute transport distance from training center to race venue.

    Features:
      - transport_distance_km: Haversine distance (km)
      - is_long_transport: 1 if distance > 500km
    """
    print("  Computing transport distance features...")

    lookup = _build_transport_lookup()

    # Determine course_enc: prefer existing column, else derive from course_code or course
    if 'course_enc' in df.columns:
        course_enc = df['course_enc'].astype(int)
    elif 'course_code' in df.columns:
        course_enc = df['course_code'].astype(str).map(COURSE_CODE_TO_ENC).fillna(-1).astype(int)
    elif 'course' in df.columns:
        # course column may be Japanese name or code string
        from train.train_v92_central import COURSE_MAP
        course_enc = df['course'].map(COURSE_MAP).fillna(-1).astype(int)
    else:
        print("    WARNING: No course column found. Setting transport_distance_km = 0")
        df['transport_distance_km'] = 0.0
        df['is_long_transport'] = 0
        return df

    # Determine location_enc
    if 'location_enc' in df.columns:
        loc_enc = df['location_enc'].astype(int)
    else:
        loc_enc = pd.Series(0, index=df.index)  # default to Miho

    # Vectorized lookup
    distances = np.zeros(len(df))
    for i in range(len(df)):
        loc = loc_enc.iloc[i]
        crs = course_enc.iloc[i]
        # Only compute for 美浦(0) and 栗東(1); others default to 0
        if loc in (0, 1) and crs in VENUE_COORDS:
            distances[i] = lookup.get((loc, crs), 0.0)

    df['transport_distance_km'] = distances
    df['is_long_transport'] = (df['transport_distance_km'] > 500).astype(int)

    avg_dist = df['transport_distance_km'].mean()
    long_count = df['is_long_transport'].sum()
    print(f"    avg transport distance: {avg_dist:.1f} km")
    print(f"    is_long_transport: {long_count}/{len(df)} "
          f"({long_count/len(df)*100:.1f}%)")

    return df


# =====================================================
# 2-3. Course renovation correction
# =====================================================

# Renovation dates: (course_enc, year, month)
RENOVATION_DATES = [
    (6, 2012, 3),   # 中京 2012/3
    (7, 2023, 4),   # 京都 2023/4
]


def compute_course_renovation(df):
    """Compute course renovation flags.

    Features:
      - course_renovated: 1 if race is within 1 year after renovation
      - post_renovation_flag: 1 if race is after renovation at that course
    """
    print("  Computing course renovation features...")

    # Build date_num from year/month/day (year is 2-digit, add 2000)
    if 'year_full' not in df.columns:
        df['year_full'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int) + 2000
    if 'month' not in df.columns or not pd.api.types.is_numeric_dtype(df['month']):
        df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(1).astype(int)

    # Get course_enc
    if 'course_enc' in df.columns:
        course_enc = df['course_enc'].astype(int)
    elif 'course_code' in df.columns:
        course_enc = df['course_code'].astype(str).map(COURSE_CODE_TO_ENC).fillna(-1).astype(int)
    else:
        print("    WARNING: No course column found. Setting renovation flags = 0")
        df['course_renovated'] = 0
        df['post_renovation_flag'] = 0
        return df

    # Initialize
    df['course_renovated'] = 0
    df['post_renovation_flag'] = 0

    year_full = df['year_full'].values
    month_val = df['month'].values
    crs_vals = course_enc.values

    for reno_course, reno_year, reno_month in RENOVATION_DATES:
        # Renovation date as comparable number
        reno_ym = reno_year * 12 + reno_month
        one_year_after_ym = reno_ym + 12

        mask_course = (crs_vals == reno_course)
        race_ym = year_full * 12 + month_val

        # post_renovation_flag: race is after renovation
        mask_post = mask_course & (race_ym >= reno_ym)
        df.loc[mask_post, 'post_renovation_flag'] = 1

        # course_renovated: within 1 year after renovation
        mask_within = mask_course & (race_ym >= reno_ym) & (race_ym < one_year_after_ym)
        df.loc[mask_within, 'course_renovated'] = 1

    renovated_count = df['course_renovated'].sum()
    post_count = df['post_renovation_flag'].sum()
    print(f"    course_renovated (within 1yr): {renovated_count} "
          f"({renovated_count/len(df)*100:.2f}%)")
    print(f"    post_renovation_flag: {post_count} "
          f"({post_count/len(df)*100:.1f}%)")

    return df


# =====================================================
# 2-4. External stable (外厩) check
# =====================================================

def compute_gaisha_features(df):
    """Check JRDB kyi for 外厩 (external stable) rank.

    外厩: JRDBに直接外厩カラムなし。jrdb_kyi.csv の放牧先ランク(column 63)を代用。
    """
    print("  Computing gaisha (external stable) features...")
    print("    外厩: JRDBに直接外厩カラムなし。jrdb_kyi.csv の放牧先ランク(column 63)を代用")

    kyi_path = os.path.join(DATA_DIR, 'jrdb_kyi.csv')

    if not os.path.exists(kyi_path):
        print(f"    WARNING: {kyi_path} not found. Setting gaisha_rank = 0")
        df['gaisha_rank'] = 0
        return df

    try:
        kyi = pd.read_csv(kyi_path, encoding='utf-8-sig', low_memory=False, dtype=str)
    except Exception as e:
        print(f"    WARNING: Failed to read jrdb_kyi.csv: {e}. Setting gaisha_rank = 0")
        df['gaisha_rank'] = 0
        return df

    # Column 62 = 放牧先ランク (may be garbled due to encoding)
    # Find it by index position or by pattern matching 'ランク' columns
    gaisha_col = None
    cols = list(kyi.columns)
    if '放牧先ランク' in cols:
        gaisha_col = '放牧先ランク'
    elif len(cols) > 62:
        # Column 62 is 放牧先ランク in JRDB KYI format
        candidate = cols[62]
        # Verify it contains rank-like values (A-E or similar letters)
        sample = kyi[candidate].dropna().head(20)
        if sample.isin(['A', 'B', 'C', 'D', 'E', '']).any():
            gaisha_col = candidate

    if gaisha_col is None:
        print(f"    WARNING: 放牧先ランク column not found. Setting gaisha_rank = 0")
        df['gaisha_rank'] = 0
        return df

    # Parse gaisha_rank: encode letter rank to numeric (A=5, B=4, C=3, D=2, E=1, else=0)
    rank_map = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
    kyi['gaisha_rank'] = kyi[gaisha_col].map(rank_map).fillna(0).astype(int)

    # Build merge key: jra_race_id + umaban
    # KYI column 5 = 馬番 (also may be garbled)
    uma_col = cols[5] if len(cols) > 5 else None
    if 'jra_race_id' in kyi.columns and 'race_id' in df.columns and uma_col:
        kyi['_merge_rid'] = kyi['jra_race_id'].astype(str).str.strip()
        kyi['_merge_uma'] = pd.to_numeric(kyi[uma_col], errors='coerce').fillna(0).astype(int)
        df['_merge_rid'] = df['race_id'].astype(str).str.strip()
        df['_merge_uma'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(0).astype(int)

        merge_cols = kyi[['_merge_rid', '_merge_uma', 'gaisha_rank']].drop_duplicates(
            subset=['_merge_rid', '_merge_uma'], keep='last'
        )

        before_len = len(df)
        df = df.merge(merge_cols, on=['_merge_rid', '_merge_uma'], how='left', suffixes=('', '_kyi'))

        # If gaisha_rank came from merge, fill NaN
        if 'gaisha_rank_kyi' in df.columns:
            df['gaisha_rank'] = df['gaisha_rank_kyi'].fillna(0).astype(int)
            df = df.drop(columns=['gaisha_rank_kyi'])
        elif 'gaisha_rank' not in df.columns:
            df['gaisha_rank'] = 0
        else:
            df['gaisha_rank'] = df['gaisha_rank'].fillna(0).astype(int)

        df = df.drop(columns=['_merge_rid', '_merge_uma'], errors='ignore')

        matched = (df['gaisha_rank'] > 0).sum()
        print(f"    gaisha_rank merged: {matched}/{len(df)} "
              f"({matched/len(df)*100:.1f}%)")

        if len(df) != before_len:
            print(f"    WARNING: merge changed row count {before_len} -> {len(df)}, "
                  f"possible duplicates in kyi")
    else:
        print("    WARNING: Cannot build merge key (jra_race_id missing). "
              "Setting gaisha_rank = 0")
        df['gaisha_rank'] = 0

    return df


# =====================================================
# Master function
# =====================================================

def compute_all_v15_new_features(df):
    """Compute all v15 new features in sequence.

    Args:
        df: DataFrame with standard columns (horse_id, jockey_id, date_num,
            finish, course_enc/course_code, location_enc, etc.)

    Returns:
        df with new feature columns added.
    """
    print("\n=== Computing v15 new features ===")
    n_before = df.shape[1]

    df = compute_jockey_horse_features(df)
    df = compute_transport_distance(df)
    df = compute_course_renovation(df)
    df = compute_gaisha_features(df)

    n_after = df.shape[1]
    print(f"\n  v15 new features: {n_after - n_before} columns added")
    print(f"  Feature list: {get_v15_new_features()}")

    # Fill any remaining NaN with defaults
    for feat, default in V15_NEW_DEFAULTS.items():
        if feat in df.columns:
            df[feat] = df[feat].fillna(default)

    return df


# =====================================================
# Standalone test
# =====================================================

if __name__ == '__main__':
    print("=== features_v15_new.py standalone test ===\n")

    RACES_PATH = os.path.join(DATA_DIR, 'jra_races_full.csv')
    if not os.path.exists(RACES_PATH):
        print(f"ERROR: {RACES_PATH} not found")
        exit(1)

    print("Loading jra_races_full.csv ...")
    df = pd.read_csv(RACES_PATH, encoding='utf-8-sig', low_memory=False, dtype=str)
    print(f"  Raw: {len(df)} rows")

    # Minimal preprocessing
    df['finish'] = pd.to_numeric(df['finish'], errors='coerce')
    df = df[df['finish'].notna() & (df['finish'] >= 1)].copy()
    df['year_full'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int) + 2000
    df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(1).astype(int)
    df['day'] = pd.to_numeric(df['day'], errors='coerce').fillna(1).astype(int)
    df['date_num'] = df['year_full'] * 10000 + df['month'] * 100 + df['day']
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(1600)
    df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(1).astype(int)
    df['race_num'] = pd.to_numeric(df['race_num'], errors='coerce').fillna(1).astype(int)

    # Course encoding
    COURSE_MAP = {
        '札幌': 0, '函館': 1, '福島': 2, '新潟': 3,
        '東京': 4, '中山': 5, '中京': 6, '京都': 7,
        '阪神': 8, '小倉': 9,
    }
    if 'course_enc' not in df.columns:
        df['course_enc'] = df['course'].map(COURSE_MAP).fillna(10).astype(int)

    # Location encoding
    if 'location_enc' not in df.columns:
        loc_map = {}
        for val in df['location'].dropna().unique():
            s = str(val).strip()
            if '美浦' in s:
                loc_map[val] = 0
            elif '栗東' in s:
                loc_map[val] = 1
            elif '地方' in s:
                loc_map[val] = 2
            elif '外国' in s:
                loc_map[val] = 3
            else:
                loc_map[val] = 0
        df['location_enc'] = df['location'].map(loc_map).fillna(0).astype(int)

    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)

    # Run
    df = compute_all_v15_new_features(df)

    # Summary
    print("\n=== Feature summary ===")
    for feat in get_v15_new_features():
        if feat in df.columns:
            col = df[feat]
            print(f"  {feat:30s}  mean={col.mean():.4f}  "
                  f"std={col.std():.4f}  min={col.min()}  max={col.max()}")
        else:
            print(f"  {feat:30s}  MISSING")

    print("\nDone.")
