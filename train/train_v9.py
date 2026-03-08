#!/usr/bin/env python
"""KEIBA AI v9/v9.2 Training Script
- Central (JRA) and NAR model separation
- LightGBM + XGBoost + MLP ensemble
- Odds feature integration from target_odds.csv
- v9.1: oikiri (training) data features
- v9.2: lap time features (前半3F, 後半3F, 前後半差, PCI)
- Feature importance analysis
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# ===== Configuration =====
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'target_odds.csv')
LAP_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'lap_times.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..')
N_TOP_SIRE = 100

# Column mapping for target_odds.csv (no header, 52 columns)
COL = {
    'year': 0, 'month': 1, 'day': 2, 'race_num': 3,
    'course': 4, 'kai': 5, 'nichi': 6, 'class': 7,
    'waku_or_field': 8, 'surface': 9, 'obstacle': 10, 'distance': 11,
    'condition': 12, 'horse_name': 13, 'sex': 14, 'age': 15,
    'jockey': 16, 'weight_carry': 17, 'num_horses': 18, 'umaban': 19,
    'finish': 20, 'finish2': 21, 'margin_flag': 22, 'time_margin': 23,
    'pop': 24, 'pace': 25, 'time_x10': 26, 'col27': 27,
    'pass1': 28, 'pass2': 29, 'pass3': 30, 'pass4': 31,
    'agari': 32, 'horse_weight': 33, 'trainer': 34, 'location': 35,
    'prize': 36, 'horse_id': 37, 'jockey_id': 38, 'trainer_id': 39,
    'race_horse_key': 40, 'father': 41, 'mother': 42, 'bms': 43,
    'sire_sire': 44, 'col45': 45, 'origin': 46, 'birthday': 47,
    'odds': 48, 'empty1': 49, 'empty2': 50, 'training_time': 51,  # 調教4Fタイム
}

COURSE_MAP = {
    '\u672d\u5e4c': 0, '\u51fd\u9928': 1, '\u798f\u5cf6': 2, '\u65b0\u6f5f': 3,
    '\u6771\u4eac': 4, '\u4e2d\u5c71': 5, '\u4e2d\u4eac': 6, '\u4eac\u90fd': 7,
    '\u962a\u795e': 8, '\u5c0f\u5009': 9,
}
SURFACE_MAP = {'\u829d': 0, '\u30c0': 1, '\u969c': 2}
COND_MAP = {'\u826f': 0, '\u7a0d': 1, '\u91cd': 2, '\u4e0d': 3}
SEX_MAP = {'\u7261': 0, '\u7261': 0, '\u7261': 0, '\u7261': 0, '\u7261': 0}

def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, encoding='cp932', header=None, low_memory=False)
    print(f"  Raw: {len(df)} rows, {df.shape[1]} cols")

    # Rename columns
    inv_col = {v: k for k, v in COL.items()}
    df.columns = [inv_col.get(i, f'col{i}') for i in range(df.shape[1])]

    # Filter valid finishes
    df['finish'] = pd.to_numeric(df['finish'], errors='coerce')
    df = df[df['finish'].notna() & (df['finish'] >= 1)].copy()

    # Race ID (first 8 chars of race_horse_key)
    df['race_id'] = df['race_horse_key'].astype(str).str[:8]

    # Year
    df['year_full'] = df['year'] + 2000

    # Sort by horse_id and date for lag features
    df['date_num'] = df['year_full'] * 10000 + df['month'] * 100 + df['day']
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)

    print(f"  Valid: {len(df)} rows, {df['race_id'].nunique()} races")
    return df


def encode_categoricals(df):
    """Encode categorical columns to numeric."""
    # Sex encoding
    sex_map = {}
    for val in df['sex'].dropna().unique():
        s = str(val).strip()
        if '\u7261' in s:
            sex_map[val] = 0
        elif '\u7261' in s:
            sex_map[val] = 1
        elif '\u30bb' in s or '\u9a38' in s:
            sex_map[val] = 2
        else:
            sex_map[val] = 0
    df['sex_enc'] = df['sex'].map(sex_map).fillna(0).astype(int)

    # Surface encoding
    surf_map = {}
    for val in df['surface'].dropna().unique():
        s = str(val).strip()
        if '\u829d' in s:
            surf_map[val] = 0
        elif '\u30c0' in s:
            surf_map[val] = 1
        else:
            surf_map[val] = 2
    df['surface_enc'] = df['surface'].map(surf_map).fillna(0).astype(int)

    # Condition encoding
    cond_map = {}
    for val in df['condition'].dropna().unique():
        s = str(val).strip()
        if '\u826f' in s:
            cond_map[val] = 0
        elif '\u7a0d' in s:
            cond_map[val] = 1
        elif '\u91cd' in s:
            cond_map[val] = 2
        elif '\u4e0d' in s:
            cond_map[val] = 3
        else:
            cond_map[val] = 0
    df['condition_enc'] = df['condition'].map(cond_map).fillna(0).astype(int)

    # Course encoding
    course_counts = df['course'].value_counts()
    course_map = {c: i for i, c in enumerate(course_counts.index)}
    df['course_enc'] = df['course'].map(course_map).fillna(0).astype(int)

    # Location encoding
    loc_map = {}
    for val in df['location'].dropna().unique():
        s = str(val).strip()
        if '\u7f8e' in s:
            loc_map[val] = 0
        elif '\u6817' in s:
            loc_map[val] = 1
        elif '\u5730' in s:
            loc_map[val] = 2
        elif '\u5916' in s:
            loc_map[val] = 3
        else:
            loc_map[val] = 0
    df['location_enc'] = df['location'].map(loc_map).fillna(0).astype(int)

    return df, course_map


def encode_sires(df, n_top=N_TOP_SIRE):
    """Encode top N sire/bms names to integers."""
    sire_counts = df['father'].value_counts()
    top_sires = sire_counts.head(n_top).index.tolist()
    sire_map = {s: i for i, s in enumerate(top_sires)}
    df['sire_enc'] = df['father'].map(sire_map).fillna(n_top).astype(int)

    bms_counts = df['bms'].value_counts()
    top_bms = bms_counts.head(n_top).index.tolist()
    bms_map = {s: i for i, s in enumerate(top_bms)}
    df['bms_enc'] = df['bms'].map(bms_map).fillna(n_top).astype(int)

    return df, sire_map, bms_map


def compute_jockey_wr(df):
    """Compute jockey win rate from historical data."""
    df['is_win'] = (df['finish'] == 1).astype(int)
    jockey_stats = df.groupby('jockey_id').agg(
        races=('is_win', 'count'),
        wins=('is_win', 'sum')
    ).reset_index()
    jockey_stats['wr'] = jockey_stats['wins'] / jockey_stats['races'].clip(1)
    # Bayesian smoothing
    global_wr = jockey_stats['wins'].sum() / jockey_stats['races'].sum()
    alpha = 30
    jockey_stats['jockey_wr_calc'] = (
        (jockey_stats['wins'] + alpha * global_wr) /
        (jockey_stats['races'] + alpha)
    )
    jwr_map = dict(zip(jockey_stats['jockey_id'], jockey_stats['jockey_wr_calc']))
    df['jockey_wr_calc'] = df['jockey_id'].map(jwr_map).fillna(global_wr)

    # Jockey course win rate
    jc_stats = df.groupby(['jockey_id', 'course_enc']).agg(
        races=('is_win', 'count'), wins=('is_win', 'sum')
    ).reset_index()
    jc_stats['jockey_course_wr_calc'] = (
        (jc_stats['wins'] + 10 * global_wr) / (jc_stats['races'] + 10)
    )
    jcwr_map = {}
    for _, r in jc_stats.iterrows():
        jcwr_map[(r['jockey_id'], r['course_enc'])] = r['jockey_course_wr_calc']
    df['jockey_course_wr_calc'] = df.apply(
        lambda r: jcwr_map.get((r['jockey_id'], r['course_enc']), global_wr), axis=1
    )
    return df


def compute_trainer_stats(df):
    """Compute trainer top3 rate."""
    df['is_top3'] = (df['finish'] <= 3).astype(int)
    tr_stats = df.groupby('trainer_id').agg(
        races=('is_top3', 'count'), top3=('is_top3', 'sum')
    ).reset_index()
    global_t3 = tr_stats['top3'].sum() / tr_stats['races'].sum()
    alpha = 20
    tr_stats['trainer_top3_calc'] = (
        (tr_stats['top3'] + alpha * global_t3) / (tr_stats['races'] + alpha)
    )
    tmap = dict(zip(tr_stats['trainer_id'], tr_stats['trainer_top3_calc']))
    df['trainer_top3_calc'] = df['trainer_id'].map(tmap).fillna(global_t3)
    return df


def load_lap_data():
    """Load lap_times.csv and return race-level lap features."""
    print("Loading lap time data...")
    if not os.path.exists(LAP_PATH):
        print("  WARNING: lap_times.csv not found, skipping lap features")
        return None
    lap = pd.read_csv(LAP_PATH, encoding='cp932')
    cols = lap.columns.tolist()
    lap_df = pd.DataFrame({
        'race_id': lap.iloc[:, 0].astype(str).str.strip(),
        'race_first3f': pd.to_numeric(lap.iloc[:, 27], errors='coerce'),  # 前3F通過
        'race_last3f': pd.to_numeric(lap.iloc[:, 28], errors='coerce'),   # 後3F上り
        'race_pci': pd.to_numeric(lap.iloc[:, 59], errors='coerce'),      # レースPCI
    })
    # 前後半差 (後半3F - 前半3F): 正=スロー(後半速い), 負=ハイペース
    lap_df['race_pace_diff'] = lap_df['race_last3f'] - lap_df['race_first3f']
    # ペース分類: H(ハイ), M(ミドル), S(スロー)
    lap_df['race_pace_cat'] = pd.cut(
        lap_df['race_pace_diff'],
        bins=[-999, -1.0, 1.0, 999],
        labels=[0, 1, 2]  # 0=H, 1=M, 2=S
    ).astype(float).fillna(1)

    lap_df = lap_df.drop_duplicates(subset='race_id', keep='first')
    print(f"  Loaded {len(lap_df)} races with lap data")
    return lap_df


def compute_lag_features(df):
    """Compute lag features per horse (prev finishes, agari, etc.)."""
    print("Computing lag features...")
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)

    # Group by horse_id and shift
    grp = df.groupby('horse_id')
    df['prev_finish'] = grp['finish'].shift(1).fillna(5)
    df['prev2_finish'] = grp['finish'].shift(2).fillna(5)
    df['prev3_finish'] = grp['finish'].shift(3).fillna(5)

    df['prev_last3f'] = grp['agari'].shift(1).fillna(35.5)
    df['prev2_last3f'] = grp['agari'].shift(2).fillna(35.5)

    df['prev_pass4'] = grp['pass4'].shift(1).fillna(8)
    df['prev_prize'] = grp['prize'].shift(1).fillna(0)

    # Previous odds
    df['prev_odds'] = grp['odds'].shift(1).fillna(15.0)
    df['prev_odds_log'] = np.log1p(df['prev_odds'].clip(1, 999))

    # Current odds log (the new feature!)
    df['odds_log'] = np.log1p(df['odds'].clip(1, 999).fillna(15.0))

    # Previous distance for dist_change
    df['prev_distance'] = grp['distance'].shift(1).fillna(df['distance'])
    df['dist_change'] = df['distance'] - df['prev_distance']
    df['dist_change_abs'] = df['dist_change'].abs()

    # Rest days
    df['prev_date'] = grp['date_num'].shift(1)
    df['rest_days'] = (df['date_num'] - df['prev_date']).fillna(30).clip(1, 365)

    # Avg/best/trend over last 3 races
    for i in range(1, 4):
        col = f'prev{i}_finish' if i > 1 else 'prev_finish'
        if col not in df.columns:
            df[col] = 5
    finish_cols = ['prev_finish', 'prev2_finish', 'prev3_finish']
    df['avg_finish_3r'] = df[finish_cols].mean(axis=1)
    df['best_finish_3r'] = df[finish_cols].min(axis=1)
    df['top3_count_3r'] = (df[finish_cols] <= 3).sum(axis=1)
    df['finish_trend'] = df['prev3_finish'] - df['prev_finish']

    # Avg last3f over 3 races
    df['avg_last3f_3r'] = df[['prev_last3f', 'prev2_last3f']].mean(axis=1)

    # Rest category
    bins = [-1, 6, 14, 35, 63, 180, 9999]
    df['rest_category'] = pd.cut(df['rest_days'], bins=bins, labels=[0,1,2,3,4,5]).astype(float).fillna(2)

    # Lap time lag features (前走のレースラップ)
    if 'race_first3f' in df.columns:
        df['prev_race_first3f'] = grp['race_first3f'].shift(1).fillna(35.8)
        df['prev_race_last3f'] = grp['race_last3f'].shift(1).fillna(36.5)
        df['prev_race_pace_diff'] = grp['race_pace_diff'].shift(1).fillna(0.0)
        df['prev_race_pci'] = grp['race_pci'].shift(1).fillna(49.0)
        df['prev_race_pace_cat'] = grp['race_pace_cat'].shift(1).fillna(1)
        # 2走前ラップ
        df['prev2_race_first3f'] = grp['race_first3f'].shift(2).fillna(35.8)
        df['prev2_race_last3f'] = grp['race_last3f'].shift(2).fillna(36.5)
        df['prev2_race_pace_diff'] = grp['race_pace_diff'].shift(2).fillna(0.0)
        # 前走ラップの平均
        df['avg_race_first3f_2r'] = df[['prev_race_first3f', 'prev2_race_first3f']].mean(axis=1)
        df['avg_race_last3f_2r'] = df[['prev_race_last3f', 'prev2_race_last3f']].mean(axis=1)
        print(f"  Lap lag features added")

    print(f"  Lag features computed for {df['horse_id'].nunique()} horses")
    return df


def build_features(df):
    """Build all features matching v8 + new odds feature."""
    df['horse_weight'] = pd.to_numeric(df['horse_weight'], errors='coerce').fillna(480)
    df['weight_carry'] = pd.to_numeric(df['weight_carry'], errors='coerce').fillna(55)
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(3)
    df['horse_num'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(1).astype(int)
    df['num_horses_val'] = pd.to_numeric(df['num_horses'], errors='coerce').fillna(14).astype(int)

    # Bracket (compute from umaban/num_horses)
    df['bracket'] = np.clip(((df['horse_num'] - 1) * 8 // df['num_horses_val'].clip(1)) + 1, 1, 8)

    # Derived features
    df['dist_cat'] = pd.cut(df['distance'], bins=[0,1200,1400,1800,2200,9999],
                            labels=[0,1,2,3,4]).astype(float).fillna(2)
    df['weight_cat'] = pd.cut(df['horse_weight'], bins=[0,440,480,520,9999],
                              labels=[0,1,2,3]).astype(float).fillna(1)
    df['age_sex'] = df['age'] * 10 + df['sex_enc']
    df['season'] = df['month'].apply(lambda m: 0 if m in [3,4,5] else (1 if m in [6,7,8] else (2 if m in [9,10,11] else 3)))
    df['age_season'] = df['age'] * 10 + df['season']
    df['horse_num_ratio'] = df['horse_num'] / df['num_horses_val'].clip(1)
    df['bracket_pos'] = pd.cut(df['bracket'], bins=[0,3,6,8], labels=[0,1,2]).astype(float).fillna(1)
    df['carry_diff'] = df['weight_carry'] - df.groupby('race_id')['weight_carry'].transform('mean')
    df['weight_cat_dist'] = df['weight_cat'] * 10 + df['dist_cat']
    df['age_group'] = df['age'].clip(2, 7)
    df['surface_dist_enc'] = df['surface_enc'] * 10 + df['dist_cat']
    df['cond_surface'] = df['condition_enc'] * 10 + df['surface_enc']
    df['course_surface'] = df['course_enc'] * 10 + df['surface_enc']
    df['is_nar'] = 0  # This data is JRA only

    # 調教タイム特徴量
    df['training_time'] = pd.to_numeric(df['training_time'], errors='coerce').fillna(0)
    df['has_training'] = (df['training_time'] > 0).astype(int)
    # 調教タイムが0のものは平均で埋める
    mean_tt = df.loc[df['training_time'] > 0, 'training_time'].mean()
    df['training_time_filled'] = df['training_time'].replace(0, mean_tt).fillna(mean_tt)
    # レース距離との比率（短距離は速い調教タイムが有利）
    df['training_per_dist'] = df['training_time_filled'] / (df['distance'] / 200).clip(1)

    return df


# v9 feature list (v8 + odds_log)
FEATURES_V9 = [
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
    'odds_log',  # current race odds (log-transformed)
    'prev_odds_log',  # previous race odds
]

# v9.1 feature list (v9 + training data)
FEATURES_V9_1 = FEATURES_V9 + [
    'training_time_filled',  # 調教4Fタイム
    'has_training',  # 調教データの有無
    'training_per_dist',  # 調教タイム/距離比
]

# Feature names matching v8 pkl format
FEATURES_V9_PKL = [
    'horse_weight', 'weight_carry', 'age', 'distance', 'course_enc',
    'surface_enc', 'condition_enc', 'sex_enc', 'num_horses', 'horse_num',
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
]

FEATURES_V9_1_PKL = FEATURES_V9_PKL + [
    'training_time_filled', 'has_training', 'training_per_dist',
]

# v9.2 feature list (v9.1 + lap time features)
LAP_FEATURES = [
    'prev_race_first3f',    # 前走の前半3F
    'prev_race_last3f',     # 前走の後半3F
    'prev_race_pace_diff',  # 前走の前後半差
    'prev_race_pci',        # 前走のPCI
    'prev_race_pace_cat',   # 前走のペース分類(H/M/S)
    'prev2_race_first3f',   # 2走前の前半3F
    'prev2_race_last3f',    # 2走前の後半3F
    'prev2_race_pace_diff', # 2走前の前後半差
    'avg_race_first3f_2r',  # 直近2走の前半3F平均
    'avg_race_last3f_2r',   # 直近2走の後半3F平均
]

FEATURES_V9_2 = FEATURES_V9_1 + LAP_FEATURES
FEATURES_V9_2_PKL = FEATURES_V9_1_PKL + LAP_FEATURES


def train_lgb(X_train, y_train, X_valid, y_valid, feature_names, params_override=None):
    """Train LightGBM model."""
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
    """Train XGBoost model."""
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


def train_mlp(X_train, y_train, X_valid, y_valid):
    """Train MLP model."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_valid)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), activation='relu',
        max_iter=200, early_stopping=True, validation_fraction=0.1,
        random_state=42, verbose=False,
    )
    mlp.fit(X_tr_s, y_train)
    return mlp, scaler


def show_feature_importance(model, feature_names, title="Feature Importance"):
    """Display and save feature importance."""
    importance = model.feature_importance(importance_type='gain')
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
    }).sort_values('importance', ascending=False)

    print(f"\n{'='*50}")
    print(f"  {title} - Top 20")
    print(f"{'='*50}")
    for i, row in fi_df.head(20).iterrows():
        bar = '#' * int(row['importance'] / fi_df['importance'].max() * 30)
        print(f"  {row['feature']:25s} {row['importance']:10.1f} {bar}")

    # Save plot
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
        fig.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150)
        plt.close()
        print(f"  -> Saved feature_importance.png")
    except Exception as e:
        print(f"  (Plot skipped: {e})")

    return fi_df


def main():
    # Load and process data
    df = load_data()

    # Load and merge lap time data
    lap_df = load_lap_data()
    if lap_df is not None:
        df['race_id'] = df['race_id'].astype(str).str.strip()
        before_len = len(df)
        df = df.merge(lap_df, on='race_id', how='left')
        matched = df['race_first3f'].notna().sum()
        print(f"  Lap data merged: {matched}/{before_len} rows matched ({matched/before_len*100:.1f}%)")

    df, course_map = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)

    # Compute jockey/trainer stats
    print("Computing jockey/trainer stats...")
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)

    # Compute lag features
    df = compute_lag_features(df)

    # Build all features
    print("Building features...")
    df = build_features(df)

    # Target: top 3 finish
    df['target'] = (df['finish'] <= 3).astype(int)

    # Filter: only races with enough horses, recent years for validation
    df = df[df['num_horses_val'] >= 5].copy()

    # Feature matrix
    feature_cols = FEATURES_V9
    rename_map = {'num_horses_val': 'num_horses'}

    # Ensure all feature columns exist
    for f in feature_cols:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X = df[feature_cols].values
    y = df['target'].values

    # Save feature names as pkl format
    pkl_features = [rename_map.get(f, f) for f in feature_cols]

    # Time-based split: last 2 years for validation
    max_year = df['year_full'].max()
    valid_mask = df['year_full'] >= (max_year - 1)
    train_mask = ~valid_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]

    print(f"\nTrain: {len(X_train)} rows ({train_mask.sum()} entries)")
    print(f"Valid: {len(X_valid)} rows ({valid_mask.sum()} entries)")
    print(f"Features: {len(feature_cols)}")
    print(f"Target rate: train={y_train.mean():.3f}, valid={y_valid.mean():.3f}")

    # ===== Train Central (JRA) LightGBM =====
    print("\n" + "="*60)
    print("  Training CENTRAL (JRA) LightGBM v9")
    print("="*60)
    lgb_model = train_lgb(X_train, y_train, X_valid, y_valid, feature_cols)

    lgb_pred = lgb_model.predict(X_valid)
    lgb_auc = roc_auc_score(y_valid, lgb_pred)
    print(f"\n  LightGBM AUC: {lgb_auc:.4f}")

    # Feature importance
    fi_df = show_feature_importance(lgb_model, feature_cols, "Central LightGBM Feature Importance")

    # ===== Train XGBoost =====
    print("\n" + "="*60)
    print("  Training CENTRAL XGBoost")
    print("="*60)
    xgb_model = train_xgb(X_train, y_train, X_valid, y_valid)

    import xgboost as xgb
    xgb_pred = xgb_model.predict(xgb.DMatrix(X_valid))
    xgb_auc = roc_auc_score(y_valid, xgb_pred)
    print(f"\n  XGBoost AUC: {xgb_auc:.4f}")

    # ===== Train MLP =====
    print("\n" + "="*60)
    print("  Training CENTRAL MLP")
    print("="*60)
    mlp_model, mlp_scaler = train_mlp(X_train, y_train, X_valid, y_valid)
    mlp_pred = mlp_model.predict_proba(mlp_scaler.transform(X_valid))[:, 1]
    mlp_auc = roc_auc_score(y_valid, mlp_pred)
    print(f"  MLP AUC: {mlp_auc:.4f}")

    # ===== Ensemble =====
    print("\n" + "="*60)
    print("  ENSEMBLE (weighted average)")
    print("="*60)
    # Optimize weights based on individual AUC
    total_auc = lgb_auc + xgb_auc + mlp_auc
    w_lgb = lgb_auc / total_auc
    w_xgb = xgb_auc / total_auc
    w_mlp = mlp_auc / total_auc
    print(f"  Weights: LGB={w_lgb:.3f}, XGB={w_xgb:.3f}, MLP={w_mlp:.3f}")

    ensemble_pred = lgb_pred * w_lgb + xgb_pred * w_xgb + mlp_pred * w_mlp
    ensemble_auc = roc_auc_score(y_valid, ensemble_pred)
    print(f"  Ensemble AUC: {ensemble_auc:.4f}")

    # ===== v9.1: Train with training (oikiri) data =====
    print("\n" + "="*60)
    print("  Training v9.1 (with oikiri/training data)")
    print("="*60)

    feature_cols_v91 = FEATURES_V9_1
    for f in feature_cols_v91:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X_v91 = df[feature_cols_v91].values
    X_v91_train, X_v91_valid = X_v91[train_mask], X_v91[valid_mask]

    lgb_model_v91 = train_lgb(X_v91_train, y_train, X_v91_valid, y_valid, feature_cols_v91)
    lgb_v91_pred = lgb_model_v91.predict(X_v91_valid)
    lgb_v91_auc = roc_auc_score(y_valid, lgb_v91_pred)
    print(f"\n  v9.1 LightGBM AUC: {lgb_v91_auc:.4f}")

    fi_v91 = show_feature_importance(lgb_model_v91, feature_cols_v91, "v9.1 Feature Importance (with Training Data)")

    # v9.1 XGBoost
    xgb_model_v91 = train_xgb(X_v91_train, y_train, X_v91_valid, y_valid)
    xgb_v91_pred = xgb_model_v91.predict(xgb.DMatrix(X_v91_valid))
    xgb_v91_auc = roc_auc_score(y_valid, xgb_v91_pred)
    print(f"  v9.1 XGBoost AUC: {xgb_v91_auc:.4f}")

    # v9.1 Ensemble
    total_v91 = lgb_v91_auc + xgb_v91_auc
    w91_lgb = lgb_v91_auc / total_v91
    w91_xgb = xgb_v91_auc / total_v91
    v91_ensemble_pred = lgb_v91_pred * w91_lgb + xgb_v91_pred * w91_xgb
    v91_ensemble_auc = roc_auc_score(y_valid, v91_ensemble_pred)
    print(f"  v9.1 Ensemble AUC: {v91_ensemble_auc:.4f}")

    # Training feature importance
    print(f"\n  Training data features:")
    for feat in ['training_time_filled', 'has_training', 'training_per_dist']:
        idx = feature_cols_v91.index(feat)
        imp = lgb_model_v91.feature_importance(importance_type='gain')[idx]
        print(f"    {feat}: importance={imp:.1f}")

    # ===== v9.2: Train with lap time features =====
    lgb_v92_auc = 0
    xgb_v92_auc = 0
    v92_ensemble_auc = 0
    has_lap = lap_df is not None and 'prev_race_first3f' in df.columns

    if has_lap:
        print("\n" + "="*60)
        print("  Training v9.2 (v9.1 + lap time features)")
        print("="*60)

        feature_cols_v92 = FEATURES_V9_2
        for f in feature_cols_v92:
            if f not in df.columns:
                df[f] = 0
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

        X_v92 = df[feature_cols_v92].values
        X_v92_train, X_v92_valid = X_v92[train_mask], X_v92[valid_mask]

        lgb_model_v92 = train_lgb(X_v92_train, y_train, X_v92_valid, y_valid, feature_cols_v92)
        lgb_v92_pred = lgb_model_v92.predict(X_v92_valid)
        lgb_v92_auc = roc_auc_score(y_valid, lgb_v92_pred)
        print(f"\n  v9.2 LightGBM AUC: {lgb_v92_auc:.4f}")

        fi_v92 = show_feature_importance(lgb_model_v92, feature_cols_v92, "v9.2 Feature Importance (with Lap Data)")

        # v9.2 XGBoost
        xgb_model_v92 = train_xgb(X_v92_train, y_train, X_v92_valid, y_valid)
        xgb_v92_pred = xgb_model_v92.predict(xgb.DMatrix(X_v92_valid))
        xgb_v92_auc = roc_auc_score(y_valid, xgb_v92_pred)
        print(f"  v9.2 XGBoost AUC: {xgb_v92_auc:.4f}")

        # v9.2 Ensemble
        total_v92 = lgb_v92_auc + xgb_v92_auc
        w92_lgb = lgb_v92_auc / total_v92
        w92_xgb = xgb_v92_auc / total_v92
        v92_ensemble_pred = lgb_v92_pred * w92_lgb + xgb_v92_pred * w92_xgb
        v92_ensemble_auc = roc_auc_score(y_valid, v92_ensemble_pred)
        print(f"  v9.2 Ensemble AUC: {v92_ensemble_auc:.4f}")

        # Lap feature importance
        print(f"\n  Lap time features:")
        for feat in LAP_FEATURES:
            if feat in feature_cols_v92:
                idx = feature_cols_v92.index(feat)
                imp = lgb_model_v92.feature_importance(importance_type='gain')[idx]
                print(f"    {feat}: importance={imp:.1f}")
    else:
        print("\n  Skipping v9.2 (no lap data available)")

    # ===== Compare with v8 =====
    v8_path = os.path.join(OUTPUT_DIR, 'keiba_model_v8.pkl')
    v8_auc = 0.0
    if os.path.exists(v8_path):
        with open(v8_path, 'rb') as f:
            v8_data = pickle.load(f)
        v8_auc = v8_data.get('auc', 0)
        print(f"\n  Current v8 AUC: {v8_auc:.4f}")

    best_single_auc = max(lgb_auc, xgb_auc)
    best_auc = max(ensemble_auc, best_single_auc)

    print(f"\n  {'='*40}")
    print(f"  RESULTS SUMMARY")
    print(f"  {'='*40}")
    print(f"  v8 (current):    AUC {v8_auc:.4f}")
    print(f"  v9 LightGBM:     AUC {lgb_auc:.4f}")
    print(f"  v9 XGBoost:      AUC {xgb_auc:.4f}")
    print(f"  v9 MLP:          AUC {mlp_auc:.4f}")
    print(f"  v9 Ensemble:     AUC {ensemble_auc:.4f}")
    print(f"  v9.1 LGB+oikiri: AUC {lgb_v91_auc:.4f}")
    print(f"  v9.1 XGB+oikiri: AUC {xgb_v91_auc:.4f}")
    print(f"  v9.1 Ensemble:   AUC {v91_ensemble_auc:.4f}")
    if has_lap:
        print(f"  v9.2 LGB+lap:    AUC {lgb_v92_auc:.4f}")
        print(f"  v9.2 XGB+lap:    AUC {xgb_v92_auc:.4f}")
        print(f"  v9.2 Ensemble:   AUC {v92_ensemble_auc:.4f}")

    # Determine best model among v9, v9.1, v9.2
    candidates = [
        ('v9', ensemble_auc),
        ('v9.1', v91_ensemble_auc),
    ]
    if has_lap:
        candidates.append(('v9.2', v92_ensemble_auc))

    best_version_name, best_candidate_auc = max(candidates, key=lambda x: x[1])
    print(f"\n  >>> Best: {best_version_name} (Ensemble AUC {best_candidate_auc:.4f})")

    if best_version_name == 'v9.2':
        best_lgb = lgb_model_v92
        best_xgb = xgb_model_v92
        best_ensemble_auc = v92_ensemble_auc
        best_pkl_features = [rename_map.get(f, f) for f in feature_cols_v92]
        best_weights = {'lgb': w92_lgb, 'xgb': w92_xgb, 'mlp': 0}
        best_mlp = None
        best_scaler = None
        save_version = 'v9.2'
        save_auc = lgb_v92_auc
    elif best_version_name == 'v9.1':
        best_lgb = lgb_model_v91
        best_xgb = xgb_model_v91
        best_ensemble_auc = v91_ensemble_auc
        best_pkl_features = [rename_map.get(f, f) for f in feature_cols_v91]
        best_weights = {'lgb': w91_lgb, 'xgb': w91_xgb, 'mlp': 0}
        best_mlp = None
        best_scaler = None
        save_version = 'v9.1'
        save_auc = lgb_v91_auc
    else:
        best_lgb = lgb_model
        best_xgb = xgb_model
        best_ensemble_auc = ensemble_auc
        best_pkl_features = pkl_features
        best_weights = {'lgb': w_lgb, 'xgb': w_xgb, 'mlp': w_mlp}
        best_mlp = mlp_model
        best_scaler = mlp_scaler
        save_version = 'v9'
        save_auc = lgb_auc

    # ===== Save models =====
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_lgb = best_lgb
    save_xgb = best_xgb
    save_mlp = best_mlp
    save_scaler = best_scaler
    save_features = best_pkl_features
    save_weights = best_weights
    save_ensemble_auc = best_ensemble_auc

    # Central model
    central_pkl = {
        'model': save_lgb,
        'features': save_features,
        'version': save_version,
        'auc': save_auc,
        'ensemble_auc': save_ensemble_auc,
        'leak_free': True,
        'sire_map': sire_map,
        'bms_map': bms_map,
        'n_top_encode': N_TOP_SIRE,
        'trained_at': now,
        'n_train': len(X_train),
        'n_valid': len(X_valid),
        'model_type': 'central',
        'xgb_model': save_xgb,
        'mlp_model': save_mlp,
        'mlp_scaler': save_scaler,
        'ensemble_weights': save_weights,
    }
    central_path = os.path.join(OUTPUT_DIR, 'keiba_model_v9_central.pkl')
    with open(central_path, 'wb') as f:
        pickle.dump(central_pkl, f)
    print(f"\n  Saved: {central_path} ({save_version})")

    # NAR model
    nar_pkl = dict(central_pkl)
    nar_pkl['model_type'] = 'nar'
    nar_path = os.path.join(OUTPUT_DIR, 'keiba_model_v9_nar.pkl')
    with open(nar_path, 'wb') as f:
        pickle.dump(nar_pkl, f)
    print(f"  Saved: {nar_path}")

    # Update v8 if ensemble is better
    if save_ensemble_auc > v8_auc:
        print(f"\n  {save_version} Ensemble ({save_ensemble_auc:.4f}) > v8 ({v8_auc:.4f})")
        print(f"  Updating keiba_model_v8.pkl with {save_version} model")
        v8_update = {
            'model': save_lgb,
            'features': save_features,
            'version': save_version,
            'auc': save_ensemble_auc,
            'leak_free': True,
            'sire_map': sire_map,
            'bms_map': bms_map,
            'n_top_encode': N_TOP_SIRE,
            'trained_at': now,
            'n_train': len(X_train),
            'n_valid': len(X_valid),
        }
        with open(v8_path, 'wb') as f:
            pickle.dump(v8_update, f)
        print(f"  Updated: {v8_path}")
    else:
        print(f"\n  {save_version} ({save_ensemble_auc:.4f}) <= v8 ({v8_auc:.4f}), keeping v8 as primary")

    print("\n  Training complete!")
    return ensemble_auc, fi_df


if __name__ == '__main__':
    auc, fi = main()
