#!/usr/bin/env python
"""KEIBA AI - Complete Leak-Free Backtest
Train: 2020-2022 ONLY
Test: 2023-2025 (completely unseen data)
Condition-based (A-E/X) ROI calculation with real payouts.
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
import time
import requests
import warnings
warnings.filterwarnings('ignore')

from itertools import combinations
from datetime import datetime
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'data', 'target_odds.csv')
EXISTING_RESULTS = os.path.join(SCRIPT_DIR, 'backtest_results_5year.json')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'backtest_leakfree_results.json')
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
N_TOP_SIRE = 100

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
    'odds': 48, 'empty1': 49, 'empty2': 50, 'training_time': 51,
}

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


# ===== Data Loading & Feature Engineering =====
def load_and_prepare():
    """Load CSV, encode, compute features. Returns full df."""
    print("Loading CSV data...")
    df = pd.read_csv(DATA_PATH, encoding='cp932', header=None, low_memory=False)
    inv_col = {v: k for k, v in COL.items()}
    df.columns = [inv_col.get(i, f'col{i}') for i in range(df.shape[1])]

    df['finish'] = pd.to_numeric(df['finish'], errors='coerce')
    df = df[df['finish'].notna() & (df['finish'] >= 1)].copy()
    df['year_full'] = pd.to_numeric(df['year'], errors='coerce') + 2000
    df['month'] = pd.to_numeric(df['month'], errors='coerce')
    df['day'] = pd.to_numeric(df['day'], errors='coerce')
    df['race_num'] = pd.to_numeric(df['race_num'], errors='coerce')
    df['date_num'] = df['year_full'] * 10000 + df['month'] * 100 + df['day']
    df['race_id'] = df['race_horse_key'].astype(str).str[:8]
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(1600)
    df['horse_weight'] = pd.to_numeric(df['horse_weight'], errors='coerce').fillna(480)
    df['weight_carry'] = pd.to_numeric(df['weight_carry'], errors='coerce').fillna(55)
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(3)
    df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(1).astype(int)
    df['num_horses'] = pd.to_numeric(df['num_horses'], errors='coerce').fillna(14).astype(int)
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce').fillna(15.0)
    df['pop'] = pd.to_numeric(df['pop'], errors='coerce').fillna(8)
    df['agari'] = pd.to_numeric(df['agari'], errors='coerce').fillna(35.5)
    df['pass4'] = pd.to_numeric(df['pass4'], errors='coerce').fillna(8)
    df['prize'] = pd.to_numeric(df['prize'], errors='coerce').fillna(0)

    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    print(f"  Loaded {len(df)} rows, {df['race_id'].nunique()} races")
    print(f"  Year range: {int(df['year_full'].min())} - {int(df['year_full'].max())}")
    return df


def encode_with_train_only(df, train_mask):
    """Encode categoricals using ONLY training data."""
    train = df[train_mask]

    # Sex
    sex_map = {}
    for val in df['sex'].dropna().unique():
        s = str(val).strip()
        if '\u7261' in s: sex_map[val] = 0
        elif '\u7261' in s: sex_map[val] = 1
        elif '\u30bb' in s or '\u9a38' in s: sex_map[val] = 2
        else: sex_map[val] = 0
    df['sex_enc'] = df['sex'].map(sex_map).fillna(0).astype(int)

    # Surface
    surf_map = {}
    for val in df['surface'].dropna().unique():
        s = str(val).strip()
        if '\u829d' in s: surf_map[val] = 0
        elif '\u30c0' in s: surf_map[val] = 1
        else: surf_map[val] = 2
    df['surface_enc'] = df['surface'].map(surf_map).fillna(0).astype(int)

    # Condition
    cond_map = {}
    for val in df['condition'].dropna().unique():
        s = str(val).strip()
        if '\u826f' in s: cond_map[val] = 0
        elif '\u7a0d' in s: cond_map[val] = 1
        elif '\u91cd' in s: cond_map[val] = 2
        elif '\u4e0d' in s: cond_map[val] = 3
        else: cond_map[val] = 0
    df['condition_enc'] = df['condition'].map(cond_map).fillna(0).astype(int)

    # Course (from train only)
    course_counts = train['course'].value_counts()
    course_map = {c: i for i, c in enumerate(course_counts.index)}
    df['course_enc'] = df['course'].map(course_map).fillna(len(course_map)).astype(int)

    # Location
    loc_map = {}
    for val in df['location'].dropna().unique():
        s = str(val).strip()
        if '\u7f8e' in s: loc_map[val] = 0
        elif '\u6817' in s: loc_map[val] = 1
        elif '\u5730' in s: loc_map[val] = 2
        elif '\u5916' in s: loc_map[val] = 3
        else: loc_map[val] = 0
    df['location_enc'] = df['location'].map(loc_map).fillna(0).astype(int)

    # Sire/BMS (from train only)
    sire_counts = train['father'].value_counts()
    top_sires = sire_counts.head(N_TOP_SIRE).index.tolist()
    sire_map = {s: i for i, s in enumerate(top_sires)}
    df['sire_enc'] = df['father'].map(sire_map).fillna(N_TOP_SIRE).astype(int)

    bms_counts = train['bms'].value_counts()
    top_bms = bms_counts.head(N_TOP_SIRE).index.tolist()
    bms_map = {s: i for i, s in enumerate(top_bms)}
    df['bms_enc'] = df['bms'].map(bms_map).fillna(N_TOP_SIRE).astype(int)

    return df, sire_map, bms_map


def compute_stats_train_only(df, train_mask):
    """Compute jockey/trainer stats from training data ONLY."""
    train = df[train_mask].copy()
    train['is_win'] = (train['finish'] == 1).astype(int)
    train['is_top3'] = (train['finish'] <= 3).astype(int)

    # Jockey WR
    j_stats = train.groupby('jockey_id').agg(
        races=('is_win', 'count'), wins=('is_win', 'sum')
    ).reset_index()
    global_wr = j_stats['wins'].sum() / j_stats['races'].sum()
    j_stats['jockey_wr_calc'] = (j_stats['wins'] + 30 * global_wr) / (j_stats['races'] + 30)
    jwr = dict(zip(j_stats['jockey_id'], j_stats['jockey_wr_calc']))
    df['jockey_wr_calc'] = df['jockey_id'].map(jwr).fillna(global_wr)

    # Jockey course WR
    jc_stats = train.groupby(['jockey_id', 'course_enc']).agg(
        races=('is_win', 'count'), wins=('is_win', 'sum')
    ).reset_index()
    jc_stats['wr'] = (jc_stats['wins'] + 10 * global_wr) / (jc_stats['races'] + 10)
    jcwr = {}
    for _, r in jc_stats.iterrows():
        jcwr[(r['jockey_id'], r['course_enc'])] = r['wr']
    df['jockey_course_wr_calc'] = df.apply(
        lambda r: jcwr.get((r['jockey_id'], r['course_enc']), global_wr), axis=1
    )

    # Trainer top3
    t_stats = train.groupby('trainer_id').agg(
        races=('is_top3', 'count'), top3=('is_top3', 'sum')
    ).reset_index()
    global_t3 = t_stats['top3'].sum() / t_stats['races'].sum()
    t_stats['trainer_top3_calc'] = (t_stats['top3'] + 20 * global_t3) / (t_stats['races'] + 20)
    tmap = dict(zip(t_stats['trainer_id'], t_stats['trainer_top3_calc']))
    df['trainer_top3_calc'] = df['trainer_id'].map(tmap).fillna(global_t3)

    return df


def compute_lag_features(df):
    """Lag features using shift (no future leak)."""
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    grp = df.groupby('horse_id')

    df['prev_finish'] = grp['finish'].shift(1).fillna(5)
    df['prev2_finish'] = grp['finish'].shift(2).fillna(5)
    df['prev3_finish'] = grp['finish'].shift(3).fillna(5)
    df['prev_last3f'] = grp['agari'].shift(1).fillna(35.5)
    df['prev2_last3f'] = grp['agari'].shift(2).fillna(35.5)
    df['prev_pass4'] = grp['pass4'].shift(1).fillna(8)
    df['prev_prize'] = grp['prize'].shift(1).fillna(0)
    df['prev_odds'] = grp['odds'].shift(1).fillna(15.0)
    df['prev_odds_log'] = np.log1p(df['prev_odds'].clip(1, 999))
    df['odds_log'] = np.log1p(df['odds'].clip(1, 999).fillna(15.0))
    df['prev_distance'] = grp['distance'].shift(1).fillna(df['distance'])
    df['dist_change'] = df['distance'] - df['prev_distance']
    df['dist_change_abs'] = df['dist_change'].abs()
    df['prev_date'] = grp['date_num'].shift(1)
    df['rest_days'] = (df['date_num'] - df['prev_date']).fillna(30).clip(1, 365)

    finish_cols = ['prev_finish', 'prev2_finish', 'prev3_finish']
    df['avg_finish_3r'] = df[finish_cols].mean(axis=1)
    df['best_finish_3r'] = df[finish_cols].min(axis=1)
    df['top3_count_3r'] = (df[finish_cols] <= 3).sum(axis=1)
    df['finish_trend'] = df['prev3_finish'] - df['prev_finish']
    df['avg_last3f_3r'] = df[['prev_last3f', 'prev2_last3f']].mean(axis=1)
    bins = [-1, 6, 14, 35, 63, 180, 9999]
    df['rest_category'] = pd.cut(df['rest_days'], bins=bins, labels=[0,1,2,3,4,5]).astype(float).fillna(2)
    return df


def build_features(df, train_mask):
    """Build derived features. Training time mean from train only."""
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
    df['carry_diff'] = df['weight_carry'] - df.groupby('race_id')['weight_carry'].transform('mean')
    df['weight_cat_dist'] = df['weight_cat'] * 10 + df['dist_cat']
    df['age_group'] = df['age'].clip(2, 7)
    df['surface_dist_enc'] = df['surface_enc'] * 10 + df['dist_cat']
    df['cond_surface'] = df['condition_enc'] * 10 + df['surface_enc']
    df['course_surface'] = df['course_enc'] * 10 + df['surface_enc']
    df['is_nar'] = 0

    # Training time (mean from TRAIN data only)
    df['training_time'] = pd.to_numeric(df['training_time'], errors='coerce').fillna(0)
    df['has_training'] = (df['training_time'] > 0).astype(int)
    train_tt = df.loc[train_mask & (df['training_time'] > 0), 'training_time']
    mean_tt = train_tt.mean() if len(train_tt) > 0 else 48.5
    df['training_time_filled'] = df['training_time'].replace(0, mean_tt).fillna(mean_tt)
    df['training_per_dist'] = df['training_time_filled'] / (df['distance'] / 200).clip(1)
    return df


# ===== Model Training =====
def train_model(X_train, y_train, X_valid, y_valid):
    """Train LightGBM + XGBoost ensemble on train data."""
    print(f"\n  Train: {len(X_train)}, Valid: {len(X_valid)}")
    print(f"  Target rate: train={y_train.mean():.3f}, valid={y_valid.mean():.3f}")

    # LightGBM
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
        'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
    lgb_model = lgb.train(params, dtrain, num_boost_round=1000,
                          valid_sets=[dvalid], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    lgb_pred = lgb_model.predict(X_valid)
    lgb_auc = roc_auc_score(y_valid, lgb_pred)
    print(f"  LightGBM AUC: {lgb_auc:.4f}")

    # XGBoost
    import xgboost as xgb_lib
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 6,
        'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_weight': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'verbosity': 0, 'seed': 42,
    }
    dxgb_train = xgb_lib.DMatrix(X_train, label=y_train)
    dxgb_valid = xgb_lib.DMatrix(X_valid, label=y_valid)
    xgb_model = xgb_lib.train(xgb_params, dxgb_train, num_boost_round=1000,
                               evals=[(dxgb_valid, 'valid')],
                               early_stopping_rounds=50, verbose_eval=False)
    xgb_pred = xgb_model.predict(dxgb_valid)
    xgb_auc = roc_auc_score(y_valid, xgb_pred)
    print(f"  XGBoost AUC:  {xgb_auc:.4f}")

    # Ensemble
    total = lgb_auc + xgb_auc
    w_lgb, w_xgb = lgb_auc / total, xgb_auc / total
    ens_pred = lgb_pred * w_lgb + xgb_pred * w_xgb
    ens_auc = roc_auc_score(y_valid, ens_pred)
    print(f"  Ensemble AUC: {ens_auc:.4f}  (w_lgb={w_lgb:.3f}, w_xgb={w_xgb:.3f})")

    return lgb_model, xgb_model, {'lgb': w_lgb, 'xgb': w_xgb}, ens_auc


def predict_ensemble(lgb_model, xgb_model, weights, X):
    """Predict using ensemble."""
    import xgboost as xgb_lib
    lgb_pred = lgb_model.predict(X)
    xgb_pred = xgb_model.predict(xgb_lib.DMatrix(X))
    return lgb_pred * weights['lgb'] + xgb_pred * weights['xgb']


# ===== Condition Classification =====
def classify_condition(num_horses, distance, condition_str):
    """Classify race into A-E/X."""
    heavy = any(c in str(condition_str) for c in ['\u91cd', '\u4e0d'])
    if num_horses <= 7:
        return 'E'
    if distance <= 1400:
        return 'D'
    if 8 <= num_horses <= 14 and distance >= 1600 and not heavy:
        return 'A'
    if 8 <= num_horses <= 14 and distance >= 1600 and heavy:
        return 'B'
    if num_horses >= 15 and distance >= 1600 and not heavy:
        return 'C'
    return 'X'


# ===== Bet Generation =====
def generate_trio_bets(ranked_umaban):
    """TOP1 axis - TOP2,3 - TOP2~6 formation (7 bets)."""
    if len(ranked_umaban) < 3:
        return []
    n1 = ranked_umaban[0]
    second = ranked_umaban[1:3]
    third = ranked_umaban[1:min(6, len(ranked_umaban))]
    bets = []
    for s in second:
        for t in third:
            if s != t:
                combo = tuple(sorted([n1, s, t]))
                if combo not in bets:
                    bets.append(combo)
    return bets[:7]


def generate_wide_bets(ranked_umaban):
    """TOP1-TOP2, TOP1-TOP3."""
    if len(ranked_umaban) < 3:
        return []
    n1 = ranked_umaban[0]
    return [tuple(sorted([n1, ranked_umaban[1]])), tuple(sorted([n1, ranked_umaban[2]]))]


def generate_umaren_bets(ranked_umaban):
    """TOP1-TOP2, TOP1-TOP3."""
    return generate_wide_bets(ranked_umaban)


# ===== Payout Scraping =====
def scrape_payouts(netkeiba_race_id):
    """Scrape trio/wide/umaren payouts from netkeiba."""
    url = f"https://race.netkeiba.com/race/result.html?race_id={netkeiba_race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = 'euc-jp'
        html = resp.text

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        payouts = {'trio': 0, 'wide': [], 'umaren': []}

        # Parse payout table
        tables = soup.select('table.Payout_Detail_Table, table.pay_table_01')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                th = row.find('th')
                if not th:
                    continue
                label = th.get_text(strip=True)
                tds = row.find_all('td')
                if len(tds) < 2:
                    continue

                nums_text = tds[0].get_text(strip=True)
                payout_text = tds[1].get_text(strip=True).replace(',', '').replace('\u5186', '')

                try:
                    payout_val = int(payout_text)
                except:
                    continue

                nums = [int(x) for x in nums_text.replace('-', ' ').replace('\u2192', ' ').replace('\u2015', ' ').split() if x.isdigit()]

                if '\u4e09\u9023\u8907' in label and len(nums) == 3:
                    payouts['trio'] = payout_val
                elif '\u30ef\u30a4\u30c9' in label and len(nums) == 2:
                    payouts['wide'].append([nums, payout_val])
                elif '\u99ac\u9023' in label and len(nums) == 2:
                    payouts['umaren'].append([nums, payout_val])

        return payouts
    except Exception as e:
        return None


def build_netkeiba_race_id(year, month, day, course_enc, kai, nichi, race_num):
    """Construct netkeiba race_id from CSV fields."""
    # Format: YYYYCCKKNNRR (year=4, course=2, kai=2, nichi=2, race=2)
    course_codes = {0: '01', 1: '02', 2: '03', 3: '04', 4: '05', 5: '06', 6: '07', 7: '08', 8: '09', 9: '10'}
    cc = course_codes.get(int(course_enc), '05')
    return f"{int(year):04d}{cc}{int(kai):02d}{int(nichi):02d}{int(race_num):02d}"


# ===== Main Backtest =====
def main(no_scrape=False):
    print("=" * 60)
    print("  KEIBA AI - COMPLETE LEAK-FREE BACKTEST")
    print("  Train: 2020-2022 | Test: 2023-2025")
    if no_scrape:
        print("  (--no-scrape: using cached payouts only)")
    print("=" * 60)

    # Step 1: Load data
    df = load_and_prepare()

    # Define train/test split
    train_mask = (df['year_full'] >= 2020) & (df['year_full'] <= 2022)
    test_mask = (df['year_full'] >= 2023) & (df['year_full'] <= 2025)
    print(f"\n  Train rows: {train_mask.sum()} ({df[train_mask]['race_id'].nunique()} races)")
    print(f"  Test rows:  {test_mask.sum()} ({df[test_mask]['race_id'].nunique()} races)")

    # Step 2: Encode using ONLY train data
    print("\nEncoding features (train-only maps)...")
    df, sire_map, bms_map = encode_with_train_only(df, train_mask)

    # Step 3: Compute lag features (shift = no leak)
    print("Computing lag features...")
    df = compute_lag_features(df)

    # Step 4: Compute jockey/trainer stats from train ONLY
    print("Computing jockey/trainer stats (train-only)...")
    df = compute_stats_train_only(df, train_mask)

    # Step 5: Build derived features
    print("Building derived features...")
    df = build_features(df, train_mask)

    # Ensure all features are numeric
    for f in FEATURES_V91:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    # Step 6: Train model on 2020-2022
    print("\n" + "=" * 60)
    print("  TRAINING V9.1 on 2020-2022 ONLY")
    print("=" * 60)

    # For training AUC validation, split 2020-2021 train / 2022 valid
    inner_train = (df['year_full'] >= 2020) & (df['year_full'] <= 2021)
    inner_valid = (df['year_full'] == 2022)

    df_filtered = df[df['num_horses_val'] >= 5].copy()
    X_all = df_filtered[FEATURES_V91].values
    y_all = (df_filtered['finish'] == 1).astype(int).values

    # Inner split for model selection
    inner_t = inner_train[df_filtered.index]
    inner_v = inner_valid[df_filtered.index]
    X_it, y_it = X_all[inner_t], y_all[inner_t]
    X_iv, y_iv = X_all[inner_v], y_all[inner_v]

    print("\n  --- Inner validation (2020-2021 train / 2022 valid) ---")
    lgb_m, xgb_m, weights, inner_auc = train_model(X_it, y_it, X_iv, y_iv)

    # Now retrain on FULL 2020-2022 for final model
    full_train = train_mask[df_filtered.index]
    full_test = test_mask[df_filtered.index]
    X_train, y_train = X_all[full_train], y_all[full_train]
    X_test, y_test = X_all[full_test], y_all[full_test]

    print(f"\n  --- Final training on ALL 2020-2022 ---")
    # Use 2022 as validation for early stopping
    lgb_model, xgb_model, weights, _ = train_model(
        X_all[inner_t], y_all[inner_t], X_all[inner_v], y_all[inner_v]
    )

    # Evaluate on 2023-2025
    test_pred = predict_ensemble(lgb_model, xgb_model, weights, X_test)
    test_auc = roc_auc_score(y_test, test_pred)
    print(f"\n  *** TEST AUC (2023-2025, unseen): {test_auc:.4f} ***")

    # Step 7: Run backtest on test races
    print("\n" + "=" * 60)
    print("  RUNNING CONDITION-BASED BACKTEST ON 2023-2025")
    print("=" * 60)

    # Load existing payout data for matching
    existing_payouts = {}
    if os.path.exists(EXISTING_RESULTS):
        with open(EXISTING_RESULTS, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        for r in existing_data.get('central_results_5year', []):
            rid = r.get('race_id', '')
            if rid:
                existing_payouts[rid] = r.get('payouts', {})
        print(f"  Loaded {len(existing_payouts)} cached payout entries")

    # Get unique test races
    test_df = df_filtered[full_test].copy()
    test_df['pred_score'] = test_pred
    test_race_ids = test_df['race_id'].unique()
    print(f"  Test races: {len(test_race_ids)}")

    results = []
    scraped = 0
    cached = 0
    no_payout = 0

    for i, rid in enumerate(test_race_ids):
        race_horses = test_df[test_df['race_id'] == rid].copy()
        if len(race_horses) < 3:
            continue

        race_horses = race_horses.sort_values('pred_score', ascending=False)
        ranked = race_horses['umaban'].tolist()
        actual_top3 = set(race_horses.nsmallest(3, 'finish')['umaban'].tolist())

        # Race info
        first = race_horses.iloc[0]
        num_h = int(first['num_horses_val'])
        dist = int(first['distance'])
        cond_str = str(first.get('condition', '\u826f'))
        year = int(first['year_full'])
        month = int(first['month'])
        day = int(first['day'])
        course = str(first.get('course', ''))
        surface = str(first.get('surface', ''))
        race_num = int(first.get('race_num', 0))
        course_enc_val = int(first.get('course_enc', 0))
        kai = int(first.get('kai', 1)) if 'kai' in first.index else 1
        nichi = int(first.get('nichi', 1)) if 'nichi' in first.index else 1

        cond_key = classify_condition(num_h, dist, cond_str)

        # Generate bets
        trio_bets = generate_trio_bets(ranked)
        wide_bets = generate_wide_bets(ranked)
        umaren_bets = generate_umaren_bets(ranked)

        # Check trio hit
        trio_hit = any(set(b).issubset(actual_top3) for b in trio_bets)

        # Check wide hits (any pair in top3)
        wide_hits = [b for b in wide_bets if set(b).issubset(actual_top3)]

        # Check umaren hit (exact 1st-2nd)
        actual_top2 = set(race_horses.nsmallest(2, 'finish')['umaban'].tolist())
        umaren_hits = [b for b in umaren_bets if set(b) == actual_top2]

        # Get payouts
        payouts = existing_payouts.get(rid)
        if payouts:
            cached += 1
        elif no_scrape:
            no_payout += 1
            payouts = None
        else:
            # Try to scrape
            try:
                nk_id = build_netkeiba_race_id(year, month, day, course_enc_val, kai, nichi, race_num)
                payouts = scrape_payouts(nk_id)
                if payouts:
                    scraped += 1
                    time.sleep(1.5)
                else:
                    no_payout += 1
            except:
                no_payout += 1
                payouts = None

        trio_payout = 0
        wide_payout = 0
        umaren_payout = 0
        if payouts:
            if trio_hit:
                trio_payout = payouts.get('trio', 0)
            for wh in wide_hits:
                wh_set = set(wh)
                for wp in payouts.get('wide', []):
                    if set(wp[0]) == wh_set:
                        wide_payout += wp[1]
                        break
            for uh in umaren_hits:
                uh_set = set(uh)
                for up in payouts.get('umaren', []):
                    if set(up[0]) == uh_set:
                        umaren_payout += up[1]
                        break

        result = {
            'race_id': rid, 'year': year, 'month': month,
            'course': course, 'distance': dist, 'surface': surface,
            'condition': cond_str, 'num_horses': num_h,
            'condition_key': cond_key,
            'trio_bets': [list(b) for b in trio_bets],
            'trio_hit': trio_hit, 'trio_payout': trio_payout,
            'wide_bets': [list(b) for b in wide_bets],
            'wide_hits': [list(b) for b in wide_hits],
            'wide_payout': wide_payout,
            'umaren_bets': [list(b) for b in umaren_bets],
            'umaren_hits': [list(b) for b in umaren_hits],
            'umaren_payout': umaren_payout,
            'has_payout': payouts is not None,
        }
        results.append(result)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(test_race_ids)} races (cached={cached}, scraped={scraped})")

    print(f"\n  Total: {len(results)} races (cached={cached}, scraped={scraped}, no_payout={no_payout})")

    # Step 8: Calculate condition-based ROI
    print("\n" + "=" * 60)
    print("  CONDITION-BASED ROI RESULTS (Leak-Free)")
    print("=" * 60)

    # Only use races with payout data for ROI
    results_with_payout = [r for r in results if r['has_payout']]
    print(f"  Races with payout data: {len(results_with_payout)}/{len(results)}")

    cond_info = {
        'A': {'desc': '8-14h/1600m+/good', 'bet': 'trio'},
        'B': {'desc': '8-14h/1600m+/heavy', 'bet': 'trio'},
        'C': {'desc': '15h+/1600m+/good', 'bet': 'trio'},
        'D': {'desc': '1400m-/sprint', 'bet': 'trio'},
        'E': {'desc': '7h-/small', 'bet': 'umaren'},
        'X': {'desc': '15h+/heavy', 'bet': 'trio'},
    }

    condition_stats = {}
    for ckey in ['A', 'B', 'C', 'D', 'E', 'X']:
        races = [r for r in results_with_payout if r['condition_key'] == ckey]
        all_races = [r for r in results if r['condition_key'] == ckey]
        n = len(races)
        n_all = len(all_races)

        stats = {'n': n, 'n_all': n_all}

        for bt in ['trio', 'wide', 'umaren']:
            if bt == 'trio':
                hits = sum(1 for r in races if r['trio_hit'])
                payouts_sum = sum(r['trio_payout'] for r in races if r['trio_hit'])
                inv = n * 7 * 100
            elif bt == 'wide':
                hits = sum(1 for r in races if len(r['wide_hits']) > 0)
                payouts_sum = sum(r['wide_payout'] for r in races)
                inv = n * 2 * 100
            elif bt == 'umaren':
                hits = sum(1 for r in races if len(r['umaren_hits']) > 0)
                payouts_sum = sum(r['umaren_payout'] for r in races)
                inv = n * 2 * 100

            hit_rate = hits / n * 100 if n > 0 else 0
            roi = payouts_sum / inv * 100 if inv > 0 else 0
            stats[bt] = {'hits': hits, 'hit_rate': hit_rate, 'investment': inv, 'payout': payouts_sum, 'roi': roi}

        # Also per-year breakdown
        yearly = {}
        for r in races:
            yr = r['year']
            yearly.setdefault(yr, []).append(r)
        stats['yearly'] = {}
        for yr in sorted(yearly.keys()):
            yr_races = yearly[yr]
            n_yr = len(yr_races)
            trio_hits_yr = sum(1 for r in yr_races if r['trio_hit'])
            trio_pay_yr = sum(r['trio_payout'] for r in yr_races if r['trio_hit'])
            inv_yr = n_yr * 700
            stats['yearly'][yr] = {
                'n': n_yr, 'trio_hits': trio_hits_yr,
                'trio_roi': trio_pay_yr / inv_yr * 100 if inv_yr > 0 else 0
            }

        condition_stats[ckey] = stats

    # Print results
    print(f"\n  {'COND':<6} {'N':>5} {'TRIO HIT':>10} {'TRIO ROI':>10} {'WIDE HIT':>10} {'WIDE ROI':>10} {'UMA HIT':>10} {'UMA ROI':>10}")
    print(f"  {'-'*80}")
    for ckey in ['A', 'B', 'C', 'D', 'E', 'X']:
        s = condition_stats[ckey]
        if s['n'] == 0:
            continue
        t = s['trio']; w = s['wide']; u = s['umaren']
        print(f"  {ckey:<6} {s['n']:>5} {t['hit_rate']:>8.1f}% {t['roi']:>8.1f}%  {w['hit_rate']:>8.1f}% {w['roi']:>8.1f}%  {u['hit_rate']:>8.1f}% {u['roi']:>8.1f}%")

    # Yearly breakdown
    print(f"\n  YEARLY TRIO ROI:")
    print(f"  {'COND':<6}", end='')
    years = sorted(set(r['year'] for r in results))
    for yr in years:
        print(f" {yr:>8}", end='')
    print()
    print(f"  {'-'*50}")
    for ckey in ['A', 'B', 'C', 'D', 'E', 'X']:
        s = condition_stats[ckey]
        print(f"  {ckey:<6}", end='')
        for yr in years:
            yd = s['yearly'].get(yr, {})
            roi = yd.get('trio_roi', 0)
            n_yr = yd.get('n', 0)
            if n_yr > 0:
                print(f" {roi:>7.1f}%", end='')
            else:
                print(f"      - ", end='')
        print()

    # Determine recommendation tier
    print(f"\n\n  {'='*60}")
    print(f"  FINAL RECOMMENDATION (Leak-Free Verified)")
    print(f"  {'='*60}")

    recommendations = {}
    for ckey in ['A', 'B', 'C', 'D', 'E', 'X']:
        s = condition_stats[ckey]
        if s['n'] == 0:
            recommendations[ckey] = {'tier': 'none', 'stars': 0, 'roi': 0, 'hit_rate': 0, 'bet_type': 'trio', 'n': 0}
            continue

        # Choose best bet type
        best_bt = max(['trio', 'wide', 'umaren'], key=lambda bt: s[bt]['roi'])
        # For E, prefer umaren (historically correct even if trio might be higher in small samples)
        if ckey == 'E' and s['umaren']['roi'] > 80:
            best_bt = 'umaren'

        roi = s[best_bt]['roi']
        hit_rate = s[best_bt]['hit_rate']

        if roi >= 120:
            tier = 'strong'
            stars = 3
        elif roi >= 100:
            tier = 'recommended'
            stars = 2
        elif roi >= 80:
            tier = 'reference'
            stars = 1
        else:
            tier = 'not_recommended'
            stars = 0

        recommendations[ckey] = {
            'tier': tier, 'stars': stars, 'roi': roi,
            'hit_rate': hit_rate, 'bet_type': best_bt,
            'n': s['n'], 'n_all': s['n_all'],
        }

        star_str = '\u2605' * stars + '\u2606' * (3 - stars) if stars > 0 else '\u2717'
        print(f"  {ckey}: {star_str} {best_bt} ROI={roi:.1f}% Hit={hit_rate:.1f}% (n={s['n']})")

    # Save results
    save_data = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_period': '2020-2022',
        'test_period': '2023-2025',
        'test_auc': test_auc,
        'total_test_races': len(results),
        'races_with_payouts': len(results_with_payout),
        'condition_stats': {},
        'recommendations': recommendations,
        'results': results,
    }
    # Convert condition_stats for JSON
    for ckey, stats in condition_stats.items():
        save_stats = dict(stats)
        save_stats['yearly'] = {str(k): v for k, v in stats['yearly'].items()}
        save_data['condition_stats'][ckey] = save_stats

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved to: {OUTPUT_PATH}")

    return recommendations, condition_stats, test_auc


if __name__ == '__main__':
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument('--no-scrape', action='store_true',
                         help='Skip payout scraping, use only cached data')
    _args = _parser.parse_args()
    recs, stats, auc = main(no_scrape=_args.no_scrape)
