#!/usr/bin/env python
"""KEIBA AI - V8/V9 Leak-Free Backtest
2025/10-12月の中央100レース + 地方100レースで検証。
各レースの予測時点で当日以降のデータを一切使わない完全リークフリー。
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
import re
import sys
import time
import requests
from datetime import datetime
from itertools import combinations
from bs4 import BeautifulSoup

# ===== Config =====
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'target_odds.csv')
RESULT_PATH = os.path.join(os.path.dirname(__file__), 'backtest_results.json')
V8_PATH = os.path.join(os.path.dirname(__file__), 'keiba_model_v8.pkl')
V9C_PATH = os.path.join(os.path.dirname(__file__), 'keiba_model_v9_central.pkl')
V9N_PATH = os.path.join(os.path.dirname(__file__), 'keiba_model_v9_nar.pkl')
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
N_CENTRAL = 100
N_NAR = 100

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
    'odds': 48, 'empty1': 49, 'empty2': 50, 'col51': 51,
}


# ===== Load Models =====
def load_models():
    models = {}
    for name, path in [('v8', V8_PATH), ('v9c', V9C_PATH), ('v9n', V9N_PATH)]:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
            print(f"  Loaded {name}: AUC={models[name].get('auc', '?')}")
        else:
            print(f"  {name} not found: {path}")
    return models


# ===== Load and Prepare CSV Data =====
def load_csv():
    print("Loading CSV data...")
    df = pd.read_csv(DATA_PATH, encoding='cp932', header=None, low_memory=False)
    inv_col = {v: k for k, v in COL.items()}
    df.columns = [inv_col.get(i, f'col{i}') for i in range(df.shape[1])]

    df['finish'] = pd.to_numeric(df['finish'], errors='coerce')
    df = df[df['finish'].notna() & (df['finish'] >= 1)].copy()
    df['year_full'] = pd.to_numeric(df['year'], errors='coerce') + 2000
    df['month'] = pd.to_numeric(df['month'], errors='coerce')
    df['day'] = pd.to_numeric(df['day'], errors='coerce')
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
    return df


def encode_data(df, train_mask):
    """Encode categoricals using ONLY training data (leak-free)."""
    train = df[train_mask]

    # Sex encoding
    sex_map = {}
    for val in df['sex'].dropna().unique():
        s = str(val).strip()
        if '牡' in s: sex_map[val] = 0
        elif '牝' in s: sex_map[val] = 1
        elif 'セ' in s or '騸' in s: sex_map[val] = 2
        else: sex_map[val] = 0
    df['sex_enc'] = df['sex'].map(sex_map).fillna(0).astype(int)

    # Surface
    surf_map = {}
    for val in df['surface'].dropna().unique():
        s = str(val).strip()
        if '芝' in s: surf_map[val] = 0
        elif 'ダ' in s: surf_map[val] = 1
        else: surf_map[val] = 2
    df['surface_enc'] = df['surface'].map(surf_map).fillna(0).astype(int)

    # Condition
    cond_map = {}
    for val in df['condition'].dropna().unique():
        s = str(val).strip()
        if '良' in s: cond_map[val] = 0
        elif '稍' in s: cond_map[val] = 1
        elif '重' in s: cond_map[val] = 2
        elif '不' in s: cond_map[val] = 3
        else: cond_map[val] = 0
    df['condition_enc'] = df['condition'].map(cond_map).fillna(0).astype(int)

    # Course encoding (from training data only)
    course_counts = train['course'].value_counts()
    course_map = {c: i for i, c in enumerate(course_counts.index)}
    df['course_enc'] = df['course'].map(course_map).fillna(len(course_map)).astype(int)

    # Location
    loc_map = {}
    for val in df['location'].dropna().unique():
        s = str(val).strip()
        if '美' in s: loc_map[val] = 0
        elif '栗' in s: loc_map[val] = 1
        elif '地' in s: loc_map[val] = 2
        elif '外' in s: loc_map[val] = 3
        else: loc_map[val] = 0
    df['location_enc'] = df['location'].map(loc_map).fillna(0).astype(int)

    # Sire/BMS (from training data only)
    n_top = 100
    sire_counts = train['father'].value_counts()
    top_sires = sire_counts.head(n_top).index.tolist()
    sire_map = {s: i for i, s in enumerate(top_sires)}
    df['sire_enc'] = df['father'].map(sire_map).fillna(n_top).astype(int)

    bms_counts = train['bms'].value_counts()
    top_bms = bms_counts.head(n_top).index.tolist()
    bms_map = {s: i for i, s in enumerate(top_bms)}
    df['bms_enc'] = df['bms'].map(bms_map).fillna(n_top).astype(int)

    return df, sire_map, bms_map


def compute_stats_leakfree(df, train_mask):
    """Compute jockey/trainer stats from training data ONLY."""
    train = df[train_mask].copy()

    # Jockey win rate (Bayesian smoothed)
    train['is_win'] = (train['finish'] == 1).astype(int)
    j_stats = train.groupby('jockey_id').agg(
        races=('is_win', 'count'), wins=('is_win', 'sum')
    ).reset_index()
    global_wr = j_stats['wins'].sum() / j_stats['races'].sum()
    alpha = 30
    j_stats['jockey_wr_calc'] = (j_stats['wins'] + alpha * global_wr) / (j_stats['races'] + alpha)
    jwr = dict(zip(j_stats['jockey_id'], j_stats['jockey_wr_calc']))
    df['jockey_wr_calc'] = df['jockey_id'].map(jwr).fillna(global_wr)

    # Jockey course win rate
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

    # Trainer top3 rate
    train['is_top3'] = (train['finish'] <= 3).astype(int)
    t_stats = train.groupby('trainer_id').agg(
        races=('is_top3', 'count'), top3=('is_top3', 'sum')
    ).reset_index()
    global_t3 = t_stats['top3'].sum() / t_stats['races'].sum()
    t_stats['trainer_top3_calc'] = (t_stats['top3'] + 20 * global_t3) / (t_stats['races'] + 20)
    tmap = dict(zip(t_stats['trainer_id'], t_stats['trainer_top3_calc']))
    df['trainer_top3_calc'] = df['trainer_id'].map(tmap).fillna(global_t3)

    return df


def compute_lag_features(df):
    """Compute lag features per horse. Shift ensures no future leak."""
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


def build_features(df):
    """Build model features."""
    df['horse_num'] = df['umaban']
    df['bracket'] = np.clip(((df['horse_num'] - 1) * 8 // df['num_horses'].clip(1)) + 1, 1, 8)
    df['dist_cat'] = pd.cut(df['distance'], bins=[0,1200,1400,1800,2200,9999],
                            labels=[0,1,2,3,4]).astype(float).fillna(2)
    df['weight_cat'] = pd.cut(df['horse_weight'], bins=[0,440,480,520,9999],
                              labels=[0,1,2,3]).astype(float).fillna(1)
    df['age_sex'] = df['age'] * 10 + df['sex_enc']
    df['season'] = df['month'].apply(lambda m: 0 if m in [3,4,5] else (1 if m in [6,7,8] else (2 if m in [9,10,11] else 3)))
    df['age_season'] = df['age'] * 10 + df['season']
    df['horse_num_ratio'] = df['horse_num'] / df['num_horses'].clip(1)
    df['bracket_pos'] = pd.cut(df['bracket'], bins=[0,3,6,8], labels=[0,1,2]).astype(float).fillna(1)
    df['carry_diff'] = df['weight_carry'] - df.groupby('race_id')['weight_carry'].transform('mean')
    df['weight_cat_dist'] = df['weight_cat'] * 10 + df['dist_cat']
    df['age_group'] = df['age'].clip(2, 7)
    df['surface_dist_enc'] = df['surface_enc'] * 10 + df['dist_cat']
    df['cond_surface'] = df['condition_enc'] * 10 + df['surface_enc']
    df['course_surface'] = df['course_enc'] * 10 + df['surface_enc']
    df['is_nar'] = 0
    return df


# ===== V8/V9 Features =====
V8_FEATURES = [
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
]

V9_FEATURES = V8_FEATURES + ['odds_log', 'prev_odds_log']


def predict_v8(model_data, X, features):
    """V8 prediction (LightGBM only)."""
    m = model_data['model']
    return m.predict(X)


def predict_v9_ensemble(model_data, X, features):
    """V9 prediction (LightGBM + XGBoost + MLP ensemble)."""
    lgb_pred = model_data['model'].predict(X)

    xgb_model = model_data.get('xgb_model')
    mlp_model = model_data.get('mlp_model')
    mlp_scaler = model_data.get('mlp_scaler')
    weights = model_data.get('ensemble_weights', {'lgb': 0.4, 'xgb': 0.35, 'mlp': 0.25})

    if xgb_model and mlp_model and mlp_scaler:
        import xgboost as xgb
        xgb_pred = xgb_model.predict(xgb.DMatrix(X))
        mlp_pred = mlp_model.predict_proba(mlp_scaler.transform(X))[:, 1]
        return lgb_pred * weights['lgb'] + xgb_pred * weights['xgb'] + mlp_pred * weights['mlp']
    else:
        return lgb_pred


# ===== Bet Calculation =====
def calc_bets(top_nums):
    """Calculate bet combinations from TOP-ranked horse numbers.
    top_nums: list of umaban sorted by AI rank (best first), at least 6.
    Returns: trio_bets (7 combos), wide_bets (2 combos), umaren_bets (2 combos)
    """
    n1 = top_nums[0]  # axis
    top2 = top_nums[1] if len(top_nums) > 1 else top_nums[0]
    top3 = top_nums[2] if len(top_nums) > 2 else top2
    flow = top_nums[1:min(6, len(top_nums))]

    # Trio: TOP1 axis - TOP2,TOP3 - TOP2~TOP6 (7 combos)
    trio_bets = set()
    for s in [top2, top3]:
        for t in flow:
            combo = tuple(sorted({n1, s, t}))
            if len(combo) == 3:
                trio_bets.add(combo)
    trio_bets = [list(b) for b in sorted(trio_bets)]

    # Wide: TOP1-TOP2, TOP1-TOP3 (2 combos)
    wide_bets = [
        sorted([n1, top2]),
        sorted([n1, top3]),
    ]
    # Remove duplicate
    if wide_bets[0] == wide_bets[1]:
        wide_bets = [wide_bets[0]]

    # Umaren: TOP1-TOP2, TOP1-TOP3 (2 combos)
    umaren_bets = [
        sorted([n1, top2]),
        sorted([n1, top3]),
    ]
    if umaren_bets[0] == umaren_bets[1]:
        umaren_bets = [umaren_bets[0]]

    return trio_bets, wide_bets, umaren_bets


def check_hits(actual_top3_nums, trio_bets, wide_bets, umaren_bets):
    """Check which bets hit.
    actual_top3_nums: {finish_pos: umaban} dict for top finishers.
    """
    actual_set = set()
    actual_1st = actual_top3_nums.get(1)
    actual_2nd = actual_top3_nums.get(2)
    actual_3rd = actual_top3_nums.get(3)
    if actual_1st: actual_set.add(actual_1st)
    if actual_2nd: actual_set.add(actual_2nd)
    if actual_3rd: actual_set.add(actual_3rd)

    # Trio hit (order-independent, all 3 in top 3)
    trio_hit = False
    trio_hit_combo = None
    for combo in trio_bets:
        if set(combo) == actual_set and len(actual_set) == 3:
            trio_hit = True
            trio_hit_combo = combo
            break

    # Wide hit (2 horses in top 3, order-independent)
    wide_hits = []
    for combo in wide_bets:
        if set(combo).issubset(actual_set):
            wide_hits.append(combo)

    # Umaren hit (exact 1st-2nd match, order-independent)
    umaren_hits = []
    if actual_1st and actual_2nd:
        actual_12 = sorted([actual_1st, actual_2nd])
        for combo in umaren_bets:
            if sorted(combo) == actual_12:
                umaren_hits.append(combo)

    return trio_hit, trio_hit_combo, wide_hits, umaren_hits


# ===== Scrape Payouts from netkeiba =====
def split_by_br(td):
    """Split td content by <br/> tags, returning list of text segments."""
    parts = []
    current = ''
    for child in td.children:
        if isinstance(child, str):
            current += child.strip()
        elif child.name == 'br':
            if current:
                parts.append(current)
            current = ''
        else:
            current += child.get_text(strip=True)
    if current:
        parts.append(current)
    return parts


def scrape_payouts(netkeiba_race_id, is_nar=False):
    """Scrape wide, umaren, trio payouts from netkeiba result page.
    Returns dict: {trio: int, wide: [(combo, payout)], umaren: [(combo, payout)]}
    """
    payouts = {'trio': 0, 'wide': [], 'umaren': []}

    # Use db.netkeiba.com directly (most reliable for past races)
    url = f"https://db.netkeiba.com/race/{netkeiba_race_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "EUC-JP"
        if resp.status_code != 200:
            return payouts
        soup = BeautifulSoup(resp.text, "html.parser")

        tables = soup.find_all("table", class_="pay_table_01")
        for table in tables:
            for tr in table.find_all("tr"):
                th = tr.find("th")
                if not th:
                    continue
                th_text = th.get_text(strip=True)
                tds = tr.find_all("td")
                if len(tds) < 2:
                    continue

                if '馬連' in th_text and '馬単' not in th_text:
                    combo_text = tds[0].get_text(strip=True)
                    payout_text = tds[1].get_text(strip=True).replace(',', '')
                    nums = re.findall(r'\d+', combo_text)
                    pm = re.search(r'(\d+)', payout_text)
                    if len(nums) >= 2 and pm:
                        payouts['umaren'].append((sorted([int(nums[0]), int(nums[1])]), int(pm.group(1))))

                elif 'ワイド' in th_text:
                    # Wide has multiple combos separated by <br/>
                    combo_parts = split_by_br(tds[0])
                    payout_parts = split_by_br(tds[1])
                    for i, cp in enumerate(combo_parts):
                        nums = re.findall(r'\d+', cp)
                        if len(nums) >= 2 and i < len(payout_parts):
                            pv = int(payout_parts[i].replace(',', ''))
                            payouts['wide'].append((sorted([int(nums[0]), int(nums[1])]), pv))

                elif '3連複' in th_text or '三連複' in th_text:
                    payout_text = tds[1].get_text(strip=True).replace(',', '')
                    pm = re.search(r'(\d+)', payout_text)
                    if pm:
                        payouts['trio'] = int(pm.group(1))

    except Exception:
        pass

    return payouts


def build_netkeiba_race_id(race_8):
    """Convert CSV 8-char race_id to netkeiba 12-digit race_id.
    CSV format: VV YY M K DD (venue(2) year(2) kai(1) nichi(1) race_num(2))
    netkeiba:   YYYY VV 0M 0K DD (year(4) venue(2) kai(2) nichi(2) race_num(2))
    """
    rid = str(race_8)
    if len(rid) != 8:
        return rid
    VV = rid[0:2]   # venue code
    YY = rid[2:4]   # year (2-digit)
    M = rid[4]       # kai (1 digit)
    K = rid[5]       # nichi (1 digit)
    DD = rid[6:8]    # race number (2 digits)
    return f"20{YY}{VV}0{M}0{K}{DD}"


# ===== Main Backtest =====
def run_central_backtest(models):
    """Run backtest on 100 central races from Oct-Dec 2025 using CSV data."""
    print("\n" + "="*60)
    print("  CENTRAL BACKTEST (CSV data)")
    print("="*60)

    df = load_csv()

    # Training cutoff: everything before Oct 2025
    cutoff = 20251001
    train_mask = df['date_num'] < cutoff
    test_mask = (df['date_num'] >= cutoff) & (df['year_full'] == 2025) & (df['num_horses'] >= 8)

    print(f"  Train: {train_mask.sum()} rows")
    print(f"  Test pool: {test_mask.sum()} rows")

    # Encode using training data only
    df, sire_map, bms_map = encode_data(df, train_mask)
    df = compute_stats_leakfree(df, train_mask)
    df = compute_lag_features(df)
    df = build_features(df)

    # Get test races (sample 100)
    test_races = df[test_mask]['race_id'].unique()
    print(f"  Test races available: {len(test_races)}")

    if len(test_races) > N_CENTRAL:
        rng = np.random.RandomState(42)
        test_races = rng.choice(test_races, N_CENTRAL, replace=False)
    print(f"  Selected: {len(test_races)} races")

    # Prepare feature matrices
    for f in V8_FEATURES + ['odds_log', 'prev_odds_log']:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    v8_data = models.get('v8')
    v9_data = models.get('v9c')

    results = []
    for ri, rid in enumerate(test_races):
        race_df = df[(df['race_id'] == rid) & test_mask].copy()
        if len(race_df) < 5:
            continue

        # Race metadata
        row0 = race_df.iloc[0]
        race_meta = {
            'race_id': rid,
            'date': f"{int(row0['year_full'])}-{int(row0['month']):02d}-{int(row0['day']):02d}",
            'course': str(row0.get('course', '')),
            'distance': int(row0['distance']),
            'surface': str(row0.get('surface', '')),
            'condition': str(row0.get('condition', '')),
            'num_horses': len(race_df),
        }

        # Actual results
        actual = {}
        for _, h in race_df.iterrows():
            actual[int(h['umaban'])] = int(h['finish'])
        actual_top3 = {v: k for k, v in sorted(actual.items(), key=lambda x: x[1]) if v <= 3}
        # Invert: {finish_pos: umaban}
        finish_to_uma = {}
        for uma, fin in actual.items():
            if fin <= 3:
                finish_to_uma[fin] = uma

        # V8 prediction
        v8_scores = None
        v8_ranking = None
        if v8_data:
            X_v8 = race_df[V8_FEATURES].values
            v8_scores = predict_v8(v8_data, X_v8, V8_FEATURES)
            race_df['v8_score'] = v8_scores
            race_df['v8_rank'] = race_df['v8_score'].rank(ascending=False).astype(int)
            v8_ranking = race_df.sort_values('v8_rank')['umaban'].tolist()

        # V9 prediction
        v9_scores = None
        v9_ranking = None
        if v9_data:
            X_v9 = race_df[V9_FEATURES].values
            v9_scores = predict_v9_ensemble(v9_data, X_v9, V9_FEATURES)
            race_df['v9_score'] = v9_scores
            race_df['v9_rank'] = race_df['v9_score'].rank(ascending=False).astype(int)
            v9_ranking = race_df.sort_values('v9_rank')['umaban'].tolist()

        # Calculate bets and check hits
        result_entry = {**race_meta, 'actual_top3': finish_to_uma}

        for ver, ranking in [('v8', v8_ranking), ('v9', v9_ranking)]:
            if not ranking:
                continue
            trio_bets, wide_bets, umaren_bets = calc_bets(ranking)
            trio_hit, trio_combo, wide_hits, umaren_hits = check_hits(
                finish_to_uma, trio_bets, wide_bets, umaren_bets
            )
            result_entry[f'{ver}_top3'] = ranking[:3]
            result_entry[f'{ver}_top6'] = ranking[:6]
            result_entry[f'{ver}_trio_bets'] = trio_bets
            result_entry[f'{ver}_trio_hit'] = trio_hit
            result_entry[f'{ver}_wide_bets'] = wide_bets
            result_entry[f'{ver}_wide_hits'] = [list(w) for w in wide_hits]
            result_entry[f'{ver}_umaren_bets'] = umaren_bets
            result_entry[f'{ver}_umaren_hits'] = [list(u) for u in umaren_hits]

        results.append(result_entry)

        if (ri + 1) % 20 == 0:
            print(f"  Processed {ri+1}/{len(test_races)} races")

    print(f"  Completed: {len(results)} races processed")
    return results


def scrape_payouts_for_results(results, is_nar=False):
    """Scrape payouts for all races in results. Adds payout data in-place."""
    print(f"\n  Scraping payouts for {len(results)} races...")
    for i, r in enumerate(results):
        rid = r['race_id']
        nk_rid = build_netkeiba_race_id(rid)
        r['netkeiba_race_id'] = nk_rid
        try:
            payouts = scrape_payouts(nk_rid, is_nar=is_nar)
            r['payouts'] = payouts
            if payouts['trio'] > 0:
                r['trio_payout'] = payouts['trio']
            # Match wide payouts to bets
            for ver in ['v8', 'v9']:
                wide_hits = r.get(f'{ver}_wide_hits', [])
                if wide_hits and payouts['wide']:
                    total_wide_payout = 0
                    for hit_combo in wide_hits:
                        for pw_combo, pw_payout in payouts['wide']:
                            if sorted(hit_combo) == sorted(pw_combo):
                                total_wide_payout += pw_payout
                                break
                    r[f'{ver}_wide_payout'] = total_wide_payout

                umaren_hits = r.get(f'{ver}_umaren_hits', [])
                if umaren_hits and payouts['umaren']:
                    total_umaren_payout = 0
                    for hit_combo in umaren_hits:
                        for pu_combo, pu_payout in payouts['umaren']:
                            if sorted(hit_combo) == sorted(pu_combo):
                                total_umaren_payout += pu_payout
                                break
                    r[f'{ver}_umaren_payout'] = total_umaren_payout

        except Exception as e:
            r['payouts'] = {'error': str(e)}

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(results)} scraped")
        time.sleep(1.0)  # Rate limit

    return results


# ===== NAR Backtest via Scraping =====
def collect_nar_race_ids(n_races=100):
    """Collect NAR race IDs for Oct-Dec 2025 from db.netkeiba.com race list.
    NAR venue codes are > 10 in netkeiba's system (30+, 40+, etc.)"""
    print("\n  Collecting NAR race IDs from db.netkeiba.com...")
    all_races = []
    seen = set()
    from datetime import timedelta
    start = datetime(2025, 10, 1)
    end = datetime(2025, 12, 31)
    d = start
    dates_tried = 0
    while d <= end and len(all_races) < n_races * 5:
        date_str = d.strftime('%Y%m%d')
        try:
            url = f"https://db.netkeiba.com/race/list/{date_str}/"
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.encoding = "EUC-JP"
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.find_all("a", href=re.compile(r'/race/\d{12}'))
            for link in links:
                href = link.get("href", "")
                m = re.search(r'/race/(\d{12})/', href)
                if m:
                    rid = m.group(1)
                    venue = int(rid[4:6])
                    # NAR venues have code > 10
                    if venue > 10 and rid not in seen:
                        seen.add(rid)
                        all_races.append({'race_id': rid, 'date': date_str})
        except Exception:
            pass
        dates_tried += 1
        if dates_tried % 15 == 0:
            print(f"    {dates_tried} dates, {len(all_races)} NAR races found")
        d += timedelta(days=1)
        time.sleep(0.3)

    print(f"  Found {len(all_races)} NAR races total")
    # Sample n_races, prefer later races (more variety)
    if len(all_races) > n_races:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(all_races), n_races, replace=False)
        all_races = [all_races[i] for i in sorted(indices)]
    return all_races


def scrape_nar_race(race_id):
    """Scrape a NAR race result page for horse info + actual finishes + payouts.
    Returns: (horses_df, actual_finishes, payouts) or None on failure."""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "EUC-JP"
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Race info
        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else ""

        # Distance / surface / condition from race header
        header_div = soup.find("diary_snap") or soup.find("div", class_="data_intro")
        header_text = soup.get_text()[:3000]
        dm = re.search(r'([芝ダ障])(\d{3,4})m', header_text)
        distance = int(dm.group(2)) if dm else 1600
        surface = dm.group(1) if dm else 'ダ'
        cm = re.search(r'馬場:(良|稍重|稍|重|不良|不)', header_text)
        if not cm:
            cm = re.search(r'(良|稍重|重|不良)', header_text[:1000])
        condition = cm.group(1) if cm else '良'

        # Course from title
        course_name = ''
        for cn in ['大井','川崎','船橋','浦和','園田','姫路','門別','盛岡','水沢',
                    '金沢','笠松','名古屋','高知','佐賀','帯広']:
            if cn in title_text:
                course_name = cn
                break

        # Horse list from result table
        table = soup.find("table", class_="race_table_01")
        if not table:
            return None
        rows = table.find_all("tr")

        horses = []
        actual_finishes = {}
        for row in rows:
            tds = row.find_all("td")
            if len(tds) < 13:
                continue
            finish_text = tds[0].get_text(strip=True)
            if not finish_text.isdigit():
                continue
            finish = int(finish_text)
            umaban = int(tds[2].get_text(strip=True)) if tds[2].get_text(strip=True).isdigit() else 0
            if umaban == 0:
                continue
            actual_finishes[umaban] = finish

            horse_name = ''
            horse_link = tds[3].find("a") if len(tds) > 3 else None
            if horse_link:
                horse_name = horse_link.get_text(strip=True)

            # Jockey
            jockey_name = ''
            jockey_link = tds[6].find("a") if len(tds) > 6 else None
            if jockey_link:
                jockey_name = jockey_link.get_text(strip=True)

            # Age/Sex
            sex_age = tds[4].get_text(strip=True) if len(tds) > 4 else '牡3'
            sex = sex_age[0] if sex_age else '牡'
            age = int(sex_age[1:]) if len(sex_age) > 1 and sex_age[1:].isdigit() else 3

            # Weight carry
            kinryo = 55.0
            if len(tds) > 5:
                try:
                    kinryo = float(tds[5].get_text(strip=True))
                except:
                    pass

            # Horse weight
            hw = 480
            if len(tds) > 14:
                hw_text = tds[14].get_text(strip=True)
                hw_m = re.match(r'(\d{3,})', hw_text)
                if hw_m:
                    hw = int(hw_m.group(1))

            # Odds
            odds_val = 15.0
            if len(tds) > 9:
                try:
                    ov = float(tds[9].get_text(strip=True))
                    if 1.0 <= ov <= 999:
                        odds_val = ov
                except:
                    pass

            # Father
            father = ''
            if len(tds) > 12:
                father_link = tds[12].find("a") if tds[12].find("a") else None
                if father_link:
                    father = father_link.get_text(strip=True)

            COURSE_MAP_NAR = {
                '大井':10,'川崎':11,'船橋':12,'浦和':13,'園田':14,'姫路':15,'門別':16,
                '盛岡':17,'水沢':18,'金沢':19,'笠松':20,'名古屋':21,'高知':22,'佐賀':23,
            }
            SURFACE_MAP = {'芝':0,'ダ':1,'障':2}
            COND_MAP = {'良':0,'稍':1,'稍重':1,'重':2,'不':3,'不良':3}
            SEX_MAP = {'牡':0,'牝':1,'セ':2,'騸':2}

            horses.append({
                'horse_name': horse_name,
                'umaban': umaban,
                'horse_weight': hw,
                'weight_carry': kinryo,
                'age': age,
                'distance': distance,
                'course_enc': COURSE_MAP_NAR.get(course_name, 10),
                'surface_enc': SURFACE_MAP.get(surface, 1),
                'condition_enc': COND_MAP.get(condition, 0),
                'sex_enc': SEX_MAP.get(sex, 0),
                'horse_num': umaban,
                'bracket': min(8, max(1, (umaban - 1) * 8 // max(1, len(rows) - 1) + 1)),
                'num_horses': 14,  # placeholder, update below
                'odds': odds_val,
                'father': father,
                'jockey_name': jockey_name,
                'finish': finish,
            })

        if len(horses) < 5:
            return None

        # Update num_horses
        for h in horses:
            h['num_horses'] = len(horses)
            h['bracket'] = min(8, max(1, (h['umaban'] - 1) * 8 // max(1, len(horses)) + 1))

        # Payouts
        payouts = scrape_payouts(race_id, is_nar=True)

        race_meta = {
            'race_id': race_id,
            'date': '',
            'course': course_name,
            'distance': distance,
            'surface': surface,
            'condition': condition,
            'num_horses': len(horses),
        }

        return horses, actual_finishes, payouts, race_meta

    except Exception:
        return None


def predict_nar_race(horses, models):
    """Build features for NAR horses and predict with V8/V9."""
    df = pd.DataFrame(horses)

    # Build derived features (using defaults for past race data)
    df['jockey_wr_calc'] = 0.08
    df['jockey_course_wr_calc'] = 0.08
    df['trainer_top3_calc'] = 0.25
    df['prev_finish'] = 5
    df['prev_last3f'] = 35.5
    df['prev_pass4'] = 8
    df['prev_prize'] = 0
    df['prev2_finish'] = 5
    df['prev3_finish'] = 5
    df['avg_finish_3r'] = 5.0
    df['best_finish_3r'] = 5
    df['finish_trend'] = 0
    df['top3_count_3r'] = 0
    df['avg_last3f_3r'] = 35.5
    df['prev2_last3f'] = 35.5
    df['dist_change'] = 0
    df['dist_change_abs'] = 0
    df['rest_days'] = 30
    df['rest_category'] = 2
    df['sire_enc'] = 100
    df['bms_enc'] = 100
    df['dist_cat'] = pd.cut(df['distance'], bins=[0,1200,1400,1800,2200,9999],
                            labels=[0,1,2,3,4]).astype(float).fillna(2)
    df['weight_cat'] = pd.cut(df['horse_weight'], bins=[0,440,480,520,9999],
                              labels=[0,1,2,3]).astype(float).fillna(1)
    df['age_sex'] = df['age'] * 10 + df['sex_enc']
    df['season'] = 2  # Oct-Dec = autumn
    df['age_season'] = df['age'] * 10 + df['season']
    df['horse_num_ratio'] = df['horse_num'] / df['num_horses'].clip(1)
    df['bracket_pos'] = pd.cut(df['bracket'], bins=[0,3,6,8], labels=[0,1,2]).astype(float).fillna(1)
    df['carry_diff'] = df['weight_carry'] - df['weight_carry'].mean()
    df['weight_cat_dist'] = df['weight_cat'] * 10 + df['dist_cat']
    df['age_group'] = df['age'].clip(2, 7)
    df['surface_dist_enc'] = df['surface_enc'] * 10 + df['dist_cat']
    df['cond_surface'] = df['condition_enc'] * 10 + df['surface_enc']
    df['course_surface'] = df['course_enc'] * 10 + df['surface_enc']
    df['location_enc'] = 2  # NAR
    df['is_nar'] = 1
    df['odds_log'] = np.log1p(df['odds'].clip(1, 999))
    df['prev_odds_log'] = np.log1p(df['odds'].clip(1, 999))  # Use current odds as proxy

    rankings = {}
    v8_data = models.get('v8')
    v9_data = models.get('v9n') or models.get('v9c')

    for f in V8_FEATURES + ['odds_log', 'prev_odds_log']:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    if v8_data:
        X_v8 = df[V8_FEATURES].values
        scores = predict_v8(v8_data, X_v8, V8_FEATURES)
        df['v8_score'] = scores
        rankings['v8'] = df.sort_values('v8_score', ascending=False)['umaban'].tolist()

    if v9_data:
        X_v9 = df[V9_FEATURES].values
        scores = predict_v9_ensemble(v9_data, X_v9, V9_FEATURES)
        df['v9_score'] = scores
        rankings['v9'] = df.sort_values('v9_score', ascending=False)['umaban'].tolist()

    return rankings


def run_nar_backtest(models, existing_nar=None):
    """Run NAR backtest by scraping races from netkeiba."""
    print("\n" + "="*60)
    print("  NAR BACKTEST (scraping)")
    print("="*60)

    if existing_nar and len(existing_nar) >= N_NAR:
        print(f"  Using cached NAR results ({len(existing_nar)} races)")
        return existing_nar

    # Collect race IDs
    nar_races = collect_nar_race_ids(N_NAR)
    if not nar_races:
        print("  No NAR races found")
        return []

    print(f"  Processing {len(nar_races)} NAR races...")
    results = []
    for ri, race_info in enumerate(nar_races):
        rid = race_info['race_id']
        race_data = scrape_nar_race(rid)
        if not race_data:
            continue

        horses, actual_finishes, payouts, race_meta = race_data
        rankings = predict_nar_race(horses, models)

        # Build finish_to_uma
        finish_to_uma = {}
        for uma, fin in actual_finishes.items():
            if fin <= 3:
                finish_to_uma[fin] = uma

        result_entry = {**race_meta, 'actual_top3': finish_to_uma, 'payouts': payouts}
        if payouts.get('trio', 0) > 0:
            result_entry['trio_payout'] = payouts['trio']

        for ver, ranking in rankings.items():
            trio_bets, wide_bets, umaren_bets = calc_bets(ranking)
            trio_hit, trio_combo, wide_hits, umaren_hits = check_hits(
                finish_to_uma, trio_bets, wide_bets, umaren_bets
            )
            result_entry[f'{ver}_top3'] = ranking[:3]
            result_entry[f'{ver}_top6'] = ranking[:6]
            result_entry[f'{ver}_trio_bets'] = trio_bets
            result_entry[f'{ver}_trio_hit'] = trio_hit
            result_entry[f'{ver}_wide_bets'] = wide_bets
            result_entry[f'{ver}_wide_hits'] = [list(w) for w in wide_hits]
            result_entry[f'{ver}_umaren_bets'] = umaren_bets
            result_entry[f'{ver}_umaren_hits'] = [list(u) for u in umaren_hits]

            # Match payouts
            if wide_hits and payouts.get('wide'):
                total_wp = 0
                for hc in wide_hits:
                    for pc, pp in payouts['wide']:
                        if sorted(hc) == sorted(pc):
                            total_wp += pp
                            break
                result_entry[f'{ver}_wide_payout'] = total_wp

            if umaren_hits and payouts.get('umaren'):
                total_up = 0
                for hc in umaren_hits:
                    for pc, pp in payouts['umaren']:
                        if sorted(hc) == sorted(pc):
                            total_up += pp
                            break
                result_entry[f'{ver}_umaren_payout'] = total_up

        results.append(result_entry)

        if (ri + 1) % 10 == 0:
            print(f"    {ri+1}/{len(nar_races)} processed ({len(results)} valid)")
        time.sleep(1.5)

    print(f"  NAR: {len(results)} valid races")
    return results


# ===== Analysis =====
def analyze_results(results, label=""):
    """Analyze backtest results and return summary."""
    print(f"\n{'='*60}")
    print(f"  RESULTS: {label}")
    print(f"{'='*60}")
    n = len(results)
    if n == 0:
        print("  No results")
        return {}

    summary = {}
    for ver in ['v8', 'v9']:
        trio_hits = sum(1 for r in results if r.get(f'{ver}_trio_hit', False))
        wide_hit_races = sum(1 for r in results if len(r.get(f'{ver}_wide_hits', [])) > 0)
        umaren_hit_races = sum(1 for r in results if len(r.get(f'{ver}_umaren_hits', [])) > 0)

        # Count total bets and hits
        total_trio_bets = sum(len(r.get(f'{ver}_trio_bets', [])) for r in results)
        total_wide_bets = sum(len(r.get(f'{ver}_wide_bets', [])) for r in results)
        total_umaren_bets = sum(len(r.get(f'{ver}_umaren_bets', [])) for r in results)

        # Payouts
        trio_payouts = sum(r.get('trio_payout', 0) for r in results if r.get(f'{ver}_trio_hit'))
        trio_investment = total_trio_bets * 100
        wide_payouts = sum(r.get(f'{ver}_wide_payout', 0) for r in results)
        wide_investment = total_wide_bets * 100
        umaren_payouts = sum(r.get(f'{ver}_umaren_payout', 0) for r in results)
        umaren_investment = total_umaren_bets * 100

        trio_roi = (trio_payouts / trio_investment * 100) if trio_investment > 0 else 0
        wide_roi = (wide_payouts / wide_investment * 100) if wide_investment > 0 else 0
        umaren_roi = (umaren_payouts / umaren_investment * 100) if umaren_investment > 0 else 0

        s = {
            'n_races': n,
            'trio_hit_rate': trio_hits / n * 100,
            'trio_hits': trio_hits,
            'trio_investment': trio_investment,
            'trio_payout': trio_payouts,
            'trio_roi': trio_roi,
            'wide_hit_rate': wide_hit_races / n * 100,
            'wide_hits': wide_hit_races,
            'wide_investment': wide_investment,
            'wide_payout': wide_payouts,
            'wide_roi': wide_roi,
            'umaren_hit_rate': umaren_hit_races / n * 100,
            'umaren_hits': umaren_hit_races,
            'umaren_investment': umaren_investment,
            'umaren_payout': umaren_payouts,
            'umaren_roi': umaren_roi,
        }
        summary[ver] = s

        print(f"\n  [{ver.upper()}] {n} races")
        print(f"  -------------------------------------------------------")
        print(f"  {'Bet Type':<20} {'Hits':>6} {'Rate':>8} {'Invest':>10} {'Payout':>10} {'ROI':>8}")
        print(f"  -------------------------------------------------------")
        print(f"  {'Trio 7-bet':<20} {trio_hits:>5}{'':1} {trio_hits/n*100:>7.1f}% {trio_investment:>9,} {trio_payouts:>9,} {trio_roi:>7.1f}%")
        print(f"  {'Wide 1ax-2flow':<20} {wide_hit_races:>5}{'':1} {wide_hit_races/n*100:>7.1f}% {wide_investment:>9,} {wide_payouts:>9,} {wide_roi:>7.1f}%")
        print(f"  {'Umaren 1ax-2flow':<20} {umaren_hit_races:>5}{'':1} {umaren_hit_races/n*100:>7.1f}% {umaren_investment:>9,} {umaren_payouts:>9,} {umaren_roi:>7.1f}%")

    return summary


def analyze_losses(results, ver='v8'):
    """Analyze patterns in losing races."""
    print(f"\n  LOSS PATTERN ANALYSIS [{ver.upper()}]")
    print(f"  {'='*50}")

    losses = [r for r in results if not r.get(f'{ver}_trio_hit', False)]
    wins = [r for r in results if r.get(f'{ver}_trio_hit', False)]

    if not losses:
        print("  No losses!")
        return {}

    def avg_field(lst, key):
        vals = [r[key] for r in lst if key in r and r[key] is not None]
        if not vals: return 0
        if isinstance(vals[0], (int, float)):
            return sum(vals) / len(vals)
        return 'N/A'

    # Group by num_horses
    print(f"\n  By Field Size:")
    for label, lo, hi in [('Small (5-9)', 5, 9), ('Medium (10-14)', 10, 14), ('Large (15-18)', 15, 18)]:
        l_count = sum(1 for r in losses if lo <= r.get('num_horses', 0) <= hi)
        w_count = sum(1 for r in wins if lo <= r.get('num_horses', 0) <= hi)
        total = l_count + w_count
        rate = w_count / total * 100 if total > 0 else 0
        print(f"    {label:<20} {w_count}/{total} = {rate:.1f}% hit")

    # Group by distance
    print(f"\n  By Distance:")
    for label, lo, hi in [('Sprint (<=1400)', 0, 1400), ('Mile (1401-1800)', 1401, 1800),
                           ('Mid (1801-2200)', 1801, 2200), ('Long (2201+)', 2201, 9999)]:
        l_count = sum(1 for r in losses if lo <= r.get('distance', 0) <= hi)
        w_count = sum(1 for r in wins if lo <= r.get('distance', 0) <= hi)
        total = l_count + w_count
        rate = w_count / total * 100 if total > 0 else 0
        print(f"    {label:<20} {w_count}/{total} = {rate:.1f}% hit")

    # Group by surface
    print(f"\n  By Surface:")
    for surf in ['turf', 'dirt']:
        s_key = '芝' if surf == 'turf' else 'ダ'
        l_count = sum(1 for r in losses if s_key in str(r.get('surface', '')))
        w_count = sum(1 for r in wins if s_key in str(r.get('surface', '')))
        total = l_count + w_count
        rate = w_count / total * 100 if total > 0 else 0
        print(f"    {surf:<20} {w_count}/{total} = {rate:.1f}% hit")

    # Group by condition
    print(f"\n  By Track Condition:")
    for cond in ['良', '稍', '重', '不']:
        l_count = sum(1 for r in losses if cond in str(r.get('condition', '')))
        w_count = sum(1 for r in wins if cond in str(r.get('condition', '')))
        total = l_count + w_count
        rate = w_count / total * 100 if total > 0 else 0
        print(f"    {cond:<20} {w_count}/{total} = {rate:.1f}% hit")

    patterns = {
        'by_field_size': {},
        'by_distance': {},
        'by_surface': {},
        'by_condition': {},
    }
    return patterns


# ===== Main =====
def main():
    print("KEIBA AI V8/V9 Leak-Free Backtest")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)

    # Check for existing results
    existing = None
    if os.path.exists(RESULT_PATH):
        try:
            with open(RESULT_PATH, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            print(f"  Found existing results: {RESULT_PATH}")
            if existing.get('central_results'):
                print(f"    Central: {len(existing['central_results'])} races")
            if existing.get('nar_results'):
                print(f"    NAR: {len(existing['nar_results'])} races")
        except:
            existing = None

    models = load_models()

    # Phase 1: Central backtest (CSV-based, fast)
    central_results = None
    if existing and existing.get('central_results') and len(existing['central_results']) >= N_CENTRAL:
        print(f"\n  Using cached central results ({len(existing['central_results'])} races)")
        central_results = existing['central_results']
    else:
        central_results = run_central_backtest(models)

    # Phase 2: Scrape payouts for central races
    has_valid_payouts = any(r.get('payouts', {}).get('trio', 0) > 0 for r in central_results)
    if not has_valid_payouts:
        print("\n  Scraping payouts for central races...")
        central_results = scrape_payouts_for_results(central_results, is_nar=False)
    else:
        print("  Payouts already scraped")

    # Analyze central
    central_summary = analyze_results(central_results, "CENTRAL (JRA) Oct-Dec 2025")
    for ver in ['v8', 'v9']:
        analyze_losses(central_results, ver)

    # Save intermediate
    save_data = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'central_results': central_results,
        'central_summary': central_summary,
        'nar_results': existing.get('nar_results') if existing else None,
        'nar_summary': existing.get('nar_summary') if existing else None,
    }
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Saved to {RESULT_PATH}")

    # Phase 3: NAR backtest (scraping-based)
    nar_results = existing.get('nar_results') if existing else None
    if '--nar' in sys.argv or (nar_results and len(nar_results) >= N_NAR):
        if nar_results and len(nar_results) >= N_NAR:
            print(f"\n  Using cached NAR results ({len(nar_results)} races)")
        else:
            nar_results = run_nar_backtest(models, nar_results)

        if nar_results:
            nar_summary = analyze_results(nar_results, "NAR Oct-Dec 2025")
            for ver in ['v8', 'v9']:
                analyze_losses(nar_results, ver)

            # Save with NAR
            save_data['nar_results'] = nar_results
            save_data['nar_summary'] = nar_summary
            with open(RESULT_PATH, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n  Saved (with NAR) to {RESULT_PATH}")
    else:
        print("\n  NAR backtest: run with --nar flag (scraping ~100 races, ~30 min)")

    print("\n" + "="*60)
    print("  BACKTEST COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
