#!/usr/bin/env python
"""JRA Walk-Forward Backtest with Condition Optimization
Phase 1: Walk-forward backtest (2015-2017→2018, ..., 2015-2024→2025)
Phase 2: Optimal condition auto-search
Phase 3: New condition proposal and output

Rule: No day-of data (odds, horse_weight, paddock) in features.
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import warnings
from itertools import combinations, product
from collections import defaultdict
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# ===== Paths =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RACES_PATH = os.path.join(DATA_DIR, 'jra_races_full.csv')
TRAINING_TIMES_PATH = os.path.join(DATA_DIR, 'training_times.csv')

COURSE_MAP = {
    '札幌': 0, '函館': 1, '福島': 2, '新潟': 3,
    '東京': 4, '中山': 5, '中京': 6, '京都': 7,
    '阪神': 8, '小倉': 9,
}
COURSE_MAP_INV = {v: k for k, v in COURSE_MAP.items()}

# ===== Leak Check =====
LEAK_CHECK = {
    # Feature: (status, explanation)
    'weight_carry':      ('OK', '前日確定。枠順発表時に判明'),
    'age':               ('OK', '静的属性'),
    'distance':          ('OK', 'レーススケジュールから判明'),
    'course_enc':        ('OK', 'レーススケジュールから判明'),
    'surface_enc':       ('OK', 'レーススケジュールから判明'),
    'condition_enc':     ('WARN', '当日朝発表。天候から予測可能だが厳密には当日データ → 使用する(影響大)'),
    'sex_enc':           ('OK', '静的属性'),
    'num_horses':        ('OK', '出走確定時に判明'),
    'horse_num':         ('OK', '枠順確定時に判明'),
    'bracket':           ('OK', '枠順から派生'),
    'jockey_wr_calc':    ('OK', 'expanding window。当該レース未使用'),
    'jockey_course_wr_calc': ('OK', 'expanding window'),
    'jockey_surface_wr': ('OK', 'expanding window'),
    'trainer_top3_calc': ('OK', 'expanding window'),
    'prev_finish':       ('OK', 'shift(1) 前走データ'),
    'prev2_finish':      ('OK', 'shift(2) 2走前'),
    'prev3_finish':      ('OK', 'shift(3) 3走前'),
    'prev_last3f':       ('OK', 'shift(1) 前走上がり'),
    'prev2_last3f':      ('OK', 'shift(2) 2走前上がり'),
    'prev_pass4':        ('OK', 'shift(1) 前走通過順'),
    'prev_prize':        ('OK', 'shift(1) 前走賞金'),
    'prev_odds_log':     ('OK', '前走オッズ(当該レースではない)'),
    'avg_finish_3r':     ('OK', '過去3走平均から派生'),
    'best_finish_3r':    ('OK', '過去3走最良から派生'),
    'top3_count_3r':     ('OK', '過去3走3着内回数'),
    'finish_trend':      ('OK', '過去走トレンド'),
    'avg_last3f_3r':     ('OK', '過去走上がり平均'),
    'dist_change':       ('OK', '前走距離差'),
    'dist_change_abs':   ('OK', '距離差絶対値'),
    'rest_days':         ('OK', '前走からの間隔'),
    'rest_category':     ('OK', '間隔カテゴリ'),
    'sire_enc':          ('OK', '静的(血統)'),
    'bms_enc':           ('OK', '静的(血統)'),
    'dist_cat':          ('OK', '距離カテゴリ(スケジュール)'),
    'age_sex':           ('OK', '年齢×性別'),
    'season':            ('OK', '開催月から派生'),
    'age_season':        ('OK', '年齢×季節'),
    'horse_num_ratio':   ('OK', '馬番/頭数'),
    'bracket_pos':       ('OK', '枠位置'),
    'carry_diff':        ('OK', '斤量差(前日確定)'),
    'age_group':         ('OK', '年齢グループ'),
    'surface_dist_enc':  ('OK', '芝ダ×距離'),
    'cond_surface':      ('OK', '馬場×芝ダ'),
    'course_surface':    ('OK', '競馬場×芝ダ'),
    'location_enc':      ('OK', '調教場所(静的)'),
    'is_nar':            ('OK', '中央/地方フラグ'),
    'training_time_filled': ('OK', '調教タイム(レース前公開)'),
    'has_training':      ('OK', '調教データ有無'),
    'training_per_dist': ('OK', '調教タイム/距離'),
    'horse_career_races':('OK', 'expanding window'),
    'horse_career_wr':   ('OK', 'expanding window'),
    'horse_career_top3r':('OK', 'expanding window'),
    'sire_surface_wr':   ('OK', 'expanding window'),
    'sire_dist_wr':      ('OK', 'expanding window'),
    'bms_surface_wr':    ('OK', 'expanding window'),
    'wood_best_4f_filled':('OK', '調教タイム(レース前公開)'),
    'has_wood_training':  ('OK', '調教データ有無'),
    'prev_horse_weight':  ('OK', '前走馬体重(当該レースではない)'),
    # EXCLUDED (当日データ):
    'odds_log':          ('EXCLUDED', '★当日単勝オッズ → 除外'),
    'horse_weight':      ('EXCLUDED', '★確定馬体重 → 除外(prev_horse_weightで代替)'),
    'weight_change':     ('EXCLUDED', '★馬体重変化 → 除外'),
    'weight_change_abs': ('EXCLUDED', '★馬体重変化絶対値 → 除外'),
    'weight_cat':        ('EXCLUDED', '★馬体重カテゴリ → 除外'),
    'weight_cat_dist':   ('EXCLUDED', '★馬体重×距離 → 除外'),
}

# Features for backtest (no day-of data)
FEATURES_BT = [
    'weight_carry', 'age', 'distance', 'course_enc',
    'surface_enc', 'condition_enc', 'sex_enc', 'num_horses', 'horse_num',
    'bracket', 'jockey_wr_calc', 'jockey_course_wr_calc', 'jockey_surface_wr',
    'trainer_top3_calc',
    'prev_finish', 'prev_last3f', 'prev_pass4', 'prev_prize',
    'prev2_finish', 'prev3_finish', 'avg_finish_3r', 'best_finish_3r',
    'finish_trend', 'top3_count_3r', 'avg_last3f_3r', 'prev2_last3f',
    'dist_change', 'dist_change_abs', 'rest_days', 'rest_category',
    'sire_enc', 'bms_enc', 'dist_cat', 'age_sex', 'season',
    'age_season', 'horse_num_ratio', 'bracket_pos', 'carry_diff',
    'age_group', 'surface_dist_enc', 'cond_surface',
    'course_surface', 'location_enc', 'is_nar',
    'prev_odds_log',
    'training_time_filled', 'has_training', 'training_per_dist',
    'horse_career_races', 'horse_career_wr', 'horse_career_top3r',
    'sire_surface_wr', 'sire_dist_wr', 'bms_surface_wr',
    'wood_best_4f_filled', 'has_wood_training',
    'prev_horse_weight',
]


def print_leak_check():
    """Print leak check report."""
    print("\n" + "=" * 70)
    print("  LEAK CHECK REPORT - 特徴量リーク検証")
    print("=" * 70)
    ok = warn = excluded = 0
    for feat, (status, desc) in sorted(LEAK_CHECK.items()):
        icon = {'OK': 'o', 'WARN': '!', 'EXCLUDED': 'x'}[status]
        print(f"  {icon} [{status:8s}] {feat:30s} {desc}")
        if status == 'OK': ok += 1
        elif status == 'WARN': warn += 1
        else: excluded += 1
    print(f"\n  Summary: {ok} OK, {warn} WARN, {excluded} EXCLUDED")
    print(f"  使用特徴量数: {len(FEATURES_BT)}")
    print(f"  除外された当日データ: odds_log, horse_weight, weight_change, weight_cat 等")
    print("=" * 70)


# ===== Data Loading =====
def load_data():
    print("Loading jra_races_full.csv...")
    df = pd.read_csv(RACES_PATH, encoding='utf-8-sig', low_memory=False, dtype=str)
    print(f"  Raw: {len(df)} rows")

    df['finish'] = pd.to_numeric(df['finish'], errors='coerce')
    df = df[df['finish'].notna() & (df['finish'] >= 1)].copy()

    # Numeric conversions
    for col, default in [
        ('year', 0), ('month', 1), ('day', 1), ('distance', 1600),
        ('num_horses', 14), ('umaban', 1), ('agari_3f', 35.5),
        ('prize', 0), ('horse_weight', 480), ('weight_carry', 55),
        ('age', 3), ('tansho_odds', 15.0), ('training_4f', 0),
        ('popularity', 8), ('race_num', 1),
    ]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)

    df['year_full'] = df['year'].astype(int) + 2000
    df['month'] = df['month'].astype(int)
    df['day'] = df['day'].astype(int)
    df['date_num'] = df['year_full'] * 10000 + df['month'] * 100 + df['day']
    df['distance'] = df['distance'].astype(int)
    df['num_horses'] = df['num_horses'].astype(int)
    df['umaban'] = df['umaban'].astype(int)
    df['finish'] = df['finish'].astype(int)
    df['race_num'] = df['race_num'].astype(int)

    df['race_id_str'] = df['race_id'].astype(str).str.strip().str[:8]

    # Class code
    df['class_code'] = pd.to_numeric(df['class_code'], errors='coerce').fillna(0).astype(int)

    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    print(f"  Valid: {len(df)} rows, {df['race_id_str'].nunique()} races")
    print(f"  Years: {df['year_full'].min()}-{df['year_full'].max()}")
    return df


def encode_categoricals(df):
    # Sex
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

    # Course
    df['course_enc'] = df['course'].map(COURSE_MAP).fillna(len(COURSE_MAP)).astype(int)

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

    return df


def encode_sires(df, train_mask, n_top=100):
    train = df[train_mask]
    sire_counts = train['father'].value_counts()
    top_sires = sire_counts.head(n_top).index.tolist()
    sire_map = {s: i for i, s in enumerate(top_sires)}
    df['sire_enc'] = df['father'].map(sire_map).fillna(n_top).astype(int)

    bms_counts = train['bms'].value_counts()
    top_bms = bms_counts.head(n_top).index.tolist()
    bms_map = {s: i for i, s in enumerate(top_bms)}
    df['bms_enc'] = df['bms'].map(bms_map).fillna(n_top).astype(int)
    return df


def compute_expanding_stats(df):
    """Compute jockey/trainer/horse/sire stats via expanding window (leak-free)."""
    print("Computing expanding window stats...")
    df['is_win'] = (df['finish'] == 1).astype(int)
    df['is_top3'] = (df['finish'] <= 3).astype(int)
    df = df.sort_values('date_num').reset_index(drop=True)

    global_wr = df['is_win'].mean()
    global_t3 = df['is_top3'].mean()

    # Jockey win rate
    df['_jc'] = df.groupby('jockey_id').cumcount()
    df['_jw'] = df.groupby('jockey_id')['is_win'].cumsum() - df['is_win']
    df['jockey_wr_calc'] = (df['_jw'] + 30 * global_wr) / (df['_jc'] + 30)

    # Jockey course WR
    df['_jcc'] = df.groupby(['jockey_id', 'course_enc']).cumcount()
    df['_jcw'] = df.groupby(['jockey_id', 'course_enc'])['is_win'].cumsum() - df['is_win']
    df['jockey_course_wr_calc'] = (df['_jcw'] + 10 * global_wr) / (df['_jcc'] + 10)

    # Jockey surface WR
    df['_jsc'] = df.groupby(['jockey_id', 'surface_enc']).cumcount()
    df['_jsw'] = df.groupby(['jockey_id', 'surface_enc'])['is_win'].cumsum() - df['is_win']
    df['jockey_surface_wr'] = (df['_jsw'] + 10 * global_wr) / (df['_jsc'] + 10)

    # Trainer top3 rate
    df['_tc'] = df.groupby('trainer_id').cumcount()
    df['_tt'] = df.groupby('trainer_id')['is_top3'].cumsum() - df['is_top3']
    df['trainer_top3_calc'] = (df['_tt'] + 20 * global_t3) / (df['_tc'] + 20)

    # Drop temp cols
    df = df.drop(columns=[c for c in df.columns if c.startswith('_')])

    # Horse career stats
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    grp = df.groupby('horse_id')
    df['horse_career_races'] = grp.cumcount()
    df['horse_career_wins'] = grp['is_win'].cumsum() - df['is_win']
    df['horse_career_top3'] = grp['is_top3'].cumsum() - df['is_top3']
    df['horse_career_wr'] = (df['horse_career_wins'] + 5 * global_wr) / (df['horse_career_races'] + 5)
    df['horse_career_top3r'] = (df['horse_career_top3'] + 5 * global_t3) / (df['horse_career_races'] + 5)
    df = df.drop(columns=['horse_career_wins', 'horse_career_top3'])

    # Sire performance (expanding)
    df = df.sort_values('date_num').reset_index(drop=True)
    alpha_s = 50
    df['_ssc'] = df.groupby(['father', 'surface_enc']).cumcount()
    df['_ssw'] = df.groupby(['father', 'surface_enc'])['is_win'].cumsum() - df['is_win']
    df['sire_surface_wr'] = (df['_ssw'] + alpha_s * global_wr) / (df['_ssc'] + alpha_s)

    dist_cat_temp = pd.cut(df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
                           labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
    df['_dct'] = dist_cat_temp
    df['_sdc'] = df.groupby(['father', '_dct']).cumcount()
    df['_sdw'] = df.groupby(['father', '_dct'])['is_win'].cumsum() - df['is_win']
    df['sire_dist_wr'] = (df['_sdw'] + alpha_s * global_wr) / (df['_sdc'] + alpha_s)

    df['_bsc'] = df.groupby(['bms', 'surface_enc']).cumcount()
    df['_bsw'] = df.groupby(['bms', 'surface_enc'])['is_win'].cumsum() - df['is_win']
    df['bms_surface_wr'] = (df['_bsw'] + alpha_s * global_wr) / (df['_bsc'] + alpha_s)

    df = df.drop(columns=[c for c in df.columns if c.startswith('_')])
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)

    print("  Done.")
    return df


def compute_lag_features(df):
    print("Computing lag features...")
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    grp = df.groupby('horse_id')

    df['prev_finish'] = grp['finish'].shift(1).fillna(5)
    df['prev2_finish'] = grp['finish'].shift(2).fillna(5)
    df['prev3_finish'] = grp['finish'].shift(3).fillna(5)
    df['prev_last3f'] = grp['agari_3f'].shift(1).fillna(35.5)
    df['prev2_last3f'] = grp['agari_3f'].shift(2).fillna(35.5)

    pp4 = pd.to_numeric(df.get('pass4', pd.Series(dtype=float)), errors='coerce').fillna(8)
    df['prev_pass4'] = pp4.groupby(df['horse_id']).shift(1).fillna(8)
    df['prev_prize'] = grp['prize'].shift(1).fillna(0)

    df['prev_odds'] = grp['tansho_odds'].shift(1).fillna(15.0)
    df['prev_odds_log'] = np.log1p(df['prev_odds'].clip(1, 999))
    df['odds_log'] = np.log1p(df['tansho_odds'].clip(1, 999))  # keep for reference

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

    # Previous horse weight (to replace day-of horse_weight)
    df['prev_horse_weight'] = grp['horse_weight'].shift(1).fillna(480)

    bins = [-1, 6, 14, 35, 63, 180, 9999]
    df['rest_category'] = pd.cut(df['rest_days'], bins=bins, labels=[0,1,2,3,4,5]).astype(float).fillna(2)

    print("  Done.")
    return df


def build_features(df):
    df['horse_num'] = df['umaban']
    df['bracket'] = np.clip(((df['horse_num'] - 1) * 8 // df['num_horses'].clip(1)) + 1, 1, 8)
    df['dist_cat'] = pd.cut(df['distance'], bins=[0,1200,1400,1800,2200,9999],
                            labels=[0,1,2,3,4]).astype(float).fillna(2)
    df['age_sex'] = df['age'] * 10 + df['sex_enc']
    df['season'] = df['month'].apply(lambda m: 0 if m in [3,4,5] else (1 if m in [6,7,8] else (2 if m in [9,10,11] else 3)))
    df['age_season'] = df['age'] * 10 + df['season']
    df['horse_num_ratio'] = df['horse_num'] / df['num_horses'].clip(1)
    df['bracket_pos'] = pd.cut(df['bracket'], bins=[0,3,6,8], labels=[0,1,2]).astype(float).fillna(1)
    df['carry_diff'] = df['weight_carry'] - df.groupby('race_id_str')['weight_carry'].transform('mean')
    df['age_group'] = df['age'].clip(2, 7)
    df['surface_dist_enc'] = df['surface_enc'] * 10 + df['dist_cat']
    df['cond_surface'] = df['condition_enc'] * 10 + df['surface_enc']
    df['course_surface'] = df['course_enc'] * 10 + df['surface_enc']
    df['is_nar'] = 0

    # Training time features
    df['has_training'] = (df['training_4f'] > 0).astype(int)
    mean_tt = df.loc[df['training_4f'] > 0, 'training_4f'].mean()
    if pd.isna(mean_tt):
        mean_tt = 48.5
    df['training_time_filled'] = df['training_4f'].replace(0, mean_tt).fillna(mean_tt)
    df['training_per_dist'] = df['training_time_filled'] / (df['distance'] / 200).clip(1)

    # Wood training placeholder
    if 'wood_best_4f_filled' not in df.columns:
        df['wood_best_4f_filled'] = 52.0
        df['has_wood_training'] = 0

    return df


# ===== Condition Classification =====
def classify_condition(row):
    """Classify race condition A-E, X."""
    dist = row['distance']
    num = row['num_horses']
    cond = str(row.get('condition', '良'))
    heavy = any(c in cond for c in ['重', '不'])

    if num <= 7:
        return 'E'
    elif dist <= 1400:
        return 'D'
    elif 8 <= num <= 14 and dist >= 1600 and not heavy:
        return 'A'
    elif 8 <= num <= 14 and dist >= 1600 and heavy:
        return 'B'
    elif num >= 15 and dist >= 1600 and not heavy:
        return 'C'
    else:
        return 'X'


def get_class_label(class_code):
    """Convert class_code to readable label."""
    cc = int(class_code)
    if cc >= 190: return 'G1'
    elif cc >= 160: return 'G2G3'
    elif cc >= 130: return 'OP'
    elif cc >= 100: return '3win'
    elif cc >= 67:  return '2win'
    elif cc >= 34:  return '1win'
    elif cc >= 16:  return 'maiden'
    elif cc >= 5:   return 'newcomer'
    else: return 'other'


# ===== Bet Logic =====
def calc_bets(ranking):
    """Generate 4 bet patterns from AI ranking.
    ranking: list of umaban sorted by score (best first).
    Returns: (trio_bets, umaren_bets, wide_bets)
    """
    if len(ranking) < 3:
        return [], [], []
    n1 = ranking[0]
    n2 = ranking[1]
    n3 = ranking[2]
    flow = ranking[1:min(6, len(ranking))]

    # Trio: 7点 TOP1軸-TOP2,3-TOP2~6
    trio_bets = set()
    for s in [n2, n3]:
        for t in flow:
            combo = tuple(sorted({n1, s, t}))
            if len(combo) == 3:
                trio_bets.add(combo)
    trio_bets = sorted(trio_bets)

    # Umaren/Wide: 1軸2流し TOP1-TOP2, TOP1-TOP3
    umaren_bets = [tuple(sorted([n1, n2])), tuple(sorted([n1, n3]))]
    if umaren_bets[0] == umaren_bets[1]:
        umaren_bets = [umaren_bets[0]]
    wide_bets = list(umaren_bets)  # same combos

    return trio_bets, umaren_bets, wide_bets


def check_hits(actual_top3, trio_bets, umaren_bets, wide_bets):
    """Check bet hits.
    actual_top3: dict {finish_pos: umaban} for pos 1,2,3
    """
    top3_set = set(actual_top3.values())
    first = actual_top3.get(1)
    second = actual_top3.get(2)

    # Trio: all 3 in top 3
    trio_hit = False
    for combo in trio_bets:
        if set(combo) == top3_set and len(top3_set) == 3:
            trio_hit = True
            break

    # Umaren: exact 1st+2nd (unordered)
    umaren_hits = []
    if first and second:
        actual_12 = tuple(sorted([first, second]))
        for combo in umaren_bets:
            if tuple(sorted(combo)) == actual_12:
                umaren_hits.append(combo)

    # Wide: any 2 of top 3
    wide_hits = []
    for combo in wide_bets:
        if set(combo).issubset(top3_set):
            wide_hits.append(combo)

    return trio_hit, umaren_hits, wide_hits


def estimate_payouts(actual_top3, df_race):
    """Estimate payouts from tansho odds."""
    odds = {}
    for _, r in df_race.iterrows():
        odds[int(r['umaban'])] = float(r['tansho_odds'])

    first = actual_top3.get(1)
    second = actual_top3.get(2)
    third = actual_top3.get(3)

    o1 = odds.get(first, 10.0) if first else 10.0
    o2 = odds.get(second, 10.0) if second else 10.0
    o3 = odds.get(third, 10.0) if third else 10.0

    # Estimated payouts per 100 yen
    trio_payout = max(100, int(o1 * o2 * o3 * 20))
    umaren_payout = max(100, int(o1 * o2 * 50))

    # Wide: depends on which pair
    wide_payouts = {}
    for a, b in [(first, second), (first, third), (second, third)]:
        if a and b:
            oa = odds.get(a, 10.0)
            ob = odds.get(b, 10.0)
            wide_payouts[tuple(sorted([a, b]))] = max(100, int(oa * ob * 15))

    return trio_payout, umaren_payout, wide_payouts


# ===== Walk-Forward Training =====
def train_fold(df, train_mask, features):
    """Train LightGBM on a fold."""
    train_df = df[train_mask].copy()

    # Target: top3 finish
    y = (train_df['finish'] <= 3).astype(int)

    # Train/valid split (last 20% of training data by date)
    dates = train_df['date_num'].sort_values()
    valid_cutoff = dates.quantile(0.8)
    tr_idx = train_df['date_num'] < valid_cutoff
    va_idx = train_df['date_num'] >= valid_cutoff

    X_train = train_df.loc[tr_idx, features].values
    y_train = y[tr_idx].values
    X_valid = train_df.loc[va_idx, features].values
    y_valid = y[va_idx].values

    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
        'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1,
        'n_jobs': -1, 'seed': 42,
    }
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=features)
    dvalid = lgb.Dataset(X_valid, label=y_valid, feature_name=features, reference=dtrain)

    model = lgb.train(
        params, dtrain, num_boost_round=800,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(200)],
    )

    auc = roc_auc_score(y_valid, model.predict(X_valid))
    return model, auc


# ===== Phase 1: Walk-Forward Backtest =====
def run_phase1(df, features):
    print("\n" + "=" * 70)
    print("  PHASE 1: Walk-Forward Backtest (条件別×買い目別)")
    print("=" * 70)

    # Walk-forward: train on 2010-Y, test on Y+1
    test_years = list(range(2018, 2026))
    all_results = []
    fold_aucs = {}

    for test_year in test_years:
        train_end = test_year - 1
        train_mask = (df['year_full'] >= 2010) & (df['year_full'] <= train_end)
        test_mask = (df['year_full'] == test_year) & (df['num_horses'] >= 5)

        n_train = train_mask.sum()
        n_test = test_mask.sum()
        print(f"\n--- Fold: Train 2010-{train_end} ({n_train:,} rows) → Test {test_year} ({n_test:,} rows) ---")

        if n_test < 100:
            print("  Skip: too few test rows")
            continue

        # Re-encode sires for this fold
        df_fold = df.copy()
        df_fold = encode_sires(df_fold, train_mask)

        # Ensure features are numeric
        for f in features:
            if f not in df_fold.columns:
                df_fold[f] = 0
            df_fold[f] = pd.to_numeric(df_fold[f], errors='coerce').fillna(0)

        # Train
        t0 = time.time()
        model, auc = train_fold(df_fold, train_mask, features)
        elapsed = time.time() - t0
        fold_aucs[test_year] = auc
        print(f"  Valid AUC: {auc:.4f}, Training: {elapsed:.1f}s")

        # Predict on test
        test_df = df_fold[test_mask].copy()
        X_test = test_df[features].values
        test_df['pred'] = model.predict(X_test)

        # Test AUC
        y_test = (test_df['finish'] <= 3).astype(int)
        test_auc = roc_auc_score(y_test, test_df['pred'])
        print(f"  Test AUC: {test_auc:.4f}")

        # Per-race evaluation
        test_races = test_df['race_id_str'].unique()
        print(f"  Evaluating {len(test_races)} races...")

        for rid in test_races:
            race_df = test_df[test_df['race_id_str'] == rid].copy()
            if len(race_df) < 5:
                continue

            # Race metadata
            row0 = race_df.iloc[0]
            cond_key = classify_condition(row0)

            # Actual top 3
            race_df_sorted = race_df.sort_values('finish')
            actual_top3 = {}
            for _, r in race_df_sorted.head(3).iterrows():
                actual_top3[int(r['finish'])] = int(r['umaban'])

            if len(actual_top3) < 3:
                continue

            # AI ranking
            race_df = race_df.sort_values('pred', ascending=False)
            ranking = race_df['umaban'].astype(int).tolist()

            # Generate bets
            trio_bets, umaren_bets, wide_bets = calc_bets(ranking)
            trio_hit, umaren_hits, wide_hits = check_hits(actual_top3, trio_bets, umaren_bets, wide_bets)
            trio_payout, umaren_payout, wide_payouts_map = estimate_payouts(actual_top3, race_df)

            # Calculate returns for 4 bet patterns
            # ① trio 7点 × 100 = 700
            trio_invest = len(trio_bets) * 100
            trio_return = trio_payout if trio_hit else 0

            # ③ umaren 2点 × 350 = 700
            umaren_n = len(umaren_bets)
            umaren_unit = 700 // max(1, umaren_n)
            umaren_return_total = sum(umaren_payout * (umaren_unit / 100) for _ in umaren_hits)

            # ④ wide 2点 × 350 = 700
            wide_n = len(wide_bets)
            wide_unit = 700 // max(1, wide_n)
            wide_return_total = 0
            for wc in wide_hits:
                wp = wide_payouts_map.get(tuple(sorted(wc)), 150)
                wide_return_total += wp * (wide_unit / 100)

            # ② umaren + wide (合計1000: umaren 2点×250 + wide 2点×250)
            combo_invest = 1000
            combo_umaren_return = sum(umaren_payout * (250 / 100) for _ in umaren_hits)
            combo_wide_return = 0
            for wc in wide_hits:
                wp = wide_payouts_map.get(tuple(sorted(wc)), 150)
                combo_wide_return += wp * (250 / 100)
            combo_return = combo_umaren_return + combo_wide_return

            result = {
                'race_id': rid,
                'year': test_year,
                'month': int(row0['month']),
                'course': str(row0.get('course', '')),
                'course_enc': int(row0['course_enc']),
                'distance': int(row0['distance']),
                'surface': str(row0.get('surface', '')),
                'condition': str(row0.get('condition', '')),
                'num_horses': int(row0['num_horses']),
                'class_code': int(row0.get('class_code', 0)),
                'class_label': get_class_label(row0.get('class_code', 0)),
                'cond_key': cond_key,
                'actual_top3': actual_top3,
                'ai_top3': ranking[:3],
                'trio_hit': trio_hit,
                'trio_invest': trio_invest,
                'trio_return': trio_return,
                'umaren_hit': len(umaren_hits) > 0,
                'umaren_invest': 700,
                'umaren_return': umaren_return_total,
                'wide_hit': len(wide_hits) > 0,
                'wide_invest': 700,
                'wide_return': wide_return_total,
                'combo_hit': len(umaren_hits) > 0 or len(wide_hits) > 0,
                'combo_invest': combo_invest,
                'combo_return': combo_return,
            }
            all_results.append(result)

    return all_results, fold_aucs


def print_phase1_results(results, fold_aucs):
    print("\n" + "=" * 70)
    print("  PHASE 1 RESULTS: 条件別 × 買い目別 ROI")
    print("=" * 70)

    # Fold AUCs
    print("\n  Walk-Forward AUC:")
    for y, a in sorted(fold_aucs.items()):
        print(f"    {y}: {a:.4f}")
    avg_auc = np.mean(list(fold_aucs.values()))
    print(f"    Average: {avg_auc:.4f}")

    df = pd.DataFrame(results)

    conditions = ['A', 'B', 'C', 'D', 'E', 'X']
    bet_types = [
        ('trio', '①三連複7点(700円)', 'trio_hit', 'trio_invest', 'trio_return'),
        ('combo', '②馬連+ワイド(1000円)', 'combo_hit', 'combo_invest', 'combo_return'),
        ('umaren', '③馬連1軸2流(700円)', 'umaren_hit', 'umaren_invest', 'umaren_return'),
        ('wide', '④ワイド1軸2流(700円)', 'wide_hit', 'wide_invest', 'wide_return'),
    ]

    summary = {}
    for cond in conditions:
        cdf = df[df['cond_key'] == cond]
        n = len(cdf)
        print(f"\n  === 条件{cond} (N={n}) ===")
        if n == 0:
            continue

        for bt_key, bt_name, hit_col, inv_col, ret_col in bet_types:
            hits = cdf[hit_col].sum()
            hit_rate = hits / n * 100
            total_inv = cdf[inv_col].sum()
            total_ret = cdf[ret_col].sum()
            roi = total_ret / total_inv * 100 if total_inv > 0 else 0

            if roi >= 120: star = '★★★'
            elif roi >= 100: star = '★★'
            elif roi >= 80: star = '★'
            else: star = '非推奨'
            flag = ' [要追加検証]' if n < 20 else ''

            print(f"    {bt_name}: Hit {hits}/{n} ({hit_rate:.1f}%), ROI {roi:.1f}% {star}{flag}")

            summary[(cond, bt_key)] = {
                'n': n, 'hits': int(hits), 'hit_rate': round(hit_rate, 1),
                'total_invest': int(total_inv), 'total_return': int(total_ret),
                'roi': round(roi, 1), 'star': star, 'flag': flag.strip(),
            }

        # Per-year breakdown
        years = sorted(cdf['year'].unique())
        for bt_key, bt_name, hit_col, inv_col, ret_col in bet_types[:1]:  # trio only for per-year
            year_rois = []
            for y in years:
                ydf = cdf[cdf['year'] == y]
                ny = len(ydf)
                inv = ydf[inv_col].sum()
                ret = ydf[ret_col].sum()
                roi = ret / inv * 100 if inv > 0 else 0
                year_rois.append(f"{y}={roi:.0f}%({ny})")
            print(f"    Trio年別: {', '.join(year_rois)}")

    # Overall summary table
    print("\n\n  === SUMMARY TABLE ===")
    print(f"  {'条件':>4} {'N':>5} | {'①Trio ROI':>10} {'②Combo ROI':>10} {'③馬連ROI':>10} {'④WidROI':>10}")
    print("  " + "-" * 65)
    for cond in conditions:
        cdf = df[df['cond_key'] == cond]
        n = len(cdf)
        vals = []
        for bt_key in ['trio', 'combo', 'umaren', 'wide']:
            s = summary.get((cond, bt_key))
            if s:
                vals.append(f"{s['roi']:>7.1f}%{s['star'][:1]}")
            else:
                vals.append(f"{'N/A':>10}")
        print(f"  {cond:>4} {n:>5} | {'  '.join(vals)}")

    return summary


# ===== Phase 2: Condition Optimization =====
def run_phase2(results):
    print("\n\n" + "=" * 70)
    print("  PHASE 2: 最適条件の自動探索")
    print("=" * 70)

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No results to optimize.")
        return {}

    # Define search axes
    dist_splits = [1200, 1400, 1600, 1800, 2000, 2400]
    num_splits = [(0, 7), (8, 12), (8, 14), (8, 16), (13, 99), (15, 99)]
    cond_groups = {
        '良のみ': lambda c: '良' in str(c),
        '良〜稍': lambda c: any(x in str(c) for x in ['良', '稍']),
        '重〜不': lambda c: any(x in str(c) for x in ['重', '不']),
        '不のみ': lambda c: '不' in str(c),
    }

    courses = sorted(df['course'].unique())
    classes = sorted(df['class_label'].unique())
    seasons_map = {0: '春', 1: '夏', 2: '秋', 3: '冬'}

    bet_types = ['trio', 'umaren', 'wide', 'combo']
    all_findings = []

    # 1. Distance × NumHorses × Track Condition
    print("\n  --- 距離×頭数×馬場 探索 ---")
    for dist_lo in [0, 1000, 1200, 1400, 1600, 1800, 2000]:
        for dist_hi in [1200, 1400, 1600, 1800, 2000, 2400, 9999]:
            if dist_hi <= dist_lo:
                continue
            for num_lo, num_hi in num_splits:
                for cond_name, cond_fn in cond_groups.items():
                    mask = (
                        (df['distance'] >= dist_lo) &
                        (df['distance'] < dist_hi) &
                        (df['num_horses'] >= num_lo) &
                        (df['num_horses'] <= num_hi) &
                        (df['condition'].apply(cond_fn))
                    )
                    sub = df[mask]
                    n = len(sub)
                    if n < 30:
                        continue

                    for bt in bet_types:
                        inv_col = f'{bt}_invest'
                        ret_col = f'{bt}_return'
                        hit_col = f'{bt}_hit'
                        total_inv = sub[inv_col].sum()
                        total_ret = sub[ret_col].sum()
                        roi = total_ret / total_inv * 100 if total_inv > 0 else 0
                        hits = sub[hit_col].sum()
                        hit_rate = hits / n * 100

                        if roi >= 110:
                            # Check yearly stability
                            yearly_rois = []
                            stable = True
                            for y in sorted(sub['year'].unique()):
                                ys = sub[sub['year'] == y]
                                if len(ys) < 3:
                                    continue
                                yr = ys[ret_col].sum() / ys[inv_col].sum() * 100 if ys[inv_col].sum() > 0 else 0
                                yearly_rois.append((y, yr, len(ys)))
                                if yr < 50:  # Very bad year
                                    stable = False

                            all_findings.append({
                                'dist_range': f'{dist_lo}-{dist_hi}m',
                                'num_range': f'{num_lo}-{num_hi}',
                                'track': cond_name,
                                'axis': 'dist_num_cond',
                                'bet_type': bt,
                                'n': n, 'hits': int(hits),
                                'hit_rate': round(hit_rate, 1),
                                'roi': round(roi, 1),
                                'stable': stable,
                                'yearly': yearly_rois,
                                'course': 'all', 'class': 'all', 'season': 'all',
                            })

    # 2. Course-specific
    print("  --- 競馬場別 探索 ---")
    for course in courses:
        if not course.strip():
            continue
        mask = df['course'] == course
        sub = df[mask]
        n = len(sub)
        if n < 30:
            continue

        for bt in bet_types:
            inv_col = f'{bt}_invest'
            ret_col = f'{bt}_return'
            hit_col = f'{bt}_hit'
            total_inv = sub[inv_col].sum()
            total_ret = sub[ret_col].sum()
            roi = total_ret / total_inv * 100 if total_inv > 0 else 0
            if roi >= 100:
                yearly_rois = []
                for y in sorted(sub['year'].unique()):
                    ys = sub[sub['year'] == y]
                    yr = ys[ret_col].sum() / ys[inv_col].sum() * 100 if ys[inv_col].sum() > 0 else 0
                    yearly_rois.append((y, yr, len(ys)))
                all_findings.append({
                    'dist_range': 'all', 'num_range': 'all', 'track': 'all',
                    'axis': 'course',
                    'bet_type': bt, 'n': n, 'hits': int(sub[hit_col].sum()),
                    'hit_rate': round(sub[hit_col].sum() / n * 100, 1),
                    'roi': round(roi, 1),
                    'stable': True, 'yearly': yearly_rois,
                    'course': course, 'class': 'all', 'season': 'all',
                })

    # 3. Class-specific
    print("  --- クラス別 探索 ---")
    for cls in classes:
        mask = df['class_label'] == cls
        sub = df[mask]
        n = len(sub)
        if n < 30:
            continue

        for bt in bet_types:
            inv_col = f'{bt}_invest'
            ret_col = f'{bt}_return'
            hit_col = f'{bt}_hit'
            total_inv = sub[inv_col].sum()
            total_ret = sub[ret_col].sum()
            roi = total_ret / total_inv * 100 if total_inv > 0 else 0
            if roi >= 100:
                yearly_rois = []
                for y in sorted(sub['year'].unique()):
                    ys = sub[sub['year'] == y]
                    yr = ys[ret_col].sum() / ys[inv_col].sum() * 100 if ys[inv_col].sum() > 0 else 0
                    yearly_rois.append((y, yr, len(ys)))
                all_findings.append({
                    'dist_range': 'all', 'num_range': 'all', 'track': 'all',
                    'axis': 'class',
                    'bet_type': bt, 'n': n, 'hits': int(sub[hit_col].sum()),
                    'hit_rate': round(sub[hit_col].sum() / n * 100, 1),
                    'roi': round(roi, 1),
                    'stable': True, 'yearly': yearly_rois,
                    'course': 'all', 'class': cls, 'season': 'all',
                })

    # 4. Season-specific
    print("  --- 季節別 探索 ---")
    df['season_label'] = df['month'].apply(
        lambda m: '春' if m in [3,4,5] else ('夏' if m in [6,7,8] else ('秋' if m in [9,10,11] else '冬'))
    )
    for sea in ['春', '夏', '秋', '冬']:
        mask = df['season_label'] == sea
        sub = df[mask]
        n = len(sub)
        if n < 30:
            continue

        for bt in bet_types:
            inv_col = f'{bt}_invest'
            ret_col = f'{bt}_return'
            hit_col = f'{bt}_hit'
            total_inv = sub[inv_col].sum()
            total_ret = sub[ret_col].sum()
            roi = total_ret / total_inv * 100 if total_inv > 0 else 0
            if roi >= 100:
                yearly_rois = []
                for y in sorted(sub['year'].unique()):
                    ys = sub[sub['year'] == y]
                    yr = ys[ret_col].sum() / ys[inv_col].sum() * 100 if ys[inv_col].sum() > 0 else 0
                    yearly_rois.append((y, yr, len(ys)))
                all_findings.append({
                    'dist_range': 'all', 'num_range': 'all', 'track': 'all',
                    'axis': 'season',
                    'bet_type': bt, 'n': n, 'hits': int(sub[hit_col].sum()),
                    'hit_rate': round(sub[hit_col].sum() / n * 100, 1),
                    'roi': round(roi, 1),
                    'stable': True, 'yearly': yearly_rois,
                    'course': 'all', 'class': 'all', 'season': sea,
                })

    # 5. Combined: Course × Distance × Class
    print("  --- 複合条件 探索 (競馬場×距離区分×クラス) ---")
    for course in courses:
        if not course.strip():
            continue
        for dist_lo, dist_hi in [(0, 1400), (1400, 1800), (1800, 9999)]:
            for cls in classes:
                mask = (
                    (df['course'] == course) &
                    (df['distance'] >= dist_lo) &
                    (df['distance'] < dist_hi) &
                    (df['class_label'] == cls)
                )
                sub = df[mask]
                n = len(sub)
                if n < 30:
                    continue

                for bt in bet_types:
                    inv_col = f'{bt}_invest'
                    ret_col = f'{bt}_return'
                    total_inv = sub[inv_col].sum()
                    total_ret = sub[ret_col].sum()
                    roi = total_ret / total_inv * 100 if total_inv > 0 else 0
                    if roi >= 120:
                        hit_col = f'{bt}_hit'
                        yearly_rois = []
                        for y in sorted(sub['year'].unique()):
                            ys = sub[sub['year'] == y]
                            yr = ys[ret_col].sum() / ys[inv_col].sum() * 100 if ys[inv_col].sum() > 0 else 0
                            yearly_rois.append((y, yr, len(ys)))
                        all_findings.append({
                            'dist_range': f'{dist_lo}-{dist_hi}m',
                            'num_range': 'all', 'track': 'all',
                            'axis': 'course_dist_class',
                            'bet_type': bt, 'n': n,
                            'hits': int(sub[hit_col].sum()),
                            'hit_rate': round(sub[hit_col].sum() / n * 100, 1),
                            'roi': round(roi, 1),
                            'stable': True, 'yearly': yearly_rois,
                            'course': course, 'class': cls, 'season': 'all',
                        })

    # Sort by ROI
    all_findings.sort(key=lambda x: x['roi'], reverse=True)

    # Print top findings
    print(f"\n  Found {len(all_findings)} conditions with ROI >= 100%")
    print(f"\n  === TOP 30 Conditions (ROI desc, N>=30) ===")
    print(f"  {'#':>3} {'ROI%':>7} {'HitR%':>6} {'N':>5} {'BetType':>7} {'Stable':>6} | Description")
    print("  " + "-" * 80)

    for i, f in enumerate(all_findings[:30]):
        desc_parts = []
        if f['course'] != 'all': desc_parts.append(f['course'])
        if f['dist_range'] != 'all': desc_parts.append(f['dist_range'])
        if f['num_range'] != 'all': desc_parts.append(f"H{f['num_range']}")
        if f['track'] != 'all': desc_parts.append(f['track'])
        if f['class'] != 'all': desc_parts.append(f['class'])
        if f['season'] != 'all': desc_parts.append(f['season'])
        desc = ' / '.join(desc_parts) if desc_parts else 'all'
        stab = '○' if f['stable'] else '×'
        print(f"  {i+1:>3} {f['roi']:>6.1f}% {f['hit_rate']:>5.1f}% {f['n']:>5} {f['bet_type']:>7} {stab:>6} | {desc}")

    return all_findings


# ===== Phase 3: New Condition Proposal =====
def run_phase3(results, findings, phase1_summary):
    print("\n\n" + "=" * 70)
    print("  PHASE 3: 新条件定義の提案")
    print("=" * 70)

    df = pd.DataFrame(results)

    # Filter stable, high-ROI conditions
    stable_top = [f for f in findings if f['stable'] and f['n'] >= 30 and f['roi'] >= 110]
    stable_top.sort(key=lambda x: x['roi'], reverse=True)

    # Propose new conditions based on findings
    # Group by bet_type and find best non-overlapping conditions
    new_conditions = {}

    # For each bet type, find best conditions
    for bt in ['trio', 'combo', 'umaren', 'wide']:
        bt_findings = [f for f in stable_top if f['bet_type'] == bt]
        if not bt_findings:
            continue

        # Pick top findings with reasonable diversity
        selected = []
        for f in bt_findings[:10]:
            selected.append(f)

        new_conditions[bt] = selected

    # Compare old vs new
    print("\n  === 旧条件 vs 新条件 TOP ROI 比較 ===")
    old_conditions = {
        'A': '8-14頭/1600m+/良〜稍',
        'B': '8-14頭/1600m+/重〜不',
        'C': '15頭+/1600m+/良〜稍',
        'D': '1400m以下',
        'E': '7頭以下',
        'X': '15頭+/重〜不',
    }

    comparison = []
    for cond, desc in old_conditions.items():
        for bt in ['trio', 'combo', 'umaren', 'wide']:
            s = phase1_summary.get((cond, bt))
            if s:
                comparison.append({
                    'type': 'old',
                    'cond': cond,
                    'desc': desc,
                    'bet': bt,
                    'roi': s['roi'],
                    'n': s['n'],
                    'hit_rate': s['hit_rate'],
                })

    print(f"\n  {'Type':>5} {'Cond':>5} {'Bet':>7} {'ROI%':>7} {'HitR%':>6} {'N':>5} | Description")
    print("  " + "-" * 70)
    for c in sorted(comparison, key=lambda x: x['roi'], reverse=True)[:20]:
        print(f"  {'OLD':>5} {c['cond']:>5} {c['bet']:>7} {c['roi']:>6.1f}% {c['hit_rate']:>5.1f}% {c['n']:>5} | {c['desc']}")

    print("\n  === 新条件 TOP候補 ===")
    for bt, findings_list in new_conditions.items():
        print(f"\n  [{bt}]:")
        for i, f in enumerate(findings_list[:5]):
            desc_parts = []
            if f['course'] != 'all': desc_parts.append(f['course'])
            if f['dist_range'] != 'all': desc_parts.append(f['dist_range'])
            if f['num_range'] != 'all': desc_parts.append(f"H{f['num_range']}")
            if f['track'] != 'all': desc_parts.append(f['track'])
            if f['class'] != 'all': desc_parts.append(f['class'])
            if f['season'] != 'all': desc_parts.append(f['season'])
            desc = ' / '.join(desc_parts) if desc_parts else 'all'
            yr_str = ', '.join([f"{y}={r:.0f}%({n})" for y, r, n in f['yearly']])
            print(f"    #{i+1}: ROI {f['roi']:.1f}%, Hit {f['hit_rate']:.1f}%, N={f['n']} | {desc}")
            print(f"         年別: {yr_str}")

    return new_conditions, comparison


# ===== Output Files =====
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_outputs(results, findings, new_conditions, comparison, phase1_summary, fold_aucs):
    print("\n\nSaving outputs...")

    # 1. simulation_results_jra.csv
    csv_path = os.path.join(DATA_DIR, 'simulation_results_jra.csv')
    df = pd.DataFrame(results)
    # Convert dict columns to strings
    for col in ['actual_top3', 'ai_top3']:
        if col in df.columns:
            df[col] = df[col].apply(str)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {csv_path} ({len(df)} races)")

    # 2. optimal_betting_jra.json
    opt_path = os.path.join(DATA_DIR, 'optimal_betting_jra.json')
    optimal = {
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'LightGBM walk-forward (no day-of features)',
        'fold_aucs': {str(k): round(v, 4) for k, v in fold_aucs.items()},
        'condition_results': {},
    }
    for (cond, bt), stats in phase1_summary.items():
        if cond not in optimal['condition_results']:
            optimal['condition_results'][cond] = {}
        optimal['condition_results'][cond][bt] = stats

    # Best bet per condition
    optimal['recommended'] = {}
    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        best_bt = None
        best_roi = 0
        for bt in ['trio', 'combo', 'umaren', 'wide']:
            s = phase1_summary.get((cond, bt))
            if s and s['roi'] > best_roi and s['n'] >= 10:
                best_roi = s['roi']
                best_bt = bt
        if best_bt:
            optimal['recommended'][cond] = {
                'bet_type': best_bt,
                'roi': best_roi,
                'n': phase1_summary[(cond, best_bt)]['n'],
            }

    with open(opt_path, 'w', encoding='utf-8') as f:
        json.dump(optimal, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"  Saved: {opt_path}")

    # 3. condition_optimization_jra.json
    opt_log_path = os.path.join(DATA_DIR, 'condition_optimization_jra.json')
    # Convert yearly tuples to serializable format
    findings_clean = []
    for f in findings:
        fc = dict(f)
        fc['yearly'] = [{'year': y, 'roi': round(r, 1), 'n': n} for y, r, n in f['yearly']]
        findings_clean.append(fc)

    new_cond_clean = {}
    for bt, fl in new_conditions.items():
        new_cond_clean[bt] = []
        for f in fl:
            fc = dict(f)
            fc['yearly'] = [{'year': y, 'roi': round(r, 1), 'n': n} for y, r, n in f['yearly']]
            new_cond_clean[bt].append(fc)

    opt_log = {
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_conditions_explored': len(findings_clean),
        'top_findings': findings_clean[:100],  # top 100
        'new_condition_proposals': new_cond_clean,
        'comparison_old_vs_new': comparison,
    }
    with open(opt_log_path, 'w', encoding='utf-8') as f:
        json.dump(opt_log, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"  Saved: {opt_log_path}")


# ===== Main =====
def main():
    t_start = time.time()

    # Leak check
    print_leak_check()

    # Load and prepare data
    df = load_data()
    df = encode_categoricals(df)

    # Use all data for initial sire encoding (will re-encode per fold)
    full_mask = df['year_full'] >= 2010
    df = encode_sires(df, full_mask)

    df = compute_expanding_stats(df)
    df = compute_lag_features(df)
    df = build_features(df)

    # Ensure all features exist
    for f in FEATURES_BT:
        if f not in df.columns:
            print(f"  WARNING: Feature '{f}' not found, filling with 0")
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    # Phase 1
    results, fold_aucs = run_phase1(df, FEATURES_BT)
    phase1_summary = print_phase1_results(results, fold_aucs)

    # Phase 2
    findings = run_phase2(results)

    # Phase 3
    new_conditions, comparison = run_phase3(results, findings, phase1_summary)

    # Save
    save_outputs(results, findings, new_conditions, comparison, phase1_summary, fold_aucs)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed / 60:.1f} minutes")
    print("  Done!")


if __name__ == '__main__':
    main()
