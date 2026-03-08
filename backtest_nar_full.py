#!/usr/bin/env python
"""NAR Full Backtest: Walk-Forward + Condition Optimization

Phase 1: Merge KDSCOPE + netkeiba data, walk-forward backtest with leak checks
Phase 2: Automatic optimal condition search (with edge vs random baseline)
Phase 3: Output results for app.py update

Key Design:
- KDSCOPE data (2009-2020): 3540 races, avg 4.8 horses, NO odds
  → Used for within-KDSCOPE walk-forward (hit rate only, no ROI)
- Netkeiba data (2022): 184 races, normal fields, WITH odds + payouts
  → Used for ROI calculation with payout cache
- "Edge" = actual hit rate - random expected hit rate (accounts for field size)
"""
import os
import json
import pickle
import warnings
from datetime import datetime
from collections import defaultdict
from math import comb

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KDSCOPE_CSV = os.path.join(BASE_DIR, 'data', 'chihou_races_full.csv')
NETKEIBA_CSV = os.path.join(BASE_DIR, 'data', 'chihou_races_2020_2025.csv')
CACHE_PATH = os.path.join(BASE_DIR, 'data', 'nar_scraped_cache.json')
MODEL_PATH = os.path.join(BASE_DIR, 'keiba_model_v9_nar.pkl')

SIM_CSV = os.path.join(BASE_DIR, 'data', 'simulation_results_nar.csv')
OPT_BET_JSON = os.path.join(BASE_DIR, 'data', 'optimal_betting_nar.json')
COND_OPT_JSON = os.path.join(BASE_DIR, 'data', 'condition_optimization_nar.json')
LOG_PATH = os.path.join(BASE_DIR, 'train', 'nar_full_backtest.log')

COURSE_MAP = {42: '浦和', 43: '船橋', 44: '大井', 45: '川崎'}

FEATURES_NO_ODDS = [
    'num_horses', 'distance', 'surface_enc', 'condition_enc',
    'course_enc', 'horse_weight', 'weight_carry', 'age', 'sex_enc',
    'horse_num', 'bracket', 'jockey_wr', 'jockey_place_rate', 'trainer_wr',
    'prev_finish', 'prev2_finish', 'prev3_finish', 'avg_finish_3r',
    'best_finish_3r', 'top3_count_3r', 'finish_trend',
    'rest_days', 'rest_category', 'dist_cat', 'weight_cat', 'age_group',
    'horse_num_ratio', 'bracket_pos', 'carry_diff', 'dist_change',
    'dist_change_abs', 'is_nar',
]

FEATURES_WITH_ODDS = FEATURES_NO_ODDS + ['odds_log', 'prev_odds_log', 'pop_rank']


def log(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode())
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


# ── LEAK CHECK ──────────────────────────────────────────────
LEAK_ITEMS = {
    'finish': 'LEAK: 着順', 'finish_pos': 'LEAK: 着順',
    'finish_time': 'LEAK: 走破タイム', 'finish_time_sec': 'LEAK: 走破タイム秒',
    'target': 'LEAK: 教師ラベル',
    'tansho_payout': 'LEAK: 単勝配当', 'umaren_payout': 'LEAK: 馬連配当',
    'trio_payout': 'LEAK: 三連複配当', 'wide_payout': 'LEAK: ワイド配当',
    'weight_change': 'CAUTION: 馬体重変化(当日)',
}

SAFE_ITEMS = {
    'odds_log': 'SAFE: 前日オッズ', 'prev_odds_log': 'SAFE: 前走オッズ',
    'pop_rank': 'SAFE: 人気順(前日)', 'num_horses': 'SAFE: 出走頭数',
    'distance': 'SAFE: 距離', 'surface_enc': 'SAFE: 馬場種別',
    'condition_enc': 'CAUTION: 馬場状態(当日朝発表)',
    'course_enc': 'SAFE: 競馬場', 'horse_weight': 'CAUTION: 馬体重(前走値)',
    'weight_carry': 'SAFE: 斤量', 'age': 'SAFE: 馬齢', 'sex_enc': 'SAFE: 性別',
    'horse_num': 'SAFE: 馬番', 'bracket': 'SAFE: 枠番',
    'jockey_wr': 'SAFE: 騎手勝率(expanding)', 'jockey_place_rate': 'SAFE: 騎手複勝率',
    'trainer_wr': 'SAFE: 調教師勝率(expanding)',
    'prev_finish': 'SAFE: 前走着順', 'prev2_finish': 'SAFE: 前々走着順',
    'prev3_finish': 'SAFE: 3走前着順', 'avg_finish_3r': 'SAFE: 直近3走平均',
    'best_finish_3r': 'SAFE: 直近3走最高', 'top3_count_3r': 'SAFE: 3着内回数',
    'finish_trend': 'SAFE: トレンド', 'rest_days': 'SAFE: 休養日数',
    'rest_category': 'SAFE: 休養区分', 'dist_cat': 'SAFE: 距離区分',
    'weight_cat': 'SAFE: 体重区分', 'age_group': 'SAFE: 馬齢G',
    'horse_num_ratio': 'SAFE: 馬番比', 'bracket_pos': 'SAFE: 枠位置',
    'carry_diff': 'SAFE: 斤量差', 'dist_change': 'SAFE: 距離変化',
    'dist_change_abs': 'SAFE: 距離変化abs', 'is_nar': 'SAFE: NARフラグ',
}


def run_leak_check(features):
    log("\n" + "=" * 60)
    log("  LEAK CHECK")
    log("=" * 60)
    leaks, cautions, safe = [], [], []
    for f in features:
        if f in LEAK_ITEMS:
            leaks.append(f)
            log(f"  [LEAK]    {f}: {LEAK_ITEMS[f]}")
        elif f in SAFE_ITEMS:
            s = SAFE_ITEMS[f]
            if 'CAUTION' in s:
                cautions.append(f)
                log(f"  [CAUTION] {f}: {s}")
            else:
                safe.append(f)
                log(f"  [OK]      {f}: {s}")
        else:
            cautions.append(f)
            log(f"  [UNKNOWN] {f}: 要手動確認")
    log(f"\n  結果: {len(safe)} safe, {len(cautions)} cautions, {len(leaks)} leaks")
    return len(leaks) == 0


# ── RANDOM EXPECTED HIT RATES ──────────────────────────────
def random_hit_rate(n_horses, bet_type, n_bets):
    """Expected hit rate by random chance."""
    if n_horses < 3:
        return 0
    if bet_type == 'trio':
        total_combos = comb(n_horses, 3)
        return min(n_bets / total_combos, 1.0) if total_combos > 0 else 0
    elif bet_type in ('umaren', 'wide'):
        if bet_type == 'umaren':
            total_combos = comb(n_horses, 2)
            return min(n_bets / total_combos, 1.0) if total_combos > 0 else 0
        else:  # wide: 2 bets, each can hit if pair is in top3
            total_combos = comb(n_horses, 2)
            # P(at least 1 of 2 wide bets hits) ≈ 1 - (1 - C(3,2)/C(n,2))^2
            p_single = comb(3, 2) / total_combos if total_combos > 0 else 0
            return 1 - (1 - p_single) ** n_bets
    elif bet_type == 'combo':
        # umaren OR wide hit
        p_u = random_hit_rate(n_horses, 'umaren', 2)
        p_w = random_hit_rate(n_horses, 'wide', 2)
        return 1 - (1 - p_u) * (1 - p_w)
    return 0


# ── DATA LOADING ────────────────────────────────────────────
def load_and_merge():
    log("\n" + "=" * 60)
    log("  PHASE 0: DATA LOADING & MERGING")
    log("=" * 60)

    # --- KDSCOPE ---
    kd = pd.read_csv(KDSCOPE_CSV, encoding='utf-8-sig')
    log(f"  KDSCOPE: {len(kd)} rows, {kd.groupby(['race_date','course_code','race_no']).ngroups} races")

    kd['race_date'] = pd.to_datetime(kd['race_date'])
    kd['race_id'] = (kd['race_date'].dt.strftime('%Y%m%d') +
                     kd['course_code'].astype(str).str.zfill(2) +
                     kd['race_no'].astype(str).str.zfill(2))

    # Derive finish_pos from time for all-zero races
    def fix_finish(g):
        if g['finish_pos'].eq(0).all() and g['finish_time_sec'].notna().all():
            g = g.copy()
            g['finish_pos'] = g['finish_time_sec'].rank(method='min').astype(int)
        return g
    kd = kd.groupby('race_id', group_keys=False).apply(fix_finish)

    # For mixed races: derive position from time for 0-pos rows with valid time
    zero_with_time = (kd['finish_pos'] == 0) & kd['finish_time_sec'].notna()
    for rid in kd[zero_with_time]['race_id'].unique():
        mask = kd['race_id'] == rid
        race = kd.loc[mask].copy()
        valid = race['finish_time_sec'].notna()
        if valid.any():
            race.loc[valid, 'finish_pos'] = race.loc[valid, 'finish_time_sec'].rank(method='min').astype(int)
            kd.loc[mask] = race

    # Remove DNF (finish_pos=0 with NaN time)
    kd = kd[kd['finish_pos'] > 0]
    kd['num_horses'] = kd.groupby('race_id')['horse_id'].transform('count')
    kd = kd[kd['num_horses'] >= 3]
    log(f"  After cleanup: {len(kd)} rows, {kd.race_id.nunique()} races")

    # Map columns
    kd['horse_num'] = kd['umaban'].astype(int)
    kd['horse_weight'] = kd['weight'].astype(float)
    kd['weight_carry'] = 55.0
    kd['age'] = kd['age'].astype(int)
    kd['distance'] = kd['distance'].astype(int)
    kd['surface'] = 'ダ'
    kd['condition'] = '良'
    kd['surface_enc'] = 1
    kd['condition_enc'] = 0
    kd['sex_enc'] = kd['sex'].map({'牡': 0, '牝': 1, 'セ': 2}).fillna(0).astype(int)
    kd['course_enc'] = kd['course_code']
    kd['course'] = kd['course_code'].map(COURSE_MAP)
    kd['finish'] = kd['finish_pos'].astype(int)
    kd['target'] = (kd['finish'] <= 3).astype(int)
    kd['odds'] = np.nan
    kd['pop_rank'] = 0
    kd['jockey_name'] = kd['jockey_name'].fillna('unknown')
    kd['bracket'] = ((kd['umaban'] - 1) // max(1, kd['num_horses'].iloc[0] / 8)).astype(int).clip(1, 8)
    kd['year'] = kd['race_date'].dt.year
    kd['month'] = kd['race_date'].dt.month
    kd['class_code'] = kd['class_code']
    kd['source'] = 'kdscope'

    # --- Netkeiba ---
    nk = pd.read_csv(NETKEIBA_CSV)
    nk['race_id'] = nk['race_id'].astype(str)
    nk['year'] = nk['race_id'].str[:4].astype(int)
    nk['month'] = nk['race_id'].str[4:6].astype(int)
    nk['race_date'] = pd.to_datetime(nk['race_id'].str[:8], format='%Y%m%d', errors='coerce')
    nk['finish'] = nk['finish'].astype(int)
    nk['source'] = 'netkeiba'
    nk['class_code'] = 0
    nk['horse_num'] = nk['umaban'].astype(int)
    log(f"  Netkeiba: {len(nk)} rows, {nk.race_id.nunique()} races")

    cols = [
        'race_id', 'race_date', 'year', 'month', 'horse_id', 'horse_name',
        'umaban', 'horse_num', 'horse_weight', 'weight_carry', 'age', 'sex_enc',
        'distance', 'surface', 'surface_enc', 'condition', 'condition_enc',
        'course_enc', 'course', 'num_horses', 'bracket',
        'odds', 'pop_rank', 'jockey_name', 'finish', 'target',
        'class_code', 'source',
    ]
    for c in cols:
        if c not in kd.columns:
            kd[c] = 0
        if c not in nk.columns:
            nk[c] = 0

    merged = pd.concat([kd[cols], nk[cols]], ignore_index=True)
    merged = merged.sort_values(['race_date', 'race_id', 'umaban']).reset_index(drop=True)
    log(f"  Merged: {len(merged)} rows, {merged.race_id.nunique()} races")

    # Stats
    for src in ['kdscope', 'netkeiba']:
        sub = merged[merged.source == src]
        avg_h = sub.groupby('race_id')['umaban'].count().mean()
        log(f"    {src}: {sub.race_id.nunique()} races, avg {avg_h:.1f} horses/race")

    return merged


# ── FEATURE ENGINEERING ─────────────────────────────────────
def engineer_features(df):
    log("\n  Engineering features (expanding window, leak-free)...")
    df = df.sort_values(['race_date', 'race_id', 'umaban']).reset_index(drop=True)

    jockey_rec = defaultdict(lambda: {'w': 0, 't3': 0, 'n': 0})
    trainer_rec = defaultdict(lambda: {'w': 0, 'n': 0})
    horse_hist = defaultdict(list)

    jwr, jpr, twr = [], [], []
    pf1, pf2, pf3, af3, bf3, t3c3, ftr, pol, rd, dc = ([] for _ in range(10))

    race_order = df.drop_duplicates('race_id')[['race_id', 'race_date']].sort_values('race_date')
    n_races = len(race_order)

    for i, (_, rr) in enumerate(race_order.iterrows()):
        rid, rdate = rr['race_id'], rr['race_date']
        rows = df[df['race_id'] == rid]

        for _, row in rows.iterrows():
            jn = row['jockey_name']
            jr = jockey_rec[jn]
            jwr.append(jr['w'] / max(jr['n'], 1))
            jpr.append(jr['t3'] / max(jr['n'], 1))
            twr.append(trainer_rec.get(jn, {'w': 0, 'n': 1})['w'] /
                       max(trainer_rec.get(jn, {'n': 1})['n'], 1))

            hid = row['horse_id']
            h = horse_hist[hid]
            pf1.append(h[-1]['f'] if len(h) >= 1 else 5)
            pf2.append(h[-2]['f'] if len(h) >= 2 else 5)
            pf3.append(h[-3]['f'] if len(h) >= 3 else 5)
            r3 = [x['f'] for x in h[-3:]] if h else [5]
            af3.append(np.mean(r3))
            bf3.append(min(r3))
            t3c3.append(sum(1 for f in r3 if f <= 3))
            ftr.append(r3[-1] - r3[0] if len(r3) >= 2 else 0)
            pol.append(h[-1].get('ol', 0) if h else 0)
            if h and pd.notna(rdate) and pd.notna(h[-1]['d']):
                rd.append(max((rdate - h[-1]['d']).days, 0))
            else:
                rd.append(30)
            dc.append(row['distance'] - h[-1]['dist'] if h else 0)

        # Update records AFTER feature computation
        for _, row in rows.iterrows():
            jn = row['jockey_name']
            jockey_rec[jn]['n'] += 1
            if row['finish'] == 1:
                jockey_rec[jn]['w'] += 1
            if row['finish'] <= 3:
                jockey_rec[jn]['t3'] += 1
            trainer_rec[jn]['n'] += 1
            if row['finish'] == 1:
                trainer_rec[jn]['w'] += 1
            horse_hist[row['horse_id']].append({
                'd': rdate, 'f': row['finish'], 'dist': row['distance'],
                'ol': np.log1p(row['odds']) if pd.notna(row['odds']) else 0,
            })

        if (i + 1) % 500 == 0:
            log(f"    {i+1}/{n_races} races...")

    df['jockey_wr'] = jwr
    df['jockey_place_rate'] = jpr
    df['trainer_wr'] = twr
    df['prev_finish'] = pf1
    df['prev2_finish'] = pf2
    df['prev3_finish'] = pf3
    df['avg_finish_3r'] = af3
    df['best_finish_3r'] = bf3
    df['top3_count_3r'] = t3c3
    df['finish_trend'] = ftr
    df['prev_odds_log'] = pol
    df['rest_days'] = rd
    df['dist_change'] = dc

    df['odds_log'] = np.log1p(df['odds'].clip(1, 999).fillna(10))
    df['dist_change_abs'] = df['dist_change'].abs()
    df['dist_cat'] = pd.cut(df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
                            labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
    df['weight_cat'] = pd.cut(df['horse_weight'], bins=[0, 440, 480, 520, 9999],
                              labels=[0, 1, 2, 3]).astype(float).fillna(1)
    df['age_group'] = df['age'].clip(2, 7)
    df['horse_num_ratio'] = df['horse_num'] / df['num_horses'].clip(1)
    df['bracket_pos'] = pd.cut(df['bracket'], bins=[0, 3, 6, 8],
                                labels=[0, 1, 2]).astype(float).fillna(1)
    df['carry_diff'] = df['weight_carry'] - 55.0
    df['rest_category'] = pd.cut(df['rest_days'], bins=[-1, 14, 35, 90, 9999],
                                  labels=[0, 1, 2, 3]).astype(float).fillna(1)
    df['is_nar'] = 1
    df['pop_rank'] = pd.to_numeric(df['pop_rank'], errors='coerce').fillna(0)

    log(f"  Done: {len(df)} rows")
    return df


# ── CONDITION CLASSIFICATION ────────────────────────────────
def classify_condition(nh, dist, cond):
    heavy = any(c in str(cond) for c in ['重', '不'])
    if nh <= 7:
        return 'E'
    if dist <= 1400:
        return 'D'
    if 8 <= nh <= 14 and dist >= 1600 and not heavy:
        return 'A'
    if 8 <= nh <= 14 and dist >= 1600 and heavy:
        return 'B'
    if nh >= 15 and dist >= 1600 and not heavy:
        return 'C'
    return 'X'


# ── BET GENERATION ──────────────────────────────────────────
def calc_bets(ranking):
    if len(ranking) < 3:
        return [], [], []
    nums = ranking[:6] if len(ranking) >= 6 else ranking
    n1 = nums[0]
    trio = sorted(set(
        tuple(sorted({n1, s, t}))
        for s in nums[1:3] for t in nums[1:min(6, len(nums))]
        if len(set({n1, s, t})) == 3
    ))
    uma = [tuple(sorted([n1, nums[1]])), tuple(sorted([n1, nums[2]]))]
    wide = [tuple(sorted([n1, nums[1]])), tuple(sorted([n1, nums[2]]))]
    return trio, wide, uma


def check_hits(actual, trio_bets, wide_bets, umaren_bets):
    top3 = set(u for u, f in actual.items() if f <= 3)
    top2 = set(u for u, f in actual.items() if f <= 2)
    trio_hit = any(set(c) == top3 for c in trio_bets) if len(top3) == 3 else False
    wide_hits = [b for b in wide_bets if set(b).issubset(top3)]
    uma_hits = [b for b in umaren_bets if set(b) == top2] if len(top2) == 2 else []
    return trio_hit, wide_hits, uma_hits


# ── BET TYPE DEFINITIONS ───────────────────────────────────
BET_DEFS = {
    'trio_7': {'name': '三連複7点', 'cost': 700, 'type': 'trio'},
    'uma_wide_combo': {'name': '馬連+ワイド1軸2流し', 'cost': 1000, 'type': 'combo'},
    'umaren_2': {'name': '馬連1軸2流し', 'cost': 700, 'type': 'umaren'},
    'wide_2': {'name': 'ワイド1軸2流し', 'cost': 700, 'type': 'wide'},
}


def calc_payout(bdef, trio_hit, wide_hits, uma_hits, payouts):
    bt = bdef['type']
    if bt == 'trio':
        return payouts.get('trio', 0) if trio_hit else 0
    elif bt == 'umaren':
        if uma_hits:
            p = payouts.get('umaren', 0)
            return int(p * 3.5) if p > 0 else 0  # 700/200=3.5x per 100yen
        return 0
    elif bt == 'wide':
        if wide_hits:
            wp = payouts.get('wide', [])
            if isinstance(wp, list) and wp:
                return int(wp[0] * 3.5)
            elif isinstance(wp, (int, float)) and wp > 0:
                return int(wp * 3.5)
        return 0
    elif bt == 'combo':
        total = 0
        if uma_hits:
            p = payouts.get('umaren', 0)
            if p > 0:
                total += int(p * 2.5)  # 1000/4=250 per bet, 250/100=2.5x
        if wide_hits:
            wp = payouts.get('wide', [])
            if isinstance(wp, list) and wp:
                total += int(wp[0] * 2.5)
            elif isinstance(wp, (int, float)) and wp > 0:
                total += int(wp * 2.5)
        return total
    return 0


# ── WALK-FORWARD BACKTEST ───────────────────────────────────
def walk_forward_backtest(df, features):
    log("\n" + "=" * 60)
    log("  PHASE 1: WALK-FORWARD BACKTEST")
    log("=" * 60)

    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    log(f"  Payout cache: {len(cache)} races")

    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    years = sorted(df['year'].unique())
    log(f"  Years: {years}")

    results = []
    aucs = []
    min_train = 3

    for test_yr in [y for y in years if y >= years[0] + min_train]:
        tr = df[df['year'] < test_yr]
        te = df[df['year'] == test_yr]
        if len(tr) < 100 or len(te) < 10:
            continue

        X_tr = tr[features].values
        y_tr = tr['target'].values
        X_te = te[features].values
        y_te = te['target'].values

        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'num_leaves': 15, 'learning_rate': 0.04, 'feature_fraction': 0.8,
            'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 10,
            'reg_alpha': 0.5, 'reg_lambda': 0.5, 'verbose': -1, 'seed': 42,
        }
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=features)
        dval = lgb.Dataset(X_te, label=y_te, feature_name=features, reference=dtrain)
        model = lgb.train(params, dtrain, 1000, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

        preds = model.predict(X_te)
        try:
            auc = roc_auc_score(y_te, preds)
        except ValueError:
            auc = 0.5
        aucs.append((test_yr, auc))
        log(f"  {test_yr}: train={tr.race_id.nunique()}R test={te.race_id.nunique()}R AUC={auc:.4f}")

        te = te.copy()
        te['score'] = preds

        for rid in te['race_id'].unique():
            rdf = te[te['race_id'] == rid].sort_values('score', ascending=False)
            if len(rdf) < 3:
                continue

            nh = int(rdf['num_horses'].iloc[0])
            dist = int(rdf['distance'].iloc[0])
            cond_str = rdf['condition'].iloc[0]
            src = rdf['source'].iloc[0]
            cenc = int(rdf['course_enc'].iloc[0])
            cname = rdf['course'].iloc[0] if 'course' in rdf.columns else ''
            cls = int(rdf['class_code'].iloc[0])
            month = int(rdf['month'].iloc[0])
            season = {3: '春', 4: '春', 5: '春', 6: '夏', 7: '夏', 8: '夏',
                      9: '秋', 10: '秋', 11: '秋'}.get(month, '冬')

            ck = classify_condition(nh, dist, cond_str)
            ranking = rdf['umaban'].astype(int).tolist()
            actual = dict(zip(rdf['umaban'].astype(int), rdf['finish'].astype(int)))
            trio_b, wide_b, uma_b = calc_bets(ranking)
            trio_hit, wide_hits, uma_hits = check_hits(actual, trio_b, wide_b, uma_b)

            rc = cache.get(str(rid), {})
            payouts = rc.get('payouts', {'trio': 0, 'umaren': 0, 'wide': []})
            has_pay = bool(rc)

            r = {
                'race_id': rid, 'year': test_yr, 'month': month, 'season': season,
                'source': src, 'condition': ck, 'num_horses': nh, 'distance': dist,
                'track_cond': cond_str, 'course_enc': cenc, 'course_name': str(cname),
                'class_code': cls,
                'trio_hit': trio_hit, 'wide_hit': len(wide_hits) > 0,
                'wide_count': len(wide_hits), 'umaren_hit': len(uma_hits) > 0,
                'has_payout': has_pay,
                # Random baselines
                'rand_trio': random_hit_rate(nh, 'trio', len(trio_b)),
                'rand_wide': random_hit_rate(nh, 'wide', 2),
                'rand_umaren': random_hit_rate(nh, 'umaren', 2),
            }
            for bk, bd in BET_DEFS.items():
                r[f'{bk}_cost'] = bd['cost']
                r[f'{bk}_payout'] = calc_payout(bd, trio_hit, wide_hits, uma_hits, payouts)
            results.append(r)

    return results, aucs


# ── RESULTS ANALYSIS ────────────────────────────────────────
def analyze_results(results, label):
    log(f"\n{'=' * 60}")
    log(f"  RESULTS: {label}")
    log(f"{'=' * 60}")
    if not results:
        return {}

    rdf = pd.DataFrame(results)
    log(f"  Total: {len(rdf)} races, Years: {sorted(rdf.year.unique())}")

    summary = {}
    for cond in ['A', 'B', 'C', 'D', 'E', 'X', 'ALL']:
        cdf = rdf if cond == 'ALL' else rdf[rdf.condition == cond]
        n = len(cdf)
        if n == 0:
            continue

        cs = {'n': n}
        avg_nh = cdf['num_horses'].mean()

        for bk, bd in BET_DEFS.items():
            bt = bd['type']
            if bt == 'trio':
                hits = int(cdf['trio_hit'].sum())
                rand_col = 'rand_trio'
            elif bt == 'umaren':
                hits = int(cdf['umaren_hit'].sum())
                rand_col = 'rand_umaren'
            elif bt == 'wide':
                hits = int(cdf['wide_hit'].sum())
                rand_col = 'rand_wide'
            elif bt == 'combo':
                hits = int((cdf['umaren_hit'] | cdf['wide_hit']).sum())
                rand_col = 'rand_wide'  # approximate
            else:
                hits = 0
                rand_col = 'rand_trio'

            hr = hits / n * 100 if n > 0 else 0
            rand_hr = cdf[rand_col].mean() * 100 if rand_col in cdf.columns else 0
            edge = hr - rand_hr

            # ROI on payout-available races
            pm = cdf['has_payout']
            np_ = int(pm.sum())
            if np_ > 0:
                cost = cdf.loc[pm, f'{bk}_cost'].sum()
                pay = cdf.loc[pm, f'{bk}_payout'].sum()
                roi = pay / cost * 100 if cost > 0 else 0
            else:
                roi = 0

            stars = 3 if roi >= 120 else (2 if roi >= 100 else (1 if roi >= 80 else 0))
            flag = '★★★' if stars == 3 else ('★★' if stars == 2 else ('★' if stars == 1 else '非推奨'))
            if n < 20:
                flag += ' (要追加検証)'

            cs[bk] = {
                'hits': hits, 'hit_rate': round(hr, 1), 'random_rate': round(rand_hr, 1),
                'edge': round(edge, 1), 'n': n, 'n_pay': np_,
                'roi': round(roi, 1), 'stars': stars, 'flag': flag,
            }
        summary[cond] = cs

    # Print
    log(f"\n  {'COND':<5} {'N':>5} {'AvgH':>5} | {'BET':<22} {'HIT':>5} {'RATE':>7} {'RAND':>7} {'EDGE':>7} {'ROI':>8} {'FLAG'}")
    log(f"  {'-' * 90}")
    for cond in ['A', 'B', 'C', 'D', 'E', 'X', 'ALL']:
        if cond not in summary:
            continue
        cs = summary[cond]
        cdf_c = rdf if cond == 'ALL' else rdf[rdf.condition == cond]
        avg_h = cdf_c['num_horses'].mean()
        first = True
        for bk in BET_DEFS:
            bs = cs[bk]
            roi_s = f"{bs['roi']:.1f}%" if bs['n_pay'] > 0 else 'N/A'
            if first:
                log(f"  {cond:<5} {cs['n']:>5} {avg_h:>5.1f} | {BET_DEFS[bk]['name']:<22} {bs['hits']:>5} {bs['hit_rate']:>6.1f}% {bs['random_rate']:>6.1f}% {bs['edge']:>+6.1f}% {roi_s:>8} {bs['flag']}")
                first = False
            else:
                log(f"  {'':5} {'':>5} {'':>5} | {BET_DEFS[bk]['name']:<22} {bs['hits']:>5} {bs['hit_rate']:>6.1f}% {bs['random_rate']:>6.1f}% {bs['edge']:>+6.1f}% {roi_s:>8} {bs['flag']}")

    # Yearly
    log(f"\n  --- Yearly Hit Rate (Trio) ---")
    for yr in sorted(rdf.year.unique()):
        ydf = rdf[rdf.year == yr]
        n_ = len(ydf)
        th = ydf['trio_hit'].mean() * 100
        wh = ydf['wide_hit'].mean() * 100
        uh = ydf['umaren_hit'].mean() * 100
        rt = ydf['rand_trio'].mean() * 100
        log(f"  {yr}: N={n_:>4}, trio={th:>5.1f}%(rand {rt:>4.1f}%), wide={wh:>5.1f}%, umaren={uh:>5.1f}%")

    return summary


# ── PHASE 2: OPTIMAL CONDITION SEARCH ───────────────────────
def optimize_conditions(results):
    log("\n" + "=" * 60)
    log("  PHASE 2: OPTIMAL CONDITION SEARCH")
    log("=" * 60)

    rdf = pd.DataFrame(results)
    if len(rdf) == 0:
        return {}

    # Search axes
    dist_bounds = [800, 1000, 1200, 1400, 1600, 1800, 2000]
    horse_filters = [
        ('<=5', lambda n: n <= 5), ('<=7', lambda n: n <= 7),
        ('6-8', lambda n: 6 <= n <= 8), ('6-10', lambda n: 6 <= n <= 10),
        ('8-10', lambda n: 8 <= n <= 10), ('8-12', lambda n: 8 <= n <= 12),
        ('8-14', lambda n: 8 <= n <= 14), ('9-12', lambda n: 9 <= n <= 12),
        ('10-14', lambda n: 10 <= n <= 14), ('11+', lambda n: n >= 11),
        ('ALL', lambda n: True),
    ]
    cond_filters = [
        ('良', lambda c: c in ['良', '']),
        ('良~稍', lambda c: c in ['良', '稍重', '稍', '']),
        ('重~不', lambda c: c in ['重', '不良', '不']),
        ('ALL', lambda c: True),
    ]
    season_filters = [
        ('春', lambda s: s == '春'), ('夏', lambda s: s == '夏'),
        ('秋', lambda s: s == '秋'), ('冬', lambda s: s == '冬'),
        ('ALL', lambda s: True),
    ]

    found = []
    total = 0

    for d_lo in dist_bounds:
        for d_hi in [d for d in dist_bounds if d >= d_lo] + [9999]:
            d_label = f"{d_lo}-{d_hi}m" if d_hi < 9999 else f"{d_lo}m+"
            for h_label, h_fn in horse_filters:
                for c_label, c_fn in cond_filters:
                    for s_label, s_fn in season_filters:
                        mask = (
                            (rdf['distance'] >= d_lo) &
                            (rdf['distance'] <= d_hi) &
                            rdf['num_horses'].apply(h_fn) &
                            rdf['track_cond'].apply(c_fn) &
                            rdf['season'].apply(s_fn)
                        )
                        sub = rdf[mask]
                        n = len(sub)
                        if n < 10:
                            continue
                        total += 1
                        avg_nh = sub['num_horses'].mean()

                        for bk, bd in BET_DEFS.items():
                            bt = bd['type']
                            if bt == 'trio':
                                hits = sub['trio_hit'].sum()
                                r_col = 'rand_trio'
                            elif bt == 'umaren':
                                hits = sub['umaren_hit'].sum()
                                r_col = 'rand_umaren'
                            elif bt == 'wide':
                                hits = sub['wide_hit'].sum()
                                r_col = 'rand_wide'
                            elif bt == 'combo':
                                hits = (sub['umaren_hit'] | sub['wide_hit']).sum()
                                r_col = 'rand_wide'
                            else:
                                continue

                            hr = hits / n * 100
                            rand_hr = sub[r_col].mean() * 100
                            edge = hr - rand_hr

                            # ROI
                            pm = sub['has_payout']
                            np_ = pm.sum()
                            if np_ > 0:
                                cost = sub.loc[pm, f'{bk}_cost'].sum()
                                pay = sub.loc[pm, f'{bk}_payout'].sum()
                                roi = pay / cost * 100 if cost > 0 else 0
                            else:
                                roi = 0

                            # Yearly stability
                            yr_edges = []
                            for yr in sorted(sub['year'].unique()):
                                ym = sub['year'] == yr
                                yn = ym.sum()
                                if yn >= 5:
                                    if bt == 'trio':
                                        yh = sub.loc[ym, 'trio_hit'].sum()
                                    elif bt == 'umaren':
                                        yh = sub.loc[ym, 'umaren_hit'].sum()
                                    elif bt == 'wide':
                                        yh = sub.loc[ym, 'wide_hit'].sum()
                                    else:
                                        yh = (sub.loc[ym, 'umaren_hit'] | sub.loc[ym, 'wide_hit']).sum()
                                    yr_hr = yh / yn * 100
                                    yr_rand = sub.loc[ym, r_col].mean() * 100
                                    yr_edges.append(yr_hr - yr_rand)

                            stability = np.std(yr_edges) if len(yr_edges) >= 2 else 999

                            # Filter: meaningful edge
                            if edge >= 5 or (np_ > 0 and roi >= 100):
                                found.append({
                                    'dist': d_label, 'horses': h_label, 'track': c_label,
                                    'season': s_label, 'bet': bk, 'bet_name': bd['name'],
                                    'n': n, 'avg_horses': round(avg_nh, 1),
                                    'hits': int(hits), 'hit_rate': round(hr, 1),
                                    'random_rate': round(rand_hr, 1), 'edge': round(edge, 1),
                                    'n_pay': int(np_), 'roi': round(roi, 1),
                                    'stability': round(stability, 1),
                                    'yearly_edges': yr_edges,
                                })

    log(f"  Searched {total} combos, found {len(found)} with edge>=5% or ROI>=100%")

    # Sort by edge (not raw hit rate!)
    found.sort(key=lambda x: (-x['edge'], x['stability']))
    strong = [r for r in found if r['n'] >= 30 and r['stability'] < 20]
    log(f"  Strong (N>=30, std<20): {len(strong)}")

    # Top by edge
    log(f"\n  --- TOP 30 BY EDGE (vs Random) ---")
    log(f"  {'DIST':<12} {'H':<6} {'TRK':<5} {'SEA':<4} {'BET':<18} {'N':>4} {'AvgH':>5} {'HIT%':>6} {'RAND':>6} {'EDGE':>6} {'ROI':>7} {'STD':>5}")
    log(f"  {'-' * 105}")

    seen = set()
    count = 0
    for r in strong:
        # Deduplicate similar conditions
        key = (r['horses'], r['season'], r['bet'], r['edge'])
        if key in seen:
            continue
        seen.add(key)
        count += 1
        if count > 30:
            break
        roi_s = f"{r['roi']:.1f}%" if r['n_pay'] > 0 else 'N/A'
        log(f"  {r['dist']:<12} {r['horses']:<6} {r['track']:<5} {r['season']:<4} "
            f"{r['bet_name']:<18} {r['n']:>4} {r['avg_horses']:>5.1f} "
            f"{r['hit_rate']:>5.1f}% {r['random_rate']:>5.1f}% {r['edge']:>+5.1f}% "
            f"{roi_s:>7} {r['stability']:>4.1f}")

    # Course analysis
    log(f"\n  --- COURSE ANALYSIS ---")
    for ce in sorted(rdf['course_enc'].unique()):
        cdf = rdf[rdf.course_enc == ce]
        n = len(cdf)
        if n < 10:
            continue
        cn = cdf['course_name'].iloc[0]
        th = cdf['trio_hit'].mean() * 100
        rt = cdf['rand_trio'].mean() * 100
        wh = cdf['wide_hit'].mean() * 100
        rw = cdf['rand_wide'].mean() * 100
        log(f"  {cn}({ce}): N={n}, trio={th:.1f}%(rand {rt:.1f}%, edge {th-rt:+.1f}%), "
            f"wide={wh:.1f}%(rand {rw:.1f}%, edge {wh-rw:+.1f}%)")

    # Class analysis
    log(f"\n  --- CLASS ANALYSIS ---")
    for cls in sorted(rdf['class_code'].unique()):
        cdf = rdf[rdf.class_code == cls]
        n = len(cdf)
        if n < 10:
            continue
        th = cdf['trio_hit'].mean() * 100
        rt = cdf['rand_trio'].mean() * 100
        log(f"  Class {cls}: N={n}, trio={th:.1f}%(edge {th-rt:+.1f}%), avg_h={cdf.num_horses.mean():.1f}")

    # Season analysis
    log(f"\n  --- SEASON ANALYSIS ---")
    for s in ['春', '夏', '秋', '冬']:
        sdf = rdf[rdf.season == s]
        n = len(sdf)
        if n < 10:
            continue
        th = sdf['trio_hit'].mean() * 100
        rt = sdf['rand_trio'].mean() * 100
        log(f"  {s}: N={n}, trio={th:.1f}%(edge {th-rt:+.1f}%), avg_h={sdf.num_horses.mean():.1f}")

    return {
        'total_searched': total, 'found': len(found), 'strong': len(strong),
        'top_conditions': strong[:50], 'all_results': found[:200],
    }


# ── PHASE 3: PROPOSE NEW CONDITIONS ────────────────────────
def propose_new_conditions(opt_results, summary):
    log("\n" + "=" * 60)
    log("  PHASE 3: NEW CONDITION PROPOSAL")
    log("=" * 60)

    top = opt_results.get('top_conditions', [])

    old = {
        'A': {'desc': '8-14頭/1600m+/良~稍重', 'bet': 'trio_7', 'roi': 382.2, 'hit': 55.1},
        'B': {'desc': '8-14頭/1600m+/重~不良', 'bet': 'wide_2', 'roi': 253.0, 'hit': 50.6},
        'C': {'desc': '15頭+/1600m+/良~稍重', 'bet': 'trio_7', 'roi': 66.4, 'hit': 50.0},
        'D': {'desc': '1400m以下', 'bet': 'trio_7', 'roi': 0, 'hit': 0},
        'E': {'desc': '7頭以下', 'bet': 'umaren_2', 'roi': 148.2, 'hit': 40.0},
        'X': {'desc': '15頭+/重~不良', 'bet': 'trio_7', 'roi': 0, 'hit': 0},
    }

    # Find best new conditions with high edge and N>=30
    proposals = [r for r in top if r['edge'] >= 10 and r['n'] >= 30][:10]

    log(f"\n  --- COMPARISON: OLD CONDITIONS (V2 backtest) vs NEW (walk-forward) ---")
    log(f"  {'LABEL':<45} {'N':>5} {'HIT%':>7} {'EDGE':>7} {'ROI':>8}")
    log(f"  {'-' * 80}")
    for ck, cv in old.items():
        roi_s = f"{cv['roi']:.1f}%" if cv['roi'] > 0 else 'N/A'
        # Get walk-forward results for this condition
        wf = summary.get(ck, {})
        wf_trio = wf.get('trio_7', {})
        wf_hr = wf_trio.get('hit_rate', 0)
        wf_edge = wf_trio.get('edge', 0)
        wf_roi = wf_trio.get('roi', 0)
        wf_roi_s = f"{wf_roi:.1f}%" if wf_trio.get('n_pay', 0) > 0 else 'N/A'
        log(f"  [OLD {ck}] {cv['desc']:<40} {'':>5} {cv['hit']:>6.1f}%  {'':>6}  {roi_s:>8}")
        if wf_trio:
            log(f"  [WF  {ck}] walk-forward result{' ' * 22} {wf.get('n', 0):>5} {wf_hr:>6.1f}% {wf_edge:>+6.1f}% {wf_roi_s:>8}")

    log(f"  {'-' * 80}")
    for i, r in enumerate(proposals):
        desc = f"{r['dist']}/{r['horses']}/{r['track']}/{r['season']}"
        roi_s = f"{r['roi']:.1f}%" if r['n_pay'] > 0 else 'N/A'
        log(f"  [NEW {i+1}] {desc:<40} {r['n']:>5} {r['hit_rate']:>6.1f}% {r['edge']:>+6.1f}% {roi_s:>8} ({r['bet_name']})")

    return {'old': old, 'proposals': proposals}


# ── MAIN ────────────────────────────────────────────────────
def main():
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('')

    log("=" * 60)
    log("  NAR FULL BACKTEST & CONDITION OPTIMIZATION")
    log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    df = load_and_merge()
    df = engineer_features(df)

    leak_ok = run_leak_check(FEATURES_NO_ODDS)
    if not leak_ok:
        log("  *** ABORT: Leak detected ***")
        return

    # Phase 1: Walk-forward
    results, aucs = walk_forward_backtest(df, FEATURES_NO_ODDS)
    log(f"\n  Walk-forward: {len(results)} race results")

    summary = analyze_results(results, "WALK-FORWARD (全データ, odds無し)")

    # Separate analysis for KDSCOPE-only vs netkeiba-only
    kd_results = [r for r in results if r['source'] == 'kdscope']
    nk_results = [r for r in results if r['source'] == 'netkeiba']
    summary_kd = analyze_results(kd_results, "KDSCOPE ONLY (2012-2020)")
    summary_nk = analyze_results(nk_results, "NETKEIBA ONLY (2022)")

    # Phase 2: Optimize
    opt = optimize_conditions(results)

    # Phase 3: Propose
    proposals = propose_new_conditions(opt, summary)

    # Save outputs
    log("\n" + "=" * 60)
    log("  SAVING OUTPUTS")
    log("=" * 60)

    pd.DataFrame(results).to_csv(SIM_CSV, index=False, encoding='utf-8')
    log(f"  {SIM_CSV} ({len(results)} rows)")

    opt_bet = {
        'generated_at': datetime.now().isoformat(),
        'walk_forward_aucs': {str(y): round(a, 4) for y, a in aucs},
        'summary_all': summary,
        'summary_kdscope': summary_kd,
        'summary_netkeiba': summary_nk,
        'proposals': proposals,
    }
    with open(OPT_BET_JSON, 'w', encoding='utf-8') as f:
        json.dump(opt_bet, f, ensure_ascii=False, indent=2, default=str)
    log(f"  {OPT_BET_JSON}")

    opt_save = {k: v for k, v in opt.items() if k != 'all_results'}
    opt_save['all_results'] = opt.get('all_results', [])[:200]
    with open(COND_OPT_JSON, 'w', encoding='utf-8') as f:
        json.dump(opt_save, f, ensure_ascii=False, indent=2, default=str)
    log(f"  {COND_OPT_JSON}")

    log("\n  ALL DONE!")


if __name__ == '__main__':
    main()
