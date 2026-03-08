#!/usr/bin/env python
"""KEIBA AI NAR V3 - Enriched KDSCOPE data with tansho odds
- chihou_full_enriched.csv: 17K records (2009-2020), 54% with odds
- chihou_races_2020_2025.csv: 1.8K records (2022), 100% with odds
- Walk-forward validation (train on past, test on future)
- LightGBM + XGBoost ensemble
- Expanding window jockey/trainer/horse stats (leak-free)
"""
import sys
import os
import pandas as pd
import numpy as np
import pickle
import json
import warnings

warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..')
ENRICHED_CSV = os.path.join(OUTPUT_DIR, 'data', 'chihou_full_enriched.csv')
NETKEIBA_CSV = os.path.join(OUTPUT_DIR, 'data', 'chihou_races_2020_2025.csv')
NAR_MODEL_PATH = os.path.join(OUTPUT_DIR, 'keiba_model_v9_nar.pkl')
BACKTEST_PATH = os.path.join(OUTPUT_DIR, 'backtest_nar_condition_v3.json')
CACHE_PATH = os.path.join(OUTPUT_DIR, 'data', 'nar_scraped_cache.json')
LOG_PATH = os.path.join(OUTPUT_DIR, 'train', 'nar_v3_training.log')

NAR_FEATURES = [
    'odds_log', 'num_horses', 'distance', 'course_enc', 'age', 'sex_enc',
    'horse_weight', 'weight_carry', 'umaban', 'bracket',
    'jockey_wr', 'jockey_place_rate', 'trainer_wr', 'horse_career_wr',
    'prev_finish', 'prev2_finish', 'prev3_finish', 'avg_finish_3r',
    'best_finish_3r', 'top3_count_3r', 'finish_trend', 'prev_odds_log',
    'rest_days', 'rest_category', 'dist_cat', 'weight_cat', 'age_group',
    'horse_num_ratio', 'bracket_pos', 'carry_diff', 'dist_change',
    'dist_change_abs', 'pop_rank',
]

# Course code -> numeric encoding (consistent with app.py)
COURSE_ENC = {
    '42': 10, '43': 11, '44': 12, '45': 13,
    '46': 14, '47': 15, '48': 16, '50': 17,
    '51': 18, '54': 19, '55': 20,
}


def log(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode())
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


def load_enriched_data():
    """Load and merge KDSCOPE enriched + netkeiba data."""
    dfs = []

    # KDSCOPE enriched data
    if os.path.exists(ENRICHED_CSV):
        kd = pd.read_csv(ENRICHED_CSV)
        log(f"  KDSCOPE enriched: {len(kd)} rows")

        # Only keep records with finish_pos > 0
        kd = kd[kd['finish_pos'] > 0].copy()
        log(f"  After finish_pos filter: {len(kd)} rows")

        # Create target: top 3 finish
        kd['target'] = (kd['finish_pos'] <= 3).astype(int)

        # Create race_id from date + course + race_no
        kd['race_id'] = kd['race_date'].str.replace('-', '') + kd['course_code'].astype(str).str.zfill(2) + kd['race_no'].astype(str).str.zfill(2)

        # Map columns to common schema
        kd['odds'] = pd.to_numeric(kd['tansho_odds'], errors='coerce').fillna(0)
        kd['horse_weight'] = kd['weight']
        kd['course_enc'] = kd['course_code'].astype(str).map(COURSE_ENC).fillna(10)
        kd['surface_enc'] = 1  # NAR = dirt
        kd['condition_enc'] = 0  # Unknown from KDSCOPE
        kd['condition'] = ''
        kd['sex_enc'] = kd['sex'].map({'牡': 0, '牝': 1, 'セン': 2}).fillna(0)
        kd['weight_carry'] = 0  # Not available from NS
        kd['bracket'] = 0  # Not available from NS
        kd['pop_rank'] = 0  # Not available

        # Compute num_horses per race
        race_sizes = kd.groupby('race_id')['umaban'].count().reset_index()
        race_sizes.columns = ['race_id', 'num_horses']
        kd = kd.merge(race_sizes, on='race_id', how='left')

        kd['finish'] = kd['finish_pos']
        kd['source'] = 'kdscope'
        dfs.append(kd)

    # Netkeiba scraped data
    if os.path.exists(NETKEIBA_CSV):
        nk = pd.read_csv(NETKEIBA_CSV)
        log(f"  Netkeiba: {len(nk)} rows")

        # Already has target, odds, finish, etc.
        nk['horse_weight'] = pd.to_numeric(nk['horse_weight'], errors='coerce').fillna(0)
        nk['weight_carry'] = pd.to_numeric(nk['weight_carry'], errors='coerce').fillna(0)
        nk['odds'] = pd.to_numeric(nk['odds'], errors='coerce').fillna(0)
        nk['course_enc'] = pd.to_numeric(nk['course_enc'], errors='coerce').fillna(10)
        nk['sex_enc'] = pd.to_numeric(nk['sex_enc'], errors='coerce').fillna(0)
        nk['condition_enc'] = pd.to_numeric(nk['condition_enc'], errors='coerce').fillna(0)
        nk['condition'] = nk.get('condition', '')
        nk['pop_rank'] = pd.to_numeric(nk.get('pop_rank', 0), errors='coerce').fillna(0)

        if 'jockey_name' not in nk.columns:
            nk['jockey_name'] = ''
        if 'trainer_name' not in nk.columns:
            nk['trainer_name'] = ''
        if 'horse_name' not in nk.columns:
            nk['horse_name'] = ''
        if 'horse_id' not in nk.columns:
            nk['horse_id'] = ''

        nk['source'] = 'netkeiba'
        dfs.append(nk)

    if not dfs:
        raise ValueError("No data loaded")

    df = pd.concat(dfs, ignore_index=True, sort=False)

    # Fill missing columns
    for col in ['prev_finish', 'prev2_finish', 'prev3_finish', 'avg_finish_3r',
                'best_finish_3r', 'top3_count_3r', 'finish_trend', 'prev_odds_log',
                'rest_days', 'rest_category', 'dist_change', 'dist_change_abs']:
        if col not in df.columns:
            df[col] = 0

    log(f"  Combined: {len(df)} rows, {df['race_id'].nunique()} races")
    return df


def build_expanding_features(df):
    """Build leak-free expanding window features for jockey/trainer/horse."""
    log("  Building expanding window features...")

    # Netkeiba data doesn't have race_date - derive from race_id
    if 'race_date' not in df.columns or df['race_date'].isna().any():
        # Fill missing race_date from race_id (format: YYYYMMDDCCNN)
        mask = df['race_date'].isna() | (df['race_date'] == '')
        if 'race_id' in df.columns:
            df.loc[mask, 'race_date'] = df.loc[mask, 'race_id'].astype(str).str[:4] + '-' + \
                df.loc[mask, 'race_id'].astype(str).str[4:6] + '-' + \
                df.loc[mask, 'race_id'].astype(str).str[6:8]

    df['race_date'] = df['race_date'].astype(str)
    df = df[df['race_date'].str.match(r'^\d{4}-\d{2}-\d{2}$', na=False)].copy()
    df = df.sort_values('race_date').reset_index(drop=True)

    # Initialize stats columns
    df['jockey_wr'] = 0.08
    df['jockey_place_rate'] = 0.25
    df['trainer_wr'] = 0.10
    df['horse_career_wr'] = 0.0

    # Group by date for expanding window
    dates = sorted(df['race_date'].unique())

    jockey_wins = {}
    jockey_races = {}
    jockey_top3 = {}
    trainer_wins = {}
    trainer_races = {}
    horse_wins = {}
    horse_races = {}

    for date in dates:
        mask = df['race_date'] == date
        idx = df[mask].index

        # Apply current stats to this date's races
        for i in idx:
            j = df.at[i, 'jockey_name']
            t = df.at[i, 'trainer_name']
            h = str(df.at[i, 'horse_id']) if pd.notna(df.at[i, 'horse_id']) else ''

            if j and j in jockey_races and jockey_races[j] >= 5:
                df.at[i, 'jockey_wr'] = jockey_wins.get(j, 0) / jockey_races[j]
                df.at[i, 'jockey_place_rate'] = jockey_top3.get(j, 0) / jockey_races[j]
            if t and t in trainer_races and trainer_races[t] >= 5:
                df.at[i, 'trainer_wr'] = trainer_wins.get(t, 0) / trainer_races[t]
            if h and h in horse_races and horse_races[h] >= 2:
                df.at[i, 'horse_career_wr'] = horse_wins.get(h, 0) / horse_races[h]

        # Update stats with this date's results
        for i in idx:
            j = df.at[i, 'jockey_name']
            t = df.at[i, 'trainer_name']
            h = str(df.at[i, 'horse_id']) if pd.notna(df.at[i, 'horse_id']) else ''
            fin = df.at[i, 'finish']
            target = df.at[i, 'target']

            if j:
                jockey_races[j] = jockey_races.get(j, 0) + 1
                if target == 1:
                    jockey_wins[j] = jockey_wins.get(j, 0) + 1
                if fin <= 3:
                    jockey_top3[j] = jockey_top3.get(j, 0) + 1
            if t:
                trainer_races[t] = trainer_races.get(t, 0) + 1
                if target == 1:
                    trainer_wins[t] = trainer_wins.get(t, 0) + 1
            if h:
                horse_races[h] = horse_races.get(h, 0) + 1
                if target == 1:
                    horse_wins[h] = horse_wins.get(h, 0) + 1

    log(f"    Jockeys tracked: {len(jockey_races)}")
    log(f"    Trainers tracked: {len(trainer_races)}")
    log(f"    Horses tracked: {len(horse_races)}")

    return df


def build_derived_features(df):
    """Build derived features from base columns."""
    df['odds_log'] = np.log1p(df['odds'].clip(0, 999))
    df['dist_cat'] = pd.cut(df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
                            labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
    df['weight_cat'] = pd.cut(df['horse_weight'], bins=[0, 440, 480, 520, 9999],
                              labels=[0, 1, 2, 3]).astype(float).fillna(1)
    df['age_group'] = df['age'].clip(2, 7)
    df['horse_num_ratio'] = df['umaban'] / df['num_horses'].clip(1)
    df['bracket_pos'] = pd.cut(df['bracket'], bins=[-1, 3, 6, 99],
                               labels=[0, 1, 2]).astype(float).fillna(1)
    df['carry_diff'] = df['weight_carry'] - df['weight_carry'].mean()
    return df


def classify_condition(num_horses, distance, condition):
    heavy = any(c in str(condition) for c in ['重', '不'])
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


def calc_bets(ranking):
    if len(ranking) < 3:
        return [], [], []
    nums = ranking[:6] if len(ranking) >= 6 else ranking
    n1 = nums[0]
    second = nums[1:3]
    third = nums[1:min(6, len(nums))]
    trio_bets = sorted(set(
        tuple(sorted({n1, s, t}))
        for s in second for t in third
        if len(set({n1, s, t})) == 3
    ))
    umaren_bets = [sorted([n1, nums[1]]), sorted([n1, nums[2]])]
    wide_bets = [sorted([n1, nums[1]]), sorted([n1, nums[2]])]
    return trio_bets, wide_bets, umaren_bets


def check_hits(actual_finishes, trio_bets, wide_bets, umaren_bets):
    top3 = set(uma for uma, fin in actual_finishes.items() if fin <= 3)
    top2 = set(uma for uma, fin in actual_finishes.items() if fin <= 2)
    trio_hit = any(set(combo) == top3 for combo in trio_bets)
    wide_hits = [bet for bet in wide_bets if set(bet).issubset(top3)]
    umaren_hits = [bet for bet in umaren_bets if set(bet) == top2]
    return trio_hit, wide_hits, umaren_hits


def walk_forward_train(df):
    """Walk-forward: train on KDSCOPE (2009-2020), validate on netkeiba (2022)."""
    log("\n  Walk-forward split:")

    # Split: KDSCOPE for training, netkeiba for validation
    train_mask = df['source'] == 'kdscope'
    test_mask = df['source'] == 'netkeiba'

    # Only use records with odds for training (odds_log > 0)
    train_with_odds = train_mask & (df['odds'] > 0)

    df_train = df[train_with_odds].copy()
    df_test = df[test_mask].copy()

    log(f"    Train (KDSCOPE with odds): {len(df_train)} rows, {df_train['race_id'].nunique()} races")
    log(f"    Test (netkeiba 2022): {len(df_test)} rows, {df_test['race_id'].nunique()} races")
    log(f"    Target rate: train={df_train['target'].mean():.3f}, test={df_test['target'].mean():.3f}")

    # Ensure features are numeric
    for f_name in NAR_FEATURES:
        if f_name not in df_train.columns:
            df_train[f_name] = 0
            df_test[f_name] = 0
        df_train[f_name] = pd.to_numeric(df_train[f_name], errors='coerce').fillna(0)
        df_test[f_name] = pd.to_numeric(df_test[f_name], errors='coerce').fillna(0)

    X_train = df_train[NAR_FEATURES].values
    y_train = df_train['target'].values
    X_test = df_test[NAR_FEATURES].values
    y_test = df_test['target'].values

    # --- LightGBM ---
    log("\n  Training LightGBM V3...")
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 15, 'learning_rate': 0.04, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 10,
        'reg_alpha': 0.5, 'reg_lambda': 0.5, 'verbose': -1,
        'n_jobs': -1, 'seed': 42,
    }
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=NAR_FEATURES)
    dval = lgb.Dataset(X_test, label=y_test, feature_name=NAR_FEATURES, reference=dtrain)

    lgb_model = lgb.train(
        params, dtrain, num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    lgb_pred = lgb_model.predict(X_test)
    lgb_auc = roc_auc_score(y_test, lgb_pred)
    log(f"  LightGBM V3 AUC: {lgb_auc:.4f}")

    # --- XGBoost ---
    xgb_model = None
    xgb_auc = 0
    w_lgb, w_xgb = 1.0, 0.0
    ens_auc = lgb_auc
    try:
        import xgboost as xgb_lib
        log("  Training XGBoost V3...")
        dtrain_xgb = xgb_lib.DMatrix(X_train, label=y_train, feature_names=NAR_FEATURES)
        dtest_xgb = xgb_lib.DMatrix(X_test, label=y_test, feature_names=NAR_FEATURES)
        xgb_params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc',
            'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_weight': 15,
            'reg_alpha': 0.5, 'reg_lambda': 0.5, 'seed': 42,
            'tree_method': 'hist', 'verbosity': 0,
        }
        xgb_model = xgb_lib.train(
            xgb_params, dtrain_xgb, num_boost_round=3000,
            evals=[(dtest_xgb, 'valid')],
            early_stopping_rounds=100, verbose_eval=200,
        )
        xgb_pred = xgb_model.predict(dtest_xgb)
        xgb_auc = roc_auc_score(y_test, xgb_pred)
        log(f"  XGBoost V3 AUC: {xgb_auc:.4f}")

        total = lgb_auc + xgb_auc
        w_lgb = lgb_auc / total
        w_xgb = xgb_auc / total
        ens_pred = lgb_pred * w_lgb + xgb_pred * w_xgb
        ens_auc = roc_auc_score(y_test, ens_pred)
        log(f"  Ensemble AUC: {ens_auc:.4f}")
    except ImportError:
        log("  XGBoost not available")

    # Feature importance
    importance = lgb_model.feature_importance(importance_type='gain')
    fi = sorted(zip(NAR_FEATURES, importance), key=lambda x: x[1], reverse=True)
    log(f"\n  Feature Importance TOP 15:")
    for fname, imp in fi[:15]:
        bar = '#' * int(imp / max(fi[0][1], 1) * 25)
        log(f"    {fname:25s} {imp:10.1f} {bar}")

    return lgb_model, xgb_model, {'lgb': w_lgb, 'xgb': w_xgb}, lgb_auc, xgb_auc, ens_auc, df_test


def backtest(lgb_model, xgb_model, weights, df_test):
    """Condition-based backtest on validation data."""
    log("\n" + "=" * 60)
    log("  NAR V3 CONDITION-BASED BACKTEST")
    log("=" * 60)

    X_test = df_test[NAR_FEATURES].values
    scores = lgb_model.predict(X_test)
    if xgb_model:
        import xgboost as xgb_lib
        xgb_scores = xgb_model.predict(xgb_lib.DMatrix(X_test, feature_names=NAR_FEATURES))
        scores = scores * weights['lgb'] + xgb_scores * weights['xgb']

    df_test = df_test.copy()
    df_test['score'] = scores

    # Load payout cache
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    log(f"  Payout cache: {len(cache)} races")

    condition_results = {'A': [], 'B': [], 'C': [], 'D': [], 'E': [], 'X': []}

    for rid in df_test['race_id'].unique():
        race_df = df_test[df_test['race_id'] == rid].sort_values('score', ascending=False)
        if len(race_df) < 3:
            continue

        num_horses = int(race_df['num_horses'].iloc[0])
        distance = int(race_df['distance'].iloc[0])
        condition = race_df['condition'].iloc[0] if 'condition' in race_df.columns else ''
        cond_key = classify_condition(num_horses, distance, condition)

        ranking = race_df['umaban'].astype(int).tolist()
        actual = dict(zip(race_df['umaban'].astype(int), race_df['finish'].astype(int)))

        trio_bets, wide_bets, umaren_bets = calc_bets(ranking)
        trio_hit, wide_hits, umaren_hits = check_hits(actual, trio_bets, wide_bets, umaren_bets)

        race_cache = cache.get(str(rid), {})
        payouts = race_cache.get('payouts', {'trio': 0, 'umaren': 0, 'wide': []})

        condition_results[cond_key].append({
            'race_id': rid, 'trio_hit': trio_hit,
            'trio_payout': payouts.get('trio', 0) if trio_hit else 0,
            'wide_hits': len(wide_hits),
            'wide_payout': sum(payouts.get('wide', [])[:len(wide_hits)]) if wide_hits else 0,
            'umaren_hit': len(umaren_hits) > 0,
            'umaren_payout': payouts.get('umaren', 0) if umaren_hits else 0,
        })

    desc_map = {
        'A': '8-14h/1600m+/good', 'B': '8-14h/1600m+/heavy',
        'C': '15h+/1600m+/good', 'D': '<=1400m',
        'E': '<=7 horses', 'X': '15h+/heavy',
    }

    log(f"\n  {'COND':<4} {'DESC':<22} {'N':>4} | {'BET':<7} {'HIT':>4} {'RATE':>7} {'INVEST':>8} {'PAYOUT':>8} {'ROI':>7} {'STARS'}")
    log(f"  {'-' * 90}")

    best_conditions = {}
    for ckey in ['A', 'B', 'C', 'D', 'E', 'X']:
        races = condition_results.get(ckey, [])
        n = len(races)
        if n == 0:
            best_conditions[ckey] = {
                'n': 0, 'bet_type': 'trio', 'hit_rate': 0,
                'roi': 0, 'stars': 0, 'recommended': False,
            }
            log(f"  {ckey:<4} {desc_map.get(ckey, '?'):<22} {'N/A':>4}")
            continue

        results_by_bet = {}
        for bt, n_bets, hit_key, pay_key in [
            ('trio', 7, 'trio_hit', 'trio_payout'),
            ('umaren', 2, 'umaren_hit', 'umaren_payout'),
            ('wide', 2, 'wide_hits', 'wide_payout'),
        ]:
            if bt == 'wide':
                hits = sum(1 for r in races if r.get(hit_key, 0) > 0)
            else:
                hits = sum(1 for r in races if r.get(hit_key, False))
            investment = n * n_bets * 100
            total_payout = sum(r.get(pay_key, 0) for r in races)
            roi = total_payout / investment * 100 if investment > 0 else 0
            hit_rate = hits / n * 100
            results_by_bet[bt] = {
                'hits': hits, 'hit_rate': hit_rate,
                'investment': investment, 'payout': total_payout, 'roi': roi,
            }

        best_bt = max(results_by_bet, key=lambda b: results_by_bet[b]['hit_rate'])
        best = results_by_bet[best_bt]
        roi = best['roi']
        recommended = best['hit_rate'] >= 30
        stars = 3 if best['hit_rate'] >= 40 else (2 if best['hit_rate'] >= 35 else (1 if best['hit_rate'] >= 30 else 0))
        star_str = '*' * stars if stars > 0 else 'X'

        best_conditions[ckey] = {
            'n': n, 'bet_type': best_bt,
            'hits': best['hits'], 'hit_rate': best['hit_rate'],
            'investment': best['investment'], 'payout': best['payout'],
            'roi': roi, 'stars': stars, 'recommended': recommended,
            'all_bets': results_by_bet,
        }

        first = True
        for bt in ['trio', 'umaren', 'wide']:
            r = results_by_bet[bt]
            marker = ' <<<' if bt == best_bt else ''
            s = f' {star_str}' if bt == best_bt else ''
            if first:
                log(f"  {ckey:<4} {desc_map.get(ckey, '?'):<22} {n:>4} | {bt:<7} {r['hits']:>4} {r['hit_rate']:>6.1f}% {r['investment']:>7,} {r['payout']:>7,} {r['roi']:>6.1f}%{marker}{s}")
                first = False
            else:
                log(f"  {'':4} {'':22} {'':4} | {bt:<7} {r['hits']:>4} {r['hit_rate']:>6.1f}% {r['investment']:>7,} {r['payout']:>7,} {r['roi']:>6.1f}%{marker}")

    return best_conditions


def main():
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('')

    log("=" * 60)
    log("  KEIBA AI NAR V3 - Enriched KDSCOPE + Odds")
    log(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log("=" * 60)

    # Load V2 model for comparison
    v2_auc = 0.792
    v2_ens_auc = 0.792
    old_jockey_stats = {}
    if os.path.exists(NAR_MODEL_PATH):
        with open(NAR_MODEL_PATH, 'rb') as f:
            v2 = pickle.load(f)
        v2_auc = v2.get('auc', 0.792)
        v2_ens_auc = v2.get('ensemble_auc', 0.792)
        old_jockey_stats = v2.get('jockey_stats', {})
        log(f"  V2 AUC: {v2_auc:.4f} / Ensemble: {v2_ens_auc:.4f}")

    # Load data
    log("\n  Loading data...")
    df = load_enriched_data()

    # Build expanding window features
    df = build_expanding_features(df)

    # Build derived features
    df = build_derived_features(df)

    # Walk-forward training
    lgb_model, xgb_model, weights, lgb_auc, xgb_auc, ens_auc, df_test = walk_forward_train(df)

    # Backtest on validation set
    best_conditions = backtest(lgb_model, xgb_model, weights, df_test)

    # Results
    log(f"\n" + "=" * 60)
    log(f"  RESULTS")
    log(f"=" * 60)
    log(f"  V2 AUC: {v2_auc:.4f} / Ensemble: {v2_ens_auc:.4f}")
    log(f"  V3 LGB AUC: {lgb_auc:.4f} / Ensemble: {ens_auc:.4f}")
    log(f"  Improvement: LGB {lgb_auc - v2_auc:+.4f} / Ensemble {ens_auc - v2_ens_auc:+.4f}")

    # Save model
    best_auc = max(lgb_auc, ens_auc)
    log(f"\n  Saving V3 model (AUC: {best_auc:.4f})...")

    # Build jockey stats from all data
    jockey_stats = {}
    jockey_data = df.groupby('jockey_name').agg(
        wins=('target', 'sum'),
        races=('target', 'count'),
        top3=('finish', lambda x: (x <= 3).sum()),
    ).reset_index()
    for _, row in jockey_data.iterrows():
        if row['races'] >= 5:
            jockey_stats[row['jockey_name']] = {
                'wr': row['wins'] / row['races'],
                'place_rate': row['top3'] / row['races'],
                'races': int(row['races']),
            }

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nar_pkl = {
        'model': lgb_model,
        'xgb_model': xgb_model,
        'features': NAR_FEATURES,
        'version': 'nar_v3',
        'auc': lgb_auc,
        'ensemble_auc': ens_auc,
        'ensemble_weights': weights,
        'model_type': 'nar_dedicated',
        'trained_at': now,
        'jockey_stats': jockey_stats,
        'condition_results': {k: {
            'bet_type': v['bet_type'],
            'hit_rate': v.get('hit_rate', 0),
            'roi': v.get('roi', 0),
            'stars': v.get('stars', 0),
            'recommended': v['recommended'],
            'n': v['n'],
        } for k, v in best_conditions.items()},
    }
    with open(NAR_MODEL_PATH, 'wb') as f:
        pickle.dump(nar_pkl, f)
    log(f"  Saved: {NAR_MODEL_PATH}")

    bt_save = {
        'generated_at': now,
        'version': 'nar_v3',
        'lgb_auc': lgb_auc,
        'xgb_auc': xgb_auc,
        'ensemble_auc': ens_auc,
        'training_data': {
            'kdscope_enriched': len(df[df['source'] == 'kdscope']),
            'netkeiba': len(df[df['source'] == 'netkeiba']),
            'odds_coverage': f"{(df[df['source']=='kdscope']['odds'] > 0).mean()*100:.1f}%",
        },
        'condition_results': {k: {
            'bet_type': v['bet_type'],
            'hit_rate': v.get('hit_rate', 0),
            'roi': v.get('roi', 0),
            'stars': v.get('stars', 0),
            'recommended': v['recommended'],
            'n': v['n'],
        } for k, v in best_conditions.items()},
    }
    with open(BACKTEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(bt_save, f, ensure_ascii=False, indent=2)
    log(f"  Saved: {BACKTEST_PATH}")

    # Summary
    log(f"\n  CONDITION SUMMARY:")
    for ckey in ['A', 'B', 'C', 'D', 'E', 'X']:
        info = best_conditions[ckey]
        if info['n'] == 0:
            log(f"  {ckey}: No data")
        else:
            star_str = '*' * info['stars'] if info['stars'] > 0 else 'X'
            rec = 'REC' if info['recommended'] else 'SKIP'
            log(f"  {ckey}: {info['bet_type']} Hit {info.get('hit_rate',0):.1f}% ROI {info.get('roi',0):.1f}% (N={info['n']}) -> {star_str} {rec}")

    log(f"\n  Done!")
    return best_auc, best_conditions


if __name__ == '__main__':
    main()
