#!/usr/bin/env python
"""KEIBA AI NAR V3 Leak-Free Training
Adds new features to NAR V2a baseline:
- Distance aptitude (horse's top3 rate at distance category, expanding window)
- Jockey × course affinity (jockey win rate per course, expanding window)
- Frame advantage by course × distance (expanding window)
All features are pre-day (Pattern A leak-free).
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
from sklearn.model_selection import train_test_split
import lightgbm as lgb

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..')
CSV_PATH = os.path.join(OUTPUT_DIR, 'data', 'chihou_races_2020_2025.csv')
NAR_MODEL_PATH = os.path.join(OUTPUT_DIR, 'keiba_model_v9_nar.pkl')
CACHE_PATH = os.path.join(OUTPUT_DIR, 'data', 'nar_scraped_cache.json')

# V2a features (Pattern A leak-free baseline)
LEAK_A = {'odds_log', 'horse_weight', 'condition_enc', 'weight_cat', 'pop_rank'}
NAR_FEATURES_V2_ORIG = [
    'odds_log', 'num_horses', 'distance', 'surface_enc', 'condition_enc',
    'course_enc', 'horse_weight', 'weight_carry', 'age', 'sex_enc',
    'horse_num', 'bracket', 'jockey_wr', 'jockey_place_rate', 'trainer_wr',
    'prev_finish', 'prev2_finish', 'prev3_finish', 'avg_finish_3r',
    'best_finish_3r', 'top3_count_3r', 'finish_trend', 'prev_odds_log',
    'rest_days', 'rest_category', 'dist_cat', 'weight_cat', 'age_group',
    'horse_num_ratio', 'bracket_pos', 'carry_diff', 'dist_change',
    'dist_change_abs', 'is_nar', 'pop_rank',
]
NAR_FEATURES_V2A = [f for f in NAR_FEATURES_V2_ORIG if f not in LEAK_A]

# V3 new features
NAR_V3_NEW_FEATURES = [
    'horse_dist_top3r',        # Horse top3 rate at distance category
    'horse_surface_top3r',     # Horse top3 rate on surface type
    'jockey_course_wr',        # Jockey win rate at this course
    'frame_course_dist_wr',    # Bracket win rate by course × distance
    'horse_career_races',      # Horse career race count
    'horse_career_wr',         # Horse career win rate
    'horse_career_top3r',      # Horse career top3 rate
]
NAR_FEATURES_V3 = NAR_FEATURES_V2A + NAR_V3_NEW_FEATURES


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


def compute_nar_new_features(df):
    """Compute new features for NAR data (expanding window, leak-free)."""
    # Sort by race_id for time ordering
    df = df.sort_values('race_id').reset_index(drop=True)

    df['is_win'] = (df['finish'] == 1).astype(int)
    df['is_top3'] = (df['finish'] <= 3).astype(int)

    global_wr = df['is_win'].mean()
    global_t3 = df['is_top3'].mean()

    # 1. Distance aptitude (horse × distance category)
    print("  Computing distance aptitude...")
    alpha_dist = 5
    df['dist_cat_apt'] = pd.cut(df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
                                 labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
    df['hd_cum_races'] = df.groupby(['horse_id', 'dist_cat_apt']).cumcount()
    df['hd_cum_top3'] = df.groupby(['horse_id', 'dist_cat_apt'])['is_top3'].cumsum() - df['is_top3']
    df['horse_dist_top3r'] = (
        (df['hd_cum_top3'] + alpha_dist * global_t3) /
        (df['hd_cum_races'] + alpha_dist)
    )

    # 2. Horse surface aptitude
    print("  Computing surface aptitude...")
    df['hs_cum_races'] = df.groupby(['horse_id', 'surface_enc']).cumcount()
    df['hs_cum_top3'] = df.groupby(['horse_id', 'surface_enc'])['is_top3'].cumsum() - df['is_top3']
    df['horse_surface_top3r'] = (
        (df['hs_cum_top3'] + alpha_dist * global_t3) /
        (df['hs_cum_races'] + alpha_dist)
    )

    # 3. Jockey × course win rate
    print("  Computing jockey x course affinity...")
    alpha_jc = 10
    df['jc_cum_races'] = df.groupby(['jockey_name', 'course_enc']).cumcount()
    df['jc_cum_wins'] = df.groupby(['jockey_name', 'course_enc'])['is_win'].cumsum() - df['is_win']
    df['jockey_course_wr'] = (
        (df['jc_cum_wins'] + alpha_jc * global_wr) /
        (df['jc_cum_races'] + alpha_jc)
    )

    # 4. Frame advantage by course × distance
    print("  Computing frame advantage...")
    alpha_frm = 50
    df['frame_key'] = df['course_enc'].astype(str) + '_' + df['dist_cat_apt'].astype(str) + '_' + df['bracket'].astype(str)
    df['frm_cum_races'] = df.groupby('frame_key').cumcount()
    df['frm_cum_wins'] = df.groupby('frame_key')['is_win'].cumsum() - df['is_win']
    df['frame_course_dist_wr'] = (
        (df['frm_cum_wins'] + alpha_frm * global_wr) /
        (df['frm_cum_races'] + alpha_frm)
    )

    # 5. Horse career features
    print("  Computing horse career features...")
    alpha_hc = 5
    df['hc_cum_races'] = df.groupby('horse_id').cumcount()
    df['hc_cum_wins'] = df.groupby('horse_id')['is_win'].cumsum() - df['is_win']
    df['hc_cum_top3'] = df.groupby('horse_id')['is_top3'].cumsum() - df['is_top3']
    df['horse_career_races'] = df['hc_cum_races']
    df['horse_career_wr'] = (
        (df['hc_cum_wins'] + alpha_hc * global_wr) /
        (df['hc_cum_races'] + alpha_hc)
    )
    df['horse_career_top3r'] = (
        (df['hc_cum_top3'] + alpha_hc * global_t3) /
        (df['hc_cum_races'] + alpha_hc)
    )

    # Clean up temp columns
    drop_cols = [c for c in df.columns if any(c.startswith(p) for p in
                 ['hd_cum_', 'hs_cum_', 'jc_cum_', 'frm_cum_', 'hc_cum_'])]
    drop_cols.extend(['dist_cat_apt', 'frame_key', 'is_win', 'is_top3'])
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def train_and_backtest(df, features, jockey_stats, label, cache):
    """Train model and run backtest for a given feature set."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Features: {len(features)}")
    print(f"={'=' * 60}")

    df_work = df.copy()
    df_work['jockey_wr'] = df_work['jockey_name'].map(lambda j: jockey_stats.get(j, {}).get('wr', 0.08))
    df_work['jockey_place_rate'] = df_work['jockey_name'].map(lambda j: jockey_stats.get(j, {}).get('place_rate', 0.25))
    df_work['trainer_wr'] = 0.10
    df_work['odds_log'] = np.log1p(df_work['odds'].clip(1, 999))
    df_work['dist_cat'] = pd.cut(df_work['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
                                  labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
    df_work['weight_cat'] = pd.cut(df_work['horse_weight'], bins=[0, 440, 480, 520, 9999],
                                    labels=[0, 1, 2, 3]).astype(float).fillna(1)
    df_work['age_group'] = df_work['age'].clip(2, 7)
    df_work['horse_num_ratio'] = df_work['horse_num'] / df_work['num_horses'].clip(1)
    df_work['bracket_pos'] = pd.cut(df_work['bracket'], bins=[0, 3, 6, 8],
                                     labels=[0, 1, 2]).astype(float).fillna(1)
    df_work['carry_diff'] = df_work['weight_carry'] - df_work['weight_carry'].mean()
    df_work['is_nar'] = 1

    # Compute new features if needed
    new_feat_needed = [f for f in features if f in NAR_V3_NEW_FEATURES]
    if new_feat_needed:
        df_work = compute_nar_new_features(df_work)

    for f_name in features:
        if f_name not in df_work.columns:
            df_work[f_name] = 0
        df_work[f_name] = pd.to_numeric(df_work[f_name], errors='coerce').fillna(0)

    X = df_work[features].values
    y = df_work['target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Target rate: train={y_train.mean():.3f}, test={y_test.mean():.3f}")

    # LightGBM
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 15, 'learning_rate': 0.04, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 10,
        'reg_alpha': 0.5, 'reg_lambda': 0.5, 'verbose': -1,
        'n_jobs': -1, 'seed': 42,
    }
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=features)
    dtest = lgb.Dataset(X_test, label=y_test, feature_name=features, reference=dtrain)

    lgb_model = lgb.train(
        params, dtrain, num_boost_round=2000,
        valid_sets=[dtest],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    lgb_pred = lgb_model.predict(X_test)
    lgb_auc = roc_auc_score(y_test, lgb_pred)
    print(f"  LightGBM AUC: {lgb_auc:.4f}")

    # XGBoost
    xgb_model = None
    xgb_auc = 0
    w_lgb, w_xgb = 1.0, 0.0
    try:
        import xgboost as xgb_lib
        dtrain_xgb = xgb_lib.DMatrix(X_train, label=y_train)
        dtest_xgb = xgb_lib.DMatrix(X_test, label=y_test)
        xgb_params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc',
            'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_weight': 15,
            'reg_alpha': 0.5, 'reg_lambda': 0.5, 'seed': 42,
            'tree_method': 'hist', 'verbosity': 0,
        }
        xgb_model = xgb_lib.train(
            xgb_params, dtrain_xgb, num_boost_round=2000,
            evals=[(dtest_xgb, 'valid')],
            early_stopping_rounds=50, verbose_eval=100,
        )
        xgb_pred = xgb_model.predict(dtest_xgb)
        xgb_auc = roc_auc_score(y_test, xgb_pred)
        print(f"  XGBoost AUC: {xgb_auc:.4f}")

        total = lgb_auc + xgb_auc
        w_lgb = lgb_auc / total
        w_xgb = xgb_auc / total
        ens_pred = lgb_pred * w_lgb + xgb_pred * w_xgb
        ens_auc = roc_auc_score(y_test, ens_pred)
        print(f"  Ensemble AUC: {ens_auc:.4f}")
    except ImportError:
        ens_auc = lgb_auc

    # Feature importance
    importance = lgb_model.feature_importance(importance_type='gain')
    fi = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
    print(f"\n  Feature Importance TOP 15:")
    for fname, imp in fi[:15]:
        bar = '#' * int(imp / fi[0][1] * 25)
        print(f"    {fname:25s} {imp:10.1f} {bar}")

    # Backtest
    all_scores = lgb_model.predict(df_work[features].values)
    if xgb_model:
        import xgboost as xgb_lib
        xgb_all = xgb_model.predict(xgb_lib.DMatrix(df_work[features].values))
        all_scores = all_scores * w_lgb + xgb_all * w_xgb

    df_work['score'] = all_scores
    condition_results = {'A': [], 'B': [], 'C': [], 'D': [], 'E': [], 'X': []}

    for rid in df_work['race_id'].unique():
        race_df = df_work[df_work['race_id'] == rid].sort_values('score', ascending=False)
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
            'race_id': rid,
            'trio_hit': trio_hit,
            'trio_payout': payouts.get('trio', 0) if trio_hit else 0,
            'wide_hits': len(wide_hits),
            'wide_payout': sum(payouts.get('wide', [])[:len(wide_hits)]) if wide_hits else 0,
            'umaren_hit': len(umaren_hits) > 0,
            'umaren_payout': payouts.get('umaren', 0) if umaren_hits else 0,
        })

    print(f"\n  {'COND':<4} {'N':>4} | {'BET':<7} {'HIT':>4} {'RATE':>7} {'INVEST':>8} {'PAYOUT':>8} {'ROI':>7}")
    print(f"  {'-' * 60}")

    best_conditions = {}
    for ckey in ['A', 'B', 'C', 'D', 'E', 'X']:
        races = condition_results.get(ckey, [])
        n = len(races)
        if n == 0:
            best_conditions[ckey] = {'n': 0, 'bet_type': 'trio', 'hit_rate': 0, 'roi': 0, 'recommended': False}
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

        best_bt = max(results_by_bet, key=lambda b: results_by_bet[b]['roi'])
        best = results_by_bet[best_bt]
        roi = best['roi']
        recommended = roi >= 80

        best_conditions[ckey] = {
            'n': n, 'bet_type': best_bt,
            'hits': best['hits'], 'hit_rate': best['hit_rate'],
            'roi': roi, 'recommended': recommended,
        }

        for bt in ['trio', 'umaren', 'wide']:
            r = results_by_bet[bt]
            marker = ' <<<' if bt == best_bt else ''
            print(f"  {ckey if bt == 'trio' else '':4} {n if bt == 'trio' else '':>4} | {bt:<7} {r['hits']:>4} {r['hit_rate']:>6.1f}% {r['investment']:>7,} {r['payout']:>7,} {r['roi']:>6.1f}%{marker}")

    return {
        'lgb_model': lgb_model, 'xgb_model': xgb_model,
        'lgb_auc': lgb_auc, 'xgb_auc': xgb_auc, 'ens_auc': ens_auc,
        'w_lgb': w_lgb, 'w_xgb': w_xgb,
        'best_conditions': best_conditions,
        'features': features,
    }


def main():
    print("=" * 60)
    print("  KEIBA AI NAR V3 LEAK-FREE TRAINING")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  New features: {NAR_V3_NEW_FEATURES}")
    print("=" * 60)

    # Load existing model for jockey stats
    with open(NAR_MODEL_PATH, 'rb') as f:
        v2 = pickle.load(f)
    jockey_stats = v2.get('jockey_stats', {})
    print(f"  Existing jockey stats: {len(jockey_stats)} jockeys")

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    print(f"  CSV: {len(df)} rows, {df['race_id'].nunique()} races")

    # Load cache
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r', encoding='utf-8') as cf:
            cache = json.load(cf)

    # Train V2a baseline and V3
    res_v2a = train_and_backtest(df, NAR_FEATURES_V2A, jockey_stats, "V2a BASELINE (Pattern A)", cache)
    res_v3 = train_and_backtest(df, NAR_FEATURES_V3, jockey_stats, "V3 (+ new features)", cache)

    # Summary
    print("\n" + "=" * 60)
    print("  NAR RESULTS COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<30} {'LGB AUC':>10} {'XGB AUC':>10} {'Ensemble':>10}")
    print(f"  {'-' * 60}")
    print(f"  {'V2a (current production)':<30} {res_v2a['lgb_auc']:>10.4f} {res_v2a['xgb_auc']:>10.4f} {res_v2a['ens_auc']:>10.4f}")
    print(f"  {'V3 (+ new features)':<30} {res_v3['lgb_auc']:>10.4f} {res_v3['xgb_auc']:>10.4f} {res_v3['ens_auc']:>10.4f}")
    print(f"\n  Improvement: {res_v3['ens_auc'] - res_v2a['ens_auc']:+.4f}")

    # Save if improved
    improved = res_v3['ens_auc'] > res_v2a['ens_auc']
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if improved:
        print(f"\n  *** NAR V3 IMPROVED! ({res_v3['ens_auc']:.4f} > {res_v2a['ens_auc']:.4f}) ***")

        nar_pkl = {
            'model': res_v3['lgb_model'],
            'xgb_model': res_v3['xgb_model'],
            'features': NAR_FEATURES_V3,
            'version': 'nar_v3_leakfree',
            'auc': res_v3['lgb_auc'],
            'ensemble_auc': res_v3['ens_auc'],
            'ensemble_weights': {'lgb': res_v3['w_lgb'], 'xgb': res_v3['w_xgb']},
            'model_type': 'nar_dedicated',
            'trained_at': now,
            'jockey_stats': jockey_stats,
            'leak_free': True,
            'leak_pattern': 'A',
            'leak_removed': sorted(LEAK_A),
            'new_features_added': NAR_V3_NEW_FEATURES,
            'baseline_auc': res_v2a['ens_auc'],
            'condition_results': {k: {
                'bet_type': v['bet_type'], 'hit_rate': v.get('hit_rate', 0),
                'roi': v.get('roi', 0), 'recommended': v['recommended'], 'n': v['n'],
            } for k, v in res_v3['best_conditions'].items()},
        }

        with open(NAR_MODEL_PATH, 'wb') as f:
            pickle.dump(nar_pkl, f)
        print(f"  Saved: {NAR_MODEL_PATH}")
    else:
        print(f"\n  NAR V3 did NOT improve ({res_v3['ens_auc']:.4f} <= {res_v2a['ens_auc']:.4f})")
        print(f"  Keeping current V2a production model.")

    # Save results
    results = {
        'generated_at': now,
        'v2a_baseline': {'lgb': res_v2a['lgb_auc'], 'xgb': res_v2a['xgb_auc'], 'ensemble': res_v2a['ens_auc']},
        'v3': {'lgb': res_v3['lgb_auc'], 'xgb': res_v3['xgb_auc'], 'ensemble': res_v3['ens_auc']},
        'improved': improved,
        'improvement': res_v3['ens_auc'] - res_v2a['ens_auc'],
        'new_features': NAR_V3_NEW_FEATURES,
    }

    comp_path = os.path.join(OUTPUT_DIR, 'v3_training_results_nar.json')
    with open(comp_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved results: {comp_path}")

    print("\n  NAR V3 training complete!")
    return results


if __name__ == '__main__':
    main()
