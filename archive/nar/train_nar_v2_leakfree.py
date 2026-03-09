#!/usr/bin/env python
"""KEIBA AI NAR V2 Leak-Free Training
Pattern A: Strict leak-free (pre-day) - removes odds_log, horse_weight, condition_enc, pop_rank
Pattern B: Pre-race info OK - removes only odds_log, pop_rank (derived from final odds)
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
BACKTEST_PATH = os.path.join(OUTPUT_DIR, 'backtest_nar_condition.json')
CACHE_PATH = os.path.join(OUTPUT_DIR, 'data', 'nar_scraped_cache.json')

# Original NAR features (with leak)
NAR_FEATURES_ORIG = [
    'odds_log', 'num_horses', 'distance', 'surface_enc', 'condition_enc',
    'course_enc', 'horse_weight', 'weight_carry', 'age', 'sex_enc',
    'horse_num', 'bracket', 'jockey_wr', 'jockey_place_rate', 'trainer_wr',
    'prev_finish', 'prev2_finish', 'prev3_finish', 'avg_finish_3r',
    'best_finish_3r', 'top3_count_3r', 'finish_trend', 'prev_odds_log',
    'rest_days', 'rest_category', 'dist_cat', 'weight_cat', 'age_group',
    'horse_num_ratio', 'bracket_pos', 'carry_diff', 'dist_change',
    'dist_change_abs', 'is_nar', 'pop_rank',
]

# Pattern A: Strict leak-free (pre-day info only)
LEAK_A = {'odds_log', 'horse_weight', 'condition_enc', 'weight_cat', 'pop_rank'}
NAR_FEATURES_A = [f for f in NAR_FEATURES_ORIG if f not in LEAK_A]

# Pattern B: Pre-race info OK (remove only final-odds-derived features)
LEAK_B = {'odds_log', 'pop_rank'}
NAR_FEATURES_B = [f for f in NAR_FEATURES_ORIG if f not in LEAK_B]


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


def train_and_backtest(df, features, jockey_stats, label, cache):
    """Train model and run backtest for a given feature set."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Features: {len(features)}")
    print(f"={'=' * 60}")

    # Build features
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

    # Condition analysis
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
    print("  KEIBA AI NAR V2 LEAK-FREE TRAINING")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Load existing model for jockey stats
    with open(NAR_MODEL_PATH, 'rb') as f:
        v1 = pickle.load(f)
    jockey_stats = v1.get('jockey_stats', {})
    print(f"  Existing jockey stats: {len(jockey_stats)} jockeys")

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    print(f"  CSV: {len(df)} rows, {df['race_id'].nunique()} races")

    # Load cache
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r', encoding='utf-8') as cf:
            cache = json.load(cf)

    # Train all three variants
    res_orig = train_and_backtest(df, NAR_FEATURES_ORIG, jockey_stats, "ORIGINAL (with leak)", cache)
    res_a = train_and_backtest(df, NAR_FEATURES_A, jockey_stats, "PATTERN A (strict leak-free)", cache)
    res_b = train_and_backtest(df, NAR_FEATURES_B, jockey_stats, "PATTERN B (pre-race OK)", cache)

    # Summary
    print("\n" + "=" * 60)
    print("  NAR RESULTS COMPARISON")
    print("=" * 60)
    print(f"  {'Pattern':<30} {'LGB AUC':>10} {'XGB AUC':>10} {'Ensemble':>10}")
    print(f"  {'-' * 60}")
    print(f"  {'Original (with leak)':<30} {res_orig['lgb_auc']:>10.4f} {res_orig['xgb_auc']:>10.4f} {res_orig['ens_auc']:>10.4f}")
    print(f"  {'Pattern A (strict leak-free)':<30} {res_a['lgb_auc']:>10.4f} {res_a['xgb_auc']:>10.4f} {res_a['ens_auc']:>10.4f}")
    print(f"  {'Pattern B (pre-race OK)':<30} {res_b['lgb_auc']:>10.4f} {res_b['xgb_auc']:>10.4f} {res_b['ens_auc']:>10.4f}")
    print(f"\n  AUC drop from leak removal:")
    print(f"  Pattern A: {res_a['ens_auc'] - res_orig['ens_auc']:+.4f}")
    print(f"  Pattern B: {res_b['ens_auc'] - res_orig['ens_auc']:+.4f}")

    # Save Pattern A as production
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    nar_pkl_a = {
        'model': res_a['lgb_model'],
        'xgb_model': res_a['xgb_model'],
        'features': NAR_FEATURES_A,
        'version': 'nar_v2a_leakfree',
        'auc': res_a['lgb_auc'],
        'ensemble_auc': res_a['ens_auc'],
        'ensemble_weights': {'lgb': res_a['w_lgb'], 'xgb': res_a['w_xgb']},
        'model_type': 'nar_dedicated',
        'trained_at': now,
        'jockey_stats': jockey_stats,
        'leak_free': True,
        'leak_pattern': 'A',
        'leak_removed': sorted(LEAK_A),
        'condition_results': {k: {
            'bet_type': v['bet_type'], 'hit_rate': v.get('hit_rate', 0),
            'roi': v.get('roi', 0), 'recommended': v['recommended'], 'n': v['n'],
        } for k, v in res_a['best_conditions'].items()},
    }

    with open(NAR_MODEL_PATH, 'wb') as f:
        pickle.dump(nar_pkl_a, f)
    print(f"\n  Saved Pattern A (production): {NAR_MODEL_PATH}")

    # Save backtest
    bt_save = {
        'generated_at': now,
        'version': 'nar_v2a_leakfree',
        'lgb_auc': res_a['lgb_auc'],
        'ensemble_auc': res_a['ens_auc'],
        'leak_pattern': 'A',
        'leak_removed': sorted(LEAK_A),
        'condition_results': {k: {
            'bet_type': v['bet_type'], 'hit_rate': v.get('hit_rate', 0),
            'roi': v.get('roi', 0), 'recommended': v['recommended'], 'n': v['n'],
        } for k, v in res_a['best_conditions'].items()},
        'pattern_b_reference': {
            'lgb_auc': res_b['lgb_auc'],
            'ensemble_auc': res_b['ens_auc'],
            'condition_results': {k: {
                'bet_type': v['bet_type'], 'hit_rate': v.get('hit_rate', 0),
                'roi': v.get('roi', 0), 'recommended': v['recommended'], 'n': v['n'],
            } for k, v in res_b['best_conditions'].items()},
        },
    }
    with open(BACKTEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(bt_save, f, ensure_ascii=False, indent=2)
    print(f"  Saved backtest: {BACKTEST_PATH}")

    # Save comparison
    comparison = {
        'generated_at': now,
        'original': {'lgb': res_orig['lgb_auc'], 'xgb': res_orig['xgb_auc'], 'ensemble': res_orig['ens_auc']},
        'pattern_a': {
            'lgb': res_a['lgb_auc'], 'xgb': res_a['xgb_auc'], 'ensemble': res_a['ens_auc'],
            'removed': sorted(LEAK_A),
            'conditions': {k: {'n': v['n'], 'bet_type': v['bet_type'], 'hit_rate': v.get('hit_rate', 0),
                               'roi': v.get('roi', 0)} for k, v in res_a['best_conditions'].items()},
        },
        'pattern_b': {
            'lgb': res_b['lgb_auc'], 'xgb': res_b['xgb_auc'], 'ensemble': res_b['ens_auc'],
            'removed': sorted(LEAK_B),
            'conditions': {k: {'n': v['n'], 'bet_type': v['bet_type'], 'hit_rate': v.get('hit_rate', 0),
                               'roi': v.get('roi', 0)} for k, v in res_b['best_conditions'].items()},
        },
    }
    comp_path = os.path.join(OUTPUT_DIR, 'leak_comparison_nar.json')
    with open(comp_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"  Saved comparison: {comp_path}")

    print("\n  NAR leak-free training complete!")
    return res_orig, res_a, res_b


if __name__ == '__main__':
    main()
