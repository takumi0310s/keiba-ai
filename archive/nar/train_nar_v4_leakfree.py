#!/usr/bin/env python
"""KEIBA AI NAR V4 Leak-Free Training
Major fix: Use nar_merged.csv (14K rows, all distances) instead of
chihou_races_2020_2025.csv (1.8K rows, 1600m only).
Time-based split instead of random split.
Pattern A leak-free.
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
CSV_PATH = os.path.join(OUTPUT_DIR, 'data', 'nar_merged.csv')
NAR_MODEL_PATH = os.path.join(OUTPUT_DIR, 'keiba_model_v9_nar.pkl')
CACHE_PATH = os.path.join(OUTPUT_DIR, 'data', 'nar_scraped_cache.json')

# Pattern A leak-free: remove race-day info
LEAK_A = {'odds_log', 'horse_weight', 'condition_enc', 'weight_cat', 'pop_rank'}

# Base features (available in nar_merged.csv)
NAR_FEATURES_BASE = [
    'num_horses', 'distance', 'surface_enc', 'course_enc',
    'weight_carry', 'age', 'sex_enc',
    'horse_num', 'bracket', 'jockey_wr', 'jockey_place_rate', 'trainer_wr',
    'prev_finish', 'prev2_finish', 'prev3_finish', 'avg_finish_3r',
    'best_finish_3r', 'top3_count_3r', 'finish_trend', 'prev_odds_log',
    'rest_days', 'rest_category', 'dist_cat', 'age_group',
    'horse_num_ratio', 'bracket_pos', 'carry_diff', 'dist_change',
    'dist_change_abs', 'is_nar',
]

# New V4 features (computed from data, all leak-free)
NAR_V4_NEW = [
    'horse_dist_top3r',
    'horse_surface_top3r',
    'jockey_course_wr',
    'frame_course_dist_wr',
    'horse_career_races',
    'horse_career_wr',
    'horse_career_top3r',
]

NAR_FEATURES_V4 = NAR_FEATURES_BASE + NAR_V4_NEW


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


def compute_expanding_features(df):
    """Compute expanding-window features (leak-free)."""
    df = df.sort_values('race_id').reset_index(drop=True)

    df['is_win'] = (df['finish'] == 1).astype(int)
    df['is_top3'] = (df['finish'] <= 3).astype(int)

    global_wr = df['is_win'].mean()
    global_t3 = df['is_top3'].mean()

    # Horse distance aptitude
    alpha = 5
    df['dist_cat_apt'] = pd.cut(df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
                                 labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
    df['hd_r'] = df.groupby(['horse_id', 'dist_cat_apt']).cumcount()
    df['hd_t3'] = df.groupby(['horse_id', 'dist_cat_apt'])['is_top3'].cumsum() - df['is_top3']
    df['horse_dist_top3r'] = (df['hd_t3'] + alpha * global_t3) / (df['hd_r'] + alpha)

    # Horse surface aptitude
    df['hs_r'] = df.groupby(['horse_id', 'surface_enc']).cumcount()
    df['hs_t3'] = df.groupby(['horse_id', 'surface_enc'])['is_top3'].cumsum() - df['is_top3']
    df['horse_surface_top3r'] = (df['hs_t3'] + alpha * global_t3) / (df['hs_r'] + alpha)

    # Jockey x course win rate
    alpha_jc = 10
    df['jc_r'] = df.groupby(['jockey_name', 'course_enc']).cumcount()
    df['jc_w'] = df.groupby(['jockey_name', 'course_enc'])['is_win'].cumsum() - df['is_win']
    df['jockey_course_wr'] = (df['jc_w'] + alpha_jc * global_wr) / (df['jc_r'] + alpha_jc)

    # Frame advantage by course x distance
    alpha_frm = 50
    df['fk'] = df['course_enc'].astype(str) + '_' + df['dist_cat_apt'].astype(str) + '_' + df['bracket'].astype(str)
    df['fr_r'] = df.groupby('fk').cumcount()
    df['fr_w'] = df.groupby('fk')['is_win'].cumsum() - df['is_win']
    df['frame_course_dist_wr'] = (df['fr_w'] + alpha_frm * global_wr) / (df['fr_r'] + alpha_frm)

    # Horse career
    df['hc_r'] = df.groupby('horse_id').cumcount()
    df['hc_w'] = df.groupby('horse_id')['is_win'].cumsum() - df['is_win']
    df['hc_t3'] = df.groupby('horse_id')['is_top3'].cumsum() - df['is_top3']
    df['horse_career_races'] = df['hc_r']
    df['horse_career_wr'] = (df['hc_w'] + alpha * global_wr) / (df['hc_r'] + alpha)
    df['horse_career_top3r'] = (df['hc_t3'] + alpha * global_t3) / (df['hc_r'] + alpha)

    # Cleanup
    drop = [c for c in df.columns if any(c.startswith(p) for p in ['hd_', 'hs_', 'jc_', 'fr_', 'hc_'])]
    drop += ['dist_cat_apt', 'fk', 'is_win', 'is_top3']
    df = df.drop(columns=[c for c in drop if c in df.columns], errors='ignore')

    return df


def train_and_evaluate(df, features, label, use_time_split=True):
    """Train and evaluate a model."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Features: {len(features)}")
    print(f"={'=' * 60}")

    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X = df[features].values
    y = df['target'].values

    if use_time_split:
        # Time-based split: last 20% of races by race_id order
        n = len(df)
        split_idx = int(n * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        df_test = df.iloc[split_idx:].copy()
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        df_test = None

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Target rate: train={y_train.mean():.3f}, test={y_test.mean():.3f}")

    # Distance distribution in test
    if df_test is not None:
        dist_counts = df_test['distance'].value_counts().sort_index()
        print(f"  Test distance distribution: {dist_counts.to_dict()}")

    # LightGBM
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'learning_rate': 0.04, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 20,
        'reg_alpha': 0.3, 'reg_lambda': 0.3, 'verbose': -1,
        'n_jobs': -1, 'seed': 42,
    }
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=features)
    dtest = lgb.Dataset(X_test, label=y_test, feature_name=features, reference=dtrain)
    lgb_model = lgb.train(
        params, dtrain, num_boost_round=2000,
        valid_sets=[dtest],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
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
            'reg_alpha': 0.3, 'reg_lambda': 0.3, 'seed': 42,
            'tree_method': 'hist', 'verbosity': 0,
        }
        xgb_model = xgb_lib.train(
            xgb_params, dtrain_xgb, num_boost_round=2000,
            evals=[(dtest_xgb, 'valid')],
            early_stopping_rounds=50, verbose_eval=200,
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
        bar = '#' * int(imp / max(fi[0][1], 1) * 25)
        print(f"    {fname:25s} {imp:10.1f} {bar}")

    # Backtest by condition
    all_X = df[features].values
    all_scores = lgb_model.predict(all_X)
    if xgb_model:
        import xgboost as xgb_lib
        xgb_all = xgb_model.predict(xgb_lib.DMatrix(all_X))
        all_scores = all_scores * w_lgb + xgb_all * w_xgb

    df_bt = df.copy()
    df_bt['score'] = all_scores

    # Load cache for payouts
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r', encoding='utf-8') as cf:
            cache = json.load(cf)

    condition_results = {'A': [], 'B': [], 'C': [], 'D': [], 'E': [], 'X': []}
    for rid in df_bt['race_id'].unique():
        race_df = df_bt[df_bt['race_id'] == rid].sort_values('score', ascending=False)
        if len(race_df) < 3:
            continue
        num_h = int(race_df['num_horses'].iloc[0])
        dist = int(race_df['distance'].iloc[0])
        cond = race_df['condition'].iloc[0] if 'condition' in race_df.columns else ''
        ck = classify_condition(num_h, dist, cond)

        ranking = race_df['umaban'].astype(int).tolist()
        actual = dict(zip(race_df['umaban'].astype(int), race_df['finish'].astype(int)))
        trio_bets, wide_bets, umaren_bets = calc_bets(ranking)
        trio_hit, wide_hits, umaren_hits = check_hits(actual, trio_bets, wide_bets, umaren_bets)

        rc = cache.get(str(rid), {})
        payouts = rc.get('payouts', {'trio': 0, 'umaren': 0, 'wide': []})
        condition_results[ck].append({
            'race_id': rid,
            'trio_hit': trio_hit,
            'trio_payout': payouts.get('trio', 0) if trio_hit else 0,
            'wide_hits': len(wide_hits),
            'wide_payout': sum(payouts.get('wide', [])[:len(wide_hits)]) if wide_hits else 0,
            'umaren_hit': len(umaren_hits) > 0,
            'umaren_payout': payouts.get('umaren', 0) if umaren_hits else 0,
        })

    print(f"\n  {'COND':<4} {'N':>5} | {'BET':<7} {'HIT':>4} {'RATE':>7}")
    print(f"  {'-' * 40}")
    best_conditions = {}
    for ck in ['A', 'B', 'C', 'D', 'E', 'X']:
        races = condition_results.get(ck, [])
        n = len(races)
        if n == 0:
            best_conditions[ck] = {'n': 0, 'bet_type': 'trio', 'hit_rate': 0, 'roi': 0, 'recommended': False}
            continue

        results_by_bet = {}
        for bt, n_bets, hit_key, pay_key in [
            ('trio', 7, 'trio_hit', 'trio_payout'),
            ('umaren', 2, 'umaren_hit', 'umaren_payout'),
            ('wide', 2, 'wide_hits', 'wide_payout'),
        ]:
            hits = sum(1 for r in races if (r.get(hit_key, 0) > 0 if bt == 'wide' else r.get(hit_key, False)))
            investment = n * n_bets * 100
            total_payout = sum(r.get(pay_key, 0) for r in races)
            roi = total_payout / investment * 100 if investment > 0 else 0
            hit_rate = hits / n * 100
            results_by_bet[bt] = {'hits': hits, 'hit_rate': hit_rate, 'roi': roi}

        best_bt = max(results_by_bet, key=lambda b: results_by_bet[b]['roi'])
        best = results_by_bet[best_bt]
        recommended = best['roi'] >= 80
        best_conditions[ck] = {
            'n': n, 'bet_type': best_bt,
            'hits': best['hits'], 'hit_rate': best['hit_rate'],
            'roi': best['roi'], 'recommended': recommended,
        }
        for bt in ['trio', 'umaren', 'wide']:
            r = results_by_bet[bt]
            marker = ' <<<' if bt == best_bt else ''
            print(f"  {ck if bt == 'trio' else '':4} {n if bt == 'trio' else '':>5} | {bt:<7} {r['hits']:>4} {r['hit_rate']:>6.1f}%{marker}")

    return {
        'lgb_model': lgb_model, 'xgb_model': xgb_model,
        'lgb_auc': lgb_auc, 'xgb_auc': xgb_auc, 'ens_auc': ens_auc,
        'w_lgb': w_lgb, 'w_xgb': w_xgb,
        'best_conditions': best_conditions,
    }


def main():
    print("=" * 60)
    print("  KEIBA AI NAR V4 LEAK-FREE TRAINING")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Data: nar_merged.csv (all distances)")
    print("=" * 60)

    # Load data
    df = pd.read_csv(CSV_PATH)
    print(f"  CSV: {len(df)} rows, {df['race_id'].nunique()} races")
    print(f"  Distance distribution:")
    print(f"    {df['distance'].value_counts().sort_index().to_dict()}")
    print(f"  Source distribution:")
    if 'source' in df.columns:
        print(f"    {df['source'].value_counts().to_dict()}")

    # Sort by race_id for time-based operations
    df = df.sort_values('race_id').reset_index(drop=True)

    # Ensure target column
    if 'target' not in df.columns:
        df['target'] = (df['finish'] <= 3).astype(int)

    # Load existing model for jockey stats fallback
    jockey_stats = {}
    if os.path.exists(NAR_MODEL_PATH):
        with open(NAR_MODEL_PATH, 'rb') as f:
            v2 = pickle.load(f)
        jockey_stats = v2.get('jockey_stats', {})
        prev_auc = v2.get('ensemble_auc', 0)
        print(f"  Previous model AUC: {prev_auc:.4f}")
    else:
        prev_auc = 0

    # Compute new features
    print("\n  Computing expanding-window features...")
    df = compute_expanding_features(df)

    # Train V2a baseline (base features, random split like original)
    print("\n  === BASELINE: V2a (base features, 1600m-only equivalent) ===")
    df_1600 = df[df['distance'] == 1600].copy()
    res_v2a_1600 = train_and_evaluate(df_1600, NAR_FEATURES_BASE, "V2a baseline (1600m only)", use_time_split=False)

    # Train V4 base (all distances, time split)
    res_v4_base = train_and_evaluate(df.copy(), NAR_FEATURES_BASE, "V4 base (all distances, time split)")

    # Train V4 full (all distances + new features)
    res_v4_full = train_and_evaluate(df.copy(), NAR_FEATURES_V4, "V4 full (all distances + new features)")

    # Summary
    print("\n" + "=" * 60)
    print("  NAR RESULTS COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<40} {'LGB':>8} {'XGB':>8} {'Ens':>8}")
    print(f"  {'-' * 64}")
    print(f"  {'V2a baseline (1600m only, random)':<40} {res_v2a_1600['lgb_auc']:>8.4f} {res_v2a_1600['xgb_auc']:>8.4f} {res_v2a_1600['ens_auc']:>8.4f}")
    print(f"  {'V4 base (all dist, time split)':<40} {res_v4_base['lgb_auc']:>8.4f} {res_v4_base['xgb_auc']:>8.4f} {res_v4_base['ens_auc']:>8.4f}")
    print(f"  {'V4 full (all dist + new features)':<40} {res_v4_full['lgb_auc']:>8.4f} {res_v4_full['xgb_auc']:>8.4f} {res_v4_full['ens_auc']:>8.4f}")

    # Determine best
    best_res = res_v4_full
    best_label = 'V4 full'
    best_features = NAR_FEATURES_V4
    if res_v4_base['ens_auc'] > res_v4_full['ens_auc']:
        best_res = res_v4_base
        best_label = 'V4 base'
        best_features = NAR_FEATURES_BASE

    print(f"\n  Best: {best_label} (AUC {best_res['ens_auc']:.4f})")

    # Save
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nar_pkl = {
        'model': best_res['lgb_model'],
        'xgb_model': best_res['xgb_model'],
        'features': best_features,
        'version': f'nar_v4_leakfree',
        'auc': best_res['lgb_auc'],
        'ensemble_auc': best_res['ens_auc'],
        'ensemble_weights': {'lgb': best_res['w_lgb'], 'xgb': best_res['w_xgb']},
        'model_type': 'nar_dedicated',
        'trained_at': now,
        'jockey_stats': jockey_stats,
        'leak_free': True,
        'leak_pattern': 'A',
        'leak_removed': sorted(LEAK_A),
        'data_source': 'nar_merged.csv',
        'n_rows': len(df),
        'distance_range': f"{int(df['distance'].min())}-{int(df['distance'].max())}m",
        'condition_results': {k: {
            'bet_type': v['bet_type'], 'hit_rate': v.get('hit_rate', 0),
            'roi': v.get('roi', 0), 'recommended': v['recommended'], 'n': v['n'],
        } for k, v in best_res['best_conditions'].items()},
    }

    with open(NAR_MODEL_PATH, 'wb') as f:
        pickle.dump(nar_pkl, f)
    print(f"\n  Saved: {NAR_MODEL_PATH}")

    # Save results
    results = {
        'generated_at': now,
        'data': {'file': 'nar_merged.csv', 'rows': len(df), 'distances': df['distance'].value_counts().sort_index().to_dict()},
        'v2a_1600only': {'lgb': res_v2a_1600['lgb_auc'], 'xgb': res_v2a_1600['xgb_auc'], 'ens': res_v2a_1600['ens_auc']},
        'v4_base': {'lgb': res_v4_base['lgb_auc'], 'xgb': res_v4_base['xgb_auc'], 'ens': res_v4_base['ens_auc']},
        'v4_full': {'lgb': res_v4_full['lgb_auc'], 'xgb': res_v4_full['xgb_auc'], 'ens': res_v4_full['ens_auc']},
        'best': best_label,
        'conditions': {k: {'n': v['n'], 'bet_type': v['bet_type'], 'hit_rate': v.get('hit_rate', 0),
                            'roi': v.get('roi', 0)} for k, v in best_res['best_conditions'].items()},
    }

    import json
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            return super().default(obj)

    rpath = os.path.join(OUTPUT_DIR, 'v4_training_results_nar.json')
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NpEncoder)
    print(f"  Saved: {rpath}")

    print("\n  NAR V4 training complete!")
    return results


if __name__ == '__main__':
    main()
