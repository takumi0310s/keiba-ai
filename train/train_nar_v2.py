#!/usr/bin/env python
"""KEIBA AI NAR V2 - Tuned hyperparameters + condition-based backtest
- Retrains on chihou_races_2020_2025.csv (netkeiba scraped data)
- chihou_races_full.csv (KDSCOPE): 0 jockey overlap, 6% horse overlap with stale data
  -> Not useful for training enrichment, kept for reference only
- Tuned LightGBM (num_leaves=15, lr=0.04, reg=0.5) + XGBoost ensemble
- Condition-based backtest with ROI/stars rating
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
LOG_PATH = os.path.join(OUTPUT_DIR, 'train', 'nar_v2_training.log')

NAR_FEATURES = [
    'odds_log', 'num_horses', 'distance', 'surface_enc', 'condition_enc',
    'course_enc', 'horse_weight', 'weight_carry', 'age', 'sex_enc',
    'horse_num', 'bracket', 'jockey_wr', 'jockey_place_rate', 'trainer_wr',
    'prev_finish', 'prev2_finish', 'prev3_finish', 'avg_finish_3r',
    'best_finish_3r', 'top3_count_3r', 'finish_trend', 'prev_odds_log',
    'rest_days', 'rest_category', 'dist_cat', 'weight_cat', 'age_group',
    'horse_num_ratio', 'bracket_pos', 'carry_diff', 'dist_change',
    'dist_change_abs', 'is_nar', 'pop_rank',
]


def log(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode())
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


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


def main():
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('')

    log("=" * 60)
    log("  KEIBA AI NAR V2 - Tuned Hyperparameters")
    log(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log("=" * 60)

    # Load V1 model for jockey stats
    with open(NAR_MODEL_PATH, 'rb') as f:
        v1 = pickle.load(f)
    jockey_stats = v1.get('jockey_stats', {})
    v1_auc = v1.get('auc', 0.789)
    v1_ens_auc = v1.get('ensemble_auc', 0.789)
    log(f"  V1 AUC: {v1_auc:.4f} / Ensemble: {v1_ens_auc:.4f}")
    log(f"  V1 jockey stats: {len(jockey_stats)} jockeys")

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    log(f"  CSV: {len(df)} rows, {df['race_id'].nunique()} races")

    # Build features (same as V1)
    df['jockey_wr'] = df['jockey_name'].map(lambda j: jockey_stats.get(j, {}).get('wr', 0.08))
    df['jockey_place_rate'] = df['jockey_name'].map(lambda j: jockey_stats.get(j, {}).get('place_rate', 0.25))
    df['trainer_wr'] = 0.10
    df['odds_log'] = np.log1p(df['odds'].clip(1, 999))
    df['dist_cat'] = pd.cut(df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
                            labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
    df['weight_cat'] = pd.cut(df['horse_weight'], bins=[0, 440, 480, 520, 9999],
                              labels=[0, 1, 2, 3]).astype(float).fillna(1)
    df['age_group'] = df['age'].clip(2, 7)
    df['horse_num_ratio'] = df['horse_num'] / df['num_horses'].clip(1)
    df['bracket_pos'] = pd.cut(df['bracket'], bins=[0, 3, 6, 8],
                                labels=[0, 1, 2]).astype(float).fillna(1)
    df['carry_diff'] = df['weight_carry'] - df['weight_carry'].mean()
    df['is_nar'] = 1

    for f_name in NAR_FEATURES:
        if f_name not in df.columns:
            df[f_name] = 0
        df[f_name] = pd.to_numeric(df[f_name], errors='coerce').fillna(0)

    X = df[NAR_FEATURES].values
    y = df['target'].values

    # Same split as V1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    log(f"  Target rate: train={y_train.mean():.3f}, test={y_test.mean():.3f}")

    # --- LightGBM V2 (tuned: fewer leaves, stronger regularization) ---
    log("\n  Training LightGBM V2...")
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 15, 'learning_rate': 0.04, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 10,
        'reg_alpha': 0.5, 'reg_lambda': 0.5, 'verbose': -1,
        'n_jobs': -1, 'seed': 42,
    }
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=NAR_FEATURES)
    dtest = lgb.Dataset(X_test, label=y_test, feature_name=NAR_FEATURES, reference=dtrain)

    lgb_model = lgb.train(
        params, dtrain, num_boost_round=2000,
        valid_sets=[dtest],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    lgb_pred = lgb_model.predict(X_test)
    lgb_auc = roc_auc_score(y_test, lgb_pred)
    log(f"  LightGBM V2 AUC: {lgb_auc:.4f} (V1: {v1_auc:.4f})")

    # --- XGBoost V2 ---
    xgb_model = None
    xgb_auc = 0
    try:
        import xgboost as xgb_lib
        log("  Training XGBoost V2...")
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
        log(f"  XGBoost V2 AUC: {xgb_auc:.4f}")

        total = lgb_auc + xgb_auc
        w_lgb = lgb_auc / total
        w_xgb = xgb_auc / total
        ens_pred = lgb_pred * w_lgb + xgb_pred * w_xgb
        ens_auc = roc_auc_score(y_test, ens_pred)
        log(f"  Ensemble AUC: {ens_auc:.4f} (V1: {v1_ens_auc:.4f})")
    except ImportError:
        w_lgb, w_xgb = 1.0, 0.0
        ens_auc = lgb_auc
        log("  XGBoost not available")

    weights = {'lgb': w_lgb, 'xgb': w_xgb}

    # Feature importance
    importance = lgb_model.feature_importance(importance_type='gain')
    fi = sorted(zip(NAR_FEATURES, importance), key=lambda x: x[1], reverse=True)
    log(f"\n  Feature Importance TOP 15:")
    for fname, imp in fi[:15]:
        bar = '#' * int(imp / fi[0][1] * 25)
        log(f"    {fname:25s} {imp:10.1f} {bar}")

    # --- Backtest on ALL 184 races ---
    log("\n" + "=" * 60)
    log("  NAR CONDITION-BASED BACKTEST (184 races)")
    log("=" * 60)

    # Load cache for payouts
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r', encoding='utf-8') as cf:
            cache = json.load(cf)
    log(f"  Cache: {len(cache)} races with payouts")

    # Predict on all data and backtest per race
    all_scores = lgb_model.predict(X)
    if xgb_model:
        xgb_all = xgb_model.predict(xgb_lib.DMatrix(X))
        all_scores = all_scores * w_lgb + xgb_all * w_xgb

    df['score'] = all_scores
    condition_results = {'A': [], 'B': [], 'C': [], 'D': [], 'E': [], 'X': []}

    for rid in df['race_id'].unique():
        race_df = df[df['race_id'] == rid].sort_values('score', ascending=False)
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

    # Analyze results
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

        best_bt = max(results_by_bet, key=lambda b: results_by_bet[b]['roi'])
        best = results_by_bet[best_bt]
        roi = best['roi']
        recommended = roi >= 80
        stars = 3 if roi >= 120 else (2 if roi >= 100 else (1 if roi >= 80 else 0))
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

    # Results comparison
    best_auc = max(lgb_auc, ens_auc)
    log(f"\n" + "=" * 60)
    log(f"  RESULTS")
    log(f"=" * 60)
    log(f"  V1 LGB AUC: {v1_auc:.4f} / Ensemble: {v1_ens_auc:.4f}")
    log(f"  V2 LGB AUC: {lgb_auc:.4f} / Ensemble: {ens_auc:.4f}")
    log(f"  Improvement: LGB {lgb_auc - v1_auc:+.4f} / Ensemble {ens_auc - v1_ens_auc:+.4f}")

    if best_auc > v1_ens_auc:
        log(f"\n  V2 WINS! Saving as production model...")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nar_pkl = {
            'model': lgb_model,
            'xgb_model': xgb_model,
            'features': NAR_FEATURES,
            'version': 'nar_v2',
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
            'version': 'nar_v2',
            'lgb_auc': lgb_auc,
            'ensemble_auc': ens_auc,
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
    else:
        log(f"\n  V2 did not beat V1. NOT updating production model.")

    # Summary
    log(f"\n  CONDITION SUMMARY:")
    for ckey in ['A', 'B', 'C', 'D', 'E', 'X']:
        info = best_conditions[ckey]
        if info['n'] == 0:
            log(f"  {ckey}: No data")
        else:
            star_str = '*' * info['stars'] if info['stars'] > 0 else 'X'
            rec = 'REC' if info['recommended'] else 'SKIP'
            log(f"  {ckey}: {info['bet_type']} ROI {info['roi']:.1f}% (N={info['n']}) -> {star_str} {rec}")

    log(f"\n  Done!")
    return best_auc, best_conditions


if __name__ == '__main__':
    main()
