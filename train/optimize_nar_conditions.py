#!/usr/bin/env python
"""NAR Condition Optimization with Short-Distance Focus
Train separate models for KDSCOPE (short) and netkeiba (1600m),
then exhaustively search for profitable conditions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

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
from itertools import product

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..')
MERGED_PATH = os.path.join(OUTPUT_DIR, 'data', 'nar_merged.csv')
NAR_MODEL_PATH = os.path.join(OUTPUT_DIR, 'keiba_model_v9_nar.pkl')
CACHE_PATH = os.path.join(OUTPUT_DIR, 'data', 'nar_scraped_cache.json')

# Pattern A leak-free
LEAK_A = {'odds_log', 'horse_weight', 'condition_enc', 'weight_cat', 'pop_rank'}

FEATURES_BASE = [
    'num_horses', 'distance', 'surface_enc', 'course_enc',
    'weight_carry', 'age', 'sex_enc',
    'horse_num', 'bracket', 'jockey_wr', 'jockey_place_rate', 'trainer_wr',
    'prev_finish', 'prev2_finish', 'prev3_finish', 'avg_finish_3r',
    'best_finish_3r', 'top3_count_3r', 'finish_trend', 'prev_odds_log',
    'rest_days', 'rest_category', 'dist_cat', 'age_group',
    'horse_num_ratio', 'bracket_pos', 'carry_diff', 'dist_change',
    'dist_change_abs', 'is_nar',
]

# Course code mapping
COURSE_NAMES = {42: 'ooi', 43: 'funabashi', 44: 'urawa', 45: 'kawasaki'}

# Distance band definitions
DIST_BANDS = {
    'D800': (0, 800),
    'D900': (801, 1000),
    'D1200': (1001, 1200),
    'D1400': (1201, 1400),
    'D1500': (1401, 1500),
    'D1600': (1501, 9999),
}

# Head count groups
HEAD_GROUPS = {
    'H_tiny': (1, 4),    # 1-4 horses
    'H_small': (5, 7),   # 5-7 horses
    'H_mid': (8, 10),    # 8-10 horses
    'H_large': (11, 99), # 11+ horses
}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


def calc_bets(ranking):
    """Generate betting combinations."""
    if len(ranking) < 3:
        return [], [], []
    nums = ranking[:6] if len(ranking) >= 6 else ranking
    n1 = nums[0]
    second = nums[1:3]
    third = nums[1:min(6, len(nums))]
    trio = sorted(set(tuple(sorted({n1, s, t})) for s in second for t in third if len(set({n1, s, t})) == 3))
    umaren = [sorted([n1, nums[1]]), sorted([n1, nums[2]])]
    wide = [sorted([n1, nums[1]]), sorted([n1, nums[2]])]
    return trio, wide, umaren


def check_hits(actual, trio, wide, umaren):
    top3 = set(u for u, f in actual.items() if f <= 3)
    top2 = set(u for u, f in actual.items() if f <= 2)
    trio_hit = any(set(c) == top3 for c in trio)
    wide_hits = [b for b in wide if set(b).issubset(top3)]
    umaren_hits = [b for b in umaren if set(b) == top2]
    return trio_hit, wide_hits, umaren_hits


def train_model(df, features, label):
    """Train LGB model on given data with random split."""
    print(f"\n  Training: {label}")
    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X = df[features].values
    y = df['target'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 15, 'learning_rate': 0.04, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 10,
        'reg_alpha': 0.5, 'reg_lambda': 0.5, 'verbose': -1,
        'n_jobs': -1, 'seed': 42,
    }
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=features)
    dtest = lgb.Dataset(X_te, label=y_te, feature_name=features, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=2000,
                      valid_sets=[dtest],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    pred = model.predict(X_te)
    auc = roc_auc_score(y_te, pred)
    print(f"  {label} AUC: {auc:.4f} (train={len(X_tr)}, test={len(X_te)})")
    return model, auc


def score_races(df, model, features):
    """Score all races and return df with scores."""
    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    df = df.copy()
    df['score'] = model.predict(df[features].values)
    return df


def backtest_races(df, cache):
    """Run backtest on scored dataframe, return per-race results."""
    results = []
    for rid in df['race_id'].unique():
        race = df[df['race_id'] == rid].sort_values('score', ascending=False)
        if len(race) < 3:
            continue

        num_h = int(race['num_horses'].iloc[0])
        dist = int(race['distance'].iloc[0])
        cond = race['condition'].iloc[0] if 'condition' in race.columns else ''
        course = int(race['course_enc'].iloc[0])

        ranking = race['umaban'].astype(int).tolist()
        actual = dict(zip(race['umaban'].astype(int), race['finish'].astype(int)))
        trio, wide, umaren = calc_bets(ranking)
        trio_hit, wide_hits, umaren_hits = check_hits(actual, trio, wide, umaren)

        rc = cache.get(str(rid), {})
        payouts = rc.get('payouts', {'trio': 0, 'umaren': 0, 'wide': []})

        results.append({
            'race_id': rid,
            'distance': dist,
            'num_horses': num_h,
            'condition': cond,
            'course_enc': course,
            'trio_hit': trio_hit,
            'trio_payout': payouts.get('trio', 0) if trio_hit else 0,
            'umaren_hit': len(umaren_hits) > 0,
            'umaren_payout': payouts.get('umaren', 0) if umaren_hits else 0,
            'wide_hits': len(wide_hits),
            'wide_payout': sum(payouts.get('wide', [])[:len(wide_hits)]) if wide_hits else 0,
        })
    return pd.DataFrame(results)


def evaluate_condition(bt_df, condition_mask, min_n=30):
    """Evaluate a condition across all bet types."""
    sub = bt_df[condition_mask]
    n = len(sub)
    if n < min_n:
        return None

    bet_results = {}
    for bt, n_bets, hit_col, pay_col in [
        ('trio', 7, 'trio_hit', 'trio_payout'),
        ('umaren', 2, 'umaren_hit', 'umaren_payout'),
        ('wide', 2, 'wide_hits', 'wide_payout'),
    ]:
        if bt == 'wide':
            hits = (sub[hit_col] > 0).sum()
        else:
            hits = sub[hit_col].sum()
        hit_rate = hits / n * 100
        investment = n * n_bets * 100
        payout = sub[pay_col].sum()
        roi = payout / investment * 100 if investment > 0 else 0
        bet_results[bt] = {
            'hits': int(hits), 'hit_rate': round(hit_rate, 1),
            'investment': int(investment), 'payout': int(payout),
            'roi': round(roi, 1),
        }

    best_bt = max(bet_results, key=lambda b: bet_results[b]['hit_rate'])
    return {
        'n': n,
        'bets': bet_results,
        'best_bet': best_bt,
        'best_hit_rate': bet_results[best_bt]['hit_rate'],
        'best_roi': bet_results[best_bt]['roi'],
    }


def grid_search_conditions(bt_df, min_n=30):
    """Exhaustive condition search."""
    print("\n  Grid searching conditions...")
    all_conditions = []

    # Distance bands
    for dname, (dlo, dhi) in DIST_BANDS.items():
        dist_mask = (bt_df['distance'] >= dlo) & (bt_df['distance'] <= dhi)

        # Distance only
        res = evaluate_condition(bt_df, dist_mask, min_n)
        if res:
            all_conditions.append({'name': dname, 'filters': {'dist': dname}, **res})

        # Distance x head count
        for hname, (hlo, hhi) in HEAD_GROUPS.items():
            head_mask = (bt_df['num_horses'] >= hlo) & (bt_df['num_horses'] <= hhi)
            combined = dist_mask & head_mask
            res = evaluate_condition(bt_df, combined, min_n)
            if res:
                all_conditions.append({'name': f'{dname}_{hname}', 'filters': {'dist': dname, 'heads': hname}, **res})

        # Distance x course (only for large courses)
        for cenc in bt_df['course_enc'].unique():
            course_mask = bt_df['course_enc'] == cenc
            combined = dist_mask & course_mask
            res = evaluate_condition(bt_df, combined, min_n)
            if res:
                cname = f'C{cenc}'
                all_conditions.append({'name': f'{dname}_{cname}', 'filters': {'dist': dname, 'course': cenc}, **res})

                # Distance x course x head count
                for hname, (hlo, hhi) in HEAD_GROUPS.items():
                    head_mask = (bt_df['num_horses'] >= hlo) & (bt_df['num_horses'] <= hhi)
                    triple = dist_mask & course_mask & head_mask
                    res = evaluate_condition(bt_df, triple, min_n)
                    if res:
                        all_conditions.append({
                            'name': f'{dname}_{cname}_{hname}',
                            'filters': {'dist': dname, 'course': cenc, 'heads': hname},
                            **res
                        })

    # Head count only (across all distances)
    for hname, (hlo, hhi) in HEAD_GROUPS.items():
        head_mask = (bt_df['num_horses'] >= hlo) & (bt_df['num_horses'] <= hhi)
        res = evaluate_condition(bt_df, head_mask, min_n)
        if res:
            all_conditions.append({'name': f'ALL_{hname}', 'filters': {'heads': hname}, **res})

    # Sort by best hit rate
    all_conditions.sort(key=lambda x: x['best_hit_rate'], reverse=True)
    return all_conditions


def walk_forward_validate(df, features, conditions, cache, n_splits=3):
    """Validate conditions across time splits."""
    print("\n  Walk-forward validation...")
    df = df.sort_values('race_id').reset_index(drop=True)
    n = len(df)
    fold_size = n // (n_splits + 1)

    stable_conditions = []

    for cond in conditions:
        fold_results = []
        for fold in range(n_splits):
            train_end = fold_size * (fold + 2)
            test_start = train_end
            test_end = min(train_end + fold_size, n)
            if test_end <= test_start:
                continue

            df_train = df.iloc[:train_end].copy()
            df_test = df.iloc[test_start:test_end].copy()

            # Train model on fold
            model, _ = train_model(df_train, features, f"WF fold {fold}")

            # Score test
            df_scored = score_races(df_test, model, features)
            bt = backtest_races(df_scored, cache)
            if len(bt) == 0:
                continue

            # Apply condition filter
            mask = pd.Series([True] * len(bt))
            filters = cond['filters']
            if 'dist' in filters:
                dlo, dhi = DIST_BANDS[filters['dist']]
                mask &= (bt['distance'] >= dlo) & (bt['distance'] <= dhi)
            if 'heads' in filters:
                hlo, hhi = HEAD_GROUPS[filters['heads']]
                mask &= (bt['num_horses'] >= hlo) & (bt['num_horses'] <= hhi)
            if 'course' in filters:
                mask &= bt['course_enc'] == filters['course']

            res = evaluate_condition(bt, mask, min_n=5)  # Lower min for WF folds
            if res:
                fold_results.append(res)

        if len(fold_results) >= 2:
            avg_hit = np.mean([r['best_hit_rate'] for r in fold_results])
            min_hit = min(r['best_hit_rate'] for r in fold_results)
            stability = min_hit / max(avg_hit, 1)
            cond['wf_folds'] = len(fold_results)
            cond['wf_avg_hit'] = round(avg_hit, 1)
            cond['wf_min_hit'] = round(min_hit, 1)
            cond['wf_stability'] = round(stability, 2)
            if stability >= 0.5 and avg_hit >= 30:
                stable_conditions.append(cond)

    stable_conditions.sort(key=lambda x: x['wf_avg_hit'], reverse=True)
    return stable_conditions


def select_final_conditions(stable_conditions, bt_df):
    """Select non-overlapping conditions that cover all races."""
    print("\n  Selecting final condition set...")

    # We want conditions that:
    # 1. Are stable (WF validated)
    # 2. Cover different segments
    # 3. Have high hit rates

    # Priority: specific (3-filter) > medium (2-filter) > broad (1-filter)
    # But also consider N and stability

    selected = []
    covered_races = set()

    # First pass: select best condition per distance band
    for dname in DIST_BANDS:
        candidates = [c for c in stable_conditions if c['filters'].get('dist') == dname]
        if not candidates:
            continue
        # Prefer conditions with higher hit rate and sufficient N
        candidates.sort(key=lambda x: (x['wf_avg_hit'], x['n']), reverse=True)
        best = candidates[0]
        selected.append(best)

    # Also add head-count-only conditions if they add value
    for hname in HEAD_GROUPS:
        candidates = [c for c in stable_conditions
                      if c['filters'].get('heads') == hname and 'dist' not in c['filters']]
        if candidates:
            best = candidates[0]
            if best not in selected and best['wf_avg_hit'] > 50:
                selected.append(best)

    return selected


def main():
    print("=" * 60)
    print("  NAR CONDITION OPTIMIZATION")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Load data
    df = pd.read_csv(MERGED_PATH)
    df['target'] = (df['finish'] <= 3).astype(int)
    print(f"  Data: {len(df)} rows, {df['race_id'].nunique()} races")

    # Load cache
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r', encoding='utf-8') as f:
            cache = json.load(f)

    # Load existing jockey stats
    jockey_stats = {}
    if os.path.exists(NAR_MODEL_PATH):
        with open(NAR_MODEL_PATH, 'rb') as f:
            prev = pickle.load(f)
        jockey_stats = prev.get('jockey_stats', {})

    # ================================================================
    # 1. DISTANCE-BAND ANALYSIS
    # ================================================================
    print("\n" + "=" * 60)
    print("  1. DISTANCE-BAND ANALYSIS")
    print("=" * 60)

    # Train model on ALL data (random split)
    model_all, auc_all = train_model(df.copy(), FEATURES_BASE, "ALL data combined")

    # Score all races
    df_scored = score_races(df.copy(), model_all, FEATURES_BASE)

    # Backtest all
    bt_all = backtest_races(df_scored, cache)
    print(f"\n  Total backtest races: {len(bt_all)}")

    # Per distance band stats
    print(f"\n  {'Band':<10} {'N':>6} {'Trio Hit':>8} {'Trio%':>7} {'Umaren%':>8} {'Wide%':>7}")
    print(f"  {'-' * 50}")
    for dname, (dlo, dhi) in DIST_BANDS.items():
        mask = (bt_all['distance'] >= dlo) & (bt_all['distance'] <= dhi)
        sub = bt_all[mask]
        n = len(sub)
        if n == 0:
            continue
        trio_hr = sub['trio_hit'].sum() / n * 100
        um_hr = sub['umaren_hit'].sum() / n * 100
        wd_hr = (sub['wide_hits'] > 0).sum() / n * 100
        print(f"  {dname:<10} {n:>6} {int(sub['trio_hit'].sum()):>8} {trio_hr:>6.1f}% {um_hr:>7.1f}% {wd_hr:>6.1f}%")

    # ================================================================
    # 2. GRID SEARCH CONDITIONS
    # ================================================================
    print("\n" + "=" * 60)
    print("  2. GRID SEARCH CONDITIONS (N>=30)")
    print("=" * 60)

    all_conditions = grid_search_conditions(bt_all, min_n=30)
    print(f"\n  Found {len(all_conditions)} conditions with N>=30")

    # Show top 20
    print(f"\n  {'Rank':<4} {'Name':<30} {'N':>5} {'BestBet':<8} {'HitRate':>7} {'ROI':>7}")
    print(f"  {'-' * 65}")
    for i, c in enumerate(all_conditions[:20]):
        print(f"  {i+1:<4} {c['name']:<30} {c['n']:>5} {c['best_bet']:<8} {c['best_hit_rate']:>6.1f}% {c['best_roi']:>6.1f}%")

    # ================================================================
    # 3. WALK-FORWARD VALIDATION
    # ================================================================
    print("\n" + "=" * 60)
    print("  3. WALK-FORWARD VALIDATION (top conditions)")
    print("=" * 60)

    # Take top 30 conditions for WF validation
    top_conditions = all_conditions[:30]
    stable = walk_forward_validate(df.copy(), FEATURES_BASE, top_conditions, cache, n_splits=3)

    print(f"\n  Stable conditions (WF validated): {len(stable)}")
    print(f"\n  {'Name':<30} {'N':>5} {'Hit%':>6} {'WF_Avg':>7} {'WF_Min':>7} {'Stab':>5}")
    print(f"  {'-' * 65}")
    for c in stable[:15]:
        print(f"  {c['name']:<30} {c['n']:>5} {c['best_hit_rate']:>5.1f}% {c['wf_avg_hit']:>6.1f}% {c['wf_min_hit']:>6.1f}% {c['wf_stability']:>4.2f}")

    # ================================================================
    # 4. SELECT FINAL CONDITIONS
    # ================================================================
    print("\n" + "=" * 60)
    print("  4. FINAL CONDITION SET")
    print("=" * 60)

    final = select_final_conditions(stable, bt_all)

    # Build condition profiles for app.py
    new_profiles = {}
    condition_map = {}  # For classify function

    for i, c in enumerate(final):
        key = chr(ord('A') + i) if i < 6 else f'Z{i}'
        filters = c['filters']

        # Build description
        parts = []
        if 'dist' in filters:
            dlo, dhi = DIST_BANDS[filters['dist']]
            parts.append(f'{dlo}-{dhi}m')
        if 'heads' in filters:
            hlo, hhi = HEAD_GROUPS[filters['heads']]
            parts.append(f'{hlo}-{hhi}head')
        if 'course' in filters:
            parts.append(f'C{filters["course"]}')
        desc = ' / '.join(parts)

        # Determine bet type and stats
        bets = c['bets']
        best_bt = c['best_bet']
        best_stats = bets[best_bt]

        # Bet label
        if best_bt == 'trio':
            bet_label = 'trio 7pt'
            investment = 700
        elif best_bt == 'umaren':
            bet_label = 'umaren 2pt'
            investment = 700
        else:
            bet_label = 'wide 2pt'
            investment = 700

        profile = {
            'label': f'NAR {key}',
            'desc': desc,
            'bet_type': best_bt,
            'bet_label': bet_label,
            'investment': investment,
            'roi': best_stats['roi'],
            'hit_rate': best_stats['hit_rate'],
            'recommended': best_stats['hit_rate'] >= 30,
            'wf_n': c.get('wf_folds', 0) * c['n'] // 3,
            'n_backtest': c['n'],
            'wf_avg_hit': c.get('wf_avg_hit', 0),
            'wf_stability': c.get('wf_stability', 0),
            'filters': filters,
        }
        new_profiles[key] = profile
        condition_map[key] = filters

        trio_info = bets['trio']
        um_info = bets['umaren']
        wd_info = bets['wide']
        print(f"\n  {key}: {c['name']} (N={c['n']})")
        print(f"    Trio:   Hit {trio_info['hit_rate']}%, ROI {trio_info['roi']}%")
        print(f"    Umaren: Hit {um_info['hit_rate']}%, ROI {um_info['roi']}%")
        print(f"    Wide:   Hit {wd_info['hit_rate']}%, ROI {wd_info['roi']}%")
        print(f"    Best: {best_bt} (Hit {best_stats['hit_rate']}%)")
        print(f"    WF: avg={c.get('wf_avg_hit',0)}%, stability={c.get('wf_stability',0)}")

    # Also add 1600m conditions from existing V2a if available
    # Keep old conditions for 1600m data
    print(f"\n  Adding 1600m conditions from existing V2a model...")
    # Load V2a model for 1600m scoring
    with open(NAR_MODEL_PATH, 'rb') as f:
        v2a = pickle.load(f)
    v2a_model = v2a['model']
    v2a_features = v2a['features']

    df_1600 = df[df['distance'] == 1600].copy()
    if len(df_1600) > 0:
        # Apply V2a jockey stats
        for jname in df_1600['jockey_name'].unique():
            js = jockey_stats.get(jname, {'wr': 0.08, 'place_rate': 0.25})
            df_1600.loc[df_1600['jockey_name'] == jname, 'jockey_wr'] = js.get('wr', 0.08)
            df_1600.loc[df_1600['jockey_name'] == jname, 'jockey_place_rate'] = js.get('place_rate', 0.25)

        for f in v2a_features:
            if f not in df_1600.columns:
                df_1600[f] = 0
            df_1600[f] = pd.to_numeric(df_1600[f], errors='coerce').fillna(0)

        df_1600['score'] = v2a_model.predict(df_1600[v2a_features].values)
        bt_1600 = backtest_races(df_1600, cache)

        # Evaluate 1600m conditions
        for hname, (hlo, hhi) in [('H_mid', (8, 10)), ('H_large', (11, 99))]:
            head_mask = (bt_1600['num_horses'] >= hlo) & (bt_1600['num_horses'] <= hhi)
            res = evaluate_condition(bt_1600, head_mask, min_n=10)
            if res:
                key = chr(ord('A') + len(new_profiles))
                new_profiles[key] = {
                    'label': f'NAR {key}',
                    'desc': f'1600m+ / {hlo}-{hhi}head (V2a)',
                    'bet_type': res['best_bet'],
                    'bet_label': f'{res["best_bet"]}',
                    'investment': 700,
                    'roi': res['bets'][res['best_bet']]['roi'],
                    'hit_rate': res['best_hit_rate'],
                    'recommended': res['best_hit_rate'] >= 30,
                    'n_backtest': res['n'],
                    'wf_n': 0,
                    'filters': {'dist': 'D1600', 'heads': hname},
                    'note': 'V2a model, small sample',
                }
                print(f"\n  {key}: 1600m/{hname} (N={res['n']}, V2a model)")
                for bt in ['trio', 'umaren', 'wide']:
                    info = res['bets'][bt]
                    print(f"    {bt}: Hit {info['hit_rate']}%, ROI {info['roi']}%")

    # ================================================================
    # 5. SAVE RESULTS
    # ================================================================
    print("\n" + "=" * 60)
    print("  5. SAVING RESULTS")
    print("=" * 60)

    # Save optimal betting config
    optimal = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_auc': auc_all,
        'data_rows': len(df),
        'data_races': int(df['race_id'].nunique()),
        'profiles': new_profiles,
    }
    opt_path = os.path.join(OUTPUT_DIR, 'data', 'optimal_betting_nar_v2.json')
    with open(opt_path, 'w', encoding='utf-8') as f:
        json.dump(optimal, f, ensure_ascii=False, indent=2, cls=NpEncoder)
    print(f"  Saved: {opt_path}")

    # Save full search log
    search_log = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'all_conditions': [{k: v for k, v in c.items()} for c in all_conditions],
        'stable_conditions': [{k: v for k, v in c.items()} for c in stable],
        'final_profiles': new_profiles,
    }
    log_path = os.path.join(OUTPUT_DIR, 'data', 'condition_optimization_nar_v2.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(search_log, f, ensure_ascii=False, indent=2, cls=NpEncoder)
    print(f"  Saved: {log_path}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("  OLD vs NEW CONDITIONS COMPARISON")
    print("=" * 60)
    print(f"\n  OLD (current app.py):")
    old_profiles = {
        'A': 'NAR A: 8-14head/1600m+/good (N=69, trio 65.2%, ROI 366%)',
        'B': 'NAR B: 8-14head/1600m+/heavy (N=83, trio 49.4%, ROI 432%)',
        'C': 'NAR C: 15+head/1600m+/good (N=2, INSUFFICIENT)',
        'D': 'NAR D: <=1400m sprint (NO DATA)',
        'E': 'NAR E: <=7head (N=30, umaren 60%, ROI 350%)',
        'X': 'NAR X: 15+/heavy (NO DATA)',
    }
    for k, v in old_profiles.items():
        print(f"    {v}")

    print(f"\n  NEW (optimized):")
    for k, p in new_profiles.items():
        rec = 'REC' if p['recommended'] else '---'
        print(f"    {k}: {p['desc']} (N={p.get('n_backtest',0)}, {p['bet_type']} {p['hit_rate']}%, ROI {p['roi']}%) [{rec}]")

    return new_profiles


if __name__ == '__main__':
    profiles = main()
