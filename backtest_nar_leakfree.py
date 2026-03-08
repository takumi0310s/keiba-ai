#!/usr/bin/env python
"""NAR Leak-Free Backtest with Multi-Axis Condition Optimization
Uses Pattern A model (strict leak-free) with actual payouts from cache.
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
import time
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'chihou_races_2020_2025.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'keiba_model_v9_nar.pkl')
CACHE_PATH = os.path.join(BASE_DIR, 'data', 'nar_scraped_cache.json')

COURSE_MAP = {'大井': 44, '船橋': 43, '浦和': 44, '川崎': 45, '門別': 30, '園田': 27}


def classify_condition(num_horses, distance, condition):
    heavy = any(c in str(condition) for c in ['重', '不'])
    if num_horses <= 7: return 'E'
    if distance <= 1400: return 'D'
    if 8 <= num_horses <= 14 and distance >= 1600 and not heavy: return 'A'
    if 8 <= num_horses <= 14 and distance >= 1600 and heavy: return 'B'
    if num_horses >= 15 and distance >= 1600 and not heavy: return 'C'
    return 'X'


def get_axes(row):
    dist = int(row['distance'])
    nh = int(row['num_horses'])
    condition = str(row.get('condition', '良'))

    dist_label = 'sprint' if dist <= 1200 else ('mile' if dist <= 1600 else ('middle' if dist <= 2200 else 'long'))
    nh_label = 'small' if nh <= 7 else ('medium' if nh <= 14 else 'large')
    heavy = any(c in condition for c in ['重', '不'])
    cond_label = 'heavy' if heavy else 'good'
    course = str(row.get('course', ''))
    surface = str(row.get('surface', 'ダ'))
    surf_label = 'turf' if '芝' in surface else 'dirt'

    return {
        'cond_key': classify_condition(nh, dist, condition),
        'dist': dist_label, 'nh': nh_label, 'cond': cond_label,
        'course': course, 'surface': surf_label,
    }


def calc_bets(ranking):
    if len(ranking) < 3:
        return [], [], []
    nums = ranking[:6] if len(ranking) >= 6 else ranking
    n1 = nums[0]
    second = nums[1:3]
    third = nums[1:min(6, len(nums))]
    trio = sorted(set(
        tuple(sorted({n1, s, t}))
        for s in second for t in third
        if len(set({n1, s, t})) == 3
    ))
    umaren = [sorted([n1, nums[1]]), sorted([n1, nums[2]])]
    wide = [sorted([n1, nums[1]]), sorted([n1, nums[2]])]
    return trio, wide, umaren


def check_hits(actual, trio_bets, wide_bets, umaren_bets):
    top3 = set(uma for uma, fin in actual.items() if fin <= 3)
    top2 = set(uma for uma, fin in actual.items() if fin <= 2)
    trio_hit = any(set(c) == top3 for c in trio_bets)
    wide_hits = [b for b in wide_bets if set(b).issubset(top3)]
    umaren_hits = [b for b in umaren_bets if set(b) == top2]
    return trio_hit, wide_hits, umaren_hits


def analyze_conditions(results, min_n=5):
    """Multi-axis condition analysis with actual payouts."""
    axes = [
        ('cond_key', 'Condition A-X'),
        ('dist', 'Distance'),
        ('nh', 'NumHorses'),
        ('cond', 'TrackCond'),
        ('course', 'Course'),
    ]
    cross_axes = [
        (('cond_key', 'cond'), 'Condition×Track'),
        (('dist', 'cond'), 'Distance×Track'),
    ]

    all_analysis = {}
    for axis_key, axis_name in axes:
        groups = defaultdict(list)
        for r in results:
            groups[r['axes'][axis_key]].append(r)
        all_analysis[axis_name] = _analyze_groups(groups, min_n)

    for (k1, k2), axis_name in cross_axes:
        groups = defaultdict(list)
        for r in results:
            key = f"{r['axes'][k1]}_{r['axes'][k2]}"
            groups[key].append(r)
        all_analysis[axis_name] = _analyze_groups(groups, min_n)

    return all_analysis


def _analyze_groups(groups, min_n):
    analysis = {}
    for label, races in sorted(groups.items()):
        n = len(races)
        if n < min_n:
            continue
        bet_results = {}
        # All bet types: 700 yen per race
        # trio: 7×100, umaren: 2×350, wide: 2×350
        for bt in ['trio', 'umaren', 'wide']:
            if bt == 'trio':
                hits = sum(1 for r in races if r['trio_hit'])
                payout = sum(r['trio_payout'] for r in races)
            elif bt == 'umaren':
                hits = sum(1 for r in races if r['umaren_hit'])
                # Scale umaren payout: cache has per-100-yen, we bet 350 each
                payout = sum(r['umaren_payout'] * 3.5 for r in races)
            else:
                hits = sum(1 for r in races if r['wide_hits'] > 0)
                payout = sum(r['wide_payout'] * 3.5 for r in races)
            investment = n * 700  # Fixed 700 yen per race
            roi = payout / investment * 100 if investment > 0 else 0
            hit_rate = hits / n * 100
            bet_results[bt] = {
                'hits': hits, 'hit_rate': round(hit_rate, 1),
                'investment': investment, 'payout': int(payout),
                'roi': round(roi, 1),
            }
        best_bt = max(bet_results, key=lambda b: bet_results[b]['roi'])
        best_roi = bet_results[best_bt]['roi']
        stars = 3 if best_roi >= 120 else (2 if best_roi >= 100 else (1 if best_roi >= 80 else 0))
        analysis[label] = {
            'n': n, 'best_bet': best_bt, 'best_roi': best_roi,
            'stars': stars, 'recommended': best_roi >= 80,
            'bets': bet_results,
        }
    return analysis


def main():
    print("=" * 60)
    print("  NAR LEAK-FREE BACKTEST (Pattern A)")
    print("=" * 60)

    # Load model
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    lgb_model = model_data['model']
    xgb_model = model_data.get('xgb_model')
    features = model_data['features']
    jockey_stats = model_data.get('jockey_stats', {})
    weights = model_data.get('ensemble_weights', {'lgb': 1, 'xgb': 0})
    print(f"  Model: {model_data.get('version', '?')}")
    print(f"  Features: {len(features)}")
    print(f"  Jockey stats: {len(jockey_stats)} jockeys")

    # Load data
    df = pd.read_csv(CSV_PATH)
    print(f"  CSV: {len(df)} rows, {df['race_id'].nunique()} races")

    # Load cache (actual payouts)
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    print(f"  Cache: {len(cache)} races with payouts")

    # Build features
    df['jockey_wr'] = df['jockey_name'].map(lambda j: jockey_stats.get(j, {}).get('wr', 0.08))
    df['jockey_place_rate'] = df['jockey_name'].map(lambda j: jockey_stats.get(j, {}).get('place_rate', 0.25))
    df['trainer_wr'] = 0.10
    df['odds_log'] = np.log1p(df['odds'].clip(1, 999))
    df['dist_cat'] = pd.cut(df['distance'], bins=[0,1200,1400,1800,2200,9999],
                            labels=[0,1,2,3,4]).astype(float).fillna(2)
    df['weight_cat'] = pd.cut(df['horse_weight'], bins=[0,440,480,520,9999],
                              labels=[0,1,2,3]).astype(float).fillna(1)
    df['age_group'] = df['age'].clip(2, 7)
    df['horse_num_ratio'] = df['horse_num'] / df['num_horses'].clip(1)
    df['bracket_pos'] = pd.cut(df['bracket'], bins=[0,3,6,8], labels=[0,1,2]).astype(float).fillna(1)
    df['carry_diff'] = df['weight_carry'] - df['weight_carry'].mean()
    df['is_nar'] = 1

    for f_name in features:
        if f_name not in df.columns:
            df[f_name] = 0
        df[f_name] = pd.to_numeric(df[f_name], errors='coerce').fillna(0)

    # Predict
    X = df[features].values
    scores = lgb_model.predict(X)
    if xgb_model:
        import xgboost as xgb
        xgb_scores = xgb_model.predict(xgb.DMatrix(X))
        scores = scores * weights['lgb'] + xgb_scores * weights['xgb']
    df['score'] = scores

    # Per-race evaluation with actual payouts
    all_results = []
    for rid in df['race_id'].unique():
        race_df = df[df['race_id'] == rid].sort_values('score', ascending=False)
        if len(race_df) < 3:
            continue

        row0 = race_df.iloc[0]
        axes = get_axes(row0)

        ranking = race_df['umaban'].astype(int).tolist()
        actual = dict(zip(race_df['umaban'].astype(int), race_df['finish'].astype(int)))

        trio_bets, wide_bets, umaren_bets = calc_bets(ranking)
        trio_hit, wide_hits, umaren_hits = check_hits(actual, trio_bets, wide_bets, umaren_bets)

        # Actual payouts from cache
        race_cache = cache.get(str(rid), {})
        payouts = race_cache.get('payouts', {'trio': 0, 'umaren': 0, 'wide': []})

        trio_payout = payouts.get('trio', 0) if trio_hit else 0
        umaren_payout = payouts.get('umaren', 0) if umaren_hits else 0
        wide_payout = 0
        if wide_hits:
            wide_list = payouts.get('wide', [])
            wide_payout = sum(wide_list[:len(wide_hits)]) if wide_list else 0

        all_results.append({
            'race_id': rid,
            'axes': axes,
            'trio_hit': trio_hit, 'trio_payout': trio_payout,
            'umaren_hit': len(umaren_hits) > 0, 'umaren_payout': umaren_payout,
            'wide_hits': len(wide_hits), 'wide_payout': wide_payout,
        })

    print(f"\n  Evaluated: {len(all_results)} races")

    # Analysis (use min_n=5 for NAR since data is limited)
    analysis = analyze_conditions(all_results, min_n=5)

    for axis_name, groups in analysis.items():
        print(f"\n  --- {axis_name} ---")
        print(f"  {'Label':<25} {'N':>5} {'Best':>7} {'HitR':>6} {'ROI':>7} {'Stars'}")
        for label, info in sorted(groups.items(), key=lambda x: x[1]['best_roi'], reverse=True):
            best = info['bets'][info['best_bet']]
            stars_str = '*' * info['stars'] if info['stars'] > 0 else 'X'
            print(f"  {label:<25} {info['n']:>5} {info['best_bet']:>7} {best['hit_rate']:>5.1f}% {best['roi']:>6.1f}% {stars_str}")

    # Save JSON
    output = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_data.get('version', 'nar_v2a'),
        'total_races': len(all_results),
        'conditions': {},
    }

    cond_analysis = analysis.get('Condition A-X', {})
    for label, info in cond_analysis.items():
        output['conditions'][label] = {
            'n': info['n'], 'best_bet': info['best_bet'],
            'best_roi': info['best_roi'], 'stars': info['stars'],
            'recommended': info['recommended'], 'bets': info['bets'],
        }

    output['multi_axis'] = {}
    for axis_name, groups in analysis.items():
        if axis_name == 'Condition A-X':
            continue
        output['multi_axis'][axis_name] = {
            label: {
                'n': info['n'], 'best_bet': info['best_bet'],
                'best_roi': info['best_roi'], 'stars': info['stars'],
                'recommended': info['recommended'],
                'trio': info['bets']['trio'],
                'umaren': info['bets']['umaren'],
                'wide': info['bets']['wide'],
            } for label, info in groups.items()
        }

    out_path = os.path.join(BASE_DIR, 'data', 'optimal_betting_nar_leakfree.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")

    print("\n  NAR backtest complete!")
    return output


if __name__ == '__main__':
    main()
