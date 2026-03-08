#!/usr/bin/env python
"""Central Walk-Forward Backtest with Multi-Axis Condition Optimization
Uses Pattern A (strict leak-free) features.
Walk-forward: train 2010~(Y-1), test Y, for Y in 2020-2025.
Multi-axis: distance/num_horses/track_condition/course/class/season
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))

from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features, COURSE_MAP, N_TOP_SIRE,
)
from train_v92_leakfree import FEATURES_PATTERN_A, LEAK_FEATURES_A

COURSE_MAP_INV = {v: k for k, v in COURSE_MAP.items()}
CLASS_LABELS = {0: '新馬', 1: '未勝利', 2: '1勝', 3: '2勝', 4: '3勝', 5: 'OP',
                10: 'G3', 11: 'G2', 12: 'G1', 14: 'リステッド', 16: 'OP特別'}
TEST_YEARS = list(range(2020, 2026))


def classify_condition(row):
    nh = int(row.get('num_horses', row.get('num_horses_val', 14)))
    dist = int(row['distance'])
    cond = int(row.get('condition_enc', 0))
    heavy = cond >= 2
    if nh <= 7: return 'E'
    if dist <= 1400: return 'D'
    if 8 <= nh <= 14 and dist >= 1600 and not heavy: return 'A'
    if 8 <= nh <= 14 and dist >= 1600 and heavy: return 'B'
    if nh >= 15 and dist >= 1600 and not heavy: return 'C'
    return 'X'


def get_axes(row):
    """Extract multi-axis labels for condition exploration."""
    dist = int(row['distance'])
    nh = int(row.get('num_horses', row.get('num_horses_val', 14)))
    cond = int(row.get('condition_enc', 0))
    course = int(row['course_enc'])
    cls = int(row.get('class_code', 0))
    month = int(row['month'])
    surface = int(row['surface_enc'])

    dist_label = 'sprint' if dist <= 1200 else ('mile' if dist <= 1600 else ('middle' if dist <= 2200 else 'long'))
    nh_label = 'small' if nh <= 7 else ('medium' if nh <= 14 else 'large')
    cond_label = 'good' if cond <= 1 else 'heavy'
    course_name = COURSE_MAP_INV.get(course, f'c{course}')
    cls_label = CLASS_LABELS.get(cls, f'cls{cls}')
    season = 'spring' if month in [3,4,5] else ('summer' if month in [6,7,8] else ('autumn' if month in [9,10,11] else 'winter'))
    surf_label = 'turf' if surface == 0 else ('dirt' if surface == 1 else 'obstacle')

    return {
        'cond_key': classify_condition(row),
        'dist': dist_label, 'nh': nh_label, 'cond': cond_label,
        'course': course_name, 'class': cls_label, 'season': season,
        'surface': surf_label,
    }


def calc_trio_bets(ranking):
    if len(ranking) < 3: return []
    nums = ranking[:6] if len(ranking) >= 6 else ranking
    n1 = nums[0]
    second = nums[1:3]
    third = nums[1:min(6, len(nums))]
    return sorted(set(
        tuple(sorted({n1, s, t}))
        for s in second for t in third
        if len(set({n1, s, t})) == 3
    ))


def calc_umaren_bets(ranking):
    if len(ranking) < 3: return []
    return [sorted([ranking[0], ranking[1]]), sorted([ranking[0], ranking[2]])]


def calc_wide_bets(ranking):
    return calc_umaren_bets(ranking)


def estimate_payouts(actual_top3, race_df):
    """Estimate payouts from tansho odds."""
    odds = {int(r['umaban']): float(r['tansho_odds']) for _, r in race_df.iterrows()}
    o1 = odds.get(actual_top3.get(1), 10.0)
    o2 = odds.get(actual_top3.get(2), 10.0)
    o3 = odds.get(actual_top3.get(3), 10.0)
    trio_pay = max(100, int(o1 * o2 * o3 * 20))
    umaren_pay = max(100, int(o1 * o2 * 50))
    wide_pays = {}
    for a, b in [(actual_top3.get(1), actual_top3.get(2)),
                 (actual_top3.get(1), actual_top3.get(3)),
                 (actual_top3.get(2), actual_top3.get(3))]:
        if a and b:
            wide_pays[tuple(sorted([a, b]))] = max(100, int(odds.get(a, 10) * odds.get(b, 10) * 15))
    return trio_pay, umaren_pay, wide_pays


def check_bets(actual_top3, trio_bets, umaren_bets, wide_bets):
    top3_set = set(actual_top3.values())
    top2_set = {actual_top3.get(1), actual_top3.get(2)}
    trio_hit = any(set(c) == top3_set for c in trio_bets) if len(top3_set) == 3 else False
    umaren_hits = [b for b in umaren_bets if set(b) == top2_set]
    wide_hits = [b for b in wide_bets if set(b).issubset(top3_set)]
    return trio_hit, umaren_hits, wide_hits


def encode_sires_fold(df, train_mask, n_top=N_TOP_SIRE):
    """Encode sires using only training data."""
    train_df = df[train_mask]
    sire_counts = train_df['father'].value_counts()
    top_sires = sire_counts.head(n_top).index.tolist()
    sire_map = {s: i for i, s in enumerate(top_sires)}
    df['sire_enc'] = df['father'].map(sire_map).fillna(n_top).astype(int)
    bms_counts = train_df['bms'].value_counts()
    top_bms = bms_counts.head(n_top).index.tolist()
    bms_map = {s: i for i, s in enumerate(top_bms)}
    df['bms_enc'] = df['bms'].map(bms_map).fillna(n_top).astype(int)
    return df


def train_lgb_fold(X_train, y_train, X_valid, y_valid, features):
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
        'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
    }
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=features)
    dvalid = lgb.Dataset(X_valid, label=y_valid, feature_name=features, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=500,
                       valid_sets=[dvalid],
                       callbacks=[lgb.early_stopping(30), lgb.log_evaluation(200)])
    return model


def analyze_conditions(results, min_n=30):
    """Analyze results by multiple axes and bet types."""
    axes_to_analyze = [
        ('cond_key', 'Condition A-X'),
        ('dist', 'Distance'),
        ('nh', 'NumHorses'),
        ('cond', 'TrackCond'),
        ('course', 'Course'),
        ('class', 'Class'),
        ('season', 'Season'),
        ('surface', 'Surface'),
    ]
    cross_axes = [
        (('cond_key', 'surface'), 'Condition×Surface'),
        (('dist', 'cond'), 'Distance×TrackCond'),
        (('course', 'dist'), 'Course×Distance'),
        (('surface', 'dist'), 'Surface×Distance'),
    ]

    all_analysis = {}

    for axis_key, axis_name in axes_to_analyze:
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
        # All bet types use 700 yen per race investment
        # trio: 7 × 100, umaren: 2 × 350, wide: 2 × 350
        for bt, hit_key, pay_key in [
            ('trio', 'trio_hit', 'trio_return'),
            ('umaren', 'umaren_hit', 'umaren_return'),
            ('wide', 'wide_hit', 'wide_return'),
        ]:
            if bt == 'wide':
                hits = sum(1 for r in races if r.get('wide_hits', 0) > 0)
            elif bt == 'umaren':
                hits = sum(1 for r in races if r.get(hit_key, False))
            else:
                hits = sum(1 for r in races if r.get(hit_key, False))
            investment = n * 700  # Fixed 700 yen per race
            payout = sum(r.get(pay_key, 0) for r in races)
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
    print("=" * 70)
    print("  CENTRAL WALK-FORWARD LEAK-FREE BACKTEST")
    print(f"  Features: Pattern A ({len(FEATURES_PATTERN_A)} features)")
    print(f"  Removed: {sorted(LEAK_FEATURES_A)}")
    print(f"  Test years: {TEST_YEARS}")
    print("=" * 70)

    # Load and prepare data
    df = load_data()
    lap_df = load_lap_data()
    if lap_df is not None:
        df = df.merge(lap_df, on='race_id_str', how='left')

    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)

    tt_data = load_training_times()
    df = merge_training_features(df, tt_data)
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)
    df = compute_lag_features(df)
    df = build_features(df)

    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()

    # Ensure all features are numeric
    features = list(FEATURES_PATTERN_A)
    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"\nData ready: {len(df)} rows, {df['race_id_str'].nunique()} races")

    # Walk-forward
    all_results = []
    fold_aucs = {}

    for test_year in TEST_YEARS:
        train_mask = (df['year_full'] >= 2010) & (df['year_full'] < test_year)
        test_mask = df['year_full'] == test_year

        n_train = train_mask.sum()
        n_test = test_mask.sum()
        if n_test < 100:
            print(f"\n  Year {test_year}: Skip (n_test={n_test})")
            continue

        print(f"\n--- Fold: Train 2010-{test_year-1} ({n_train:,}) → Test {test_year} ({n_test:,}) ---")

        # Re-encode sires for this fold
        df_fold = df.copy()
        df_fold = encode_sires_fold(df_fold, train_mask)
        for f in features:
            if f not in df_fold.columns:
                df_fold[f] = 0
            df_fold[f] = pd.to_numeric(df_fold[f], errors='coerce').fillna(0)

        # Train/valid split within training data
        train_df = df_fold[train_mask]
        y_train = train_df['target'].values
        dates = train_df['date_num']
        valid_cutoff = dates.quantile(0.85)
        tr_idx = dates < valid_cutoff
        va_idx = dates >= valid_cutoff

        X_tr = train_df.loc[tr_idx, features].values
        y_tr = y_train[tr_idx.values]
        X_va = train_df.loc[va_idx, features].values
        y_va = y_train[va_idx.values]

        t0 = time.time()
        model = train_lgb_fold(X_tr, y_tr, X_va, y_va, features)
        elapsed = time.time() - t0
        va_pred = model.predict(X_va)
        va_auc = roc_auc_score(y_va, va_pred)
        print(f"  Valid AUC: {va_auc:.4f} ({elapsed:.0f}s)")

        # Predict on test
        test_df = df_fold[test_mask].copy()
        X_test = test_df[features].values
        test_df['pred'] = model.predict(X_test)
        y_test = test_df['target'].values
        test_auc = roc_auc_score(y_test, test_df['pred'])
        fold_aucs[test_year] = test_auc
        print(f"  Test AUC: {test_auc:.4f}")

        # Evaluate each race
        test_races = test_df['race_id_str'].unique()
        year_hits = 0
        year_total = 0

        for rid in test_races:
            race_df = test_df[test_df['race_id_str'] == rid].copy()
            if len(race_df) < 5:
                continue

            row0 = race_df.iloc[0]
            axes = get_axes(row0)

            # Actual top 3
            race_sorted = race_df.sort_values('finish')
            actual_top3 = {}
            for _, r in race_sorted.head(3).iterrows():
                actual_top3[int(r['finish'])] = int(r['umaban'])
            if len(actual_top3) < 3:
                continue

            # AI ranking
            race_df = race_df.sort_values('pred', ascending=False)
            ranking = race_df['umaban'].astype(int).tolist()

            # Bets
            trio_bets = calc_trio_bets(ranking)
            umaren_bets = calc_umaren_bets(ranking)
            wide_bets = calc_wide_bets(ranking)
            trio_hit, umaren_hits, wide_hits = check_bets(actual_top3, trio_bets, umaren_bets, wide_bets)

            # Payouts
            trio_pay, umaren_pay, wide_pays = estimate_payouts(actual_top3, race_df)

            trio_return = trio_pay if trio_hit else 0
            umaren_return = umaren_pay * 3.5 * len(umaren_hits) if umaren_hits else 0
            wide_return = sum(wide_pays.get(tuple(sorted(w)), 150) * 3.5 for w in wide_hits)

            year_total += 1
            if trio_hit: year_hits += 1

            all_results.append({
                'race_id': rid, 'year': test_year,
                'axes': axes,
                'trio_hit': trio_hit, 'trio_return': trio_return,
                'umaren_hit': len(umaren_hits) > 0, 'umaren_return': umaren_return,
                'wide_hits': len(wide_hits), 'wide_return': wide_return,
            })

        print(f"  Races: {year_total}, Trio hits: {year_hits} ({year_hits/max(1,year_total)*100:.1f}%)")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD RESULTS ({len(all_results)} races)")
    print(f"{'=' * 70}")
    print(f"  Year AUCs: {', '.join(f'{y}={a:.4f}' for y, a in fold_aucs.items())}")
    avg_auc = np.mean(list(fold_aucs.values()))
    print(f"  Average AUC: {avg_auc:.4f}")

    # Multi-axis analysis
    analysis = analyze_conditions(all_results, min_n=30)

    # Print results
    for axis_name, groups in analysis.items():
        print(f"\n  --- {axis_name} ---")
        print(f"  {'Label':<25} {'N':>5} {'Best':>6} {'HitR':>6} {'ROI':>7} {'Stars'}")
        for label, info in sorted(groups.items(), key=lambda x: x[1]['best_roi'], reverse=True):
            best = info['bets'][info['best_bet']]
            stars_str = '*' * info['stars'] if info['stars'] > 0 else 'X'
            print(f"  {label:<25} {info['n']:>5} {info['best_bet']:>6} {best['hit_rate']:>5.1f}% {best['roi']:>6.1f}% {stars_str}")

    # Save JSON
    output = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'v9.2a_leakfree (Pattern A)',
        'features_removed': sorted(LEAK_FEATURES_A),
        'n_features': len(FEATURES_PATTERN_A),
        'test_years': TEST_YEARS,
        'total_races': len(all_results),
        'fold_aucs': fold_aucs,
        'avg_auc': round(avg_auc, 4),
        'conditions': {},
    }

    # Extract recommended conditions
    cond_analysis = analysis.get('Condition A-X', {})
    for label, info in cond_analysis.items():
        output['conditions'][label] = {
            'n': info['n'],
            'best_bet': info['best_bet'],
            'best_roi': info['best_roi'],
            'stars': info['stars'],
            'recommended': info['recommended'],
            'bets': info['bets'],
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
                'trio_hit_rate': info['bets']['trio']['hit_rate'],
                'trio_roi': info['bets']['trio']['roi'],
            } for label, info in groups.items()
        }

    out_path = os.path.join(BASE_DIR, 'data', 'optimal_betting_jra_leakfree.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")

    print("\n  Central backtest complete!")
    return output


if __name__ == '__main__':
    main()
