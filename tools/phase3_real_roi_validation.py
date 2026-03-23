#!/usr/bin/env python
"""Phase 3: Real Payout ROI Validation

Uses ONLY jra_payouts.csv actual payouts. No estimated formulas.
Existing model, existing logic, verification only.
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import warnings
import traceback
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, BASE_DIR)

from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features, COURSE_MAP, N_TOP_SIRE,
)
from train_v92_leakfree import FEATURES_PATTERN_A, LEAK_FEATURES_A

from backtest_central_leakfree import (
    classify_condition, get_axes, calc_trio_bets, calc_umaren_bets,
    calc_wide_bets, encode_sires_fold, train_lgb_fold,
)
from calc_actual_roi import (
    load_payouts, parse_trio_nums, parse_umaren_nums, parse_wide_data,
    get_actual_payouts, calc_actual_returns,
)

ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts', 'phase2')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

TEST_YEARS = [2023, 2024, 2025]

report = {
    'timestamp': None,
    'payout_source': 'jra_payouts.csv (JRA official)',
    'test_years': TEST_YEARS,
    'task1_real_roi': {},
    'task2_bet_count_analysis': {},
    'task3_score_filter': {},
    'task4_race_filter': {},
    'task5_bootstrap_ci': {},
    'task6_drawdown': {},
    'final_diagnosis': {},
}


def save_report():
    report['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    path = os.path.join(DATA_DIR, 'phase3_real_roi_validation.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Report saved: {path}")


def prepare_data():
    print("=" * 70)
    print("  DATA PREPARATION")
    print("=" * 70)

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

    features = list(FEATURES_PATTERN_A)
    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"  Data ready: {len(df)} rows")
    return df, features


def run_wf_and_collect(df, features, payout_lookup, years):
    """Walk-forward backtest with actual payouts. Returns per-race results."""
    all_results = []

    for test_year in years:
        train_mask = (df['year_full'] >= 2010) & (df['year_full'] < test_year)
        test_mask = df['year_full'] == test_year
        if test_mask.sum() < 100:
            continue

        df_fold = df.copy()
        df_fold = encode_sires_fold(df_fold, train_mask)
        for f in features:
            if f not in df_fold.columns:
                df_fold[f] = 0
            df_fold[f] = pd.to_numeric(df_fold[f], errors='coerce').fillna(0)

        train_df = df_fold[train_mask]
        y_train = train_df['target'].values
        dates = train_df['date_num']
        vc = dates.quantile(0.85)
        tr_idx = dates < vc
        va_idx = dates >= vc

        model = train_lgb_fold(
            train_df.loc[tr_idx, features].values, y_train[tr_idx.values],
            train_df.loc[va_idx, features].values, y_train[va_idx.values],
            features)

        test_df = df_fold[test_mask].copy()
        test_df['pred'] = model.predict(test_df[features].values)
        auc = roc_auc_score(test_df['target'].values, test_df['pred'])
        print(f"  WF {test_year}: AUC {auc:.4f}")

        for rid in test_df['race_id_str'].unique():
            race_df = test_df[test_df['race_id_str'] == rid].copy()
            if len(race_df) < 5:
                continue

            row0 = race_df.iloc[0]
            cond_key = classify_condition(row0)

            race_sorted = race_df.sort_values('finish')
            actual_top3 = {}
            for _, r in race_sorted.head(3).iterrows():
                actual_top3[int(r['finish'])] = int(r['umaban'])
            if len(actual_top3) < 3:
                continue

            race_df_ranked = race_df.sort_values('pred', ascending=False)
            ranking = race_df_ranked['umaban'].astype(int).tolist()
            scores = race_df_ranked['pred'].tolist()

            # Standard bets
            trio_bets = calc_trio_bets(ranking)
            umaren_bets = calc_umaren_bets(ranking)
            wide_bets = calc_wide_bets(ranking)

            # Actual payouts
            payout_info = get_actual_payouts(payout_lookup, rid)
            if payout_info is None:
                continue  # Skip races without payout data

            (trio_hit, trio_return,
             umaren_hit, umaren_return,
             wide_hits, wide_return) = calc_actual_returns(
                payout_info, trio_bets, umaren_bets, wide_bets)

            # Num horses, distance, condition for filters
            nh = int(row0.get('num_horses', row0.get('num_horses_val', 14)))
            dist = int(row0['distance'])
            cond_enc = int(row0.get('condition_enc', 0))
            surface = int(row0.get('surface_enc', 0))
            cls = int(row0.get('class_code', 0))

            all_results.append({
                'race_id': rid,
                'year': test_year,
                'cond_key': cond_key,
                'date_num': int(row0['date_num']),
                'num_horses': nh,
                'distance': dist,
                'condition_enc': cond_enc,
                'surface_enc': surface,
                'class_code': cls,
                'ranking': ranking,
                'scores': scores,
                'top1_score': scores[0] if scores else 0,
                'top3_avg_score': float(np.mean(scores[:3])) if len(scores) >= 3 else 0,
                # Trio (7点 × 100円 = 700円)
                'trio_hit': trio_hit,
                'trio_return': trio_return,  # per 100 yen
                'trio_investment': 700,
                # Umaren (2点 × 350円 = 700円)
                'umaren_hit': umaren_hit,
                'umaren_return': umaren_return * 3.5 if umaren_hit else 0,
                'umaren_investment': 700,
                # Wide (2点 × 350円 = 700円)
                'wide_hit': wide_hits > 0,
                'wide_return': wide_return * 3.5 if wide_hits > 0 else 0,
                'wide_investment': 700,
            })

    return all_results


def calc_roi(results, bet_type='trio'):
    """Calculate ROI from results list."""
    if not results:
        return {'roi': 0, 'bets': 0, 'hit_rate': 0, 'total_return': 0, 'total_investment': 0, 'hits': 0}
    n = len(results)
    hit_key = f'{bet_type}_hit'
    return_key = f'{bet_type}_return'
    inv_key = f'{bet_type}_investment'
    hits = sum(1 for r in results if r.get(hit_key, False))
    total_return = sum(r.get(return_key, 0) for r in results)
    total_inv = sum(r.get(inv_key, 700) for r in results)
    roi = total_return / total_inv * 100 if total_inv > 0 else 0
    return {
        'roi': round(roi, 1),
        'bets': n,
        'hit_rate': round(hits / n * 100, 1) if n > 0 else 0,
        'total_return': int(total_return),
        'total_investment': int(total_inv),
        'hits': hits,
    }


# ============================================================
# TASK 1: Real ROI by year and condition
# ============================================================
def task1_real_roi(all_results):
    print("\n" + "=" * 70)
    print("  TASK 1: REAL PAYOUT ROI (jra_payouts.csv)")
    print("=" * 70)
    try:
        # Overall
        total = calc_roi(all_results, 'trio')
        total_uma = calc_roi(all_results, 'umaren')
        total_wide = calc_roi(all_results, 'wide')
        print(f"  TOTAL: Trio ROI={total['roi']:.1f}%, Hit={total['hit_rate']:.1f}%, N={total['bets']}")
        print(f"         Umaren ROI={total_uma['roi']:.1f}%, Wide ROI={total_wide['roi']:.1f}%")

        report['task1_real_roi']['total'] = {
            'trio': total, 'umaren': total_uma, 'wide': total_wide
        }

        # By year
        for yr in TEST_YEARS:
            yr_results = [r for r in all_results if r['year'] == yr]
            yr_roi = calc_roi(yr_results, 'trio')
            report['task1_real_roi'][str(yr)] = {
                'trio': yr_roi,
                'umaren': calc_roi(yr_results, 'umaren'),
                'wide': calc_roi(yr_results, 'wide'),
            }
            print(f"  {yr}: Trio ROI={yr_roi['roi']:.1f}%, Hit={yr_roi['hit_rate']:.1f}%, N={yr_roi['bets']}")

        # By condition
        for ck in ['A', 'B', 'C', 'D', 'E', 'X']:
            ck_results = [r for r in all_results if r['cond_key'] == ck]
            if not ck_results:
                continue
            ck_roi = calc_roi(ck_results, 'trio')
            ck_uma = calc_roi(ck_results, 'umaren')
            report['task1_real_roi'][f'cond_{ck}'] = {
                'trio': ck_roi, 'umaren': ck_uma,
            }
            print(f"  Cond {ck}: Trio ROI={ck_roi['roi']:.1f}%, Uma ROI={ck_uma['roi']:.1f}%, N={ck_roi['bets']}")

        save_report()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['task1_real_roi']['error'] = str(e)
        save_report()


# ============================================================
# TASK 2: Bet Count vs ROI
# ============================================================
def task2_bet_count(df, features, payout_lookup):
    print("\n" + "=" * 70)
    print("  TASK 2: BET COUNT vs ROI ANALYSIS")
    print("=" * 70)
    try:
        # We need to vary the trio bet structure
        # Current: TOP1 axis - TOP2,3 - TOP2~6 = 7 points
        # Top1 axis 2-flow: TOP1 - TOP2 - TOP2~4 = fewer points
        # Top2 axis: TOP1,TOP2 axis - TOP3~6 = more points
        # Top4: expand to TOP7

        strategies = {}

        for test_year in TEST_YEARS:
            train_mask = (df['year_full'] >= 2010) & (df['year_full'] < test_year)
            test_mask = df['year_full'] == test_year
            if test_mask.sum() < 100:
                continue

            df_fold = df.copy()
            df_fold = encode_sires_fold(df_fold, train_mask)
            for f in features:
                if f not in df_fold.columns:
                    df_fold[f] = 0
                df_fold[f] = pd.to_numeric(df_fold[f], errors='coerce').fillna(0)

            train_df = df_fold[train_mask]
            y_train = train_df['target'].values
            dates = train_df['date_num']
            vc = dates.quantile(0.85)
            model = train_lgb_fold(
                train_df.loc[dates < vc, features].values, y_train[(dates < vc).values],
                train_df.loc[dates >= vc, features].values, y_train[(dates >= vc).values],
                features)

            test_df = df_fold[test_mask].copy()
            test_df['pred'] = model.predict(test_df[features].values)

            for rid in test_df['race_id_str'].unique():
                race_df = test_df[test_df['race_id_str'] == rid].copy()
                if len(race_df) < 5:
                    continue

                race_sorted = race_df.sort_values('finish')
                actual_top3 = {}
                for _, r in race_sorted.head(3).iterrows():
                    actual_top3[int(r['finish'])] = int(r['umaban'])
                if len(actual_top3) < 3:
                    continue

                payout_info = get_actual_payouts(payout_lookup, rid)
                if payout_info is None:
                    continue

                actual_trio = parse_trio_nums(payout_info.get('trio_nums', ''))
                trio_payout = int(payout_info.get('trio_payout', 0))

                race_df = race_df.sort_values('pred', ascending=False)
                ranking = race_df['umaban'].astype(int).tolist()

                # Strategy 1: TOP1 axis, 2nd=TOP2, 3rd=TOP2~4 (narrow)
                def make_bets_narrow(r):
                    if len(r) < 3: return []
                    n1 = r[0]
                    second = r[1:2]
                    third = r[1:min(4, len(r))]
                    return sorted(set(
                        tuple(sorted({n1, s, t}))
                        for s in second for t in third
                        if len(set({n1, s, t})) == 3
                    ))

                # Strategy 2: Current (TOP1 axis, 2nd=TOP2,3, 3rd=TOP2~6) = 7 points
                def make_bets_current(r):
                    return calc_trio_bets(r)

                # Strategy 3: TOP1,TOP2 axis (wider)
                def make_bets_wide(r):
                    if len(r) < 4: return []
                    n1, n2 = r[0], r[1]
                    others = r[2:min(7, len(r))]
                    bets = set()
                    for t in others:
                        bets.add(tuple(sorted({n1, n2, t})))
                    for s in r[1:3]:
                        for t in r[1:min(7, len(r))]:
                            combo = tuple(sorted({n1, s, t}))
                            if len(combo) == 3:
                                bets.add(combo)
                    return sorted(bets)

                # Strategy 4: TOP1 axis, 2nd=TOP2~4, 3rd=TOP2~7 (very wide)
                def make_bets_vwide(r):
                    if len(r) < 4: return []
                    n1 = r[0]
                    second = r[1:min(4, len(r))]
                    third = r[1:min(7, len(r))]
                    return sorted(set(
                        tuple(sorted({n1, s, t}))
                        for s in second for t in third
                        if len(set({n1, s, t})) == 3
                    ))

                for name, bet_fn in [
                    ('narrow_3pt', make_bets_narrow),
                    ('current_7pt', make_bets_current),
                    ('wide_axis2', make_bets_wide),
                    ('vwide_15pt', make_bets_vwide),
                ]:
                    if name not in strategies:
                        strategies[name] = []

                    bets = bet_fn(ranking)
                    n_bets = len(bets)
                    hit = False
                    if actual_trio and n_bets > 0:
                        hit = any(frozenset(b) == actual_trio for b in bets)

                    strategies[name].append({
                        'hit': hit,
                        'return': trio_payout if hit else 0,
                        'investment': n_bets * 100,
                        'n_bets': n_bets,
                    })

        # Summarize
        for name, results in strategies.items():
            n = len(results)
            hits = sum(1 for r in results if r['hit'])
            total_return = sum(r['return'] for r in results)
            total_inv = sum(r['investment'] for r in results)
            avg_bets = np.mean([r['n_bets'] for r in results])
            roi = total_return / total_inv * 100 if total_inv > 0 else 0

            report['task2_bet_count_analysis'][name] = {
                'roi': round(roi, 1),
                'hit_rate': round(hits / n * 100, 1) if n > 0 else 0,
                'hits': hits,
                'races': n,
                'avg_bets_per_race': round(float(avg_bets), 1),
                'total_investment': int(total_inv),
                'total_return': int(total_return),
            }
            print(f"  {name:15s}: ROI={roi:>7.1f}%, Hit={hits/n*100:>5.1f}%, "
                  f"avg_bets={avg_bets:.1f}, N={n}")

        save_report()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['task2_bet_count_analysis']['error'] = str(e)
        save_report()


# ============================================================
# TASK 3: Score/Probability Filter
# ============================================================
def task3_score_filter(all_results):
    print("\n" + "=" * 70)
    print("  TASK 3: SCORE THRESHOLD FILTER")
    print("=" * 70)
    try:
        # Filter by top1 score or top3 average score
        thresholds = {
            'all_races': lambda r: True,
            'top1_score_gt_0.25': lambda r: r['top1_score'] > 0.25,
            'top1_score_gt_0.30': lambda r: r['top1_score'] > 0.30,
            'top1_score_gt_0.35': lambda r: r['top1_score'] > 0.35,
            'top3_avg_gt_0.20': lambda r: r['top3_avg_score'] > 0.20,
            'top3_avg_gt_0.25': lambda r: r['top3_avg_score'] > 0.25,
            'score_gap_top1_top4_gt_0.05': lambda r: (
                r['scores'][0] - r['scores'][3] > 0.05 if len(r['scores']) > 3 else False
            ),
        }

        for name, filter_fn in thresholds.items():
            filtered = [r for r in all_results if filter_fn(r)]
            roi = calc_roi(filtered, 'trio')
            report['task3_score_filter'][name] = roi
            pct = len(filtered) / len(all_results) * 100 if all_results else 0
            print(f"  {name:35s}: ROI={roi['roi']:>7.1f}%, Hit={roi['hit_rate']:>5.1f}%, "
                  f"N={roi['bets']} ({pct:.0f}% of races)")

        save_report()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['task3_score_filter']['error'] = str(e)
        save_report()


# ============================================================
# TASK 4: Race Filter Analysis
# ============================================================
def task4_race_filter(all_results):
    print("\n" + "=" * 70)
    print("  TASK 4: RACE FILTER ANALYSIS")
    print("=" * 70)
    try:
        filters = {
            # Num horses
            'nh_5_7': lambda r: r['num_horses'] <= 7,
            'nh_8_14': lambda r: 8 <= r['num_horses'] <= 14,
            'nh_15+': lambda r: r['num_horses'] >= 15,
            # Distance
            'dist_sprint_1400': lambda r: r['distance'] <= 1400,
            'dist_mile_1600_2000': lambda r: 1600 <= r['distance'] <= 2000,
            'dist_long_2200+': lambda r: r['distance'] >= 2200,
            # Condition
            'cond_good': lambda r: r['condition_enc'] <= 1,
            'cond_heavy': lambda r: r['condition_enc'] >= 2,
            # Surface
            'turf': lambda r: r['surface_enc'] == 0,
            'dirt': lambda r: r['surface_enc'] == 1,
            # Class
            'class_low': lambda r: r['class_code'] <= 2,
            'class_mid': lambda r: 3 <= r['class_code'] <= 5,
            'class_high': lambda r: r['class_code'] >= 10,
        }

        for name, filter_fn in filters.items():
            filtered = [r for r in all_results if filter_fn(r)]
            if not filtered:
                continue
            roi = calc_roi(filtered, 'trio')
            roi_uma = calc_roi(filtered, 'umaren')
            report['task4_race_filter'][name] = {
                'trio': roi, 'umaren': roi_uma,
            }
            print(f"  {name:25s}: Trio ROI={roi['roi']:>7.1f}%, Uma ROI={roi_uma['roi']:>7.1f}%, "
                  f"Hit={roi['hit_rate']:>5.1f}%, N={roi['bets']}")

        save_report()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['task4_race_filter']['error'] = str(e)
        save_report()


# ============================================================
# TASK 5: Bootstrap CI
# ============================================================
def task5_bootstrap(all_results):
    print("\n" + "=" * 70)
    print("  TASK 5: BOOTSTRAP CONFIDENCE INTERVAL")
    print("=" * 70)
    try:
        n = len(all_results)
        n_bootstrap = 1000
        bootstrap_rois = []

        for i in range(n_bootstrap):
            rng = np.random.RandomState(i)
            indices = rng.choice(n, size=n, replace=True)
            sample = [all_results[j] for j in indices]
            roi = calc_roi(sample, 'trio')
            bootstrap_rois.append(roi['roi'])

        bootstrap_rois = np.array(bootstrap_rois)
        mean_roi = float(np.mean(bootstrap_rois))
        ci_lower = float(np.percentile(bootstrap_rois, 2.5))
        ci_upper = float(np.percentile(bootstrap_rois, 97.5))
        prob_above_100 = float(np.mean(bootstrap_rois > 100))

        report['task5_bootstrap_ci'] = {
            'n_races': n,
            'n_bootstrap': n_bootstrap,
            'mean_roi': round(mean_roi, 1),
            'ci_95_lower': round(ci_lower, 1),
            'ci_95_upper': round(ci_upper, 1),
            'prob_roi_above_100': round(prob_above_100 * 100, 1),
            'std_roi': round(float(np.std(bootstrap_rois)), 1),
        }

        print(f"  Bootstrap (n={n_bootstrap}):")
        print(f"    Mean ROI: {mean_roi:.1f}%")
        print(f"    95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]")
        print(f"    P(ROI > 100%): {prob_above_100*100:.1f}%")

        save_report()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['task5_bootstrap_ci']['error'] = str(e)
        save_report()


# ============================================================
# TASK 6: Drawdown Analysis
# ============================================================
def task6_drawdown(all_results):
    print("\n" + "=" * 70)
    print("  TASK 6: DRAWDOWN & RISK ANALYSIS")
    print("=" * 70)
    try:
        sorted_results = sorted(all_results, key=lambda r: (r['date_num'], r['race_id']))

        balance = 0
        peak = 0
        max_dd = 0
        max_dd_pct = 0
        consecutive_losses = 0
        max_consecutive_losses = 0
        dd_start = None
        max_recovery_races = 0
        current_recovery = 0
        in_drawdown = False

        balance_series = []
        roi_series = []
        cum_invest = 0
        cum_return = 0

        for r in sorted_results:
            cum_invest += r['trio_investment']
            pnl = r['trio_return'] - r['trio_investment']
            cum_return += r['trio_return']
            balance += pnl

            roi = cum_return / cum_invest * 100 if cum_invest > 0 else 0
            roi_series.append(round(roi, 1))
            balance_series.append(int(balance))

            if balance > peak:
                peak = balance
                if in_drawdown:
                    in_drawdown = False
                    max_recovery_races = max(max_recovery_races, current_recovery)

            dd = peak - balance
            if dd > max_dd:
                max_dd = dd
                if peak > 0:
                    max_dd_pct = dd / peak * 100

            if pnl < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                if not in_drawdown:
                    in_drawdown = True
                    current_recovery = 0
                current_recovery += 1
            else:
                consecutive_losses = 0
                if in_drawdown:
                    current_recovery += 1

        # Sample for JSON
        step = max(1, len(roi_series) // 200)

        report['task6_drawdown'] = {
            'total_races': len(sorted_results),
            'final_balance': int(balance),
            'final_roi': roi_series[-1] if roi_series else 0,
            'max_drawdown_yen': int(max_dd),
            'max_drawdown_pct': round(max_dd_pct, 1),
            'max_consecutive_losses': max_consecutive_losses,
            'max_recovery_races': max_recovery_races,
            'peak_balance': int(peak),
            'min_balance': min(balance_series) if balance_series else 0,
            'balance_series': balance_series[::step],
            'roi_series': roi_series[::step],
        }

        print(f"  Total races: {len(sorted_results)}")
        print(f"  Final balance: {balance:,} yen")
        print(f"  Final ROI: {roi_series[-1]:.1f}%")
        print(f"  Max drawdown: {max_dd:,} yen ({max_dd_pct:.1f}%)")
        print(f"  Max consecutive losses: {max_consecutive_losses}")
        print(f"  Max recovery: {max_recovery_races} races")

        # Plot
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            ax1.plot(balance_series, linewidth=0.5, color='green')
            ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax1.set_title('Cumulative P&L (Real Payouts, 2023-2025)')
            ax1.set_ylabel('Balance (yen)')
            ax1.grid(True, alpha=0.3)

            ax2.plot(roi_series, linewidth=0.5)
            ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Break-even')
            ax2.set_title('Cumulative ROI (Real Payouts)')
            ax2.set_ylabel('ROI (%)')
            ax2.set_xlabel('Race #')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(os.path.join(ARTIFACTS_DIR, 'real_roi_curve.png'), dpi=150)
            plt.close(fig)
            print(f"  Saved: artifacts/phase2/real_roi_curve.png")
        except Exception as e:
            print(f"  Warning: Plot failed: {e}")

        save_report()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['task6_drawdown']['error'] = str(e)
        save_report()


# ============================================================
# MAIN
# ============================================================
def main():
    start_time = time.time()
    print("=" * 70)
    print("  PHASE 3: REAL PAYOUT ROI VALIDATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Source: jra_payouts.csv (JRA official payouts)")
    print("  NO estimated formulas used")
    print("=" * 70)

    # Check jra_payouts.csv exists
    payout_path = os.path.join(DATA_DIR, 'jra_payouts.csv')
    if not os.path.exists(payout_path):
        print(f"\n  ERROR: {payout_path} not found!")
        print("  Run: python scrape_jra_payouts.py --year 2023")
        print("       python scrape_jra_payouts.py --year 2024")
        print("       python scrape_jra_payouts.py --year 2025")
        report['error'] = 'jra_payouts.csv not found'
        save_report()
        return

    df, features = prepare_data()

    print("\nLoading JRA payout data...")
    payout_lookup = load_payouts()

    print("\nRunning walk-forward backtest with REAL payouts...")
    all_results = run_wf_and_collect(df, features, payout_lookup, TEST_YEARS)
    print(f"  Collected {len(all_results)} races with payout data")

    if not all_results:
        print("  ERROR: No results collected!")
        report['error'] = 'No results collected'
        save_report()
        return

    # Run all tasks
    task1_real_roi(all_results)
    task2_bet_count(df, features, payout_lookup)
    task3_score_filter(all_results)
    task4_race_filter(all_results)
    task5_bootstrap(all_results)
    task6_drawdown(all_results)

    # Final diagnosis
    t1 = report.get('task1_real_roi', {}).get('total', {}).get('trio', {})
    t5 = report.get('task5_bootstrap_ci', {})

    real_roi = t1.get('roi', 0)
    ci_lower = t5.get('ci_95_lower', 0)
    prob_above_100 = t5.get('prob_roi_above_100', 0)

    report['final_diagnosis'] = {
        'real_trio_roi': real_roi,
        'bootstrap_95ci': f"[{ci_lower}, {t5.get('ci_95_upper', 0)}]",
        'prob_profitable': prob_above_100,
        'verdict': 'PROFITABLE' if ci_lower > 100 else ('MARGINAL' if real_roi > 100 else 'NOT PROFITABLE'),
    }

    elapsed = time.time() - start_time
    report['elapsed_seconds'] = round(elapsed, 1)
    save_report()

    print(f"\n{'=' * 70}")
    print(f"  PHASE 3 COMPLETE ({elapsed/60:.1f} minutes)")
    print(f"  Real Trio ROI: {real_roi:.1f}%")
    print(f"  95% CI: [{ci_lower:.1f}%, {t5.get('ci_95_upper', 0):.1f}%]")
    print(f"  P(profitable): {prob_above_100:.1f}%")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
