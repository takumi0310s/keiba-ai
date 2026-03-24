#!/usr/bin/env python
"""Phase 4: Live Validation - Out-of-sample 2026 + production verification

Uses ONLY real payouts (jra_payouts.csv + netkeiba scraping for 2026).
No model changes. Existing v10 model and betting logic.
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import warnings
import traceback
import sqlite3
import requests
import re
from bs4 import BeautifulSoup
from collections import defaultdict
from datetime import datetime, timedelta

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

ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts', 'phase4')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

report = {}


def save_report():
    report['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    path = os.path.join(DATA_DIR, 'phase4_live_validation.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Report saved: {path}")


def calc_roi(results, bet_type='trio'):
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
        'roi': round(roi, 1), 'bets': n,
        'hit_rate': round(hits / n * 100, 1) if n > 0 else 0,
        'total_return': int(total_return), 'total_investment': int(total_inv), 'hits': hits,
    }


def bootstrap_ci(results, bet_type='trio', n_boot=1000):
    n = len(results)
    if n == 0:
        return {'mean': 0, 'ci_lower': 0, 'ci_upper': 0, 'prob_above_100': 0}
    rois = []
    for i in range(n_boot):
        rng = np.random.RandomState(i)
        sample = [results[j] for j in rng.choice(n, size=n, replace=True)]
        r = calc_roi(sample, bet_type)
        rois.append(r['roi'])
    rois = np.array(rois)
    return {
        'mean': round(float(np.mean(rois)), 1),
        'ci_lower': round(float(np.percentile(rois, 2.5)), 1),
        'ci_upper': round(float(np.percentile(rois, 97.5)), 1),
        'prob_above_100': round(float(np.mean(rois > 100)) * 100, 1),
    }


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


def run_wf_with_payouts(df, features, payout_lookup, years):
    """Walk-forward backtest with actual payouts."""
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
        model = train_lgb_fold(
            train_df.loc[dates < vc, features].values, y_train[(dates < vc).values],
            train_df.loc[dates >= vc, features].values, y_train[(dates >= vc).values],
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
            trio_bets = calc_trio_bets(ranking)
            umaren_bets = calc_umaren_bets(ranking)
            wide_bets = calc_wide_bets(ranking)
            payout_info = get_actual_payouts(payout_lookup, rid)
            if payout_info is None:
                continue
            (trio_hit, trio_return, umaren_hit, umaren_return,
             wide_hits, wide_return) = calc_actual_returns(
                payout_info, trio_bets, umaren_bets, wide_bets)
            nh = int(row0.get('num_horses', row0.get('num_horses_val', 14)))
            dist = int(row0['distance'])
            cond_enc = int(row0.get('condition_enc', 0))
            surface = int(row0.get('surface_enc', 0))
            cls_code = int(row0.get('class_code', 0))
            all_results.append({
                'race_id': rid, 'year': test_year, 'cond_key': cond_key,
                'date_num': int(row0['date_num']),
                'num_horses': nh, 'distance': dist, 'condition_enc': cond_enc,
                'surface_enc': surface, 'class_code': cls_code,
                'top1_score': scores[0] if scores else 0,
                'top3_avg_score': float(np.mean(scores[:3])) if len(scores) >= 3 else 0,
                'trio_hit': trio_hit, 'trio_return': trio_return, 'trio_investment': 700,
                'umaren_hit': umaren_hit,
                'umaren_return': umaren_return * 3.5 if umaren_hit else 0,
                'umaren_investment': 700,
                'wide_hit': wide_hits > 0,
                'wide_return': wide_return * 3.5 if wide_hits > 0 else 0,
                'wide_investment': 700,
            })
    return all_results


# ============================================================
# TASK 1: 2026 OOS via jra_payouts.csv scraping
# ============================================================
def task1_oos_2026(df, features, payout_lookup):
    """Out-of-sample test using 2026 Jan-Feb data from jra_payouts.csv."""
    print("\n" + "=" * 70)
    print("  TASK 1: 2026 OUT-OF-SAMPLE TEST (REAL PAYOUTS)")
    print("=" * 70)
    try:
        # First scrape 2026 payouts if not already present
        has_2026 = False
        payout_path = os.path.join(DATA_DIR, 'jra_payouts.csv')
        if os.path.exists(payout_path):
            pdf = pd.read_csv(payout_path, encoding='utf-8-sig', dtype=str)
            n_2026 = pdf['race_date'].str.startswith('2026').sum()
            if n_2026 > 0:
                has_2026 = True
                print(f"  Already have {n_2026} rows for 2026")

        if not has_2026:
            print("  Scraping 2026 payouts from JRA...")
            import subprocess
            result = subprocess.run(
                [sys.executable, os.path.join(BASE_DIR, 'scrape_jra_payouts.py'), '--year', '2026'],
                capture_output=True, text=True, timeout=7200)
            print(f"  Scraper exit code: {result.returncode}")
            if result.returncode != 0:
                print(f"  Scraper stderr: {result.stderr[-500:]}")

        # Reload payouts
        payout_lookup = load_payouts()

        # Check how many 2026 races we have in jra_races_full.csv
        # The model uses data up to 2025 for training. 2026 is truly OOS.
        # But jra_races_full.csv may not have 2026 data yet.
        # Check if 2026 data exists:
        test_mask_2026 = df['year_full'] == 2026
        n_2026_in_data = test_mask_2026.sum()

        if n_2026_in_data < 100:
            print(f"  Only {n_2026_in_data} rows for 2026 in jra_races_full.csv")
            print("  2026 data insufficient for full OOS test.")
            print("  Falling back to 2025 as quasi-OOS (trained up to 2024)")

            # Use 2025 data with model trained up to 2024
            results_2025 = run_wf_with_payouts(df, features, payout_lookup, [2025])

            if results_2025:
                # Filter to Jan-Feb for fair comparison
                all_oos = results_2025  # Use all 2025
                # Monthly breakdown
                monthly = {}
                for r in all_oos:
                    dn = str(r['date_num'])
                    if len(dn) >= 6:
                        month = dn[4:6]
                    else:
                        month = '??'
                    if month not in monthly:
                        monthly[month] = []
                    monthly[month].append(r)

                total = calc_roi(all_oos, 'trio')
                ci = bootstrap_ci(all_oos, 'trio')
                uma = calc_roi(all_oos, 'umaren')
                wide = calc_roi(all_oos, 'wide')

                monthly_out = {}
                for m, mrs in sorted(monthly.items()):
                    mr = calc_roi(mrs, 'trio')
                    monthly_out[m] = mr

                ci_lower = ci['ci_lower']
                if ci_lower > 100:
                    verdict = 'valid'
                elif total['roi'] > 100:
                    verdict = 'uncertain'
                else:
                    verdict = 'invalid'

                report['oos_test_2026'] = {
                    'note': '2026 data insufficient. Using 2025 as quasi-OOS (model trained up to 2024)',
                    'roi_trio': total['roi'],
                    'roi_umaren': uma['roi'],
                    'roi_wide': wide['roi'],
                    'ci_lower': ci['ci_lower'],
                    'ci_upper': ci['ci_upper'],
                    'prob_above_100': ci['prob_above_100'],
                    'hit_rate': total['hit_rate'],
                    'race_count': total['bets'],
                    'monthly_breakdown': monthly_out,
                    'verdict': verdict,
                }
                print(f"  Trio ROI: {total['roi']:.1f}% [{ci['ci_lower']:.1f}%, {ci['ci_upper']:.1f}%]")
                print(f"  Verdict: {verdict}")
            else:
                report['oos_test_2026'] = {'error': 'No results for 2025'}
        else:
            # Full 2026 OOS
            results_2026 = run_wf_with_payouts(df, features, payout_lookup, [2026])
            # Filter Jan-Feb
            jan_feb = [r for r in results_2026 if str(r['date_num'])[4:6] in ('01', '02')]
            all_oos = jan_feb if jan_feb else results_2026

            total = calc_roi(all_oos, 'trio')
            ci = bootstrap_ci(all_oos, 'trio')

            ci_lower = ci['ci_lower']
            verdict = 'valid' if ci_lower > 100 else ('uncertain' if total['roi'] > 100 else 'invalid')

            report['oos_test_2026'] = {
                'roi_trio': total['roi'],
                'roi_umaren': calc_roi(all_oos, 'umaren')['roi'],
                'roi_wide': calc_roi(all_oos, 'wide')['roi'],
                'ci_lower': ci['ci_lower'],
                'ci_upper': ci['ci_upper'],
                'prob_above_100': ci['prob_above_100'],
                'hit_rate': total['hit_rate'],
                'race_count': total['bets'],
                'verdict': verdict,
            }
            print(f"  Trio ROI: {total['roi']:.1f}% [{ci['ci_lower']:.1f}%, {ci['ci_upper']:.1f}%]")
            print(f"  Verdict: {verdict}")

        save_report()
        return all_oos if 'all_oos' in dir() else []
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['oos_test_2026'] = {'error': str(e)}
        save_report()
        return []


# ============================================================
# TASK 2: Live validation from DB
# ============================================================
def task2_live_validation():
    print("\n" + "=" * 70)
    print("  TASK 2: LIVE VALIDATION (keiba_race_results.db)")
    print("=" * 70)
    try:
        db_path = os.path.join(BASE_DIR, 'keiba_race_results.db')
        if not os.path.exists(db_path):
            report['live_validation'] = {'error': 'DB not found'}
            save_report()
            return

        conn = sqlite3.connect(db_path)
        results = conn.execute("""
            SELECT race_id, hit_trio, payout, bet_condition, bet_type, buy_recommended,
                   hit_wide, wide_payout, umaren_bets, trio_bets
            FROM race_results
        """).fetchall()
        conn.close()

        if not results:
            report['live_validation'] = {'error': 'No results in DB', 'race_count': 0}
            save_report()
            return

        n = len(results)
        recommended = [r for r in results if r[5] == 1]
        n_rec = len(recommended)
        trio_hits = sum(1 for r in recommended if r[1] == 1)
        trio_payout_total = sum(r[2] for r in recommended if r[1] == 1)
        investment = n_rec * 700

        actual_roi = trio_payout_total / investment * 100 if investment > 0 else 0
        hit_rate = trio_hits / n_rec * 100 if n_rec > 0 else 0

        report['live_validation'] = {
            'actual_roi': round(actual_roi, 1),
            'backtest_roi': 225.8,  # From Phase 3
            'prediction_match_rate': 'N/A (single-day data, insufficient for comparison)',
            'drift_analysis': 'Insufficient data for drift analysis (only 1 day recorded)',
            'race_count': n_rec,
            'trio_hits': trio_hits,
            'total_payout': int(trio_payout_total),
            'investment': investment,
            'note': f'Only {n_rec} races from 1 day (2026-03-14). Need more data for meaningful comparison.',
        }

        print(f"  Live races: {n_rec} (recommended)")
        print(f"  Trio hits: {trio_hits}")
        print(f"  Live ROI: {actual_roi:.1f}%")
        print(f"  Note: Only 1 day of data - insufficient for statistical comparison")
        save_report()

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['live_validation'] = {'error': str(e)}
        save_report()


# ============================================================
# TASK 3: ROI Structure Breakdown
# ============================================================
def task3_roi_breakdown(all_results):
    print("\n" + "=" * 70)
    print("  TASK 3: ROI STRUCTURE BREAKDOWN")
    print("=" * 70)
    try:
        hits = [r for r in all_results if r['trio_hit']]
        payouts = [r['trio_return'] for r in hits]

        n = len(all_results)
        n_hits = len(hits)
        total_return = sum(r['trio_return'] for r in all_results)
        total_inv = n * 700

        # Top-N exclusion
        payouts_sorted = sorted(payouts, reverse=True)
        exclude_results = {}
        for ex_n in [5, 10, 20]:
            if len(payouts_sorted) > ex_n:
                excluded = sum(payouts_sorted[:ex_n])
                remaining = total_return - excluded
                ex_roi = remaining / total_inv * 100
            else:
                ex_roi = 0
            exclude_results[f'exclude_top{ex_n}'] = round(ex_roi, 1)

        dep_level = 'low'
        if exclude_results.get('exclude_top5', 0) < 100:
            dep_level = 'high'
        elif exclude_results.get('exclude_top10', 0) < 100:
            dep_level = 'medium'

        report['roi_breakdown'] = {
            'avg_payout_on_hit': round(float(np.mean(payouts)), 0) if payouts else 0,
            'median_payout_on_hit': round(float(np.median(payouts)), 0) if payouts else 0,
            'std_payout': round(float(np.std(payouts)), 0) if payouts else 0,
            'max_payout': int(max(payouts)) if payouts else 0,
            'min_payout': int(min(payouts)) if payouts else 0,
            'hit_rate': round(n_hits / n * 100, 1) if n > 0 else 0,
            'investment_per_race': 700,
            'expected_profit_per_race': round((total_return - total_inv) / n, 0) if n > 0 else 0,
            'roi': round(total_return / total_inv * 100, 1) if total_inv > 0 else 0,
            'exclude_top5_roi': exclude_results.get('exclude_top5', 0),
            'exclude_top10_roi': exclude_results.get('exclude_top10', 0),
            'exclude_top20_roi': exclude_results.get('exclude_top20', 0),
            'high_payout_dependency': dep_level,
            'top10_payouts': [int(p) for p in payouts_sorted[:10]],
        }

        print(f"  Hits: {n_hits}/{n} ({n_hits/n*100:.1f}%)")
        print(f"  Avg payout: {np.mean(payouts):,.0f}, Median: {np.median(payouts):,.0f}")
        print(f"  Expected profit/race: {(total_return - total_inv) / n:,.0f} yen")
        print(f"  Exclude top5 ROI: {exclude_results.get('exclude_top5', 0):.1f}%")
        print(f"  Dependency: {dep_level}")

        # Histogram
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(payouts, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(x=np.mean(payouts), color='r', linestyle='--',
                       label=f'Mean: {np.mean(payouts):,.0f}')
            ax.axvline(x=np.median(payouts), color='g', linestyle='--',
                       label=f'Median: {np.median(payouts):,.0f}')
            ax.set_xlabel('Payout (yen per 100 yen)')
            ax.set_ylabel('Count')
            ax.set_title(f'Real Payout Distribution (N={len(payouts)} hits)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(ARTIFACTS_DIR, 'payout_histogram.png'), dpi=150)
            plt.close(fig)
        except Exception:
            pass

        save_report()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['roi_breakdown'] = {'error': str(e)}
        save_report()


# ============================================================
# TASK 4: Condition-dependent analysis
# ============================================================
def task4_condition_analysis(all_results):
    print("\n" + "=" * 70)
    print("  TASK 4: CONDITION DEPENDENCY ANALYSIS")
    print("=" * 70)
    try:
        analyses = {}

        filter_defs = {
            'by_field_size': {
                '7_or_less': lambda r: r['num_horses'] <= 7,
                '8_to_14': lambda r: 8 <= r['num_horses'] <= 14,
                '15_plus': lambda r: r['num_horses'] >= 15,
            },
            'by_track_condition': {
                'good': lambda r: r['condition_enc'] == 0,
                'slightly_heavy': lambda r: r['condition_enc'] == 1,
                'heavy': lambda r: r['condition_enc'] == 2,
                'bad': lambda r: r['condition_enc'] == 3,
            },
            'by_distance': {
                'sprint_1400': lambda r: r['distance'] <= 1400,
                'mile_1600_2000': lambda r: 1600 <= r['distance'] <= 2000,
                'long_2200': lambda r: r['distance'] >= 2200,
            },
            'by_surface': {
                'turf': lambda r: r['surface_enc'] == 0,
                'dirt': lambda r: r['surface_enc'] == 1,
            },
            'by_class': {
                'maiden': lambda r: r['class_code'] <= 1,
                'class_1_2': lambda r: 2 <= r['class_code'] <= 3,
                'class_3_OP': lambda r: 4 <= r['class_code'] <= 5,
                'graded': lambda r: r['class_code'] >= 10,
            },
        }

        for axis_name, filters in filter_defs.items():
            axis_results = {}
            for label, fn in filters.items():
                filtered = [r for r in all_results if fn(r)]
                if len(filtered) < 20:
                    continue
                roi = calc_roi(filtered, 'trio')
                ci = bootstrap_ci(filtered, 'trio')
                axis_results[label] = {
                    'roi': roi['roi'], 'hit_rate': roi['hit_rate'], 'bets': roi['bets'],
                    'ci_lower': ci['ci_lower'], 'ci_upper': ci['ci_upper'],
                    'hits': roi['hits'],
                }
                print(f"  {axis_name}/{label:20s}: ROI={roi['roi']:>7.1f}% "
                      f"[{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}] "
                      f"Hit={roi['hit_rate']:.1f}% N={roi['bets']}")
            analyses[axis_name] = axis_results

        report['condition_analysis'] = analyses
        save_report()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['condition_analysis'] = {'error': str(e)}
        save_report()


# ============================================================
# TASK 5: EV-based bet filtering
# ============================================================
def task5_bet_filter(all_results):
    print("\n" + "=" * 70)
    print("  TASK 5: BET FILTER (EV-BASED)")
    print("=" * 70)
    try:
        # EV proxy: top3_avg_score (model confidence)
        # Higher score = higher expected hit probability
        # We use score-based percentile filtering
        scores = [r['top3_avg_score'] for r in all_results]
        percentiles = {
            'top10': np.percentile(scores, 90),
            'top20': np.percentile(scores, 80),
            'top30': np.percentile(scores, 70),
            'top50': np.percentile(scores, 50),
        }

        filter_results = {}
        for name, threshold in [
            ('top10', percentiles['top10']),
            ('top20', percentiles['top20']),
            ('top30', percentiles['top30']),
            ('top50', percentiles['top50']),
            ('all', 0),
        ]:
            filtered = [r for r in all_results if r['top3_avg_score'] >= threshold]
            roi = calc_roi(filtered, 'trio')
            ci = bootstrap_ci(filtered, 'trio')
            profit = roi['total_return'] - roi['total_investment']
            per_race = profit / roi['bets'] if roi['bets'] > 0 else 0

            filter_results[name] = {
                'roi': roi['roi'],
                'ci': {'lower': ci['ci_lower'], 'upper': ci['ci_upper']},
                'bets': roi['bets'],
                'hits': roi['hits'],
                'hit_rate': roi['hit_rate'],
                'profit': int(profit),
                'profit_per_race': round(per_race, 0),
                'threshold': round(float(threshold), 4),
            }
            print(f"  {name:6s}: ROI={roi['roi']:>7.1f}% [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}] "
                  f"N={roi['bets']} Hit={roi['hit_rate']:.1f}% Profit/R={per_race:,.0f}")

        report['bet_filter'] = filter_results
        save_report()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['bet_filter'] = {'error': str(e)}
        save_report()


# ============================================================
# TASK 6: Risk Assessment
# ============================================================
def task6_risk(all_results):
    print("\n" + "=" * 70)
    print("  TASK 6: RISK ASSESSMENT")
    print("=" * 70)
    try:
        sorted_results = sorted(all_results, key=lambda r: (r['date_num'], r.get('race_id', '')))

        balance = 0
        peak = 0
        max_dd = 0
        consecutive_losses = 0
        max_consecutive_losses = 0
        max_recovery = 0
        in_dd = False
        dd_counter = 0

        balance_series = []
        daily_pnl = defaultdict(int)

        for r in sorted_results:
            pnl = r['trio_return'] - r['trio_investment']
            balance += pnl
            balance_series.append(balance)
            daily_pnl[r['date_num']] += pnl

            if balance > peak:
                peak = balance
                if in_dd:
                    max_recovery = max(max_recovery, dd_counter)
                    in_dd = False
            dd = peak - balance
            if dd > max_dd:
                max_dd = dd
            if pnl < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                if not in_dd:
                    in_dd = True
                    dd_counter = 0
                dd_counter += 1
            else:
                consecutive_losses = 0
                if in_dd:
                    dd_counter += 1

        # Bankruptcy simulation (30,000 yen initial, 700/race)
        n_sim = 5000
        bankrupt_count = 0
        for s in range(n_sim):
            rng = np.random.RandomState(s)
            capital = 30000
            for _ in range(1000):
                idx = rng.randint(0, len(sorted_results))
                r = sorted_results[idx]
                capital += r['trio_return'] - r['trio_investment']
                if capital < 0:
                    bankrupt_count += 1
                    break

        bankruptcy_prob = bankrupt_count / n_sim * 100

        daily_values = list(daily_pnl.values())
        daily_mean = float(np.mean(daily_values)) if daily_values else 0
        daily_std = float(np.std(daily_values)) if daily_values else 0

        max_dd_pct = max_dd / max(1, peak) * 100 if peak > 0 else 0

        step = max(1, len(balance_series) // 200)

        report['risk'] = {
            'max_drawdown_amount': int(max_dd),
            'max_drawdown_pct': round(max_dd_pct, 1),
            'max_losing_streak': max_consecutive_losses,
            'recovery_races': max_recovery,
            'bankruptcy_prob': round(bankruptcy_prob, 2),
            'daily_pnl_mean': round(daily_mean, 0),
            'daily_pnl_std': round(daily_std, 0),
            'initial_capital': 30000,
            'balance_series': balance_series[::step],
        }

        print(f"  Max DD: {max_dd:,} yen ({max_dd_pct:.1f}%)")
        print(f"  Max consecutive losses: {max_consecutive_losses}")
        print(f"  Recovery: {max_recovery} races")
        print(f"  Bankruptcy prob (30k, 1000R): {bankruptcy_prob:.2f}%")
        print(f"  Daily P&L: mean={daily_mean:,.0f}, std={daily_std:,.0f}")

        # Capital curve
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 5))
            # Show capital starting from 30000
            capital_series = [30000 + b for b in balance_series]
            ax.plot(capital_series, linewidth=0.5, color='green')
            ax.axhline(y=30000, color='r', linestyle='--', alpha=0.5, label='Initial capital')
            ax.set_xlabel('Race #')
            ax.set_ylabel('Capital (yen)')
            ax.set_title('Capital Curve (Initial: 30,000 yen, Real Payouts)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(ARTIFACTS_DIR, 'capital_curve.png'), dpi=150)
            plt.close(fig)
            print(f"  Saved: artifacts/phase4/capital_curve.png")
        except Exception:
            pass

        save_report()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['risk'] = {'error': str(e)}
        save_report()


# ============================================================
# MAIN
# ============================================================
def main():
    start_time = time.time()
    print("=" * 70)
    print("  PHASE 4: LIVE VALIDATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    df, features = prepare_data()

    print("\nLoading JRA payout data...")
    payout_lookup = load_payouts()

    # Get Phase 3 results (2023-2025) for tasks 3-6
    print("\nRunning walk-forward with real payouts (2023-2025)...")
    phase3_results = run_wf_with_payouts(df, features, payout_lookup, [2023, 2024, 2025])
    print(f"  Phase 3 base: {len(phase3_results)} races")

    # Task 1: 2026 OOS
    oos_results = task1_oos_2026(df, features, payout_lookup)

    # Task 2: Live DB validation
    task2_live_validation()

    # Combine Phase 3 + OOS for tasks 3-6
    all_results = phase3_results + oos_results
    print(f"\n  Combined dataset: {len(all_results)} races")

    # Task 3-6
    task3_roi_breakdown(all_results)
    task4_condition_analysis(all_results)
    task5_bet_filter(all_results)
    task6_risk(all_results)

    # Final verdict
    oos = report.get('oos_test_2026', {})
    t3 = report.get('roi_breakdown', {})
    t5 = report.get('bet_filter', {})
    t6 = report.get('risk', {})

    oos_verdict = oos.get('verdict', 'unknown')
    real_roi = t3.get('roi', 0)
    dep = t3.get('high_payout_dependency', 'unknown')
    ci_l = oos.get('ci_lower', 0)
    ci_u = oos.get('ci_upper', 0)
    bankruptcy = t6.get('bankruptcy_prob', 100)

    if oos_verdict == 'valid' and dep != 'high' and bankruptcy < 1:
        future_validity = 'high'
        confidence = 'high'
        production_ready = True
    elif oos_verdict in ('valid', 'uncertain') and bankruptcy < 5:
        future_validity = 'medium'
        confidence = 'medium'
        production_ready = True
    else:
        future_validity = 'low'
        confidence = 'low'
        production_ready = False

    report['final_verdict'] = {
        'future_validity': future_validity,
        'confidence': confidence,
        'realistic_roi_range': f"{ci_l:.0f}%-{ci_u:.0f}%",
        'production_ready': production_ready,
        'recommended_next_action': (
            'Continue live operation with current model. Monitor monthly ROI.'
            if production_ready else
            'Investigate OOS degradation before deploying.'
        ),
    }

    elapsed = time.time() - start_time
    report['elapsed_seconds'] = round(elapsed, 1)
    save_report()

    print(f"\n{'=' * 70}")
    print(f"  PHASE 4 COMPLETE ({elapsed/60:.1f} minutes)")
    print(f"  Future validity: {future_validity}")
    print(f"  Production ready: {production_ready}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
