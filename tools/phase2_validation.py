#!/usr/bin/env python
"""Phase 2 Validation: ROI reliability verification

7 validation tasks to check if estimated ROI ~3627% is realistic.
No model changes, no optimization — verification only.
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
    calc_wide_bets, check_bets, estimate_payouts, encode_sires_fold,
    train_lgb_fold,
)

ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts', 'phase2')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

TEST_YEARS = [2023, 2024, 2025]
WF_SPLITS = [
    (2020, 2021),
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
    (2024, 2025),
]

report = {
    'timestamp': None,
    'task1_yearly_roi': {},
    'task2_roi_curve': {},
    'task3_shuffle_test': {},
    'task4_no_odds_test': {},
    'task5_random_bet': {},
    'task6_walk_forward': {},
    'task7_payout_dependency': {},
    'final_diagnosis': {},
}


def save_report():
    """Save current report state."""
    report['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    path = os.path.join(DATA_DIR, 'phase2_validation_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Report saved: {path}")


def prepare_data():
    """Load and prepare data using existing pipeline."""
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

    print(f"  Data ready: {len(df)} rows, {df['race_id_str'].nunique()} races")
    return df, features


def run_backtest_year(df, features, test_year):
    """Run WF backtest for a single test year. Returns per-race results list."""
    train_mask = (df['year_full'] >= 2010) & (df['year_full'] < test_year)
    test_mask = df['year_full'] == test_year
    n_test = test_mask.sum()
    if n_test < 100:
        return [], 0.0

    df_fold = df.copy()
    df_fold = encode_sires_fold(df_fold, train_mask)
    for f in features:
        if f not in df_fold.columns:
            df_fold[f] = 0
        df_fold[f] = pd.to_numeric(df_fold[f], errors='coerce').fillna(0)

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

    model = train_lgb_fold(X_tr, y_tr, X_va, y_va, features)

    test_df = df_fold[test_mask].copy()
    X_test = test_df[features].values
    test_df['pred'] = model.predict(X_test)
    test_auc = roc_auc_score(test_df['target'].values, test_df['pred'])

    results = []
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

        race_df = race_df.sort_values('pred', ascending=False)
        ranking = race_df['umaban'].astype(int).tolist()

        trio_bets = calc_trio_bets(ranking)
        umaren_bets = calc_umaren_bets(ranking)
        wide_bets = calc_wide_bets(ranking)

        trio_hit, umaren_hits, wide_hits = check_bets(
            actual_top3, trio_bets, umaren_bets, wide_bets)
        trio_pay, umaren_pay, wide_pays = estimate_payouts(actual_top3, race_df)

        trio_return = trio_pay if trio_hit else 0
        umaren_return = umaren_pay * 3.5 * len(umaren_hits) if umaren_hits else 0

        results.append({
            'race_id': rid,
            'year': test_year,
            'cond_key': cond_key,
            'trio_hit': trio_hit,
            'trio_return': trio_return,
            'investment': 700,
            'date_num': int(row0['date_num']),
        })

    return results, test_auc


def summarize_results(results):
    """Compute ROI summary from results list."""
    if not results:
        return {'roi': 0, 'bets': 0, 'hit_rate': 0, 'total_return': 0, 'total_investment': 0}
    n = len(results)
    hits = sum(1 for r in results if r['trio_hit'])
    total_return = sum(r['trio_return'] for r in results)
    total_inv = n * 700
    return {
        'roi': round(total_return / total_inv * 100, 1) if total_inv > 0 else 0,
        'bets': n,
        'hit_rate': round(hits / n * 100, 1) if n > 0 else 0,
        'total_return': int(total_return),
        'total_investment': total_inv,
        'hits': hits,
    }


# ============================================================
# TASK 1: Yearly ROI Decomposition
# ============================================================
def task1_yearly_roi(df, features):
    print("\n" + "=" * 70)
    print("  TASK 1: YEARLY ROI DECOMPOSITION")
    print("=" * 70)
    try:
        all_results = []
        for yr in TEST_YEARS:
            t0 = time.time()
            results, auc = run_backtest_year(df, features, yr)
            elapsed = time.time() - t0
            summary = summarize_results(results)
            summary['auc'] = round(auc, 4)
            report['task1_yearly_roi'][str(yr)] = summary
            all_results.extend(results)
            print(f"  {yr}: ROI {summary['roi']:.1f}%, bets={summary['bets']}, "
                  f"hit_rate={summary['hit_rate']:.1f}%, AUC={auc:.4f} ({elapsed:.0f}s)")

        total = summarize_results(all_results)
        report['task1_yearly_roi']['total'] = total
        print(f"  TOTAL: ROI {total['roi']:.1f}%, bets={total['bets']}, hit_rate={total['hit_rate']:.1f}%")
        save_report()
        return all_results
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['task1_yearly_roi']['error'] = str(e)
        save_report()
        return []


# ============================================================
# TASK 2: Cumulative ROI Curve
# ============================================================
def task2_roi_curve(all_results):
    print("\n" + "=" * 70)
    print("  TASK 2: CUMULATIVE ROI CURVE")
    print("=" * 70)
    try:
        if not all_results:
            report['task2_roi_curve'] = {'error': 'No results from task1'}
            save_report()
            return

        # Sort by date
        sorted_results = sorted(all_results, key=lambda r: (r['date_num'], r['race_id']))

        cum_return = 0
        cum_invest = 0
        roi_series = []
        peak_balance = 0
        max_dd = 0
        balance_series = []

        for r in sorted_results:
            cum_invest += 700
            cum_return += r['trio_return']
            balance = cum_return - cum_invest
            roi = cum_return / cum_invest * 100 if cum_invest > 0 else 0
            roi_series.append(round(roi, 1))

            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / max(1, peak_balance + cum_invest) * 100
            if dd > max_dd:
                max_dd = dd
            balance_series.append(int(balance))

        # Sample every 50 races for JSON size
        step = max(1, len(roi_series) // 200)
        roi_sampled = roi_series[::step]
        balance_sampled = balance_series[::step]

        report['task2_roi_curve'] = {
            'total_races': len(roi_series),
            'final_roi': roi_series[-1] if roi_series else 0,
            'max_drawdown_pct': round(max_dd, 2),
            'roi_series': roi_sampled,
            'balance_series': balance_sampled,
            'min_balance': min(balance_series) if balance_series else 0,
            'max_balance': max(balance_series) if balance_series else 0,
        }

        print(f"  Total races: {len(roi_series)}")
        print(f"  Final ROI: {roi_series[-1]:.1f}%")
        print(f"  Max drawdown: {max_dd:.2f}%")
        print(f"  Balance range: {min(balance_series):,} ~ {max(balance_series):,}")

        # Plot
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            ax1.plot(roi_series, linewidth=0.5)
            ax1.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Break-even (100%)')
            ax1.set_xlabel('Race #')
            ax1.set_ylabel('Cumulative ROI (%)')
            ax1.set_title('Cumulative ROI Over Time (Test 2023-2025)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(balance_series, linewidth=0.5, color='green')
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Race #')
            ax2.set_ylabel('Cumulative P&L (yen)')
            ax2.set_title('Cumulative Balance')
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(os.path.join(ARTIFACTS_DIR, 'roi_curve.png'), dpi=150)
            plt.close(fig)
            print(f"  Saved: artifacts/phase2/roi_curve.png")
        except Exception as e:
            print(f"  Warning: Plot failed: {e}")

        save_report()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['task2_roi_curve'] = {'error': str(e)}
        save_report()


# ============================================================
# TASK 3: Label Shuffle Test
# ============================================================
def task3_shuffle_test(df, features, original_roi):
    print("\n" + "=" * 70)
    print("  TASK 3: LABEL SHUFFLE TEST (LEAK DETECTION)")
    print("=" * 70)
    try:
        n_shuffles = 5
        shuffled_rois = []

        for trial in range(n_shuffles):
            t0 = time.time()
            rng = np.random.RandomState(42 + trial)

            trial_results = []
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

                train_df = df_fold[train_mask].copy()

                # SHUFFLE labels within training data
                shuffled_target = train_df['target'].values.copy()
                rng.shuffle(shuffled_target)

                dates = train_df['date_num']
                valid_cutoff = dates.quantile(0.85)
                tr_idx = dates < valid_cutoff
                va_idx = dates >= valid_cutoff

                X_tr = train_df.loc[tr_idx, features].values
                y_tr = shuffled_target[tr_idx.values]
                X_va = train_df.loc[va_idx, features].values
                y_va = shuffled_target[va_idx.values]

                model = train_lgb_fold(X_tr, y_tr, X_va, y_va, features)

                test_df = df_fold[test_mask].copy()
                test_df['pred'] = model.predict(test_df[features].values)

                for rid in test_df['race_id_str'].unique():
                    race_df = test_df[test_df['race_id_str'] == rid].copy()
                    if len(race_df) < 5:
                        continue

                    row0 = race_df.iloc[0]
                    race_sorted = race_df.sort_values('finish')
                    actual_top3 = {}
                    for _, r in race_sorted.head(3).iterrows():
                        actual_top3[int(r['finish'])] = int(r['umaban'])
                    if len(actual_top3) < 3:
                        continue

                    race_df = race_df.sort_values('pred', ascending=False)
                    ranking = race_df['umaban'].astype(int).tolist()
                    trio_bets = calc_trio_bets(ranking)
                    trio_hit, _, _ = check_bets(actual_top3, trio_bets, [], [])
                    trio_pay, _, _ = estimate_payouts(actual_top3, race_df)
                    trio_return = trio_pay if trio_hit else 0

                    trial_results.append({
                        'trio_hit': trio_hit,
                        'trio_return': trio_return,
                    })

            summary = summarize_results(trial_results)
            shuffled_rois.append(summary['roi'])
            elapsed = time.time() - t0
            print(f"  Shuffle {trial+1}/{n_shuffles}: ROI {summary['roi']:.1f}%, "
                  f"hits={summary['hits']}/{summary['bets']} ({elapsed:.0f}s)")

        avg_shuffled = round(np.mean(shuffled_rois), 1) if shuffled_rois else 0
        leak_suspected = avg_shuffled >= 100

        report['task3_shuffle_test'] = {
            'original_roi': original_roi,
            'shuffled_rois': shuffled_rois,
            'shuffled_avg_roi': avg_shuffled,
            'leak_suspected': leak_suspected,
            'verdict': 'LEAK/BUG SUSPECTED' if leak_suspected else 'PASS (no leak detected)',
        }

        print(f"\n  Original ROI: {original_roi:.1f}%")
        print(f"  Shuffled avg ROI: {avg_shuffled:.1f}%")
        print(f"  Verdict: {'LEAK/BUG SUSPECTED' if leak_suspected else 'PASS'}")
        save_report()

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['task3_shuffle_test'] = {'error': str(e)}
        save_report()


# ============================================================
# TASK 4: No-Odds Test
# ============================================================
def task4_no_odds(df, features, original_roi):
    print("\n" + "=" * 70)
    print("  TASK 4: ODDS REMOVAL TEST (MARKET DEPENDENCY)")
    print("=" * 70)
    try:
        # Identify all odds-related features in Pattern A
        odds_features = [f for f in features if 'odds' in f.lower() or 'pop' in f.lower()]
        print(f"  Odds-related features found: {odds_features}")

        if not odds_features:
            print("  No odds features in Pattern A (already removed). Testing prev_odds_log...")
            odds_features = ['prev_odds_log']

        features_no_odds = [f for f in features if f not in odds_features]
        print(f"  Features: {len(features)} -> {len(features_no_odds)} (removed {odds_features})")

        all_results = []
        for test_year in TEST_YEARS:
            train_mask = (df['year_full'] >= 2010) & (df['year_full'] < test_year)
            test_mask = df['year_full'] == test_year
            if test_mask.sum() < 100:
                continue

            df_fold = df.copy()
            df_fold = encode_sires_fold(df_fold, train_mask)
            for f in features_no_odds:
                if f not in df_fold.columns:
                    df_fold[f] = 0
                df_fold[f] = pd.to_numeric(df_fold[f], errors='coerce').fillna(0)

            train_df = df_fold[train_mask]
            y_train = train_df['target'].values
            dates = train_df['date_num']
            valid_cutoff = dates.quantile(0.85)
            tr_idx = dates < valid_cutoff
            va_idx = dates >= valid_cutoff

            X_tr = train_df.loc[tr_idx, features_no_odds].values
            y_tr = y_train[tr_idx.values]
            X_va = train_df.loc[va_idx, features_no_odds].values
            y_va = y_train[va_idx.values]

            model = train_lgb_fold(X_tr, y_tr, X_va, y_va, features_no_odds)

            test_df = df_fold[test_mask].copy()
            test_df['pred'] = model.predict(test_df[features_no_odds].values)
            auc = roc_auc_score(test_df['target'].values, test_df['pred'])

            for rid in test_df['race_id_str'].unique():
                race_df = test_df[test_df['race_id_str'] == rid].copy()
                if len(race_df) < 5:
                    continue
                row0 = race_df.iloc[0]
                race_sorted = race_df.sort_values('finish')
                actual_top3 = {}
                for _, r in race_sorted.head(3).iterrows():
                    actual_top3[int(r['finish'])] = int(r['umaban'])
                if len(actual_top3) < 3:
                    continue

                race_df = race_df.sort_values('pred', ascending=False)
                ranking = race_df['umaban'].astype(int).tolist()
                trio_bets = calc_trio_bets(ranking)
                trio_hit, _, _ = check_bets(actual_top3, trio_bets, [], [])
                trio_pay, _, _ = estimate_payouts(actual_top3, race_df)

                all_results.append({
                    'trio_hit': trio_hit,
                    'trio_return': trio_pay if trio_hit else 0,
                    'investment': 700,
                    'year': test_year,
                })

            print(f"  {test_year}: AUC {auc:.4f}")

        summary = summarize_results(all_results)
        drop = original_roi - summary['roi']

        report['task4_no_odds_test'] = {
            'removed_features': odds_features,
            'n_features_original': len(features),
            'n_features_remaining': len(features_no_odds),
            'roi': summary['roi'],
            'bets': summary['bets'],
            'hit_rate': summary['hit_rate'],
            'original_roi': original_roi,
            'performance_drop': round(drop, 1),
            'drop_pct': round(drop / original_roi * 100, 1) if original_roi > 0 else 0,
        }

        print(f"\n  Original ROI: {original_roi:.1f}%")
        print(f"  No-odds ROI: {summary['roi']:.1f}%")
        print(f"  Drop: {drop:.1f}% ({report['task4_no_odds_test']['drop_pct']:.1f}%)")
        save_report()

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['task4_no_odds_test'] = {'error': str(e)}
        save_report()


# ============================================================
# TASK 5: Random Bet Comparison
# ============================================================
def task5_random_bet(df, features, original_roi, original_results):
    print("\n" + "=" * 70)
    print("  TASK 5: RANDOM BET COMPARISON")
    print("=" * 70)
    try:
        # Pre-compute race data once (avoid iterrows in inner loop)
        race_data = []
        for test_year in TEST_YEARS:
            test_mask = df['year_full'] == test_year
            test_df = df[test_mask].copy()

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

                all_umaban = race_df['umaban'].astype(int).tolist()
                top3_set = frozenset(actual_top3.values())

                # Pre-compute estimated trio payout (doesn't depend on ranking)
                odds = {int(r['umaban']): float(r['tansho_odds'])
                        for _, r in race_df.iterrows()}
                o1 = odds.get(actual_top3.get(1), 10.0)
                o2 = odds.get(actual_top3.get(2), 10.0)
                o3 = odds.get(actual_top3.get(3), 10.0)
                trio_pay = max(100, int(o1 * o2 * o3 * 20))

                race_data.append({
                    'all_umaban': np.array(all_umaban),
                    'top3_set': top3_set,
                    'trio_pay': trio_pay,
                })

        n_races = len(race_data)
        print(f"  Races for random test: {n_races}")
        print(f"  Pre-computed payout data. Running 1000 trials...")

        n_trials = 1000
        random_rois = []

        for trial in range(n_trials):
            rng = np.random.RandomState(trial)
            total_return = 0

            for rd in race_data:
                # Random ranking
                ranking = rng.permutation(rd['all_umaban']).tolist()
                # calc_trio_bets inline for speed
                if len(ranking) < 3:
                    continue
                nums = ranking[:6] if len(ranking) >= 6 else ranking
                n1 = nums[0]
                second = nums[1:3]
                third = nums[1:min(6, len(nums))]
                trio_bets = set()
                for s in second:
                    for t in third:
                        combo = frozenset({n1, s, t})
                        if len(combo) == 3:
                            trio_bets.add(combo)

                if rd['top3_set'] in trio_bets:
                    total_return += rd['trio_pay']

            total_invest = n_races * 700
            roi = total_return / total_invest * 100 if total_invest > 0 else 0
            random_rois.append(roi)

            if (trial + 1) % 200 == 0:
                print(f"  Trial {trial+1}/{n_trials}: avg ROI so far = {np.mean(random_rois):.1f}%")

        random_rois = np.array(random_rois)
        mean_roi = float(np.mean(random_rois))
        median_roi = float(np.median(random_rois))
        ci_lower = float(np.percentile(random_rois, 2.5))
        ci_upper = float(np.percentile(random_rois, 97.5))
        model_outside = original_roi > ci_upper or original_roi < ci_lower

        report['task5_random_bet'] = {
            'model_roi': original_roi,
            'random_mean_roi': round(mean_roi, 1),
            'random_median_roi': round(median_roi, 1),
            'random_95ci_lower': round(ci_lower, 1),
            'random_95ci_upper': round(ci_upper, 1),
            'random_std': round(float(np.std(random_rois)), 1),
            'model_outside_95ci': model_outside,
            'n_trials': n_trials,
            'n_races': n_races,
        }

        print(f"\n  Model ROI: {original_roi:.1f}%")
        print(f"  Random mean: {mean_roi:.1f}%")
        print(f"  Random 95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]")
        print(f"  Model outside 95% CI: {model_outside}")

        if not model_outside:
            print("  WARNING: Model ROI is within random 95% CI!")
        else:
            print("  Model significantly beats random.")

        save_report()

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['task5_random_bet'] = {'error': str(e)}
        save_report()


# ============================================================
# TASK 6: Walk-Forward Verification
# ============================================================
def task6_walk_forward(df, features):
    print("\n" + "=" * 70)
    print("  TASK 6: WALK-FORWARD VERIFICATION (OVERFIT CHECK)")
    print("=" * 70)
    try:
        for train_end, test_year in WF_SPLITS:
            t0 = time.time()
            results, auc = run_backtest_year(df, features, test_year)
            elapsed = time.time() - t0
            summary = summarize_results(results)
            summary['auc'] = round(auc, 4)
            summary['train_period'] = f"2010-{train_end}"
            report['task6_walk_forward'][str(test_year)] = summary
            print(f"  Train 2010-{train_end} -> Test {test_year}: "
                  f"ROI {summary['roi']:.1f}%, bets={summary['bets']}, "
                  f"hit_rate={summary['hit_rate']:.1f}%, AUC={auc:.4f} ({elapsed:.0f}s)")

        # Check consistency
        rois = [report['task6_walk_forward'][str(yr)]['roi']
                for yr in range(2021, 2026) if str(yr) in report['task6_walk_forward']]
        if rois:
            roi_std = np.std(rois)
            roi_mean = np.mean(rois)
            report['task6_walk_forward']['summary'] = {
                'mean_roi': round(float(roi_mean), 1),
                'std_roi': round(float(roi_std), 1),
                'cv': round(float(roi_std / roi_mean * 100), 1) if roi_mean > 0 else 0,
                'all_above_100': all(r > 100 for r in rois),
            }
            print(f"\n  Mean ROI: {roi_mean:.1f}%, Std: {roi_std:.1f}%, CV: {roi_std/max(1,roi_mean)*100:.1f}%")
            print(f"  All years > 100%: {all(r > 100 for r in rois)}")

        save_report()

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['task6_walk_forward']['error'] = str(e)
        save_report()


# ============================================================
# TASK 7: High Payout Dependency Analysis
# ============================================================
def task7_payout_dependency(all_results, original_roi):
    print("\n" + "=" * 70)
    print("  TASK 7: HIGH PAYOUT DEPENDENCY ANALYSIS")
    print("=" * 70)
    try:
        if not all_results:
            report['task7_payout_dependency'] = {'error': 'No results'}
            save_report()
            return

        # Get all hits with their payouts
        hits = [(r['trio_return'], i) for i, r in enumerate(all_results) if r['trio_hit']]
        hits_sorted = sorted(hits, key=lambda x: -x[0])  # Descending by payout

        total_n = len(all_results)
        total_inv = total_n * 700
        total_return = sum(r['trio_return'] for r in all_results)
        full_roi = total_return / total_inv * 100

        # Exclude top-N hits
        exclusion_results = {}
        for exclude_n in [5, 10, 20]:
            if len(hits_sorted) <= exclude_n:
                excluded_return = sum(p for p, _ in hits_sorted)
            else:
                excluded_return = sum(p for p, _ in hits_sorted[:exclude_n])
            remaining_return = total_return - excluded_return
            remaining_roi = remaining_return / total_inv * 100
            exclusion_results[f'exclude_top{exclude_n}'] = {
                'roi': round(remaining_roi, 1),
                'excluded_return': int(excluded_return),
                'remaining_return': int(remaining_return),
            }
            print(f"  Exclude top {exclude_n}: ROI {remaining_roi:.1f}% "
                  f"(removed {excluded_return:,} yen from returns)")

        # Payout distribution of hits
        hit_payouts = [p for p, _ in hits]

        # Determine dependency level
        ex5_roi = exclusion_results.get('exclude_top5', {}).get('roi', 0)
        ex10_roi = exclusion_results.get('exclude_top10', {}).get('roi', 0)
        ex20_roi = exclusion_results.get('exclude_top20', {}).get('roi', 0)

        if ex5_roi < 100:
            dep_level = 'high'
        elif ex10_roi < 100:
            dep_level = 'medium'
        else:
            dep_level = 'low'

        report['task7_payout_dependency'] = {
            'full_roi': round(full_roi, 1),
            'total_hits': len(hits),
            'total_bets': total_n,
            'exclude_top5_roi': ex5_roi,
            'exclude_top10_roi': ex10_roi,
            'exclude_top20_roi': ex20_roi,
            'exclusion_details': exclusion_results,
            'dependency_level': dep_level,
            'top5_payouts': [int(p) for p, _ in hits_sorted[:5]] if len(hits_sorted) >= 5 else [],
            'top10_payouts': [int(p) for p, _ in hits_sorted[:10]] if len(hits_sorted) >= 10 else [],
            'payout_stats': {
                'mean': round(float(np.mean(hit_payouts)), 0) if hit_payouts else 0,
                'median': round(float(np.median(hit_payouts)), 0) if hit_payouts else 0,
                'std': round(float(np.std(hit_payouts)), 0) if hit_payouts else 0,
                'max': int(max(hit_payouts)) if hit_payouts else 0,
                'min': int(min(hit_payouts)) if hit_payouts else 0,
                'p25': round(float(np.percentile(hit_payouts, 25)), 0) if hit_payouts else 0,
                'p75': round(float(np.percentile(hit_payouts, 75)), 0) if hit_payouts else 0,
            },
        }

        print(f"\n  Full ROI: {full_roi:.1f}%")
        print(f"  Total hits: {len(hits)}/{total_n}")
        print(f"  Top 5 payouts: {[int(p) for p, _ in hits_sorted[:5]]}")
        print(f"  Dependency level: {dep_level.upper()}")

        # Histogram
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(hit_payouts, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(x=np.mean(hit_payouts), color='r', linestyle='--',
                       label=f'Mean: {np.mean(hit_payouts):,.0f}')
            ax.axvline(x=np.median(hit_payouts), color='g', linestyle='--',
                       label=f'Median: {np.median(hit_payouts):,.0f}')
            ax.set_xlabel('Payout (yen per 100 yen bet)')
            ax.set_ylabel('Count')
            ax.set_title(f'Hit Payout Distribution (N={len(hit_payouts)})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(ARTIFACTS_DIR, 'payout_distribution.png'), dpi=150)
            plt.close(fig)
            print(f"  Saved: artifacts/phase2/payout_distribution.png")
        except Exception as e:
            print(f"  Warning: Histogram failed: {e}")

        save_report()

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report['task7_payout_dependency'] = {'error': str(e)}
        save_report()


# ============================================================
# FINAL DIAGNOSIS
# ============================================================
def final_diagnosis():
    print("\n" + "=" * 70)
    print("  FINAL DIAGNOSIS")
    print("=" * 70)

    t3 = report.get('task3_shuffle_test', {})
    t4 = report.get('task4_no_odds_test', {})
    t5 = report.get('task5_random_bet', {})
    t6 = report.get('task6_walk_forward', {})
    t7 = report.get('task7_payout_dependency', {})

    leak_suspected = t3.get('leak_suspected', False)
    overfit = False
    wf_summary = t6.get('summary', {})
    if wf_summary and not wf_summary.get('all_above_100', True):
        overfit = True

    market_dep = 'low'
    drop_pct = t4.get('drop_pct', 0)
    if drop_pct > 30:
        market_dep = 'high'
    elif drop_pct > 15:
        market_dep = 'medium'

    payout_dep = t7.get('dependency_level', 'unknown')

    model_outside = t5.get('model_outside_95ci', True)

    # Overall reliability
    issues = []
    if leak_suspected:
        issues.append('leak_detected')
    if overfit:
        issues.append('overfit_detected')
    if not model_outside:
        issues.append('no_advantage_over_random')
    if payout_dep == 'high':
        issues.append('high_payout_dependency')

    if len(issues) >= 2:
        reliability = 'low'
    elif len(issues) == 1:
        reliability = 'medium'
    else:
        reliability = 'high'

    diagnosis = {
        'leak_suspected': leak_suspected,
        'overfit_suspected': overfit,
        'market_dependency': market_dep,
        'payout_dependency': payout_dep,
        'model_beats_random': model_outside,
        'roi_reliability': reliability,
        'issues': issues if issues else ['none'],
    }

    report['final_diagnosis'] = diagnosis

    print(f"  Leak suspected: {leak_suspected}")
    print(f"  Overfit suspected: {overfit}")
    print(f"  Market dependency: {market_dep}")
    print(f"  Payout dependency: {payout_dep}")
    print(f"  Model beats random: {model_outside}")
    print(f"  ROI reliability: {reliability}")
    if issues:
        print(f"  Issues: {issues}")

    save_report()


# ============================================================
# MAIN
# ============================================================
def main():
    start_time = time.time()
    print("=" * 70)
    print("  PHASE 2 VALIDATION SUITE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Purpose: Verify ROI reliability (no model changes)")
    print("=" * 70)

    df, features = prepare_data()

    # Task 1: Yearly ROI
    all_results = task1_yearly_roi(df, features)
    original_roi = report['task1_yearly_roi'].get('total', {}).get('roi', 0)

    # Task 2: ROI Curve
    task2_roi_curve(all_results)

    # Task 3: Shuffle Test
    task3_shuffle_test(df, features, original_roi)

    # Task 4: No-Odds Test
    task4_no_odds(df, features, original_roi)

    # Task 5: Random Bet
    task5_random_bet(df, features, original_roi, all_results)

    # Task 6: Walk-Forward
    task6_walk_forward(df, features)

    # Task 7: Payout Dependency
    task7_payout_dependency(all_results, original_roi)

    # Final Diagnosis
    final_diagnosis()

    elapsed = time.time() - start_time
    report['elapsed_seconds'] = round(elapsed, 1)
    report['note'] = ('ROI values use estimated payouts (o1*o2*o3*20) which are ~2x actual. '
                      'Relative comparisons are valid. '
                      'See CLAUDE.md for actual JRA payout ROI baseline.')
    save_report()

    print(f"\n{'=' * 70}")
    print(f"  VALIDATION COMPLETE ({elapsed/60:.1f} minutes)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
