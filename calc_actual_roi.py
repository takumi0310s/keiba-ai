#!/usr/bin/env python
"""実配当ROI計算スクリプト

jra_payouts.csv（JRA公式配当データ）とウォークフォワードバックテストを
組み合わせて、実配当ベースのROIを条件別・券種別に算出する。

出力:
- data/actual_roi_results.json
- コンソールに推定ROI vs 実ROI比較表
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

from backtest_central_leakfree import (
    classify_condition, get_axes, calc_trio_bets, calc_umaren_bets,
    calc_wide_bets, check_bets, estimate_payouts, encode_sires_fold,
    train_lgb_fold,
)

TEST_YEARS = list(range(2020, 2026))

# Course code (TARGET JV) → course name mapping
COURSE_CODE_TO_NAME = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉',
}
COURSE_NAME_TO_CODE = {v: k for k, v in COURSE_CODE_TO_NAME.items()}

# Nichi encoding: 1-9 → '1'-'9', 10 → 'A', 11 → 'B', 12 → 'C'
NICHI_TO_CHAR = {i: str(i) for i in range(1, 10)}
NICHI_TO_CHAR.update({10: 'A', 11: 'B', 12: 'C'})


def load_payouts():
    """jra_payouts.csvをロードしてルックアップ辞書を作成"""
    path = os.path.join(BASE_DIR, 'data', 'jra_payouts.csv')
    pay_df = pd.read_csv(path, encoding='utf-8', dtype=str)
    print(f"  Payouts loaded: {len(pay_df)} records")

    # Build lookup: race_id_str (8 chars) → payout info
    # race_id_str format: CC(2) + YY(2) + K(1) + N(1, hex for 10+) + RR(2)
    lookup = {}
    for _, row in pay_df.iterrows():
        date_str = str(row['race_date'])  # YYYYMMDD
        course_name = str(row['course'])
        kai_int = int(row['kai'])
        nichi_int = int(row['nichi'])
        race_num = str(row['race_num']).zfill(2)

        cc = COURSE_NAME_TO_CODE.get(course_name)
        if cc is None:
            continue

        year_2d = date_str[2:4]  # YYYY → YY
        nichi_char = NICHI_TO_CHAR.get(nichi_int, str(nichi_int))

        # race_id_str (8 chars) = CC(2) + YY(2) + K(1) + N(1) + RR(2)
        key = f"{cc}{year_2d}{kai_int}{nichi_char}{race_num}"

        lookup[key] = {
            'race_date': date_str,
            'trio_nums': str(row.get('trio_nums', '')),
            'trio_payout': int(row.get('trio_payout', 0)) if str(row.get('trio_payout', '0')).isdigit() else 0,
            'umaren_nums': str(row.get('umaren_nums', '')),
            'umaren_payout': int(row.get('umaren_payout', 0)) if str(row.get('umaren_payout', '0')).isdigit() else 0,
            'wide_nums': str(row.get('wide_nums', '')),
            'wide_payouts': str(row.get('wide_payouts', '')),
        }

    print(f"  Lookup built: {len(lookup)} entries")
    # Check coverage for test years
    for yr in TEST_YEARS:
        yr_2d = str(yr)[2:]
        n = sum(1 for k in lookup if k[2:4] == yr_2d)
        print(f"    {yr}: {n} races")
    return lookup


def parse_trio_nums(trio_str):
    """Parse trio_nums like '1-5-7' → frozenset({1, 5, 7})"""
    if not trio_str or trio_str == 'nan':
        return None
    try:
        parts = trio_str.split('-')
        return frozenset(int(x) for x in parts)
    except (ValueError, AttributeError):
        return None


def parse_umaren_nums(umaren_str):
    """Parse umaren_nums like '3-7' → frozenset({3, 7})"""
    if not umaren_str or umaren_str == 'nan':
        return None
    try:
        parts = umaren_str.split('-')
        return frozenset(int(x) for x in parts)
    except (ValueError, AttributeError):
        return None


def parse_wide_data(wide_nums_str, wide_payouts_str):
    """Parse wide data: '3-7/7-15/3-15' and '360/780/1040' → {frozenset: payout}"""
    result = {}
    if not wide_nums_str or wide_nums_str == 'nan':
        return result
    try:
        nums_list = wide_nums_str.split('/')
        pays_list = wide_payouts_str.split('/')
        for ns, ps in zip(nums_list, pays_list):
            parts = ns.split('-')
            key = frozenset(int(x) for x in parts)
            result[key] = int(ps)
    except (ValueError, AttributeError):
        pass
    return result


def get_actual_payouts(payout_lookup, race_id_str):
    """Look up actual payouts for a race. race_id_str already includes race_num."""
    return payout_lookup.get(race_id_str)


def calc_actual_returns(payout_info, trio_bets, umaren_bets, wide_bets):
    """Calculate actual returns using real payout data."""
    trio_hit = False
    trio_return = 0
    umaren_hit = False
    umaren_return = 0
    wide_hit_count = 0
    wide_return = 0

    if payout_info is None:
        return trio_hit, trio_return, umaren_hit, umaren_return, wide_hit_count, wide_return

    # Trio check
    actual_trio = parse_trio_nums(payout_info['trio_nums'])
    if actual_trio:
        for bet in trio_bets:
            if frozenset(bet) == actual_trio:
                trio_hit = True
                trio_return = payout_info['trio_payout']  # 100円あたりの配当
                break

    # Umaren check
    actual_umaren = parse_umaren_nums(payout_info['umaren_nums'])
    if actual_umaren:
        for bet in umaren_bets:
            if frozenset(bet) == actual_umaren:
                umaren_hit = True
                umaren_return = payout_info['umaren_payout']  # 100円あたりの配当
                break

    # Wide check - multiple combinations can hit
    actual_wide = parse_wide_data(payout_info['wide_nums'], payout_info['wide_payouts'])
    for bet in wide_bets:
        bet_set = frozenset(bet)
        if bet_set in actual_wide:
            wide_hit_count += 1
            wide_return += actual_wide[bet_set]

    return trio_hit, trio_return, umaren_hit, umaren_return, wide_hit_count, wide_return


def analyze_groups_actual(groups, min_n=30):
    """Analyze groups with actual ROI."""
    analysis = {}
    for label, races in sorted(groups.items()):
        n = len(races)
        if n < min_n:
            continue

        bet_results = {}
        for bt in ['trio', 'umaren', 'wide']:
            if bt == 'trio':
                # 7点 × 100円 = 700円
                hits = sum(1 for r in races if r.get('actual_trio_hit', False))
                investment = n * 700
                payout = sum(r.get('actual_trio_return', 0) for r in races)
            elif bt == 'umaren':
                # 2点 × 350円 = 700円
                hits = sum(1 for r in races if r.get('actual_umaren_hit', False))
                investment = n * 700
                payout = sum(r.get('actual_umaren_return', 0) * 3.5 for r in races)
            else:  # wide
                # 2点 × 350円 = 700円
                hits = sum(1 for r in races if r.get('actual_wide_hits', 0) > 0)
                investment = n * 700
                payout = sum(r.get('actual_wide_return', 0) * 3.5 for r in races)

            roi = payout / investment * 100 if investment > 0 else 0
            hit_rate = hits / n * 100

            bet_results[bt] = {
                'hits': hits, 'hit_rate': round(hit_rate, 1),
                'investment': investment, 'payout': int(payout),
                'roi': round(roi, 1),
            }

        # Also calculate estimated ROI for comparison
        est_results = {}
        for bt in ['trio', 'umaren', 'wide']:
            if bt == 'trio':
                hits = sum(1 for r in races if r.get('trio_hit', False))
                investment = n * 700
                payout = sum(r.get('trio_return', 0) for r in races)
            elif bt == 'umaren':
                hits = sum(1 for r in races if r.get('umaren_hit', False))
                investment = n * 700
                payout = sum(r.get('umaren_return', 0) for r in races)
            else:
                hits = sum(1 for r in races if r.get('wide_hits', 0) > 0)
                investment = n * 700
                payout = sum(r.get('wide_return', 0) for r in races)

            est_results[bt] = {
                'roi': round(payout / investment * 100 if investment > 0 else 0, 1),
            }

        best_bt = max(bet_results, key=lambda b: bet_results[b]['roi'])
        best_roi = bet_results[best_bt]['roi']

        analysis[label] = {
            'n': n,
            'best_bet': best_bt,
            'best_roi': best_roi,
            'recommended': best_roi >= 100,
            'bets': bet_results,
            'estimated': est_results,
        }

    return analysis


def main():
    print("=" * 70)
    print("  実配当ROI計算 (JRA公式配当データ × WFバックテスト)")
    print(f"  Features: Pattern A ({len(FEATURES_PATTERN_A)} features)")
    print(f"  Test years: {TEST_YEARS}")
    print("=" * 70)

    # Load payout data
    print("\n[1] 配当データ読み込み...")
    payout_lookup = load_payouts()

    # Load and prepare race data
    print("\n[2] レースデータ準備...")
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

    # Walk-forward backtest
    print("\n[3] ウォークフォワードバックテスト...")
    all_results = []
    fold_aucs = {}
    matched_count = 0
    unmatched_count = 0

    for test_year in TEST_YEARS:
        train_mask = (df['year_full'] >= 2010) & (df['year_full'] < test_year)
        test_mask = df['year_full'] == test_year
        n_test = test_mask.sum()
        if n_test < 100:
            continue

        print(f"\n  --- Train 2010-{test_year-1} → Test {test_year} ({n_test:,}) ---")

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

        t0 = time.time()
        model = train_lgb_fold(X_tr, y_tr, X_va, y_va, features)
        elapsed = time.time() - t0

        test_df_fold = df_fold[test_mask].copy()
        X_test = test_df_fold[features].values
        test_df_fold['pred'] = model.predict(X_test)
        test_auc = roc_auc_score(test_df_fold['target'].values, test_df_fold['pred'])
        fold_aucs[test_year] = test_auc
        print(f"  AUC: {test_auc:.4f} ({elapsed:.0f}s)")

        # Evaluate each race
        test_races = test_df_fold['race_id_str'].unique()
        year_trio_hits = 0
        year_total = 0
        year_matched = 0

        for rid in test_races:
            race_df = test_df_fold[test_df_fold['race_id_str'] == rid].copy()
            if len(race_df) < 5:
                continue

            row0 = race_df.iloc[0]
            axes = get_axes(row0)
            race_num = int(row0['race_num'])

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

            # Estimated payouts (existing method)
            trio_hit, umaren_hits, wide_hits = check_bets(actual_top3, trio_bets, umaren_bets, wide_bets)
            trio_pay_est, umaren_pay_est, wide_pays_est = estimate_payouts(actual_top3, race_df)
            trio_return_est = trio_pay_est if trio_hit else 0
            umaren_return_est = umaren_pay_est * 3.5 * len(umaren_hits) if umaren_hits else 0
            wide_return_est = sum(wide_pays_est.get(tuple(sorted(w)), 150) * 3.5 for w in wide_hits)

            # Actual payouts from JRA data
            # race_id_str already contains race_num (CC+YY+K+N+RR)
            payout_info = get_actual_payouts(payout_lookup, rid)
            (actual_trio_hit, actual_trio_return,
             actual_umaren_hit, actual_umaren_return,
             actual_wide_hits, actual_wide_return) = calc_actual_returns(
                payout_info, trio_bets, umaren_bets, wide_bets)

            if payout_info is not None:
                year_matched += 1
                matched_count += 1
            else:
                unmatched_count += 1

            year_total += 1
            if actual_trio_hit:
                year_trio_hits += 1

            all_results.append({
                'race_id': rid, 'year': test_year, 'race_num': race_num,
                'axes': axes,
                'has_payout': payout_info is not None,
                # Estimated
                'trio_hit': trio_hit, 'trio_return': trio_return_est,
                'umaren_hit': len(umaren_hits) > 0, 'umaren_return': umaren_return_est,
                'wide_hits': len(wide_hits), 'wide_return': wide_return_est,
                # Actual
                'actual_trio_hit': actual_trio_hit, 'actual_trio_return': actual_trio_return,
                'actual_umaren_hit': actual_umaren_hit, 'actual_umaren_return': actual_umaren_return,
                'actual_wide_hits': actual_wide_hits, 'actual_wide_return': actual_wide_return,
            })

        print(f"  Races: {year_total}, Matched payouts: {year_matched}, Trio hits: {year_trio_hits}")

    print(f"\n  Payout matching: {matched_count}/{matched_count+unmatched_count} "
          f"({matched_count/(matched_count+unmatched_count)*100:.1f}%)")

    # === Analysis ===
    print(f"\n{'=' * 70}")
    print(f"  [4] 条件別 実配当ROI分析 ({len(all_results)} races)")
    print(f"{'=' * 70}")

    # Filter to only matched races for actual ROI
    matched_results = [r for r in all_results if r['has_payout']]
    print(f"  配当データあり: {len(matched_results)} races")

    # Condition analysis
    cond_groups = defaultdict(list)
    for r in matched_results:
        cond_groups[r['axes']['cond_key']].append(r)

    cond_analysis = analyze_groups_actual(cond_groups, min_n=10)

    # Print comparison table
    print(f"\n  {'Cond':<6} {'N':>5} | {'trio的中':>7} {'trio実ROI':>9} {'trio推ROI':>9} | "
          f"{'uma的中':>7} {'uma実ROI':>9} | {'wide的中':>7} {'wide実ROI':>9} | {'Best':>5} {'推奨'}")
    print(f"  {'-' * 100}")

    for label in ['A', 'B', 'C', 'D', 'E', 'X']:
        if label not in cond_analysis:
            continue
        info = cond_analysis[label]
        t = info['bets']['trio']
        u = info['bets']['umaren']
        w = info['bets']['wide']
        et = info['estimated']['trio']
        rec = '○' if info['recommended'] else '×'
        print(f"  {label:<6} {info['n']:>5} | {t['hit_rate']:>5.1f}% {t['roi']:>8.1f}% {et['roi']:>8.1f}% | "
              f"{u['hit_rate']:>5.1f}% {u['roi']:>8.1f}% | {w['hit_rate']:>5.1f}% {w['roi']:>8.1f}% | "
              f"{info['best_bet']:>5} {rec}")

    # Year-by-year stability
    print(f"\n  --- 年別ROI安定性 (trio 7点) ---")
    print(f"  {'Year':<6} {'Cond':<6} {'N':>5} {'Hit%':>6} {'実ROI':>8} {'推ROI':>8}")
    print(f"  {'-' * 50}")

    for test_year in TEST_YEARS:
        year_results = [r for r in matched_results if r['year'] == test_year]
        for cond_key in ['A', 'B', 'C', 'D', 'E', 'X']:
            cond_races = [r for r in year_results if r['axes']['cond_key'] == cond_key]
            if len(cond_races) < 5:
                continue
            n = len(cond_races)
            trio_hits = sum(1 for r in cond_races if r['actual_trio_hit'])
            trio_pay = sum(r['actual_trio_return'] for r in cond_races)
            trio_est = sum(r['trio_return'] for r in cond_races)
            inv = n * 700
            actual_roi = trio_pay / inv * 100
            est_roi = trio_est / inv * 100
            print(f"  {test_year:<6} {cond_key:<6} {n:>5} {trio_hits/n*100:>5.1f}% {actual_roi:>7.1f}% {est_roi:>7.1f}%")

    # Multi-axis analysis
    all_axis_analysis = {}
    for axis_key in ['cond_key', 'dist', 'nh', 'cond', 'surface']:
        groups = defaultdict(list)
        for r in matched_results:
            groups[r['axes'][axis_key]].append(r)
        all_axis_analysis[axis_key] = analyze_groups_actual(groups, min_n=30)

    # Optimal bet determination
    print(f"\n  --- 条件別 最適券種判定 (実配当ベース) ---")
    optimal_bets = {}
    for label in ['A', 'B', 'C', 'D', 'E', 'X']:
        if label not in cond_analysis:
            continue
        info = cond_analysis[label]
        best = info['best_bet']
        best_roi = info['best_roi']
        optimal_bets[label] = {
            'bet_type': best,
            'roi': best_roi,
            'recommended': info['recommended'],
        }
        status = '★推奨' if best_roi >= 100 else '非推奨'
        print(f"  {label}: {best} ROI {best_roi:.1f}% {status}")

    # === Save results ===
    print(f"\n[5] 結果保存...")

    output = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'v9.3_leakfree (Pattern A)',
        'n_features': len(FEATURES_PATTERN_A),
        'test_years': TEST_YEARS,
        'total_races': len(all_results),
        'matched_races': len(matched_results),
        'match_rate': round(matched_count / (matched_count + unmatched_count) * 100, 1),
        'fold_aucs': {str(k): v for k, v in fold_aucs.items()},
        'avg_auc': round(np.mean(list(fold_aucs.values())), 4),
        'conditions': {},
        'year_stability': {},
        'optimal_bets': optimal_bets,
    }

    # Conditions detail
    for label in ['A', 'B', 'C', 'D', 'E', 'X']:
        if label not in cond_analysis:
            continue
        info = cond_analysis[label]
        output['conditions'][label] = {
            'n': info['n'],
            'best_bet': info['best_bet'],
            'best_roi_actual': info['best_roi'],
            'recommended': info['recommended'],
            'actual_roi': {bt: info['bets'][bt] for bt in ['trio', 'umaren', 'wide']},
            'estimated_roi': {bt: info['estimated'][bt] for bt in ['trio', 'umaren', 'wide']},
        }

    # Year stability
    for test_year in TEST_YEARS:
        year_results = [r for r in matched_results if r['year'] == test_year]
        year_data = {}
        for cond_key in ['A', 'B', 'C', 'D', 'E', 'X']:
            cond_races = [r for r in year_results if r['axes']['cond_key'] == cond_key]
            if len(cond_races) < 5:
                continue
            n = len(cond_races)
            inv = n * 700
            year_data[cond_key] = {
                'n': n,
                'trio': {
                    'hits': sum(1 for r in cond_races if r['actual_trio_hit']),
                    'hit_rate': round(sum(1 for r in cond_races if r['actual_trio_hit']) / n * 100, 1),
                    'actual_roi': round(sum(r['actual_trio_return'] for r in cond_races) / inv * 100, 1),
                    'estimated_roi': round(sum(r['trio_return'] for r in cond_races) / inv * 100, 1),
                },
            }
        output['year_stability'][str(test_year)] = year_data

    out_path = os.path.join(BASE_DIR, 'data', 'actual_roi_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {out_path}")

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  最終サマリー")
    print(f"{'=' * 70}")
    print(f"  Average AUC: {output['avg_auc']:.4f}")
    print(f"  配当マッチ率: {output['match_rate']:.1f}%")
    print(f"\n  条件別 実配当ROI:")
    for label in ['A', 'B', 'C', 'D', 'E', 'X']:
        if label in cond_analysis:
            info = cond_analysis[label]
            t = info['bets']['trio']
            est = info['estimated']['trio']
            status = '★' if t['roi'] >= 100 else '×'
            print(f"    {label}: trio実ROI {t['roi']:>7.1f}% (推定{est['roi']:>7.1f}%) "
                  f"的中{t['hit_rate']:>5.1f}% N={info['n']} {status}")

    print(f"\n  完了!")
    return output


if __name__ == '__main__':
    main()
