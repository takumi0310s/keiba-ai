#!/usr/bin/env python
"""Phase 10: 市場依存性テスト
オッズ関連特徴量を完全除外したモデルを再学習し、AUC/ROIを比較。
Pattern Aは既にodds_logを除外済みだが、prev_odds_logは残っている。
prev_odds_logも除外して真の能力予測力を検証。
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

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, BASE_DIR)

from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features, compute_distance_aptitude,
    compute_frame_advantage,
)
from train_v92_leakfree import FEATURES_PATTERN_A, LEAK_FEATURES_A

from backtest_central_leakfree import (
    classify_condition, calc_trio_bets, calc_umaren_bets,
    encode_sires_fold, train_lgb_fold, get_axes,
)

from calc_actual_roi import load_payouts, calc_actual_returns

TEST_YEARS = list(range(2020, 2026))

# オッズ関連特徴量（Pattern Aに含まれるもの）
ODDS_FEATURES = {'prev_odds_log'}
# pop_rank, odds_log はPattern Aに含まれない（既に除外済み）


def run_wf_with_features(df, features, payout_lookup, label):
    """指定された特徴量でWFバックテストを実行"""
    fold_aucs = {}
    all_results = []

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
        valid_cutoff = dates.quantile(0.85)
        tr_idx = dates < valid_cutoff
        va_idx = dates >= valid_cutoff

        model = train_lgb_fold(
            train_df.loc[tr_idx, features].values, y_train[tr_idx.values],
            train_df.loc[va_idx, features].values, y_train[va_idx.values],
            features
        )

        test_df = df_fold[test_mask].copy()
        test_df['pred'] = model.predict(test_df[features].values)
        test_auc = roc_auc_score(test_df['target'].values, test_df['pred'])
        fold_aucs[test_year] = test_auc

        for rid in test_df['race_id_str'].unique():
            race_df = test_df[test_df['race_id_str'] == rid].copy()
            if len(race_df) < 5:
                continue

            row0 = race_df.iloc[0]
            cond = classify_condition(row0)

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
            wide_bets = [[ranking[0], ranking[1]], [ranking[0], ranking[2]]]

            payout_info = payout_lookup.get(rid)
            (actual_trio_hit, actual_trio_return,
             actual_umaren_hit, actual_umaren_return,
             actual_wide_hits, actual_wide_return) = calc_actual_returns(
                payout_info, trio_bets, umaren_bets, wide_bets)

            all_results.append({
                'cond': cond, 'year': test_year,
                'actual_trio_hit': actual_trio_hit,
                'actual_trio_return': actual_trio_return,
                'actual_umaren_hit': actual_umaren_hit,
                'actual_umaren_return': actual_umaren_return,
                'has_payout': payout_info is not None,
            })

    avg_auc = np.mean(list(fold_aucs.values()))
    print(f"  [{label}] Avg AUC: {avg_auc:.4f}, Folds: {fold_aucs}")

    # Condition ROI
    matched = [r for r in all_results if r['has_payout']]
    cond_roi = {}
    total_inv = 0
    total_pay = 0

    for cond_key in ['A', 'B', 'C', 'D', 'E', 'X']:
        races = [r for r in matched if r['cond'] == cond_key]
        n = len(races)
        if n < 10:
            continue

        bt = 'umaren' if cond_key == 'E' else 'trio'
        hits = sum(1 for r in races if r[f'actual_{bt}_hit'])
        inv = n * 700
        pay = sum(r[f'actual_{bt}_return'] * (3.5 if bt == 'umaren' else 1) for r in races)

        total_inv += inv
        total_pay += pay

        cond_roi[cond_key] = {
            'n': n, 'bet_type': bt,
            'hit_rate': round(hits / n * 100, 1),
            'roi': round(pay / inv * 100, 1),
        }

    overall_roi = total_pay / total_inv * 100 if total_inv > 0 else 0

    return {
        'label': label,
        'n_features': len(features),
        'avg_auc': round(avg_auc, 4),
        'fold_aucs': {str(k): round(v, 4) for k, v in fold_aucs.items()},
        'overall_roi': round(overall_roi, 1),
        'condition_roi': cond_roi,
        'n_races': len(matched),
    }


def main():
    print("=" * 60)
    print("  Phase 10: Market Dependency Test")
    print("=" * 60)

    # Data prep
    print("\n[1] Data preparation...")
    df = load_data()
    lap_df = load_lap_data()
    if lap_df is not None:
        df = df.merge(lap_df, on='race_id_str', how='left')

    df = encode_categoricals(df)
    df, _, _ = encode_sires(df)
    tt_data = load_training_times()
    df = merge_training_features(df, tt_data)
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)
    df = compute_lag_features(df)
    df = build_features(df)
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)

    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()

    for f in FEATURES_PATTERN_A:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    payout_lookup = load_payouts()

    # === Test 1: Pattern A (baseline, includes prev_odds_log) ===
    print("\n[2] Baseline: Pattern A (with prev_odds_log)...")
    features_baseline = list(FEATURES_PATTERN_A)
    result_baseline = run_wf_with_features(df, features_baseline, payout_lookup, "Pattern A (baseline)")

    # === Test 2: No odds features at all ===
    print("\n[3] No-odds: Pattern A minus prev_odds_log...")
    features_no_odds = [f for f in FEATURES_PATTERN_A if f not in ODDS_FEATURES]
    result_no_odds = run_wf_with_features(df, features_no_odds, payout_lookup, "No-odds")

    # === Comparison ===
    auc_diff = result_no_odds['avg_auc'] - result_baseline['avg_auc']
    roi_diff = result_no_odds['overall_roi'] - result_baseline['overall_roi']

    # Dependency judgment
    if abs(auc_diff) < 0.02 and result_no_odds['overall_roi'] > 100:
        dependency = "LOW"
        judgment = "True ability prediction model"
        detail = "AUC drop < 0.02 and ROI maintained > 100%. Model relies on genuine horse ability, not market information."
    elif abs(auc_diff) < 0.02:
        dependency = "LOW-MEDIUM"
        judgment = "Mostly ability-based but ROI affected"
        detail = "AUC stable but ROI dropped. Odds info helps with bet sizing but not ranking."
    else:
        dependency = "HIGH"
        judgment = "Market-dependent model"
        detail = "Significant AUC drop indicates model relies heavily on market odds information."

    print(f"\n{'=' * 60}")
    print(f"  Market Dependency Results")
    print(f"{'=' * 60}")
    print(f"  Baseline AUC:  {result_baseline['avg_auc']:.4f} ({result_baseline['n_features']} features)")
    print(f"  No-odds AUC:   {result_no_odds['avg_auc']:.4f} ({result_no_odds['n_features']} features)")
    print(f"  AUC diff:      {auc_diff:+.4f}")
    print(f"  Baseline ROI:  {result_baseline['overall_roi']:.1f}%")
    print(f"  No-odds ROI:   {result_no_odds['overall_roi']:.1f}%")
    print(f"  ROI diff:      {roi_diff:+.1f}%")
    print(f"  Dependency:    {dependency}")
    print(f"  Judgment:      {judgment}")

    print(f"\n  Condition ROI comparison:")
    print(f"  {'Cond':<6} {'Baseline':>10} {'No-odds':>10} {'Diff':>8}")
    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        b_roi = result_baseline['condition_roi'].get(cond, {}).get('roi', 0)
        n_roi = result_no_odds['condition_roi'].get(cond, {}).get('roi', 0)
        diff = n_roi - b_roi
        print(f"  {cond:<6} {b_roi:>9.1f}% {n_roi:>9.1f}% {diff:>+7.1f}%")

    # Save
    output = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'removed_features': sorted(ODDS_FEATURES),
        'note': 'Pattern A already excludes odds_log. This test additionally removes prev_odds_log.',
        'baseline': result_baseline,
        'no_odds': result_no_odds,
        'comparison': {
            'auc_diff': round(auc_diff, 4),
            'roi_diff': round(roi_diff, 1),
            'dependency_level': dependency,
            'judgment': judgment,
            'detail': detail,
        },
    }

    out_path = os.path.join(BASE_DIR, 'data', 'market_dependency_test.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")

    return output


if __name__ == '__main__':
    main()
