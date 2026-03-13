#!/usr/bin/env python
"""■7 年別AUC・ROI推移
2020-2025の各年でPattern A AUC + 条件別実配当ROI + 全体ROI。
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
from train_v92_leakfree import FEATURES_PATTERN_A

from backtest_central_leakfree import (
    classify_condition, calc_trio_bets, calc_umaren_bets,
    encode_sires_fold, train_lgb_fold,
)

from calc_actual_roi import (
    load_payouts, calc_actual_returns, parse_trio_nums, parse_umaren_nums,
    parse_wide_data,
)

TEST_YEARS = list(range(2020, 2026))


def main():
    print("=" * 60)
    print("  ■7 年別AUC・ROI推移")
    print("=" * 60)

    # Data prep
    print("\n[1] データ準備...")
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

    features = list(FEATURES_PATTERN_A)
    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    payout_lookup = load_payouts()

    # Walk-forward
    print("\n[2] ウォークフォワード...")
    yearly_data = {}

    for test_year in TEST_YEARS:
        train_mask = (df['year_full'] >= 2010) & (df['year_full'] < test_year)
        test_mask = df['year_full'] == test_year
        if test_mask.sum() < 100:
            continue

        print(f"\n  --- {test_year} ---")
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
        print(f"  AUC: {test_auc:.4f}")

        # Race-level results
        year_results = defaultdict(list)

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

            year_results[cond].append({
                'actual_trio_hit': actual_trio_hit,
                'actual_trio_return': actual_trio_return,
                'actual_umaren_hit': actual_umaren_hit,
                'actual_umaren_return': actual_umaren_return,
                'actual_wide_return': actual_wide_return,
                'has_payout': payout_info is not None,
            })

        # Yearly summary
        year_data = {
            'auc': round(test_auc, 4),
            'conditions': {},
            'overall': {},
        }

        total_races = 0
        total_trio_pay = 0
        total_uma_pay = 0

        for cond_key in ['A', 'B', 'C', 'D', 'E', 'X']:
            races = [r for r in year_results[cond_key] if r['has_payout']]
            n = len(races)
            if n < 5:
                year_data['conditions'][cond_key] = {'n': n, 'note': 'insufficient data'}
                continue

            inv = n * 700
            trio_hits = sum(1 for r in races if r['actual_trio_hit'])
            trio_pay = sum(r['actual_trio_return'] for r in races)
            uma_hits = sum(1 for r in races if r['actual_umaren_hit'])
            uma_pay = sum(r['actual_umaren_return'] * 3.5 for r in races)

            total_races += n
            total_trio_pay += trio_pay
            total_uma_pay += uma_pay

            # Use appropriate bet type per condition
            if cond_key == 'E':
                best_roi = uma_pay / inv * 100
                best_hit = uma_hits / n * 100
                best_type = 'umaren'
            else:
                best_roi = trio_pay / inv * 100
                best_hit = trio_hits / n * 100
                best_type = 'trio'

            year_data['conditions'][cond_key] = {
                'n': n,
                'bet_type': best_type,
                'hit_rate': round(best_hit, 1),
                'actual_roi': round(best_roi, 1),
                'trio_roi': round(trio_pay / inv * 100, 1),
                'umaren_roi': round(uma_pay / inv * 100, 1),
            }

        total_inv = total_races * 700
        year_data['overall'] = {
            'total_races': total_races,
            'trio_roi': round(total_trio_pay / total_inv * 100, 1) if total_inv > 0 else 0,
            'umaren_roi': round(total_uma_pay / total_inv * 100, 1) if total_inv > 0 else 0,
        }

        yearly_data[str(test_year)] = year_data
        print(f"  Overall trio ROI: {year_data['overall']['trio_roi']:.1f}%, Races: {total_races}")

    # Anomaly detection
    print(f"\n{'=' * 60}")
    print(f"  [3] 年別推移サマリー")
    print(f"{'=' * 60}")

    warnings_list = []
    aucs = [yearly_data[str(y)]['auc'] for y in TEST_YEARS if str(y) in yearly_data]
    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    print(f"\n  年別AUC:")
    for year in TEST_YEARS:
        key = str(year)
        if key not in yearly_data:
            continue
        auc = yearly_data[key]['auc']
        flag = ''
        if auc < avg_auc - 2 * std_auc:
            flag = ' !! 異常に低い'
            warnings_list.append(f'{year} AUC={auc:.4f} is significantly below average')
        elif auc > avg_auc + 2 * std_auc:
            flag = ' !! 異常に高い'
            warnings_list.append(f'{year} AUC={auc:.4f} is significantly above average')
        print(f"  {year}: {auc:.4f}{flag}")
    print(f"  平均: {avg_auc:.4f} (std: {std_auc:.4f})")

    print(f"\n  年別 条件別ROI:")
    print(f"  {'Year':<6} {'AUC':>6} | {'A':>7} {'B':>7} {'C':>7} {'D':>7} {'E':>7} {'X':>7} | {'Total':>7}")
    print(f"  {'-' * 75}")

    for year in TEST_YEARS:
        key = str(year)
        if key not in yearly_data:
            continue
        yd = yearly_data[key]
        line = f"  {year:<6} {yd['auc']:>6.4f} |"
        for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
            cd = yd['conditions'].get(cond, {})
            roi = cd.get('actual_roi', cd.get('trio_roi', 0))
            n = cd.get('n', 0)
            if n < 5:
                line += f" {'N/A':>7}"
            else:
                flag = '*' if roi < 50 else ''
                line += f" {roi:>6.1f}%{flag}"
                if roi < 50 and n >= 20:
                    warnings_list.append(f'{year} cond {cond}: ROI={roi:.1f}% (n={n})')
        line += f" | {yd['overall']['trio_roi']:>6.1f}%"
        print(line)

    if warnings_list:
        print(f"\n  !! 警告:")
        for w in warnings_list:
            print(f"    - {w}")
    else:
        print(f"\n  OK 異常値なし")

    # Save
    output = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_years': TEST_YEARS,
        'yearly_data': yearly_data,
        'summary': {
            'avg_auc': round(avg_auc, 4),
            'std_auc': round(std_auc, 4),
            'min_auc': round(min(aucs), 4),
            'max_auc': round(max(aucs), 4),
            'auc_trend': 'stable' if std_auc < 0.005 else ('improving' if aucs[-1] > aucs[0] else 'declining'),
        },
        'warnings': warnings_list,
    }

    out_path = os.path.join(BASE_DIR, 'data', 'yearly_performance.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  保存: {out_path}")

    return output


if __name__ == '__main__':
    main()
