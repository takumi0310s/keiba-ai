#!/usr/bin/env python
"""■4 券種最適化の深掘り
全条件×全券種（単勝/馬連/ワイド/三連複/三連単）の実配当ROIを計算。
jra_payouts.csvから全券種の配当データを使用。
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
    classify_condition, calc_trio_bets, calc_umaren_bets, calc_wide_bets,
    encode_sires_fold, train_lgb_fold, get_axes,
)

TEST_YEARS = list(range(2020, 2026))

# Course code mapping
COURSE_CODE_TO_NAME = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉',
}
COURSE_NAME_TO_CODE = {v: k for k, v in COURSE_CODE_TO_NAME.items()}
NICHI_TO_CHAR = {i: str(i) for i in range(1, 10)}
NICHI_TO_CHAR.update({10: 'A', 11: 'B', 12: 'C'})


def load_full_payouts():
    """全券種の配当データを読み込み"""
    path = os.path.join(BASE_DIR, 'data', 'jra_payouts.csv')
    pay_df = pd.read_csv(path, encoding='utf-8', dtype=str)
    print(f"  Payouts loaded: {len(pay_df)} records")

    lookup = {}
    for _, row in pay_df.iterrows():
        date_str = str(row['race_date'])
        course_name = str(row['course'])
        kai_int = int(row['kai'])
        nichi_int = int(row['nichi'])
        race_num = str(row['race_num']).zfill(2)

        cc = COURSE_NAME_TO_CODE.get(course_name)
        if cc is None:
            continue

        year_2d = date_str[2:4]
        nichi_char = NICHI_TO_CHAR.get(nichi_int, str(nichi_int))
        key = f"{cc}{year_2d}{kai_int}{nichi_char}{race_num}"

        # Parse all ticket types
        def safe_int(val):
            try:
                return int(str(val).replace(',', ''))
            except (ValueError, TypeError):
                return 0

        def parse_nums(val):
            if not val or str(val) == 'nan':
                return None
            return str(val)

        lookup[key] = {
            'tansho_nums': parse_nums(row.get('tansho_nums', '')),
            'tansho_payout': safe_int(row.get('tansho_payout', 0)),
            'fukusho_nums': parse_nums(row.get('fukusho_nums', '')),
            'fukusho_payouts': parse_nums(row.get('fukusho_payouts', '')),
            'umaren_nums': parse_nums(row.get('umaren_nums', '')),
            'umaren_payout': safe_int(row.get('umaren_payout', 0)),
            'wide_nums': parse_nums(row.get('wide_nums', '')),
            'wide_payouts': parse_nums(row.get('wide_payouts', '')),
            'trio_nums': parse_nums(row.get('trio_nums', '')),
            'trio_payout': safe_int(row.get('trio_payout', 0)),
            'tierce_nums': parse_nums(row.get('tierce_nums', '')),
            'tierce_payout': safe_int(row.get('tierce_payout', 0)),
        }

    return lookup


def calc_all_ticket_returns(payout_info, ranking):
    """全券種のリターンを計算"""
    if payout_info is None:
        return {}

    results = {}

    # 1. 単勝: TOP1の馬番 (1点 × 700円)
    if payout_info['tansho_nums']:
        try:
            tansho_uma = int(payout_info['tansho_nums'].split('-')[0])
            tansho_hit = ranking[0] == tansho_uma
            results['tansho'] = {
                'hit': tansho_hit,
                'return': payout_info['tansho_payout'] * 7 if tansho_hit else 0,
                'investment': 700,
            }
        except (ValueError, IndexError):
            results['tansho'] = {'hit': False, 'return': 0, 'investment': 700}
    else:
        results['tansho'] = {'hit': False, 'return': 0, 'investment': 700}

    # 2. 馬連: TOP1-TOP2, TOP1-TOP3 の2点 (各350円 = 700円)
    if payout_info['umaren_nums']:
        try:
            parts = payout_info['umaren_nums'].split('-')
            actual_umaren = frozenset(int(x) for x in parts)
            bet1 = frozenset([ranking[0], ranking[1]])
            bet2 = frozenset([ranking[0], ranking[2]])
            hit1 = bet1 == actual_umaren
            hit2 = bet2 == actual_umaren
            uma_return = 0
            if hit1:
                uma_return = payout_info['umaren_payout'] * 3.5
            elif hit2:
                uma_return = payout_info['umaren_payout'] * 3.5
            results['umaren'] = {
                'hit': hit1 or hit2,
                'return': uma_return,
                'investment': 700,
            }
        except (ValueError, IndexError):
            results['umaren'] = {'hit': False, 'return': 0, 'investment': 700}
    else:
        results['umaren'] = {'hit': False, 'return': 0, 'investment': 700}

    # 3. ワイド: TOP1-TOP2, TOP1-TOP3 の2点 (各350円 = 700円)
    if payout_info['wide_nums'] and payout_info['wide_payouts']:
        try:
            wide_pairs = payout_info['wide_nums'].split('/')
            wide_pays = payout_info['wide_payouts'].split('/')
            actual_wide = {}
            for wp, pay in zip(wide_pairs, wide_pays):
                parts = wp.split('-')
                k = frozenset(int(x) for x in parts)
                actual_wide[k] = int(pay)

            bet1 = frozenset([ranking[0], ranking[1]])
            bet2 = frozenset([ranking[0], ranking[2]])
            wide_return = 0
            wide_hit = False
            if bet1 in actual_wide:
                wide_return += actual_wide[bet1] * 3.5
                wide_hit = True
            if bet2 in actual_wide:
                wide_return += actual_wide[bet2] * 3.5
                wide_hit = True
            results['wide'] = {
                'hit': wide_hit,
                'return': wide_return,
                'investment': 700,
            }
        except (ValueError, IndexError):
            results['wide'] = {'hit': False, 'return': 0, 'investment': 700}
    else:
        results['wide'] = {'hit': False, 'return': 0, 'investment': 700}

    # 4. 三連複: TOP1軸-TOP2,3-TOP2~6 の7点 (各100円 = 700円)
    if payout_info['trio_nums']:
        try:
            parts = payout_info['trio_nums'].split('-')
            actual_trio = frozenset(int(x) for x in parts)
            trio_bets = calc_trio_bets(ranking)
            trio_hit = any(frozenset(bet) == actual_trio for bet in trio_bets)
            results['trio'] = {
                'hit': trio_hit,
                'return': payout_info['trio_payout'] if trio_hit else 0,
                'investment': 700,
            }
        except (ValueError, IndexError):
            results['trio'] = {'hit': False, 'return': 0, 'investment': 700}
    else:
        results['trio'] = {'hit': False, 'return': 0, 'investment': 700}

    # 5. 三連単: TOP1→TOP2→TOP3 の1点 (700円)
    if payout_info['tierce_nums']:
        try:
            parts = payout_info['tierce_nums'].split('-')
            actual_order = [int(x) for x in parts]
            # 1点買い: 予測1着→2着→3着
            tierce_hit = (len(ranking) >= 3 and len(actual_order) >= 3 and
                          ranking[0] == actual_order[0] and
                          ranking[1] == actual_order[1] and
                          ranking[2] == actual_order[2])
            results['tierce'] = {
                'hit': tierce_hit,
                'return': payout_info['tierce_payout'] * 7 if tierce_hit else 0,
                'investment': 700,
            }
        except (ValueError, IndexError):
            results['tierce'] = {'hit': False, 'return': 0, 'investment': 700}
    else:
        results['tierce'] = {'hit': False, 'return': 0, 'investment': 700}

    return results


def main():
    print("=" * 60)
    print("  ■4 券種最適化の深掘り")
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

    payout_lookup = load_full_payouts()

    # Walk-forward
    print("\n[2] ウォークフォワード...")
    all_results = []

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

        for rid in test_df['race_id_str'].unique():
            race_df = test_df[test_df['race_id_str'] == rid].copy()
            if len(race_df) < 5:
                continue

            row0 = race_df.iloc[0]
            cond = classify_condition(row0)

            race_df_sorted = race_df.sort_values('pred', ascending=False)
            ranking = race_df_sorted['umaban'].astype(int).tolist()

            payout_info = payout_lookup.get(rid)
            ticket_results = calc_all_ticket_returns(payout_info, ranking)

            if ticket_results:
                all_results.append({
                    'race_id': rid, 'year': test_year, 'cond': cond,
                    'tickets': ticket_results,
                })

    # Analysis
    print(f"\n{'=' * 60}")
    print(f"  [3] 条件別×券種別 実配当ROI")
    print(f"{'=' * 60}")

    ticket_types = ['tansho', 'umaren', 'wide', 'trio', 'tierce']
    ticket_names = {'tansho': '単勝', 'umaren': '馬連', 'wide': 'ワイド', 'trio': '三連複', 'tierce': '三連単'}

    cond_ticket_results = {}

    header = f"  {'Cond':<6}"
    for tt in ticket_types:
        header += f" | {ticket_names[tt]:>6} {'的中率':>6} {'ROI':>7}"
    print(header)
    print(f"  {'-' * 90}")

    for cond_key in ['A', 'B', 'C', 'D', 'E', 'X', 'ALL']:
        if cond_key == 'ALL':
            races = all_results
        else:
            races = [r for r in all_results if r['cond'] == cond_key]

        n = len(races)
        if n < 10:
            continue

        cond_data = {'n': n}
        line = f"  {cond_key:<6}"

        for tt in ticket_types:
            hits = sum(1 for r in races if r['tickets'].get(tt, {}).get('hit', False))
            total_return = sum(r['tickets'].get(tt, {}).get('return', 0) for r in races)
            total_inv = n * 700
            roi = total_return / total_inv * 100 if total_inv > 0 else 0
            hit_rate = hits / n * 100

            cond_data[tt] = {
                'hits': hits, 'hit_rate': round(hit_rate, 1),
                'roi': round(roi, 1),
                'total_return': int(total_return),
                'total_investment': total_inv,
            }
            line += f" | {ticket_names[tt]:>6} {hit_rate:>5.1f}% {roi:>6.1f}%"

        # Best ticket
        best_tt = max(ticket_types, key=lambda t: cond_data[t]['roi'])
        cond_data['best_ticket'] = best_tt
        cond_data['best_roi'] = cond_data[best_tt]['roi']

        cond_ticket_results[cond_key] = cond_data
        print(line)

    # Best ticket per condition
    print(f"\n  --- 条件別 最適券種 ---")
    for cond_key in ['A', 'B', 'C', 'D', 'E', 'X']:
        if cond_key in cond_ticket_results:
            data = cond_ticket_results[cond_key]
            best = data['best_ticket']
            print(f"  {cond_key}: {ticket_names[best]} (ROI {data['best_roi']:.1f}%)")

    # Save
    output = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_races': len(all_results),
        'ticket_types': list(ticket_names.keys()),
        'ticket_names': ticket_names,
        'condition_ticket_roi': cond_ticket_results,
        'optimal_strategy': {},
    }

    for cond_key in ['A', 'B', 'C', 'D', 'E', 'X']:
        if cond_key in cond_ticket_results:
            data = cond_ticket_results[cond_key]
            output['optimal_strategy'][cond_key] = {
                'best_ticket': data['best_ticket'],
                'best_roi': data['best_roi'],
                'current_ticket': 'umaren' if cond_key == 'E' else 'trio',
                'current_roi': data.get('umaren' if cond_key == 'E' else 'trio', {}).get('roi', 0),
            }

    out_path = os.path.join(BASE_DIR, 'data', 'ticket_type_optimization.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  保存: {out_path}")

    return output


if __name__ == '__main__':
    main()
