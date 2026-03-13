#!/usr/bin/env python
"""■3 EVフィルタ導入分析 + ■9 EV>1.0フィルタのROI変化
各レースのEV = 予測確率 × 想定配当 を計算。
EV > 1.0 のレースのみ購入する場合のROI変化をシミュレーション。
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import warnings
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
    check_bets, estimate_payouts, encode_sires_fold, train_lgb_fold, get_axes,
)

from calc_actual_roi import load_payouts, calc_actual_returns

TEST_YEARS = list(range(2020, 2026))


def compute_race_ev(race_df, ranking, trio_bets, umaren_bets):
    """レースのEV（期待値）を計算。

    EV = Σ(各買い目の的中確率 × 配当) / 投資額

    三連複の場合:
    - AI予測スコアからTOP3に入る確率を推定
    - 7点買いの各組み合わせの的中確率を計算
    - 想定配当 = オッズから推定
    """
    scores = race_df.sort_values('pred', ascending=False)
    total_score = scores['pred'].sum()
    if total_score <= 0:
        return 0.0, {}

    # 各馬のスコアを確率に変換（softmax的）
    probs = {}
    for _, row in scores.iterrows():
        uma = int(row['umaban'])
        probs[uma] = row['pred'] / total_score

    # 三連複の各買い目のEV計算
    # P(combo) ≈ P(top1 in top3) × P(top2 in top3) × P(top3 in top3) × correction
    # 簡易的にスコア積で近似
    trio_ev = 0
    odds_map = {}
    for _, row in race_df.iterrows():
        odds_map[int(row['umaban'])] = float(row['tansho_odds'])

    for bet in trio_bets:
        if len(bet) != 3:
            continue
        a, b, c = bet
        p_a = probs.get(a, 0)
        p_b = probs.get(b, 0)
        p_c = probs.get(c, 0)
        # TOP3に入る確率（独立近似は不正確だが簡易）
        p_combo = p_a * p_b * p_c * 6  # 3! for ordering
        p_combo = min(p_combo * 20, 0.3)  # calibration cap

        # 想定三連複配当 = o1 * o2 * o3 * 20（推定式）
        o_a = odds_map.get(a, 10)
        o_b = odds_map.get(b, 10)
        o_c = odds_map.get(c, 10)
        est_payout = max(100, o_a * o_b * o_c * 20)

        trio_ev += p_combo * est_payout

    investment = 700  # 7点 × 100円
    ev_ratio = trio_ev / investment if investment > 0 else 0

    # 馬連の場合
    umaren_ev = 0
    for bet in umaren_bets:
        if len(bet) != 2:
            continue
        a, b = bet
        p_a = probs.get(a, 0)
        p_b = probs.get(b, 0)
        p_combo = p_a * p_b * 2 * 15  # calibration
        p_combo = min(p_combo, 0.5)
        o_a = odds_map.get(a, 10)
        o_b = odds_map.get(b, 10)
        est_payout = max(100, o_a * o_b * 50)
        umaren_ev += p_combo * est_payout

    umaren_inv = 700
    umaren_ev_ratio = umaren_ev / umaren_inv if umaren_inv > 0 else 0

    return ev_ratio, {
        'trio_ev': round(ev_ratio, 3),
        'umaren_ev': round(umaren_ev_ratio, 3),
    }


def main():
    print("=" * 60)
    print("  ■3/■9 EVフィルタ分析")
    print("=" * 60)

    # Load data
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

    # Load actual payouts
    print("\n[2] 配当データ読み込み...")
    payout_lookup = load_payouts()

    # Walk-forward with EV calculation
    print("\n[3] ウォークフォワード + EV計算...")
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

            # EV計算
            ev_ratio, ev_detail = compute_race_ev(race_df, ranking, trio_bets, umaren_bets)

            # 的中チェック
            trio_hit, umaren_hits, wide_hits = check_bets(actual_top3, trio_bets, umaren_bets, wide_bets)
            trio_pay_est, umaren_pay_est, _ = estimate_payouts(actual_top3, race_df)

            # 実配当
            payout_info = payout_lookup.get(rid)
            from calc_actual_roi import calc_actual_returns as _calc
            (actual_trio_hit, actual_trio_return,
             actual_umaren_hit, actual_umaren_return,
             actual_wide_hits, actual_wide_return) = _calc(
                payout_info, trio_bets, umaren_bets, wide_bets)

            all_results.append({
                'race_id': rid, 'year': test_year, 'cond': cond,
                'ev_trio': ev_detail.get('trio_ev', 0),
                'ev_umaren': ev_detail.get('umaren_ev', 0),
                'trio_hit': trio_hit,
                'trio_return_est': trio_pay_est if trio_hit else 0,
                'umaren_hit': len(umaren_hits) > 0,
                'umaren_return_est': umaren_pay_est * 3.5 * len(umaren_hits) if umaren_hits else 0,
                'actual_trio_hit': actual_trio_hit,
                'actual_trio_return': actual_trio_return,
                'actual_umaren_hit': actual_umaren_hit,
                'actual_umaren_return': actual_umaren_return,
                'has_payout': payout_info is not None,
            })

    # === EV Filter Analysis ===
    print(f"\n{'=' * 60}")
    print(f"  [4] EVフィルタ分析 ({len(all_results)} races)")
    print(f"{'=' * 60}")

    matched = [r for r in all_results if r['has_payout']]

    ev_thresholds = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    ev_analysis = {}

    print(f"\n  {'EV閾値':>8} | {'N':>6} | {'trio的中率':>9} | {'trio実ROI':>9} | {'uma実ROI':>9}")
    print(f"  {'-' * 60}")

    for thresh in ev_thresholds:
        for cond_key in ['ALL', 'A', 'B', 'C', 'D', 'E', 'X']:
            if cond_key == 'ALL':
                races = [r for r in matched if r['ev_trio'] >= thresh]
            else:
                races = [r for r in matched if r['cond'] == cond_key and r['ev_trio'] >= thresh]

            n = len(races)
            if n < 10:
                continue

            trio_hits = sum(1 for r in races if r['actual_trio_hit'])
            trio_pay = sum(r['actual_trio_return'] for r in races)
            uma_hits = sum(1 for r in races if r['actual_umaren_hit'])
            uma_pay = sum(r['actual_umaren_return'] * 3.5 for r in races)
            inv = n * 700

            trio_roi = trio_pay / inv * 100
            uma_roi = uma_pay / inv * 100
            hit_rate = trio_hits / n * 100

            key = f"ev>={thresh}_{cond_key}"
            ev_analysis[key] = {
                'threshold': thresh, 'condition': cond_key,
                'n': n, 'trio_hit_rate': round(hit_rate, 1),
                'trio_actual_roi': round(trio_roi, 1),
                'umaren_actual_roi': round(uma_roi, 1),
            }

            if cond_key == 'ALL':
                print(f"  EV>={thresh:<4} | {n:>6} | {hit_rate:>8.1f}% | {trio_roi:>8.1f}% | {uma_roi:>8.1f}%")

    # Condition-level EV analysis
    print(f"\n  --- 条件別 EVフィルタ効果 (EV>=1.0 vs 全件) ---")
    print(f"  {'Cond':<6} | {'全件N':>6} {'全件ROI':>8} | {'EV>=1 N':>8} {'EV>=1 ROI':>10}")
    print(f"  {'-' * 55}")

    cond_ev_comparison = {}
    for cond_key in ['A', 'B', 'C', 'D', 'E', 'X']:
        all_cond = [r for r in matched if r['cond'] == cond_key]
        ev_cond = [r for r in matched if r['cond'] == cond_key and r['ev_trio'] >= 1.0]

        if len(all_cond) < 10:
            continue

        bt = 'umaren' if cond_key == 'E' else 'trio'
        actual_key = f'actual_{bt}_hit'
        return_key = f'actual_{bt}_return'

        n_all = len(all_cond)
        pay_all = sum(r[return_key] * (3.5 if bt == 'umaren' else 1) for r in all_cond)
        roi_all = pay_all / (n_all * 700) * 100

        n_ev = len(ev_cond)
        if n_ev > 0:
            pay_ev = sum(r[return_key] * (3.5 if bt == 'umaren' else 1) for r in ev_cond)
            roi_ev = pay_ev / (n_ev * 700) * 100
        else:
            roi_ev = 0

        cond_ev_comparison[cond_key] = {
            'all_n': n_all, 'all_roi': round(roi_all, 1),
            'ev_filtered_n': n_ev, 'ev_filtered_roi': round(roi_ev, 1),
            'improvement': round(roi_ev - roi_all, 1) if n_ev > 0 else None,
        }

        ev_n_str = f"{n_ev}" if n_ev > 0 else "N/A"
        ev_roi_str = f"{roi_ev:.1f}%" if n_ev > 0 else "N/A"
        print(f"  {cond_key:<6} | {n_all:>6} {roi_all:>7.1f}% | {ev_n_str:>8} {ev_roi_str:>10}")

    # Save
    output = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_races': len(all_results),
        'matched_races': len(matched),
        'ev_threshold_analysis': ev_analysis,
        'condition_ev_comparison': cond_ev_comparison,
        'ev_distribution': {
            'mean': round(np.mean([r['ev_trio'] for r in all_results]), 3),
            'median': round(np.median([r['ev_trio'] for r in all_results]), 3),
            'p25': round(np.percentile([r['ev_trio'] for r in all_results], 25), 3),
            'p75': round(np.percentile([r['ev_trio'] for r in all_results], 75), 3),
            'pct_above_1': round(sum(1 for r in all_results if r['ev_trio'] >= 1.0) / len(all_results) * 100, 1),
        },
        'recommendation': 'EVフィルタの効果は条件や閾値によって異なる。EV>=1.0でフィルタすると投票対象レース数が減少し、ROIが改善する場合がある。ただし推定EVはモデル予測精度に依存するため、過信は禁物。',
    }

    out_path = os.path.join(BASE_DIR, 'data', 'ev_filter_analysis.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  保存: {out_path}")

    return output


if __name__ == '__main__':
    main()
