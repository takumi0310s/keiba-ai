#!/usr/bin/env python
"""■2 目的変数の検証
3パターンの目的変数でAUCとROIを比較。
- パターン1: 1着確率（binary: 1着=1）
- パターン2: 複勝確率（binary: 3着以内=1）← 現行
- パターン3: EV近似（確率×配当）
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
    compute_lag_features, build_features, N_TOP_SIRE,
)
from train_v92_leakfree import FEATURES_PATTERN_A

from backtest_central_leakfree import (
    classify_condition, calc_trio_bets, calc_umaren_bets, encode_sires_fold,
    check_bets, estimate_payouts,
)

TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
}


def run_wf_target(df, features, target_col, label, test_years=TEST_YEARS):
    """Walk-forward for a given target column."""
    fold_aucs = {}
    all_results = []

    for test_year in test_years:
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
        y_train = train_df[target_col].values
        dates = train_df['date_num']
        valid_cutoff = dates.quantile(0.85)
        tr_idx = dates < valid_cutoff
        va_idx = dates >= valid_cutoff

        X_tr = train_df.loc[tr_idx, features].values
        y_tr = y_train[tr_idx.values]
        X_va = train_df.loc[va_idx, features].values
        y_va = y_train[va_idx.values]

        # Train
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=features)
        dvalid = lgb.Dataset(X_va, label=y_va, feature_name=features, reference=dtrain)
        model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=500,
                         valid_sets=[dvalid],
                         callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

        # Predict
        test_df = df_fold[test_mask].copy()
        test_df['pred'] = model.predict(test_df[features].values)

        # AUC against TOP3 target (standard evaluation)
        y_test_top3 = (test_df['finish'] <= 3).astype(int).values
        try:
            test_auc = roc_auc_score(y_test_top3, test_df['pred'])
        except ValueError:
            test_auc = 0.5
        fold_aucs[test_year] = test_auc

        # ROI evaluation
        for rid in test_df['race_id_str'].unique():
            race_df = test_df[test_df['race_id_str'] == rid].copy()
            if len(race_df) < 5:
                continue

            row0 = race_df.iloc[0]
            cond = classify_condition(row0)
            nh = int(row0.get('num_horses', row0.get('num_horses_val', 14)))
            dist = int(row0['distance'])

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
            trio_hit, umaren_hits, _ = check_bets(actual_top3, trio_bets, umaren_bets, [])
            trio_pay, umaren_pay, _ = estimate_payouts(actual_top3, race_df)

            all_results.append({
                'cond': cond, 'year': test_year,
                'trio_hit': trio_hit,
                'trio_return': trio_pay if trio_hit else 0,
                'umaren_hit': len(umaren_hits) > 0,
                'umaren_return': umaren_pay * 3.5 * len(umaren_hits) if umaren_hits else 0,
            })

    avg_auc = np.mean(list(fold_aucs.values())) if fold_aucs else 0

    # Condition ROI
    cond_roi = {}
    for cond_key in ['A', 'B', 'C', 'D', 'E', 'X']:
        races = [r for r in all_results if r['cond'] == cond_key]
        n = len(races)
        if n < 30:
            continue
        bt = 'umaren' if cond_key == 'E' else 'trio'
        hits = sum(1 for r in races if r[f'{bt}_hit'])
        inv = n * 700
        pay = sum(r[f'{bt}_return'] for r in races)
        cond_roi[cond_key] = {
            'n': n, 'bet_type': bt,
            'hit_rate': round(hits / n * 100, 1),
            'roi': round(pay / inv * 100, 1) if inv > 0 else 0,
        }

    total_inv = len(all_results) * 700
    total_pay = sum(r['trio_return'] for r in all_results)
    overall_roi = total_pay / total_inv * 100 if total_inv > 0 else 0

    print(f"  [{label}] AUC={avg_auc:.4f}, Overall trio ROI={overall_roi:.1f}%, Folds={fold_aucs}")

    return {
        'label': label,
        'avg_auc': round(avg_auc, 4),
        'fold_aucs': {str(k): round(v, 4) for k, v in fold_aucs.items()},
        'overall_estimated_roi': round(overall_roi, 1),
        'condition_roi': cond_roi,
        'n_races': len(all_results),
    }


def main():
    print("=" * 60)
    print("  ■2 目的変数の検証")
    print("=" * 60)

    # Prepare data
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

    # Distance aptitude + frame advantage
    from train_v92_central import compute_distance_aptitude, compute_frame_advantage
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)

    df = df[df['num_horses_val'] >= 5].copy()

    features = list(FEATURES_PATTERN_A)
    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    # === Pattern 1: Win (1着) ===
    print("\n[2] パターン1: 1着確率...")
    df['target_win'] = (df['finish'] == 1).astype(int)
    result1 = run_wf_target(df, features, 'target_win', 'Win (1着=1)')

    # === Pattern 2: Place (3着以内) - 現行 ===
    print("\n[3] パターン2: 複勝確率（現行）...")
    df['target_top3'] = (df['finish'] <= 3).astype(int)
    result2 = run_wf_target(df, features, 'target_top3', 'Place (3着以内=1)')

    # === Pattern 3: EV近似 ===
    print("\n[4] パターン3: EV近似...")
    # EV = P(win) × odds ≈ 的中時のリターン / 投資
    # 三連複の場合: 3着以内に入る確率に配当期待値で重み付け
    # 近似: finish <= 3 をベースに、着順が良いほど重み大
    df['target_ev'] = 0.0
    df.loc[df['finish'] == 1, 'target_ev'] = 1.0
    df.loc[df['finish'] == 2, 'target_ev'] = 0.7
    df.loc[df['finish'] == 3, 'target_ev'] = 0.5
    df.loc[df['finish'] == 4, 'target_ev'] = 0.1
    df.loc[df['finish'] == 5, 'target_ev'] = 0.05
    # Clip to [0,1] for binary classification
    result3 = run_wf_target(df, features, 'target_ev', 'EV weighted (着順重み付け)')

    # Summary
    results = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'current_target': 'Place (finish <= 3)',
        'patterns': {
            'win': result1,
            'place_current': result2,
            'ev_weighted': result3,
        },
        'comparison': {
            'best_auc': max(result1['avg_auc'], result2['avg_auc'], result3['avg_auc']),
            'best_roi': None,
            'recommendation': None,
        }
    }

    # Determine best
    all_results = [('win', result1), ('place_current', result2), ('ev_weighted', result3)]
    best_auc = max(all_results, key=lambda x: x[1]['avg_auc'])
    best_roi = max(all_results, key=lambda x: x[1]['overall_estimated_roi'])

    results['comparison']['best_auc_pattern'] = best_auc[0]
    results['comparison']['best_auc'] = best_auc[1]['avg_auc']
    results['comparison']['best_roi_pattern'] = best_roi[0]
    results['comparison']['best_roi'] = best_roi[1]['overall_estimated_roi']
    results['comparison']['recommendation'] = (
        f"AUC最高: {best_auc[0]} ({best_auc[1]['avg_auc']:.4f}), "
        f"ROI最高: {best_roi[0]} ({best_roi[1]['overall_estimated_roi']:.1f}%). "
        f"現行(place)のAUC={result2['avg_auc']:.4f}は実績があり、変更リスクに見合わない場合は現状維持推奨。"
    )

    print(f"\n{'=' * 60}")
    print(f"  目的変数比較結果")
    print(f"{'=' * 60}")
    for name, res in all_results:
        print(f"  {name:<20} AUC={res['avg_auc']:.4f}  ROI(trio est)={res['overall_estimated_roi']:.1f}%")
    print(f"\n  推奨: {results['comparison']['recommendation']}")

    out_path = os.path.join(BASE_DIR, 'data', 'target_variable_comparison.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存: {out_path}")

    return results


if __name__ == '__main__':
    main()
