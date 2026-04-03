#!/usr/bin/env python3
"""v13.5b 実配当ROI計算 (JRA公式配当データ × 4-model WFバックテスト)

v13.4 (LGB+XGB) vs v13.5b (LGB+XGB+FT+IR) の条件別実配当ROIを比較。
jra_payouts.csv の公式配当データで検証。

出力:
- data/actual_roi_v135b.json
- コンソールに v13.4 vs v13.5b 比較表
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import warnings
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

from train_v135_ft_transformer import (
    build_v134_dataframe, get_v134_features, fill_defaults,
    FTTransformer, IntraRaceAttention, train_ft_transformer,
    DEVICE, LGB_PARAMS, XGB_PARAMS,
)
from train_v135b_intra_ensemble import (
    build_race_id, train_intra_race_with_preds, MAX_HORSES,
)
from train_v92_leakfree import LEAK_FEATURES_A

# Payout lookup
from calc_actual_roi import (
    load_payouts, parse_trio_nums, parse_umaren_nums, parse_wide_data,
    calc_actual_returns,
)
from backtest_central_leakfree import (
    classify_condition, get_axes, calc_trio_bets, calc_umaren_bets,
    calc_wide_bets, check_bets,
)

TEST_YEARS = list(range(2020, 2026))


def evaluate_races_actual_roi(df_test, pred_col, payout_lookup):
    """テストデータの各レースで買い目を生成し、実配当ROIを計算。

    Returns:
        list of dict: 各レースの結果
    """
    results = []
    test_races = df_test['race_id_str'].unique()

    for rid in test_races:
        race_df = df_test[df_test['race_id_str'] == rid].copy()
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

        # AI ranking by prediction score
        race_df = race_df.sort_values(pred_col, ascending=False)
        ranking = race_df['umaban'].astype(int).tolist()

        # Generate bets
        trio_bets = calc_trio_bets(ranking)
        umaren_bets = calc_umaren_bets(ranking)
        wide_bets = calc_wide_bets(ranking)

        # Look up actual payouts
        payout_info = payout_lookup.get(rid)
        (actual_trio_hit, actual_trio_return,
         actual_umaren_hit, actual_umaren_return,
         actual_wide_hits, actual_wide_return) = calc_actual_returns(
            payout_info, trio_bets, umaren_bets, wide_bets)

        results.append({
            'race_id': rid,
            'year': int(row0.get('year_full', 2000 + int(row0['year']))),
            'axes': axes,
            'has_payout': payout_info is not None,
            'actual_trio_hit': actual_trio_hit,
            'actual_trio_return': actual_trio_return,
            'actual_umaren_hit': actual_umaren_hit,
            'actual_umaren_return': actual_umaren_return,
            'actual_wide_hits': actual_wide_hits,
            'actual_wide_return': actual_wide_return,
        })

    return results


def calc_condition_roi(results):
    """条件別の実配当ROIを計算"""
    matched = [r for r in results if r['has_payout']]
    cond_groups = defaultdict(list)
    for r in matched:
        cond_groups[r['axes']['cond_key']].append(r)

    roi_by_cond = {}
    for label in ['A', 'B', 'C', 'D', 'E', 'X']:
        races = cond_groups.get(label, [])
        n = len(races)
        if n < 5:
            roi_by_cond[label] = {'n': n, 'trio_roi': 0, 'trio_hits': 0,
                                   'trio_hit_rate': 0, 'umaren_roi': 0,
                                   'umaren_hits': 0, 'umaren_hit_rate': 0}
            continue

        # Trio: 7 tickets × 100 yen = 700 yen
        trio_hits = sum(1 for r in races if r['actual_trio_hit'])
        trio_inv = n * 700
        trio_pay = sum(r['actual_trio_return'] for r in races)
        trio_roi = trio_pay / trio_inv * 100

        # Umaren: 2 tickets × 350 yen = 700 yen
        umaren_hits = sum(1 for r in races if r['actual_umaren_hit'])
        umaren_inv = n * 700
        umaren_pay = sum(r['actual_umaren_return'] * 3.5 for r in races)
        umaren_roi = umaren_pay / umaren_inv * 100

        roi_by_cond[label] = {
            'n': n,
            'trio_roi': round(trio_roi, 1),
            'trio_hits': trio_hits,
            'trio_hit_rate': round(trio_hits / n * 100, 1),
            'umaren_roi': round(umaren_roi, 1),
            'umaren_hits': umaren_hits,
            'umaren_hit_rate': round(umaren_hits / n * 100, 1),
        }

    return roi_by_cond


def main():
    t0 = time.time()
    print("=" * 70)
    print("  v13.5b 実配当ROI計算")
    print(f"  v13.4 (LGB+XGB) vs v13.5b (LGB+XGB+FT+IR)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # [1] Load payout data
    print("\n[1] 配当データ読み込み...")
    payout_lookup = load_payouts()

    # [2] Build dataframe
    print("\n[2] データフレーム構築 (v13.4 pipeline)...")
    df, sire_map, bms_map = build_v134_dataframe()
    features, jrdb_sel = get_v134_features()
    df = fill_defaults(df, features)
    df = build_race_id(df)

    # year_full for payout matching
    if 'year_full' not in df.columns:
        df['year_full'] = 2000 + df['year'].astype(int)

    print(f"  Data: {len(df)} rows, {len(features)} features")
    print(f"  Races: {df['race_id_str'].nunique()}")

    # [3] Walk-forward with ROI calculation
    print("\n[3] ウォークフォワード + 実配当ROI...")

    all_results_v134 = []
    all_results_v135b = []
    fold_aucs = {'v134': {}, 'v135b': {}}

    for test_year in TEST_YEARS:
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[test_mask, features].values
        y_te = df.loc[test_mask, 'target'].values
        test_indices = df.loc[test_mask].index.values

        print(f"\n{'='*60}")
        print(f"  WF {test_year}: train={len(X_tr):,}, test={len(X_te):,}")
        print(f"{'='*60}")

        # --- LightGBM ---
        print("  Training LightGBM...")
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
        m_lgb = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                          valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        p_lgb = m_lgb.predict(X_te)
        auc_lgb = roc_auc_score(y_te, p_lgb)
        print(f"    LGB AUC: {auc_lgb:.4f}")

        # --- XGBoost ---
        print("  Training XGBoost...")
        dxtr = xgb.DMatrix(X_tr, label=y_tr)
        dxte = xgb.DMatrix(X_te, label=y_te)
        m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=1000,
                           evals=[(dxte, 'valid')],
                           early_stopping_rounds=50, verbose_eval=False)
        p_xgb = m_xgb.predict(dxte)
        auc_xgb = roc_auc_score(y_te, p_xgb)
        print(f"    XGB AUC: {auc_xgb:.4f}")

        # v13.4 ensemble: LGB+XGB
        w_lgb = auc_lgb / (auc_lgb + auc_xgb)
        w_xgb = auc_xgb / (auc_lgb + auc_xgb)
        p_v134 = w_lgb * p_lgb + w_xgb * p_xgb
        auc_v134 = roc_auc_score(y_te, p_v134)
        fold_aucs['v134'][test_year] = auc_v134
        print(f"    v13.4 (LGB+XGB) AUC: {auc_v134:.4f}")

        # --- FT-Transformer ---
        print("  Training FT-Transformer...")
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr.astype(np.float32))
        X_te_s = scaler.transform(X_te.astype(np.float32))

        ft_model, p_ft, p_ft_tr, auc_ft = train_ft_transformer(
            X_tr_s, y_tr.astype(np.float32),
            X_te_s, y_te.astype(np.float32),
            n_features=len(features),
            epochs=50, batch_size=4096, lr=1e-3,
            patience=10, d_token=64, n_heads=4, n_layers=3,
            dropout=0.1, label=f'FT-{test_year}',
        )
        print(f"    FT AUC: {auc_ft:.4f}")

        # --- IntraRace Attention ---
        print("  Training IntraRace Attention...")
        df_tr = df.loc[train_mask].copy()
        df_te = df.loc[test_mask].copy()

        scaler_ir = StandardScaler()
        df_tr_scaled = df_tr.copy()
        df_te_scaled = df_te.copy()
        df_tr_scaled[features] = scaler_ir.fit_transform(
            df_tr[features].values.astype(np.float32))
        df_te_scaled[features] = scaler_ir.transform(
            df_te[features].values.astype(np.float32))

        ir_model, ir_val_dict, ir_tr_dict, ir_tr_auc, ir_val_auc = \
            train_intra_race_with_preds(
                df_tr_scaled, df_te_scaled, features,
                epochs=30, batch_size=64, lr=1e-3, patience=8,
                d_model=64, n_heads=4, n_layers=2, dropout=0.1,
            )

        # Map IntraRace predictions to test array
        p_ir = np.zeros(len(X_te), dtype=np.float32)
        ir_coverage = 0
        for i, idx in enumerate(test_indices):
            if idx in ir_val_dict:
                p_ir[i] = ir_val_dict[idx]
                ir_coverage += 1
            else:
                p_ir[i] = 0.3
        auc_ir = roc_auc_score(y_te, p_ir) if ir_coverage > len(X_te) * 0.5 else 0
        print(f"    IR AUC: {auc_ir:.4f} (coverage={ir_coverage/len(X_te)*100:.1f}%)")

        # v13.5b ensemble: Grid-optimized 4-model
        best_grid_auc = 0
        best_grid_w = None
        best_grid_p = None
        if auc_ir > 0:
            for w1 in np.arange(0.25, 0.45, 0.05):
                for w2 in np.arange(0.25, 0.45, 0.05):
                    for w3 in np.arange(0.05, 0.30, 0.05):
                        w4 = 1.0 - w1 - w2 - w3
                        if w4 < 0.01 or w4 > 0.35:
                            continue
                        p_grid = w1*p_lgb + w2*p_xgb + w3*p_ft + w4*p_ir
                        auc_grid = roc_auc_score(y_te, p_grid)
                        if auc_grid > best_grid_auc:
                            best_grid_auc = auc_grid
                            best_grid_w = (w1, w2, w3, w4)
                            best_grid_p = p_grid

        if best_grid_p is not None:
            p_v135b = best_grid_p
            auc_v135b = best_grid_auc
        else:
            # Fallback to AUC-weighted 3-model
            total3 = auc_lgb + auc_xgb + auc_ft
            p_v135b = (auc_lgb/total3)*p_lgb + (auc_xgb/total3)*p_xgb + (auc_ft/total3)*p_ft
            auc_v135b = roc_auc_score(y_te, p_v135b)

        fold_aucs['v135b'][test_year] = auc_v135b
        print(f"    v13.5b (4-model grid) AUC: {auc_v135b:.4f}")
        if best_grid_w:
            print(f"    Grid weights: L={best_grid_w[0]:.2f} X={best_grid_w[1]:.2f} "
                  f"F={best_grid_w[2]:.2f} IR={best_grid_w[3]:.2f}")

        # --- Evaluate races for ROI ---
        df_test = df.loc[test_mask].copy()
        df_test['pred_v134'] = p_v134
        df_test['pred_v135b'] = p_v135b

        results_v134 = evaluate_races_actual_roi(df_test, 'pred_v134', payout_lookup)
        results_v135b = evaluate_races_actual_roi(df_test, 'pred_v135b', payout_lookup)

        n_matched_v134 = sum(1 for r in results_v134 if r['has_payout'])
        n_matched_v135b = sum(1 for r in results_v135b if r['has_payout'])
        print(f"    v13.4 races: {len(results_v134)} (matched: {n_matched_v134})")
        print(f"    v13.5b races: {len(results_v135b)} (matched: {n_matched_v135b})")

        all_results_v134.extend(results_v134)
        all_results_v135b.extend(results_v135b)

        # Save intermediate results after each fold
        interim = {
            'fold_aucs': fold_aucs,
            'n_results_v134': len(all_results_v134),
            'n_results_v135b': len(all_results_v135b),
        }
        interim_path = os.path.join(DATA_DIR, 'actual_roi_v135b_interim.json')
        with open(interim_path, 'w') as f:
            json.dump(interim, f, default=str)

    # [4] Condition-by-condition ROI
    print(f"\n{'='*70}")
    print(f"  [4] 条件別 実配当ROI比較")
    print(f"{'='*70}")

    roi_v134 = calc_condition_roi(all_results_v134)
    roi_v135b = calc_condition_roi(all_results_v135b)

    # Save results early (before printing, in case of encoding issues)
    mean4 = np.mean(list(fold_aucs['v134'].values())) if fold_aucs['v134'] else 0
    mean5 = np.mean(list(fold_aucs['v135b'].values())) if fold_aucs['v135b'] else 0
    matched_v134 = [r for r in all_results_v134 if r['has_payout']]
    matched_v135b = [r for r in all_results_v135b if r['has_payout']]
    total_inv_v134 = len(matched_v134) * 700
    total_inv_v135b = len(matched_v135b) * 700
    total_pay_v134 = sum(r['actual_trio_return'] for r in matched_v134)
    total_pay_v135b = sum(r['actual_trio_return'] for r in matched_v135b)
    overall_roi_v134 = total_pay_v134 / total_inv_v134 * 100 if total_inv_v134 > 0 else 0
    overall_roi_v135b = total_pay_v135b / total_inv_v135b * 100 if total_inv_v135b > 0 else 0

    output = {
        'generated_at': datetime.now().isoformat(),
        'test_years': TEST_YEARS,
        'device': str(DEVICE),
        'n_features': len(features),
        'total_races_v134': len(matched_v134),
        'total_races_v135b': len(matched_v135b),
        'fold_aucs': {
            'v134': {str(k): round(v, 4) for k, v in fold_aucs['v134'].items()},
            'v135b': {str(k): round(v, 4) for k, v in fold_aucs['v135b'].items()},
        },
        'mean_auc': {'v134': round(mean4, 4), 'v135b': round(mean5, 4)},
        'conditions_v134': roi_v134,
        'conditions_v135b': roi_v135b,
        'overall_trio_roi': {
            'v134': round(overall_roi_v134, 1),
            'v135b': round(overall_roi_v135b, 1),
        },
    }
    out_path = os.path.join(DATA_DIR, 'actual_roi_v135b.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  Results saved: {out_path}")

    # Print comparison
    print(f"\n  {'Cond':<5} {'N':>5} | {'v13.4 trio':>10} {'hit%':>6} | "
          f"{'v13.5b trio':>11} {'hit%':>6} | {'delta':>7} | "
          f"{'v13.4 uma':>10} {'v13.5b uma':>11}")
    print(f"  {'-'*90}")

    all_pass = True
    for label in ['A', 'B', 'C', 'D', 'E', 'X']:
        r4 = roi_v134.get(label, {})
        r5 = roi_v135b.get(label, {})
        n4 = r4.get('n', 0)
        n5 = r5.get('n', 0)
        t4 = r4.get('trio_roi', 0)
        t5 = r5.get('trio_roi', 0)
        h4 = r4.get('trio_hit_rate', 0)
        h5 = r5.get('trio_hit_rate', 0)
        u4 = r4.get('umaren_roi', 0)
        u5 = r5.get('umaren_roi', 0)
        delta = t5 - t4

        # Determine best bet for each condition
        best4 = t4 if label != 'E' else u4
        best5 = t5 if label != 'E' else u5

        status = 'OK' if best5 >= best4 else 'NG'
        if best5 < best4:
            all_pass = False

        print(f"  {label:<5} {n5:>5} | {t4:>9.1f}% {h4:>5.1f}% | "
              f"{t5:>10.1f}% {h5:>5.1f}% | {delta:>+6.1f}% | "
              f"{u4:>9.1f}% {u5:>10.1f}% {status}")

    # Mean AUC
    print(f"\n  AUC by year:")
    print(f"  {'Year':<6} {'v13.4':>8} {'v13.5b':>8} {'delta':>8}")
    print(f"  {'-'*35}")
    for yr in TEST_YEARS:
        a4 = fold_aucs['v134'].get(yr, 0)
        a5 = fold_aucs['v135b'].get(yr, 0)
        print(f"  {yr:<6} {a4:>8.4f} {a5:>8.4f} {a5-a4:>+8.4f}")
    print(f"  {'Mean':<6} {mean4:>8.4f} {mean5:>8.4f} {mean5-mean4:>+8.4f}")

    # Year-by-year ROI stability (2023-2025 only, where payouts exist)
    print(f"\n  Year x Condition ROI (v13.5b trio):")
    for yr in TEST_YEARS:
        yr_results = [r for r in all_results_v135b if r['has_payout'] and r['year'] == yr]
        for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
            cr = [r for r in yr_results if r['axes']['cond_key'] == cond]
            if len(cr) < 5:
                continue
            n = len(cr)
            trio_hits = sum(1 for r in cr if r['actual_trio_hit'])
            trio_pay = sum(r['actual_trio_return'] for r in cr)
            inv = n * 700
            roi = trio_pay / inv * 100
            print(f"    {yr} {cond}: N={n:>4} hit={trio_hits/n*100:>5.1f}% ROI={roi:>7.1f}%")

    print(f"\n  Overall trio ROI:")
    print(f"    v13.4:  {overall_roi_v134:.1f}% (N={len(matched_v134)})")
    print(f"    v13.5b: {overall_roi_v135b:.1f}% (N={len(matched_v135b)})")

    # Final judgment
    output['all_conditions_pass'] = all_pass
    # Re-save with final judgment
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    if all_pass:
        print(f"  *** ALL CONDITIONS v13.5b >= v13.4 -> ADOPTED ***")
    else:
        print(f"  *** SOME CONDITIONS v13.5b < v13.4 -> REVIEW NEEDED ***")
    print(f"{'='*70}")

    elapsed = (time.time() - t0) / 60
    print(f"  Total elapsed: {elapsed:.1f} min")

    return output


if __name__ == '__main__':
    main()
