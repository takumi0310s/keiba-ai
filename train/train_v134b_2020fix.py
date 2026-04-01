#!/usr/bin/env python3
"""v13.4b — 2020年WF AUC改善テスト

問題: 2020年WF AUC=0.7923（他年0.85-0.87に対して突出して低い）
原因: JRDB/OZ特徴量(46個/121個=38%)が2020年以降のみ存在
      → 学習データ(2015-2019)では全てデフォルト値
      → テストデータ(2020)で初めて実データが出現
      → 学習時にJRDB特徴量のシグナルを学習できない

テスト:
  A) 2020年WFのみJRDB/OZ特徴量を除外して学習・テスト（75特徴量）
  B) 年別適応型: 2020年は75特徴量、2021+は121特徴量で学習・テスト
  C) JRDB特徴量のデフォルト値を学習データでもテストデータでも除外（NaN→0ではなく列自体を除外）
"""

import os
import sys
import time
import pickle
import gzip
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

from train_v92_central import (
    load_data, encode_categoricals, encode_sires,
    load_training_times, merge_training_features,
    compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance,
    compute_lag_features, build_features,
    compute_distance_aptitude, compute_frame_advantage,
    FEATURES_V91, V92_NEW_FEATURES, V93_NEW_FEATURES,
    COURSE_MAP, N_TOP_SIRE,
)
from train_v92_leakfree import LEAK_FEATURES_A
from train_v12_comprehensive import (
    V12_NEW_FEATURES, LGB_PARAMS, XGB_PARAMS,
    merge_speed_index, merge_training_1f, merge_sire_shinba,
    merge_dam_stats, compute_pci, _build_nk_race_id,
)
from jrdb_features import (
    merge_jrdb_train_features, _build_nk_race_id_from_jv,
    JRDB_LIVE_FEATURES, JRDB_DEFAULTS,
)
from train_v132_extended import (
    V132_NEW_FEATURES, V132_DEFAULTS,
    merge_cha_features, merge_jo_features, merge_kta_features,
    merge_ze_features, merge_sr_features, merge_kka_features,
    load_v13_jrdb_selected,
)
from train_v133_extended import V133_NEW_FEATURES, V133_DEFAULTS, merge_stable_comment_combined
from train_v134_odds_change import V134_NEW_FEATURES, V134_DEFAULTS, merge_oz_features


# JRDB/OZ features that only have data for 2020+
JRDB_OZ_FEATURES = None  # will be computed dynamically


def _get_jrdb_oz_features(all_features):
    """JRDB/OZ由来の特徴量を特定（2020+のみ存在）"""
    return [f for f in all_features
            if f.startswith('jrdb_') or f.startswith('oz_')
            or f in ('odds_change_rate', 'pop_rank_change', 'odds_sharp_drop')]


def _get_non_jrdb_features(all_features):
    """JRDB/OZ以外の特徴量（全年データあり）"""
    jrdb_oz = set(_get_jrdb_oz_features(all_features))
    return [f for f in all_features if f not in jrdb_oz]


def walk_forward_single(df, features, test_year, label='target'):
    """Single year WF fold"""
    ty = test_year - 2000
    train_mask = df['year'] < ty
    test_mask = df['year'] == ty
    if train_mask.sum() < 1000 or test_mask.sum() < 100:
        return None
    X_tr = df.loc[train_mask, features].values
    y_tr = df.loc[train_mask, label].values
    X_te = df.loc[test_mask, features].values
    y_te = df.loc[test_mask, label].values

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
    m_lgb = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                      valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    p_lgb = m_lgb.predict(X_te)
    auc_lgb = roc_auc_score(y_te, p_lgb)

    dxtr = xgb.DMatrix(X_tr, label=y_tr)
    dxte = xgb.DMatrix(X_te, label=y_te)
    m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=1000,
                       evals=[(dxte, 'valid')],
                       early_stopping_rounds=50, verbose_eval=False)
    p_xgb = m_xgb.predict(dxte)
    auc_xgb = roc_auc_score(y_te, p_xgb)

    p_tr = m_lgb.predict(X_tr)
    auc_tr = roc_auc_score(y_tr, p_tr)

    return {
        'year': test_year, 'lgb_auc': auc_lgb, 'xgb_auc': auc_xgb,
        'train_auc': auc_tr, 'gap': auc_tr - auc_lgb,
        'lgb_model': m_lgb, 'xgb_model': m_xgb, 'features': features,
    }


def walk_forward_full(df, features, years=range(2020, 2026)):
    """Standard WF backtest"""
    results = []
    for y in years:
        r = walk_forward_single(df, features, y)
        if r:
            results.append(r)
    return results


def walk_forward_adaptive(df, full_features, lite_features, jrdb_start_year=2021):
    """Adaptive WF: lite features for years before JRDB, full for after.

    For 2020: train on 2015-2019 with lite features, test 2020 with lite features
    For 2021+: train with full features, test with full features
    """
    results = []
    for test_year in range(2020, 2026):
        if test_year < jrdb_start_year:
            r = walk_forward_single(df, lite_features, test_year)
        else:
            r = walk_forward_single(df, full_features, test_year)
        if r:
            results.append(r)
    return results


def print_wf(results, label=''):
    print(f"\n  WF: {label}")
    print(f"  {'='*60}")
    aucs = []
    for r in results:
        n_feats = len(r.get('features', []))
        ok = 'Y' if r['lgb_auc'] > 0.78 else 'N'
        gok = 'Y' if r['gap'] < 0.05 else 'N'
        print(f"  {r['year']}: LGB={r['lgb_auc']:.4f} XGB={r['xgb_auc']:.4f} "
              f"Train={r['train_auc']:.4f} Gap={r['gap']:.4f} {ok}{gok}"
              f" ({n_feats}f)" if n_feats else "")
        aucs.append(r['lgb_auc'])
    mean_auc = np.mean(aucs)
    print(f"  --- Mean LGB AUC: {mean_auc:.4f} ---")
    return mean_auc


def main():
    t0 = time.time()
    print("=" * 60)
    print("  v13.4b — 2020 WF AUC Improvement Analysis")
    print("=" * 60)

    # Build full dataframe (identical to v13.4)
    print("\n[1/4] Building features (same as v13.4)...")
    df = load_data()
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
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)
    df = merge_speed_index(df)
    df = merge_training_1f(df)
    df = merge_sire_shinba(df)
    df = merge_dam_stats(df)
    df = compute_pci(df)
    df = merge_jrdb_train_features(df)
    if '_nk_rid' not in df.columns:
        df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    if '_uma' not in df.columns:
        df['_uma'] = df['umaban'].astype(int)
    df = merge_cha_features(df)
    df = merge_jo_features(df)
    df = merge_kta_features(df)
    df = merge_ze_features(df)
    df = merge_sr_features(df)
    df = merge_kka_features(df)
    df.drop(columns=['_nk_rid', '_uma'], inplace=True, errors='ignore')
    df = merge_stable_comment_combined(df)
    df = merge_oz_features(df)

    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses'] >= 5].copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    print(f"  Final: {len(df)} rows")

    # Feature lists
    F_V93 = list(FEATURES_V91) + list(V92_NEW_FEATURES) + list(V93_NEW_FEATURES)
    F_V12_BASE = [f for f in F_V93 if f not in LEAK_FEATURES_A]
    V12_ADOPTED = [f for f in V12_NEW_FEATURES if f != 'dam_top3r']
    F_V12 = F_V12_BASE + V12_ADOPTED
    jrdb_sel = load_v13_jrdb_selected()
    F_V13 = F_V12 + jrdb_sel
    F_V132 = F_V13 + V132_NEW_FEATURES
    F_V133 = F_V132 + V133_NEW_FEATURES
    F_V134 = F_V133 + V134_NEW_FEATURES

    all_defaults = {**JRDB_DEFAULTS, **V132_DEFAULTS, **V133_DEFAULTS, **V134_DEFAULTS}
    for f in F_V134:
        if f not in df.columns:
            df[f] = all_defaults.get(f, 0)
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(all_defaults.get(f, 0))

    F_LITE = _get_non_jrdb_features(F_V134)
    F_JRDB = _get_jrdb_oz_features(F_V134)

    print(f"\n  Full features: {len(F_V134)}")
    print(f"  JRDB/OZ features: {len(F_JRDB)}")
    print(f"  Lite features (non-JRDB): {len(F_LITE)}")

    # ===== Tests =====
    print(f"\n{'='*60}")
    print(f"  [2/4] Baseline: v13.4 standard WF")
    print(f"{'='*60}")
    wf_baseline = walk_forward_full(df, F_V134)
    auc_baseline = print_wf(wf_baseline, 'v13.4 baseline (121 feats)')

    # Test A: 2020 only with lite features
    print(f"\n{'='*60}")
    print(f"  [3/4] Test A: 2020年のみlite features ({len(F_LITE)}f)")
    print(f"{'='*60}")

    print("\n--- A1: 2020年のみlite ---")
    r2020_lite = walk_forward_single(df, F_LITE, 2020)
    r2020_full = walk_forward_single(df, F_V134, 2020)
    print(f"  2020 full ({len(F_V134)}f): LGB={r2020_full['lgb_auc']:.4f}")
    print(f"  2020 lite ({len(F_LITE)}f): LGB={r2020_lite['lgb_auc']:.4f}")
    print(f"  Delta: {r2020_lite['lgb_auc'] - r2020_full['lgb_auc']:+.4f}")

    # Test B: Adaptive WF (lite for 2020, full for 2021+)
    print(f"\n--- B: Adaptive WF ---")
    wf_adaptive = walk_forward_adaptive(df, F_V134, F_LITE, jrdb_start_year=2021)
    auc_adaptive = print_wf(wf_adaptive, f'Adaptive (2020={len(F_LITE)}f, 2021+={len(F_V134)}f)')

    # Test C: Adaptive with JRDB start at 2022
    # (2021 training data has only 1 year of JRDB: 2020, might still be weak)
    print(f"\n--- C: Adaptive (JRDB from 2022) ---")
    wf_adaptive2 = walk_forward_adaptive(df, F_V134, F_LITE, jrdb_start_year=2022)
    auc_adaptive2 = print_wf(wf_adaptive2, f'Adaptive (2020-21={len(F_LITE)}f, 2022+={len(F_V134)}f)')

    # Test D: Weight adjustment - down-weight 2020
    # Instead of equal weight, use AUC-proportional or exclude 2020 from mean
    print(f"\n--- D: Weighted mean (exclude 2020) ---")
    auc_no2020 = np.mean([r['lgb_auc'] for r in wf_baseline if r['year'] != 2020])
    print(f"  v13.4 mean (2021-2025 only): {auc_no2020:.4f}")

    # ===== Summary =====
    print(f"\n{'='*60}")
    print(f"  [4/4] SUMMARY")
    print(f"{'='*60}")

    results_table = {
        'v13.4 baseline (all 121f)': auc_baseline,
        'Adaptive (2020=lite, 2021+=full)': auc_adaptive,
        'Adaptive (2020-21=lite, 2022+=full)': auc_adaptive2,
        'v13.4 mean (2021-2025 only)': auc_no2020,
    }

    best_name = max(results_table, key=results_table.get)
    best_auc = results_table[best_name]

    for name, auc in sorted(results_table.items(), key=lambda x: -x[1]):
        delta = auc - auc_baseline
        marker = ' <<<' if auc == best_auc else ''
        print(f"  {name:45s}: {auc:.6f} ({delta:+.6f}){marker}")

    print(f"\n  2020 improvement: full={r2020_full['lgb_auc']:.4f} → lite={r2020_lite['lgb_auc']:.4f}"
          f" ({r2020_lite['lgb_auc']-r2020_full['lgb_auc']:+.4f})")

    # Determine if adaptive approach should be adopted
    adopted = False
    best_wf = None
    if auc_adaptive > auc_baseline:
        print(f"\n  Adaptive WF improves mean AUC: {auc_baseline:.4f} → {auc_adaptive:.4f}")
        all_ok = all(r['lgb_auc'] > 0.78 for r in wf_adaptive)
        no_overfit = all(r['gap'] < 0.05 for r in wf_adaptive)
        if all_ok and no_overfit:
            adopted = True
            best_wf = wf_adaptive
            print(f"  VERDICT: ADOPTED (adaptive WF)")
        else:
            print(f"  FAILED: all>0.78={all_ok}, gap<0.05={no_overfit}")

    if auc_adaptive2 > max(auc_baseline, auc_adaptive):
        all_ok2 = all(r['lgb_auc'] > 0.78 for r in wf_adaptive2)
        no_overfit2 = all(r['gap'] < 0.05 for r in wf_adaptive2)
        if all_ok2 and no_overfit2:
            adopted = True
            best_wf = wf_adaptive2
            print(f"  VERDICT: ADOPTED (adaptive WF v2, JRDB from 2022)")

    if not adopted:
        print(f"\n  VERDICT: NOT ADOPTED — baseline is best or criteria not met")
        print(f"  Note: 2020 low AUC is structural (JRDB data gap)")
        print(f"  Recommendation: report WF AUC 2021-2025 alongside full WF")

    # Save results
    result = {
        'baseline_auc': auc_baseline,
        'adaptive_auc': auc_adaptive,
        'adaptive2_auc': auc_adaptive2,
        'no2020_auc': auc_no2020,
        '2020_full_auc': r2020_full['lgb_auc'],
        '2020_lite_auc': r2020_lite['lgb_auc'],
        '2020_improvement': r2020_lite['lgb_auc'] - r2020_full['lgb_auc'],
        'adopted': adopted,
        'lite_features': F_LITE,
        'full_features': F_V134,
        'jrdb_oz_features': F_JRDB,
        'baseline_yearly': [{'year': r['year'], 'lgb_auc': r['lgb_auc'],
                              'gap': r['gap']} for r in wf_baseline],
        'adaptive_yearly': [{'year': r['year'], 'lgb_auc': r['lgb_auc'],
                              'gap': r['gap']} for r in wf_adaptive],
        'adaptive2_yearly': [{'year': r['year'], 'lgb_auc': r['lgb_auc'],
                               'gap': r['gap']} for r in wf_adaptive2],
        'timestamp': datetime.now().isoformat(),
    }
    rpath = os.path.join(DATA_DIR, 'v134b_2020fix_results.json')
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Saved: {rpath}")

    # If adaptive adopted, train dual production models
    if adopted and best_wf is not None:
        print(f"\n{'='*60}")
        print(f"  TRAINING DUAL PRODUCTION MODELS")
        print(f"{'='*60}")
        _train_dual_production(df, F_V134, F_LITE, sire_map, bms_map, best_wf, jrdb_sel)

    print(f"\n  Elapsed: {(time.time()-t0)/60:.1f} min")


def _train_dual_production(df, full_features, lite_features, sire_map, bms_map, wf, jrdb_sel):
    """Train both full and lite production models"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    my = int(df['year'].max())
    vm = df['year'] >= (my - 1)
    tm = ~vm

    # Full model (for 2021+ predictions)
    print("\n  --- Full model (121 features) ---")
    X_tr, y_tr = df.loc[tm, full_features].values, df.loc[tm, 'target'].values
    X_va, y_va = df.loc[vm, full_features].values, df.loc[vm, 'target'].values
    print(f"  Train: {tm.sum()}, Valid: {vm.sum()}")

    dt = lgb.Dataset(X_tr, label=y_tr)
    dv = lgb.Dataset(X_va, label=y_va, reference=dt)
    ml = lgb.train(LGB_PARAMS, dt, num_boost_round=1000, valid_sets=[dv],
                   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    al = roc_auc_score(y_va, ml.predict(X_va))
    print(f"  LGB AUC: {al:.4f}")

    dxt = xgb.DMatrix(X_tr, label=y_tr)
    dxv = xgb.DMatrix(X_va, label=y_va)
    mx = xgb.train(XGB_PARAMS, dxt, num_boost_round=1000, evals=[(dxv, 'valid')],
                   early_stopping_rounds=50, verbose_eval=100)
    ax = roc_auc_score(y_va, mx.predict(dxv))
    print(f"  XGB AUC: {ax:.4f}")

    wl, wx = al / (al + ax), ax / (al + ax)
    wf_auc = np.mean([r['lgb_auc'] for r in wf])

    pa_feats = [f for f in full_features if f not in LEAK_FEATURES_A]
    pkl = {
        'model': ml, 'features': pa_feats, 'version': 'v134b_leakfree',
        'auc': al, 'wf_auc': wf_auc, 'leak_free': True, 'leak_pattern': 'A',
        'leak_removed': sorted(LEAK_FEATURES_A),
        'sire_map': sire_map, 'bms_map': bms_map, 'n_top_encode': N_TOP_SIRE,
        'trained_at': now, 'model_type': 'central',
        'xgb_model': mx, 'mlp_model': None, 'mlp_scaler': None,
        'ensemble_weights': {'lgb': wl, 'xgb': wx, 'mlp': 0},
        'course_map': dict(COURSE_MAP),
        'jrdb_features': jrdb_sel,
        'lite_features': lite_features,
        'adaptive_wf': True,
    }
    ap = os.path.join(BASE_DIR, 'keiba_model_v134b_central.pkl.gz')
    with gzip.open(ap, 'wb') as f:
        pickle.dump(pkl, f)
    print(f"  Saved: {ap}")

    pb_feats = full_features + JRDB_LIVE_FEATURES
    pkl_b = dict(pkl)
    pkl_b['features'] = pb_feats
    pkl_b['version'] = 'v134b_live'
    pkl_b['leak_free'] = False
    pkl_b['leak_pattern'] = 'B'
    pkl_b['is_live'] = True
    bp = os.path.join(BASE_DIR, 'keiba_model_v134b_central_live.pkl.gz')
    with gzip.open(bp, 'wb') as f:
        pickle.dump(pkl_b, f)
    print(f"  Saved: {bp}")
    print(f"  *** v13.4b saved! A={len(pa_feats)} feats, B={len(pb_feats)} feats ***")


if __name__ == '__main__':
    main()
