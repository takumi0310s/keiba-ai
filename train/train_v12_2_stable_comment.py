#!/usr/bin/env python3
"""v12.2再学習 — stable_comment_scoreのみ追加投入

v12（74特徴量, WF AUC 0.8037）にstable_comment_score（1特徴量）を追加。
netkeibaプレミアム厩舎コメントスコア(-3〜+3)。
カバレッジ向上(84-85%)により再テスト。

採用基準:
  - WF AUC > 0.8037
  - 全年AUC > 0.78
  - gap < 0.05
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
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Import base functions
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
    merge_speed_index, merge_training_1f, merge_sire_shinba,
    merge_dam_stats, compute_pci,
    V12_NEW_FEATURES, LGB_PARAMS, XGB_PARAMS,
    walk_forward_backtest,
    compute_feature_importance, train_production_model,
    _build_nk_race_id,
)


def print_wf_results(results, label=''):
    """WF結果表示（cp932互換版）"""
    print(f"\n{'='*60}")
    print(f"  WF Results: {label}")
    print(f"{'='*60}")
    aucs = []
    for r in results:
        aucs.append(r['lgb_auc'])
        ok = 'OK' if r['lgb_auc'] > 0.78 else 'NG'
        gap_ok = 'OK' if r['gap'] < 0.05 else 'NG'
        print(f"  {r['year']}: LGB={r['lgb_auc']:.4f} XGB={r['xgb_auc']:.4f} "
              f"Train={r['train_auc']:.4f} Gap={r['gap']:.4f} [{ok}][{gap_ok}]")
    mean_auc = np.mean(aucs)
    print(f"  --- Mean LGB AUC: {mean_auc:.4f} ---")
    return mean_auc


def merge_stable_comment_score(df):
    """厩舎コメントスコアをマージ。スコア(-3〜+3)、未マッチは0(=中立)"""
    csv_path = os.path.join(DATA_DIR, 'netkeiba_stable_comments.csv')
    if not os.path.exists(csv_path):
        print("  WARNING: netkeiba_stable_comments.csv not found")
        df['stable_comment_score'] = 0
        return df

    sc = pd.read_csv(csv_path, encoding='utf-8-sig', dtype={'race_id': str, 'umaban': str})
    sc['score'] = pd.to_numeric(sc['score'], errors='coerce').fillna(0)

    df = _build_nk_race_id(df)
    df['_uma_str'] = df['umaban'].astype(int).astype(str)
    sc['_nk_rid'] = sc['race_id'].astype(str).str.zfill(12)
    sc['_uma_str'] = sc['umaban'].astype(str)

    sc_dedup = sc.drop_duplicates(subset=['_nk_rid', '_uma_str'], keep='last')
    merged = df.merge(
        sc_dedup[['_nk_rid', '_uma_str', 'score']],
        on=['_nk_rid', '_uma_str'], how='left', suffixes=('', '_sc')
    )

    # 未マッチは0(中立)で埋める（meanではなく0が自然）
    merged['stable_comment_score'] = merged['score'].fillna(0)

    matched = merged['score'].notna().sum()
    total = len(df)
    print(f"  Stable Comment: {matched}/{total} matched ({matched/total*100:.1f}%)")

    # 年別カバレッジ
    if 'year' in df.columns:
        temp = df.copy()
        temp['_matched'] = merged['score'].notna()
        for y in sorted(temp['year'].unique()):
            mask = temp['year'] == y
            m = temp.loc[mask, '_matched'].sum()
            t = mask.sum()
            pct = m / t * 100 if t > 0 else 0
            print(f"    Year {int(y)+2000}: {m}/{t} ({pct:.1f}%)")

    df['stable_comment_score'] = merged['stable_comment_score'].values
    df.drop(columns=['_uma_str'], inplace=True, errors='ignore')
    return df


def main():
    start_time = time.time()
    print("=" * 60)
    print("  v12.2 Re-training: +stable_comment_score")
    print("=" * 60)

    # Phase 1: Load & encode
    print("\n[1/6] Loading data...")
    df = load_data()
    print(f"  Loaded: {len(df)} rows")
    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)
    print(f"  Encoded: sires={len(sire_map)}, bms={len(bms_map)}")

    # Phase 2: Training data merge
    print("\n[2/6] Merging training data...")
    tt_data = load_training_times()
    df = merge_training_features(df, tt_data)

    # Phase 3: Historical stats
    print("\n[3/6] Computing historical stats...")
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)
    df = compute_lag_features(df)
    df = build_features(df)
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)

    # Phase 4: V12 features
    print("\n[4/6] Merging v12 features...")
    df = merge_speed_index(df)
    df = merge_training_1f(df)
    df = merge_sire_shinba(df)
    df = merge_dam_stats(df)
    df = compute_pci(df)

    # Phase 4b: V12.2 new feature
    print("\n[4b] Merging stable_comment_score...")
    df = merge_stable_comment_score(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses'] >= 5].copy()

    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    print(f"\n  Final dataset: {len(df)} rows, years {int(df['year'].min())+2000}-{int(df['year'].max())+2000}")

    # Feature lists
    FEATURES_V93 = list(FEATURES_V91) + list(V92_NEW_FEATURES) + list(V93_NEW_FEATURES)
    FEATURES_BASELINE = [f for f in FEATURES_V93 if f not in LEAK_FEATURES_A]

    # V12 adopted features (dam_top3rは除外済み)
    V12_ADOPTED = [f for f in V12_NEW_FEATURES if f != 'dam_top3r']
    FEATURES_V12 = FEATURES_BASELINE + V12_ADOPTED

    # V12.2 = V12 + stable_comment_score
    FEATURES_V12_2 = FEATURES_V12 + ['stable_comment_score']

    # Ensure all features exist
    for f in FEATURES_V12_2:
        if f not in df.columns:
            print(f"  WARNING: {f} not in df, filling with 0")
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"\n  V12 features (baseline): {len(FEATURES_V12)}")
    print(f"  V12.2 features (+stable_comment_score): {len(FEATURES_V12_2)}")

    # ===== WF Backtest: V12 baseline =====
    print("\n[5/6] Walk-forward backtesting...")
    print("\n--- V12 Baseline ---")
    wf_v12 = walk_forward_backtest(df, FEATURES_V12)
    auc_v12 = print_wf_results(wf_v12, 'V12 Baseline')

    # ===== WF Backtest: V12.2 =====
    print("\n--- V12.2 (+stable_comment_score) ---")
    wf_v12_2 = walk_forward_backtest(df, FEATURES_V12_2)
    auc_v12_2 = print_wf_results(wf_v12_2, 'V12.2')

    # Feature importance
    imp_ranked = compute_feature_importance(wf_v12_2, FEATURES_V12_2)
    print("\n  Top 20 features by importance:")
    for i, (f, v) in enumerate(imp_ranked[:20]):
        marker = " ★NEW" if f == 'stable_comment_score' else ""
        print(f"  {i+1:2d}. {f:35s} {v:12.1f}{marker}")

    # stable_comment_score rank
    sc_rank = next((i+1 for i, (f, _) in enumerate(imp_ranked) if f == 'stable_comment_score'), len(imp_ranked))
    sc_imp = next((v for f, v in imp_ranked if f == 'stable_comment_score'), 0)
    print(f"\n  stable_comment_score: rank {sc_rank}/{len(FEATURES_V12_2)}, importance {sc_imp:.1f}")

    # ===== 採用判定 =====
    print(f"\n{'='*60}")
    print(f"  ADOPTION DECISION")
    print(f"{'='*60}")

    improvement = auc_v12_2 - auc_v12
    all_years_ok = all(r['lgb_auc'] > 0.78 for r in wf_v12_2)
    no_overfit = all(r['gap'] < 0.05 for r in wf_v12_2)
    auc_improved = auc_v12_2 > 0.8037  # Current v12 baseline

    print(f"  V12 AUC:       {auc_v12:.4f}")
    print(f"  V12.2 AUC:     {auc_v12_2:.4f}")
    print(f"  Improvement:   {improvement:+.5f}")
    print(f"  All years>0.78: {'PASS' if all_years_ok else 'FAIL'}")
    print(f"  No overfitting: {'PASS' if no_overfit else 'FAIL'}")
    print(f"  AUC > 0.8037:   {'PASS' if auc_improved else 'FAIL'}")

    # 年別詳細比較
    print(f"\n  Year-by-year comparison:")
    print(f"  {'Year':>6} {'V12':>8} {'V12.2':>8} {'Diff':>9}")
    for r12, r122 in zip(wf_v12, wf_v12_2):
        diff = r122['lgb_auc'] - r12['lgb_auc']
        sign = '+' if diff >= 0 else ''
        print(f"  {r12['year']:>6} {r12['lgb_auc']:.4f}   {r122['lgb_auc']:.4f}   {sign}{diff:.5f}")

    adopted = auc_improved and all_years_ok and no_overfit
    print(f"\n  VERDICT: {'ADOPTED as v12.2' if adopted else 'NOT ADOPTED (v12 maintained)'}")

    # Save results
    result_data = {
        'version': 'v12.2',
        'new_feature': 'stable_comment_score',
        'v12_auc': auc_v12,
        'v12_2_auc': auc_v12_2,
        'improvement': improvement,
        'adopted': adopted,
        'all_years_ok': all_years_ok,
        'no_overfit': no_overfit,
        'auc_improved': auc_improved,
        'feature_rank': sc_rank,
        'feature_importance': sc_imp,
        'n_features_v12': len(FEATURES_V12),
        'n_features_v12_2': len(FEATURES_V12_2),
        'features_v12_2': FEATURES_V12_2,
        'yearly_results_v12': [{
            'year': r['year'], 'lgb_auc': r['lgb_auc'], 'xgb_auc': r['xgb_auc'],
            'train_auc': r['train_auc'], 'gap': r['gap']
        } for r in wf_v12],
        'yearly_results_v12_2': [{
            'year': r['year'], 'lgb_auc': r['lgb_auc'], 'xgb_auc': r['xgb_auc'],
            'train_auc': r['train_auc'], 'gap': r['gap']
        } for r in wf_v12_2],
        'timestamp': datetime.now().isoformat(),
    }

    result_path = os.path.join(DATA_DIR, 'v12_2_training_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved: {result_path}")

    # ===== 本番モデル学習 =====
    if adopted:
        print(f"\n{'='*60}")
        print(f"  TRAINING PRODUCTION MODEL (v12.2)")
        print(f"{'='*60}")
        train_v12_2_production(df, FEATURES_V12_2, sire_map, bms_map, wf_v12_2)
    else:
        print(f"\n  v12 maintained. No model update.")

    elapsed = (time.time() - start_time) / 60
    print(f"\n  Total elapsed: {elapsed:.1f} min")


def train_v12_2_production(df, features, sire_map, bms_map, wf_results):
    """v12.2本番モデル学習・保存"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    OUTPUT_DIR = BASE_DIR

    max_year = int(df['year'].max())
    valid_mask = df['year'] >= (max_year - 1)
    train_mask = ~valid_mask

    X_train = df.loc[train_mask, features].values
    y_train = df.loc[train_mask, 'target'].values
    X_valid = df.loc[valid_mask, features].values
    y_valid = df.loc[valid_mask, 'target'].values

    print(f"  Train: {train_mask.sum()}, Valid: {valid_mask.sum()}")
    print(f"  Features: {len(features)}")

    # LGB
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
    lgb_model = lgb.train(
        LGB_PARAMS, dtrain, num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    lgb_auc = roc_auc_score(y_valid, lgb_model.predict(X_valid))
    print(f"  LGB AUC: {lgb_auc:.4f}")

    # XGB
    dxtr = xgb.DMatrix(X_train, label=y_train)
    dxte = xgb.DMatrix(X_valid, label=y_valid)
    xgb_model = xgb.train(
        XGB_PARAMS, dxtr, num_boost_round=1000,
        evals=[(dxte, 'valid')],
        early_stopping_rounds=50, verbose_eval=100
    )
    xgb_auc = roc_auc_score(y_valid, xgb_model.predict(dxte))
    print(f"  XGB AUC: {xgb_auc:.4f}")

    # Ensemble weights
    total = lgb_auc + xgb_auc
    w_lgb = lgb_auc / total
    w_xgb = xgb_auc / total
    ens_pred = w_lgb * lgb_model.predict(X_valid) + w_xgb * xgb_model.predict(dxte)
    ens_auc = roc_auc_score(y_valid, ens_pred)
    print(f"  Ensemble AUC: {ens_auc:.4f} (LGB {w_lgb:.3f} + XGB {w_xgb:.3f})")

    wf_auc = np.mean([r['lgb_auc'] for r in wf_results])

    # Pattern A (leak-free)
    pattern_a_features = [f for f in features if f not in LEAK_FEATURES_A]
    pkl_a = {
        'model': lgb_model,
        'features': pattern_a_features,
        'version': 'v12.2_leakfree',
        'auc': lgb_auc,
        'ensemble_auc': ens_auc,
        'wf_auc': wf_auc,
        'leak_free': True,
        'leak_pattern': 'A',
        'leak_removed': sorted(LEAK_FEATURES_A),
        'sire_map': sire_map,
        'bms_map': bms_map,
        'n_top_encode': N_TOP_SIRE,
        'trained_at': now,
        'n_train': int(train_mask.sum()),
        'n_valid': int(valid_mask.sum()),
        'model_type': 'central',
        'xgb_model': xgb_model,
        'mlp_model': None,
        'mlp_scaler': None,
        'ensemble_weights': {'lgb': w_lgb, 'xgb': w_xgb, 'mlp': 0},
        'course_map': dict(COURSE_MAP),
        'v12_new_features': V12_NEW_FEATURES,
        'v12_2_new_feature': 'stable_comment_score',
    }

    a_path = os.path.join(OUTPUT_DIR, 'keiba_model_v12_central.pkl.gz')
    with gzip.open(a_path, 'wb') as f:
        pickle.dump(pkl_a, f)
    print(f"  Pattern A saved: {a_path}")

    # Pattern B (live)
    pkl_b = dict(pkl_a)
    pkl_b['features'] = features
    pkl_b['version'] = 'v12.2_live'
    pkl_b['leak_free'] = False
    pkl_b['leak_pattern'] = 'B'
    pkl_b['is_live'] = True

    b_path = os.path.join(OUTPUT_DIR, 'keiba_model_v12_central_live.pkl.gz')
    with gzip.open(b_path, 'wb') as f:
        pickle.dump(pkl_b, f)
    print(f"  Pattern B saved: {b_path}")

    # V8 backup
    v8_path = os.path.join(OUTPUT_DIR, 'keiba_model_v8.pkl')
    v8_pkl = dict(pkl_a)
    v8_pkl['version'] = 'v12.2_backup'
    with open(v8_path, 'wb') as f:
        pickle.dump(v8_pkl, f)
    print(f"  V8 backup saved: {v8_path}")

    print(f"\n  *** v12.2 Production model saved! ***")
    print(f"  Pattern A: {len(pattern_a_features)} features")
    print(f"  Pattern B: {len(features)} features")
    print(f"  WF AUC: {wf_auc:.4f}")


if __name__ == '__main__':
    main()
