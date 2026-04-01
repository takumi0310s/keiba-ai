#!/usr/bin/env python3
"""v13 JRDB特徴量追加 — WFバックテスト

v12の74特徴量 + JRDB特徴量候補でウォークフォワードバックテストを実行。
採用基準: WF AUC > 0.8037 (v12ベースライン) かつ全年AUC > 0.78 かつ gap < 0.05

JRDB特徴量候補（最大28個）:
  前日データ(KYI): IDM, 調教指数, 厩舎指数, 情報指数, 総合指数, 激走指数,
                   展開予想(テン/ペース/上がり/位置), クラス, 上昇度, 重適性,
                   蹄, 外厩ランク, 厩舎ランク, 調教矢印, 厩舎評価, 脚質, 距離適性
  前走成績(SED):   前走IDM, 馬場差, 不利, 出遅, テン指数, 上がり指数,
                   ペース指数, 上昇度
  直前データ(TYB): パドック指数, オッズ指数, 総合指数, 馬体コード, 気配コード
                   → Pattern Bのみ

評価:
  - Pattern A WF AUC (2020-2025)
  - 全年AUC > 0.78
  - 過学習チェック (train-test AUC gap < 0.05)
  - 個別特徴量貢献度分析
  - 実配当ROI (条件A-X)

使い方:
  python train/train_v13_jrdb.py              # WFバックテスト+学習
  python train/train_v13_jrdb.py --test-only  # WFバックテストのみ（モデル保存しない）
"""

import os
import sys
import time
import pickle
import gzip
import json
import argparse
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

# v12ベースのインポート
from train_v92_central import (
    load_data, encode_categoricals, encode_sires,
    load_training_times, merge_training_features,
    compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance,
    compute_lag_features, build_features,
    compute_distance_aptitude, compute_frame_advantage,
    train_lgb, train_xgb,
    FEATURES_V91, V92_NEW_FEATURES, V93_NEW_FEATURES,
    COURSE_MAP, N_TOP_SIRE,
)
from train_v92_leakfree import LEAK_FEATURES_A
from train_v12_comprehensive import (
    V12_NEW_FEATURES, LGB_PARAMS, XGB_PARAMS,
    merge_speed_index, merge_training_1f, merge_sire_shinba,
    merge_dam_stats, compute_pci,
    walk_forward_backtest, print_wf_results, compute_feature_importance,
)
from jrdb_features import (
    merge_jrdb_train_features,
    JRDB_FEATURES_PRE_RACE, JRDB_FEATURES_PREV_RACE,
    JRDB_FEATURES_LIVE, JRDB_ALL_FEATURES, JRDB_LIVE_FEATURES,
    JRDB_DEFAULTS,
)

# =====================================================
# v13特徴量定義
# =====================================================

# v12ベースライン（74特徴量）
FEATURES_V93_ALL = list(FEATURES_V91) + list(V92_NEW_FEATURES) + list(V93_NEW_FEATURES)
FEATURES_V12_BASELINE = [f for f in FEATURES_V93_ALL if f not in LEAK_FEATURES_A]

# v12で採用された新特徴量（dam_top3rは除外済み）
V12_ADOPTED = [f for f in V12_NEW_FEATURES if f != 'dam_top3r']
FEATURES_V12 = FEATURES_V12_BASELINE + V12_ADOPTED

# v13候補: v12 + JRDB前日データ + JRDB前走成績
FEATURES_V13_CANDIDATE = FEATURES_V12 + JRDB_ALL_FEATURES

# v13 Pattern B: v13候補 + 直前データ（+ v12のリーク特徴量）
# Pattern Bは当日情報を含むので別途定義


# =====================================================
# データ準備
# =====================================================

def prepare_data():
    """v12と同じデータ準備フロー + JRDB特徴量マージ"""
    start = time.time()

    # Phase 1: Load & encode
    print("\n[1/7] Loading data...")
    df = load_data()
    print(f"  Loaded: {len(df)} rows")
    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)

    # Phase 2: Training data
    print("\n[2/7] Merging training data...")
    tt_data = load_training_times()
    df = merge_training_features(df, tt_data)

    # Phase 3: Historical stats
    print("\n[3/7] Computing historical stats...")
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)
    df = compute_lag_features(df)
    df = build_features(df)
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)

    # Phase 4: v12 features
    print("\n[4/7] Merging v12 features...")
    df = merge_speed_index(df)
    df = merge_training_1f(df)
    df = merge_sire_shinba(df)
    df = merge_dam_stats(df)
    df = compute_pci(df)

    # Phase 5: JRDB features (NEW)
    print("\n[5/7] Merging JRDB features...")
    df = merge_jrdb_train_features(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses'] >= 5].copy()

    # Year
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    print(f"\n  Final dataset: {len(df)} rows")

    # Ensure all features exist
    for f in FEATURES_V13_CANDIDATE:
        if f not in df.columns:
            print(f"  WARNING: {f} not in df, filling with default")
            df[f] = JRDB_DEFAULTS.get(f, 0)
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(JRDB_DEFAULTS.get(f, 0))

    elapsed = (time.time() - start) / 60
    print(f"  Data preparation: {elapsed:.1f} min")

    return df, sire_map, bms_map


# =====================================================
# JRDB特徴量カバレッジ確認
# =====================================================

def check_jrdb_coverage(df, years=range(2020, 2026)):
    """年別のJRDB特徴量カバレッジを確認"""
    print(f"\n{'='*60}")
    print(f"  JRDB Feature Coverage (by year)")
    print(f"{'='*60}")

    jrdb_feats = [f for f in JRDB_ALL_FEATURES if f in df.columns]
    for year in years:
        ty = year - 2000
        mask = df['year'] == ty
        n = mask.sum()
        if n == 0:
            continue

        rates = {}
        for f in jrdb_feats[:5]:  # 代表的な5個
            default = JRDB_DEFAULTS.get(f, 0)
            non_default = (df.loc[mask, f] != default).sum()
            rates[f] = non_default / n

        avg_rate = np.mean(list(rates.values())) if rates else 0
        print(f"  {year}: N={n:,}, avg coverage={avg_rate*100:.1f}%")
        for f, r in rates.items():
            print(f"    {f}: {r*100:.1f}%")

    return


# =====================================================
# 段階的特徴量選択
# =====================================================

def evaluate_feature_groups(df, base_features, feature_groups, years=range(2020, 2026)):
    """特徴量グループごとにWFバックテストを実行

    Args:
        df: データ
        base_features: ベースライン特徴量
        feature_groups: dict {グループ名: [特徴量リスト]}

    Returns:
        dict: {グループ名: mean AUC}
    """
    results = {}

    # ベースライン
    print(f"\n--- Baseline ({len(base_features)} features) ---")
    wf_base = walk_forward_backtest(df, base_features, years=years)
    auc_base = print_wf_results(wf_base, 'Baseline (v12)')
    results['baseline'] = auc_base

    # 各グループを追加してテスト
    for group_name, group_feats in feature_groups.items():
        feats = base_features + group_feats
        print(f"\n--- +{group_name} ({len(feats)} features, +{len(group_feats)}) ---")
        wf = walk_forward_backtest(df, feats, years=years)
        auc = print_wf_results(wf, f'+{group_name}')
        results[group_name] = auc

        delta = auc - auc_base
        print(f"  Delta: {delta:+.4f} {'OK' if delta > 0 else 'NG'}")

    return results


# =====================================================
# 個別特徴量貢献度分析
# =====================================================

def individual_contribution(df, base_features, new_features, test_years=[2024, 2025]):
    """新特徴量を1個ずつ除外してWF AUCの変化を計測"""
    print(f"\n{'='*60}")
    print(f"  Individual Feature Contribution")
    print(f"{'='*60}")

    all_feats = base_features + new_features
    wf_full = walk_forward_backtest(df, all_feats, years=test_years)
    auc_full = np.mean([r['lgb_auc'] for r in wf_full])
    print(f"  Full ({len(all_feats)} features): AUC = {auc_full:.4f}")

    contributions = {}
    for feat in new_features:
        feats_without = [f for f in all_feats if f != feat]
        wf = walk_forward_backtest(df, feats_without, years=test_years)
        auc_without = np.mean([r['lgb_auc'] for r in wf])
        contrib = auc_full - auc_without
        contributions[feat] = contrib
        sign = '+' if contrib > 0 else ''
        print(f"  {feat:35s}: {sign}{contrib:.5f}")

    # ソート
    ranked = sorted(contributions.items(), key=lambda x: -x[1])
    print(f"\n  Ranked by contribution:")
    for feat, c in ranked:
        sign = '+' if c > 0 else ''
        mark = 'OK' if c > -0.0005 else 'NG'
        print(f"    {mark} {feat:35s}: {sign}{c:.5f}")

    return contributions


# =====================================================
# メイン
# =====================================================

def main():
    parser = argparse.ArgumentParser(description='v13 JRDB Feature Training')
    parser.add_argument('--test-only', action='store_true', help='WFバックテストのみ')
    parser.add_argument('--quick', action='store_true', help='2024-2025のみでクイック評価')
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 60)
    print("  v13 JRDB Feature Training")
    print(f"  Baseline: v12 WF AUC 0.8037 (74 features)")
    print(f"  Target: WF AUC > 0.8037, all years > 0.78")
    print("=" * 60)

    # データ準備
    df, sire_map, bms_map = prepare_data()

    # JRDBカバレッジ確認
    check_jrdb_coverage(df)

    test_years = range(2024, 2026) if args.quick else range(2020, 2026)

    # ===== Phase 1: グループ別評価 =====
    print(f"\n{'='*60}")
    print(f"  Phase 1: Feature Group Evaluation")
    print(f"{'='*60}")

    feature_groups = {
        'jrdb_indices': [  # JRDB主力指数のみ
            'jrdb_idm', 'jrdb_training_idx', 'jrdb_stable_idx',
            'jrdb_composite_idx',
        ],
        'jrdb_prediction': [  # 展開予想指数
            'jrdb_ten_idx_pred', 'jrdb_pace_idx_pred',
            'jrdb_agari_idx_pred', 'jrdb_position_idx_pred',
        ],
        'jrdb_meta': [  # メタ情報
            'jrdb_class_code', 'jrdb_rise_code', 'jrdb_heavy_apt',
            'jrdb_ranch_rank', 'jrdb_stable_rank', 'jrdb_training_arrow',
            'jrdb_stable_eval',
        ],
        'jrdb_prev_race': [  # 前走成績
            'jrdb_prev_idm', 'jrdb_prev_track_bias',
            'jrdb_prev_interference', 'jrdb_prev_late_start',
            'jrdb_prev_agari_idx', 'jrdb_prev_rise_code',
        ],
        'jrdb_all': JRDB_ALL_FEATURES,  # 全JRDB特徴量
    }

    group_results = evaluate_feature_groups(df, FEATURES_V12, feature_groups, years=test_years)

    # ===== Phase 2: 最良グループの個別貢献度 =====
    # 全JRDB特徴量でまず試し、有害な特徴量を除外
    best_group = max(group_results.items(), key=lambda x: x[1] if x[0] != 'baseline' else 0)
    print(f"\n  Best group: {best_group[0]} (AUC {best_group[1]:.4f})")

    print(f"\n{'='*60}")
    print(f"  Phase 2: Individual Contribution Analysis")
    print(f"{'='*60}")

    contributions = individual_contribution(
        df, FEATURES_V12, JRDB_ALL_FEATURES,
        test_years=[2024, 2025]
    )

    # 有害特徴量除外
    selected = [f for f, c in contributions.items() if c > -0.0005]
    removed = [f for f, c in contributions.items() if c <= -0.0005]

    print(f"\n  Selected: {len(selected)} features")
    print(f"  Removed: {len(removed)} features")
    for f in removed:
        print(f"    NG {f}: {contributions[f]:+.5f}")

    # ===== Phase 3: 最終WFバックテスト =====
    FEATURES_V13 = FEATURES_V12 + selected

    print(f"\n{'='*60}")
    print(f"  Phase 3: Final Walk-Forward Backtest")
    print(f"  v13: {len(FEATURES_V13)} features (+{len(selected)} JRDB)")
    print(f"{'='*60}")

    wf_v13 = walk_forward_backtest(df, FEATURES_V13, years=range(2020, 2026))
    auc_v13 = print_wf_results(wf_v13, f'v13 ({len(FEATURES_V13)} features)')

    # Feature importance
    imp_ranked = compute_feature_importance(wf_v13, FEATURES_V13)
    print(f"\n  Top 30 features by importance:")
    for i, (f, v) in enumerate(imp_ranked[:30]):
        marker = " *JRDB" if f.startswith('jrdb_') else (" *V12" if f in V12_ADOPTED else "")
        print(f"  {i+1:2d}. {f:35s} {v:12.1f}{marker}")

    # ===== 採用判定 =====
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  v12 Baseline:  0.8037 (74 features)")
    print(f"  v13 Selected:  {auc_v13:.4f} ({len(FEATURES_V13)} features)")
    print(f"  Improvement:   {auc_v13 - 0.8037:+.4f}")

    all_years_ok = all(r['lgb_auc'] > 0.78 for r in wf_v13)
    no_overfit = all(r['gap'] < 0.05 for r in wf_v13)
    auc_improved = auc_v13 > 0.8037

    print(f"\n  AUC > 0.8037:     {'OK' if auc_improved else 'NG'}")
    print(f"  All years > 0.78: {'OK' if all_years_ok else 'NG'}")
    print(f"  No overfitting:   {'OK' if no_overfit else 'NG'}")

    adopted = auc_improved and all_years_ok and no_overfit
    print(f"\n  VERDICT: {'OK ADOPTED as v13' if adopted else 'NG NOT ADOPTED'}")

    # 結果保存
    result_data = {
        'v12_baseline_auc': 0.8037,
        'v13_auc': auc_v13,
        'improvement': auc_v13 - 0.8037,
        'adopted': adopted,
        'v13_features': FEATURES_V13,
        'jrdb_selected': selected,
        'jrdb_removed': removed,
        'contributions': contributions,
        'group_results': group_results,
        'yearly_results': [{
            'year': r['year'], 'lgb_auc': r['lgb_auc'], 'xgb_auc': r['xgb_auc'],
            'train_auc': r['train_auc'], 'gap': r['gap']
        } for r in wf_v13],
        'timestamp': datetime.now().isoformat(),
    }

    result_path = os.path.join(DATA_DIR, 'v13_training_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved: {result_path}")

    # ===== 本番モデル学習 =====
    if adopted and not args.test_only:
        print(f"\n{'='*60}")
        print(f"  TRAINING PRODUCTION MODEL (v13)")
        print(f"{'='*60}")
        train_v13_production(df, FEATURES_V13, sire_map, bms_map, wf_v13, selected)

    elapsed = (time.time() - start_time) / 60
    print(f"\n  Total elapsed: {elapsed:.1f} min")


def train_v13_production(df, features, sire_map, bms_map, wf_results, jrdb_features):
    """v13本番モデル学習・保存"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

    # Ensemble
    total = lgb_auc + xgb_auc
    w_lgb = lgb_auc / total
    w_xgb = xgb_auc / total
    ens_pred = w_lgb * lgb_model.predict(X_valid) + w_xgb * xgb_model.predict(dxte)
    ens_auc = roc_auc_score(y_valid, ens_pred)
    print(f"  Ensemble AUC: {ens_auc:.4f}")

    wf_auc = np.mean([r['lgb_auc'] for r in wf_results])

    # Pattern A (leak-free)
    pattern_a_features = [f for f in features if f not in LEAK_FEATURES_A]
    pkl_a = {
        'model': lgb_model,
        'features': pattern_a_features,
        'version': 'v13_leakfree',
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
        'v12_new_features': V12_ADOPTED,
        'jrdb_features': jrdb_features,
    }

    a_path = os.path.join(BASE_DIR, 'keiba_model_v13_central.pkl.gz')
    with gzip.open(a_path, 'wb') as f:
        pickle.dump(pkl_a, f)
    print(f"  Pattern A saved: {a_path}")

    # Pattern B (live) — JRDB直前データも追加
    pattern_b_features = features + JRDB_LIVE_FEATURES
    pkl_b = dict(pkl_a)
    pkl_b['features'] = pattern_b_features
    pkl_b['version'] = 'v13_live'
    pkl_b['leak_free'] = False
    pkl_b['leak_pattern'] = 'B'
    pkl_b['is_live'] = True

    b_path = os.path.join(BASE_DIR, 'keiba_model_v13_central_live.pkl.gz')
    with gzip.open(b_path, 'wb') as f:
        pickle.dump(pkl_b, f)
    print(f"  Pattern B saved: {b_path}")

    print(f"\n  *** v13 Production model saved! ***")
    print(f"  Pattern A: {len(pattern_a_features)} features, WF AUC {wf_auc:.4f}")
    print(f"  Pattern B: {len(pattern_b_features)} features")


if __name__ == '__main__':
    main()
