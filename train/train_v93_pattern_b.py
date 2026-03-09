#!/usr/bin/env python
"""KEIBA AI v9.3 Pattern B Training (Central) - 実運用専用モデル

設計思想:
- Pattern A（リークフリー）をベースに、当日情報を追加した実運用モデル
- 学習データ: 過去の全レース（当日情報込み）
- AUC評価: Pattern Aで行う（モデルの真の実力評価）
- Pattern BのAUCは参考値

追加する当日特徴量:
- odds_log: 当日単勝オッズ（log変換）
- horse_weight: 当日馬体重
- condition_enc: 当日馬場状態
- weight_change / weight_change_abs: 馬体重増減
- weight_cat / weight_cat_dist: 体重カテゴリ派生
- cond_surface: 馬場×芝ダート
- weather_enc: 天候（晴/曇/雨/雪）
- pop_rank: 人気順位
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features,
    compute_distance_aptitude, compute_frame_advantage,
    COURSE_MAP, N_TOP_SIRE,
    FEATURES_V93,
    train_lgb, train_xgb, show_feature_importance,
)
from train_v92_leakfree import (
    LEAK_FEATURES_A,
    classify_condition, calc_trio_bets, backtest_condition,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
OUTPUT_DIR = BASE_DIR

# Pattern B: 全V9.3特徴量 + 当日追加特徴量
# LEAK_FEATURES_A を除外せず全て使う
LIVE_EXTRA_FEATURES = [
    'weather_enc',     # 天候: 晴=0, 曇=1, 雨=2, 小雨=2, 雪=3
    'pop_rank',        # 人気順位（オッズから算出）
]

FEATURES_PATTERN_B = FEATURES_V93 + LIVE_EXTRA_FEATURES
FEATURES_PATTERN_B_PKL = [f if f != 'num_horses_val' else 'num_horses' for f in FEATURES_PATTERN_B]

# Pattern A features (for reference AUC)
FEATURES_PATTERN_A = [f for f in FEATURES_V93 if f not in LEAK_FEATURES_A]

# Weather encoding
WEATHER_MAP = {'晴': 0, '曇': 1, '小雨': 2, '雨': 2, '雪': 3}


def add_weather_feature(df):
    """天候特徴量を追加（学習データ用）"""
    if 'weather' in df.columns:
        df['weather_enc'] = df['weather'].map(WEATHER_MAP).fillna(0).astype(int)
    else:
        # 天候データがない場合はデフォルト（晴=0）
        df['weather_enc'] = 0
    return df


def add_pop_rank(df):
    """人気順位を追加（オッズから算出）"""
    if 'tansho_odds' in df.columns:
        # レースごとにオッズ順位を計算
        df['pop_rank'] = df.groupby('race_id_str')['tansho_odds'].rank(method='min').fillna(8)
    else:
        df['pop_rank'] = 8
    return df


def main():
    print("=" * 70)
    print("  KEIBA AI v9.3 PATTERN B TRAINING (CENTRAL) - 実運用モデル")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("  当日情報（オッズ・馬体重・馬場・天候）込みで学習")
    print("=" * 70)

    # === Data Loading ===
    df = load_data()
    lap_df = load_lap_data()
    if lap_df is not None:
        df = df.merge(lap_df, on='race_id_str', how='left')
        matched = df['race_first3f'].notna().sum()
        print(f"  Lap data merged: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")

    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)

    tt_data = load_training_times()
    df = merge_training_features(df, tt_data)
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)
    df = compute_lag_features(df)

    print("Building features...")
    df = build_features(df)
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)

    # 当日追加特徴量
    df = add_weather_feature(df)
    df = add_pop_rank(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()
    y = df['target'].values

    # Time-based split
    max_year = df['year_full'].max()
    valid_mask = df['year_full'] >= (max_year - 1)
    train_mask = ~valid_mask
    y_train, y_valid = y[train_mask], y[valid_mask]

    print(f"\nTrain: {train_mask.sum()}, Valid: {valid_mask.sum()}")
    print(f"Target rate: train={y_train.mean():.3f}, valid={y_valid.mean():.3f}")

    # === Pattern A Baseline (reference AUC) ===
    print("\n" + "=" * 70)
    print("  REFERENCE: Pattern A (リークフリー) AUC")
    print(f"  Features: {len(FEATURES_PATTERN_A)}")
    print("=" * 70)

    for f in FEATURES_PATTERN_A:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X_a = df[FEATURES_PATTERN_A].values
    lgb_a = train_lgb(X_a[train_mask], y_train, X_a[valid_mask], y_valid, FEATURES_PATTERN_A)
    lgb_a_pred = lgb_a.predict(X_a[valid_mask])
    lgb_a_auc = roc_auc_score(y_valid, lgb_a_pred)

    import xgboost as xgb
    xgb_a = train_xgb(X_a[train_mask], y_train, X_a[valid_mask], y_valid)
    xgb_a_pred = xgb_a.predict(xgb.DMatrix(X_a[valid_mask]))
    xgb_a_auc = roc_auc_score(y_valid, xgb_a_pred)

    total_a = lgb_a_auc + xgb_a_auc
    w_lgb_a = lgb_a_auc / total_a
    w_xgb_a = xgb_a_auc / total_a
    a_pred = lgb_a_pred * w_lgb_a + xgb_a_pred * w_xgb_a
    a_auc = roc_auc_score(y_valid, a_pred)
    print(f"\n  Pattern A AUC: LGB {lgb_a_auc:.4f} / XGB {xgb_a_auc:.4f} / Ensemble {a_auc:.4f}")

    # === Pattern B (当日情報込み) ===
    print("\n" + "=" * 70)
    print("  PATTERN B: 当日情報込み（実運用モデル）")
    print(f"  Features: {len(FEATURES_PATTERN_B)}")
    print(f"  当日追加: odds_log, horse_weight, condition_enc, weight_change,")
    print(f"            weight_cat, weight_cat_dist, cond_surface, weather_enc, pop_rank")
    print("=" * 70)

    for f in FEATURES_PATTERN_B:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X_b = df[FEATURES_PATTERN_B].values
    lgb_b = train_lgb(X_b[train_mask], y_train, X_b[valid_mask], y_valid, FEATURES_PATTERN_B)
    lgb_b_pred = lgb_b.predict(X_b[valid_mask])
    lgb_b_auc = roc_auc_score(y_valid, lgb_b_pred)

    xgb_b = train_xgb(X_b[train_mask], y_train, X_b[valid_mask], y_valid)
    xgb_b_pred = xgb_b.predict(xgb.DMatrix(X_b[valid_mask]))
    xgb_b_auc = roc_auc_score(y_valid, xgb_b_pred)

    total_b = lgb_b_auc + xgb_b_auc
    w_lgb_b = lgb_b_auc / total_b
    w_xgb_b = xgb_b_auc / total_b
    b_pred = lgb_b_pred * w_lgb_b + xgb_b_pred * w_xgb_b
    b_auc = roc_auc_score(y_valid, b_pred)
    print(f"\n  Pattern B AUC: LGB {lgb_b_auc:.4f} / XGB {xgb_b_auc:.4f} / Ensemble {b_auc:.4f}")
    print(f"  Pattern A → B improvement: {b_auc - a_auc:+.4f} (参考値)")

    fi_b = show_feature_importance(lgb_b, FEATURES_PATTERN_B, "Pattern B (Live)")

    # === Feature importance analysis: day-of-race features ===
    print("\n  当日特徴量の重要度:")
    day_features = list(LEAK_FEATURES_A) + LIVE_EXTRA_FEATURES
    for f in day_features:
        if f in fi_b:
            print(f"    {f}: {fi_b[f]:,.0f}")

    # === Condition Backtest (using Pattern B predictions) ===
    valid_df = df[valid_mask].copy()
    bt_b = backtest_condition(valid_df, lgb_b, xgb_b, FEATURES_PATTERN_B, w_lgb_b, w_xgb_b, "Pattern B")

    # === Summary ===
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  {'Model':<35} {'LGB AUC':>10} {'XGB AUC':>10} {'Ensemble':>10}")
    print(f"  {'-' * 65}")
    print(f"  {'Pattern A (リークフリー/評価用)':<35} {lgb_a_auc:>10.4f} {xgb_a_auc:>10.4f} {a_auc:>10.4f}")
    print(f"  {'Pattern B (当日情報込み/実運用)':<35} {lgb_b_auc:>10.4f} {xgb_b_auc:>10.4f} {b_auc:>10.4f}")
    print(f"\n  ※モデル評価はPattern A AUCで行う。Pattern B AUCは参考値。")

    # === Save Pattern B model ===
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pkl = {
        'model': lgb_b,
        'features': FEATURES_PATTERN_B_PKL,
        'version': 'v9.3_live',
        'auc': lgb_b_auc,
        'ensemble_auc': b_auc,
        'pattern_a_auc': a_auc,  # リークフリーAUC（真の評価）
        'leak_free': False,
        'leak_pattern': 'B',
        'live_features': sorted(list(LEAK_FEATURES_A) + LIVE_EXTRA_FEATURES),
        'sire_map': sire_map,
        'bms_map': bms_map,
        'n_top_encode': N_TOP_SIRE,
        'trained_at': now,
        'n_train': int(train_mask.sum()),
        'n_valid': int(valid_mask.sum()),
        'model_type': 'central_live',
        'xgb_model': xgb_b,
        'mlp_model': None,
        'mlp_scaler': None,
        'ensemble_weights': {'lgb': w_lgb_b, 'xgb': w_xgb_b, 'mlp': 0},
        'course_map': dict(COURSE_MAP),
        'weather_map': WEATHER_MAP,
        'condition_backtest': bt_b,
    }

    live_path = os.path.join(OUTPUT_DIR, 'keiba_model_v9_central_live.pkl')
    with open(live_path, 'wb') as f:
        pickle.dump(pkl, f)
    print(f"\n  Saved: {live_path}")

    # Save results JSON
    results = {
        'generated_at': now,
        'pattern_a': {'lgb': lgb_a_auc, 'xgb': xgb_a_auc, 'ensemble': a_auc,
                       'n_features': len(FEATURES_PATTERN_A)},
        'pattern_b': {'lgb': lgb_b_auc, 'xgb': xgb_b_auc, 'ensemble': b_auc,
                       'n_features': len(FEATURES_PATTERN_B)},
        'improvement': b_auc - a_auc,
        'live_features': sorted(list(LEAK_FEATURES_A) + LIVE_EXTRA_FEATURES),
        'feature_importance_top20': dict(sorted(fi_b.items(), key=lambda x: -x[1])[:20]),
    }

    comp_path = os.path.join(OUTPUT_DIR, 'v93_pattern_b_results.json')
    with open(comp_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved results: {comp_path}")

    print("\n  Pattern B training complete!")
    print("  ※実運用ではこのモデル(keiba_model_v9_central_live.pkl)を使用")
    print("  ※モデル更新判定はPattern AのAUCで行う")
    return results


if __name__ == '__main__':
    main()
