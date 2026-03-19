#!/usr/bin/env python
"""オッズ依存度低減実験 v2
Pattern Bのodds_log/pop_rank依存度を73.8%→65%以下に下げつつAUC 0.8450以上を維持する。

手法:
1. Noise injection: odds_logに±10%ランダムノイズ
2. Feature fraction down: odds特徴量を含むサンプリング比率を下げる
3. Column drop: pop_rank除去 (odds_logのみ保持)
4. Two-stage: オッズなし事前学習→オッズありファインチューニング

結果: data/odds_reduction_v2_report.json
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features,
    compute_distance_aptitude, compute_frame_advantage,
    compute_running_style_features,
    COURSE_MAP, N_TOP_SIRE,
    FEATURES_V93_PACE,
    train_lgb, train_xgb, show_feature_importance,
)
from train_v92_leakfree import LEAK_FEATURES_A
from train_v93_pattern_b import (
    LIVE_EXTRA_FEATURES, add_weather_feature, add_pop_rank,
)

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')

FEATURES_B = FEATURES_V93_PACE + LIVE_EXTRA_FEATURES
ODDS_FEATURES = ['odds_log', 'pop_rank', 'prev_odds_log']
TARGET_AUC = 0.8450
TARGET_DEPENDENCY = 65.0


def calc_odds_dependency(model, feature_names):
    """オッズ関連特徴量の重要度比率を計算"""
    importance = model.feature_importance(importance_type='gain')
    fi = dict(zip(feature_names, importance))
    total = sum(importance)
    if total == 0:
        return 0.0, fi
    odds_imp = sum(fi.get(f, 0) for f in ODDS_FEATURES)
    dependency = odds_imp / total * 100
    return dependency, fi


def main():
    start_time = time.time()
    print("=" * 70)
    print("  ODDS DEPENDENCY REDUCTION v2")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Target: AUC >= {TARGET_AUC} AND dependency <= {TARGET_DEPENDENCY}%")
    print("=" * 70)

    # === Data Loading ===
    df = load_data()
    lap_df = load_lap_data()
    if lap_df is not None:
        df = df.merge(lap_df, on='race_id_str', how='left')
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
    df = compute_running_style_features(df)
    df = add_weather_feature(df)
    df = add_pop_rank(df)

    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()
    y = df['target'].values

    max_year = df['year_full'].max()
    valid_mask = df['year_full'] >= (max_year - 1)
    train_mask = ~valid_mask
    y_train, y_valid = y[train_mask], y[valid_mask]

    for f in FEATURES_B:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"\nTrain: {train_mask.sum()}, Valid: {valid_mask.sum()}")
    print(f"Features: {len(FEATURES_B)}")

    results = {}

    # ============================================================
    #  BASELINE: Current Pattern B
    # ============================================================
    print("\n" + "=" * 70)
    print("  [0] BASELINE: Current Pattern B")
    print("=" * 70)

    X_b = df[FEATURES_B].values
    lgb_base = train_lgb(X_b[train_mask], y_train, X_b[valid_mask], y_valid, FEATURES_B)
    lgb_base_pred = lgb_base.predict(X_b[valid_mask])
    lgb_base_auc = roc_auc_score(y_valid, lgb_base_pred)

    import xgboost as xgb
    xgb_base = train_xgb(X_b[train_mask], y_train, X_b[valid_mask], y_valid)
    xgb_base_pred = xgb_base.predict(xgb.DMatrix(X_b[valid_mask]))
    xgb_base_auc = roc_auc_score(y_valid, xgb_base_pred)

    total_w = lgb_base_auc + xgb_base_auc
    base_pred = lgb_base_pred * (lgb_base_auc / total_w) + xgb_base_pred * (xgb_base_auc / total_w)
    base_auc = roc_auc_score(y_valid, base_pred)

    base_dep, base_fi = calc_odds_dependency(lgb_base, FEATURES_B)
    print(f"  AUC: {base_auc:.4f}, Odds dependency: {base_dep:.1f}%")

    results['baseline'] = {
        'auc': base_auc, 'lgb_auc': lgb_base_auc, 'xgb_auc': xgb_base_auc,
        'odds_dependency': round(base_dep, 1),
        'odds_log_imp': round(base_fi.get('odds_log', 0), 1),
        'pop_rank_imp': round(base_fi.get('pop_rank', 0), 1),
        'prev_odds_log_imp': round(base_fi.get('prev_odds_log', 0), 1),
    }

    # ============================================================
    #  METHOD 1: Noise Injection (±10% on odds_log)
    # ============================================================
    print("\n" + "=" * 70)
    print("  [1] NOISE INJECTION: odds_log ±10% random noise")
    print("=" * 70)

    df_noisy = df.copy()
    np.random.seed(42)
    noise = 1.0 + np.random.uniform(-0.10, 0.10, size=len(df_noisy))
    df_noisy['odds_log'] = df_noisy['odds_log'] * noise
    # Also add noise to pop_rank (shuffle within small groups)
    noise_pr = np.random.uniform(-0.5, 0.5, size=len(df_noisy))
    df_noisy['pop_rank'] = (df_noisy['pop_rank'] + noise_pr).clip(1)

    X_noisy = df_noisy[FEATURES_B].values
    lgb_m1 = train_lgb(X_noisy[train_mask], y_train, X_b[valid_mask], y_valid, FEATURES_B)
    lgb_m1_pred = lgb_m1.predict(X_b[valid_mask])
    lgb_m1_auc = roc_auc_score(y_valid, lgb_m1_pred)

    xgb_m1 = train_xgb(X_noisy[train_mask], y_train, X_b[valid_mask], y_valid)
    xgb_m1_pred = xgb_m1.predict(xgb.DMatrix(X_b[valid_mask]))
    xgb_m1_auc = roc_auc_score(y_valid, xgb_m1_pred)

    total_m1 = lgb_m1_auc + xgb_m1_auc
    m1_pred = lgb_m1_pred * (lgb_m1_auc / total_m1) + xgb_m1_pred * (xgb_m1_auc / total_m1)
    m1_auc = roc_auc_score(y_valid, m1_pred)

    m1_dep, m1_fi = calc_odds_dependency(lgb_m1, FEATURES_B)
    print(f"  AUC: {m1_auc:.4f} (diff: {m1_auc - base_auc:+.4f}), Odds dep: {m1_dep:.1f}%")

    results['noise_injection'] = {
        'auc': m1_auc, 'lgb_auc': lgb_m1_auc, 'xgb_auc': xgb_m1_auc,
        'odds_dependency': round(m1_dep, 1),
        'auc_diff': round(m1_auc - base_auc, 4),
    }

    # ============================================================
    #  METHOD 2: Feature fraction down for odds
    # ============================================================
    print("\n" + "=" * 70)
    print("  [2] FEATURE FRACTION DOWN: 0.8 → 0.5 + higher regularization")
    print("=" * 70)

    params_m2 = {
        'feature_fraction': 0.5,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
    }
    lgb_m2 = train_lgb(X_b[train_mask], y_train, X_b[valid_mask], y_valid, FEATURES_B, params_override=params_m2)
    lgb_m2_pred = lgb_m2.predict(X_b[valid_mask])
    lgb_m2_auc = roc_auc_score(y_valid, lgb_m2_pred)

    m2_dep, m2_fi = calc_odds_dependency(lgb_m2, FEATURES_B)
    print(f"  LGB AUC: {lgb_m2_auc:.4f}, Odds dep: {m2_dep:.1f}%")

    results['feature_fraction_down'] = {
        'lgb_auc': lgb_m2_auc,
        'odds_dependency': round(m2_dep, 1),
        'auc_diff': round(lgb_m2_auc - lgb_base_auc, 4),
    }

    # ============================================================
    #  METHOD 3: Drop pop_rank (keep odds_log only)
    # ============================================================
    print("\n" + "=" * 70)
    print("  [3] DROP pop_rank: keep only odds_log")
    print("=" * 70)

    features_m3 = [f for f in FEATURES_B if f != 'pop_rank']
    X_m3 = df[features_m3].values
    lgb_m3 = train_lgb(X_m3[train_mask], y_train, X_m3[valid_mask], y_valid, features_m3)
    lgb_m3_pred = lgb_m3.predict(X_m3[valid_mask])
    lgb_m3_auc = roc_auc_score(y_valid, lgb_m3_pred)

    xgb_m3 = train_xgb(X_m3[train_mask], y_train, X_m3[valid_mask], y_valid)
    xgb_m3_pred = xgb_m3.predict(xgb.DMatrix(X_m3[valid_mask]))
    xgb_m3_auc = roc_auc_score(y_valid, xgb_m3_pred)

    total_m3 = lgb_m3_auc + xgb_m3_auc
    m3_pred = lgb_m3_pred * (lgb_m3_auc / total_m3) + xgb_m3_pred * (xgb_m3_auc / total_m3)
    m3_auc = roc_auc_score(y_valid, m3_pred)

    m3_dep, m3_fi = calc_odds_dependency(lgb_m3, features_m3)
    print(f"  AUC: {m3_auc:.4f} (diff: {m3_auc - base_auc:+.4f}), Odds dep: {m3_dep:.1f}%")

    results['drop_pop_rank'] = {
        'auc': m3_auc, 'lgb_auc': lgb_m3_auc, 'xgb_auc': xgb_m3_auc,
        'odds_dependency': round(m3_dep, 1),
        'auc_diff': round(m3_auc - base_auc, 4),
        'n_features': len(features_m3),
    }

    # ============================================================
    #  METHOD 4: Two-stage (pre-train without odds → fine-tune with odds)
    # ============================================================
    print("\n" + "=" * 70)
    print("  [4] TWO-STAGE: pre-train no-odds → fine-tune with odds")
    print("=" * 70)

    features_no_odds = [f for f in FEATURES_B if f not in ['odds_log', 'pop_rank']]
    X_no_odds = df[features_no_odds].values

    # Stage 1: Pre-train without odds (500 rounds)
    lgb_stage1 = train_lgb(X_no_odds[train_mask], y_train,
                           X_no_odds[valid_mask], y_valid, features_no_odds,
                           params_override={'num_leaves': 31, 'learning_rate': 0.03})
    stage1_pred = lgb_stage1.predict(X_no_odds[valid_mask])
    stage1_auc = roc_auc_score(y_valid, stage1_pred)
    print(f"  Stage 1 (no-odds): AUC {stage1_auc:.4f}")

    # Stage 2: Fine-tune with odds (use stage1 predictions as additional feature)
    df['stage1_pred'] = lgb_stage1.predict(X_no_odds)
    features_m4 = FEATURES_B + ['stage1_pred']
    for f in features_m4:
        if f not in df.columns:
            df[f] = 0
    X_m4 = df[features_m4].values

    lgb_m4 = train_lgb(X_m4[train_mask], y_train, X_m4[valid_mask], y_valid, features_m4,
                       params_override={'learning_rate': 0.03, 'num_leaves': 47})
    lgb_m4_pred = lgb_m4.predict(X_m4[valid_mask])
    lgb_m4_auc = roc_auc_score(y_valid, lgb_m4_pred)

    m4_dep, m4_fi = calc_odds_dependency(lgb_m4, features_m4)
    print(f"  Stage 2 (with odds+stage1): AUC {lgb_m4_auc:.4f}, Odds dep: {m4_dep:.1f}%")

    results['two_stage'] = {
        'lgb_auc': lgb_m4_auc,
        'stage1_auc': stage1_auc,
        'odds_dependency': round(m4_dep, 1),
        'auc_diff': round(lgb_m4_auc - lgb_base_auc, 4),
        'stage1_importance': round(m4_fi.get('stage1_pred', 0), 1),
    }

    # ============================================================
    #  METHOD 5: Noise + Feature fraction combined
    # ============================================================
    print("\n" + "=" * 70)
    print("  [5] COMBINED: Noise injection + regularization")
    print("=" * 70)

    params_m5 = {
        'feature_fraction': 0.6,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
    }
    lgb_m5 = train_lgb(X_noisy[train_mask], y_train, X_b[valid_mask], y_valid, FEATURES_B, params_override=params_m5)
    lgb_m5_pred = lgb_m5.predict(X_b[valid_mask])
    lgb_m5_auc = roc_auc_score(y_valid, lgb_m5_pred)

    xgb_m5 = train_xgb(X_noisy[train_mask], y_train, X_b[valid_mask], y_valid)
    xgb_m5_pred = xgb_m5.predict(xgb.DMatrix(X_b[valid_mask]))
    xgb_m5_auc = roc_auc_score(y_valid, xgb_m5_pred)

    total_m5 = lgb_m5_auc + xgb_m5_auc
    m5_pred = lgb_m5_pred * (lgb_m5_auc / total_m5) + xgb_m5_pred * (xgb_m5_auc / total_m5)
    m5_auc = roc_auc_score(y_valid, m5_pred)

    m5_dep, m5_fi = calc_odds_dependency(lgb_m5, FEATURES_B)
    print(f"  AUC: {m5_auc:.4f} (diff: {m5_auc - base_auc:+.4f}), Odds dep: {m5_dep:.1f}%")

    results['combined_noise_reg'] = {
        'auc': m5_auc, 'lgb_auc': lgb_m5_auc, 'xgb_auc': xgb_m5_auc,
        'odds_dependency': round(m5_dep, 1),
        'auc_diff': round(m5_auc - base_auc, 4),
    }

    # Clean up temp column
    df.drop(columns=['stage1_pred'], errors='ignore', inplace=True)

    # ============================================================
    #  SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Method':<30} {'AUC':>8} {'Dep%':>6} {'AUC>=0.845':>10} {'Dep<=65':>8} {'PASS':>6}")
    print(f"  {'-' * 68}")

    best_method = None
    best_auc = 0

    for method, r in results.items():
        auc_val = r.get('auc', r.get('lgb_auc', 0))
        dep_val = r.get('odds_dependency', 100)
        auc_ok = auc_val >= TARGET_AUC
        dep_ok = dep_val <= TARGET_DEPENDENCY
        passed = auc_ok and dep_ok
        print(f"  {method:<30} {auc_val:>8.4f} {dep_val:>5.1f}% {'OK' if auc_ok else 'NG':>10} {'OK' if dep_ok else 'NG':>8} {'PASS' if passed else 'FAIL':>6}")
        if passed and auc_val > best_auc:
            best_auc = auc_val
            best_method = method

    if best_method:
        print(f"\n  >>> BEST METHOD: {best_method} (AUC {best_auc:.4f})")
        print(f"  >>> 本番採用可能")
    else:
        print(f"\n  >>> No method met both criteria.")
        print(f"  >>> 依存度低減は実験結果として記録。本番採用なし。")

    # Save report
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = {
        'generated_at': now,
        'target_auc': TARGET_AUC,
        'target_dependency': TARGET_DEPENDENCY,
        'results': results,
        'best_method': best_method,
        'best_auc': best_auc if best_method else None,
        'conclusion': f'{best_method} adopted' if best_method else 'No method met criteria',
    }

    report_path = os.path.join(DATA_DIR, 'odds_reduction_v2_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Report saved: {report_path}")

    elapsed = time.time() - start_time
    print(f"\n  Completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == '__main__':
    main()
