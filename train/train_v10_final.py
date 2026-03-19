#!/usr/bin/env python
"""KEIBA AI v10 Final Ensemble Training (Central)
V9.3ペース特徴量統合 + LightGBM + XGBoost + CatBoost 3モデルアンサンブル
- Pattern A: リークフリー評価用
- Pattern B: 当日情報込み実運用
- スタッキング（LogisticRegression meta-learner）
- ウォークフォワード 2020-2025
- 結果: data/v10_training_report.json
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
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb_lib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features,
    compute_distance_aptitude, compute_frame_advantage,
    compute_running_style_features,
    COURSE_MAP, N_TOP_SIRE,
    FEATURES_V93_PACE, FEATURES_V93_PACE_PKL,
    train_lgb, train_xgb,
)
from train_v92_leakfree import (
    LEAK_FEATURES_A,
    classify_condition, calc_trio_bets, backtest_condition,
)
from train_v93_pattern_b import (
    LIVE_EXTRA_FEATURES, WEATHER_MAP,
    add_weather_feature, add_pop_rank,
)

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = BASE_DIR

# Pattern A: leak-free
FEATURES_A = [f for f in FEATURES_V93_PACE if f not in LEAK_FEATURES_A]
FEATURES_A_PKL = [f if f != 'num_horses_val' else 'num_horses' for f in FEATURES_A]

# Pattern B: all + live extras
FEATURES_B = FEATURES_V93_PACE + LIVE_EXTRA_FEATURES
FEATURES_B_PKL = [f if f != 'num_horses_val' else 'num_horses' for f in FEATURES_B]

# Current baseline (V9.3 Pattern A)
BASELINE_AUC = 0.8095


def train_catboost(X_train, y_train, X_valid, y_valid, feature_names):
    """Train CatBoost model."""
    from catboost import CatBoostClassifier, Pool
    pool_train = Pool(X_train, label=y_train, feature_names=feature_names)
    pool_valid = Pool(X_valid, label=y_valid, feature_names=feature_names)

    cb = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=100,
        eval_metric='AUC',
        early_stopping_rounds=50,
        task_type='CPU',
    )
    cb.fit(pool_train, eval_set=pool_valid, verbose=100)
    pred = cb.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, pred)
    return cb, auc, pred


def train_fold_v10(X_train, y_train, X_valid, y_valid, feature_names):
    """Train LGB + XGB + CatBoost on one fold."""
    # LightGBM
    lgb_model = train_lgb(X_train, y_train, X_valid, y_valid, feature_names)
    lgb_pred = lgb_model.predict(X_valid)
    lgb_auc = roc_auc_score(y_valid, lgb_pred)

    # XGBoost
    xgb_model = train_xgb(X_train, y_train, X_valid, y_valid)
    xgb_pred = xgb_model.predict(xgb_lib.DMatrix(X_valid))
    xgb_auc = roc_auc_score(y_valid, xgb_pred)

    # CatBoost
    cb_model, cb_auc, cb_pred = train_catboost(
        X_train, y_train, X_valid, y_valid, feature_names)

    return {
        'lgb': {'model': lgb_model, 'pred': lgb_pred, 'auc': lgb_auc},
        'xgb': {'model': xgb_model, 'pred': xgb_pred, 'auc': xgb_auc},
        'cb':  {'model': cb_model,  'pred': cb_pred,  'auc': cb_auc},
    }


def compute_ensemble_v10(fold, y_valid):
    """Compute weighted average and stacking ensemble."""
    lgb_pred = fold['lgb']['pred']
    xgb_pred = fold['xgb']['pred']
    cb_pred = fold['cb']['pred']

    lgb_auc = fold['lgb']['auc']
    xgb_auc = fold['xgb']['auc']
    cb_auc = fold['cb']['auc']

    # Weighted average (by AUC)
    total_auc = lgb_auc + xgb_auc + cb_auc
    w_lgb = lgb_auc / total_auc
    w_xgb = xgb_auc / total_auc
    w_cb = cb_auc / total_auc

    avg_pred = lgb_pred * w_lgb + xgb_pred * w_xgb + cb_pred * w_cb
    avg_auc = roc_auc_score(y_valid, avg_pred)

    # 2-model ensemble (LGB+XGB only, for comparison)
    total_2 = lgb_auc + xgb_auc
    avg2_pred = lgb_pred * (lgb_auc / total_2) + xgb_pred * (xgb_auc / total_2)
    avg2_auc = roc_auc_score(y_valid, avg2_pred)

    # Stacking (LogisticRegression meta-learner)
    meta_X = np.column_stack([lgb_pred, xgb_pred, cb_pred])
    from sklearn.model_selection import cross_val_predict
    meta_lr = LogisticRegression(C=1.0, random_state=42, max_iter=500)
    try:
        stack_pred = cross_val_predict(
            meta_lr, meta_X, y_valid, cv=3, method='predict_proba')[:, 1]
        stack_auc = roc_auc_score(y_valid, stack_pred)
    except Exception:
        stack_auc = avg_auc
        stack_pred = avg_pred

    meta_lr.fit(meta_X, y_valid)

    return {
        'weights': {'lgb': w_lgb, 'xgb': w_xgb, 'cb': w_cb},
        'avg_auc': avg_auc,
        'avg2_auc': avg2_auc,
        'stack_auc': stack_auc,
        'meta_model': meta_lr,
        'avg_pred': avg_pred,
        'stack_pred': stack_pred,
    }


def walk_forward_v10(df, features, test_years=None):
    """Walk-forward validation 2020-2025."""
    if test_years is None:
        test_years = list(range(2020, 2026))

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD V10 (LGB+XGB+CatBoost)")
    print(f"  Features: {len(features)}")
    print(f"  Test years: {test_years}")
    print(f"{'='*70}")

    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    all_y = []
    all_preds = {'lgb': [], 'xgb': [], 'cb': [], 'avg': [], 'avg2': [], 'stack': []}
    fold_details = []

    for test_year in test_years:
        train_mask = df['year_full'] < test_year
        test_mask = df['year_full'] == test_year
        if test_mask.sum() == 0:
            continue

        X_train = df.loc[train_mask, features].values
        y_train = df.loc[train_mask, 'target'].values
        X_test = df.loc[test_mask, features].values
        y_test = df.loc[test_mask, 'target'].values

        print(f"\n  Fold {test_year}: train={len(X_train)}, test={len(X_test)}")
        t0 = time.time()

        fold = train_fold_v10(X_train, y_train, X_test, y_test, features)
        ens = compute_ensemble_v10(fold, y_test)

        elapsed = time.time() - t0
        print(f"    LGB={fold['lgb']['auc']:.4f}  XGB={fold['xgb']['auc']:.4f}  "
              f"CB={fold['cb']['auc']:.4f}  AVG3={ens['avg_auc']:.4f}  "
              f"AVG2={ens['avg2_auc']:.4f}  STACK={ens['stack_auc']:.4f}  ({elapsed:.0f}s)")

        fold_details.append({
            'year': test_year,
            'lgb_auc': fold['lgb']['auc'],
            'xgb_auc': fold['xgb']['auc'],
            'cb_auc': fold['cb']['auc'],
            'avg3_auc': ens['avg_auc'],
            'avg2_auc': ens['avg2_auc'],
            'stack_auc': ens['stack_auc'],
            'n_test': len(X_test),
        })

        all_y.extend(y_test.tolist())
        all_preds['lgb'].extend(fold['lgb']['pred'].tolist())
        all_preds['xgb'].extend(fold['xgb']['pred'].tolist())
        all_preds['cb'].extend(fold['cb']['pred'].tolist())
        all_preds['avg'].extend(ens['avg_pred'].tolist())
        all_preds['avg2'].extend((fold['lgb']['pred'] * (fold['lgb']['auc'] / (fold['lgb']['auc'] + fold['xgb']['auc'])) +
                                   fold['xgb']['pred'] * (fold['xgb']['auc'] / (fold['lgb']['auc'] + fold['xgb']['auc']))).tolist())
        all_preds['stack'].extend(ens['stack_pred'].tolist())

    # Overall AUC
    y_all = np.array(all_y)
    overall = {}
    for key in all_preds:
        overall[key] = roc_auc_score(y_all, np.array(all_preds[key]))

    print(f"\n  {'='*50}")
    print(f"  OVERALL WF AUC")
    print(f"  {'='*50}")
    for k, v in sorted(overall.items(), key=lambda x: -x[1]):
        marker = " <<<" if v == max(overall.values()) else ""
        print(f"  {k:10s}: {v:.4f}{marker}")

    return fold_details, overall


def main():
    start_time = time.time()
    print("=" * 70)
    print("  KEIBA AI v10 FINAL ENSEMBLE TRAINING")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Models: LightGBM + XGBoost + CatBoost")
    print(f"  Baseline: Pattern A AUC {BASELINE_AUC}")
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
    df = compute_running_style_features(df)

    # Weather/pop_rank for Pattern B
    df = add_weather_feature(df)
    df = add_pop_rank(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()

    print(f"\n  Final dataset: {len(df)} rows, {df['race_id_str'].nunique()} races")
    print(f"  Pattern A features: {len(FEATURES_A)}")
    print(f"  Pattern B features: {len(FEATURES_B)}")

    # ============================================================
    #  WALK-FORWARD Pattern A
    # ============================================================
    print("\n" + "=" * 70)
    print("  [1/4] WALK-FORWARD PATTERN A")
    print("=" * 70)

    wf_details_a, wf_overall_a = walk_forward_v10(df, FEATURES_A)

    # ============================================================
    #  FINAL MODEL Pattern A
    # ============================================================
    print("\n" + "=" * 70)
    print("  [2/4] FINAL MODEL PATTERN A")
    print("=" * 70)

    max_year = df['year_full'].max()
    valid_mask = df['year_full'] >= (max_year - 1)
    train_mask = ~valid_mask

    for f in FEATURES_A:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X_train_a = df.loc[train_mask, FEATURES_A].values
    y_train = df.loc[train_mask, 'target'].values
    X_valid_a = df.loc[valid_mask, FEATURES_A].values
    y_valid = df.loc[valid_mask, 'target'].values

    print(f"  Train: {len(X_train_a)}, Valid: {len(X_valid_a)}")

    fold_a = train_fold_v10(X_train_a, y_train, X_valid_a, y_valid, FEATURES_A)
    ens_a = compute_ensemble_v10(fold_a, y_valid)

    print(f"\n  Pattern A Final:")
    print(f"    LGB={fold_a['lgb']['auc']:.4f}  XGB={fold_a['xgb']['auc']:.4f}  "
          f"CB={fold_a['cb']['auc']:.4f}")
    print(f"    AVG3={ens_a['avg_auc']:.4f}  AVG2={ens_a['avg2_auc']:.4f}  "
          f"STACK={ens_a['stack_auc']:.4f}")

    # Feature importance
    importance = fold_a['lgb']['model'].feature_importance(importance_type='gain')
    fi = sorted(zip(FEATURES_A, importance), key=lambda x: x[1], reverse=True)
    print(f"\n  Feature Importance TOP 20:")
    for fname, imp in fi[:20]:
        print(f"    {fname:30s} {imp:10.1f}")

    # Condition backtest
    valid_df = df[valid_mask].copy()
    bt_a = backtest_condition(
        valid_df, fold_a['lgb']['model'], fold_a['xgb']['model'],
        FEATURES_A, ens_a['weights']['lgb'], ens_a['weights']['xgb'],
        "V10 Pattern A")

    # ============================================================
    #  FINAL MODEL Pattern B
    # ============================================================
    print("\n" + "=" * 70)
    print("  [3/4] FINAL MODEL PATTERN B")
    print("=" * 70)

    for f in FEATURES_B:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X_train_b = df.loc[train_mask, FEATURES_B].values
    X_valid_b = df.loc[valid_mask, FEATURES_B].values

    fold_b = train_fold_v10(X_train_b, y_train, X_valid_b, y_valid, FEATURES_B)
    ens_b = compute_ensemble_v10(fold_b, y_valid)

    print(f"\n  Pattern B Final:")
    print(f"    LGB={fold_b['lgb']['auc']:.4f}  XGB={fold_b['xgb']['auc']:.4f}  "
          f"CB={fold_b['cb']['auc']:.4f}")
    print(f"    AVG3={ens_b['avg_auc']:.4f}  AVG2={ens_b['avg2_auc']:.4f}  "
          f"STACK={ens_b['stack_auc']:.4f}")

    # ============================================================
    #  DECISION & SAVE
    # ============================================================
    print("\n" + "=" * 70)
    print("  [4/4] DECISION & SAVE")
    print("=" * 70)

    # Determine best WF method
    best_wf_method = max(wf_overall_a, key=wf_overall_a.get)
    best_wf_auc = wf_overall_a[best_wf_method]

    # Determine best final AUC
    final_options = {
        'avg3': ens_a['avg_auc'],
        'avg2': ens_a['avg2_auc'],
        'stack': ens_a['stack_auc'],
        'lgb': fold_a['lgb']['auc'],
    }
    best_final_method = max(final_options, key=final_options.get)
    best_final_auc = final_options[best_final_method]

    improved = best_final_auc > BASELINE_AUC

    print(f"\n  Current baseline (V9.3): {BASELINE_AUC}")
    print(f"  Best WF AUC: {best_wf_auc:.4f} ({best_wf_method})")
    print(f"  Best Final AUC: {best_final_auc:.4f} ({best_final_method})")
    print(f"  Improvement: {best_final_auc - BASELINE_AUC:+.4f}")
    print(f"  Decision: {'>>> UPDATE PRODUCTION <<<' if improved else 'KEEP V9.3 (no improvement)'}")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if improved:
        # Save Pattern A
        pkl_a = {
            'model': fold_a['lgb']['model'],
            'xgb_model': fold_a['xgb']['model'],
            'cb_model': fold_a['cb']['model'],
            'features': FEATURES_A_PKL,
            'version': 'v10_ensemble',
            'auc': fold_a['lgb']['auc'],
            'ensemble_auc': best_final_auc,
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
            'mlp_model': None,
            'mlp_scaler': None,
            'ensemble_weights': {
                'lgb': ens_a['weights']['lgb'],
                'xgb': ens_a['weights']['xgb'],
                'mlp': 0,
                'cb': ens_a['weights']['cb'],
            },
            'meta_model': ens_a['meta_model'],
            'course_map': dict(COURSE_MAP),
            'condition_backtest': bt_a,
            'wf_overall': wf_overall_a,
        }

        central_path = os.path.join(OUTPUT_DIR, 'keiba_model_v9_central.pkl')
        with open(central_path, 'wb') as f:
            pickle.dump(pkl_a, f)
        print(f"\n  Saved Pattern A (production): {central_path}")

        v8_path = os.path.join(OUTPUT_DIR, 'keiba_model_v8.pkl')
        with open(v8_path, 'wb') as f:
            pickle.dump(pkl_a, f)
        print(f"  Saved backup: {v8_path}")

        # Save Pattern B
        pkl_b = {
            'model': fold_b['lgb']['model'],
            'xgb_model': fold_b['xgb']['model'],
            'cb_model': fold_b['cb']['model'],
            'features': FEATURES_B_PKL,
            'version': 'v10_live',
            'auc': fold_b['lgb']['auc'],
            'ensemble_auc': ens_b['avg_auc'],
            'pattern_a_auc': best_final_auc,
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
            'mlp_model': None,
            'mlp_scaler': None,
            'ensemble_weights': {
                'lgb': ens_b['weights']['lgb'],
                'xgb': ens_b['weights']['xgb'],
                'mlp': 0,
                'cb': ens_b['weights']['cb'],
            },
            'meta_model': ens_b['meta_model'],
            'course_map': dict(COURSE_MAP),
            'weather_map': WEATHER_MAP,
        }

        live_path = os.path.join(OUTPUT_DIR, 'keiba_model_v9_central_live.pkl')
        with open(live_path, 'wb') as f:
            pickle.dump(pkl_b, f)
        print(f"  Saved Pattern B (production): {live_path}")

    # Save results JSON
    results = {
        'generated_at': now,
        'baseline_auc': BASELINE_AUC,
        'pattern_a': {
            'n_features': len(FEATURES_A),
            'features': FEATURES_A_PKL,
            'lgb_auc': fold_a['lgb']['auc'],
            'xgb_auc': fold_a['xgb']['auc'],
            'cb_auc': fold_a['cb']['auc'],
            'avg3_auc': ens_a['avg_auc'],
            'avg2_auc': ens_a['avg2_auc'],
            'stack_auc': ens_a['stack_auc'],
            'weights': ens_a['weights'],
        },
        'pattern_b': {
            'n_features': len(FEATURES_B),
            'lgb_auc': fold_b['lgb']['auc'],
            'xgb_auc': fold_b['xgb']['auc'],
            'cb_auc': fold_b['cb']['auc'],
            'avg3_auc': ens_b['avg_auc'],
            'avg2_auc': ens_b['avg2_auc'],
            'stack_auc': ens_b['stack_auc'],
        },
        'walk_forward': {
            'overall_auc': wf_overall_a,
            'best_method': best_wf_method,
            'best_auc': best_wf_auc,
            'fold_details': wf_details_a,
        },
        'decision': {
            'improved': improved,
            'best_final_auc': best_final_auc,
            'best_method': best_final_method,
            'improvement': best_final_auc - BASELINE_AUC,
            'production_updated': improved,
        },
        'feature_importance_top20': {fname: round(imp, 1) for fname, imp in fi[:20]},
    }

    report_path = os.path.join(DATA_DIR, 'v10_training_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {report_path}")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  V10 TRAINING COMPLETE ({elapsed:.0f}s / {elapsed/60:.1f}min)")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    main()
