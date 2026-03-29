#!/usr/bin/env python3
"""v12.1再学習 — prev_review_score + shinba_eval_score追加

新規特徴量:
  1. prev_review_score: 前走不利スコア(-2~+1, shift(1)でリークフリー)
  2. shinba_eval_score: 新馬評価スコア(-3~+3, 2024-2025のみ, 他は0埋め)

ベースライン: v12 WF AUC 0.8037 (74特徴量)
採用基準: WF AUC > 0.8037, 全年AUC > 0.78, gap < 0.05
"""

import os
import sys
import io
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

# Fix cp932 encoding on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

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
    _build_nk_race_id, merge_speed_index, merge_training_1f,
    merge_sire_shinba, merge_dam_stats, compute_pci,
    walk_forward_backtest, print_wf_results, compute_feature_importance,
    train_production_model,
)


V121_NEW_FEATURES = [
    'prev_review_score',
    'shinba_eval_score',
]


def merge_prev_review_score(df):
    """前走不利スコア(shift(1), リークフリー)
    race_review CSVからhorse単位でshift(1)して前走の不利スコアをマージ
    """
    print("  Merging prev_review_score...")
    csv_path = os.path.join(DATA_DIR, 'netkeiba_race_review.csv')
    if not os.path.exists(csv_path):
        print("  WARNING: netkeiba_race_review.csv not found")
        df['prev_review_score'] = 0
        return df

    rr = pd.read_csv(csv_path, encoding='utf-8', dtype={'race_id': str, 'umaban': str})
    rr['review_score'] = pd.to_numeric(rr['review_score'], errors='coerce').fillna(0)

    # Build netkeiba race_id for df
    df = _build_nk_race_id(df)
    df['_uma_str'] = df['umaban'].astype(int).astype(str)

    # Build lookup: (nk_race_id, umaban) -> review_score
    rr['_nk_rid'] = rr['race_id'].astype(str).str.zfill(12)
    rr['_uma_str'] = rr['umaban'].astype(str)
    rr_dedup = rr.drop_duplicates(subset=['_nk_rid', '_uma_str'], keep='last')
    rr_lookup = dict(zip(zip(rr_dedup['_nk_rid'], rr_dedup['_uma_str']),
                         rr_dedup['review_score']))

    # For each horse, the current race's review_score (to be shifted)
    df['_current_review'] = df.apply(
        lambda row: rr_lookup.get((row['_nk_rid'], row['_uma_str']), 0), axis=1
    )

    # shift(1) per horse: prev race's review_score
    df = df.sort_values(['horse_id', 'year', 'month', 'day', 'race_num']).reset_index(drop=True)
    df['prev_review_score'] = df.groupby('horse_id')['_current_review'].shift(1).fillna(0)

    matched = (df['_current_review'] != 0).sum()
    prev_filled = (df['prev_review_score'] != 0).sum()
    print(f"  Review scores matched: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")
    print(f"  Prev review filled: {prev_filled}/{len(df)} ({prev_filled/len(df)*100:.1f}%)")

    df.drop(columns=['_current_review', '_uma_str'], inplace=True, errors='ignore')
    return df


def merge_shinba_eval_score(df):
    """新馬評価スコア (2024-2025のみ、他は0)"""
    print("  Merging shinba_eval_score...")
    csv_path = os.path.join(DATA_DIR, 'netkeiba_shinba_eval.csv')
    if not os.path.exists(csv_path):
        print("  WARNING: netkeiba_shinba_eval.csv not found")
        df['shinba_eval_score'] = 0
        return df

    se = pd.read_csv(csv_path, encoding='utf-8', dtype={'race_id': str, 'umaban': str})
    se['comment_score'] = pd.to_numeric(se['comment_score'], errors='coerce').fillna(0)

    df = _build_nk_race_id(df)
    df['_uma_str'] = df['umaban'].astype(int).astype(str)

    se['_nk_rid'] = se['race_id'].astype(str).str.zfill(12)
    se['_uma_str'] = se['umaban'].astype(str)
    se_dedup = se.drop_duplicates(subset=['_nk_rid', '_uma_str'], keep='last')
    se_lookup = dict(zip(zip(se_dedup['_nk_rid'], se_dedup['_uma_str']),
                         se_dedup['comment_score']))

    df['shinba_eval_score'] = df.apply(
        lambda row: se_lookup.get((row['_nk_rid'], row['_uma_str']), 0), axis=1
    )

    matched = (df['shinba_eval_score'] != 0).sum()
    print(f"  Shinba eval matched: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")

    df.drop(columns=['_uma_str'], inplace=True, errors='ignore')
    return df


def main():
    start_time = time.time()
    print("=" * 60)
    print("  v12.1 Re-training (+prev_review_score, +shinba_eval_score)")
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

    # Phase 4: v12 features
    print("\n[4/6] Merging v12 features...")
    df = merge_speed_index(df)
    df = merge_training_1f(df)
    df = merge_sire_shinba(df)
    df = merge_dam_stats(df)
    df = compute_pci(df)

    # Phase 5: v12.1 new features
    print("\n[5/6] Merging v12.1 new features...")
    df = merge_prev_review_score(df)
    df = merge_shinba_eval_score(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses'] >= 5].copy()

    if 'year_full' not in df.columns:
        df['year_full'] = df['year'] + 2000

    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    print(f"\n  Final dataset: {len(df)} rows, years {int(df['year'].min())+2000}-{int(df['year'].max())+2000}")

    # Feature lists
    FEATURES_V93 = list(FEATURES_V91) + list(V92_NEW_FEATURES) + list(V93_NEW_FEATURES)
    FEATURES_BASELINE = [f for f in FEATURES_V93 if f not in LEAK_FEATURES_A]

    # v12 adopted features (dam_top3r removed in v12)
    V12_ADOPTED = [f for f in V12_NEW_FEATURES if f != 'dam_top3r']
    FEATURES_V12 = FEATURES_BASELINE + V12_ADOPTED
    FEATURES_V121 = FEATURES_V12 + V121_NEW_FEATURES

    # Ensure all features exist
    for f in FEATURES_V121:
        if f not in df.columns:
            print(f"  WARNING: {f} not in df, filling with 0")
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"\n  V12 features: {len(FEATURES_V12)}")
    print(f"  V12.1 features: {len(FEATURES_V121)} (+{len(V121_NEW_FEATURES)})")

    # ===== WF: Baseline (v12) =====
    print("\n[6/6] Walk-forward backtesting...")
    print("\n--- Pattern 1: v12 Baseline ---")
    wf_baseline = walk_forward_backtest(df, FEATURES_V12)
    auc_baseline = print_wf_results(wf_baseline, 'v12 Baseline')

    # ===== WF: +both features =====
    print("\n--- Pattern 2: v12.1 (+prev_review_score, +shinba_eval_score) ---")
    wf_v121 = walk_forward_backtest(df, FEATURES_V121)
    auc_v121 = print_wf_results(wf_v121, 'v12.1')

    # ===== Individual contribution =====
    print("\n--- Individual feature contribution ---")
    for nf in V121_NEW_FEATURES:
        feats_single = FEATURES_V12 + [nf]
        wf_single = walk_forward_backtest(df, feats_single)
        auc_single = np.mean([r['lgb_auc'] for r in wf_single])
        diff = auc_single - auc_baseline
        print(f"  {nf:35s}: AUC {auc_single:.4f} ({diff:+.5f})")

    # Feature importance
    imp_ranked = compute_feature_importance(wf_v121, FEATURES_V121)
    print("\n  Top 25 features by importance:")
    for i, (f, v) in enumerate(imp_ranked[:25]):
        marker = " ★NEW" if f in V121_NEW_FEATURES else (" ★V12" if f in V12_ADOPTED else "")
        print(f"  {i+1:2d}. {f:35s} {v:12.1f}{marker}")

    # ===== 採用判定 =====
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  V12 Baseline AUC: {auc_baseline:.4f} ({len(FEATURES_V12)} features)")
    print(f"  V12.1 AUC:        {auc_v121:.4f} ({len(FEATURES_V121)} features)")
    print(f"  Improvement:      {auc_v121 - auc_baseline:+.5f}")

    all_years_ok = all(r['lgb_auc'] > 0.78 for r in wf_v121)
    no_overfit = all(r['gap'] < 0.05 for r in wf_v121)
    auc_improved = auc_v121 > 0.8037  # v12 baseline

    print(f"\n  All years > 0.78: {'✓' if all_years_ok else '✗'}")
    print(f"  No overfitting:   {'✓' if no_overfit else '✗'}")
    print(f"  AUC > 0.8037:     {'✓' if auc_improved else '✗'}")

    adopted = auc_improved and all_years_ok and no_overfit

    # Check which new features help
    harmful = []
    for nf in V121_NEW_FEATURES:
        feats_without = [f for f in FEATURES_V121 if f != nf]
        wf_without = walk_forward_backtest(df, feats_without, years=[2024, 2025])
        auc_without = np.mean([r['lgb_auc'] for r in wf_without])
        auc_full_subset = np.mean([r['lgb_auc'] for r in wf_v121 if r['year'] in [2024, 2025]])
        contribution = auc_full_subset - auc_without
        if contribution < -0.0005:
            harmful.append(nf)
            print(f"  HARMFUL: {nf} contribution={contribution:+.5f}")

    # If some features are harmful, try without them
    if harmful:
        selected_new = [f for f in V121_NEW_FEATURES if f not in harmful]
        if selected_new:
            FEATURES_SELECTED = FEATURES_V12 + selected_new
            print(f"\n--- Pattern 3: Selected (without {harmful}) ---")
            wf_selected = walk_forward_backtest(df, FEATURES_SELECTED)
            auc_selected = print_wf_results(wf_selected, f'Selected ({len(selected_new)} new)')

            if auc_selected > auc_v121:
                print(f"  Selected is better, using Pattern 3")
                wf_v121 = wf_selected
                auc_v121 = auc_selected
                FEATURES_V121 = FEATURES_SELECTED
                V121_NEW_FEATURES_FINAL = selected_new
            else:
                V121_NEW_FEATURES_FINAL = V121_NEW_FEATURES
        else:
            V121_NEW_FEATURES_FINAL = []
            adopted = False
            print(f"  All new features harmful, NOT ADOPTED")
    else:
        V121_NEW_FEATURES_FINAL = V121_NEW_FEATURES

    # Re-evaluate after potential feature selection
    auc_improved = auc_v121 > 0.8037
    all_years_ok = all(r['lgb_auc'] > 0.78 for r in wf_v121)
    no_overfit = all(r['gap'] < 0.05 for r in wf_v121)
    adopted = auc_improved and all_years_ok and no_overfit

    print(f"\n  VERDICT: {'✓ ADOPTED as v12.1' if adopted else '✗ NOT ADOPTED'}")

    # Save results
    result_data = {
        'baseline_auc': auc_baseline,
        'v121_auc': auc_v121,
        'improvement': auc_v121 - auc_baseline,
        'adopted': adopted,
        'new_features': V121_NEW_FEATURES_FINAL,
        'harmful_features': harmful,
        'yearly_results': [{
            'year': r['year'], 'lgb_auc': r['lgb_auc'], 'xgb_auc': r['xgb_auc'],
            'train_auc': r['train_auc'], 'gap': r['gap'],
        } for r in wf_v121],
        'baseline_yearly': [{
            'year': r['year'], 'lgb_auc': r['lgb_auc'],
        } for r in wf_baseline],
        'timestamp': datetime.now().isoformat(),
    }

    result_path = os.path.join(DATA_DIR, 'v121_training_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved: {result_path}")

    # ===== 本番モデル学習 =====
    if adopted:
        print(f"\n{'='*60}")
        print(f"  TRAINING PRODUCTION MODEL (v12.1)")
        print(f"{'='*60}")
        train_production_model_v121(df, FEATURES_V121, sire_map, bms_map, wf_v121,
                                    V121_NEW_FEATURES_FINAL)

    elapsed = (time.time() - start_time) / 60
    print(f"\n  Elapsed: {elapsed:.1f} min")
    return adopted, result_data


def train_production_model_v121(df, features, sire_map, bms_map, wf_results, new_features):
    """v12.1本番モデル学習・保存"""
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

    # Ensemble
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
        'version': 'v12.1_leakfree',
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
        'v12_new_features': list(V12_NEW_FEATURES) + new_features,
    }

    a_path = os.path.join(OUTPUT_DIR, 'keiba_model_v12_central.pkl.gz')
    with gzip.open(a_path, 'wb') as f:
        pickle.dump(pkl_a, f)
    print(f"  Pattern A saved: {a_path}")

    # Pattern B (live)
    pkl_b = dict(pkl_a)
    pkl_b['features'] = features
    pkl_b['version'] = 'v12.1_live'
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
    v8_pkl['version'] = 'v12.1_backup'
    with open(v8_path, 'wb') as f:
        pickle.dump(v8_pkl, f)
    print(f"  V8 backup saved: {v8_path}")

    print(f"\n  *** v12.1 Production model saved! ***")
    print(f"  Pattern A: {len(pattern_a_features)} features, AUC {lgb_auc:.4f}")
    print(f"  Pattern B: {len(features)} features")
    print(f"  WF AUC: {wf_auc:.4f}")


if __name__ == '__main__':
    main()
