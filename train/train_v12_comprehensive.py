#!/usr/bin/env python3
"""v12総合再学習 — 新特徴量8個を同時投入してWFバックテスト

新規特徴量:
  1. index_max (最高指数)
  2. index_run1 (前走指数)
  3. index_avg5 (5走平均指数)
  4. time_1f_last (追切ラスト1F)
  5. training_intensity_enc (調教強度)
  6. sire_shinba_top3r (種牡馬新馬成績)
  7. dam_top3r (母産駒成績)
  8. pci (ペースチェンジ指数)

評価:
  - Pattern A WF AUC (2020-2025)
  - 全年AUC > 0.78
  - 実配当ROI (条件A-X)
  - 過学習チェック (train-test AUC gap < 0.05)
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
    train_lgb, train_xgb,
    FEATURES_V91, V92_NEW_FEATURES, V93_NEW_FEATURES,
    COURSE_MAP, N_TOP_SIRE,
)
from train_v92_leakfree import LEAK_FEATURES_A

# ===== 新規特徴量 =====

V12_NEW_FEATURES = [
    'index_max_filled',
    'index_run1_filled',
    'index_avg5_filled',
    'time_1f_last_filled',
    'training_intensity_enc',
    'sire_shinba_top3r',
    'dam_top3r',
    'pci',
]

# LGB params (current best)
LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_child_samples': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'verbose': -1, 'seed': 42,
}

XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'max_depth': 6, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'seed': 42, 'tree_method': 'hist',
}


def _build_nk_race_id(df):
    """JV race_id + 数値カラムからnetkeiba形式race_idを構築"""
    if '_nk_rid' in df.columns:
        return df
    jv_rid = df['race_id'].astype(str).str.zfill(10)
    course_code = jv_rid.str[:2]
    year_2d = jv_rid.str[2:4]
    # nichi列を使用（hex問題回避）
    kai = df['kai'].astype(int).apply(lambda x: f'{x:02d}')
    nichi = df['nichi'].astype(int).apply(lambda x: f'{x:02d}')
    race_num = jv_rid.str[6:8]
    df['_nk_rid'] = '20' + year_2d + course_code + kai + nichi + race_num
    return df


def merge_speed_index(df):
    """speed_index CSVをnetkeiba race_id+umabanでマージ"""
    csv_path = os.path.join(DATA_DIR, 'netkeiba_speed_index.csv')
    if not os.path.exists(csv_path):
        print("  WARNING: netkeiba_speed_index.csv not found")
        df['index_max_filled'] = 0
        df['index_run1_filled'] = 0
        df['index_avg5_filled'] = 0
        return df

    si = pd.read_csv(csv_path, encoding='utf-8', dtype={'race_id': str, 'umaban': str})
    si['index_max'] = pd.to_numeric(si['index_max'], errors='coerce')
    si['index_avg5'] = pd.to_numeric(si['index_avg5'], errors='coerce')
    if 'index_run1' in si.columns:
        si['index_run1'] = pd.to_numeric(si['index_run1'], errors='coerce')
    else:
        si['index_run1'] = np.nan

    df = _build_nk_race_id(df)
    df['_uma_str'] = df['umaban'].astype(int).astype(str)
    si['_nk_rid'] = si['race_id'].astype(str).str.zfill(12)
    si['_uma_str'] = si['umaban'].astype(str)

    si_dedup = si.drop_duplicates(subset=['_nk_rid', '_uma_str'], keep='last')
    merged = df.merge(
        si_dedup[['_nk_rid', '_uma_str', 'index_max', 'index_run1', 'index_avg5']],
        on=['_nk_rid', '_uma_str'], how='left', suffixes=('', '_si')
    )

    mean_max = merged['index_max'].dropna().mean()
    mean_run1 = merged['index_run1'].dropna().mean()
    mean_avg5 = merged['index_avg5'].dropna().mean()

    merged['index_max_filled'] = merged['index_max'].fillna(mean_max if not np.isnan(mean_max) else 0)
    merged['index_run1_filled'] = merged['index_run1'].fillna(mean_run1 if not np.isnan(mean_run1) else 0)
    merged['index_avg5_filled'] = merged['index_avg5'].fillna(mean_avg5 if not np.isnan(mean_avg5) else 0)

    matched = merged['index_max'].notna().sum()
    print(f"  Speed Index: {matched}/{len(df)} matched ({matched/len(df)*100:.1f}%)")

    df['index_max_filled'] = merged['index_max_filled'].values
    df['index_run1_filled'] = merged['index_run1_filled'].values
    df['index_avg5_filled'] = merged['index_avg5_filled'].values
    df.drop(columns=['_uma_str'], inplace=True, errors='ignore')
    return df


def merge_training_1f(df):
    """training_times CSVからラスト1Fと強度をマージ"""
    csv_path = os.path.join(DATA_DIR, 'netkeiba_training_times.csv')
    if not os.path.exists(csv_path):
        print("  WARNING: netkeiba_training_times.csv not found")
        df['time_1f_last_filled'] = 12.5
        df['training_intensity_enc'] = 0
        return df

    tt = pd.read_csv(csv_path, encoding='utf-8', dtype={'race_id': str, 'umaban': str})
    tt['time_1f'] = pd.to_numeric(tt.get('time_1f', pd.Series(dtype=float)), errors='coerce')

    intensity_map = {'一杯': 3, '強め': 2, '馬なり': 1, '': 0}
    if 'intensity' in tt.columns:
        tt['intensity_enc'] = tt['intensity'].map(
            lambda x: intensity_map.get(str(x).strip(), 0) if pd.notna(x) else 0
        )
    else:
        tt['intensity_enc'] = 0

    df = _build_nk_race_id(df)
    df['_uma_str'] = df['umaban'].astype(int).astype(str)
    tt['_nk_rid'] = tt['race_id'].astype(str).str.zfill(12)
    tt['_uma_str'] = tt['umaban'].astype(str)

    tt_dedup = tt.drop_duplicates(subset=['_nk_rid', '_uma_str'], keep='last')
    merged = df.merge(
        tt_dedup[['_nk_rid', '_uma_str', 'time_1f', 'intensity_enc']],
        on=['_nk_rid', '_uma_str'], how='left', suffixes=('', '_tt')
    )

    mean_1f = merged['time_1f'].dropna().mean()
    merged['time_1f_last_filled'] = merged['time_1f'].fillna(mean_1f if not np.isnan(mean_1f) else 12.5)
    merged['training_intensity_enc'] = merged['intensity_enc'].fillna(0)

    matched = merged['time_1f'].notna().sum()
    print(f"  Training 1F: {matched}/{len(df)} matched ({matched/len(df)*100:.1f}%)")

    df['time_1f_last_filled'] = merged['time_1f_last_filled'].values
    df['training_intensity_enc'] = merged['training_intensity_enc'].values
    df.drop(columns=['_uma_str'], inplace=True, errors='ignore')
    return df


def merge_sire_shinba(df):
    """種牡馬新馬成績をexpanding windowで計算（リークフリー）
    新馬戦(class_code==15)のデータのみ、各レース時点の過去データから計算
    """
    print("  Computing sire_shinba_top3r (expanding window, leak-free)...")
    if 'father' not in df.columns:
        df['sire_shinba_top3r'] = 0.22
        return df

    df = df.sort_values(['year', 'month', 'day', 'race_num']).reset_index(drop=True)
    sire_top3r = np.full(len(df), np.nan)
    alpha = 20
    prior = 0.22

    # Only update from shinba races (class_code==15)
    sire_cumsum = {}  # father -> (total, top3_count)
    for i in range(len(df)):
        f = df.iloc[i]['father']
        if pd.isna(f) or f == '':
            continue
        # Get current stats BEFORE this race
        if f in sire_cumsum:
            total, t3 = sire_cumsum[f]
            if total > 0:
                sire_top3r[i] = (t3 + alpha * prior) / (total + alpha)
        # Only update from shinba races
        cc = df.iloc[i].get('class_code', 0)
        if cc == 15:
            fin = df.iloc[i].get('finish', 99)
            if pd.notna(fin) and fin > 0:
                if f not in sire_cumsum:
                    sire_cumsum[f] = (0, 0)
                old_total, old_t3 = sire_cumsum[f]
                sire_cumsum[f] = (old_total + 1, old_t3 + (1 if fin <= 3 else 0))

    df['sire_shinba_top3r'] = sire_top3r
    mean_val = np.nanmean(sire_top3r)
    df['sire_shinba_top3r'] = df['sire_shinba_top3r'].fillna(mean_val if not np.isnan(mean_val) else prior)
    valid = np.sum(~np.isnan(sire_top3r))
    print(f"  Sire shinba (expanding): {valid}/{len(df)} computed ({valid/len(df)*100:.1f}%)")
    return df


def merge_dam_stats(df):
    """母産駒成績をexpanding windowで計算（リークフリー）
    各レースの時点で、当該レースを除く過去データのみから母馬の複勝率を算出
    """
    print("  Computing dam_top3r (expanding window, leak-free)...")
    if 'mother' not in df.columns:
        df['dam_top3r'] = 0.22
        print("  WARNING: 'mother' column not found")
        return df

    df = df.sort_values(['year', 'month', 'day', 'race_num']).reset_index(drop=True)
    dam_top3r = np.full(len(df), np.nan)
    # Bayesian prior
    alpha = 10
    prior = 0.22

    # Expanding window: for each row, use all previous rows with same mother
    mother_cumsum = {}  # mother -> (total, top3_count)
    for i in range(len(df)):
        m = df.iloc[i]['mother']
        if pd.isna(m) or m == '':
            continue
        # Get current stats BEFORE this race
        if m in mother_cumsum:
            total, t3 = mother_cumsum[m]
            if total > 0:
                dam_top3r[i] = (t3 + alpha * prior) / (total + alpha)
        # Update cumsum with this race's result
        fin = df.iloc[i].get('finish', 99)
        if pd.notna(fin) and fin > 0:
            if m not in mother_cumsum:
                mother_cumsum[m] = (0, 0)
            old_total, old_t3 = mother_cumsum[m]
            mother_cumsum[m] = (old_total + 1, old_t3 + (1 if fin <= 3 else 0))

    df['dam_top3r'] = dam_top3r
    mean_val = np.nanmean(dam_top3r)
    df['dam_top3r'] = df['dam_top3r'].fillna(mean_val if not np.isnan(mean_val) else prior)
    valid = np.sum(~np.isnan(dam_top3r))
    print(f"  Dam stats (expanding): {valid}/{len(df)} computed ({valid/len(df)*100:.1f}%)")
    return df


def compute_pci(df):
    """PCI (Pace Change Index) = 後半3F / 前半3F"""
    if 'prev_race_first3f' in df.columns and 'prev_race_last3f' in df.columns:
        first3f = df['prev_race_first3f'].replace(0, np.nan)
        last3f = df['prev_race_last3f'].replace(0, np.nan)
        pci = last3f / first3f
        mean_pci = pci.dropna().mean()
        df['pci'] = pci.fillna(mean_pci if not np.isnan(mean_pci) else 1.0)
        valid = pci.notna().sum()
        print(f"  PCI: {valid}/{len(df)} computed ({valid/len(df)*100:.1f}%)")
    else:
        df['pci'] = 1.0
        print("  PCI: columns not found, using default 1.0")
    return df


# ===== ウォークフォワード =====

def walk_forward_backtest(df, features, label='target', years=range(2020, 2026)):
    """年単位WFバックテスト"""
    results = []
    for test_year in years:
        # 2桁年
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty

        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        X_train = df.loc[train_mask, features].values
        y_train = df.loc[train_mask, label].values
        X_test = df.loc[test_mask, features].values
        y_test = df.loc[test_mask, label].values

        # LGB
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)
        lgb_model = lgb.train(
            LGB_PARAMS, dtrain, num_boost_round=1000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        lgb_pred = lgb_model.predict(X_test)
        from sklearn.metrics import roc_auc_score
        lgb_auc = roc_auc_score(y_test, lgb_pred)

        # XGB
        dxtr = xgb.DMatrix(X_train, label=y_train)
        dxte = xgb.DMatrix(X_test, label=y_test)
        xgb_model = xgb.train(
            XGB_PARAMS, dxtr, num_boost_round=1000,
            evals=[(dxte, 'valid')],
            early_stopping_rounds=50, verbose_eval=False
        )
        xgb_pred = xgb_model.predict(dxte)
        xgb_auc = roc_auc_score(y_test, xgb_pred)

        # Train AUC (for overfitting check)
        lgb_train_pred = lgb_model.predict(X_train)
        train_auc = roc_auc_score(y_train, lgb_train_pred)

        results.append({
            'year': test_year,
            'lgb_auc': lgb_auc,
            'xgb_auc': xgb_auc,
            'train_auc': train_auc,
            'gap': train_auc - lgb_auc,
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
            'lgb_model': lgb_model,
            'xgb_model': xgb_model,
            'lgb_pred': lgb_pred,
            'xgb_pred': xgb_pred,
            'y_test': y_test,
            'test_df': df.loc[test_mask].copy(),
        })

    return results


def print_wf_results(results, label=''):
    """WF結果を表示"""
    print(f"\n{'='*60}")
    print(f"  WF Results: {label}")
    print(f"{'='*60}")
    aucs = []
    for r in results:
        aucs.append(r['lgb_auc'])
        ok = 'OK' if r['lgb_auc'] > 0.78 else 'NG'
        gap_ok = 'OK' if r['gap'] < 0.05 else 'NG'
        print(f"  {r['year']}: LGB={r['lgb_auc']:.4f} XGB={r['xgb_auc']:.4f} "
              f"Train={r['train_auc']:.4f} Gap={r['gap']:.4f} {ok} {gap_ok}")
    mean_auc = np.mean(aucs)
    print(f"  --- Mean LGB AUC: {mean_auc:.4f} ---")
    return mean_auc


def compute_feature_importance(results, features):
    """特徴量重要度を集計"""
    imp = np.zeros(len(features))
    for r in results:
        imp += r['lgb_model'].feature_importance(importance_type='gain')
    imp /= len(results)
    ranked = sorted(zip(features, imp), key=lambda x: -x[1])
    return ranked


# ===== メイン =====

def main():
    start_time = time.time()
    print("=" * 60)
    print("  v12 Comprehensive Re-training")
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

    # Phase 4: New v12 features
    print("\n[4/6] Merging v12 new features...")
    df = merge_speed_index(df)
    df = merge_training_1f(df)
    df = merge_sire_shinba(df)
    df = merge_dam_stats(df)
    df = compute_pci(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses'] >= 5].copy()

    # Convert year for WF
    if 'year_full' not in df.columns:
        df['year_full'] = df['year'] + 2000

    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    print(f"\n  Final dataset: {len(df)} rows, years {int(df['year'].min())+2000}-{int(df['year'].max())+2000}")

    # Feature lists
    FEATURES_V93 = list(FEATURES_V91) + list(V92_NEW_FEATURES) + list(V93_NEW_FEATURES)
    FEATURES_BASELINE = [f for f in FEATURES_V93 if f not in LEAK_FEATURES_A]
    FEATURES_V12 = FEATURES_BASELINE + V12_NEW_FEATURES

    # Ensure all features exist
    for f in FEATURES_V12:
        if f not in df.columns:
            print(f"  WARNING: {f} not in df, filling with 0")
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"\n  Baseline features: {len(FEATURES_BASELINE)}")
    print(f"  V12 features: {len(FEATURES_V12)} (+{len(V12_NEW_FEATURES)})")

    # ===== Pattern 1: Baseline =====
    print("\n[5/6] Walk-forward backtesting...")
    print("\n--- Pattern 1: Baseline (current 67 features) ---")
    wf_baseline = walk_forward_backtest(df, FEATURES_BASELINE)
    auc_baseline = print_wf_results(wf_baseline, 'Baseline')

    # ===== Pattern 2: +all 8 features =====
    print("\n--- Pattern 2: +all_8 (75 features) ---")
    wf_all = walk_forward_backtest(df, FEATURES_V12)
    auc_all = print_wf_results(wf_all, '+all_8')

    # Feature importance
    imp_ranked = compute_feature_importance(wf_all, FEATURES_V12)
    print("\n  Top 20 features by importance:")
    for i, (f, v) in enumerate(imp_ranked[:20]):
        marker = " ★NEW" if f in V12_NEW_FEATURES else ""
        print(f"  {i+1:2d}. {f:35s} {v:12.1f}{marker}")

    # ===== Pattern 3: Selected (remove harmful features) =====
    # Check which new features improve AUC
    print("\n--- Individual feature contribution ---")
    new_feat_scores = {}
    for nf in V12_NEW_FEATURES:
        feats_without = [f for f in FEATURES_V12 if f != nf]
        wf_without = walk_forward_backtest(df, feats_without, years=[2024, 2025])
        auc_without = np.mean([r['lgb_auc'] for r in wf_without])
        # Contribution = full AUC - AUC without this feature
        auc_full_subset = np.mean([r['lgb_auc'] for r in wf_all if r['year'] in [2024, 2025]])
        contribution = auc_full_subset - auc_without
        new_feat_scores[nf] = contribution
        sign = '+' if contribution > 0 else ''
        print(f"  {nf:35s}: {sign}{contribution:.5f}")

    # Select only positive contributors
    selected_new = [f for f, c in new_feat_scores.items() if c > -0.0005]
    removed_new = [f for f, c in new_feat_scores.items() if c <= -0.0005]
    FEATURES_SELECTED = FEATURES_BASELINE + selected_new

    print(f"\n  Selected: {selected_new}")
    print(f"  Removed: {removed_new}")

    if selected_new:
        print(f"\n--- Pattern 3: Selected ({len(FEATURES_SELECTED)} features) ---")
        wf_selected = walk_forward_backtest(df, FEATURES_SELECTED)
        auc_selected = print_wf_results(wf_selected, f'+selected ({len(selected_new)} new)')
    else:
        print("\n  No features selected, skipping Pattern 3")
        wf_selected = wf_all
        auc_selected = auc_all
        FEATURES_SELECTED = FEATURES_V12

    # ===== 結果サマリー =====
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline AUC:  {auc_baseline:.4f} ({len(FEATURES_BASELINE)} features)")
    print(f"  +all_8 AUC:    {auc_all:.4f} ({len(FEATURES_V12)} features)")
    print(f"  +selected AUC: {auc_selected:.4f} ({len(FEATURES_SELECTED)} features)")
    print(f"  Improvement:   {auc_selected - auc_baseline:+.4f}")

    # 採用判定
    best_auc = max(auc_all, auc_selected)
    best_label = '+all_8' if auc_all >= auc_selected else '+selected'
    best_wf = wf_all if auc_all >= auc_selected else wf_selected
    best_features = FEATURES_V12 if auc_all >= auc_selected else FEATURES_SELECTED

    all_years_ok = all(r['lgb_auc'] > 0.78 for r in best_wf)
    no_overfit = all(r['gap'] < 0.05 for r in best_wf)
    auc_improved = best_auc > 0.8017

    print(f"\n  Best pattern: {best_label} (AUC {best_auc:.4f})")
    print(f"  All years > 0.78: {'✓' if all_years_ok else '✗'}")
    print(f"  No overfitting:   {'✓' if no_overfit else '✗'}")
    print(f"  AUC > 0.8017:     {'✓' if auc_improved else '✗'}")

    adopted = auc_improved and all_years_ok and no_overfit
    print(f"\n  VERDICT: {'✓ ADOPTED as v12' if adopted else '✗ NOT ADOPTED'}")

    # Save results
    result_data = {
        'baseline_auc': auc_baseline,
        'all8_auc': auc_all,
        'selected_auc': auc_selected,
        'best_pattern': best_label,
        'best_auc': best_auc,
        'adopted': adopted,
        'new_features': V12_NEW_FEATURES,
        'selected_features': selected_new if selected_new else V12_NEW_FEATURES,
        'removed_features': removed_new,
        'feature_contributions': new_feat_scores,
        'yearly_results': [{
            'year': r['year'], 'lgb_auc': r['lgb_auc'], 'xgb_auc': r['xgb_auc'],
            'train_auc': r['train_auc'], 'gap': r['gap']
        } for r in best_wf],
        'timestamp': datetime.now().isoformat(),
    }

    result_path = os.path.join(DATA_DIR, 'v12_training_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved: {result_path}")

    # ===== 本番モデル学習（adoptedの場合のみ） =====
    if adopted:
        print(f"\n{'='*60}")
        print(f"  TRAINING PRODUCTION MODEL (v12)")
        print(f"{'='*60}")
        train_production_model(df, best_features, sire_map, bms_map, best_wf)

    elapsed = (time.time() - start_time) / 60
    print(f"  Elapsed: {elapsed:.1f} min")


def train_production_model(df, features, sire_map, bms_map, wf_results):
    """フルデータで本番モデルを学習・保存"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    OUTPUT_DIR = BASE_DIR

    # 直近2年をvalidation、それ以前をtrain
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
    from sklearn.metrics import roc_auc_score
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

    # WF AUC summary
    wf_auc = np.mean([r['lgb_auc'] for r in wf_results])

    # ===== Pattern A (leak-free) =====
    pattern_a_features = [f for f in features if f not in LEAK_FEATURES_A]
    pkl_a = {
        'model': lgb_model,
        'features': pattern_a_features,
        'version': 'v12_leakfree',
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
    }

    a_path = os.path.join(OUTPUT_DIR, 'keiba_model_v12_central.pkl.gz')
    with gzip.open(a_path, 'wb') as f:
        pickle.dump(pkl_a, f)
    print(f"  Pattern A saved: {a_path}")

    # ===== Pattern B (live) =====
    pkl_b = dict(pkl_a)
    pkl_b['features'] = features  # All features including leak ones
    pkl_b['version'] = 'v12_live'
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
    v8_pkl['version'] = 'v12_backup'
    with open(v8_path, 'wb') as f:
        pickle.dump(v8_pkl, f)
    print(f"  V8 backup saved: {v8_path}")

    print(f"\n  *** v12 Production model saved! ***")
    print(f"  Pattern A: {len(pattern_a_features)} features, AUC {lgb_auc:.4f}")
    print(f"  Pattern B: {len(features)} features")
    print(f"  WF AUC: {wf_auc:.4f}")


if __name__ == '__main__':
    main()
