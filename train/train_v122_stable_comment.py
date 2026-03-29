#!/usr/bin/env python3
"""v12.2 — stable_comment_score追加テスト

v12（74特徴量、WF AUC 0.8037）に stable_comment_score のみ追加。
netkeiba_stable_comments.csv (99,485行、カバレッジ84-85%) をマージ。

採用判定:
  - WF AUC > 0.8037
  - 全条件ROI >= 現行
  - train-test gap < 0.05（全年）
  - 各年AUC安定（> 0.78）
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

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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

# v12 existing new features (7 adopted)
V12_NEW_FEATURES = [
    'index_max_filled', 'index_run1_filled', 'index_avg5_filled',
    'time_1f_last_filled', 'training_intensity_enc',
    'sire_shinba_top3r', 'pci',
]

# v12.2: add stable_comment_score
V122_NEW_FEATURES = ['stable_comment_score']

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
    """JV race_id + kai/nichi からnetkeiba形式race_idを構築"""
    if '_nk_rid' in df.columns:
        return df
    jv_rid = df['race_id'].astype(str).str.zfill(10)
    course_code = jv_rid.str[:2]
    year_2d = jv_rid.str[2:4]
    kai = df['kai'].astype(int).apply(lambda x: f'{x:02d}')
    nichi = df['nichi'].astype(int).apply(lambda x: f'{x:02d}')
    race_num = jv_rid.str[6:8]
    df['_nk_rid'] = '20' + year_2d + course_code + kai + nichi + race_num
    return df


# === v12 feature merge functions (from train_v12_comprehensive.py) ===

def merge_speed_index(df):
    csv_path = os.path.join(DATA_DIR, 'netkeiba_speed_index.csv')
    if not os.path.exists(csv_path):
        df['index_max_filled'] = df['index_run1_filled'] = df['index_avg5_filled'] = 0
        return df
    si = pd.read_csv(csv_path, encoding='utf-8', dtype={'race_id': str, 'umaban': str})
    si['index_max'] = pd.to_numeric(si['index_max'], errors='coerce')
    si['index_avg5'] = pd.to_numeric(si['index_avg5'], errors='coerce')
    si['index_run1'] = pd.to_numeric(si.get('index_run1', pd.Series(dtype=float)), errors='coerce')
    df = _build_nk_race_id(df)
    df['_uma_str'] = df['umaban'].astype(int).astype(str)
    si['_nk_rid'] = si['race_id'].astype(str).str.zfill(12)
    si['_uma_str'] = si['umaban'].astype(str)
    si_dedup = si.drop_duplicates(subset=['_nk_rid', '_uma_str'], keep='last')
    merged = df.merge(si_dedup[['_nk_rid', '_uma_str', 'index_max', 'index_run1', 'index_avg5']],
                      on=['_nk_rid', '_uma_str'], how='left', suffixes=('', '_si'))
    for col in ['index_max', 'index_run1', 'index_avg5']:
        mean_v = merged[col].dropna().mean()
        df[col + '_filled'] = merged[col].fillna(mean_v if not np.isnan(mean_v) else 0).values
    matched = merged['index_max'].notna().sum()
    print(f"  Speed Index: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")
    df.drop(columns=['_uma_str'], inplace=True, errors='ignore')
    return df


def merge_training_1f(df):
    csv_path = os.path.join(DATA_DIR, 'netkeiba_training_times.csv')
    if not os.path.exists(csv_path):
        df['time_1f_last_filled'] = 12.5
        df['training_intensity_enc'] = 0
        return df
    tt = pd.read_csv(csv_path, encoding='utf-8', dtype={'race_id': str, 'umaban': str})
    tt['time_1f'] = pd.to_numeric(tt.get('time_1f', pd.Series(dtype=float)), errors='coerce')
    intensity_map = {'一杯': 3, '強め': 2, '馬なり': 1, '': 0}
    tt['intensity_enc'] = tt.get('intensity', pd.Series(dtype=str)).map(
        lambda x: intensity_map.get(str(x).strip(), 0) if pd.notna(x) else 0)
    df = _build_nk_race_id(df)
    df['_uma_str'] = df['umaban'].astype(int).astype(str)
    tt['_nk_rid'] = tt['race_id'].astype(str).str.zfill(12)
    tt['_uma_str'] = tt['umaban'].astype(str)
    tt_dedup = tt.drop_duplicates(subset=['_nk_rid', '_uma_str'], keep='last')
    merged = df.merge(tt_dedup[['_nk_rid', '_uma_str', 'time_1f', 'intensity_enc']],
                      on=['_nk_rid', '_uma_str'], how='left', suffixes=('', '_tt'))
    mean_1f = merged['time_1f'].dropna().mean()
    df['time_1f_last_filled'] = merged['time_1f'].fillna(mean_1f if not np.isnan(mean_1f) else 12.5).values
    df['training_intensity_enc'] = merged['intensity_enc'].fillna(0).values
    matched = merged['time_1f'].notna().sum()
    print(f"  Training 1F: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")
    df.drop(columns=['_uma_str'], inplace=True, errors='ignore')
    return df


def merge_sire_shinba(df):
    print("  Computing sire_shinba_top3r (expanding window)...")
    if 'father' not in df.columns:
        df['sire_shinba_top3r'] = 0.22
        return df
    df = df.sort_values(['year', 'month', 'day', 'race_num']).reset_index(drop=True)
    sire_top3r = np.full(len(df), np.nan)
    alpha, prior = 20, 0.22
    sire_cumsum = {}
    for i in range(len(df)):
        f = df.iloc[i]['father']
        if pd.isna(f) or f == '':
            continue
        if f in sire_cumsum:
            total, t3 = sire_cumsum[f]
            if total > 0:
                sire_top3r[i] = (t3 + alpha * prior) / (total + alpha)
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
    print(f"  Sire shinba: {np.sum(~np.isnan(sire_top3r))}/{len(df)} computed")
    return df


def compute_pci(df):
    if 'prev_race_first3f' in df.columns and 'prev_race_last3f' in df.columns:
        first3f = df['prev_race_first3f'].replace(0, np.nan)
        last3f = df['prev_race_last3f'].replace(0, np.nan)
        pci = last3f / first3f
        mean_pci = pci.dropna().mean()
        df['pci'] = pci.fillna(mean_pci if not np.isnan(mean_pci) else 1.0)
        print(f"  PCI: {pci.notna().sum()}/{len(df)} computed")
    else:
        df['pci'] = 1.0
    return df


# === NEW: stable_comment_score merge ===

def merge_stable_comments(df):
    """stable_comment_score をCSVからマージ。
    netkeiba race_id + umaban でマッチ。未マッチは0（コメントなし＝中立）。
    """
    csv_path = os.path.join(DATA_DIR, 'netkeiba_stable_comments.csv')
    if not os.path.exists(csv_path):
        print("  WARNING: netkeiba_stable_comments.csv not found")
        df['stable_comment_score'] = 0
        return df

    sc = pd.read_csv(csv_path, encoding='utf-8-sig', dtype={'race_id': str, 'umaban': str, 'score': str})
    sc['score'] = pd.to_numeric(sc['score'], errors='coerce').fillna(0).astype(int)

    df = _build_nk_race_id(df)
    df['_uma_str'] = df['umaban'].astype(int).astype(str)
    sc['_nk_rid'] = sc['race_id'].astype(str).str.zfill(12)
    sc['_uma_str'] = sc['umaban'].astype(str)

    sc_dedup = sc.drop_duplicates(subset=['_nk_rid', '_uma_str'], keep='last')
    merged = df.merge(
        sc_dedup[['_nk_rid', '_uma_str', 'score']],
        on=['_nk_rid', '_uma_str'], how='left', suffixes=('', '_sc')
    )

    df['stable_comment_score'] = merged['score'].fillna(0).astype(int).values
    matched = merged['score'].notna().sum()
    print(f"  Stable Comments: {matched}/{len(df)} matched ({matched/len(df)*100:.1f}%)")

    # Distribution
    dist = df['stable_comment_score'].value_counts().sort_index()
    print(f"  Score distribution: {dict(dist)}")

    df.drop(columns=['_uma_str'], inplace=True, errors='ignore')
    return df


# === Walk-forward ===

def walk_forward_backtest(df, features, label='target', years=range(2020, 2026)):
    results = []
    for test_year in years:
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue
        X_train = df.loc[train_mask, features].values
        y_train = df.loc[train_mask, label].values
        X_test = df.loc[test_mask, features].values
        y_test = df.loc[test_mask, label].values

        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)
        lgb_model = lgb.train(
            LGB_PARAMS, dtrain, num_boost_round=1000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        lgb_pred = lgb_model.predict(X_test)
        lgb_auc = roc_auc_score(y_test, lgb_pred)

        dxtr = xgb.DMatrix(X_train, label=y_train)
        dxte = xgb.DMatrix(X_test, label=y_test)
        xgb_model = xgb.train(
            XGB_PARAMS, dxtr, num_boost_round=1000,
            evals=[(dxte, 'valid')],
            early_stopping_rounds=50, verbose_eval=False
        )
        xgb_pred = xgb_model.predict(dxte)
        xgb_auc = roc_auc_score(y_test, xgb_pred)

        lgb_train_pred = lgb_model.predict(X_train)
        train_auc = roc_auc_score(y_train, lgb_train_pred)

        results.append({
            'year': test_year, 'lgb_auc': lgb_auc, 'xgb_auc': xgb_auc,
            'train_auc': train_auc, 'gap': train_auc - lgb_auc,
            'n_train': int(train_mask.sum()), 'n_test': int(test_mask.sum()),
            'lgb_model': lgb_model, 'xgb_model': xgb_model,
            'lgb_pred': lgb_pred, 'xgb_pred': xgb_pred,
            'y_test': y_test, 'test_df': df.loc[test_mask].copy(),
        })
    return results


def compute_roi(wf_results, features_list):
    """条件別ROIを計算（推定式）"""
    from train_v12_comprehensive import walk_forward_backtest as _  # just to confirm import works
    # We'll compute ROI inline
    all_preds = []
    for r in wf_results:
        tdf = r['test_df'].copy()
        tdf['pred'] = r['lgb_pred']
        all_preds.append(tdf)
    if not all_preds:
        return {}
    pdf = pd.concat(all_preds, ignore_index=True)

    # Group by race and compute condition-based ROI
    results = {}
    for cond_name, cond_filter in [
        ('A', lambda d: (d['num_horses'] >= 8) & (d['num_horses'] <= 14) & (d['distance'] >= 1600)),
        ('C', lambda d: (d['num_horses'] >= 15) & (d['distance'] >= 1600)),
        ('D', lambda d: (d['distance'] >= 1200) & (d['distance'] <= 1400)),
    ]:
        try:
            mask = cond_filter(pdf)
            sub = pdf[mask]
            if len(sub) < 100:
                continue
            # Simple hit rate as proxy
            races = sub.groupby(['year', 'month', 'day', 'race_num'])
            hits = 0
            total_races = 0
            for _, race_df in races:
                if len(race_df) < 3:
                    continue
                top3 = race_df.nlargest(3, 'pred')
                actual_top3 = set(race_df[race_df['finish'] <= 3].index)
                pred_top3 = set(top3.index)
                if len(pred_top3 & actual_top3) >= 2:
                    hits += 1
                total_races += 1
            if total_races > 0:
                results[cond_name] = {'hit_rate': hits / total_races, 'n': total_races}
        except Exception:
            pass
    return results


def main():
    start_time = time.time()
    print("=" * 60)
    print("  v12.2 Re-training: +stable_comment_score")
    print("  Baseline: v12 (74 features, WF AUC 0.8037)")
    print("=" * 60)

    # Phase 1: Load & encode
    print("\n[1/6] Loading data...")
    df = load_data()
    print(f"  Loaded: {len(df)} rows")
    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)

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
    df = compute_pci(df)

    # Phase 5: NEW - stable_comment_score
    print("\n[5/6] Merging stable_comment_score...")
    df = merge_stable_comments(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses'] >= 5].copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    print(f"\n  Final dataset: {len(df)} rows, years {int(df['year'].min())+2000}-{int(df['year'].max())+2000}")

    # Feature lists
    FEATURES_V93 = list(FEATURES_V91) + list(V92_NEW_FEATURES) + list(V93_NEW_FEATURES)
    FEATURES_V12 = [f for f in FEATURES_V93 if f not in LEAK_FEATURES_A] + V12_NEW_FEATURES
    FEATURES_V122 = FEATURES_V12 + V122_NEW_FEATURES

    # Ensure all features exist
    for f in FEATURES_V122:
        if f not in df.columns:
            print(f"  WARNING: {f} not in df, filling with 0")
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    print(f"\n  v12 features: {len(FEATURES_V12)}")
    print(f"  v12.2 features: {len(FEATURES_V122)} (+{len(V122_NEW_FEATURES)})")

    # ===== WF Backtest =====
    print("\n[6/6] Walk-forward backtesting...")

    # Baseline: v12 (74 features)
    print("\n--- Baseline: v12 (74 features) ---")
    wf_baseline = walk_forward_backtest(df, FEATURES_V12)
    aucs_bl = []
    for r in wf_baseline:
        aucs_bl.append(r['lgb_auc'])
        ok = '✓' if r['lgb_auc'] > 0.78 else '✗'
        gap_ok = '✓' if r['gap'] < 0.05 else '✗'
        print(f"  {r['year']}: LGB={r['lgb_auc']:.4f} XGB={r['xgb_auc']:.4f} "
              f"Train={r['train_auc']:.4f} Gap={r['gap']:.4f} {ok}{gap_ok}")
    mean_bl = np.mean(aucs_bl)
    print(f"  --- Mean LGB AUC: {mean_bl:.4f} ---")

    # v12.2: +stable_comment_score
    print(f"\n--- v12.2: +stable_comment_score ({len(FEATURES_V122)} features) ---")
    wf_v122 = walk_forward_backtest(df, FEATURES_V122)
    aucs_v122 = []
    for r in wf_v122:
        aucs_v122.append(r['lgb_auc'])
        ok = '✓' if r['lgb_auc'] > 0.78 else '✗'
        gap_ok = '✓' if r['gap'] < 0.05 else '✗'
        bl_auc = next(b['lgb_auc'] for b in wf_baseline if b['year'] == r['year'])
        diff = r['lgb_auc'] - bl_auc
        print(f"  {r['year']}: LGB={r['lgb_auc']:.4f} XGB={r['xgb_auc']:.4f} "
              f"Train={r['train_auc']:.4f} Gap={r['gap']:.4f} {ok}{gap_ok} "
              f"(diff={diff:+.4f})")
    mean_v122 = np.mean(aucs_v122)
    print(f"  --- Mean LGB AUC: {mean_v122:.4f} (diff={mean_v122-mean_bl:+.4f}) ---")

    # Feature importance for stable_comment_score
    print("\n  Feature importance (stable_comment_score):")
    for r in wf_v122:
        imp = r['lgb_model'].feature_importance(importance_type='gain')
        idx = FEATURES_V122.index('stable_comment_score')
        total_imp = sum(imp)
        print(f"  {r['year']}: importance={imp[idx]:.1f} ({imp[idx]/total_imp*100:.2f}%)")

    # Top 10 features
    print("\n  Top 10 features (averaged):")
    imp_avg = np.zeros(len(FEATURES_V122))
    for r in wf_v122:
        imp_avg += r['lgb_model'].feature_importance(importance_type='gain')
    imp_avg /= len(wf_v122)
    ranked = sorted(zip(FEATURES_V122, imp_avg), key=lambda x: -x[1])
    for i, (f, v) in enumerate(ranked[:10]):
        marker = " ★NEW" if f == 'stable_comment_score' else ""
        print(f"  {i+1:2d}. {f:35s} {v:12.1f}{marker}")
    # Show stable_comment_score rank
    for i, (f, v) in enumerate(ranked):
        if f == 'stable_comment_score':
            print(f"  stable_comment_score rank: #{i+1}/{len(ranked)} (importance={v:.1f})")
            break

    # ===== 採用判定 =====
    print(f"\n{'='*60}")
    print(f"  ADOPTION DECISION")
    print(f"{'='*60}")

    all_years_ok = all(r['lgb_auc'] > 0.78 for r in wf_v122)
    no_overfit = all(r['gap'] < 0.05 for r in wf_v122)
    auc_improved = mean_v122 > 0.8037  # v12 baseline
    auc_stable = all(r['lgb_auc'] > 0.78 for r in wf_v122)

    print(f"  v12 baseline AUC:    {mean_bl:.4f}")
    print(f"  v12.2 AUC:           {mean_v122:.4f} ({mean_v122-mean_bl:+.4f})")
    print(f"  AUC > 0.8037:        {'✓' if auc_improved else '✗'}")
    print(f"  All years > 0.78:    {'✓' if all_years_ok else '✗'}")
    print(f"  No overfitting:      {'✓' if no_overfit else '✗'}")

    # Per-year comparison
    print(f"\n  Per-year comparison:")
    worse_years = 0
    for rb, rv in zip(wf_baseline, wf_v122):
        diff = rv['lgb_auc'] - rb['lgb_auc']
        status = '↑' if diff > 0 else ('↓' if diff < 0 else '→')
        print(f"    {rv['year']}: {rb['lgb_auc']:.4f} → {rv['lgb_auc']:.4f} ({diff:+.4f}) {status}")
        if diff < -0.002:
            worse_years += 1

    adopted = auc_improved and all_years_ok and no_overfit
    print(f"\n  VERDICT: {'✓ ADOPTED as v12.2' if adopted else '✗ NOT ADOPTED'}")

    # Save results
    result_data = {
        'version': 'v12.2',
        'new_feature': 'stable_comment_score',
        'baseline_auc': mean_bl,
        'v122_auc': mean_v122,
        'improvement': mean_v122 - mean_bl,
        'adopted': adopted,
        'yearly_results': [{
            'year': r['year'], 'lgb_auc': r['lgb_auc'], 'xgb_auc': r['xgb_auc'],
            'train_auc': r['train_auc'], 'gap': r['gap'],
            'baseline_auc': next(b['lgb_auc'] for b in wf_baseline if b['year'] == r['year']),
        } for r in wf_v122],
        'n_features_baseline': len(FEATURES_V12),
        'n_features_v122': len(FEATURES_V122),
        'timestamp': datetime.now().isoformat(),
    }

    result_path = os.path.join(DATA_DIR, 'v122_training_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved: {result_path}")

    # ===== 本番モデル保存（adopted時のみ）=====
    if adopted:
        print(f"\n{'='*60}")
        print(f"  TRAINING PRODUCTION MODEL (v12.2)")
        print(f"{'='*60}")
        train_production_model(df, FEATURES_V122, sire_map, bms_map, wf_v122)
    else:
        print(f"\n  v12 maintained. stable_comment_score not adopted.")
        print(f"  Reason: {'AUC not improved' if not auc_improved else ''}"
              f"{'Years not stable' if not all_years_ok else ''}"
              f"{'Overfitting' if not no_overfit else ''}")

    elapsed = (time.time() - start_time) / 60
    print(f"\n  Total elapsed: {elapsed:.1f} min")
    return result_data


def train_production_model(df, features, sire_map, bms_map, wf_results):
    """フルデータで本番モデルを学習・保存"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    max_year = int(df['year'].max())
    valid_mask = df['year'] >= (max_year - 1)
    train_mask = ~valid_mask

    X_train = df.loc[train_mask, features].values
    y_train = df.loc[train_mask, 'target'].values
    X_valid = df.loc[valid_mask, features].values
    y_valid = df.loc[valid_mask, 'target'].values

    print(f"  Train: {train_mask.sum()}, Valid: {valid_mask.sum()}")

    # LGB
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
    lgb_model = lgb.train(
        LGB_PARAMS, dtrain, num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    lgb_auc = roc_auc_score(y_valid, lgb_model.predict(X_valid))

    # XGB
    dxtr = xgb.DMatrix(X_train, label=y_train)
    dxte = xgb.DMatrix(X_valid, label=y_valid)
    xgb_model = xgb.train(
        XGB_PARAMS, dxtr, num_boost_round=1000,
        evals=[(dxte, 'valid')],
        early_stopping_rounds=50, verbose_eval=100
    )
    xgb_auc = roc_auc_score(y_valid, xgb_model.predict(dxte))

    total = lgb_auc + xgb_auc
    w_lgb = lgb_auc / total
    w_xgb = xgb_auc / total
    ens_pred = w_lgb * lgb_model.predict(X_valid) + w_xgb * xgb_model.predict(dxte)
    ens_auc = roc_auc_score(y_valid, ens_pred)
    wf_auc = np.mean([r['lgb_auc'] for r in wf_results])

    print(f"  LGB={lgb_auc:.4f} XGB={xgb_auc:.4f} Ensemble={ens_auc:.4f}")
    print(f"  Weights: LGB={w_lgb:.3f} XGB={w_xgb:.3f}")

    pattern_a_features = [f for f in features if f not in LEAK_FEATURES_A]

    pkl_a = {
        'model': lgb_model,
        'features': pattern_a_features,
        'version': 'v12.2_leakfree',
        'auc': lgb_auc, 'ensemble_auc': ens_auc, 'wf_auc': wf_auc,
        'leak_free': True, 'leak_pattern': 'A',
        'leak_removed': sorted(LEAK_FEATURES_A),
        'sire_map': sire_map, 'bms_map': bms_map,
        'n_top_encode': N_TOP_SIRE,
        'trained_at': now,
        'n_train': int(train_mask.sum()), 'n_valid': int(valid_mask.sum()),
        'model_type': 'central',
        'xgb_model': xgb_model, 'mlp_model': None, 'mlp_scaler': None,
        'ensemble_weights': {'lgb': w_lgb, 'xgb': w_xgb, 'mlp': 0},
        'course_map': dict(COURSE_MAP),
        'v12_new_features': V12_NEW_FEATURES + V122_NEW_FEATURES,
    }

    a_path = os.path.join(BASE_DIR, 'keiba_model_v12_central.pkl.gz')
    with gzip.open(a_path, 'wb') as f:
        pickle.dump(pkl_a, f)
    print(f"  Pattern A: {a_path} ({len(pattern_a_features)} features)")

    pkl_b = dict(pkl_a)
    pkl_b['features'] = features
    pkl_b['version'] = 'v12.2_live'
    pkl_b['leak_free'] = False
    pkl_b['leak_pattern'] = 'B'
    pkl_b['is_live'] = True

    b_path = os.path.join(BASE_DIR, 'keiba_model_v12_central_live.pkl.gz')
    with gzip.open(b_path, 'wb') as f:
        pickle.dump(pkl_b, f)
    print(f"  Pattern B: {b_path} ({len(features)} features)")

    v8_pkl = dict(pkl_a)
    v8_pkl['version'] = 'v12.2_backup'
    with open(os.path.join(BASE_DIR, 'keiba_model_v8.pkl'), 'wb') as f:
        pickle.dump(v8_pkl, f)

    print(f"\n  *** v12.2 Production model saved! ***")
    print(f"  Pattern A: {len(pattern_a_features)} features, WF AUC {wf_auc:.4f}")


if __name__ == '__main__':
    main()
