#!/usr/bin/env python3
"""v13.4 馬体重長期トレンド特徴量

v13.3(115特徴量, WF AUC 0.84685) に対して:
  1. weight_ma5 - 過去5走の馬体重移動平均
  2. weight_trend - 過去5走の馬体重線形トレンド（傾き）
  3. weight_peak_diff - ピーク馬体重からの乖離

全てexpanding window（過去走のみ使用）でリークなし。
当レースのhorse_weightは使わず、前走以前の馬体重のみを参照。

採用基準: WF AUC > 0.84685 (v13.3) かつ全年AUC > 0.78 かつ gap < 0.05
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
from train_v133_extended import (
    V133_NEW_FEATURES, V133_DEFAULTS,
    merge_stable_comment_combined,
)


# =====================================================
# v13.4 新規特徴量: 馬体重トレンド
# =====================================================

V134_NEW_FEATURES = [
    'weight_ma5',        # 過去5走馬体重移動平均
    'weight_trend',      # 過去5走馬体重線形トレンド（傾き）
    'weight_peak_diff',  # ピーク馬体重からの乖離
]

V134_DEFAULTS = {
    'weight_ma5': 0,         # 不明=0 (mean fillは後で適用)
    'weight_trend': 0,       # トレンドなし
    'weight_peak_diff': 0,   # 乖離なし
}


def compute_weight_trends(df):
    """馬体重長期トレンド特徴量を計算（expanding window, 過去走のみ）

    horse_weightは当レースの馬体重だがPattern Aではリーク対象。
    ここでは「前走以前の馬体重」のみを使い、トレンドを計算する。
    → 当レースのhorse_weightは参照しない。

    計算ロジック:
    - 馬IDでソートし、時系列順に過去走の馬体重を蓄積
    - weight_ma5: 直近5走の馬体重平均（5走未満なら全走平均）
    - weight_trend: 直近5走の線形回帰傾き（2走以上必要）
    - weight_peak_diff: 過去走のピーク馬体重 - 直近馬体重
    """
    print("  Computing weight trends (expanding window)...")

    # horse_weightを数値化
    hw = pd.to_numeric(df['horse_weight'], errors='coerce')

    # 結果格納
    weight_ma5 = np.full(len(df), np.nan)
    weight_trend = np.full(len(df), np.nan)
    weight_peak_diff = np.full(len(df), np.nan)

    # horse_idでグループ化、時系列順にソート済みと仮定
    # dfはload_data()後のデータなのでyear/month/dayでソートされている
    horse_ids = df['horse_id'].values
    unique_horses = df['horse_id'].unique()

    # horse_idごとにインデックスを事前計算
    horse_groups = df.groupby('horse_id', sort=False).indices

    processed = 0
    for hid, idxs in horse_groups.items():
        # 時系列順のインデックス（dfの行順＝時系列順）
        idxs_sorted = sorted(idxs)
        weights = hw.iloc[idxs_sorted].values

        # 各レースについて、「そのレースより前の馬体重」のみ使用
        past_weights = []
        for i, idx in enumerate(idxs_sorted):
            w = weights[i]

            if len(past_weights) > 0:
                # 過去走データがある場合
                recent = past_weights[-5:]  # 直近5走

                # weight_ma5: 直近5走の平均
                weight_ma5[idx] = np.mean(recent)

                # weight_trend: 直近5走の線形傾き
                if len(recent) >= 2:
                    x = np.arange(len(recent))
                    # np.polyfit degree 1 → [slope, intercept]
                    slope = np.polyfit(x, recent, 1)[0]
                    weight_trend[idx] = slope

                # weight_peak_diff: ピーク馬体重 - 直近馬体重
                peak = max(past_weights)
                weight_peak_diff[idx] = peak - recent[-1]

            # 現在のレースの馬体重を過去走リストに追加（NaNでなければ）
            if not np.isnan(w) and w > 0:
                past_weights.append(w)

        processed += 1
        if processed % 10000 == 0:
            print(f"    {processed}/{len(unique_horses)} horses processed...")

    df['weight_ma5'] = weight_ma5
    df['weight_trend'] = weight_trend
    df['weight_peak_diff'] = weight_peak_diff

    # NaN fill (初出走の馬など)
    # weight_ma5: 全体平均で埋める
    global_mean_weight = hw.dropna().mean()
    df['weight_ma5'] = df['weight_ma5'].fillna(global_mean_weight)
    df['weight_trend'] = df['weight_trend'].fillna(0)
    df['weight_peak_diff'] = df['weight_peak_diff'].fillna(0)

    # Coverage report
    has_ma5 = (df['weight_ma5'] != global_mean_weight).sum()
    has_trend = (df['weight_trend'] != 0).sum()
    has_peak = (df['weight_peak_diff'] != 0).sum()
    total = len(df)
    print(f"    weight_ma5:       {has_ma5}/{total} ({has_ma5/total*100:.1f}%) non-default")
    print(f"    weight_trend:     {has_trend}/{total} ({has_trend/total*100:.1f}%) non-zero")
    print(f"    weight_peak_diff: {has_peak}/{total} ({has_peak/total*100:.1f}%) non-zero")

    return df


# =====================================================
# WF Backtest (v13.3から流用)
# =====================================================

def walk_forward_backtest(df, features, label='target', years=range(2020, 2026)):
    results = []
    for test_year in years:
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue
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

        results.append({
            'year': test_year, 'lgb_auc': auc_lgb, 'xgb_auc': auc_xgb,
            'train_auc': auc_tr, 'gap': auc_tr - auc_lgb,
            'lgb_model': m_lgb, 'xgb_model': m_xgb,
        })
    return results


def print_wf(results, label=''):
    print(f"\n  WF: {label}")
    print(f"  {'='*60}")
    aucs = []
    for r in results:
        aucs.append(r['lgb_auc'])
        ok = 'Y' if r['lgb_auc'] > 0.78 else 'N'
        gok = 'Y' if r['gap'] < 0.05 else 'N'
        print(f"  {r['year']}: LGB={r['lgb_auc']:.4f} XGB={r['xgb_auc']:.4f} "
              f"Train={r['train_auc']:.4f} Gap={r['gap']:.4f} {ok}{gok}")
    mean_auc = np.mean(aucs)
    print(f"  --- Mean LGB AUC: {mean_auc:.5f} ---")
    return mean_auc


def feature_importance(results, features):
    imp = np.zeros(len(features))
    for r in results:
        imp += r['lgb_model'].feature_importance(importance_type='gain')
    imp /= len(results)
    return sorted(zip(features, imp), key=lambda x: -x[1])


# =====================================================
# Main
# =====================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("  v13.4 Horse Weight Trend Features")
    print(f"  Baseline: v13.3 WF AUC 0.84685 (115 features)")
    print(f"  New: weight_ma5, weight_trend, weight_peak_diff")
    print("=" * 60)

    # Phase 1-7: identical to v13.3
    print("\n[1/9] Loading data...")
    df = load_data()
    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)

    print("\n[2/9] Training data...")
    tt_data = load_training_times()
    df = merge_training_features(df, tt_data)

    print("\n[3/9] Historical stats...")
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)
    df = compute_lag_features(df)
    df = build_features(df)
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)

    print("\n[4/9] v12 features...")
    df = merge_speed_index(df)
    df = merge_training_1f(df)
    df = merge_sire_shinba(df)
    df = merge_dam_stats(df)
    df = compute_pci(df)

    print("\n[5/9] v13 JRDB features...")
    df = merge_jrdb_train_features(df)

    if '_nk_rid' not in df.columns:
        df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    if '_uma' not in df.columns:
        df['_uma'] = df['umaban'].astype(int)

    print("\n[6/9] v13.2 JRDB extended features...")
    df = merge_cha_features(df)
    df = merge_jo_features(df)
    df = merge_kta_features(df)
    df = merge_ze_features(df)
    df = merge_sr_features(df)
    df = merge_kka_features(df)

    df.drop(columns=['_nk_rid', '_uma'], inplace=True, errors='ignore')

    print("\n[7/9] v13.3 features...")
    df = merge_stable_comment_combined(df)

    # Phase 8: v13.4 NEW - horse weight trends
    print("\n[8/9] v13.4 horse weight trend features...")
    df = compute_weight_trends(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses'] >= 5].copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    print(f"\n  Final: {len(df)} rows")

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

    # Fill missing
    all_defaults = {**JRDB_DEFAULTS, **V132_DEFAULTS, **V133_DEFAULTS, **V134_DEFAULTS}
    for f in F_V134:
        if f not in df.columns:
            df[f] = all_defaults.get(f, 0)
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(all_defaults.get(f, 0))

    print(f"  v13.3: {len(F_V133)} features (baseline)")
    print(f"  v13.4: {len(F_V134)} features (+{len(V134_NEW_FEATURES)})")

    # ===== WF Backtest =====
    print(f"\n{'='*60}")
    print(f"  [9/9] Walk-Forward Backtest")
    print(f"{'='*60}")

    # v13.3 baseline (re-run for fair comparison with same data)
    print("\n--- v13.3 baseline ---")
    wf133 = walk_forward_backtest(df, F_V133)
    auc133 = print_wf(wf133, f'v13.3 ({len(F_V133)} feats)')

    # v13.4 with weight trend features
    print("\n--- v13.4 +weight_trends ---")
    wf134 = walk_forward_backtest(df, F_V134)
    auc134 = print_wf(wf134, f'v13.4 ({len(F_V134)} feats)')

    # Feature importance
    imp = feature_importance(wf134, F_V134)
    print(f"\n  Top 30 by importance:")
    for i, (f, v) in enumerate(imp[:30]):
        tag = " *NEW" if f in V134_NEW_FEATURES else ""
        print(f"  {i+1:2d}. {f:35s} {v:12.1f}{tag}")

    # New feature positions
    print(f"\n  New feature positions:")
    for f_name in V134_NEW_FEATURES:
        for i, (f, v) in enumerate(imp):
            if f == f_name:
                print(f"    {f_name}: rank {i+1}/{len(F_V134)}, importance {v:.1f}")
                break

    # Individual contribution check
    print(f"\n  Individual contribution check:")
    for feat in V134_NEW_FEATURES:
        feats_wo = [f for f in F_V134 if f != feat]
        wf_wo = walk_forward_backtest(df, feats_wo)
        auc_wo = np.mean([r['lgb_auc'] for r in wf_wo])
        contrib = auc134 - auc_wo
        print(f"    {feat:25s}: with={auc134:.5f}, without={auc_wo:.5f}, contrib={contrib:+.5f}")

    # All 3 combined contribution
    feats_wo_all = [f for f in F_V134 if f not in V134_NEW_FEATURES]
    wf_wo_all = walk_forward_backtest(df, feats_wo_all)
    auc_wo_all = np.mean([r['lgb_auc'] for r in wf_wo_all])
    combined_contrib = auc134 - auc_wo_all
    print(f"    {'ALL 3 COMBINED':25s}: with={auc134:.5f}, without={auc_wo_all:.5f}, contrib={combined_contrib:+.5f}")

    # Adoption check
    baseline_auc = 0.84685  # v13.3
    delta = auc134 - baseline_auc
    all_ok = all(r['lgb_auc'] > 0.78 for r in wf134)
    no_overfit = all(r['gap'] < 0.05 for r in wf134)
    adopted = auc134 > baseline_auc and all_ok and no_overfit

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  v13.3 baseline:    {auc133:.5f} ({len(F_V133)} feats)")
    print(f"  v13.4 (+weight):   {auc134:.5f} ({len(F_V134)} feats)")
    print(f"  Delta:             {delta:+.5f}")
    print(f"  Combined contrib:  {combined_contrib:+.5f}")
    print(f"  AUC > 0.84685:     {'Y' if auc134 > baseline_auc else 'N'}")
    print(f"  All years > 0.78:  {'Y' if all_ok else 'N'}")
    print(f"  No overfitting:    {'Y' if no_overfit else 'N'}")
    print(f"\n  VERDICT: {'ADOPTED as v13.4' if adopted else 'NOT ADOPTED'}")

    # Yearly comparison
    print(f"\n  Year-by-year comparison:")
    print(f"  {'Year':6s} {'v13.3':>8s} {'v13.4':>8s} {'Delta':>8s} {'Gap134':>8s} {'OK':>4s}")
    for r3, r4 in zip(wf133, wf134):
        d = r4['lgb_auc'] - r3['lgb_auc']
        gok = 'Y' if r4['gap'] < 0.05 else 'N'
        print(f"  {r4['year']:6d} {r3['lgb_auc']:.5f} {r4['lgb_auc']:.5f} {d:+.5f} {r4['gap']:.4f} {gok}")

    # Save results
    result = {
        'v133_baseline_auc': baseline_auc,
        'v133_rerun_auc': auc133,
        'v134_auc': auc134,
        'delta': delta,
        'adopted': adopted,
        'new_features': V134_NEW_FEATURES,
        'individual_contributions': {},
        'combined_contribution': combined_contrib,
        'v133_yearly': [{'year': r['year'], 'lgb_auc': r['lgb_auc'], 'xgb_auc': r['xgb_auc'],
                          'train_auc': r['train_auc'], 'gap': r['gap']} for r in wf133],
        'v134_yearly': [{'year': r['year'], 'lgb_auc': r['lgb_auc'], 'xgb_auc': r['xgb_auc'],
                          'train_auc': r['train_auc'], 'gap': r['gap']} for r in wf134],
        'best_features': F_V134 if adopted else F_V133,
        'timestamp': datetime.now().isoformat(),
    }
    rpath = os.path.join(DATA_DIR, 'v134_training_results.json')
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Saved: {rpath}")

    # If adopted, train production model
    if adopted:
        print(f"\n{'='*60}")
        print(f"  TRAINING PRODUCTION MODEL (v13.4)")
        print(f"{'='*60}")
        _train_production(df, F_V134, sire_map, bms_map, wf134, jrdb_sel, 'v13.4')

    print(f"\n  Elapsed: {(time.time()-t0)/60:.1f} min")


def _train_production(df, features, sire_map, bms_map, wf, jrdb_sel, label):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    my = int(df['year'].max())
    vm = df['year'] >= (my - 1)
    tm = ~vm
    X_tr, y_tr = df.loc[tm, features].values, df.loc[tm, 'target'].values
    X_va, y_va = df.loc[vm, features].values, df.loc[vm, 'target'].values
    print(f"  Train: {tm.sum()}, Valid: {vm.sum()}, Features: {len(features)}")

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

    pa_feats = [f for f in features if f not in LEAK_FEATURES_A]
    v132_new = [f for f in V132_NEW_FEATURES if f in features]
    v133_new = [f for f in V133_NEW_FEATURES if f in features]
    v134_new = [f for f in V134_NEW_FEATURES if f in features]

    pkl = {
        'model': ml, 'features': pa_feats, 'version': 'v134_leakfree',
        'auc': al, 'wf_auc': wf_auc, 'leak_free': True, 'leak_pattern': 'A',
        'leak_removed': sorted(LEAK_FEATURES_A),
        'sire_map': sire_map, 'bms_map': bms_map, 'n_top_encode': N_TOP_SIRE,
        'trained_at': now, 'model_type': 'central',
        'xgb_model': mx, 'mlp_model': None, 'mlp_scaler': None,
        'ensemble_weights': {'lgb': wl, 'xgb': wx, 'mlp': 0},
        'course_map': dict(COURSE_MAP),
        'jrdb_features': jrdb_sel, 'v132_features': v132_new,
        'v133_features': v133_new, 'v134_features': v134_new,
        'label': label,
    }
    ap = os.path.join(BASE_DIR, 'keiba_model_v134_central.pkl.gz')
    with gzip.open(ap, 'wb') as f:
        pickle.dump(pkl, f)
    print(f"  Pattern A: {ap}")

    pb_feats = features + JRDB_LIVE_FEATURES
    pkl_b = dict(pkl)
    pkl_b['features'] = pb_feats
    pkl_b['version'] = 'v134_live'
    pkl_b['leak_free'] = False
    pkl_b['leak_pattern'] = 'B'
    pkl_b['is_live'] = True
    bp = os.path.join(BASE_DIR, 'keiba_model_v134_central_live.pkl.gz')
    with gzip.open(bp, 'wb') as f:
        pickle.dump(pkl_b, f)
    print(f"  Pattern B: {bp}")
    print(f"\n  *** v13.4 saved! A={len(pa_feats)} feats, B={len(pb_feats)} feats ***")


if __name__ == '__main__':
    main()
