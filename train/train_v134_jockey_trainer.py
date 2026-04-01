#!/usr/bin/env python3
"""v13.4 騎手×調教師相性特徴量

v13.3(115特徴量, WF AUC 0.84685) に対して5特徴量を追加検証:
  1. jt_wr       - 騎手×調教師の過去組み合わせ勝率 (expanding, alpha=20)
  2. jt_top3r    - 騎手×調教師の過去組み合わせ複勝率 (expanding, alpha=20)
  3. jockey_course_wr_v2 - 騎手×競馬場(course_code)の勝率 (expanding, alpha=15)
  4. trainer_course_wr   - 調教師×競馬場(course_code)の勝率 (expanding, alpha=15)
  5. jockey_dist_wr      - 騎手×距離カテゴリの勝率 (expanding, alpha=15)

データソース: jra_races_full.csv (jockey_id, trainer_id, course_code, distance)
採用基準: WF AUC > 0.84685, gap < 0.05, 全年AUC > 0.78
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
# v13.4 新規特徴量
# =====================================================

V134_NEW_FEATURES = [
    'jt_wr',              # 騎手×調教師 勝率
    'jt_top3r',           # 騎手×調教師 複勝率
    'jockey_course_wr_v2',# 騎手×競馬場 勝率 (course_code使用)
    'trainer_course_wr',  # 調教師×競馬場 勝率
    'jockey_dist_wr',     # 騎手×距離カテゴリ 勝率
]

V134_DEFAULTS = {f: 0.0 for f in V134_NEW_FEATURES}


def compute_jockey_trainer_features(df):
    """騎手×調教師 / 騎手×競馬場 / 調教師×競馬場 / 騎手×距離カテゴリの相性特徴量

    全てexpanding window (cumsum - current) でリークフリー。
    Bayesian smoothing (alpha prior) で低サンプル時の過学習を防止。
    """
    print("  Computing jockey×trainer interaction features...")

    df = df.sort_values('date_num').reset_index(drop=True)

    # Ensure is_win / is_top3 exist
    if 'is_win' not in df.columns:
        df['is_win'] = (df['finish'] == 1).astype(int)
    if 'is_top3' not in df.columns:
        df['is_top3'] = (df['finish'] <= 3).astype(int)

    global_wr = df['is_win'].mean()
    global_t3 = df['is_top3'].mean()

    # dist_cat for jockey_dist_wr (same bins as existing)
    df['_dist_cat_jd'] = pd.cut(
        df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # --- 1. 騎手×調教師 勝率 (alpha=20) ---
    jt_key = ['jockey_id', 'trainer_id']
    df['_jt_cum_races'] = df.groupby(jt_key).cumcount()
    df['_jt_cum_wins'] = df.groupby(jt_key)['is_win'].cumsum() - df['is_win']
    df['_jt_cum_top3'] = df.groupby(jt_key)['is_top3'].cumsum() - df['is_top3']
    alpha_jt = 20
    df['jt_wr'] = (df['_jt_cum_wins'] + alpha_jt * global_wr) / (df['_jt_cum_races'] + alpha_jt)
    df['jt_top3r'] = (df['_jt_cum_top3'] + alpha_jt * global_t3) / (df['_jt_cum_races'] + alpha_jt)

    # --- 2. 騎手×競馬場 勝率 (alpha=15, course_code使用) ---
    jc_key = ['jockey_id', 'course_code']
    df['_jc2_cum_races'] = df.groupby(jc_key).cumcount()
    df['_jc2_cum_wins'] = df.groupby(jc_key)['is_win'].cumsum() - df['is_win']
    alpha_jc = 15
    df['jockey_course_wr_v2'] = (df['_jc2_cum_wins'] + alpha_jc * global_wr) / (df['_jc2_cum_races'] + alpha_jc)

    # --- 3. 調教師×競馬場 勝率 (alpha=15) ---
    tc_key = ['trainer_id', 'course_code']
    df['_tc_cum_races'] = df.groupby(tc_key).cumcount()
    df['_tc_cum_wins'] = df.groupby(tc_key)['is_win'].cumsum() - df['is_win']
    alpha_tc = 15
    df['trainer_course_wr'] = (df['_tc_cum_wins'] + alpha_tc * global_wr) / (df['_tc_cum_races'] + alpha_tc)

    # --- 4. 騎手×距離カテゴリ 勝率 (alpha=15) ---
    jd_key = ['jockey_id', '_dist_cat_jd']
    df['_jd_cum_races'] = df.groupby(jd_key).cumcount()
    df['_jd_cum_wins'] = df.groupby(jd_key)['is_win'].cumsum() - df['is_win']
    alpha_jd = 15
    df['jockey_dist_wr'] = (df['_jd_cum_wins'] + alpha_jd * global_wr) / (df['_jd_cum_races'] + alpha_jd)

    # Cleanup
    drop_cols = [c for c in df.columns if c.startswith('_jt_') or c.startswith('_jc2_')
                 or c.startswith('_tc_') or c.startswith('_jd_') or c == '_dist_cat_jd']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Sort back for lag features
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)

    # Coverage stats
    for f in V134_NEW_FEATURES:
        nonzero = (df[f] != V134_DEFAULTS.get(f, 0)).mean() * 100
        print(f"    {f}: {nonzero:.1f}% non-default")

    return df


# =====================================================
# WF Backtest (same as v13.3)
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
    print(f"  --- Mean LGB AUC: {mean_auc:.4f} ---")
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
    print("  v13.4 騎手×調教師 相性特徴量")
    print(f"  Baseline: v13.3 WF AUC 0.84685 (115 features)")
    print(f"  New: {len(V134_NEW_FEATURES)} features")
    for f in V134_NEW_FEATURES:
        print(f"    - {f}")
    print("=" * 60)

    # Phase 1-7: same as v13.3
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

    print("\n[7/9] v13.3 stable comment...")
    df = merge_stable_comment_combined(df)

    # Phase 8: v13.4 NEW features
    print("\n[8/9] v13.4 jockey×trainer interaction features...")
    df = compute_jockey_trainer_features(df)

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

    # v13.3 baseline
    print("\n--- v13.3 baseline ---")
    wf133 = walk_forward_backtest(df, F_V133)
    auc133 = print_wf(wf133, f'v13.3 ({len(F_V133)} feats)')

    # v13.4 all 5 features
    print("\n--- v13.4 +5 jockey×trainer features ---")
    wf134 = walk_forward_backtest(df, F_V134)
    auc134 = print_wf(wf134, f'v13.4 ({len(F_V134)} feats)')

    # Individual contribution check
    print(f"\n  Individual feature contributions:")
    contribs = {}
    for feat in V134_NEW_FEATURES:
        feats_wo = [f for f in F_V134 if f != feat]
        wf_wo = walk_forward_backtest(df, feats_wo)
        auc_wo = np.mean([r['lgb_auc'] for r in wf_wo])
        contrib = auc134 - auc_wo
        contribs[feat] = contrib
        print(f"    {feat:30s}: {contrib:+.5f}")

    # Try subsets: remove features with negative contribution
    negative_feats = [f for f, c in contribs.items() if c < -0.0005]
    if negative_feats:
        print(f"\n  Removing negative contributors: {negative_feats}")
        F_V134_PRUNED = [f for f in F_V134 if f not in negative_feats]
        wf134p = walk_forward_backtest(df, F_V134_PRUNED)
        auc134p = print_wf(wf134p, f'v13.4 pruned ({len(F_V134_PRUNED)} feats)')
    else:
        F_V134_PRUNED = F_V134
        wf134p = wf134
        auc134p = auc134

    # Feature importance (full v13.4)
    imp = feature_importance(wf134, F_V134)
    print(f"\n  Top 30 by importance:")
    for i, (f, v) in enumerate(imp[:30]):
        tag = " *NEW" if f in V134_NEW_FEATURES else ""
        print(f"  {i+1:2d}. {f:35s} {v:12.1f}{tag}")

    # New features positions
    print(f"\n  New feature importance positions:")
    for i, (f, v) in enumerate(imp):
        if f in V134_NEW_FEATURES:
            print(f"    {f:30s}: rank {i+1}/{len(F_V134)}, importance {v:.1f}, contrib {contribs[f]:+.5f}")

    # Determine best
    candidates = [
        ('v13.3', auc133, wf133, F_V133),
        ('v13.4', auc134, wf134, F_V134),
    ]
    if negative_feats:
        candidates.append(('v13.4-pruned', auc134p, wf134p, F_V134_PRUNED))

    best_label, best_auc, best_wf, best_feats = max(candidates, key=lambda x: x[1])

    # Adoption check
    baseline_auc = 0.84685
    delta = best_auc - baseline_auc
    all_ok = all(r['lgb_auc'] > 0.78 for r in best_wf)
    no_overfit = all(r['gap'] < 0.05 for r in best_wf)
    adopted = best_auc > baseline_auc and all_ok and no_overfit

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  v13.3 baseline:   {auc133:.5f} ({len(F_V133)} feats)")
    print(f"  v13.4 full:       {auc134:.5f} ({len(F_V134)} feats)")
    if negative_feats:
        print(f"  v13.4 pruned:     {auc134p:.5f} ({len(F_V134_PRUNED)} feats)")
    print(f"  Best:             {best_auc:.5f} ({best_label})")
    print(f"  Delta vs v13.3:   {delta:+.5f}")
    print(f"  AUC > 0.84685:    {'Y' if best_auc > baseline_auc else 'N'}")
    print(f"  All years > 0.78: {'Y' if all_ok else 'N'}")
    print(f"  No overfitting:   {'Y' if no_overfit else 'N'}")
    print(f"\n  VERDICT: {'ADOPTED as v13.4' if adopted else 'NOT ADOPTED'}")

    # Save results
    adopted_feats = [f for f in V134_NEW_FEATURES if f not in negative_feats] if negative_feats else V134_NEW_FEATURES
    result = {
        'v133_baseline_auc': auc133,
        'v134_auc': auc134,
        'v134_pruned_auc': auc134p if negative_feats else None,
        'best_auc': best_auc,
        'best_label': best_label,
        'delta': delta,
        'adopted': adopted,
        'new_features': V134_NEW_FEATURES,
        'contributions': contribs,
        'negative_feats': negative_feats,
        'adopted_features': adopted_feats if adopted else [],
        'best_features': best_feats,
        'yearly': [{'year': r['year'], 'lgb_auc': r['lgb_auc'], 'xgb_auc': r['xgb_auc'],
                     'train_auc': r['train_auc'], 'gap': r['gap']} for r in best_wf],
        'v134_yearly': [{'year': r['year'], 'lgb_auc': r['lgb_auc'], 'xgb_auc': r['xgb_auc'],
                          'train_auc': r['train_auc'], 'gap': r['gap']} for r in wf134],
        'v133_yearly': [{'year': r['year'], 'lgb_auc': r['lgb_auc'], 'xgb_auc': r['xgb_auc'],
                          'train_auc': r['train_auc'], 'gap': r['gap']} for r in wf133],
        'timestamp': datetime.now().isoformat(),
    }
    rpath = os.path.join(DATA_DIR, 'v134_training_results.json')
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Saved: {rpath}")

    # Train production model if adopted
    if adopted:
        print(f"\n{'='*60}")
        print(f"  TRAINING PRODUCTION MODEL (v13.4)")
        print(f"{'='*60}")
        _train_production(df, best_feats, sire_map, bms_map, best_wf, jrdb_sel, best_label)

    elapsed = (time.time() - t0) / 60
    print(f"\n  Elapsed: {elapsed:.1f} min")


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


if __name__ == '__main__':
    main()
