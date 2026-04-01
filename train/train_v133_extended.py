#!/usr/bin/env python3
"""v13.3 拡張データカバレッジ + stable_comment_score

v13.2(114特徴量, WF AUC 0.8468) に対して:
  1. Speed Index 2020年追加 → index特徴量のWF2020カバレッジ向上
  2. Training Times 2020-2024フルカバレッジ → 調教特徴量カバレッジ向上
  3. 厩舎コメント 2020-2025 → stable_comment_score 新規追加
  4. データ分析(race_analysis) 2020-2025 → stable_comment_scoreソース統合

採用基準: WF AUC > 0.8468 (v13.2) かつ全年AUC > 0.78 かつ gap < 0.05
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


# =====================================================
# v13.3 新規特徴量
# =====================================================

V133_NEW_FEATURES = [
    'stable_comment_score',  # 厩舎コメントスコア(-3〜+3)
]

V133_DEFAULTS = {
    'stable_comment_score': 0,  # 未マッチ=中立
}


def merge_stable_comment_combined(df):
    """厩舎コメントスコアをマージ（race_analysis + stable_comments統合）

    2ソースを統合して最大カバレッジを実現:
    - netkeiba_race_analysis.csv (comment.html, 特別/新馬レース, 2020-2025)
    - netkeiba_stable_comments.csv (bulk scrape, 2025+)
    """
    sources = [
        ('netkeiba_race_analysis.csv', ['race_id', 'umaban', 'score']),
        ('netkeiba_stable_comments.csv', ['race_id', 'umaban', 'score']),
    ]

    all_sc = []
    for fname, cols in sources:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"    {fname}: not found")
            continue
        sc = pd.read_csv(path, encoding='utf-8-sig', dtype={'race_id': str, 'umaban': str},
                         usecols=cols)
        sc['score'] = pd.to_numeric(sc['score'], errors='coerce').fillna(0)
        sc['_nk_rid'] = sc['race_id'].astype(str).str.zfill(12)
        sc['_uma_str'] = sc['umaban'].astype(str)
        print(f"    {fname}: {len(sc)} rows")
        all_sc.append(sc[['_nk_rid', '_uma_str', 'score']])

    if not all_sc:
        print("    WARNING: No stable comment data found")
        df['stable_comment_score'] = 0
        return df

    # Combine and deduplicate (race_analysis first, stable_comments overwrites)
    combined = pd.concat(all_sc, ignore_index=True)
    combined = combined.drop_duplicates(subset=['_nk_rid', '_uma_str'], keep='last')
    print(f"    Combined unique: {len(combined)} entries")

    df = _build_nk_race_id(df)
    df['_uma_str'] = df['umaban'].astype(int).astype(str)

    merged = df.merge(combined, on=['_nk_rid', '_uma_str'], how='left', suffixes=('', '_sc'))
    df['stable_comment_score'] = merged['score'].fillna(0).values

    matched = merged['score'].notna().sum()
    total = len(df)
    print(f"    Matched: {matched}/{total} ({matched/total*100:.1f}%)")

    # Year-level coverage
    if 'year' in df.columns:
        temp_match = merged['score'].notna()
        for y in sorted(df['year'].unique()):
            mask = df['year'] == y
            m = temp_match[mask].sum()
            t = mask.sum()
            print(f"      Year {int(y)+2000}: {m}/{t} ({m/t*100:.1f}%)")

    df.drop(columns=['_uma_str'], inplace=True, errors='ignore')
    return df


# =====================================================
# WF Backtest
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
    print("  v13.3 Extended Data Coverage + Stable Comment Score")
    print(f"  Baseline: v13.2 WF AUC 0.8468 (114 features)")
    print(f"  Changes: expanded SI/TT coverage + stable_comment_score")
    print("=" * 60)

    # Phase 1-4: identical to v13.2
    print("\n[1/8] Loading data...")
    df = load_data()
    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)

    print("\n[2/8] Training data...")
    tt_data = load_training_times()
    df = merge_training_features(df, tt_data)

    print("\n[3/8] Historical stats...")
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)
    df = compute_lag_features(df)
    df = build_features(df)
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)

    print("\n[4/8] v12 features (expanded SI/TT data)...")
    df = merge_speed_index(df)
    df = merge_training_1f(df)
    df = merge_sire_shinba(df)
    df = merge_dam_stats(df)
    df = compute_pci(df)

    print("\n[5/8] v13 JRDB features...")
    df = merge_jrdb_train_features(df)

    if '_nk_rid' not in df.columns:
        df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    if '_uma' not in df.columns:
        df['_uma'] = df['umaban'].astype(int)

    print("\n[6/8] v13.2 JRDB extended features...")
    df = merge_cha_features(df)
    df = merge_jo_features(df)
    df = merge_kta_features(df)
    df = merge_ze_features(df)
    df = merge_sr_features(df)
    df = merge_kka_features(df)

    df.drop(columns=['_nk_rid', '_uma'], inplace=True, errors='ignore')

    # Phase 7: v13.3 NEW features
    print("\n[7/8] v13.3 new features...")
    df = merge_stable_comment_combined(df)

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

    # Fill missing
    all_defaults = {**JRDB_DEFAULTS, **V132_DEFAULTS, **V133_DEFAULTS}
    for f in F_V133:
        if f not in df.columns:
            df[f] = all_defaults.get(f, 0)
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(all_defaults.get(f, 0))

    print(f"  v13.2: {len(F_V132)} features (baseline)")
    print(f"  v13.3: {len(F_V133)} features (+{len(V133_NEW_FEATURES)})")

    # Coverage check for new features
    print(f"\n  v13.3 new feature coverage:")
    for f in V133_NEW_FEATURES:
        default = V133_DEFAULTS.get(f, 0)
        rate = (df[f] != default).mean() * 100
        print(f"    {f}: {rate:.1f}% non-default")

    # ===== WF Backtest =====
    print(f"\n{'='*60}")
    print(f"  [8/8] Walk-Forward Backtest")
    print(f"{'='*60}")

    # v13.2 baseline (same features, but with expanded data)
    print("\n--- v13.2 baseline (expanded data) ---")
    wf132 = walk_forward_backtest(df, F_V132)
    auc132 = print_wf(wf132, f'v13.2 ({len(F_V132)} feats, expanded data)')

    # v13.3 with new feature
    print("\n--- v13.3 +stable_comment_score ---")
    wf133 = walk_forward_backtest(df, F_V133)
    auc133 = print_wf(wf133, f'v13.3 ({len(F_V133)} feats)')

    # Feature importance
    imp = feature_importance(wf133, F_V133)
    print(f"\n  Top 30 by importance:")
    for i, (f, v) in enumerate(imp[:30]):
        tag = " *NEW" if f in V133_NEW_FEATURES else (" *JRDB" if f.startswith('jrdb_') else "")
        print(f"  {i+1:2d}. {f:35s} {v:12.1f}{tag}")

    # stable_comment_score position
    for i, (f, v) in enumerate(imp):
        if f == 'stable_comment_score':
            print(f"\n  stable_comment_score: rank {i+1}/{len(F_V133)}, importance {v:.1f}")
            break

    # Individual contribution check (full WF)
    print(f"\n  Contribution check (stable_comment_score):")
    feats_wo = [f for f in F_V133 if f != 'stable_comment_score']
    wf_wo = walk_forward_backtest(df, feats_wo)
    auc_wo = np.mean([r['lgb_auc'] for r in wf_wo])
    contrib = auc133 - auc_wo
    print(f"    With: {auc133:.4f}, Without: {auc_wo:.4f}, Contribution: {contrib:+.5f}")

    # Determine best result
    # Compare: v13.2-expanded vs v13.3
    if auc133 > auc132:
        best_auc = auc133
        best_wf = wf133
        best_feats = F_V133
        best_label = 'v13.3'
    else:
        best_auc = auc132
        best_wf = wf132
        best_feats = F_V132
        best_label = 'v13.2-expanded'

    # Adoption check
    baseline_auc = 0.8468  # v13.2 original
    delta = best_auc - baseline_auc
    all_ok = all(r['lgb_auc'] > 0.78 for r in best_wf)
    no_overfit = all(r['gap'] < 0.05 for r in best_wf)
    adopted = best_auc > baseline_auc and all_ok and no_overfit

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  v13.2 original:   0.8468 (114 feats)")
    print(f"  v13.2 expanded:   {auc132:.4f} ({len(F_V132)} feats, +SI2020/TT)")
    print(f"  v13.3 (+comment): {auc133:.4f} ({len(F_V133)} feats)")
    print(f"  Best:             {best_auc:.4f} ({best_label})")
    print(f"  Delta vs v13.2:   {delta:+.4f}")
    print(f"  comment contrib:  {contrib:+.5f}")
    print(f"  AUC > 0.8468:     {'Y' if best_auc > baseline_auc else 'N'}")
    print(f"  All years > 0.78: {'Y' if all_ok else 'N'}")
    print(f"  No overfitting:   {'Y' if no_overfit else 'N'}")
    print(f"\n  VERDICT: {'ADOPTED as v13.3' if adopted else 'NOT ADOPTED'}")

    # If v13.2-expanded beats original but v13.3 doesn't add value,
    # still adopt v13.2-expanded as v13.3
    if not adopted and auc132 > baseline_auc:
        alt_ok = all(r['lgb_auc'] > 0.78 for r in wf132)
        alt_no_overfit = all(r['gap'] < 0.05 for r in wf132)
        if alt_ok and alt_no_overfit:
            print(f"  → v13.2-expanded ({auc132:.4f}) beats original baseline!")
            print(f"  → Adopting as v13.3 WITHOUT stable_comment_score")
            best_auc = auc132
            best_wf = wf132
            best_feats = F_V132
            best_label = 'v13.3 (data-expanded)'
            adopted = True
            delta = auc132 - baseline_auc

    # Save results
    result = {
        'v132_original_auc': 0.8468,
        'v132_expanded_auc': auc132,
        'v133_auc': auc133,
        'best_auc': best_auc,
        'best_label': best_label,
        'delta': delta,
        'adopted': adopted,
        'stable_comment_contribution': contrib,
        'new_features': V133_NEW_FEATURES,
        'best_features': best_feats,
        'yearly': [{'year': r['year'], 'lgb_auc': r['lgb_auc'], 'xgb_auc': r['xgb_auc'],
                     'train_auc': r['train_auc'], 'gap': r['gap']} for r in best_wf],
        'v132_expanded_yearly': [{'year': r['year'], 'lgb_auc': r['lgb_auc'], 'xgb_auc': r['xgb_auc'],
                                   'train_auc': r['train_auc'], 'gap': r['gap']} for r in wf132],
        'v133_yearly': [{'year': r['year'], 'lgb_auc': r['lgb_auc'], 'xgb_auc': r['xgb_auc'],
                          'train_auc': r['train_auc'], 'gap': r['gap']} for r in wf133],
        'timestamp': datetime.now().isoformat(),
    }
    rpath = os.path.join(DATA_DIR, 'v133_training_results.json')
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Saved: {rpath}")

    if adopted:
        print(f"\n{'='*60}")
        print(f"  TRAINING PRODUCTION MODEL (v13.3)")
        print(f"{'='*60}")
        _train_production(df, best_feats, sire_map, bms_map, best_wf, jrdb_sel, best_label)

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

    pkl = {
        'model': ml, 'features': pa_feats, 'version': 'v133_leakfree',
        'auc': al, 'wf_auc': wf_auc, 'leak_free': True, 'leak_pattern': 'A',
        'leak_removed': sorted(LEAK_FEATURES_A),
        'sire_map': sire_map, 'bms_map': bms_map, 'n_top_encode': N_TOP_SIRE,
        'trained_at': now, 'model_type': 'central',
        'xgb_model': mx, 'mlp_model': None, 'mlp_scaler': None,
        'ensemble_weights': {'lgb': wl, 'xgb': wx, 'mlp': 0},
        'course_map': dict(COURSE_MAP),
        'jrdb_features': jrdb_sel, 'v132_features': v132_new,
        'v133_features': v133_new, 'label': label,
    }
    ap = os.path.join(BASE_DIR, 'keiba_model_v133_central.pkl.gz')
    with gzip.open(ap, 'wb') as f:
        pickle.dump(pkl, f)
    print(f"  Pattern A: {ap}")

    pb_feats = features + JRDB_LIVE_FEATURES
    pkl_b = dict(pkl)
    pkl_b['features'] = pb_feats
    pkl_b['version'] = 'v133_live'
    pkl_b['leak_free'] = False
    pkl_b['leak_pattern'] = 'B'
    pkl_b['is_live'] = True
    bp = os.path.join(BASE_DIR, 'keiba_model_v133_central_live.pkl.gz')
    with gzip.open(bp, 'wb') as f:
        pickle.dump(pkl_b, f)
    print(f"  Pattern B: {bp}")
    print(f"\n  *** v13.3 saved! A={len(pa_feats)} feats, B={len(pb_feats)} feats ***")


if __name__ == '__main__':
    main()
