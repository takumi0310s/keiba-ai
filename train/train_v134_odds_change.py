#!/usr/bin/env python3
"""v13.4 オッズ時系列変化特徴量テスト

v13.3(115特徴量, WF AUC 0.84685) に対して:
  jrdb_oz.csv（基準オッズ＝前日19-20時発表）と確定オッズの差分から
  市場情報の変化を特徴量化。

新規特徴量:
  1. oz_tansho_base_log   - 基準単勝オッズ(log変換)
  2. oz_fukusho_base_log  - 基準複勝オッズ(log変換)
  3. oz_base_pop_rank     - 基準人気順位
  4. odds_change_rate     - 単勝オッズ変化率 (base→確定)
  5. pop_rank_change      - 人気順変化 (base→確定)
  6. odds_sharp_drop      - オッズ急落フラグ(20%以上下落)

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
from train_v133_extended import V133_NEW_FEATURES, V133_DEFAULTS, merge_stable_comment_combined


# =====================================================
# v13.4 オッズ変化特徴量
# =====================================================

V134_NEW_FEATURES = [
    'oz_tansho_base_log',   # 基準単勝オッズ(log)
    'oz_fukusho_base_log',  # 基準複勝オッズ(log)
    'oz_base_pop_rank',     # 基準人気順位
    'odds_change_rate',     # 単勝オッズ変化率 (base→確定)
    'pop_rank_change',      # 人気順変化 (base→確定)
    'odds_sharp_drop',      # オッズ急落フラグ(20%以上下落)
]

V134_DEFAULTS = {
    'oz_tansho_base_log': 2.3,    # log(10) ≈ 2.3 (平均的なオッズ)
    'oz_fukusho_base_log': 0.7,   # log(2) ≈ 0.7
    'oz_base_pop_rank': 8,        # 中央値
    'odds_change_rate': 0.0,      # 変化なし
    'pop_rank_change': 0,         # 変化なし
    'odds_sharp_drop': 0,         # 急落なし
}


def merge_oz_features(df):
    """JRDB OZ（基準オッズ）から特徴量を生成・マージ

    OZ: 前日19-20時に発表される基準オッズ（前日オッズ）
    確定オッズ: jra_races_full.csvのtansho_odds（レース確定後）

    特徴量:
    1. oz_tansho_base_log: 基準単勝オッズのlog
    2. oz_fukusho_base_log: 基準複勝オッズのlog
    3. oz_base_pop_rank: 基準オッズから算出した人気順
    4. odds_change_rate: (base - confirmed) / base (正=人気下落、負=人気上昇)
    5. pop_rank_change: base_rank - confirmed_rank (正=人気上昇)
    6. odds_sharp_drop: 確定オッズが基準の80%以下 → 1
    """
    oz_path = os.path.join(DATA_DIR, 'jrdb_oz.csv')
    if not os.path.exists(oz_path):
        print("    OZ: not found")
        for f in V134_NEW_FEATURES:
            df[f] = V134_DEFAULTS[f]
        return df

    oz = pd.read_csv(oz_path, encoding='utf-8-sig', dtype={'race_id': str})

    # Wide → Long: tansho_01..18 → (race_id, umaban, base_tansho)
    tansho_cols = [f'tansho_{i:02d}' for i in range(1, 19)]
    fukusho_cols = [f'fukusho_{i:02d}' for i in range(1, 19)]

    records = []
    for _, row in oz.iterrows():
        rid = str(row['race_id']).zfill(12)
        for i in range(1, 19):
            t_col = f'tansho_{i:02d}'
            f_col = f'fukusho_{i:02d}'
            t_val = pd.to_numeric(row.get(t_col), errors='coerce')
            f_val = pd.to_numeric(row.get(f_col), errors='coerce')
            if pd.notna(t_val) and t_val > 0:
                records.append({
                    '_nk_rid': rid,
                    '_uma_str': str(i),
                    'base_tansho': t_val,
                    'base_fukusho': f_val if pd.notna(f_val) and f_val > 0 else np.nan,
                })

    oz_long = pd.DataFrame(records)
    print(f"    OZ: {len(oz_long)} horse-level records from {len(oz)} races")

    # Compute base popularity rank (lower odds = higher popularity)
    oz_long['oz_base_pop_rank'] = oz_long.groupby('_nk_rid')['base_tansho'].rank(
        method='min', ascending=True
    ).astype(int)

    # Log transform
    oz_long['oz_tansho_base_log'] = np.log(oz_long['base_tansho'].clip(lower=1.0))
    oz_long['oz_fukusho_base_log'] = np.log(oz_long['base_fukusho'].clip(lower=1.0))
    oz_long['oz_fukusho_base_log'] = oz_long['oz_fukusho_base_log'].fillna(
        V134_DEFAULTS['oz_fukusho_base_log'])

    # Merge to df
    df = _build_nk_race_id(df)
    df['_uma_str'] = df['umaban'].astype(int).astype(str)

    oz_dedup = oz_long.drop_duplicates(subset=['_nk_rid', '_uma_str'], keep='last')
    merge_cols = ['_nk_rid', '_uma_str', 'oz_tansho_base_log', 'oz_fukusho_base_log',
                  'oz_base_pop_rank', 'base_tansho']
    merged = df.merge(oz_dedup[merge_cols], on=['_nk_rid', '_uma_str'], how='left',
                      suffixes=('', '_oz'))

    # Fill base features
    df['oz_tansho_base_log'] = merged['oz_tansho_base_log'].fillna(
        V134_DEFAULTS['oz_tansho_base_log']).values
    df['oz_fukusho_base_log'] = merged['oz_fukusho_base_log'].fillna(
        V134_DEFAULTS['oz_fukusho_base_log']).values
    df['oz_base_pop_rank'] = merged['oz_base_pop_rank'].fillna(
        V134_DEFAULTS['oz_base_pop_rank']).values

    # === Change features: base vs confirmed ===
    # confirmed odds from jra_races_full.csv
    confirmed_odds = pd.to_numeric(df['tansho_odds'], errors='coerce') if 'tansho_odds' in df.columns else None
    confirmed_pop = pd.to_numeric(df['popularity'], errors='coerce') if 'popularity' in df.columns else None
    base_tansho = merged['base_tansho']

    if confirmed_odds is not None and base_tansho is not None:
        # odds_change_rate: (base - confirmed) / base
        # positive = odds dropped (horse became more popular)
        # negative = odds rose (horse became less popular)
        valid = base_tansho.notna() & confirmed_odds.notna() & (base_tansho > 0)
        change_rate = pd.Series(0.0, index=df.index)
        change_rate[valid] = (base_tansho[valid].values - confirmed_odds[valid].values) / base_tansho[valid].values
        change_rate = change_rate.clip(-2.0, 2.0)  # clip extreme values
        df['odds_change_rate'] = change_rate.values
    else:
        df['odds_change_rate'] = 0.0

    if confirmed_pop is not None:
        # pop_rank_change: base_rank - confirmed_rank
        # positive = moved up in popularity
        base_rank = merged['oz_base_pop_rank']
        valid = base_rank.notna() & confirmed_pop.notna()
        rank_change = pd.Series(0, index=df.index)
        rank_change[valid] = (base_rank[valid].values - confirmed_pop[valid].values).astype(int)
        df['pop_rank_change'] = rank_change.values
    else:
        df['pop_rank_change'] = 0

    # odds_sharp_drop: confirmed <= base * 0.8 (20%+ drop)
    if confirmed_odds is not None and base_tansho is not None:
        valid = base_tansho.notna() & confirmed_odds.notna() & (base_tansho > 0)
        sharp_drop = pd.Series(0, index=df.index)
        sharp_drop[valid] = (confirmed_odds[valid].values <= base_tansho[valid].values * 0.8).astype(int)
        df['odds_sharp_drop'] = sharp_drop.values
    else:
        df['odds_sharp_drop'] = 0

    # Coverage stats
    matched = merged['base_tansho'].notna().sum()
    total = len(df)
    print(f"    Matched: {matched}/{total} ({matched/total*100:.1f}%)")

    if 'year' in df.columns:
        temp_match = merged['base_tansho'].notna()
        for y in sorted(df['year'].unique()):
            mask = df['year'] == y
            m = temp_match[mask].sum()
            t = mask.sum()
            print(f"      Year {int(y)+2000}: {m}/{t} ({m/t*100:.1f}%)")

    # Stats
    if matched > 0:
        valid_change = df.loc[merged['base_tansho'].notna(), 'odds_change_rate']
        print(f"    odds_change_rate: mean={valid_change.mean():.3f}, std={valid_change.std():.3f}")
        sharp_count = df['odds_sharp_drop'].sum()
        print(f"    odds_sharp_drop: {int(sharp_count)} ({sharp_count/total*100:.1f}%)")

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
    print("  v13.4 Odds Time-Series Change Features")
    print(f"  Baseline: v13.3 WF AUC 0.84685 (115 features)")
    print(f"  New: +{len(V134_NEW_FEATURES)} features from JRDB OZ")
    print("=" * 60)

    # Phase 1-6: identical to v13.3
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

    # Phase 8: v13.4 NEW - OZ features
    print("\n[8/9] v13.4 OZ odds change features...")
    df = merge_oz_features(df)

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

    # Coverage
    print(f"\n  v13.4 new feature coverage:")
    for f in V134_NEW_FEATURES:
        default = V134_DEFAULTS.get(f, 0)
        rate = (df[f] != default).mean() * 100
        print(f"    {f}: {rate:.1f}% non-default")

    # ===== WF Backtest =====
    print(f"\n{'='*60}")
    print(f"  [9/9] Walk-Forward Backtest")
    print(f"{'='*60}")

    # v13.3 baseline
    print("\n--- v13.3 baseline ---")
    wf133 = walk_forward_backtest(df, F_V133)
    auc133 = print_wf(wf133, f'v13.3 ({len(F_V133)} feats)')

    # v13.4 all new
    print("\n--- v13.4 +all OZ features ---")
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
    for i, (f, v) in enumerate(imp):
        if f in V134_NEW_FEATURES:
            print(f"    {f}: rank {i+1}/{len(F_V134)}, importance {v:.1f}")

    # Individual contributions (full WF)
    print(f"\n  Individual contributions (full WF 2020-2025):")
    contribs = {}
    for nf in V134_NEW_FEATURES:
        feats_wo = [f for f in F_V134 if f != nf]
        wf_wo = walk_forward_backtest(df, feats_wo)
        auc_wo = np.mean([r['lgb_auc'] for r in wf_wo])
        c = auc134 - auc_wo
        contribs[nf] = c
        sign = '+' if c > 0 else ''
        mark = 'Y' if c > -0.0005 else 'N'
        print(f"    {mark} {nf:30s}: {sign}{c:.5f}")

    # Select features (threshold: > -0.0005)
    selected = [f for f, c in contribs.items() if c > -0.0005]
    removed = [f for f, c in contribs.items() if c <= -0.0005]
    F_V134_SEL = F_V133 + selected

    if removed:
        print(f"\n  Removed {len(removed)}: {removed}")
        print(f"\n--- v13.4 selected ({len(F_V134_SEL)} feats) ---")
        wf134s = walk_forward_backtest(df, F_V134_SEL)
        auc134s = print_wf(wf134s, f'v13.4 selected')
    else:
        wf134s = wf134
        auc134s = auc134
        F_V134_SEL = F_V134

    # Determine best
    best_auc = max(auc134, auc134s)
    best_wf = wf134 if auc134 >= auc134s else wf134s
    best_feats = F_V134 if auc134 >= auc134s else F_V134_SEL

    baseline_auc = 0.84685
    delta = best_auc - baseline_auc
    all_ok = all(r['lgb_auc'] > 0.78 for r in best_wf)
    no_overfit = all(r['gap'] < 0.05 for r in best_wf)
    adopted = best_auc > baseline_auc and all_ok and no_overfit

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  v13.3:          {auc133:.6f} ({len(F_V133)} feats)")
    print(f"  v13.4 all:      {auc134:.6f} ({len(F_V134)} feats)")
    print(f"  v13.4 selected: {auc134s:.6f} ({len(F_V134_SEL)} feats)")
    print(f"  Delta vs v13.3: {delta:+.6f}")
    print(f"  AUC > 0.84685:    {'Y' if best_auc > baseline_auc else 'N'}")
    print(f"  All years > 0.78: {'Y' if all_ok else 'N'}")
    print(f"  No overfitting:   {'Y' if no_overfit else 'N'}")
    print(f"\n  VERDICT: {'ADOPTED as v13.4' if adopted else 'NOT ADOPTED'}")

    # Save results
    result = {
        'v133_auc': auc133,
        'v134_all_auc': auc134,
        'v134_sel_auc': auc134s,
        'best_auc': best_auc,
        'delta': delta,
        'adopted': adopted,
        'new_features': V134_NEW_FEATURES,
        'selected': selected,
        'removed': removed,
        'contributions': contribs,
        'best_features': best_feats,
        'yearly': [{'year': r['year'], 'lgb_auc': r['lgb_auc'], 'xgb_auc': r['xgb_auc'],
                     'train_auc': r['train_auc'], 'gap': r['gap']} for r in best_wf],
        'timestamp': datetime.now().isoformat(),
    }
    rpath = os.path.join(DATA_DIR, 'v134_training_results.json')
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Saved: {rpath}")

    if adopted:
        print(f"\n{'='*60}")
        print(f"  TRAINING PRODUCTION MODEL (v13.4)")
        print(f"{'='*60}")
        _train_production(df, best_feats, sire_map, bms_map, best_wf, jrdb_sel, selected)

    print(f"\n  Elapsed: {(time.time()-t0)/60:.1f} min")


def _train_production(df, features, sire_map, bms_map, wf, jrdb_sel, oz_sel):
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
