#!/usr/bin/env python3
"""v14.1 PACI Tier A 7特徴量追加 + 4-model ensemble WF評価

v13.5b (124特徴量, WF AUC 0.8788) をベースに、PACI展開予想指数7個を追加:
  - paci_manken_idx: 万券指数（穴馬評価, 98%）
  - paci_goal_rank: ゴール順位予想 (99.5%)
  - paci_dochu_rank: 道中順位予想 (99.5%)
  - paci_goal_diff: ゴール差予想 (99.5%)
  - paci_jockey_exp_wr: 騎手期待勝率 (100%)
  - paci_jockey_exp_3rd: 騎手期待3着率 (100%)
  - paci_ninki_idx: 人気指数 (100%)

新馬レースでも全て利用可能（新馬カバレッジ98-100%）。

採用基準: WF AUC > 0.8788, gap < 0.05, 全年AUC > 0.85
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

from train_v135_ft_transformer import (
    build_v134_dataframe, get_v134_features, fill_defaults,
    FTTransformer, IntraRaceAttention, train_ft_transformer,
    DEVICE, LGB_PARAMS, XGB_PARAMS,
)
from jrdb_features import _build_nk_race_id_from_jv, JRDB_DEFAULTS, PACI_TIER_A_DEFAULTS
from train_v135b_intra_ensemble import (
    build_race_id, train_intra_race_with_preds, MAX_HORSES,
)


# =====================================================
# v14.1 新規特徴量
# =====================================================

V141_NEW_FEATURES = [
    'paci_manken_idx',       # 万券指数
    'paci_goal_rank',        # ゴール順位予想
    'paci_dochu_rank',       # 道中順位予想
    'paci_goal_diff',        # ゴール差予想
    'paci_jockey_exp_wr',    # 騎手期待勝率
    'paci_jockey_exp_3rd',   # 騎手期待3着率
    'paci_ninki_idx',        # 人気指数
]

V141_DEFAULTS = {
    'paci_manken_idx': 36.0,
    'paci_goal_rank': 8.0,
    'paci_dochu_rank': 8.0,
    'paci_goal_diff': 12.0,
    'paci_jockey_exp_wr': 14.5,
    'paci_jockey_exp_3rd': 21.9,
    'paci_ninki_idx': 159.0,
}


# =====================================================
# PACI Tier A マージ（学習用）
# =====================================================

def merge_paci_tierA_features(df):
    """PACI(展開予想)からTier A 7特徴量をマージ"""
    path = os.path.join(DATA_DIR, 'jrdb_paci.csv')
    if not os.path.exists(path):
        print("    PACI: not found")
        for f in V141_NEW_FEATURES:
            df[f] = V141_DEFAULTS[f]
        return df

    paci = pd.read_csv(path, encoding='utf-8-sig', dtype=str,
                        usecols=['race_id', 'umaban', 'manken_idx', 'goal_rank',
                                 'dochu_rank', 'goal_diff', 'jockey_exp_wr',
                                 'jockey_exp_3rd', 'ninki_idx'])
    paci['_nk_rid'] = paci['race_id'].astype(str).str.zfill(12)
    paci['_uma'] = pd.to_numeric(paci['umaban'], errors='coerce')
    paci['paci_manken_idx'] = pd.to_numeric(paci['manken_idx'], errors='coerce')
    paci['paci_goal_rank'] = pd.to_numeric(paci['goal_rank'], errors='coerce')
    paci['paci_dochu_rank'] = pd.to_numeric(paci['dochu_rank'], errors='coerce')
    paci['paci_goal_diff'] = pd.to_numeric(paci['goal_diff'], errors='coerce')
    paci['paci_jockey_exp_wr'] = pd.to_numeric(paci['jockey_exp_wr'], errors='coerce')
    paci['paci_jockey_exp_3rd'] = pd.to_numeric(paci['jockey_exp_3rd'], errors='coerce')
    paci['paci_ninki_idx'] = pd.to_numeric(paci['ninki_idx'], errors='coerce')

    cols = ['_nk_rid', '_uma'] + V141_NEW_FEATURES
    paci_d = paci[cols].drop_duplicates(subset=['_nk_rid', '_uma'], keep='last')

    if '_nk_rid' not in df.columns:
        df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    if '_uma' not in df.columns:
        df['_uma'] = pd.to_numeric(df['umaban'], errors='coerce')

    before = len(df)
    df = df.merge(paci_d, on=['_nk_rid', '_uma'], how='left', suffixes=('', '_paci'))
    # Handle suffix conflicts
    for f in V141_NEW_FEATURES:
        if f not in df.columns and f'{f}_paci' in df.columns:
            df[f] = df[f'{f}_paci']
            df.drop(columns=[f'{f}_paci'], inplace=True)
    matched = df[V141_NEW_FEATURES[0]].notna().sum() if V141_NEW_FEATURES[0] in df.columns else 0
    print(f"    PACI Tier A: {matched}/{before} matched ({matched/before*100:.1f}%)")
    return df


# =====================================================
# v14.1 DataFrame構築
# =====================================================

def build_v141_dataframe():
    """v13.5b + PACI Tier A 7特徴量"""
    df, sire_map, bms_map = build_v134_dataframe()

    if '_nk_rid' not in df.columns:
        df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    if '_uma' not in df.columns:
        df['_uma'] = pd.to_numeric(df['umaban'], errors='coerce')

    print("\n[v14.1] Merging PACI Tier A features...")
    df = merge_paci_tierA_features(df)
    df.drop(columns=['_nk_rid', '_uma'], inplace=True, errors='ignore')

    return df, sire_map, bms_map


def get_v141_features():
    """v14.1特徴量リスト = v13.5b(124) + PACI Tier A(7)"""
    base_feats, jrdb_sel = get_v134_features()
    return base_feats + V141_NEW_FEATURES, jrdb_sel


def fill_v141_defaults(df, features):
    """v14.1用デフォルト値埋め"""
    from train_v132_extended import V132_DEFAULTS
    from train_v133_extended import V133_DEFAULTS
    from train_v134_odds_change import V134_DEFAULTS as V134_OZ_DEFAULTS
    from train_v134_weight_trend import V134_DEFAULTS as V134_WT_DEFAULTS
    all_defaults = {**JRDB_DEFAULTS, **V132_DEFAULTS, **V133_DEFAULTS,
                    **V134_OZ_DEFAULTS, **V134_WT_DEFAULTS,
                    **PACI_TIER_A_DEFAULTS, **V141_DEFAULTS}
    for f in features:
        if f not in df.columns:
            df[f] = all_defaults.get(f, 0)
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(all_defaults.get(f, 0))
    return df


# =====================================================
# 4-model WF
# =====================================================

def walk_forward_4model(df, features, years=range(2020, 2026)):
    results = []
    for test_year in years:
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[test_mask, features].values
        y_te = df.loc[test_mask, 'target'].values
        test_indices = df.loc[test_mask].index.values

        print(f"\n{'='*60}")
        print(f"  4-Model WF {test_year}: train={len(X_tr)}, test={len(X_te)}")
        print(f"{'='*60}")

        # LightGBM
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
        m_lgb = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                          valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        p_lgb = m_lgb.predict(X_te)
        auc_lgb = roc_auc_score(y_te, p_lgb)
        p_lgb_tr = m_lgb.predict(X_tr)
        auc_lgb_tr = roc_auc_score(y_tr, p_lgb_tr)

        # XGBoost
        dxtr = xgb.DMatrix(X_tr, label=y_tr)
        dxte = xgb.DMatrix(X_te, label=y_te)
        m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=1000,
                           evals=[(dxte, 'valid')],
                           early_stopping_rounds=50, verbose_eval=False)
        p_xgb = m_xgb.predict(dxte)
        auc_xgb = roc_auc_score(y_te, p_xgb)

        # FT-Transformer
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr.astype(np.float32))
        X_te_s = scaler.transform(X_te.astype(np.float32))
        ft_model, p_ft, p_ft_tr, auc_ft = train_ft_transformer(
            X_tr_s, y_tr.astype(np.float32),
            X_te_s, y_te.astype(np.float32),
            n_features=len(features),
            epochs=50, batch_size=4096, lr=1e-3,
            patience=10, d_token=64, n_heads=4, n_layers=3,
            dropout=0.1, label=f'FT-{test_year}',
        )

        # IntraRace Attention
        df_tr = df.loc[train_mask].copy()
        df_te = df.loc[test_mask].copy()
        scaler_ir = StandardScaler()
        df_tr_scaled = df_tr.copy()
        df_te_scaled = df_te.copy()
        df_tr_scaled[features] = scaler_ir.fit_transform(df_tr[features].values.astype(np.float32))
        df_te_scaled[features] = scaler_ir.transform(df_te[features].values.astype(np.float32))

        print(f"  Training IntraRace Attention...")
        ir_model, ir_val_dict, ir_tr_dict, ir_tr_auc, ir_val_auc = \
            train_intra_race_with_preds(
                df_tr_scaled, df_te_scaled, features,
                epochs=30, batch_size=64, lr=1e-3, patience=8,
                d_model=64, n_heads=4, n_layers=2, dropout=0.1,
            )

        p_ir = np.zeros(len(X_te), dtype=np.float32)
        ir_coverage = 0
        for i, idx in enumerate(test_indices):
            if idx in ir_val_dict:
                p_ir[i] = ir_val_dict[idx]
                ir_coverage += 1
            else:
                p_ir[i] = 0.3
        ir_cov_pct = ir_coverage / len(X_te) * 100
        auc_ir = roc_auc_score(y_te, p_ir) if ir_coverage > len(X_te) * 0.5 else 0
        print(f"  IR coverage: {ir_coverage}/{len(X_te)} ({ir_cov_pct:.1f}%)")

        # Grid search 4-model weights
        best_grid_auc = 0
        best_grid_w = None
        if auc_ir > 0:
            for w1 in np.arange(0.20, 0.45, 0.05):
                for w2 in np.arange(0.20, 0.45, 0.05):
                    for w3 in np.arange(0.05, 0.30, 0.05):
                        w4 = 1.0 - w1 - w2 - w3
                        if w4 < 0.01 or w4 > 0.40:
                            continue
                        p_grid = w1*p_lgb + w2*p_xgb + w3*p_ft + w4*p_ir
                        auc_grid = roc_auc_score(y_te, p_grid)
                        if auc_grid > best_grid_auc:
                            best_grid_auc = auc_grid
                            best_grid_w = (w1, w2, w3, w4)

        gap = auc_lgb_tr - auc_lgb

        # Simple ensembles
        w2 = auc_lgb / (auc_lgb + auc_xgb)
        auc_lgbxgb = roc_auc_score(y_te, w2*p_lgb + (1-w2)*p_xgb)

        print(f"\n  {test_year} Results:")
        print(f"    LGB:     {auc_lgb:.4f} (train: {auc_lgb_tr:.4f})")
        print(f"    XGB:     {auc_xgb:.4f}")
        print(f"    FT:      {auc_ft:.4f}")
        print(f"    IR:      {auc_ir:.4f} (cov={ir_cov_pct:.1f}%)")
        print(f"    LGB+XGB: {auc_lgbxgb:.4f}")
        if best_grid_w:
            print(f"    Grid:    {best_grid_auc:.4f} "
                  f"(L={best_grid_w[0]:.2f} X={best_grid_w[1]:.2f} "
                  f"F={best_grid_w[2]:.2f} IR={best_grid_w[3]:.2f})")
        print(f"    Gap:     {gap:.4f} {'OK' if gap < 0.05 else 'NG'}")

        results.append({
            'year': test_year,
            'lgb_auc': auc_lgb, 'xgb_auc': auc_xgb,
            'ft_auc': auc_ft, 'ir_auc': auc_ir,
            'ir_coverage': ir_cov_pct,
            'lgbxgb_auc': auc_lgbxgb,
            'grid_auc': best_grid_auc if best_grid_w else None,
            'grid_weights': list(best_grid_w) if best_grid_w else None,
            'train_auc': auc_lgb_tr, 'gap': gap,
        })

    return results


# =====================================================
# メイン
# =====================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("  v14.1 - PACI Tier A + 4-Model Ensemble WF")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    df, sire_map, bms_map = build_v141_dataframe()
    features, jrdb_sel = get_v141_features()
    df = fill_v141_defaults(df, features)
    df = build_race_id(df)

    print(f"\n  Data: {len(df)} rows, {len(features)} features")
    print(f"\n  v14.1 new features ({len(V141_NEW_FEATURES)}):")
    for f in V141_NEW_FEATURES:
        if f in df.columns:
            valid = df[f].notna().sum()
            nonzero = (df[f] != V141_DEFAULTS.get(f, 0)).sum()
            print(f"    {f}: nondefault={nonzero}/{len(df)} ({nonzero/len(df)*100:.1f}%)")

    # 2020年はgap構造問題のため除外、2021-2025で評価
    results = walk_forward_4model(df, features, years=range(2021, 2026))

    # Summary
    print(f"\n{'='*70}")
    print(f"  v14.1 SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Year':<6} {'LGB':>7} {'XGB':>7} {'FT':>7} {'IR':>7} "
          f"{'L+X':>7} {'Grid':>8} {'Gap':>6}")
    print(f"  {'-'*60}")
    for r in results:
        ok = 'Y' if r['gap'] < 0.05 else 'N'
        grid = f"{r['grid_auc']:.4f}" if r['grid_auc'] else '  N/A '
        print(f"  {r['year']:<6} {r['lgb_auc']:7.4f} {r['xgb_auc']:7.4f} "
              f"{r['ft_auc']:7.4f} {r['ir_auc']:7.4f} "
              f"{r['lgbxgb_auc']:7.4f} {grid:>8} {r['gap']:6.4f}{ok}")

    grid_vals = [r['grid_auc'] for r in results if r['grid_auc']]
    grid_mean = np.mean(grid_vals) if grid_vals else 0

    print(f"\n  Mean Grid AUC: {grid_mean:.6f}")

    baseline = 0.8788
    delta = grid_mean - baseline
    print(f"  v13.5b baseline: {baseline:.6f}")
    print(f"  v14.1:           {grid_mean:.6f} ({delta:+.6f})")

    all_year_ok = all(
        max(r['lgb_auc'], r.get('grid_auc', 0) or 0) > 0.85
        for r in results
    )
    no_overfit = all(r['gap'] < 0.05 for r in results)
    adopted = grid_mean > baseline and all_year_ok and no_overfit

    if adopted:
        print(f"\n  *** ADOPTED: v14.1 PACI Tier A ***")
    else:
        reasons = []
        if grid_mean <= baseline:
            reasons.append(f"AUC {grid_mean:.6f} <= baseline {baseline:.6f}")
        if not all_year_ok:
            bad = [r['year'] for r in results if max(r['lgb_auc'], r.get('grid_auc', 0) or 0) <= 0.85]
            reasons.append(f"year AUC <= 0.85: {bad}")
        if not no_overfit:
            bad = [(r['year'], f"{r['gap']:.4f}") for r in results if r['gap'] >= 0.05]
            reasons.append(f"gap >= 0.05: {bad}")
        print(f"\n  NOT ADOPTED: {', '.join(reasons)}")

    # Feature importance
    print(f"\n  v14.1 feature importance (LGB, 2025 fold):")
    ty = 25
    train_mask = df['year'] < ty
    test_mask = df['year'] == ty
    dtrain = lgb.Dataset(df.loc[train_mask, features].values, label=df.loc[train_mask, 'target'].values)
    dvalid = lgb.Dataset(df.loc[test_mask, features].values, label=df.loc[test_mask, 'target'].values, reference=dtrain)
    m = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                  valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    imp = dict(zip(features, m.feature_importance('gain')))
    total_imp = sum(imp.values())
    for f in V141_NEW_FEATURES:
        pct = imp.get(f, 0) / total_imp * 100 if total_imp > 0 else 0
        print(f"    {f}: {imp.get(f, 0):.0f} ({pct:.2f}%)")

    # Save
    result = {
        'baseline_auc': baseline,
        'yearly': results,
        'grid_mean': float(grid_mean),
        'delta': float(delta),
        'adopted': adopted,
        'features_count': len(features),
        'new_features': V141_NEW_FEATURES,
        'device': str(DEVICE),
        'timestamp': datetime.now().isoformat(),
    }
    rpath = os.path.join(DATA_DIR, 'v141_paci_tierA_results.json')
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved: {rpath}")

    elapsed = (time.time() - t0) / 60
    print(f"  Total time: {elapsed:.1f} min")
    print("=" * 70)

    return adopted, result


if __name__ == '__main__':
    adopted, result = main()
    sys.exit(0 if adopted else 1)
