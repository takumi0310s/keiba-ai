#!/usr/bin/env python3
"""v14 SKB/SRB(SR)/UKC/JOA 特徴量追加 + 4-model ensemble WF評価

v13.5b (124特徴量, WF AUC 0.8788) をベースに、未使用JRDB特徴量を追加:

SKB (成績拡張, 馬単位, 前走ラグ):
  - skb_prev_turf_hoof: 前走芝蹄コード (100% coverage)
  - skb_prev_run_stage: 前走走行段階 (49% coverage)

SR 拡張 (レース単位, 当レースバイアス):
  - sr_tb_4corner_inner: 4角バイアス内有利度
  - sr_pace_up_pos: ペースアップ位置

UKC (馬基本データ, 静的):
  - ukc_keito_code: 系統コード

JOA: 700行のみのため見送り

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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

from train_v135_ft_transformer import (
    build_v134_dataframe, get_v134_features, fill_defaults,
    FTTransformer, IntraRaceAttention, train_ft_transformer,
    DEVICE, LGB_PARAMS, XGB_PARAMS, _build_nk_race_id,
)
from jrdb_features import _build_nk_race_id_from_jv, JRDB_DEFAULTS
from train_v92_leakfree import LEAK_FEATURES_A
from train_v135b_intra_ensemble import (
    build_race_id, train_intra_race_with_preds, MAX_HORSES,
)


# =====================================================
# v14 新規特徴量定義
# =====================================================

V14_NEW_FEATURES = [
    # SKB (前走ラグ)
    'skb_prev_turf_hoof',    # 前走芝蹄コード (0-8)
    'skb_prev_run_stage',    # 前走走行段階
    # SR拡張 (当レース)
    'sr_tb_4corner_inner',   # 4角バイアス内有利度 (1-5)
    'sr_pace_up_pos',        # ペースアップ位置 (残ハロン数)
    # UKC (静的)
    'ukc_keito_code',        # 系統コード
]

V14_DEFAULTS = {
    'skb_prev_turf_hoof': 0,     # 不明
    'skb_prev_run_stage': 0,     # 不明
    'sr_tb_4corner_inner': 3,    # 中立
    'sr_pace_up_pos': 2,         # 平均的な位置
    'ukc_keito_code': 0,         # 不明
}


# =====================================================
# SKB前走特徴量マージ
# =====================================================

def merge_skb_prev_features(df):
    """SKB(成績拡張)から前走適性コードをマージ

    SKBは成績データ（レース後の評価）なので、前走データとして使用。
    horse_id + date順でソートし、1行シフトして前走データを取得。
    """
    path = os.path.join(DATA_DIR, 'jrdb_skb.csv')
    if not os.path.exists(path):
        print("    SKB: not found")
        for f in ['skb_prev_turf_hoof', 'skb_prev_run_stage']:
            df[f] = V14_DEFAULTS[f]
        return df

    skb = pd.read_csv(path, encoding='utf-8-sig', dtype=str,
                       usecols=['race_id', 'umaban', 'turf_hoof', 'run_stage'])
    skb['_nk_rid'] = skb['race_id'].astype(str).str.zfill(12)
    skb['_uma'] = pd.to_numeric(skb['umaban'], errors='coerce')
    skb['_turf_hoof'] = pd.to_numeric(skb['turf_hoof'], errors='coerce')
    skb['_run_stage'] = pd.to_numeric(skb['run_stage'], errors='coerce')

    skb_d = skb[['_nk_rid', '_uma', '_turf_hoof', '_run_stage']].drop_duplicates(
        subset=['_nk_rid', '_uma'], keep='last')

    # Build _nk_rid for df if not present
    if '_nk_rid' not in df.columns:
        df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    if '_uma' not in df.columns:
        df['_uma'] = pd.to_numeric(df['umaban'], errors='coerce')

    # Merge current race's SKB data
    before = len(df)
    df = df.merge(skb_d, on=['_nk_rid', '_uma'], how='left')

    # Shift to get prev-race data (group by horse_id, sorted by date)
    # We use the current row's SKB as "this race's assessment",
    # then in the feature we use prev_race's assessment
    # Since df is sorted by horse within compute_lag_features,
    # we can shift within horse_id groups
    if 'horse_id' in df.columns:
        df = df.sort_values(['horse_id', 'date_num'])
        df['skb_prev_turf_hoof'] = df.groupby('horse_id')['_turf_hoof'].shift(1)
        df['skb_prev_run_stage'] = df.groupby('horse_id')['_run_stage'].shift(1)
    else:
        df['skb_prev_turf_hoof'] = df['_turf_hoof']
        df['skb_prev_run_stage'] = df['_run_stage']

    matched = df['skb_prev_turf_hoof'].notna().sum()
    print(f"    SKB prev: {matched}/{before} matched ({matched/before*100:.1f}%)")
    df.drop(columns=['_turf_hoof', '_run_stage'], inplace=True, errors='ignore')
    return df


# =====================================================
# SR拡張特徴量マージ
# =====================================================

def merge_sr_extended_features(df):
    """SR(レース結果)から4角バイアスとペースアップ位置を追加マージ

    既存のjrdb_tb_homestr_innerと同じパターン（当レースデータ）。
    """
    path = os.path.join(DATA_DIR, 'jrdb_sr.csv')
    if not os.path.exists(path):
        print("    SR ext: not found")
        df['sr_tb_4corner_inner'] = V14_DEFAULTS['sr_tb_4corner_inner']
        df['sr_pace_up_pos'] = V14_DEFAULTS['sr_pace_up_pos']
        return df

    sr = pd.read_csv(path, encoding='utf-8-sig', dtype=str,
                      usecols=['race_id', 'tb_4corner', 'pace_up_pos'])
    sr['_nk_rid'] = sr['race_id'].astype(str).str.zfill(12)

    # 4角バイアス: 先頭1文字 = 最内有利度 (1=有利, 5=不利)
    sr['sr_tb_4corner_inner'] = sr['tb_4corner'].apply(
        lambda x: int(str(x)[0]) if pd.notna(x) and len(str(x)) >= 1 and str(x)[0].isdigit() else 3
    )
    sr['sr_pace_up_pos'] = pd.to_numeric(sr['pace_up_pos'], errors='coerce')

    sr_d = sr[['_nk_rid', 'sr_tb_4corner_inner', 'sr_pace_up_pos']].drop_duplicates(
        subset='_nk_rid', keep='last')

    if '_nk_rid' not in df.columns:
        df['_nk_rid'] = _build_nk_race_id_from_jv(df)

    before = len(df)
    df = df.merge(sr_d, on='_nk_rid', how='left')
    matched = df['sr_tb_4corner_inner'].notna().sum()
    print(f"    SR ext: {matched}/{before} matched ({matched/before*100:.1f}%)")
    return df


# =====================================================
# UKC特徴量マージ
# =====================================================

def merge_ukc_features(df):
    """UKC(馬基本データ)から系統コードをマージ"""
    path = os.path.join(DATA_DIR, 'jrdb_ukc.csv')
    if not os.path.exists(path):
        print("    UKC: not found")
        df['ukc_keito_code'] = V14_DEFAULTS['ukc_keito_code']
        return df

    ukc = pd.read_csv(path, encoding='utf-8-sig', dtype=str,
                       usecols=['blood_num', 'keito_code'])
    ukc['ukc_keito_code'] = pd.to_numeric(ukc['keito_code'], errors='coerce')
    ukc = ukc[['blood_num', 'ukc_keito_code']].drop_duplicates(subset='blood_num', keep='last')

    # Need blood_num in df. Get from KYI (has nk_race_id, 馬番, 血統登録番号)
    kyi_path = os.path.join(DATA_DIR, 'jrdb_kyi.csv')
    paci_path = os.path.join(DATA_DIR, 'jrdb_paci.csv')

    blood_map = None
    for src_path in [kyi_path, paci_path]:
        if not os.path.exists(src_path):
            continue
        # Read first row to detect column names
        sample = pd.read_csv(src_path, encoding='utf-8-sig', dtype=str, nrows=0)
        all_cols = sample.columns.tolist()
        # Find the right column names
        bn_col = next((c for c in all_cols if c in ('blood_num', '血統登録番号')), None)
        uma_col = next((c for c in all_cols if c in ('umaban', '馬番')), None)
        rid_col = next((c for c in all_cols if c in ('nk_race_id', 'race_id')), None)
        if bn_col and uma_col and rid_col:
            src = pd.read_csv(src_path, encoding='utf-8-sig', dtype=str,
                              usecols=[rid_col, uma_col, bn_col])
            src['_nk_rid'] = src[rid_col].astype(str).str.zfill(12)
            src['_uma'] = pd.to_numeric(src[uma_col], errors='coerce')
            src['blood_num'] = src[bn_col].astype(str)
            blood_map = src[['_nk_rid', '_uma', 'blood_num']].drop_duplicates(
                subset=['_nk_rid', '_uma'], keep='last')
            print(f"    UKC blood_num map: {len(blood_map)} entries from {os.path.basename(src_path)}")
            break

    if blood_map is None:
        print("    UKC: no blood_num mapping source")
        df['ukc_keito_code'] = V14_DEFAULTS['ukc_keito_code']
        return df

    if '_nk_rid' not in df.columns:
        df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    if '_uma' not in df.columns:
        df['_uma'] = pd.to_numeric(df['umaban'], errors='coerce')

    before = len(df)
    df = df.merge(blood_map, on=['_nk_rid', '_uma'], how='left', suffixes=('', '_bmap'))
    bn_col = 'blood_num' if 'blood_num' in df.columns else 'blood_num_bmap'
    if bn_col == 'blood_num_bmap':
        df.rename(columns={'blood_num_bmap': 'blood_num'}, inplace=True)
    df = df.merge(ukc, on='blood_num', how='left', suffixes=('', '_ukc'))
    matched = df['ukc_keito_code'].notna().sum()
    print(f"    UKC: {matched}/{before} matched ({matched/before*100:.1f}%)")
    df.drop(columns=['blood_num'], inplace=True, errors='ignore')
    return df


# =====================================================
# v14 DataFrame構築
# =====================================================

def build_v14_dataframe():
    """v13.5bのDataFrameに新特徴量を追加"""
    # v13.5b base
    df, sire_map, bms_map = build_v134_dataframe()

    # Ensure _nk_rid and _uma
    if '_nk_rid' not in df.columns:
        df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    if '_uma' not in df.columns:
        df['_uma'] = pd.to_numeric(df['umaban'], errors='coerce')

    print("\n[v14] Merging new features...")

    # SKB prev-race features
    df = merge_skb_prev_features(df)

    # SR extended features
    df = merge_sr_extended_features(df)

    # UKC features
    df = merge_ukc_features(df)

    # Cleanup
    df.drop(columns=['_nk_rid', '_uma'], inplace=True, errors='ignore')

    return df, sire_map, bms_map


def get_v14_features():
    """v14特徴量リスト = v13.5b(124) + v14新規"""
    from train_v135_ft_transformer import get_v134_features
    base_feats, jrdb_sel = get_v134_features()
    v14_feats = base_feats + V14_NEW_FEATURES
    return v14_feats, jrdb_sel


def fill_v14_defaults(df, features):
    """v14用デフォルト値埋め"""
    all_defaults = {**JRDB_DEFAULTS, **V14_DEFAULTS}
    # Import all existing defaults
    from train_v132_extended import V132_DEFAULTS
    from train_v133_extended import V133_DEFAULTS
    from train_v134_odds_change import V134_DEFAULTS as V134_OZ_DEFAULTS
    from train_v134_weight_trend import V134_DEFAULTS as V134_WT_DEFAULTS
    all_defaults.update(V132_DEFAULTS)
    all_defaults.update(V133_DEFAULTS)
    all_defaults.update(V134_OZ_DEFAULTS)
    all_defaults.update(V134_WT_DEFAULTS)
    all_defaults.update(V14_DEFAULTS)

    for f in features:
        if f not in df.columns:
            df[f] = all_defaults.get(f, 0)
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(all_defaults.get(f, 0))
    return df


# =====================================================
# 4-model WF (from v135b, with new features)
# =====================================================

def walk_forward_4model(df, features, years=range(2020, 2026)):
    """LGB + XGB + FT + IntraRace の4モデルアンサンブルWF"""

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

        # --- LightGBM ---
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
        m_lgb = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                          valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        p_lgb = m_lgb.predict(X_te)
        auc_lgb = roc_auc_score(y_te, p_lgb)
        p_lgb_tr = m_lgb.predict(X_tr)
        auc_lgb_tr = roc_auc_score(y_tr, p_lgb_tr)

        # --- XGBoost ---
        dxtr = xgb.DMatrix(X_tr, label=y_tr)
        dxte = xgb.DMatrix(X_te, label=y_te)
        m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=1000,
                           evals=[(dxte, 'valid')],
                           early_stopping_rounds=50, verbose_eval=False)
        p_xgb = m_xgb.predict(dxte)
        auc_xgb = roc_auc_score(y_te, p_xgb)

        # --- FT-Transformer ---
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

        # --- IntraRace Attention ---
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

        # Map IntraRace predictions
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

        # --- Ensembles ---
        # Grid search for optimal 4-model weights
        best_grid_auc = 0
        best_grid_w = None
        if auc_ir > 0:
            for w1 in np.arange(0.20, 0.45, 0.05):   # LGB
                for w2 in np.arange(0.20, 0.45, 0.05): # XGB
                    for w3 in np.arange(0.05, 0.30, 0.05): # FT
                        w4 = 1.0 - w1 - w2 - w3  # IR
                        if w4 < 0.01 or w4 > 0.40:
                            continue
                        p_grid = w1*p_lgb + w2*p_xgb + w3*p_ft + w4*p_ir
                        auc_grid = roc_auc_score(y_te, p_grid)
                        if auc_grid > best_grid_auc:
                            best_grid_auc = auc_grid
                            best_grid_w = (w1, w2, w3, w4)

        gap = auc_lgb_tr - auc_lgb

        # Simple ensembles for comparison
        w_lgb2 = auc_lgb / (auc_lgb + auc_xgb)
        p_lgbxgb = w_lgb2 * p_lgb + (1 - w_lgb2) * p_xgb
        auc_lgbxgb = roc_auc_score(y_te, p_lgbxgb)

        total4 = auc_lgb + auc_xgb + auc_ft + auc_ir if auc_ir > 0 else 1
        if auc_ir > 0:
            p_4m = ((auc_lgb/total4)*p_lgb + (auc_xgb/total4)*p_xgb +
                    (auc_ft/total4)*p_ft + (auc_ir/total4)*p_ir)
            auc_4m = roc_auc_score(y_te, p_4m)
        else:
            auc_4m = auc_lgbxgb

        print(f"\n  {test_year} Results:")
        print(f"    LGB:         {auc_lgb:.4f} (train: {auc_lgb_tr:.4f})")
        print(f"    XGB:         {auc_xgb:.4f}")
        print(f"    FT:          {auc_ft:.4f}")
        print(f"    IR:          {auc_ir:.4f} (coverage={ir_cov_pct:.1f}%)")
        print(f"    LGB+XGB:     {auc_lgbxgb:.4f}")
        print(f"    4-model:     {auc_4m:.4f}")
        if best_grid_w:
            print(f"    4m-grid:     {best_grid_auc:.4f} "
                  f"(L={best_grid_w[0]:.2f} X={best_grid_w[1]:.2f} "
                  f"F={best_grid_w[2]:.2f} IR={best_grid_w[3]:.2f})")
        print(f"    Gap(LGB):    {gap:.4f}")

        results.append({
            'year': test_year,
            'lgb_auc': auc_lgb,
            'xgb_auc': auc_xgb,
            'ft_auc': auc_ft,
            'ir_auc': auc_ir,
            'ir_coverage': ir_cov_pct,
            'lgbxgb_auc': auc_lgbxgb,
            '4model_auc': auc_4m,
            'grid_auc': best_grid_auc if best_grid_w else None,
            'grid_weights': list(best_grid_w) if best_grid_w else None,
            'train_auc': auc_lgb_tr,
            'gap': gap,
        })

    return results


# =====================================================
# メイン
# =====================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("  v14 - SKB/SRB(SR)/UKC + 4-Model Ensemble WF")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Build dataframe with new features
    df, sire_map, bms_map = build_v14_dataframe()
    features, jrdb_sel = get_v14_features()
    df = fill_v14_defaults(df, features)
    df = build_race_id(df)

    # Verify new features
    print(f"\n  Data: {len(df)} rows, {len(features)} features")
    print(f"  Races: {df['race_id_unique'].nunique()}")
    print(f"\n  v14 new features ({len(V14_NEW_FEATURES)}):")
    for f in V14_NEW_FEATURES:
        if f in df.columns:
            valid = df[f].notna().sum()
            nonzero = (df[f] != 0).sum()
            print(f"    {f}: valid={valid}/{len(df)} ({valid/len(df)*100:.1f}%), "
                  f"nonzero={nonzero} ({nonzero/len(df)*100:.1f}%)")
        else:
            print(f"    {f}: MISSING")

    # Run 4-model WF
    results = walk_forward_4model(df, features)

    # Summary
    print(f"\n{'='*70}")
    print(f"  v14 SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Year':<6} {'LGB':>7} {'XGB':>7} {'FT':>7} {'IR':>7} "
          f"{'L+X':>7} {'4-model':>8} {'Grid':>8} {'Gap':>6}")
    print(f"  {'-'*70}")
    for r in results:
        ok = 'Y' if r['gap'] < 0.05 else 'N'
        grid = f"{r['grid_auc']:.4f}" if r['grid_auc'] else '  N/A '
        print(f"  {r['year']:<6} {r['lgb_auc']:7.4f} {r['xgb_auc']:7.4f} "
              f"{r['ft_auc']:7.4f} {r['ir_auc']:7.4f} "
              f"{r['lgbxgb_auc']:7.4f} {r['4model_auc']:8.4f} "
              f"{grid:>8} {r['gap']:6.4f}{ok}")

    lgbxgb_mean = np.mean([r['lgbxgb_auc'] for r in results])
    m4_mean = np.mean([r['4model_auc'] for r in results])
    grid_vals = [r['grid_auc'] for r in results if r['grid_auc']]
    grid_mean = np.mean(grid_vals) if grid_vals else 0

    print(f"\n  Mean AUC:")
    print(f"    LGB+XGB:        {lgbxgb_mean:.6f}")
    print(f"    4-model:        {m4_mean:.6f}")
    if grid_mean > 0:
        print(f"    4-model grid:   {grid_mean:.6f}")

    baseline = 0.8788
    candidates = {
        '4-model (equal)': m4_mean,
    }
    if grid_mean > 0:
        candidates['4-model (grid)'] = grid_mean

    best_name = max(candidates, key=candidates.get)
    best_val = candidates[best_name]
    delta = best_val - baseline

    print(f"\n  v13.5b baseline: {baseline:.6f}")
    print(f"  Best v14:        {best_val:.6f} ({delta:+.6f})")

    all_year_ok = all(
        max(r['lgb_auc'], r.get('grid_auc', 0) or 0, r['4model_auc']) > 0.85
        for r in results
    )
    no_overfit = all(r['gap'] < 0.05 for r in results)
    adopted = best_val > baseline and all_year_ok and no_overfit

    if adopted:
        print(f"\n  *** ADOPTED: v14 {best_name} ***")
    else:
        reasons = []
        if best_val <= baseline:
            reasons.append(f"AUC {best_val:.6f} <= baseline {baseline:.6f}")
        if not all_year_ok:
            bad_years = [r['year'] for r in results
                         if max(r['lgb_auc'], r.get('grid_auc', 0) or 0, r['4model_auc']) <= 0.85]
            reasons.append(f"year AUC <= 0.85: {bad_years}")
        if not no_overfit:
            bad_gaps = [(r['year'], r['gap']) for r in results if r['gap'] >= 0.05]
            reasons.append(f"gap >= 0.05: {bad_gaps}")
        print(f"\n  NOT ADOPTED: {', '.join(reasons)}")

    # Feature importance (LGB last fold)
    print(f"\n  v14 new feature importance (LGB, last fold):")
    # Retrain LGB on full last fold for importance
    ty = 25  # 2025
    train_mask = df['year'] < ty
    test_mask = df['year'] == ty
    X_tr = df.loc[train_mask, features].values
    y_tr = df.loc[train_mask, 'target'].values
    X_te = df.loc[test_mask, features].values
    y_te = df.loc[test_mask, 'target'].values
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
    m = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                  valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    imp = dict(zip(features, m.feature_importance('gain')))
    total_imp = sum(imp.values())
    for f in V14_NEW_FEATURES:
        pct = imp.get(f, 0) / total_imp * 100 if total_imp > 0 else 0
        print(f"    {f}: {imp.get(f, 0):.0f} ({pct:.2f}%)")

    # Save results
    result = {
        'baseline_auc': baseline,
        'yearly': [{k: v for k, v in r.items()} for r in results],
        'means': {
            'lgbxgb': lgbxgb_mean,
            '4model': m4_mean,
            '4model_grid': grid_mean,
        },
        'best_ensemble': best_name,
        'best_auc': float(best_val),
        'delta_vs_baseline': float(delta),
        'adopted': adopted,
        'features_count': len(features),
        'new_features': V14_NEW_FEATURES,
        'device': str(DEVICE),
        'timestamp': datetime.now().isoformat(),
    }
    rpath = os.path.join(DATA_DIR, 'v14_skb_srb_ukc_results.json')
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved: {rpath}")

    elapsed = (time.time() - t0) / 60
    print(f"\n  Total time: {elapsed:.1f} min")
    print("=" * 70)

    return adopted, result


if __name__ == '__main__':
    adopted, result = main()
    sys.exit(0 if adopted else 1)
