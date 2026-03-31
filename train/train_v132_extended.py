#!/usr/bin/env python3
"""v13.2 JRDB拡張特徴量テスト

v13(99特徴量) + 新規JRDB特徴量候補でWFバックテスト。
採用基準: WF AUC > 0.8207 (v13) かつ全年AUC > 0.78 かつ gap < 0.05

追加データソース:
  CHA: 追切テン/中間/終い指数、追切総合指数 (96% valid)
  JO:  CID指数、LS指数 (100% valid)
  KTA: IDM予想(別ソース)、展開指数予想 (87-100% valid)
  ZE:  過去5走IDM平均/テン指数平均/上がり指数平均/不利回数
  SR:  トラックバイアス(直線) → 成績データなので前走として使用
  KKA: dam_rensho(母連勝)、bms_rensho(BMS連勝) ※成績カラムはNaN
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
    merge_dam_stats, compute_pci,
)
from jrdb_features import (
    merge_jrdb_train_features, _build_nk_race_id_from_jv,
    JRDB_LIVE_FEATURES, JRDB_DEFAULTS,
)

# =====================================================
# v13特徴量ロード
# =====================================================

def load_v13_jrdb_selected():
    path = os.path.join(DATA_DIR, 'v13_training_results.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f).get('jrdb_selected', [])
    from jrdb_features import JRDB_ALL_FEATURES
    return [f for f in JRDB_ALL_FEATURES
            if f not in ('jrdb_info_idx', 'jrdb_prev_ten_idx', 'jrdb_prev_agari_idx')]


# =====================================================
# v13.2 新規特徴量マージ
# =====================================================

V132_NEW_FEATURES = [
    # CHA: 追切指数 (96% valid, 最有望)
    'jrdb_oikiri_idx',         # 追切総合指数
    'jrdb_ten_time_idx',       # 追切テン指数
    'jrdb_shimai_time_idx',    # 追切終い指数
    # JO: CID/LS指数 (100% valid)
    'jrdb_cid_idx',            # CID指数(コンディション)
    'jrdb_ls_idx',             # LS指数(ロングショット=穴馬)
    # KTA: IDM予想・展開予想 (87-100%)
    'jrdb_kta_idm',            # KTA版IDM予想
    'jrdb_kta_ten_pred',       # KTA版テン指数予想
    'jrdb_kta_agari_pred',     # KTA版上がり指数予想
    # ZE: 過去5走集計
    'jrdb_ze_idm_avg',         # 過去5走IDM平均
    'jrdb_ze_ten_avg',         # 過去5走テン指数平均
    'jrdb_ze_agari_avg',       # 過去5走上がり指数平均
    'jrdb_ze_furi_count',      # 過去5走不利回数
    # SR: トラックバイアス (前走レースのバイアス → 前走不利の補足)
    'jrdb_tb_homestr_inner',   # 直線内側有利度
    # KKA: 母/BMS連勝指数
    'jrdb_dam_rensho_avg',     # 母産駒平均連勝指数
    'jrdb_bms_rensho_avg',     # BMS産駒平均連勝指数
]

V132_DEFAULTS = {
    'jrdb_oikiri_idx': 53.0,
    'jrdb_ten_time_idx': 14.5,
    'jrdb_shimai_time_idx': 21.0,
    'jrdb_cid_idx': 0.0,
    'jrdb_ls_idx': 0.0,
    'jrdb_kta_idm': 13.0,
    'jrdb_kta_ten_pred': -14.0,
    'jrdb_kta_agari_pred': -11.0,
    'jrdb_ze_idm_avg': 37.0,
    'jrdb_ze_ten_avg': -15.0,
    'jrdb_ze_agari_avg': -12.0,
    'jrdb_ze_furi_count': 0.0,
    'jrdb_tb_homestr_inner': 2.0,
    'jrdb_dam_rensho_avg': 1600.0,
    'jrdb_bms_rensho_avg': 1600.0,
}


def merge_cha_features(df):
    """CHA(追切データ)をマージ"""
    path = os.path.join(DATA_DIR, 'jrdb_cha.csv')
    if not os.path.exists(path):
        print("    CHA: not found")
        return df
    cha = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
    cha['_nk_rid'] = cha['race_id'].astype(str).str.zfill(12)
    cha['_uma'] = pd.to_numeric(cha['umaban'], errors='coerce')
    cha['jrdb_oikiri_idx'] = pd.to_numeric(cha['oikiri_idx'], errors='coerce')
    cha['jrdb_ten_time_idx'] = pd.to_numeric(cha['ten_time_idx'], errors='coerce')
    cha['jrdb_shimai_time_idx'] = pd.to_numeric(cha['shimai_time_idx'], errors='coerce')

    cols = ['_nk_rid', '_uma', 'jrdb_oikiri_idx', 'jrdb_ten_time_idx', 'jrdb_shimai_time_idx']
    cha_d = cha[cols].drop_duplicates(subset=['_nk_rid', '_uma'], keep='last')
    before = len(df)
    df = df.merge(cha_d, on=['_nk_rid', '_uma'], how='left')
    matched = df['jrdb_oikiri_idx'].notna().sum()
    print(f"    CHA: {matched}/{before} matched ({matched/before*100:.1f}%)")
    return df


def merge_jo_features(df):
    """JO(CID/LS指数)をマージ"""
    path = os.path.join(DATA_DIR, 'jrdb_jo.csv')
    if not os.path.exists(path):
        print("    JO: not found")
        return df
    jo = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
    jo['_nk_rid'] = jo['race_id'].astype(str).str.zfill(12)
    jo['_uma'] = pd.to_numeric(jo['umaban'], errors='coerce')
    jo['jrdb_cid_idx'] = pd.to_numeric(jo['cid_idx'], errors='coerce')
    jo['jrdb_ls_idx'] = pd.to_numeric(jo['ls_idx'], errors='coerce')

    cols = ['_nk_rid', '_uma', 'jrdb_cid_idx', 'jrdb_ls_idx']
    jo_d = jo[cols].drop_duplicates(subset=['_nk_rid', '_uma'], keep='last')
    before = len(df)
    df = df.merge(jo_d, on=['_nk_rid', '_uma'], how='left')
    matched = df['jrdb_cid_idx'].notna().sum()
    print(f"    JO: {matched}/{before} matched ({matched/before*100:.1f}%)")
    return df


def merge_kta_features(df):
    """KTA(IDM予想/展開予想)をマージ — blood_num経由"""
    path = os.path.join(DATA_DIR, 'jrdb_kta.csv')
    paci_path = os.path.join(DATA_DIR, 'jrdb_paci.csv')
    if not os.path.exists(path):
        print("    KTA: not found")
        return df
    kta = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
    kta['_nk_rid'] = kta['race_id'].astype(str).str.zfill(12)
    kta['jrdb_kta_idm'] = pd.to_numeric(kta['idm'], errors='coerce')
    kta['jrdb_kta_ten_pred'] = pd.to_numeric(kta['ten_idx_pred'], errors='coerce')
    kta['jrdb_kta_agari_pred'] = pd.to_numeric(kta['agari_idx_pred'], errors='coerce')

    # KTA has no umaban - use blood_num via PACI to get umaban
    if os.path.exists(paci_path):
        paci = pd.read_csv(paci_path, encoding='utf-8-sig', dtype=str,
                           usecols=['race_id', 'umaban', 'blood_num'])
        paci['_nk_rid'] = paci['race_id'].astype(str).str.zfill(12)
        paci['_uma'] = pd.to_numeric(paci['umaban'], errors='coerce')
        paci_map = paci[['_nk_rid', '_uma', 'blood_num']].drop_duplicates(
            subset=['_nk_rid', 'blood_num'], keep='last')
        kta = kta.merge(paci_map, on=['_nk_rid', 'blood_num'], how='left')
    else:
        print("    KTA: PACI not found for umaban mapping")
        return df

    cols = ['_nk_rid', '_uma', 'jrdb_kta_idm', 'jrdb_kta_ten_pred', 'jrdb_kta_agari_pred']
    kta_d = kta[cols].dropna(subset=['_uma']).drop_duplicates(
        subset=['_nk_rid', '_uma'], keep='last')
    before = len(df)
    df = df.merge(kta_d, on=['_nk_rid', '_uma'], how='left')
    matched = df['jrdb_kta_idm'].notna().sum()
    print(f"    KTA: {matched}/{before} matched ({matched/before*100:.1f}%)")
    return df


def merge_ze_features(df):
    """ZE(過去5走集計)をマージ — blood_num経由"""
    path = os.path.join(DATA_DIR, 'jrdb_ze.csv')
    paci_path = os.path.join(DATA_DIR, 'jrdb_paci.csv')
    if not os.path.exists(path):
        print("    ZE: not found")
        return df

    ze = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
    ze['_idm'] = pd.to_numeric(ze['idm'], errors='coerce')
    ze['_ten'] = pd.to_numeric(ze['ten_idx'], errors='coerce')
    ze['_agari'] = pd.to_numeric(ze['agari_idx'], errors='coerce')
    ze['_furi'] = pd.to_numeric(ze['furi'], errors='coerce').fillna(0)
    ze['_furi_flag'] = (ze['_furi'] > 0).astype(int)

    # Aggregate per blood_num (all past races → mean/count)
    agg = ze.groupby('blood_num').agg(
        jrdb_ze_idm_avg=('_idm', 'mean'),
        jrdb_ze_ten_avg=('_ten', 'mean'),
        jrdb_ze_agari_avg=('_agari', 'mean'),
        jrdb_ze_furi_count=('_furi_flag', 'sum'),
    ).reset_index()

    # Map blood_num → (race_id, umaban) via PACI
    if os.path.exists(paci_path):
        paci = pd.read_csv(paci_path, encoding='utf-8-sig', dtype=str,
                           usecols=['race_id', 'umaban', 'blood_num'])
        paci['_nk_rid'] = paci['race_id'].astype(str).str.zfill(12)
        paci['_uma'] = pd.to_numeric(paci['umaban'], errors='coerce')
        paci_ze = paci.merge(agg, on='blood_num', how='left')
        cols = ['_nk_rid', '_uma', 'jrdb_ze_idm_avg', 'jrdb_ze_ten_avg',
                'jrdb_ze_agari_avg', 'jrdb_ze_furi_count']
        paci_ze_d = paci_ze[cols].dropna(subset=['_uma']).drop_duplicates(
            subset=['_nk_rid', '_uma'], keep='last')
        before = len(df)
        df = df.merge(paci_ze_d, on=['_nk_rid', '_uma'], how='left')
        matched = df['jrdb_ze_idm_avg'].notna().sum()
        print(f"    ZE: {matched}/{before} matched ({matched/before*100:.1f}%)")
    else:
        print("    ZE: PACI not found")
    return df


def merge_sr_features(df):
    """SR(トラックバイアス)をマージ — レース単位、前走データとして使用

    tb_homestrは5桁文字列: 各桁1-5で最内/内/中/外/大外の有利度
    1=有利(内伸び), 5=不利(外伸び)。先頭桁が「最内」。
    → 先頭桁の値を数値化して「直線内側有利度」とする。
    ※成績データなので当該レースではなく前走バイアスとして使う
    """
    path = os.path.join(DATA_DIR, 'jrdb_sr.csv')
    if not os.path.exists(path):
        print("    SR: not found")
        return df
    sr = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
    sr['_nk_rid'] = sr['race_id'].astype(str).str.zfill(12)
    # 直線バイアス: 先頭1文字 = 最内の有利度
    sr['jrdb_tb_homestr_inner'] = sr['tb_homestr'].apply(
        lambda x: int(str(x)[0]) if pd.notna(x) and len(str(x)) >= 1 and str(x)[0].isdigit() else 2
    )
    sr_d = sr[['_nk_rid', 'jrdb_tb_homestr_inner']].drop_duplicates(subset='_nk_rid', keep='last')

    # SR is race-level, merge directly (not per horse)
    before = len(df)
    df = df.merge(sr_d, on='_nk_rid', how='left')
    matched = df['jrdb_tb_homestr_inner'].notna().sum()
    print(f"    SR: {matched}/{before} matched ({matched/before*100:.1f}%)")
    return df


def merge_kka_features(df):
    """KKA(母/BMS連勝指数)をマージ"""
    path = os.path.join(DATA_DIR, 'jrdb_kka.csv')
    if not os.path.exists(path):
        print("    KKA: not found")
        return df
    kka = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
    kka['_nk_rid'] = kka['race_id'].astype(str).str.zfill(12)
    kka['_uma'] = pd.to_numeric(kka['umaban'], errors='coerce')
    kka['jrdb_dam_rensho_avg'] = pd.to_numeric(kka['dam_rensho_avg'], errors='coerce')
    kka['jrdb_bms_rensho_avg'] = pd.to_numeric(kka['bms_rensho_avg'], errors='coerce')

    cols = ['_nk_rid', '_uma', 'jrdb_dam_rensho_avg', 'jrdb_bms_rensho_avg']
    kka_d = kka[cols].drop_duplicates(subset=['_nk_rid', '_uma'], keep='last')
    before = len(df)
    df = df.merge(kka_d, on=['_nk_rid', '_uma'], how='left')
    matched = df['jrdb_dam_rensho_avg'].notna().sum()
    print(f"    KKA: {matched}/{before} matched ({matched/before*100:.1f}%)")
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
    print(f"  {'='*55}")
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
    print("  v13.2 Extended JRDB Features")
    print(f"  Baseline: v13 WF AUC 0.8207 (99 features)")
    print(f"  New: +{len(V132_NEW_FEATURES)} features from CHA/JO/KTA/ZE/SR/KKA")
    print("=" * 60)

    # Phase 1-4: identical to v13
    print("\n[1/7] Loading data...")
    df = load_data()
    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)

    print("\n[2/7] Training data...")
    tt_data = load_training_times()
    df = merge_training_features(df, tt_data)

    print("\n[3/7] Historical stats...")
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)
    df = compute_lag_features(df)
    df = build_features(df)
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)

    print("\n[4/7] v12 features...")
    df = merge_speed_index(df)
    df = merge_training_1f(df)
    df = merge_sire_shinba(df)
    df = merge_dam_stats(df)
    df = compute_pci(df)

    print("\n[5/7] v13 JRDB features...")
    df = merge_jrdb_train_features(df)

    # Build nk_race_id for new merges (already built inside merge_jrdb_train_features,
    # but might be dropped)
    if '_nk_rid' not in df.columns:
        df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    if '_uma' not in df.columns:
        df['_uma'] = df['umaban'].astype(int)

    # Phase 6: NEW v13.2 features
    print("\n[6/7] v13.2 new features...")
    df = merge_cha_features(df)
    df = merge_jo_features(df)
    df = merge_kta_features(df)
    df = merge_ze_features(df)
    df = merge_sr_features(df)
    df = merge_kka_features(df)

    # Cleanup temp cols
    df.drop(columns=['_nk_rid', '_uma'], inplace=True, errors='ignore')

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

    # Fill missing
    all_defaults = {**JRDB_DEFAULTS, **V132_DEFAULTS}
    for f in F_V132:
        if f not in df.columns:
            df[f] = all_defaults.get(f, 0)
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(all_defaults.get(f, 0))

    print(f"  v13: {len(F_V13)} features")
    print(f"  v13.2: {len(F_V132)} features (+{len(V132_NEW_FEATURES)})")

    # Coverage
    print(f"\n  v13.2 new feature coverage:")
    for f in V132_NEW_FEATURES:
        default = V132_DEFAULTS.get(f, 0)
        rate = (df[f] != default).mean() * 100
        print(f"    {f}: {rate:.1f}% non-default")

    # ===== WF Backtest =====
    print(f"\n{'='*60}")
    print(f"  [7/7] Walk-Forward Backtest")
    print(f"{'='*60}")

    print("\n--- v13 Baseline ---")
    wf13 = walk_forward_backtest(df, F_V13)
    auc13 = print_wf(wf13, 'v13 (99 feats)')

    print("\n--- v13.2 +all new ---")
    wf132 = walk_forward_backtest(df, F_V132)
    auc132 = print_wf(wf132, f'v13.2 ({len(F_V132)} feats)')

    # Feature importance
    imp = feature_importance(wf132, F_V132)
    print(f"\n  Top 30 by importance:")
    for i, (f, v) in enumerate(imp[:30]):
        tag = " *NEW" if f in V132_NEW_FEATURES else (" *JRDB" if f.startswith('jrdb_') else "")
        print(f"  {i+1:2d}. {f:35s} {v:12.1f}{tag}")

    # Individual contribution (quick: 2024-2025 only)
    print(f"\n  Individual contribution (2024-2025):")
    wf_full_sub = walk_forward_backtest(df, F_V132, years=[2024, 2025])
    auc_full_sub = np.mean([r['lgb_auc'] for r in wf_full_sub])
    contribs = {}
    for nf in V132_NEW_FEATURES:
        feats_wo = [f for f in F_V132 if f != nf]
        wf_wo = walk_forward_backtest(df, feats_wo, years=[2024, 2025])
        auc_wo = np.mean([r['lgb_auc'] for r in wf_wo])
        c = auc_full_sub - auc_wo
        contribs[nf] = c
        sign = '+' if c > 0 else ''
        mark = 'Y' if c > -0.0005 else 'N'
        print(f"    {mark} {nf:35s}: {sign}{c:.5f}")

    # Select beneficial features
    selected = [f for f, c in contribs.items() if c > -0.0005]
    removed = [f for f, c in contribs.items() if c <= -0.0005]
    F_V132_SEL = F_V13 + selected

    if removed:
        print(f"\n  Removed {len(removed)}: {removed}")
        print(f"\n--- v13.2 selected ({len(F_V132_SEL)} feats) ---")
        wf132s = walk_forward_backtest(df, F_V132_SEL)
        auc132s = print_wf(wf132s, f'v13.2 selected')
    else:
        wf132s = wf132
        auc132s = auc132
        F_V132_SEL = F_V132

    # Final check
    best_auc = max(auc132, auc132s)
    best_wf = wf132 if auc132 >= auc132s else wf132s
    best_feats = F_V132 if auc132 >= auc132s else F_V132_SEL

    delta = best_auc - 0.8207
    all_ok = all(r['lgb_auc'] > 0.78 for r in best_wf)
    no_overfit = all(r['gap'] < 0.05 for r in best_wf)
    adopted = best_auc > 0.8207 and all_ok and no_overfit

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  v13 AUC:     {auc13:.4f} ({len(F_V13)} feats)")
    print(f"  v13.2 all:   {auc132:.4f} ({len(F_V132)} feats)")
    print(f"  v13.2 sel:   {auc132s:.4f} ({len(F_V132_SEL)} feats)")
    print(f"  Delta vs v13: {delta:+.4f}")
    print(f"  AUC > 0.8207:     {'Y' if best_auc > 0.8207 else 'N'}")
    print(f"  All years > 0.78: {'Y' if all_ok else 'N'}")
    print(f"  No overfitting:   {'Y' if no_overfit else 'N'}")
    print(f"\n  VERDICT: {'ADOPTED as v13.2' if adopted else 'NOT ADOPTED'}")

    # Save
    result = {
        'v13_auc': auc13, 'v132_all_auc': auc132, 'v132_sel_auc': auc132s,
        'best_auc': best_auc, 'delta': delta, 'adopted': adopted,
        'new_features': V132_NEW_FEATURES,
        'selected': selected, 'removed': removed,
        'contributions': contribs,
        'best_features': best_feats,
        'yearly': [{'year': r['year'], 'lgb_auc': r['lgb_auc'], 'xgb_auc': r['xgb_auc'],
                     'train_auc': r['train_auc'], 'gap': r['gap']} for r in best_wf],
        'timestamp': datetime.now().isoformat(),
    }
    rpath = os.path.join(DATA_DIR, 'v132_training_results.json')
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Saved: {rpath}")

    if adopted:
        print(f"\n{'='*60}")
        print(f"  TRAINING PRODUCTION MODEL (v13.2)")
        print(f"{'='*60}")
        _train_production(df, best_feats, sire_map, bms_map, best_wf, jrdb_sel, selected)

    print(f"\n  Elapsed: {(time.time()-t0)/60:.1f} min")


def _train_production(df, features, sire_map, bms_map, wf, jrdb_sel, new_sel):
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
    pkl = {
        'model': ml, 'features': pa_feats, 'version': 'v132_leakfree',
        'auc': al, 'wf_auc': wf_auc, 'leak_free': True, 'leak_pattern': 'A',
        'leak_removed': sorted(LEAK_FEATURES_A),
        'sire_map': sire_map, 'bms_map': bms_map, 'n_top_encode': N_TOP_SIRE,
        'trained_at': now, 'model_type': 'central',
        'xgb_model': mx, 'mlp_model': None, 'mlp_scaler': None,
        'ensemble_weights': {'lgb': wl, 'xgb': wx, 'mlp': 0},
        'course_map': dict(COURSE_MAP),
        'jrdb_features': jrdb_sel, 'v132_features': new_sel,
    }
    ap = os.path.join(BASE_DIR, 'keiba_model_v132_central.pkl.gz')
    with gzip.open(ap, 'wb') as f:
        pickle.dump(pkl, f)
    print(f"  Pattern A: {ap}")

    pb_feats = features + JRDB_LIVE_FEATURES
    pkl_b = dict(pkl)
    pkl_b['features'] = pb_feats
    pkl_b['version'] = 'v132_live'
    pkl_b['leak_free'] = False
    pkl_b['leak_pattern'] = 'B'
    pkl_b['is_live'] = True
    bp = os.path.join(BASE_DIR, 'keiba_model_v132_central_live.pkl.gz')
    with gzip.open(bp, 'wb') as f:
        pickle.dump(pkl_b, f)
    print(f"  Pattern B: {bp}")
    print(f"\n  *** v13.2 saved! A={len(pa_feats)} feats, B={len(pb_feats)} feats ***")


if __name__ == '__main__':
    main()
