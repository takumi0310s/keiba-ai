#!/usr/bin/env python3
"""v15候補特徴量スクリーニング

v14.1ベース(131特徴量, Pattern A)に候補特徴量を1つずつ追加し、
2025年1年foldでLGB AUC差分を簡易計測。

有望候補（+0.0005以上）のみフルWF対象とする。
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'seed': 42,
}


def screen_candidates():
    from train_v141_paci_tierA import build_v141_dataframe, get_v141_features, fill_v141_defaults
    from jrdb_features import JRDB_DEFAULTS, _build_nk_race_id_from_jv

    print("Loading v14.1 base data...")
    df, sire_map, bms_map = build_v141_dataframe()
    base_feats, jrdb_sel = get_v141_features()
    df = fill_v141_defaults(df, base_feats)

    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses'] >= 5].copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)

    # Baseline: 2025 fold, LGB only
    train = df[df['year'] < 25]
    test = df[df['year'] == 25]
    print(f"Train: {len(train)}, Test(2025): {len(test)}")
    print(f"Base features: {len(base_feats)}")

    X_train = train[base_feats].values.astype(np.float32)
    y_train = train['target'].values
    X_test = test[base_feats].values.astype(np.float32)
    y_test = test['target'].values

    d_train = lgb.Dataset(X_train, y_train)
    d_val = lgb.Dataset(X_test, y_test, reference=d_train)
    model_base = lgb.train(LGB_PARAMS, d_train, num_boost_round=1000,
                           valid_sets=[d_val], callbacks=[lgb.early_stopping(50, verbose=False)])
    base_pred = model_base.predict(X_test)
    base_auc = roc_auc_score(y_test, base_pred)
    print(f"\n>>> BASELINE AUC (2025, LGB): {base_auc:.6f}")

    # =====================================================
    # Build candidate features
    # =====================================================
    if '_nk_rid' not in df.columns:
        df['_nk_rid'] = _build_nk_race_id_from_jv(df)
    if '_uma' not in df.columns:
        df['_uma'] = pd.to_numeric(df.get('umaban', 0), errors='coerce')

    # --- A1: PACI Tier B marks ---
    paci_path = os.path.join(DATA_DIR, 'jrdb_paci.csv')
    paci_b_feats = []
    if os.path.exists(paci_path):
        paci = pd.read_csv(paci_path, encoding='utf-8-sig', dtype=str)
        mark_cols = ['sogo_mark', 'idm_mark', 'jockey_mark', 'train_mark']
        for mc in mark_cols:
            if mc in paci.columns:
                paci[mc] = pd.to_numeric(paci[mc], errors='coerce').fillna(0)
        paci['_uma'] = pd.to_numeric(paci['umaban'], errors='coerce')
        paci['_nk_rid'] = paci['race_id'].astype(str).str.zfill(12)
        paci_merge = paci[['_nk_rid', '_uma'] + mark_cols].drop_duplicates(subset=['_nk_rid', '_uma'])
        df = df.merge(paci_merge, on=['_nk_rid', '_uma'], how='left', suffixes=('', '_paci_b'))
        for mc in mark_cols:
            col = f'paci_{mc}'
            df[col] = df[mc].fillna(0) if mc in df.columns else 0
            paci_b_feats.append(col)
        print(f"  PACI Tier B: {len(paci_b_feats)} features merged")

    # --- A2: KKA dam_rensho ---
    kka_path = os.path.join(DATA_DIR, 'jrdb_kka.csv')
    kka_feats = []
    if os.path.exists(kka_path):
        kka = pd.read_csv(kka_path, encoding='utf-8-sig', dtype=str)
        kka['_uma'] = pd.to_numeric(kka['umaban'], errors='coerce')
        kka['_nk_rid'] = kka['race_id'].astype(str).str.zfill(12)
        for c in ['dam_rensho_avg', 'bms_rensho_avg']:
            if c in kka.columns:
                kka[c] = pd.to_numeric(kka[c], errors='coerce').fillna(0)
                kka_feats.append(c)
        kka_merge = kka[['_nk_rid', '_uma'] + kka_feats].drop_duplicates(subset=['_nk_rid', '_uma'])
        df = df.merge(kka_merge, on=['_nk_rid', '_uma'], how='left', suffixes=('', '_kka'))
        for c in kka_feats:
            df[c] = df[c].fillna(0)
        print(f"  KKA dam/bms rensho: {len(kka_feats)} features merged")

    # --- E1: PCI (already in features as 'pci') ---
    # pci should already exist from v12

    # --- E2: age_peak_dist ---
    df['age_peak_dist'] = 0.0
    age_col = 'age' if 'age' in df.columns else 'horse_age'
    dist_col = 'distance' if 'distance' in df.columns else 'dist'
    if age_col in df.columns and dist_col in df.columns:
        _age = df[age_col].fillna(3)
        _dist = df[dist_col].fillna(1600)
        # Peak age varies by distance: short=3-4, mid=3-5, long=4-6
        _peak = np.where(_dist <= 1400, 3.5, np.where(_dist <= 2000, 4.0, 5.0))
        df['age_peak_dist'] = np.abs(_age - _peak)

    # --- E3: jockey_switch_wr_diff (would need expanding window, use proxy) ---
    # Skip complex computation, use simple proxy
    df['jockey_switch'] = 0  # placeholder

    # --- E6: SKB heavy_apt (already merged as jrdb_heavy_apt_skb in some versions) ---
    skb_feats = []
    skb_path = os.path.join(DATA_DIR, 'jrdb_skb.csv')
    if os.path.exists(skb_path):
        skb = pd.read_csv(skb_path, encoding='utf-8-sig', dtype=str)
        skb['_uma'] = pd.to_numeric(skb['umaban'], errors='coerce')
        skb['_nk_rid'] = skb['race_id'].astype(str).str.zfill(12)
        for c in ['kishi_code_1', 'baba_code_1', 'kyaku_code_1']:
            if c in skb.columns:
                skb[c] = pd.to_numeric(skb[c], errors='coerce').fillna(0)
                cname = f'skb_{c}'
                df[cname] = 0
                skb_feats.append(cname)
        if skb_feats:
            skb_merge = skb[['_nk_rid', '_uma'] + [c.replace('skb_', '') for c in skb_feats]].drop_duplicates(subset=['_nk_rid', '_uma'])
            df = df.merge(skb_merge, on=['_nk_rid', '_uma'], how='left', suffixes=('', '_skb2'))
            for c in skb_feats:
                raw = c.replace('skb_', '')
                if raw in df.columns:
                    df[c] = df[raw].fillna(0)
            print(f"  SKB codes: {len(skb_feats)} features merged")

    # Rebuild train/test
    train = df[df['year'] < 25]
    test = df[df['year'] == 25]

    # =====================================================
    # Screen each candidate
    # =====================================================
    candidates = {}

    def eval_candidate(name, new_feats):
        if not new_feats:
            return
        feats = base_feats + new_feats
        for f in new_feats:
            if f not in df.columns:
                df[f] = 0
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

        X_tr = train[feats].values.astype(np.float32)
        X_te = test[feats].values.astype(np.float32)
        d_tr = lgb.Dataset(X_tr, y_train)
        d_va = lgb.Dataset(X_te, y_test, reference=d_tr)
        m = lgb.train(LGB_PARAMS, d_tr, num_boost_round=1000,
                      valid_sets=[d_va], callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = m.predict(X_te)
        auc = roc_auc_score(y_test, pred)
        diff = auc - base_auc
        status = "ADOPT" if diff > 0.0005 else ("MARGINAL" if diff > 0 else "REJECT")
        print(f"  {name}: AUC={auc:.6f} diff={diff:+.6f} [{status}]")
        candidates[name] = {'auc': auc, 'diff': round(diff, 6), 'status': status, 'features': new_feats}

    print(f"\n{'='*60}")
    print(f"  Screening candidates (base AUC={base_auc:.6f})")
    print(f"{'='*60}\n")

    # A1: PACI Tier B marks (all at once)
    eval_candidate("A1_PACI_TierB", paci_b_feats)

    # A1 individual marks
    for mc in paci_b_feats:
        eval_candidate(f"A1_{mc}", [mc])

    # A2: KKA dam/bms rensho
    eval_candidate("A2_KKA_rensho", kka_feats)
    for c in kka_feats:
        eval_candidate(f"A2_{c}", [c])

    # E2: age_peak_dist
    eval_candidate("E2_age_peak_dist", ['age_peak_dist'])

    # E6: SKB codes
    if skb_feats:
        eval_candidate("E6_SKB_codes", skb_feats)

    # =====================================================
    # Save results
    # =====================================================
    results = {
        'baseline_auc': round(base_auc, 6),
        'baseline_features': len(base_feats),
        'candidates': candidates,
        'timestamp': pd.Timestamp.now().isoformat(),
    }

    out_path = os.path.join(DATA_DIR, 'v15_screening_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    adopt = [k for k, v in candidates.items() if v['status'] == 'ADOPT']
    marginal = [k for k, v in candidates.items() if v['status'] == 'MARGINAL']
    reject = [k for k, v in candidates.items() if v['status'] == 'REJECT']
    print(f"  ADOPT (>+0.0005): {len(adopt)}")
    for a in adopt:
        print(f"    {a}: {candidates[a]['diff']:+.6f}")
    print(f"  MARGINAL (0 to +0.0005): {len(marginal)}")
    for m in marginal:
        print(f"    {m}: {candidates[m]['diff']:+.6f}")
    print(f"  REJECT (<0): {len(reject)}")

    return results


if __name__ == '__main__':
    screen_candidates()
