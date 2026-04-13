#!/usr/bin/env python
"""v16 premium新特徴量 + AM8 backtest 一括実行

Steps:
  1. cache: _v15_train_df_cache.pkl を build_v15_dataframe() から構築（~30-60min）
     すでに存在すればロードのみ
  2. 5新特徴量(v16 premium) を既存dfにマージ
  3. WF #1: v16 all-in (155特徴量) ablation判定
  4. WF #2: AM8 (A+B only) 138特徴量 evaluation
  5. 採用判定 → data/v16_wf_results.json に結果保存

採用基準: WF mean AUC > 0.8858, 全年 AUC > 0.85, max gap < 0.05

Usage:
    nohup python -u train/run_v16_and_am8_wf.py > logs/v16_wf.log 2>&1 &
"""
import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

CACHE_PATH = os.path.join(DATA_DIR, '_v15_train_df_cache.pkl')
OUT_JSON = os.path.join(DATA_DIR, 'v16_wf_results.json')
BASELINE_AUC = 0.8858  # user target
USER_BASELINE_FROM_REPORT = 0.8856  # v15_master_report.json


def build_or_load_cache():
    if os.path.exists(CACHE_PATH):
        print(f"[CACHE] load existing: {CACHE_PATH}")
        with open(CACHE_PATH, 'rb') as f:
            d = pickle.load(f)
        return d['df'], d.get('sire_map'), d.get('bms_map'), d.get('v15_features')

    print(f"[CACHE] not found - building v15 dataframe (may take 30-60min)")
    from train_v15_master import build_v15_dataframe, get_v15_all_features, fill_v15_defaults
    t0 = time.time()
    df, sire_map, bms_map = build_v15_dataframe()
    print(f"[CACHE] build_v15_dataframe done: {len(df)} rows in {(time.time()-t0)/60:.1f}min")

    v15_features, _ = get_v15_all_features()
    df = fill_v15_defaults(df, v15_features)

    with open(CACHE_PATH, 'wb') as f:
        pickle.dump({'df': df, 'sire_map': sire_map, 'bms_map': bms_map,
                      'v15_features': v15_features}, f)
    print(f"[CACHE] saved: {CACHE_PATH}")
    return df, sire_map, bms_map, v15_features


def run_wf(df, features, label):
    from train_v15_master import walk_forward_4model, summarize_results, check_acceptance, WF_YEARS
    print(f"\n{'='*70}\n  WF RUN: {label}  (features={len(features)})\n{'='*70}")
    t0 = time.time()
    results = walk_forward_4model(df, features, years=WF_YEARS, label=label)
    mean = summarize_results(results, label=label)
    adopted, reasons = check_acceptance(results, mean, baseline=USER_BASELINE_FROM_REPORT)
    user_adopted = mean > BASELINE_AUC and adopted
    elapsed = (time.time() - t0) / 60
    print(f"\n[{label}] mean WF AUC = {mean:.6f}  elapsed={elapsed:.1f}min")
    print(f"[{label}] adopted(base={USER_BASELINE_FROM_REPORT}): {adopted}, user_target({BASELINE_AUC}): {user_adopted}")
    if not adopted:
        print(f"[{label}] reasons: {reasons}")
    return {
        'label': label,
        'n_features': len(features),
        'mean_auc': mean,
        'per_year': results,
        'adopted_vs_base': adopted,
        'adopted_vs_user_target': user_adopted,
        'reasons': reasons,
        'elapsed_min': elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-ablation', action='store_true', help='Skip per-feature ablation (faster)')
    ap.add_argument('--only-ablation', action='store_true', help='Only run per-feature ablation')
    args = ap.parse_args()

    t_start = time.time()
    out = {'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}

    # Step 1-2: Cache + features
    df, sire_map, bms_map, v15_features = build_or_load_cache()
    print(f"\n[INFO] v15 features: {len(v15_features)}")

    # race_id_unique を IntraRaceAttention 用に準備
    from train_v135b_intra_ensemble import build_race_id
    df = build_race_id(df)

    from features_v16_premium import compute_all_v16_premium_features, get_v16_premium_features
    print(f"\n[STEP 2] merging v16 premium features...")
    df = compute_all_v16_premium_features(df)
    v16_feats = get_v16_premium_features()
    print(f"[INFO] v16 premium features: {v16_feats}")

    # Fill NaN in v16 feats (defensive)
    from features_v16_premium import V16_PREMIUM_DEFAULTS
    for f in v16_feats:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(V16_PREMIUM_DEFAULTS[f])

    # Step 3: WF #1 - v15 + v16 all-in (155 features)
    r1 = None
    if not args.only_ablation:
        combined_features = list(v15_features) + v16_feats
        r1 = run_wf(df, combined_features, 'v15+v16_all_in')
        out['v16_all_in'] = r1

    # Step 3b: Ablation - each v16 feature individually on top of v15
    if not args.skip_ablation:
        out['v16_ablation'] = {}
        for feat in v16_feats:
            ab_feats = list(v15_features) + [feat]
            r_ab = run_wf(df, ab_feats, f'v15+{feat}')
            out['v16_ablation'][feat] = r_ab

    # Step 4: WF #2 - AM8 only (A+B) on v15 feature set
    with open(os.path.join(DATA_DIR, 'feature_availability_am8.json'), 'r', encoding='utf-8') as f:
        av = json.load(f)
    am8_feats_all = av['A']['features'] + av['B']['features']
    # Keep only those existing in df
    am8_feats = [f for f in am8_feats_all if f in df.columns]
    missing = [f for f in am8_feats_all if f not in df.columns]
    if missing:
        print(f"[WARN] AM8 missing in df: {len(missing)}: {missing[:10]}")
    r_am8 = run_wf(df, am8_feats, 'AM8_A+B_only')
    out['am8'] = r_am8
    out['am8']['missing_features'] = missing

    # Summary
    out['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    out['total_elapsed_min'] = (time.time() - t_start) / 60
    out['baseline_from_report'] = USER_BASELINE_FROM_REPORT
    out['user_target'] = BASELINE_AUC

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[DONE] results saved: {OUT_JSON}")
    print(f"  total elapsed: {out['total_elapsed_min']:.1f}min")

    # Notify
    try:
        from tools.notify import send_discord
        msg_lines = [
            f"v16 all_in: {r1['mean_auc']:.4f} ({'ADOPTED' if r1['adopted_vs_user_target'] else 'REJECTED'})",
            f"AM8 only: {r_am8['mean_auc']:.4f}",
        ]
        for feat, r in out.get('v16_ablation', {}).items():
            msg_lines.append(f"  +{feat}: {r['mean_auc']:.4f}")
        adopted_flag = r1['adopted_vs_user_target'] if r1 else False
        send_discord("v16/AM8 WF完了", "\n".join(msg_lines),
                     color="green" if adopted_flag else "yellow",
                     channel="updates")
    except Exception as e:
        print(f"[WARN] notify failed: {e}")


if __name__ == '__main__':
    main()
