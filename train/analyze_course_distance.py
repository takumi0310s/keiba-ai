#!/usr/bin/env python
"""Analyze AUC by course and distance to identify where specialized models help."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score

from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features,
    compute_distance_aptitude, compute_frame_advantage,
    COURSE_MAP, train_lgb, train_xgb,
)
from train_v92_leakfree import LEAK_FEATURES_A, FEATURES_PATTERN_A
from train_v93_leakfree import FEATURES_V93_PATTERN_A

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')


def prepare_data():
    """Load and prepare full dataset."""
    df = load_data()
    lap_df = load_lap_data()
    if lap_df is not None:
        df = df.merge(lap_df, on='race_id_str', how='left')
    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)
    tt_data = load_training_times()
    df = merge_training_features(df, tt_data)
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)
    df = compute_lag_features(df)
    df = build_features(df)
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)

    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()

    features = FEATURES_V93_PATTERN_A
    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    return df, features, sire_map, bms_map


def analyze_auc_by_segment(df, features):
    """Compute AUC by course and distance segments using the global model."""
    max_year = df['year_full'].max()
    valid_mask = df['year_full'] >= (max_year - 1)
    train_mask = ~valid_mask
    y = df['target'].values

    # Train global model
    X = df[features].values
    lgb_model = train_lgb(X[train_mask], y[train_mask], X[valid_mask], y[valid_mask], features)
    preds = lgb_model.predict(X[valid_mask])
    global_auc = roc_auc_score(y[valid_mask], preds)
    print(f"\nGlobal AUC: {global_auc:.4f}")

    valid_df = df[valid_mask].copy()
    valid_df['pred'] = preds

    # Distance categories
    dist_bins = [(0, 1200, '~1200'), (1201, 1400, '1400'), (1401, 1600, '1600'),
                 (1601, 1800, '1800'), (1801, 2000, '2000'), (2001, 9999, '2200+')]

    valid_df['dist_label'] = '?'
    for lo, hi, label in dist_bins:
        mask = (valid_df['distance'] >= lo) & (valid_df['distance'] <= hi)
        valid_df.loc[mask, 'dist_label'] = label

    inv_course = {v: k for k, v in COURSE_MAP.items()}

    results = {'global_auc': global_auc, 'by_course': {}, 'by_distance': {}, 'by_course_distance': {}}

    # By course
    print(f"\n{'Course':<8} {'N':>7} {'AUC':>8} {'Delta':>8}")
    print("-" * 35)
    for cenc in sorted(valid_df['course_enc'].unique()):
        cname = inv_course.get(cenc, f'C{cenc}')
        mask = valid_df['course_enc'] == cenc
        sub = valid_df[mask]
        if len(sub) < 100 or sub['target'].nunique() < 2:
            continue
        auc = roc_auc_score(sub['target'], sub['pred'])
        delta = auc - global_auc
        results['by_course'][cname] = {'n': len(sub), 'auc': round(auc, 4), 'delta': round(delta, 4)}
        print(f"{cname:<8} {len(sub):>7} {auc:>8.4f} {delta:>+8.4f}")

    # By distance
    print(f"\n{'Distance':<8} {'N':>7} {'AUC':>8} {'Delta':>8}")
    print("-" * 35)
    for _, _, label in dist_bins:
        mask = valid_df['dist_label'] == label
        sub = valid_df[mask]
        if len(sub) < 100 or sub['target'].nunique() < 2:
            continue
        auc = roc_auc_score(sub['target'], sub['pred'])
        delta = auc - global_auc
        results['by_distance'][label] = {'n': len(sub), 'auc': round(auc, 4), 'delta': round(delta, 4)}
        print(f"{label:<8} {len(sub):>7} {auc:>8.4f} {delta:>+8.4f}")

    # By course x distance (only large segments)
    print(f"\n{'Course-Dist':<16} {'N':>7} {'AUC':>8} {'Delta':>8}")
    print("-" * 43)
    for cenc in sorted(valid_df['course_enc'].unique()):
        cname = inv_course.get(cenc, f'C{cenc}')
        for _, _, dlabel in dist_bins:
            mask = (valid_df['course_enc'] == cenc) & (valid_df['dist_label'] == dlabel)
            sub = valid_df[mask]
            if len(sub) < 200 or sub['target'].nunique() < 2:
                continue
            auc = roc_auc_score(sub['target'], sub['pred'])
            delta = auc - global_auc
            key = f"{cname}_{dlabel}"
            results['by_course_distance'][key] = {'n': len(sub), 'auc': round(auc, 4), 'delta': round(delta, 4)}
            marker = " ***" if abs(delta) > 0.01 else ""
            print(f"{key:<16} {len(sub):>7} {auc:>8.4f} {delta:>+8.4f}{marker}")

    return results, lgb_model


def main():
    print("=" * 60)
    print("  COURSE x DISTANCE AUC ANALYSIS")
    print("=" * 60)
    df, features, _, _ = prepare_data()
    results, _ = analyze_auc_by_segment(df, features)

    out_path = os.path.join(BASE_DIR, 'analysis_course_distance.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
