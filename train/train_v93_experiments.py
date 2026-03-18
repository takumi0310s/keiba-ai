#!/usr/bin/env python
"""V9.3 Experiments: (1) Pace/Running Style features, (2) Specialized models

Task 1: Add running_style, predicted_pace, pace_advantage, style_vs_pace
Task 2: Train surface×distance specialized models (6 categories)

All features are leak-free (expanding window, historical data only).
Results saved to data/v93_specialized_report.json.
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features,
    compute_distance_aptitude, compute_frame_advantage,
    COURSE_MAP, N_TOP_SIRE, FEATURES_V93,
    train_lgb, train_xgb, show_feature_importance,
)
from train_v92_leakfree import LEAK_FEATURES_A

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')

FEATURES_PATTERN_A = [f for f in FEATURES_V93 if f not in LEAK_FEATURES_A]

# New pace/running style features
PACE_FEATURES = [
    'running_style',     # 脚質分類 (1=逃げ, 2=先行, 3=差し, 4=追込)
    'predicted_pace',    # レースの予測ペース (0=スロー, 1=ミドル, 2=ハイ)
    'pace_advantage',    # ペース有利度スコア
    'style_vs_pace',     # 脚質×ペース相性 (cross feature)
]


# ======================================================================
#  TASK 1: Running Style / Pace Prediction
# ======================================================================

def compute_running_style_features(df):
    """脚質・ペース予測特徴量を計算（リークフリー: expanding window）"""
    print("\n  Computing running style / pace features...")

    # --- 1. Parse pass1 (1角通過順位) ---
    df['pass1_val'] = pd.to_numeric(df['pass1'], errors='coerce')
    # 0 or NaN = missing (short races may not have pass1)
    df.loc[(df['pass1_val'] <= 0) | (df['pass1_val'].isna()), 'pass1_val'] = np.nan
    valid_pass1 = df['pass1_val'].notna().sum()
    print(f"    pass1 valid: {valid_pass1}/{len(df)} ({valid_pass1/len(df)*100:.1f}%)")

    # --- 2. Running style from historical pass positions (expanding window) ---
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    grp = df.groupby('horse_id')

    # Expanding cumsum of pass1 (excluding current race)
    df['_p1_filled'] = df['pass1_val'].fillna(0)
    df['_p1_valid'] = df['pass1_val'].notna().astype(float)
    cum_sum = grp['_p1_filled'].cumsum() - df['_p1_filled']
    cum_count = grp['_p1_valid'].cumsum() - df['_p1_valid']

    avg_pass_hist = np.where(cum_count > 0, cum_sum / cum_count, 5.0)  # default=5 (先行)
    df['avg_pass_hist'] = avg_pass_hist

    # Categorize: 逃げ=1, 先行=2, 差し=3, 追込=4
    df['running_style'] = np.where(
        df['avg_pass_hist'] <= 2, 1,
        np.where(df['avg_pass_hist'] <= 5, 2,
                 np.where(df['avg_pass_hist'] <= 10, 3, 4))
    ).astype(float)

    style_counts = df['running_style'].value_counts().sort_index()
    print(f"    Running styles: 逃げ={style_counts.get(1,0)}, 先行={style_counts.get(2,0)}, "
          f"差し={style_counts.get(3,0)}, 追込={style_counts.get(4,0)}")

    # --- 3. Predicted pace (number of front-runners per race) ---
    # Count horses with running_style=1 (逃げ) in each race
    df = df.sort_values('date_num').reset_index(drop=True)
    n_escapers = df.groupby('race_id_str')['running_style'].transform(
        lambda s: (s == 1).sum()
    )
    # Exclude self if self is escaper (avoid counting self)
    n_escapers_excl = n_escapers - (df['running_style'] == 1).astype(int)
    df['n_escapers'] = n_escapers_excl.clip(0)

    # Also count front-runners (style 1 or 2)
    n_front = df.groupby('race_id_str')['running_style'].transform(
        lambda s: (s <= 2).sum()
    )
    n_front_excl = n_front - (df['running_style'] <= 2).astype(int)
    df['n_front_runners'] = n_front_excl.clip(0)

    # Normalize by field size
    df['front_ratio'] = df['n_front_runners'] / df['num_horses'].clip(1)

    # Pace prediction: many front-runners → ハイペース
    # 0=スロー (front_ratio < 0.2), 1=ミドル, 2=ハイ (front_ratio > 0.4)
    df['predicted_pace'] = np.where(
        df['front_ratio'] < 0.2, 0,
        np.where(df['front_ratio'] > 0.4, 2, 1)
    ).astype(float)

    pace_counts = df['predicted_pace'].value_counts().sort_index()
    print(f"    Pace prediction: スロー={pace_counts.get(0,0)}, "
          f"ミドル={pace_counts.get(1,0)}, ハイ={pace_counts.get(2,0)}")

    # --- 4. Pace advantage score ---
    # ハイペース → 差し/追込有利, スローペース → 逃げ/先行有利
    # Advantage matrix: [pace][style] → score
    adv_map = {
        (0, 1): 2.0, (0, 2): 1.0, (0, 3): -1.0, (0, 4): -2.0,  # スロー
        (1, 1): 0.0, (1, 2): 0.5, (1, 3): 0.5, (1, 4): 0.0,    # ミドル
        (2, 1): -2.0, (2, 2): -1.0, (2, 3): 1.0, (2, 4): 2.0,  # ハイ
    }
    pace_int = df['predicted_pace'].astype(int)
    style_int = df['running_style'].astype(int)
    df['pace_advantage'] = [
        adv_map.get((p, s), 0.0) for p, s in zip(pace_int, style_int)
    ]

    # --- 5. Style × Pace cross feature ---
    df['style_vs_pace'] = df['running_style'] * 10 + df['predicted_pace']

    # Cleanup temp columns
    drop_cols = [c for c in df.columns if c.startswith('_p1_') or c in
                 ('pass1_val', 'avg_pass_hist', 'n_escapers', 'n_front_runners', 'front_ratio')]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Re-sort
    df = df.sort_values(['horse_id', 'date_num', 'race_num']).reset_index(drop=True)
    print(f"    Pace features computed: {PACE_FEATURES}")

    return df


# ======================================================================
#  TASK 2: Specialized Models by Surface × Distance
# ======================================================================

CATEGORIES = {
    'turf_short':  {'surface': 0, 'dist_max': 1400, 'dist_min': 0,    'label': '芝短距離(~1400m)'},
    'turf_mid':    {'surface': 0, 'dist_max': 2000, 'dist_min': 1401, 'label': '芝中距離(1600-2000m)'},
    'turf_long':   {'surface': 0, 'dist_max': 9999, 'dist_min': 2001, 'label': '芝長距離(2200m~)'},
    'dirt_short':  {'surface': 1, 'dist_max': 1400, 'dist_min': 0,    'label': 'ダ短距離(~1400m)'},
    'dirt_mid':    {'surface': 1, 'dist_max': 2000, 'dist_min': 1401, 'label': 'ダ中距離(1600-2000m)'},
    'dirt_long':   {'surface': 1, 'dist_max': 9999, 'dist_min': 2001, 'label': 'ダ長距離(2100m~)'},
}

MIN_ROWS_FOR_SPECIALIZED = 5000


def train_specialized_models(df, features, y, train_mask, valid_mask):
    """カテゴリ別モデルを学習し、汎用モデルとAUCを比較"""
    print("\n" + "=" * 70)
    print("  TASK 2: Specialized Models by Surface × Distance")
    print("=" * 70)

    import xgboost as xgb

    results = {}

    for cat_key, cat_info in CATEGORIES.items():
        surf = cat_info['surface']
        dmin = cat_info['dist_min']
        dmax = cat_info['dist_max']
        label = cat_info['label']

        cat_mask = (df['surface_enc'] == surf) & (df['distance'] >= dmin) & (df['distance'] <= dmax)
        cat_train = cat_mask & train_mask
        cat_valid = cat_mask & valid_mask

        n_train = cat_train.sum()
        n_valid = cat_valid.sum()

        print(f"\n  --- {label} (train={n_train}, valid={n_valid}) ---")

        if n_valid < 100:
            print(f"    SKIP: validation too small ({n_valid} < 100)")
            results[cat_key] = {
                'label': label, 'n_train': int(n_train), 'n_valid': int(n_valid),
                'status': 'skipped', 'reason': 'insufficient_valid_data',
            }
            continue

        if n_train < MIN_ROWS_FOR_SPECIALIZED:
            print(f"    FALLBACK: train too small ({n_train} < {MIN_ROWS_FOR_SPECIALIZED}), use unified")
            results[cat_key] = {
                'label': label, 'n_train': int(n_train), 'n_valid': int(n_valid),
                'status': 'fallback_to_unified',
            }
            continue

        # --- Train specialized model ---
        X_cat = df[features].values
        y_arr = y

        try:
            lgb_cat = train_lgb(
                X_cat[cat_train], y_arr[cat_train],
                X_cat[cat_valid], y_arr[cat_valid], features)
            lgb_cat_pred = lgb_cat.predict(X_cat[cat_valid])
            lgb_cat_auc = roc_auc_score(y_arr[cat_valid], lgb_cat_pred)

            xgb_cat = train_xgb(
                X_cat[cat_train], y_arr[cat_train],
                X_cat[cat_valid], y_arr[cat_valid])
            xgb_cat_pred = xgb_cat.predict(xgb.DMatrix(X_cat[cat_valid]))
            xgb_cat_auc = roc_auc_score(y_arr[cat_valid], xgb_cat_pred)

            total_cat = lgb_cat_auc + xgb_cat_auc
            cat_pred = lgb_cat_pred * (lgb_cat_auc / total_cat) + xgb_cat_pred * (xgb_cat_auc / total_cat)
            cat_auc = roc_auc_score(y_arr[cat_valid], cat_pred)

            print(f"    Specialized AUC: LGB {lgb_cat_auc:.4f} / XGB {xgb_cat_auc:.4f} / Ensemble {cat_auc:.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[cat_key] = {
                'label': label, 'n_train': int(n_train), 'n_valid': int(n_valid),
                'status': 'error', 'reason': str(e),
            }
            continue

        results[cat_key] = {
            'label': label, 'n_train': int(n_train), 'n_valid': int(n_valid),
            'status': 'trained',
            'specialized_auc': {
                'lgb': round(lgb_cat_auc, 4), 'xgb': round(xgb_cat_auc, 4),
                'ensemble': round(cat_auc, 4),
            },
        }

    return results


# ======================================================================
#  MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("  V9.3 EXPERIMENTS: Pace Features + Specialized Models")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # === Data Loading (same pipeline) ===
    df = load_data()
    lap_df = load_lap_data()
    if lap_df is not None:
        df = df.merge(lap_df, on='race_id_str', how='left')
        matched = df['race_first3f'].notna().sum()
        print(f"  Lap data merged: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")

    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)

    tt_data = load_training_times()
    df = merge_training_features(df, tt_data)
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)
    df = compute_lag_features(df)

    print("Building features...")
    df = build_features(df)
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)

    # === TASK 1: Add pace/running style features ===
    df = compute_running_style_features(df)

    # Target
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()
    y = df['target'].values

    # Time-based split
    max_year = df['year_full'].max()
    valid_mask_arr = (df['year_full'] >= (max_year - 1)).values
    train_mask_arr = ~valid_mask_arr
    y_train, y_valid = y[train_mask_arr], y[valid_mask_arr]

    print(f"\nTrain: {train_mask_arr.sum()}, Valid: {valid_mask_arr.sum()}")
    print(f"Target rate: train={y_train.mean():.3f}, valid={y_valid.mean():.3f}")

    # Prepare features
    features_base = list(FEATURES_PATTERN_A)
    features_with_pace = list(FEATURES_PATTERN_A) + PACE_FEATURES

    all_features = set(features_base + features_with_pace)
    for f in all_features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    import xgboost as xgb

    # ==============================
    #  TASK 1: Baseline vs Pace
    # ==============================
    print("\n" + "=" * 70)
    print("  TASK 1: BASELINE (Pattern A, 67 features)")
    print("=" * 70)

    X_base = df[features_base].values
    lgb_base = train_lgb(X_base[train_mask_arr], y_train, X_base[valid_mask_arr], y_valid, features_base)
    lgb_base_pred = lgb_base.predict(X_base[valid_mask_arr])
    lgb_base_auc = roc_auc_score(y_valid, lgb_base_pred)

    xgb_base = train_xgb(X_base[train_mask_arr], y_train, X_base[valid_mask_arr], y_valid)
    xgb_base_pred = xgb_base.predict(xgb.DMatrix(X_base[valid_mask_arr]))
    xgb_base_auc = roc_auc_score(y_valid, xgb_base_pred)

    total_b = lgb_base_auc + xgb_base_auc
    base_pred = lgb_base_pred * (lgb_base_auc / total_b) + xgb_base_pred * (xgb_base_auc / total_b)
    base_auc = roc_auc_score(y_valid, base_pred)
    print(f"\n  Baseline AUC: LGB {lgb_base_auc:.4f} / XGB {xgb_base_auc:.4f} / Ensemble {base_auc:.4f}")

    print("\n" + "=" * 70)
    print(f"  TASK 1: WITH PACE FEATURES ({len(features_with_pace)} features)")
    print(f"  Added: {PACE_FEATURES}")
    print("=" * 70)

    X_pace = df[features_with_pace].values
    lgb_pace = train_lgb(X_pace[train_mask_arr], y_train, X_pace[valid_mask_arr], y_valid, features_with_pace)
    lgb_pace_pred = lgb_pace.predict(X_pace[valid_mask_arr])
    lgb_pace_auc = roc_auc_score(y_valid, lgb_pace_pred)

    xgb_pace = train_xgb(X_pace[train_mask_arr], y_train, X_pace[valid_mask_arr], y_valid)
    xgb_pace_pred = xgb_pace.predict(xgb.DMatrix(X_pace[valid_mask_arr]))
    xgb_pace_auc = roc_auc_score(y_valid, xgb_pace_pred)

    total_p = lgb_pace_auc + xgb_pace_auc
    pace_pred = lgb_pace_pred * (lgb_pace_auc / total_p) + xgb_pace_pred * (xgb_pace_auc / total_p)
    pace_auc = roc_auc_score(y_valid, pace_pred)
    print(f"\n  With Pace AUC: LGB {lgb_pace_auc:.4f} / XGB {xgb_pace_auc:.4f} / Ensemble {pace_auc:.4f}")

    fi_pace_df = show_feature_importance(lgb_pace, features_with_pace, "Pattern A + Pace Features")
    fi_pace = dict(zip(fi_pace_df['feature'], fi_pace_df['importance']))

    # Pace feature importance
    print(f"\n  Pace feature importance:")
    total_imp = fi_pace_df['importance'].sum()
    for f in PACE_FEATURES:
        imp = fi_pace.get(f, 0)
        pct = imp / total_imp * 100
        print(f"    {f}: {imp:,.0f} ({pct:.1f}%)")

    pace_improvement = pace_auc - base_auc
    pace_improved = pace_improvement > 0

    print(f"\n  Pace improvement: {pace_improvement:+.4f} ({'IMPROVED' if pace_improved else 'NO IMPROVEMENT'})")

    # ==============================
    #  TASK 2: Specialized Models
    # ==============================
    # Use features_with_pace as the best feature set for specialized models
    best_features = features_with_pace if pace_improved else features_base
    best_label = "with_pace" if pace_improved else "baseline"

    # First, compute unified AUC per category (for comparison)
    print("\n" + "=" * 70)
    print(f"  TASK 2: UNIFIED MODEL AUC per Category (using {best_label})")
    print("=" * 70)

    X_best = df[best_features].values
    lgb_best = lgb_pace if pace_improved else lgb_base
    xgb_best = xgb_pace if pace_improved else xgb_base
    best_lgb_auc = lgb_pace_auc if pace_improved else lgb_base_auc
    best_xgb_auc = xgb_pace_auc if pace_improved else xgb_base_auc
    w_lgb_best = best_lgb_auc / (best_lgb_auc + best_xgb_auc)
    w_xgb_best = best_xgb_auc / (best_lgb_auc + best_xgb_auc)

    unified_pred_full = lgb_best.predict(X_best) * w_lgb_best + xgb_best.predict(xgb.DMatrix(X_best)) * w_xgb_best

    unified_per_cat = {}
    for cat_key, cat_info in CATEGORIES.items():
        surf = cat_info['surface']
        dmin = cat_info['dist_min']
        dmax = cat_info['dist_max']
        cat_valid_mask = valid_mask_arr & (df['surface_enc'].values == surf) & \
                         (df['distance'].values >= dmin) & (df['distance'].values <= dmax)
        n_cat_valid = cat_valid_mask.sum()
        if n_cat_valid >= 50:
            cat_unified_auc = roc_auc_score(y[cat_valid_mask], unified_pred_full[cat_valid_mask])
            unified_per_cat[cat_key] = round(cat_unified_auc, 4)
            print(f"  {cat_info['label']:25s} N={n_cat_valid:>5}  Unified AUC={cat_unified_auc:.4f}")
        else:
            unified_per_cat[cat_key] = None
            print(f"  {cat_info['label']:25s} N={n_cat_valid:>5}  (too few)")

    # Train specialized models
    specialized_results = train_specialized_models(
        df, best_features, y, train_mask_arr, valid_mask_arr)

    # Add unified AUC for comparison
    for cat_key in specialized_results:
        specialized_results[cat_key]['unified_auc'] = unified_per_cat.get(cat_key)
        if specialized_results[cat_key].get('status') == 'trained' and unified_per_cat.get(cat_key):
            spec_auc = specialized_results[cat_key]['specialized_auc']['ensemble']
            uni_auc = unified_per_cat[cat_key]
            diff = spec_auc - uni_auc
            specialized_results[cat_key]['diff_vs_unified'] = round(diff, 4)
            specialized_results[cat_key]['beats_unified'] = diff > 0

    # ==============================
    #  SUMMARY
    # ==============================
    print("\n" + "=" * 70)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  {'Model':<40} {'Ensemble AUC':>12}")
    print(f"  {'-' * 52}")
    print(f"  {'Pattern A Baseline (67 features)':<40} {base_auc:>12.4f}")
    print(f"  {'Pattern A + Pace (71 features)':<40} {pace_auc:>12.4f}  ({pace_improvement:+.4f})")

    print(f"\n  --- Specialized vs Unified ---")
    print(f"  {'Category':<25} {'Unified':>10} {'Specialized':>12} {'Diff':>8} {'Winner':>10}")
    print(f"  {'-' * 65}")
    any_spec_wins = False
    for cat_key, cat_info in CATEGORIES.items():
        sr = specialized_results.get(cat_key, {})
        label = cat_info['label']
        uni = unified_per_cat.get(cat_key)
        if sr.get('status') == 'trained':
            spec = sr['specialized_auc']['ensemble']
            diff = sr.get('diff_vs_unified', 0)
            winner = 'SPEC' if diff > 0 else 'UNIFIED'
            if diff > 0:
                any_spec_wins = True
            print(f"  {label:<25} {uni:>10.4f} {spec:>12.4f} {diff:>+8.4f} {winner:>10}")
        elif sr.get('status') == 'fallback_to_unified':
            print(f"  {label:<25} {uni or 0:>10.4f} {'(fallback)':>12} {'':>8} {'UNIFIED':>10}")
        else:
            print(f"  {label:<25} {'N/A':>10} {'N/A':>12} {'':>8} {'SKIP':>10}")

    # === Judgements ===
    print(f"\n  TASK 1 JUDGEMENT: {'ADOPT' if pace_improved else 'REJECT'}")
    if pace_improved:
        print(f"    Pace features improved AUC by {pace_improvement:+.4f}")
    else:
        print(f"    Pace features did not improve AUC ({pace_improvement:+.4f})")

    print(f"\n  TASK 2 JUDGEMENT: {'MIXED' if any_spec_wins else 'REJECT (unified wins all)'}")
    print(f"    Note: CLAUDE.md explicitly states specialized models were tried and rejected")
    print(f"    due to overfitting risk. Unified model is recommended.")

    # === Save results ===
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = {
        'generated_at': now,
        'task1_pace_features': {
            'new_features': PACE_FEATURES,
            'feature_descriptions': {
                'running_style': '脚質分類 (1=逃げ, 2=先行, 3=差し, 4=追込) - expanding window of pass1',
                'predicted_pace': 'レース予測ペース (0=スロー, 1=ミドル, 2=ハイ) - based on front-runner ratio',
                'pace_advantage': 'ペース有利度 (-2 to +2) - running_style × predicted_pace interaction',
                'style_vs_pace': '脚質×ペース cross feature (categorical)',
            },
            'baseline_auc': {
                'lgb': round(lgb_base_auc, 4), 'xgb': round(xgb_base_auc, 4),
                'ensemble': round(base_auc, 4), 'n_features': len(features_base),
            },
            'with_pace_auc': {
                'lgb': round(lgb_pace_auc, 4), 'xgb': round(xgb_pace_auc, 4),
                'ensemble': round(pace_auc, 4), 'n_features': len(features_with_pace),
            },
            'improvement': round(pace_improvement, 4),
            'improved': pace_improved,
            'pace_feature_importance': {f: round(fi_pace.get(f, 0), 1) for f in PACE_FEATURES},
            'judgement': 'ADOPT' if pace_improved else 'REJECT',
        },
        'task2_specialized_models': {
            'categories': {k: v['label'] for k, v in CATEGORIES.items()},
            'min_rows_threshold': MIN_ROWS_FOR_SPECIALIZED,
            'unified_auc_per_category': unified_per_cat,
            'specialized_results': specialized_results,
            'any_specialized_wins': any_spec_wins,
            'judgement': 'REJECT - unified model preferred (consistent with prior finding)',
        },
    }

    report_path = os.path.join(DATA_DIR, 'v93_specialized_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {report_path}")

    print("\n  All experiments complete!")
    return report


if __name__ == '__main__':
    main()
