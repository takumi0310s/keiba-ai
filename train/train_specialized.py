#!/usr/bin/env python
"""KEIBA AI Specialized Models Training
Trains distance-specialized models and evaluates vs global model.
Pattern A leak-free. Walk-forward validated.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_training_times,
    merge_training_features, compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance, load_lap_data,
    compute_lag_features, build_features,
    compute_distance_aptitude, compute_frame_advantage,
    COURSE_MAP, N_TOP_SIRE,
    train_lgb, train_xgb, show_feature_importance,
)
from train_v92_leakfree import (
    LEAK_FEATURES_A, classify_condition, calc_trio_bets,
)
from train_v93_leakfree import FEATURES_V93_PATTERN_A

from routing_model import RoutingModel

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
OUTPUT_DIR = BASE_DIR

# Distance groups
DIST_GROUPS = {
    'sprint': (0, 1400, 'sprint <=1400m'),
    'middle': (1401, 2000, 'middle 1600-2000m'),
    'long': (2001, 9999, 'long 2200m+'),
}

INV_COURSE = {v: k for k, v in COURSE_MAP.items()}


def prepare_data():
    """Load and prepare full dataset."""
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
    df = build_features(df)
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)

    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()

    features = list(FEATURES_V93_PATTERN_A)
    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    return df, features, sire_map, bms_map


def get_dist_group(distance):
    for name, (lo, hi, _) in DIST_GROUPS.items():
        if lo <= distance <= hi:
            return name
    return 'middle'


def train_specialized_lgb(X_train, y_train, X_valid, y_valid, feature_names, group_name):
    """Train LGB with group-optimized hyperparameters."""
    # Different hyperparams for different distance groups
    if group_name == 'sprint':
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'num_leaves': 47, 'learning_rate': 0.05, 'feature_fraction': 0.75,
            'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 40,
            'reg_alpha': 0.2, 'reg_lambda': 0.2, 'verbose': -1,
            'n_jobs': -1, 'seed': 42,
        }
    elif group_name == 'long':
        # Less data, more regularization
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'num_leaves': 31, 'learning_rate': 0.04, 'feature_fraction': 0.7,
            'bagging_fraction': 0.75, 'bagging_freq': 5, 'min_child_samples': 60,
            'reg_alpha': 0.3, 'reg_lambda': 0.3, 'verbose': -1,
            'n_jobs': -1, 'seed': 42,
        }
    else:  # middle
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
            'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
            'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1,
            'n_jobs': -1, 'seed': 42,
        }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    dvalid = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_names, reference=dtrain)
    model = lgb.train(
        params, dtrain, num_boost_round=1500,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
    )
    return model


def walk_forward_evaluate(df, features, models_dict, dist_feature_idx, n_folds=3):
    """Walk-forward evaluation: train on years [start..Y-2], validate on [Y-1..Y]."""
    years = sorted(df['year_full'].unique())
    min_year = min(years)
    max_year = max(years)

    # Create fold boundaries: last n_folds 2-year windows
    fold_starts = []
    for i in range(n_folds):
        val_end = max_year - i * 2
        val_start = val_end - 1
        if val_start <= min_year + 4:  # Need at least 5 years of training
            break
        fold_starts.append((val_start, val_end))
    fold_starts.reverse()

    print(f"\n  Walk-forward: {len(fold_starts)} folds")
    wf_results = []

    for val_start, val_end in fold_starts:
        train_mask = df['year_full'] < val_start
        valid_mask = (df['year_full'] >= val_start) & (df['year_full'] <= val_end)

        if train_mask.sum() < 10000 or valid_mask.sum() < 1000:
            continue

        y_train = df.loc[train_mask, 'target'].values
        y_valid = df.loc[valid_mask, 'target'].values

        # Global model
        X_train = df.loc[train_mask, features].values
        X_valid = df.loc[valid_mask, features].values
        lgb_global = train_lgb(X_train, y_train, X_valid, y_valid, features)
        global_pred = lgb_global.predict(X_valid)
        global_auc = roc_auc_score(y_valid, global_pred)

        # Specialized models
        spec_models = {}
        spec_preds = np.zeros(valid_mask.sum())
        distances_valid = df.loc[valid_mask, 'distance'].values

        for gname, (lo, hi, desc) in DIST_GROUPS.items():
            g_train = train_mask & (df['distance'] >= lo) & (df['distance'] <= hi)
            g_valid = valid_mask & (df['distance'] >= lo) & (df['distance'] <= hi)

            if g_train.sum() < 3000 or g_valid.sum() < 300:
                spec_models[gname] = lgb_global  # fallback
                continue

            X_gt = df.loc[g_train, features].values
            y_gt = df.loc[g_train, 'target'].values
            X_gv = df.loc[g_valid, features].values
            y_gv = df.loc[g_valid, 'target'].values

            spec_lgb = train_specialized_lgb(X_gt, y_gt, X_gv, y_gv, features, gname)
            spec_models[gname] = spec_lgb

            # Get prediction indices in validation set
            g_valid_local = (distances_valid >= lo) & (distances_valid <= hi)
            spec_preds[g_valid_local] = spec_lgb.predict(X_valid[g_valid_local])

        # Fill any gaps with global
        unfilled = spec_preds == 0
        if unfilled.any():
            spec_preds[unfilled] = lgb_global.predict(X_valid[unfilled])

        routing_auc = roc_auc_score(y_valid, spec_preds)

        # Per-group AUC comparison
        group_aucs = {}
        for gname, (lo, hi, desc) in DIST_GROUPS.items():
            g_mask = (distances_valid >= lo) & (distances_valid <= hi)
            if g_mask.sum() < 100:
                continue
            y_g = y_valid[g_mask]
            if len(set(y_g)) < 2:
                continue
            gauc_global = roc_auc_score(y_g, global_pred[g_mask])
            gauc_spec = roc_auc_score(y_g, spec_preds[g_mask])
            group_aucs[gname] = {
                'n': int(g_mask.sum()),
                'global': round(gauc_global, 4),
                'specialized': round(gauc_spec, 4),
                'delta': round(gauc_spec - gauc_global, 4),
            }

        fold_result = {
            'val_years': f"{val_start}-{val_end}",
            'n_train': int(train_mask.sum()),
            'n_valid': int(valid_mask.sum()),
            'global_auc': round(global_auc, 4),
            'routing_auc': round(routing_auc, 4),
            'delta': round(routing_auc - global_auc, 4),
            'group_aucs': group_aucs,
        }
        wf_results.append(fold_result)

        print(f"  Fold {val_start}-{val_end}: Global {global_auc:.4f} / Routing {routing_auc:.4f} (delta {routing_auc - global_auc:+.4f})")
        for gname, ga in group_aucs.items():
            marker = "+" if ga['delta'] > 0 else ""
            print(f"    {gname:10s}: Global {ga['global']:.4f} / Spec {ga['specialized']:.4f} ({marker}{ga['delta']:.4f}) N={ga['n']}")

    return wf_results


def backtest_condition(df, routing_model, features, label=""):
    """Run condition-based backtest."""
    print(f"\n  --- {label} Condition Backtest ---")
    X = df[features].values
    scores = routing_model.predict(X)
    df = df.copy()
    df['score'] = scores

    condition_results = {}
    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        condition_results[cond] = {'n': 0, 'trio_hits': 0}

    for rid, race_df in df.groupby('race_id_str'):
        race_df = race_df.sort_values('score', ascending=False)
        if len(race_df) < 3:
            continue
        num_horses = int(race_df['num_horses_val'].iloc[0])
        distance = int(race_df['distance'].iloc[0])
        cond_enc = int(race_df['condition_enc'].iloc[0]) if 'condition_enc' in race_df.columns else 0
        cond_key = classify_condition(num_horses, distance, cond_enc)

        ranking = race_df['umaban'].astype(int).tolist()
        actual_top3 = set(race_df[race_df['finish'] <= 3]['umaban'].astype(int).tolist())
        trio_bets = calc_trio_bets(ranking)
        trio_hit = any(set(combo) == actual_top3 for combo in trio_bets) if len(actual_top3) == 3 else False

        cr = condition_results[cond_key]
        cr['n'] += 1
        if trio_hit:
            cr['trio_hits'] += 1

    print(f"  {'COND':<4} {'N':>5} {'TRIO_HIT':>8} {'HIT_RATE':>8}")
    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        cr = condition_results[cond]
        n = cr['n']
        if n == 0:
            print(f"  {cond:<4} {'N/A':>5}")
            continue
        hit_rate = cr['trio_hits'] / n * 100
        print(f"  {cond:<4} {n:>5} {cr['trio_hits']:>8} {hit_rate:>7.1f}%")
    return condition_results


def main():
    print("=" * 70)
    print("  KEIBA AI SPECIALIZED MODEL TRAINING")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    df, features, sire_map, bms_map = prepare_data()
    dist_feature_idx = features.index('distance')
    print(f"  Distance feature index: {dist_feature_idx}")

    max_year = df['year_full'].max()
    valid_mask = df['year_full'] >= (max_year - 1)
    train_mask = ~valid_mask
    y = df['target'].values

    # ================================================================
    # 1. GLOBAL BASELINE (V9.3 Pattern A)
    # ================================================================
    print("\n" + "=" * 70)
    print("  GLOBAL BASELINE (V9.3 Pattern A)")
    print("=" * 70)
    X = df[features].values
    lgb_global = train_lgb(X[train_mask], y[train_mask], X[valid_mask], y[valid_mask], features)
    global_pred = lgb_global.predict(X[valid_mask])
    global_auc = roc_auc_score(y[valid_mask], global_pred)
    print(f"\n  Global LGB AUC: {global_auc:.4f}")

    # ================================================================
    # 2. DISTANCE-SPECIALIZED MODELS
    # ================================================================
    print("\n" + "=" * 70)
    print("  DISTANCE-SPECIALIZED MODELS")
    print("=" * 70)

    spec_models = {}
    spec_results = {}
    distances_valid = df.loc[valid_mask, 'distance'].values

    for gname, (lo, hi, desc) in DIST_GROUPS.items():
        print(f"\n  --- {gname}: {desc} ---")
        g_train = train_mask & (df['distance'] >= lo) & (df['distance'] <= hi)
        g_valid = valid_mask & (df['distance'] >= lo) & (df['distance'] <= hi)

        n_train = g_train.sum()
        n_valid = g_valid.sum()
        print(f"  Train: {n_train}, Valid: {n_valid}")

        if n_train < 5000 or n_valid < 500:
            print(f"  Insufficient data, using global model")
            spec_models[gname] = lgb_global
            spec_results[gname] = {'used_global': True}
            continue

        X_gt = df.loc[g_train, features].values
        y_gt = df.loc[g_train, 'target'].values
        X_gv = df.loc[g_valid, features].values
        y_gv = df.loc[g_valid, 'target'].values

        spec_lgb = train_specialized_lgb(X_gt, y_gt, X_gv, y_gv, features, gname)
        spec_pred = spec_lgb.predict(X_gv)
        spec_auc = roc_auc_score(y_gv, spec_pred)

        # Compare to global on same segment
        g_valid_mask = (distances_valid >= lo) & (distances_valid <= hi)
        global_on_segment = global_pred[g_valid_mask]
        global_seg_auc = roc_auc_score(y_gv, global_on_segment)

        delta = spec_auc - global_seg_auc
        better = delta > 0

        print(f"  Global on {gname}: {global_seg_auc:.4f}")
        print(f"  Specialized {gname}: {spec_auc:.4f} (delta {delta:+.4f}) {'BETTER' if better else 'WORSE'}")

        spec_models[gname] = spec_lgb if better else lgb_global
        spec_results[gname] = {
            'n_train': n_train, 'n_valid': n_valid,
            'global_auc': round(global_seg_auc, 4),
            'spec_auc': round(spec_auc, 4),
            'delta': round(delta, 4),
            'adopted': better,
            'used_global': not better,
        }

        if better:
            fi = show_feature_importance(spec_lgb, features, f"Specialized {gname}")

    # ================================================================
    # 3. ROUTING MODEL EVALUATION
    # ================================================================
    print("\n" + "=" * 70)
    print("  ROUTING MODEL EVALUATION")
    print("=" * 70)

    routing = RoutingModel(spec_models, dist_feature_idx, fallback_model=lgb_global)
    routing_pred = routing.predict(X[valid_mask])
    routing_auc = roc_auc_score(y[valid_mask], routing_pred)

    print(f"\n  Global AUC:  {global_auc:.4f}")
    print(f"  Routing AUC: {routing_auc:.4f} (delta {routing_auc - global_auc:+.4f})")

    # Per-condition comparison
    valid_df = df[valid_mask].copy()
    valid_df['global_pred'] = global_pred
    valid_df['routing_pred'] = routing_pred

    # ================================================================
    # 4. WALK-FORWARD VALIDATION
    # ================================================================
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION")
    print("=" * 70)
    wf_results = walk_forward_evaluate(df, features, spec_models, dist_feature_idx, n_folds=3)

    # Average WF improvement
    if wf_results:
        avg_delta = np.mean([r['delta'] for r in wf_results])
        all_positive = all(r['delta'] >= 0 for r in wf_results)
        print(f"\n  WF Average Delta: {avg_delta:+.4f}")
        print(f"  All folds positive: {all_positive}")
    else:
        avg_delta = 0
        all_positive = False

    # ================================================================
    # 5. CONDITION BACKTEST
    # ================================================================
    print("\n" + "=" * 70)
    print("  CONDITION BACKTEST")
    print("=" * 70)
    bt_routing = backtest_condition(valid_df, routing, features, "Routing Model")

    # ================================================================
    # 6. DECISION & SAVE
    # ================================================================
    print("\n" + "=" * 70)
    print("  FINAL DECISION")
    print("=" * 70)

    improved = routing_auc > global_auc and avg_delta > 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Also train XGB for routing (use global XGB as we can't route XGB easily)
    import xgboost as xgb_lib
    xgb_global = train_xgb(X[train_mask], y[train_mask], X[valid_mask], y[valid_mask])
    xgb_pred = xgb_global.predict(xgb_lib.DMatrix(X[valid_mask]))
    xgb_auc = roc_auc_score(y[valid_mask], xgb_pred)
    print(f"  XGB Global AUC: {xgb_auc:.4f}")

    # Ensemble: routing LGB + global XGB
    total = routing_auc + xgb_auc
    w_lgb = routing_auc / total
    w_xgb = xgb_auc / total
    ens_pred = routing_pred * w_lgb + xgb_pred * w_xgb
    ens_auc = roc_auc_score(y[valid_mask], ens_pred)
    print(f"  Ensemble (Routing LGB + Global XGB): {ens_auc:.4f}")

    # Compare to previous production (V9.3)
    prev_pkl_path = os.path.join(OUTPUT_DIR, 'keiba_model_v9_central.pkl')
    with open(prev_pkl_path, 'rb') as f:
        prev_pkl = pickle.load(f)
    prev_ens_auc = prev_pkl.get('ensemble_auc', 0)
    print(f"\n  Previous production AUC: {prev_ens_auc:.4f}")
    print(f"  New routing ensemble AUC: {ens_auc:.4f}")
    print(f"  Delta: {ens_auc - prev_ens_auc:+.4f}")

    overall_improved = ens_auc > prev_ens_auc

    features_pkl = [f if f != 'num_horses_val' else 'num_horses' for f in features]

    if overall_improved:
        print(f"\n  *** ROUTING MODEL IMPROVED! Updating production. ***")

        pkl = {
            'model': routing,  # RoutingModel with .predict()
            'features': features_pkl,
            'version': 'v9.3s_specialized',
            'auc': routing_auc,
            'ensemble_auc': ens_auc,
            'leak_free': True,
            'leak_pattern': 'A',
            'leak_removed': sorted(LEAK_FEATURES_A),
            'sire_map': sire_map,
            'bms_map': bms_map,
            'n_top_encode': N_TOP_SIRE,
            'trained_at': now,
            'n_train': int(train_mask.sum()),
            'n_valid': int(valid_mask.sum()),
            'model_type': 'central',
            'xgb_model': xgb_global,
            'mlp_model': None,
            'mlp_scaler': None,
            'ensemble_weights': {'lgb': w_lgb, 'xgb': w_xgb, 'mlp': 0},
            'course_map': dict(COURSE_MAP),
            'condition_backtest': bt_routing,
            'specialized_results': spec_results,
            'walk_forward_results': wf_results,
            'global_baseline_auc': global_auc,
        }

        with open(prev_pkl_path, 'wb') as f:
            pickle.dump(pkl, f)
        print(f"  Saved: {prev_pkl_path}")

        v8_path = os.path.join(OUTPUT_DIR, 'keiba_model_v8.pkl')
        v8_pkl = dict(pkl)
        v8_pkl['auc'] = ens_auc
        with open(v8_path, 'wb') as f:
            pickle.dump(v8_pkl, f)
        print(f"  Saved backup: {v8_path}")
    else:
        print(f"\n  Routing model did NOT improve overall. Keeping V9.3.")

    # Save detailed results
    results = {
        'generated_at': now,
        'global_auc': global_auc,
        'routing_lgb_auc': routing_auc,
        'xgb_auc': xgb_auc,
        'ensemble_auc': ens_auc,
        'prev_production_auc': prev_ens_auc,
        'improved': overall_improved,
        'delta': ens_auc - prev_ens_auc,
        'specialized': spec_results,
        'walk_forward': wf_results,
        'wf_avg_delta': avg_delta if wf_results else 0,
        'condition_backtest': {k: {'n': v['n'], 'trio_hits': v['trio_hits'],
                                    'hit_rate': v['trio_hits']/v['n']*100 if v['n'] > 0 else 0}
                               for k, v in bt_routing.items()},
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    results_path = os.path.join(OUTPUT_DIR, 'specialized_training_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NpEncoder)
    print(f"  Saved: {results_path}")

    print("\n  Specialized model training complete!")
    return results


if __name__ == '__main__':
    main()
