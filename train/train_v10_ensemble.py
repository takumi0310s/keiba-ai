#!/usr/bin/env python
"""KEIBA AI v10 Ensemble Training (Central + NAR)
- FIX: training_times.csv horse_id 10-digit → 8-digit conversion
- Ensemble: LightGBM + XGBoost + MLP (3-model weighted average + stacking)
- Pattern A strict leak-free
- Walk-forward validation (train on ~(Y-1), test on Y)
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
import warnings
import time
warnings.filterwarnings('ignore')

from datetime import datetime
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb_lib

# Reuse from v92
sys.path.insert(0, os.path.dirname(__file__))
from train_v92_central import (
    load_data, encode_categoricals, encode_sires, load_lap_data,
    compute_jockey_wr, compute_trainer_stats,
    compute_horse_career, compute_sire_performance,
    compute_distance_aptitude, compute_frame_advantage,
    compute_lag_features, build_features,
    COURSE_MAP, N_TOP_SIRE, FEATURES_V92, FEATURES_V92_PKL,
    FEATURES_V93, FEATURES_V93_PKL,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = BASE_DIR
TRAINING_TIMES_PATH = os.path.join(DATA_DIR, 'training_times.csv')

# Pattern A leak features to remove
LEAK_FEATURES_A = {
    'odds_log', 'horse_weight', 'condition_enc',
    'weight_change', 'weight_change_abs',
    'weight_cat', 'weight_cat_dist', 'cond_surface',
}

# Pattern A features (V9.3 base, leak removed)
FEATURES_V10 = [f for f in FEATURES_V93 if f not in LEAK_FEATURES_A]
FEATURES_V10_PKL = [f if f != 'num_horses_val' else 'num_horses' for f in FEATURES_V10]


def load_training_times_fixed():
    """Load training times with horse_id 10→8 digit fix."""
    if not os.path.exists(TRAINING_TIMES_PATH):
        print("  WARNING: training_times.csv not found")
        return None

    print("Loading training times (with horse_id fix)...")
    tt = pd.read_csv(TRAINING_TIMES_PATH, encoding='utf-8-sig', dtype=str)
    print(f"  Raw: {len(tt)} training records")

    tt['time_4f'] = pd.to_numeric(tt['time_4f'], errors='coerce')
    tt['time_3f'] = pd.to_numeric(tt['time_3f'], errors='coerce')
    tt['date'] = pd.to_numeric(tt['date'], errors='coerce').fillna(0).astype(int)

    tt = tt[tt['time_4f'].notna() & (tt['time_4f'] > 30) & (tt['time_4f'] < 80)]

    # FIX: Convert horse_id from 10-digit (YYYYIIIIII) to 8-digit (YYIIIIII)
    has_hid = tt['horse_id'].notna() & (tt['horse_id'].astype(str).str.strip() != '')
    tt_with_hid = tt[has_hid].copy()
    tt_no_hid = tt[~has_hid].copy()

    original_count = len(tt_with_hid)
    tt_with_hid['horse_id_raw'] = tt_with_hid['horse_id'].astype(str).str.strip()
    # 10-digit → 8-digit: drop first 2 chars of year (2019104442 → 19104442)
    mask_10 = tt_with_hid['horse_id_raw'].str.len() == 10
    tt_with_hid.loc[mask_10, 'horse_id'] = tt_with_hid.loc[mask_10, 'horse_id_raw'].str[2:]
    tt_with_hid.loc[~mask_10, 'horse_id'] = tt_with_hid.loc[~mask_10, 'horse_id_raw']
    converted = mask_10.sum()
    print(f"  Horse_id fix: {converted}/{original_count} converted (10→8 digit)")

    tt = pd.concat([tt_with_hid, tt_no_hid], ignore_index=True)

    wood = tt[tt['training_type'] == 'wood'].copy()
    sakaro = tt[tt['training_type'] == 'sakaro'].copy()
    print(f"  Wood: {len(wood)} valid, Sakaro: {len(sakaro)} valid")

    return {'wood': wood, 'sakaro': sakaro}


def _agg_training_by_period(tt_df, key_col):
    """Aggregate training data into year-month periods for efficient merge.
    Returns DataFrame with key, year_month, best_4f, best_3f, count."""
    if len(tt_df) == 0 or key_col not in tt_df.columns:
        return pd.DataFrame()
    clean = tt_df[tt_df[key_col].notna()].copy()
    clean[key_col] = clean[key_col].astype(str).str.strip()
    clean = clean[clean[key_col] != '']
    if len(clean) == 0:
        return pd.DataFrame()

    clean['date_int'] = clean['date'].astype(int)
    # Create year_month bucket: YYYYMM
    clean['ym'] = clean['date_int'] // 100

    # Also create next-month bucket (training in late month serves next month's races)
    agg = clean.groupby([key_col, 'ym']).agg(
        best_4f=('time_4f', 'min'),
        best_3f=('time_3f', 'min'),
        count=('time_4f', 'count'),
    ).reset_index()
    agg.columns = [key_col, 'ym', 'best_4f', 'best_3f', 'count']
    return agg


def merge_training_features_fixed(df, tt_data):
    """Merge training features with fixed horse_id matching.
    Uses year-month aggregation for speed (approximate 14-day window)."""
    if tt_data is None:
        for col in ['wood_best_4f', 'wood_count_2w', 'wood_best_4f_filled',
                     'has_wood_training', 'sakaro_best_4f', 'sakaro_best_3f',
                     'sakaro_count_2w', 'sakaro_best_4f_filled', 'sakaro_best_3f_filled',
                     'has_sakaro_training', 'total_training_count']:
            df[col] = 0
        return df

    print("Merging training features (fixed horse_id)...", flush=True)
    wood = tt_data['wood']
    sakaro = tt_data['sakaro']

    # Compute year_month for races
    df['_ym'] = df['date_num'] // 100

    # === Wood: join by horse_id ===
    t0 = time.time()
    wood_agg = _agg_training_by_period(wood, 'horse_id')
    if len(wood_agg) > 0:
        # Merge on horse_id + same year_month (approximate 14-day window)
        wood_agg = wood_agg.rename(columns={'horse_id': '_wkey', 'ym': '_ym',
                                             'best_4f': 'wood_best_4f',
                                             'best_3f': '_wood_3f',
                                             'count': 'wood_count_2w'})
        df['_wkey'] = df['horse_id'].astype(str).str.strip()
        df = df.merge(wood_agg[['_wkey', '_ym', 'wood_best_4f', 'wood_count_2w']],
                      on=['_wkey', '_ym'], how='left')
        # Also try previous month for early-month races
        wood_agg2 = wood_agg.copy()
        wood_agg2['_ym'] = wood_agg2['_ym'] + 1  # shift to next month
        wood_agg2 = wood_agg2.rename(columns={'wood_best_4f': '_w4f_prev', 'wood_count_2w': '_wc_prev'})
        df = df.merge(wood_agg2[['_wkey', '_ym', '_w4f_prev', '_wc_prev']],
                      on=['_wkey', '_ym'], how='left')
        # Combine: use current month if available, else previous month
        df['wood_best_4f'] = df['wood_best_4f'].fillna(df['_w4f_prev'])
        df['wood_count_2w'] = df['wood_count_2w'].fillna(df['_wc_prev']).fillna(0).astype(int)
        df = df.drop(columns=['_wkey', '_w4f_prev', '_wc_prev'], errors='ignore')
    else:
        df['wood_best_4f'] = np.nan
        df['wood_count_2w'] = 0

    matched = df['wood_best_4f'].notna().sum()
    print(f"  Wood matched: {matched}/{len(df)} ({matched/len(df)*100:.1f}%) ({time.time()-t0:.1f}s)", flush=True)

    wood_mean = df.loc[df['wood_best_4f'].notna(), 'wood_best_4f'].mean()
    df['wood_best_4f_filled'] = df['wood_best_4f'].fillna(wood_mean if not np.isnan(wood_mean) else 52.0)
    df['has_wood_training'] = df['wood_best_4f'].notna().astype(int)

    # === Sakaro: join by horse_name ===
    t0 = time.time()
    sakaro_agg = _agg_training_by_period(sakaro, 'horse_name')
    if len(sakaro_agg) > 0:
        sakaro_agg = sakaro_agg.rename(columns={'horse_name': '_skey', 'ym': '_ym',
                                                 'best_4f': 'sakaro_best_4f',
                                                 'best_3f': 'sakaro_best_3f',
                                                 'count': 'sakaro_count_2w'})
        df['_skey'] = df['horse_name'].astype(str).str.strip()
        df = df.merge(sakaro_agg[['_skey', '_ym', 'sakaro_best_4f', 'sakaro_best_3f', 'sakaro_count_2w']],
                      on=['_skey', '_ym'], how='left')
        # Previous month fallback
        sakaro_agg2 = sakaro_agg.copy()
        sakaro_agg2['_ym'] = sakaro_agg2['_ym'] + 1
        sakaro_agg2 = sakaro_agg2.rename(columns={
            'sakaro_best_4f': '_s4f_prev', 'sakaro_best_3f': '_s3f_prev', 'sakaro_count_2w': '_sc_prev'})
        df = df.merge(sakaro_agg2[['_skey', '_ym', '_s4f_prev', '_s3f_prev', '_sc_prev']],
                      on=['_skey', '_ym'], how='left')
        df['sakaro_best_4f'] = df['sakaro_best_4f'].fillna(df['_s4f_prev'])
        df['sakaro_best_3f'] = df['sakaro_best_3f'].fillna(df['_s3f_prev'])
        df['sakaro_count_2w'] = df['sakaro_count_2w'].fillna(df['_sc_prev']).fillna(0).astype(int)
        df = df.drop(columns=['_skey', '_s4f_prev', '_s3f_prev', '_sc_prev'], errors='ignore')
    else:
        df['sakaro_best_4f'] = np.nan
        df['sakaro_best_3f'] = np.nan
        df['sakaro_count_2w'] = 0

    sak_matched = df['sakaro_best_4f'].notna().sum()
    print(f"  Sakaro matched: {sak_matched}/{len(df)} ({sak_matched/len(df)*100:.1f}%) ({time.time()-t0:.1f}s)", flush=True)

    sak_mean_4f = df.loc[df['sakaro_best_4f'].notna(), 'sakaro_best_4f'].mean()
    sak_mean_3f = df.loc[df['sakaro_best_3f'].notna(), 'sakaro_best_3f'].mean()
    df['sakaro_best_4f_filled'] = df['sakaro_best_4f'].fillna(sak_mean_4f if not np.isnan(sak_mean_4f) else 53.0)
    df['sakaro_best_3f_filled'] = df['sakaro_best_3f'].fillna(sak_mean_3f if not np.isnan(sak_mean_3f) else 39.0)
    df['has_sakaro_training'] = df['sakaro_best_4f'].notna().astype(int)
    df['total_training_count'] = df['wood_count_2w'] + df['sakaro_count_2w']

    df = df.drop(columns=['_ym'], errors='ignore')
    return df


def train_mlp(X_train, y_train, X_valid, y_valid):
    """Train MLP (sklearn MLPClassifier)."""
    from sklearn.neural_network import MLPClassifier

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_valid)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_tr_s, y_train)
    pred = mlp.predict_proba(X_va_s)[:, 1]
    auc = roc_auc_score(y_valid, pred)
    return mlp, scaler, auc


def train_fold(X_train, y_train, X_valid, y_valid, feature_names):
    """Train LGB + XGB + MLP on one fold, return models and predictions."""
    # LightGBM
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
        'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1,
        'n_jobs': -1, 'seed': 42,
    }
    dtrain_lgb = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    dvalid_lgb = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_names, reference=dtrain_lgb)
    lgb_model = lgb.train(
        lgb_params, dtrain_lgb, num_boost_round=1000,
        valid_sets=[dvalid_lgb],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    lgb_pred = lgb_model.predict(X_valid)
    lgb_auc = roc_auc_score(y_valid, lgb_pred)

    # XGBoost
    dtrain_xgb = xgb_lib.DMatrix(X_train, label=y_train)
    dvalid_xgb = xgb_lib.DMatrix(X_valid, label=y_valid)
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'min_child_weight': 50,
        'reg_alpha': 0.1, 'reg_lambda': 0.1, 'seed': 42,
        'tree_method': 'hist', 'verbosity': 0,
    }
    xgb_model = xgb_lib.train(
        xgb_params, dtrain_xgb, num_boost_round=1000,
        evals=[(dvalid_xgb, 'valid')],
        early_stopping_rounds=50, verbose_eval=0,
    )
    xgb_pred = xgb_model.predict(dvalid_xgb)
    xgb_auc = roc_auc_score(y_valid, xgb_pred)

    # MLP
    mlp_model, mlp_scaler, mlp_auc = train_mlp(X_train, y_train, X_valid, y_valid)
    mlp_pred = mlp_model.predict_proba(mlp_scaler.transform(X_valid))[:, 1]

    return {
        'lgb': {'model': lgb_model, 'pred': lgb_pred, 'auc': lgb_auc},
        'xgb': {'model': xgb_model, 'pred': xgb_pred, 'auc': xgb_auc},
        'mlp': {'model': mlp_model, 'scaler': mlp_scaler, 'pred': mlp_pred, 'auc': mlp_auc},
    }


def compute_ensemble(fold_results, y_valid):
    """Compute weighted average and stacking ensemble."""
    lgb_pred = fold_results['lgb']['pred']
    xgb_pred = fold_results['xgb']['pred']
    mlp_pred = fold_results['mlp']['pred']

    lgb_auc = fold_results['lgb']['auc']
    xgb_auc = fold_results['xgb']['auc']
    mlp_auc = fold_results['mlp']['auc']

    # Weighted average (by AUC)
    total_auc = lgb_auc + xgb_auc + mlp_auc
    w_lgb = lgb_auc / total_auc
    w_xgb = xgb_auc / total_auc
    w_mlp = mlp_auc / total_auc

    avg_pred = lgb_pred * w_lgb + xgb_pred * w_xgb + mlp_pred * w_mlp
    avg_auc = roc_auc_score(y_valid, avg_pred)

    # Stacking (LogisticRegression meta-learner)
    meta_X = np.column_stack([lgb_pred, xgb_pred, mlp_pred])
    # Use cross-validation within valid set for stacking
    from sklearn.model_selection import cross_val_predict
    meta_lr = LogisticRegression(C=1.0, random_state=42, max_iter=500)
    try:
        stack_pred = cross_val_predict(meta_lr, meta_X, y_valid, cv=3, method='predict_proba')[:, 1]
        stack_auc = roc_auc_score(y_valid, stack_pred)
    except Exception:
        stack_auc = avg_auc
        stack_pred = avg_pred

    # Final meta-model fit on all data
    meta_lr.fit(meta_X, y_valid)

    return {
        'weights': {'lgb': w_lgb, 'xgb': w_xgb, 'mlp': w_mlp},
        'avg_auc': avg_auc,
        'stack_auc': stack_auc,
        'meta_model': meta_lr,
        'avg_pred': avg_pred,
        'stack_pred': stack_pred,
    }


def walk_forward_central(df, features, feature_names_pkl, test_years=None):
    """Walk-forward training for central model."""
    if test_years is None:
        test_years = list(range(2020, 2026))

    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD ENSEMBLE (Central)")
    print(f"  Features: {len(features)} (Pattern A leak-free)")
    print(f"  Test years: {test_years}")
    print(f"{'='*60}")

    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    all_fold_results = []
    all_y_true = []
    all_preds = {'lgb': [], 'xgb': [], 'mlp': [], 'avg': [], 'stack': []}

    for test_year in test_years:
        train_mask = df['year_full'] < test_year
        test_mask = df['year_full'] == test_year

        if test_mask.sum() == 0:
            continue

        X_train = df.loc[train_mask, features].values
        y_train = df.loc[train_mask, 'target'].values
        X_test = df.loc[test_mask, features].values
        y_test = df.loc[test_mask, 'target'].values

        print(f"\n  Fold {test_year}: train={len(X_train)}, test={len(X_test)}")
        t0 = time.time()

        fold = train_fold(X_train, y_train, X_test, y_test, features)
        ens = compute_ensemble(fold, y_test)

        elapsed = time.time() - t0
        print(f"    LGB={fold['lgb']['auc']:.4f}  XGB={fold['xgb']['auc']:.4f}  "
              f"MLP={fold['mlp']['auc']:.4f}  AVG={ens['avg_auc']:.4f}  "
              f"STACK={ens['stack_auc']:.4f}  ({elapsed:.0f}s)")

        all_fold_results.append({
            'year': test_year, 'fold': fold, 'ensemble': ens,
            'n_test': len(X_test),
        })
        all_y_true.extend(y_test.tolist())
        all_preds['lgb'].extend(fold['lgb']['pred'].tolist())
        all_preds['xgb'].extend(fold['xgb']['pred'].tolist())
        all_preds['mlp'].extend(fold['mlp']['pred'].tolist())
        all_preds['avg'].extend(ens['avg_pred'].tolist())
        all_preds['stack'].extend(ens['stack_pred'].tolist())

    # Overall AUC
    y_all = np.array(all_y_true)
    overall = {}
    for key in all_preds:
        p = np.array(all_preds[key])
        overall[key] = roc_auc_score(y_all, p)

    print(f"\n  {'='*50}")
    print(f"  OVERALL WF AUC (Central)")
    print(f"  {'='*50}")
    print(f"  LightGBM:     {overall['lgb']:.4f}")
    print(f"  XGBoost:      {overall['xgb']:.4f}")
    print(f"  MLP:          {overall['mlp']:.4f}")
    print(f"  Weighted Avg: {overall['avg']:.4f}")
    print(f"  Stacking:     {overall['stack']:.4f}")

    best_method = max(overall, key=overall.get)
    print(f"  >>> Best: {best_method} (AUC {overall[best_method]:.4f})")

    return all_fold_results, overall


def train_final_central(df, features, feature_names_pkl):
    """Train final production model on all data up to max_year-1, validate on max_year."""
    print(f"\n{'='*60}")
    print(f"  FINAL CENTRAL MODEL TRAINING")
    print(f"{'='*60}")

    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    max_year = df['year_full'].max()
    train_mask = df['year_full'] < max_year
    valid_mask = df['year_full'] == max_year

    X_train = df.loc[train_mask, features].values
    y_train = df.loc[train_mask, 'target'].values
    X_valid = df.loc[valid_mask, features].values
    y_valid = df.loc[valid_mask, 'target'].values

    print(f"  Train: {len(X_train)} (< {max_year}), Valid: {len(X_valid)} ({max_year})")

    fold = train_fold(X_train, y_train, X_valid, y_valid, features)
    ens = compute_ensemble(fold, y_valid)

    print(f"  LGB={fold['lgb']['auc']:.4f}  XGB={fold['xgb']['auc']:.4f}  "
          f"MLP={fold['mlp']['auc']:.4f}  AVG={ens['avg_auc']:.4f}  "
          f"STACK={ens['stack_auc']:.4f}")

    # Feature importance
    importance = fold['lgb']['model'].feature_importance(importance_type='gain')
    fi = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
    print(f"\n  Feature Importance TOP 20:")
    for fname, imp in fi[:20]:
        bar = '#' * int(imp / fi[0][1] * 25)
        print(f"    {fname:30s} {imp:10.1f} {bar}")

    return fold, ens


# ==================== NAR Section ====================

NAR_CSV_PATH = os.path.join(DATA_DIR, 'chihou_races_2020_2025.csv')
NAR_CACHE_PATH = os.path.join(DATA_DIR, 'nar_scraped_cache.json')

NAR_FEATURES_ORIG = [
    'odds_log', 'num_horses', 'distance', 'surface_enc', 'condition_enc',
    'course_enc', 'horse_weight', 'weight_carry', 'age', 'sex_enc',
    'horse_num', 'bracket', 'jockey_wr', 'jockey_place_rate', 'trainer_wr',
    'prev_finish', 'prev2_finish', 'prev3_finish', 'avg_finish_3r',
    'best_finish_3r', 'top3_count_3r', 'finish_trend', 'prev_odds_log',
    'rest_days', 'rest_category', 'dist_cat', 'weight_cat', 'age_group',
    'horse_num_ratio', 'bracket_pos', 'carry_diff', 'dist_change',
    'dist_change_abs', 'is_nar', 'pop_rank',
]

NAR_LEAK_A = {'odds_log', 'horse_weight', 'condition_enc', 'weight_cat', 'pop_rank'}
NAR_FEATURES_A = [f for f in NAR_FEATURES_ORIG if f not in NAR_LEAK_A]


def train_nar_ensemble(jockey_stats):
    """Train NAR ensemble model (LGB + XGB + MLP)."""
    print(f"\n{'='*60}")
    print(f"  NAR ENSEMBLE TRAINING (Pattern A)")
    print(f"{'='*60}")

    df = pd.read_csv(NAR_CSV_PATH)
    print(f"  CSV: {len(df)} rows, {df['race_id'].nunique()} races")

    # Build features
    df['target'] = (df['finish'] <= 3).astype(int)
    df['jockey_wr'] = df['jockey_name'].map(lambda j: jockey_stats.get(j, {}).get('wr', 0.08))
    df['jockey_place_rate'] = df['jockey_name'].map(lambda j: jockey_stats.get(j, {}).get('place_rate', 0.25))
    df['trainer_wr'] = 0.10
    df['odds_log'] = np.log1p(df['odds'].clip(1, 999))
    df['dist_cat'] = pd.cut(df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
                             labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
    df['weight_cat'] = pd.cut(df['horse_weight'], bins=[0, 440, 480, 520, 9999],
                               labels=[0, 1, 2, 3]).astype(float).fillna(1)
    df['age_group'] = df['age'].clip(2, 7)
    df['horse_num_ratio'] = df['horse_num'] / df['num_horses'].clip(1)
    df['bracket_pos'] = pd.cut(df['bracket'], bins=[0, 3, 6, 8],
                                labels=[0, 1, 2]).astype(float).fillna(1)
    df['carry_diff'] = df['weight_carry'] - df['weight_carry'].mean()
    df['is_nar'] = 1

    # Lag features are pre-computed in NAR CSV
    # Only compute missing ones
    if 'prev_odds_log' not in df.columns:
        df['prev_odds_log'] = np.log1p(15.0)
    if 'rest_days' not in df.columns:
        df['rest_days'] = 30
    if 'rest_category' not in df.columns:
        df['rest_category'] = 2

    features = NAR_FEATURES_A
    for f_name in features:
        if f_name not in df.columns:
            df[f_name] = 0
        df[f_name] = pd.to_numeric(df[f_name], errors='coerce').fillna(0)

    X = df[features].values
    y = df['target'].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    fold = train_fold(X_train, y_train, X_test, y_test, features)
    ens = compute_ensemble(fold, y_test)

    print(f"  LGB={fold['lgb']['auc']:.4f}  XGB={fold['xgb']['auc']:.4f}  "
          f"MLP={fold['mlp']['auc']:.4f}  AVG={ens['avg_auc']:.4f}  "
          f"STACK={ens['stack_auc']:.4f}")

    # Feature importance
    importance = fold['lgb']['model'].feature_importance(importance_type='gain')
    fi = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
    print(f"\n  NAR Feature Importance TOP 15:")
    for fname, imp in fi[:15]:
        bar = '#' * int(imp / fi[0][1] * 25)
        print(f"    {fname:25s} {imp:10.1f} {bar}")

    return fold, ens, features, jockey_stats, df


def main():
    start_time = time.time()
    print("=" * 60)
    print("  KEIBA AI v10 ENSEMBLE TRAINING")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Pattern A (strict leak-free)")
    print("=" * 60)

    # ==================== CENTRAL ====================
    print("\n" + "=" * 60)
    print("  [1/4] LOADING CENTRAL DATA")
    print("=" * 60)

    df = load_data()

    # Lap data
    lap_df = load_lap_data()
    if lap_df is not None:
        df = df.merge(lap_df, on='race_id_str', how='left')
        matched = df['race_first3f'].notna().sum()
        print(f"  Lap data: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")

    df = encode_categoricals(df)
    df, sire_map, bms_map = encode_sires(df)

    # Training times with FIXED horse_id
    tt_data = load_training_times_fixed()
    df = merge_training_features_fixed(df, tt_data)

    # Expanding window features
    df = compute_jockey_wr(df)
    df = compute_trainer_stats(df)
    df = compute_horse_career(df)
    df = compute_sire_performance(df)

    # compute_lag_features before build_features (no bracket dependency)
    df = compute_lag_features(df)

    print("Building features...")
    df = build_features(df)

    # These need 'bracket' from build_features
    df = compute_distance_aptitude(df)
    df = compute_frame_advantage(df)
    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses_val'] >= 5].copy()

    print(f"  Final dataset: {len(df)} rows, {df['race_id_str'].nunique()} races")

    # ==================== WALK-FORWARD ====================
    print("\n" + "=" * 60)
    print("  [2/4] WALK-FORWARD VALIDATION (Central)")
    print("=" * 60)

    wf_results, wf_overall = walk_forward_central(df, FEATURES_V10, FEATURES_V10_PKL)

    # ==================== FINAL CENTRAL MODEL ====================
    print("\n" + "=" * 60)
    print("  [3/4] FINAL CENTRAL MODEL")
    print("=" * 60)

    central_fold, central_ens = train_final_central(df, FEATURES_V10, FEATURES_V10_PKL)

    # Determine best ensemble method
    best_central_method = max(wf_overall, key=wf_overall.get)
    best_central_auc = wf_overall[best_central_method]

    # Reference: current production AUC
    current_auc = 0.8083  # Pattern A from leak_comparison_central.json
    improved = best_central_auc > current_auc

    print(f"\n  Current production AUC: {current_auc:.4f}")
    print(f"  Best WF ensemble AUC:  {best_central_auc:.4f} ({best_central_method})")
    print(f"  Improvement: {best_central_auc - current_auc:+.4f} {'>>> IMPROVED!' if improved else '(no improvement)'}")

    # ==================== NAR ====================
    print("\n" + "=" * 60)
    print("  [4/4] NAR ENSEMBLE")
    print("=" * 60)

    # Load existing jockey stats
    nar_model_path = os.path.join(OUTPUT_DIR, 'keiba_model_v9_nar.pkl')
    with open(nar_model_path, 'rb') as f:
        nar_old = pickle.load(f)
    jockey_stats = nar_old.get('jockey_stats', {})
    print(f"  Jockey stats: {len(jockey_stats)} jockeys")

    nar_fold, nar_ens, nar_features, _, nar_df = train_nar_ensemble(jockey_stats)

    nar_current_auc = 0.8243  # Pattern A from leak_comparison_nar.json
    nar_best_auc = max(nar_ens['avg_auc'], nar_ens['stack_auc'])
    nar_best_method = 'stack' if nar_ens['stack_auc'] > nar_ens['avg_auc'] else 'avg'
    nar_improved = nar_best_auc > nar_current_auc

    print(f"\n  NAR current AUC: {nar_current_auc:.4f}")
    print(f"  NAR best ensemble: {nar_best_auc:.4f} ({nar_best_method})")
    print(f"  Improvement: {nar_best_auc - nar_current_auc:+.4f} {'>>> IMPROVED!' if nar_improved else '(no improvement)'}")

    # ==================== SAVE ====================
    print(f"\n{'='*60}")
    print(f"  SAVING MODELS")
    print(f"{'='*60}")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine ensemble weights for central
    c_lgb_auc = central_fold['lgb']['auc']
    c_xgb_auc = central_fold['xgb']['auc']
    c_mlp_auc = central_fold['mlp']['auc']
    c_total = c_lgb_auc + c_xgb_auc + c_mlp_auc
    c_weights = {'lgb': c_lgb_auc / c_total, 'xgb': c_xgb_auc / c_total, 'mlp': c_mlp_auc / c_total}

    central_pkl = {
        'model': central_fold['lgb']['model'],
        'xgb_model': central_fold['xgb']['model'],
        'mlp_model': central_fold['mlp']['model'],
        'mlp_scaler': central_fold['mlp']['scaler'],
        'features': FEATURES_V10_PKL,
        'version': 'v10_ensemble',
        'auc': c_lgb_auc,
        'ensemble_auc': central_ens['avg_auc'],
        'stack_auc': central_ens['stack_auc'],
        'ensemble_weights': c_weights,
        'meta_model': central_ens['meta_model'],
        'leak_free': True,
        'leak_pattern': 'A',
        'leak_removed': sorted(LEAK_FEATURES_A),
        'sire_map': sire_map,
        'bms_map': bms_map,
        'n_top_encode': N_TOP_SIRE,
        'trained_at': now,
        'model_type': 'central',
        'course_map': dict(COURSE_MAP),
        'wf_overall': wf_overall,
        'training_fix': 'horse_id 10→8 digit conversion',
    }

    if improved:
        central_path = os.path.join(OUTPUT_DIR, 'keiba_model_v9_central.pkl')
        with open(central_path, 'wb') as f:
            pickle.dump(central_pkl, f)
        print(f"  Central (production): {central_path}")

        v8_path = os.path.join(OUTPUT_DIR, 'keiba_model_v8.pkl')
        with open(v8_path, 'wb') as f:
            pickle.dump(central_pkl, f)
        print(f"  Central (v8 backup): {v8_path}")
    else:
        ref_path = os.path.join(OUTPUT_DIR, 'keiba_model_v10_central_ref.pkl')
        with open(ref_path, 'wb') as f:
            pickle.dump(central_pkl, f)
        print(f"  Central (reference only): {ref_path}")

    # NAR
    n_lgb_auc = nar_fold['lgb']['auc']
    n_xgb_auc = nar_fold['xgb']['auc']
    n_mlp_auc = nar_fold['mlp']['auc']
    n_total = n_lgb_auc + n_xgb_auc + n_mlp_auc
    n_weights = {'lgb': n_lgb_auc / n_total, 'xgb': n_xgb_auc / n_total, 'mlp': n_mlp_auc / n_total}

    nar_pkl = {
        'model': nar_fold['lgb']['model'],
        'xgb_model': nar_fold['xgb']['model'],
        'mlp_model': nar_fold['mlp']['model'],
        'mlp_scaler': nar_fold['mlp']['scaler'],
        'features': nar_features,
        'version': 'nar_v3_ensemble',
        'auc': n_lgb_auc,
        'ensemble_auc': nar_ens['avg_auc'],
        'stack_auc': nar_ens['stack_auc'],
        'ensemble_weights': n_weights,
        'meta_model': nar_ens['meta_model'],
        'leak_free': True,
        'leak_pattern': 'A',
        'leak_removed': sorted(NAR_LEAK_A),
        'model_type': 'nar_dedicated',
        'trained_at': now,
        'jockey_stats': jockey_stats,
    }

    if nar_improved:
        with open(nar_model_path, 'wb') as f:
            pickle.dump(nar_pkl, f)
        print(f"  NAR (production): {nar_model_path}")
    else:
        ref_path = os.path.join(OUTPUT_DIR, 'keiba_model_v10_nar_ref.pkl')
        with open(ref_path, 'wb') as f:
            pickle.dump(nar_pkl, f)
        print(f"  NAR (reference only): {ref_path}")

    # Save results JSON
    results = {
        'generated_at': now,
        'training_fix': 'horse_id 10→8 digit conversion for training_times.csv',
        'central': {
            'wf_auc': wf_overall,
            'final_lgb_auc': c_lgb_auc,
            'final_xgb_auc': c_xgb_auc,
            'final_mlp_auc': c_mlp_auc,
            'final_avg_auc': central_ens['avg_auc'],
            'final_stack_auc': central_ens['stack_auc'],
            'best_method': best_central_method,
            'best_wf_auc': best_central_auc,
            'current_auc': current_auc,
            'improved': improved,
            'weights': c_weights,
            'features': FEATURES_V10_PKL,
            'n_features': len(FEATURES_V10_PKL),
        },
        'nar': {
            'lgb_auc': n_lgb_auc,
            'xgb_auc': n_xgb_auc,
            'mlp_auc': n_mlp_auc,
            'avg_auc': nar_ens['avg_auc'],
            'stack_auc': nar_ens['stack_auc'],
            'best_method': nar_best_method,
            'best_auc': nar_best_auc,
            'current_auc': nar_current_auc,
            'improved': nar_improved,
            'weights': n_weights,
            'features': nar_features,
            'n_features': len(nar_features),
        },
    }

    results_path = os.path.join(OUTPUT_DIR, 'train_v10_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Results: {results_path}")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  Central: LGB={c_lgb_auc:.4f} XGB={c_xgb_auc:.4f} MLP={c_mlp_auc:.4f}")
    print(f"           AVG={central_ens['avg_auc']:.4f} STACK={central_ens['stack_auc']:.4f}")
    print(f"           WF best: {best_central_method}={best_central_auc:.4f} (was {current_auc:.4f})")
    print(f"  NAR:     LGB={n_lgb_auc:.4f} XGB={n_xgb_auc:.4f} MLP={n_mlp_auc:.4f}")
    print(f"           AVG={nar_ens['avg_auc']:.4f} STACK={nar_ens['stack_auc']:.4f}")
    print(f"           (was {nar_current_auc:.4f})")

    return results


if __name__ == '__main__':
    main()
