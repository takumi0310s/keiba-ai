#!/usr/bin/env python
"""NAR Model V4 - Train on scraped nar_all_races.csv (4800+ races)
Replaces V2 which only had 184 races from KDSCOPE.
Uses LightGBM + XGBoost ensemble.
"""
import os
import sys
import json
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SCRAPED_CSV = os.path.join(DATA_DIR, 'nar_all_races.csv')
OLD_CSV = os.path.join(DATA_DIR, 'chihou_races_2020_2025.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'keiba_model_v9_nar.pkl')
RESULT_PATH = os.path.join(BASE_DIR, 'train_nar_v4_results.json')

COURSE_MAP = {
    '大井': 10, '川崎': 11, '船橋': 12, '浦和': 13, '園田': 14, '姫路': 15,
    '門別': 16, '盛岡': 17, '水沢': 18, '金沢': 19, '笠松': 20, '名古屋': 21,
    '高知': 22, '佐賀': 23, '帯広': 24,
}
SURFACE_MAP = {'芝': 0, 'ダ': 1, '障': 2}
COND_MAP = {'良': 0, '稍': 1, '稍重': 1, '重': 2, '不': 3, '不良': 3}
SEX_MAP = {'牡': 0, '牝': 1, 'セ': 2, '騸': 2}

FEATURES = [
    'odds_log', 'num_horses', 'distance', 'surface_enc', 'condition_enc',
    'course_enc', 'horse_weight', 'weight_carry', 'age', 'sex_enc',
    'horse_num', 'bracket', 'horse_num_ratio', 'bracket_pos',
    'carry_diff', 'dist_cat', 'weight_cat', 'age_group',
    'jockey_wr', 'jockey_place_rate', 'pop_rank',
    'is_nar',
]


def load_and_prepare_data():
    """Load scraped CSV and prepare features."""
    print('\n  Loading data...', flush=True)

    df = pd.read_csv(SCRAPED_CSV, encoding='utf-8')
    print(f'  Scraped: {len(df)} rows, {df["race_id"].nunique()} races', flush=True)

    # Note: old KDSCOPE data (184 races) has different schema, not worth merging
    # with 4800+ new scraped races

    # Filter out non-numeric finish
    df['finish'] = pd.to_numeric(df['finish'], errors='coerce')
    df = df.dropna(subset=['finish'])
    df['finish'] = df['finish'].astype(int)
    df = df[df['finish'] > 0]

    # Target: top 3
    df['target'] = (df['finish'] <= 3).astype(int)

    # Parse sex/age
    df['sex'] = df['sex_age'].astype(str).str[0]
    df['age'] = pd.to_numeric(df['sex_age'].astype(str).str[1:], errors='coerce').fillna(4)

    # Build features
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce').fillna(30)
    df['odds_log'] = np.log1p(df['odds'].clip(1, 999))
    df['num_horses'] = pd.to_numeric(df['num_horses'], errors='coerce').fillna(10)
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(1600)
    df['horse_weight'] = pd.to_numeric(df['horse_weight'], errors='coerce').fillna(470)
    df['weight_carry'] = pd.to_numeric(df['weight_carry'], errors='coerce').fillna(55)
    df['horse_num'] = pd.to_numeric(df['horse_num'], errors='coerce').fillna(5)
    df['bracket'] = pd.to_numeric(df['bracket'], errors='coerce').fillna(4)
    df['pop_rank'] = pd.to_numeric(df['pop_rank'], errors='coerce').fillna(5)

    df['surface_enc'] = df['surface'].map(SURFACE_MAP).fillna(1)
    df['condition_enc'] = df['condition'].map(COND_MAP).fillna(0)
    df['course_enc'] = df['course'].map(COURSE_MAP).fillna(10)
    df['sex_enc'] = df['sex'].map(SEX_MAP).fillna(0)

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

    # Compute jockey stats (expanding window - leak-free)
    df['race_date_int'] = pd.to_numeric(df['race_date'], errors='coerce').fillna(0).astype(int)
    df = df.sort_values('race_date_int')

    print('  Computing jockey stats (leak-free)...', flush=True)
    jockey_wins = {}
    jockey_runs = {}
    jockey_top3 = {}
    jwr_list = []
    jpr_list = []

    for _, row in df.iterrows():
        j = row['jockey_name']
        runs = jockey_runs.get(j, 0)
        wins = jockey_wins.get(j, 0)
        t3 = jockey_top3.get(j, 0)
        jwr_list.append(wins / runs if runs >= 10 else 0.08)
        jpr_list.append(t3 / runs if runs >= 10 else 0.25)
        jockey_runs[j] = runs + 1
        if row['finish'] == 1:
            jockey_wins[j] = wins + 1
        if row['finish'] <= 3:
            jockey_top3[j] = t3 + 1

    df['jockey_wr'] = jwr_list
    df['jockey_place_rate'] = jpr_list

    # Final jockey stats for production
    jockey_stats = {}
    for j in jockey_runs:
        if jockey_runs[j] >= 5:
            jockey_stats[j] = {
                'wr': round(jockey_wins.get(j, 0) / jockey_runs[j], 4),
                'place_rate': round(jockey_top3.get(j, 0) / jockey_runs[j], 4),
                'runs': jockey_runs[j],
            }

    print(f'  Jockey stats: {len(jockey_stats)} jockeys', flush=True)
    print(f'  Features: {len(FEATURES)}', flush=True)
    print(f'  Target rate: {df["target"].mean():.3f}', flush=True)

    return df, jockey_stats


def train_and_evaluate(df, jockey_stats):
    """Train LGB + XGB with 80/20 split (only 2025 data = 1 year)."""
    print('\n  Training...', flush=True)

    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0

    X = df[FEATURES].values
    y = df['target'].values

    # Split by month: Jan-Sep train, Oct-Dec test (temporal split)
    df['month'] = df['race_date'].astype(str).str[4:6].astype(int)
    months = sorted(df['month'].unique())
    if len(months) > 3:
        # Last 3 months as test
        test_months = months[-3:]
        train_mask = ~df['month'].isin(test_months)
        test_mask = df['month'].isin(test_months)
    else:
        # Random split
        np.random.seed(42)
        idx = np.random.permutation(len(df))
        split = int(len(idx) * 0.8)
        train_mask = pd.Series(False, index=df.index)
        test_mask = pd.Series(False, index=df.index)
        train_mask.iloc[idx[:split]] = True
        test_mask.iloc[idx[split:]] = True

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    print(f'  Train: {len(train_idx)} rows, Test: {len(test_idx)} rows', flush=True)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # LightGBM
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 15, 'learning_rate': 0.04,
        'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5,
        'reg_alpha': 0.5, 'reg_lambda': 0.5,
        'verbose': -1, 'n_jobs': -1,
    }

    dtrain = lgb.Dataset(X_train, y_train)
    dval = lgb.Dataset(X_test, y_test, reference=dtrain)
    lgb_model = lgb.train(
        lgb_params, dtrain, num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    lgb_pred = lgb_model.predict(X_test)
    lgb_auc = roc_auc_score(y_test, lgb_pred)
    print(f'  LGB AUC: {lgb_auc:.4f}', flush=True)

    # XGBoost
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'max_depth': 4, 'learning_rate': 0.04,
        'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 0.5, 'reg_lambda': 0.5,
        'verbosity': 0, 'nthread': -1,
    }

    xgb_dtrain = xgb.DMatrix(X_train, y_train)
    xgb_dval = xgb.DMatrix(X_test, y_test)
    xgb_model = xgb.train(
        xgb_params, xgb_dtrain, num_boost_round=500,
        evals=[(xgb_dval, 'val')],
        early_stopping_rounds=50, verbose_eval=False,
    )

    xgb_pred = xgb_model.predict(xgb_dval)
    xgb_auc = roc_auc_score(y_test, xgb_pred)
    print(f'  XGB AUC: {xgb_auc:.4f}', flush=True)

    # Ensemble
    total_auc = lgb_auc + xgb_auc
    w_lgb = lgb_auc / total_auc
    w_xgb = xgb_auc / total_auc
    ens_pred = lgb_pred * w_lgb + xgb_pred * w_xgb
    ens_auc = roc_auc_score(y_test, ens_pred)
    print(f'  Ensemble AUC: {ens_auc:.4f} (w_lgb={w_lgb:.3f}, w_xgb={w_xgb:.3f})', flush=True)

    # Feature importance
    importance = lgb_model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(FEATURES, importance), key=lambda x: -x[1])
    print('\n  Top 10 features:', flush=True)
    for fname, imp in feat_imp[:10]:
        print(f'    {fname}: {imp:.0f}', flush=True)

    # Final model on all data
    print('\n  Training final model on all data...', flush=True)
    dtrain_all = lgb.Dataset(X, y)
    lgb_final = lgb.train(lgb_params, dtrain_all, num_boost_round=lgb_model.best_iteration)

    xgb_dtrain_all = xgb.DMatrix(X, y)
    xgb_final = xgb.train(xgb_params, xgb_dtrain_all, num_boost_round=xgb_model.best_iteration)

    lgb_all = lgb_final.predict(X)
    xgb_all = xgb_final.predict(xgb_dtrain_all)
    final_pred = lgb_all * w_lgb + xgb_all * w_xgb
    final_auc = roc_auc_score(y, final_pred)
    print(f'  Final all-data AUC: {final_auc:.4f}', flush=True)

    return {
        'lgb_model': lgb_final, 'xgb_model': xgb_final,
        'lgb_auc': lgb_auc, 'xgb_auc': xgb_auc,
        'ensemble_auc': ens_auc, 'final_auc': final_auc,
        'weights': {'lgb': w_lgb, 'xgb': w_xgb},
        'feat_imp': feat_imp, 'jockey_stats': jockey_stats,
    }


def main():
    print('=' * 60, flush=True)
    print('  NAR MODEL V4 TRAINING', flush=True)
    print(f'  {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    print('=' * 60, flush=True)

    df, jockey_stats = load_and_prepare_data()
    results = train_and_evaluate(df, jockey_stats)

    # Compare with current model
    current_auc = 0
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            old = pickle.load(f)
        current_auc = old.get('auc', old.get('lgb_auc', 0))
        print(f'\n  Current model AUC: {current_auc:.4f}', flush=True)

    new_auc = results['ensemble_auc']
    improved = new_auc > current_auc
    print(f'  New ensemble AUC: {new_auc:.4f}', flush=True)
    print(f'  Improved: {improved}', flush=True)

    # Save model
    model_data = {
        'model': results['lgb_model'],
        'xgb_model': results['xgb_model'],
        'features': FEATURES,
        'ensemble_weights': results['weights'],
        'jockey_stats': results['jockey_stats'],
        'version': 'nar_v4',
        'auc': new_auc,
        'lgb_auc': results['lgb_auc'],
        'xgb_auc': results['xgb_auc'],
        'n_races': int(df['race_id'].nunique()),
        'n_rows': int(len(df)),
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    if improved or current_auc == 0:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model_data, f)
        print(f'  Model UPDATED: {MODEL_PATH}', flush=True)
    else:
        alt_path = os.path.join(BASE_DIR, 'keiba_model_nar_v4.pkl')
        with open(alt_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f'  No improvement - saved to {alt_path}', flush=True)

    # Save results
    result_json = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data': {
            'scraped_races': int(df['race_id'].nunique()),
            'scraped_rows': int(len(df)),
        },
        'lgb_auc': round(results['lgb_auc'], 4),
        'xgb_auc': round(results['xgb_auc'], 4),
        'ensemble_auc': round(results['ensemble_auc'], 4),
        'final_auc': round(results['final_auc'], 4),
        'current_auc': round(current_auc, 4),
        'improved': improved,
        'weights': {k: round(v, 4) for k, v in results['weights'].items()},
        'top_features': [(f, round(float(i), 1)) for f, i in results['feat_imp'][:15]],
    }

    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    print(f'  Results: {RESULT_PATH}', flush=True)

    print('\n  Done!', flush=True)
    return result_json


if __name__ == '__main__':
    main()
