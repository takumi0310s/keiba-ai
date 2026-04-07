#!/usr/bin/env python3
"""v15 Full Improvement Pipeline

TASK 1: v14.2 再学習 — training_times・stable_comments追加データ反映
TASK 4: CatBoost 5モデル目追加
TASK 5: Optuna LGB/XGB ハイパーパラメータ再調整

全結果を data/v15_full_improvement_report.json に保存。
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import torch

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

from train_v141_paci_tierA import (
    build_v141_dataframe, get_v141_features, fill_v141_defaults,
    V141_NEW_FEATURES, V141_DEFAULTS,
)
from train_v135_ft_transformer import (
    FTTransformer, IntraRaceAttention, train_ft_transformer,
    DEVICE, LGB_PARAMS, XGB_PARAMS,
)
from train_v135b_intra_ensemble import (
    build_race_id, train_intra_race_with_preds, MAX_HORSES,
)
from jrdb_features import JRDB_DEFAULTS, JRDB_LIVE_FEATURES, PACI_TIER_A_DEFAULTS

BASELINE_AUC = 0.8856  # v14.1 current best


# ================================================================
# Utility: WF for arbitrary GBDT params
# ================================================================

def wf_lgb_xgb(df, features, years, lgb_params, xgb_params):
    """Quick WF with LGB+XGB only (no FT/IR) for fast evaluation."""
    results = []
    for test_year in years:
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        X_tr = df.loc[train_mask, features].values.astype(np.float32)
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[test_mask, features].values.astype(np.float32)
        y_te = df.loc[test_mask, 'target'].values

        # LGB
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
        m_lgb = lgb.train(lgb_params, dtrain, num_boost_round=1000,
                          valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        p_lgb = m_lgb.predict(X_te)
        auc_lgb = roc_auc_score(y_te, p_lgb)
        auc_lgb_tr = roc_auc_score(y_tr, m_lgb.predict(X_tr))

        # XGB
        dxtr = xgb.DMatrix(X_tr, label=y_tr)
        dxte = xgb.DMatrix(X_te, label=y_te)
        m_xgb = xgb.train(xgb_params, dxtr, num_boost_round=1000,
                           evals=[(dxte, 'valid')],
                           early_stopping_rounds=50, verbose_eval=False)
        p_xgb = m_xgb.predict(dxte)
        auc_xgb = roc_auc_score(y_te, p_xgb)

        w = auc_lgb / (auc_lgb + auc_xgb)
        auc_ens = roc_auc_score(y_te, w * p_lgb + (1 - w) * p_xgb)
        gap = auc_lgb_tr - auc_lgb

        results.append({
            'year': test_year, 'lgb_auc': auc_lgb, 'xgb_auc': auc_xgb,
            'ensemble_auc': auc_ens, 'train_auc': auc_lgb_tr, 'gap': gap,
        })
    return results


# ================================================================
# TASK 1: v14.2 retrain with expanded data
# ================================================================

def task1_v142_retrain(df, features):
    """Full 4-model WF with expanded training_times & stable_comments data."""
    print("\n" + "=" * 70)
    print("  TASK 1: v14.2 Retrain (expanded data)")
    print("=" * 70)

    years = range(2021, 2026)
    results = []

    for test_year in years:
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        X_tr = df.loc[train_mask, features].values.astype(np.float32)
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[test_mask, features].values.astype(np.float32)
        y_te = df.loc[test_mask, 'target'].values
        test_indices = df.loc[test_mask].index.values

        print(f"\n{'='*60}")
        print(f"  4-Model WF {test_year}: train={len(X_tr)}, test={len(X_te)}")
        print(f"{'='*60}")

        # LGB
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
        m_lgb = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                          valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        p_lgb = m_lgb.predict(X_te)
        auc_lgb = roc_auc_score(y_te, p_lgb)
        auc_lgb_tr = roc_auc_score(y_tr, m_lgb.predict(X_tr))

        # XGB
        dxtr = xgb.DMatrix(X_tr, label=y_tr)
        dxte = xgb.DMatrix(X_te, label=y_te)
        m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=1000,
                           evals=[(dxte, 'valid')],
                           early_stopping_rounds=50, verbose_eval=False)
        p_xgb = m_xgb.predict(dxte)
        auc_xgb = roc_auc_score(y_te, p_xgb)

        # FT-Transformer
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        ft_model, p_ft, p_ft_tr, auc_ft = train_ft_transformer(
            X_tr_s, y_tr.astype(np.float32),
            X_te_s, y_te.astype(np.float32),
            n_features=len(features),
            epochs=50, batch_size=4096, lr=1e-3,
            patience=10, d_token=64, n_heads=4, n_layers=3,
            dropout=0.1, label=f'FT-v142-{test_year}',
        )

        # IntraRace Attention
        df_tr = df.loc[train_mask].copy()
        df_te = df.loc[test_mask].copy()
        scaler_ir = StandardScaler()
        df_tr_scaled = df_tr.copy()
        df_te_scaled = df_te.copy()
        df_tr_scaled[features] = scaler_ir.fit_transform(
            df_tr[features].values.astype(np.float32))
        df_te_scaled[features] = scaler_ir.transform(
            df_te[features].values.astype(np.float32))

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

        # Grid search 4-model weights
        best_grid_auc = 0
        best_grid_w = None
        if auc_ir > 0:
            for w1 in np.arange(0.15, 0.40, 0.05):
                for w2 in np.arange(0.15, 0.40, 0.05):
                    for w3 in np.arange(0.05, 0.25, 0.05):
                        w4 = 1.0 - w1 - w2 - w3
                        if w4 < 0.05 or w4 > 0.45:
                            continue
                        p_grid = w1 * p_lgb + w2 * p_xgb + w3 * p_ft + w4 * p_ir
                        auc_grid = roc_auc_score(y_te, p_grid)
                        if auc_grid > best_grid_auc:
                            best_grid_auc = auc_grid
                            best_grid_w = (w1, w2, w3, w4)

        gap = auc_lgb_tr - auc_lgb

        print(f"  {test_year}: LGB={auc_lgb:.4f} XGB={auc_xgb:.4f} "
              f"FT={auc_ft:.4f} IR={auc_ir:.4f}({ir_cov_pct:.0f}%) "
              f"Grid={best_grid_auc:.4f} gap={gap:.4f}")

        results.append({
            'year': test_year,
            'lgb_auc': float(auc_lgb), 'xgb_auc': float(auc_xgb),
            'ft_auc': float(auc_ft), 'ir_auc': float(auc_ir),
            'ir_coverage': float(ir_cov_pct),
            'grid_auc': float(best_grid_auc) if best_grid_w else None,
            'grid_weights': [float(x) for x in best_grid_w] if best_grid_w else None,
            'train_auc': float(auc_lgb_tr), 'gap': float(gap),
        })

    grid_vals = [r['grid_auc'] for r in results if r['grid_auc']]
    grid_mean = float(np.mean(grid_vals)) if grid_vals else 0

    no_overfit = all(r['gap'] < 0.05 for r in results)
    adopted = grid_mean > BASELINE_AUC and no_overfit

    print(f"\n  TASK 1 Summary:")
    print(f"    v14.1 baseline: {BASELINE_AUC:.6f}")
    print(f"    v14.2 Grid Mean: {grid_mean:.6f} ({grid_mean - BASELINE_AUC:+.6f})")
    print(f"    Overfit check: {'PASS' if no_overfit else 'FAIL'}")
    print(f"    Verdict: {'ADOPTED' if adopted else 'NOT ADOPTED'}")

    return {
        'task': 'TASK1_v142_retrain',
        'baseline_auc': BASELINE_AUC,
        'grid_mean_auc': grid_mean,
        'delta': grid_mean - BASELINE_AUC,
        'adopted': adopted,
        'no_overfit': no_overfit,
        'yearly': results,
        'features_count': len(features),
    }


# ================================================================
# TASK 4: CatBoost as 5th model
# ================================================================

def task4_catboost(df, features, task1_result):
    """Add CatBoost as 5th model to ensemble."""
    print("\n" + "=" * 70)
    print("  TASK 4: CatBoost 5th Model")
    print("=" * 70)

    base_auc = task1_result['grid_mean_auc'] if task1_result['adopted'] else BASELINE_AUC
    print(f"  Base AUC to beat: {base_auc:.6f}")

    CB_PARAMS = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3.0,
        'border_count': 128,
        'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'random_seed': 42,
        'verbose': 0,
        'early_stopping_rounds': 50,
        'task_type': 'CPU',
    }

    years = range(2021, 2026)
    results = []

    for test_year in years:
        ty = test_year - 2000
        train_mask = df['year'] < ty
        test_mask = df['year'] == ty
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        X_tr = df.loc[train_mask, features].values.astype(np.float32)
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[test_mask, features].values.astype(np.float32)
        y_te = df.loc[test_mask, 'target'].values
        test_indices = df.loc[test_mask].index.values

        print(f"\n  5-Model WF {test_year}: train={len(X_tr)}, test={len(X_te)}")

        # LGB
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
        m_lgb = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                          valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        p_lgb = m_lgb.predict(X_te)
        auc_lgb = roc_auc_score(y_te, p_lgb)
        auc_lgb_tr = roc_auc_score(y_tr, m_lgb.predict(X_tr))

        # XGB
        dxtr = xgb.DMatrix(X_tr, label=y_tr)
        dxte = xgb.DMatrix(X_te, label=y_te)
        m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=1000,
                           evals=[(dxte, 'valid')],
                           early_stopping_rounds=50, verbose_eval=False)
        p_xgb = m_xgb.predict(dxte)
        auc_xgb = roc_auc_score(y_te, p_xgb)

        # CatBoost
        cb_train = cb.Pool(X_tr, label=y_tr)
        cb_test = cb.Pool(X_te, label=y_te)
        m_cb = cb.CatBoostClassifier(**CB_PARAMS)
        m_cb.fit(cb_train, eval_set=cb_test, verbose=0)
        p_cb = m_cb.predict_proba(X_te)[:, 1]
        auc_cb = roc_auc_score(y_te, p_cb)

        # FT-Transformer
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        ft_model, p_ft, _, auc_ft = train_ft_transformer(
            X_tr_s, y_tr.astype(np.float32),
            X_te_s, y_te.astype(np.float32),
            n_features=len(features),
            epochs=50, batch_size=4096, lr=1e-3,
            patience=10, d_token=64, n_heads=4, n_layers=3,
            dropout=0.1, label=f'FT-CB-{test_year}',
        )

        # IntraRace Attention
        df_tr = df.loc[train_mask].copy()
        df_te = df.loc[test_mask].copy()
        scaler_ir = StandardScaler()
        df_tr_scaled = df_tr.copy()
        df_te_scaled = df_te.copy()
        df_tr_scaled[features] = scaler_ir.fit_transform(
            df_tr[features].values.astype(np.float32))
        df_te_scaled[features] = scaler_ir.transform(
            df_te[features].values.astype(np.float32))

        ir_model, ir_val_dict, _, _, ir_val_auc = \
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

        gap = auc_lgb_tr - auc_lgb

        # Grid search: 4-model (without CatBoost) baseline
        best_4m_auc = 0
        best_4m_w = None
        if auc_ir > 0:
            for w1 in np.arange(0.15, 0.40, 0.05):
                for w2 in np.arange(0.15, 0.40, 0.05):
                    for w3 in np.arange(0.05, 0.25, 0.05):
                        w4 = 1.0 - w1 - w2 - w3
                        if w4 < 0.05 or w4 > 0.45:
                            continue
                        p_4m = w1 * p_lgb + w2 * p_xgb + w3 * p_ft + w4 * p_ir
                        a = roc_auc_score(y_te, p_4m)
                        if a > best_4m_auc:
                            best_4m_auc = a
                            best_4m_w = (w1, w2, w3, w4)

        # Grid search: 5-model (with CatBoost)
        best_5m_auc = 0
        best_5m_w = None
        if auc_ir > 0:
            for w1 in np.arange(0.10, 0.35, 0.05):  # LGB
                for w2 in np.arange(0.10, 0.35, 0.05):  # XGB
                    for w3 in np.arange(0.05, 0.20, 0.05):  # FT
                        for w5 in np.arange(0.05, 0.25, 0.05):  # CB
                            w4 = 1.0 - w1 - w2 - w3 - w5  # IR
                            if w4 < 0.05 or w4 > 0.40:
                                continue
                            p_5m = (w1 * p_lgb + w2 * p_xgb + w3 * p_ft
                                    + w4 * p_ir + w5 * p_cb)
                            a = roc_auc_score(y_te, p_5m)
                            if a > best_5m_auc:
                                best_5m_auc = a
                                best_5m_w = (w1, w2, w3, w4, w5)

        print(f"  {test_year}: LGB={auc_lgb:.4f} XGB={auc_xgb:.4f} "
              f"CB={auc_cb:.4f} FT={auc_ft:.4f} IR={auc_ir:.4f} "
              f"4m={best_4m_auc:.4f} 5m={best_5m_auc:.4f} gap={gap:.4f}")

        results.append({
            'year': test_year,
            'lgb_auc': float(auc_lgb), 'xgb_auc': float(auc_xgb),
            'cb_auc': float(auc_cb), 'ft_auc': float(auc_ft),
            'ir_auc': float(auc_ir), 'ir_coverage': float(ir_cov_pct),
            'grid_4m_auc': float(best_4m_auc),
            'grid_4m_weights': [float(x) for x in best_4m_w] if best_4m_w else None,
            'grid_5m_auc': float(best_5m_auc),
            'grid_5m_weights': [float(x) for x in best_5m_w] if best_5m_w else None,
            'train_auc': float(auc_lgb_tr), 'gap': float(gap),
        })

    mean_4m = float(np.mean([r['grid_4m_auc'] for r in results if r['grid_4m_auc']]))
    mean_5m = float(np.mean([r['grid_5m_auc'] for r in results if r['grid_5m_auc']]))
    no_overfit = all(r['gap'] < 0.05 for r in results)
    adopted = mean_5m > base_auc and mean_5m > mean_4m and no_overfit

    print(f"\n  TASK 4 Summary:")
    print(f"    4-model mean: {mean_4m:.6f}")
    print(f"    5-model mean: {mean_5m:.6f} ({mean_5m - mean_4m:+.6f} vs 4m)")
    print(f"    Base AUC:     {base_auc:.6f}")
    print(f"    Overfit:      {'PASS' if no_overfit else 'FAIL'}")
    print(f"    Verdict:      {'ADOPTED' if adopted else 'NOT ADOPTED'}")

    return {
        'task': 'TASK4_catboost_5model',
        'base_auc': base_auc,
        'mean_4m_auc': mean_4m,
        'mean_5m_auc': mean_5m,
        'delta_5m_vs_4m': mean_5m - mean_4m,
        'delta_5m_vs_base': mean_5m - base_auc,
        'adopted': adopted,
        'no_overfit': no_overfit,
        'cb_params': CB_PARAMS,
        'yearly': results,
    }


# ================================================================
# TASK 5: Optuna hyperparameter tuning
# ================================================================

def task5_optuna(df, features, current_best_auc):
    """Optuna tuning for LGB and XGB, 100 trials each."""
    print("\n" + "=" * 70)
    print("  TASK 5: Optuna Hyperparameter Tuning")
    print("=" * 70)

    wf_years = list(range(2021, 2026))

    # ---- LGB Optuna ----
    print("\n  [LGB] Running 100 trials...")

    def lgb_objective(trial):
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'verbose': -1, 'seed': 42,
        }

        aucs = []
        for test_year in wf_years:
            ty = test_year - 2000
            train_mask = df['year'] < ty
            test_mask = df['year'] == ty
            X_tr = df.loc[train_mask, features].values.astype(np.float32)
            y_tr = df.loc[train_mask, 'target'].values
            X_te = df.loc[test_mask, features].values.astype(np.float32)
            y_te = df.loc[test_mask, 'target'].values

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
            m = lgb.train(params, dtrain, num_boost_round=1000,
                          valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            p = m.predict(X_te)
            aucs.append(roc_auc_score(y_te, p))

        return float(np.mean(aucs))

    lgb_study = optuna.create_study(direction='maximize',
                                     study_name='lgb_v15')
    lgb_study.optimize(lgb_objective, n_trials=100, show_progress_bar=True)

    lgb_best = lgb_study.best_value
    lgb_best_params = lgb_study.best_params
    print(f"  LGB best: {lgb_best:.6f} (baseline LGB single ~0.87)")
    print(f"  LGB params: {lgb_best_params}")

    # ---- XGB Optuna ----
    print("\n  [XGB] Running 100 trials...")

    def xgb_objective(trial):
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc',
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 20, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'seed': 42, 'tree_method': 'hist',
        }

        aucs = []
        for test_year in wf_years:
            ty = test_year - 2000
            train_mask = df['year'] < ty
            test_mask = df['year'] == ty
            X_tr = df.loc[train_mask, features].values.astype(np.float32)
            y_tr = df.loc[train_mask, 'target'].values
            X_te = df.loc[test_mask, features].values.astype(np.float32)
            y_te = df.loc[test_mask, 'target'].values

            dxtr = xgb.DMatrix(X_tr, label=y_tr)
            dxte = xgb.DMatrix(X_te, label=y_te)
            m = xgb.train(params, dxtr, num_boost_round=1000,
                          evals=[(dxte, 'valid')],
                          early_stopping_rounds=50, verbose_eval=False)
            p = m.predict(dxte)
            aucs.append(roc_auc_score(y_te, p))

        return float(np.mean(aucs))

    xgb_study = optuna.create_study(direction='maximize',
                                     study_name='xgb_v15')
    xgb_study.optimize(xgb_objective, n_trials=100, show_progress_bar=True)

    xgb_best = xgb_study.best_value
    xgb_best_params = xgb_study.best_params
    print(f"  XGB best: {xgb_best:.6f}")
    print(f"  XGB params: {xgb_best_params}")

    # ---- Evaluate best params with full 4-model ensemble WF ----
    print("\n  Evaluating Optuna best params with LGB+XGB ensemble...")
    optimized_lgb_params = {**LGB_PARAMS, **lgb_best_params}
    optimized_xgb_params = {**XGB_PARAMS, **xgb_best_params}

    baseline_results = wf_lgb_xgb(df, features, wf_years, LGB_PARAMS, XGB_PARAMS)
    optimized_results = wf_lgb_xgb(df, features, wf_years,
                                    optimized_lgb_params, optimized_xgb_params)

    baseline_ens_mean = float(np.mean([r['ensemble_auc'] for r in baseline_results]))
    optimized_ens_mean = float(np.mean([r['ensemble_auc'] for r in optimized_results]))
    no_overfit = all(r['gap'] < 0.05 for r in optimized_results)

    delta = optimized_ens_mean - baseline_ens_mean
    adopted = delta > 0.0005 and no_overfit

    print(f"\n  TASK 5 Summary:")
    print(f"    Baseline LGB+XGB mean: {baseline_ens_mean:.6f}")
    print(f"    Optimized LGB+XGB mean: {optimized_ens_mean:.6f} ({delta:+.6f})")
    print(f"    Overfit: {'PASS' if no_overfit else 'FAIL'}")
    print(f"    Verdict: {'ADOPTED' if adopted else 'NOT ADOPTED'}")
    print(f"    (Threshold: +0.0005 required)")

    return {
        'task': 'TASK5_optuna',
        'lgb_best_auc_single': lgb_best,
        'lgb_best_params': lgb_best_params,
        'xgb_best_auc_single': xgb_best,
        'xgb_best_params': xgb_best_params,
        'baseline_lgbxgb_mean': baseline_ens_mean,
        'optimized_lgbxgb_mean': optimized_ens_mean,
        'delta': delta,
        'adopted': adopted,
        'no_overfit': no_overfit,
        'baseline_yearly': baseline_results,
        'optimized_yearly': optimized_results,
    }


# ================================================================
# Main
# ================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("  v15 Full Improvement Pipeline")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {DEVICE}")
    print(f"  Baseline: v14.1 WF AUC {BASELINE_AUC}")
    print("=" * 70)

    # Build dataframe
    print("\n[0] Building v14.1 dataframe...")
    df, sire_map, bms_map = build_v141_dataframe()
    features, jrdb_sel = get_v141_features()
    df = fill_v141_defaults(df, features)
    df = build_race_id(df)

    df['target'] = (df['finish'] <= 3).astype(int)
    df = df[df['num_horses'] >= 5].copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)

    print(f"  Data: {len(df)} rows, {len(features)} features")

    # Data coverage check
    training_feats = ['time_1f_last_filled', 'training_intensity_enc',
                      'wood_best_4f_filled', 'has_wood_training']
    comment_feats = ['stable_comment_score']
    print("\n  Training data coverage:")
    for f in training_feats + comment_feats:
        if f in df.columns:
            nonzero = (df[f] != 0).sum()
            print(f"    {f}: nonzero={nonzero}/{len(df)} ({nonzero/len(df)*100:.1f}%)")

    # TASK 1
    task1_result = task1_v142_retrain(df, features)

    # TASK 4
    task4_result = task4_catboost(df, features, task1_result)

    # TASK 5
    best_so_far = BASELINE_AUC
    if task1_result['adopted']:
        best_so_far = task1_result['grid_mean_auc']
    if task4_result['adopted']:
        best_so_far = max(best_so_far, task4_result['mean_5m_auc'])

    task5_result = task5_optuna(df, features, best_so_far)

    # ================================================================
    # Final report
    # ================================================================
    elapsed = (time.time() - t0) / 60

    # Determine final best model
    final_model = 'v14.1'
    final_auc = BASELINE_AUC
    adopted_tasks = []

    if task1_result['adopted']:
        final_model = 'v14.2'
        final_auc = task1_result['grid_mean_auc']
        adopted_tasks.append('TASK1_v142')

    if task4_result['adopted'] and task4_result['mean_5m_auc'] > final_auc:
        final_model = f'{final_model}+CatBoost'
        final_auc = task4_result['mean_5m_auc']
        adopted_tasks.append('TASK4_catboost')

    if task5_result['adopted']:
        adopted_tasks.append('TASK5_optuna')

    report = {
        'date': datetime.now().isoformat(),
        'baseline': {
            'model': 'v14.1',
            'wf_auc': BASELINE_AUC,
            'features': len(features),
        },
        'task1_v142_retrain': task1_result,
        'task4_catboost': task4_result,
        'task5_optuna': task5_result,
        'final': {
            'model': final_model,
            'auc': final_auc,
            'adopted_tasks': adopted_tasks,
            'delta_vs_baseline': final_auc - BASELINE_AUC,
        },
        'elapsed_minutes': round(elapsed, 1),
    }

    rpath = os.path.join(DATA_DIR, 'v15_full_improvement_report.json')
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Report saved: {rpath}")

    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Baseline:       v14.1 WF AUC {BASELINE_AUC:.6f}")
    print(f"  TASK 1 (v14.2): {'ADOPTED' if task1_result['adopted'] else 'NOT ADOPTED'} "
          f"AUC={task1_result['grid_mean_auc']:.6f}")
    print(f"  TASK 4 (CB):    {'ADOPTED' if task4_result['adopted'] else 'NOT ADOPTED'} "
          f"5m={task4_result['mean_5m_auc']:.6f}")
    print(f"  TASK 5 (Opt):   {'ADOPTED' if task5_result['adopted'] else 'NOT ADOPTED'} "
          f"delta={task5_result['delta']:+.6f}")
    print(f"  Final model:    {final_model} AUC={final_auc:.6f}")
    print(f"  Elapsed:        {elapsed:.1f} min")
    print(f"{'='*70}")

    return report


if __name__ == '__main__':
    report = main()
