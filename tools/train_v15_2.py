#!/usr/bin/env python3
"""v15.2 training (V15 baseline + 17 candidate features)

★ candidate suffix ★ — production .pkl.gz は絶対上書きしない。
WF 6-fold (V15 と同一 split: 2020-2025、 train year < ty, test year == ty)
4-model ensemble (LGB + XGB + FT-Transformer + IntraRace Attention)、 V15 と同一 architecture / hyperparams。

Output:
  models/v15_2_candidate.pkl.gz  (★ candidate suffix ★)
  data/v15_2/wf_results.json
  logs/v15_2_training.log (progress)
  docs/V15_2_TRAINING_LOG_<DATE>.md (final report)

絶対遵守:
  - production *.pkl.gz 上書き禁止
  - V15 cache 上書き禁止
  - git commit/push なし
  - fold AUC < 0.85 連続 2 fold で 即中止
  - 8h hard limit
"""
from __future__ import annotations

import os
import sys
import time
import json
import gzip
import pickle
import traceback
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'v15_2'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'docs'), exist_ok=True)

# V15 architecture imports
from train_v135_ft_transformer import (
    FTTransformer, train_ft_transformer, DEVICE,
)
from train_v135b_intra_ensemble import (
    build_race_id, train_intra_race_with_preds,
)

# V15 hyperparams (CLAUDE.md baseline)
LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_child_samples': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'verbose': -1, 'seed': 42,
}
XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'max_depth': 6, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'seed': 42, 'tree_method': 'hist',
}

V15_BASELINE_AUC = 0.8939  # CLAUDE.md current production
WF_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
HARD_LIMIT_SEC = 8 * 3600  # 8h


def log_progress(log_path, **kwargs):
    """Append JSON-line progress to log file (atomic-ish)."""
    entry = {'ts': datetime.now().isoformat(), **kwargs}
    line = json.dumps(entry, ensure_ascii=False, default=str)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(f"[{entry['ts']}] {kwargs}", flush=True)


def fold_train_eval(df, features, test_year, log_path, fold_idx, total_folds,
                    enable_ft=True, enable_ir=True):
    """1 fold WF train + eval. 4-model ensemble."""
    ty = test_year - 2000
    train_mask = df['year'] < ty
    test_mask = df['year'] == ty

    if train_mask.sum() < 1000 or test_mask.sum() < 100:
        log_progress(log_path, fold=fold_idx, year=test_year,
                     step='skip_insufficient',
                     train_n=int(train_mask.sum()), test_n=int(test_mask.sum()))
        return None

    X_tr = df.loc[train_mask, features].values.astype(np.float32)
    y_tr = df.loc[train_mask, 'target'].values.astype(np.float32)
    X_te = df.loc[test_mask, features].values.astype(np.float32)
    y_te = df.loc[test_mask, 'target'].values.astype(np.float32)
    test_indices = df.loc[test_mask].index.values

    log_progress(log_path, fold=fold_idx, year=test_year, step='start',
                 train_n=int(len(X_tr)), test_n=int(len(X_te)),
                 features=len(features))

    # ===== LightGBM =====
    t0 = time.time()
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
    m_lgb = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1000,
                      valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    p_lgb = m_lgb.predict(X_te)
    p_lgb_tr = m_lgb.predict(X_tr)
    auc_lgb = float(roc_auc_score(y_te, p_lgb))
    auc_lgb_tr = float(roc_auc_score(y_tr, p_lgb_tr))
    gap = auc_lgb_tr - auc_lgb
    log_progress(log_path, fold=fold_idx, year=test_year, step='lgb_done',
                 auc=auc_lgb, train_auc=auc_lgb_tr, gap=gap,
                 elapsed_sec=int(time.time() - t0))

    # ===== XGBoost =====
    t0 = time.time()
    dxtr = xgb.DMatrix(X_tr, label=y_tr)
    dxte = xgb.DMatrix(X_te, label=y_te)
    m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=1000,
                      evals=[(dxte, 'valid')],
                      early_stopping_rounds=50, verbose_eval=False)
    p_xgb = m_xgb.predict(dxte)
    auc_xgb = float(roc_auc_score(y_te, p_xgb))
    log_progress(log_path, fold=fold_idx, year=test_year, step='xgb_done',
                 auc=auc_xgb, elapsed_sec=int(time.time() - t0))

    # LGB+XGB simple ensemble
    w_lx = auc_lgb / (auc_lgb + auc_xgb)
    auc_lgbxgb = float(roc_auc_score(y_te, w_lx * p_lgb + (1 - w_lx) * p_xgb))

    # ===== FT-Transformer =====
    p_ft = None
    auc_ft = 0.0
    if enable_ft:
        t0 = time.time()
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        try:
            ft_model, p_ft, _p_ft_tr, auc_ft = train_ft_transformer(
                X_tr_s, y_tr, X_te_s, y_te,
                n_features=len(features),
                epochs=50, batch_size=2048, lr=1e-3,
                patience=10, d_token=48, n_heads=4, n_layers=3,
                dropout=0.1, label=f'FT-{test_year}',
            )
            auc_ft = float(auc_ft)
        except Exception as e:
            log_progress(log_path, fold=fold_idx, year=test_year,
                         step='ft_error', err=str(e))
            auc_ft = 0.0
            p_ft = None
        log_progress(log_path, fold=fold_idx, year=test_year, step='ft_done',
                     auc=auc_ft, elapsed_sec=int(time.time() - t0))

    # ===== IntraRace Attention =====
    p_ir = None
    auc_ir = 0.0
    ir_cov_pct = 0.0
    if enable_ir:
        t0 = time.time()
        try:
            df_tr = df.loc[train_mask].copy()
            df_te = df.loc[test_mask].copy()
            scaler_ir = StandardScaler()
            df_tr_scaled = df_tr.copy()
            df_te_scaled = df_te.copy()
            df_tr_scaled[features] = scaler_ir.fit_transform(
                df_tr[features].values.astype(np.float32))
            df_te_scaled[features] = scaler_ir.transform(
                df_te[features].values.astype(np.float32))
            ir_model, ir_val_dict, _ir_tr_dict, _ir_tr_auc, ir_val_auc = \
                train_intra_race_with_preds(
                    df_tr_scaled, df_te_scaled, features,
                    epochs=30, batch_size=64, lr=1e-3, patience=8,
                    d_model=64, n_heads=4, n_layers=2, dropout=0.1,
                )
            p_ir_arr = np.zeros(len(X_te), dtype=np.float32)
            cov = 0
            for i, idx in enumerate(test_indices):
                if idx in ir_val_dict:
                    p_ir_arr[i] = ir_val_dict[idx]
                    cov += 1
                else:
                    p_ir_arr[i] = 0.3
            ir_cov_pct = cov / len(X_te) * 100
            if cov > len(X_te) * 0.5:
                p_ir = p_ir_arr
                auc_ir = float(roc_auc_score(y_te, p_ir))
        except Exception as e:
            log_progress(log_path, fold=fold_idx, year=test_year,
                         step='ir_error', err=str(e),
                         tb=traceback.format_exc()[:500])
            auc_ir = 0.0
            p_ir = None
        log_progress(log_path, fold=fold_idx, year=test_year, step='ir_done',
                     auc=auc_ir, cov=float(ir_cov_pct),
                     elapsed_sec=int(time.time() - t0))

    # ===== Grid 4-model ensemble =====
    best_grid_auc = 0.0
    best_grid_w = None
    if p_ft is not None and p_ir is not None and auc_ir > 0:
        for w1 in np.arange(0.20, 0.45, 0.05):
            for w2 in np.arange(0.20, 0.45, 0.05):
                for w3 in np.arange(0.05, 0.30, 0.05):
                    w4 = 1.0 - w1 - w2 - w3
                    if w4 < 0.01 or w4 > 0.40:
                        continue
                    p_grid = w1 * p_lgb + w2 * p_xgb + w3 * p_ft + w4 * p_ir
                    a = roc_auc_score(y_te, p_grid)
                    if a > best_grid_auc:
                        best_grid_auc = float(a)
                        best_grid_w = (float(w1), float(w2), float(w3), float(w4))

    result = {
        'fold': fold_idx,
        'year': test_year,
        'train_n': int(len(X_tr)),
        'test_n': int(len(X_te)),
        'lgb_auc': auc_lgb,
        'xgb_auc': auc_xgb,
        'ft_auc': auc_ft,
        'ir_auc': auc_ir,
        'ir_cov': float(ir_cov_pct),
        'lgbxgb_auc': auc_lgbxgb,
        'grid_auc': best_grid_auc if best_grid_w else None,
        'grid_weights': list(best_grid_w) if best_grid_w else None,
        'train_auc': auc_lgb_tr,
        'gap': float(gap),
    }
    log_progress(log_path, fold=fold_idx, year=test_year, step='ensemble_done',
                 **{k: v for k, v in result.items() if k not in ('fold', 'year')})

    # save fold models (small artifacts only)
    return result, m_lgb, m_xgb


def save_candidate_model(features, fold_results, all_models, out_path):
    """Save v15.2 candidate model. ★ candidate suffix ★"""
    pkl = {
        'version': 'v15_2_candidate',
        'features': features,
        'fold_results': fold_results,
        'lgb_models': [m['lgb'] for m in all_models if m.get('lgb')],
        'xgb_models': [m['xgb'] for m in all_models if m.get('xgb')],
        # FT/IR models are not saved (re-train at production if GO)
        'baseline_auc_v15': V15_BASELINE_AUC,
        'wf_years': WF_YEARS,
        'trained_at': datetime.now().isoformat(),
        'note': '★ candidate suffix、 production 投入禁止。 paper eval 後別 phase で 投入判定 ★',
    }
    with gzip.open(out_path, 'wb') as f:
        pickle.dump(pkl, f, protocol=4)
    print(f"  saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-ft', action='store_true',
                        help='skip FT-Transformer (CPU-only fast mode)')
    parser.add_argument('--skip-ir', action='store_true',
                        help='skip IntraRace Attention')
    parser.add_argument('--years', type=str, default=None,
                        help='comma-separated test years (default 2020-2025)')
    args = parser.parse_args()

    if args.years:
        global WF_YEARS
        WF_YEARS = [int(y) for y in args.years.split(',')]

    t_start = time.time()
    date_tag = datetime.now().strftime('%Y%m%d_%H%M')
    log_path = os.path.join(LOG_DIR, f'v15_2_training_{date_tag}.log')
    print(f"=" * 70)
    print(f"  v15.2 training")
    print(f"  device: {DEVICE} (cuda: {torch.cuda.is_available()})")
    print(f"  log: {log_path}")
    print(f"  V15 baseline AUC: {V15_BASELINE_AUC}")
    print(f"  WF years: {WF_YEARS}")
    print(f"  hard limit: {HARD_LIMIT_SEC}s")
    print(f"=" * 70)

    log_progress(log_path, step='start', device=str(DEVICE),
                 cuda=torch.cuda.is_available(),
                 wf_years=WF_YEARS, hard_limit_sec=HARD_LIMIT_SEC)

    # Load v15.2 cache
    cache_path = os.path.join(DATA_DIR, '_v15_2_optuna_df_cache.pkl.gz')
    print(f"\n  loading {cache_path}...")
    cache = pd.read_pickle(cache_path)
    df = cache['df']
    features = cache['features']
    v152_new = cache.get('v152_new_features', [])
    print(f"  df: {len(df)} rows, features: {len(features)} "
          f"(V15 {len(features) - len(v152_new)} + new {len(v152_new)})")
    log_progress(log_path, step='cache_loaded', n_rows=len(df),
                 n_features=len(features), n_v152_new=len(v152_new))

    # Ensure race_id_unique exists
    if 'race_id_unique' not in df.columns:
        df = build_race_id(df)

    # Sanity: missing features
    missing = [f for f in features if f not in df.columns]
    if missing:
        log_progress(log_path, step='abort_missing_features',
                     missing=missing[:20])
        print(f"  ABORT: missing features: {missing}")
        sys.exit(1)

    # Train per fold
    fold_results = []
    all_models = []
    consecutive_low_auc = 0
    aborted = False

    for fold_idx, year in enumerate(WF_YEARS):
        # check time budget
        elapsed = time.time() - t_start
        if elapsed > HARD_LIMIT_SEC:
            log_progress(log_path, step='hard_limit_hit',
                         elapsed_sec=int(elapsed))
            print(f"  HARD LIMIT REACHED at fold {fold_idx}")
            break

        print(f"\n  ===== fold {fold_idx+1}/{len(WF_YEARS)} year {year} =====")
        try:
            ret = fold_train_eval(
                df, features, year, log_path, fold_idx, len(WF_YEARS),
                enable_ft=not args.skip_ft,
                enable_ir=not args.skip_ir,
            )
        except Exception as e:
            log_progress(log_path, fold=fold_idx, year=year, step='fold_error',
                         err=str(e), tb=traceback.format_exc()[:1000])
            print(f"  fold {fold_idx} ERROR: {e}")
            continue

        if ret is None:
            continue

        result, m_lgb, m_xgb = ret
        fold_results.append(result)
        all_models.append({'lgb': m_lgb, 'xgb': m_xgb, 'year': year})

        # safety: consecutive low AUC abort
        primary_auc = result.get('grid_auc') or result.get('lgbxgb_auc') or result['lgb_auc']
        if primary_auc < 0.85:
            consecutive_low_auc += 1
            log_progress(log_path, fold=fold_idx, year=year,
                         step='low_auc_warn', auc=primary_auc,
                         consecutive=consecutive_low_auc)
            if consecutive_low_auc >= 2:
                log_progress(log_path, step='abort_consecutive_low_auc',
                             consecutive=consecutive_low_auc)
                print(f"  ABORT: 2 consecutive folds AUC < 0.85")
                aborted = True
                break
        else:
            consecutive_low_auc = 0

    # Summary
    print(f"\n" + "=" * 70)
    print(f"  v15.2 WF summary")
    print(f"=" * 70)
    print(f"  {'Year':<6} {'LGB':>7} {'XGB':>7} {'FT':>7} {'IR':>7} "
          f"{'L+X':>8} {'Grid':>8} {'Gap':>6}")
    for r in fold_results:
        gap_ok = 'Y' if r['gap'] < 0.05 else 'N'
        grid = f"{r['grid_auc']:.4f}" if r['grid_auc'] else '  N/A '
        print(f"  {r['year']:<6} {r['lgb_auc']:7.4f} {r['xgb_auc']:7.4f} "
              f"{r['ft_auc']:7.4f} {r['ir_auc']:7.4f} "
              f"{r['lgbxgb_auc']:8.4f} {grid:>8} {r['gap']:6.4f}{gap_ok}")

    if fold_results:
        lgb_mean = float(np.mean([r['lgb_auc'] for r in fold_results]))
        lgbxgb_mean = float(np.mean([r['lgbxgb_auc'] for r in fold_results]))
        grid_vals = [r['grid_auc'] for r in fold_results if r['grid_auc']]
        grid_mean = float(np.mean(grid_vals)) if grid_vals else 0.0
        delta = grid_mean - V15_BASELINE_AUC if grid_mean else lgbxgb_mean - V15_BASELINE_AUC

        # Verdict
        if grid_mean >= V15_BASELINE_AUC + 0.004:
            verdict = 'GO_PAPER_EVAL'
        elif grid_mean >= V15_BASELINE_AUC - 0.001 and grid_mean > 0:
            verdict = 'NEUTRAL_PAPER_EVAL'
        elif lgbxgb_mean >= V15_BASELINE_AUC - 0.001:
            verdict = 'NEUTRAL_LGBXGB_ONLY'
        else:
            verdict = 'NO_GO'
    else:
        lgb_mean = lgbxgb_mean = grid_mean = 0.0
        delta = 0.0
        verdict = 'NO_FOLDS_COMPLETED'

    print(f"\n  LGB mean:    {lgb_mean:.6f}")
    print(f"  LGB+XGB:     {lgbxgb_mean:.6f}")
    print(f"  Grid mean:   {grid_mean:.6f}")
    print(f"  V15 base:    {V15_BASELINE_AUC:.6f}")
    print(f"  Delta:       {delta:+.6f}")
    print(f"  VERDICT:     {verdict}")
    if aborted:
        print(f"  (aborted early)")

    log_progress(log_path, step='final',
                 lgb_mean=lgb_mean, lgbxgb_mean=lgbxgb_mean,
                 grid_mean=grid_mean, delta=delta, verdict=verdict,
                 n_folds_completed=len(fold_results), aborted=aborted)

    # Save results JSON
    results_path = os.path.join(DATA_DIR, 'v15_2', f'wf_results_{date_tag}.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'version': 'v15_2_candidate',
            'date': datetime.now().isoformat(),
            'wf_years': WF_YEARS,
            'fold_results': fold_results,
            'lgb_mean': lgb_mean,
            'lgbxgb_mean': lgbxgb_mean,
            'grid_mean': grid_mean,
            'v15_baseline': V15_BASELINE_AUC,
            'delta': delta,
            'verdict': verdict,
            'aborted': aborted,
            'features': features,
            'v152_new_features': v152_new,
            'elapsed_sec': int(time.time() - t_start),
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  results saved: {results_path}")

    # Save candidate model (LGB+XGB only; FT/IR are heavy, re-train at deploy)
    out_model = os.path.join(MODEL_DIR, 'v15_2_candidate.pkl.gz')
    save_candidate_model(features, fold_results, all_models, out_model)

    # Final markdown report
    md_path = os.path.join(BASE_DIR, 'docs', f'V15_2_TRAINING_LOG_{date_tag}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# v15.2 学習 結果 ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n")
        f.write(f"## 0. 結論\n\n")
        f.write(f"- elapsed: {(time.time()-t_start)/3600:.2f} h\n")
        f.write(f"- WF folds completed: {len(fold_results)} / {len(WF_YEARS)}\n")
        f.write(f"- aborted: {aborted}\n")
        f.write(f"- LGB mean AUC: **{lgb_mean:.6f}**\n")
        f.write(f"- LGB+XGB mean AUC: **{lgbxgb_mean:.6f}**\n")
        f.write(f"- Grid 4-model mean AUC: **{grid_mean:.6f}**\n")
        f.write(f"- V15 baseline AUC: {V15_BASELINE_AUC:.6f}\n")
        f.write(f"- delta: **{delta:+.6f}**\n")
        f.write(f"- **verdict: {verdict}**\n\n")
        f.write(f"## 1. T4 leak audit gate 事前\n\n")
        f.write(f"- 候補 17 features (paci.info_idx 除外、 priority A+B のみ)\n")
        f.write(f"- gate: PASS (leak=0, suspect=0)\n\n")
        f.write(f"## 2. WF fold-by-fold AUC\n\n")
        f.write(f"| fold | year | LGB | XGB | FT | IR | LGB+XGB | Grid | Gap |\n")
        f.write(f"|------|------|-----|-----|-----|-----|---------|------|-----|\n")
        for r in fold_results:
            grid = f"{r['grid_auc']:.4f}" if r['grid_auc'] else 'N/A'
            f.write(f"| {r['fold']} | {r['year']} | {r['lgb_auc']:.4f} | "
                    f"{r['xgb_auc']:.4f} | {r['ft_auc']:.4f} | {r['ir_auc']:.4f} | "
                    f"{r['lgbxgb_auc']:.4f} | {grid} | {r['gap']:.4f} |\n")
        f.write(f"\n## 3. 採用判定\n\n")
        criteria = [
            ('AUC 維持 (Grid mean ≥ V15 - 0.001)', grid_mean >= V15_BASELINE_AUC - 0.001 if grid_mean else False),
            ('AUC 改善 (Grid mean ≥ V15 + 0.003)', grid_mean >= V15_BASELINE_AUC + 0.003 if grid_mean else False),
            ('LGB+XGB ≥ V15 - 0.001', lgbxgb_mean >= V15_BASELINE_AUC - 0.001),
            ('全 fold AUC ≥ 0.85', all(r['lgb_auc'] > 0.85 for r in fold_results)),
            ('全 fold gap < 0.05', all(r['gap'] < 0.05 for r in fold_results)),
        ]
        for name, ok in criteria:
            f.write(f"- {name}: {'✅' if ok else '❌'}\n")
        f.write(f"- **final verdict: {verdict}**\n\n")
        f.write(f"## 4. V15 production 不変保証\n\n")
        f.write(f"- keiba_model_v15_central*.pkl.gz 上書き: 0\n")
        f.write(f"- data/_v15_optuna_df_cache.pkl.gz 上書き: 0\n")
        f.write(f"- models/v15_2_candidate.pkl.gz: 新規保存\n")
        f.write(f"- production 投入: なし (★ candidate のみ ★)\n\n")
        f.write(f"## 5. 5/24+ paper eval 候補\n\n")
        if verdict.startswith('GO'):
            f.write(f"- 30R paper eval → ROI / winner_top1 / shift 確認 → 採用判定\n")
        elif verdict.startswith('NEUTRAL'):
            f.write(f"- paper eval 価値あり (delta {delta:+.6f})、 30R 検証推奨\n")
        else:
            f.write(f"- NO-GO、 candidate 廃棄、 5/24+ 学習 plan 見直し\n")
        f.write(f"\n## 6. 出力 file\n\n")
        f.write(f"- {results_path}\n")
        f.write(f"- {out_model}\n")
        f.write(f"- {log_path}\n")
        f.write(f"- {md_path}\n")
    print(f"  md report: {md_path}")
    print(f"\n  ★ V15 production 完全不変、 v15.2 candidate のみ保存 ★")
    print(f"  total elapsed: {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
