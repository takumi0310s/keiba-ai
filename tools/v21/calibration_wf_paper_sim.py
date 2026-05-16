#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sub-task 15: Paper simulation of 4 calibration methods on LIVE V15 data.

Joins data/daily_predictions/YYYYMMDD.csv (top1_score) +
       data/daily_results/YYYYMMDD.csv  (top1_finish, trio_payout, profit, investment)

For each calibration method (Platt/Isotonic/Beta/Temperature):
  - calibrator trained on WF OOF predictions (data/v21/calibration_wf/wf_oof_predictions.parquet)
  - apply calibrator to live top1_score → p_cal
  - simulate ROI under raw vs calibrated EV scoring (paper, no production impact)

ROI is the actually-recorded profit/investment (V15 strategy was already executed),
so calibration here is *informational* — it tells us if p_cal-ranking changes
which races would have been picked under same EV threshold.

NEVER touched: production files, scheduler, V15 model.

Output: data/v21/calibration_wf/paper_sim_summary.json + paper_sim_per_method.csv

Usage:
    python tools/v21/calibration_wf_paper_sim.py
"""
import argparse
import glob
import json
import os
import pickle
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WF_DIR = os.path.join(BASE_DIR, 'data', 'v21', 'calibration_wf')


def load_oof():
    p = os.path.join(WF_DIR, 'wf_oof_predictions.parquet')
    if os.path.exists(p):
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    p = os.path.join(WF_DIR, 'wf_oof_predictions.csv')
    if os.path.exists(p):
        return pd.read_csv(p)
    raise FileNotFoundError('WF OOF predictions not found — run calibration_wf_reeval.py first')


def fit_all_calibrators(y, p):
    """Train Platt, Isotonic, Beta, Temperature on (y, p_raw)."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)

    # Platt
    platt = LogisticRegression(C=1e9, solver='lbfgs', max_iter=1000)
    platt.fit(p.reshape(-1, 1), y)

    # Isotonic
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p, y)

    # Beta
    X = np.column_stack([np.log(p), -np.log(1 - p)])
    beta = LogisticRegression(C=1e9, solver='lbfgs', max_iter=1000)
    beta.fit(X, y)

    # Temperature
    import torch, torch.nn as nn
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logit = np.log(p / (1 - p)).astype(np.float32)
    yf = y.astype(np.float32)
    logit_t = torch.from_numpy(logit).to(dev)
    y_t = torch.from_numpy(yf).to(dev)
    T = nn.Parameter(torch.ones(1, device=dev))
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=200)
    bce = nn.BCEWithLogitsLoss()
    def closure():
        optimizer.zero_grad()
        loss = bce(logit_t / T, y_t)
        loss.backward()
        return loss
    optimizer.step(closure)
    temp = {'T': float(T.detach().cpu().item())}

    return {
        'platt': platt,
        'isotonic': iso,
        'beta': beta,
        'temperature': temp,
    }


def apply_calibrator(name, cal, p):
    eps = 1e-12
    p = np.clip(np.asarray(p), eps, 1 - eps)
    if name == 'platt':
        return cal.predict_proba(p.reshape(-1, 1))[:, 1]
    if name == 'isotonic':
        return cal.transform(p)
    if name == 'beta':
        X = np.column_stack([np.log(p), -np.log(1 - p)])
        return cal.predict_proba(X)[:, 1]
    if name == 'temperature':
        T = cal['T']
        logit = np.log(p / (1 - p))
        return 1.0 / (1.0 + np.exp(-logit / T))
    raise ValueError(name)


def load_live_data():
    """Join daily_predictions + daily_results on race_id (central only, non-NAR)."""
    pred_dir = os.path.join(BASE_DIR, 'data', 'daily_predictions')
    res_dir = os.path.join(BASE_DIR, 'data', 'daily_results')

    pred_files = sorted(glob.glob(os.path.join(pred_dir, '2026*.csv')))
    res_files = sorted(glob.glob(os.path.join(res_dir, '2026*.csv')))

    # exclude NAR files and partials
    pred_files = [f for f in pred_files if 'nar_' not in os.path.basename(f).lower()
                  and '_prerace' not in os.path.basename(f).lower()
                  and '.bak' not in os.path.basename(f).lower()]
    res_files = [f for f in res_files if 'nar_' not in os.path.basename(f).lower()
                 and '.bak' not in os.path.basename(f).lower()]

    preds = []
    for f in pred_files:
        date = os.path.basename(f).replace('.csv', '')
        try:
            df = pd.read_csv(f, encoding='utf-8')
            if 'top1_score' not in df.columns or 'race_id' not in df.columns:
                continue
            df['date'] = date
            preds.append(df[['date', 'race_id', 'top1_score', 'condition', 'num_horses', 'distance']])
        except Exception as e:
            print(f'  WARN: pred {f}: {e}')

    results = []
    for f in res_files:
        date = os.path.basename(f).replace('.csv', '')
        try:
            df = pd.read_csv(f, encoding='utf-8')
            if 'race_id' not in df.columns:
                continue
            df['date'] = date
            keep_cols = ['date', 'race_id', 'top1_finish', 'top2_finish', 'top3_finish',
                         'trio_hit', 'trio_payout', 'umaren_hit', 'umaren_payout',
                         'investment', 'profit', 'bet_type']
            keep_cols = [c for c in keep_cols if c in df.columns]
            results.append(df[keep_cols])
        except Exception as e:
            print(f'  WARN: result {f}: {e}')

    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    res_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    print(f'  loaded {len(pred_df)} pred rows, {len(res_df)} result rows')

    # join on (date, race_id)
    pred_df['race_id'] = pred_df['race_id'].astype(str)
    res_df['race_id'] = res_df['race_id'].astype(str)
    merged = pred_df.merge(res_df, on=['date', 'race_id'], how='inner', suffixes=('', '_r'))
    print(f'  inner join: {len(merged)} rows')

    # label: top3 hit at top1 → did top1 horse finish <=3 (proxy for V15 confidence calibration target)
    def _to_int_safe(x):
        try:
            return int(float(x))
        except Exception:
            return np.nan
    merged['top1_finish_int'] = merged['top1_finish'].apply(_to_int_safe)
    merged['top1_is_top3'] = (merged['top1_finish_int'] <= 3).astype(int)
    merged.loc[merged['top1_finish_int'].isna(), 'top1_is_top3'] = np.nan

    # clean numeric
    for c in ['top1_score', 'investment', 'profit', 'trio_payout', 'umaren_payout']:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors='coerce')

    merged = merged.dropna(subset=['top1_score', 'top1_is_top3'])
    print(f'  after dropna: {len(merged)} rows (with score+label)')
    return merged


def compute_metrics_for_method(df, name, cal):
    p_raw = df['top1_score'].values
    if name == 'baseline':
        p_cal = p_raw
    else:
        p_cal = apply_calibrator(name, cal, p_raw)

    y = df['top1_is_top3'].values
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, p_cal)) if len(set(y)) > 1 else float('nan')
    brier = float(np.mean((p_cal - y) ** 2))
    # ECE
    bins = np.linspace(0, 1, 11)
    n = len(p_cal)
    ec = 0.0
    for i in range(10):
        if i == 9:
            mask = (p_cal >= bins[i]) & (p_cal <= bins[i+1])
        else:
            mask = (p_cal >= bins[i]) & (p_cal < bins[i+1])
        m = int(mask.sum())
        if m == 0:
            continue
        ec += (m / n) * abs(y[mask].mean() - p_cal[mask].mean())

    # ROI simulation: for each row, V15 has already invested 'investment' yen.
    # We simulate: bet if p_cal >= threshold (mirror real strategy threshold).
    # Baseline strategy (現実): all races where top1_score was deemed sufficient — production already filtered.
    # For paper sim, we use the same set of rows (no re-filter) but report what calibration says.
    inv = df['investment'].fillna(0).values
    profit = df['profit'].fillna(0).values
    realized_pnl = float(profit.sum())
    realized_inv = float(inv.sum())
    realized_roi = float((realized_inv + realized_pnl) / realized_inv) if realized_inv > 0 else float('nan')

    # Threshold sweep: bet only rows with p_cal >= th
    thresholds = [0.0, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50]
    th_results = []
    for th in thresholds:
        mask = p_cal >= th
        m_inv = float(inv[mask].sum())
        m_pnl = float(profit[mask].sum())
        roi = float((m_inv + m_pnl) / m_inv) if m_inv > 0 else float('nan')
        th_results.append({
            'threshold': float(th),
            'n_bets': int(mask.sum()),
            'investment': m_inv,
            'profit': m_pnl,
            'roi': roi,
        })

    return {
        'method': name,
        'n_samples': int(len(df)),
        'auc': auc,
        'brier': brier,
        'ece': float(ec),
        'p_cal_min': float(p_cal.min()),
        'p_cal_max': float(p_cal.max()),
        'p_cal_mean': float(p_cal.mean()),
        'p_cal_std': float(p_cal.std()),
        'realized_pnl': realized_pnl,
        'realized_inv': realized_inv,
        'realized_roi': realized_roi,
        'threshold_sweep': th_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof-train-year', type=int, default=None,
                    help='Use only this year of OOF for calibrator training (default: all years)')
    args = ap.parse_args()

    print("=" * 70)
    print("  Sub-task 15: Paper simulation")
    print("=" * 70)

    print('\n[1/4] Loading WF OOF predictions...')
    oof = load_oof()
    print(f'  loaded {len(oof)} rows, years: {sorted(oof["year"].unique())}')

    if args.oof_train_year is not None:
        oof = oof[oof['year'] == args.oof_train_year]
        print(f'  filter year={args.oof_train_year}: {len(oof)} rows')

    print('\n[2/4] Fitting calibrators on WF OOF...')
    y_train = oof['y_true'].values.astype(int)
    p_train = oof['p_raw'].values.astype(float)
    cals = fit_all_calibrators(y_train, p_train)
    print(f'  trained: {list(cals.keys())}')

    print('\n[3/4] Loading live data (daily_predictions + daily_results)...')
    live = load_live_data()
    if len(live) == 0:
        print('  ERROR: no live data')
        return 1

    print(f'\n  live label distribution: {live["top1_is_top3"].value_counts().to_dict()}')
    print(f'  raw top1_score: min={live["top1_score"].min():.4f}, '
          f'max={live["top1_score"].max():.4f}, '
          f'mean={live["top1_score"].mean():.4f}')

    print('\n[4/4] Evaluating each method on live data...')
    results = {}
    for name in ['baseline', 'platt', 'isotonic', 'beta', 'temperature']:
        cal = cals.get(name) if name != 'baseline' else None
        r = compute_metrics_for_method(live, name, cal)
        results[name] = r
        print(f'  {name:<12}: AUC={r["auc"]:.4f}, Brier={r["brier"]:.4f}, '
              f'ECE={r["ece"]:.4f}, '
              f'p_cal_range=[{r["p_cal_min"]:.3f}, {r["p_cal_max"]:.3f}], '
              f'realized_ROI={r["realized_roi"]:.3f}')

    # Save
    out_json = os.path.join(WF_DIR, 'paper_sim_summary.json')
    out = {
        'meta': {
            'task': 'Sub-task 15 paper sim',
            'oof_train_year': args.oof_train_year,
            'n_oof_train': int(len(p_train)),
            'n_live_samples': int(len(live)),
            'live_date_range': [str(live['date'].min()), str(live['date'].max())],
            'generated_at': datetime.now().isoformat(),
        },
        'results': results,
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print(f'\n  paper_sim saved: {out_json}')

    # Per-method threshold sweep csv for easy reading
    rows = []
    for name, r in results.items():
        for s in r['threshold_sweep']:
            rows.append({
                'method': name,
                'threshold': s['threshold'],
                'n_bets': s['n_bets'],
                'investment': s['investment'],
                'profit': s['profit'],
                'roi': s['roi'],
            })
    pd.DataFrame(rows).to_csv(os.path.join(WF_DIR, 'paper_sim_threshold_sweep.csv'),
                              index=False, encoding='utf-8')
    print(f'  threshold sweep saved.')

    print("\n[OK] Sub-task 15 paper sim complete.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
