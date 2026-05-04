"""v18/v19 Platt scaling 試作.

OOS 2025 data を使って calibrate (sigmoid Platt scaling)。
calibrated 確率で reliability bin を再計算 → before/after 比較。
calibrator (LR) を pickle 保存。

Usage:
  python tools/calibrate_v18_v19.py
"""
import sys, os, io, json, pickle
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)


def reliability_bins(y_true, p, bins=None):
    if bins is None:
        bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01]
    rows = []
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p < hi)
        n = mask.sum()
        if n == 0: continue
        actual = y_true[mask].mean()
        mean_p = p[mask].mean()
        rows.append({'bin': f'[{lo:.2f},{hi:.2f})', 'n': int(n),
                      'mean_p': float(mean_p), 'actual': float(actual),
                      'gap': float(actual - mean_p)})
    return rows


def platt_calibrate(p_raw, y):
    """Platt scaling: 1D logistic regression on raw probabilities."""
    # Use logit of p as feature for stability
    eps = 1e-8
    p_clipped = np.clip(p_raw, eps, 1-eps)
    X = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)  # logit
    lr = LogisticRegression(C=1e8, solver='lbfgs')
    lr.fit(X, y)
    return lr


def apply_calibrator(lr, p_raw):
    eps = 1e-8
    p_clipped = np.clip(p_raw, eps, 1-eps)
    X = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)
    return lr.predict_proba(X)[:, 1]


def main():
    print("=" * 60)
    print(f"v18/v19 Platt Scaling Calibration ({datetime.now()})")
    print("=" * 60)

    results = {}

    for name, csv_path, target_col, model_save_path in [
        ('v18_tansho', 'data/v18/v18_tansho_oos_2025.csv', 'is_win', 'data/v18/models/v18_tansho_calibrator.pkl'),
        ('v19_fukusho', 'data/v18/v19_fukusho_oos_2025.csv', 'is_top3', 'data/v18/models/v19_fukusho_calibrator.pkl'),
    ]:
        if not os.path.exists(csv_path):
            print(f"\n  {name}: SKIP - {csv_path} 不在")
            continue
        df = pd.read_csv(csv_path)
        y = df[target_col].astype(int).values
        p_raw = df['p_ens'].astype(float).values
        n = len(df)
        print(f"\n--- {name} ---")
        print(f"  n={n}, base rate={y.mean():.4f}")

        # Before
        b_before = brier_score_loss(y, p_raw)
        ll_before = log_loss(y, np.clip(p_raw, 1e-7, 1-1e-7))
        print(f"  BEFORE: Brier={b_before:.4f}, LogLoss={ll_before:.4f}")

        rb_before = reliability_bins(y, p_raw)
        print(f"  Reliability bins (before):")
        for r in rb_before:
            print(f"    {r['bin']:15s} n={r['n']:>6} mean_p={r['mean_p']:.3f} actual={r['actual']:.3f} gap={r['gap']:+.3f}")

        # Train Platt
        lr = platt_calibrate(p_raw, y)
        coef, intercept = float(lr.coef_[0][0]), float(lr.intercept_[0])
        print(f"  Platt: coef={coef:.4f}, intercept={intercept:.4f}")

        # After
        p_cal = apply_calibrator(lr, p_raw)
        b_after = brier_score_loss(y, p_cal)
        ll_after = log_loss(y, np.clip(p_cal, 1e-7, 1-1e-7))
        print(f"  AFTER: Brier={b_after:.4f}, LogLoss={ll_after:.4f}")
        print(f"  Δ: Brier {b_after-b_before:+.4f}, LogLoss {ll_after-ll_before:+.4f}")

        rb_after = reliability_bins(y, p_cal)
        print(f"  Reliability bins (after):")
        for r in rb_after:
            print(f"    {r['bin']:15s} n={r['n']:>6} mean_p={r['mean_p']:.3f} actual={r['actual']:.3f} gap={r['gap']:+.3f}")

        # Save
        os.makedirs('data/v18/models', exist_ok=True)
        with open(model_save_path, 'wb') as f:
            pickle.dump(lr, f)
        print(f"  Saved {model_save_path}")

        results[name] = {
            'n': n,
            'base_rate': float(y.mean()),
            'brier_before': b_before, 'brier_after': b_after,
            'logloss_before': ll_before, 'logloss_after': ll_after,
            'platt_coef': coef, 'platt_intercept': intercept,
            'reliability_before': rb_before,
            'reliability_after': rb_after,
        }

    # Save summary
    with open('data/v18/calibration_5_4_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print("\nSaved data/v18/calibration_5_4_summary.json")


if __name__ == '__main__':
    main()
