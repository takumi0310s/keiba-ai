#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B4: Confidence calibration (isotonic + Platt scaling).

V15 4-ensemble の出力 (top3 確率) は miscalibrated の可能性。 isotonic regression と
Platt scaling で 真の P(top3) ≒ 出力 になるよう校正。 Kelly / EV 計算の前提。

【V15 投資保護】 predict_core.py / V15 model 一切触らず、 predict 出力に対する **post-process** 適用。

Usage:
    # 校正器学習 (CSV: pred,label の 2 列)
    python tools/calibrate_confidence.py fit --input data/backtest_preds.csv --out data/calibrator_v15.pkl

    # 校正適用 (predicted_prob → calibrated_prob)
    python tools/calibrate_confidence.py apply --input data/v15_predictions.csv --calibrator data/calibrator_v15.pkl --out data/v15_preds_calibrated.csv

    # 動作確認 (synthetic data で demo)
    python tools/calibrate_confidence.py demo

Input CSV (fit): pred (V15 model output 0-1), label (actual top3 = 1/0)
Output: calibrator pickle (isotonic + Platt 両方保存、 適用時に選択可)

【参考】 V15 (0.8939 AUC) でも calibration error は典型的に 5-15% あり得る。
Brier Score / ECE で改善幅を測定。
"""
import argparse
import json
import os
import pickle
import sys
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_metrics(y_true, y_pred):
    """Brier Score + ECE (Expected Calibration Error) 計算."""
    import numpy as np
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    brier = float(np.mean((y_pred - y_true) ** 2))

    # ECE: 10 bin
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    n = len(y_pred)
    for i in range(10):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i+1])
        if i == 9:
            mask = (y_pred >= bins[i]) & (y_pred <= bins[i+1])
        m = mask.sum()
        if m == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_pred[mask].mean()
        ece += (m / n) * abs(bin_acc - bin_conf)
    return brier, float(ece)


def fit_calibrators(y_true, y_pred):
    """Isotonic regression + Platt scaling 両方学習."""
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_pred, y_true)

    # Platt: sigmoid(a * x + b)
    platt = LogisticRegression(C=1e9, solver='lbfgs')
    platt.fit(np.asarray(y_pred).reshape(-1, 1), y_true)

    return iso, platt


def apply_calibrator(calibrator, y_pred, method='isotonic'):
    """Apply calibrator. method = 'isotonic' or 'platt'."""
    import numpy as np
    y_pred = np.asarray(y_pred).reshape(-1, 1) if method == 'platt' else np.asarray(y_pred)
    if method == 'isotonic':
        return calibrator['isotonic'].transform(y_pred)
    elif method == 'platt':
        return calibrator['platt'].predict_proba(y_pred)[:, 1]
    else:
        raise ValueError(f'unknown method: {method}')


def cmd_fit(args):
    import pandas as pd
    df = pd.read_csv(args.input)
    if 'pred' not in df.columns or 'label' not in df.columns:
        print('[ERROR] CSV must have columns: pred, label')
        return 1
    y_pred = df['pred'].values
    y_true = df['label'].values
    print(f'[INFO] training samples: {len(y_pred)}')

    brier_raw, ece_raw = compute_metrics(y_true, y_pred)
    print(f'[BEFORE] Brier={brier_raw:.4f}, ECE={ece_raw:.4f}')

    iso, platt = fit_calibrators(y_true, y_pred)
    calibrator = {'isotonic': iso, 'platt': platt}

    iso_pred = apply_calibrator(calibrator, y_pred, 'isotonic')
    platt_pred = apply_calibrator(calibrator, y_pred, 'platt')
    brier_iso, ece_iso = compute_metrics(y_true, iso_pred)
    brier_platt, ece_platt = compute_metrics(y_true, platt_pred)
    print(f'[AFTER iso ] Brier={brier_iso:.4f}, ECE={ece_iso:.4f}')
    print(f'[AFTER platt] Brier={brier_platt:.4f}, ECE={ece_platt:.4f}')

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({
            'isotonic': iso,
            'platt': platt,
            'metrics': {
                'before': {'brier': brier_raw, 'ece': ece_raw},
                'after_iso': {'brier': brier_iso, 'ece': ece_iso},
                'after_platt': {'brier': brier_platt, 'ece': ece_platt},
            },
            'trained_at': datetime.now().isoformat(),
            'n_samples': int(len(y_pred)),
        }, f)
    print(f'[OK] calibrator saved: {out_path}')
    return 0


def cmd_apply(args):
    import pandas as pd
    df = pd.read_csv(args.input)
    if 'pred' not in df.columns:
        print('[ERROR] CSV must have column: pred')
        return 1

    with open(args.calibrator, 'rb') as f:
        cal = pickle.load(f)

    df['pred_calibrated_iso'] = apply_calibrator(cal, df['pred'].values, 'isotonic')
    df['pred_calibrated_platt'] = apply_calibrator(cal, df['pred'].values, 'platt')

    df.to_csv(args.out, index=False)
    print(f'[OK] calibrated predictions saved: {args.out}')
    return 0


def cmd_demo(args):
    """Synthetic data で fit + apply のフルパス動作確認."""
    import numpy as np
    np.random.seed(42)
    n = 2000

    # 真の確率 p_true
    p_true = np.random.beta(2, 8, n)
    # ラベル sampling
    y_true = (np.random.rand(n) < p_true).astype(int)
    # 観測 model = miscalibrated (中域 寄せ + sigmoid 歪み)
    y_pred = 0.5 * p_true + 0.5 * (p_true ** 0.5)
    y_pred = np.clip(y_pred + np.random.normal(0, 0.05, n), 0.001, 0.999)

    brier_raw, ece_raw = compute_metrics(y_true, y_pred)
    print(f'[DEMO BEFORE] Brier={brier_raw:.4f}, ECE={ece_raw:.4f}')

    iso, platt = fit_calibrators(y_true, y_pred)
    cal = {'isotonic': iso, 'platt': platt}
    iso_pred = apply_calibrator(cal, y_pred, 'isotonic')
    platt_pred = apply_calibrator(cal, y_pred, 'platt')
    brier_iso, ece_iso = compute_metrics(y_true, iso_pred)
    brier_platt, ece_platt = compute_metrics(y_true, platt_pred)
    print(f'[DEMO AFTER iso ] Brier={brier_iso:.4f}, ECE={ece_iso:.4f} (-{ece_raw-ece_iso:.4f})')
    print(f'[DEMO AFTER platt] Brier={brier_platt:.4f}, ECE={ece_platt:.4f} (-{ece_raw-ece_platt:.4f})')
    print('[DEMO] fit + apply 動作確認 OK')
    return 0


def main():
    ap = argparse.ArgumentParser(description='Confidence calibration (B4)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    fit_p = sub.add_parser('fit', help='Fit calibrator from CSV (pred, label)')
    fit_p.add_argument('--input', required=True, help='CSV with columns pred, label')
    fit_p.add_argument('--out', default='data/calibrator_v15.pkl', help='output pickle path')

    app_p = sub.add_parser('apply', help='Apply calibrator to CSV')
    app_p.add_argument('--input', required=True)
    app_p.add_argument('--calibrator', required=True)
    app_p.add_argument('--out', required=True)

    sub.add_parser('demo', help='Synthetic data で動作確認')

    args = ap.parse_args()
    if args.cmd == 'fit':
        return cmd_fit(args)
    elif args.cmd == 'apply':
        return cmd_apply(args)
    elif args.cmd == 'demo':
        return cmd_demo(args)
    return 1


if __name__ == '__main__':
    sys.exit(main())
