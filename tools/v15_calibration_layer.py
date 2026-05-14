"""V15 prediction prob → 真の確率 校正 layer.

V15 production の prob output は logistic で 過信 傾向 (well-known issue)。
data/calibrator_v15_pilot.pkl (Isotonic + Platt) で 真の確率 化 → EV 計算精度 up。

V15 .pkl.gz / predict_core.py / daily_predict.py 完全不変。
本 module は **後段 layer**、 V15 prob を入力 → 校正後 prob 出力。

Usage:
    from tools.v15_calibration_layer import calibrate_v15_prob, calibrated_ev

    calibrated = calibrate_v15_prob(raw_prob, method='isotonic')
    ev = calibrated_ev(prob_calibrated, odds, place_factor=0.8)
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
CALIBRATOR_PKL = BASE / 'data' / 'calibrator_v15_pilot.pkl'

_calib_cache = None


def _load_calibrator():
    global _calib_cache
    if _calib_cache is not None:
        return _calib_cache
    if not CALIBRATOR_PKL.exists():
        print(f'[WARN] calibrator not found: {CALIBRATOR_PKL}')
        return None
    with open(CALIBRATOR_PKL, 'rb') as f:
        _calib_cache = pickle.load(f)
    return _calib_cache


def calibrate_v15_prob(raw_prob, method: str = 'isotonic'):
    """V15 raw prob を 校正.

    Args:
        raw_prob: float or np.ndarray、 V15 model.predict() output (top3 確率)
        method: 'isotonic' (ECE 0.000、 推奨) or 'platt' (ECE 0.005)

    Returns:
        校正後 prob (same shape)
    """
    calib = _load_calibrator()
    if calib is None:
        return raw_prob

    if method == 'isotonic':
        model = calib.get('isotonic')
        if model is None:
            return raw_prob
        if np.isscalar(raw_prob):
            return float(model.transform([raw_prob])[0])
        return model.transform(np.asarray(raw_prob))
    elif method == 'platt':
        model = calib.get('platt')
        if model is None:
            return raw_prob
        # LogisticRegression expects 2D input
        scalar_input = np.isscalar(raw_prob)
        x = np.asarray(raw_prob).reshape(-1, 1)
        out = model.predict_proba(x)[:, 1]
        return float(out[0]) if scalar_input else out
    else:
        raise ValueError(f'unknown method: {method}')


def calibrated_ev(prob, odds, ticket_type: str = 'trio',
                  place_factor: float = 1.0) -> float:
    """校正後 prob と オッズ から EV 計算.

    EV = prob * odds * place_factor - 1 (買い目 1 単位 = 100 円 想定)
    EV > 1.0 → 期待値 +、 EV > 1.2 → 強推奨、 EV < 0.8 → 取消検討

    Args:
        prob: 校正後 prob (0-1)
        odds: 配当 倍率 (例 trio 30倍 → 30.0)
        ticket_type: 'trio' / 'umaren' / 'wide' / 'tansho'
        place_factor: 0.8 = 三連複 7点で 1 点 hit、 0.5 = ワイド 4点で 1 点等

    Returns:
        EV (期待値、 1.0 が break-even)
    """
    if prob is None or odds is None or odds <= 0:
        return 0.0
    return float(prob) * float(odds) * float(place_factor)


def get_calibration_metrics() -> dict:
    """Calibration の before / after metrics 確認用."""
    calib = _load_calibrator()
    if calib is None:
        return {}
    return calib.get('metrics', {})


if __name__ == '__main__':
    # 簡単 test
    metrics = get_calibration_metrics()
    print('=== V15 calibrator metrics ===')
    for stage, m in metrics.items():
        print(f'  {stage}: {m}')

    # raw prob 0.3 / 0.6 / 0.85 を 校正
    print('\n=== Calibration sample ===')
    for raw in [0.2, 0.3, 0.5, 0.7, 0.85, 0.95]:
        iso = calibrate_v15_prob(raw, 'isotonic')
        platt = calibrate_v15_prob(raw, 'platt')
        print(f'  raw={raw:.2f} → iso={iso:.4f} platt={platt:.4f}')

    # EV 例 (trio 7点、 当該 1 点が hit)
    print('\n=== EV example (trio 7点 で TOP1 軸馬) ===')
    odds_list = [10, 20, 50, 100]
    for raw_p in [0.4, 0.6, 0.8]:
        for odds in odds_list:
            iso_p = calibrate_v15_prob(raw_p, 'isotonic')
            ev_raw = calibrated_ev(raw_p, odds, place_factor=1.0/7)
            ev_iso = calibrated_ev(iso_p, odds, place_factor=1.0/7)
            print(f'  raw_p={raw_p:.2f}, odds={odds}: EV_raw={ev_raw:.3f}, EV_iso={ev_iso:.3f}')
