"""
強-5: calibrator_overlay_v2 テスト (7 tests)

★ V15 production 不変、 paper eval 用 module のテスト ★
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.calibrator_overlay_v2 import (  # noqa: E402
    ODDS_BIN_LABELS,
    ODDS_BINS,
    PROB_MAX,
    PROB_MIN,
    IsotonicCalibrator,
    OddsBinCalibrator,
    VenueCalibrator,
    calibrate_v2,
    evaluate_calibration,
    crossval_brier,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture()
def synthetic_data():
    """合成 data: y_pred ほど y_true=1 が多い (monotonic 傾向)."""
    rng = np.random.default_rng(42)
    n = 200
    y_pred = rng.uniform(0.15, 0.85, n)
    # slightly noisy monotonic label
    noise = rng.normal(0, 0.1, n)
    y_true = (y_pred + noise > 0.5).astype(float)
    return y_pred, y_true


@pytest.fixture()
def venue_data(synthetic_data):
    """per-venue 用: 4 会場に分割 (各 50 サンプル)."""
    y_pred, y_true = synthetic_data
    venues = np.array(["東京"] * 50 + ["中山"] * 50 + ["阪神"] * 50 + ["京都"] * 50)
    return y_pred, y_true, venues


@pytest.fixture()
def odds_data(synthetic_data):
    """per-odds-bin 用: 各 bin に 30+ サンプルが入るよう odds を割り当て."""
    y_pred, y_true = synthetic_data
    n = len(y_pred)
    rng = np.random.default_rng(99)
    # spread across bins (0.0-∞) — ensure exact length matches y_pred
    bins_lo = [1.0, 1.5, 3.0, 5.0, 10.0, 20.0]
    bins_hi = [1.5, 3.0, 5.0, 10.0, 20.0, 50.0]
    n_bins = len(bins_lo)
    # allocate floor(n/n_bins) per bin, then pad last bin to reach n
    n_each = n // n_bins
    remainder = n - n_each * n_bins
    chunks = []
    for i, (lo, hi) in enumerate(zip(bins_lo, bins_hi)):
        size = n_each + (1 if i < remainder else 0)
        chunks.append(rng.uniform(lo, hi, size))
    odds = np.concatenate(chunks)
    assert len(odds) == n, f"odds length mismatch: {len(odds)} != {n}"
    return y_pred, y_true, odds


# ============================================================
# Test 1: IsotonicCalibrator fit + predict
# ============================================================

def test_isotonic_calibrator_fit_predict(synthetic_data):
    """fit 後 predict が [PROB_MIN, PROB_MAX] 範囲内の値を返す."""
    y_pred, y_true = synthetic_data
    cal = IsotonicCalibrator()
    cal.fit(y_pred, y_true)
    out = cal.predict(y_pred)

    assert out.shape == y_pred.shape
    assert np.all(out >= PROB_MIN), f"下限違反: min={out.min()}"
    assert np.all(out <= PROB_MAX), f"上限違反: max={out.max()}"


# ============================================================
# Test 2: IsotonicCalibrator monotonic constraint
# ============================================================

def test_isotonic_calibrator_monotonic(synthetic_data):
    """calibrated scores が元 score に対して単調非減少 (isotonic 制約確認)."""
    y_pred, y_true = synthetic_data
    cal = IsotonicCalibrator()
    cal.fit(y_pred, y_true)

    # sorted inputs に対して calibrated も単調非減少 (または平坦) であること
    sorted_pred = np.sort(y_pred)
    calibrated  = cal.predict(sorted_pred)

    diffs = np.diff(calibrated)
    assert np.all(diffs >= -1e-9), (
        f"monotonic 違反: min diff={diffs.min():.6f}"
    )


# ============================================================
# Test 3: VenueCalibrator fit + predict
# ============================================================

def test_venue_calibrator_fit_predict(venue_data):
    """per-venue calibrator が各会場で値を返す."""
    y_pred, y_true, venues = venue_data
    vcal = VenueCalibrator(min_samples=30)
    vcal.fit(y_pred, y_true, venues)

    for v in ["東京", "中山", "阪神", "京都"]:
        # 各会場の入力で predict が動く
        mask = venues == v
        out = vcal.predict(y_pred[mask], v)
        assert out.shape == y_pred[mask].shape
        assert np.all(out >= PROB_MIN)
        assert np.all(out <= PROB_MAX)

    # venue_summary が正しいカウントを返す
    summary = vcal.venue_summary()
    assert "東京" in summary
    assert summary["東京"]["n"] == 50


# ============================================================
# Test 4: OddsBinCalibrator fit + predict
# ============================================================

def test_odds_bin_calibrator_fit_predict(odds_data):
    """per-odds-bin calibrator が各 bin で値を返す."""
    y_pred, y_true, odds = odds_data
    ocal = OddsBinCalibrator(min_samples=20)
    ocal.fit(y_pred, y_true, odds)

    out = ocal.predict(y_pred, odds)
    assert out.shape == y_pred.shape
    assert np.all(out >= PROB_MIN)
    assert np.all(out <= PROB_MAX)

    # bin_summary が bin ラベルを含む
    summary = ocal.bin_summary()
    for lb in ODDS_BIN_LABELS:
        assert lb in summary

    # scalar predict も動作確認
    scalar_out = ocal.predict(0.5, 5.0)
    assert isinstance(scalar_out, float)
    assert PROB_MIN <= scalar_out <= PROB_MAX


# ============================================================
# Test 5: calibrate_v2 returns scores (method='none')
# ============================================================

def test_calibrate_v2_returns_scores(synthetic_data):
    """calibrate_v2(method='none') は raw scores を [PROB_MIN, PROB_MAX] で返す."""
    y_pred, _ = synthetic_data
    out = calibrate_v2(y_pred, method="none")
    assert out.shape == y_pred.shape
    assert np.all(out >= PROB_MIN)
    assert np.all(out <= PROB_MAX)


# ============================================================
# Test 6: calibrate_v2 警告 + fallback (calibrator 未保存時)
# ============================================================

def test_calibrate_v2_fallback_on_missing_file(synthetic_data, tmp_path):
    """calibrator pkl 不在時に UserWarning + raw scores を返す."""
    y_pred, _ = synthetic_data
    missing_path = tmp_path / "nonexistent.pkl"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = calibrate_v2(y_pred, method="isotonic", calibrator_path=str(missing_path))

    assert len(caught) >= 1
    assert any(issubclass(w.category, UserWarning) for w in caught)
    # fallback で raw scores (clipped) が返る
    assert out.shape == y_pred.shape
    np.testing.assert_allclose(out, np.clip(y_pred, PROB_MIN, PROB_MAX), atol=1e-9)


# ============================================================
# Test 7: evaluate_calibration + crossval_brier
# ============================================================

def test_evaluate_calibration_and_crossval(synthetic_data):
    """evaluate_calibration と crossval_brier が正常値を返す."""
    y_pred, y_true = synthetic_data

    # evaluate_calibration (calibrated あり)
    cal = IsotonicCalibrator()
    cal.fit(y_pred, y_true)
    calibrated = cal.predict(y_pred)
    metrics = evaluate_calibration(y_pred, y_true, calibrated)

    assert "raw_brier" in metrics
    assert "cal_brier" in metrics
    assert "delta_brier" in metrics
    assert metrics["raw_brier"] > 0.0
    assert metrics["cal_brier"] > 0.0
    # delta は計算できていること (NaN でない)
    assert not np.isnan(metrics["delta_brier"])

    # crossval_brier
    cv = crossval_brier(y_pred, y_true, n_splits=5)
    assert "cv_brier" in cv
    assert "raw_brier" in cv
    assert "delta_brier" in cv
    assert cv["n_samples"] == len(y_true)
    assert cv["n_splits"] == 5
    # CV Brier は [0, 1] 範囲
    assert 0.0 <= cv["cv_brier"] <= 1.0
    assert 0.0 <= cv["raw_brier"] <= 1.0


# ============================================================
# Test 8: save / load round-trip (IsotonicCalibrator)
# ============================================================

def test_isotonic_save_load_roundtrip(synthetic_data, tmp_path):
    """IsotonicCalibrator の save/load round-trip で同じ predict 結果を返す."""
    y_pred, y_true = synthetic_data
    cal = IsotonicCalibrator()
    cal.fit(y_pred, y_true)
    pred_before = cal.predict(y_pred)

    pkl_path = tmp_path / "iso_test.pkl"
    cal.save(pkl_path)

    loaded = IsotonicCalibrator.load(pkl_path)
    pred_after = loaded.predict(y_pred)

    np.testing.assert_allclose(pred_before, pred_after, atol=1e-12)


# ============================================================
# Test 9: calibrate_v2 invalid method raises ValueError
# ============================================================

def test_calibrate_v2_invalid_method_raises():
    """不明な method は ValueError を送出する."""
    with pytest.raises(ValueError, match="未知の method"):
        calibrate_v2(np.array([0.5]), method="unknown_method")


# ============================================================
# Test 10: VenueCalibrator global fallback for unknown venue
# ============================================================

def test_venue_calibrator_fallback_unknown_venue(venue_data):
    """未知の会場 (test set) は global calibrator で fallback される."""
    y_pred, y_true, venues = venue_data
    vcal = VenueCalibrator(min_samples=10)
    vcal.fit(y_pred, y_true, venues)

    # "小倉" は fit に含まれていない → global fallback
    out = vcal.predict(y_pred[:10], "小倉")
    assert out.shape == (10,)
    assert np.all(out >= PROB_MIN)
    assert np.all(out <= PROB_MAX)
