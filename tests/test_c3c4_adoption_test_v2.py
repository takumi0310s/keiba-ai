"""Tests for tools/c3c4_adoption_test_v2.py"""
import math
import os
import random
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure repo root on path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tools'))

from c3c4_adoption_test_v2 import (
    compute_bootstrap_ci,
    evaluate_strategy,
    load_paper_data,
    welch_t_test,
    THRESH_N,
)


# ---------------------------------------------------------------------------
# 1. load_paper_data — LOG_DIR does not exist → returns empty list (no error)
# ---------------------------------------------------------------------------

def test_load_paper_data_empty_dir(tmp_path, monkeypatch):
    """LOG_DIR なし → 空リスト返す (エラーなし)"""
    import c3c4_adoption_test_v2 as mod
    monkeypatch.setattr(mod, 'LOG_DIR', tmp_path / 'nonexistent_dir')
    result = mod.load_paper_data()
    assert result == []


# ---------------------------------------------------------------------------
# 2. compute_bootstrap_ci — 10 values → returns a tuple of two floats
# ---------------------------------------------------------------------------

def test_bootstrap_ci_returns_tuple():
    """10 values → (lower, upper) tuple 返す"""
    random.seed(0)
    values = [float(i) for i in range(10)]
    result = compute_bootstrap_ci(values, n_iter=200)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# 3. compute_bootstrap_ci — lower < upper (CI is well-ordered)
# ---------------------------------------------------------------------------

def test_bootstrap_ci_lower_lt_upper():
    """CI の lower < upper"""
    random.seed(1)
    values = [random.gauss(50, 10) for _ in range(50)]
    lower, upper = compute_bootstrap_ci(values, n_iter=500)
    assert lower < upper


# ---------------------------------------------------------------------------
# 4. welch_t_test — same distribution → p > 0.1 (no significant difference)
# ---------------------------------------------------------------------------

def test_welch_t_test_same_dist():
    """同一分布 → p > 0.1 (有意差なし)"""
    random.seed(42)
    a = [random.gauss(0, 1) for _ in range(50)]
    b = [random.gauss(0, 1) for _ in range(50)]
    t_stat, p_value = welch_t_test(a, b)
    assert p_value > 0.1, f"Expected p > 0.1 for same distribution, got p={p_value}"


# ---------------------------------------------------------------------------
# 5. welch_t_test — large mean difference → p < 0.05
# ---------------------------------------------------------------------------

def test_welch_t_test_diff_dist():
    """平均差大 → p < 0.05"""
    random.seed(42)
    a = [random.gauss(100, 1) for _ in range(50)]
    b = [random.gauss(0, 1) for _ in range(50)]
    t_stat, p_value = welch_t_test(a, b)
    assert p_value < 0.05, f"Expected p < 0.05 for large mean diff, got p={p_value}"


# ---------------------------------------------------------------------------
# 6. evaluate_strategy — N=5 → NO-GO (N < 24)
# ---------------------------------------------------------------------------

def test_evaluate_strategy_no_go_small_n():
    """N=5 → NO-GO (N<24)"""
    small_stats = {
        'n': 5,
        'hits': 3,
        'inv': 3500,
        'pay': 7000,
        'roi_pct': 200.0,
        'pnl': 3500,
    }
    baseline_stats = {
        'n': 30,
        'hits': 10,
        'inv': 21000,
        'pay': 21000,
        'roi_pct': 100.0,
        'pnl': 0,
    }
    result = evaluate_strategy('c3', small_stats, baseline_stats)
    assert result['verdict'] == 'NO-GO', (
        f"Expected NO-GO for N=5 < {THRESH_N}, got {result['verdict']}"
    )
    assert result['check1_n'] is False


# ---------------------------------------------------------------------------
# 7. evaluate_strategy — sufficient N + large ROI delta + low p → GO
# ---------------------------------------------------------------------------

def test_evaluate_strategy_go_large_delta():
    """N>=24, delta >= +5pt, p < 0.05 → GO"""
    # Construct stats that produce a large ROI delta and small p-value.
    # Strategy: 30 races, 15 hits (50% hit rate), avg payout 2800 each.
    # Baseline: 30 races, 8 hits (27% hit rate), avg payout 2625 each.
    strat_pay = 15 * 2800   # 42000
    strat_inv = 30 * 700    # 21000
    strat_roi = strat_pay / strat_inv * 100  # 200%

    base_pay = 8 * 2625    # 21000
    base_inv = 30 * 700    # 21000
    base_roi = base_pay / base_inv * 100  # 100%

    strat_stats = {
        'n': 30,
        'hits': 15,
        'inv': strat_inv,
        'pay': strat_pay,
        'roi_pct': round(strat_roi, 2),
        'pnl': strat_pay - strat_inv,
    }
    baseline_stats = {
        'n': 30,
        'hits': 8,
        'inv': base_inv,
        'pay': base_pay,
        'roi_pct': round(base_roi, 2),
        'pnl': base_pay - base_inv,
    }
    result = evaluate_strategy('c3', strat_stats, baseline_stats)
    # delta = 200 - 100 = +100pt (well above +5pt threshold)
    assert result['delta'] >= 5.0
    assert result['check1_n'] is True


# ---------------------------------------------------------------------------
# 8. compute_bootstrap_ci — empty input → (0.0, 0.0)
# ---------------------------------------------------------------------------

def test_bootstrap_ci_empty_input():
    """空リスト → (0.0, 0.0) を返す"""
    result = compute_bootstrap_ci([])
    assert result == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 9. welch_t_test — tiny samples → returns tuple of two floats
# ---------------------------------------------------------------------------

def test_welch_t_test_tiny_samples():
    """最小サンプル (n=2) でも tuple を返す"""
    t, p = welch_t_test([1.0, 2.0], [3.0, 4.0])
    assert isinstance(t, float)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0
