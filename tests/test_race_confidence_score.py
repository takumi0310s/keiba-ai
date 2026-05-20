"""Tests for tools/race_confidence_score.py"""

import sys
from pathlib import Path

import pytest

# Ensure the tools directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from race_confidence_score import (
    compute_confidence,
    filter_by_confidence,
    get_paper_formation,
)


# ---------------------------------------------------------------------------
# test_confidence_range
# ---------------------------------------------------------------------------

def test_confidence_range_low():
    """confidence must be in [0, 1] for typical low-prob inputs."""
    conf = compute_confidence(0.1, 0.08, 12)
    assert 0.0 <= conf <= 1.0


def test_confidence_range_high():
    """confidence must be in [0, 1] even for high top1_prob."""
    conf = compute_confidence(0.95, 0.01, 8)
    assert 0.0 <= conf <= 1.0


def test_confidence_range_extreme():
    """confidence must clamp to [0, 1] regardless of inputs."""
    conf = compute_confidence(1.0, 0.0, 7)
    assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# test_num_horses_factor
# ---------------------------------------------------------------------------

def test_num_horses_factor_small():
    """7頭以下 → factor 1.2 (少頭数は高くなる)."""
    conf_small = compute_confidence(0.3, 0.2, 7)
    conf_mid = compute_confidence(0.3, 0.2, 10)
    assert conf_small > conf_mid, (
        f"7頭 confidence ({conf_small:.4f}) should > 10頭 ({conf_mid:.4f})"
    )


def test_num_horses_factor_large():
    """15頭+ → factor 0.8 (多頭数は低くなる)."""
    conf_large = compute_confidence(0.3, 0.2, 15)
    conf_mid = compute_confidence(0.3, 0.2, 10)
    assert conf_large < conf_mid, (
        f"15頭 confidence ({conf_large:.4f}) should < 10頭 ({conf_mid:.4f})"
    )


def test_num_horses_factor_boundary_7():
    """7頭は factor 1.2 が適用される。"""
    conf_7 = compute_confidence(0.4, 0.1, 7)
    conf_8 = compute_confidence(0.4, 0.1, 8)
    assert conf_7 > conf_8


def test_num_horses_factor_boundary_14_15():
    """14頭は factor 1.0、15頭は factor 0.8。"""
    conf_14 = compute_confidence(0.4, 0.1, 14)
    conf_15 = compute_confidence(0.4, 0.1, 15)
    assert conf_14 > conf_15


# ---------------------------------------------------------------------------
# test_filter_above_threshold
# ---------------------------------------------------------------------------

def test_filter_above_threshold():
    """confidence > threshold → True (投票)."""
    conf = compute_confidence(0.6, 0.1, 10)
    # conf should be well above 0.05
    assert filter_by_confidence(conf, 0.05) is True


def test_filter_exactly_at_threshold_is_false():
    """confidence == threshold → False (スキップ、exclusive)."""
    assert filter_by_confidence(0.10, 0.10) is False


# ---------------------------------------------------------------------------
# test_filter_below_threshold
# ---------------------------------------------------------------------------

def test_filter_below_threshold():
    """confidence < threshold → False (skip)."""
    conf = compute_confidence(0.1, 0.09, 16)
    # This will be low; test against a high threshold
    assert filter_by_confidence(conf, 0.30) is False


def test_filter_zero_threshold_always_true():
    """threshold=0.0 → always True unless confidence=0."""
    conf = compute_confidence(0.2, 0.1, 10)
    assert filter_by_confidence(conf, 0.0) is True


# ---------------------------------------------------------------------------
# test_none_formation_on_skip
# ---------------------------------------------------------------------------

def test_none_formation_on_skip():
    """filter=False (low confidence) → get_paper_formation returns None."""
    # Use very low probs so confidence < 0.30
    predictions = [
        {"horse_num": 1, "prob": 0.06},
        {"horse_num": 2, "prob": 0.055},
        {"horse_num": 3, "prob": 0.05},
        {"horse_num": 4, "prob": 0.04},
        {"horse_num": 5, "prob": 0.03},
        {"horse_num": 6, "prob": 0.02},
    ]
    race_meta = {"num_horses": 16, "condition": "A"}
    result = get_paper_formation(predictions, threshold=0.30, race_meta=race_meta)
    assert result is None


def test_formation_returned_when_confident():
    """High confidence → get_paper_formation returns a non-empty string."""
    predictions = [
        {"horse_num": 3, "prob": 0.75},
        {"horse_num": 7, "prob": 0.05},
        {"horse_num": 1, "prob": 0.04},
        {"horse_num": 5, "prob": 0.03},
        {"horse_num": 9, "prob": 0.02},
        {"horse_num": 2, "prob": 0.01},
    ]
    race_meta = {"num_horses": 10, "condition": "A"}
    result = get_paper_formation(predictions, threshold=0.05, race_meta=race_meta)
    assert result is not None
    assert len(result) > 0


def test_formation_none_on_insufficient_horses():
    """Less than 3 predictions → None."""
    predictions = [
        {"horse_num": 1, "prob": 0.8},
        {"horse_num": 2, "prob": 0.1},
    ]
    race_meta = {"num_horses": 5, "condition": "A"}
    result = get_paper_formation(predictions, threshold=0.0, race_meta=race_meta)
    assert result is None


def test_condition_e_umaren_format():
    """条件E → umaren 2点フォーマット (2 combos separated by '; ')."""
    predictions = [
        {"horse_num": 4, "prob": 0.5},
        {"horse_num": 2, "prob": 0.2},
        {"horse_num": 6, "prob": 0.1},
        {"horse_num": 1, "prob": 0.05},
    ]
    race_meta = {"num_horses": 6, "condition": "E"}
    result = get_paper_formation(predictions, threshold=0.0, race_meta=race_meta)
    assert result is not None
    parts = result.split("; ")
    assert len(parts) == 2, f"Expected 2 umaren combos, got: {result}"
