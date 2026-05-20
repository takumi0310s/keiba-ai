"""Tests for tools/c4_param_tune.py"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.c4_param_tune import (
    TRAIN_END,
    TRAIN_START,
    TEST_END,
    TEST_START,
    compute_roi_for_subset,
    investment_for_n_tickets,
    load_data,
    run_grid_search,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_row(race_id: int, date: int, condition: str, distance: float,
              trio_hit: float, actual_payout: float, investment: int = 700,
              status: str = "settled") -> dict:
    return {
        "race_id": race_id,
        "date": date,
        "condition": condition,
        "distance": distance,
        "trio_hit": trio_hit,
        "actual_payout": actual_payout,
        "investment": investment,
        "status": status,
        "course": "東京",
        "race_num": 1,
        "race_name": "test",
        "num_horses": 10.0,
        "surface": "芝",
        "track_condition": "良",
    }


def _minimal_df(rows: list[dict]) -> pd.DataFrame:
    """Create a minimal DataFrame that load_data would produce."""
    df = pd.DataFrame(rows)
    df["date"] = df["date"].astype(int)
    df["distance"] = df["distance"].astype(float)
    df["actual_payout"] = df["actual_payout"].astype(float)
    df["investment"] = df["investment"].astype(int)
    df["trio_hit"] = df["trio_hit"].astype(float)
    return df


# ---------------------------------------------------------------------------
# Test 1: train/test split — no race_id overlap
# ---------------------------------------------------------------------------

def test_train_test_split_no_leak():
    """train/test に同じ race_id が含まれないことを確認。"""
    rows = [
        _make_row(1001, 20260401, "A", 1600, 1, 5000),
        _make_row(1002, 20260503, "A", 2000, 0, 0),
        _make_row(1003, 20260504, "A", 1600, 1, 3000),
        _make_row(1004, 20260517, "A", 1800, 0, 0),
    ]
    df = _minimal_df(rows)

    cond_a = df[(df["condition"] == "A") & (df["status"] == "settled")]
    train = cond_a[(cond_a["date"] >= TRAIN_START) & (cond_a["date"] <= TRAIN_END)]
    test = cond_a[(cond_a["date"] >= TEST_START) & (cond_a["date"] <= TEST_END)]

    train_ids = set(train["race_id"])
    test_ids = set(test["race_id"])
    overlap = train_ids & test_ids

    assert len(overlap) == 0, f"Leak: race_ids in both train and test: {overlap}"
    assert len(train) == 2
    assert len(test) == 2


# ---------------------------------------------------------------------------
# Test 2: ROI calculation accuracy
# ---------------------------------------------------------------------------

def test_roi_calculation():
    """既知データで ROI 計算が正確か確認。"""
    rows = [
        _make_row(2001, 20260410, "A", 2000, 1, 6000),   # hit
        _make_row(2002, 20260411, "A", 2000, 0, 0),       # miss
        _make_row(2003, 20260412, "A", 2000, 1, 12000),   # hit
        _make_row(2004, 20260413, "A", 2000, 0, 0),       # miss
    ]
    df = _minimal_df(rows)

    # 4 races, 4 tickets each, 100 yen/ticket → 400 yen/race
    n_tickets = 4
    invest_per_race = investment_for_n_tickets(n_tickets)  # 400
    expected_invest = invest_per_race * 4  # 1600
    expected_payout = 6000 + 0 + 12000 + 0  # 18000
    expected_roi = expected_payout / expected_invest * 100  # 1125.0

    result = compute_roi_for_subset(df, n_tickets)
    assert result["n"] == 4
    assert result["invest_sum"] == expected_invest
    assert abs(result["payout_sum"] - expected_payout) < 0.01
    assert abs(result["roi_pct"] - expected_roi) < 0.1

    # hit3_rate = 2/4 = 0.5
    assert abs(result["hit3_rate"] - 0.5) < 0.001


def test_roi_calculation_zero_invest():
    """空の subset で ROI=0 になるか確認。"""
    # Build an empty DataFrame with the correct columns
    columns = ["race_id", "date", "condition", "distance", "trio_hit",
               "actual_payout", "investment", "status", "course",
               "race_num", "race_name", "num_horses", "surface", "track_condition"]
    df = pd.DataFrame(columns=columns)
    df = df.astype({"trio_hit": float, "actual_payout": float, "investment": int})
    result = compute_roi_for_subset(df, 7)
    assert result["n"] == 0
    assert result["roi_pct"] == 0.0
    assert result["invest_sum"] == 0


# ---------------------------------------------------------------------------
# Test 3: overfit detection
# ---------------------------------------------------------------------------

def test_overfit_detection():
    """in-sample ROI >> hold-out ROI で overfit フラグが立つか確認。"""
    # Train: 5 hits with big payout → very high ROI
    train_rows = [
        _make_row(3001, 20260401, "A", 2000, 1, 20000),
        _make_row(3002, 20260402, "A", 2000, 1, 20000),
        _make_row(3003, 20260403, "A", 2000, 1, 20000),
        _make_row(3004, 20260404, "A", 2000, 1, 20000),
        _make_row(3005, 20260405, "A", 2000, 1, 20000),
    ]
    # Test: all misses → ROI = 0%
    test_rows = [
        _make_row(3010, 20260504, "A", 2000, 0, 0),
        _make_row(3011, 20260505, "A", 2000, 0, 0),
        _make_row(3012, 20260506, "A", 2000, 0, 0),
        _make_row(3013, 20260507, "A", 2000, 0, 0),
        _make_row(3014, 20260508, "A", 2000, 0, 0),
    ]
    all_rows = train_rows + test_rows
    df = _minimal_df(all_rows)

    cond_a = df[(df["condition"] == "A") & (df["status"] == "settled")]
    train = cond_a[(cond_a["date"] >= TRAIN_START) & (cond_a["date"] <= TRAIN_END)]
    test = cond_a[(cond_a["date"] >= TEST_START) & (cond_a["date"] <= TEST_END)]

    train_metrics = compute_roi_for_subset(train, 7)
    test_metrics = compute_roi_for_subset(test, 7)

    in_sample_roi = train_metrics["roi_pct"]
    holdout_roi = test_metrics["roi_pct"]
    overfit_flag = (in_sample_roi - holdout_roi) >= 20.0

    # Train ROI should be very high (>> 100%), test = 0%
    assert in_sample_roi > 100.0, f"Expected high train ROI, got {in_sample_roi}"
    assert holdout_roi == 0.0, f"Expected 0% test ROI, got {holdout_roi}"
    assert overfit_flag is True, "Expected overfit_flag=True"


# ---------------------------------------------------------------------------
# Test 4: run_grid_search produces expected structure
# ---------------------------------------------------------------------------

def test_run_grid_search_structure():
    """run_grid_search の出力が期待する構造を持つか確認。"""
    rows = []
    rid = 4000
    # Generate enough rows to test (Cond A, various distances, train + test)
    for date in [20260410, 20260420, 20260430, 20260504, 20260510]:
        for dist in [1600.0, 1800.0, 2000.0]:
            rows.append(_make_row(rid, date, "A", dist, 0, 0))
            rid += 1

    df = _minimal_df(rows)
    results = run_grid_search(df)

    # Should have 5 distance ranges × 4 ticket counts = 20 entries
    assert len(results) == 20

    for r in results:
        assert "c4_dist_lo" in r
        assert "c4_dist_hi" in r
        assert "n_tickets" in r
        assert "train" in r
        assert "test" in r
        assert "overfit_flag" in r
        assert isinstance(r["overfit_flag"], bool)
        assert "roi_pct" in r["train"]
        assert "hit3_rate" in r["train"]
        assert "n" in r["train"]

    # Exactly one entry should be is_current=True (1600-1800, 7 tickets)
    current_entries = [r for r in results if r.get("is_current")]
    assert len(current_entries) == 1
    assert current_entries[0]["c4_dist_lo"] == 1600
    assert current_entries[0]["c4_dist_hi"] == 1800
    assert current_entries[0]["n_tickets"] == 7


# ---------------------------------------------------------------------------
# Test 5: investment_for_n_tickets
# ---------------------------------------------------------------------------

def test_investment_per_ticket():
    """100円/枚の投資額計算。"""
    assert investment_for_n_tickets(7) == 700
    assert investment_for_n_tickets(6) == 600
    assert investment_for_n_tickets(10) == 1000
