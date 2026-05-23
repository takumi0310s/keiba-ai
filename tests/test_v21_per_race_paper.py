"""Tests for tools/v21_per_race_paper.py

3 tests:
  1. test_load_v21_model_graceful — missing file → returns None, no exception
  2. test_fire_race_notify_swallows_exception — if prediction fails, no exception propagates
  3. test_discord_message_contains_paper_label — message always has required labels
"""
from __future__ import annotations

import gzip
import json
import pickle
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "tools"))


# ---------------------------------------------------------------------------
# Test 1: load_v21_model graceful on missing file
# ---------------------------------------------------------------------------
def test_load_v21_model_graceful(tmp_path):
    """load_v21_model returns None (no exception) when model file is absent."""
    import importlib
    import tools.v21_per_race_paper as mod

    # Temporarily point V21_MODEL_PATH to a non-existent path
    original = mod.V21_MODEL_PATH
    try:
        mod.V21_MODEL_PATH = tmp_path / "nonexistent_v21_model.pkl.gz"
        result = mod.load_v21_model()
        assert result is None, "Expected None when file is missing"
    finally:
        mod.V21_MODEL_PATH = original


def test_load_v21_model_graceful_corrupt(tmp_path):
    """load_v21_model returns None (no exception) when model file is corrupt."""
    import tools.v21_per_race_paper as mod

    bad_file = tmp_path / "bad_model.pkl.gz"
    bad_file.write_bytes(b"this is not valid gzip data")

    original = mod.V21_MODEL_PATH
    try:
        mod.V21_MODEL_PATH = bad_file
        result = mod.load_v21_model()
        assert result is None, "Expected None for corrupt file"
    finally:
        mod.V21_MODEL_PATH = original


# ---------------------------------------------------------------------------
# Test 2: fire_race_notify swallows all exceptions
# ---------------------------------------------------------------------------
def test_fire_race_notify_swallows_exception():
    """
    fire_race_notify must not raise even if predict_v21 raises an unhandled exception.
    This is the contract that keeps the scheduler alive.
    """
    import tools.v21_per_race_paper as mod

    race_info = {
        "race_id": "202605240511",
        "course": "東京",
        "race_num": 11,
        "start_time": "15:30",
        "start_dt": datetime.now() + timedelta(minutes=30),
    }

    dummy_model = {"model": object(), "features": ["f1"], "version": "v21_test"}

    # Patch predict_v21 to raise RuntimeError
    with mock.patch.object(mod, "predict_v21", side_effect=RuntimeError("boom")):
        # Patch record_paper_log to no-op
        with mock.patch.object(mod, "record_paper_log", return_value=None):
            # Patch fetch_tyb_for_race to return None
            with mock.patch.object(mod, "fetch_tyb_for_race", return_value=None):
                # Should not raise
                try:
                    mod.fire_race_notify(race_info, dummy_model)
                except Exception as e:
                    pytest.fail(f"fire_race_notify raised unexpectedly: {e}")


def test_fire_race_notify_swallows_discord_exception():
    """
    fire_race_notify must not raise even if send_discord raises.
    """
    import tools.v21_per_race_paper as mod

    race_info = {
        "race_id": "202605240512",
        "course": "中山",
        "race_num": 12,
        "start_time": "16:00",
        "start_dt": datetime.now() + timedelta(minutes=30),
    }

    dummy_model = {"model": object(), "features": ["f1"], "version": "v21_test"}

    good_pred = {
        "race_id": "202605240512",
        "race_name": "テストレース",
        "rinfo": {"course": "中山", "race_num": 12, "surface": "芝",
                  "condition": "良", "distance": 2000},
        "cond_key": "A",
        "bet_type": "trio",
        "bets": [[1, 2, 3]],
        "top5": [1, 2, 3, 4, 5],
        "top5_names": ["A", "B", "C", "D", "E"],
        "top1": 1,
        "top1_name": "A",
        "top1_score": 0.75,
        "distance": 2000,
        "num_horses": 14,
        "strategy_pass": True,
        "strategy_skip_reason": "",
        "formation": [[1, 2, 3]],
        "tyb_injected": False,
    }

    with mock.patch.object(mod, "predict_v21", return_value=good_pred):
        with mock.patch.object(mod, "record_paper_log", return_value=None):
            with mock.patch.object(mod, "fetch_tyb_for_race", return_value=None):
                with mock.patch.object(mod, "send_discord", side_effect=ConnectionError("no network")):
                    try:
                        mod.fire_race_notify(race_info, dummy_model)
                    except Exception as e:
                        pytest.fail(f"fire_race_notify raised unexpectedly: {e}")


# ---------------------------------------------------------------------------
# Test 3: Discord message always contains paper label
# ---------------------------------------------------------------------------
def test_discord_message_contains_paper_label():
    """
    build_discord_message must always include:
    - 【V21 paper】 header
    - 投票しないでください
    - paper warning footer
    """
    import tools.v21_per_race_paper as mod

    pred = {
        "race_name": "テストS",
        "rinfo": {"course": "東京", "race_num": 11, "surface": "芝",
                  "condition": "良", "distance": 1600},
        "cond_key": "A",
        "bet_type": "trio",
        "bets": [[1, 2, 3], [1, 2, 4]],
        "top5": [1, 2, 3, 4, 5],
        "top5_names": ["ホースA", "ホースB", "ホースC", "ホースD", "ホースE"],
        "top1": 1,
        "top1_name": "ホースA",
        "top1_score": 0.82,
        "distance": 1600,
        "num_horses": 14,
        "strategy_pass": True,
        "strategy_skip_reason": "",
        "formation": [[1, 2, 3]],
        "tyb_injected": False,
    }

    race_info = {"course": "東京", "race_num": 11}

    msg = mod.build_discord_message("202605240511", pred, None, race_info)

    assert "【V21 paper" in msg, "Message must contain 【V21 paper】 header"
    assert "投票しないでください" in msg, "Message must contain 投票しないでください"
    assert "paper予測" in msg or "paper" in msg.lower(), "Message must mention paper"
    assert "V15" in msg, "Message must reference V15"


def test_discord_message_contains_paper_label_when_filter_skips():
    """
    Even when strategy filter skips (strategy_pass=False), message must contain paper label.
    """
    import tools.v21_per_race_paper as mod

    pred = {
        "race_name": "テスト特別",
        "rinfo": {"course": "東京", "race_num": 5, "surface": "芝",
                  "condition": "良", "distance": 1800},
        "cond_key": "E",
        "bet_type": "umaren",
        "bets": [],
        "top5": [2, 5],
        "top5_names": ["ホースX", "ホースY"],
        "top1": 2,
        "top1_name": "ホースX",
        "top1_score": 0.61,
        "distance": 1800,
        "num_horses": 6,
        "strategy_pass": False,
        "strategy_skip_reason": "strategy_7_cond_E",
        "formation": [],
        "tyb_injected": False,
    }

    race_info = {"course": "東京", "race_num": 5}

    msg = mod.build_discord_message("202605240505", pred, None, race_info)

    assert "【V21 paper" in msg, "Message must contain 【V21 paper】 header even when skipped"
    assert "投票しないでください" in msg, "Warning must always be present"
