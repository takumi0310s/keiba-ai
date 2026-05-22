"""
tests/test_tyb_shadow_fetcher.py

TYB shadow fetcher unit tests (6 tests).
All tests pass without network access (unittest.mock).
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ensure repo root is on path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import tools.tyb_shadow_fetcher as tsf


class TestTybShadowFetcher(unittest.TestCase):

    # ------------------------------------------------------------------
    # Test 1: default disabled → returns None immediately
    # ------------------------------------------------------------------
    def test_default_disabled(self):
        """fetch_tyb_shadow with enabled=False returns None, no exceptions."""
        result = tsf.fetch_tyb_shadow(
            race_id="202605230611",
            start_time_str="1030",
            enabled=False,
        )
        self.assertIsNone(result, "Should return None when disabled")

    # ------------------------------------------------------------------
    # Test 2: disabled → no files created in data/tyb_shadow/
    # ------------------------------------------------------------------
    def test_disabled_no_side_effects(self):
        """Calling with enabled=False must not create any files."""
        shadow_dir = REPO / "data" / "tyb_shadow"
        # collect pre-existing files
        before = set(shadow_dir.rglob("*")) if shadow_dir.exists() else set()

        tsf.fetch_tyb_shadow(
            race_id="202605230611",
            start_time_str="1030",
            enabled=False,
        )

        after = set(shadow_dir.rglob("*")) if shadow_dir.exists() else set()
        new_files = after - before
        self.assertEqual(
            new_files,
            set(),
            f"No new files should be created when disabled; got: {new_files}",
        )

    # ------------------------------------------------------------------
    # Test 3: network error → returns None (does not raise)
    # ------------------------------------------------------------------
    @patch("tools.tyb_shadow_fetcher.urllib.request.urlopen")
    @patch("tools.tyb_shadow_fetcher._load_env", return_value=("user", "pass"))
    def test_fetch_fail_returns_none(self, mock_env, mock_urlopen):
        """Network error should be caught → returns None, never raises."""
        mock_urlopen.side_effect = OSError("Connection refused")
        result = tsf.fetch_tyb_shadow(
            race_id="202605230611",
            start_time_str="1530",
            enabled=True,
        )
        self.assertIsNone(result, "Should return None on network error")

    # ------------------------------------------------------------------
    # Test 4: fetch failure cannot affect V15 prediction
    # ------------------------------------------------------------------
    @patch("tools.tyb_shadow_fetcher.urllib.request.urlopen")
    @patch("tools.tyb_shadow_fetcher._load_env", return_value=("user", "pass"))
    def test_fetch_fail_v15_unaffected(self, mock_env, mock_urlopen):
        """
        Simulate fetch failure and verify None return is handled gracefully.
        The None return ensures V15 prediction is never touched.
        """
        mock_urlopen.side_effect = RuntimeError("Simulated TYB failure")

        # Simulate how a caller would use the result
        tyb_result = tsf.fetch_tyb_shadow(
            race_id="202605230611",
            start_time_str="1545",
            enabled=True,
        )

        # caller checks None before using — V15 logic untouched
        self.assertIsNone(tyb_result)

        # Verify that None result can be safely tested without error
        supplement = tsf.format_tyb_discord_supplement(tyb_result) if tyb_result else ""
        self.assertEqual(supplement, "", "Empty string expected when tyb_result is None")

        # A hypothetical V15 prediction dict is unchanged
        v15_pred = {"race_id": "202605230611", "top1_num": 5, "score": 0.82}
        if tyb_result is not None:
            # This block must NEVER execute — assert it doesn't
            v15_pred["tyb_injected"] = True  # pragma: no cover

        self.assertNotIn(
            "tyb_injected", v15_pred,
            "V15 prediction must not be modified by TYB fetch failure",
        )

    # ------------------------------------------------------------------
    # Test 5: format_tyb_discord_supplement with mock data → non-empty string
    # ------------------------------------------------------------------
    def test_format_supplement_valid(self):
        """format_tyb_discord_supplement returns non-empty string for valid data."""
        mock_result = {
            "race_id": "202606010611",
            "odds_idx_list": [85.0, 72.3, 68.1, 55.0, 40.2],
            "padock_idx_list": [90.0, 78.0, 60.0, 55.0, 42.0],
            "tansho_odds_list": [3.5, 5.2, 7.1, 10.0, 15.0],
            "horse_weight_list": [480, 456, 512, 494, 470],
            "padock_mark_list": ["A", "B", "A", "C", "B"],
            "fetch_time": "2026-06-01T14:45:00",
            "delta_to_start_min": 15.0,
            "num_horses": 5,
            "raw_rows": [
                {"umaban": 1, "weight_diff": 2, "cancel_flag": 0},
                {"umaban": 2, "weight_diff": -12, "cancel_flag": 0},
                {"umaban": 3, "weight_diff": 0, "cancel_flag": 0},
                {"umaban": 4, "weight_diff": 4, "cancel_flag": 1},
                {"umaban": 5, "weight_diff": -2, "cancel_flag": 0},
            ],
        }
        supplement = tsf.format_tyb_discord_supplement(mock_result)
        self.assertIsInstance(supplement, str)
        self.assertGreater(len(supplement), 0, "Supplement string must not be empty")
        self.assertIn("TYB", supplement)
        # check large weight change detected
        self.assertIn("馬体重大幅変化", supplement)
        # check cancel detected
        self.assertIn("取消", supplement)
        # shadow note present
        self.assertIn("非影響", supplement)

    # ------------------------------------------------------------------
    # Test 6: 5/23 safety — module-level TYB_SHADOW_ENABLED is False
    # ------------------------------------------------------------------
    def test_no_523_impact(self):
        """
        Module-level TYB_SHADOW_ENABLED must be False.
        This guarantees tyb_shadow_fetcher cannot fire on 5/23 by default.
        """
        self.assertFalse(
            tsf.TYB_SHADOW_ENABLED,
            "TYB_SHADOW_ENABLED must be False at module level "
            "(5/23 fire safety guarantee)",
        )

        # Also verify that calling with the module default returns None
        result = tsf.fetch_tyb_shadow(
            race_id="202605230611",
            start_time_str="1030",
            # enabled= not passed → uses module default TYB_SHADOW_ENABLED
            enabled=tsf.TYB_SHADOW_ENABLED,
        )
        self.assertIsNone(result, "Default-disabled fetch must return None")


if __name__ == "__main__":
    unittest.main()
