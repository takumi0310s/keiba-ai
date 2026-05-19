"""Tests for strategy-level anomaly detection (D-6 2026-05-19)"""
from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.anomaly_auto_detector import check_strategy_anomaly


class TestStrategyAnomalyDetection(unittest.TestCase):

    def test_single_day_roi_drop(self):
        history = [{'roi': -60.0, 'n': 10, 'hits': 0, 'date': '20260520'}]
        result = check_strategy_anomaly('c4', history)
        self.assertIsNotNone(result)
        self.assertIn('単日', result['details'])

    def test_no_anomaly_normal(self):
        history = [{'roi': 110.0, 'n': 8, 'hits': 2, 'date': '20260520'}]
        result = check_strategy_anomaly('c4', history)
        self.assertIsNone(result)

    def test_consecutive_negative(self):
        history = [
            {'roi': 80.0, 'n': 5, 'hits': 1, 'date': '20260518'},
            {'roi': 85.0, 'n': 5, 'hits': 1, 'date': '20260519'},
            {'roi': 90.0, 'n': 5, 'hits': 1, 'date': '20260520'},
        ]
        result = check_strategy_anomaly('c3', history)
        self.assertIsNotNone(result)
        self.assertIn('連続', result['details'])

    def test_zero_hit_rate(self):
        history = [{'roi': 0.0, 'n': 7, 'hits': 0, 'date': '20260520'}]
        result = check_strategy_anomaly('b1', history)
        self.assertIsNotNone(result)
        self.assertIn('hit = 0', result['details'])

    def test_action_is_paper_only(self):
        history = [{'roi': -60.0, 'n': 10, 'hits': 0, 'date': '20260520'}]
        result = check_strategy_anomaly('c4', history)
        self.assertEqual(result['action'], 'paper_only_auto_switch')

    def test_empty_history(self):
        result = check_strategy_anomaly('actual', [])
        self.assertIsNone(result)

    def test_consecutive_needs_exactly_3(self):
        # 2 日連続 negative は anomaly にならない
        history = [
            {'roi': 80.0, 'n': 5, 'hits': 1, 'date': '20260519'},
            {'roi': 90.0, 'n': 5, 'hits': 1, 'date': '20260520'},
        ]
        result = check_strategy_anomaly('c3', history)
        self.assertIsNone(result)

    def test_hit_zero_below_threshold_n(self):
        # N = 4 (< threshold 5) → hit=0 でも anomaly にならない
        history = [{'roi': 0.0, 'n': 4, 'hits': 0, 'date': '20260520'}]
        result = check_strategy_anomaly('c4', history)
        # roi=0 < -50? No. hit check: n=4 < 5? No anomaly from hit check.
        # roi=0 is not < -50, so no single_day anomaly either.
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
