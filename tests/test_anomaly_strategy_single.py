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
        # 6/11 Fable sweep: roi semantics 修正 (roi_pct<50 が正) に伴い、
        # roi 起因の anomaly を出さない fixture (roi=60>=50) に変更。テスト意図は hit=0 の N 閾値のみ。
        history = [{'roi': 60.0, 'n': 4, 'hits': 0, 'date': '20260520'}]
        result = check_strategy_anomaly('c4', history)
        self.assertIsNone(result)

    def test_single_day_roi_pct_scale(self):
        """6/11 修正の本丸: roi_pct スケール (0-100+) で 単日 ROI<50% が発火する。
        旧実装は閾値 -50 (別スケール) + 'roi'キー誤りで永久不発だった。"""
        result = check_strategy_anomaly('c4', [{'roi': 30.0, 'n': 6, 'hits': 1, 'date': '20260607'}])
        self.assertIsNotNone(result)
        self.assertIn('< 50%', result['details'])
        # 賭けゼロの日 (n=0, roi_pct=0) は対象外
        result = check_strategy_anomaly('c4', [{'roi': 0.0, 'n': 0, 'hits': 0, 'date': '20260609'}])
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
