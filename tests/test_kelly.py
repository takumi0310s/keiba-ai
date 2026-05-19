"""Unit tests for Kelly criterion module (D-1 2026-05-19)"""
import sys
import os
import unittest

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.kelly_criterion import calc_kelly_fraction, calc_dynamic_bet_amount


class TestKellyFraction(unittest.TestCase):
    def test_positive_edge(self):
        """High probability + decent odds → positive Kelly fraction."""
        f = calc_kelly_fraction(0.35, 50.0)
        self.assertGreater(f, 0.0)

    def test_zero_edge_low_prob(self):
        """Very low probability → Kelly fraction is 0 (no positive edge)."""
        f = calc_kelly_fraction(0.02, 50.0)
        self.assertGreaterEqual(f, 0.0)

    def test_safety_factor_proportional(self):
        """Doubling safety_factor should double the returned fraction."""
        f1 = calc_kelly_fraction(0.35, 50.0, safety_factor=0.25)
        f2 = calc_kelly_fraction(0.35, 50.0, safety_factor=0.50)
        self.assertAlmostEqual(f2, f1 * 2.0, places=5)

    def test_zero_score(self):
        """score=0 → 0."""
        self.assertEqual(calc_kelly_fraction(0.0, 50.0), 0.0)

    def test_unity_odds(self):
        """odds=1 → 0 (no payout beyond stake)."""
        self.assertEqual(calc_kelly_fraction(0.5, 1.0), 0.0)


class TestDynamicBetAmount(unittest.TestCase):
    def test_within_range(self):
        """Result must be in [300, 700]."""
        scores = [0.35, 0.22, 0.15, 0.10, 0.08, 0.05]
        bet = calc_dynamic_bet_amount(scores)
        self.assertGreaterEqual(bet, 300)
        self.assertLessEqual(bet, 700)

    def test_multiple_of_100(self):
        """Result must be a multiple of 100."""
        scores = [0.35, 0.22, 0.15]
        bet = calc_dynamic_bet_amount(scores)
        self.assertEqual(bet % 100, 0)

    def test_low_roi_reduces_bet(self):
        """Poor recent ROI (<80%) should produce a bet <= normal-ROI bet."""
        scores = [0.35, 0.22, 0.15]
        bet_normal = calc_dynamic_bet_amount(scores, rolling_roi=1.0)
        bet_bad    = calc_dynamic_bet_amount(scores, rolling_roi=0.70)
        self.assertLessEqual(bet_bad, bet_normal)

    def test_empty_scores_returns_min(self):
        """Empty scores list → min_bet (300)."""
        bet = calc_dynamic_bet_amount([])
        self.assertGreaterEqual(bet, 300)

    def test_high_roi_does_not_exceed_max(self):
        """Even with excellent rolling ROI, bet must not exceed max_bet."""
        scores = [0.35, 0.22, 0.15]
        bet = calc_dynamic_bet_amount(scores, rolling_roi=2.0, max_bet=700)
        self.assertLessEqual(bet, 700)


if __name__ == '__main__':
    unittest.main(verbosity=2)
