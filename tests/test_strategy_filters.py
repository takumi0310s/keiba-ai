"""
Regression tests for strategy layer filters (C3, C4, B-1, B-2, C-2).
Added: 2026-05-19 (Day 1 A-4 → D-5 active化)

Logic source: tools/race_auto_notify.py predict_and_notify() inline strategies.
Pure-logic helpers: tools/strategy_filters.py
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.strategy_filters import (
    should_skip_c4,
    build_trio_bets,
    should_skip_b1,
    should_bet_b2,
    should_skip_c2,
)


class TestC4Filter(unittest.TestCase):
    """C4: Cond-A 1600-1800m skip"""

    def test_c4_skip_cond_a_1600(self):
        """Cond-A + 1600m -> skip"""
        self.assertTrue(should_skip_c4('A', 1600))

    def test_c4_skip_cond_a_1800(self):
        """Cond-A + 1800m -> skip"""
        self.assertTrue(should_skip_c4('A', 1800))

    def test_c4_no_skip_cond_a_2000(self):
        """Cond-A + 2000m -> no skip"""
        self.assertFalse(should_skip_c4('A', 2000))

    def test_c4_no_skip_cond_c_1600(self):
        """Cond-C + 1600m -> no skip (only Cond-A affected)"""
        self.assertFalse(should_skip_c4('C', 1600))

    def test_c4_no_skip_boundary_1599(self):
        """Cond-A + 1599m -> no skip (just below boundary)"""
        self.assertFalse(should_skip_c4('A', 1599))

    def test_c4_no_skip_boundary_1801(self):
        """Cond-A + 1801m -> no skip (above upper bound 1800)"""
        self.assertFalse(should_skip_c4('A', 1801))


class TestC3Formation(unittest.TestCase):
    """C3: pos2 (T1-T2-T4) formation removal"""

    def test_c3_removes_bet2(self):
        """C3 enabled -> 7 bets -> 6 bets (bet2 removed)"""
        bets_normal = build_trio_bets(1, 2, 3, 4, 5, 6, apply_c3=False)
        bets_c3 = build_trio_bets(1, 2, 3, 4, 5, 6, apply_c3=True)
        self.assertEqual(len(bets_normal), 7)
        self.assertEqual(len(bets_c3), 6)
        # bet2 = (1,2,4) should be absent
        self.assertNotIn((1, 2, 4), bets_c3)

    def test_c3_all_other_bets_present(self):
        """C3: bet2 removed, all 6 remaining bets intact"""
        bets_c3 = build_trio_bets(1, 2, 3, 4, 5, 6, apply_c3=True)
        expected = [(1, 2, 3), (1, 2, 5), (1, 2, 6), (1, 3, 4), (1, 3, 5), (1, 3, 6)]
        for e in expected:
            self.assertIn(e, bets_c3, f"Missing bet {e}")

    def test_c3_disabled_keeps_7(self):
        """C3 disabled -> 7 bets retained"""
        bets = build_trio_bets(1, 2, 3, 4, 5, 6, apply_c3=False)
        self.assertEqual(len(bets), 7)


class TestB1Filter(unittest.TestCase):
    """B-1: skip when top1 horse is 1-ban_ninkyo"""

    def test_b1_skip_pop1(self):
        """top1 pop_rank=1 -> skip"""
        self.assertTrue(should_skip_b1(1))

    def test_b1_no_skip_pop2(self):
        """top1 pop_rank=2 -> no skip"""
        self.assertFalse(should_skip_b1(2))

    def test_b1_no_skip_pop3(self):
        """top1 pop_rank=3 -> no skip"""
        self.assertFalse(should_skip_b1(3))


class TestB2Filter(unittest.TestCase):
    """B-2: bet only when top1 is divergence (pop_rank >= 3)"""

    def test_b2_bet_pop3(self):
        """top1 pop_rank=3 -> bet"""
        self.assertTrue(should_bet_b2(3))

    def test_b2_bet_pop5(self):
        """top1 pop_rank=5 -> bet"""
        self.assertTrue(should_bet_b2(5))

    def test_b2_no_bet_pop1(self):
        """top1 pop_rank=1 -> no bet"""
        self.assertFalse(should_bet_b2(1))

    def test_b2_no_bet_pop2(self):
        """top1 pop_rank=2 -> no bet (not divergence)"""
        self.assertFalse(should_bet_b2(2))


class TestC2Filter(unittest.TestCase):
    """C-2: odds band filter"""

    def test_c2_skip_low_odds(self):
        """odds < 1.5 -> skip"""
        self.assertTrue(should_skip_c2(1.3))

    def test_c2_skip_high_odds(self):
        """odds > 20.0 -> skip"""
        self.assertTrue(should_skip_c2(21.0))

    def test_c2_no_skip_normal_odds(self):
        """odds 3.0 -> no skip"""
        self.assertFalse(should_skip_c2(3.0))

    def test_c2_no_skip_boundary_15(self):
        """odds exactly 1.5 -> no skip (boundary inclusive)"""
        self.assertFalse(should_skip_c2(1.5))

    def test_c2_no_skip_boundary_20(self):
        """odds exactly 20.0 -> no skip (boundary inclusive)"""
        self.assertFalse(should_skip_c2(20.0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
