"""
Regression tests for strategy layer filters (C3, C4, B-1, B-2, C-2)
Added: 2026-05-19 (Day 1 A-4)
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestC4Filter(unittest.TestCase):
    """C4: Cond-A 1600-1800m skip"""

    def test_c4_skip_cond_a_1600(self):
        """Cond-A + 1600m -> skip"""
        try:
            from tools.race_auto_notify import should_skip_c4
            self.assertTrue(should_skip_c4('A', 1600))
        except ImportError:
            self.skipTest("should_skip_c4 not yet implemented")

    def test_c4_skip_cond_a_1800(self):
        """Cond-A + 1800m -> skip"""
        try:
            from tools.race_auto_notify import should_skip_c4
            self.assertTrue(should_skip_c4('A', 1800))
        except ImportError:
            self.skipTest("should_skip_c4 not yet implemented")

    def test_c4_no_skip_cond_a_2000(self):
        """Cond-A + 2000m -> no skip"""
        try:
            from tools.race_auto_notify import should_skip_c4
            self.assertFalse(should_skip_c4('A', 2000))
        except ImportError:
            self.skipTest("should_skip_c4 not yet implemented")

    def test_c4_no_skip_cond_c_1600(self):
        """Cond-C + 1600m -> no skip (only Cond-A affected)"""
        try:
            from tools.race_auto_notify import should_skip_c4
            self.assertFalse(should_skip_c4('C', 1600))
        except ImportError:
            self.skipTest("should_skip_c4 not yet implemented")

    def test_c4_no_skip_boundary_1599(self):
        """Cond-A + 1599m -> no skip (just below boundary)"""
        try:
            from tools.race_auto_notify import should_skip_c4
            self.assertFalse(should_skip_c4('A', 1599))
        except ImportError:
            self.skipTest("should_skip_c4 not yet implemented")

    def test_c4_skip_boundary_1801(self):
        """Cond-A + 1801m -> skip (just above boundary)"""
        try:
            from tools.race_auto_notify import should_skip_c4
            self.assertTrue(should_skip_c4('A', 1801))
        except ImportError:
            self.skipTest("should_skip_c4 not yet implemented")


class TestC3Formation(unittest.TestCase):
    """C3: pos2 (T1-T2-T4) formation removal"""

    def test_c3_removes_bet2(self):
        """C3 enabled -> 7 bets -> 6 bets (bet2 removed)"""
        try:
            from tools.race_auto_notify import build_trio_bets
            bets_normal = build_trio_bets(1, 2, 3, 4, 5, 6, apply_c3=False)
            bets_c3 = build_trio_bets(1, 2, 3, 4, 5, 6, apply_c3=True)
            self.assertEqual(len(bets_normal), 7)
            self.assertEqual(len(bets_c3), 6)
            # bet2 = (1,2,4) should be absent
            self.assertNotIn(tuple(sorted([1, 2, 4])), [tuple(sorted(b)) for b in bets_c3])
        except ImportError:
            self.skipTest("build_trio_bets not yet implemented")

    def test_c3_all_other_bets_present(self):
        """C3: bet2 removed, all 6 remaining bets intact"""
        try:
            from tools.race_auto_notify import build_trio_bets
            bets_c3 = build_trio_bets(1, 2, 3, 4, 5, 6, apply_c3=True)
            expected = [(1, 2, 3), (1, 2, 5), (1, 2, 6), (1, 3, 4), (1, 3, 5), (1, 3, 6)]
            bets_sorted = [tuple(sorted(b)) for b in bets_c3]
            for e in expected:
                self.assertIn(e, bets_sorted, f"Missing bet {e}")
        except ImportError:
            self.skipTest("build_trio_bets not yet implemented")

    def test_c3_disabled_keeps_7(self):
        """C3 disabled -> 7 bets retained"""
        try:
            from tools.race_auto_notify import build_trio_bets
            bets = build_trio_bets(1, 2, 3, 4, 5, 6, apply_c3=False)
            self.assertEqual(len(bets), 7)
        except ImportError:
            self.skipTest("build_trio_bets not yet implemented")


class TestB1Filter(unittest.TestCase):
    """B-1: skip when top1 horse is 1-ban_ninkyo"""

    def test_b1_skip_pop1(self):
        """top1 pop_rank=1 -> skip"""
        try:
            from tools.race_auto_notify import should_skip_b1
            self.assertTrue(should_skip_b1(1))
        except ImportError:
            self.skipTest("should_skip_b1 not yet implemented")

    def test_b1_no_skip_pop2(self):
        """top1 pop_rank=2 -> no skip"""
        try:
            from tools.race_auto_notify import should_skip_b1
            self.assertFalse(should_skip_b1(2))
        except ImportError:
            self.skipTest("should_skip_b1 not yet implemented")

    def test_b1_no_skip_pop3(self):
        """top1 pop_rank=3 -> no skip"""
        try:
            from tools.race_auto_notify import should_skip_b1
            self.assertFalse(should_skip_b1(3))
        except ImportError:
            self.skipTest("should_skip_b1 not yet implemented")


class TestB2Filter(unittest.TestCase):
    """B-2: bet only when top1 is divergence (pop_rank >= 3)"""

    def test_b2_bet_pop3(self):
        """top1 pop_rank=3 -> bet"""
        try:
            from tools.race_auto_notify import should_bet_b2
            self.assertTrue(should_bet_b2(3))
        except ImportError:
            self.skipTest("should_bet_b2 not yet implemented")

    def test_b2_bet_pop5(self):
        """top1 pop_rank=5 -> bet"""
        try:
            from tools.race_auto_notify import should_bet_b2
            self.assertTrue(should_bet_b2(5))
        except ImportError:
            self.skipTest("should_bet_b2 not yet implemented")

    def test_b2_no_bet_pop1(self):
        """top1 pop_rank=1 -> no bet"""
        try:
            from tools.race_auto_notify import should_bet_b2
            self.assertFalse(should_bet_b2(1))
        except ImportError:
            self.skipTest("should_bet_b2 not yet implemented")

    def test_b2_no_bet_pop2(self):
        """top1 pop_rank=2 -> no bet (not divergence)"""
        try:
            from tools.race_auto_notify import should_bet_b2
            self.assertFalse(should_bet_b2(2))
        except ImportError:
            self.skipTest("should_bet_b2 not yet implemented")


class TestC2Filter(unittest.TestCase):
    """C-2: odds band filter"""

    def test_c2_skip_low_odds(self):
        """odds < 1.5 -> skip"""
        try:
            from tools.race_auto_notify import should_skip_c2
            self.assertTrue(should_skip_c2(1.3))
        except ImportError:
            self.skipTest("should_skip_c2 not yet implemented")

    def test_c2_skip_high_odds(self):
        """odds > 20.0 -> skip"""
        try:
            from tools.race_auto_notify import should_skip_c2
            self.assertTrue(should_skip_c2(21.0))
        except ImportError:
            self.skipTest("should_skip_c2 not yet implemented")

    def test_c2_no_skip_normal_odds(self):
        """odds 3.0 -> no skip"""
        try:
            from tools.race_auto_notify import should_skip_c2
            self.assertFalse(should_skip_c2(3.0))
        except ImportError:
            self.skipTest("should_skip_c2 not yet implemented")

    def test_c2_no_skip_boundary_15(self):
        """odds exactly 1.5 -> no skip (boundary inclusive)"""
        try:
            from tools.race_auto_notify import should_skip_c2
            self.assertFalse(should_skip_c2(1.5))
        except ImportError:
            self.skipTest("should_skip_c2 not yet implemented")

    def test_c2_no_skip_boundary_20(self):
        """odds exactly 20.0 -> no skip (boundary inclusive)"""
        try:
            from tools.race_auto_notify import should_skip_c2
            self.assertFalse(should_skip_c2(20.0))
        except ImportError:
            self.skipTest("should_skip_c2 not yet implemented")


if __name__ == '__main__':
    unittest.main(verbosity=2)
