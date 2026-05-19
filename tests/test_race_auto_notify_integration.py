"""Integration tests for race_auto_notify strategy layer (D-5 2026-05-19).

Tests that strategy flags/constants exist in race_auto_notify.py and
that the module is syntactically valid.  Does NOT import the module at
module level to avoid heavy side-effects (Discord / scheduler / etc.).
"""
import unittest
import sys
import os
import py_compile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

RAN_PATH = os.path.join(os.path.dirname(__file__), '..', 'tools', 'race_auto_notify.py')


class TestRaceAutoNotifySyntax(unittest.TestCase):
    """race_auto_notify.py が構文エラーなく読み込めること"""

    def test_no_syntax_errors(self):
        """race_auto_notify.py に syntax error がないこと"""
        # py_compile.compile returns the .pyc path on success (not None)
        # Use doraise=True so it raises SyntaxError on failure instead
        try:
            py_compile.compile(RAN_PATH, doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"Syntax error in race_auto_notify.py: {e}")

    def test_file_exists(self):
        """tools/race_auto_notify.py が存在すること"""
        self.assertTrue(os.path.isfile(RAN_PATH))


class TestStrategyFlagsInSource(unittest.TestCase):
    """race_auto_notify.py ソース内に各 STRATEGY フラグが存在すること"""

    def setUp(self):
        with open(RAN_PATH, encoding='utf-8') as f:
            self.src = f.read()

    def test_strategy_c4_enabled_present(self):
        """STRATEGY_C4_ENABLED が定義されていること"""
        self.assertIn('STRATEGY_C4_ENABLED', self.src)

    def test_strategy_c3_enabled_present(self):
        """STRATEGY_C3_ENABLED が定義されていること"""
        self.assertIn('STRATEGY_C3_ENABLED', self.src)

    def test_strategy_b1_paper_only_present(self):
        """STRATEGY_B1_PAPER_ONLY が定義されていること"""
        self.assertIn('STRATEGY_B1_PAPER_ONLY', self.src)

    def test_strategy_b2_paper_only_present(self):
        """STRATEGY_B2_PAPER_ONLY が定義されていること"""
        self.assertIn('STRATEGY_B2_PAPER_ONLY', self.src)

    def test_strategy_c2_paper_only_present(self):
        """STRATEGY_C2_PAPER_ONLY が定義されていること"""
        self.assertIn('STRATEGY_C2_PAPER_ONLY', self.src)

    def test_strategy_c1_paper_only_present(self):
        """STRATEGY_C1_PAPER_ONLY が定義されていること"""
        self.assertIn('STRATEGY_C1_PAPER_ONLY', self.src)

    def test_c4_condition_logic_correct(self):
        """C4 logic: cond_key == 'A' and 1600 <= distance <= 1800 がソース内に存在"""
        self.assertIn("cond_key == 'A' and 1600 <= distance <= 1800", self.src)

    def test_b2_min_pop_rank_is_3(self):
        """B2 MIN_POP_RANK デフォルト値が 3 であること"""
        self.assertIn('STRATEGY_B2_MIN_POP_RANK = 3', self.src)

    def test_c2_low_odds_threshold(self):
        """C2 低オッズ閾値 1.5 がソース内に存在"""
        self.assertIn('< 1.5', self.src)

    def test_c2_high_odds_threshold(self):
        """C2 高オッズ閾値 20.0 がソース内に存在"""
        self.assertIn('> 20.0', self.src)


class TestStrategyFiltersModule(unittest.TestCase):
    """tools/strategy_filters.py の関数シグネチャと基本動作"""

    def setUp(self):
        from tools.strategy_filters import (
            should_skip_c4,
            should_skip_b1,
            should_bet_b2,
            should_skip_c2,
            build_trio_bets,
        )
        self.should_skip_c4 = should_skip_c4
        self.should_skip_b1 = should_skip_b1
        self.should_bet_b2 = should_bet_b2
        self.should_skip_c2 = should_skip_c2
        self.build_trio_bets = build_trio_bets

    def test_all_functions_callable(self):
        """全 strategy_filters 関数が呼び出し可能であること"""
        self.assertTrue(callable(self.should_skip_c4))
        self.assertTrue(callable(self.should_skip_b1))
        self.assertTrue(callable(self.should_bet_b2))
        self.assertTrue(callable(self.should_skip_c2))
        self.assertTrue(callable(self.build_trio_bets))

    def test_build_trio_bets_returns_list(self):
        """build_trio_bets の戻り値が list であること"""
        result = self.build_trio_bets(1, 2, 3, 4, 5, 6)
        self.assertIsInstance(result, list)

    def test_build_trio_bets_elements_are_tuples(self):
        """build_trio_bets の各要素が tuple であること"""
        result = self.build_trio_bets(1, 2, 3, 4, 5, 6)
        for bet in result:
            self.assertIsInstance(bet, tuple)

    def test_build_trio_bets_each_bet_has_3_horses(self):
        """trio 各 bet が 3 頭であること"""
        result = self.build_trio_bets(1, 2, 3, 4, 5, 6)
        for bet in result:
            self.assertEqual(len(bet), 3)

    def test_strategy_filters_returns_bool(self):
        """各フィルタ関数が bool を返すこと"""
        self.assertIsInstance(self.should_skip_c4('A', 1600), bool)
        self.assertIsInstance(self.should_skip_b1(1), bool)
        self.assertIsInstance(self.should_bet_b2(3), bool)
        self.assertIsInstance(self.should_skip_c2(3.0), bool)


if __name__ == '__main__':
    unittest.main(verbosity=2)
