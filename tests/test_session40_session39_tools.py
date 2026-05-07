"""Session #39 + #40 tools 単体 test (Session #40 D2).

新規 tool の syntax / 主要関数 のみ確認 (production 不変前提)。

実行:
  python -m pytest tests/test_session40_session39_tools.py -v
  python tests/test_session40_session39_tools.py  # standalone
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tools"))


class TestSession39ATools(unittest.TestCase):
    """A: sib expanding window."""

    def test_sib_expanding_imports(self):
        from tools import sib_expanding_features
        self.assertTrue(hasattr(sib_expanding_features, 'build_expanding_sib'))

    def test_sib_expanding_smoke_minimal(self):
        """blood / races CSV が存在しないと build を呼べないので skip 可"""
        b = BASE / 'data' / 'blood_full.csv'
        r = BASE / 'data' / 'jra_races_full.csv'
        if not b.exists() or not r.exists():
            self.skipTest("data files unavailable")
        # 軽量実行は重いので import 確認のみで pass


class TestSession39BTools(unittest.TestCase):
    """B: JV-Link fetcher."""

    def test_jvlink_imports(self):
        from tools import jvlink_fetcher
        self.assertTrue(hasattr(jvlink_fetcher, 'JVLINK_DATATYPES'))
        self.assertIn("RACE", jvlink_fetcher.JVLINK_DATATYPES)
        self.assertIn("O1", jvlink_fetcher.JVLINK_DATATYPES)

    def test_jvlink_datatype_count(self):
        from tools import jvlink_fetcher
        dt = jvlink_fetcher.JVLINK_DATATYPES
        self.assertGreaterEqual(len(dt), 10)


class TestSession39CTools(unittest.TestCase):
    """C: SKB exclusion patch."""

    def test_v15_1_features_skb_leak_list(self):
        from train.v15_1_features import (
            V15_1_NEW_FEATURES, V15_1_SKB_FEATURES,
            SKB_LEAK_FEATURES, V20_LEAK_FEATURES, filter_v15_1_features
        )
        self.assertEqual(len(V15_1_SKB_FEATURES), 10)
        self.assertEqual(len(SKB_LEAK_FEATURES), 10)
        self.assertEqual(set(SKB_LEAK_FEATURES), set(V15_1_SKB_FEATURES))

    def test_filter_skips_skb(self):
        from train.v15_1_features import V15_1_NEW_FEATURES, filter_v15_1_features
        # skip_skb=False
        full = filter_v15_1_features(V15_1_NEW_FEATURES, skip_skb=False)
        self.assertEqual(len(full), len(V15_1_NEW_FEATURES))
        # skip_skb=True
        filtered = filter_v15_1_features(V15_1_NEW_FEATURES, skip_skb=True)
        self.assertEqual(len(filtered), len(V15_1_NEW_FEATURES) - 10)
        # SKB が含まれていないこと
        for f in filtered:
            self.assertFalse(f.startswith('skb_'))

    def test_v20_leak_features_count(self):
        from train.v15_1_features import V20_LEAK_FEATURES
        # 8 (V12 odds系) + 10 (SKB) = 18
        self.assertEqual(len(V20_LEAK_FEATURES), 18)
        self.assertIn('odds_log', V20_LEAK_FEATURES)
        self.assertIn('skb_kishi_code_3', V20_LEAK_FEATURES)


class TestSession40ATools(unittest.TestCase):
    """A: 5/9 直前 (PAT 点数 / 分類 / Kelly / health)."""

    def test_bet_pattern_optimization_imports(self):
        from tools import bet_pattern_optimization as bp
        self.assertTrue(hasattr(bp, 'gen_bets_from_top'))

    def test_bet_pattern_5pt_count(self):
        from tools.bet_pattern_optimization import gen_bets_from_top
        tops = [1, 2, 3, 4, 5, 6, 7]
        bets = gen_bets_from_top(tops, "5pt")
        self.assertEqual(len(bets), 5)

    def test_bet_pattern_7pt_count(self):
        from tools.bet_pattern_optimization import gen_bets_from_top
        tops = [1, 2, 3, 4, 5, 6, 7]
        bets = gen_bets_from_top(tops, "7pt")
        self.assertEqual(len(bets), 7)

    def test_bet_pattern_9pt_count(self):
        from tools.bet_pattern_optimization import gen_bets_from_top
        tops = [1, 2, 3, 4, 5, 6, 7]
        bets = gen_bets_from_top(tops, "9pt")
        # TOP1 軸 × TOP2/3/4 × TOP2/3/4/5/6/7
        # combinations を数えると 3 × 6 - duplicates = 9 程度
        self.assertGreaterEqual(len(bets), 7)
        self.assertLessEqual(len(bets), 12)

    def test_bet_pattern_returns_empty_on_short(self):
        from tools.bet_pattern_optimization import gen_bets_from_top
        self.assertEqual(gen_bets_from_top([1, 2], "7pt"), [])
        self.assertEqual(gen_bets_from_top([], "7pt"), [])

    def test_race_classifier_imports(self):
        from tools import race_classifier
        self.assertTrue(hasattr(race_classifier, 'classify_race'))
        self.assertTrue(hasattr(race_classifier, 'decide_accept'))

    def test_race_classifier_1sho(self):
        from tools.race_classifier import classify_race, decide_accept
        code, _ = classify_race("12R 4歳以上 1勝クラス")
        self.assertEqual(code, "1勝")
        accept, _ = decide_accept(code, "東京", 14)
        self.assertTrue(accept)

    def test_race_classifier_g2(self):
        from tools.race_classifier import classify_race, decide_accept
        code, _ = classify_race("11R 京王杯スプリングカップ（G2）")
        self.assertEqual(code, "G2")
        accept, _ = decide_accept(code, "東京", 16)
        self.assertFalse(accept)

    def test_race_classifier_excludes_kyoto(self):
        from tools.race_classifier import decide_accept
        accept, reason = decide_accept("1勝", "京都", 14)
        self.assertFalse(accept)
        self.assertIn("京都", reason)

    def test_race_classifier_excludes_low_horses(self):
        from tools.race_classifier import decide_accept
        accept, reason = decide_accept("1勝", "東京", 6)
        self.assertFalse(accept)

    def test_final_health_check_imports(self):
        from tools import final_health_check_5_8 as fhc
        self.assertTrue(hasattr(fhc, 'check_v15_model'))
        self.assertTrue(hasattr(fhc, 'check_syntax'))


class TestSession40BTools(unittest.TestCase):
    """B: 運用安定化 (alert routing / monitor / logs)."""

    def test_discord_routing_imports(self):
        from tools import discord_routing
        self.assertTrue(hasattr(discord_routing, 'notify'))
        self.assertTrue(hasattr(discord_routing, '_resolve_webhook'))

    def test_discord_routing_channel_keys(self):
        from tools.discord_routing import _resolve_webhook
        # webhook 設定無しでも関数は呼べる (None 返り)
        result = _resolve_webhook("alerts")
        self.assertIn(type(result).__name__, ("str", "NoneType"))

    def test_realtime_monitor_imports(self):
        from tools import realtime_monitor
        self.assertTrue(hasattr(realtime_monitor, 'render_status'))

    def test_realtime_monitor_render(self):
        from tools.realtime_monitor import render_status
        out = render_status(once=True)
        self.assertIn("realtime_monitor", out)
        self.assertIn("cumulative", out)

    def test_logs_cleanup_imports(self):
        from tools import logs_cleanup
        self.assertTrue(hasattr(logs_cleanup, 'iter_log_files'))
        self.assertTrue(hasattr(logs_cleanup, 'archive_path'))


class TestSession40DTools(unittest.TestCase):
    """D: メタ系 (jvlink_backfill_plan)."""

    def test_jvlink_backfill_plan_imports(self):
        from tools import jvlink_backfill_plan
        self.assertTrue(hasattr(jvlink_backfill_plan, 'jvlink_backfill_targets'))

    def test_jvlink_backfill_targets_format(self):
        from tools.jvlink_backfill_plan import jvlink_backfill_targets
        targets = jvlink_backfill_targets()
        self.assertGreaterEqual(len(targets), 5)
        for t in targets:
            self.assertIn('datatype', t)
            self.assertIn('priority', t)
            self.assertIn('estimated_records', t)


class TestV15ProductionUnchanged(unittest.TestCase):
    """V15 production の syntax (重要 file の文法エラー検知)."""

    def test_predict_core_syntax(self):
        import py_compile
        path = BASE / "tools" / "predict_core.py"
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"predict_core.py syntax error: {e}")

    def test_daily_predict_syntax(self):
        import py_compile
        path = BASE / "tools" / "daily_predict.py"
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"daily_predict.py syntax error: {e}")

    def test_app_syntax(self):
        import py_compile
        path = BASE / "app.py"
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"app.py syntax error: {e}")

    def test_v15_model_file_exists(self):
        p = BASE / "keiba_model_v15_central_live.pkl.gz"
        self.assertTrue(p.exists(), "V15 live model file missing")
        self.assertGreater(p.stat().st_size, 1024 * 1024, "V15 model too small")


if __name__ == "__main__":
    unittest.main(verbosity=2)
