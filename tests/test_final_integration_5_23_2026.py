"""
F-2 最終 integration test — 5/23 fire 前 final safety net
2026-05-19 実装

V15 -> C3+C4 -> 戦略⑦案C -> race_auto_notify -> race_notify_log_v2 -> aggregator
の全 chain を 1R 単位で verify。

★ NEVER ルール 遵守 ★:
- V15 .pkl.gz / predict_core / daily_predict / app.py logic 変更なし
- 実 live fetch なし (mock/stub のみ)
- schtasks /Create なし
- destructive git 操作なし
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'tools'))

# ---------------------------------------------------------------------------
# Mock data: ヴィクトリアマイル想定 (5/16-5/17 相当)
# ---------------------------------------------------------------------------
MOCK_RACE = {
    'race_id': '202605020811',   # 東京11R ヴィクトリアマイル想定
    'venue': '東京',
    'cond_key': 'C',
    'distance': 1600,
    'surface': '芝',
    'condition': '良',
    'predictions': [
        {'horse_num': 1,  'score': 0.85, 'odds': 2.5,  'pop_rank': 1, 'top_num': True},
        {'horse_num': 3,  'score': 0.72, 'odds': 5.1,  'pop_rank': 2, 'top_num': True},
        {'horse_num': 7,  'score': 0.68, 'odds': 8.3,  'pop_rank': 3, 'top_num': True},
        {'horse_num': 11, 'score': 0.61, 'odds': 12.0, 'pop_rank': 4, 'top_num': True},
        {'horse_num': 5,  'score': 0.55, 'odds': 15.0, 'pop_rank': 5, 'top_num': False},
        {'horse_num': 9,  'score': 0.48, 'odds': 20.0, 'pop_rank': 6, 'top_num': False},
    ],
    'real_top3': [3, 1, 7],
    'trio_payout': 1820,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStrategyC4Filter(unittest.TestCase):
    """C4: Cond-A 1600-1800m drag skip"""

    def setUp(self):
        from tools.strategy_filters import should_skip_c4
        self.should_skip_c4 = should_skip_c4

    def test_strategy_c4_cond_a_1600_1800_skip(self):
        """C4 フィルタ: Cond-A 1600m → skip"""
        self.assertTrue(self.should_skip_c4('A', 1600))
        self.assertTrue(self.should_skip_c4('A', 1700))
        self.assertTrue(self.should_skip_c4('A', 1800))

    def test_strategy_c4_cond_a_1900_no_skip(self):
        """C4 フィルタ: Cond-A 1900m → no skip (boundary 1800 を超える)"""
        self.assertFalse(self.should_skip_c4('A', 1900))
        self.assertFalse(self.should_skip_c4('A', 2000))
        self.assertFalse(self.should_skip_c4('A', 2400))

    def test_strategy_c4_cond_c_1600_no_skip(self):
        """C4 フィルタ: Cond-C 1600m → no skip (C4 は Cond-A のみ対象)"""
        self.assertFalse(self.should_skip_c4('C', 1600))
        self.assertFalse(self.should_skip_c4('B', 1700))
        self.assertFalse(self.should_skip_c4('D', 1600))
        self.assertFalse(self.should_skip_c4('X', 1800))


class TestStrategyC3Formation(unittest.TestCase):
    """C3: pos2 (T1-T2-T4) bet 除外 → 6点"""

    def setUp(self):
        from tools.strategy_filters import build_trio_bets
        self.build_trio_bets = build_trio_bets

    def test_strategy_c3_build_trio_bets_6points(self):
        """C3 enabled: trio bets → 6点 (T1-T2-T4 bet 除外)"""
        bets = self.build_trio_bets(1, 3, 7, 11, 5, 9, apply_c3=True)
        self.assertEqual(len(bets), 6, f"Expected 6 bets, got {len(bets)}: {bets}")
        # bet2 = (n1=1, n2=3, n4=11) → sorted = (1,3,11) は除外される
        self.assertNotIn((1, 3, 11), bets, "C3 should remove (1,3,11) bet")

    def test_strategy_c3_build_trio_bets_7points_disabled(self):
        """C3 disabled: 7点 保持"""
        bets = self.build_trio_bets(1, 3, 7, 11, 5, 9, apply_c3=False)
        self.assertEqual(len(bets), 7, f"Expected 7 bets, got {len(bets)}: {bets}")
        # bet2 は残る
        self.assertIn((1, 3, 11), bets, "C3 disabled: (1,3,11) should remain")


class TestKellyCriterion(unittest.TestCase):
    """Kelly criterion bet amount calculation"""

    def setUp(self):
        from tools.kelly_criterion import calc_dynamic_bet_amount
        self.calc_dynamic_bet_amount = calc_dynamic_bet_amount

    def test_kelly_bet_300_700_range(self):
        """Kelly: 返値が ¥300-¥700 の範囲内"""
        scores = [p['score'] for p in MOCK_RACE['predictions']]
        amount = self.calc_dynamic_bet_amount(scores, bankroll=50000)
        self.assertGreaterEqual(amount, 300, f"Bet {amount} below min 300")
        self.assertLessEqual(amount, 700, f"Bet {amount} above max 700")

    def test_kelly_high_score_high_bet(self):
        """score=0.85, odds=5.0 → ¥700 近辺 (max cap)"""
        scores = [0.85, 0.72, 0.68]
        amount = self.calc_dynamic_bet_amount(scores, bankroll=50000, max_bet=700)
        self.assertGreaterEqual(amount, 300)
        self.assertLessEqual(amount, 700)


class TestRollbackState(unittest.TestCase):
    """strategy_rollback: state 読み込み + anomaly 適用"""

    def test_rollback_state_default_enabled(self):
        """rollback state なし → (True, True) — デフォルト全 enabled"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from tools.strategy_rollback import get_rollback_state, ROLLBACK_STATE_FILE
            # state file が 存在しない tmpdir を使って check
            fake_state_path = Path(tmpdir) / 'strategy_rollback_state.json'
            with mock.patch('tools.strategy_rollback.ROLLBACK_STATE_FILE', fake_state_path):
                from tools import strategy_rollback as rb
                rb.ROLLBACK_STATE_FILE = fake_state_path
                state = rb.get_rollback_state()
            self.assertTrue(state.get('c4_enabled', True), "c4_enabled should default True")
            self.assertTrue(state.get('c3_enabled', True), "c3_enabled should default True")

    def test_rollback_no_anomaly_no_change(self):
        """anomaly list 空 → state 変わらず (enabled のまま)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_state_path = Path(tmpdir) / 'strategy_rollback_state.json'
            with mock.patch('tools.strategy_rollback.ROLLBACK_STATE_FILE', fake_state_path):
                import tools.strategy_rollback as rb
                rb.ROLLBACK_STATE_FILE = fake_state_path
                result = rb.check_and_apply_rollback([])
            self.assertTrue(result['c4_enabled'])
            self.assertTrue(result['c3_enabled'])

    def test_rollback_c4_anomaly_disables_c4(self):
        """c4 anomaly → c4_enabled=False"""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_state_path = Path(tmpdir) / 'strategy_rollback_state.json'
            import tools.strategy_rollback as rb
            orig = rb.ROLLBACK_STATE_FILE
            rb.ROLLBACK_STATE_FILE = fake_state_path
            try:
                anomalies = [{
                    'strategy': 'c4',
                    'anomalies': ['単日 ROI -60.0pt < -50pt'],
                    'action': 'paper_only_auto_switch',
                    'details': '単日 ROI -60.0pt < -50pt',
                }]
                result = rb.check_and_apply_rollback(anomalies)
                self.assertFalse(result['c4_enabled'], "c4 anomaly should disable c4")
                self.assertFalse(result['c3_enabled'], "c4 anomaly should also disable c3")
            finally:
                rb.ROLLBACK_STATE_FILE = orig


class TestRaceNotifyLogV2(unittest.TestCase):
    """race_notify_log_v2: phase2 schema + strategy_formations フィールド"""

    def test_race_notify_log_v2_phase2_schema(self):
        """log_phase2 が strategy_formations フィールドを保持 (8 keys 全て存在)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ['RACE_NOTIFY_LOG_V2_ROOT'] = tmpdir
            try:
                import tools.race_notify_log_v2 as rnl
                race_meta = {
                    'cond_key': MOCK_RACE['cond_key'],
                    'distance': MOCK_RACE['distance'],
                    'venue': MOCK_RACE['venue'],
                }
                rnl.log_phase2(
                    race_id=MOCK_RACE['race_id'],
                    race_meta=race_meta,
                    formation_actual=[(1, 3, 7), (1, 3, 11)],
                    vote_time_odds={'1': 2.5, '3': 5.1},
                    strategy_7c_skip=False,
                    cond_key=MOCK_RACE['cond_key'],
                    bet_type='trio',
                    predictions=MOCK_RACE['predictions'],
                    date_str='20260523',
                )
                p2_file = Path(tmpdir) / '20260523' / 'phase2' / f"{MOCK_RACE['race_id']}.json"
                self.assertTrue(p2_file.exists(), "phase2 JSON not written")
                with open(p2_file, encoding='utf-8') as f:
                    data = json.load(f)
                self.assertIn('strategy_formations', data, "strategy_formations missing from phase2 log")
                sf = data['strategy_formations']
                expected_keys = ['actual', 'c3', 'c4', 'c3c4', 'no_1pop', 'divergence', 'ev_filter', 'odds_filter']
                for k in expected_keys:
                    self.assertIn(k, sf, f"strategy_formations missing key: {k}")
            finally:
                del os.environ['RACE_NOTIFY_LOG_V2_ROOT']


class TestAggregator(unittest.TestCase):
    """aggregator: aggregate_day が 8 strategy keys 持つ"""

    def test_aggregator_strategy_stats_keys(self):
        """aggregate_day が strategy_stats に 8 keys 返す"""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ['RACE_NOTIFY_LOG_V2_ROOT'] = tmpdir
            try:
                import tools.race_notify_log_v2 as rnl
                import tools.race_notify_log_v2_aggregator as agg
                # 既存の LOG_DIR を tmpdir で上書き
                agg.LOG_DIR = Path(tmpdir)

                # phase2 + phase3 を書き込む
                race_meta = {
                    'cond_key': 'C',
                    'distance': 1600,
                    'venue': '東京',
                }
                rnl.log_phase2(
                    race_id='202605020811',
                    race_meta=race_meta,
                    formation_actual=[(1, 3, 7)],
                    strategy_7c_skip=False,
                    cond_key='C',
                    bet_type='trio',
                    predictions=MOCK_RACE['predictions'],
                    date_str='20260523',
                )
                rnl.log_phase3(
                    race_id='202605020811',
                    real_top3=[3, 1, 7],
                    real_payouts={'trio': 1820},
                    hit_miss={'trio_hit': True},
                    date_str='20260523',
                )
                result = agg.aggregate_day('20260523')
                self.assertIn('strategy_stats', result, "aggregate_day missing strategy_stats")
                ss = result['strategy_stats']
                expected_keys = ['actual', 'c3', 'c4', 'c3c4', 'no_1pop', 'divergence', 'ev_filter', 'odds_filter']
                for k in expected_keys:
                    self.assertIn(k, ss, f"strategy_stats missing key: {k}")
                self.assertEqual(len(ss), 8, f"Expected 8 strategy keys, got {len(ss)}")
            finally:
                del os.environ['RACE_NOTIFY_LOG_V2_ROOT']
                # restore
                import importlib
                importlib.reload(agg)


class TestAnomalyScan(unittest.TestCase):
    """anomaly_auto_detector: run_strategy_anomaly_scan が list 返す"""

    def test_anomaly_scan_returns_list(self):
        """run_strategy_anomaly_scan が list 返す (空でも OK)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from tools.anomaly_auto_detector import run_strategy_anomaly_scan
            result = run_strategy_anomaly_scan(log_dir=tmpdir)
            self.assertIsInstance(result, list, f"Expected list, got {type(result)}")


class TestStrategyChainC3C4Combined(unittest.TestCase):
    """C4 skip + C3 → 戦略⑦案C 全体 chain 動作"""

    def test_strategy_chain_c3c4_combined(self):
        """C4 skip (Cond-A 1600m) → c4/c3c4=None; C3 有効 → c3 が 6 bets"""
        from tools.strategy_filters import should_skip_c4, build_trio_bets
        import tools.race_notify_log_v2 as rnl

        # Cond-A 1600m レース
        race_a = {
            'race_id': '202605020512',
            'cond_key': 'A',
            'distance': 1600,
            'venue': '東京',
        }
        preds = MOCK_RACE['predictions']

        # step 1: C4 skip check
        c4_skip = should_skip_c4(race_a['cond_key'], race_a['distance'])
        self.assertTrue(c4_skip, "C4 should skip Cond-A 1600m")

        # step 2: C3 formation (6 bets)
        nums = [p['horse_num'] for p in preds[:6]]
        bets_c3 = build_trio_bets(*nums, apply_c3=True)
        self.assertEqual(len(bets_c3), 6, f"C3 should yield 6 bets, got {len(bets_c3)}")

        # step 3: build_strategy_formations
        sf = rnl.build_strategy_formations(preds, race_a)
        # c4 / c3c4 は skip (Cond-A 1600m)
        self.assertIsNone(sf.get('c4'), "c4 should be None for Cond-A 1600m")
        self.assertIsNone(sf.get('c3c4'), "c3c4 should be None for Cond-A 1600m")
        # actual / c3 は生成される (戦略⑦ base_skip は 京都/B/E/X のみ)
        self.assertIsNotNone(sf.get('actual'), "actual should not be None for Cond-A")
        self.assertIsNotNone(sf.get('c3'), "c3 should not be None for Cond-A")
        # c3 は 6 bets
        self.assertEqual(len(sf['c3']), 6, f"c3 should have 6 bets, got {len(sf.get('c3', []))}")


class TestAppPySyntax(unittest.TestCase):
    """app.py が python -m py_compile PASS (syntax check)"""

    def test_app_py_syntax_valid(self):
        """app.py syntax check via py_compile"""
        app_path = BASE_DIR / 'app.py'
        self.assertTrue(app_path.exists(), "app.py not found")
        result = subprocess.run(
            [sys.executable, '-m', 'py_compile', str(app_path)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode, 0,
            f"app.py syntax error:\n{result.stderr}"
        )


class TestDailyDiscordReport(unittest.TestCase):
    """build_report_message が 5 フィールド含む"""

    def test_daily_discord_report_build_message(self):
        """build_report_message が date / 本日 / 累計 / 撤退余裕 の各フィールドを含む"""
        from tools.daily_discord_report import build_report_message
        stats = {
            'total_n': 100,
            'total_pnl': -6920,
            'total_roi': 98.34,
            'today_n': 5,
            'today_pnl': 700,
            'today_roi': 120.0,
            'strat_c_n': 0,
            'strat_c_pnl': 0,
            'strat_c_roi': 0.0,
            'has_strat_c': False,
            'withdrawal_margin': 43080,
        }
        msg = build_report_message(stats, date_str='20260523')
        # 5 フィールド: date label, 本日, 累計, ROI, 撤退
        self.assertIn('2026/05/23', msg, "date label missing")
        self.assertIn('本日', msg, "'本日' field missing")
        self.assertIn('累計', msg, "'累計' field missing")
        self.assertIn('ROI', msg, "'ROI' field missing")
        self.assertIn('撤退', msg, "'撤退' field missing")


class TestAdminVerifyV2Imports(unittest.TestCase):
    """admin_verify_v2.py が import エラーなし"""

    def test_admin_verify_v2_imports(self):
        """admin_verify_v2.py import 確認"""
        admin_path = BASE_DIR / 'tools' / 'admin_verify_v2.py'
        self.assertTrue(admin_path.exists(), "tools/admin_verify_v2.py not found")
        result = subprocess.run(
            [sys.executable, '-c', f'import importlib.util; '
             f'spec = importlib.util.spec_from_file_location("admin_verify_v2", r"{admin_path}"); '
             f'mod = importlib.util.module_from_spec(spec); '
             f'spec.loader.exec_module(mod)'],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
        )
        self.assertEqual(
            result.returncode, 0,
            f"admin_verify_v2.py import error:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# Additional / bonus tests for robustness
# ---------------------------------------------------------------------------

class TestStrategyFormationsExtra(unittest.TestCase):
    """build_strategy_formations edge cases"""

    def test_build_strategy_formations_cond_c_has_actual(self):
        """Cond-C 1600m: actual / c3 / c4 / c3c4 全て生成 (base_skip=False)"""
        import tools.race_notify_log_v2 as rnl
        preds = MOCK_RACE['predictions']
        race_meta = {'cond_key': 'C', 'distance': 1600, 'venue': '東京'}
        sf = rnl.build_strategy_formations(preds, race_meta)
        # Cond-C は base_skip=False、 C4 skip=False (A のみ)
        self.assertIsNotNone(sf.get('actual'), "Cond-C actual should not be None")
        self.assertIsNotNone(sf.get('c4'), "Cond-C c4 should not be None (c4 only skips Cond-A)")
        self.assertIsNotNone(sf.get('c3c4'), "Cond-C c3c4 should not be None")

    def test_build_strategy_formations_kyoto_base_skip(self):
        """京都 venue → base_skip=True → actual / c3 / c4 / c3c4 全て None"""
        import tools.race_notify_log_v2 as rnl
        preds = MOCK_RACE['predictions']
        race_meta = {'cond_key': 'A', 'distance': 2000, 'venue': '京都'}
        sf = rnl.build_strategy_formations(preds, race_meta)
        self.assertIsNone(sf.get('actual'), "京都 actual should be None")
        self.assertIsNone(sf.get('c3'), "京都 c3 should be None")
        self.assertIsNone(sf.get('c4'), "京都 c4 should be None")

    def test_compute_strategy_results_hit(self):
        """compute_strategy_results: 的中 → hit=True / pnl = payout - 700"""
        import tools.race_notify_log_v2 as rnl
        # 3着が 3,1,7 → trio_payout 1820
        formations = {
            'actual': [[1, 3, 7]],
            'c3': [[1, 3, 7]],
        }
        for k in ['c4', 'c3c4', 'no_1pop', 'divergence', 'ev_filter', 'odds_filter']:
            formations[k] = None
        results = rnl.compute_strategy_results(
            formations,
            real_top3=[3, 1, 7],
            real_payout_trio=1820,
        )
        self.assertTrue(results['actual']['hit'], "actual should hit")
        self.assertEqual(results['actual']['payout'], 1820)
        self.assertEqual(results['actual']['pnl'], 1820 - 700)
        self.assertTrue(results['c3']['hit'])
        self.assertTrue(results['c4']['skipped'], "c4 formation=None should be skipped")


if __name__ == '__main__':
    unittest.main(verbosity=2)
