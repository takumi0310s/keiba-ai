"""
tests/test_strategy_rollback.py
Unit tests for tools/strategy_rollback.py (E-1 rollback mechanism)

6 required tests:
  1. get_rollback_state: file missing → default (enabled)
  2. set_rollback_state: JSON written correctly
  3. check_and_apply_rollback: anomaly present → c4/c3 disabled
  4. check_and_apply_rollback: no anomaly + cooldown NOT expired → no restore
  5. check_and_apply_rollback: no anomaly + cooldown expired → restored
  6. get_effective_flags: state file missing → (True, True)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tools.strategy_rollback as rb


class TestGetRollbackState(unittest.TestCase):
    """Test 1: file missing → default enabled state"""

    def test_default_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / 'nonexistent.json'
            with mock.patch.object(rb, 'ROLLBACK_STATE_FILE', missing_path):
                state = rb.get_rollback_state()
        self.assertTrue(state['c4_enabled'])
        self.assertTrue(state['c3_enabled'])
        self.assertEqual(state['reason'], '')
        self.assertEqual(state['timestamp'], '')

    def test_default_when_file_corrupt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / 'bad.json'
            bad_path.write_text("not-json", encoding='utf-8')
            with mock.patch.object(rb, 'ROLLBACK_STATE_FILE', bad_path):
                state = rb.get_rollback_state()
        self.assertTrue(state['c4_enabled'])
        self.assertTrue(state['c3_enabled'])


class TestSetRollbackState(unittest.TestCase):
    """Test 2: set_rollback_state writes correct JSON"""

    def test_json_written_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'state.json'
            with mock.patch.object(rb, 'ROLLBACK_STATE_FILE', path):
                rb.set_rollback_state(c4_enabled=False, c3_enabled=False, reason='test reason')
                with open(path, encoding='utf-8') as fp:
                    data = json.load(fp)
        self.assertFalse(data['c4_enabled'])
        self.assertFalse(data['c3_enabled'])
        self.assertEqual(data['reason'], 'test reason')
        self.assertIn('timestamp', data)
        self.assertNotEqual(data['timestamp'], '')

    def test_restore_written_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'state.json'
            with mock.patch.object(rb, 'ROLLBACK_STATE_FILE', path):
                rb.set_rollback_state(c4_enabled=True, c3_enabled=True, reason='restored')
                state = rb.get_rollback_state()
        self.assertTrue(state['c4_enabled'])
        self.assertTrue(state['c3_enabled'])


class TestCheckAndApplyRollbackAnomaly(unittest.TestCase):
    """Test 3: anomaly present → c4/c3 disabled"""

    def test_c4_anomaly_disables_both(self):
        anomalies = [{'strategy': 'c4', 'details': '単日 ROI -60pt', 'action': 'paper_only_auto_switch'}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'state.json'
            with mock.patch.object(rb, 'ROLLBACK_STATE_FILE', path):
                state = rb.check_and_apply_rollback(anomalies)
        self.assertFalse(state['c4_enabled'])
        self.assertFalse(state['c3_enabled'])
        self.assertIn('anomaly', state['reason'])

    def test_c3_anomaly_disables_both(self):
        anomalies = [{'strategy': 'c3', 'details': '連続 3 日 ROI < 100%', 'action': 'paper_only_auto_switch'}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'state.json'
            with mock.patch.object(rb, 'ROLLBACK_STATE_FILE', path):
                state = rb.check_and_apply_rollback(anomalies)
        self.assertFalse(state['c4_enabled'])
        self.assertFalse(state['c3_enabled'])

    def test_c3c4_anomaly_disables_both(self):
        anomalies = [{'strategy': 'c3c4', 'details': 'hit = 0/5 races', 'action': 'paper_only_auto_switch'}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'state.json'
            with mock.patch.object(rb, 'ROLLBACK_STATE_FILE', path):
                state = rb.check_and_apply_rollback(anomalies)
        self.assertFalse(state['c4_enabled'])
        self.assertFalse(state['c3_enabled'])

    def test_unrelated_strategy_anomaly_no_disable(self):
        # 'actual' or 'no_1pop' anomaly should NOT disable C4/C3
        anomalies = [{'strategy': 'actual', 'details': '連続 3 日 ROI < 100%', 'action': 'paper_only_auto_switch'}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'state.json'
            with mock.patch.object(rb, 'ROLLBACK_STATE_FILE', path):
                state = rb.check_and_apply_rollback(anomalies)
        self.assertTrue(state['c4_enabled'])
        self.assertTrue(state['c3_enabled'])


class TestCheckAndApplyRollbackCooldown(unittest.TestCase):
    """Test 4: no anomaly + cooldown NOT expired → disabled state kept"""

    def _make_recent_disabled_state(self, tmpdir: str) -> Path:
        path = Path(tmpdir) / 'state.json'
        # Rollback was applied 2 days ago (< 7 day cooldown)
        ts = (datetime.now() - timedelta(days=2)).isoformat()
        state = {'c4_enabled': False, 'c3_enabled': False, 'reason': 'anomaly detected: test', 'timestamp': ts}
        path.write_text(json.dumps(state), encoding='utf-8')
        return path

    def test_no_restore_within_cooldown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_recent_disabled_state(tmpdir)
            with mock.patch.object(rb, 'ROLLBACK_STATE_FILE', path):
                state = rb.check_and_apply_rollback([])  # no anomaly
        # Still disabled because cooldown not expired
        self.assertFalse(state['c4_enabled'])
        self.assertFalse(state['c3_enabled'])


class TestCheckAndApplyRollbackRestore(unittest.TestCase):
    """Test 5: no anomaly + cooldown expired → restored to enabled"""

    def _make_old_disabled_state(self, tmpdir: str) -> Path:
        path = Path(tmpdir) / 'state.json'
        # Rollback was applied 8 days ago (> 7 day cooldown)
        ts = (datetime.now() - timedelta(days=8)).isoformat()
        state = {'c4_enabled': False, 'c3_enabled': False, 'reason': 'anomaly detected: test', 'timestamp': ts}
        path.write_text(json.dumps(state), encoding='utf-8')
        return path

    def test_restore_after_cooldown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_old_disabled_state(tmpdir)
            with mock.patch.object(rb, 'ROLLBACK_STATE_FILE', path):
                state = rb.check_and_apply_rollback([])  # no anomaly
        # Restored after cooldown
        self.assertTrue(state['c4_enabled'])
        self.assertTrue(state['c3_enabled'])
        self.assertIn('restored', state['reason'].lower())


class TestGetEffectiveFlags(unittest.TestCase):
    """Test 6: state file missing → (True, True)"""

    def test_defaults_to_true_true_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / 'no_state.json'
            with mock.patch.object(rb, 'ROLLBACK_STATE_FILE', missing_path):
                c4, c3 = rb.get_effective_flags()
        self.assertTrue(c4)
        self.assertTrue(c3)

    def test_returns_false_false_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'state.json'
            ts = datetime.now().isoformat()
            state = {'c4_enabled': False, 'c3_enabled': False, 'reason': 'test', 'timestamp': ts}
            path.write_text(json.dumps(state), encoding='utf-8')
            with mock.patch.object(rb, 'ROLLBACK_STATE_FILE', path):
                c4, c3 = rb.get_effective_flags()
        self.assertFalse(c4)
        self.assertFalse(c3)

    def test_returns_true_true_on_exception(self):
        with mock.patch.object(rb, 'get_rollback_state', side_effect=RuntimeError('boom')):
            c4, c3 = rb.get_effective_flags()
        self.assertTrue(c4)
        self.assertTrue(c3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
