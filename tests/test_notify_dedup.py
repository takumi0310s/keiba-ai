"""Session #59: send_discord dedup 機構の test。

Discord Webhook 自体は mock し、 dedup 動作のみ検証。
"""
import os
import sys
import time
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

import notify  # noqa: E402


class FakeResponse:
    def __init__(self, status_code=204):
        self.status_code = status_code


class TestSendDiscordDedup(unittest.TestCase):
    def setUp(self):
        # 一時 cache file に切替
        self._tmp = tempfile.mkdtemp()
        self._orig_cache_path = notify._DEDUP_CACHE_PATH
        notify._DEDUP_CACHE_PATH = os.path.join(self._tmp, 'cache.json')
        # webhook URL 用 env を mock
        notify._ENV_CACHE = {'DISCORD_WEBHOOK_UPDATES': 'https://example.invalid/webhook',
                             'DISCORD_WEBHOOK_BETS': 'https://example.invalid/bets'}

    def tearDown(self):
        notify._DEDUP_CACHE_PATH = self._orig_cache_path
        notify._ENV_CACHE = None
        # cleanup
        try:
            os.remove(os.path.join(self._tmp, 'cache.json'))
        except FileNotFoundError:
            pass
        os.rmdir(self._tmp)

    @patch.object(notify.requests, 'post')
    def test_first_send_calls_post(self, mock_post):
        mock_post.return_value = FakeResponse(204)
        ok = notify.send_discord("title-A", "msg-A", channel="updates")
        self.assertTrue(ok)
        self.assertEqual(mock_post.call_count, 1)

    @patch.object(notify.requests, 'post')
    def test_duplicate_within_window_skipped(self, mock_post):
        mock_post.return_value = FakeResponse(204)
        # 1 回目: POST される
        ok1 = notify.send_discord("title-B", "msg-B", channel="updates")
        # 2 回目: 直後 → skip (POST 呼ばれない)
        ok2 = notify.send_discord("title-B", "msg-B", channel="updates")
        self.assertTrue(ok1)
        self.assertTrue(ok2)
        self.assertEqual(mock_post.call_count, 1, "2 回目は skip されるはず")

    @patch.object(notify.requests, 'post')
    def test_different_message_not_skipped(self, mock_post):
        mock_post.return_value = FakeResponse(204)
        ok1 = notify.send_discord("title-C", "msg-C-1", channel="updates")
        ok2 = notify.send_discord("title-C", "msg-C-2", channel="updates")
        self.assertTrue(ok1)
        self.assertTrue(ok2)
        self.assertEqual(mock_post.call_count, 2, "異なる msg は両方送信")

    @patch.object(notify.requests, 'post')
    def test_different_channel_not_skipped(self, mock_post):
        mock_post.return_value = FakeResponse(204)
        ok1 = notify.send_discord("title-D", "msg-D", channel="updates")
        ok2 = notify.send_discord("title-D", "msg-D", channel="bets")
        self.assertEqual(mock_post.call_count, 2, "channel 違いは両方送信")

    @patch.object(notify.requests, 'post')
    def test_dedup_disabled(self, mock_post):
        mock_post.return_value = FakeResponse(204)
        # dedup_window_sec=0 で完全無効化
        ok1 = notify.send_discord("title-E", "msg-E", channel="updates", dedup_window_sec=0)
        ok2 = notify.send_discord("title-E", "msg-E", channel="updates", dedup_window_sec=0)
        self.assertEqual(mock_post.call_count, 2, "dedup_window_sec=0 なら両方送信")

    @patch.object(notify.requests, 'post')
    def test_dedup_after_window_expires(self, mock_post):
        mock_post.return_value = FakeResponse(204)
        notify.send_discord("title-F", "msg-F", channel="updates", dedup_window_sec=1)
        time.sleep(1.2)
        notify.send_discord("title-F", "msg-F", channel="updates", dedup_window_sec=1)
        self.assertEqual(mock_post.call_count, 2, "window 経過後は再送")

    @patch.object(notify.requests, 'post')
    def test_failed_send_not_cached(self, mock_post):
        # 1 回目: 500 で失敗
        mock_post.return_value = FakeResponse(500)
        ok1 = notify.send_discord("title-G", "msg-G", channel="updates")
        self.assertFalse(ok1)
        # 2 回目: 200 で成功 (失敗時は cache 入れないので skip しない)
        mock_post.return_value = FakeResponse(200)
        ok2 = notify.send_discord("title-G", "msg-G", channel="updates")
        self.assertTrue(ok2)
        # POST: 1 回目 retry 3 回 + 2 回目 1 回 = 計 4 回
        self.assertGreaterEqual(mock_post.call_count, 2,
                                "失敗後は cache 入らず 2 回目も送信される")


if __name__ == '__main__':
    unittest.main()
