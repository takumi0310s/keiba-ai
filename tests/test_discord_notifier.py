"""tools/discord_notifier.py のテスト (実送信はモック)."""
from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from tools import discord_notifier as dn  # noqa: E402


def test_dedup_skip_within_ttl(tmp_path, monkeypatch):
    """同じ dedup_key を TTL 内に 2 回送ると 2 回目スキップ。"""
    monkeypatch.setattr(dn, "STATE_PATH", tmp_path / "state.json")
    with patch("tools.discord_notifier.subprocess.run") as mock_run:
        r1 = dn.notify("t", "s", "b", severity="warning", dedup_key="K1", ttl_sec=100)
        r2 = dn.notify("t", "s", "b", severity="warning", dedup_key="K1", ttl_sec=100)
    assert r1 is True
    assert r2 is False
    assert mock_run.call_count == 1


def test_critical_ignores_dedup(tmp_path, monkeypatch):
    """severity=critical は dedup 無視で必ず送信。"""
    monkeypatch.setattr(dn, "STATE_PATH", tmp_path / "state.json")
    with patch("tools.discord_notifier.subprocess.run") as mock_run:
        r1 = dn.notify("t", "s", "b", severity="critical", dedup_key="K2")
        r2 = dn.notify("t", "s", "b", severity="critical", dedup_key="K2")
    assert r1 is True
    assert r2 is True
    assert mock_run.call_count == 2


def test_no_dedup_key_always_sends(tmp_path, monkeypatch):
    """dedup_key なしなら毎回送信。"""
    monkeypatch.setattr(dn, "STATE_PATH", tmp_path / "state.json")
    with patch("tools.discord_notifier.subprocess.run") as mock_run:
        r1 = dn.notify("t", "s", "b", severity="info")
        r2 = dn.notify("t", "s", "b", severity="info")
    assert r1 is True
    assert r2 is True
    assert mock_run.call_count == 2


def test_dedup_expires_after_ttl(tmp_path, monkeypatch):
    """TTL 経過後は再送信可能。"""
    monkeypatch.setattr(dn, "STATE_PATH", tmp_path / "state.json")
    # 手動で古い timestamp を記録
    old = (datetime.datetime.now() - datetime.timedelta(hours=1)).isoformat()
    (tmp_path / "state.json").write_text(
        json.dumps({"K3": old}), encoding="utf-8")
    with patch("tools.discord_notifier.subprocess.run") as mock_run:
        r = dn.notify("t", "s", "b", severity="warning", dedup_key="K3", ttl_sec=60)
    assert r is True
    assert mock_run.call_count == 1
