"""tools/am3_fire_check.py の軽量テスト."""
from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from tools import am3_fire_check  # noqa: E402


def test_missing_log_returns_critical(tmp_path, monkeypatch):
    """ログが存在しない日付を指定 → critical."""
    monkeypatch.setattr(am3_fire_check, "BASE", tmp_path)
    (tmp_path / "logs").mkdir()
    result = am3_fire_check.check_am3_fire(target_date="20260420")
    assert result["status"] == "critical"
    assert "ログファイル未検出" in result["message"]
    assert "recovery" in result


def test_small_log_returns_critical(tmp_path, monkeypatch):
    """ログが 2000B 未満 → critical."""
    monkeypatch.setattr(am3_fire_check, "BASE", tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    log_path = logs / "premium_scrape_20260420.log"
    log_path.write_text("tiny log\n" * 10, encoding="utf-8")

    # mtime を 03:05 に設定 (発火後)
    t = datetime.datetime(2026, 4, 20, 3, 5).timestamp()
    os.utime(log_path, (t, t))

    result = am3_fire_check.check_am3_fire(target_date="20260420")
    assert result["status"] == "critical"
    assert "サイズ異常" in result["message"]


def test_large_log_with_guard_keyword_returns_warning(tmp_path, monkeypatch):
    """ログが大きいが SCRAPER-GUARD を含む → warning."""
    monkeypatch.setattr(am3_fire_check, "BASE", tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    log_path = logs / "premium_scrape_20260420.log"
    content = "x" * 3000 + "\n[SCRAPER-GUARD] 停止\n"
    log_path.write_text(content, encoding="utf-8")
    t = datetime.datetime(2026, 4, 20, 3, 10).timestamp()
    os.utime(log_path, (t, t))

    result = am3_fire_check.check_am3_fire(target_date="20260420")
    assert result["status"] == "warning"
    assert "SCRAPER-GUARD" in result["keyword"]


def test_healthy_large_log_returns_ok(tmp_path, monkeypatch):
    """ログが 2000B+ かつエラーなし → ok."""
    monkeypatch.setattr(am3_fire_check, "BASE", tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    log_path = logs / "premium_scrape_20260420.log"
    log_path.write_text("normal progress " * 200, encoding="utf-8")
    t = datetime.datetime(2026, 4, 20, 3, 10).timestamp()
    os.utime(log_path, (t, t))

    result = am3_fire_check.check_am3_fire(target_date="20260420")
    assert result["status"] == "ok"
    assert result["size"] >= 2000
