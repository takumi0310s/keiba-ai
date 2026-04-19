"""tools/morning_dashboard.py のテスト."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from tools import morning_dashboard as md  # noqa: E402


def test_format_empty():
    body = md.format_dashboard("20260420", {}, {})
    assert "Morning Dashboard - 2026/04/20" in body
    assert "未実行" in body
    assert "未実行" in body


def test_format_with_results():
    results = {
        "DailyPremiumScrape": {"status": "ok", "message": "正常発火", "size": 15000},
        "DailyJrdbKyi":       {"status": "ok", "message": "正常発火", "size": 3000},
        "DailyPredict":       {"status": "warning", "message": "SCRAPER-GUARD 検出"},
    }
    pre_fire = {"overall": "ok", "checks": [
        {"name": "SCRAPER-GUARD", "severity": "ok", "msg": "ALLOW"},
    ]}
    body = md.format_dashboard("20260420", results, pre_fire)
    assert "[OK]" in body
    assert "[WARN]" in body
    assert "DailyPremiumScrape" in body
    assert "CRITICAL: 0" in body
    assert "WARNING: 1" in body


def test_format_critical():
    results = {"DailyPremiumScrape": {"status": "critical", "message": "ログ未検出"}}
    body = md.format_dashboard("20260420", results, {})
    assert "[NG]" in body
    assert "要確認" in body
    assert "CRITICAL: 1" in body


def test_load_results_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(md, "BASE", tmp_path)
    assert md.load_results("20260420") == {}
    assert md.load_pre_fire("20260420") == {}


def test_load_results_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(md, "BASE", tmp_path)
    d = tmp_path / "data" / "fire_check_results"
    d.mkdir(parents=True)
    (d / "20260420.json").write_text(
        json.dumps({"DailyPredict": {"status": "ok"}}), encoding="utf-8")
    r = md.load_results("20260420")
    assert r["DailyPredict"]["status"] == "ok"
