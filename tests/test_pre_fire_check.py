"""tools/pre_fire_check.py の軽量テスト."""
from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from tools import pre_fire_check as pfc  # noqa: E402


def test_scraper_guard_mon_am3_allowed():
    """Mon 03:00 で daily_premium_scrape caller なら ALLOW。"""
    with patch("tools.pre_fire_check.datetime") as mdt:
        mdt.datetime.now.return_value = datetime.datetime(2026, 4, 20, 2, 55)
        mdt.timedelta = datetime.timedelta
        mdt.datetime.strptime = datetime.datetime.strptime
        # 実際の check 実行
        r = pfc.check_scraper_guard()
    # Mon の premium 特例で allowed
    assert r["ok"] is True
    assert r["severity"] == "ok"


def test_check_cookie_missing_env(tmp_path, monkeypatch):
    """.env が欠落 → critical."""
    monkeypatch.setattr(pfc, "BASE", tmp_path)
    r = pfc.check_cookie()
    assert r["ok"] is False
    assert r["severity"] == "critical"


def test_check_cookie_valid(tmp_path, monkeypatch):
    """.env に NETKEIBA_COOKIE= が十分に長ければ ok。"""
    monkeypatch.setattr(pfc, "BASE", tmp_path)
    env = tmp_path / ".env"
    long_val = "a" * 100
    env.write_text(f"NETKEIBA_COOKIE={long_val}\n", encoding="utf-8")
    r = pfc.check_cookie()
    assert r["ok"] is True


def test_check_dirs_missing(tmp_path, monkeypatch):
    """必要ディレクトリが全て存在しなければ warning。"""
    monkeypatch.setattr(pfc, "BASE", tmp_path)
    r = pfc.check_dirs()
    assert r["ok"] is False
    assert r["severity"] == "warning"


def test_check_dirs_ok(tmp_path, monkeypatch):
    """全ディレクトリ作成後は ok。"""
    monkeypatch.setattr(pfc, "BASE", tmp_path)
    for d in ["data", "logs", "data/daily_predictions", "data/weekly_premium_cache"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    r = pfc.check_dirs()
    assert r["ok"] is True


def test_run_all_checks_structure():
    """run_all_checks が 6 チェック返す."""
    checks, overall = pfc.run_all_checks()
    assert len(checks) == 6
    assert overall in ("ok", "warning", "critical")
