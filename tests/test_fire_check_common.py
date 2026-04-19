"""tools/fire_check_common.py のテスト."""
from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from tools.fire_check_common import FireCheckConfig, check_fire  # noqa: E402


def _make_cfg(tmp_path, task_name="Test", min_size=100, csv_rows=0):
    log = tmp_path / "logs" / f"{task_name}.log"
    csv_p = tmp_path / "data" / f"{task_name}.csv"
    return FireCheckConfig(
        task_name=task_name,
        log_candidates=[log],
        expected_time=datetime.datetime(2026, 4, 20, 8, 0),
        min_size=min_size,
        recovery_command="python tools/test.py",
        csv_candidates=[csv_p],
        min_csv_rows=csv_rows,
    )


def test_missing_log_critical(tmp_path):
    cfg = _make_cfg(tmp_path)
    r = check_fire(cfg)
    assert r["status"] == "critical"
    assert "未検出" in r["message"]


def test_small_log_critical(tmp_path):
    cfg = _make_cfg(tmp_path, min_size=500)
    (tmp_path / "logs").mkdir()
    log = cfg.log_candidates[0]
    log.write_text("small\n" * 5, encoding="utf-8")
    t = datetime.datetime(2026, 4, 20, 8, 5).timestamp()
    os.utime(log, (t, t))
    r = check_fire(cfg)
    assert r["status"] == "critical"
    assert "サイズ異常" in r["message"]


def test_ok_log(tmp_path):
    cfg = _make_cfg(tmp_path, min_size=100)
    (tmp_path / "logs").mkdir()
    log = cfg.log_candidates[0]
    log.write_text("healthy progress " * 30, encoding="utf-8")
    t = datetime.datetime(2026, 4, 20, 8, 5).timestamp()
    os.utime(log, (t, t))
    r = check_fire(cfg)
    assert r["status"] == "ok"


def test_keyword_warning(tmp_path):
    cfg = _make_cfg(tmp_path, min_size=100)
    (tmp_path / "logs").mkdir()
    log = cfg.log_candidates[0]
    log.write_text("x" * 1000 + "\nTraceback: ...\n", encoding="utf-8")
    t = datetime.datetime(2026, 4, 20, 8, 5).timestamp()
    os.utime(log, (t, t))
    r = check_fire(cfg)
    assert r["status"] == "warning"
    assert r["keyword"] == "Traceback"


def test_csv_fallback_ok(tmp_path):
    """ログ無くても CSV 充足なら ok."""
    cfg = _make_cfg(tmp_path, min_size=100, csv_rows=5)
    (tmp_path / "data").mkdir()
    csv_p = cfg.csv_candidates[0]
    csv_p.write_text("h1,h2\n" + "a,b\n" * 10, encoding="utf-8")
    r = check_fire(cfg)
    assert r["status"] == "ok"
    assert r["rows"] >= 5


def test_stale_log_critical(tmp_path):
    """expected_time より古い mtime → critical."""
    cfg = _make_cfg(tmp_path, min_size=100)
    (tmp_path / "logs").mkdir()
    log = cfg.log_candidates[0]
    log.write_text("x" * 500, encoding="utf-8")
    # mtime を 08:00 より前にする (07:00)
    t = datetime.datetime(2026, 4, 20, 7, 0).timestamp()
    os.utime(log, (t, t))
    r = check_fire(cfg)
    assert r["status"] == "critical"
    assert "未更新" in r["message"]
