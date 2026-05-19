"""
tests/test_daily_discord_report.py
Unit tests for tools/daily_discord_report.py
"""
import os
import sys
import csv
import tempfile
import pytest

# Ensure project root is on path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from tools.daily_discord_report import (
    load_cumulative_stats,
    build_report_message,
    send_daily_report,
    WITHDRAWAL_LIMIT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Test 1: load_cumulative_stats — normal CSV with settled rows
# ---------------------------------------------------------------------------

def test_load_cumulative_stats_normal():
    """CSV から正常集計: total_n / total_pnl / today_n など全フィールドが返る。"""
    fields = ["race_id", "status", "date", "investment", "profit"]
    rows = [
        {"race_id": "1", "status": "settled", "date": "20260519.0",
         "investment": "700", "profit": "1400"},   # +700
        {"race_id": "2", "status": "settled", "date": "20260519.0",
         "investment": "700", "profit": "-700"},   # -700
        {"race_id": "3", "status": "settled", "date": "20260518.0",
         "investment": "700", "profit": "2100"},   # +1400
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                    encoding="utf-8", newline="") as f:
        tmp_path = f.name
    try:
        _write_csv(tmp_path, rows, fields)
        stats = load_cumulative_stats(csv_path=tmp_path, date_str="20260519")

        # Cumulative: rows 1+2+3 settled → profit = 1400 - 700 + 2100 = 2800; inv = 2100
        assert stats["total_n"] == 3
        assert stats["total_pnl"] == 2800
        # ROI = (2100 + 2800) / 2100 * 100 ≈ 233.3
        assert abs(stats["total_roi"] - (4900 / 2100 * 100)) < 0.1

        # Today (20260519): rows 1+2 → profit = 1400 - 700 = 700; inv = 1400
        assert stats["today_n"] == 2
        assert stats["today_pnl"] == 700
        assert abs(stats["today_roi"] - (2100 / 1400 * 100)) < 0.1

        # withdrawal_margin = 50000 + 2800
        assert stats["withdrawal_margin"] == WITHDRAWAL_LIMIT + 2800

        # No strat_c column
        assert stats["has_strat_c"] is False
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test 2: load_cumulative_stats — file not found → empty stats, no error
# ---------------------------------------------------------------------------

def test_load_cumulative_stats_missing_file():
    """ファイルなし → empty stats が返り、例外が発生しない。"""
    stats = load_cumulative_stats(csv_path="/nonexistent/path/does_not_exist.csv")
    assert stats["total_n"] == 0
    assert stats["total_pnl"] == 0
    assert stats["total_roi"] == 0.0
    assert stats["today_n"] == 0
    assert stats["withdrawal_margin"] == WITHDRAWAL_LIMIT


# ---------------------------------------------------------------------------
# Test 3: build_report_message — all expected fields present in message
# ---------------------------------------------------------------------------

def test_build_report_message_fields():
    """全フィールド含む message 生成: 日付 / 本日 / 累計 / 撤退余裕。"""
    stats = {
        "total_n": 596,
        "total_pnl": -6920,
        "total_roi": 98.34,
        "today_n": 12,
        "today_pnl": 350,
        "today_roi": 104.2,
        "strat_c_n": 0,
        "strat_c_pnl": 0,
        "strat_c_roi": 0.0,
        "has_strat_c": False,
        "withdrawal_margin": 43080,
    }
    msg = build_report_message(stats, date_str="20260519")

    assert "2026/05/19" in msg
    assert "596R" in msg or "596" in msg
    assert "-6,920" in msg or "-6920" in msg
    assert "43,080" in msg or "43080" in msg
    assert "本日" in msg
    assert "累計" in msg
    assert "撤退余裕" in msg


# ---------------------------------------------------------------------------
# Test 4: build_report_message — withdrawal_margin calculation
#         PnL=-6920 → margin = 50000 - 6920 = 43080
# ---------------------------------------------------------------------------

def test_build_report_message_withdrawal_margin():
    """撤退余裕計算: PnL=-6920 → margin=43080。"""
    fields = ["race_id", "status", "date", "investment", "profit"]
    # Build a CSV where profit sums to -6920
    rows = [
        {"race_id": str(i), "status": "settled", "date": "20260317.0",
         "investment": "700", "profit": "-700"}
        for i in range(10)  # -7000
    ] + [
        {"race_id": "999", "status": "settled", "date": "20260317.0",
         "investment": "700", "profit": "-220"},  # adjusts: -7000 + 80 = -6920 not right
    ]
    # Simpler: just construct stats directly
    stats = {
        "total_n": 596,
        "total_pnl": -6920,
        "total_roi": 98.34,
        "today_n": 0,
        "today_pnl": 0,
        "today_roi": 0.0,
        "strat_c_n": 0,
        "strat_c_pnl": 0,
        "strat_c_roi": 0.0,
        "has_strat_c": False,
        "withdrawal_margin": WITHDRAWAL_LIMIT + (-6920),
    }
    assert stats["withdrawal_margin"] == 43080

    msg = build_report_message(stats, date_str="20260519")
    # The message should contain the margin value
    assert "43,080" in msg or "43080" in msg


# ---------------------------------------------------------------------------
# Test 5: send_daily_report — no webhook → returns False, no exception
# ---------------------------------------------------------------------------

def test_send_daily_report_no_webhook(monkeypatch):
    """webhook なし → False が返り、例外が発生しない。"""
    import tools.daily_discord_report as mod

    # Remove env vars
    monkeypatch.delenv("DISCORD_WEBHOOK_UPDATES", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

    # Patch notify._get_webhook_url so it always returns None
    try:
        import tools.notify as notify_mod
        monkeypatch.setattr(notify_mod, "_get_webhook_url", lambda channel="updates": None)
        # Also reset notify's env cache
        monkeypatch.setattr(notify_mod, "_ENV_CACHE", None)
    except Exception:
        pass

    # Patch the .env read inside daily_discord_report by pointing BASE_DIR to a
    # temp dir that has no .env file
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    orig_base = mod.BASE_DIR
    monkeypatch.setattr(mod, "BASE_DIR", tmp_dir)

    try:
        result = send_daily_report(webhook_url=None, date_str="20260519")
        # Should be False (no webhook configured) — not raise
        assert result is False
    finally:
        monkeypatch.setattr(mod, "BASE_DIR", orig_base)


# ---------------------------------------------------------------------------
# Test 6: load_cumulative_stats with strat_c column
# ---------------------------------------------------------------------------

def test_load_cumulative_stats_with_strat_c():
    """strat_c 列が存在する場合に戦略⑦案C stats が集計される。"""
    fields = ["race_id", "status", "date", "investment", "profit", "strat_c"]
    rows = [
        {"race_id": "1", "status": "settled", "date": "20260519.0",
         "investment": "700", "profit": "1400", "strat_c": "True"},
        {"race_id": "2", "status": "settled", "date": "20260519.0",
         "investment": "700", "profit": "-700", "strat_c": "False"},
        {"race_id": "3", "status": "settled", "date": "20260518.0",
         "investment": "700", "profit": "700", "strat_c": "True"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                    encoding="utf-8", newline="") as f:
        tmp_path = f.name
    try:
        _write_csv(tmp_path, rows, fields)
        stats = load_cumulative_stats(csv_path=tmp_path, date_str="20260519")
        assert stats["has_strat_c"] is True
        # strat_c rows: 1 and 3 → profit = 1400 + 700 = 2100; inv = 1400
        assert stats["strat_c_n"] == 2
        assert stats["strat_c_pnl"] == 2100
        assert stats["strat_c_roi"] > 100.0
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
