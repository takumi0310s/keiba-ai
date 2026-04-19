"""明日 (2026-04-20 Mon) 自動発火タスク群の SCRAPER-GUARD 挙動検証.

NOTE: 元の指示では「2026-04-20 日曜」だったが、実際は月曜 (Mon, weekday=0)。
本日 4/19 が日曜、明日 4/20 は月曜である。
従って明日発火する週末タスク (RaceAutoNotify_Sun, JrdbHealthCheck_Sun, DailyResults_Sun)
は次回 4/26 (Sun) に持ち越し。明日実発火するのは:
    - 03:00 DailyPremiumScrape
    - 06:00 DailyJrdbKyi
    - 08:00 DailyPredict
    - 08:00 WeeklyReport

の 4 タスク。本スクリプトは各時刻の SCRAPER-GUARD 挙動を実機シミュレーションする。

Usage:
    python tools/verify_scraper_guard_sunday.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from unittest.mock import patch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from tools.scraper_guard import check_scraping_allowed, is_scraping_allowed  # noqa: E402


def _eval(time_str: str, caller: str | None) -> str:
    """指定時刻・caller で check_scraping_allowed(mode='exit') を実行し ALLOW/STOP を返す."""
    mock_dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    # is_scraping_allowed(now=...) を直接モックする方が安全
    real_is_allowed = is_scraping_allowed
    def fake_is_allowed(now=None, caller=None):
        return real_is_allowed(now=mock_dt, caller=caller)
    with patch("tools.scraper_guard.is_scraping_allowed", side_effect=fake_is_allowed):
        try:
            check_scraping_allowed(caller=caller, mode="exit")
            return "ALLOW"
        except SystemExit:
            return "STOP"


def main() -> int:
    test_cases = [
        # === 明日 (Mon 4/20) 実発火タスク ===
        ("2026-04-20 03:00:00", "daily_premium_scrape", "ALLOW", "AM3:00 DailyPremiumScrape (Mon)"),
        ("2026-04-20 06:00:00", "daily_jrdb_kyi",        "ALLOW", "AM6:00 DailyJrdbKyi (Mon, boundary)"),
        ("2026-04-20 08:00:00", "daily_predict",         "ALLOW", "AM8:00 DailyPredict (Mon)"),
        ("2026-04-20 08:00:00", None,                    "ALLOW", "AM8:00 WeeklyReport (no caller, but past guard)"),
        # === 次の土日 (Sat 4/25 / Sun 4/26) 週末タスクの先行検証 ===
        ("2026-04-25 03:00:00", "daily_premium_scrape", "ALLOW", "AM3:00 DailyPremiumScrape (Sat early slot)"),
        ("2026-04-25 07:30:00", "jrdb_health_check",    "ALLOW", "AM7:30 JrdbHealthCheck_Sat"),
        ("2026-04-25 08:00:00", "daily_predict",        "ALLOW", "AM8:00 DailyPredict (Sat)"),
        ("2026-04-25 08:45:00", "race_auto_notify",     "ALLOW", "AM8:45 RaceAutoNotify_Sat"),
        ("2026-04-25 18:00:00", "daily_results",        "ALLOW", "PM18:00 DailyResults_Sat (no caller in script)"),
        ("2026-04-26 03:00:00", "daily_premium_scrape", "ALLOW", "AM3:00 DailyPremiumScrape (Sun early slot)"),
        ("2026-04-26 08:45:00", "race_auto_notify",     "ALLOW", "AM8:45 RaceAutoNotify_Sun"),
        # === 回帰: 引数なし / 無関係 caller は従来通り停止 ===
        ("2026-04-20 03:00:00", None,                    "STOP",  "Mon 03:00 (no caller, default)"),
        ("2026-04-20 03:00:00", "bulk_scrape_upset",     "STOP",  "Mon 03:00 (non-operational caller)"),
        ("2026-04-20 03:00:00", "scrape_master_index",   "STOP",  "Mon 03:00 (non-operational caller)"),
        ("2026-04-25 08:00:00", None,                    "STOP",  "Sat 08:00 (no caller)"),
        ("2026-04-26 08:00:00", "bulk_scrape_upset",     "STOP",  "Sun 08:00 (non-operational caller)"),
    ]

    results = []
    for time_str, caller, expected, desc in test_cases:
        actual = _eval(time_str, caller)
        status = "OK" if actual == expected else "NG"
        results.append((status, time_str, caller or "(none)", expected, actual, desc))
        print(f"[{status}] {time_str} caller={(caller or '(none)'):<24s} expect={expected:<5s} got={actual:<5s}  {desc}")

    ng = [r for r in results if r[0] == "NG"]
    print()
    if ng:
        print(f"❌ {len(ng)} NG / {len(results)} total")
        for r in ng:
            print(f"   - {r[1]} caller={r[2]} expect={r[3]} got={r[4]}  ({r[5]})")
        return 1
    print(f"✅ ALL PASS {len(results)}/{len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
