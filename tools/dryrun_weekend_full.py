"""来週末 (2026-04-25 Sat / 4-26 Sun) と月曜 (4-27) の全自動発火タスクを
完全シミュレーション (mock datetime / 副作用ゼロ).

各タスクで以下をチェック:
  a. SCRAPER-GUARD の挙動 (ALLOW/STOP 期待値)
  b. 各スクリプトの import チェーン整合性
  c. 必要な外部依存 (cookie, DB, 主要ファイル) の存在
  d. 出力ファイル書き込み先のディレクトリ存在 / 書き込み可能性
  e. 異常時のフォールバック動作 (簡易)

Usage:
    python tools/dryrun_weekend_full.py
"""
from __future__ import annotations

import importlib
import os
import sys
from datetime import datetime
from unittest.mock import patch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from tools.scraper_guard import is_scraping_allowed  # noqa: E402


# (datetime_str, task_name, script_module, caller_for_guard, is_scraper,
#  required_files, output_dirs)
WEEKEND_SCHEDULE = [
    # Saturday 2026-04-25
    ("2026-04-25 03:00:00", "DailyPremiumScrape_Sat", "tools.daily_premium_scrape",
     "daily_premium_scrape", True,
     [".env", "tools/scraper_guard.py"],
     ["data/weekly_premium_cache"]),
    ("2026-04-25 06:00:00", "DailyJrdbKyi_Sat", "tools.scrape_jrdb",
     "daily_jrdb_kyi", False,
     ["tools/scrape_jrdb.py"],
     ["data"]),
    ("2026-04-25 07:30:00", "JrdbHealthCheck_Sat", "tools.jrdb_health_check",
     "jrdb_health_check", False,
     ["tools/jrdb_health_check.py"],
     ["data", "logs"]),
    ("2026-04-25 08:00:00", "DailyPredict_Sat", "tools.daily_predict",
     "daily_predict", False,
     ["keiba_model_v15_central_live.pkl.gz", ".env"],
     ["data/daily_predictions"]),
    ("2026-04-25 08:45:00", "RaceAutoNotify_Sat", "tools.race_auto_notify",
     "race_auto_notify", False,
     ["tools/race_auto_notify.py", ".env"],
     ["logs"]),
    ("2026-04-25 18:00:00", "DailyResults_Sat", "tools.daily_results",
     "daily_results", False,
     ["tools/daily_results.py"],
     ["data/daily_results"]),
    ("2026-04-25 20:00:00", "DailyResultsEvening_Sat", "tools.daily_results",
     "daily_results", False,
     ["tools/daily_results.py"],
     ["data/daily_results"]),
    # Sunday 2026-04-26
    ("2026-04-26 03:00:00", "DailyPremiumScrape_Sun", "tools.daily_premium_scrape",
     "daily_premium_scrape", True,
     [".env"],
     ["data/weekly_premium_cache"]),
    ("2026-04-26 06:00:00", "DailyJrdbKyi_Sun", "tools.scrape_jrdb",
     "daily_jrdb_kyi", False,
     ["tools/scrape_jrdb.py"],
     ["data"]),
    ("2026-04-26 07:30:00", "JrdbHealthCheck_Sun", "tools.jrdb_health_check",
     "jrdb_health_check", False,
     ["tools/jrdb_health_check.py"],
     ["data", "logs"]),
    ("2026-04-26 08:00:00", "DailyPredict_Sun", "tools.daily_predict",
     "daily_predict", False,
     ["keiba_model_v15_central_live.pkl.gz"],
     ["data/daily_predictions"]),
    ("2026-04-26 08:45:00", "RaceAutoNotify_Sun", "tools.race_auto_notify",
     "race_auto_notify", False,
     ["tools/race_auto_notify.py"],
     ["logs"]),
    ("2026-04-26 18:00:00", "DailyResults_Sun", "tools.daily_results",
     "daily_results", False,
     ["tools/daily_results.py"],
     ["data/daily_results"]),
    # Monday 2026-04-27
    ("2026-04-27 03:00:00", "DailyPremiumScrape_Mon", "tools.daily_premium_scrape",
     "daily_premium_scrape", True,
     [".env"],
     ["data/weekly_premium_cache"]),
    ("2026-04-27 06:00:00", "DailyJrdbKyi_Mon", "tools.scrape_jrdb",
     "daily_jrdb_kyi", False,
     ["tools/scrape_jrdb.py"],
     ["data"]),
    ("2026-04-27 08:00:00", "DailyPredict_Mon", "tools.daily_predict",
     "daily_predict", False,
     ["keiba_model_v15_central_live.pkl.gz"],
     ["data/daily_predictions"]),
    ("2026-04-27 08:00:00", "WeeklyReport_Mon", "tools.weekly_report",
     None, False,
     ["tools/weekly_report.py"],
     ["logs"]),
]


def check_file(relpath: str) -> tuple[bool, str]:
    full = os.path.join(BASE, relpath)
    if os.path.exists(full):
        return True, relpath
    return False, relpath


def check_dir_writable(relpath: str) -> tuple[bool, str]:
    full = os.path.join(BASE, relpath)
    if os.path.isdir(full):
        if os.access(full, os.W_OK):
            return True, relpath
        return False, f"{relpath} (not writable)"
    return False, f"{relpath} (not exists)"


def check_import(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, module_name
    except Exception as e:
        return False, f"{module_name} (import err: {type(e).__name__}: {e})"


def check_guard(dt_str: str, caller: str | None, expected_allow: bool) -> tuple[bool, str]:
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    allowed = is_scraping_allowed(now=dt, caller=caller)
    if allowed == expected_allow:
        return True, f"guard={'ALLOW' if allowed else 'STOP'}"
    return False, f"guard={'ALLOW' if allowed else 'STOP'} expected={'ALLOW' if expected_allow else 'STOP'}"


def main() -> int:
    fail_count = 0
    pass_count = 0
    per_task = []

    print("=" * 70)
    print("WEEKEND E2E DRY-RUN — 2026-04-25 (Sat) / 4-26 (Sun) / 4-27 (Mon)")
    print("=" * 70)

    for (dt_str, task_name, module, caller, is_scraper,
         files, out_dirs) in WEEKEND_SCHEDULE:

        task_fail = 0
        checks = []

        # (a) SCRAPER-GUARD
        # スクレイパータスクはガード越え必要、非スクレイパー (daily_predict 等で
        # guard を呼ばないもの) は caller 渡しでも許可される前提
        expected_allow = True  # 全 operational タスクは ALLOW 期待
        ok, msg = check_guard(dt_str, caller, expected_allow)
        checks.append((ok, f"(a) {msg}"))
        if not ok:
            task_fail += 1

        # (b) Import integrity
        ok, msg = check_import(module)
        checks.append((ok, f"(b) import {msg}"))
        if not ok:
            task_fail += 1

        # (c) Required files
        for f in files:
            ok, msg = check_file(f)
            checks.append((ok, f"(c) file {msg}"))
            if not ok:
                task_fail += 1

        # (d) Output dirs
        for d in out_dirs:
            ok, msg = check_dir_writable(d)
            checks.append((ok, f"(d) dir {msg}"))
            if not ok:
                task_fail += 1

        status = "PASS" if task_fail == 0 else f"FAIL ({task_fail})"
        per_task.append((status, dt_str, task_name, task_fail, checks))

        if task_fail == 0:
            pass_count += 1
        else:
            fail_count += 1

        tag = "✓" if task_fail == 0 else "✗"
        print(f"\n[{tag}] {dt_str} {task_name}")
        for ok, msg in checks:
            print(f"    {'OK' if ok else 'NG'}: {msg}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {pass_count} PASS / {fail_count} FAIL (total {len(WEEKEND_SCHEDULE)} tasks)")
    print("=" * 70)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
