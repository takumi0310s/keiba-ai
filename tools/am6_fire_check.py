"""AM6:00 DailyJrdbKyi の発火確認 + Discord 通知.

Usage:
    python tools/am6_fire_check.py
    python tools/am6_fire_check.py --silent
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from tools.fire_check_common import FireCheckConfig, check_fire, save_result, notify_result  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--silent", action="store_true")
    args = p.parse_args()

    today = datetime.date.today() if not args.date else datetime.datetime.strptime(args.date, "%Y%m%d").date()
    ymd = today.strftime("%Y%m%d")

    cfg = FireCheckConfig(
        task_name="DailyJrdbKyi",
        log_candidates=[BASE / f"logs/jrdb_kyi_auto_{ymd}.log"],
        expected_time=datetime.datetime.combine(today, datetime.time(6, 0)),
        min_size=500,
        error_keywords=["Traceback", "Exception", "ERROR"],
        recovery_command=f"python tools/daily_jrdb_kyi.bat",
    )
    r = check_fire(cfg)
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    save_result(cfg.task_name, r)
    if not args.silent:
        notify_result(cfg.task_name, r)
    return 0 if r["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
