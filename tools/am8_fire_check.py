"""AM8:00 DailyPredict の発火確認 + Discord 通知.

DailyPredict は予測処理が 30-45 分かかるため grace period を広くとる。
ログが長大なので CSV (data/daily_predictions/{YYYYMMDD}.csv) の行数でも判定。

Usage:
    python tools/am8_fire_check.py
    python tools/am8_fire_check.py --silent
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

    # 非開催日 (Mon-Fri) なら CSV ゼロでも OK なので min_csv_rows=0、
    # 週末 (Sat/Sun) なら 30 行以上期待
    is_weekend = today.weekday() >= 5
    min_rows = 30 if is_weekend else 0

    cfg = FireCheckConfig(
        task_name="DailyPredict",
        log_candidates=[
            BASE / f"logs/daily_predict_{ymd}.log",
            BASE / "logs/daily_predict.log",
        ],
        expected_time=datetime.datetime.combine(today, datetime.time(8, 0)),
        min_size=2000,
        error_keywords=["Traceback", "Exception", "CRITICAL"],
        recovery_command=f"SCRAPER_GUARD_DISABLE=1 python tools/daily_predict.py --date {ymd}",
        csv_candidates=[BASE / f"data/daily_predictions/{ymd}.csv"],
        min_csv_rows=min_rows,
    )
    r = check_fire(cfg)
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    save_result(cfg.task_name, r)
    if not args.silent:
        notify_result(cfg.task_name, r)
    return 0 if r["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
