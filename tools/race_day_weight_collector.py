"""当日馬体重 collector (Session #48 B、 dev/two-stage).

各 R 70 分前 trigger で 当日馬体重を取得。
- 主軸: JV-Link WF datatype (5/16+ 32-bit Python venv)
- fallback: netkeiba 出馬表 polling (各 R 70 分前)
- retry 3 回、 失敗時 Discord alert

usage:
  python tools/race_day_weight_collector.py --race-id 202605020412
  python tools/race_day_weight_collector.py --date 20260509 --course 東京

V15 production 完全独立、 dev/two-stage 専用。
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")


def fetch_via_jvlink(race_id: str) -> dict:
    """JV-Link WF (馬体重情報、 32-bit COM、 5/16+) から取得.

    本 Session では design のみ。
    """
    return {
        "source": "JV-Link WF",
        "status": "deferred",
        "design": {
            "tool": "tools/jvlink_fetcher_v2.py --datatype WF --date {date}",
            "trigger": "32-bit Python venv (5/16+)、 各 R 65 分前 schtasks",
            "fields": ["umaban", "horse_weight_current", "horse_weight_change"],
        },
    }


def fetch_via_netkeiba(race_id: str) -> dict:
    """netkeiba 出馬表 polling (各 R 70 分前 update) から馬体重 取得.

    本 Session では design + skeleton のみ。 実 polling は 5/16+ schtasks。
    """
    return {
        "source": "netkeiba scraping",
        "status": "deferred",
        "design": {
            "url_pattern": "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}",
            "selector": "table tr.HorseList td 馬体重列",
            "polling_interval": "5 分毎、 各 R 70 分前 から開始",
            "stop_condition": "馬体重 全頭取得 or レース 30 分前",
        },
    }


def collect_weights_with_retry(race_id: str, max_retries: int = 3) -> dict:
    """retry 付き 馬体重 取得 (本 Session では design)."""
    attempts = []

    for attempt in range(max_retries):
        # primary: JV-Link
        r = fetch_via_jvlink(race_id)
        attempts.append({"attempt": attempt + 1, "result": r})
        if r.get("status") == "ok":
            return {"status": "ok", "weights": r.get("data", {}), "attempts": attempts}

        # fallback: netkeiba
        r2 = fetch_via_netkeiba(race_id)
        attempts.append({"attempt": attempt + 1, "fallback": r2})
        if r2.get("status") == "ok":
            return {"status": "ok", "weights": r2.get("data", {}), "attempts": attempts}

    return {
        "status": "deferred",
        "race_id": race_id,
        "attempts": attempts,
        "design": "5/16+ schtasks で 各 R 65 分前 trigger、 retry 3 + Discord alert",
    }


def main():
    p = argparse.ArgumentParser(description="race_day_weight_collector (Session #48 B)")
    p.add_argument("--race-id", default=None)
    p.add_argument("--date", default=None)
    p.add_argument("--course", default=None)
    p.add_argument("--out", default="data/v18/session_48_weight_collector_test.json")
    args = p.parse_args()

    print("=" * 60)
    print("race_day_weight_collector (Session #48 B、 dev/two-stage)")
    print("=" * 60)

    if args.race_id:
        result = collect_weights_with_retry(args.race_id)
        print(f"\nrace_id: {args.race_id}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"\n[design only] date={args.date}, course={args.course}")
        print("実 collection は 5/16+ schtasks で 各 R 65 分前 trigger")

    out_path = BASE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "race_id": args.race_id,
        "date": args.date,
        "course": args.course,
        "status": "design",
        "production_trigger": "5/16+ schtasks 各 R 65 分前",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
