#!/usr/bin/env python
"""netkeiba 回復監視 + race_auto_notify 自動再起動

netkeiba が 400 から回復したら schtasks で RaceAutoNotify_Sun を再起動する。

Usage:
    python tools/netkeiba_watchdog.py          # 5分間隔で監視
    python tools/netkeiba_watchdog.py --interval 60  # 60秒間隔
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime

import requests

BASE_URL = "https://race.netkeiba.com/race/shutuba.html"
TEST_RACE_ID = "202605021006"  # 東京6R (今日)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
TASK_NAME = r"\keiba-ai\RaceAutoNotify_Sun"
RACE_END_HOUR = 17  # 17:00以降は監視不要


def check_netkeiba() -> int:
    try:
        r = requests.get(f"{BASE_URL}?race_id={TEST_RACE_ID}",
                         headers=HEADERS, timeout=10)
        return r.status_code
    except Exception:
        return -1


def restart_task() -> bool:
    result = subprocess.run(
        ["schtasks", "/run", "/TN", TASK_NAME],
        capture_output=True, text=True
    )
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300, help="チェック間隔(秒) default=300")
    args = ap.parse_args()

    print(f"[watchdog] netkeiba 回復監視開始 (interval={args.interval}s)")
    print(f"[watchdog] 回復確認後 → {TASK_NAME} 再起動")

    while True:
        now = datetime.now()
        if now.hour >= RACE_END_HOUR:
            print(f"[watchdog] {now.strftime('%H:%M')} レース終了時間帯 → 監視終了")
            return 0

        status = check_netkeiba()
        ts = now.strftime("%H:%M:%S")

        if status == 200:
            print(f"[{ts}] netkeiba 回復! (200) → race_auto_notify 再起動")
            ok = restart_task()
            if ok:
                print(f"[{ts}] 再起動成功")
                return 0
            else:
                print(f"[{ts}] 再起動失敗 (権限不足?) → 手動実行: schtasks /run /TN \"{TASK_NAME}\"")
                return 1
        elif status == 400:
            print(f"[{ts}] netkeiba 400 (レート制限中) → {args.interval}秒後に再確認")
        else:
            print(f"[{ts}] netkeiba 応答異常 (status={status}) → {args.interval}秒後に再確認")

        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
