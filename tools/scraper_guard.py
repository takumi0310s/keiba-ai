"""レース開催時間帯のスクレイピング自動停止ガード。

金曜22:00 〜 月曜06:00 (JST) の間はnetkeiba系スクレイパーを停止する。
週末レース運用中にスクレイピング負荷をかけないための保護。

使い方:
    from tools.scraper_guard import check_scraping_allowed
    check_scraping_allowed()  # ガード時間帯なら sleep(600) でループ待機し、
                              # 時間帯を抜けたら自動で処理続行

    # ループ内で定期チェックする場合 (推奨):
    from tools.scraper_guard import check_scraping_allowed
    for item in items:
        check_scraping_allowed()  # 走行中にガード帯入りしたらここで自動停止→再開
        process(item)

    # 明示的に exit したい場合:
    check_scraping_allowed(mode="exit")
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime


SLEEP_INTERVAL_SEC = 600  # 10分おきに再チェック


def is_scraping_allowed(now: datetime | None = None) -> bool:
    """金曜22:00〜月曜06:00の間はFalseを返す。"""
    if os.environ.get("SCRAPER_GUARD_DISABLE") == "1":
        return True
    now = now or datetime.now()
    wd = now.weekday()  # Mon=0 .. Sun=6
    h = now.hour
    if wd == 4 and h >= 22:          # Fri 22:00+
        return False
    if wd == 5:                       # Sat all day
        return False
    if wd == 6:                       # Sun all day
        return False
    if wd == 0 and h < 6:             # Mon 00:00-05:59
        return False
    return True


def wait_until_allowed(interval: int = SLEEP_INTERVAL_SEC) -> None:
    """ガード時間帯なら許可されるまで sleep(interval) でブロックする。"""
    notified = False
    while not is_scraping_allowed():
        if not notified:
            now = datetime.now().strftime("%Y-%m-%d %H:%M %a")
            print(f"[SCRAPER-GUARD] {now} — 週末レース時間帯のためスクレイピング停止 (Fri22:00〜Mon06:00)")
            print(f"[SCRAPER-GUARD] {interval}秒おきにチェックし、月曜06:00に自動再開します")
            print("[SCRAPER-GUARD] SCRAPER_GUARD_DISABLE=1 を設定すると強制実行できます")
            notified = True
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("[SCRAPER-GUARD] 中断されました")
            sys.exit(0)
    if notified:
        now = datetime.now().strftime("%Y-%m-%d %H:%M %a")
        print(f"[SCRAPER-GUARD] {now} — 時間帯を抜けたためスクレイピングを再開します")


def check_scraping_allowed(exit_code: int = 0, mode: str = "wait") -> None:
    """ガード時間帯中の振る舞いを切り替える。

    mode="wait" (デフォルト): sleep ループで待機し、時間帯を抜けたら自動再開
    mode="exit": 旧挙動。sys.exit(exit_code) で即終了
    """
    if is_scraping_allowed():
        return
    if mode == "exit":
        now = datetime.now().strftime("%Y-%m-%d %H:%M %a")
        print(f"[SCRAPER-GUARD] {now} — 週末レース時間帯のためスクレイピング停止 (Fri22:00〜Mon06:00)")
        print("[SCRAPER-GUARD] SCRAPER_GUARD_DISABLE=1 を設定すると強制実行できます")
        sys.exit(exit_code)
    wait_until_allowed()
