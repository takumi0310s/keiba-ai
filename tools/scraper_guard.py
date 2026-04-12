"""レース開催時間帯のスクレイピング自動停止ガード。

金曜22:00 〜 月曜06:00 (JST) の間はnetkeiba系スクレイパーを停止する。
週末レース運用中にスクレイピング負荷をかけないための保護。

使い方:
    from tools.scraper_guard import check_scraping_allowed
    check_scraping_allowed()  # ブロック時は sys.exit(0)

    # ループ内で継続的に確認する場合:
    if not is_scraping_allowed():
        break
"""
from __future__ import annotations

import os
import sys
from datetime import datetime


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


def check_scraping_allowed(exit_code: int = 0) -> None:
    """ブロック時間帯ならメッセージを出してsys.exitする。"""
    if is_scraping_allowed():
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M %a")
    print(f"[SCRAPER-GUARD] {now} — 週末レース時間帯のためスクレイピング停止 (Fri22:00〜Mon06:00)")
    print("[SCRAPER-GUARD] SCRAPER_GUARD_DISABLE=1 を設定すると強制実行できます")
    sys.exit(exit_code)
