"""レース開催時間帯のスクレイピング自動停止ガード。

金曜22:00 〜 月曜06:00 (JST) の間はnetkeiba系スクレイパーを停止する。
週末レース運用中にスクレイピング負荷をかけないための保護。

ただし、以下の「運用必須」呼び出し元は週末ガードを自動バイパスする:
    - daily_predict
    - race_auto_notify
    - notify_bets_all_in_one
    - jrdb_health_check
    - daily_jrdb_kyi
環境変数 KEIBA_OPERATIONAL_MODE=1 でも同様にバイパスされる。

daily_premium_scrape は特例: 土日月の 03:00-05:59 の早朝スロットのみ許可
（タスクスケジューラ AM3:00 daily起動が誤停止する事故を防止。
 Mon 早朝はガード窓 Mon00:00-06:00 と重なるため Mon も特例対象）。

使い方:
    from tools.scraper_guard import check_scraping_allowed
    check_scraping_allowed()  # 引数なし: 従来通り週末停止→後方互換

    # 運用タスク (許可される):
    check_scraping_allowed(caller="daily_predict")

    # 明示的exit:
    check_scraping_allowed(mode="exit")
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime


SLEEP_INTERVAL_SEC = 600  # 10分おきに再チェック

# 週末ガードをバイパスする運用必須タスク
OPERATIONAL_CALLERS = frozenset({
    "daily_predict",
    "race_auto_notify",
    "notify_bets_all_in_one",
    "jrdb_health_check",
    "daily_jrdb_kyi",
    "daily_results",
})


def _is_operational_mode() -> bool:
    return os.environ.get("KEIBA_OPERATIONAL_MODE") == "1"


def _premium_scrape_early_slot(now: datetime) -> bool:
    """daily_premium_scrape 特例: 土日月の03:00-05:59は許可。

    タスクスケジューラ AM3:00 daily起動が SCRAPER-GUARD で誤停止する事故を防ぐ。
    Sat/Sun/Mon の早朝 (00:00-05:59) はガード窓内のため特例対象。
    平日 Tue-Fri 03:00 はガード対象外なので特例不要。
    """
    return now.weekday() in (5, 6, 0) and 3 <= now.hour < 6


def is_scraping_allowed(now: datetime | None = None, caller: str | None = None) -> bool:
    """金曜22:00〜月曜06:00の間はFalseを返す。

    caller が OPERATIONAL_CALLERS に含まれる、または KEIBA_OPERATIONAL_MODE=1 なら常に True。
    caller="daily_premium_scrape" かつ 土日 03:00-05:59 は True（早朝スロット許可）。
    """
    if os.environ.get("SCRAPER_GUARD_DISABLE") == "1":
        return True
    if caller in OPERATIONAL_CALLERS:
        return True
    if _is_operational_mode():
        return True
    now = now or datetime.now()
    if caller == "daily_premium_scrape" and _premium_scrape_early_slot(now):
        return True
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


def wait_until_allowed(interval: int = SLEEP_INTERVAL_SEC, caller: str | None = None) -> None:
    """ガード時間帯なら許可されるまで sleep(interval) でブロックする。"""
    notified = False
    while not is_scraping_allowed(caller=caller):
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


def check_scraping_allowed(caller: str | None = None, mode: str = "wait", exit_code: int = 0) -> None:
    """ガード時間帯中の振る舞いを切り替える。

    Args:
        caller: 呼び出し元識別子。OPERATIONAL_CALLERS のいずれか、または
                "daily_premium_scrape"（03:00-05:59 土日許可の特例）、
                その他名称（従来通りガード対象）。
        mode: "wait"=sleepループ待機 (default)、"exit"=即終了
        exit_code: mode="exit" 時の終了コード
    """
    if is_scraping_allowed(caller=caller):
        if caller in OPERATIONAL_CALLERS or _is_operational_mode():
            reason = "operational" if caller in OPERATIONAL_CALLERS else "env_operational_mode"
            print(f"[SCRAPER-GUARD] BYPASS: caller={caller or '(none)'} reason={reason}")
        return
    if mode == "exit":
        now = datetime.now().strftime("%Y-%m-%d %H:%M %a")
        print(f"[SCRAPER-GUARD] {now} — 週末レース時間帯のためスクレイピング停止 (Fri22:00〜Mon06:00) caller={caller or '(none)'}")
        print("[SCRAPER-GUARD] SCRAPER_GUARD_DISABLE=1 を設定すると強制実行できます")
        sys.exit(exit_code)
    wait_until_allowed(caller=caller)
