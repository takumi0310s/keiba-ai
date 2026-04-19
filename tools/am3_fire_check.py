"""AM3:00 DailyPremiumScrape の発火確認 + Discord 通知.

毎日 03:15 に Keiba-AM3FireCheck から実行され、AM3:00 DailyPremiumScrape が
正常発火したかを判定し、結果を Discord に投稿する。

判定基準:
    ok: ログ 2000B 以上 + エラーキーワードなし
    warning: エラーキーワード検出 ("SCRAPER-GUARD" 停止ログ等)
    critical: ログ未検出 / サイズ異常 / 発火時刻後未更新

Usage:
    python tools/am3_fire_check.py
    python tools/am3_fire_check.py --silent   # Discord 通知スキップ
    python tools/am3_fire_check.py --date 20260420  # 特定日チェック (デバッグ用)
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")


def check_am3_fire(target_date: str | None = None, now: datetime.datetime | None = None) -> dict:
    """AM3:00 の発火状況を判定。

    Args:
        target_date: YYYYMMDD 形式。None なら today。
        now: テスト用の現在時刻モック。
    """
    if now is None:
        now = datetime.datetime.now()
    if target_date is None:
        target_date = now.strftime("%Y%m%d")

    log_candidates = [
        BASE / f"logs/premium_scrape_{target_date}.log",
        BASE / f"logs/daily_premium_scrape_{target_date}.log",
    ]
    log_file: Path | None = None
    for candidate in log_candidates:
        if candidate.exists():
            log_file = candidate
            break

    if log_file is None:
        return {
            "status": "critical",
            "message": "ログファイル未検出",
            "candidates": [str(p) for p in log_candidates],
            "recovery": "SCRAPER_GUARD_DISABLE=1 python tools/daily_premium_scrape.py",
        }

    # 期待発火時刻 (当日 03:00)
    try:
        dt = datetime.datetime.strptime(target_date, "%Y%m%d")
    except ValueError:
        return {"status": "critical", "message": f"date format invalid: {target_date}"}
    expected_fire = dt.replace(hour=3, minute=0, second=0, microsecond=0)

    stat = log_file.stat()
    size = stat.st_size
    mtime = datetime.datetime.fromtimestamp(stat.st_mtime)

    # 発火予定時刻より前かつ現時刻が3:00前ならスキップ (テスト/デバッグ用)
    # 通常の本番は 03:15 実行なので current time > expected_fire が保証される
    if mtime < expected_fire:
        return {
            "status": "critical",
            "message": f"ログ未更新 (最終更新 {mtime.isoformat()}, 期待 {expected_fire.isoformat()} 以降)",
            "size": size,
            "recovery": "SCRAPER_GUARD_DISABLE=1 python tools/daily_premium_scrape.py",
        }

    # サイズで成功判定
    if size < 2000:
        try:
            tail = log_file.read_text(encoding="utf-8", errors="replace")[-500:]
        except Exception as e:
            tail = f"(read err: {e})"
        return {
            "status": "critical",
            "message": f"ログサイズ異常 {size}B (最低 2000B 期待)",
            "size": size,
            "mtime": mtime.isoformat(),
            "log_tail": tail,
            "recovery": "SCRAPER_GUARD_DISABLE=1 python tools/daily_premium_scrape.py",
        }

    # エラーキーワードで警告判定
    try:
        tail = log_file.read_text(encoding="utf-8", errors="replace")[-3000:]
    except Exception as e:
        tail = f"(read err: {e})"

    error_keywords = ["SCRAPER-GUARD", "Traceback", "ERROR", "Exception", "IP banned"]
    for kw in error_keywords:
        if kw in tail:
            return {
                "status": "warning",
                "message": f"ログに '{kw}' を検出",
                "keyword": kw,
                "size": size,
                "mtime": mtime.isoformat(),
                "log_tail": tail[-500:],
            }

    return {
        "status": "ok",
        "message": "AM3:00 DailyPremiumScrape 正常発火",
        "size": size,
        "mtime": mtime.isoformat(),
        "log": str(log_file),
    }


def notify_discord(result: dict) -> None:
    """既存の tools/notify_done.py を経由して Discord に投稿."""
    status = result.get("status", "critical")
    if status == "ok":
        title = "AM3:00 正常発火"
        subtitle = "OK"
        body = f"ログ {result.get('size')}B / 更新 {result.get('mtime')}"
        color = "green"
    elif status == "warning":
        title = "AM3:00 警告"
        subtitle = "要確認"
        body = result.get("message", "")
        tail = result.get("log_tail", "")
        if tail:
            body += "\n\nlog_tail:\n" + tail[:400]
        color = "yellow"
    else:
        title = "CRITICAL: AM3:00 失敗"
        subtitle = "要手動介入"
        body = result.get("message", "")
        rec = result.get("recovery", "")
        if rec:
            body += "\n\nリカバリ:\n" + rec
        color = "red"

    try:
        subprocess.run(
            [sys.executable, str(BASE / "tools/notify_done.py"),
             title, subtitle, body, "--color", color],
            check=False,
            timeout=30,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except Exception as e:
        print(f"[WARN] Discord 通知失敗: {e}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=None, help="対象日 YYYYMMDD (default: today)")
    p.add_argument("--silent", action="store_true", help="Discord 通知しない")
    args = p.parse_args()

    result = check_am3_fire(target_date=args.date)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    if not args.silent:
        notify_discord(result)

    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
