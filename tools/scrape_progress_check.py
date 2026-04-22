"""scrape_missing 進捗確認 (軽量、毎朝AM7:00)

logs/scrape_missing_*.log の最終行と更新時刻を確認し、
- 60分以上更新なし → STALL警告
- 進捗あり → 通常通知
- ログ無し → スキップ

Discord channel: updates
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import sys
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
LOG_DIR = BASE / "logs"

sys.path.insert(0, str(BASE))


def latest_scrape_log() -> Path | None:
    files = sorted(LOG_DIR.glob("scrape_missing_*.log"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def tail(path: Path, n: int = 5) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.readlines()[-n:]
    except Exception:
        return []


def main() -> int:
    log = latest_scrape_log()
    if log is None:
        print("[scrape_progress] no scrape_missing log found")
        return 0

    age_min = (dt.datetime.now().timestamp() - log.stat().st_mtime) / 60
    last_lines = "".join(tail(log, 5)).strip()
    size_mb = log.stat().st_size / 1024 / 1024

    stalled = age_min > 60
    color = "yellow" if stalled else "green"
    title = f"Scrape Progress {'STALL' if stalled else 'OK'}"
    body = (
        f"log: {log.name}\n"
        f"size: {size_mb:.2f}MB  age: {age_min:.0f}min\n"
        f"```\n{last_lines[:800]}\n```"
    )
    print(f"[scrape_progress] {title}")
    print(body)

    try:
        from tools.notify import send_discord
        send_discord(title, body, color=color, channel="updates")
    except Exception as e:
        print(f"[scrape_progress] discord notify failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
