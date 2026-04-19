"""朝の1発ダッシュボード.

data/fire_check_results/YYYYMMDD.json を読み、Pre/AM3/AM6/AM8 の発火状況を
整形表示 + Discord 送信。

Usage:
    python tools/morning_dashboard.py              # 今朝 (today) の結果
    python tools/morning_dashboard.py --date 20260420
    python tools/morning_dashboard.py --silent     # Discord 送信しない
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


ICONS = {
    "ok":       "[OK]",
    "warning":  "[WARN]",
    "critical": "[NG]",
    "pending":  "[...]",
    "unknown":  "[?]",
}


def load_results(ymd: str) -> dict:
    path = BASE / "data" / "fire_check_results" / f"{ymd}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_pre_fire(ymd: str) -> dict:
    path = BASE / "data" / "fire_check_results" / f"pre_fire_check_{ymd}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def format_dashboard(ymd: str, results: dict, pre_fire: dict) -> str:
    dt = datetime.datetime.strptime(ymd, "%Y%m%d")
    wd = dt.strftime("%a")
    lines = []
    lines.append("+-----------------------------------------+")
    lines.append(f"| Morning Dashboard - {dt.strftime('%Y/%m/%d')} ({wd})      |")
    lines.append("+-----------------------------------------+")
    lines.append("")

    # Pre-Fire-Check
    lines.append("## Pre-Fire-Check (AM02:55)")
    if pre_fire:
        overall = pre_fire.get("overall", "unknown")
        icon = ICONS.get(overall, ICONS["unknown"])
        lines.append(f"  {icon} {overall.upper()}")
        for c in pre_fire.get("checks", []):
            sev = c.get("severity", "unknown")
            i = ICONS.get(sev, ICONS["unknown"])
            lines.append(f"    {i} {c.get('name', '?')}: {c.get('msg', '')}")
    else:
        lines.append(f"  {ICONS['pending']} 未実行")
    lines.append("")

    # 各タスク
    lines.append("## 自動タスク発火状況")
    task_order = [
        ("DailyPremiumScrape", "AM03:00"),
        ("DailyJrdbKyi",       "AM06:00"),
        ("DailyPredict",       "AM08:00"),
    ]
    for name, time_str in task_order:
        r = results.get(name)
        if not r:
            lines.append(f"  {ICONS['pending']} {time_str} {name} - 未実行")
            continue
        status = r.get("status", "unknown")
        icon = ICONS.get(status, ICONS["unknown"])
        msg = r.get("message", "")
        size = r.get("size")
        rows = r.get("rows")
        detail = ""
        if size:
            detail += f"size={size}B"
        if rows:
            detail += (f", rows={rows}" if detail else f"rows={rows}")
        if detail:
            msg += f" ({detail})"
        lines.append(f"  {icon} {time_str} {name} - {msg}")
    lines.append("")

    # サマリー
    critical_cnt = sum(1 for r in results.values() if r.get("status") == "critical")
    warning_cnt = sum(1 for r in results.values() if r.get("status") == "warning")
    lines.append("## サマリー")
    lines.append(f"  CRITICAL: {critical_cnt}")
    lines.append(f"  WARNING: {warning_cnt}")
    if critical_cnt == 0 and warning_cnt == 0:
        lines.append("  手動介入: 不要")
    else:
        lines.append("  手動介入: 要確認")

    return "\n".join(lines)


def send_discord(body: str, title: str = "Morning Dashboard") -> None:
    try:
        subprocess.run(
            [sys.executable, str(BASE / "tools/notify_done.py"),
             title, "", body, "--color", "blue"],
            check=False, timeout=30,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except Exception as e:
        print(f"[WARN] Discord 送信失敗: {e}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--silent", action="store_true")
    args = p.parse_args()

    ymd = args.date if args.date else datetime.date.today().strftime("%Y%m%d")
    results = load_results(ymd)
    pre_fire = load_pre_fire(ymd)
    body = format_dashboard(ymd, results, pre_fire)
    print(body)
    if not args.silent:
        send_discord(body, title=f"Morning Dashboard {ymd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
