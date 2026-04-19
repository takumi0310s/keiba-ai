"""毎晩 23:00 頃に自動実行し、翌日発火予定タスクを事前チェック.

チェック内容:
1. 翌日発火予定の Keiba タスクを列挙
2. 各タスクの最終結果を確認 (Ctrl+C / 強制終了履歴があれば警告)
3. 必要なファイル (モデル, .env, Cookie) の存在確認
4. scraper_guard の翌日時刻での挙動確認

異常があれば Discord 緊急通知、正常なら軽い通知。

Usage:
    python tools/nightly_sanity_check.py
    python tools/nightly_sanity_check.py --date 2026-04-20   # 特定日の確認
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from tools.scraper_guard import is_scraping_allowed  # noqa: E402

REQUIRED_FILES = [
    "keiba_model_v15_central_live.pkl.gz",
    ".env",
    "tools/scraper_guard.py",
]


def notify_discord(title: str, msg: str, color: str = "blue") -> None:
    """Discord 通知 (notify_done.py 経由)."""
    try:
        subprocess.run(
            [sys.executable, os.path.join(BASE, "tools", "notify_done.py"),
             title, msg, "--color", color],
            timeout=30, env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
    except Exception as e:
        print(f"[WARN] Discord 通知失敗: {e}")


def get_tomorrow_tasks(target_date: datetime) -> list[dict]:
    """PowerShell で 翌日発火予定の Keiba タスクを取得."""
    ymd = target_date.strftime("%Y-%m-%d")
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        (
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Get-ScheduledTask | Where-Object { "
            "  $_.TaskName -match 'Keiba|DailyPredict|DailyPremium|DailyJrdb|DailyResults"
            "|JrdbHealth|RaceAutoNotify|WeeklyReport|ProcessWatchdog|DriftDetector' "
            "} | ForEach-Object { "
            "  $info = Get-ScheduledTaskInfo -TaskName $_.TaskName -TaskPath $_.TaskPath; "
            "  $a = $_.Actions[0]; "
            "  [PSCustomObject]@{ "
            "    TaskName=$_.TaskName; State=$_.State.ToString(); "
            "    NextRun=$info.NextRunTime.ToString('o'); "
            "    LastResult=$info.LastTaskResult; Execute=$a.Execute "
            "  } "
            "} | ConvertTo-Json -Depth 2"
        ),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return []
    try:
        data = json.loads(r.stdout)
    except Exception:
        return []
    if isinstance(data, dict):
        data = [data]

    # 翌日発火予定のみフィルタ
    tomorrow_tasks = []
    for t in data:
        nr = t.get("NextRun", "")
        if nr and nr.startswith(ymd):
            tomorrow_tasks.append(t)
    return tomorrow_tasks


def check_required_files() -> list[tuple[bool, str]]:
    results = []
    for f in REQUIRED_FILES:
        p = os.path.join(BASE, f)
        results.append((os.path.exists(p), f))
    return results


def check_guard_for_tasks(tasks: list[dict]) -> list[tuple[bool, str]]:
    """各タスクの NextRun 時刻で guard を評価."""
    results = []
    caller_map = {
        "DailyPremiumScrape": "daily_premium_scrape",
        "DailyPredict": "daily_predict",
        "DailyJrdbKyi": "daily_jrdb_kyi",
        "RaceAutoNotify": "race_auto_notify",
        "JrdbHealthCheck": "jrdb_health_check",
        "DailyResults": "daily_results",
    }
    for t in tasks:
        name = t.get("TaskName", "")
        nr = t.get("NextRun", "")
        caller = None
        for key, val in caller_map.items():
            if key in name:
                caller = val
                break
        if not nr:
            continue
        try:
            dt = datetime.fromisoformat(nr.replace("Z", "+00:00"))
            # TZ を local に合わせる
            dt = dt.replace(tzinfo=None)
            allowed = is_scraping_allowed(now=dt, caller=caller)
            results.append((allowed,
                            f"{name} @ {dt.strftime('%H:%M')} caller={caller or '(none)'} → {'ALLOW' if allowed else 'STOP'}"))
        except Exception as e:
            results.append((False, f"{name}: guard check err {e}"))
    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=None,
                   help="チェック対象日 YYYY-MM-DD (default: 翌日)")
    p.add_argument("--silent", action="store_true", help="Discord 通知しない")
    args = p.parse_args()

    if args.date:
        target = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        target = datetime.now() + timedelta(days=1)

    print(f"[nightly_sanity_check] 対象日: {target.strftime('%Y-%m-%d (%a)')}")

    issues: list[str] = []
    warnings: list[str] = []

    # 1. 翌日発火予定タスク列挙
    tasks = get_tomorrow_tasks(target)
    print(f"\n[1] 翌日発火予定タスク: {len(tasks)} 件")
    for t in tasks:
        name = t.get("TaskName", "?")
        nr = t.get("NextRun", "?")
        state = t.get("State", "?")
        lr = t.get("LastResult", 0)
        print(f"  - {name}: state={state}, next={nr}, last_result={lr}")
        # Ctrl+C 履歴を warning
        if int(lr or 0) in (3221225786, 3221226091):
            warnings.append(f"{name}: 過去に Ctrl+C 強制終了あり (last={lr})")
        if state not in ("Ready", "Running"):
            issues.append(f"{name}: state={state} (Ready/Running 以外)")

    # 2. 必要ファイル存在確認
    print(f"\n[2] 必要ファイル確認")
    for ok, fname in check_required_files():
        print(f"  {'OK' if ok else 'NG'}: {fname}")
        if not ok:
            issues.append(f"必要ファイル欠落: {fname}")

    # 3. Guard 挙動確認
    print(f"\n[3] SCRAPER-GUARD 挙動確認")
    guard_results = check_guard_for_tasks(tasks)
    for ok, msg in guard_results:
        print(f"  {'OK' if ok else 'NG'}: {msg}")
        if not ok:
            issues.append(f"Guard で停止: {msg}")

    # まとめ
    print("\n" + "=" * 60)
    if issues:
        print(f"❌ ISSUES: {len(issues)}")
        for i in issues:
            print(f"  - {i}")
    if warnings:
        print(f"⚠ WARNINGS: {len(warnings)}")
        for w in warnings:
            print(f"  - {w}")
    if not issues and not warnings:
        print("✅ 全チェック PASS")
    print("=" * 60)

    # Discord 通知
    if not args.silent:
        title = f"nightly_sanity_check {target.strftime('%Y-%m-%d')}"
        if issues:
            body = f"❌ {len(issues)}件の問題:\n" + "\n".join(f"- {x}" for x in issues)
            if warnings:
                body += f"\n⚠ {len(warnings)}件の警告:\n" + "\n".join(f"- {x}" for x in warnings)
            notify_discord(title, body, color="red")
        elif warnings:
            body = f"⚠ {len(warnings)}件の警告 (動作継続):\n" + "\n".join(f"- {x}" for x in warnings)
            body += f"\n\n翌日発火予定タスク {len(tasks)} 件は ALL READY"
            notify_discord(title, body, color="yellow")
        else:
            notify_discord(title, f"✅ 翌日発火予定 {len(tasks)} 件 ALL READY", color="green")

    return 0 if not issues else 2


if __name__ == "__main__":
    sys.exit(main())
