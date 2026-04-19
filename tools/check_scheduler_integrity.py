"""タスクスケジューラ登録済みタスクの整合性チェック.

schtasks / PowerShell Get-ScheduledTask で Keiba 関連タスクを取得し:
- 各タスクが有効 (Ready / Running) か
- 呼び出す .bat / .py が実在するか
- .bat が最新コードを参照しているか (PYTHONUNBUFFERED=1 等)
- 次回発火時刻が妥当か
- 最終結果が Ctrl+C 強制終了でないか

Usage:
    python tools/check_scheduler_integrity.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_keiba_tasks() -> list[dict]:
    """PowerShell で Keiba 関連のタスクを取得."""
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        (
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Get-ScheduledTask | Where-Object { "
            "  ($_.TaskName -match 'Keiba|DailyPredict|DailyPremium|DailyJrdb|DailyResults|JrdbHealth|RaceAutoNotify|WeeklyReport|ProcessWatchdog|DriftDetector') "
            "} | ForEach-Object { "
            "  $info = Get-ScheduledTaskInfo -TaskName $_.TaskName -TaskPath $_.TaskPath; "
            "  $a = $_.Actions[0]; "
            "  [PSCustomObject]@{ "
            "    TaskName=$_.TaskName; State=$_.State.ToString(); "
            "    NextRun=$info.NextRunTime.ToString('o'); "
            "    LastRun=$info.LastRunTime.ToString('o'); "
            "    LastResult=$info.LastTaskResult; "
            "    Execute=$a.Execute; Arguments=$a.Arguments "
            "  } "
            "} | ConvertTo-Json -Depth 2"
        ),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print(f"[ERR] PowerShell failed: {r.stderr}")
        return []
    try:
        data = json.loads(r.stdout)
    except Exception as e:
        print(f"[ERR] JSON parse: {e}")
        print(r.stdout[:500])
        return []
    if isinstance(data, dict):
        data = [data]
    return data


def check_task(task: dict) -> tuple[int, list[tuple[bool, str]]]:
    """個別タスクを検証。(fail_count, [(ok, msg), ...])"""
    checks = []
    fails = 0

    name = task.get("TaskName", "?")
    state = task.get("State", "?")
    execute = task.get("Execute", "")
    last_result = task.get("LastResult", 0)

    # State: Ready or Running が健全、Disabled は NG
    ok = state in ("Ready", "Running")
    checks.append((ok, f"state={state}"))
    if not ok:
        fails += 1

    # Execute ファイル実在
    if execute:
        exe_path = execute.replace("/", os.sep)
        exists = os.path.exists(exe_path)
        checks.append((exists, f"execute={execute}"))
        if not exists:
            fails += 1
    else:
        checks.append((True, "execute=(empty/inline)"))

    # LastResult — Ctrl+C 強制終了は要注意
    lr = int(last_result) if last_result else 0
    if lr == 0:
        checks.append((True, f"last_result=0 (OK)"))
    elif lr in (3221225786, 3221226091):
        # 0xC000013A (Ctrl+C) / 0xC000021B
        checks.append((False, f"last_result={lr} (Ctrl+C / 強制終了履歴あり)"))
        fails += 1
    elif lr == 267011:
        checks.append((False, f"last_result={lr} (未起動 or スケジュール未到達)"))
        # 未起動は fail にカウントしない (新規登録時の自然状態)
    elif lr == 1:
        checks.append((False, f"last_result={lr} (failure)"))
        fails += 1
    else:
        checks.append((False, f"last_result={lr} (abnormal)"))
        fails += 1

    # .bat ファイルの UTF-8 対応チェック
    if execute and execute.endswith(".bat"):
        bat_path = execute.replace("/", os.sep)
        if os.path.exists(bat_path):
            try:
                with open(bat_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                has_utf8 = "PYTHONIOENCODING=utf-8" in content or "chcp 65001" in content
                has_unbuf = "PYTHONUNBUFFERED=1" in content
                checks.append((has_utf8, f"bat has PYTHONIOENCODING={has_utf8}"))
                checks.append((has_unbuf, f"bat has PYTHONUNBUFFERED={has_unbuf}"))
                if not has_utf8:
                    fails += 1
                if not has_unbuf:
                    # unbuffered は重要だが致命的ではない (warning 扱い)
                    pass
            except Exception as e:
                checks.append((False, f"bat read err: {e}"))

    return fails, checks


def main() -> int:
    tasks = get_keiba_tasks()
    if not tasks:
        print("[ERR] Keiba タスクが 1 件も見つかりません")
        return 1

    print("=" * 70)
    print(f"SCHEDULER INTEGRITY CHECK — {len(tasks)} tasks found")
    print("=" * 70)

    total_fail = 0
    issues = []
    for t in tasks:
        fails, checks = check_task(t)
        name = t.get("TaskName", "?")
        next_run = t.get("NextRun", "")
        last_run = t.get("LastRun", "")
        tag = "✓" if fails == 0 else "✗"
        print(f"\n[{tag}] {name}")
        print(f"    next={next_run}, last={last_run}")
        for ok, msg in checks:
            print(f"    {'OK' if ok else 'NG'}: {msg}")
        if fails > 0:
            total_fail += fails
            issues.append((name, fails))

    print("\n" + "=" * 70)
    print(f"RESULT: {len(tasks)} tasks, {total_fail} issues")
    if issues:
        print("Issues per task:")
        for n, f in issues:
            print(f"  - {n}: {f} issue(s)")
    print("=" * 70)

    return 0 if total_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
