"""AM2:55 Pre-Fire-Check — AM3:00 DailyPremiumScrape 発火前の予防チェック.

毎日 02:55 に Keiba-PreFireCheck から実行され、AM3:00 発火が成功する見込みかを
6つの観点から事前検証。万一問題があれば 4 分以内に手動介入可能な時間帯に
Discord 緊急通知する。

チェック項目:
    1. SCRAPER-GUARD dry-run (daily_premium_scrape 特例が効くか)
    2. netkeiba Cookie 有効性 (.env NETKEIBA_COOKIE 存在 + 非空)
    3. 必要ディレクトリ書き込み権限 (data/, logs/, data/daily_predictions/, data/weekly_premium_cache/)
    4. JRDB 疎通 (HEAD リクエストのみ、実データ取得せず IP バン回避)
    5. ディスク容量 (>= 5GB)
    6. タスクスケジューラ有効確認 (DailyPremiumScrape が Ready + NextRun が AM3:00)

Usage:
    python tools/pre_fire_check.py
    python tools/pre_fire_check.py --silent    # Discord 通知スキップ
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))


def check_scraper_guard() -> dict:
    """Guard が daily_premium_scrape caller で AM3:00 に ALLOW するか."""
    try:
        from tools.scraper_guard import is_scraping_allowed
    except Exception as e:
        return {"ok": False, "severity": "critical", "msg": f"scraper_guard import err: {e}"}
    # 翌日03:00 (= 次の AM3:00 発火時刻) をシミュレート
    now = datetime.datetime.now()
    if now.hour < 3:
        target = now.replace(hour=3, minute=0, second=0, microsecond=0)
    else:
        target = (now + datetime.timedelta(days=1)).replace(hour=3, minute=0, second=0, microsecond=0)
    allowed = is_scraping_allowed(now=target, caller="daily_premium_scrape")
    if allowed:
        return {"ok": True, "severity": "ok",
                "msg": f"ALLOW @ {target.strftime('%Y-%m-%d %H:%M %a')} (daily_premium_scrape 特例)"}
    return {"ok": False, "severity": "critical",
            "msg": f"BLOCKED @ {target.strftime('%Y-%m-%d %H:%M %a')} — SCRAPER-GUARD が誤停止する",
            "recovery": "tools/scraper_guard.py の _premium_scrape_early_slot を修正"}


def check_cookie() -> dict:
    """.env の NETKEIBA_COOKIE が存在 + 非空."""
    env_path = BASE / ".env"
    if not env_path.exists():
        return {"ok": False, "severity": "critical", "msg": ".env が存在しない"}
    try:
        content = env_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"ok": False, "severity": "warning", "msg": f".env read err: {e}"}
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("NETKEIBA_COOKIE="):
            val = line.split("=", 1)[1].strip().strip("'\"")
            if len(val) > 50:
                return {"ok": True, "severity": "ok", "msg": f"Cookie OK ({len(val)} 文字)"}
            return {"ok": False, "severity": "critical",
                    "msg": f"NETKEIBA_COOKIE が短すぎ ({len(val)} 文字)",
                    "recovery": "python tools/refresh_cookie.py"}
    return {"ok": False, "severity": "critical",
            "msg": "NETKEIBA_COOKIE が .env に未設定",
            "recovery": "python tools/refresh_cookie.py"}


def check_dirs() -> dict:
    """必要ディレクトリの書き込み権限."""
    dirs = ["data", "logs", "data/daily_predictions", "data/weekly_premium_cache"]
    missing = []
    not_writable = []
    for d in dirs:
        p = BASE / d
        if not p.exists():
            missing.append(d)
        elif not os.access(p, os.W_OK):
            not_writable.append(d)
    if missing:
        return {"ok": False, "severity": "warning",
                "msg": f"ディレクトリ欠落: {missing}",
                "recovery": f"mkdir -p {' '.join(missing)}"}
    if not_writable:
        return {"ok": False, "severity": "critical",
                "msg": f"書き込み権限なし: {not_writable}"}
    return {"ok": True, "severity": "ok", "msg": f"書き込み権限 OK ({len(dirs)} dirs)"}


def check_jrdb_reachable() -> dict:
    """JRDB へ HEAD リクエスト (軽量疎通確認)."""
    try:
        import urllib.request
        # JRDB のトップページだけ、HEAD リクエスト
        req = urllib.request.Request("http://www.jrdb.com/", method="HEAD")
        with urllib.request.urlopen(req, timeout=5) as resp:
            code = resp.status
        if 200 <= code < 400:
            return {"ok": True, "severity": "ok", "msg": f"JRDB 疎通 OK (HTTP {code})"}
        return {"ok": False, "severity": "warning",
                "msg": f"JRDB HTTP {code}"}
    except Exception as e:
        return {"ok": False, "severity": "warning",
                "msg": f"JRDB 疎通失敗: {type(e).__name__}"}


def check_disk() -> dict:
    """data ドライブ空き >= 5GB."""
    try:
        total, used, free = shutil.disk_usage(str(BASE))
        free_gb = free / (1024 ** 3)
        if free_gb >= 5.0:
            return {"ok": True, "severity": "ok",
                    "msg": f"空き {free_gb:.1f} GB"}
        if free_gb >= 1.0:
            return {"ok": False, "severity": "warning",
                    "msg": f"空き {free_gb:.1f} GB (<5GB warning)"}
        return {"ok": False, "severity": "critical",
                "msg": f"空き {free_gb:.1f} GB (<1GB CRITICAL)"}
    except Exception as e:
        return {"ok": False, "severity": "warning", "msg": f"disk check err: {e}"}


def check_task_scheduler() -> dict:
    """DailyPremiumScrape が Ready + NextRun が AM3:00."""
    try:
        cmd = [
            "powershell.exe", "-NoProfile", "-Command",
            (
                "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                "$t = Get-ScheduledTask | Where-Object { $_.TaskName -eq 'DailyPremiumScrape' } | Select-Object -First 1; "
                "if (-not $t) { throw 'DailyPremiumScrape not found' } "
                "$info = Get-ScheduledTaskInfo -TaskName $t.TaskName -TaskPath $t.TaskPath; "
                "[PSCustomObject]@{ "
                "  State=$t.State.ToString(); "
                "  TaskPath=$t.TaskPath; "
                "  NextRun=$info.NextRunTime.ToString('o') "
                "} | ConvertTo-Json"
            ),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            return {"ok": False, "severity": "critical",
                    "msg": f"DailyPremiumScrape タスク未登録 / 取得失敗: {(r.stderr or '')[:200]}"}
        if not r.stdout.strip():
            return {"ok": False, "severity": "warning",
                    "msg": "task info 取得空 (タイミング問題の可能性)"}
        data = json.loads(r.stdout)
        state = data.get("State", "")
        next_run = data.get("NextRun", "")
        if state not in ("Ready", "Running"):
            return {"ok": False, "severity": "critical",
                    "msg": f"state={state} (Ready/Running 以外)"}
        # NextRun が 03:00 時刻か (分まで確認)
        if "T03:00" in next_run or "T03:01" in next_run:
            return {"ok": True, "severity": "ok",
                    "msg": f"Ready, next={next_run}"}
        return {"ok": False, "severity": "warning",
                "msg": f"NextRun が AM3:00 でない: {next_run}"}
    except Exception as e:
        return {"ok": False, "severity": "warning", "msg": f"task check err: {e}"}


def run_all_checks() -> tuple[list[tuple[str, dict]], str]:
    """全チェック実行。(results, overall_severity) を返す."""
    checks = [
        ("SCRAPER-GUARD",    check_scraper_guard()),
        ("Cookie",           check_cookie()),
        ("Directories",      check_dirs()),
        ("JRDB reachable",   check_jrdb_reachable()),
        ("Disk space",       check_disk()),
        ("Task Scheduler",   check_task_scheduler()),
    ]
    # 総合判定: critical 含めば critical、warning 含めば warning、全OK なら ok
    if any(r.get("severity") == "critical" for _, r in checks):
        overall = "critical"
    elif any(r.get("severity") == "warning" for _, r in checks):
        overall = "warning"
    else:
        overall = "ok"
    return checks, overall


def notify_discord(checks: list[tuple[str, dict]], overall: str) -> None:
    """Discord に結果投稿."""
    if overall == "ok":
        title = "Pre-Fire-Check OK"
        subtitle = f"AM3:00 発火見込み ({len(checks)}/{len(checks)} OK)"
        color = "green"
    elif overall == "warning":
        title = "Pre-Fire-Check 警告"
        subtitle = "動作継続見込みだが要確認"
        color = "yellow"
    else:
        title = "CRITICAL: Pre-Fire-Check 失敗"
        subtitle = "AM3:00 発火が失敗する見込み @everyone 要手動介入"
        color = "red"

    lines = []
    for name, r in checks:
        icon = "OK" if r["ok"] else ("WARN" if r["severity"] == "warning" else "NG")
        lines.append(f"[{icon}] {name}: {r['msg']}")
        if not r["ok"] and r.get("recovery"):
            lines.append(f"      → {r['recovery']}")
    body = "\n".join(lines)

    try:
        subprocess.run(
            [sys.executable, str(BASE / "tools/notify_done.py"),
             title, subtitle, body, "--color", color],
            check=False, timeout=30,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except Exception as e:
        print(f"[WARN] Discord 通知失敗: {e}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--silent", action="store_true", help="Discord 通知しない")
    args = p.parse_args()

    # Windows cp932 で ✓/⚠/✗ が UnicodeEncodeError になるため stdout を utf-8 に切替
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    checks, overall = run_all_checks()

    print("=" * 60)
    print(f"PRE-FIRE-CHECK @ {datetime.datetime.now().isoformat()}")
    print("=" * 60)
    for name, r in checks:
        if r["ok"]:
            icon = "OK"
        elif r["severity"] == "warning":
            icon = "WARN"
        else:
            icon = "NG"
        print(f" [{icon}] {name}: {r['msg']}")
        if not r["ok"] and r.get("recovery"):
            print(f"      recovery: {r['recovery']}")
    print("=" * 60)
    print(f"OVERALL: {overall.upper()}")

    # 結果 JSON 保存 (他ツールから参照できるよう)
    results_dir = BASE / "data" / "fire_check_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "timestamp": datetime.datetime.now().isoformat(),
        "overall": overall,
        "checks": [{"name": n, **r} for n, r in checks],
    }
    out_path = results_dir / f"pre_fire_check_{datetime.date.today().strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {out_path}")

    if not args.silent:
        notify_discord(checks, overall)

    # critical 時は非ゼロ終了で Task Scheduler ログにも残す
    return 0 if overall == "ok" else (1 if overall == "warning" else 2)


if __name__ == "__main__":
    sys.exit(main())
