"""5/9 当日 リアルタイム監視 CLI (Session #40 B3).

既存 tools/dashboard.py (Streamlit base) と独立した、 軽量 polling 監視。
5/9 当日 朝 PC 動作中に terminal で常時 開いておくと最新状況が分かる。

監視項目 (5 秒 polling):
1. 累計収支 (data/cumulative_results.csv)
2. daily_predict log の最新行
3. race_auto_notify 直近 trigger
4. JRDB / Cookie 鮮度
5. schtasks の次回 trigger 時刻

usage:
  python tools/realtime_monitor.py
  python tools/realtime_monitor.py --interval 10
  python tools/realtime_monitor.py --once  # 1 回だけ表示

V15 production 完全不変 (read-only monitor)。
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")


def get_cumulative() -> tuple[int, int, int]:
    """(rows_total, profit_sum, last_date_int)"""
    p = BASE / "data" / "cumulative_results.csv"
    if not p.exists():
        return (0, 0, 0)
    try:
        import pandas as pd
        df = pd.read_csv(p, low_memory=False)
        df['profit_num'] = pd.to_numeric(df['profit'], errors='coerce').fillna(0)
        df['date'] = df['date'].astype(str).str.replace(r'\.0$', '', regex=True)
        last = df['date'].max() if len(df) else "N/A"
        try:
            last_int = int(last)
        except Exception:
            last_int = 0
        return (len(df), int(df['profit_num'].sum()), last_int)
    except Exception:
        return (0, 0, 0)


def get_daily_predict_status() -> str:
    """logs/ 配下の最新 daily_predict log 行"""
    log_dir = BASE / "logs"
    if not log_dir.exists():
        return "no logs/ dir"
    candidates = sorted(log_dir.glob("daily_predict*.log"), key=os.path.getmtime, reverse=True)
    if not candidates:
        candidates = sorted(log_dir.glob("*.log"), key=os.path.getmtime, reverse=True)
    if not candidates:
        return "no log files"
    latest = candidates[0]
    age_sec = time.time() - latest.stat().st_mtime
    try:
        last_line = ""
        with open(latest, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 1024))
            tail = f.read().decode("utf-8", errors="replace")
            for line in tail.splitlines()[::-1]:
                if line.strip():
                    last_line = line.strip()
                    break
        return f"{latest.name} (mtime: {age_sec/60:.0f} min ago) > {last_line[:120]}"
    except Exception as e:
        return f"err: {e}"


def get_jrdb_freshness() -> str:
    p = BASE / "data" / "jrdb" / "extracted" / "Bac"
    if not p.exists():
        return "no extracted/Bac"
    files = list(p.glob("BAC*.txt"))
    if not files:
        return "0 BAC files"
    dates = []
    for f in files:
        n = f.stem
        if len(n) == 9:
            dates.append("20" + n[3:5] + n[5:7] + n[7:9])
    return f"latest: {max(dates) if dates else '?'}"


def get_next_schtasks() -> str:
    try:
        result = subprocess.run(
            ["schtasks", "/Query", "/FO", "CSV", "/V"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return "schtasks query fail"
        lines = result.stdout.splitlines()
        # フィルタ: Keiba- prefix の running task のみ
        keiba_lines = [l for l in lines if "Keiba-" in l or "Daily" in l]
        # next-run-time 抽出 (Locale 依存だが best effort)
        future = []
        for l in keiba_lines[:50]:
            cells = [c.strip('"') for c in l.split('","')]
            if len(cells) >= 4 and "20" in cells[2]:  # 雑な date 含み
                future.append((cells[0][:30], cells[2][:20]))
        if not future:
            return "(schtasks 解析簡略)"
        return ", ".join(f"{n}@{t}" for n, t in future[:3])
    except Exception as e:
        return f"err: {e}"


def get_cookie_freshness() -> str:
    """data/cookies.json の mtime"""
    p = BASE / "data" / "cookies.json"
    if not p.exists():
        return "no cookies.json"
    age_h = (time.time() - p.stat().st_mtime) / 3600
    return f"age={age_h:.1f}h"


def render_status(once: bool = False) -> str:
    rows, profit, last = get_cumulative()
    dp_status = get_daily_predict_status()
    jrdb = get_jrdb_freshness()
    cookie = get_cookie_freshness()
    next_st = get_next_schtasks()
    now = datetime.datetime.now()

    lines = [
        "=" * 70,
        f"realtime_monitor  ({now:%Y-%m-%d %H:%M:%S})",
        "=" * 70,
        f"  cumulative: rows={rows}, profit={profit:+,d} JPY, last_date={last}",
        f"  retire margin: {profit - (-50_000):+,d} JPY  (line=-50,000)",
        f"  daily_predict: {dp_status}",
        f"  JRDB: {jrdb}",
        f"  Cookie: {cookie}",
        f"  schtasks (Keiba-*): {next_st}",
        "=" * 70,
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="5/9 当日 リアルタイム監視 (Session #40 B3)")
    p.add_argument("--interval", type=int, default=5, help="polling 間隔 (秒)")
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    if args.once:
        print(render_status(once=True))
        return

    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print(render_status())
            print(f"\n  next refresh in {args.interval}s ... (Ctrl+C で終了)")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nbye.")


if __name__ == "__main__":
    main()
