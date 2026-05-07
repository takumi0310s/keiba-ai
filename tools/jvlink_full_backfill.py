"""JV-Link 6 年分 (2020-2025) full backfill (Session #41 E).

V20 (Phase 3 後半 6/9-30) 学習 data 主軸用に、 過去 6 年分の JRA レースデータを
JV-Link 経由で段階的に取得。 schtasks Keiba-JvlinkBackfillNightly で夜間自動化。

期間: 2020/01/01 - 2025/12/31 (約 2,200 日)
datatype: RACE / SE / HR / O1 / TCOV / WOOD / BLOD = 7 種

推定 size:
  RACE:  約 5-8 GB (2,200 日 × 2-4 MB)
  SE:    約 15-25 GB (馬毎、 出走数大)
  HR:    約 1-2 GB
  O1:    約 8-15 GB
  TCOV:  約 5-10 GB
  WOOD:  約 2-5 GB
  BLOD:  約 0.5-1 GB
  ----
  計:    約 36-66 GB (raw)、 圧縮後 約 25-50 GB

usage (32-bit venv):
  # 段階的 (推奨): 1 か月分ずつ
  python tools\\jvlink_full_backfill.py --from 20200101 --to 20200131
  # 1 年分
  python tools\\jvlink_full_backfill.py --from 20200101 --to 20201231 --datatypes RACE,SE,HR,O1
  # resume mode (既 fetch を skip)
  python tools\\jvlink_full_backfill.py --resume

V15 production 完全不変 (data/jvlink_full/ 新規 path)。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import time
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
sys.path.insert(0, str(BASE / "tools"))


def daterange_monthly(start_date: str, end_date: str):
    """月単位で (month_start, month_end) tuple を yield."""
    s = datetime.datetime.strptime(start_date, "%Y%m%d").date()
    e = datetime.datetime.strptime(end_date, "%Y%m%d").date()
    cur = s.replace(day=1)
    while cur <= e:
        # 月末
        if cur.month == 12:
            month_end = cur.replace(year=cur.year+1, month=1, day=1) - datetime.timedelta(days=1)
        else:
            month_end = cur.replace(month=cur.month+1, day=1) - datetime.timedelta(days=1)
        yield (cur.strftime("%Y%m%d"), min(month_end, e).strftime("%Y%m%d"))
        if cur.month == 12:
            cur = cur.replace(year=cur.year+1, month=1, day=1)
        else:
            cur = cur.replace(month=cur.month+1, day=1)


def daterange_daily(start_date: str, end_date: str):
    s = datetime.datetime.strptime(start_date, "%Y%m%d").date()
    e = datetime.datetime.strptime(end_date, "%Y%m%d").date()
    cur = s
    while cur <= e:
        yield cur.strftime("%Y%m%d")
        cur += datetime.timedelta(days=1)


def is_already_fetched(date: str, datatype: str, out_dir: Path) -> bool:
    p = out_dir / datatype / f"{date}_meta.json"
    if not p.exists(): return False
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
        return meta.get("n_records", 0) >= 0  # 取得済み (0 records でも完了 mark)
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser(description="JV-Link 6 年分 full backfill (Session #41 E)")
    p.add_argument("--from", dest="from_date", default="20200101")
    p.add_argument("--to", dest="to_date", default="20251231")
    p.add_argument("--datatypes", default="RACE,SE,HR,O1,TCOV,WOOD,BLOD")
    p.add_argument("--option", type=int, default=4)
    p.add_argument("--out-dir", default="data/jvlink_full")
    p.add_argument("--parse", action="store_true")
    p.add_argument("--sid", default="UNKNOWN")
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="既 fetch を skip")
    p.add_argument("--monthly", action="store_true", help="月単位で fetch (default: daily)")
    p.add_argument("--sleep-between-days", type=float, default=2.0)
    p.add_argument("--max-runs", type=int, default=None, help="1 回の実行で fetch する run 数上限")
    args = p.parse_args()

    arch = platform.architecture()[0]
    if arch != "32bit" and not args.dry_run:
        print(f"[!] WARN: 64-bit Python では JV-Link COM 接続不可 (arch={arch})")
        print(f"[!] 32-bit venv (C:\\Users\\takum\\jvlink-venv\\) で実行してください")
        print(f"[!] --dry-run なら plan のみ表示")
        sys.exit(1) if not args.dry_run else None

    datatypes = [d.strip() for d in args.datatypes.split(",") if d.strip()]
    out_dir = BASE / args.out_dir

    # plan 表示
    if args.monthly:
        ranges = list(daterange_monthly(args.from_date, args.to_date))
        n_units = len(ranges)
    else:
        dates = list(daterange_daily(args.from_date, args.to_date))
        ranges = [(d, d) for d in dates]
        n_units = len(dates)

    total_runs = n_units * len(datatypes)
    print("=" * 70)
    print(f"JV-Link full backfill plan ({args.from_date} - {args.to_date})")
    print("=" * 70)
    print(f"  granularity: {'monthly' if args.monthly else 'daily'}")
    print(f"  date units: {n_units}")
    print(f"  datatypes ({len(datatypes)}): {datatypes}")
    print(f"  estimated runs: {total_runs}")
    print(f"  estimated time: {total_runs * 30}s ~ {total_runs * 30 // 3600}h")
    print(f"  estimated size: 36-66 GB (depend on datatypes)")
    print(f"  out_dir: {args.out_dir}")
    print(f"  resume: {args.resume}")
    print(f"  parse: {args.parse}")
    print(f"  max_runs: {args.max_runs}")

    if args.dry_run:
        print("\n[DRY-RUN] 実行 skip。 plan のみ表示")
        return

    # 実 fetch
    try:
        from jvlink_fetcher_v2 import fetch_to_csv
    except ImportError as e:
        print(f"[!] jvlink_fetcher_v2 import 失敗: {e}")
        sys.exit(1)

    summary = {}
    runs_done = 0
    t0 = time.time()
    for from_d, to_d in ranges:
        days_in_range = list(daterange_daily(from_d, to_d))
        for d in days_in_range:
            if args.max_runs is not None and runs_done >= args.max_runs:
                print(f"\n[stop] max_runs={args.max_runs} reached")
                break
            day_summary = {}
            for dt in datatypes:
                if args.resume and is_already_fetched(d, dt, out_dir):
                    day_summary[dt] = {"records": -1, "status": "skipped (resume)"}
                    continue
                try:
                    fromtime = d + "000000"
                    n = fetch_to_csv(
                        dataspec=dt,
                        fromtime=fromtime,
                        out_dir=str(out_dir.relative_to(BASE)),
                        sid=args.sid,
                        option=args.option,
                        max_records=args.max_records,
                        parse=args.parse,
                    )
                    day_summary[dt] = {"records": n, "status": "ok"}
                    runs_done += 1
                except Exception as e:
                    print(f"  [ERROR] {dt} {d}: {e}")
                    day_summary[dt] = {"records": 0, "status": f"error: {e}"}
                time.sleep(1.0)
            summary[d] = day_summary
            time.sleep(args.sleep_between_days)
        if args.max_runs is not None and runs_done >= args.max_runs:
            break

    elapsed = time.time() - t0
    print("\n========================= summary =========================")
    print(f"  elapsed: {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    print(f"  runs done: {runs_done}")

    out_path = BASE / "data" / "v18" / f"jvlink_full_backfill_{args.from_date}_{args.to_date}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "from": args.from_date,
        "to": args.to_date,
        "datatypes": datatypes,
        "elapsed_sec": elapsed,
        "runs_done": runs_done,
        "summary": summary,
        "fetched_at": datetime.datetime.now().isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  summary: {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
