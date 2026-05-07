"""JV-Link 5/1-5/7 backfill script (Session #41 C).

5/1-5/7 期間の JRA レース全データを JV-Link 経由で取得。 32-bit Python venv で実行。

取得対象:
- RACE: 各日 30-40 ファイル想定
- SE:   馬毎レース情報
- HR:   払戻 (jra_payouts.csv 4/6 停止 解消データ)
- O1:   単複オッズ (paci 自前算出 用)

usage (32-bit venv):
  python tools\\jvlink_backfill_5_1_5_7.py
  python tools\\jvlink_backfill_5_1_5_7.py --from 20260501 --to 20260507
  python tools\\jvlink_backfill_5_1_5_7.py --datatypes RACE,HR --dry-run

V15 production 完全不変 (新規 data/jvlink/ への保存のみ)。
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


def daterange(start_date: str, end_date: str):
    """YYYYMMDD start <= d <= end の各日付を yield."""
    s = datetime.datetime.strptime(start_date, "%Y%m%d").date()
    e = datetime.datetime.strptime(end_date, "%Y%m%d").date()
    cur = s
    while cur <= e:
        yield cur.strftime("%Y%m%d")
        cur += datetime.timedelta(days=1)


def main():
    p = argparse.ArgumentParser(description="JV-Link 5/1-5/7 backfill (Session #41 C)")
    p.add_argument("--from", dest="from_date", default="20260501")
    p.add_argument("--to", dest="to_date", default="20260507")
    p.add_argument("--datatypes", default="RACE,SE,HR,O1")
    p.add_argument("--option", type=int, default=4)
    p.add_argument("--out-dir", default="data/jvlink")
    p.add_argument("--parse", action="store_true")
    p.add_argument("--sid", default="UNKNOWN")
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sleep-between-days", type=float, default=2.0)
    args = p.parse_args()

    arch = platform.architecture()[0]
    if arch != "32bit" and not args.dry_run:
        print(f"[!] WARN: 64-bit Python では JV-Link COM 接続不可 (arch={arch})")
        print(f"[!] 32-bit venv (C:\\Users\\takum\\jvlink-venv\\) で実行してください")
        print(f"[!] --dry-run なら plan のみ表示")

    datatypes = [d.strip() for d in args.datatypes.split(",") if d.strip()]
    dates = list(daterange(args.from_date, args.to_date))

    print("=" * 70)
    print(f"JV-Link backfill plan ({args.from_date} - {args.to_date})")
    print("=" * 70)
    print(f"  dates ({len(dates)}): {dates}")
    print(f"  datatypes ({len(datatypes)}): {datatypes}")
    print(f"  option: {args.option}")
    print(f"  parse: {args.parse}")
    print(f"  out_dir: {args.out_dir}")
    print(f"  estimated runs: {len(dates) * len(datatypes)}")
    print(f"  estimated time: {len(dates) * len(datatypes) * 30}s "
          f"~ {len(dates) * len(datatypes) * 30 // 60}min")

    if args.dry_run:
        print("\n[DRY-RUN] 実行 skip")
        for d in dates:
            for dt in datatypes:
                print(f"  fetch {dt} {d}")
        return

    # 実 fetch (32-bit venv 前提)
    try:
        from jvlink_fetcher_v2 import fetch_to_csv
    except ImportError as e:
        print(f"[!] jvlink_fetcher_v2 import 失敗: {e}")
        sys.exit(1)

    summary = {}
    total_records = 0
    t0 = time.time()
    for d in dates:
        print(f"\n========================== {d} ==========================")
        day_summary = {}
        for dt in datatypes:
            print(f"\n  --- {dt} ---")
            try:
                fromtime = d + "000000"
                n = fetch_to_csv(
                    dataspec=dt,
                    fromtime=fromtime,
                    out_dir=args.out_dir,
                    sid=args.sid,
                    option=args.option,
                    max_records=args.max_records,
                    parse=args.parse,
                )
                day_summary[dt] = {"records": n, "status": "ok"}
                total_records += n
            except Exception as e:
                print(f"  [ERROR] {dt} {d}: {e}")
                day_summary[dt] = {"records": 0, "status": f"error: {e}"}
            time.sleep(1.0)
        summary[d] = day_summary
        time.sleep(args.sleep_between_days)

    elapsed = time.time() - t0
    print("\n========================= summary =========================")
    print(f"  elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  total records: {total_records:,}")
    for d, day in summary.items():
        ok = sum(1 for v in day.values() if v.get('status') == 'ok')
        nrec = sum(v.get('records', 0) for v in day.values())
        print(f"  {d}: {ok}/{len(day)} OK, {nrec:,} records")

    out_path = BASE / "data" / "v18" / f"jvlink_backfill_summary_{args.from_date}_{args.to_date}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "from": args.from_date,
        "to": args.to_date,
        "datatypes": datatypes,
        "elapsed_sec": elapsed,
        "total_records": total_records,
        "summary": summary,
        "fetched_at": datetime.datetime.now().isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  summary written: {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
