"""Session #67 A: 5/9 全 36R 結果取得.

`tools/daily_results.fetch_race_result()` を呼んで netkeiba から着順 + 払戻取得。
data/results/20260509_results.csv に書き出す。

Usage:
    python tools/session_67_fetch_results.py
    python tools/session_67_fetch_results.py --date 20260509
    python tools/session_67_fetch_results.py --date 20260509 --retry 3
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tools"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="20260509")
    p.add_argument("--retry", type=int, default=2)
    p.add_argument("--sleep", type=float, default=0.7)
    args = p.parse_args()

    pred_csv = BASE / "data" / "daily_predictions" / f"{args.date}.csv"
    if not pred_csv.exists():
        print(f"[FAIL] no daily_predictions for {args.date}", file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    pred_df = pd.read_csv(pred_csv, dtype=str)

    from daily_results import fetch_race_result

    out_dir = BASE / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.date}_results.csv"

    rows = []
    n_ok = n_fail = 0
    fields = ["race_id", "course", "race_num", "race_name", "num_horses",
              "finish_1", "finish_2", "finish_3",
              "trio_nums", "umaren_nums",
              "payout_tansho", "payout_umaren", "payout_trio", "payout_wide",
              "fetch_status"]

    for _, r in pred_df.iterrows():
        rid = str(r.get("race_id", ""))
        course = str(r.get("course", ""))
        rnum = str(r.get("race_num", ""))
        rname = str(r.get("race_name", ""))
        nh = str(r.get("num_horses", ""))

        result = None
        for attempt in range(args.retry + 1):
            try:
                result = fetch_race_result(rid)
                if result and result.get("trio_nums"):
                    break
            except Exception as e:
                print(f"  [{rid}] attempt {attempt+1} exception: {e}")
            if attempt < args.retry:
                time.sleep(2.0)

        row = {
            "race_id": rid, "course": course, "race_num": rnum,
            "race_name": rname, "num_horses": nh,
            "finish_1": "", "finish_2": "", "finish_3": "",
            "trio_nums": "", "umaren_nums": "",
            "payout_tansho": 0, "payout_umaren": 0, "payout_trio": 0, "payout_wide": 0,
            "fetch_status": "fail",
        }

        if result and result.get("trio_nums"):
            n_ok += 1
            trio = result["trio_nums"]
            uma = result.get("umaren_nums") or []
            row["finish_1"] = trio[0] if len(trio) >= 1 else ""
            row["finish_2"] = trio[1] if len(trio) >= 2 else ""
            row["finish_3"] = trio[2] if len(trio) >= 3 else ""
            row["trio_nums"] = "-".join(str(n) for n in trio)
            row["umaren_nums"] = "-".join(str(n) for n in uma)
            payouts = result.get("payouts", {})
            row["payout_tansho"] = payouts.get("tansho", 0)
            row["payout_umaren"] = payouts.get("umaren", 0)
            row["payout_trio"] = payouts.get("trio", 0)
            row["payout_wide"] = payouts.get("wide", 0)
            row["fetch_status"] = "ok"
            print(f"  [OK] {course} R{rnum} {rname}: {row['trio_nums']} (trio ¥{row['payout_trio']:,})")
        else:
            n_fail += 1
            print(f"  [FAIL] {course} R{rnum} {rname}: no result")

        rows.append(row)
        time.sleep(args.sleep)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== summary ===")
    print(f"  total: {len(rows)}, ok: {n_ok}, fail: {n_fail}")
    print(f"  out: {out_csv.relative_to(BASE)}")


if __name__ == "__main__":
    main()
