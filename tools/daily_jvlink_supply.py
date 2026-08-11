# -*- coding: utf-8 -*-
"""JV-Link 日次供給ジョブ (2026-08-12 第1弾-2。JRDB daily_jrdb_supply と同構造)。

  1. jv_daily_fetch.ps1 (SysWOW64 32bit) : RACE(SE/HR/RA/O1-O6等)+SLOP+WOOD+DIFF を
     checkpoint 差分で生保存 (data/jvlink/daily/*.dat)
  2. jv_daily_parse.py : record type 別に CSV へ append+dedup (冪等・raw恒久保存)
  3. 供給ヘルス JSON → data/T1v2_audit/jv_health_<date>.json (T1v2 --source-check が参照)

usage: python tools/daily_jvlink_supply.py [--skip-fetch]
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
PS32 = r"C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe"
AUDIT = os.path.join(BASE, "data", "T1v2_audit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true")
    a = ap.parse_args()
    t0 = datetime.now()
    ok = True
    if not a.skip_fetch:
        r = subprocess.run([PS32, "-NoProfile", "-ExecutionPolicy", "Bypass",
                            "-File", os.path.join(BASE, "tools", "jv_daily_fetch.ps1")],
                           cwd=BASE, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=1800)
        print("\n".join((r.stdout or "").strip().splitlines()[-8:]))
        ok &= (r.returncode == 0)
    r2 = subprocess.run([sys.executable, os.path.join(BASE, "tools", "jv_daily_parse.py")],
                        cwd=BASE, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=1800)
    print("\n".join((r2.stdout or "").strip().splitlines()[-12:]))
    ok &= (r2.returncode == 0)

    # 供給ヘルス
    import pandas as pd
    h = {"date": datetime.now().strftime("%Y%m%d"), "checked_at": datetime.now().isoformat()}
    for name, col in [("se", "_event_date"), ("hr", "_event_date"), ("hc", "train_date")]:
        p = os.path.join(BASE, "data", "jvlink", f"jv_{name}.csv")
        try:
            d = pd.read_csv(p, usecols=[col], dtype=str, encoding="utf-8-sig")
            h[f"{name}_latest"] = str(d[col].max())
            h[f"{name}_rows"] = int(len(d))
        except Exception as e:
            h[f"{name}_error"] = str(e)[:80]
    os.makedirs(AUDIT, exist_ok=True)
    json.dump(h, open(os.path.join(AUDIT, f"jv_health_{h['date']}.json"), "w",
                      encoding="utf-8"), ensure_ascii=False)
    print(f"jv health: {json.dumps(h, ensure_ascii=False)}")
    print(f"Done in {(datetime.now()-t0).total_seconds():.0f}s ({'ALL OK' if ok else 'SOME FAILED'})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
