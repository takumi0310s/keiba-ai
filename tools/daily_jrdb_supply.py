# -*- coding: utf-8 -*-
"""JRDB 日次供給ジョブ (2026-08-11 供給復旧)。

旧チェーンの死因 (v15_autopsy 追補):
  - daily_jrdb_kyi.bat の scrape_jrdb --date TODAY が「0日間」計算で恒常0件
  - 実供給源は金曜 FridayWeekendScrape の週次Paciバンドル → 6/12 を最後に停止 → 全滅
本ジョブは実証済み経路 (datazip 日次zip + 日次lzh直フェッチ + 種別単独フル再構築) に一本化。

フロー (全て subprocess 隔離 — モジュール横断 stdout ラッパ差替の GC-close 地雷対策):
  1. download_jrdb.py            : Kyi/Cyb/Skb/Tyb/Kka/Sed の 2026 日次zip (skip-existing)
  2. jrdb_daily_fix_fetch.py     : KTA/KKA 日次lzh (直近10日窓)
  3. download_parse_jrdb_extra   : JOA 日次 + jrdb_jo.csv 再構築
  4. 再構築: fable_rebuild(tyb/skb) + batch2(srb, skip-download)
            + jrdb_rebuild_driver(sed/cyb/kta/kka/paci) + jrdb_rebuild_kyi
  5. 供給ヘルス JSON → data/T1v2_audit/supply_health_<date>.json (T1v2 --source-check が参照)

usage: python tools/daily_jrdb_supply.py [--skip-download]
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
PY = sys.executable
AUDIT_DIR = os.path.join(BASE, "data", "T1v2_audit")


def run(label, args, timeout=900):
    print(f"\n=== {label} ===", flush=True)
    r = subprocess.run([PY] + args, cwd=BASE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    tail = "\n".join((r.stdout or "").strip().splitlines()[-6:])
    print(tail)
    if r.returncode != 0:
        err = "\n".join((r.stderr or "").strip().splitlines()[-4:])
        print(f"  [FAIL rc={r.returncode}] {err}")
    return r.returncode == 0


def supply_health():
    """kyi/paci/sed の内容鮮度 + 馬名解決率を測る (T1v2 source-check の入力)。"""
    import pandas as pd
    h = {"date": datetime.now().strftime("%Y%m%d"), "checked_at": datetime.now().isoformat()}
    try:
        kyi = pd.read_csv(os.path.join(BASE, "data", "jrdb_kyi.csv"),
                          usecols=["nk_race_id", "馬名"], dtype=str, encoding="utf-8-sig")
        rid = kyi["nk_race_id"].astype(str)
        # nk_race_id は開催回ベースで暦日を持たない → extracted KYI ファイル名で内容鮮度を測る
        kyi_files = sorted(glob.glob(os.path.join(BASE, "data", "jrdb", "extracted", "Kyi", "KYI*.txt")))
        latest_file = os.path.basename(kyi_files[-1])[3:9] if kyi_files else None  # yymmdd
        h["kyi_latest_file"] = f"20{latest_file}" if latest_file else None
        # 最新開催日の KYI 行で馬名解決率
        latest_rids = None
        if latest_file:
            import io
            raw = open(kyi_files[-1], "rb").read()
            h["kyi_latest_rows"] = raw.count(b"\n")
        # 馬名解決率: 2026 行のうち馬名が非空
        r26 = kyi[rid.str.startswith("2026")]
        named = r26["馬名"].astype(str).str.strip().replace("nan", "")
        h["name_resolution_2026"] = round(float((named != "").mean()), 4) if len(r26) else None
        h["kyi_2026_rows"] = int(len(r26))
    except Exception as e:
        h["kyi_error"] = str(e)
    try:
        import pandas as pd
        sed = pd.read_csv(os.path.join(BASE, "data", "jrdb_sed.csv"),
                          usecols=["yyyymmdd"], dtype=str, encoding="utf-8-sig")
        h["sed_latest"] = str(sed["yyyymmdd"].max())
    except Exception as e:
        h["sed_error"] = str(e)
    os.makedirs(AUDIT_DIR, exist_ok=True)
    out = os.path.join(AUDIT_DIR, f"supply_health_{h['date']}.json")
    json.dump(h, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\n=== supply health ===\n{json.dumps(h, ensure_ascii=False, indent=1)}")
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    t0 = datetime.now()
    ok = True
    if not args.skip_download:
        start10 = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
        ok &= run("1. datazip 日次zip (Kyi/Cyb/Skb/Tyb/Kka/Sed 2026)",
                  ["tools/download_jrdb.py", "--types", "Kyi", "Cyb", "Skb", "Tyb", "Kka", "Sed",
                   "--years", "2026"])
        ok &= run("2. KTA/KKA 日次lzh (直近10日)",
                  ["tools/jrdb_daily_fix_fetch.py", "--types", "kta", "kka", "--start", start10])
        ok &= run("3. JOA 日次 + jo 再構築",
                  ["tools/download_parse_jrdb_extra.py", "--types", "jo"])
    ok &= run("4a. rebuild tyb/skb", ["tools/fable_rebuild_type_20260612.py", "--types", "tyb", "skb"])
    ok &= run("4b. rebuild srb", ["tools/download_parse_jrdb_batch2.py", "--types", "srb", "--skip-download"])
    ok &= run("4c. rebuild sed/cyb/kta/kka/paci",
              ["tools/jrdb_rebuild_driver.py", "--types", "sed", "cyb", "kta", "kka", "paci"])
    ok &= run("4d. rebuild kyi", ["tools/jrdb_rebuild_kyi.py"])
    supply_health()
    print(f"\nDone in {(datetime.now()-t0).total_seconds():.0f}s  ({'ALL OK' if ok else 'SOME FAILED'})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
