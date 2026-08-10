# -*- coding: utf-8 -*-
"""騎手勝率マスタ更新 — JRDB版 (2026-08-11 供給復旧 item2)。

旧 update_jockey_wr.py は netkeiba プロフィールスクレイプ (解約方針に反する +
mojibake でクラッシュ)。本スクリプトは JRDB のみで再構築:
  - jrdb_sed.csv (直近365日の着順) × jrdb_paci.csv (jockey_name) → 勝率
  - 出力: jockey_wr.json (旧フォーマット {騎手名: 勝率} 互換)
  - 旧エントリで新集計に無い騎手名は保持 (netkeiba表記名の解決を維持)
"""
import json, os, sys
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

OUT = os.path.join(BASE, "jockey_wr.json")
STAMP = os.path.join(BASE, "jockey_wr_updated.txt")
MIN_RIDES = 30  # 最低騎乗数


def main():
    cutoff = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    sed = pd.read_csv(os.path.join(BASE, "data", "jrdb_sed.csv"),
                      usecols=["race_id", "umaban", "finish", "jockey_code", "yyyymmdd"],
                      dtype=str, encoding="utf-8-sig")
    sed = sed[(sed["yyyymmdd"] >= cutoff)]
    sed["win"] = (pd.to_numeric(sed["finish"], errors="coerce") == 1).astype(int)
    print(f"SED 直近365日: {len(sed):,}騎乗 (cutoff {cutoff})")

    paci = pd.read_csv(os.path.join(BASE, "data", "jrdb_paci.csv"),
                       usecols=["jockey_code", "jockey_name"], dtype=str, encoding="utf-8-sig")
    paci["jockey_name"] = paci["jockey_name"].astype(str).str.strip()
    name_map = (paci[paci["jockey_name"] != ""]
                .drop_duplicates("jockey_code", keep="last")
                .set_index("jockey_code")["jockey_name"].to_dict())
    print(f"騎手コード→名 map: {len(name_map):,}")

    g = sed.groupby("jockey_code").agg(rides=("win", "size"), wins=("win", "sum"))
    g = g[g["rides"] >= MIN_RIDES]
    new_wr = {}
    for code, row in g.iterrows():
        nm = name_map.get(code)
        if not nm or nm == "nan":
            continue
        new_wr[nm] = round(float(row["wins"] / row["rides"]), 6)
    print(f"新集計 (JRDB, 騎乗{MIN_RIDES}+): {len(new_wr)}騎手")

    old = {}
    if os.path.exists(OUT):
        try:
            old = json.load(open(OUT, encoding="utf-8"))
        except Exception:
            pass
    merged = dict(old)
    merged.update(new_wr)  # JRDB 新値を優先、旧のみの表記名は保持
    json.dump(merged, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(STAMP, "w", encoding="utf-8").write(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " (JRDB SED/paci)")
    print(f"保存: jockey_wr.json {len(old)} → {len(merged)}騎手 (JRDB更新 {len(new_wr)})")
    top = sorted(new_wr.items(), key=lambda kv: -kv[1])[:5]
    print("勝率上位:", [(n, f"{w:.3f}") for n, w in top])
    return 0


if __name__ == "__main__":
    sys.exit(main())
