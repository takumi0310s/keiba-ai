# -*- coding: utf-8 -*-
"""KYI 単独再構築 (rebuild_driver から分離。scrape_jrdb の stdout/stderr 差し替えと
他モジュールのラッパ GC-close が衝突するため、専用インタプリタで scrape_jrdb のみ import)。
ソース = data/jrdb/extracted/Kyi/KYI*.txt (2020+ = 現行CSVと同フロア)。
安全装置: 現行ヘッダ列整合 ABORT / 履歴縮小 ABORT / 原子的置換。
"""
import glob, os, sys
BASE = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "tools"))
# scrape_jrdb を最初に import させ stdout/stderr の所有権を渡す (以後差し替え無し)
from scrape_jrdb import (parse_fixed_length, KYI_FIELDS,           # noqa: E402
                         jrdb_to_jra_race_id, jrdb_to_netkeiba_race_id)
import pandas as pd  # noqa: E402

EXTRACT = os.path.join(BASE, "data", "jrdb", "extracted", "Kyi")
OUT = os.path.join(BASE, "data", "jrdb_kyi.csv")


def main():
    files = sorted(glob.glob(os.path.join(EXTRACT, "KYI*.txt")))
    files = [f for f in files if os.path.basename(f)[3:5] >= "20"]
    print(f"KYI: {len(files)} files (2020+)")
    dfs, errors = [], 0
    for i, fp in enumerate(files):
        try:
            with open(fp, "rb") as f:
                df = parse_fixed_length(f.read(), KYI_FIELDS)
            df["jra_race_id"] = df.apply(jrdb_to_jra_race_id, axis=1)
            df["nk_race_id"] = df.apply(jrdb_to_netkeiba_race_id, axis=1)
            for c in ["馬名", "放牧先", "入厩年月日"]:
                if c in df.columns:
                    df[c] = df[c].str.strip()
            dfs.append(df)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR {os.path.basename(fp)}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(files)}]")
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["jra_race_id", "馬番"], keep="last")
    print(f"parsed+dedup: {len(df):,} rows ({errors} errors)")

    cur_header = pd.read_csv(OUT, nrows=0, encoding="utf-8-sig").columns.tolist()
    missing = [c for c in cur_header if c not in df.columns]
    if missing:
        print(f"ABORT: 現行列欠落 {missing[:8]}")
        return 1
    cur_rows = sum(1 for _ in open(OUT, encoding="utf-8-sig", errors="replace")) - 1
    if len(df) < cur_rows:
        print(f"ABORT: 縮小 {len(df):,} < {cur_rows:,}")
        return 1
    df = df[cur_header]
    tmp = OUT + ".new"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, OUT)
    print(f"Saved: jrdb_kyi.csv {cur_rows:,} → {len(df):,} rows (+{len(df)-cur_rows:,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
