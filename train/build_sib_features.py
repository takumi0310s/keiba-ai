"""Build sib_*_exp features for V20 from netkeiba_siblings_expanding.csv.

Match rate: 100% (race_id + horse_id format matches V15 cache directly).

Output: data/v20/sib_features_v15cache.parquet
  Index: original V15 cache index
  Columns: sib_top3_rate_exp, sib_shinba_wr_exp, sib_total_races_exp, sib_total_offspring_exp

Usage:
  python train/build_sib_features.py
"""
from __future__ import annotations
import sys, os, time, pickle
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIB_CSV  = os.path.join(BASE, "data", "netkeiba_siblings_expanding.csv")
CACHE_PKL = os.path.join(BASE, "data", "_v15_train_df_cache.pkl")
OUT_DIR  = os.path.join(BASE, "data", "v20")
OUT_FILE = os.path.join(OUT_DIR, "sib_features_v15cache.parquet")
os.makedirs(OUT_DIR, exist_ok=True)

SIB_COLS = ["sib_top3_rate_exp", "sib_shinba_wr_exp", "sib_total_races_exp", "sib_total_offspring_exp"]


def main():
    print("Loading sib_expanding.csv ...")
    t0 = time.time()
    sib = pd.read_csv(SIB_CSV, dtype={"race_id": str, "horse_id": str},
                      usecols=["race_id", "horse_id"] + SIB_COLS)
    sib["horse_id"] = sib["horse_id"].apply(
        lambda x: str(int(float(x))).zfill(8) if x.replace(".", "").isdigit() else x
    )
    print(f"  sib rows: {len(sib):,}, elapsed: {time.time()-t0:.1f}s")

    print("\nLoading V15 cache ...")
    t0 = time.time()
    with open(CACHE_PKL, "rb") as f:
        d = pickle.load(f)
    df = d["df"].copy()
    df["_orig_idx"] = df.index
    df["_race_id"] = df["race_id"].astype(str).str.strip()
    df["_horse_id"] = df["horse_id"].astype(str).str.strip().apply(
        lambda x: str(int(float(x))).zfill(8) if x.replace(".", "").isdigit() else x
    )
    print(f"  df shape: {df.shape}, elapsed: {time.time()-t0:.0f}s")

    print("\nMerging ...")
    t0 = time.time()
    merged = df[["_orig_idx", "_race_id", "_horse_id"]].merge(
        sib.rename(columns={"race_id": "_race_id", "horse_id": "_horse_id"}),
        on=["_race_id", "_horse_id"],
        how="left",
    )
    matched = merged[SIB_COLS[0]].notna().sum()
    print(f"  match: {matched:,}/{len(df):,} = {matched/len(df)*100:.1f}%")
    print(f"  elapsed: {time.time()-t0:.1f}s")

    for c in SIB_COLS:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)
        nz = (merged[c] > 0).sum()
        print(f"  {c}: mean={merged[c].mean():.4f}, nonzero={nz:,} ({nz/len(merged)*100:.1f}%)")

    result = merged.set_index("_orig_idx")[SIB_COLS].sort_index()
    result.to_parquet(OUT_FILE)
    size_mb = os.path.getsize(OUT_FILE) / 1e6
    print(f"\n[SAVED] {OUT_FILE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
