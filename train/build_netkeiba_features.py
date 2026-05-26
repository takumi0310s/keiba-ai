"""Build netkeiba-sourced features for V20.

Features:
  track_index_nk: Race-level track bias from netkeiba_track_bias.csv
                  Complementary to JRDB jrdb_tb_homestr_inner (corr=0.36).
                  Coverage: 2020-2025 (52% fill in V15 cache).

Output: data/v20/netkeiba_race_features_v15cache.parquet
  Index: original V15 cache index
  Columns: track_index_nk

Usage:
  python train/build_netkeiba_features.py
"""
from __future__ import annotations
import sys, os, time, pickle
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIAS_CSV  = os.path.join(BASE, "data", "netkeiba_track_bias.csv")
CACHE_PKL = os.path.join(BASE, "data", "_v15_train_df_cache.pkl")
OUT_DIR   = os.path.join(BASE, "data", "v20")
OUT_FILE  = os.path.join(OUT_DIR, "netkeiba_race_features_v15cache.parquet")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE, "train"))
from v15_1_features import netkeiba_to_jrdb_internal


def main():
    # ===== 1. Load track_bias =====
    print("Loading netkeiba_track_bias.csv ...")
    t0 = time.time()
    tb = pd.read_csv(BIAS_CSV)
    tb["race_id"] = tb["race_id"].astype(str)
    tb["jrdb_id"] = tb["race_id"].apply(netkeiba_to_jrdb_internal)
    tb_valid = tb[tb["jrdb_id"] != ""].copy()
    # One value per race (take first occurrence)
    tb_race = tb_valid.groupby("jrdb_id")["track_index"].first().reset_index()
    tb_race.columns = ["race_id_str", "track_index_nk"]
    print(f"  track_bias unique races: {len(tb_race)}, elapsed: {time.time()-t0:.1f}s")
    print(f"  track_index_nk: mean={tb_race['track_index_nk'].mean():.2f}, "
          f"std={tb_race['track_index_nk'].std():.2f}, "
          f"min={tb_race['track_index_nk'].min():.0f}, max={tb_race['track_index_nk'].max():.0f}")

    # ===== 2. Load V15 cache =====
    print("\nLoading V15 cache ...")
    t0 = time.time()
    with open(CACHE_PKL, "rb") as f:
        d = pickle.load(f)
    df = d["df"].copy()
    df["_orig_idx"] = df.index
    df["race_id_str"] = df["race_id_str"].astype(str)
    print(f"  shape={df.shape}, elapsed={time.time()-t0:.0f}s")

    # ===== 3. Merge =====
    print("\nMerging track_bias ...")
    merged = df[["_orig_idx", "race_id_str"]].merge(tb_race, on="race_id_str", how="left")
    fill_rate = merged["track_index_nk"].notna().mean()
    print(f"  fill rate: {fill_rate:.1%}")

    # Fill NaN with global mean (so model learns "missing = neutral bias")
    global_mean = merged["track_index_nk"].mean()
    merged["track_index_nk"] = merged["track_index_nk"].fillna(global_mean)

    # ===== 4. Coverage by year =====
    df["_y"] = df["year"].apply(lambda y: 2000 + int(y) if int(y) <= 30 else 1900 + int(y))
    merged["_y"] = df["_y"].values
    print("\nCoverage (actual value, not filled) by year:")
    tb_set = set(tb_race["race_id_str"].values)
    for yr in sorted(df["_y"].unique()):
        sub = df[df["_y"] == yr]
        has = sub["race_id_str"].isin(tb_set).sum()
        print(f"  {yr}: {has}/{len(sub)} ({has/len(sub)*100:.0f}%)")

    # ===== 5. Save =====
    result = merged.set_index("_orig_idx")[["track_index_nk"]].sort_index()
    result.to_parquet(OUT_FILE)
    size_mb = os.path.getsize(OUT_FILE) / 1e6
    print(f"\n[SAVED] {OUT_FILE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
