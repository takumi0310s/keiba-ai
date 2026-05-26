"""Build prev_race_first3f / prev_race_last3f / prev_race_pace_diff from netkeiba_race_lap.csv.

Strategy:
  1. Load netkeiba_race_lap.csv (19,982 unique races, 2020-2025)
  2. Convert to JRDB 8-char race_id_str format
  3. For each horse in V15 cache: sort by date_num, prev_race_id = shift(1)
  4. Join prev_race_id with lap lookup → get real first3f/last3f
  5. Save as data/v20/lap_features_v15cache.parquet for merge in V20 training

Output columns (per V15 cache row index):
  prev_race_first3f_real  (NaN if prev race not in lap table)
  prev_race_last3f_real
  prev_race_pace_diff_real

Usage:
  python train/build_lap_features.py
"""
import sys, os, time, pickle
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "train"))

from v15_1_features import netkeiba_to_jrdb_internal

LAP_CSV   = os.path.join(BASE, "data", "netkeiba_race_lap.csv")
CACHE_PKL = os.path.join(BASE, "data", "_v15_train_df_cache.pkl")
OUT_DIR   = os.path.join(BASE, "data", "v20")
OUT_FILE  = os.path.join(OUT_DIR, "lap_features_v15cache.parquet")

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # ===== 1. Load lap CSV =====
    print("Loading netkeiba_race_lap.csv ...")
    t0 = time.time()
    lap = pd.read_csv(LAP_CSV, encoding="utf-8")
    lap["race_id"] = lap["race_id"].astype(str)
    lap = lap.drop_duplicates("race_id").reset_index(drop=True)
    lap["jrdb_id"] = lap["race_id"].apply(netkeiba_to_jrdb_internal)
    lap = lap[lap["jrdb_id"] != ""]
    lap_lookup = lap.set_index("jrdb_id")[["pace_first_half", "pace_second_half"]].copy()
    lap_lookup.columns = ["prev_race_first3f_real", "prev_race_last3f_real"]
    print(f"  lap rows: {len(lap_lookup)}, elapsed: {time.time()-t0:.1f}s")

    # ===== 2. Load V15 cache =====
    print("\nLoading V15 cache ...")
    t0 = time.time()
    with open(CACHE_PKL, "rb") as f:
        d = pickle.load(f)
    df = d["df"].copy()
    print(f"  shape={df.shape}, elapsed={time.time()-t0:.0f}s")

    # ===== 3. For each horse: sort by date_num, compute prev_race_id_str =====
    print("\nComputing prev_race_id_str ...")
    t0 = time.time()

    df["_orig_idx"] = df.index
    df["race_id_str"] = df["race_id_str"].astype(str)
    df["horse_id"]   = df["horse_id"].astype(str)

    # Sort within horse groups by date_num
    df_sorted = df.sort_values(["horse_id", "date_num"]).copy()

    # prev_race_id_str = previous row's race_id_str within same horse
    df_sorted["prev_race_id_str_computed"] = (
        df_sorted.groupby("horse_id")["race_id_str"]
        .shift(1)
    )
    print(f"  prev computed, elapsed: {time.time()-t0:.1f}s")
    print(f"  prev race available: {df_sorted['prev_race_id_str_computed'].notna().sum():,} / {len(df_sorted):,}")

    # ===== 4. Join with lap lookup =====
    print("\nJoining with lap lookup ...")
    t0 = time.time()

    prev_ids = df_sorted["prev_race_id_str_computed"].values
    first3f = np.full(len(df_sorted), np.nan)
    last3f  = np.full(len(df_sorted), np.nan)

    for i, pid in enumerate(prev_ids):
        if pd.notna(pid) and pid in lap_lookup.index:
            first3f[i] = lap_lookup.at[pid, "prev_race_first3f_real"]
            last3f[i]  = lap_lookup.at[pid, "prev_race_last3f_real"]

    df_sorted["prev_race_first3f_real"] = first3f
    df_sorted["prev_race_last3f_real"]  = last3f
    df_sorted["prev_race_pace_diff_real"] = first3f - last3f

    fill_rate = (~np.isnan(first3f)).mean()
    print(f"  fill rate: {fill_rate:.1%}  ({(~np.isnan(first3f)).sum():,} / {len(df_sorted):,})")
    print(f"  elapsed: {time.time()-t0:.1f}s")

    # ===== 5. Coverage by year =====
    df_sorted["_y"] = df_sorted["year"].apply(lambda y: 2000 + int(y) if int(y) <= 30 else 1900 + int(y))
    print("\nCoverage by year:")
    for yr in sorted(df_sorted["_y"].unique()):
        sub = df_sorted[df_sorted["_y"] == yr]
        filled = sub["prev_race_first3f_real"].notna().sum()
        print(f"  {yr}: {filled}/{len(sub)} ({filled/len(sub)*100:.0f}%)")

    # Stats on real values
    valid = df_sorted["prev_race_first3f_real"].dropna()
    print(f"\nprev_race_first3f_real: mean={valid.mean():.2f}, std={valid.std():.2f}, "
          f"min={valid.min():.1f}, max={valid.max():.1f}")
    valid2 = df_sorted["prev_race_pace_diff_real"].dropna()
    print(f"prev_race_pace_diff_real: mean={valid2.mean():.2f}, std={valid2.std():.2f}, "
          f"min={valid2.min():.1f}, max={valid2.max():.1f}")

    # ===== 6. Save =====
    out_cols = ["_orig_idx", "prev_race_first3f_real", "prev_race_last3f_real",
                "prev_race_pace_diff_real"]
    result = df_sorted[out_cols].set_index("_orig_idx").sort_index()
    result.to_parquet(OUT_FILE)
    print(f"\n[SAVED] {OUT_FILE} ({os.path.getsize(OUT_FILE)/1e6:.1f} MB)")
    print(f"  non-null rows: {result['prev_race_first3f_real'].notna().sum():,}")


if __name__ == "__main__":
    main()
