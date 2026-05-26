"""Build sire_shinba_top3r_exp from V15 cache (expanding window).

sire_shinba_top3r was constant 0.22 (static blood CSV) → replace with
expanding window computed from the V15 cache itself (2015-2025).

Logic:
  - "shinba" = age==2 horses
  - For each race row: sire_shinba_top3r_exp = (# prior shinba races by same sire
    where finish<=3) / (# prior shinba races by same sire)
  - "prior" = date_num strictly BEFORE current race date
  - sire_enc==100 (unknown sire) → fill with global mean

Output: data/v20/blod_features_v15cache.parquet
  Index: original V15 cache index
  Columns: sire_shinba_top3r_exp, sire_shinba_runs_exp

Usage:
  python train/build_blod_features.py
"""
from __future__ import annotations
import sys, os, time, pickle
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PKL = os.path.join(BASE, "data", "_v15_train_df_cache.pkl")
OUT_DIR   = os.path.join(BASE, "data", "v20")
OUT_FILE  = os.path.join(OUT_DIR, "blod_features_v15cache.parquet")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # ===== 1. Load V15 cache =====
    print("Loading V15 cache ...")
    t0 = time.time()
    with open(CACHE_PKL, "rb") as f:
        d = pickle.load(f)
    df = d["df"].copy()
    df["_orig_idx"] = df.index
    df["_y"] = df["year"].apply(lambda y: 2000 + int(y) if int(y) <= 30 else 1900 + int(y))
    print(f"  shape={df.shape}, elapsed={time.time()-t0:.0f}s")

    # ===== 2. Build shinba events (age==2, known sire) =====
    print("\nBuilding shinba events ...")
    shinba = df[df["age"] == 2].copy()
    shinba["is_top3"] = (shinba["finish"] <= 3).astype(int)
    print(f"  total shinba rows: {len(shinba)}")
    print(f"  known sire (enc!=100): {(shinba['sire_enc'] != 100).sum()}")

    # Aggregate by (sire_enc, date_num) — multiple horses same day possible
    daily = (
        shinba.groupby(["sire_enc", "date_num"])
        .agg(top3_count=("is_top3", "sum"), total_count=("is_top3", "count"))
        .reset_index()
        .sort_values(["sire_enc", "date_num"])
    )

    # Expanding cumsum, shifted so we exclude current date's races
    daily["cum_top3"] = daily.groupby("sire_enc")["top3_count"].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )
    daily["cum_count"] = daily.groupby("sire_enc")["total_count"].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )
    # Rate: prior shinba top3 rate for this sire as of this date
    daily["sire_shinba_top3r_exp"] = (
        daily["cum_top3"] / daily["cum_count"].clip(lower=1)
    )
    # When cum_count==0 (first shinba race for this sire): rate = NaN → fill with 0.22 (V15 constant)
    daily.loc[daily["cum_count"] == 0, "sire_shinba_top3r_exp"] = np.nan

    print(f"  daily events: {len(daily)}")
    valid = daily[daily["cum_count"] > 0]["sire_shinba_top3r_exp"]
    print(f"  sire_shinba_top3r_exp: mean={valid.mean():.3f}, std={valid.std():.3f}")

    # ===== 3. Merge into full df via merge_asof (per-sire, as-of date) =====
    print("\nMerging into full df via merge_asof ...")
    t0 = time.time()

    df_key = df[["_orig_idx", "sire_enc", "date_num"]].copy().sort_values("date_num")
    daily_lookup = daily[["sire_enc", "date_num", "sire_shinba_top3r_exp", "cum_count"]].sort_values("date_num")

    merged = pd.merge_asof(
        df_key,
        daily_lookup.rename(columns={"cum_count": "sire_shinba_runs_exp"}),
        on="date_num",
        by="sire_enc",
        direction="backward",  # latest entry <= date_num
    )
    print(f"  merged shape: {merged.shape}, elapsed: {time.time()-t0:.1f}s")

    # Global mean for fill (NaN = no prior shinba from this sire or unknown sire)
    global_mean = valid.mean()
    print(f"  global_mean (fallback): {global_mean:.3f}")
    merged["sire_shinba_top3r_exp"] = merged["sire_shinba_top3r_exp"].fillna(global_mean)
    merged["sire_shinba_runs_exp"] = merged["sire_shinba_runs_exp"].fillna(0)

    # ===== 4. Coverage by year =====
    merged["_y"] = merged["date_num"].apply(
        lambda d: int(str(d)[:4]) if pd.notna(d) else np.nan
    )
    # Non-trivial: runs_exp > 0 means we have at least 1 prior shinba event
    merged["_has_prior"] = merged["sire_shinba_runs_exp"] > 0
    print("\nCoverage (prior shinba events > 0) by year:")
    for yr in sorted(merged["_y"].dropna().unique().astype(int)):
        sub = merged[merged["_y"] == yr]
        has = sub["_has_prior"].sum()
        print(f"  {yr}: {has}/{len(sub)} ({has/len(sub)*100:.0f}%)")

    # ===== 5. Save =====
    result = merged.set_index("_orig_idx")[
        ["sire_shinba_top3r_exp", "sire_shinba_runs_exp"]
    ].sort_index()
    result.to_parquet(OUT_FILE)
    size_mb = os.path.getsize(OUT_FILE) / 1e6
    print(f"\n[SAVED] {OUT_FILE} ({size_mb:.1f} MB)")
    print(f"  non-NaN top3r: {result['sire_shinba_top3r_exp'].notna().sum():,}")
    print(f"  non-zero runs: {(result['sire_shinba_runs_exp'] > 0).sum():,}")
    print(f"  mean top3r: {result['sire_shinba_top3r_exp'].mean():.4f}")


if __name__ == "__main__":
    main()
