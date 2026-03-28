"""
Compute sibling (兄弟馬) and mother's production statistics from existing CSV data.
No scraping needed - uses blood_full.csv + jra_races_full.csv.

Output: data/netkeiba_siblings.csv
"""
import pandas as pd
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    t0 = time.time()

    # --- Load blood data (horse_id -> mother mapping) ---
    print("Loading blood_full.csv ...")
    blood = pd.read_csv(
        os.path.join(BASE_DIR, "data", "blood_full.csv"),
        encoding="utf-8",
        usecols=["horse_id", "horse_name", "mother"],
    )
    blood["horse_id"] = blood["horse_id"].astype(str).str.strip()
    blood["mother"] = blood["mother"].str.strip()
    blood["horse_name"] = blood["horse_name"].str.strip()
    print(f"  blood rows: {len(blood):,}, unique mothers: {blood['mother'].nunique():,}")

    # --- Load race results ---
    print("Loading jra_races_full.csv ...")
    races = pd.read_csv(
        os.path.join(BASE_DIR, "data", "jra_races_full.csv"),
        encoding="utf-8",
        usecols=["horse_id", "finish", "class_code"],
    )
    races["horse_id"] = races["horse_id"].astype(str).str.strip()
    races["finish"] = pd.to_numeric(races["finish"], errors="coerce")
    races["class_code"] = pd.to_numeric(races["class_code"], errors="coerce")
    races = races.dropna(subset=["finish"])
    print(f"  race rows (valid finish): {len(races):,}")

    # --- Merge: attach mother to each race row ---
    print("Merging ...")
    merged = races.merge(blood[["horse_id", "mother"]], on="horse_id", how="inner")
    print(f"  merged rows: {len(merged):,}")

    # --- Aggregate per mother ---
    print("Computing per-mother stats ...")

    # Basic stats
    grp = merged.groupby("mother")
    stats = pd.DataFrame()
    stats["total_races"] = grp["finish"].count()
    stats["total_offspring"] = grp["horse_id"].nunique()
    stats["win_rate"] = grp["finish"].apply(lambda s: (s == 1).mean())
    stats["top3_rate"] = grp["finish"].apply(lambda s: (s <= 3).mean())
    stats["avg_finish"] = grp["finish"].mean()
    stats["best_class"] = grp["class_code"].min()

    # Shinba (新馬, class_code==15) win rate
    shinba = merged[merged["class_code"] == 15]
    shinba_grp = shinba.groupby("mother")
    shinba_stats = pd.DataFrame()
    shinba_stats["shinba_races"] = shinba_grp["finish"].count()
    shinba_stats["shinba_win_rate"] = shinba_grp["finish"].apply(lambda s: (s == 1).mean())

    # Join shinba stats
    stats = stats.join(shinba_stats, how="left")
    stats["shinba_races"] = stats["shinba_races"].fillna(0).astype(int)
    stats["shinba_win_rate"] = stats["shinba_win_rate"].fillna(0.0)

    # Filter: total_races >= 3
    stats = stats[stats["total_races"] >= 3].copy()
    stats = stats.reset_index()

    # Round floats
    for col in ["win_rate", "top3_rate", "avg_finish", "shinba_win_rate"]:
        stats[col] = stats[col].round(4)
    stats["best_class"] = stats["best_class"].astype(int)

    # Select final columns (keep shinba_races internally for summary)
    out_cols = [
        "mother", "total_offspring", "total_races", "win_rate", "top3_rate",
        "avg_finish", "best_class", "shinba_win_rate",
    ]
    save_cols = out_cols  # CSV output columns
    stats = stats.sort_values("total_races", ascending=False)

    # --- Save ---
    out_path = os.path.join(BASE_DIR, "data", "netkeiba_siblings.csv")
    stats[save_cols].to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(f"  Total mothers: {len(stats):,}")
    print(f"  Total rows: {len(stats):,}")

    # --- Summary: top 10 by shinba_win_rate (min 5 shinba races) ---
    top_shinba = stats[stats["shinba_races"] >= 5].nlargest(10, "shinba_win_rate")
    print(f"\n=== Top 10 mothers by shinba_win_rate (min 5 shinba races) ===")
    print(top_shinba[["mother", "total_offspring", "total_races", "shinba_win_rate", "win_rate", "top3_rate"]].to_string(index=False))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
