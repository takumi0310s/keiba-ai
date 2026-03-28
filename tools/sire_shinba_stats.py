"""
種牡馬の新馬戦成績を集計するスクリプト
data/jra_races_full.csv から class_code==15 (新馬) のデータを抽出し、
父馬別の成績を計算して data/sire_shinba_stats.csv に保存する。
"""
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def main():
    csv_path = os.path.join(DATA_DIR, "jra_races_full.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    print("Loading jra_races_full.csv ...")
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    print(f"  Total rows: {len(df):,}")

    # Filter: year 15-25 (2015-2025), class_code 15 (新馬)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["class_code"] = pd.to_numeric(df["class_code"], errors="coerce")
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

    maiden = df[(df["year"] >= 15) & (df["year"] <= 25) & (df["class_code"] == 15)].copy()
    print(f"  Maiden races (class_code==15, 2015-2025): {len(maiden):,} entries")

    # Clean father column
    maiden["father"] = maiden["father"].astype(str).str.strip()
    maiden = maiden[maiden["father"].notna() & (maiden["father"] != "") & (maiden["father"] != "nan")]
    maiden = maiden[maiden["finish"].notna() & (maiden["finish"] > 0)]
    print(f"  After cleaning: {len(maiden):,} entries")

    # Surface flags
    maiden["is_turf"] = maiden["surface"].astype(str).str.contains("芝")
    maiden["is_dirt"] = maiden["surface"].astype(str).str.contains("ダ")

    # Distance categories
    maiden["is_sprint"] = maiden["distance"] <= 1400
    maiden["is_mile"] = maiden["distance"] == 1600
    maiden["is_middle"] = (maiden["distance"] >= 1800) & (maiden["distance"] <= 2000)
    maiden["is_long"] = maiden["distance"] >= 2200

    # Helper: top3 rate for a subset
    def top3_rate(sub):
        if len(sub) == 0:
            return float("nan")
        return (sub["finish"] <= 3).mean()

    # Group by sire
    print("Calculating sire stats ...")
    results = []
    for sire, g in maiden.groupby("father"):
        n = len(g)
        if n < 5:
            continue
        row = {
            "sire": sire,
            "total_runners": n,
            "win_rate": round((g["finish"] == 1).mean(), 4),
            "top2_rate": round((g["finish"] <= 2).mean(), 4),
            "top3_rate": round((g["finish"] <= 3).mean(), 4),
            "avg_finish": round(g["finish"].mean(), 2),
            "turf_top3r": round(top3_rate(g[g["is_turf"]]), 4) if g["is_turf"].any() else float("nan"),
            "dirt_top3r": round(top3_rate(g[g["is_dirt"]]), 4) if g["is_dirt"].any() else float("nan"),
            "sprint_top3r": round(top3_rate(g[g["is_sprint"]]), 4) if g["is_sprint"].any() else float("nan"),
            "mile_top3r": round(top3_rate(g[g["is_mile"]]), 4) if g["is_mile"].any() else float("nan"),
            "middle_top3r": round(top3_rate(g[g["is_middle"]]), 4) if g["is_middle"].any() else float("nan"),
            "long_top3r": round(top3_rate(g[g["is_long"]]), 4) if g["is_long"].any() else float("nan"),
        }
        results.append(row)

    out = pd.DataFrame(results)
    out = out.sort_values("total_runners", ascending=False).reset_index(drop=True)

    # Save
    out_path = os.path.join(DATA_DIR, "sire_shinba_stats.csv")
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(f"  Total sires (>=5 runners): {len(out)}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total maiden entries (2015-2025): {len(maiden):,}")
    print(f"Total sires with >=5 runners:     {len(out)}")
    print(f"Overall win rate:                  {maiden['finish'].eq(1).mean():.4f}")
    print(f"Overall top3 rate:                 {(maiden['finish'] <= 3).mean():.4f}")
    print(f"Overall avg finish:                {maiden['finish'].mean():.2f}")

    # Top 10 by win_rate (min 20 runners)
    top = out[out["total_runners"] >= 20].nlargest(10, "win_rate")
    print(f"\n--- Top 10 Sires by Win Rate (min 20 runners) ---")
    print(f"{'Sire':<24s} {'N':>5s} {'WinR':>6s} {'Top3R':>6s} {'AvgFin':>7s} {'Turf3':>6s} {'Dirt3':>6s}")
    print("-" * 70)
    for _, r in top.iterrows():
        print(f"{r['sire']:<24s} {r['total_runners']:5.0f} {r['win_rate']:6.3f} {r['top3_rate']:6.3f} {r['avg_finish']:7.2f} "
              f"{r['turf_top3r']:6.3f} {r['dirt_top3r']:6.3f}" if pd.notna(r['dirt_top3r']) else
              f"{r['sire']:<24s} {r['total_runners']:5.0f} {r['win_rate']:6.3f} {r['top3_rate']:6.3f} {r['avg_finish']:7.2f} "
              f"{r['turf_top3r']:6.3f}    N/A")

    # Top 10 by total_runners
    top_n = out.nlargest(10, "total_runners")
    print(f"\n--- Top 10 Sires by Runner Count ---")
    print(f"{'Sire':<24s} {'N':>5s} {'WinR':>6s} {'Top3R':>6s} {'AvgFin':>7s}")
    print("-" * 50)
    for _, r in top_n.iterrows():
        print(f"{r['sire']:<24s} {r['total_runners']:5.0f} {r['win_rate']:6.3f} {r['top3_rate']:6.3f} {r['avg_finish']:7.2f}")


if __name__ == "__main__":
    main()
