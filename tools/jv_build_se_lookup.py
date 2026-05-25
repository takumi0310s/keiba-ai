"""Build prev_race_last3f lookup from bulk SE TSV data.

Input:  data/jvlink/se_bulk/se_bulk_YYYYMMDD.tsv (output of jv_bulk_se_fetch.ps1)
Output: data/jvlink/se_agari3f_lookup.parquet
        index: (horse_id, race_date)
        cols:  agari3f (seconds, float)
               finish_time (seconds, float)

Usage:
  python tools/jv_build_se_lookup.py
  python tools/jv_build_se_lookup.py --input data/jvlink/se_bulk/se_bulk_20200101.tsv
"""
import argparse
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent


def build_lookup(tsv_paths: list[Path], out_path: Path) -> None:
    import pandas as pd

    dfs = []
    for p in tsv_paths:
        df = pd.read_csv(p, sep="\t", dtype=str)
        # rename tab header bug (tsv has backtick in header from script)
        df.columns = [c.replace("`", "").strip() for c in df.columns]
        dfs.append(df)
        print(f"  Loaded {len(df):>8,} rows from {p.name}")

    data = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(data):,}")

    # Convert numeric cols
    data["agari3f"] = pd.to_numeric(data["agari3f"], errors="coerce")
    data["finish_time"] = pd.to_numeric(data["finish_time"], errors="coerce")
    data["race_date"] = data["race_date"].str.strip()
    data["horse_id"] = data["horse_id"].str.strip()

    # Drop rows without valid agari3f
    valid = data.dropna(subset=["agari3f"])
    print(f"Valid agari3f rows: {len(valid):,} ({100*len(valid)/max(len(data),1):.1f}%)")

    # Keep the most recent record per (horse_id, race_date) in case of duplicates
    result = (
        valid
        .sort_values("race_date")
        .drop_duplicates(subset=["horse_id", "race_date"], keep="last")
        [["horse_id", "race_date", "agari3f", "finish_time"]]
    )
    result = result.set_index(["horse_id", "race_date"])
    print(f"Unique (horse_id, race_date) pairs: {len(result):,}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, engine="pyarrow")
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="*", help="TSV input file(s) (default: all in se_bulk/)")
    parser.add_argument("--out", default="data/jvlink/se_agari3f_lookup.parquet")
    args = parser.parse_args()

    if args.input:
        tsv_paths = [Path(p) for p in args.input]
    else:
        bulk_dir = BASE / "data/jvlink/se_bulk"
        tsv_paths = sorted(bulk_dir.glob("se_bulk_*.tsv"))
        if not tsv_paths:
            print(f"No TSV files found in {bulk_dir}. Run jv_bulk_se_fetch.ps1 first.")
            sys.exit(1)

    out_path = BASE / args.out
    print(f"Building SE lookup from {len(tsv_paths)} file(s) → {out_path}")
    build_lookup(tsv_paths, out_path)


if __name__ == "__main__":
    main()
