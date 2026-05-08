"""V20 interaction features (Session #57 B).

10 件の interaction (組み合わせ) features を expanding window + Bayesian smoothing で計算。
当該レース除外 (cumsum-current pattern) でリーク完全防止。

input:
  data/_v15_train_df_cache.pkl ('df' に 145 V15 features + raw cols)

output:
  data/v20/interaction_features.csv (race_id + horse_id + 10 interaction features)

Usage:
  python tools/v20_interaction_features.py [--alpha-scale 1.0]

Session #57 B (2026-05-09).
"""
import sys
import os
import io
import time
import argparse
import pickle
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


INTERACTION_SPEC = [
    # name, [keys], alpha (Bayesian smoothing prior weight)
    ("int_horse_jockey_top3r",   ["horse_id", "jockey_id"],          3),
    ("int_jockey_course_top3r",  ["jockey_id", "course_enc"],        10),
    ("int_jockey_distcat_top3r", ["jockey_id", "dist_cat"],          10),
    ("int_jockey_baba_top3r",    ["jockey_id", "condition_enc"],     5),
    ("int_jockey_class_top3r",   ["jockey_id", "class_code"],        5),
    ("int_trainer_course_top3r", ["trainer_id", "course_enc"],       10),
    ("int_sire_course_top3r",    ["sire_enc", "course_enc"],         30),
    ("int_sire_distcat_top3r",   ["sire_enc", "dist_cat"],           30),
    ("int_sire_baba_top3r",      ["sire_enc", "condition_enc"],      20),
    ("int_jockey_trainer_top3r", ["jockey_id", "trainer_id"],        5),
]


def ensure_dist_cat(df: pd.DataFrame) -> pd.DataFrame:
    if "dist_cat" in df.columns:
        return df
    bins = [0, 1300, 1600, 2000, 2400, 9999]
    df["dist_cat"] = pd.cut(df["distance"], bins=bins, labels=False, right=False).astype("Int64")
    df["dist_cat"] = df["dist_cat"].fillna(2).astype(int)
    return df


def compute_interaction(df: pd.DataFrame, name: str, keys: list, alpha: float, prior: float) -> pd.Series:
    """Expanding-window top3 rate with Bayesian smoothing for a (multi-key) groupby.

    Returns a Series aligned with df.index.
    """
    # cumsum-current pattern: sum of past hits, count of past races BEFORE current
    grp = df.groupby(keys, sort=False)
    cum_sum = grp["is_top3"].cumsum() - df["is_top3"]
    cum_cnt = grp.cumcount()  # 0-indexed: number of past races

    feat = (cum_sum + alpha * prior) / (cum_cnt + alpha)
    return feat.astype("float32")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha-scale", type=float, default=1.0,
                    help="multiplier for alpha (shrinkage strength). default 1.0")
    ap.add_argument("--cache",
                    default="data/_v15_train_df_cache.pkl",
                    help="V15 train cache path")
    ap.add_argument("--out",
                    default="data/v20/interaction_features.csv",
                    help="output CSV path")
    args = ap.parse_args()

    print("=" * 60)
    print("V20 interaction features (Session #57 B)")
    print(f"alpha-scale: {args.alpha_scale}")
    print("=" * 60)

    # === load cache ===
    t0 = time.time()
    print(f"\nLoading {args.cache} ...")
    with open(args.cache, "rb") as f:
        d = pickle.load(f)
    df = d["df"]
    print(f"  {len(df):,} rows, {df.shape[1]} cols, {time.time()-t0:.1f}s")

    # === ensure target / sort cols ===
    if "is_top3" not in df.columns:
        df["is_top3"] = (df["finish"] <= 3).astype(int)
    if "date_num" not in df.columns:
        raise SystemExit("date_num required for expanding window")

    # required keys
    needed_cols = {"horse_id", "jockey_id", "trainer_id", "sire_enc",
                   "course_enc", "condition_enc", "class_code", "distance",
                   "race_id", "date_num"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing cols: {missing}")

    df = ensure_dist_cat(df)

    # sort by date for expanding (stable to keep within-race order deterministic)
    print("\nSorting by date_num ...")
    df = df.sort_values(["date_num", "race_id"], kind="mergesort").reset_index(drop=True)

    prior = float(df["is_top3"].mean())
    print(f"global is_top3 prior: {prior:.4f}")

    # === compute each interaction ===
    feats_out = {}
    for name, keys, alpha_base in INTERACTION_SPEC:
        alpha = alpha_base * args.alpha_scale
        ts = time.time()
        # ensure keys exist & are non-null
        sub = df[keys + ["is_top3"]].copy()
        # cast keys to nullable-safe types
        for k in keys:
            if sub[k].dtype.kind not in ("i", "u", "f"):
                # encode object/string keys to category codes
                sub[k] = sub[k].astype("category").cat.codes.astype("int32")
            else:
                # fill NaN
                sub[k] = pd.to_numeric(sub[k], errors="coerce").fillna(-1).astype("int64")
        sub["is_top3"] = sub["is_top3"].astype("int8")
        # compute expanding
        cum_sum = sub.groupby(keys, sort=False)["is_top3"].cumsum() - sub["is_top3"]
        cum_cnt = sub.groupby(keys, sort=False).cumcount()
        feat = (cum_sum + alpha * prior) / (cum_cnt + alpha)
        feats_out[name] = feat.astype("float32").values

        nonzero = int((feats_out[name] > 0).sum())
        mean = float(feats_out[name].mean())
        std = float(feats_out[name].std())
        print(f"  [{name}] alpha={alpha:.1f}, mean={mean:.4f}, std={std:.4f}, "
              f"nonzero={nonzero:,}/{len(df):,}, {time.time()-ts:.1f}s")

    # === assemble output ===
    out = pd.DataFrame({
        "race_id": df["race_id"].astype(str).values,
        "horse_id": df["horse_id"].astype(str).values,
    })
    for name, _, _ in INTERACTION_SPEC:
        out[name] = feats_out[name]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    sz_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"\nSaved {args.out} ({len(out):,} rows, {sz_mb:.1f} MB)")
    print(f"Total: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
