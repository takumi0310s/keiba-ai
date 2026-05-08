"""sib_expanding 4 variant 探索 (Session #42 F).

Session #41 D で sib_top3_rate_exp / sib_shinba_wr_exp の単純 expanding 版を実装。
本 Session で 4 variant を試作:

a) window size 最適化 (3 / 5 / 10 / 全期間)
b) weight decay (直近重視 vs 全期間均等)
c) 性別・距離別 細分化
d) JRA-NAR 横断 (本 Session では JRA only に絞り、 NAR 統合は Phase 3 後半 6/9-13)

各 variant の出力を data/netkeiba_siblings_expanding_<variant>.csv に保存。
Phase 3 (5/24+) で実 retro 比較し、 best variant を V18/V19 v2 に統合。

usage:
  python tools/sib_expanding_variants.py --variant a
  python tools/sib_expanding_variants.py --variant b --decay 0.95
  python tools/sib_expanding_variants.py --variant c

V15 production 完全不変 (新規 csv 出力のみ)。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")


def load_blood_and_races():
    print(f"[sib_var] loading blood ...")
    blood = pd.read_csv(
        BASE / "data" / "blood_full.csv",
        encoding="utf-8",
        usecols=["horse_id", "mother"],
    )
    blood["horse_id"] = blood["horse_id"].astype(str).str.strip()
    blood["mother"] = blood["mother"].fillna("").astype(str).str.strip()
    blood = blood[blood["mother"] != ""]
    print(f"  blood: {len(blood):,}, mothers: {blood['mother'].nunique():,}")

    print(f"[sib_var] loading races ...")
    races = pd.read_csv(
        BASE / "data" / "jra_races_full.csv",
        encoding="utf-8",
        usecols=[
            "year", "month", "day", "horse_id", "race_id",
            "finish", "class_code", "sex", "distance",
        ],
        low_memory=False,
    )
    def _norm_id(x):
        try: return str(int(float(str(x).strip())))
        except: return str(x).strip()
    races["horse_id"] = races["horse_id"].apply(_norm_id)
    races["finish"] = pd.to_numeric(races["finish"], errors="coerce")
    races["class_code"] = pd.to_numeric(races["class_code"], errors="coerce")
    races["distance"] = pd.to_numeric(races["distance"], errors="coerce")
    races = races.dropna(subset=["finish", "horse_id"])

    races["year_int"] = pd.to_numeric(races["year"], errors="coerce")
    races["year_full"] = races["year_int"].apply(
        lambda y: 2000 + int(y) if pd.notna(y) and int(y) <= 30 else None
    )
    races = races.dropna(subset=["year_full"])
    races["date"] = pd.to_datetime(
        races["year_full"].astype(int).astype(str)
        + "-" + races["month"].astype(str).str.zfill(2)
        + "-" + races["day"].astype(str).str.zfill(2),
        errors="coerce",
    )
    races = races.dropna(subset=["date"])

    # blood merge
    races = races.merge(blood[["horse_id", "mother"]], on="horse_id", how="left")
    races = races[races["mother"].notna() & (races["mother"] != "")]
    print(f"  races with mother: {len(races):,}")
    return races.sort_values(["mother", "date", "race_id"]).reset_index(drop=True)


# ===== Variant A: window size limited =====

def variant_a_window(races: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """過去 N 走 window で expanding (default 5、 Session #41 D は無制限)."""
    print(f"\n[variant A] window={window}走 限定")
    races = races.copy()
    races["is_top3"] = (races["finish"] <= 3).astype(int)
    races["is_shinba_win"] = ((races["class_code"] == 15) & (races["finish"] == 1)).astype(int)
    races["is_shinba"] = (races["class_code"] == 15).astype(int)

    grp = races.groupby("mother", sort=False)

    # rolling window: 過去 window 走 (current 含まず)
    races["mother_recent_top3"] = grp["is_top3"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )
    races["mother_recent_races"] = grp["is_top3"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).count()
    )
    races["mother_recent_shinba_win"] = grp["is_shinba_win"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )
    races["mother_recent_shinba_runs"] = grp["is_shinba"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )

    alpha = 5.0
    races[f"sib_top3_rate_exp_w{window}"] = (
        (races["mother_recent_top3"].fillna(0) + alpha * 0.30)
        / (races["mother_recent_races"].fillna(0) + alpha)
    )
    races[f"sib_shinba_wr_exp_w{window}"] = (
        (races["mother_recent_shinba_win"].fillna(0) + alpha * 0.10)
        / (races["mother_recent_shinba_runs"].fillna(0) + alpha)
    )
    return races[["race_id", "horse_id", "mother",
                  f"sib_top3_rate_exp_w{window}",
                  f"sib_shinba_wr_exp_w{window}"]]


# ===== Variant B: exponential weight decay =====

def variant_b_decay(races: pd.DataFrame, decay: float = 0.95) -> pd.DataFrame:
    """直近重視 (exponential decay)、 古いレースは weight decay^N."""
    print(f"\n[variant B] decay={decay} (直近重視)")
    races = races.copy()
    races["is_top3"] = (races["finish"] <= 3).astype(int)
    races["is_shinba_win"] = ((races["class_code"] == 15) & (races["finish"] == 1)).astype(int)
    races["is_shinba"] = (races["class_code"] == 15).astype(int)

    out_top3 = []
    out_shinba = []
    out_runs = []

    # mother 単位 で expanding decay weighted sum
    for mother, grp_df in races.groupby("mother", sort=False):
        n = len(grp_df)
        # decay weights: w_i = decay^(n - i - 1)、 直近 n-1 番目が w=1、 0 番目が w=decay^(n-1)
        # current row 除外: shift(1) と同等処理
        cum_top3 = 0.0
        cum_shinba = 0.0
        cum_runs = 0.0
        for i, (idx, row) in enumerate(grp_df.iterrows()):
            # current は自身を除外
            out_top3.append(cum_top3)
            out_shinba.append(cum_shinba)
            out_runs.append(cum_runs)
            # update for next iteration
            cum_top3 = cum_top3 * decay + row["is_top3"]
            cum_shinba = cum_shinba * decay + row["is_shinba_win"]
            cum_runs = cum_runs * decay + 1

    # races は groupby loop の順序と一致しないため、 sort されたまま処理
    sorted_races = races.copy()
    sorted_races[f"sib_top3_decay{decay}"] = out_top3
    sorted_races[f"sib_shinba_decay{decay}"] = out_shinba
    sorted_races[f"sib_runs_decay{decay}"] = out_runs

    alpha = 5.0
    sorted_races[f"sib_top3_rate_exp_d{decay}"] = (
        (sorted_races[f"sib_top3_decay{decay}"] + alpha * 0.30)
        / (sorted_races[f"sib_runs_decay{decay}"] + alpha)
    )
    sorted_races[f"sib_shinba_wr_exp_d{decay}"] = (
        (sorted_races[f"sib_shinba_decay{decay}"] + alpha * 0.10)
        / (sorted_races[f"sib_runs_decay{decay}"] + alpha)
    )
    return sorted_races[[
        "race_id", "horse_id", "mother",
        f"sib_top3_rate_exp_d{decay}",
        f"sib_shinba_wr_exp_d{decay}",
    ]]


# ===== Variant C: 性別・距離別細分化 =====

def variant_c_segmented(races: pd.DataFrame) -> pd.DataFrame:
    """性別・距離 bucket で別々に expanding 集計."""
    print(f"\n[variant C] 性別・距離 細分化")
    races = races.copy()
    races["is_top3"] = (races["finish"] <= 3).astype(int)

    # 性別: 牡=0, 牝=1, セン=2 (簡易)
    sex_map = {"牡": 0, "牝": 1, "セン": 2}
    races["sex_code"] = races["sex"].map(sex_map).fillna(0).astype(int)
    # 距離 bucket: 短=<1500, 中=1500-2000, 長=>2000
    races["dist_bucket"] = pd.cut(
        races["distance"], bins=[0, 1500, 2000, 99999],
        labels=[0, 1, 2], include_lowest=True
    ).astype(int)

    # mother × sex × dist で expanding
    races_sorted = races.sort_values(["mother", "sex_code", "dist_bucket", "date"])
    grp = races_sorted.groupby(["mother", "sex_code", "dist_bucket"], sort=False)
    races_sorted["mother_seg_cum_top3"] = grp["is_top3"].cumsum() - races_sorted["is_top3"]
    races_sorted["mother_seg_cum_races"] = grp.cumcount()

    alpha = 5.0
    races_sorted["sib_top3_rate_exp_seg"] = (
        (races_sorted["mother_seg_cum_top3"] + alpha * 0.30)
        / (races_sorted["mother_seg_cum_races"] + alpha)
    )
    return races_sorted[[
        "race_id", "horse_id", "mother", "sex_code", "dist_bucket",
        "sib_top3_rate_exp_seg",
    ]]


def main():
    p = argparse.ArgumentParser(description="sib_expanding 4 variant (Session #42 F)")
    p.add_argument("--variant", default="all", choices=["all", "a", "b", "c"])
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--decay", type=float, default=0.95)
    p.add_argument("--out-dir", default="data")
    args = p.parse_args()

    t0 = time.time()
    races = load_blood_and_races()
    print(f"[sib_var] data load: {time.time()-t0:.1f}s")

    out_dir = BASE / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.variant in ("all", "a"):
        for w in [3, 5, 10]:
            df = variant_a_window(races, window=w)
            out_path = out_dir / f"netkeiba_siblings_expanding_w{w}.csv"
            df.to_csv(out_path, index=False, encoding="utf-8")
            print(f"  written: {out_path.relative_to(BASE)} ({len(df):,} rows)")

    if args.variant in ("all", "b"):
        for d in [0.90, 0.95, 0.98]:
            df = variant_b_decay(races, decay=d)
            out_path = out_dir / f"netkeiba_siblings_expanding_decay{d}.csv"
            df.to_csv(out_path, index=False, encoding="utf-8")
            print(f"  written: {out_path.relative_to(BASE)} ({len(df):,} rows)")

    if args.variant in ("all", "c"):
        df = variant_c_segmented(races)
        out_path = out_dir / "netkeiba_siblings_expanding_segmented.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  written: {out_path.relative_to(BASE)} ({len(df):,} rows)")

    print(f"\n  total elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
