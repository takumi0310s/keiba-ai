"""V20 expanding features (Session #55 B).

V15 145 features の history-aggregation 系 6 件を window 限定 expanding 化する PoC。
V18/V19 sib_w5 (+0.0689 AUC) と同じ pattern を適用。

期待: +0.005-0.015 AUC (V20 構築の本命戦略)

usage:
    python tools/v20_expanding_features.py --years 2020 2025
    -> data/v20/expanding_features_v1.parquet

絶対遵守:
- main 不変、 V15 model 不変
- jra_races_full.csv は read-only
- dev/v20-expanding 専用
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC = DATA / "jra_races_full.csv"
OUT_DIR = DATA / "v20"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "expanding_features_v1.parquet"


def _make_race_dt(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        df["year"].astype(str)
        + "-"
        + df["month"].astype(str).str.zfill(2)
        + "-"
        + df["day"].astype(str).str.zfill(2),
        errors="coerce",
    )


def _expanding_rolling_rate(
    df: pd.DataFrame,
    *,
    group_col: str,
    target_col: str,
    window: int,
    out_col: str,
    alpha: float = 5.0,
    prior: float = 0.30,
) -> pd.Series:
    """Bayesian-smoothed rolling expanding rate (current row excluded).

    For each row r in group g:
        rate = (sum_prev_N(target) + alpha * prior) / (count_prev_N + alpha)
    where prev_N is the last `window` records in g (current row excluded).

    Implementation: shift(1) then rolling(window) — exclude current row.
    """
    df_sorted = df.sort_values([group_col, "race_dt", "race_id", "umaban"])
    g = df_sorted.groupby(group_col, sort=False)[target_col]
    rolled_sum = g.shift(1).rolling(window=window, min_periods=1).sum()
    rolled_cnt = g.shift(1).rolling(window=window, min_periods=1).count()
    rate = (rolled_sum + alpha * prior) / (rolled_cnt + alpha)
    rate.name = out_col
    return rate.reindex(df.index)


def main(years: tuple[int, int]) -> None:
    print(f"[B] read {SRC}")
    df = pd.read_csv(
        SRC,
        usecols=[
            "year",
            "month",
            "day",
            "race_id",
            "horse_id",
            "jockey_id",
            "trainer_id",
            "umaban",
            "finish",
            "abnormal_code",
            "course",
            "distance",
            "surface",
        ],
        low_memory=False,
    )
    # year は 2-digit (15 = 2015) → 4-digit に正規化
    df["year_full"] = df["year"].apply(lambda y: 2000 + int(y) if int(y) < 80 else 1900 + int(y))
    df = df[(df["year_full"] >= years[0]) & (df["year_full"] <= years[1])].copy()
    print(f"[B] rows after year filter ({years[0]}-{years[1]}): {len(df):,}")

    df["race_dt"] = pd.to_datetime(
        df["year_full"].astype(str)
        + "-"
        + df["month"].astype(str).str.zfill(2)
        + "-"
        + df["day"].astype(str).str.zfill(2),
        errors="coerce",
    )
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce")
    df["target_top3"] = (df["finish"].between(1, 3)).astype(int)
    df["target_win"] = (df["finish"] == 1).astype(int)

    valid = df["abnormal_code"].fillna(0) == 0
    df = df.loc[valid].reset_index(drop=True)
    print(f"[B] rows after abnormal filter: {len(df):,}")

    print("[B] compute 6 expanding features (window 限定)...")

    # 1. jockey_wr_w30 (騎手 直近 30 R 勝率)
    df["jockey_wr_w30"] = _expanding_rolling_rate(
        df,
        group_col="jockey_id",
        target_col="target_win",
        window=30,
        out_col="jockey_wr_w30",
        alpha=5.0,
        prior=0.10,
    )

    # 2. jockey_top3_w30 (騎手 直近 30 R 複勝率)
    df["jockey_top3_w30"] = _expanding_rolling_rate(
        df,
        group_col="jockey_id",
        target_col="target_top3",
        window=30,
        out_col="jockey_top3_w30",
        alpha=5.0,
        prior=0.30,
    )

    # 3. trainer_top3_w90 (調教師 直近 90 R)
    df["trainer_top3_w90"] = _expanding_rolling_rate(
        df,
        group_col="trainer_id",
        target_col="target_top3",
        window=90,
        out_col="trainer_top3_w90",
        alpha=10.0,
        prior=0.30,
    )

    # 4. horse_career_top3_w5 (馬 直近 5 走複勝率)
    df["horse_career_top3_w5"] = _expanding_rolling_rate(
        df,
        group_col="horse_id",
        target_col="target_top3",
        window=5,
        out_col="horse_career_top3_w5",
        alpha=2.0,
        prior=0.30,
    )

    # 5. horse_career_wr_w5 (馬 直近 5 走勝率)
    df["horse_career_wr_w5"] = _expanding_rolling_rate(
        df,
        group_col="horse_id",
        target_col="target_win",
        window=5,
        out_col="horse_career_wr_w5",
        alpha=2.0,
        prior=0.10,
    )

    # 6. horse_career_top3_w10 (馬 直近 10 走複勝率, 中期 trend)
    df["horse_career_top3_w10"] = _expanding_rolling_rate(
        df,
        group_col="horse_id",
        target_col="target_top3",
        window=10,
        out_col="horse_career_top3_w10",
        alpha=3.0,
        prior=0.30,
    )

    df.rename(columns={"year_full": "year_4digit"}, inplace=True)
    out_cols = [
        "race_id",
        "horse_id",
        "umaban",
        "year_4digit",
        "jockey_wr_w30",
        "jockey_top3_w30",
        "trainer_top3_w90",
        "horse_career_top3_w5",
        "horse_career_wr_w5",
        "horse_career_top3_w10",
    ]
    out = df[out_cols].copy()

    print()
    print("[B] coverage (non-NaN rate):")
    for c in out_cols[4:]:
        cov = (1 - out[c].isna().mean()) * 100
        print(f"  {c}: {cov:.1f}%")

    print()
    print("[B] sample stats (mean / std):")
    for c in out_cols[4:]:
        m = out[c].mean()
        s = out[c].std()
        print(f"  {c}: mean={m:.4f} std={s:.4f}")

    print()
    print(f"[B] write {OUT_PATH}")
    out.to_parquet(OUT_PATH, index=False)
    print(f"[B] done, rows={len(out):,}, file_size={OUT_PATH.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs=2, type=int, default=[2020, 2025])
    args = p.parse_args()
    main(tuple(args.years))
