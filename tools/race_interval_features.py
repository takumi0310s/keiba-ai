"""レース間隔 features (Session #47 B、 dev/sprint2).

前走からの日数 + カテゴリ別 hit rate + 距離 interaction。

カテゴリ:
- 連闘 (1-7 日)
- 中1週 (8-14 日)
- 中2-4週 (15-28 日)
- 中5-8週 (29-56 日)
- 休み明け (57+ 日)

features:
- days_since_prev_race (int)
- interval_category (0-4)
- interval_category × distance interaction
- interval_top3_rate_history (馬の同 category での過去 top3 率、 expanding)

V15 production 完全独立、 dev/sprint2 のみ。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")


INTERVAL_BINS = [0, 7, 14, 28, 56, 999]
INTERVAL_LABELS = ["連闘", "中1週", "中2-4週", "中5-8週", "休み明け"]
INTERVAL_CODES = [0, 1, 2, 3, 4]


def categorize_interval(days: int) -> int:
    """days → category code (0-4)"""
    for i in range(len(INTERVAL_BINS) - 1):
        if INTERVAL_BINS[i] <= days < INTERVAL_BINS[i + 1]:
            return INTERVAL_CODES[i]
    return INTERVAL_CODES[-1]


def compute_interval_features(history_df: pd.DataFrame, horse_id: str,
                               race_date: str) -> dict:
    """馬 history から 前走間隔 + 同 category 過去成績 features 計算."""
    h = history_df[history_df["horse_id"].astype(str) == str(horse_id)].copy()
    h["date_s"] = h["date"].astype(str)
    h = h[h["date_s"] < race_date].sort_values("date_s", ascending=False)

    out = {
        "days_since_prev_race": -1,
        "interval_category": -1,
        "interval_distance_interaction": -1,
        "interval_top3_rate_history": 0.30,
    }

    if len(h) == 0:
        return out

    prev = h.iloc[0]
    try:
        prev_date = pd.to_datetime(prev["date_s"], format="%Y%m%d")
        curr_date = pd.to_datetime(race_date, format="%Y%m%d")
        days = (curr_date - prev_date).days
        out["days_since_prev_race"] = days
        out["interval_category"] = categorize_interval(days)
    except Exception:
        return out

    # interval × distance interaction (current race の distance を取得不能のため、 prev 流用 簡易)
    try:
        prev_dist = pd.to_numeric(prev.get("distance"), errors="coerce")
        if pd.notna(prev_dist):
            out["interval_distance_interaction"] = int(out["interval_category"] * 1000 + (int(prev_dist) // 200))
    except Exception:
        pass

    # 過去 同 category での top3 率 (expanding)
    try:
        h["finish_num"] = pd.to_numeric(h.get("finish"), errors="coerce")
        h["top3"] = (h["finish_num"] <= 3).astype(int)
        # 各 prev row の interval category を計算 (簡略: 直前 prev のみ)
        # 簡易 overall top3 率 を 使用
        if len(h) >= 3:
            out["interval_top3_rate_history"] = round(h["top3"].mean(), 4)
    except Exception:
        pass

    return out


def backtest_distribution() -> dict:
    """jra_races_full から interval 分布 + category 別 top3 率を確認."""
    p = BASE / "data" / "jra_races_full.csv"
    if not p.exists():
        return {"available": False}

    df = pd.read_csv(p, usecols=["horse_id", "year", "month", "day", "finish"], low_memory=False)
    df["horse_id"] = df["horse_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["year_full"] = pd.to_numeric(df["year"], errors="coerce").apply(
        lambda y: 2000 + int(y) if pd.notna(y) and int(y) <= 30 else None
    )
    df = df.dropna(subset=["year_full"])
    df["date"] = pd.to_datetime(
        df["year_full"].astype(int).astype(str)
        + "-" + df["month"].astype(str).str.zfill(2)
        + "-" + df["day"].astype(str).str.zfill(2),
        errors="coerce"
    )
    df = df.dropna(subset=["date"]).sort_values(["horse_id", "date"])

    df["prev_date"] = df.groupby("horse_id")["date"].shift(1)
    df["days_since_prev"] = (df["date"] - df["prev_date"]).dt.days
    df["finish_num"] = pd.to_numeric(df["finish"], errors="coerce")
    df["top3"] = (df["finish_num"] <= 3).astype(int)

    valid = df.dropna(subset=["days_since_prev"])
    valid = valid[valid["days_since_prev"] > 0]

    cat_stats = {}
    for code, label, lo, hi in zip(INTERVAL_CODES, INTERVAL_LABELS,
                                     INTERVAL_BINS[:-1], INTERVAL_BINS[1:]):
        mask = (valid["days_since_prev"] >= lo) & (valid["days_since_prev"] < hi)
        sub = valid[mask]
        cat_stats[label] = {
            "code": code,
            "n_races": len(sub),
            "top3_rate": round(sub["top3"].mean() * 100, 2) if len(sub) > 0 else 0,
        }

    return {
        "available": True,
        "n_total_with_prev": int(len(valid)),
        "by_category": cat_stats,
    }


def cli():
    p = argparse.ArgumentParser(description="race_interval_features (Session #47 B)")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--out", default="data/v18/sprint2_race_interval_backtest.json")
    args = p.parse_args()

    if args.backtest:
        result = backtest_distribution()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        out_path = BASE / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  written: {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    cli()
