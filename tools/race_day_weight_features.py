"""当日馬体重 features 計算 (Session #48 B、 dev/two-stage).

馬の過去履歴 + 当日体重 から features 化:
- current_weight (kg)
- weight_change_kg (前走比)
- weight_change_pct (%)
- weight_vs_3r_avg (kg)
- weight_vs_same_dist_avg (kg)
- weight_extreme_change_flag (±10kg 超)

usage:
  from tools.race_day_weight_features import compute_weight_features
  feats = compute_weight_features(history_df, current_weight, current_distance)

V15 production 完全独立、 dev/two-stage 専用。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")


def compute_weight_features(history_df: pd.DataFrame, current_weight: float,
                            current_distance: int = None,
                            current_course: str = None) -> dict:
    """馬の過去履歴 + 当日体重 から features 計算.

    Args:
        history_df: 馬の過去 race 履歴 (date 順、 horse_weight 列)
        current_weight: 当日馬体重 (kg)
        current_distance: 現 race 距離
        current_course: 現 race コース

    Returns:
        dict of features
    """
    out = {
        "current_weight": current_weight,
        "weight_change_kg": 0.0,
        "weight_change_pct": 0.0,
        "weight_vs_3r_avg": 0.0,
        "weight_vs_same_dist_avg": 0.0,
        "weight_extreme_change_flag": 0,
        "history_n": 0,
    }

    if not current_weight or current_weight <= 0:
        return out

    h = history_df.copy()
    h["weight_num"] = pd.to_numeric(h.get("horse_weight"), errors="coerce")
    h = h.dropna(subset=["weight_num"])
    if len(h) == 0:
        return out

    # date 順 (新→旧)
    if "date" in h.columns:
        h = h.sort_values("date", ascending=False)
    elif "year" in h.columns and "month" in h.columns and "day" in h.columns:
        h["_date_calc"] = (h["year"].astype(str).str.zfill(2) +
                           h["month"].astype(str).str.zfill(2) +
                           h["day"].astype(str).str.zfill(2))
        h = h.sort_values("_date_calc", ascending=False)

    out["history_n"] = len(h)
    weights = h["weight_num"].tolist()

    # 前走比
    prev_weight = weights[0]
    out["weight_change_kg"] = round(current_weight - prev_weight, 1)
    if prev_weight > 0:
        out["weight_change_pct"] = round((current_weight - prev_weight) / prev_weight * 100, 2)

    # 過去 3 走 平均との差
    if len(weights) >= 3:
        avg3 = np.mean(weights[:3])
        out["weight_vs_3r_avg"] = round(current_weight - avg3, 1)
    elif len(weights) >= 1:
        out["weight_vs_3r_avg"] = round(current_weight - weights[0], 1)

    # 同距離 過去比較
    if current_distance and "distance" in h.columns:
        same_dist = h[pd.to_numeric(h["distance"], errors="coerce") == current_distance]
        if len(same_dist) > 0:
            avg = same_dist["weight_num"].mean()
            out["weight_vs_same_dist_avg"] = round(current_weight - avg, 1)

    # ±10kg 超 flag
    if abs(out["weight_change_kg"]) >= 10:
        out["weight_extreme_change_flag"] = 1

    return out


def backtest_weight_features() -> dict:
    """過去 retro で 体重変化 vs top3 の相関 backtest."""
    p = BASE / "data" / "jra_races_full.csv"
    if not p.exists():
        return {"available": False}

    df = pd.read_csv(p, usecols=["horse_id", "year", "month", "day",
                                  "horse_weight", "distance", "finish"],
                      low_memory=False)
    df["horse_id"] = df["horse_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["weight_num"] = pd.to_numeric(df["horse_weight"], errors="coerce")
    df["finish_num"] = pd.to_numeric(df["finish"], errors="coerce")
    df = df.dropna(subset=["weight_num", "finish_num"])
    df["target"] = (df["finish_num"] <= 3).astype(int)

    # 簡易: horse_weight 自体 vs top3 corr
    c1 = df[["weight_num", "target"]].corr().iloc[0, 1]

    # 体重変化 (group ごと sort + diff)
    df = df.sort_values(["horse_id", "year", "month", "day"])
    df["prev_weight"] = df.groupby("horse_id")["weight_num"].shift(1)
    df["weight_change"] = df["weight_num"] - df["prev_weight"]
    df_with_prev = df.dropna(subset=["weight_change"])
    c2 = df_with_prev[["weight_change", "target"]].corr().iloc[0, 1]

    return {
        "available": True,
        "n_total": int(len(df)),
        "n_with_prev_weight": int(len(df_with_prev)),
        "weight_corr_target": round(c1, 4),
        "weight_change_corr_target": round(c2, 4),
        "interpretation": "weight_change > 0 (増量) は top3 率に弱い relation",
    }


def cli():
    p = argparse.ArgumentParser(description="race_day_weight_features (Session #48 B)")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--out", default="data/v18/session_48_weight_features_backtest.json")
    args = p.parse_args()

    if args.backtest:
        result = backtest_weight_features()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        out_path = BASE / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  written: {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    cli()
