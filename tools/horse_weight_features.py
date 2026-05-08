"""馬体重変化 features (Session #47 A、 dev/sprint2).

直近 3 走の体重 trend + 変化率 + 同条件比較。

features:
- weight_trend_3r: trend (-1: 減量、 0: 安定、 +1: 増量)
- weight_change_pct_3r: 平均変化率 (%)
- weight_vs_same_cond: 同 course/distance での過去体重との差 (kg)
- weight_extreme_change_3r_count: ±10kg 超 変化の 過去 3 走 count

usage:
  from tools.horse_weight_features import compute_horse_weight_features
  feats = compute_horse_weight_features(df_race)  # race row data + history

V15 production 完全独立、 dev/sprint2 のみ。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")


def compute_horse_weight_features(history_df: pd.DataFrame,
                                  current_horse_id: str,
                                  current_course: str = None,
                                  current_distance: int = None) -> dict:
    """馬の過去体重から features を計算.

    history_df: 馬の過去 race 履歴 DataFrame、 columns: horse_id, date, course, distance, horse_weight
    current_horse_id: 対象馬の horse_id
    current_course: (optional) 現 race の course (同条件比較用)
    current_distance: (optional) 同上

    Returns:
        dict of features
    """
    h = history_df[history_df["horse_id"].astype(str) == str(current_horse_id)].copy()
    if "date" in h.columns:
        h = h.sort_values("date", ascending=False)
    h["weight_num"] = pd.to_numeric(h.get("horse_weight"), errors="coerce")
    h = h.dropna(subset=["weight_num"])

    out = {
        "weight_trend_3r": 0,
        "weight_change_pct_3r": 0.0,
        "weight_vs_same_cond": 0.0,
        "weight_extreme_change_3r_count": 0,
        "weight_history_n": len(h),
    }

    if len(h) == 0:
        return out

    last3 = h.head(3)
    weights = last3["weight_num"].tolist()

    # trend (-1/0/+1)
    if len(weights) >= 2:
        diff = weights[0] - weights[-1]
        if diff > 4: out["weight_trend_3r"] = 1
        elif diff < -4: out["weight_trend_3r"] = -1

    # 変化率 (mean of consecutive % changes)
    if len(weights) >= 2:
        pct_changes = []
        for i in range(len(weights) - 1):
            curr, prev = weights[i], weights[i + 1]
            if prev > 0:
                pct_changes.append((curr - prev) / prev * 100)
        if pct_changes:
            out["weight_change_pct_3r"] = round(np.mean(pct_changes), 2)

    # ±10kg 超 変化 count
    extreme_count = 0
    for i in range(len(weights) - 1):
        if abs(weights[i] - weights[i + 1]) >= 10:
            extreme_count += 1
    out["weight_extreme_change_3r_count"] = extreme_count

    # 同条件 (course + distance) との差
    if current_course and current_distance and len(h) > 0:
        same = h[(h["course"].astype(str) == str(current_course))
                & (pd.to_numeric(h.get("distance"), errors="coerce") == current_distance)]
        if len(same) > 0:
            avg_same = same["weight_num"].mean()
            current_weight = weights[0] if weights else None
            if current_weight:
                out["weight_vs_same_cond"] = round(current_weight - avg_same, 1)

    return out


def backtest_contribution(target_dates: list[str]) -> dict:
    """過去 retro data で 体重変化 features の AUC contribution 簡易測定."""
    summary = {"available": False}
    p = BASE / "data" / "jra_races_full.csv"
    if not p.exists():
        return summary
    try:
        df = pd.read_csv(p, usecols=["horse_id", "year", "month", "day", "course",
                                      "distance", "horse_weight", "finish"],
                          low_memory=False)
        df["horse_id"] = df["horse_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        df["weight_num"] = pd.to_numeric(df["horse_weight"], errors="coerce")
        df["finish_num"] = pd.to_numeric(df["finish"], errors="coerce")
        df = df.dropna(subset=["weight_num", "finish_num"])
        df["target"] = (df["finish_num"] <= 3).astype(int)
        # group by horse, compute per-horse stat
        grp = df.groupby("horse_id")
        # mean weight + variance (single feature for backtest correlation)
        weight_var = grp["weight_num"].std().rename("horse_weight_std").reset_index()
        # merge back
        merged = df.merge(weight_var, on="horse_id", how="left")
        merged["horse_weight_std"] = merged["horse_weight_std"].fillna(0)
        # corr with target
        c = merged[["horse_weight_std", "target"]].corr().iloc[0, 1]
        summary = {
            "available": True,
            "n": len(merged),
            "horse_weight_std_corr_target": round(c, 4),
            "interpretation": "high std = unstable horse、 corr negative if 不安定 → top3 less",
        }
    except Exception as e:
        summary["error"] = str(e)[:120]
    return summary


def cli():
    p = argparse.ArgumentParser(description="horse_weight_features (Session #47 A)")
    p.add_argument("--horse-id", default=None)
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--out", default="data/v18/sprint2_horse_weight_backtest.json")
    args = p.parse_args()

    if args.backtest:
        result = backtest_contribution([])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        out_path = BASE / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  written: {out_path.relative_to(BASE)}")
        return

    if args.horse_id:
        # sample test (jra_races_full 必要)
        p_full = BASE / "data" / "jra_races_full.csv"
        if not p_full.exists():
            print("[!] jra_races_full.csv 不在")
            sys.exit(1)
        history = pd.read_csv(p_full, usecols=["horse_id", "year", "month", "day",
                                                "course", "distance", "horse_weight"],
                              low_memory=False)
        history["horse_id"] = history["horse_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        history["date"] = (history["year"].astype(str).str.zfill(2) +
                           history["month"].astype(str).str.zfill(2) +
                           history["day"].astype(str).str.zfill(2))
        feats = compute_horse_weight_features(history, args.horse_id)
        print(json.dumps(feats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
