"""騎手 graph features (Session #47 E、 dev/sprint2).

過去 3 年のレース data から騎手 graph 構築:
- node: 騎手
- edge: 同 race に出走 (重み = 共出走回数)
- 中心性 metrics: degree、 (簡易 betweenness)、 pagerank-like

features 化:
- jockey_degree (出走回数 = node degree)
- jockey_top_partner_count (頻繁 共出走の騎手数)
- jockey_top_partner_top3_rate (上位 partner の top3 率 平均)

V15 production 完全独立、 dev/sprint2 のみ。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

BASE = Path(r"C:/Users/takum/keiba-ai")


def build_jockey_graph(races_df: pd.DataFrame) -> dict:
    """各 race の出走騎手 set から co-occurrence graph 構築."""
    # group by race_id → 出走騎手 list
    grp = races_df.groupby("race_id")["jockey"].apply(list)
    edges = defaultdict(int)  # (jockey_a, jockey_b) → 共出走回数
    degrees = Counter()       # jockey → 出走回数

    for race_id, jockeys in grp.items():
        unique_jockeys = list(set(j for j in jockeys if j and pd.notna(j)))
        for j in unique_jockeys:
            degrees[j] += 1
        for i in range(len(unique_jockeys)):
            for k in range(i + 1, len(unique_jockeys)):
                a, b = sorted([unique_jockeys[i], unique_jockeys[k]])
                edges[(a, b)] += 1

    return {"degrees": dict(degrees), "edges": dict(edges)}


def compute_jockey_features(graph: dict, jockey_top3_rates: dict,
                            target_jockey: str, top_n: int = 10) -> dict:
    """ある jockey の network features 計算."""
    degrees = graph.get("degrees", {})
    edges = graph.get("edges", {})

    out = {
        "jockey_degree": int(degrees.get(target_jockey, 0)),
        "jockey_top_partner_count": 0,
        "jockey_top_partner_top3_rate_avg": 0.0,
        "jockey_isolation_score": 1.0,  # 0=hub, 1=isolated
    }

    # 共出走 上位 partner 抽出
    co_occurrences = []
    for (a, b), count in edges.items():
        if a == target_jockey:
            co_occurrences.append((b, count))
        elif b == target_jockey:
            co_occurrences.append((a, count))

    co_occurrences.sort(key=lambda x: -x[1])
    top_partners = co_occurrences[:top_n]
    out["jockey_top_partner_count"] = len(top_partners)

    if top_partners:
        rates = []
        for partner, _ in top_partners:
            r = jockey_top3_rates.get(partner)
            if r is not None:
                rates.append(r)
        if rates:
            out["jockey_top_partner_top3_rate_avg"] = round(np.mean(rates), 4)

    # isolation: degree / max_degree (近似)
    if degrees:
        max_d = max(degrees.values())
        out["jockey_isolation_score"] = round(1 - (out["jockey_degree"] / max_d), 4)

    return out


def backtest_summary() -> dict:
    """過去 3 年 (2022-2025) で 騎手 network 統計."""
    p = BASE / "data" / "jra_races_full.csv"
    if not p.exists():
        return {"available": False}

    df = pd.read_csv(p, usecols=["race_id", "jockey", "year", "finish"], low_memory=False)
    df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["year_num"] >= 22) & (df["year_num"] <= 25)]
    df["finish_num"] = pd.to_numeric(df["finish"], errors="coerce")
    df = df.dropna(subset=["jockey", "finish_num"])
    df["top3"] = (df["finish_num"] <= 3).astype(int)

    print(f"[network] data: {len(df):,} rows、 unique races {df['race_id'].nunique():,}")

    graph = build_jockey_graph(df)
    print(f"[network] graph: {len(graph['degrees'])} jockeys, {len(graph['edges'])} edges")

    # 各 jockey の top3 率
    jockey_stats = df.groupby("jockey").agg(
        n_rides=("top3", "count"),
        top3_rate=("top3", "mean"),
    ).reset_index()
    jockey_top3 = dict(zip(jockey_stats["jockey"], jockey_stats["top3_rate"]))

    # 上位 5 jockey の network features
    top_jockeys = jockey_stats.nlargest(5, "n_rides")["jockey"].tolist()
    samples = []
    for j in top_jockeys:
        feats = compute_jockey_features(graph, jockey_top3, j)
        feats["jockey"] = j
        feats["top3_rate"] = round(jockey_top3.get(j, 0), 4)
        samples.append(feats)

    return {
        "available": True,
        "n_jockeys": len(graph["degrees"]),
        "n_edges": len(graph["edges"]),
        "n_races": int(df["race_id"].nunique()),
        "top_5_jockey_samples": samples,
    }


def cli():
    p = argparse.ArgumentParser(description="jockey_network (Session #47 E)")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--out", default="data/v18/sprint2_jockey_network_backtest.json")
    args = p.parse_args()

    if args.backtest:
        result = backtest_summary()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        out_path = BASE / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\n  written: {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    cli()
