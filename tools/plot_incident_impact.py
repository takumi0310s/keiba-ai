"""4/19 SCRAPER-GUARD 事故による機会損失を可視化.

data/daily_predictions/20260419.csv と data/cumulative_results.csv (過去実績) を基に:
- 午前 (R1-R6) と午後 (R7-R12) のレース件数
- 条件別件数
- 条件別の過去実績 ROI を適用した期待プロフィット
- 事故で失った機会損失額

出力:
- report/incident_impact_20260419.png
- report/incident_impact_20260419_data.tsv
"""
from __future__ import annotations

import os
import sys
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

# 条件別の実績 ROI (CLAUDE.md 2026-03-14〜04-18 324R dedup 成績)
HISTORICAL_ROI = {
    "A": 1.229,
    "B": 0.000,
    "C": 1.238,
    "D": 1.443,
    "E": 0.132,
    "X": 0.138,
}
INVESTMENT_PER_RACE = 700


def main() -> int:
    pred_path = os.path.join(BASE, "data", "daily_predictions", "20260419.csv")
    if not os.path.exists(pred_path):
        print(f"[ERROR] 予測ファイル未検出: {pred_path}")
        return 1

    df = pd.read_csv(pred_path)
    # BOM 対応
    df.columns = [c.lstrip("\ufeff") for c in df.columns]

    # 午前 (R1-R6) / 午後 (R7-R12) で分割
    df["period"] = df["race_num"].apply(lambda n: "morning" if int(n) <= 6 else "afternoon")

    # 条件別集計
    summary = []
    for period in ("morning", "afternoon"):
        sub = df[df["period"] == period]
        for cond in sorted(sub["condition"].unique()):
            n = int((sub["condition"] == cond).sum())
            if n == 0:
                continue
            roi = HISTORICAL_ROI.get(cond, 1.0)
            invested = n * INVESTMENT_PER_RACE
            expected_return = invested * roi
            expected_profit = expected_return - invested
            summary.append({
                "period": period,
                "condition": cond,
                "n_races": n,
                "invested": invested,
                "expected_roi": roi,
                "expected_return": expected_return,
                "expected_profit": expected_profit,
            })
    sm = pd.DataFrame(summary)

    # 総計
    morning_total = sm[sm["period"] == "morning"]
    afternoon_total = sm[sm["period"] == "afternoon"]
    total_morning_invested = int(morning_total["invested"].sum())
    total_morning_expected_profit = int(morning_total["expected_profit"].sum())
    total_afternoon_invested = int(afternoon_total["invested"].sum())
    total_afternoon_expected_profit = int(afternoon_total["expected_profit"].sum())

    print("=" * 60)
    print("4/19 INCIDENT IMPACT ANALYSIS")
    print("=" * 60)
    print(f"Morning  (R1-R6): {len(morning_total):2d} conds, "
          f"N={int(morning_total['n_races'].sum())}R, "
          f"invest={total_morning_invested:,}円, "
          f"expected_profit={total_morning_expected_profit:+,}円 (LOST)")
    print(f"Afternoon(R7-R12): {len(afternoon_total):2d} conds, "
          f"N={int(afternoon_total['n_races'].sum())}R, "
          f"invest={total_afternoon_invested:,}円, "
          f"expected_profit={total_afternoon_expected_profit:+,}円")
    print(f"Machine loss from skipping morning: {total_morning_expected_profit:+,}円")

    # TSV 出力
    tsv_path = os.path.join(REPORT_DIR, "incident_impact_20260419_data.tsv")
    sm.to_csv(tsv_path, sep="\t", index=False)
    print(f"TSV saved: {tsv_path}")

    # 棒グラフ: 条件別 × morning/afternoon の期待プロフィット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    pivot = sm.pivot(index="condition", columns="period", values="expected_profit").fillna(0)
    pivot = pivot.reindex(sorted(pivot.index))
    x = range(len(pivot))
    w = 0.35
    ax1.bar([i - w/2 for i in x], pivot.get("morning", 0), w, label="Morning (LOST)", color="#d62728")
    ax1.bar([i + w/2 for i in x], pivot.get("afternoon", 0), w, label="Afternoon (bet)", color="#2ca02c")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(pivot.index)
    ax1.set_ylabel("Expected Profit (JPY)")
    ax1.set_title("Expected Profit by Condition — Morning vs Afternoon")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 条件別 race count
    pivot_n = sm.pivot(index="condition", columns="period", values="n_races").fillna(0)
    pivot_n = pivot_n.reindex(sorted(pivot_n.index))
    ax2.bar([i - w/2 for i in x], pivot_n.get("morning", 0), w, label="Morning (LOST)", color="#d62728")
    ax2.bar([i + w/2 for i in x], pivot_n.get("afternoon", 0), w, label="Afternoon (bet)", color="#2ca02c")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(pivot_n.index)
    ax2.set_ylabel("Race Count")
    ax2.set_title("Race Count by Condition")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"2026-04-19 SCRAPER-GUARD Incident Impact\n"
        f"Morning LOST: {total_morning_expected_profit:+,}円 "
        f"(invested: {total_morning_invested:,}円) "
        f"| Afternoon: {total_afternoon_expected_profit:+,}円",
        fontsize=11,
    )
    fig.tight_layout()
    png_path = os.path.join(REPORT_DIR, "incident_impact_20260419.png")
    fig.savefig(png_path, dpi=120)
    print(f"PNG saved: {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
