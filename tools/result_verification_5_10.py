"""5/10 朝 結果照合 自動化 (Session #42 D).

5/9 投資結果 と V15 / V18 / V19 / sib_exp 各 model 予測との比較を 自動集計。

source:
- data/daily_predictions/20260509.csv (V15 朝の予測)
- data/daily_results/20260509.csv (5/9 18:00 自動結果照合 後)
- data/v18/v18v19_sib_exp_v1/v18_lgb_sib_exp_v1.txt (sib_exp model、 retro 用)

集計指標:
- 採用 R 数 (案B改 1勝 + 戦略⑦) と 全 R 数
- V15 trio hit / ROI
- V18 単勝 winner_top1 / V19 複勝 top3 (sib含 / sib_exp で 比較)
- 撤退余裕 (累計 + 5/9 profit)
- 5/16 投入判定 への材料

usage:
  python tools/result_verification_5_10.py
  python tools/result_verification_5_10.py --date 20260509

V15 production 完全不変 (read-only 集計)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")


def aggregate_v15(date: str) -> dict:
    rp = BASE / "data" / "daily_results" / f"{date}.csv"
    if not rp.exists():
        return {"available": False, "reason": f"{rp.name} 不在"}
    df = pd.read_csv(rp, encoding='utf-8-sig', low_memory=False)
    settled = df[df.get('status', '') == 'settled'].copy()
    if len(settled) == 0:
        return {"available": False, "reason": "settled が 0"}

    # 案B改 (1勝 + 戦略⑦)
    case_b = settled.copy()
    case_b['investment'] = pd.to_numeric(case_b['investment'], errors='coerce').fillna(0)
    case_b['actual_payout'] = pd.to_numeric(case_b['actual_payout'], errors='coerce').fillna(0)
    case_b['profit'] = pd.to_numeric(case_b['profit'], errors='coerce').fillna(0)
    case_b['trio_hit'] = pd.to_numeric(case_b['trio_hit'], errors='coerce').fillna(0)

    if 'race_name' in case_b.columns:
        mask_1sho = case_b['race_name'].fillna('').astype(str).str.contains('1勝', na=False)
        case_b = case_b[mask_1sho]
    if 'course' in case_b.columns:
        case_b = case_b[case_b['course'].astype(str) != '京都']
    if 'condition' in case_b.columns:
        case_b = case_b[case_b['condition'].astype(str).isin(['A', 'C', 'D', 'X'])]  # B (重~不良) / E (頭数<=7) 除外
    if 'track_condition' in case_b.columns:
        case_b = case_b[~case_b['track_condition'].fillna('').astype(str).isin(['重', '不良'])]

    full_inv = float(settled['investment'].sum() if 'investment' in settled else 0)
    full_pay = float(settled['actual_payout'].sum() if 'actual_payout' in settled else 0)
    full_profit = float(settled['profit'].sum() if 'profit' in settled else 0)

    b_inv = float(case_b['investment'].sum())
    b_pay = float(case_b['actual_payout'].sum())
    b_profit = float(case_b['profit'].sum())
    b_hit = int(case_b['trio_hit'].sum())
    b_n = len(case_b)

    return {
        "available": True,
        "date": date,
        "v15_full": {
            "n_races": len(settled),
            "inv": int(full_inv),
            "pay": int(full_pay),
            "profit": int(full_profit),
            "roi_pct": round(full_pay / full_inv * 100, 2) if full_inv > 0 else 0,
        },
        "v15_case_b": {
            "n_races": b_n,
            "inv": int(b_inv),
            "pay": int(b_pay),
            "profit": int(b_profit),
            "trio_hit": b_hit,
            "hit_rate_pct": round(b_hit / b_n * 100, 2) if b_n > 0 else 0,
            "roi_pct": round(b_pay / b_inv * 100, 2) if b_inv > 0 else 0,
        },
    }


def aggregate_cumulative(through_date: str) -> dict:
    rp = BASE / "data" / "cumulative_results.csv"
    if not rp.exists():
        return {"available": False}
    df = pd.read_csv(rp, low_memory=False)
    df['profit_num'] = pd.to_numeric(df['profit'], errors='coerce').fillna(0)
    df['date'] = df['date'].astype(str).str.replace(r'\.0$', '', regex=True)
    df = df[df['date'] <= through_date]
    raw_total = int(df['profit_num'].sum())
    return {
        "available": True,
        "through_date": through_date,
        "raw_cumulative_jpy": raw_total,
        "retire_margin_jpy": raw_total - (-50_000),
        "user_real_per_claude_md": 13530,  # CLAUDE.md USER 実投資 累計 (5/6 真相確定値)
    }


def judge_5_16_go(v15_today: dict, cumulative: dict) -> dict:
    """5/16 V18/V19 投入 GO/no-go 判定 (Session #42 H plan v2 §4.1)."""
    if not v15_today.get("available"):
        return {"verdict": "data_missing", "go_probability_pct": None}

    b = v15_today.get("v15_case_b", {})
    if b.get("n_races", 0) == 0:
        return {"verdict": "data_missing_case_b", "go_probability_pct": None}

    profit = b.get("profit", 0)
    roi = b.get("roi_pct", 0)

    if profit >= 1000:
        return {"verdict": "大成功", "go_probability_pct": 85, "recommendation": "V18 sib_exp 単独 trial 推奨"}
    if profit >= 400:
        return {"verdict": "期待通り", "go_probability_pct": 75, "recommendation": "V18 sib_exp 単独 trial OK"}
    if profit >= 0:
        return {"verdict": "微益", "go_probability_pct": 65, "recommendation": "V18 sib_exp 単独 trial 慎重"}
    if profit >= -700:
        return {"verdict": "微損", "go_probability_pct": 45, "recommendation": "V15 単独継続 推奨"}
    if profit >= -1400:
        return {"verdict": "損失", "go_probability_pct": 30, "recommendation": "V15 単独継続、 5/22 再判定"}
    return {"verdict": "大損失", "go_probability_pct": 15, "recommendation": "V18/V19 NO-GO、 V15 単独継続"}


def main():
    p = argparse.ArgumentParser(description="5/10 朝 結果照合 (Session #42 D)")
    p.add_argument("--date", default="20260509")
    p.add_argument("--out", default="data/v18/result_verification_5_10.json")
    args = p.parse_args()

    print("=" * 70)
    print(f"5/10 朝 結果照合 ({args.date})")
    print("=" * 70)

    v15 = aggregate_v15(args.date)
    print(f"\n=== V15 retro {args.date} ===")
    if v15.get("available"):
        print(f"  V15 全 {v15['v15_full']['n_races']} races: ROI {v15['v15_full']['roi_pct']}%, profit {v15['v15_full']['profit']:+,d} 円")
        print(f"  V15 案B改 {v15['v15_case_b']['n_races']} races: ROI {v15['v15_case_b']['roi_pct']}%, "
              f"hit {v15['v15_case_b']['hit_rate_pct']}%, profit {v15['v15_case_b']['profit']:+,d} 円")
    else:
        print(f"  data 不在: {v15.get('reason', '?')}")
        print(f"  → 5/9 18:00 DailyResults 自動実行 後に再 run")

    cum = aggregate_cumulative(args.date)
    print(f"\n=== 累計収支 (through {args.date}) ===")
    if cum.get("available"):
        print(f"  raw cumulative: {cum['raw_cumulative_jpy']:+,d} 円")
        print(f"  retire margin: {cum['retire_margin_jpy']:+,d} 円")
        print(f"  USER 実 (CLAUDE.md): +13,530 円")

    judge = judge_5_16_go(v15, cum)
    print(f"\n=== 5/16 V18/V19 投入判定 ===")
    print(f"  verdict: {judge.get('verdict')}")
    if judge.get("go_probability_pct") is not None:
        print(f"  GO 確率: {judge['go_probability_pct']}%")
        print(f"  recommendation: {judge.get('recommendation', '?')}")

    summary = {
        "date": args.date,
        "v15_today": v15,
        "cumulative": cum,
        "judge_5_16": judge,
    }

    out_path = BASE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")

    # 撤退判定
    if cum.get("available"):
        margin = cum.get("retire_margin_jpy", 0)
        if margin < 0:
            print(f"\n  [CRITICAL] 撤退ライン到達")
        elif margin < 40000:
            print(f"\n  [WARN] 撤退余裕 残り {margin:+,d} 円 (40,000 円未満)")


if __name__ == "__main__":
    main()
