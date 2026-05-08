"""拡張 retro 4/12-5/5 集計 (Session #42 C).

V15 (現状 case B改 baseline) / V18 (既存 sib含) / V18 sib_exp (Session #41 D)
の 各 model retro 結果を 4/12-5/5 期間で集計。

source:
- V15 retro: data/daily_predictions/<date>.csv + data/daily_results/<date>.csv
   (既存 daily_predict + 結果照合済 data から ROI 計算)
- V18 sib含 retro: 既存 5/2-5/3 retro (data/v18/v18_tansho_oos_2025.csv は BT のみなので注意)
- V18/V19 sib_exp retro: Session #41 D の data/v18/v18v19_sib_exp_v1/sib_exp_retro_5_2_5_3_predictions.csv

V15 case B改 戦略 (実投資想定):
- 採用: 1勝クラス (12R)
- 戦略⑦ filter: 06_特別 / 京都 / 条件E / 条件B 除外
- 三連複 7点 700円固定

usage:
  python tools/extended_retro_4_12_5_5.py
  python tools/extended_retro_4_12_5_5.py --dates 20260418,20260419,20260425,20260426,20260502,20260503

V15 production 完全不変 (read-only 集計)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")


# ===== V15 retro (daily_predictions + daily_results 集計) =====

def aggregate_v15_retro(dates: List[str]) -> dict:
    """各 date の daily_results を集計して V15 retro 統計を計算."""
    rows = []
    for d in dates:
        rp = BASE / "data" / "daily_results" / f"{d}.csv"
        if not rp.exists():
            print(f"[V15] skip {d} (no daily_results)")
            continue
        try:
            df = pd.read_csv(rp, encoding='utf-8-sig', low_memory=False)
            df['_date'] = d
            rows.append(df)
        except Exception as e:
            print(f"[V15] {d} read error: {e}")
            continue

    if not rows:
        return {"n_races": 0}

    all_df = pd.concat(rows, ignore_index=True)

    # 戦略⑦ filter: 06_特別 / 京都 / 条件E (頭数<=7) / 条件B (重~不良)
    # daily_results に condition / num_horses / race_name 列があるか確認
    print(f"[V15] columns: {list(all_df.columns)[:20]}")

    n_total = len(all_df)
    # status='settled' に絞る
    if 'status' in all_df.columns:
        settled = all_df[all_df['status'] == 'settled'].copy()
    else:
        settled = all_df.copy()
    n_settled = len(settled)

    # trio_hit / actual_payout / investment / profit
    for col in ['trio_hit', 'actual_payout', 'investment', 'profit']:
        if col in settled.columns:
            settled[col] = pd.to_numeric(settled[col], errors='coerce').fillna(0)

    n_total_races = n_settled
    total_inv = float(settled['investment'].sum()) if 'investment' in settled else 0
    total_pay = float(settled['actual_payout'].sum()) if 'actual_payout' in settled else 0
    total_profit = float(settled['profit'].sum()) if 'profit' in settled else 0
    trio_hit_count = int(settled['trio_hit'].sum()) if 'trio_hit' in settled else 0

    roi = (total_pay / total_inv * 100) if total_inv > 0 else 0
    hit_rate = (trio_hit_count / n_total_races * 100) if n_total_races > 0 else 0

    # 案B改 想定 (1勝クラス + 戦略⑦ filter) の subset
    # 戦略⑦: 06_特別 / 京都 / 条件E (頭数<=7) / 条件B (重~不良) 除外
    case_b = settled.copy()
    case_b['actual_payout'] = pd.to_numeric(case_b.get('actual_payout', 0), errors='coerce').fillna(0)
    case_b['investment'] = pd.to_numeric(case_b.get('investment', 0), errors='coerce').fillna(0)
    case_b['profit'] = pd.to_numeric(case_b.get('profit', 0), errors='coerce').fillna(0)
    case_b['trio_hit'] = pd.to_numeric(case_b.get('trio_hit', 0), errors='coerce').fillna(0)

    if 'race_name' in case_b.columns:
        case_b['race_name'] = case_b['race_name'].fillna('').astype(str)
        mask_1sho = case_b['race_name'].str.contains('1勝', na=False)
        case_b = case_b[mask_1sho]
    # 京都 除外 (戦略⑦)
    if 'course' in case_b.columns:
        case_b = case_b[case_b['course'].astype(str) != '京都']
    # 条件 E 除外 (頭数<=7) — daily_results の condition 列が V15 class A-X
    if 'condition' in case_b.columns:
        case_b = case_b[case_b['condition'].astype(str) != 'E']
    # 条件 B 除外 (重~不良)
    if 'condition' in case_b.columns:
        case_b = case_b[case_b['condition'].astype(str) != 'B']
    # track_condition 列があれば 重/不良 直接除外も
    if 'track_condition' in case_b.columns:
        tc = case_b['track_condition'].fillna('').astype(str)
        case_b = case_b[~tc.isin(['重', '不良'])]

    n_b = len(case_b)
    b_inv = float(case_b['investment'].sum())
    b_pay = float(case_b['actual_payout'].sum())
    b_profit = float(case_b['profit'].sum())
    b_roi = (b_pay / b_inv * 100) if b_inv > 0 else 0
    b_hit = int(case_b['trio_hit'].sum())
    b_hit_rate = (b_hit / n_b * 100) if n_b > 0 else 0

    return {
        "n_races_total": n_total_races,
        "v15_full": {
            "inv": int(total_inv),
            "pay": int(total_pay),
            "profit": int(total_profit),
            "trio_hit": trio_hit_count,
            "hit_rate_pct": round(hit_rate, 2),
            "roi_pct": round(roi, 2),
        },
        "v15_case_b": {
            "n_races": n_b,
            "inv": int(b_inv),
            "pay": int(b_pay),
            "profit": int(b_profit),
            "trio_hit": b_hit,
            "hit_rate_pct": round(b_hit_rate, 2),
            "roi_pct": round(b_roi, 2),
        },
    }


# ===== V18/V19 sib_exp retro 集計 =====

def aggregate_sib_exp_retro() -> dict:
    """Session #41 D 出力 から V18/V19 sib_exp の retro 統計を計算."""
    rp = BASE / "data" / "v18" / "v18v19_sib_exp_v1" / "sib_exp_retro_5_2_5_3_predictions.csv"
    if not rp.exists():
        return {"available": False}
    df = pd.read_csv(rp)
    df_known = df[df['winner_known'] == 1]

    # winner_top1
    top1 = df_known.loc[df_known.groupby('race_id')['p_tansho'].idxmax()]
    winner_top1 = top1['is_win'].mean()

    # top3 (any of top 3 by p_tansho is winner)
    def top3_hit(g):
        top3 = g.nlargest(3, 'p_tansho')
        return any(top3['is_win'].values)
    top3_per_race = df_known.groupby('race_id', group_keys=False).apply(top3_hit)
    top3_hit_rate = top3_per_race.mean()

    # 単純 ROI 試算 (case B 三連複 7点 700円固定 想定)
    # → 実際は trio 払戻データが必要、 ここでは EV ベースのみ
    n_races = df_known['race_id'].nunique()

    return {
        "available": True,
        "n_races": int(n_races),
        "winner_top1_pct": round(float(winner_top1) * 100, 2),
        "top3_hit_rate_pct": round(float(top3_hit_rate) * 100, 2),
        "p_tansho_mean": round(float(df_known['p_tansho'].mean()), 4),
        "p_tansho_max": round(float(df_known['p_tansho'].max()), 4),
    }


# ===== V18 既存 (sib 含 ens) retro 集計 =====

def aggregate_v18_existing_retro() -> dict:
    """既存 5/2-5/3 retro (sib 含) から winner_top1 を計算."""
    rp_v18 = BASE / "data" / "v18" / "v18v19_retraining" / "no_sib_retro_5_2_5_3_predictions.csv"
    if not rp_v18.exists():
        return {"available": False}
    df = pd.read_csv(rp_v18)
    df_known = df[df['winner_known'] == 1]
    top1 = df_known.loc[df_known.groupby('race_id')['p_tansho'].idxmax()]
    winner_top1_no_sib = top1['is_win'].mean()
    n_races = df_known['race_id'].nunique()

    # OLD (sib 含 ens) は no_sib_live_retro_metrics.json から
    rp_metrics = BASE / "data" / "v18" / "v18v19_retraining" / "no_sib_live_retro_metrics.json"
    old_winner_top1 = None
    if rp_metrics.exists():
        try:
            m = json.loads(rp_metrics.read_text(encoding='utf-8'))
            old_winner_top1 = m.get('live', {}).get('OLD', {}).get('winner_top1')
        except Exception:
            pass

    return {
        "available": True,
        "n_races": int(n_races),
        "no_sib_winner_top1_pct": round(float(winner_top1_no_sib) * 100, 2),
        "old_winner_top1_pct": round(float(old_winner_top1) * 100, 2) if old_winner_top1 else None,
    }


def main():
    p = argparse.ArgumentParser(description="拡張 retro 4/12-5/5 (Session #42 C)")
    p.add_argument("--dates", default="20260418,20260419,20260425,20260426,20260502,20260503,20260505")
    p.add_argument("--out", default="data/v18/extended_retro_4_12_5_5_5_8.json")
    args = p.parse_args()

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    print(f"[retro] target dates ({len(dates)}): {dates}")

    print("\n=== V15 retro (daily_results 集計) ===")
    v15 = aggregate_v15_retro(dates)
    print(json.dumps(v15, ensure_ascii=False, indent=2))

    print("\n=== V18 既存 (sib 含) + no_sib (Session #37) ===")
    v18_old = aggregate_v18_existing_retro()
    print(json.dumps(v18_old, ensure_ascii=False, indent=2))

    print("\n=== V18 sib_exp (Session #41 D) ===")
    sib_exp = aggregate_sib_exp_retro()
    print(json.dumps(sib_exp, ensure_ascii=False, indent=2))

    summary = {
        "dates": dates,
        "v15": v15,
        "v18_existing": v18_old,
        "v18_sib_exp": sib_exp,
    }

    out_path = BASE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")

    # === 5/9 戦略 final 検証 ===
    print("\n=== 5/9 戦略 final 検証 ===")
    if v15.get('v15_case_b', {}).get('n_races', 0) > 0:
        b = v15['v15_case_b']
        print(f"  V15 案B改 retro {len(dates)} dates × ~{b['n_races']/len(dates):.1f}R/date:")
        print(f"    n_races: {b['n_races']}")
        print(f"    investment: {b['inv']:,d} JPY")
        print(f"    payout: {b['pay']:,d} JPY")
        print(f"    profit: {b['profit']:+,d} JPY")
        print(f"    hit_rate: {b['hit_rate_pct']}%")
        print(f"    ROI: {b['roi_pct']}%")
        print(f"  → BT 想定 161% [CI 135-222%] vs 実 {b['roi_pct']}%")


if __name__ == "__main__":
    main()
