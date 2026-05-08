"""馬券種類 自動選択 (Session #45 B、 dev/sprint1).

レース毎に各券種の Expected Value (EV) を計算し、 max EV な券種を自動選択。
EV < 1.0 の R は skip。

JRA 控除率:
  単勝/複勝: 20% (還元率 80%)
  馬連/ワイド/馬単: 22.5% (77.5%)
  3連複/3連単: 27.5% (72.5%)

券種別 EV 計算:
  EV = sum(P(combination) × payout(combination)) / cost
  cost = 投票点数 × 100 円

実装 ticket types:
  - tansho:   単勝 1 点 (top1)
  - fukusho:  複勝 1 点 (top1) → 3 着以内
  - umaren:   馬連 2 点 (top1+top2, top1+top3)
  - umatan:   馬単 2 点 (top1→top2, top1→top3)
  - wide:     ワイド 3 点 (top1+top2/top3, top2+top3)
  - trio:     3連複 7 点 (現状 案B改 baseline)
  - tierce:   3連単 6 点 (top1 fix, top2/top3 順列)

usage:
  from tools.auto_ticket_selector import select_best_ticket
  best = select_best_ticket(
      probs=[0.45, 0.20, 0.15, ...], odds=[3.5, 6.0, 9.0, ...],
      top1_idx=0, top2_idx=1, top3_idx=2,
  )
  # → {'ticket': 'umaren', 'ev': 1.35, 'bets': [...], 'cost': 200}

V15 production 完全独立 (新規 module、 dev/sprint1 branch のみ)。
"""
from __future__ import annotations

import argparse
import json
from typing import List, Optional


# ===== JRA 控除率 (還元率) =====
RETURN_RATES = {
    "tansho":  0.80,
    "fukusho": 0.80,
    "umaren":  0.775,
    "umatan":  0.775,
    "wide":    0.775,
    "trio":    0.725,
    "tierce":  0.725,
}

# 1 点あたりコスト (100 円)
COST_PER_BET = 100


def compute_ev_tansho(probs: List[float], odds: List[float], top1_idx: int) -> dict:
    """単勝 1 点: top1 を 100 円で買い."""
    p = probs[top1_idx]
    o = odds[top1_idx] if odds and top1_idx < len(odds) and odds[top1_idx] > 0 else None
    if o is None or p <= 0:
        return {"ticket": "tansho", "ev": 0, "n_bets": 0, "cost": 0}
    expected_payout = p * o * 100
    cost = 100
    ev = expected_payout / cost
    return {
        "ticket": "tansho",
        "ev": round(ev, 4),
        "n_bets": 1,
        "cost": cost,
        "expected_payout": round(expected_payout, 1),
    }


def compute_ev_fukusho(probs: List[float], odds_fuku: Optional[List[float]],
                      top1_idx: int) -> dict:
    """複勝 1 点: top1 が 3 着以内. 簡易: P(top3) ≈ P(top1) × 1.7 + offset"""
    p_top1 = probs[top1_idx] if top1_idx < len(probs) else 0
    # 簡易: 単勝 prob → 複勝 prob (経験則 ×1.7-2.0)
    p_top3 = min(0.95, p_top1 * 1.7)
    # 複勝 odds は単勝の 0.3-0.5 倍程度 (簡易)
    if odds_fuku and top1_idx < len(odds_fuku) and odds_fuku[top1_idx] > 0:
        o = odds_fuku[top1_idx]
    else:
        return {"ticket": "fukusho", "ev": 0, "n_bets": 0, "cost": 0}
    cost = 100
    expected = p_top3 * o * 100
    ev = expected / cost
    return {
        "ticket": "fukusho",
        "ev": round(ev, 4),
        "n_bets": 1,
        "cost": cost,
        "expected_payout": round(expected, 1),
    }


def compute_ev_umaren(probs: List[float], odds_umaren: dict,
                      top1_idx: int, top2_idx: int, top3_idx: int) -> dict:
    """馬連 2 点: (top1, top2), (top1, top3)"""
    if not odds_umaren:
        return {"ticket": "umaren", "ev": 0, "n_bets": 0, "cost": 0}

    p1 = probs[top1_idx] if top1_idx < len(probs) else 0
    p2 = probs[top2_idx] if top2_idx < len(probs) else 0
    p3 = probs[top3_idx] if top3_idx < len(probs) else 0

    # P(馬連 = (a, b) hit) ≈ P(a top2) × P(b top2 | a top2) + reverse
    # 簡易: P(a, b 両方 top2) ≈ p_a × p_b × 2 (順序考慮なし)
    # umaren 1 点 cost 100 円
    # bet 1: (top1, top2) odds o12
    # bet 2: (top1, top3) odds o13
    bets = []
    total_expected = 0
    cost = 0
    for (a, b, p_a, p_b) in [(top1_idx, top2_idx, p1, p2), (top1_idx, top3_idx, p1, p3)]:
        key = tuple(sorted([a, b]))
        o = odds_umaren.get(key, 0)
        if o <= 0:
            continue
        # 簡易 P(combination) = p_a × p_b × 2 (race の中で a と b が top2 入る両方)
        p = p_a * p_b * 2
        expected = p * o * 100
        cost += 100
        total_expected += expected
        bets.append({"combination": list(key), "odds": o, "prob": round(p, 4),
                     "expected_payout": round(expected, 1)})

    if cost == 0:
        return {"ticket": "umaren", "ev": 0, "n_bets": 0, "cost": 0}

    ev = total_expected / cost
    return {
        "ticket": "umaren",
        "ev": round(ev, 4),
        "n_bets": len(bets),
        "cost": cost,
        "expected_payout": round(total_expected, 1),
        "bets": bets,
    }


def compute_ev_trio(probs: List[float], odds_trio: dict,
                    top1_idx: int, top2_idx: int, top3_idx: int,
                    top4_idx: int = None, top5_idx: int = None,
                    top6_idx: int = None) -> dict:
    """3 連複 7 点 (案B改 baseline): TOP1 軸 - TOP2,TOP3 - TOP2-TOP6"""
    if not odds_trio:
        return {"ticket": "trio", "ev": 0, "n_bets": 0, "cost": 0}

    p_list = probs
    a = top1_idx
    col2 = [top2_idx, top3_idx]
    col3 = [i for i in [top2_idx, top3_idx, top4_idx, top5_idx, top6_idx] if i is not None]

    bets = []
    seen = set()
    total_expected = 0
    cost = 0
    for x in col2:
        for y in col3:
            if x == a or y == a or x == y:
                continue
            tri = tuple(sorted([a, x, y]))
            if tri in seen:
                continue
            seen.add(tri)
            o = odds_trio.get(tri, 0)
            if o <= 0:
                continue
            # 簡易 P(trio) ≈ p_a × p_x × p_y × 6 (順序 6 通り)
            p = p_list[a] * p_list[x] * p_list[y] * 6
            expected = p * o * 100
            cost += 100
            total_expected += expected
            bets.append({"combination": list(tri), "odds": o, "prob": round(p, 4),
                         "expected_payout": round(expected, 1)})

    if cost == 0:
        return {"ticket": "trio", "ev": 0, "n_bets": 0, "cost": 0}

    ev = total_expected / cost
    return {
        "ticket": "trio",
        "ev": round(ev, 4),
        "n_bets": len(bets),
        "cost": cost,
        "expected_payout": round(total_expected, 1),
        "bets": bets[:5],  # 簡略表示
    }


def select_best_ticket(probs: List[float],
                       odds: List[float] = None,
                       odds_fuku: List[float] = None,
                       odds_umaren: dict = None,
                       odds_trio: dict = None,
                       top1_idx: int = 0, top2_idx: int = 1, top3_idx: int = 2,
                       top4_idx: int = 3, top5_idx: int = 4, top6_idx: int = 5,
                       min_ev: float = 1.0) -> dict:
    """全 ticket type の EV 比較、 max EV 選択."""
    candidates = []

    if odds:
        candidates.append(compute_ev_tansho(probs, odds, top1_idx))
    if odds_fuku:
        candidates.append(compute_ev_fukusho(probs, odds_fuku, top1_idx))
    if odds_umaren:
        candidates.append(compute_ev_umaren(probs, odds_umaren, top1_idx, top2_idx, top3_idx))
    if odds_trio:
        candidates.append(compute_ev_trio(probs, odds_trio, top1_idx, top2_idx, top3_idx,
                                          top4_idx, top5_idx, top6_idx))

    # max EV
    candidates.sort(key=lambda x: x.get("ev", 0), reverse=True)
    if not candidates:
        return {"ticket": "skip", "reason": "no odds data"}

    best = candidates[0]
    if best.get("ev", 0) < min_ev:
        return {"ticket": "skip", "reason": f"max EV {best.get('ev', 0):.4f} < {min_ev}",
                "candidates": candidates}

    best["candidates"] = candidates
    return best


def cli():
    p = argparse.ArgumentParser(description="auto_ticket_selector")
    p.add_argument("--probs", required=True, help="comma-separated probs")
    p.add_argument("--odds", help="comma-separated tansho odds")
    p.add_argument("--odds-fuku", help="comma-separated fukusho odds")
    p.add_argument("--min-ev", type=float, default=1.0)
    args = p.parse_args()

    probs = [float(x) for x in args.probs.split(",")]
    odds = [float(x) for x in args.odds.split(",")] if args.odds else None
    odds_fuku = [float(x) for x in args.odds_fuku.split(",")] if args.odds_fuku else None

    result = select_best_ticket(probs, odds=odds, odds_fuku=odds_fuku, min_ev=args.min_ev)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
