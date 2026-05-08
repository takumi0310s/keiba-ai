"""動的 Kelly criterion (Session #45 A、 dev/sprint1).

現状: V15 案B改 で固定 Eighth Kelly (1/8 = 12.5%)。
改善: model の予測 confidence で動的調整 (Quarter / Eighth / Sixteenth)。

- top1_prob > 0.4         → Quarter Kelly  (1/4 = 25%、 高自信、 max 投資)
- top1_prob 0.30 - 0.4    → Eighth Kelly   (1/8 = 12.5%、 baseline)
- top1_prob 0.25 - 0.30   → Sixteenth Kelly (1/16 = 6.25%、 控えめ)
- top1_prob < 0.25        → skip            (期待 EV 低、 投票しない)

数式:
  bet_amount = bankroll × kelly_fraction × (b * p - q) / b
    b: net 配当倍率 (= odds - 1)
    p: 推定勝率 (top1_prob)
    q: 1 - p

実 production では 案B改 700円/R (Eighth Kelly 概算) を baseline とし、
本 module は overlay で 1.0x / 0.5x / 1.5x の係数で調整。

V15 production 完全独立 (新規 module、 predict_core 不変)。

usage:
  from tools.dynamic_kelly import compute_kelly_fraction
  frac, mode = compute_kelly_fraction(top1_prob=0.42, top1_odds=4.5)
  # → frac=0.25 (Quarter Kelly)、 mode='HIGH_CONFIDENCE'

  # 案B改 baseline 700円 に係数掛けて 動的調整
  base_bet = 700
  adjusted = base_bet * (frac / 0.125)  # Eighth = 0.125 を baseline
  # 高自信時 adjusted = 700 * (0.25 / 0.125) = 1,400 円

  # CLI
  python tools/dynamic_kelly.py --top1-prob 0.42 --odds 4.5
"""
from __future__ import annotations

import argparse
import json
from typing import Tuple


# ===== 設定 =====
KELLY_THRESHOLDS = [
    # (lower_prob, upper_prob, fraction, mode_name)
    (0.40, 1.00, 0.25,    "HIGH_CONFIDENCE"),    # Quarter Kelly
    (0.30, 0.40, 0.125,   "BASELINE"),            # Eighth Kelly (案B改 同等)
    (0.25, 0.30, 0.0625,  "LOW_CONFIDENCE"),      # Sixteenth Kelly
    (0.00, 0.25, 0.0,     "SKIP"),                # 投票しない
]

BASELINE_FRACTION = 0.125  # 案B改 Eighth Kelly
BASELINE_BET = 700  # 案B改 投資額


def compute_kelly_fraction(top1_prob: float, top1_odds: float = None) -> Tuple[float, str]:
    """top1_prob から動的 Kelly fraction を返す.

    Returns:
        (kelly_fraction, mode_name)
    """
    for lo, hi, frac, mode in KELLY_THRESHOLDS:
        if lo <= top1_prob < hi:
            return frac, mode
    # fallback (top1_prob >= 1.0 等)
    return 0.0, "INVALID"


def compute_bet_size(top1_prob: float, top1_odds: float = None,
                     bankroll: float = None,
                     base_bet: float = BASELINE_BET) -> dict:
    """動的 Kelly + 案B改 baseline overlay で bet 額計算.

    Args:
        top1_prob: 推定勝率 (0-1)
        top1_odds: top1 odds (None なら kelly multiplier のみ)
        bankroll: 現在資産 (Full Kelly 計算用、 None なら overlay only)
        base_bet: baseline 投資額 (default 700円、 案B改)

    Returns:
        {
            "kelly_fraction": float,
            "mode": str,
            "multiplier": float (1x baseline = 1.0),
            "bet_amount": int,
            "skip": bool,
        }
    """
    frac, mode = compute_kelly_fraction(top1_prob, top1_odds)

    if mode == "SKIP":
        return {
            "kelly_fraction": 0.0,
            "mode": "SKIP",
            "multiplier": 0.0,
            "bet_amount": 0,
            "skip": True,
            "reason": f"top1_prob {top1_prob:.4f} < 0.25 threshold",
        }

    # baseline (Eighth Kelly = 0.125) に対する 比率
    multiplier = frac / BASELINE_FRACTION
    bet_amount = int(base_bet * multiplier)

    return {
        "kelly_fraction": frac,
        "mode": mode,
        "multiplier": multiplier,
        "bet_amount": bet_amount,
        "skip": False,
        "top1_prob": top1_prob,
        "base_bet": base_bet,
    }


def cli():
    p = argparse.ArgumentParser(description="動的 Kelly criterion")
    p.add_argument("--top1-prob", type=float, required=True)
    p.add_argument("--odds", type=float, default=None)
    p.add_argument("--bankroll", type=float, default=None)
    p.add_argument("--base-bet", type=float, default=BASELINE_BET)
    args = p.parse_args()

    result = compute_bet_size(args.top1_prob, args.odds, args.bankroll, args.base_bet)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
