"""レース skip optimizer (Session #45 C、 dev/sprint1).

現状: V15 案B改 で全 1勝クラス R に投票。
改善: 「自信度」 metric が低い R は skip。

skip_score = f(top1_prob, prob_dist, race_grade, num_horses)
threshold tuning: 0.6 / 0.7 / 0.8 で grid search

判定要素:
1. top1_prob - top2_prob (分布 sharpness)
2. top1_prob 絶対値
3. num_horses (頭数 多 → skip しやすい)
4. race_grade (重賞 → skip 候補強)

usage:
  from tools.race_skip_optimizer import should_skip_race
  skip, reason, score = should_skip_race(
      top1_prob=0.42, top2_prob=0.31, num_horses=14, race_grade='1勝')

V15 production 完全独立 (新規 module、 dev/sprint1 branch のみ)。
"""
from __future__ import annotations

import argparse
import json
from typing import Tuple


# ===== 設定 =====
SKIP_THRESHOLDS = {
    "loose":  0.50,   # 緩い (skip 少)
    "medium": 0.60,   # 中
    "strict": 0.70,   # 厳しい (skip 多)
}

DEFAULT_THRESHOLD = 0.60  # medium


def compute_skip_score(top1_prob: float, top2_prob: float = None,
                       num_horses: int = None, race_grade: str = None) -> float:
    """skip_score 計算 (高いほど skip 推奨).

    高 score = 「自信度低い → skip」 を 意味する。
    低 score = 「予測自信あり → 投票」

    Returns:
        skip_score: 0.0-1.0
    """
    # base: 1 - top1_prob (top1_prob 高いほど skip しない)
    score = 1.0 - top1_prob  # 0.40 prob → 0.60 score

    # top1-top2 差 (sharp distribution → 投票) — 反映で score 下げる
    if top2_prob is not None:
        sharpness = top1_prob - top2_prob  # 0-1 (high = sharp)
        score -= sharpness * 0.3  # sharp 1.0 で -0.3 lower

    # num_horses bonus (多頭 → skip しやすい)
    if num_horses is not None and num_horses >= 16:
        score += 0.05

    # race_grade penalty (重賞 → skip 候補)
    if race_grade and race_grade in ('G1', 'G2', 'G3'):
        score += 0.10  # 重賞 +0.10 で skip 推奨

    return max(0.0, min(1.0, score))


def should_skip_race(top1_prob: float, top2_prob: float = None,
                     num_horses: int = None, race_grade: str = None,
                     threshold: float = DEFAULT_THRESHOLD) -> Tuple[bool, str, float]:
    """skip 判定.

    Returns:
        (skip: bool, reason: str, score: float)
    """
    score = compute_skip_score(top1_prob, top2_prob, num_horses, race_grade)

    if score >= threshold:
        return True, f"skip_score {score:.4f} ≥ {threshold}", score
    return False, f"OK (skip_score {score:.4f} < {threshold})", score


def cli():
    p = argparse.ArgumentParser(description="race skip optimizer")
    p.add_argument("--top1-prob", type=float, required=True)
    p.add_argument("--top2-prob", type=float, default=None)
    p.add_argument("--num-horses", type=int, default=None)
    p.add_argument("--race-grade", default=None)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = p.parse_args()

    skip, reason, score = should_skip_race(
        args.top1_prob, args.top2_prob, args.num_horses, args.race_grade,
        threshold=args.threshold,
    )
    print(json.dumps({
        "skip": skip,
        "reason": reason,
        "skip_score": round(score, 4),
        "threshold": args.threshold,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
