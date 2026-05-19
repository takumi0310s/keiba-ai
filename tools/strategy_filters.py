"""
Pure-logic strategy filter functions extracted from race_auto_notify.py.
These are stateless helpers used for unit testing and potential reuse.

Source: tools/race_auto_notify.py predict_and_notify() inline strategies
Added: 2026-05-19 (D-5)
"""


def should_skip_c4(cond_key: str, distance: int) -> bool:
    """STRATEGY_C4: Cond-A 1600-1800m drag skip.

    production active — confirmed +8.62pt (重-2 audit)
    """
    return cond_key == 'A' and 1600 <= distance <= 1800


def should_skip_b1(top1_pop_rank: int) -> bool:
    """STRATEGY_B1: skip when V15 top1 == market 1-ban (paper eval only).

    Returns True if top1_pop_rank == 1.
    """
    return top1_pop_rank == 1


def should_bet_b2(top1_pop_rank: int, min_pop_rank: int = 3) -> bool:
    """STRATEGY_B2: bet only when top1 is divergence (pop_rank >= min).

    Returns True when top1_pop_rank >= min_pop_rank (V15 vs market divergence).
    """
    if top1_pop_rank <= 0:
        return True  # unknown pop_rank → do not filter
    return top1_pop_rank >= min_pop_rank


def should_skip_c2(top1_odds: float, course: str = '') -> bool:
    """STRATEGY_C2: odds band filter (paper eval only).

    Skip when:
    - odds < 1.5  (over-favorite)
    - odds > 20.0 (extreme longshot)
    - Tokyo course AND 5.0 <= odds <= 10.0 (Tokyo band)
    """
    if top1_odds <= 0:
        return False
    if top1_odds < 1.5:
        return True
    if top1_odds > 20.0:
        return True
    if '東京' in course and 5.0 <= top1_odds <= 10.0:
        return True
    return False


def build_trio_bets(
    n1: int, n2: int, n3: int, n4: int, n5: int, n6: int,
    apply_c3: bool = True
) -> list:
    """Generate standard 7-bet trio formation and optionally apply C3 filter.

    Formation: TOP1(n1) x [n2,n3] x [n2,n3,n4,n5,n6]  — 7 bets.
    STRATEGY_C3: remove (n1, n2, n4) — pos2 T1-T2-T4 — reduces to 6 bets.

    Returns list of sorted 3-tuples.
    """
    nums = [n1, n2, n3, n4, n5, n6]
    second = nums[1:3]    # n2, n3
    third = nums[1:6]     # n2..n6
    bets: set = set()
    for s in second:
        for t in third:
            combo = tuple(sorted({n1, s, t}))
            if len(combo) == 3:
                bets.add(combo)
    result = sorted(bets)

    if apply_c3:
        # Remove bet2: (n1, n2, n4)
        bet2 = tuple(sorted([n1, n2, n4]))
        result = [b for b in result if b != bet2]

    return result
