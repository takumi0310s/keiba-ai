"""Wide (ワイド) ticket 提案 helper.

V15 prediction TOP3 から wide 4 点 (順序問わず 3 着以内 2 頭組合せ) を 提案。
trio より hit rate 2-3x、 配当 約 1/3。 投資 安定化 用。

V15 不変、 後段 layer。 daily_predict / race_auto_notify 不変、 本 module は 別途 呼出。

Wide formation (4 点):
- TOP1 - TOP2
- TOP1 - TOP3
- TOP1 - TOP4
- TOP2 - TOP3

各 100 円 = 400 円/race (trio 700 円 + wide 400 円 = 1100 円/race) — オプション、 user 判断。

Usage:
    from tools.wide_ticket_helper import generate_wide_formation

    bets = generate_wide_formation(top_horses)  # [(1,2), (1,3), (1,4), (2,3)]
    bets_str = format_wide_bets(bets)            # "1-2; 1-3; 1-4; 2-3"
"""
from __future__ import annotations

from typing import List, Tuple


def generate_wide_formation(top_horses: list, formation: str = 'safe4') -> List[Tuple[int, int]]:
    """TOP horses list から wide 買い目 生成.

    Args:
        top_horses: TOP1 から umaban list (例 [1, 5, 3, 7, 9])
        formation: 'safe4' (4 点、 推奨)、 'wide5' (5 点)、 'aggressive3' (3 点 攻め)

    Returns:
        list of (umaban_a, umaban_b) tuples、 sorted で重複なし
    """
    if len(top_horses) < 4:
        return []

    t1, t2, t3, t4 = top_horses[:4]

    if formation == 'safe4':
        # TOP1 中心 4 点
        bets = [
            tuple(sorted((t1, t2))),
            tuple(sorted((t1, t3))),
            tuple(sorted((t1, t4))),
            tuple(sorted((t2, t3))),
        ]
    elif formation == 'wide5':
        # TOP1-2-3-4-5 fully 連携 5 点
        if len(top_horses) < 5:
            return generate_wide_formation(top_horses, 'safe4')
        t5 = top_horses[4]
        bets = [
            tuple(sorted((t1, t2))),
            tuple(sorted((t1, t3))),
            tuple(sorted((t1, t4))),
            tuple(sorted((t1, t5))),
            tuple(sorted((t2, t3))),
        ]
    elif formation == 'aggressive3':
        # 攻め 3 点 (高配当狙い)
        bets = [
            tuple(sorted((t1, t3))),
            tuple(sorted((t1, t4))),
            tuple(sorted((t2, t4))),
        ]
    else:
        raise ValueError(f'unknown formation: {formation}')

    # dedup
    seen = set()
    out = []
    for b in bets:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def format_wide_bets(bets: List[Tuple[int, int]]) -> str:
    """買い目を '1-2; 1-3; ...' format に."""
    return '; '.join('-'.join(str(n) for n in b) for b in bets)


def estimate_wide_hit_rate(trio_hit_rate: float, condition: str = 'A') -> float:
    """trio hit rate から wide hit rate 推定 (経験則).

    Wide は 2 頭 が 3 着内なら hit、 trio は 3 頭 全 3 着内必要。
    経験則: wide hit rate ≒ trio hit rate × 2.5
    """
    base = trio_hit_rate * 2.5
    # 条件別 補正 (経験 から)
    multiplier = {
        'A': 1.0, 'B': 0.95, 'C': 0.92,
        'D': 0.88, 'E': 0.85, 'X': 0.90,
    }.get(condition, 1.0)
    return min(0.95, base * multiplier)


def estimate_wide_payout_range(trio_payout: float) -> Tuple[float, float]:
    """trio 配当 から wide 配当 範囲 推定."""
    # Wide 配当 は trio の 30-50% (経験則)
    low = trio_payout * 0.25
    high = trio_payout * 0.45
    return low, high


if __name__ == '__main__':
    # Sample test
    horses = [1, 5, 3, 7, 9, 11]
    for f in ['safe4', 'wide5', 'aggressive3']:
        bets = generate_wide_formation(horses, formation=f)
        print(f'{f}: {format_wide_bets(bets)}')

    print()
    for cond in 'ABCDEX':
        for tr_hit in [0.20, 0.30, 0.45]:
            wh = estimate_wide_hit_rate(tr_hit, cond)
            print(f'  {cond} trio {tr_hit:.0%} → wide hit ~ {wh:.0%}')
