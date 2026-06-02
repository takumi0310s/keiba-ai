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


# =====================================================================
# 投票判定 (買う / 見送り) — 戦略⑦ フィルタの単一真実源
# Added: 2026-05-29. race_auto_notify.py predict_and_notify() の inline
# 早期 return フィルタを抽出。 ★ 従来の投票判断ロジックは完全不変 ★。
# 障害競走は呼び出し側で完全除外済みの前提 (このヘルパは平地のみ扱う)。
# =====================================================================

# skip_reason → 日本語表示ラベル (通知ヘッダ用)
SKIP_REASON_LABELS = {
    'distance_le_1000': '1000m以下 (非推奨 ROI85%)',
    'strategy_7_06_tokubetsu': '平場特別 (戦略⑦)',
    'strategy_7_kyoto_p0_2_5_17': '京都平場 (戦略⑦案C)',
    'strategy_c4_condA_1600_1800': '条件A 1600-1800m (C4)',
    'strategy_7_cond_E': '条件E 少頭数 (戦略⑦)',
    'strategy_7_cond_B': '条件B 重~不良 (戦略⑦)',
    'strategy_7_cond_X_p0_2_5_17': '条件X (戦略⑦案C)',
}

_GRADED_TOKENS = ['G1', 'G2', 'G3', 'GⅠ', 'GⅡ', 'GⅢ']
_LISTED_TOKENS = ['L)', '(L)', 'OP)', '(OP)']
_OPEN_TOKUBETSU_TOKENS = ['杯', '賞', 'ステークス', 'カップ', 'ハンデ']


def evaluate_bet_decision(race_name, course, distance, cond_key,
                          is_graded=None, is_listed=None,
                          is_open_tokubetsu=None, c4_effective=True):
    """戦略⑦ 投票判定。 (should_bet, skip_reason) を返す。

    should_bet=True / skip_reason=None  → 🟢 買い (投票対象、 従来通知レース)
    should_bet=False / skip_reason=str  → ⚪ 見送り (投票対象外・参考)

    ★ race_auto_notify.py の inline フィルタと同一順序・同一条件 ★。
    順序: distance≤1000 → 06特別 → 京都 → C4 → 条件E → 条件B → 条件X。
    最初に一致した理由を返す (元の早期 return と等価)。

    Note: 障害は呼び出し側で除外済み前提。 cond_key は予測後に確定する
    条件分類 (A/B/C/D/E/X)。
    """
    rn = str(race_name or '')
    crs = str(course or '')
    try:
        dist = int(distance or 0)
    except (TypeError, ValueError):
        dist = 0

    if is_graded is None:
        is_graded = any(g in rn for g in _GRADED_TOKENS)
    if is_listed is None:
        is_listed = any(s in rn for s in _LISTED_TOKENS)
    if is_open_tokubetsu is None:
        is_open_tokubetsu = any(s in rn for s in _OPEN_TOKUBETSU_TOKENS)

    # 1. 1000m以下 (非推奨)
    if 0 < dist <= 1000:
        return False, 'distance_le_1000'
    # 2. 06_特別 (G/L/OPEN特別 ではない平場特別)
    if '特別' in rn and not (is_graded or is_listed or is_open_tokubetsu):
        return False, 'strategy_7_06_tokubetsu'
    # 3. 京都 (P0-2 案C、 5/17 適用) — Graded/Listed は除外しない
    if crs == '京都' and not (is_graded or is_listed):
        return False, 'strategy_7_kyoto_p0_2_5_17'
    # 4. STRATEGY_C4: Cond-A 1600-1800m drag
    if c4_effective and cond_key == 'A' and 1600 <= dist <= 1800:
        return False, 'strategy_c4_condA_1600_1800'
    # 5. 条件E (頭数<=7)
    if cond_key == 'E':
        return False, 'strategy_7_cond_E'
    # 6. 条件B (重~不良馬場)
    if cond_key == 'B':
        return False, 'strategy_7_cond_B'
    # 7. 条件X (P0-2 案C、 5/17 適用) — Graded/Listed は除外しない
    if cond_key == 'X' and not (is_graded or is_listed):
        return False, 'strategy_7_cond_X_p0_2_5_17'

    return True, None


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
