"""
V15 top-5 rerank — paper evaluation only.
Production 投入は 6/17 採用判定後。

入力:
  scores    : pd.Series, index=馬番(int), values=スコア(float)
  odds      : pd.Series, index=馬番(int), values=単勝オッズ(float)  or None
  condition : str, 'A'|'B'|'C'|'D'|'E'|'X'

出力: reranked top-5 馬番リスト (list[int])

5 rerank candidates
  a. probability tempering   — score^(1/T)
  b. odds divergence boost   — score × (1 - 1/odds)
  c. top-5 spread            — 近似スコア(<threshold)の馬を下位へ
  d. rank average            — (score_rank + raw_score_rank) / 2
  e. condition-aware         — A/C は top1 重視, D/X は top3-5 重視

★ 注意: predict_core.py / daily_predict.py / app.py / race_auto_notify.py は
         一切改変禁止。このファイルは独立 post-process layer。
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _top_n(series: pd.Series, n: int = 5) -> list[int]:
    """Return top-n horse numbers sorted by descending score."""
    return list(series.sort_values(ascending=False).head(n).index.astype(int))


def _validate_scores(scores: pd.Series) -> pd.Series:
    """Coerce index to int, clip negatives, raise if empty."""
    if scores is None or len(scores) == 0:
        raise ValueError("scores must be a non-empty pd.Series")
    scores = scores.copy()
    scores.index = scores.index.astype(int)
    scores = scores.clip(lower=0.0)
    return scores


# ---------------------------------------------------------------------------
# Rerank a: probability tempering
# ---------------------------------------------------------------------------

def rerank_tempering(scores: pd.Series, T: float = 0.8, n: int = 5) -> list[int]:
    """Probability tempering rerank.

    T < 1: scores が均等化される (低スコア馬の相対確率が上昇)
    T = 1: 変化なし
    T > 1: scores がより尖る (高スコア馬の相対確率がさらに上昇)

    Args:
        scores : pd.Series, index=馬番, values=スコア(float, 0-1)
        T      : temperature。小さいほど分布が平坦になる。
        n      : 返す頭数 (default 5)

    Returns:
        list[int]: reranked top-n 馬番
    """
    scores = _validate_scores(scores)
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    # スコアを正規化して確率化
    s = scores.values.astype(float)
    s_sum = s.sum()
    if s_sum <= 0:
        return _top_n(scores, n)
    probs = s / s_sum
    # temper: raise to 1/T then renormalize
    probs_t = np.power(np.clip(probs, 1e-9, None), 1.0 / T)
    probs_t = probs_t / probs_t.sum()
    tempered = pd.Series(probs_t, index=scores.index)
    return _top_n(tempered, n)


# ---------------------------------------------------------------------------
# Rerank b: odds divergence boost
# ---------------------------------------------------------------------------

def rerank_odds_divergence(
    scores: pd.Series,
    odds: Optional[pd.Series],
    alpha: float = 0.5,
    n: int = 5,
) -> list[int]:
    """Market divergence boost rerank.

    rerank_score = score × (1 + alpha × divergence)
    divergence   = AI確率 / 市場確率 - 1  (正 = AIが市場より高く評価)

    AIスコアが高いのにオッズが高い馬（市場過小評価）を上位へ。
    odds が None の場合は素のスコアで返す。

    Args:
        scores : pd.Series, index=馬番
        odds   : pd.Series, index=馬番, values=単勝オッズ  or None
        alpha  : divergence の重み (0=無効, 1=強)
        n      : 返す頭数

    Returns:
        list[int]: reranked top-n 馬番
    """
    scores = _validate_scores(scores)
    if odds is None or len(odds) == 0:
        return _top_n(scores, n)

    odds = odds.copy()
    odds.index = odds.index.astype(int)
    # 有効なオッズのみ使用 (>1.0)
    odds = odds[odds > 1.0]

    s_sum = scores.sum()
    if s_sum <= 0:
        return _top_n(scores, n)

    ai_probs = scores / s_sum * 3.0   # 複勝圏確率推定 (×3)
    ai_probs = ai_probs.clip(upper=0.95)

    rerank_scores = scores.copy().astype(float)
    for uma in scores.index:
        o = odds.get(uma, None)
        if o is None or o <= 1.0:
            continue
        market_prob = 1.0 / o
        ai_prob = ai_probs.get(uma, 0.0)
        divergence = (ai_prob / (market_prob + 1e-9)) - 1.0
        # boost: positive divergence → uprank, negative → downrank
        rerank_scores[uma] = float(scores[uma]) * (1.0 + alpha * np.clip(divergence, -0.9, 5.0))

    return _top_n(rerank_scores, n)


# ---------------------------------------------------------------------------
# Rerank c: top-5 spread (近似スコア分散)
# ---------------------------------------------------------------------------

def rerank_spread(
    scores: pd.Series,
    threshold: float = 0.02,
    penalty: float = 0.5,
    n: int = 5,
) -> list[int]:
    """Near-tie spread rerank.

    スコア差が threshold 未満の隣接馬ペアは「実質同スコア」と見なし、
    その後続馬のスコアに penalty を乗じて下位化する。
    → top-5 内の近似スコア馬を整理し、より差のある馬を上位へ引き上げる。

    Args:
        scores    : pd.Series, index=馬番
        threshold : このスコア差未満は「近似」と判定
        penalty   : 近似馬への乗数 (0 < penalty < 1)
        n         : 返す頭数

    Returns:
        list[int]: reranked top-n 馬番
    """
    scores = _validate_scores(scores)
    sorted_s = scores.sort_values(ascending=False)
    penalized = sorted_s.copy().astype(float)

    vals = sorted_s.values
    for i in range(1, len(vals)):
        diff = abs(float(vals[i - 1]) - float(vals[i]))
        if diff < threshold:
            penalized.iloc[i] = penalized.iloc[i] * penalty

    return _top_n(penalized, n)


# ---------------------------------------------------------------------------
# Rerank d: score + rank average
# ---------------------------------------------------------------------------

def rerank_rank_average(scores: pd.Series, n: int = 5) -> list[int]:
    """Rank average rerank.

    final_rank = (raw_score_rank + normalized_score_rank) / 2

    normalized_score_rank: スコアを [0, 1] 正規化した値でランク付け。
    raw_score_rank       : 元のスコア値でランク付け。

    ノイズを相殺しつつ相対順位を安定させる効果を狙う。

    Args:
        scores : pd.Series, index=馬番
        n      : 返す頭数

    Returns:
        list[int]: reranked top-n 馬番
    """
    scores = _validate_scores(scores)
    if len(scores) == 0:
        return []
    s = scores.copy().astype(float)

    # raw rank (lower = better)
    raw_rank = s.rank(ascending=False, method='average')

    # normalize then rank
    s_min, s_max = s.min(), s.max()
    if s_max > s_min:
        s_norm = (s - s_min) / (s_max - s_min)
    else:
        s_norm = pd.Series(np.ones(len(s)), index=s.index)
    norm_rank = s_norm.rank(ascending=False, method='average')

    # average rank → sort ascending (lower = better) → take top-n
    avg_rank = (raw_rank + norm_rank) / 2.0
    top_idx = avg_rank.sort_values(ascending=True).head(n).index
    return list(top_idx.astype(int))


# ---------------------------------------------------------------------------
# Rerank e: condition-aware
# ---------------------------------------------------------------------------

def rerank_condition_aware(
    scores: pd.Series,
    condition: str,
    n: int = 5,
) -> list[int]:
    """Condition-aware rerank.

    条件別に top1 重視 / top3-5 多様化の戦略を切り替える。

    条件 A/E  → top1 強調 (T=1.5: scores 尖らせ)
    条件 C    → top1 重視 (T=1.2)
    条件 D/X  → top3-5 多様化 (T=0.7: scores 均等化)
    条件 B    → 変化なし (T=1.0)
    その他    → 変化なし (T=1.0)

    根拠:
      - 条件 A/C は大荷差 (8-14頭/1600m+/良稍): AI確率が高い馬が圧倒的に強い
      - 条件 D (短距離): 混戦傾向、上位候補を広く取る
      - 条件 X (15頭+/重不良): 荒れやすく穴馬が台頭しやすい

    Args:
        scores    : pd.Series, index=馬番
        condition : 'A'|'B'|'C'|'D'|'E'|'X'
        n         : 返す頭数

    Returns:
        list[int]: reranked top-n 馬番
    """
    T_MAP = {
        'A': 1.5,
        'B': 1.0,
        'C': 1.2,
        'D': 0.7,
        'E': 1.5,
        'X': 0.7,
    }
    T = T_MAP.get(str(condition).upper(), 1.0)
    return rerank_tempering(scores, T=T, n=n)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

_METHOD_MAP = {
    'tempering': rerank_tempering,
    'odds_divergence': rerank_odds_divergence,
    'spread': rerank_spread,
    'rank_average': rerank_rank_average,
    'condition_aware': rerank_condition_aware,
    'baseline': _top_n,  # 変化なし (比較用)
}

VALID_METHODS = list(_METHOD_MAP.keys())


def get_top5_reranked(
    scores: pd.Series,
    method: str = 'tempering',
    odds: Optional[pd.Series] = None,
    condition: str = 'A',
    n: int = 5,
    **kwargs,
) -> list[int]:
    """統一エントリポイント。

    Args:
        scores    : pd.Series, index=馬番, values=スコア(float)
                    predict_core.py の df['スコア'] を Series 化して渡す。
                    例: scores = df.set_index('馬番')['スコア']
        method    : 'tempering' | 'odds_divergence' | 'spread' |
                    'rank_average' | 'condition_aware' | 'baseline'
        odds      : pd.Series, index=馬番, values=単勝オッズ  (method='odds_divergence' 時に必須)
        condition : 'A'-'X'  (method='condition_aware' 時に使用)
        n         : top-n 頭数 (default=5)
        **kwargs  : 各 rerank 関数へのパラメータ (T, alpha, threshold 等)

    Returns:
        list[int]: reranked top-n 馬番リスト

    Example:
        import pandas as pd
        from tools.v15_top5_rerank import get_top5_reranked

        # predict_core.py の predict_race() 後:
        #   df_sorted = predict_race(df, model_data, ...)
        scores = df_sorted.set_index('馬番')['スコア']

        top5 = get_top5_reranked(scores, method='condition_aware', condition='D')
    """
    if method not in _METHOD_MAP:
        raise ValueError(f"method must be one of {VALID_METHODS}, got '{method}'")

    scores = _validate_scores(scores)

    if method == 'baseline':
        return _top_n(scores, n)
    if method == 'tempering':
        T = kwargs.get('T', 0.8)
        return rerank_tempering(scores, T=T, n=n)
    if method == 'odds_divergence':
        alpha = kwargs.get('alpha', 0.5)
        return rerank_odds_divergence(scores, odds, alpha=alpha, n=n)
    if method == 'spread':
        threshold = kwargs.get('threshold', 0.02)
        penalty = kwargs.get('penalty', 0.5)
        return rerank_spread(scores, threshold=threshold, penalty=penalty, n=n)
    if method == 'rank_average':
        return rerank_rank_average(scores, n=n)
    if method == 'condition_aware':
        return rerank_condition_aware(scores, condition=condition, n=n)

    # fallback (unreachable)
    return _top_n(scores, n)


# ---------------------------------------------------------------------------
# Simulation helper (paper backtest 用)
# ---------------------------------------------------------------------------

def simulate_rerank_hit_rate(
    scores_list: list[pd.Series],
    actual_top3_list: list[list[int]],
    method: str = 'tempering',
    n: int = 5,
    **kwargs,
) -> dict:
    """Paper simulation: top-n hit rate comparison.

    hit = actual top3 の全馬が predicted top-n に含まれる割合 (trio coverage)

    Args:
        scores_list      : list of pd.Series per race
        actual_top3_list : list of [1着馬番, 2着馬番, 3着馬番] per race
        method           : rerank method
        n                : top-n (default 5)
        **kwargs         : rerank params

    Returns:
        dict with keys: hit_rate, n_races, method
    """
    hits = 0
    valid = 0
    for scores, actual_top3 in zip(scores_list, actual_top3_list):
        try:
            reranked = get_top5_reranked(scores, method=method, n=n, **kwargs)
        except Exception:
            continue
        if len(actual_top3) < 3:
            continue
        reranked_set = set(reranked)
        if all(h in reranked_set for h in actual_top3[:3]):
            hits += 1
        valid += 1

    return {
        'method': method,
        'n': n,
        'n_races': valid,
        'hits': hits,
        'hit_rate': hits / valid if valid > 0 else 0.0,
    }
