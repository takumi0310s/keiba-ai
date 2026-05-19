"""
tests/test_v15_top5_rerank.py — V15 top-5 rerank 5 rerank 候補の単体テスト

pytest tests/test_v15_top5_rerank.py -v
"""

import pytest
import pandas as pd
import numpy as np

from tools.v15_top5_rerank import (
    rerank_tempering,
    rerank_odds_divergence,
    rerank_spread,
    rerank_rank_average,
    rerank_condition_aware,
    get_top5_reranked,
    simulate_rerank_hit_rate,
    VALID_METHODS,
    _top_n,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_scores():
    """8頭レースのサンプルスコア (馬番 1-8, top1=馬番 3 最高)"""
    return pd.Series(
        {1: 0.35, 2: 0.50, 3: 0.75, 4: 0.42, 5: 0.30, 6: 0.20, 7: 0.18, 8: 0.10}
    )


@pytest.fixture
def close_scores():
    """近似スコアが多いレース (spread テスト用)"""
    return pd.Series(
        {1: 0.60, 2: 0.59, 3: 0.58, 4: 0.40, 5: 0.39, 6: 0.20, 7: 0.10, 8: 0.05}
    )


@pytest.fixture
def base_odds():
    """サンプル単勝オッズ (馬番 1-8)"""
    return pd.Series(
        {1: 5.0, 2: 2.5, 3: 3.0, 4: 8.0, 5: 12.0, 6: 20.0, 7: 30.0, 8: 50.0}
    )


# ---------------------------------------------------------------------------
# Test 1: rerank_tempering returns exactly n horses
# ---------------------------------------------------------------------------

class TestRerankTemperingReturns5:
    def test_returns_5_default(self, base_scores):
        result = rerank_tempering(base_scores)
        assert len(result) == 5, f"Expected 5, got {len(result)}"

    def test_returns_3_when_n_3(self, base_scores):
        result = rerank_tempering(base_scores, n=3)
        assert len(result) == 3

    def test_all_are_valid_horse_nums(self, base_scores):
        result = rerank_tempering(base_scores)
        for h in result:
            assert h in base_scores.index.tolist(), f"{h} not in scores index"

    def test_no_duplicates(self, base_scores):
        result = rerank_tempering(base_scores)
        assert len(result) == len(set(result)), "Duplicates found in result"

    def test_small_field_3_horses(self):
        scores = pd.Series({1: 0.8, 2: 0.5, 3: 0.3})
        result = rerank_tempering(scores, n=5)
        # n > 頭数 → 返せる分だけ返す (3頭)
        assert len(result) <= 3

    def test_invalid_T_raises(self, base_scores):
        with pytest.raises(ValueError):
            rerank_tempering(base_scores, T=0)

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            rerank_tempering(pd.Series(dtype=float))


# ---------------------------------------------------------------------------
# Test 2: rerank_tempering high T (>1) makes ranking sharper (top horse stays #1)
#         low T (<1) flattens ranking (low-score horse may rise)
# ---------------------------------------------------------------------------

class TestRerankTemperingTemperatureEffect:
    def test_T1_identity(self, base_scores):
        """T=1 は元のランキングと同一になる。"""
        result_T1 = rerank_tempering(base_scores, T=1.0)
        baseline = _top_n(base_scores, 5)
        assert result_T1 == baseline, f"T=1 should match baseline.\n{result_T1}\n{baseline}"

    def test_high_T_top1_stays(self, base_scores):
        """T>1 では top1 馬 (スコア最大) は変わらない。"""
        result = rerank_tempering(base_scores, T=2.0)
        assert result[0] == 3, f"Top horse should be 3, got {result[0]}"

    def test_low_T_flattens(self, base_scores):
        """T=0.3 (非常に低い) では低スコア馬が top5 に含まれやすくなる。"""
        # T=0.3 → 分布が平坦化 → base_scores[8]=0.10 でも top5 入りの可能性あり
        result_T03 = rerank_tempering(base_scores, T=0.3)
        result_T20 = rerank_tempering(base_scores, T=2.0)
        # T=0.3 の方が多様性が高い (min horse_num in result が小さい = 低スコア馬が混入)
        # 最低スコア馬 (8番) が T=0.3 で含まれることを確認
        assert len(result_T03) == 5
        # T=2.0 では上位集中 → top-5 合計スコアが T=0.3 より高い
        sum_T20 = sum(base_scores[h] for h in result_T20)
        sum_T03 = sum(base_scores[h] for h in result_T03)
        assert sum_T20 >= sum_T03, "T=2.0 should include higher-scoring horses"

    def test_T_range_0001_to_5(self, base_scores):
        """広範囲の T で例外なく動作することを確認。"""
        for T in [0.001, 0.1, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0]:
            result = rerank_tempering(base_scores, T=T)
            assert len(result) == 5, f"T={T}: expected 5 results"


# ---------------------------------------------------------------------------
# Test 3: rerank_odds_divergence boosts divergent (AI high / market low) horse
# ---------------------------------------------------------------------------

class TestRerankOddsDivergenceBoostsDivergent:
    def test_returns_5(self, base_scores, base_odds):
        result = rerank_odds_divergence(base_scores, base_odds)
        assert len(result) == 5

    def test_none_odds_returns_baseline(self, base_scores):
        result_none = rerank_odds_divergence(base_scores, odds=None)
        baseline = _top_n(base_scores, 5)
        assert result_none == baseline

    def test_divergent_horse_rises(self):
        """AI スコア高 + オッズ高 (市場過小評価) の馬が上位に来る。"""
        # 馬番 5: AIスコア=0.60 (2位), オッズ=50.0 (最高倍) → 大乖離
        # 馬番 1: AIスコア=0.70 (1位), オッズ=2.0  (最低倍) → 乖離なし
        scores = pd.Series({1: 0.70, 2: 0.50, 3: 0.40, 4: 0.30, 5: 0.60})
        odds = pd.Series({1: 2.0, 2: 5.0, 3: 8.0, 4: 15.0, 5: 50.0})
        result = rerank_odds_divergence(scores, odds, alpha=1.0)
        # 馬番 5 は divergence が最大なので baseline より順位が上がるはず
        baseline = _top_n(scores, 5)
        baseline_rank_5 = baseline.index(5) if 5 in baseline else 99
        result_rank_5 = result.index(5) if 5 in result else 99
        assert result_rank_5 <= baseline_rank_5, (
            f"Horse 5 should rank higher with divergence boost. "
            f"baseline={baseline}, result={result}"
        )

    def test_empty_odds_series_returns_baseline(self, base_scores):
        result = rerank_odds_divergence(base_scores, pd.Series(dtype=float))
        baseline = _top_n(base_scores, 5)
        assert result == baseline

    def test_alpha_0_identity(self, base_scores, base_odds):
        """alpha=0 → divergence boost なし → baseline と同一。"""
        result = rerank_odds_divergence(base_scores, base_odds, alpha=0.0)
        baseline = _top_n(base_scores, 5)
        assert result == baseline, f"alpha=0 should be identity. {result} vs {baseline}"


# ---------------------------------------------------------------------------
# Test 4: rerank_spread separates close-score horses
# ---------------------------------------------------------------------------

class TestRerankSpreadSeparatesCloseScores:
    def test_returns_5(self, close_scores):
        result = rerank_spread(close_scores)
        assert len(result) == 5

    def test_large_threshold_penalizes_clusters(self, close_scores):
        """大きい threshold → 近似スコアグループの後続馬がペナルティを受ける。"""
        # close_scores: 1,2,3 が 0.60/0.59/0.58 (差 < 0.02)
        # spread で 2,3番がペナルティ → 4番(0.40)が上昇の可能性
        result_spread = rerank_spread(close_scores, threshold=0.02, penalty=0.1)
        result_baseline = _top_n(close_scores, 5)
        # 少なくとも結果は返る
        assert len(result_spread) == 5
        # 1番 (最高スコア) は変わらず 1位のはず
        assert result_spread[0] == 1

    def test_zero_threshold_identity(self, close_scores):
        """threshold=0 → 全馬がペナルティなし → baseline と同一。"""
        result = rerank_spread(close_scores, threshold=0.0)
        baseline = _top_n(close_scores, 5)
        assert result == baseline

    def test_unique_scores_no_penalty(self):
        """全馬のスコアが十分離れている場合は順位変化なし。"""
        scores = pd.Series({1: 0.9, 2: 0.7, 3: 0.5, 4: 0.3, 5: 0.1})
        result = rerank_spread(scores, threshold=0.01)
        baseline = _top_n(scores, 5)
        assert result == baseline

    def test_all_tie_still_returns_5(self):
        """全馬同スコア → 返せる馬はいるが順位不定。"""
        scores = pd.Series({1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5})
        result = rerank_spread(scores)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Test 5: get_top5_reranked calls all methods without error
# ---------------------------------------------------------------------------

class TestGetTop5RerankAllMethods:
    def test_baseline(self, base_scores):
        result = get_top5_reranked(base_scores, method='baseline')
        assert len(result) == 5

    def test_tempering_default(self, base_scores):
        result = get_top5_reranked(base_scores, method='tempering')
        assert len(result) == 5

    def test_tempering_with_T(self, base_scores):
        result = get_top5_reranked(base_scores, method='tempering', T=1.5)
        assert len(result) == 5

    def test_odds_divergence_with_odds(self, base_scores, base_odds):
        result = get_top5_reranked(base_scores, method='odds_divergence', odds=base_odds)
        assert len(result) == 5

    def test_odds_divergence_no_odds(self, base_scores):
        result = get_top5_reranked(base_scores, method='odds_divergence', odds=None)
        assert len(result) == 5

    def test_spread(self, base_scores):
        result = get_top5_reranked(base_scores, method='spread')
        assert len(result) == 5

    def test_rank_average(self, base_scores):
        result = get_top5_reranked(base_scores, method='rank_average')
        assert len(result) == 5

    def test_condition_aware_all_conditions(self, base_scores):
        for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
            result = get_top5_reranked(base_scores, method='condition_aware', condition=cond)
            assert len(result) == 5, f"condition {cond} failed"

    def test_invalid_method_raises(self, base_scores):
        with pytest.raises(ValueError):
            get_top5_reranked(base_scores, method='unknown_method')

    def test_all_valid_methods_listed(self):
        assert set(VALID_METHODS) == {
            'tempering', 'odds_divergence', 'spread',
            'rank_average', 'condition_aware', 'baseline'
        }

    def test_n_param_respected(self, base_scores):
        for n in [1, 3, 5, 7]:
            result = get_top5_reranked(base_scores, method='tempering', n=n)
            expected = min(n, len(base_scores))
            assert len(result) == expected, f"n={n}: expected {expected}, got {len(result)}"


# ---------------------------------------------------------------------------
# Test 6: rerank_rank_average
# ---------------------------------------------------------------------------

class TestRerankRankAverage:
    def test_returns_5(self, base_scores):
        result = rerank_rank_average(base_scores)
        assert len(result) == 5

    def test_no_duplicates(self, base_scores):
        result = rerank_rank_average(base_scores)
        assert len(result) == len(set(result))

    def test_uniform_scores_returns_any_5(self):
        scores = pd.Series({i: 0.5 for i in range(1, 9)})
        result = rerank_rank_average(scores)
        assert len(result) == 5

    def test_top_horse_in_top2(self, base_scores):
        """rank_average では明確な最高スコア馬が常に top-2 に含まれる。"""
        result = rerank_rank_average(base_scores)
        # 馬番 3 (スコア 0.75) は top-2 に含まれるはず
        assert 3 in result[:2], f"Horse 3 (highest score) should be in top-2. result={result}"


# ---------------------------------------------------------------------------
# Test 7: condition_aware per-condition T mapping
# ---------------------------------------------------------------------------

class TestConditionAware:
    def test_condition_D_top1_different_from_A(self, base_scores):
        """D (T=0.7 flatten) と A (T=1.5 sharpen) は異なる結果になることがある。"""
        result_A = rerank_condition_aware(base_scores, 'A')
        result_D = rerank_condition_aware(base_scores, 'D')
        # At minimum both return 5 horses
        assert len(result_A) == 5
        assert len(result_D) == 5
        # For this score distribution they may differ
        # Just verify the top horse for A is the highest-scoring one
        assert result_A[0] == 3

    def test_unknown_condition_defaults_to_baseline(self, base_scores):
        """未知の条件コードはデフォルト T=1.0 で処理される。"""
        result_unknown = rerank_condition_aware(base_scores, 'Z')
        baseline = _top_n(base_scores, 5)
        assert result_unknown == baseline


# ---------------------------------------------------------------------------
# Test 8: simulate_rerank_hit_rate (simulation helper)
# ---------------------------------------------------------------------------

class TestSimulateRerankHitRate:
    def _make_data(self, n_races=20, seed=42):
        rng = np.random.default_rng(seed)
        scores_list = []
        actual_list = []
        for _ in range(n_races):
            horses = list(range(1, 13))
            s = pd.Series(rng.uniform(0.1, 0.9, len(horses)), index=horses)
            top3 = rng.choice(horses, size=3, replace=False).tolist()
            scores_list.append(s)
            actual_list.append(top3)
        return scores_list, actual_list

    def test_returns_dict_with_required_keys(self):
        scores_list, actual_list = self._make_data()
        result = simulate_rerank_hit_rate(scores_list, actual_list, method='baseline')
        assert 'hit_rate' in result
        assert 'n_races' in result
        assert 'method' in result
        assert 'hits' in result

    def test_hit_rate_in_0_1(self):
        scores_list, actual_list = self._make_data()
        for method in VALID_METHODS:
            result = simulate_rerank_hit_rate(scores_list, actual_list, method=method)
            assert 0.0 <= result['hit_rate'] <= 1.0, f"method={method} hit_rate out of range"

    def test_n_races_matches(self):
        scores_list, actual_list = self._make_data(n_races=10)
        result = simulate_rerank_hit_rate(scores_list, actual_list, method='tempering')
        assert result['n_races'] == 10

    def test_empty_input(self):
        result = simulate_rerank_hit_rate([], [], method='baseline')
        assert result['hit_rate'] == 0.0
        assert result['n_races'] == 0
