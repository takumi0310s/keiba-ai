"""tests/test_v15_market_divergence_v2.py

最低 3 test:
  - test_divergence_score_range  : pop_rank 1-5+ で 0.0-1.0 範囲内
  - test_skip_on_low_divergence  : score < 0.3 で get_paper_formation が None 返却
  - test_backtest_runs           : csv があれば例外なく完走

追加テスト:
  - test_divergence_score_values : 各 pop_rank の期待値を確認
  - test_classify_action         : threshold ごとの action 分類
  - test_formation_structure     : formation 返却時の構造確認
  - test_backtest_proxy_mode     : pop_rank 列なし CSV でも完走
"""
from __future__ import annotations

import csv
import io
import os
import tempfile
from pathlib import Path

import pytest

# test 対象をインポート
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

from v15_market_divergence_v2 import (
    backtest_divergence_v2,
    classify_action,
    compute_divergence_score,
    get_paper_formation,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO / 'data' / 'cumulative_results.csv'


# ---------------------------------------------------------------------------
# compute_divergence_score
# ---------------------------------------------------------------------------

class TestDivergenceScoreRange:
    """test_divergence_score_range: pop_rank 1-5+ で 0.0-1.0 範囲内"""

    def test_all_ranks_in_range(self):
        for rank in range(1, 18 + 1):
            score = compute_divergence_score(rank)
            assert 0.0 <= score <= 1.0, f"pop_rank={rank} score={score} out of [0,1]"

    def test_unknown_rank_zero_returns_default(self):
        score = compute_divergence_score(0)
        assert 0.0 <= score <= 1.0

    def test_negative_rank_returns_default(self):
        score = compute_divergence_score(-1)
        assert 0.0 <= score <= 1.0


class TestDivergenceScoreValues:
    """各 pop_rank の期待スコア値を確認"""

    def test_rank_1_is_zero(self):
        assert compute_divergence_score(1) == 0.0

    def test_rank_2(self):
        assert compute_divergence_score(2) == 0.3

    def test_rank_3(self):
        assert compute_divergence_score(3) == 0.6

    def test_rank_4(self):
        assert compute_divergence_score(4) == 0.8

    def test_rank_5_plus_is_one(self):
        for rank in [5, 6, 7, 10, 18]:
            assert compute_divergence_score(rank) == 1.0, f"rank={rank} should be 1.0"

    def test_monotone_non_decreasing(self):
        """rank が上がるほど score は非減少"""
        scores = [compute_divergence_score(r) for r in range(1, 8)]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], f"not monotone at rank {i+1}"


# ---------------------------------------------------------------------------
# get_paper_formation
# ---------------------------------------------------------------------------

class TestSkipOnLowDivergence:
    """test_skip_on_low_divergence: score < 0.3 で None 返却"""

    def _make_preds(self, pop_rank_top1):
        return [
            {'horse_num': i + 1, 'pop_rank': pop_rank_top1 if i == 0 else i + 2}
            for i in range(6)
        ]

    def test_rank1_returns_none(self):
        preds = self._make_preds(1)
        result = get_paper_formation(preds, {}, divergence_threshold=0.3)
        assert result is None

    def test_rank2_with_threshold_04_returns_none(self):
        """rank=2 score=0.3 で threshold=0.4 なら skip"""
        preds = self._make_preds(2)
        result = get_paper_formation(preds, {}, divergence_threshold=0.4)
        assert result is None

    def test_rank2_with_threshold_03_returns_formation(self):
        """rank=2 score=0.3 で threshold=0.3 なら bet (score >= threshold)"""
        preds = self._make_preds(2)
        result = get_paper_formation(preds, {}, divergence_threshold=0.3)
        assert result is not None

    def test_insufficient_predictions_returns_none(self):
        preds = [{'horse_num': 1, 'pop_rank': 5}]
        result = get_paper_formation(preds, {}, divergence_threshold=0.3)
        assert result is None

    def test_empty_predictions_returns_none(self):
        result = get_paper_formation([], {}, divergence_threshold=0.3)
        assert result is None


class TestFormationStructure:
    """formation 返却時の構造確認"""

    def _make_preds(self, pop_rank=5):
        return [
            {'horse_num': i + 1, 'pop_rank': pop_rank if i == 0 else i + 5}
            for i in range(6)
        ]

    def test_returns_list(self):
        preds = self._make_preds(5)
        result = get_paper_formation(preds, {}, divergence_threshold=0.3)
        assert isinstance(result, list)

    def test_each_element_is_tuple_or_list_of_3(self):
        preds = self._make_preds(5)
        result = get_paper_formation(preds, {}, divergence_threshold=0.3)
        assert result is not None
        for bet in result:
            assert len(bet) == 3, f"bet {bet} should have 3 elements"

    def test_up_to_7_bets(self):
        preds = self._make_preds(5)
        result = get_paper_formation(preds, {}, divergence_threshold=0.3)
        assert result is not None
        assert len(result) <= 7

    def test_bets_are_sorted(self):
        preds = self._make_preds(5)
        result = get_paper_formation(preds, {}, divergence_threshold=0.3)
        assert result is not None
        for bet in result:
            assert list(bet) == sorted(bet), f"bet {bet} is not sorted"

    def test_top1_in_every_bet(self):
        preds = self._make_preds(5)
        result = get_paper_formation(preds, {}, divergence_threshold=0.3)
        top1_num = preds[0]['horse_num']
        assert result is not None
        for bet in result:
            assert top1_num in bet, f"top1 {top1_num} missing in bet {bet}"


# ---------------------------------------------------------------------------
# classify_action
# ---------------------------------------------------------------------------

class TestClassifyAction:
    def test_below_threshold_is_skip(self):
        assert classify_action(0.0, 0.3) == 'skip'
        assert classify_action(0.29, 0.3) == 'skip'

    def test_at_threshold_is_standard(self):
        assert classify_action(0.3, 0.3) == 'standard'

    def test_mid_range_is_standard(self):
        assert classify_action(0.5, 0.3) == 'standard'
        assert classify_action(0.7, 0.3) == 'standard'

    def test_above_07_is_boost(self):
        assert classify_action(0.8, 0.3) == 'boost'
        assert classify_action(1.0, 0.3) == 'boost'


# ---------------------------------------------------------------------------
# backtest_divergence_v2
# ---------------------------------------------------------------------------

class TestBacktestRuns:
    """test_backtest_runs: csv があれば例外なく完走"""

    def test_real_csv_if_exists(self):
        """実データ CSV があれば例外なく完走する"""
        if not DEFAULT_CSV.exists():
            pytest.skip('cumulative_results.csv not found')

        result = backtest_divergence_v2(
            csv_path=DEFAULT_CSV,
            thresholds=[0.3, 0.5, 0.7],
        )
        assert isinstance(result, dict)
        assert 'results' in result
        assert 'has_pop_rank' in result
        assert 'proxy_mode' in result

    def test_result_keys_present(self):
        """結果 dict に必須キーが存在する"""
        if not DEFAULT_CSV.exists():
            pytest.skip('cumulative_results.csv not found')

        result = backtest_divergence_v2(DEFAULT_CSV, thresholds=[0.3])
        if result['results']:
            r = result['results']['0.3']
            for key in ('n_bet', 'n_skip', 'n_hit', 'hit_rate', 'skip_rate', 'roi', 'pnl'):
                assert key in r, f"key '{key}' missing in result"

    def test_roi_is_numeric(self):
        if not DEFAULT_CSV.exists():
            pytest.skip('cumulative_results.csv not found')

        result = backtest_divergence_v2(DEFAULT_CSV, thresholds=[0.3])
        if result['results']:
            assert isinstance(result['results']['0.3']['roi'], float)


class TestBacktestProxyMode:
    """test_backtest_proxy_mode: pop_rank 列なし CSV でも完走"""

    def _make_csv(self, rows: list[dict]) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, encoding='utf-8', newline=''
        )
        if not rows:
            tmp.write('')
            tmp.close()
            return Path(tmp.name)
        writer = csv.DictWriter(tmp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        tmp.close()
        return Path(tmp.name)

    def test_no_pop_rank_column(self):
        """pop_rank 列なし CSV で proxy モードが有効になる"""
        rows = [
            {
                'race_id': '2026060501011',
                'status': 'settled',
                'date': '20260505',
                'top1_finish': '2',
                'trio_hit': '0',
                'trio_payout': '0',
                'trio_bets_str': '1-3-5',
                'trio_result': '2-4-6',
                'top2_finish': '4',
                'top3_finish': '6',
            }
        ] * 5
        csv_path = self._make_csv(rows)
        try:
            result = backtest_divergence_v2(csv_path, thresholds=[0.3])
            assert result['proxy_mode'] is True
            assert result['has_pop_rank'] is False
            assert isinstance(result['results'], dict)
        finally:
            os.unlink(csv_path)

    def test_with_pop_rank_column(self):
        """pop_rank 列あり CSV で has_pop_rank=True"""
        rows = [
            {
                'race_id': '2026060501011',
                'status': 'settled',
                'date': '20260505',
                'pop_rank': '4',
                'top1_finish': '1',
                'trio_hit': '1',
                'trio_payout': '1500',
                'trio_bets_str': '1-3-5',
                'trio_result': '1-3-5',
                'top2_finish': '3',
                'top3_finish': '5',
            }
        ] * 3
        csv_path = self._make_csv(rows)
        try:
            result = backtest_divergence_v2(csv_path, thresholds=[0.3])
            assert result['has_pop_rank'] is True
            assert result['proxy_mode'] is False
        finally:
            os.unlink(csv_path)

    def test_empty_settled_rows(self):
        """settled 行ゼロでも例外なく返す"""
        rows = [
            {
                'race_id': '2026060501011',
                'status': 'pending',
                'date': '20260505',
                'top1_finish': '',
                'trio_hit': '',
                'trio_payout': '0',
                'trio_bets_str': '',
                'trio_result': '',
                'top2_finish': '',
                'top3_finish': '',
            }
        ]
        csv_path = self._make_csv(rows)
        try:
            result = backtest_divergence_v2(csv_path, thresholds=[0.3])
            assert isinstance(result, dict)
            # settled なしなら results は空
            assert result['results'] == {}
        finally:
            os.unlink(csv_path)
