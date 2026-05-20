"""tests for tools/v15_top5_score_weight.py"""
import os
import sys
import tempfile
import csv

import pytest

# プロジェクトルートをパスに追加
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools.v15_top5_score_weight import (
    weight_change_factor,
    compute_weighted_score,
    rank_by_weighted_score,
    get_paper_formation,
    build_trio_formation,
    backtest_on_csv,
)


# ---------------------------------------------------------------------------
# test_weight_change_factor
# ---------------------------------------------------------------------------

class TestWeightChangeFactor:
    def test_within_5kg_returns_1(self):
        assert weight_change_factor(0.0) == 1.0
        assert weight_change_factor(5.0) == 1.0
        assert weight_change_factor(-5.0) == 1.0

    def test_between_5_and_10kg_returns_09(self):
        assert weight_change_factor(6.0) == 0.9
        assert weight_change_factor(-10.0) == 0.9
        assert weight_change_factor(10.0) == 0.9

    def test_above_10kg_returns_08(self):
        assert weight_change_factor(11.0) == 0.8
        assert weight_change_factor(-15.0) == 0.8
        assert weight_change_factor(100.0) == 0.8

    def test_none_returns_1(self):
        assert weight_change_factor(None) == 1.0

    def test_invalid_string_returns_1(self):
        assert weight_change_factor('abc') == 1.0

    def test_numeric_string_works(self):
        assert weight_change_factor('8') == 0.9


# ---------------------------------------------------------------------------
# test_score_ordering
# ---------------------------------------------------------------------------

class TestScoreOrdering:
    """weighted_score 降順で top-5 が選ばれるか"""

    def test_order_descending(self):
        preds = [
            {'num': 5, 'score': 0.50},
            {'num': 1, 'score': 0.80},
            {'num': 3, 'score': 0.65},
            {'num': 7, 'score': 0.45},
            {'num': 9, 'score': 0.70},
        ]
        ranked = rank_by_weighted_score(preds)
        scores = [r['weighted_score'] for r in ranked]
        assert scores == sorted(scores, reverse=True), "weighted_score must be descending"

    def test_top1_is_highest_score(self):
        preds = [
            {'num': 2, 'score': 0.60},
            {'num': 4, 'score': 0.90},
            {'num': 6, 'score': 0.30},
        ]
        ranked = rank_by_weighted_score(preds)
        assert int(ranked[0]['num']) == 4

    def test_weight_change_reorders(self):
        """大きな体重変動があると score が高くても順位が下がる"""
        preds = [
            {'num': 1, 'score': 0.80, 'weight_change': 20.0},   # factor 0.8 → ws=0.64
            {'num': 2, 'score': 0.75, 'weight_change': 0.0},    # factor 1.0 → ws=0.75
        ]
        ranked = rank_by_weighted_score(preds)
        assert int(ranked[0]['num']) == 2, "Horse 2 should rank first after weight penalty"

    def test_weighted_score_in_output(self):
        preds = [{'num': 1, 'score': 0.5}]
        ranked = rank_by_weighted_score(preds)
        assert 'weighted_score' in ranked[0]

    def test_empty_list_returns_empty(self):
        assert rank_by_weighted_score([]) == []


# ---------------------------------------------------------------------------
# test_formation_output_format
# ---------------------------------------------------------------------------

class TestFormationOutputFormat:
    def _make_preds(self, nums, scores):
        return [{'num': n, 'score': s} for n, s in zip(nums, scores)]

    def test_format_contains_dashes_and_semicolons(self):
        preds = self._make_preds([1, 3, 5, 7, 9], [0.9, 0.8, 0.7, 0.6, 0.5])
        formation = get_paper_formation(preds)
        assert '-' in formation, "formation should contain '-'"
        assert ';' in formation, "formation should contain ';'"

    def test_all_bets_are_ascending(self):
        preds = self._make_preds([9, 7, 5, 3, 1], [0.9, 0.8, 0.7, 0.6, 0.5])
        formation = get_paper_formation(preds)
        bets = [b.strip() for b in formation.split(';')]
        for bet in bets:
            parts = [int(x) for x in bet.split('-')]
            assert parts == sorted(parts), f"bet {bet} not in ascending order"

    def test_up_to_7_bets(self):
        preds = self._make_preds([1, 3, 5, 7, 9], [0.9, 0.8, 0.7, 0.6, 0.5])
        formation = get_paper_formation(preds)
        bets = [b.strip() for b in formation.split(';') if b.strip()]
        assert len(bets) <= 7

    def test_top1_appears_in_all_bets(self):
        preds = [
            {'num': 2, 'score': 0.9},
            {'num': 4, 'score': 0.8},
            {'num': 6, 'score': 0.7},
            {'num': 8, 'score': 0.6},
            {'num': 10, 'score': 0.5},
        ]
        ranked = rank_by_weighted_score(preds)
        top1 = int(ranked[0]['num'])
        formation = get_paper_formation(preds)
        bets = [b.strip() for b in formation.split(';') if b.strip()]
        for bet in bets:
            nums = [int(x) for x in bet.split('-')]
            assert top1 in nums, f"top1 ({top1}) should appear in bet {bet}"


# ---------------------------------------------------------------------------
# test_get_paper_formation — edge cases
# ---------------------------------------------------------------------------

class TestGetPaperFormation:
    def test_empty_predictions_no_crash(self):
        result = get_paper_formation([])
        assert result == ''

    def test_single_horse_no_crash(self):
        result = get_paper_formation([{'num': 1, 'score': 0.9}])
        # 3頭未満なので formation 組めない → '' か空
        assert isinstance(result, str)

    def test_two_horses_no_crash(self):
        result = get_paper_formation([
            {'num': 1, 'score': 0.9},
            {'num': 2, 'score': 0.8},
        ])
        assert isinstance(result, str)

    def test_missing_weight_change_no_crash(self):
        preds = [
            {'num': 1, 'score': 0.9},
            {'num': 3, 'score': 0.8},
            {'num': 5, 'score': 0.7},
        ]
        result = get_paper_formation(preds)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_none_weight_change_graceful(self):
        preds = [
            {'num': 1, 'score': 0.9, 'weight_change': None},
            {'num': 3, 'score': 0.8, 'weight_change': None},
            {'num': 5, 'score': 0.7, 'weight_change': None},
        ]
        result = get_paper_formation(preds)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# test_backtest_runs
# ---------------------------------------------------------------------------

class TestBacktestRuns:
    def _write_minimal_csv(self, path, rows):
        """最小限の cumulative_results.csv を書き出す。"""
        fieldnames = [
            'race_id', 'course', 'race_num', 'race_name', 'condition',
            'num_horses', 'distance', 'surface', 'track_condition',
            'top1_num', 'top1_name', 'top1_score',
            'top2_num', 'top3_num',
            'top1_finish', 'top2_finish', 'top3_finish',
            'trio_bets', 'trio_result', 'bet_type',
            'trio_hit', 'trio_payout', 'umaren_payout',
            'actual_payout', 'investment', 'profit', 'status', 'date',
            'umaren_hit', 'trio_bets_str',
            'top2_name', 'top2_score', 'top3_name', 'top3_score', 'top4_num',
        ]
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                full_row = {k: '' for k in fieldnames}
                full_row.update(row)
                writer.writerow(full_row)

    def _make_row(self, race_id='202601010101', status='settled', bet_type='trio',
                  n1=1, n2=3, n3=5, s1=0.8, s2=0.7, s3=0.6,
                  f1=1, f2=2, f3=3, trio_result='1-3-5', trio_payout=1500):
        return {
            'race_id': race_id,
            'status': status,
            'bet_type': bet_type,
            'top1_num': n1, 'top1_score': s1,
            'top2_num': n2, 'top2_score': s2,
            'top3_num': n3, 'top3_score': s3,
            'top1_finish': f1, 'top2_finish': f2, 'top3_finish': f3,
            'trio_result': trio_result,
            'trio_payout': trio_payout,
            'investment': 700,
        }

    def test_backtest_runs_on_real_csv(self):
        """実 CSV が存在すれば例外なく完走する"""
        csv_path = os.path.join(REPO, 'data', 'cumulative_results.csv')
        if not os.path.exists(csv_path):
            pytest.skip('cumulative_results.csv not found')
        result = backtest_on_csv(csv_path)
        assert isinstance(result, dict)
        assert 'n_valid' in result
        assert 'original_hit3' in result
        assert 'weighted_hit3' in result

    def test_backtest_runs_on_synthetic_csv(self):
        """合成 CSV でも例外なく完走する"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            tmp_path = f.name
        try:
            rows = [
                self._make_row('race001', trio_result='1-3-5', trio_payout=1500),
                self._make_row('race002', n1=2, n2=4, n3=6, s1=0.75, s2=0.65, s3=0.55,
                               f1=1, f2=2, f3=3, trio_result='2-4-6', trio_payout=800),
                self._make_row('race003', bet_type='umaren'),  # skip non-trio
            ]
            self._write_minimal_csv(tmp_path, rows)
            result = backtest_on_csv(tmp_path)
            assert result['n_valid'] == 2
            assert 0.0 <= result['original_hit3'] <= 1.0
        finally:
            os.unlink(tmp_path)

    def test_backtest_hit_detection(self):
        """hit 判定が正しく機能するか"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            tmp_path = f.name
        try:
            # top1=1,top2=3,top3=5 で trio_result=1-3-5 → hit
            rows = [self._make_row('r001', n1=1, n2=3, n3=5, trio_result='1-3-5', trio_payout=2000)]
            self._write_minimal_csv(tmp_path, rows)
            result = backtest_on_csv(tmp_path)
            assert result['original_hit3'] == 1.0
        finally:
            os.unlink(tmp_path)

    def test_backtest_empty_csv_no_crash(self):
        """valid rows がゼロでも crash しない"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            tmp_path = f.name
        try:
            # score が空 → valid 0
            rows = [{'race_id': 'r999', 'status': 'settled', 'bet_type': 'trio',
                     'top1_num': '', 'top1_score': '', 'top2_num': '', 'top2_score': '',
                     'top3_num': '', 'top3_score': ''}]
            self._write_minimal_csv(tmp_path, rows)
            result = backtest_on_csv(tmp_path)
            assert result['n_valid'] == 0
            assert result['original_hit3'] == 0.0
        finally:
            os.unlink(tmp_path)
