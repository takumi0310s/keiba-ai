"""tests/test_v15_ensemble_blender.py

V15 ensemble blender の最低限テスト (3件)
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

# プロジェクトルートを sys.path に追加
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.v15_ensemble_blender import (
    load_models,
    blend_predictions,
    get_paper_formation,
    WEIGHT_CONFIGS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_model_data(n_features: int = 10) -> dict:
    """lightgbm なしでも動くダミー model dict を返す。"""

    class _DummyLGB:
        """predict() だけ実装した最小 stub。"""
        def __init__(self, n_feat):
            self._n = n_feat

        def num_feature(self):
            return self._n

        def predict(self, X):
            # 馬番1 が常に高スコアになるよう、行インデックスで降順スコア返す
            n = X.shape[0]
            return np.linspace(0.9, 0.1, n)

    return {
        "model": _DummyLGB(n_features),
        "xgb_model": None,
        "ensemble_weights": {"lgb": 1.0, "xgb": 0.0},
        "features": [f"feat_{i}" for i in range(n_features)],
    }


def _make_models_list(n_models: int = 1, n_features: int = 10):
    names = ["v15", "v15_full", "v15_2"]
    return [
        {"name": names[i], "data": _make_dummy_model_data(n_features)}
        for i in range(n_models)
    ]


def _make_race_df(n_horses: int = 8, n_features: int = 10) -> pd.DataFrame:
    cols = [f"feat_{i}" for i in range(n_features)]
    data = np.random.rand(n_horses, n_features)
    return pd.DataFrame(data, columns=cols)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSingleModelFallback:
    """test_single_model_fallback: candidate なしでも V15 単独で動くか"""

    def test_blend_with_single_model(self):
        """モデルが1件だけでも blend_predictions が正常に値を返す。"""
        models = _make_models_list(n_models=1)
        race_df = _make_race_df(n_horses=10)
        weights = [1.0, 0.0, 0.0]

        probs = blend_predictions(race_df, models, weights)

        assert isinstance(probs, np.ndarray), "np.ndarray を返すこと"
        assert probs.shape == (10,), f"馬数に合った shape: expected (10,), got {probs.shape}"
        assert np.all(probs >= 0) and np.all(probs <= 1.0 + 1e-6), "確率は [0, 1] に収まること"

    def test_formation_generated_with_single_model(self):
        """モデル1件で formation が生成できること。"""
        models = _make_models_list(n_models=1)
        race_df = _make_race_df(n_horses=12)
        horse_nums = list(range(1, 13))

        probs = blend_predictions(race_df, models, [1.0])
        formation = get_paper_formation(probs, horse_nums, condition="A")

        assert formation["bet_type"] == "trio"
        assert len(formation["trio_bets"]) > 0, "trio_bets が空でないこと"
        assert formation["top1_num"] in horse_nums


class TestWeightsSumToOne:
    """test_weights_sum_to_one: weights が 1.0 に正規化されるか"""

    def test_unnormalized_weights_produce_valid_prob(self):
        """正規化前の weights [2.0, 1.0] でも確率が [0, 1] に収まること。"""
        models = _make_models_list(n_models=2)
        race_df = _make_race_df(n_horses=8)
        weights = [2.0, 1.0]  # 合計 3.0 → 内部で正規化

        probs = blend_predictions(race_df, models, weights)

        assert probs.shape == (8,)
        assert np.all(probs >= 0) and np.all(probs <= 1.0 + 1e-6)

    def test_zero_weights_fallback(self):
        """weights が全て 0 でも crash しないこと (均等割り当て)。"""
        models = _make_models_list(n_models=2)
        race_df = _make_race_df(n_horses=6)
        weights = [0.0, 0.0]  # 全ゼロ → 均等

        probs = blend_predictions(race_df, models, weights)

        assert probs.shape == (6,)
        # 全ゼロ weights は内部で均等化されるため、全ゼロにはならない
        assert not np.all(probs == 0.0), "全ゼロ確率は異常"

    def test_weight_configs_are_normalized(self):
        """WEIGHT_CONFIGS の各エントリーの重みの合計が 1.0 に近いこと。"""
        for name, w in WEIGHT_CONFIGS.items():
            total = sum(w)
            assert abs(total - 1.0) < 1e-6, (
                f"WEIGHT_CONFIGS['{name}'] の合計が 1.0 でない: {total}"
            )


class TestFormationDeterministic:
    """test_formation_deterministic: 同入力で同 formation が出るか"""

    def test_same_input_same_formation(self):
        """同じ probs と horse_nums で get_paper_formation を2回呼ぶと同じ結果になること。"""
        np.random.seed(0)
        probs = np.random.rand(12)
        horse_nums = list(range(1, 13))

        f1 = get_paper_formation(probs.copy(), horse_nums[:], condition="A")
        f2 = get_paper_formation(probs.copy(), horse_nums[:], condition="A")

        assert f1["top1_num"] == f2["top1_num"], "top1 が一致すること"
        assert f1["trio_bets"] == f2["trio_bets"], "trio_bets が一致すること"

    def test_condition_e_returns_umaren(self):
        """条件 E では bet_type == 'umaren' かつ trio_bets が空であること。"""
        probs = np.array([0.8, 0.6, 0.3, 0.2, 0.1])
        horse_nums = [1, 2, 3, 4, 5]

        formation = get_paper_formation(probs, horse_nums, condition="E")

        assert formation["bet_type"] == "umaren"
        assert formation["trio_bets"] == []
        assert len(formation["umaren_bets"]) >= 1

    def test_formation_top1_is_highest_prob(self):
        """top1_num は最も prob が高い馬番であること。"""
        probs = np.array([0.1, 0.9, 0.5, 0.3])
        horse_nums = [3, 7, 11, 14]

        formation = get_paper_formation(probs, horse_nums, condition="A")

        # index 1 (prob=0.9) → horse_num=7 が top1 であるべき
        assert formation["top1_num"] == 7

    def test_trio_bets_always_sorted(self):
        """trio_bets の各組み合わせは昇順ソートされていること。"""
        np.random.seed(99)
        probs = np.random.rand(16)
        horse_nums = list(range(1, 17))

        formation = get_paper_formation(probs, horse_nums, condition="A")

        for bet in formation["trio_bets"]:
            assert bet == sorted(bet), f"trio_bet が昇順でない: {bet}"
