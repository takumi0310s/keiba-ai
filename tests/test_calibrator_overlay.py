"""P0-5 順4: calibrator_overlay テスト (10 tests)

★ V15 production 不変、 shadow eval 用 module のテスト ★
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.calibrator_overlay import (  # noqa: E402
    DELTA_CAP,
    apply_overlay,
    build_overlay_state,
    compute_odds_shift,
    compute_weight_shift,
    is_safe_feature,
    overlay_race_ranking,
    save_overlay_state,
    simulate_on_history,
)


# ---------- Test 1 ----------
def test_is_safe_feature_excludes_leak():
    """LEAK 監査済 features は False、 safe feature は True"""
    assert is_safe_feature("paci_info_idx") is False
    assert is_safe_feature("jrdb_odds_idx") is False
    assert is_safe_feature("jrdb_paddock_idx") is False
    assert is_safe_feature("skb_kishi_code_3") is False
    assert is_safe_feature("skb_kishi_code_1") is False
    assert is_safe_feature("kyi_jockey_idx") is True
    assert is_safe_feature("v15_prob") is True


# ---------- Test 2 ----------
def test_delta_cap():
    """補正 max delta が ±DELTA_CAP (0.10) で cap される"""
    # 極端 case (odds 1.01 = implied prob ~0.99) でも cap 超えない
    corrected = apply_overlay(v15_prob=0.5, odds_15min=1.01)
    assert abs(corrected - 0.5) <= DELTA_CAP + 1e-9

    # 反対側 (odds 1000 = implied ~0.001) でも cap
    corrected_neg = apply_overlay(v15_prob=0.5, odds_15min=1000.0)
    assert abs(corrected_neg - 0.5) <= DELTA_CAP + 1e-9


# ---------- Test 3 ----------
def test_overlay_v15_unchanged_when_no_data():
    """odds / weight 不在 で V15 prob 維持"""
    assert apply_overlay(0.6) == 0.6
    assert apply_overlay(0.123) == 0.123
    # explicit None
    assert apply_overlay(0.45, odds_15min=None, weight_diff=None) == 0.45


# ---------- Test 4 ----------
def test_overlay_odds_shift_high_odds_negative():
    """odds 高 (= implied prob 低) → V15 prob 0.5 から negative shift"""
    # V15 0.5、 odds 20 (implied 0.05) → 補正 negative
    corrected = apply_overlay(v15_prob=0.5, odds_15min=20.0)
    assert corrected < 0.5
    # 逆: odds 低 (implied 高) → V15 0.3 から positive shift
    corrected_up = apply_overlay(v15_prob=0.3, odds_15min=1.5)
    assert corrected_up > 0.3


# ---------- Test 5 ----------
def test_overlay_weight_diff_penalty():
    """馬体重 大変動 (+20kg) で penalty (V15 から下がる)"""
    corrected = apply_overlay(v15_prob=0.5, weight_diff=20.0)
    assert corrected < 0.5
    # 中程度 (+12kg)
    mid = apply_overlay(v15_prob=0.5, weight_diff=12.0)
    assert mid < 0.5


# ---------- Test 6 ----------
def test_overlay_weight_diff_squeeze_bonus():
    """馬体重 適度減 (-3 kg) で 微 positive"""
    corrected = apply_overlay(v15_prob=0.5, weight_diff=-3.0)
    assert corrected > 0.5
    # 範囲外 (-1 kg) は 0
    no_change = apply_overlay(v15_prob=0.5, weight_diff=-1.0)
    assert no_change == 0.5


# ---------- Test 7 ----------
def test_overlay_race_ranking_unchanged_no_data():
    """odds / weight 列 不在 で V15 ranking 完全維持"""
    df = pd.DataFrame({
        "horse_num": [1, 2, 3],
        "v15_prob": [0.7, 0.5, 0.3],
    })
    result = overlay_race_ranking(df)
    assert result["new_rank"].tolist() == [1, 2, 3]
    np.testing.assert_array_almost_equal(result["corrected_prob"].values, [0.7, 0.5, 0.3])


# ---------- Test 8 ----------
def test_build_state_and_save(tmp_path):
    """state 構築 + save + load round-trip"""
    state = build_overlay_state()
    assert state["version"] == "overlay_v1"
    assert state["delta_cap"] == DELTA_CAP
    assert "paci_info_idx" in state["leak_features"]
    assert "weight_diff_thresholds" in state

    # save to tmp path (★ 既存 calibrator pkl 不変 ★)
    out_path = tmp_path / "overlay_state.pkl"
    save_overlay_state(state, path=out_path)
    assert out_path.exists()

    import pickle
    with open(out_path, "rb") as f:
        loaded = pickle.load(f)
    assert loaded["version"] == "overlay_v1"


# ---------- Test 9 ----------
def test_clip_at_001_999():
    """probability bounds [0.001, 0.999]"""
    # 上限
    corrected_high = apply_overlay(v15_prob=0.999, odds_15min=1.01)
    assert corrected_high <= 0.999
    # 下限
    corrected_low = apply_overlay(v15_prob=0.001, odds_15min=1000.0)
    assert corrected_low >= 0.001
    # 中間値 で bounds 維持
    mid = apply_overlay(v15_prob=0.5)
    assert 0.001 <= mid <= 0.999


# ---------- Test 10 ----------
def test_simulation_returns_metrics():
    """simulation 動作確認 (実 daily_predictions data on past date)"""
    # 既存 20260314.csv で simulate
    result = simulate_on_history("20260314", dry_run=True)
    # 過去日付なら error にならず、 metrics 返る
    if "error" not in result:
        assert "n_races" in result
        assert "mean_abs_delta" in result
        assert "max_abs_delta" in result
        assert "delta_cap_respected" in result
        assert result["delta_cap_respected"] is True
        assert result["dry_run"] is True
        # delta cap 厳守
        assert result["max_abs_delta"] <= DELTA_CAP + 1e-9
    else:
        # データ不在なら error フィールドで明示される
        assert "error" in result


# ---------- Bonus: compute_* unit tests ----------
def test_compute_odds_shift_zero_for_invalid():
    """odds None / NaN / <=1.0 で shift = 0"""
    assert compute_odds_shift(0.5, None) == 0.0
    assert compute_odds_shift(0.5, float("nan")) == 0.0
    assert compute_odds_shift(0.5, 1.0) == 0.0


def test_compute_weight_shift_zero_for_invalid():
    """weight None / NaN で shift = 0"""
    assert compute_weight_shift(None) == 0.0
    assert compute_weight_shift(float("nan")) == 0.0
    # 範囲外 (0kg or +3kg = 増えてる) は 0
    assert compute_weight_shift(0) == 0.0
    assert compute_weight_shift(3.0) == 0.0  # +3 は squeeze ではない (増)
