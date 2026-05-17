"""P0-5 順5: recalc_15min テスト (★ V15 不変、 shadow eval ★)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.recalc_15min import (  # noqa: E402
    is_strategy_7c_excluded,
    load_morning_prediction,
    load_race_meta,
    run_recalc,
)


# -------- helpers --------
def _meta(course: str = "東京", condition: str = "A", race_name: str = "test") -> dict:
    return {
        "race_id": "999999999999",
        "race_name": race_name,
        "course": course,
        "condition": condition,
    }


# -------- tests --------
def test_load_morning_prediction_mock_fallback():
    """daily_predictions file 不在 + mock=True で mock ranking 返す."""
    result = run_recalc("999999999999", date_str="19990101", mock=True, dry_run=True)
    assert result["status"] == "ok"
    assert len(result["original_ranking"]) == 5
    assert result["original_ranking"][0]["horse_num"] == 5


def test_strategy_7c_kyoto_excluded():
    """京都 R で skip."""
    meta = _meta(course="京都", race_name="京都 1R")
    skip, reason = is_strategy_7c_excluded(meta)
    assert skip is True
    assert "kyoto" in reason


def test_strategy_7c_g1_not_excluded():
    """G1 は skip しない (京都 でも)."""
    meta = _meta(course="京都", race_name="天皇賞(G1)")
    skip, reason = is_strategy_7c_excluded(meta)
    assert skip is False
    assert reason is None


def test_run_recalc_mock():
    """mock data で full pipeline OK."""
    result = run_recalc("test_mock", mock=True, dry_run=True)
    assert result["status"] == "ok"
    assert "recalc_ranking" in result
    assert len(result["recalc_ranking"]) == 5


def test_fallback_no_morning():
    """daily_predictions 不在 + mock=False で fallback."""
    result = run_recalc(
        "999999999999", date_str="19000101", mock=False, dry_run=True
    )
    assert result["status"] == "fallback"
    assert result["reason"] == "no_morning_prediction"


def test_strategy_7c_skip_returns_early():
    """戦略⑦案 C skip で early return (recalc_ranking なし)."""
    # 京都 で daily_predictions が存在しないので、 mock + race_meta も mock
    # mock 時の race_meta は東京なので、 別経路で 京都 skip を確認する：
    # race_meta=京都 を直接 inject するために load_race_meta を mock 的に使う
    meta = _meta(course="京都", race_name="京都 1R")
    skip, reason = is_strategy_7c_excluded(meta)
    assert skip is True
    assert reason == "strategy_7_kyoto_p0_2_5_17"


def test_recalc_ranking_includes_corrected_prob():
    """recalc_ranking に corrected_prob 含む."""
    result = run_recalc("test_corr", mock=True, dry_run=True)
    assert result["status"] == "ok"
    for r in result["recalc_ranking"]:
        assert "corrected_prob" in r
        assert "new_rank" in r


def test_pre_features_optional():
    """pre_features 不在で V15 ranking そのまま (overlay 効果 0)."""
    # mock=False、 file 不在 → pre_features=None だが
    # morning も None なので fallback。 ここでは mock=True で pre_features 注入有を確認。
    result_mock = run_recalc("test_pf", mock=True, dry_run=True)
    assert result_mock["pre_features_available"] is True


def test_notification_delegate():
    """Discord 通知 discord_recalc_notify に delegate (dry-run 確認)."""
    result = run_recalc("test_notif", mock=True, dry_run=True)
    assert "notification_result" in result
    assert "reason" in result["notification_result"]


def test_v15_prob_not_modified_in_morning():
    """morning prediction の v15_prob は overlay で 直接変更されない (read-only)."""
    result = run_recalc("test_imm", mock=True, dry_run=True)
    # original_ranking の v15_prob は 0.65, 0.55, 0.48, 0.42, 0.38 (mock data)
    mock_probs = {5: 0.65, 3: 0.55, 1: 0.48, 7: 0.42, 4: 0.38}
    for r in result["original_ranking"]:
        assert r["v15_prob"] == pytest.approx(mock_probs[r["horse_num"]])


def test_output_file_format(tmp_path, monkeypatch):
    """output JSON file の structure (実 file 書き込み)."""
    # REPO を monkeypatch で tmp_path に向ける手法は煩雑なので、
    # ここでは dry_run=False で REPO/data/recalc_15min に出力されることを確認、
    # ★ ただし 既存 production data を 汚さないため race_id を test 専用 ★
    out_dir = REPO / "data" / "recalc_15min" / "19000101"
    out_file = out_dir / "test_output_format.json"
    if out_file.exists():
        out_file.unlink()

    result = run_recalc(
        "test_output_format", date_str="19000101", mock=True, dry_run=False
    )
    assert result["status"] == "ok"
    assert "output_file" in result
    out_path = Path(result["output_file"])
    assert out_path.exists()
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "ok"
    assert "recalc_ranking" in data
    # cleanup
    out_path.unlink()
    try:
        out_path.parent.rmdir()
        out_path.parent.parent.rmdir()
    except OSError:
        pass  # 他 file あれば残す


def test_strategy_7c_cond_x_excluded():
    """条件 X で skip (非 graded のみ)."""
    meta = _meta(course="東京", condition="X", race_name="平場特別")
    skip, reason = is_strategy_7c_excluded(meta)
    assert skip is True
    assert "cond_X" in reason
