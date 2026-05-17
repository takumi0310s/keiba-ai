"""P0-5 順6: live_data_fetcher テスト (★ 全 mock、 実 fetch 0 ★).

15 patterns:
  1.  is_safe_to_fetch blocks 5/17 (G1 day)
  2.  is_safe_to_fetch allows 5/24 (post-G1)
  3.  fetch_jvlink_o1 mock returns valid dict
  4.  fetch_jvlink_o1 real raises NotImplementedError
  5.  fetch_jvlink_tcov mock returns valid dict
  6.  fetch_jvlink_tcov real raises NotImplementedError
  7.  fetch_netkeiba_pre_odds mock returns valid dict
  8.  fetch_netkeiba_pre_odds real raises NotImplementedError
  9.  merge_sources with all three sources OK
  10. merge_sources with partial sources OK
  11. RATE_LIMIT_SEC respects <60 req/min (>=1.0 sec)
  12. fetch_pre_features full mock pipeline OK
  13. fetch_pre_features blocked on G1 day when mock=False
  14. fetch_pre_features dry_run skips file write
  15. V15 production files unchanged (sanity)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ---------- safety guard ----------

def test_is_safe_to_fetch_blocks_g1_day_5_17():
    """5/17 中は fetch 拒否 (G1 day blocklist)."""
    from tools.live_data_fetcher import is_safe_to_fetch

    fake_now = datetime(2026, 5, 17, 10, 0, 0)
    safe, reason = is_safe_to_fetch(now=fake_now)
    assert safe is False
    assert reason == "g1_day_blocklist"


def test_is_safe_to_fetch_allows_5_24():
    """5/24 (post-G1) は fetch OK."""
    from tools.live_data_fetcher import is_safe_to_fetch

    fake_now = datetime(2026, 5, 24, 10, 0, 0)
    safe, reason = is_safe_to_fetch(now=fake_now)
    assert safe is True
    assert reason == "ok"


# ---------- jvlink_o1 ----------

def test_fetch_jvlink_o1_mock_returns_data():
    """mock=True で O1 data 返す."""
    from tools.live_data_fetcher import fetch_jvlink_o1

    result = fetch_jvlink_o1("202605020611", dry_run=True)
    assert "odds_snapshot" in result
    assert "tansho_odds" in result
    assert "fukusho_odds" in result
    assert "popularity" in result
    assert result["source"] == "jvlink_o1_mock"


def test_fetch_jvlink_o1_real_blocked():
    """dry_run=False で NotImplementedError (★ real fetch は 5/24+ ★)."""
    from tools.live_data_fetcher import fetch_jvlink_o1

    with pytest.raises(NotImplementedError):
        fetch_jvlink_o1("test", dry_run=False)


# ---------- jvlink_tcov ----------

def test_fetch_jvlink_tcov_mock_returns_data():
    """mock=True で TCOV data 返す."""
    from tools.live_data_fetcher import fetch_jvlink_tcov

    result = fetch_jvlink_tcov("202605020611", dry_run=True)
    assert "baba_condition" in result
    assert "moisture_rate" in result
    assert "cushion_value" in result
    assert result["source"] == "jvlink_tcov_mock"


def test_fetch_jvlink_tcov_real_blocked():
    """dry_run=False で NotImplementedError."""
    from tools.live_data_fetcher import fetch_jvlink_tcov

    with pytest.raises(NotImplementedError):
        fetch_jvlink_tcov("test", dry_run=False)


# ---------- netkeiba ----------

def test_fetch_netkeiba_pre_mock_returns_data():
    """mock=True で netkeiba pre data 返す."""
    from tools.live_data_fetcher import fetch_netkeiba_pre_odds

    result = fetch_netkeiba_pre_odds("202605020611", dry_run=True)
    assert "horse_weights" in result
    assert "weight_diff" in result
    assert result["source"] == "netkeiba_pre_mock"


def test_fetch_netkeiba_pre_real_blocked():
    """dry_run=False で NotImplementedError (★ IP ban safety ★)."""
    from tools.live_data_fetcher import fetch_netkeiba_pre_odds

    with pytest.raises(NotImplementedError):
        fetch_netkeiba_pre_odds("test", dry_run=False)


# ---------- merge ----------

def test_merge_sources_all_three():
    """3 source merge OK."""
    from tools.live_data_fetcher import (
        fetch_jvlink_o1,
        fetch_jvlink_tcov,
        fetch_netkeiba_pre_odds,
        merge_sources,
    )

    o1 = fetch_jvlink_o1("test", dry_run=True)
    tcov = fetch_jvlink_tcov("test", dry_run=True)
    nk = fetch_netkeiba_pre_odds("test", dry_run=True)
    merged = merge_sources(o1, tcov, nk)

    assert set(merged["sources_used"]) == {"jvlink_o1", "jvlink_tcov", "netkeiba_pre"}
    assert "odds_snapshot" in merged
    assert "baba_condition" in merged
    assert "horse_weights" in merged


def test_merge_sources_partial():
    """1 source 不在でも merge."""
    from tools.live_data_fetcher import fetch_jvlink_o1, merge_sources

    o1 = fetch_jvlink_o1("test", dry_run=True)
    merged = merge_sources(o1, None, None)

    assert merged["sources_used"] == ["jvlink_o1"]
    assert "odds_snapshot" in merged
    assert "baba_condition" not in merged
    assert "horse_weights" not in merged


# ---------- rate limit ----------

def test_rate_limit_under_60_per_min():
    """RATE_LIMIT_SEC が >= 1.0 sec を確認 (<= 60 req/min)."""
    from tools.live_data_fetcher import RATE_LIMIT_SEC

    assert RATE_LIMIT_SEC >= 1.0
    # 60 req/min = 1.0 sec/req、 余裕を持って 1.5 設定
    requests_per_min = 60.0 / RATE_LIMIT_SEC
    assert requests_per_min <= 60.0


# ---------- full pipeline ----------

def test_fetch_pre_features_mock_full_pipeline():
    """full pipeline mock OK."""
    from tools.live_data_fetcher import fetch_pre_features

    result = fetch_pre_features(
        "202605020611",
        date_str="20260517",
        dry_run=True,
        mock=True,
    )
    assert result["status"] == "ok"
    assert "merge_timestamp" in result
    assert len(result["sources_used"]) == 3


def test_fetch_pre_features_blocked_on_g1_day_no_mock():
    """G1 day + mock=False で blocked 返す."""
    from tools.live_data_fetcher import fetch_pre_features

    fake_now = datetime(2026, 5, 17, 10, 0, 0)
    with patch("tools.live_data_fetcher.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = datetime  # 他 datetime call は本物
        result = fetch_pre_features(
            "202605020611",
            date_str="20260517",
            dry_run=False,
            mock=False,
        )
    assert result["status"] == "blocked"
    assert result["reason"] == "g1_day_blocklist"


def test_fetch_pre_features_dry_run_skips_file_write(tmp_path):
    """dry_run=True で file 書き込み skip."""
    from tools.live_data_fetcher import fetch_pre_features

    result = fetch_pre_features(
        "202605020611",
        date_str="20260517",
        dry_run=True,
        mock=True,
    )
    assert "output_file" not in result
    assert result["status"] == "ok"


# ---------- V15 sanity ----------

def test_v15_production_files_unchanged():
    """V15 production file が touch されていないことを sanity check.

    本 test は file 存在 + 命名規約のみ確認 (中身 hash までは未確認、 既存 regression test に委譲)。
    """
    v15_pa = REPO / "keiba_model_v135_central.pkl.gz"
    v15_pb = REPO / "keiba_model_v135_central_live.pkl.gz"
    # 片方でも存在すれば OK (環境により live のみ / pa のみあり得る)
    assert v15_pa.exists() or v15_pb.exists(), (
        "V15 production model files not found — this test ensures we didn't "
        "accidentally relocate/delete them."
    )

    # live_data_fetcher.py が production file を import / open していないか軽くチェック
    # (docstring / comment での言及は OK、 実 import / open のみ禁止)
    fetcher = REPO / "tools" / "live_data_fetcher.py"
    text = fetcher.read_text(encoding="utf-8")
    forbidden_imports = [
        "import predict_core",
        "from predict_core",
        "import daily_predict",
        "from daily_predict",
        "import race_auto_notify",
        "from race_auto_notify",
        "keiba_model_v135",  # model file 名は code に出るべきでない
        "cumulative_results.csv",
    ]
    for needle in forbidden_imports:
        assert needle not in text, (
            f"live_data_fetcher.py contains forbidden reference: {needle!r}"
        )
