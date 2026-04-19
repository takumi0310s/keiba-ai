"""scraper_guard のホワイトリスト・特例ロジックを pytest で検証。

6時刻 × 複数 caller の組み合わせで、is_scraping_allowed() と
check_scraping_allowed() が期待通りに許可/拒否されることをテストする。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from tools.scraper_guard import (  # noqa: E402
    is_scraping_allowed,
    check_scraping_allowed,
    OPERATIONAL_CALLERS,
)


# 6時刻: Fri22時・土AM3・土AM8・日AM8・月AM5:59・月AM6:01
T_FRI_22 = datetime(2026, 4, 17, 22, 0)   # Fri (guarded start)
T_SAT_03 = datetime(2026, 4, 18, 3, 30)   # Sat 03:30 (guarded, premium early slot)
T_SAT_08 = datetime(2026, 4, 18, 8, 0)    # Sat 08:00 (guarded)
T_SUN_08 = datetime(2026, 4, 19, 8, 0)    # Sun 08:00 (guarded)
T_MON_0559 = datetime(2026, 4, 20, 5, 59) # Mon 05:59 (guarded)
T_MON_0601 = datetime(2026, 4, 20, 6, 1)  # Mon 06:01 (allowed)

GUARDED_TIMES = [T_FRI_22, T_SAT_03, T_SAT_08, T_SUN_08, T_MON_0559]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """各テスト前に SCRAPER_GUARD_DISABLE / KEIBA_OPERATIONAL_MODE を解除"""
    monkeypatch.delenv("SCRAPER_GUARD_DISABLE", raising=False)
    monkeypatch.delenv("KEIBA_OPERATIONAL_MODE", raising=False)


# -------- is_scraping_allowed: デフォルト（caller なし）挙動 --------

@pytest.mark.parametrize("t", GUARDED_TIMES)
def test_default_blocked_in_guard_window(t):
    """引数なしでガード時間帯は False"""
    assert is_scraping_allowed(now=t) is False


def test_default_allowed_mon_after_6am():
    """月曜06:01はガード抜けて True"""
    assert is_scraping_allowed(now=T_MON_0601) is True


# -------- ホワイトリスト caller は常に許可 --------

@pytest.mark.parametrize("caller", sorted(OPERATIONAL_CALLERS))
@pytest.mark.parametrize("t", GUARDED_TIMES)
def test_operational_callers_always_allowed(caller, t):
    """daily_predict など運用タスクは週末でも True"""
    assert is_scraping_allowed(now=t, caller=caller) is True


# -------- KEIBA_OPERATIONAL_MODE env でも許可 --------

@pytest.mark.parametrize("t", GUARDED_TIMES)
def test_env_operational_mode_allows(monkeypatch, t):
    monkeypatch.setenv("KEIBA_OPERATIONAL_MODE", "1")
    assert is_scraping_allowed(now=t) is True


# -------- ホワイトリスト外 caller は従来通り停止 --------

@pytest.mark.parametrize("t", GUARDED_TIMES)
def test_non_operational_caller_still_blocked(t):
    """bulk_scrape_upset 等は引数ありでもガードされる（ホワイトリスト外）"""
    assert is_scraping_allowed(now=t, caller="bulk_scrape_upset") is False


# -------- daily_premium_scrape 特例: 土日03:00-05:59 のみ許可 --------

def test_premium_scrape_sat_early_allowed():
    """土曜03:30 は premium_scrape 特例で許可"""
    assert is_scraping_allowed(now=T_SAT_03, caller="daily_premium_scrape") is True


def test_premium_scrape_sun_early_allowed():
    """日曜04:30 も許可"""
    t = datetime(2026, 4, 19, 4, 30)
    assert is_scraping_allowed(now=t, caller="daily_premium_scrape") is True


def test_premium_scrape_sat_mid_blocked():
    """土曜08:00 は premium_scrape でもガード(早朝スロット外)"""
    assert is_scraping_allowed(now=T_SAT_08, caller="daily_premium_scrape") is False


def test_premium_scrape_mon_early_blocked():
    """月曜05:59 は premium_scrape 特例対象外 (土日のみ)"""
    assert is_scraping_allowed(now=T_MON_0559, caller="daily_premium_scrape") is False


def test_premium_scrape_weekday_normal_allowed():
    """平日10:00 は当然許可"""
    t = datetime(2026, 4, 21, 10, 0)  # Tue
    assert is_scraping_allowed(now=t, caller="daily_premium_scrape") is True


# -------- check_scraping_allowed: mode="exit" で sys.exit 発動確認 --------

def test_check_exit_mode_blocks_default(monkeypatch):
    """引数なし + guarded時刻 + mode=exit → SystemExit"""
    import tools.scraper_guard as sg
    monkeypatch.setattr(sg, "is_scraping_allowed",
                        lambda now=None, caller=None: False)
    with pytest.raises(SystemExit):
        check_scraping_allowed(mode="exit")


def test_check_exit_operational_caller_passes(monkeypatch):
    """ホワイトリストcaller → SystemExit せず return"""
    import tools.scraper_guard as sg
    # is_scraping_allowed 本物を使い、guardedな時刻を固定したいが
    # now 引数を渡せないので、そのまま OPERATIONAL_CALLERS チェックを通る
    check_scraping_allowed(caller="daily_predict", mode="exit")  # 例外なし = pass


def test_check_env_operational_mode_passes(monkeypatch):
    monkeypatch.setenv("KEIBA_OPERATIONAL_MODE", "1")
    check_scraping_allowed(caller="bulk_scrape_upset", mode="exit")  # 例外なし


# -------- SCRAPER_GUARD_DISABLE env は全時刻で許可 --------

def test_disable_env_overrides_all(monkeypatch):
    monkeypatch.setenv("SCRAPER_GUARD_DISABLE", "1")
    for t in GUARDED_TIMES:
        assert is_scraping_allowed(now=t) is True
