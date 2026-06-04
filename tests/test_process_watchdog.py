"""process_watchdog_v2 のロジックテスト。

実プロセスには依存させず、モック化した process_checker と log_base で
- 鮮度閾値の境界
- 時間帯制御
- STALE/MISSING/ALIVE の判定
をテストする。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
    sys.path.insert(0, os.path.join(BASE, 'tools'))

from tools import process_watchdog_v2 as wd  # noqa: E402


# 基準時刻（平日昼・本番時間帯）
NOW_ACTIVE = datetime(2026, 4, 21, 12, 0)    # Tue 12:00
NOW_EARLY = datetime(2026, 4, 21, 6, 30)     # 06:30 (時間帯外)
NOW_LATE = datetime(2026, 4, 21, 18, 30)     # 18:30 (時間帯外)
NOW_BOUNDARY_OPEN = datetime(2026, 4, 21, 7, 0)   # 07:00 ちょうど → 時間帯内
NOW_BOUNDARY_CLOSE = datetime(2026, 4, 21, 18, 0) # 18:00 ちょうど → 時間帯外


# ===== is_active_hours =====

@pytest.mark.parametrize("t,expected", [
    (NOW_ACTIVE, True),
    (NOW_EARLY, False),
    (NOW_LATE, False),
    (NOW_BOUNDARY_OPEN, True),      # 07:00 is start (inclusive)
    (NOW_BOUNDARY_CLOSE, False),    # 18:00 is end (exclusive)
])
def test_is_active_hours(t, expected):
    assert wd.is_active_hours(t) is expected


# ===== latest_log_mtime / is_log_stale =====

def test_log_stale_when_no_files(tmp_path):
    """ログファイルが一つもない → stale=True, mtime=None"""
    stale, mt = wd.is_log_stale('nothing*.log', 600, now=NOW_ACTIVE, base_dir=str(tmp_path))
    assert stale is True
    assert mt is None


def test_log_fresh(tmp_path):
    """now から 30秒前に更新したログは stale_sec=600 なら fresh"""
    p = tmp_path / 'daily_predict_20260421.log'
    p.write_text('recent')
    # mtime を NOW_ACTIVE - 30秒に設定
    target_ts = NOW_ACTIVE.timestamp() - 30
    os.utime(p, (target_ts, target_ts))
    stale, mt = wd.is_log_stale('daily_predict*.log', 600, now=NOW_ACTIVE, base_dir=str(tmp_path))
    assert stale is False
    assert mt == pytest.approx(target_ts, abs=1)


def test_log_stale_over_threshold(tmp_path):
    """stale_sec=600、now-mtime=700 → stale=True"""
    p = tmp_path / 'race_auto_notify_20260421.log'
    p.write_text('old')
    target_ts = NOW_ACTIVE.timestamp() - 700
    os.utime(p, (target_ts, target_ts))
    stale, _ = wd.is_log_stale('race_auto_notify*.log', 600, now=NOW_ACTIVE, base_dir=str(tmp_path))
    assert stale is True


def test_log_stale_uses_newest_when_multiple(tmp_path):
    """複数ログがあれば最新の mtime を使う"""
    old = tmp_path / 'daily_predict_20260420.log'
    new = tmp_path / 'daily_predict_20260421.log'
    old.write_text('old'); new.write_text('new')
    os.utime(old, (NOW_ACTIVE.timestamp() - 10_000, NOW_ACTIVE.timestamp() - 10_000))
    os.utime(new, (NOW_ACTIVE.timestamp() - 60, NOW_ACTIVE.timestamp() - 60))
    stale, mt = wd.is_log_stale('daily_predict*.log', 600, now=NOW_ACTIVE, base_dir=str(tmp_path))
    assert stale is False
    assert mt == pytest.approx(NOW_ACTIVE.timestamp() - 60, abs=1)


# ===== check_target =====

def _fake_process_alive(match: str, mapping: dict) -> bool:
    return mapping.get(match, False)


def test_check_target_alive(tmp_path):
    """プロセス生存 + ログ fresh → ALIVE"""
    p = tmp_path / 'daily_predict_20260421.log'
    p.write_text('ok')
    os.utime(p, (NOW_ACTIVE.timestamp() - 30, NOW_ACTIVE.timestamp() - 30))
    target = wd.TARGETS[0]  # daily_predict
    r = wd.check_target(target, now=NOW_ACTIVE, log_base=str(tmp_path),
                        process_checker=lambda m: _fake_process_alive(m, {target.process_match: True}))
    assert r['status'] == 'ALIVE'


def test_check_target_stale(tmp_path):
    """プロセス生存だがログ古い → STALE (Fortran ゾンビ想定)"""
    target = wd.TARGETS[0]
    p = tmp_path / 'daily_predict_20260421.log'
    p.write_text('old')
    os.utime(p, (NOW_ACTIVE.timestamp() - 10_000, NOW_ACTIVE.timestamp() - 10_000))
    r = wd.check_target(target, now=NOW_ACTIVE, log_base=str(tmp_path),
                        process_checker=lambda m: _fake_process_alive(m, {target.process_match: True}))
    assert r['status'] == 'STALE'


def test_check_target_missing(tmp_path):
    """プロセスなし + ログも古い(or なし) → MISSING"""
    target = wd.TARGETS[0]
    r = wd.check_target(target, now=NOW_ACTIVE, log_base=str(tmp_path),
                        process_checker=lambda m: False)
    assert r['status'] == 'MISSING'


# ===== restart_target 時間帯制御 =====

def test_restart_skipped_outside_hours(monkeypatch):
    """時間帯外 (06:30) は再起動せず skipped_reason=outside_active_hours"""
    target = wd.TARGETS[0]
    called = {}
    def fake_popen(*args, **kwargs):
        called['popen'] = True
        class _P: pid = 99999
        return _P()
    r = wd.restart_target(target, dry_run=False, now=NOW_EARLY, popen=fake_popen)
    assert r['restarted'] is False
    assert r['skipped_reason'] == 'outside_active_hours'
    assert 'popen' not in called


def test_restart_dry_run_active_hours():
    """dry_run=True なら時間帯内でも再起動せず skipped=dry_run"""
    target = wd.TARGETS[0]
    r = wd.restart_target(target, dry_run=True, now=NOW_ACTIVE,
                          popen=lambda *a, **k: (_ for _ in ()).throw(AssertionError("popen called")))
    assert r['restarted'] is False
    assert r['skipped_reason'] == 'dry_run'


def test_restart_executes_in_active_hours(monkeypatch, tmp_path):
    """時間帯内 + dry_run=False → popen 呼び出し+env付与"""
    target = wd.TARGETS[0]
    monkeypatch.setattr(wd, 'LOG_DIR', str(tmp_path))
    captured = {}
    class _FakeProc:
        pid = 12345
    def fake_popen(cmd, **kwargs):
        captured['cmd'] = cmd
        captured['env'] = kwargs.get('env', {})
        captured['cwd'] = kwargs.get('cwd')
        return _FakeProc()
    # 再起動cap の台帳をテスト用に隔離(実台帳を汚さない・他テストと干渉しない)
    r = wd.restart_target(target, dry_run=False, now=NOW_ACTIVE, popen=fake_popen,
                          ledger_path=str(tmp_path / 'led.json'))
    assert r['restarted'] is True
    assert r['pid'] == 12345
    assert captured['cmd'] == target.restart_cmd
    assert captured['env'].get('SCRAPER_GUARD_DISABLE') == '1'
    assert captured['env'].get('KEIBA_OPERATIONAL_MODE') == '1'


def test_restart_exception_caught(monkeypatch, tmp_path):
    """popen が例外を投げても restart_target は dict を返す"""
    target = wd.TARGETS[0]
    monkeypatch.setattr(wd, 'LOG_DIR', str(tmp_path))
    def bad_popen(*a, **k):
        raise RuntimeError("boom")
    # 台帳隔離(cap guard を通過させ、popen 例外パスを検証する)
    r = wd.restart_target(target, dry_run=False, now=NOW_ACTIVE, popen=bad_popen,
                          ledger_path=str(tmp_path / 'led.json'))
    assert r['restarted'] is False
    assert r['skipped_reason'].startswith('exception')


# ===== TARGETS sanity =====

def test_targets_defined():
    names = {t.name for t in wd.TARGETS}
    assert 'daily_predict' in names
    assert 'race_auto_notify' in names


def test_targets_stale_sec_values():
    # Session #31 で誤発火防止のため stale_sec を引き上げ済(daily 30→60min / race 10→30min)。
    # テスト側が旧値のままだったので実コード値に追従(本番ロジックは不変)。
    by_name = {t.name: t for t in wd.TARGETS}
    assert by_name['daily_predict'].stale_sec == 60 * 60
    assert by_name['race_auto_notify'].stale_sec == 30 * 60
