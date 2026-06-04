#!/usr/bin/env python3
"""process_watchdog_v2 の再起動 cap(per-day上限 + 連続間隔ガード)のユニットテスト。
★5/9 の再起動ループ(Discord spam)再発防止の前提。watchdog は kill-switch で停止中=本番無影響★。
台帳ファイル永続化なので watchdog 自身がプロセス再起動しても cap が効くことを固定する。
"""
import os
import sys
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(BASE, 'tools')
for p in (TOOLS, BASE):
    if p not in sys.path:
        sys.path.insert(0, p)

import process_watchdog_v2 as w


class _FakeProc:
    pid = 4242


def _target():
    return w.WatchTarget(
        name='race_auto_notify', log_glob='x*.log', stale_sec=1,
        process_match='tools/race_auto_notify.py',
        restart_cmd=['python', '-u', 'tools/race_auto_notify.py'],
        restart_log_pattern='x_{date}.log')


def _restart(now, ledger, **kw):
    """active_hours/dry_run を回避し、popen をフェイクにして restart_target を呼ぶ。"""
    return w.restart_target(
        _target(), dry_run=False, now=now,
        popen=lambda *a, **k: _FakeProc(),
        active_checker=lambda _n: True,
        ledger_path=str(ledger), **kw)


def test_daily_cap_blocks_after_limit(tmp_path):
    """間隔ガードを無効化(min_interval=0)し、当日3回まで→4回目は daily_cap_reached。"""
    ledger = tmp_path / 'led.json'
    base = datetime(2026, 6, 6, 10, 0, 0)
    results = []
    for i in range(4):
        # 各回 5分ずつ進める(間隔ガードは0なので effect なし=capのみ検証)
        r = _restart(base + timedelta(minutes=5 * i), ledger, max_per_day=3, min_interval_sec=0)
        results.append(r)
    assert [r['restarted'] for r in results] == [True, True, True, False]
    assert results[3]['skipped_reason'] == 'daily_cap_reached'
    assert results[2]['day_count'] == 3


def test_interval_guard_blocks_rapid_restart(tmp_path):
    """連続再起動: 1回目OK、min_interval内の2回目は interval_guard で阻止。"""
    ledger = tmp_path / 'led.json'
    base = datetime(2026, 6, 6, 10, 0, 0)
    r1 = _restart(base, ledger, max_per_day=3, min_interval_sec=600)
    r2 = _restart(base + timedelta(seconds=120), ledger, max_per_day=3, min_interval_sec=600)  # 2分後=阻止
    r3 = _restart(base + timedelta(seconds=900), ledger, max_per_day=3, min_interval_sec=600)  # 15分後=許可
    assert r1['restarted'] is True
    assert r2['restarted'] is False and r2['skipped_reason'] == 'interval_guard'
    assert r3['restarted'] is True


def test_ledger_persists_across_calls(tmp_path):
    """台帳がファイル永続=新しい restart_target 呼び出し(=watchdogプロセス再起動相当)でも cap が効く。"""
    ledger = tmp_path / 'led.json'
    base = datetime(2026, 6, 6, 10, 0, 0)
    for i in range(3):
        _restart(base + timedelta(minutes=20 * i), ledger, max_per_day=3, min_interval_sec=600)
    assert os.path.exists(ledger)  # 永続化された
    # 別プロセス相当(モジュール状態に依存せず)でも台帳から cap 判定される
    guard = w.restart_guard('race_auto_notify', base + timedelta(minutes=120),
                            ledger_path=str(ledger), max_per_day=3, min_interval_sec=600)
    assert guard['allowed'] is False and guard['reason'] == 'daily_cap_reached'


def test_different_days_reset_count(tmp_path):
    """日付が変われば cap はリセット(翌日は再び3回使える)。"""
    ledger = tmp_path / 'led.json'
    day1 = datetime(2026, 6, 6, 10, 0, 0)
    for i in range(3):
        _restart(day1 + timedelta(minutes=20 * i), ledger, max_per_day=3, min_interval_sec=0)
    # 翌日
    day2 = datetime(2026, 6, 7, 10, 0, 0)
    r = _restart(day2, ledger, max_per_day=3, min_interval_sec=0)
    assert r['restarted'] is True
    assert r['day_count'] == 1


def test_dry_run_does_not_consume_budget(tmp_path):
    """dry_run は台帳を消費しない(cap を食わない)。"""
    ledger = tmp_path / 'led.json'
    now = datetime(2026, 6, 6, 10, 0, 0)
    r = w.restart_target(_target(), dry_run=True, now=now,
                         popen=lambda *a, **k: _FakeProc(),
                         active_checker=lambda _n: True, ledger_path=str(ledger))
    assert r['skipped_reason'] == 'dry_run'
    assert not os.path.exists(ledger)  # 台帳に何も書かない


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-v']))
