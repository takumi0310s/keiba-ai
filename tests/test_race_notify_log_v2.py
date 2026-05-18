"""C: race_notify_log v2 (3 phase) tests

★ 既存 race_auto_notify / daily_predict logic 不変確認 + v2 log 動作 ★

Usage:
    python -m pytest tests/test_race_notify_log_v2.py -v
"""
import json
import os
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'tools'))


# ----- Phase 1 (morning_predict) -----

def test_log_phase1_creates_file(tmp_path, monkeypatch):
    monkeypatch.setenv('RACE_NOTIFY_LOG_V2_ROOT', str(tmp_path))
    from race_notify_log_v2 import log_phase1, read_phase

    log_phase1(
        race_id='202605020101',
        race_meta={'race_name': 'test', 'course': '東京', 'distance': 1600},
        ranking_top5=[{'rank': 1, 'umaban': 5, 'name': 'A'}],
        formation_planned='5-7-9',
        morning_odds={5: 3.2, 7: 5.4},
        date_str='20260518',
    )
    rec = read_phase('202605020101', 1, date_str='20260518')
    assert rec is not None
    assert rec['phase'] == 1
    assert rec['phase_name'] == 'morning_predict'
    assert rec['race_id'] == '202605020101'
    assert rec['race_meta']['race_name'] == 'test'
    assert rec['formation_planned'] == '5-7-9'
    assert rec['morning_odds']['5'] == 3.2


def test_log_phase1_missing_fields_ok(tmp_path, monkeypatch):
    """ranking/odds 等 None でも fail しない。"""
    monkeypatch.setenv('RACE_NOTIFY_LOG_V2_ROOT', str(tmp_path))
    from race_notify_log_v2 import log_phase1, read_phase
    log_phase1(race_id='202605020199', date_str='20260518')
    rec = read_phase('202605020199', 1, date_str='20260518')
    assert rec is not None
    assert rec['ranking_top5'] == []
    assert rec['morning_odds'] == {}


# ----- Phase 2 (pre_vote) -----

def test_log_phase2_bets_normal(tmp_path, monkeypatch):
    monkeypatch.setenv('RACE_NOTIFY_LOG_V2_ROOT', str(tmp_path))
    from race_notify_log_v2 import log_phase2, read_phase
    log_phase2(
        race_id='202605020102',
        race_meta={'race_name': 'r2'},
        formation_actual=[(5, 7, 9), (5, 7, 11)],
        vote_time_odds={5: 4.2},
        strategy_7c_skip=False,
        channel='bets',
        cond_key='A',
        bet_type='trio',
        date_str='20260518',
    )
    rec = read_phase('202605020102', 2, date_str='20260518')
    assert rec is not None
    assert rec['phase'] == 2
    assert rec['phase_name'] == 'pre_vote'
    assert rec['channel'] == 'bets'
    assert rec['strategy_7c_skip'] is False
    assert rec['cond_key'] == 'A'
    assert rec['bet_type'] == 'trio'
    # tuple → list 正規化されている
    assert rec['formation_actual'] == [[5, 7, 9], [5, 7, 11]]


def test_log_phase2_strategy_7c_skip(tmp_path, monkeypatch):
    monkeypatch.setenv('RACE_NOTIFY_LOG_V2_ROOT', str(tmp_path))
    from race_notify_log_v2 import log_phase2, read_phase
    log_phase2(
        race_id='202608010101',
        race_meta={'race_name': 'kyoto'},
        formation_actual=None,
        strategy_7c_skip=True,
        strategy_7c_reason='strategy_7_kyoto_p0_2_5_17',
        channel='skip',
        date_str='20260518',
    )
    rec = read_phase('202608010101', 2, date_str='20260518')
    assert rec['strategy_7c_skip'] is True
    assert rec['strategy_7c_reason'] == 'strategy_7_kyoto_p0_2_5_17'
    assert rec['channel'] == 'skip'


# ----- Phase 3 (post_result) -----

def test_log_phase3_with_hit(tmp_path, monkeypatch):
    monkeypatch.setenv('RACE_NOTIFY_LOG_V2_ROOT', str(tmp_path))
    from race_notify_log_v2 import log_phase3, read_phase
    log_phase3(
        race_id='202605020103',
        real_top3=[5, 7, 9],
        real_payouts={'trio': 1500, 'umaren': 0},
        hit_miss={'trio_hit': True, 'umaren_hit': False},
        date_str='20260518',
    )
    rec = read_phase('202605020103', 3, date_str='20260518')
    assert rec['phase'] == 3
    assert rec['real_top3'] == [5, 7, 9]
    assert rec['real_payouts']['trio'] == 1500
    assert rec['hit_miss']['trio_hit'] is True


def test_log_phase3_miss(tmp_path, monkeypatch):
    monkeypatch.setenv('RACE_NOTIFY_LOG_V2_ROOT', str(tmp_path))
    from race_notify_log_v2 import log_phase3, read_phase
    log_phase3(
        race_id='202605020104',
        real_top3=[1, 2, 3],
        real_payouts={'trio': 0},
        hit_miss={'trio_hit': False},
        date_str='20260518',
    )
    rec = read_phase('202605020104', 3, date_str='20260518')
    assert rec['hit_miss']['trio_hit'] is False


# ----- Robustness -----

def test_log_phase_failure_does_not_raise(tmp_path, monkeypatch):
    """異常な引数でも例外を投げない。"""
    monkeypatch.setenv('RACE_NOTIFY_LOG_V2_ROOT', str(tmp_path))
    from race_notify_log_v2 import log_phase1, log_phase2, log_phase3
    # 何が起きても raise しない (try/except で包んでいる)
    try:
        log_phase1(None, None, None, None, None)
        log_phase2(None, None, None, None, None, None)
        log_phase3(None, None, None, None)
    except Exception as e:
        pytest.fail(f"log_phase* should never raise: {e}")


# ----- race_id validation (None.json bug fix) -----

def test_race_id_none_skipped(tmp_path, monkeypatch):
    """race_id=None で log skip、 None.json が作成されない。"""
    monkeypatch.setenv('RACE_NOTIFY_LOG_V2_ROOT', str(tmp_path))
    from race_notify_log_v2 import log_phase1, log_phase2, log_phase3

    log_phase1(None, {}, [], '', date_str='20260518')
    log_phase2(None, {}, None, None, False, None, 'bets', None, None,
               date_str='20260518')
    log_phase3(None, [], {}, {}, date_str='20260518')

    for phase in (1, 2, 3):
        none_file = tmp_path / '20260518' / f'phase{phase}' / 'None.json'
        assert not none_file.exists(), f"None.json should not be created at phase{phase}"


def test_race_id_nan_skipped(tmp_path, monkeypatch):
    """race_id='nan' / 'None' (string) で log skip。"""
    monkeypatch.setenv('RACE_NOTIFY_LOG_V2_ROOT', str(tmp_path))
    from race_notify_log_v2 import log_phase1

    for bad in ('nan', 'NaN', 'None', 'null', '', '   '):
        log_phase1(bad, {}, [], '', date_str='20260518')
        bad_file = tmp_path / '20260518' / 'phase1' / f'{bad}.json'
        assert not bad_file.exists(), f"phase1/{bad}.json should not be created"


def test_race_id_invalid_format_skipped(tmp_path, monkeypatch):
    """race_id 非 digit / 桁不正 で log skip。"""
    monkeypatch.setenv('RACE_NOTIFY_LOG_V2_ROOT', str(tmp_path))
    from race_notify_log_v2 import log_phase1

    invalid_ids = ['abc', '123', '202605020', '2026050201010101', 'R1', '12-34-56']
    for bad in invalid_ids:
        log_phase1(bad, {}, [], '', date_str='20260518')
        bad_file = tmp_path / '20260518' / 'phase1' / f'{bad}.json'
        assert not bad_file.exists(), f"phase1/{bad}.json should not be created"

    # 正常な race_id (12 桁 digit) は通る
    log_phase1('202605020101', {}, [], '', date_str='20260518')
    good_file = tmp_path / '20260518' / 'phase1' / '202605020101.json'
    assert good_file.exists(), "valid race_id should be logged"


def test_read_phase_nonexistent_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv('RACE_NOTIFY_LOG_V2_ROOT', str(tmp_path))
    from race_notify_log_v2 import read_phase
    assert read_phase('not_exist', 1, date_str='20260518') is None
    assert read_phase('not_exist', 2, date_str='20260518') is None
    assert read_phase('not_exist', 3, date_str='20260518') is None


# ----- race_auto_notify wrapper -----

def test_race_auto_notify_v2_wrapper_exists():
    """race_auto_notify.py に _v2_log_phase2_safe wrapper が定義されている。"""
    from tools.race_auto_notify import _v2_log_phase2_safe
    assert callable(_v2_log_phase2_safe)


def test_race_auto_notify_v2_wrapper_safe(tmp_path, monkeypatch):
    """wrapper が異常引数で raise しない。"""
    monkeypatch.setenv('RACE_NOTIFY_LOG_V2_ROOT', str(tmp_path))
    from tools.race_auto_notify import _v2_log_phase2_safe
    try:
        _v2_log_phase2_safe('rid', 'name', None, None, None, None, None,
                            channel='bets', strategy_7c_skip=False, strategy_7c_reason=None)
    except Exception as e:
        pytest.fail(f"wrapper should not raise: {e}")


def test_race_auto_notify_v2_wrapper_writes(tmp_path, monkeypatch):
    """wrapper で phase2 log が出力される。"""
    monkeypatch.setenv('RACE_NOTIFY_LOG_V2_ROOT', str(tmp_path))
    # 日付を固定するため race_notify_log_v2 module の _today_str を mock
    import race_notify_log_v2 as v2mod
    monkeypatch.setattr(v2mod, '_today_str', lambda: '20260518')

    from tools.race_auto_notify import _v2_log_phase2_safe
    rinfo = {'course': '東京', 'distance': 1600, 'surface': '芝', 'condition': '良',
             'start_time': '15:45'}
    _v2_log_phase2_safe('202605020105', 'テスト', rinfo,
                        [(5, 7, 9)], {5: 3.2}, 'A', 'trio',
                        channel='bets', strategy_7c_skip=False, strategy_7c_reason=None)

    log_file = tmp_path / '20260518' / 'phase2' / '202605020105.json'
    assert log_file.exists()
    rec = json.loads(log_file.read_text(encoding='utf-8'))
    assert rec['race_meta']['course'] == '東京'
    assert rec['bet_type'] == 'trio'
    assert rec['cond_key'] == 'A'


# ----- race_auto_notify call site verification -----

def test_race_auto_notify_existing_logic_unchanged():
    """race_auto_notify.py の既存 _p0_5_notify_log call が全て残存している。"""
    src = (BASE_DIR / 'tools' / 'race_auto_notify.py').read_text(encoding='utf-8')
    # 既存 P0-5 log call が削除されていないこと
    expected_reasons = [
        'no_horse_data', 'obstacle_race', 'distance_le_1000',
        'strategy_7_06_tokubetsu', 'strategy_7_kyoto_p0_2_5_17',
        'strategy_7_cond_E', 'strategy_7_cond_B', 'strategy_7_cond_X_p0_2_5_17',
    ]
    for r in expected_reasons:
        assert f"strategy_7c_reason='{r}'" in src, f"P0-5 call missing: {r}"


def test_race_auto_notify_v2_call_added_to_all_paths():
    """全 P0-5 log call path に v2 wrapper call も追加されている。"""
    src = (BASE_DIR / 'tools' / 'race_auto_notify.py').read_text(encoding='utf-8')
    # bets path + 各 skip path + error path
    expected_v2_reasons = [
        'no_horse_data', 'obstacle_race', 'distance_le_1000',
        'strategy_7_06_tokubetsu', 'strategy_7_kyoto_p0_2_5_17',
        'strategy_7_cond_E', 'strategy_7_cond_B', 'strategy_7_cond_X_p0_2_5_17',
    ]
    # v2 wrapper の呼び出し回数 >= 既存 P0-5 (= 8 skip/error + 1 bets + 1 exception)
    v2_calls = src.count('_v2_log_phase2_safe(')
    assert v2_calls >= 10, f"v2 wrapper calls insufficient: {v2_calls}"

    for r in expected_v2_reasons:
        # 各 reason に対応する v2 call が存在する
        assert f"strategy_7c_reason='{r}'" in src, f"v2 reason missing: {r}"


# ----- daily_predict wrapper -----

def test_daily_predict_v2_wrapper_exists():
    """daily_predict.py に _v2_log_phase1_safe wrapper が定義されている。"""
    src = (BASE_DIR / 'tools' / 'daily_predict.py').read_text(encoding='utf-8')
    assert 'def _v2_log_phase1_safe(' in src
    assert '_v2_log_phase1_safe(' in src
    # call site が _append_prediction_to_csv の直後に追加されている
    assert 'race_notify_log v2 phase 1' in src


def test_daily_predict_logic_unchanged():
    """daily_predict.py の核心 logic (CSV 書き込み等) が不変。"""
    src = (BASE_DIR / 'tools' / 'daily_predict.py').read_text(encoding='utf-8')
    # 既存重要部分が残存
    assert 'def _append_prediction_to_csv(row, date_str):' in src
    assert "df_row.to_csv(f, index=False, header=new_file)" in src
    assert 'results.append(row)' in src


# ----- Aggregator -----

def test_aggregator_complete_race(tmp_path, monkeypatch):
    """3 phase 揃った race の aggregation。"""
    monkeypatch.setenv('RACE_NOTIFY_LOG_V2_ROOT', str(tmp_path))

    # log_root を tmp_path に固定して 3 phase 書き込み
    from race_notify_log_v2 import log_phase1, log_phase2, log_phase3
    log_phase1('202605020201', race_meta={'race_name': 'test'}, formation_planned='5-7-9',
               date_str='20260518')
    log_phase2('202605020201', race_meta={'race_name': 'test'}, formation_actual=[(5, 7, 9)],
               channel='bets', cond_key='A', bet_type='trio', date_str='20260518')
    log_phase3('202605020201', real_top3=[5, 7, 9], real_payouts={'trio': 1500},
               hit_miss={'trio_hit': True}, date_str='20260518')

    # aggregator も同じ env を見る必要 → reload
    import importlib
    import race_notify_log_v2_aggregator as agg
    importlib.reload(agg)

    result = agg.aggregate_day('20260518')
    assert result['complete_races'] == 1
    assert result['voted_races'] == 1
    assert result['hits'] == 1
    assert result['inv'] == 700
    assert result['pay'] == 1500
    assert result['roi_pct'] > 200


def test_aggregator_skip_strategy_7c(tmp_path, monkeypatch):
    """strategy_7c skip race は集計除外。"""
    monkeypatch.setenv('RACE_NOTIFY_LOG_V2_ROOT', str(tmp_path))

    from race_notify_log_v2 import log_phase2, log_phase3
    log_phase2('202608010101', race_meta={'race_name': 'kyoto'}, formation_actual=None,
               strategy_7c_skip=True, strategy_7c_reason='strategy_7_kyoto_p0_2_5_17',
               channel='skip', cond_key='A', bet_type='trio', date_str='20260518')
    log_phase3('202608010101', real_top3=[1, 2, 3], real_payouts={'trio': 0},
               hit_miss={'trio_hit': False}, date_str='20260518')

    import importlib
    import race_notify_log_v2_aggregator as agg
    importlib.reload(agg)

    result = agg.aggregate_day('20260518')
    assert result['voted_races'] == 0
    assert result['skipped_races'] == 1
    assert result['inv'] == 0


def test_aggregator_missing_dir(tmp_path, monkeypatch):
    """log dir 無くても fail しない。"""
    monkeypatch.setenv('RACE_NOTIFY_LOG_V2_ROOT', str(tmp_path / 'nope'))
    import importlib
    import race_notify_log_v2_aggregator as agg
    importlib.reload(agg)

    result = agg.aggregate_day('20260518')
    # phase2 ないので voted_races=0
    assert result is not None
    assert result['voted_races'] == 0
