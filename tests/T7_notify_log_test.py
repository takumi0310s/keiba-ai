"""T7: P0-5 race_notify_log 動作テスト (★ 既存 logic 不変確認 ★)

race_auto_notify.py に追加した _p0_5_notify_log 関数のテスト。
既存 logic は 1 行も変更しないため、 ここでは log 出力 file IO のみ検証する。

Usage:
    python -m pytest tests/T7_notify_log_test.py -v
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def test_notify_log_function_exists():
    """_p0_5_notify_log 関数が race_auto_notify.py に存在する"""
    from tools.race_auto_notify import _p0_5_notify_log
    assert callable(_p0_5_notify_log), "_p0_5_notify_log must be callable"


def test_notify_log_writes_json(tmp_path, monkeypatch):
    """log file が data/race_notify_log/YYYYMMDD.json に作成される"""
    # data/race_notify_log の親ディレクトリを tmp_path に向ける
    fake_base = tmp_path
    (fake_base / 'data').mkdir(parents=True, exist_ok=True)

    # race_auto_notify.py 内で Path(__file__).resolve().parents[1] を使うので、
    # tools/ 直下の擬似 module path を作る
    fake_tools = fake_base / 'tools'
    fake_tools.mkdir(parents=True, exist_ok=True)
    fake_module = fake_tools / 'race_auto_notify.py'

    # 実 module の関数を import
    from tools.race_auto_notify import _p0_5_notify_log

    # __file__ を一時的に書き換えて log dir を tmp_path に向ける
    import tools.race_auto_notify as mod
    orig_file = mod.__file__
    try:
        mod.__file__ = str(fake_module)
        race_id = '202605020999'
        race_name = 'テストレース'
        notified_at = datetime.now().isoformat()
        _p0_5_notify_log(race_id, race_name, notified_at, channel='bets',
                         strategy_7c_skip=False, strategy_7c_reason=None)

        date_str = datetime.now().strftime('%Y%m%d')
        log_file = fake_base / 'data' / 'race_notify_log' / f'{date_str}.json'
        assert log_file.exists(), f"log file not created: {log_file}"

        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        assert isinstance(logs, list), "log file must contain a list"
        assert len(logs) >= 1, "log file must contain at least 1 entry"
        last = logs[-1]
        assert last['race_id'] == race_id
        assert last['race_name'] == race_name
        assert last['channel'] == 'bets'
        assert last['strategy_7c_skip'] is False
    finally:
        mod.__file__ = orig_file


def test_notify_log_appends(tmp_path):
    """同一 file への append 動作確認 (複数 race の通知が累積する)"""
    import tools.race_auto_notify as mod
    from tools.race_auto_notify import _p0_5_notify_log

    fake_base = tmp_path
    fake_tools = fake_base / 'tools'
    fake_tools.mkdir(parents=True, exist_ok=True)
    fake_module = fake_tools / 'race_auto_notify.py'

    orig_file = mod.__file__
    try:
        mod.__file__ = str(fake_module)
        for i in range(3):
            _p0_5_notify_log(f'race_{i}', f'name_{i}',
                             datetime.now().isoformat(), channel='bets')

        date_str = datetime.now().strftime('%Y%m%d')
        log_file = fake_base / 'data' / 'race_notify_log' / f'{date_str}.json'
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        # 直近 3 件分 (この test 専用 dir なので 3 件のみ)
        assert len(logs) == 3, f"expected 3 entries, got {len(logs)}"
        assert logs[0]['race_id'] == 'race_0'
        assert logs[2]['race_id'] == 'race_2'
    finally:
        mod.__file__ = orig_file


def test_notify_log_strategy_7c_skip_logged(tmp_path):
    """strategy_7c_skip=True + reason が正しく log される"""
    import tools.race_auto_notify as mod
    from tools.race_auto_notify import _p0_5_notify_log

    fake_base = tmp_path
    fake_tools = fake_base / 'tools'
    fake_tools.mkdir(parents=True, exist_ok=True)
    fake_module = fake_tools / 'race_auto_notify.py'

    orig_file = mod.__file__
    try:
        mod.__file__ = str(fake_module)
        _p0_5_notify_log('202605020888', '京都10R', datetime.now().isoformat(),
                         channel='skip', strategy_7c_skip=True,
                         strategy_7c_reason='strategy_7_kyoto_p0_2_5_17')

        date_str = datetime.now().strftime('%Y%m%d')
        log_file = fake_base / 'data' / 'race_notify_log' / f'{date_str}.json'
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        last = logs[-1]
        assert last['strategy_7c_skip'] is True
        assert last['strategy_7c_reason'] == 'strategy_7_kyoto_p0_2_5_17'
        assert last['channel'] == 'skip'
    finally:
        mod.__file__ = orig_file


def test_notify_log_failure_does_not_raise():
    """log fail (file IO error) は exception を投げず stderr 出力のみ"""
    from tools.race_auto_notify import _p0_5_notify_log
    # log 関数自体は try/except で包まれているので、
    # 異常な引数でも raise しないこと
    try:
        _p0_5_notify_log(None, None, None, channel=None,
                         strategy_7c_skip=False, strategy_7c_reason=None)
    except Exception as e:
        assert False, f"_p0_5_notify_log should never raise: {e}"


def test_existing_notify_paths_have_log_call():
    """既存通知 path (Discord notify / strategy_7 skip) すべてに log call が追加されている

    既存 logic 1 行不変、 log call 1 行追加のみ確認。
    """
    fpath = Path(BASE_DIR) / 'tools' / 'race_auto_notify.py'
    with open(fpath, 'r', encoding='utf-8') as f:
        src = f.read()

    # Discord 通知後 (send_discord(..., channel="bets") 直後) に log call があること
    assert '_p0_5_notify_log(race_id, race_name, datetime.now().isoformat(), channel=\'bets\'' in src, \
        "bets 通知後の log call が見つからない"

    # strategy_7 各 skip path に log call があること
    expected_reasons = [
        'no_horse_data', 'obstacle_race', 'distance_le_1000',
        'strategy_7_06_tokubetsu', 'strategy_7_kyoto_p0_2_5_17',
        'strategy_7_cond_E', 'strategy_7_cond_B', 'strategy_7_cond_X_p0_2_5_17',
    ]
    for reason in expected_reasons:
        assert f"strategy_7c_reason='{reason}'" in src, \
            f"strategy_7 skip log call missing for reason: {reason}"


if __name__ == '__main__':
    # pytest なしの直接実行用 (簡易)
    import tempfile
    tests = [
        test_notify_log_function_exists,
        test_notify_log_failure_does_not_raise,
        test_existing_notify_paths_have_log_call,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
    # tmp_path を必要とする test は pytest 経由でのみ実行
    print(f"\n{passed}/{len(tests)} direct tests passed (use pytest for full suite)")
