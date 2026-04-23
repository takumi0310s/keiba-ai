"""payout 整合性テスト

regression_test.py から呼び出される。
- daily_results.py の result_row 辞書に actual_payout キーが含まれること
- cumulative_results.csv の HIT (trio_hit=1 or umaren_hit=1) で payout=0 が無いこと
- daily_results.py の actual_payout 計算式が trio/umaren に対応
"""
import os
import sys
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def test_daily_results_includes_actual_payout():
    """daily_results.py の result_row に 'actual_payout' キーがあること"""
    fp = os.path.join(BASE_DIR, 'tools', 'daily_results.py')
    with open(fp, 'r', encoding='utf-8') as f:
        src = f.read()
    # actual_payout が result_row に含まれている
    assert "'actual_payout': actual_payout" in src, \
        "daily_results.py の result_row に 'actual_payout' キーが欠落している (4/23 payout バグ再発)"


def test_actual_payout_formula_present():
    """actual_payout = trio_payout if trio_hit else (umaren_payout if umaren_hit else 0) 計算式"""
    fp = os.path.join(BASE_DIR, 'tools', 'daily_results.py')
    with open(fp, 'r', encoding='utf-8') as f:
        src = f.read()
    # 計算式の存在確認 (正規表現でゆるくマッチ)
    pat = r'actual_payout\s*=\s*trio_payout\s+if\s+trio_hit'
    assert re.search(pat, src), \
        "actual_payout の計算式が見つからない (trio/umaren からの導出)"


def test_cumulative_results_no_hit_with_zero_payout():
    """cumulative_results.csv で HIT 件は actual_payout > 0 (NaN含まず)"""
    csv = os.path.join(BASE_DIR, 'data', 'cumulative_results.csv')
    if not os.path.exists(csv):
        return  # ファイルなしはskip
    import pandas as pd
    df = pd.read_csv(csv, encoding='utf-8-sig')
    if len(df) == 0:
        return
    hit = df[(df.get('trio_hit', 0) == 1) | (df.get('umaren_hit', 0) == 1)]
    if len(hit) == 0:
        return  # 的中なしならskip
    # NaN または 0 を異常とする
    bad = hit[hit['actual_payout'].fillna(0) == 0]
    assert len(bad) == 0, \
        f"HITレースで actual_payout=0/NaN が {len(bad)} 件: race_ids={list(bad['race_id'].head(5))}"


def test_payout_critical_alert_in_daily_results():
    """HIT で payout=0 検知時に Discord CRITICAL 通知が送られるコードが存在"""
    fp = os.path.join(BASE_DIR, 'tools', 'daily_results.py')
    with open(fp, 'r', encoding='utf-8') as f:
        src = f.read()
    assert 'CRITICAL payout=0' in src, \
        "daily_results.py に CRITICAL payout=0 通知が無い"


if __name__ == '__main__':
    tests = [
        test_daily_results_includes_actual_payout,
        test_actual_payout_formula_present,
        test_cumulative_results_no_hit_with_zero_payout,
        test_payout_critical_alert_in_daily_results,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f'  PASS: {t.__name__}')
        except AssertionError as e:
            print(f'  FAIL: {t.__name__}: {e}')
            failed += 1
        except Exception as e:
            print(f'  ERROR: {t.__name__}: {e}')
            failed += 1
    print(f'\n{len(tests) - failed} passed, {failed} failed out of {len(tests)} tests')
    sys.exit(0 if failed == 0 else 1)
