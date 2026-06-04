#!/usr/bin/env python3
"""今日(2026-06-04/05)の新ツールの挙動を固定するスモークテスト(将来の回帰防止)。
対象: race_day_check / per_race_coverage_check / paper_trade_s2b。
★本番ロジックは一切変えない・新ツールの安全挙動(誤通知しない/スクレイプしない)を固定★。
高速・ヘルメティック(tmp_path + 純粋関数中心)。重いWF学習(from-oof)は対象外。
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(BASE, 'tools')
for p in (TOOLS, BASE):
    if p not in sys.path:
        sys.path.insert(0, p)


# ============ race_day_check ============
def test_race_day_info_nonrace_day(tmp_path):
    """KYI が無いディレクトリ → 非開催日と判定(平日で誤検知しない)。"""
    import race_day_check as rdc
    info = rdc.jrdb_race_day_info('20260605', str(tmp_path))
    assert info['is_race_day'] is False
    assert info['n_races'] == 0
    assert info['source'] == ''


def test_race_day_info_race_day(tmp_path):
    """KYI(出走表)を置く → 開催日・レース数=先頭8文字のdistinct。"""
    import race_day_check as rdc
    kyi_dir = tmp_path / 'data' / 'jrdb' / 'extracted' / 'Kyi'
    kyi_dir.mkdir(parents=True)
    # 先頭8文字=レースキー。2レース×各5頭の行(>100bytes確保のため十分長く)。
    lines = []
    for race_key in ('06260601', '06260602'):
        for uma in range(1, 6):
            lines.append(race_key + str(uma).zfill(2) + ' ' * 30 + 'HORSE')
    (kyi_dir / 'KYI260606.txt').write_text('\n'.join(lines), encoding='cp932')
    info = rdc.jrdb_race_day_info('20260606', str(tmp_path))
    assert info['is_race_day'] is True
    assert info['n_races'] == 2
    assert info['source'].endswith('KYI260606.txt')


def test_fetch_robust_nonempty_passthrough(tmp_path):
    """fetch が非空 → そのまま返す・リトライしない・JRDB参照不要。"""
    import race_day_check as rdc
    calls = {'n': 0}

    def fetch():
        calls['n'] += 1
        return [{'race_id': 'r1'}, {'race_id': 'r2'}]

    rd = rdc.fetch_race_list_robust(fetch, '20260606', str(tmp_path), log=lambda *_: None)
    assert len(rd['races']) == 2
    assert rd['fetch_failed'] is False
    assert calls['n'] == 1  # 1回で確定(リトライなし)


def test_fetch_robust_nonrace_day_silent(tmp_path):
    """fetch空 + JRDB KYIなし = 真の非開催日 → 静かに返す・notify呼ばない。"""
    import race_day_check as rdc
    notified = []
    rd = rdc.fetch_race_list_robust(
        lambda: [], '20260605', str(tmp_path),
        log=lambda *_: None, notify_fn=lambda m: notified.append(m))
    assert rd['races'] == []
    assert rd['is_race_day'] is False
    assert rd['fetch_failed'] is False
    assert notified == []  # 平日は誤警告しない


def test_fetch_robust_racetoday_failure_warns(tmp_path):
    """fetch空 + JRDB開催日 → リトライ後も空なら fetch_failed=True かつ Discord警告を呼ぶ。"""
    import race_day_check as rdc
    kyi_dir = tmp_path / 'data' / 'jrdb' / 'extracted' / 'Kyi'
    kyi_dir.mkdir(parents=True)
    (kyi_dir / 'KYI260606.txt').write_text(
        '\n'.join('06260601' + str(i).zfill(2) + ' ' * 30 for i in range(1, 8)), encoding='cp932')
    notified = []
    rd = rdc.fetch_race_list_robust(
        lambda: [], '20260606', str(tmp_path),
        log=lambda *_: None, notify_fn=lambda m: notified.append(m),
        retries=2, delay_seconds=0)  # delay=0 でテスト高速
    assert rd['fetch_failed'] is True
    assert rd['is_race_day'] is True
    assert rd['n_races_jrdb'] == 1
    assert len(notified) == 1  # フェイルラウド(1回警告)


# ============ per_race_coverage_check ============
def test_coverage_nonrace_day_no_warn():
    """非開催日(KYIなし日)→ 開催0R・漏れ0・警告なし。notify=False で副作用なし(read-only)。"""
    import per_race_coverage_check as cov
    res = cov.check_coverage('20260605', notify=False)  # 6/5=非開催(KYI無)を確認済
    assert res['is_race_day'] is False
    assert res['n_races'] == 0
    assert res['missing'] == 0
    assert res['warned'] is False


# ============ paper_trade_s2b ============
def test_paper_predict_no_dump_safe_stop():
    """V15特徴ダンプが無い日 → ★新規スクレイプせず★ None で安全停止・pred生成しない。"""
    import paper_trade_s2b as paper
    date = '20991231'  # ダンプの無い未来日
    pred_path = paper._pred_path(date)
    if os.path.exists(pred_path):
        os.remove(pred_path)
    ret = paper.predict_date(date, allow_scrape=False)
    assert ret is None
    assert not os.path.exists(pred_path)  # 予測ログを作らない=スクレイプも予測もしていない


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-v']))
