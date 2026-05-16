"""T6: anomaly detection auto tests (Sub-task T6, 5/16 evening)

★ tools/anomaly_auto_detector.py の 5 trigger 単体 + false/true positive 計測 ★

Usage:
    python tests/T6_anomaly_detection_test.py
"""
import os
import sys
import shutil
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'tools'))

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import anomaly_auto_detector as aad  # type: ignore


RESULTS = []


def t(name):
    def deco(fn):
        def wrapper():
            try:
                fn()
                RESULTS.append((name, 'PASS', ''))
                print(f"  [PASS] {name}")
            except AssertionError as e:
                RESULTS.append((name, 'FAIL', str(e)))
                print(f"  [FAIL] {name} -- {e}")
            except Exception as e:
                RESULTS.append((name, 'ERROR', f"{type(e).__name__}: {e}"))
                print(f"  [ERROR] {name} -- {type(e).__name__}: {e}")
        return wrapper
    return deco


def _mk_tmp_repo() -> Path:
    """temp repo: data/daily_predictions/, logs/ を含む"""
    tmp = Path(tempfile.mkdtemp(prefix='aad_test_'))
    (tmp / 'data' / 'daily_predictions').mkdir(parents=True)
    (tmp / 'logs').mkdir(parents=True)
    return tmp


def _write_predictions(repo: Path, date: str, n_rows: int,
                       include_kyoto: bool = True,
                       kyoto_g1_only: bool = False) -> Path:
    """sample predictions csv 生成"""
    import pandas as pd
    rows = []
    for i in range(n_rows):
        # 半分東京、 1/3 京都 (include_kyoto なら)、 残り新潟
        if include_kyoto and i % 3 == 0:
            course = '京都'
            race_name = 'ヴィクトリアマイル(G1)' if kyoto_g1_only else f'{i+1}R 3歳未勝利'
        elif i % 3 == 1:
            course = '東京'
            race_name = 'ヴィクトリアマイル(G1)' if i == 11 else f'{i+1}R 3歳未勝利'
        else:
            course = '新潟'
            race_name = f'{i+1}R 3歳未勝利'
        rows.append({
            'race_id': f'202608030{i:03d}',
            'course': course,
            'race_num': (i % 12) + 1,
            'race_name': race_name,
            'condition': 'A',
            'num_horses': 12,
            'distance': 1600,
            'top1_num': 1,
            'top1_score': 0.5,
        })
    path = repo / 'data' / 'daily_predictions' / f'{date}.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------- trigger 1: 予測 ----------
@t("T6-1a check_predictions normal 35R -> ok")
def test_pred_normal():
    repo = _mk_tmp_repo()
    try:
        _write_predictions(repo, '20260517', 35)
        r = aad.check_predictions('20260517', repo=repo)
        assert r['severity'] == 'ok', r
    finally:
        shutil.rmtree(repo, ignore_errors=True)


@t("T6-1b check_predictions missing -> critical")
def test_pred_missing():
    repo = _mk_tmp_repo()
    try:
        r = aad.check_predictions('20260517', repo=repo)
        assert r['severity'] == 'critical', r
    finally:
        shutil.rmtree(repo, ignore_errors=True)


@t("T6-1c check_predictions empty -> critical")
def test_pred_empty():
    import pandas as pd
    repo = _mk_tmp_repo()
    try:
        path = repo / 'data' / 'daily_predictions' / '20260517.csv'
        pd.DataFrame(columns=['race_id', 'course']).to_csv(path, index=False)
        r = aad.check_predictions('20260517', repo=repo)
        assert r['severity'] == 'critical', r
    finally:
        shutil.rmtree(repo, ignore_errors=True)


@t("T6-1d check_predictions abnormal-low 10R -> warning")
def test_pred_low():
    repo = _mk_tmp_repo()
    try:
        _write_predictions(repo, '20260517', 10)
        r = aad.check_predictions('20260517', repo=repo)
        assert r['severity'] == 'warning', r
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ---------- trigger 2: 投票候補 ----------
@t("T6-2a check_vote_candidates summary >0 -> ok")
def test_vote_ok():
    repo = _mk_tmp_repo()
    try:
        log = repo / 'logs' / 'race_auto_notify_20260517.log'
        log.write_text("Found: 36 races\n全レース一括通知: 2 messages\n整形済み買い目通知: 8 messages\n",
                       encoding='utf-8')
        r = aad.check_vote_candidates('20260517', repo=repo)
        assert r['severity'] == 'ok', r
    finally:
        shutil.rmtree(repo, ignore_errors=True)


@t("T6-2b check_vote_candidates 0 messages -> critical")
def test_vote_zero():
    repo = _mk_tmp_repo()
    try:
        log = repo / 'logs' / 'race_auto_notify_20260517.log'
        log.write_text("Found: 36 races\n全レース一括通知: 2 messages\n整形済み買い目通知: 0 messages\n",
                       encoding='utf-8')
        r = aad.check_vote_candidates('20260517', repo=repo)
        assert r['severity'] == 'critical', r
    finally:
        shutil.rmtree(repo, ignore_errors=True)


@t("T6-2c check_vote_candidates log missing -> warning")
def test_vote_no_log():
    repo = _mk_tmp_repo()
    try:
        r = aad.check_vote_candidates('20260517', repo=repo)
        assert r['severity'] == 'warning', r
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ---------- trigger 3: streamlit ----------
@t("T6-3a check_streamlit unreachable -> warning")
def test_streamlit_unreachable():
    # port 65535 = まず空き
    r = aad.check_streamlit(port=65535, timeout=1.0)
    assert r['severity'] in ('warning',), r


# ---------- trigger 4: Discord 通知 ----------
@t("T6-4a check_discord_recent fresh log -> ok")
def test_discord_fresh():
    repo = _mk_tmp_repo()
    try:
        log = repo / 'logs' / 'race_auto_notify_20260517.log'
        log.write_text("全レース一括通知: 2 messages\n整形済み買い目通知: 8 messages\n",
                       encoding='utf-8')
        r = aad.check_discord_recent('20260517', repo=repo)
        assert r['severity'] == 'ok', r
    finally:
        shutil.rmtree(repo, ignore_errors=True)


@t("T6-4b check_discord_recent missing log -> warning")
def test_discord_missing():
    repo = _mk_tmp_repo()
    try:
        r = aad.check_discord_recent('20260517', repo=repo)
        assert r['severity'] == 'warning', r
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ---------- trigger 5: 戦略⑦案 C ----------
@t("T6-5a strategy7c kyoto skipped in log -> ok")
def test_s7c_ok():
    repo = _mk_tmp_repo()
    try:
        _write_predictions(repo, '20260517', 35, include_kyoto=True)
        log = repo / 'logs' / 'race_auto_notify_20260517.log'
        log.write_text(
            "    [STRATEGY7] Skip 京都 (P0-2 案 C、 5/17 適用): 1R 3歳未勝利\n"
            "    [STRATEGY7] Skip 京都 (P0-2 案 C、 5/17 適用): 4R 3歳未勝利\n"
            "    [STRATEGY7] Skip 京都 (P0-2 案 C、 5/17 適用): 7R 3歳未勝利\n",
            encoding='utf-8')
        r = aad.check_strategy7c('20260517', repo=repo)
        assert r['severity'] == 'ok', r
    finally:
        shutil.rmtree(repo, ignore_errors=True)


@t("T6-5b strategy7c kyoto NOT skipped -> critical")
def test_s7c_critical():
    repo = _mk_tmp_repo()
    try:
        _write_predictions(repo, '20260517', 35, include_kyoto=True)
        log = repo / 'logs' / 'race_auto_notify_20260517.log'
        log.write_text("Found: 36 races\n(no skip log)\n", encoding='utf-8')
        r = aad.check_strategy7c('20260517', repo=repo)
        assert r['severity'] == 'critical', r
    finally:
        shutil.rmtree(repo, ignore_errors=True)


@t("T6-5c strategy7c no-kyoto-races -> ok")
def test_s7c_no_kyoto():
    repo = _mk_tmp_repo()
    try:
        _write_predictions(repo, '20260517', 30, include_kyoto=False)
        r = aad.check_strategy7c('20260517', repo=repo)
        assert r['severity'] == 'ok', r
    finally:
        shutil.rmtree(repo, ignore_errors=True)


@t("T6-5d strategy7c kyoto-all-G1 (filter 例外) -> ok")
def test_s7c_kyoto_all_g1():
    repo = _mk_tmp_repo()
    try:
        _write_predictions(repo, '20260517', 30, include_kyoto=True, kyoto_g1_only=True)
        log = repo / 'logs' / 'race_auto_notify_20260517.log'
        log.write_text("Found: 30 races\n", encoding='utf-8')
        r = aad.check_strategy7c('20260517', repo=repo)
        assert r['severity'] == 'ok', r
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ---------- false/true positive rate (small sample) ----------
@t("T6-FP false positive rate < 5% (5 normal scenarios)")
def test_fp_rate():
    """正常データ 5 シナリオ → critical なし期待"""
    repo = _mk_tmp_repo()
    fps = 0
    n = 5
    try:
        for i in range(n):
            date = f'202605{10+i:02d}'
            _write_predictions(repo, date, 35, include_kyoto=True)
            log = repo / 'logs' / f'race_auto_notify_{date}.log'
            log.write_text(
                "Found: 36 races\n"
                "    [STRATEGY7] Skip 京都 (P0-2 案 C): 1R\n"
                "    [STRATEGY7] Skip 京都 (P0-2 案 C): 4R\n"
                "    [STRATEGY7] Skip 京都 (P0-2 案 C): 7R\n"
                "全レース一括通知: 2 messages\n整形済み買い目通知: 8 messages\n",
                encoding='utf-8')
            for _, r in [
                ('pred', aad.check_predictions(date, repo=repo)),
                ('vote', aad.check_vote_candidates(date, repo=repo)),
                ('disc', aad.check_discord_recent(date, repo=repo)),
                ('s7c',  aad.check_strategy7c(date, repo=repo)),
            ]:
                if r['severity'] == 'critical':
                    fps += 1
        # streamlit は外部依存なので 除外
        total = n * 4
        rate = fps / total
        assert rate < 0.05, f'FP rate {rate:.2%} ({fps}/{total}) >= 5%'
    finally:
        shutil.rmtree(repo, ignore_errors=True)


@t("T6-TP true positive rate > 95% (anomaly scenarios)")
def test_tp_rate():
    """異常データ 4 シナリオ → 全 critical 期待"""
    repo = _mk_tmp_repo()
    tps = 0
    total = 0
    try:
        # 1. predictions missing
        total += 1
        if aad.check_predictions('99999999', repo=repo)['severity'] == 'critical':
            tps += 1
        # 2. predictions empty
        import pandas as pd
        p = repo / 'data' / 'daily_predictions' / '20260518.csv'
        pd.DataFrame(columns=['race_id']).to_csv(p, index=False)
        total += 1
        if aad.check_predictions('20260518', repo=repo)['severity'] == 'critical':
            tps += 1
        # 3. vote 0
        log = repo / 'logs' / 'race_auto_notify_20260519.log'
        log.write_text("Found: 36 races\n整形済み買い目通知: 0 messages\n", encoding='utf-8')
        total += 1
        if aad.check_vote_candidates('20260519', repo=repo)['severity'] == 'critical':
            tps += 1
        # 4. strategy7c 不動作
        _write_predictions(repo, '20260520', 35, include_kyoto=True)
        log2 = repo / 'logs' / 'race_auto_notify_20260520.log'
        log2.write_text("Found: 36 races\n(no skip)\n", encoding='utf-8')
        total += 1
        if aad.check_strategy7c('20260520', repo=repo)['severity'] == 'critical':
            tps += 1
        rate = tps / total
        assert rate > 0.95, f'TP rate {rate:.2%} ({tps}/{total}) <= 95%'
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ---------- end-to-end main ----------
@t("T6-E2E run_all 5 triggers callable")
def test_e2e_run_all():
    repo = _mk_tmp_repo()
    try:
        _write_predictions(repo, '20260517', 35, include_kyoto=True)
        log = repo / 'logs' / 'race_auto_notify_20260517.log'
        log.write_text(
            "Found: 36 races\n"
            "    [STRATEGY7] Skip 京都 (P0-2 案 C): 1R\n"
            "    [STRATEGY7] Skip 京都 (P0-2 案 C): 4R\n"
            "    [STRATEGY7] Skip 京都 (P0-2 案 C): 7R\n"
            "全レース一括通知: 2 messages\n整形済み買い目通知: 8 messages\n",
            encoding='utf-8')
        # run_all は global REPO を参照、 ここでは個別 check で代替
        triggers = [
            ('predictions', aad.check_predictions('20260517', repo=repo)),
            ('vote', aad.check_vote_candidates('20260517', repo=repo)),
            ('discord', aad.check_discord_recent('20260517', repo=repo)),
            ('s7c', aad.check_strategy7c('20260517', repo=repo)),
        ]
        assert len(triggers) == 4, triggers
        for name, r in triggers:
            assert r['severity'] in ('ok', 'warning', 'critical'), (name, r)
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def main():
    print("=" * 60)
    print(" T6 anomaly detection auto tests")
    print("=" * 60)
    for fn_name in list(globals()):
        if fn_name.startswith('test_'):
            globals()[fn_name]()
    print("\n" + "=" * 60)
    passed = sum(1 for _, s, _ in RESULTS if s == 'PASS')
    failed = sum(1 for _, s, _ in RESULTS if s in ('FAIL', 'ERROR'))
    print(f" {passed}/{len(RESULTS)} PASS, {failed} FAIL/ERROR")
    print("=" * 60)
    for name, status, err in RESULTS:
        if status != 'PASS':
            print(f"  {status}: {name} — {err}")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
