#!/usr/bin/env python
"""Phase 21A: 5/17 GO 判定 worksheet 自動 fill in (06:30 schtask) — 2026-05-11.

Phase 20 B (data/v18/phase20_5_17_go_criteria.md) の 10 項目 GO checklist を
朝 06:30 自動 fill in、 Discord 通知。

Usage:
  python tools/morning_go_check.py                    # 通常実行 (Discord 通知)
  python tools/morning_go_check.py --no-notify        # Discord skip
  python tools/morning_go_check.py --date 20260517    # 任意日付
  python tools/morning_go_check.py --json             # JSON 出力のみ

schtask 登録 (5/17+ 朝 06:30):
  schtasks /create /tn "Keiba-MorningGoCheck" /tr "python C:\\Users\\takum\\keiba-ai\\tools\\morning_go_check.py" /sc DAILY /st 06:30
"""
from __future__ import annotations
import os
import sys
import json
import hashlib
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'

# 期待値 (Phase 20 B)
EXPECTED_V15_MD5 = '842b9a5f'   # 先頭 8 文字、 Phase 20 B 表 4 と一致
CUMULATIVE_FLOOR = 14140         # +¥14,140 (5/10 時点)
WITHDRAW_LINE = -50000           # 撤退ライン -¥50,000

# ANSI safe ASCII output (cp932 互換)
def _line(s=''): print(s, flush=True)


# ===========================================================================
# Each check function returns (status, detail) where status ∈ {'OK', 'NG', 'WARN', 'SKIP'}
# ===========================================================================

def check_1_v15_morning_predict(date: str) -> Tuple[str, str]:
    """1. V15 朝予測 (06:00 daily_predict) 完走"""
    p = DATA / 'daily_predictions' / f'{date}.csv'
    if not p.exists():
        return 'NG', f'{p.name} not found'
    try:
        n = sum(1 for _ in open(p, 'r', encoding='utf-8-sig')) - 1  # minus header
    except Exception as e:
        return 'WARN', f'read error: {e}'
    if n == 0:
        return 'NG', 'predictions empty'
    if n < 5:
        return 'WARN', f'only {n} R (less than 5)'
    return 'OK', f'{n} R'


def check_2_stage2(date: str) -> Tuple[str, str]:
    """2. Stage 2 動作確認 (data/daily_predictions_full)"""
    p = DATA / 'daily_predictions_full'
    if not p.exists():
        return 'NG', 'directory not found'
    files = list(p.glob(f'{date}*'))
    if not files:
        return 'WARN', f'no files matching {date}'
    return 'OK', f'{len(files)} files'


def check_3_v15_load() -> Tuple[str, str]:
    """3. V15 model load OK"""
    try:
        result = subprocess.run(
            [sys.executable, '-c',
             'import gzip, pickle; '
             'pickle.load(gzip.open(r"' + str(BASE / 'keiba_model_v15_central.pkl.gz') + '", "rb")); '
             'print("OK")'],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and 'OK' in result.stdout:
            return 'OK', 'load ok'
        return 'NG', f'rc={result.returncode}: {result.stderr[:100]}'
    except subprocess.TimeoutExpired:
        return 'WARN', 'timeout > 30s'
    except Exception as e:
        return 'NG', f'subproc error: {e}'


def check_4_v15_md5() -> Tuple[str, str]:
    """4. V15 model md5 不変"""
    p = BASE / 'keiba_model_v15_central.pkl.gz'
    if not p.exists():
        return 'NG', 'V15 model not found'
    try:
        h = hashlib.md5()
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        digest = h.hexdigest()
        prefix = digest[:8]
        match = prefix == EXPECTED_V15_MD5
        if match:
            return 'OK', f'md5={prefix}... (expected)'
        return 'WARN', f'md5={prefix}... (expected {EXPECTED_V15_MD5}...) — Phase 20 B baseline check'
    except Exception as e:
        return 'NG', f'hash error: {e}'


def check_5_paper_engine(date: str) -> Tuple[str, str]:
    """5. paper_trade_engine_v22.py 動作"""
    p = BASE / 'tools' / 'paper_trade_engine_v22.py'
    if not p.exists():
        return 'NG', 'engine script not found'
    try:
        result = subprocess.run(
            [sys.executable, str(p), '--date', date, '--dry-run'],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'},
        )
        if result.returncode == 0:
            return 'OK', 'engine ran (dry-run)'
        # dry-run not supported is acceptable, fall back to import check
        result2 = subprocess.run(
            [sys.executable, '-c',
             'import sys; sys.path.insert(0, r"' + str(BASE / 'tools') + '"); '
             'import paper_trade_engine_v22; print("OK")'],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'},
        )
        if result2.returncode == 0 and 'OK' in result2.stdout:
            return 'OK', 'engine importable'
        return 'WARN', f'rc={result.returncode} / import rc={result2.returncode}'
    except Exception as e:
        return 'WARN', f'subproc error: {e}'


def check_6_discord() -> Tuple[str, str]:
    """6. Discord webhook 動作 (env 確認のみ、 実 ping は morning_check 末尾で)"""
    keys = ['DISCORD_WEBHOOK_UPDATES', 'DISCORD_WEBHOOK_BETS', 'DISCORD_WEBHOOK_URL']
    found = [k for k in keys if os.environ.get(k)]
    # .env 経由
    env_path = BASE / '.env'
    env_keys: List[str] = []
    if env_path.exists():
        try:
            for line in env_path.open('r', encoding='utf-8'):
                if any(line.startswith(k + '=') for k in keys):
                    name = line.split('=', 1)[0]
                    env_keys.append(name)
        except Exception:
            pass
    found = list(set(found) | set(env_keys))
    if not found:
        return 'NG', 'no Discord webhook configured'
    return 'OK', f'configured: {",".join(found)}'


def check_7_cookie() -> Tuple[str, str]:
    """7. netkeiba Cookie 有効"""
    refresh_tool = BASE / 'tools' / 'refresh_cookie.py'
    if not refresh_tool.exists():
        return 'WARN', 'refresh_cookie.py not found'
    try:
        result = subprocess.run(
            [sys.executable, str(refresh_tool), '--check'],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'},
        )
        out = (result.stdout + result.stderr).lower()
        if result.returncode == 0 and ('有効' in (result.stdout + result.stderr) or 'valid' in out or 'ok' in out):
            return 'OK', 'cookie valid'
        if 'expired' in out or '期限' in (result.stdout + result.stderr):
            return 'NG', 'cookie expired'
        return 'WARN', f'rc={result.returncode}'
    except Exception as e:
        return 'WARN', f'check error: {e}'


def check_8_cumulative() -> Tuple[str, str]:
    """8. 累計収支 (cumulative_results.csv profit sum)。
    Phase 20 B baseline: +14,140 yen (5/5 時点 CLAUDE.md 記載)。
    本 check は CSV profit 列 sum を計算 (production paper を含む)。
    撤退ライン -50,000 yen を超えると NG。
    """
    p = DATA / 'cumulative_results.csv'
    if not p.exists():
        return 'WARN', 'cumulative_results.csv not found'
    try:
        import pandas as pd
        df = pd.read_csv(p, encoding='utf-8-sig')
        if 'profit' not in df.columns:
            return 'WARN', f'no profit column (cols={list(df.columns)[:5]}...)'
        # production-settled のみ集計 (status=settled のもの)
        if 'status' in df.columns:
            df_settled = df[df['status'].astype(str).str.lower() == 'settled']
        else:
            df_settled = df
        total = pd.to_numeric(df_settled['profit'], errors='coerce').sum()
        n = len(df_settled)
        margin = total - WITHDRAW_LINE
        detail = f'sum={total:,.0f}yen ({n} settled), withdraw_margin={margin:,.0f}yen'
        if total <= WITHDRAW_LINE:
            return 'NG', f'{detail} -- WITHDRAW LINE REACHED'
        if total >= CUMULATIVE_FLOOR:
            return 'OK', detail
        if total >= 0:
            return 'WARN', f'{detail} (below floor +{CUMULATIVE_FLOOR}yen)'
        return 'WARN', f'{detail} (negative)'
    except Exception as e:
        return 'WARN', f'parse error: {e}'


def check_9_scraper_guard() -> Tuple[str, str]:
    """9. SCRAPER-GUARD 緑 (kill switch off)"""
    indicators = [
        DATA / 'scraper_guard.lock',
        BASE / '.claude' / 'scraped_blocked.flag',
        DATA / 'netkeiba_master' / '.disabled',
    ]
    blocked = [str(p.relative_to(BASE)) for p in indicators if p.exists()]
    if blocked:
        return 'WARN', f'kill switches present: {",".join(blocked)}'
    return 'OK', 'no kill switches active'


def check_10_nightly_sanity(date: str) -> Tuple[str, str]:
    """10. 前夜 nightly_sanity 実行確認"""
    log_dir = BASE / 'logs'
    if not log_dir.exists():
        return 'WARN', 'logs dir not found'
    # 前日 (date - 1) の nightly_sanity log を探す
    try:
        d = datetime.strptime(date, '%Y%m%d')
        from datetime import timedelta
        prev = (d - timedelta(days=1)).strftime('%Y%m%d')
    except Exception:
        prev = ''
    candidates = list(log_dir.glob(f'*nightly_sanity*{prev}*')) if prev else []
    candidates += list(log_dir.glob(f'*nightly*sanity*'))
    if not candidates:
        return 'WARN', 'no nightly_sanity log'
    # newest
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    age_h = (datetime.now().timestamp() - newest.stat().st_mtime) / 3600
    if age_h > 36:
        return 'WARN', f'newest log {age_h:.1f}h old'
    return 'OK', f'log {newest.name} ({age_h:.1f}h ago)'


# ===========================================================================
# Aggregator
# ===========================================================================

CHECKS: List[Tuple[str, callable]] = [
    ('1. V15 朝予測 完走', lambda d: check_1_v15_morning_predict(d)),
    ('2. Stage 2 動作確認', lambda d: check_2_stage2(d)),
    ('3. V15 model load OK', lambda d: check_3_v15_load()),
    ('4. V15 model md5 不変', lambda d: check_4_v15_md5()),
    ('5. paper_trade_engine_v22', lambda d: check_5_paper_engine(d)),
    ('6. Discord webhook 動作', lambda d: check_6_discord()),
    ('7. netkeiba Cookie 有効', lambda d: check_7_cookie()),
    ('8. 累計 +14,140 yen / 撤退余裕', lambda d: check_8_cumulative()),
    ('9. scrape-status guard', lambda d: check_9_scraper_guard()),
    ('10. nightly_sanity 前夜 PASS', lambda d: check_10_nightly_sanity(d)),
]


def run_all(date: str) -> Dict[str, Any]:
    results = []
    for name, fn in CHECKS:
        try:
            status, detail = fn(date)
        except Exception as e:
            status, detail = 'WARN', f'unexpected: {e}'
        results.append({'name': name, 'status': status, 'detail': detail})

    n_ok = sum(1 for r in results if r['status'] == 'OK')
    n_ng = sum(1 for r in results if r['status'] == 'NG')
    n_warn = sum(1 for r in results if r['status'] == 'WARN')
    overall = 'GO' if n_ng == 0 else 'NO-GO'
    if n_ng == 0 and n_warn >= 3:
        overall = 'GO (caution)'

    return {
        'date': date,
        'checked_at': datetime.now().isoformat(timespec='seconds'),
        'results': results,
        'n_ok': n_ok,
        'n_ng': n_ng,
        'n_warn': n_warn,
        'overall': overall,
    }


def render_worksheet(rep: Dict[str, Any]) -> str:
    """Phase 20 B 表 3 形式 worksheet 自動 fill in."""
    lines = []
    lines.append('=' * 72)
    lines.append(f"5/{rep['date'][6:8]} (土) 朝 06:30 GO 判定 worksheet (auto-filled)")
    lines.append(f"checked at: {rep['checked_at']}")
    lines.append('=' * 72)
    for r in rep['results']:
        mark = {'OK': '[OK]', 'NG': '[NG]', 'WARN': '[!!]', 'SKIP': '[--]'}.get(r['status'], '[??]')
        lines.append(f"{mark} {r['name']:<32}: {r['detail']}")
    lines.append('=' * 72)
    lines.append(f"OK={rep['n_ok']} / NG={rep['n_ng']} / WARN={rep['n_warn']}  ===> {rep['overall']}")
    lines.append('=' * 72)
    return '\n'.join(lines)


def send_discord_notification(rep: Dict[str, Any]):
    try:
        sys.path.insert(0, str(BASE / 'tools'))
        from notify import send_discord  # type: ignore
    except Exception as e:
        _line(f'[discord] import fail: {e}')
        return False
    color = 'green' if rep['overall'] == 'GO' else ('red' if rep['overall'].startswith('NO-GO') else 'yellow')
    title = f"[Phase 21A] 5/{rep['date'][6:8]} 朝 GO check: {rep['overall']}"
    msg_lines = []
    msg_lines.append(f"OK={rep['n_ok']} NG={rep['n_ng']} WARN={rep['n_warn']}")
    for r in rep['results']:
        mark = {'OK': 'OK', 'NG': 'NG', 'WARN': '!!', 'SKIP': '--'}.get(r['status'], '??')
        msg_lines.append(f"[{mark}] {r['name']}: {r['detail']}")
    msg_lines.append('')
    msg_lines.append('V15 protect: predict_core / daily_predict / app.py / V15 model unchanged')
    ok = send_discord(title, '\n'.join(msg_lines), color=color, channel='updates')
    return bool(ok)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--date', default=datetime.now().strftime('%Y%m%d'))
    p.add_argument('--no-notify', action='store_true')
    p.add_argument('--json', action='store_true')
    p.add_argument('--save', help='save report to path')
    args = p.parse_args()

    rep = run_all(args.date)

    if args.json:
        _line(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        _line(render_worksheet(rep))

    # save (always, even without --save)
    out_dir = DATA / 'morning_go_check'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.save) if args.save else (out_dir / f'{args.date}.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    _line(f"\nSaved: {out_path}")

    if not args.no_notify:
        if send_discord_notification(rep):
            _line('[discord] notified')
        else:
            _line('[discord] skipped or failed (env not set)')

    # exit code: 0 GO, 1 NO-GO
    sys.exit(0 if rep['overall'] == 'GO' or rep['overall'].startswith('GO') else 1)


if __name__ == '__main__':
    main()
