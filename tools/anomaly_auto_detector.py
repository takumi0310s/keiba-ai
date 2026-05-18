"""
anomaly auto detector — 5 trigger 全自動 check (Sub-task T6, 5/16 evening)

usage:
  python tools/anomaly_auto_detector.py [--date YYYYMMDD] [--no-discord] [--severity critical|warning|all]

★ V15 production 完全不変 ★、 read-only check + Discord 通知のみ。

5 trigger:
1. 予測ゼロ           — data/daily_predictions/{date}.csv 不在 or 0 rows or <20 rows
2. 投票候補 0R         — race_auto_notify_{date}.log で「整形済み買い目通知: 0 messages」
                          または 全 race [STRATEGY7] Skip
3. Streamlit 起動失敗  — port 8501 HTTP GET 失敗 (G1 day 朝の dashboard 確認用)
4. Discord 通知なし    — race_auto_notify_{date}.log の最終 mtime が 古すぎる or 通知 0 messages
5. 戦略⑦案 C 不動作    — daily_predictions に 京都 R 存在し、 race_auto_notify_{date}.log に
                          「[STRATEGY7] Skip 京都」 が 1 つも無い (= filter 通り抜け)

exit code:
  0 — ✅ 全 OK
  1 — ⚠ warning のみ
  2 — ★ critical あり

honest 注記:
  - false positive rate / true positive rate は 実 test (tests/T6_anomaly_detection_test.py) で 計測
  - detection logic は 「絶対」 ではなく 「現行 logic + log pattern」 ベース
  - rollback 推奨は warning では 行わない (streamlit 不可は投票影響 0)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Windows cp932 console で 絵文字記号印字 → utf-8 化 (schtask redirect 対策、 errors='replace' で 安全化)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# requests / pandas は遅延 import (test 軽量化)


SEVERITY_ICON = {
    'critical': '★',
    'warning': '⚠',
    'ok': '✅',
}


def _result(severity: str, msg: str, **extra) -> dict:
    out = {'severity': severity, 'msg': msg}
    out.update(extra)
    return out


# ---------- trigger 1: 予測ゼロ ----------
def check_predictions(date_str: str, repo: Path = REPO) -> dict:
    """daily_predictions/{date}.csv の存在 + 行数 check"""
    path = repo / 'data' / 'daily_predictions' / f'{date_str}.csv'
    if not path.exists():
        return _result('critical', f'predictions file 不在: data/daily_predictions/{date_str}.csv')
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception as e:
        return _result('critical', f'predictions 読込失敗: {e}')
    n = len(df)
    if n == 0:
        return _result('critical', f'predictions 0 rows')
    if n < 20:
        return _result('warning', f'predictions 異常 少 ({n} R、 通常 30+)')
    return _result('ok', f'predictions {n} R OK', n=n)


# ---------- trigger 2: 投票候補 0R ----------
def check_vote_candidates(date_str: str, repo: Path = REPO) -> dict:
    """race_auto_notify_{date}.log で 「整形済み買い目通知: N messages」 と Skip カウント"""
    log_path = repo / 'logs' / f'race_auto_notify_{date_str}.log'
    if not log_path.exists():
        return _result('warning', f'race_auto_notify_{date_str}.log 不在 (まだ起動前?)')
    try:
        text = log_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return _result('warning', f'log 読込失敗: {e}')

    # 整形済み買い目通知 messages 数 抽出
    m = re.search(r'整形済み買い目通知:\s*(\d+)\s*messages', text)
    fetched_m = re.search(r'Found:\s*(\d+)\s*races', text)
    total_races = int(fetched_m.group(1)) if fetched_m else None

    if m:
        bet_msgs = int(m.group(1))
        if bet_msgs == 0:
            return _result('critical', f'投票候補 0R (整形済み買い目通知: 0 messages、 fetched {total_races} races)')
        return _result('ok', f'投票候補 {bet_msgs} messages OK', bet_msgs=bet_msgs)

    # log まだ summary 出力前 — Skip 数で 推定
    skip_count = text.count('[STRATEGY7] Skip')
    if total_races and skip_count >= total_races:
        return _result('critical', f'全 R STRATEGY7 Skip ({skip_count}/{total_races} = 投票候補 0R)')
    return _result('warning', f'summary 未出力 (log 進行中)、 Skip={skip_count}/{total_races}')


# ---------- trigger 3: Streamlit 起動失敗 ----------
def check_streamlit(port: int = 8501, timeout: float = 5.0) -> dict:
    """port 8501 HTTP GET 200 ?"""
    try:
        import requests
    except Exception as e:
        return _result('warning', f'requests import 失敗: {e}')
    try:
        r = requests.get(f'http://localhost:{port}/', timeout=timeout)
        if r.status_code == 200:
            return _result('ok', f'streamlit :{port} OK')
        return _result('warning', f'streamlit :{port} status {r.status_code}')
    except Exception as e:
        # 投票影響 0、 warning のみ
        return _result('warning', f'streamlit :{port} unreachable: {type(e).__name__}')


# ---------- trigger 4: Discord 通知なし ----------
def check_discord_recent(date_str: str, repo: Path = REPO, within_hours: float = 1.0) -> dict:
    """race_auto_notify_{date}.log の最終更新 + 通知 messages 数 check"""
    log_path = repo / 'logs' / f'race_auto_notify_{date_str}.log'
    if not log_path.exists():
        return _result('warning', f'race_auto_notify_{date_str}.log 不在')

    mtime = datetime.fromtimestamp(log_path.stat().st_mtime)
    age_min = (datetime.now() - mtime).total_seconds() / 60

    try:
        text = log_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return _result('warning', f'log 読込失敗: {e}')

    # 「全レース一括通知: N messages」 + 「整形済み買い目通知: M messages」 合算
    total = 0
    for pat in (r'全レース一括通知:\s*(\d+)\s*messages',
                r'整形済み買い目通知:\s*(\d+)\s*messages'):
        m = re.search(pat, text)
        if m:
            total += int(m.group(1))

    # まだ start phase = mtime 新しいが messages 0 は OK
    if age_min > within_hours * 60 and total == 0:
        return _result('warning',
                       f'Discord 通知 0 messages、 log {age_min:.0f} min 前から無変化')
    return _result('ok', f'Discord notify {total} messages (log {age_min:.0f} min 前)',
                   total=total, age_min=age_min)


# ---------- trigger 5: 戦略⑦案 C 不動作 ----------
def check_strategy7c(date_str: str, repo: Path = REPO) -> dict:
    """
    daily_predictions に 京都 R あり、 かつ race_auto_notify log に
    「[STRATEGY7] Skip 京都」 が 1 つも無ければ critical (案 C 効いてない)
    G1/L/OP 特別の京都 R は除外しない仕様 (race_auto_notify.py L186) なので、
    京都 R 全てが G/L/OP 特別 の場合は ok
    """
    pred_path = repo / 'data' / 'daily_predictions' / f'{date_str}.csv'
    log_path = repo / 'logs' / f'race_auto_notify_{date_str}.log'
    if not pred_path.exists():
        return _result('warning', 'check skipped (predictions 不在)')
    try:
        import pandas as pd
        df = pd.read_csv(pred_path)
    except Exception as e:
        return _result('warning', f'predictions 読込失敗: {e}')

    # course 列 (大半の format で 「京都」 文字列)
    if 'course' not in df.columns:
        return _result('warning', 'predictions に course 列なし、 check 不可')
    kyoto_df = df[df['course'].astype(str) == '京都']
    if len(kyoto_df) == 0:
        return _result('ok', '京都 R 該当日なし (skip 不要)')

    # G/L/OP 特別 は filter 通過する仕様、 race_name から判定
    def is_excluded_from_filter(name: str) -> bool:
        s = str(name)
        if any(g in s for g in ['G1', 'G2', 'G3', 'GⅠ', 'GⅡ', 'GⅢ']):
            return True
        if any(t in s for t in ['L)', '(L)', 'OP)', '(OP)']):
            return True
        return False

    if 'race_name' in kyoto_df.columns:
        non_excluded = kyoto_df[~kyoto_df['race_name'].apply(is_excluded_from_filter)]
    else:
        non_excluded = kyoto_df

    if len(non_excluded) == 0:
        return _result('ok', f'京都 R {len(kyoto_df)} 全て G/L/OP 例外 (filter 対象外)')

    if not log_path.exists():
        return _result('warning',
                       f'京都 R {len(non_excluded)} あり、 log まだ (起動前なら OK)')
    try:
        text = log_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return _result('warning', f'log 読込失敗: {e}')
    skip_kyoto = text.count('[STRATEGY7] Skip 京都')
    if skip_kyoto == 0:
        return _result('critical',
                       f'★ 案 C 不動作疑い: 京都 R {len(non_excluded)} あり、 log に [STRATEGY7] Skip 京都 0 件')
    return _result('ok',
                   f'京都 R {len(non_excluded)} に対し Skip 京都 log {skip_kyoto} 件 確認',
                   skip_count=skip_kyoto, kyoto_races=len(non_excluded))


# ---------- discord ----------
def send_discord(severity: str, message: str, rollback_cmd: str | None = None) -> bool:
    webhook = os.environ.get('DISCORD_WEBHOOK_UPDATES') or os.environ.get('DISCORD_WEBHOOK_URL')
    if not webhook:
        return False
    try:
        import requests
    except Exception:
        return False
    icon = SEVERITY_ICON.get(severity, '?')
    body = f"{icon} **anomaly auto detection**\n{message}"
    if rollback_cmd:
        body += f"\n\nrollback 候補:\n```\n{rollback_cmd}\n```"
    try:
        r = requests.post(webhook, json={'content': body}, timeout=10)
        return r.status_code in (200, 204)
    except Exception:
        return False


# ---------- main ----------
def run_all(date_str: str) -> list[tuple[str, dict]]:
    return [
        ('predictions',      check_predictions(date_str)),
        ('vote_candidates',  check_vote_candidates(date_str)),
        ('streamlit',        check_streamlit()),
        ('discord_recent',   check_discord_recent(date_str)),
        ('strategy7c',       check_strategy7c(date_str)),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=datetime.now().strftime('%Y%m%d'))
    ap.add_argument('--no-discord', action='store_true')
    ap.add_argument('--severity', choices=['critical', 'warning', 'all'], default='all')
    args = ap.parse_args(argv)

    triggers = run_all(args.date)
    critical = sum(1 for _, r in triggers if r['severity'] == 'critical')
    warning = sum(1 for _, r in triggers if r['severity'] == 'warning')

    print(f"=== anomaly auto detection {args.date} ===")
    for name, r in triggers:
        print(f"  [{SEVERITY_ICON[r['severity']]}] {name:18} {r['msg']}")
    print(f"\nsummary: critical={critical}, warning={warning}, ok={5 - critical - warning}")

    # Discord
    if not args.no_discord and (critical > 0 or warning > 0):
        lines = [f"date={args.date}"]
        for name, r in triggers:
            if r['severity'] == 'ok':
                continue
            if args.severity == 'critical' and r['severity'] != 'critical':
                continue
            lines.append(f"{SEVERITY_ICON[r['severity']]} {name}: {r['msg']}")
        rollback = ("git revert <Sub-task 8 commit hash> --no-edit  "
                    "# 戦略⑦案 C rollback (docs/5_17_G1_DAY_CHECKLIST.md 参照)")
        send_discord('critical' if critical else 'warning', '\n'.join(lines), rollback)

    if critical > 0:
        return 2
    if warning > 0:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
