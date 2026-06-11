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
def _is_no_race_day(date_str: str, repo: Path = REPO) -> bool:
    """daily_predict.py が記録した「レースが見つかりません(非開催日)」の有無で非開催判定。
    6/11 Fable sweep: 平日(非開催)に predictions 不在を critical 誤警報していた対策
    (5/11-6/10 で critical=1 を 68 日分 Discord 送信、rollback 提案文まで付いていた)。"""
    # watchdog 化 (6/9) 以降は dated subproc ログが1次。旧 daily_predict.log は 5/4 で停止済
    candidates = [repo / 'logs' / f'daily_predict_watchdog_{date_str}_subproc.log',
                  repo / 'logs' / 'daily_predict.log']
    for log in candidates:
        if not log.exists():
            continue
        try:
            raw = log.read_bytes()
        except Exception:
            continue
        # ログは utf-8/cp932 混在 (実測) のためバイトで両対応検索
        for enc in ('utf-8', 'cp932'):
            if log.name == 'daily_predict.log':
                sec = raw.rfind(f'日次予測 - {date_str}'.encode(enc))
                if sec < 0:
                    continue
                chunk = raw[sec:sec + 8000]
            else:
                chunk = raw
            if 'レースが見つかりません'.encode(enc) in chunk:
                return True
    return False


def check_predictions(date_str: str, repo: Path = REPO) -> dict:
    """daily_predictions/{date}.csv の存在 + 行数 check"""
    path = repo / 'data' / 'daily_predictions' / f'{date_str}.csv'
    if not path.exists():
        if _is_no_race_day(date_str, repo):
            return _result('ok', f'非開催日 (daily_predict が「レースなし」を記録済) → predictions 不在は正常')
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


# === Strategy-level anomaly detection (D-6 2026-05-19) ===

# 6/11 Fable sweep: 旧実装は ss.get('roi') (実キーは roi_pct) + 閾値 -50 (roi_pct は 0-100+ スケール)
# の二重バグで全チェック永久不発だった。設計意図 (撤退3段階の「単日 ROI<50%」) に復元。
STRATEGY_ANOMALY_THRESHOLDS = {
    'single_day_roi_pct_floor': 50.0,  # 単日 ROI < 50% (n>0 の日のみ)
    'hit_rate_zero_min_n': 5,          # N >= 5 で hit = 0
    'consecutive_negative_days': 3,    # 連続 negative ROI (< 100%、n>0 の日のみ)
}


def check_strategy_anomaly(strategy_key: str, strategy_stats_history: list) -> dict | None:
    """
    Check if a specific strategy shows anomalous behavior.

    Args:
        strategy_key: strategy name (str)
        strategy_stats_history: list of daily stats dicts
            [{roi, n, hits, date}, ...]  (oldest → newest)

    Returns:
        dict {strategy, anomalies, action, details} if anomaly found, else None
    """
    if not strategy_stats_history:
        return None

    anomalies = []
    latest = strategy_stats_history[-1]

    # Check 1: 単日 ROI < 50% (賭けがあった日のみ。n=0 日の roi_pct=0 は対象外)
    roi = latest.get('roi')
    n = latest.get('n', 0)
    if roi is not None and n > 0 and roi < STRATEGY_ANOMALY_THRESHOLDS['single_day_roi_pct_floor']:
        anomalies.append(f"単日 ROI {roi:.1f}% < 50%")

    # Check 2: hit = 0 かつ N >= threshold
    hits = latest.get('hits', 1)
    if n >= STRATEGY_ANOMALY_THRESHOLDS['hit_rate_zero_min_n'] and hits == 0:
        anomalies.append(f"hit = 0/{n} races")

    # Check 3: 連続 consecutive_negative_days 日 ROI < 100% (n>0 の日のみ連続カウント)
    req = STRATEGY_ANOMALY_THRESHOLDS['consecutive_negative_days']
    bet_days = [d for d in strategy_stats_history if d.get('n', 0) > 0]
    recent = bet_days[-req:]
    if (len(recent) >= req and
            all(d.get('roi') is not None and d['roi'] < 100.0 for d in recent)):
        anomalies.append(f"直近賭け日 {len(recent)} 日連続 ROI < 100%")

    if anomalies:
        return {
            'strategy': strategy_key,
            'anomalies': anomalies,
            'action': 'paper_only_auto_switch',
            'details': '; '.join(anomalies),
        }
    return None


def run_strategy_anomaly_scan(log_dir: str | None = None) -> list:
    """
    Scan all strategy stats for anomalies from race_notify_log_v2 aggregator summaries.

    Returns list of anomaly dicts for strategies with issues.
    """
    import glob as _glob

    if log_dir is None:
        log_dir = str(REPO / 'data' / 'race_notify_log_v2_summary')

    strategies = ['actual', 'c3', 'c4', 'c3c4', 'no_1pop', 'divergence', 'ev_filter', 'odds_filter']
    strategy_history: dict = {s: [] for s in strategies}

    if os.path.isdir(log_dir):
        for fpath in sorted(_glob.glob(os.path.join(log_dir, '*.json')))[-7:]:
            try:
                with open(fpath, encoding='utf-8') as fp:
                    data = json.load(fp)
                date_str = os.path.splitext(os.path.basename(fpath))[0]
                for s in strategies:
                    ss = data.get('strategy_stats', {}).get(s, {})
                    if ss:
                        strategy_history[s].append({
                            'date': date_str,
                            # 実キーは roi_pct (旧 'roi' は常に None → 検知不発の原因)
                            'roi': ss.get('roi_pct', ss.get('roi')),
                            'n': ss.get('n', 0),
                            'hits': ss.get('hits', 0),
                        })
            except Exception:
                pass

    anomalies_found = []
    for s in strategies:
        result = check_strategy_anomaly(s, strategy_history[s])
        if result:
            anomalies_found.append(result)

    return anomalies_found


def send_strategy_anomaly_alert(anomalies: list, webhook_url: str | None = None) -> None:
    """Send Discord alert for strategy anomalies. Falls back to stdout print."""
    if not anomalies:
        return

    if webhook_url is None:
        webhook_url = (os.environ.get('DISCORD_WEBHOOK_UPDATES') or
                       os.environ.get('DISCORD_WEBHOOK_URL'))

    if not webhook_url:
        for a in anomalies:
            print(f"[ANOMALY] {a['strategy']}: {a['details']} → {a['action']}")
        return

    try:
        import json as _json
        import urllib.request
        lines = [f"⚠️ Strategy Anomaly Detected ({len(anomalies)} strategies)"]
        for a in anomalies:
            lines.append(f"• {a['strategy']}: {a['details']}")
            lines.append(f"  → {a['action']}")
        payload = {'content': '\n'.join(lines)}
        data = _json.dumps(payload).encode()
        req = urllib.request.Request(
            webhook_url, data=data,
            headers={'Content-Type': 'application/json'},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[anomaly strategy alert] Discord failed: {e}")


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
        # 6/11 Fable sweep: 未解決プレースホルダ「<Sub-task 8 commit hash>」を Discord 送信していた → 手順参照に変更
        rollback = "rollback 手順: python tools/strategy_rollback.py --check (docs/5_17_G1_DAY_CHECKLIST.md 参照)"
        send_discord('critical' if critical else 'warning', '\n'.join(lines), rollback)

    # strategy-level anomaly scan (D-6)
    strategy_anomalies = run_strategy_anomaly_scan()
    if strategy_anomalies:
        send_strategy_anomaly_alert(strategy_anomalies)
        for a in strategy_anomalies:
            print(f"  [⚠] strategy_anomaly  {a['strategy']}: {a['details']}")
        if not args.no_discord:
            pass  # already sent via send_strategy_anomaly_alert

    if critical > 0:
        return 2
    if warning > 0:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
