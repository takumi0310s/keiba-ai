#!/usr/bin/env python
"""process_watchdog v2 — ログ鮮度ベースの死活監視

既存 process_watchdog.py は PID 登録ベースで、運用タスク (daily_predict,
race_auto_notify) が未登録のため実質無効だった。
v2 はハードコード対象 + ログファイル mtime で「生きてる風ゾンビ」も検知する。

監視対象:
  - daily_predict      : logs/daily_predict*.log が 30分以上更新なしで DEAD
  - race_auto_notify   : logs/race_auto_notify*.log が 10分以上更新なしで DEAD

再起動ポリシー:
  - 07:00 - 18:00 の本番時間帯のみ再起動試行
  - それ以外は Discord 警告通知のみ
  - 再起動時 env に SCRAPER_GUARD_DISABLE=1 + KEIBA_OPERATIONAL_MODE=1 を付与
  - Discord は [updates] チャンネルに "🚨 CRITICAL" プレフィックス + 色 red

Usage:
  python tools/process_watchdog_v2.py --once     # 1回チェックして終了
  python tools/process_watchdog_v2.py            # 継続監視 (5分間隔)
  python tools/process_watchdog_v2.py --dry-run  # 検知のみ (再起動しない)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime
from typing import Callable, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

LOG_DIR = os.path.join(BASE_DIR, 'logs')
CHECK_INTERVAL = 300  # 5分

# --- 再起動 cap (5/9 の再起動ループ=Discord spam 再発防止の前提) ---
# watchdog 自身がプロセス再起動で in-memory カウンタを失っても効くよう、
# 台帳をファイル永続化する。安全な再登録の前提(再登録自体は別途承認・ここでは未登録)。
MAX_RESTARTS_PER_DAY = 3       # 同一対象を1日に再起動する上限
MIN_RESTART_INTERVAL_SEC = 600  # 連続再起動の最小間隔(10分)。短時間の反復restartを阻止
RESTART_LEDGER = os.path.join(BASE_DIR, 'data', 'v18', 'watchdog_restart_ledger.json')


def _load_ledger(path: str) -> dict:
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_ledger(path: str, ledger: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(ledger, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # 台帳保存失敗で本処理は止めない


def restart_guard(target_name: str, now: datetime, ledger_path: str = RESTART_LEDGER,
                  max_per_day: int = MAX_RESTARTS_PER_DAY,
                  min_interval_sec: int = MIN_RESTART_INTERVAL_SEC) -> dict:
    """再起動してよいかを台帳で判定。{'allowed':bool,'reason':str|None,'count':int}。
    - daily_cap_reached: 当日 max_per_day 回に到達
    - interval_guard:    前回再起動から min_interval_sec 未満
    """
    day = now.strftime('%Y%m%d')
    led = _load_ledger(ledger_path)
    rec = (led.get(day, {}) or {}).get(target_name, {}) or {}
    count = int(rec.get('count', 0))
    last_ts = rec.get('last_ts')
    if count >= max_per_day:
        return {'allowed': False, 'reason': 'daily_cap_reached', 'count': count}
    if last_ts:
        try:
            elapsed = (now - datetime.fromisoformat(last_ts)).total_seconds()
            if 0 <= elapsed < min_interval_sec:
                return {'allowed': False, 'reason': 'interval_guard', 'count': count}
        except Exception:
            pass
    return {'allowed': True, 'reason': None, 'count': count}


def record_restart(target_name: str, now: datetime, ledger_path: str = RESTART_LEDGER) -> int:
    """再起動成功を台帳に記録し、当日のカウントを返す。"""
    day = now.strftime('%Y%m%d')
    led = _load_ledger(ledger_path)
    led.setdefault(day, {})
    rec = led[day].get(target_name, {}) or {}
    rec['count'] = int(rec.get('count', 0)) + 1
    rec['last_ts'] = now.isoformat()
    led[day][target_name] = rec
    # 古い日付は掃除(直近7日だけ保持)
    if len(led) > 7:
        for k in sorted(led)[:-7]:
            led.pop(k, None)
    _save_ledger(ledger_path, led)
    return rec['count']

# 本番時間帯: [07:00, 18:00)
ACTIVE_START = dtime(7, 0)
ACTIVE_END = dtime(18, 0)


@dataclass(frozen=True)
class WatchTarget:
    name: str
    log_glob: str                # logs/ 配下のログ glob（最新1件の mtime を見る）
    stale_sec: int               # 無更新で DEAD 判定する秒数
    process_match: str           # コマンドライン一致確認用の部分文字列 (tasklist|ps)
    restart_cmd: list[str]       # 再起動コマンド (cwd=BASE_DIR)
    restart_log_pattern: str     # 再起動時に書き出すログ (YYYYMMDD 置換可)


# --- 監視対象定義 (Session #31 A3+D で閾値最適化) ---
# stale_sec: 各 task の想定実行時間 + 余裕 (誤発火防止):
#   daily_predict: ~30 min 実行 + 余裕 → 60 min stale で再起動
#   race_auto_notify: ~10 min 実行 + 余裕 → 30 min stale で再起動
TARGETS: list[WatchTarget] = [
    WatchTarget(
        name='daily_predict',
        log_glob='daily_predict*.log',
        stale_sec=60 * 60,  # 30 → 60 min (Session #31 誤発火防止)
        process_match='tools/daily_predict.py',
        restart_cmd=['python', '-u', 'tools/daily_predict.py', '--resume'],
        restart_log_pattern='daily_predict_watchdog_restart_{date}.log',
    ),
    WatchTarget(
        name='race_auto_notify',
        log_glob='race_auto_notify*.log',
        stale_sec=30 * 60,  # 10 → 30 min (Session #31 誤発火防止)
        process_match='tools/race_auto_notify.py',
        restart_cmd=['python', '-u', 'tools/race_auto_notify.py'],
        restart_log_pattern='race_auto_notify_watchdog_restart_{date}.log',
    ),
]


def _log_misfire(target_name: str, status: str, action: str, mtime: Optional[float]) -> None:
    """誤発火ログ記録 (Session #31 A3 強化)。
    再起動した記録を data/v18/process_watchdog_v2_misfires.log に追記。
    後で偽陽性判定の検証用。
    """
    try:
        log_path = os.path.join(BASE_DIR, 'data/v18/process_watchdog_v2_misfires.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        ts = datetime.now().isoformat()
        mt_str = datetime.fromtimestamp(mtime).isoformat() if mtime else "None"
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{ts}] target={target_name} status={status} action={action} last_mtime={mt_str}\n")
    except Exception:
        pass


# ===== 時間帯 =====

def is_active_hours(now: Optional[datetime] = None) -> bool:
    """07:00-18:00 の本番時間帯なら True"""
    now = now or datetime.now()
    t = now.time()
    return ACTIVE_START <= t < ACTIVE_END


# ===== ログ鮮度 =====

def latest_log_mtime(log_glob: str, base_dir: str = LOG_DIR) -> Optional[float]:
    """logs/log_glob に該当するファイルのうち最新の mtime (epoch) を返す。見つからなければ None"""
    paths = glob.glob(os.path.join(base_dir, log_glob))
    if not paths:
        return None
    return max(os.path.getmtime(p) for p in paths)


def is_log_stale(log_glob: str, stale_sec: int,
                 now: Optional[datetime] = None,
                 base_dir: str = LOG_DIR) -> tuple[bool, Optional[float]]:
    """ログが stale_sec 以上更新されていないか判定。戻り値 = (stale?, mtime or None)"""
    now_ts = (now or datetime.now()).timestamp()
    mt = latest_log_mtime(log_glob, base_dir=base_dir)
    if mt is None:
        # ログ自体がない = stale 扱い（ただし ALIVE 判定は process_alive に委ねる）
        return True, None
    return (now_ts - mt) >= stale_sec, mt


# ===== プロセス存在 =====

def process_alive(match: str) -> bool:
    """コマンドライン部分一致でプロセスを検索。Windows 前提で wmic / tasklist フォールバック"""
    if sys.platform != 'win32':
        # Linux/Mac: pgrep 相当
        try:
            out = subprocess.run(['pgrep', '-af', match], capture_output=True, text=True, timeout=5)
            return bool(out.stdout.strip())
        except Exception:
            return False
    # Windows: PowerShell CIM (wmic は 24H2 で廃止)
    try:
        ps = (
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Where-Object { $_.CommandLine -like '*" + match.replace("'", "''") + "*' } | "
            "Select-Object -First 1 | ForEach-Object { $_.ProcessId }"
        )
        out = subprocess.run(['powershell', '-NoProfile', '-Command', ps],
                             capture_output=True, text=True, timeout=15)
        return bool(out.stdout.strip())
    except Exception:
        return False


# ===== 再起動 =====

def restart_target(target: WatchTarget,
                   dry_run: bool = False,
                   now: Optional[datetime] = None,
                   popen: Callable = subprocess.Popen,
                   active_checker: Callable[[Optional[datetime]], bool] = is_active_hours,
                   ledger_path: str = RESTART_LEDGER,
                   max_per_day: int = MAX_RESTARTS_PER_DAY,
                   min_interval_sec: int = MIN_RESTART_INTERVAL_SEC) -> dict:
    """対象を再起動し、結果 dict を返す。
    {'restarted': bool, 'skipped_reason': str|None, 'pid': int|None}
    ★再起動 cap: 当日 max_per_day 回上限 + 連続 min_interval_sec 間隔ガード(5/9 spam再発防止)★
    """
    result: dict = {'restarted': False, 'skipped_reason': None, 'pid': None}
    now = now or datetime.now()
    if not active_checker(now):
        result['skipped_reason'] = 'outside_active_hours'
        return result
    if dry_run:
        result['skipped_reason'] = 'dry_run'
        return result
    guard = restart_guard(target.name, now, ledger_path, max_per_day, min_interval_sec)
    if not guard['allowed']:
        result['skipped_reason'] = guard['reason']  # daily_cap_reached / interval_guard
        return result
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['SCRAPER_GUARD_DISABLE'] = '1'
    env['KEIBA_OPERATIONAL_MODE'] = '1'
    env.setdefault('FOR_DISABLE_CONSOLE_CTRL_HANDLER', '1')
    env.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

    date_str = now.strftime('%Y%m%d')
    log_path = os.path.join(LOG_DIR, target.restart_log_pattern.format(date=date_str))
    os.makedirs(LOG_DIR, exist_ok=True)
    try:
        log_f = open(log_path, 'a', encoding='utf-8', errors='replace')
        log_f.write(f"\n=== watchdog_v2 restart {now.isoformat()} ===\n")
        log_f.flush()
        creation = 0
        if sys.platform == 'win32':
            creation = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0) \
                     | getattr(subprocess, 'DETACHED_PROCESS', 0)
        proc = popen(target.restart_cmd, cwd=BASE_DIR, stdout=log_f, stderr=subprocess.STDOUT,
                     env=env, creationflags=creation) if sys.platform == 'win32' else \
               popen(target.restart_cmd, cwd=BASE_DIR, stdout=log_f, stderr=subprocess.STDOUT,
                     env=env, start_new_session=True)
        result['restarted'] = True
        result['pid'] = getattr(proc, 'pid', None)
        result['day_count'] = record_restart(target.name, now, ledger_path)
    except Exception as e:
        result['skipped_reason'] = f'exception:{e}'
    return result


# ===== 通知 =====

def notify_critical(title: str, msg: str, color: str = 'red'):
    """Discord 強通知 (CRITICAL prefix)。notify モジュールが無くても黙って継続"""
    full_title = f"🚨 CRITICAL {title}"
    # 優先: tools/notify_done.py のシェル呼び出し
    try:
        subprocess.run(
            ['python', os.path.join(BASE_DIR, 'tools', 'notify_done.py'),
             full_title, msg, f"--color={color}"],
            cwd=BASE_DIR, check=False, timeout=20,
        )
        return
    except Exception:
        pass
    try:
        from notify import send_discord
        send_discord(full_title, msg, color=color, channel='updates')
    except Exception as e:
        print(f"[watchdog_v2] notify failed: {e}")


# ===== メイン判定 =====

def check_target(target: WatchTarget, now: Optional[datetime] = None,
                 log_base: str = LOG_DIR,
                 process_checker: Callable[[str], bool] = process_alive) -> dict:
    """単一対象の健康状態を判定し結果 dict を返す
    {'name':..., 'alive':bool, 'stale':bool, 'mtime':..., 'status':'ALIVE'|'STALE'|'MISSING'}
    """
    alive = process_checker(target.process_match)
    stale, mt = is_log_stale(target.log_glob, target.stale_sec, now=now, base_dir=log_base)
    if alive and not stale:
        status = 'ALIVE'
    elif alive and stale:
        status = 'STALE'      # プロセスは生きてるがログ止まってる (Fortran ゾンビ等)
    elif (not alive) and (not stale):
        # Session #64 fix: プロセスがいなくてもログ鮮度内 → 直近に正常終了したワンショット
        # (例: daily_predict.py は実行 → ログ書き → 終了 のサイクル)。
        # これを MISSING 扱いすると watchdog が 5 分毎に誤再起動 → Discord spam。
        status = 'COMPLETED'
    else:
        status = 'MISSING'    # プロセス自体がない、 かつログも stale_sec 以上 古い
    return {
        'name': target.name,
        'alive': alive,
        'stale': stale,
        'mtime': mt,
        'status': status,
    }


def run_once(dry_run: bool = False, now: Optional[datetime] = None) -> dict:
    """1 チェックパス。結果サマリを返す。"""
    now = now or datetime.now()
    results = []
    restarted = []
    warned = []
    for target in TARGETS:
        r = check_target(target, now=now)
        results.append(r)
        print(f"[watchdog_v2] {target.name}: {r['status']} "
              f"(alive={r['alive']}, stale={r['stale']}, mtime={r['mtime']})")
        if r['status'] in ('ALIVE', 'COMPLETED'):
            # Session #64 fix: COMPLETED (ワンショット正常終了 + ログ新鮮) は再起動しない
            continue
        # STALE or MISSING → 再起動 or 警告
        rr = restart_target(target, dry_run=dry_run, now=now)
        # Session #31 A3: 全アクション (再起動 / 時間外スキップ) を misfire log へ
        _log_misfire(target.name, r['status'],
                     'restarted' if rr['restarted'] else (rr.get('skipped_reason') or 'no_action'),
                     r.get('mtime'))
        if rr['restarted']:
            restarted.append(target.name)
            notify_critical(
                f"Watchdog restart: {target.name}",
                f"status={r['status']} last_mtime={r['mtime']}\n"
                f"再起動PID={rr['pid']} env=SCRAPER_GUARD_DISABLE+KEIBA_OPERATIONAL_MODE",
                color='red',
            )
        else:
            warned.append((target.name, rr['skipped_reason']))
            if rr['skipped_reason'] == 'outside_active_hours':
                notify_critical(
                    f"Watchdog detected dead ({target.name}, 時間帯外)",
                    f"status={r['status']} last_mtime={r['mtime']}\n"
                    f"07:00-18:00 時間帯外のため再起動は見送り。手動復旧が必要。",
                    color='yellow',
                )
    return {'results': results, 'restarted': restarted, 'warned': warned, 'now': now.isoformat()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true', help='1回チェックして終了')
    ap.add_argument('--dry-run', action='store_true', help='検知のみ、再起動しない')
    args = ap.parse_args()

    # Session #64 fix: kill-switch — Admin なしで緊急停止可能。
    # `data/v18/process_watchdog_v2.kill` を touch すれば即時 no-op で抜ける。
    kill_switch = os.path.join(BASE_DIR, 'data', 'v18', 'process_watchdog_v2.kill')
    if os.path.exists(kill_switch):
        print(f"[watchdog_v2] kill-switch active ({kill_switch}) → no-op exit")
        return

    if args.once:
        run_once(dry_run=args.dry_run)
        return

    print(f"[watchdog_v2] loop start interval={CHECK_INTERVAL}s dry_run={args.dry_run}")
    while True:
        try:
            run_once(dry_run=args.dry_run)
        except Exception as e:
            print(f"[watchdog_v2] loop error: {e}")
        time.sleep(CHECK_INTERVAL)


if __name__ == '__main__':
    main()
