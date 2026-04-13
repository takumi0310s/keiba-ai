#!/usr/bin/env python
"""スクレイピングプロセス死活監視・自動再起動

概要:
  - logs/pids/*.json にスクレイピングプロセスのPID/コマンドを記録
  - 5分ごとにプロセス生存確認 (psutil or 純Python)
  - 死んでいて scraper_guard が許可する時間帯なら自動再起動
  - 再起動時 Discord 通知

ファイルフォーマット (logs/pids/<name>.json):
    {"name": "bulk_scrape_upset", "pid": 12345, "cmd": ["python","-u",...],
     "cwd": "...", "log": "logs/xxx.log", "started_at": "ISO"}

Usage:
    # 単発チェック (1回実行)
    python tools/process_watchdog.py --once

    # 継続監視 (5分間隔、永続ループ)
    python tools/process_watchdog.py

    # PID 登録
    python tools/process_watchdog.py --register --name bulk_scrape_upset \
        --pid 12345 --cmd "python -u tools/bulk_scrape_upset.py"

    # 監視解除 (完了扱い)
    python tools/process_watchdog.py --unregister --name bulk_scrape_upset

    # 状態確認
    python tools/process_watchdog.py --list
"""
from __future__ import annotations

import os
import sys
import json
import time
import shlex
import signal
import argparse
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

LOG_DIR = os.path.join(BASE_DIR, 'logs')
PID_DIR = os.path.join(LOG_DIR, 'pids')
os.makedirs(PID_DIR, exist_ok=True)

CHECK_INTERVAL = 300  # 5分


def _is_pid_alive(pid: int) -> bool:
    """指定PIDが生存中か確認 (Windows/Linux両対応)"""
    if pid <= 0:
        return False
    try:
        import psutil  # type: ignore
        return psutil.pid_exists(pid)
    except ImportError:
        pass
    # Fallback: Windows tasklist
    if sys.platform == 'win32':
        try:
            out = subprocess.run(
                ['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
                capture_output=True, text=True, timeout=10,
            )
            return str(pid) in out.stdout
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def _list_entries() -> list[dict]:
    entries = []
    for fn in sorted(os.listdir(PID_DIR)):
        if not fn.endswith('.json'):
            continue
        p = os.path.join(PID_DIR, fn)
        try:
            with open(p, 'r', encoding='utf-8') as f:
                entries.append(json.load(f))
        except Exception as e:
            print(f"[WARN] load {p}: {e}")
    return entries


def _save_entry(entry: dict):
    p = os.path.join(PID_DIR, f"{entry['name']}.json")
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)


def _delete_entry(name: str):
    p = os.path.join(PID_DIR, f"{name}.json")
    if os.path.exists(p):
        os.remove(p)


def _scraper_guard_ok() -> bool:
    try:
        from tools.scraper_guard import is_scraping_allowed
        return is_scraping_allowed()
    except Exception:
        return True


def _restart(entry: dict) -> int | None:
    """entry['cmd'] を再実行して新しいPIDを返す"""
    cmd = entry['cmd']
    cwd = entry.get('cwd', BASE_DIR)
    log = entry.get('log', os.path.join(LOG_DIR, f"{entry['name']}_restart.log"))
    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    print(f"[watchdog] restart {entry['name']}: {' '.join(cmd)}")
    log_f = open(log, 'a', encoding='utf-8', errors='replace')
    log_f.write(f"\n=== watchdog restart {datetime.now().isoformat()} ===\n")
    log_f.flush()
    try:
        if sys.platform == 'win32':
            creation = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=log_f, stderr=subprocess.STDOUT,
                                    env=env, creationflags=creation)
        else:
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=log_f, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
        return proc.pid
    except Exception as e:
        print(f"[watchdog] restart failed {entry['name']}: {e}")
        return None


def _notify(title: str, msg: str, color: str = 'yellow'):
    try:
        from notify import send_discord
        send_discord(title, msg, color=color, channel='updates')
    except Exception as e:
        print(f"[watchdog] notify failed: {e}")


def cmd_register(args):
    cmd_list = shlex.split(args.cmd) if isinstance(args.cmd, str) else list(args.cmd)
    entry = {
        'name': args.name,
        'pid': args.pid,
        'cmd': cmd_list,
        'cwd': args.cwd or BASE_DIR,
        'log': args.log or os.path.join(LOG_DIR, f"{args.name}.log"),
        'started_at': datetime.now().isoformat(),
        'restart_count': 0,
    }
    _save_entry(entry)
    print(f"[watchdog] registered {args.name} pid={args.pid}")


def cmd_unregister(args):
    _delete_entry(args.name)
    print(f"[watchdog] unregistered {args.name}")


def cmd_list(_args):
    entries = _list_entries()
    print(f"{'NAME':<30} {'PID':>7} {'ALIVE':<6} RESTARTS  STARTED")
    for e in entries:
        alive = _is_pid_alive(int(e.get('pid', 0)))
        print(f"{e['name']:<30} {e.get('pid',0):>7} "
              f"{'YES' if alive else 'NO':<6} {e.get('restart_count',0):>8}  {e.get('started_at','?')[:19]}")


def cmd_once(_args):
    entries = _list_entries()
    if not entries:
        print("[watchdog] no entries")
        return
    guard_ok = _scraper_guard_ok()
    restarted = []
    dead = []
    for entry in entries:
        pid = int(entry.get('pid', 0))
        alive = _is_pid_alive(pid)
        if alive:
            print(f"[watchdog] {entry['name']} pid={pid} ALIVE")
            continue
        dead.append(entry['name'])
        if not guard_ok:
            print(f"[watchdog] {entry['name']} DEAD — scraper_guard blocks restart")
            continue
        new_pid = _restart(entry)
        if new_pid:
            entry['pid'] = new_pid
            entry['restart_count'] = int(entry.get('restart_count', 0)) + 1
            entry['restarted_at'] = datetime.now().isoformat()
            _save_entry(entry)
            restarted.append(entry['name'])

    if restarted:
        _notify(
            f"🔄 Watchdog restart ({len(restarted)})",
            "\n".join(f"- {n}" for n in restarted),
            color='yellow',
        )
    elif dead and not guard_ok:
        _notify(
            f"⏸ Watchdog blocked ({len(dead)} dead)",
            "scraper_guard がブロック時間帯のため再起動見送り:\n" + "\n".join(f"- {n}" for n in dead),
            color='yellow',
        )


def cmd_loop(_args):
    print(f"[watchdog] loop start interval={CHECK_INTERVAL}s")
    while True:
        try:
            cmd_once(_args)
        except Exception as e:
            print(f"[watchdog] loop error: {e}")
        time.sleep(CHECK_INTERVAL)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='mode')

    ap.add_argument('--once', action='store_true', help='1回チェックして終了')
    ap.add_argument('--list', action='store_true', help='登録済みエントリを一覧表示')
    ap.add_argument('--register', action='store_true', help='エントリを登録')
    ap.add_argument('--unregister', action='store_true', help='エントリ削除')
    ap.add_argument('--name', type=str, help='エントリ名')
    ap.add_argument('--pid', type=int, help='監視対象PID')
    ap.add_argument('--cmd', type=str, help='再起動用コマンド (空白区切り)')
    ap.add_argument('--cwd', type=str, help='作業ディレクトリ')
    ap.add_argument('--log', type=str, help='ログ出力先')
    args = ap.parse_args()

    if args.register:
        if not args.name or not args.pid or not args.cmd:
            print("--register には --name --pid --cmd 必須")
            sys.exit(1)
        cmd_register(args)
    elif args.unregister:
        if not args.name:
            print("--unregister には --name 必須")
            sys.exit(1)
        cmd_unregister(args)
    elif args.list:
        cmd_list(args)
    elif args.once:
        cmd_once(args)
    else:
        cmd_loop(args)


if __name__ == '__main__':
    main()
