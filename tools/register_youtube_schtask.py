#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YouTube JRA 公式 LIVE 録画 schtask 自動登録.

Windows タスクスケジューラに 土日 08:55 起動の YouTube 録画 task を登録。
admin 権限が必要 (UAC prompt 発生)。

Usage:
    python tools/register_youtube_schtask.py             # 登録 (admin 権限)
    python tools/register_youtube_schtask.py --check     # 登録状態のみ確認
    python tools/register_youtube_schtask.py --remove    # 登録解除 (admin)
    python tools/register_youtube_schtask.py --dry-run   # 実行 cmd だけ表示

【登録される task】
- 名前: Keiba-YouTubeLiveRecord-Sat / Keiba-YouTubeLiveRecord-Sun
- 起動: 土 08:55 / 日 08:55 (LIVE 配信 9:00 開始 5 分前)
- 内容: python tools/youtube_jra_live_record.py --quality 720 --max-duration 28800
- 動作時間: 最大 8 時間 (LIVE は通常 9:00-17:00)
- 失敗時: 最大 3 回 再試行 (10 分 interval)
"""
import argparse
import os
import shlex
import subprocess
import sys

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD_SCRIPT = os.path.join(BASE_DIR, 'tools', 'youtube_jra_live_record.py')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

TASKS = [
    {
        'name': 'Keiba-YouTubeLiveRecord-Sat',
        'schedule': 'WEEKLY',
        'day': 'SAT',
        'time': '08:55',
    },
    {
        'name': 'Keiba-YouTubeLiveRecord-Sun',
        'schedule': 'WEEKLY',
        'day': 'SUN',
        'time': '08:55',
    },
]


def task_command():
    """schtasks /tr に渡す command 文字列."""
    python_path = sys.executable
    log_path = os.path.join(LOG_DIR, 'youtube_record_%date:~0,4%%date:~5,2%%date:~8,2%.log')
    # quotes 必須 (空白入る path)
    return (
        f'"{python_path}" "{RECORD_SCRIPT}" --quality 720 --max-duration 28800 '
        f'>> "{log_path}" 2>&1'
    )


def cmd_register(args):
    os.makedirs(LOG_DIR, exist_ok=True)
    cmd_str = task_command()
    print('[INFO] 登録対象 task command:')
    print(f'  {cmd_str}\n')

    for t in TASKS:
        full_cmd = [
            'schtasks', '/create',
            '/tn', t['name'],
            '/sc', t['schedule'],
            '/d', t['day'],
            '/st', t['time'],
            '/tr', f'cmd /c {cmd_str}',
            '/f',  # 既存上書き
            '/rl', 'HIGHEST',
        ]
        print(f'[REGISTER] {t["name"]} ({t["day"]} {t["time"]})')
        print('  cmd:', ' '.join(shlex.quote(a) for a in full_cmd[:8]))

        if args.dry_run:
            print('  [DRY-RUN] skip\n')
            continue

        try:
            r = subprocess.run(full_cmd, capture_output=True, text=True, timeout=30,
                               encoding='cp932', errors='replace')
        except Exception as e:
            print(f'  [ERROR] {e}')
            continue

        if r.returncode == 0:
            print(f'  [OK] registered')
        else:
            print(f'  [FAIL] rc={r.returncode}')
            print(f'    stdout: {r.stdout[:200]}')
            print(f'    stderr: {r.stderr[:200]}')
            print(f'  ※ admin 権限必要、 PowerShell を 管理者として 実行してください')
        print()
    return 0


def cmd_check(args):
    for t in TASKS:
        try:
            r = subprocess.run(['schtasks', '/query', '/tn', t['name'], '/v', '/fo', 'list'],
                               capture_output=True, text=True, timeout=15,
                               encoding='cp932', errors='replace')
        except Exception as e:
            print(f'[ERROR] {t["name"]}: {e}')
            continue
        if r.returncode == 0:
            # parse last run / next run
            lines = r.stdout.splitlines()
            for line in lines:
                if any(k in line for k in ['Next Run Time', '次回の実行時刻', 'Last Run',
                                            '前回の実行時刻', 'Status', '状態']):
                    print(f'  {t["name"]} | {line.strip()}')
        else:
            print(f'[NG] {t["name"]} not registered (rc={r.returncode})')
    return 0


def cmd_remove(args):
    for t in TASKS:
        full_cmd = ['schtasks', '/delete', '/tn', t['name'], '/f']
        print(f'[REMOVE] {t["name"]}')
        if args.dry_run:
            print('  [DRY-RUN] skip')
            continue
        try:
            r = subprocess.run(full_cmd, capture_output=True, text=True, timeout=15,
                               encoding='cp932', errors='replace')
            if r.returncode == 0:
                print('  [OK] removed')
            else:
                print(f'  [FAIL] rc={r.returncode}: {r.stderr[:200]}')
        except Exception as e:
            print(f'  [ERROR] {e}')
    return 0


def main():
    ap = argparse.ArgumentParser(description='YouTube JRA LIVE schtask 登録')
    ap.add_argument('--check', action='store_true', help='登録状態確認')
    ap.add_argument('--remove', action='store_true', help='登録解除')
    ap.add_argument('--dry-run', dest='dry_run', action='store_true', help='cmd 表示のみ')
    args = ap.parse_args()

    if args.check:
        return cmd_check(args)
    elif args.remove:
        return cmd_remove(args)
    else:
        return cmd_register(args)


if __name__ == '__main__':
    sys.exit(main())
