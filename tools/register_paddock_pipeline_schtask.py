#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paddock_pipeline.py 自動 schtask 登録.

netkeiba paddock 動画は当日 18:00+ archive 化。 schtask で
日曜 20:00 / 月曜 20:00 に 前日 開催分 paddock × top 3 馬 自動 capture。

【V15 投資保護】 V15 production 一切 不変、 完全 background data 蓄積。

Usage:
    python tools/register_paddock_pipeline_schtask.py          # 登録
    python tools/register_paddock_pipeline_schtask.py --check
    python tools/register_paddock_pipeline_schtask.py --remove
    python tools/register_paddock_pipeline_schtask.py --dry-run
"""
import argparse
import os
import subprocess
import sys

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_SCRIPT = os.path.join(BASE_DIR, 'tools', 'video_pipeline_unified.py')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

TASKS = [
    {
        'name': 'Keiba-PaddockArchive-Sun',
        'schedule': 'WEEKLY',
        'day': 'SUN',
        'time': '20:00',
        'date_offset': -1,  # 前日 (土) の paddock を 取得
    },
    {
        'name': 'Keiba-PaddockArchive-Mon',
        'schedule': 'WEEKLY',
        'day': 'MON',
        'time': '20:00',
        'date_offset': -1,  # 前日 (日) の paddock
    },
]


def date_offset_cmd(offset):
    """Windows cmd で 当日 + offset の日付 を取得 (PowerShell 経由)."""
    return (
        f'for /f %%i in (\'powershell -NoProfile -Command "(Get-Date).AddDays({offset}).ToString(\\"yyyyMMdd\\")"\') '
        f'do (set DATESTR=%%i)'
    )


def task_command(date_offset):
    python_path = sys.executable
    log_path = os.path.join(LOG_DIR, 'paddock_archive_%date:~0,4%%date:~5,2%%date:~8,2%.log')
    # cmd /c で 日付計算 → pipeline 実行
    return (
        f'cmd /c "{date_offset_cmd(date_offset)} & "{python_path}" "{PIPELINE_SCRIPT}" '
        f'%DATESTR% --top-n 3 --fps 3 --duration 30 --sleep-between 5 '
        f'>> "{log_path}" 2>&1"'
    )


def cmd_register(args):
    os.makedirs(LOG_DIR, exist_ok=True)
    for t in TASKS:
        cmd_str = task_command(t['date_offset'])
        full_cmd = [
            'schtasks', '/create',
            '/tn', t['name'],
            '/sc', t['schedule'],
            '/d', t['day'],
            '/st', t['time'],
            '/tr', cmd_str,
            '/f',
            '/rl', 'HIGHEST',
        ]
        print(f'[REGISTER] {t["name"]} ({t["day"]} {t["time"]}、 前日 paddock)')

        if args.dry_run:
            print(f'  [DRY-RUN] tr: {cmd_str[:200]}...')
            continue

        try:
            r = subprocess.run(full_cmd, capture_output=True, text=True, timeout=30,
                               encoding='cp932', errors='replace')
            if r.returncode == 0:
                print(f'  [OK] registered')
            else:
                print(f'  [FAIL] rc={r.returncode}: {r.stderr[:200]}')
                print(f'  ※ admin 権限 必要')
        except Exception as e:
            print(f'  [ERROR] {e}')
        print()
    return 0


def cmd_check(args):
    for t in TASKS:
        try:
            r = subprocess.run(['schtasks', '/query', '/tn', t['name'], '/fo', 'list'],
                               capture_output=True, text=True, timeout=15,
                               encoding='cp932', errors='replace')
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if any(k in line for k in ['Next Run', '次回', 'Status', '状態']):
                        print(f'  {t["name"]}: {line.strip()}')
            else:
                print(f'[NG] {t["name"]} not registered')
        except Exception as e:
            print(f'[ERROR] {t["name"]}: {e}')
    return 0


def cmd_remove(args):
    for t in TASKS:
        full_cmd = ['schtasks', '/delete', '/tn', t['name'], '/f']
        print(f'[REMOVE] {t["name"]}')
        if args.dry_run:
            continue
        try:
            r = subprocess.run(full_cmd, capture_output=True, text=True, timeout=15,
                               encoding='cp932', errors='replace')
            print(f'  rc={r.returncode}')
        except Exception as e:
            print(f'  [ERROR] {e}')
    return 0


def main():
    ap = argparse.ArgumentParser(description='paddock_pipeline.py schtask 登録')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--remove', action='store_true')
    ap.add_argument('--dry-run', dest='dry_run', action='store_true')
    args = ap.parse_args()
    if args.check: return cmd_check(args)
    elif args.remove: return cmd_remove(args)
    else: return cmd_register(args)


if __name__ == '__main__':
    sys.exit(main())
