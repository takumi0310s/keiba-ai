#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 24 全 schtask 一括登録 (YouTube + paddock + 5min odds + morning briefing).

admin 権限で 1 コマンド で 5/17 開催 前必要な schtask を全部登録。

【登録 schtask】
1. Keiba-YouTubeLiveRecord-Sat (土 08:55)
2. Keiba-YouTubeLiveRecord-Sun (日 08:55)
3. Keiba-PaddockArchive-Sun (日 20:00、 前日 paddock archive)
4. Keiba-PaddockArchive-Mon (月 20:00、 前日 paddock archive)
5. Keiba-MorningBriefing-Sat (土 06:30、 朝 status)
6. Keiba-MorningBriefing-Sun (日 06:30)

Usage:
    python tools/register_all_phase24_schtasks.py
    python tools/register_all_phase24_schtasks.py --check
    python tools/register_all_phase24_schtasks.py --remove
    python tools/register_all_phase24_schtasks.py --dry-run
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


SUB_SCRIPTS = [
    {
        'name': 'YouTube schtask',
        'script': 'register_youtube_schtask.py',
    },
    {
        'name': 'Paddock archive schtask',
        'script': 'register_paddock_pipeline_schtask.py',
    },
]


def run(cmd, dry_run=False):
    print(f'[CMD] {" ".join(cmd)}')
    if dry_run:
        return True
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                           encoding='utf-8', errors='replace')
        if r.returncode == 0:
            print(f'  [OK]')
            return True
        else:
            print(f'  [FAIL] rc={r.returncode}')
            print(f'  stdout: {r.stdout[-300:]}')
            print(f'  stderr: {r.stderr[-200:]}')
            return False
    except Exception as e:
        print(f'  [ERROR] {e}')
        return False


def register_morning_briefing(args):
    """morning_briefing 専用 schtask (土 / 日 06:30)."""
    script = os.path.join(BASE_DIR, 'tools', 'morning_briefing_5_17.py')
    log_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'morning_briefing_%date:~0,4%%date:~5,2%%date:~8,2%.log')
    cmd_str = f'cmd /c ""{sys.executable}" "{script}" --discord >> "{log_path}" 2>&1"'

    for day_short in ['SAT', 'SUN']:
        task_name = f'Keiba-MorningBriefing-{day_short.title()}'
        full_cmd = [
            'schtasks', '/create',
            '/tn', task_name,
            '/sc', 'WEEKLY',
            '/d', day_short,
            '/st', '06:30',
            '/tr', cmd_str,
            '/f', '/rl', 'HIGHEST',
        ]
        print(f'[REGISTER] {task_name}')
        if args.dry_run:
            print(f'  [DRY-RUN] cmd: {cmd_str[:200]}...')
            continue
        try:
            r = subprocess.run(full_cmd, capture_output=True, text=True, timeout=30,
                               encoding='cp932', errors='replace')
            if r.returncode == 0:
                print(f'  [OK] registered')
            else:
                print(f'  [FAIL] rc={r.returncode}: {r.stderr[:200]}')
        except Exception as e:
            print(f'  [ERROR] {e}')


def cmd_register(args):
    print('=== Phase 24 全 schtask 一括登録 ===\n')

    # 1-2. YouTube + Paddock 既存 sub-script
    for sub in SUB_SCRIPTS:
        print(f'\n[{sub["name"]}]')
        script = os.path.join(BASE_DIR, 'tools', sub['script'])
        cmd = [sys.executable, script]
        if args.dry_run:
            cmd.append('--dry-run')
        run(cmd, dry_run=args.dry_run)

    # 3. Morning briefing
    print(f'\n[Morning Briefing schtask]')
    register_morning_briefing(args)

    print('\n=== 登録完了 ===')
    return 0


def cmd_check(args):
    print('=== schtask 確認 ===\n')
    all_tasks = [
        'Keiba-YouTubeLiveRecord-Sat',
        'Keiba-YouTubeLiveRecord-Sun',
        'Keiba-PaddockArchive-Sun',
        'Keiba-PaddockArchive-Mon',
        'Keiba-MorningBriefing-Sat',
        'Keiba-MorningBriefing-Sun',
    ]
    for t in all_tasks:
        try:
            r = subprocess.run(['schtasks', '/query', '/tn', t, '/fo', 'list'],
                               capture_output=True, text=True, timeout=15,
                               encoding='cp932', errors='replace')
            if r.returncode == 0:
                # 次回実行
                next_run = None
                for line in r.stdout.splitlines():
                    if any(k in line for k in ['Next Run', '次回']):
                        next_run = line.strip()
                        break
                print(f'  ✓ {t}: {next_run or "registered"}')
            else:
                print(f'  ✗ {t}: not registered')
        except Exception as e:
            print(f'  ! {t}: {e}')
    return 0


def cmd_remove(args):
    all_tasks = [
        'Keiba-YouTubeLiveRecord-Sat',
        'Keiba-YouTubeLiveRecord-Sun',
        'Keiba-PaddockArchive-Sun',
        'Keiba-PaddockArchive-Mon',
        'Keiba-MorningBriefing-Sat',
        'Keiba-MorningBriefing-Sun',
    ]
    for t in all_tasks:
        print(f'[REMOVE] {t}')
        if args.dry_run:
            continue
        try:
            r = subprocess.run(['schtasks', '/delete', '/tn', t, '/f'],
                               capture_output=True, text=True, timeout=15,
                               encoding='cp932', errors='replace')
            print(f'  rc={r.returncode}')
        except Exception as e:
            print(f'  [ERROR] {e}')
    return 0


def main():
    ap = argparse.ArgumentParser(description='Phase 24 全 schtask 一括登録')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--remove', action='store_true')
    ap.add_argument('--dry-run', dest='dry_run', action='store_true')
    args = ap.parse_args()
    if args.check: return cmd_check(args)
    elif args.remove: return cmd_remove(args)
    else: return cmd_register(args)


if __name__ == '__main__':
    sys.exit(main())
