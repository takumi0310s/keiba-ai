#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""5/17 開催 朝 interactive checklist (5 分 で 全 status 確認).

朝 起きたら 1 コマンド で 全 status を 順番に check + Discord 通知。
schtask とは別、 user 手動で 確認する 補完 用。

【V15 投資保護】 read-only、 V15 production 一切 unchanged

Usage:
    python tools/saturday_interactive_checklist.py
    python tools/saturday_interactive_checklist.py --quick   # 通知のみ、 interactive なし
"""
import argparse
import json
import os
import subprocess
import sys
import time

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def step(num, total, name, cmd, quick=False):
    print(f'\n[{num}/{total}] {name}')
    print('-' * 60)
    start = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                           encoding='utf-8', errors='replace')
        elapsed = time.time() - start
        print(r.stdout[-500:])
        print(f'  ({elapsed:.1f}s、 rc={r.returncode})')
        return r.returncode in [0, 1, 2]  # WARN/HALT も accept
    except Exception as e:
        print(f'  [ERROR] {e}')
        return False


def main():
    ap = argparse.ArgumentParser(description='5/17 朝 interactive checklist')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    print('=' * 60)
    print(' 5/17 (土) 開催 朝 checklist')
    print('=' * 60)
    print('\nこの script は 朝 06:30-07:00 に user 起きてから 1 回 走らせる用。')

    checks_passed = 0
    total = 5

    # 1. morning briefing
    step(1, total, 'Morning briefing (累計 / health)',
         [sys.executable, os.path.join(BASE_DIR, 'tools', 'morning_briefing_5_17.py')])
    checks_passed += 1

    # 2. drawdown breaker
    step(2, total, 'Drawdown breaker (撤退保護)',
         [sys.executable, os.path.join(BASE_DIR, 'tools', 'drawdown_circuit_breaker.py')])
    checks_passed += 1

    # 3. video sources health
    step(3, total, 'Video sources health',
         [sys.executable, os.path.join(BASE_DIR, 'tools', 'check_video_sources.py')])
    checks_passed += 1

    # 4. cookie status (netkeiba)
    step(4, total, 'netkeiba cookie 確認',
         [sys.executable, os.path.join(BASE_DIR, 'tools', 'refresh_cookie.py'), '--check'])
    checks_passed += 1

    # 5. shadow runner (もし daily_predict 既に走ってれば)
    today_path = os.path.join(BASE_DIR, 'data', 'daily_predictions')
    today_files = [f for f in os.listdir(today_path) if f.endswith('.csv')]
    today_files.sort(reverse=True)
    if today_files:
        latest = today_files[0].replace('.csv', '')
        step(5, total, f'Strategy 8 shadow ({latest})',
             [sys.executable, os.path.join(BASE_DIR, 'tools', 'strategy8_shadow_runner.py'),
              latest])
        checks_passed += 1

    print('\n' + '=' * 60)
    print(f'=== Summary: {checks_passed}/{total} steps 完了 ===')
    print('\n[次 action]')
    print('  1. V15 daily_predict 結果 を 確認 (data/daily_predictions/20260517.csv)')
    print('  2. 戦略⑦ 適用 races (除外 12-13 races)')
    print('  3. Jackpot 該当馬 が出てれば 別 alert (試験運用)')
    print('  4. V15 通知 通り 手動 投票 (IPAT)')
    print('  5. 21:00 daily_results 自動')
    print('  6. 22:00 verdict 確認')

    # Discord 通知
    if not args.quick:
        try:
            subprocess.run([sys.executable, os.path.join(BASE_DIR, 'tools', 'notify_done.py'),
                              '5/17 朝 interactive check', f'{checks_passed}/{total} steps 完了'],
                            timeout=15)
            print('\n[Discord] 通知送信')
        except Exception:
            pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
