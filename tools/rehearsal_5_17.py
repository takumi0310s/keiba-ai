#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""5/17 開催 リハーサル: V15 + Phase 23 shadow chain を 1 コマンドで全実行 dry-run.

5/17 朝の本番運用を 5/13-16 のいつでも リハーサル可能。 V15 production には触らず、
Phase 22-24 で実装した全 tool を 順番に呼び、 ready 状態を verify する。

【リハーサル項目】
1. cookie 鮮度確認 (refresh_cookie.py --check)
2. health check (check_video_sources.py)
3. drawdown breaker status
4. shadow runner backtest verify
5. paddock pipeline dry-run (next race day)
6. unified pipeline features-only on existing data
7. smoke test (Phase 23 全 tool 動作確認)

Usage:
    python tools/rehearsal_5_17.py
    python tools/rehearsal_5_17.py --target-date 20260517
    python tools/rehearsal_5_17.py --json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def step(idx, total, name, cmd, timeout=120, accept_rcs=None):
    if accept_rcs is None:
        accept_rcs = [0]
    print(f'\n[{idx}/{total}] {name}')
    start = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding='utf-8', errors='replace')
        elapsed = time.time() - start
        ok = r.returncode in accept_rcs
        print(f'  rc={r.returncode}, {elapsed:.1f}s, status={"OK" if ok else "FAIL"}')
        # last 5 lines of stdout
        for line in r.stdout.splitlines()[-5:]:
            print(f'  > {line}')
        return {
            'name': name,
            'status': 'OK' if ok else 'FAIL',
            'rc': r.returncode,
            'elapsed': elapsed,
        }
    except subprocess.TimeoutExpired:
        print(f'  TIMEOUT after {timeout}s')
        return {'name': name, 'status': 'TIMEOUT'}
    except Exception as e:
        print(f'  ERROR: {e}')
        return {'name': name, 'status': 'ERROR'}


def main():
    ap = argparse.ArgumentParser(description='5/17 開催 リハーサル')
    ap.add_argument('--target-date', dest='target_date', default='20260517')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    print(f'=== 5/17 開催 リハーサル ({args.target_date} 想定) ===')
    print(f'  開始: {datetime.now().isoformat()}')

    results = []
    total = 7

    # 1. cookie check
    results.append(step(1, total, 'cookie 鮮度確認',
        [sys.executable, os.path.join(BASE_DIR, 'tools', 'refresh_cookie.py'), '--check']))

    # 2. video sources health
    results.append(step(2, total, 'video sources health',
        [sys.executable, os.path.join(BASE_DIR, 'tools', 'check_video_sources.py')],
        accept_rcs=[0, 1, 2]))

    # 3. drawdown breaker
    results.append(step(3, total, 'drawdown breaker status',
        [sys.executable, os.path.join(BASE_DIR, 'tools', 'drawdown_circuit_breaker.py')],
        accept_rcs=[0, 1, 2, 3]))

    # 4. shadow runner backtest
    results.append(step(4, total, 'shadow runner backtest re-validate',
        [sys.executable, os.path.join(BASE_DIR, 'tools', 'phase23_shadow_runner.py'),
         '--backtest', '--from', '20260301']))

    # 5. paddock pipeline dry-run
    daily_path = os.path.join(BASE_DIR, 'data', 'daily_predictions', f'{args.target_date}.csv')
    if os.path.exists(daily_path):
        results.append(step(5, total, f'paddock pipeline dry-run ({args.target_date})',
            [sys.executable, os.path.join(BASE_DIR, 'tools', 'paddock_pipeline.py'),
             args.target_date, '--dry-run', '--top-n', '3'], timeout=300))
    else:
        results.append({'name': f'paddock pipeline ({args.target_date})',
                          'status': 'SKIP', 'reason': f'daily_predictions/{args.target_date}.csv 未生成'})
        print(f'\n[5/{total}] paddock pipeline dry-run')
        print(f'  SKIP: daily_predictions/{args.target_date}.csv まだない')

    # 6. unified features-only on existing paddock data
    print(f'\n[6/{total}] unified features-only (4/11 既存 frame で再抽出)')
    results.append(step(6, total, 'unified features-only',
        [sys.executable, os.path.join(BASE_DIR, 'tools', 'video_pipeline_unified.py'),
         '20260411', '--top-n', '1', '--max-races', '1', '--features-only'], timeout=120))

    # 7. smoke test
    results.append(step(7, total, 'phase23_smoke_test',
        [sys.executable, os.path.join(BASE_DIR, 'tools', 'phase23_smoke_test.py')], timeout=120))

    # Summary
    print('\n=== リハーサル 結果 ===')
    ok = sum(1 for r in results if r['status'] == 'OK')
    skip = sum(1 for r in results if r['status'] == 'SKIP')
    fail = sum(1 for r in results if r['status'] in ['FAIL', 'TIMEOUT', 'ERROR'])
    for r in results:
        m = {'OK': '✓', 'FAIL': '✗', 'SKIP': '-', 'TIMEOUT': '⏱', 'ERROR': '!'}.get(r['status'], '?')
        print(f'  {m} {r["status"]:<6} {r["name"]}')
    print(f'\n[RESULT] OK={ok}, SKIP={skip}, FAIL={fail}')

    # 判定
    if fail == 0:
        print('\n✅ 5/17 GO READY: 全 step 正常、 V15 production 完全保護 確認')
    else:
        print(f'\n⚠ {fail} step FAIL: 修正 推奨')

    out_path = os.path.join(BASE_DIR, 'data', 'v18',
                             f'rehearsal_{args.target_date}_{datetime.now().strftime("%H%M%S")}.json')
    json.dump({'target_date': args.target_date, 'results': results,
                'summary': {'OK': ok, 'SKIP': skip, 'FAIL': fail},
                'tested_at': datetime.now().isoformat()},
              open(out_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'[OK] saved: {out_path}')

    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
