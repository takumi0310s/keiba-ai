#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 23 全 tool end-to-end smoke test.

production-ready 判定の前段階として、 Phase 23 全 tool が連動して動くか 1 コマンドで確認。

【テスト項目】
1. calibrate_confidence.py demo (isotonic + Platt 動作)
2. kelly_bet_sizer.py demo (戦略⑦ 全 EV+ で bet>0)
3. exotic_optimizer.py demo (trio EV ranking)
4. build_remarks_features.py (race_review CSV → features CSV 生成済 確認)
5. build_event_effect_features.py (events CSV 生成済 確認)
6. v21_multimodal_poc.py demo
7. video_ai_body_condition.py (既存 paddock dir、 condition 出力済 確認)
8. drawdown_circuit_breaker.py (現状 status 取得)
9. phase23_shadow_runner.py manual (sample race)
10. check_video_sources.py (健全性)

Usage:
    python tools/phase23_smoke_test.py
    python tools/phase23_smoke_test.py --verbose
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


def run_test(name, cmd, timeout=120, accept_rcs=None):
    """accept_rcs: 容認する exit code 一覧 (None=0 のみ)。 WARN等 で 1-2 返す tool 用。"""
    if accept_rcs is None:
        accept_rcs = [0]
    start = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding='utf-8', errors='replace')
        elapsed = time.time() - start
        status = 'OK' if r.returncode in accept_rcs else 'FAIL'
        if r.returncode != 0 and r.returncode in accept_rcs:
            status = 'WARN'  # 正常な WARN status を OK 区別
        return {
            'name': name,
            'status': status,
            'rc': r.returncode,
            'elapsed_sec': round(elapsed, 2),
            'stdout_tail': r.stdout[-300:] if r.stdout else '',
            'stderr_tail': r.stderr[-200:] if r.stderr else '',
        }
    except subprocess.TimeoutExpired:
        return {'name': name, 'status': 'TIMEOUT', 'elapsed_sec': timeout}
    except Exception as e:
        return {'name': name, 'status': 'ERROR', 'msg': str(e)}


def file_exists_check(name, path):
    return {
        'name': name,
        'status': 'OK' if os.path.exists(path) else 'MISSING',
        'path': path,
        'size': os.path.getsize(path) if os.path.exists(path) else 0,
    }


def main():
    ap = argparse.ArgumentParser(description='Phase 23 smoke test')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    print('=== Phase 23 全 tool end-to-end smoke test ===\n')

    results = []

    # 1. calibrate_confidence demo
    print('[1/10] calibrate_confidence.py demo')
    results.append(run_test('calibrate_confidence',
        [sys.executable, os.path.join(BASE_DIR, 'tools', 'calibrate_confidence.py'), 'demo']))

    # 2. kelly_bet_sizer demo
    print('[2/10] kelly_bet_sizer.py demo')
    results.append(run_test('kelly_bet_sizer',
        [sys.executable, os.path.join(BASE_DIR, 'tools', 'kelly_bet_sizer.py'), 'demo']))

    # 3. exotic_optimizer demo
    print('[3/10] exotic_optimizer.py demo')
    results.append(run_test('exotic_optimizer',
        [sys.executable, os.path.join(BASE_DIR, 'tools', 'exotic_optimizer.py'), 'demo']))

    # 4. remarks features file 確認
    print('[4/10] race_review_features.csv 確認')
    results.append(file_exists_check('race_review_features.csv',
        os.path.join(BASE_DIR, 'data', 'race_review_features.csv')))

    # 5. event effects file 確認
    print('[5/10] event_effect_features.csv 確認')
    results.append(file_exists_check('event_effect_features.csv',
        os.path.join(BASE_DIR, 'data', 'event_effect_features.csv')))

    # 6. v21_multimodal_poc demo
    print('[6/10] v21_multimodal_poc.py demo')
    results.append(run_test('v21_multimodal_poc',
        [sys.executable, os.path.join(BASE_DIR, 'tools', 'v21_multimodal_poc.py'), 'demo']))

    # 7. body condition features 確認
    print('[7/10] body_condition_features.json (paddock 既存) 確認')
    results.append(file_exists_check('body_condition_features (paddock)',
        os.path.join(BASE_DIR, 'data', 'video_ai_features',
                      '202603010112_2022106229', 'body_condition_features.json')))

    # 8. drawdown breaker status (rc 1=WARN, 2=HALT, 3=STOP は accept)
    print('[8/10] drawdown_circuit_breaker.py status')
    results.append(run_test('drawdown_breaker',
        [sys.executable, os.path.join(BASE_DIR, 'tools', 'drawdown_circuit_breaker.py')],
        accept_rcs=[0, 1, 2, 3]))

    # 9. shadow runner manual
    print('[9/10] phase23_shadow_runner.py manual')
    results.append(run_test('shadow_runner',
        [sys.executable, os.path.join(BASE_DIR, 'tools', 'phase23_shadow_runner.py'),
         '--race-id', 'TEST_RACE_ID', '--probs', '0.65,0.13,0.10', '--odds', '3.5,5.2,8.1']))

    # 10. health check (rc 1=WARN, 2=NG は accept)
    print('[10/10] check_video_sources.py')
    results.append(run_test('video_sources_check',
        [sys.executable, os.path.join(BASE_DIR, 'tools', 'check_video_sources.py')],
        accept_rcs=[0, 1, 2]))

    # Summary
    print('\n=== Summary ===')
    ok = sum(1 for r in results if r['status'] == 'OK')
    warn = sum(1 for r in results if r['status'] == 'WARN')
    fail = sum(1 for r in results if r['status'] in ['FAIL', 'TIMEOUT', 'ERROR', 'MISSING'])
    for r in results:
        marker = {'OK': '✓', 'WARN': '⚠', 'FAIL': '✗', 'TIMEOUT': '⏱', 'ERROR': '!', 'MISSING': '?'}.get(r['status'], '?')
        size_info = f' ({r["size"]:>8,} B)' if 'size' in r else ''
        elapsed = f' ({r.get("elapsed_sec", "?")}s)' if 'elapsed_sec' in r else ''
        print(f'  {marker} {r["status"]:<8} {r["name"]:<30}{size_info}{elapsed}')
        if r['status'] not in ('OK', 'MISSING') and args.verbose:
            print(f'    stderr: {r.get("stderr_tail", "")}')

    print(f'\n[RESULT] OK={ok}, WARN={warn}, FAIL/MISSING={fail}')

    # Save
    out_path = os.path.join(BASE_DIR, 'data', 'v18', f'phase23_smoke_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    json.dump({'results': results, 'summary': {'OK': ok, 'FAIL': fail},
                'tested_at': datetime.now().isoformat()},
              open(out_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'[OK] saved: {out_path}')

    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
