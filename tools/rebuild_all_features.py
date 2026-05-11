#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""All features re-build in one command (5/17+ 朝の features refresh 用).

V15 投資保護: 全 features re-build を 1 コマンド で実行可能に。
5/17 朝 / 5/24 V20 投入前 / 任意のタイミング で 全 features を 最新 化。

【V15 投資保護】 各 build script を call、 V15 model 不変

Usage:
    python tools/rebuild_all_features.py
    python tools/rebuild_all_features.py --skip event,pace  # 特定 skip
"""
import argparse
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

BUILDERS = [
    ('event', 'tools/build_event_effect_features.py', []),
    ('pace', 'tools/build_pace_features.py', []),
    ('pace_exp', 'tools/build_pace_features_expanding.py', []),
    ('hot_streak', 'tools/build_hot_streak_features.py', []),
    ('layoff', 'tools/build_layoff_features.py', []),
    ('distance_surface', 'tools/build_distance_surface_change_features.py', []),
    ('sire', 'tools/build_sire_class_down_features.py', []),
    ('remarks', 'tools/build_remarks_features.py', []),
]

INTEGRATED = [
    ('v20_full', 'tools/v20_training_data_full_builder.py', []),
]


def run(name, script, args):
    start = time.time()
    cmd = [sys.executable, os.path.join(BASE_DIR, script)] + args
    print(f'\n[{name}] {script}')
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           encoding='utf-8', errors='replace')
        elapsed = time.time() - start
        if r.returncode == 0:
            print(f'  [OK] {elapsed:.1f}s')
            # last 2 lines of stdout
            for line in r.stdout.strip().splitlines()[-3:]:
                print(f'    {line}')
        else:
            print(f'  [FAIL] rc={r.returncode}, stderr: {r.stderr[-200:]}')
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print(f'  [TIMEOUT]')
        return False
    except Exception as e:
        print(f'  [ERROR] {e}')
        return False


def main():
    ap = argparse.ArgumentParser(description='All features re-build')
    ap.add_argument('--skip', default='', help='comma sep skip list')
    ap.add_argument('--skip-integrated', action='store_true')
    args = ap.parse_args()

    skip_set = set(s.strip() for s in args.skip.split(',') if s.strip())
    print(f'=== Rebuild all features ===')
    print(f'skip: {skip_set if skip_set else "none"}')

    results = {}
    for name, script, sargs in BUILDERS:
        if name in skip_set:
            print(f'\n[{name}] SKIPPED')
            results[name] = 'SKIP'
            continue
        ok = run(name, script, sargs)
        results[name] = 'OK' if ok else 'FAIL'

    if not args.skip_integrated:
        for name, script, sargs in INTEGRATED:
            ok = run(name, script, sargs)
            results[name] = 'OK' if ok else 'FAIL'

    print('\n=== SUMMARY ===')
    for k, v in results.items():
        m = {'OK': '✓', 'FAIL': '✗', 'SKIP': '-'}.get(v, '?')
        print(f'  {m} {v:<5} {k}')

    ok_count = sum(1 for v in results.values() if v == 'OK')
    return 0 if ok_count == len([r for r in results.values() if r != 'SKIP']) else 1


if __name__ == '__main__':
    sys.exit(main())
