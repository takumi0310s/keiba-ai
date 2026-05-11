#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V21 auto-retrain trigger (paddock coverage 一定値超えで V21 再学習).

paddock_frames dir 数が 閾値 超えたら:
1. v21_extract_all_video_features.py 実行 (新 features 抽出)
2. v21_training_data_builder.py 実行 (V21 data 再構築)
3. train_v21_lgb_xgb.py 実行 (V21 再学習)
4. Discord 通知

【V15 投資保護】 V21 model のみ更新、 V15 production 不変

Usage:
    python tools/v21_auto_retrain_trigger.py --threshold 200
    python tools/v21_auto_retrain_trigger.py --threshold 1000 --no-train  # data update のみ
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_PATH = os.path.join(BASE_DIR, 'data', 'v21_retrain_state.json')


def count_paddock_dirs():
    p = os.path.join(BASE_DIR, 'data', 'paddock_frames')
    if not os.path.exists(p):
        return 0
    return len([d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))])


def load_state():
    if not os.path.exists(STATE_PATH):
        return {'last_retrain_count': 0, 'history': []}
    try:
        return json.load(open(STATE_PATH, 'r', encoding='utf-8'))
    except Exception:
        return {'last_retrain_count': 0, 'history': []}


def save_state(state):
    with open(STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def run_step(name, cmd, timeout=1200):
    print(f'\n[{name}] start')
    try:
        r = subprocess.run(cmd, timeout=timeout, capture_output=True,
                           text=True, encoding='utf-8', errors='replace')
        ok = (r.returncode == 0)
        print(f'[{name}] {"OK" if ok else "FAIL"} (rc={r.returncode})')
        if r.stdout:
            print(r.stdout[-400:])
        if not ok and r.stderr:
            print(f'STDERR: {r.stderr[-200:]}')
        return ok
    except subprocess.TimeoutExpired:
        print(f'[{name}] TIMEOUT')
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=int, default=200,
                    help='paddock dir 閾値 (此 数 超えで retrain)')
    ap.add_argument('--no-train', action='store_true', help='data update のみ、 学習 skip')
    ap.add_argument('--force', action='store_true', help='閾値 無視で 強制実行')
    args = ap.parse_args()

    state = load_state()
    current = count_paddock_dirs()
    last = state.get('last_retrain_count', 0)

    print(f'[INFO] paddock dirs: current={current}, last_retrain={last}')
    print(f'[INFO] threshold: {args.threshold} delta={current - last}')

    if not args.force and (current - last) < args.threshold:
        print(f'[INFO] threshold 未到達 ({current - last} < {args.threshold})、 SKIP')
        return 0

    # Step 1: 新 features 抽出
    if not run_step('extract', [sys.executable, os.path.join(BASE_DIR, 'tools',
                                'v21_extract_all_video_features.py')], timeout=3600):
        return 1

    # Step 2: V21 data 再構築
    if not run_step('build', [sys.executable, os.path.join(BASE_DIR, 'tools',
                              'v21_training_data_builder.py')], timeout=600):
        return 1

    if args.no_train:
        print('[INFO] --no-train 指定、 data update のみで終了')
    else:
        # Step 3: V21 学習
        if not run_step('train', [sys.executable, os.path.join(BASE_DIR, 'train',
                                  'train_v21_lgb_xgb.py')], timeout=1800):
            return 1

    # Save state
    state['last_retrain_count'] = current
    state['history'].append({
        'timestamp': datetime.now().isoformat(),
        'paddock_count': current,
        'no_train': args.no_train,
    })
    save_state(state)
    print(f'[OK] state saved: last_retrain_count={current}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
