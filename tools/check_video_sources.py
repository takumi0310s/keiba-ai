#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""動画 / 新規 source 鮮度 health check (Phase 22-24 source 用).

既存 nightly_sanity_check.py は V15 production 範囲。 本 script は
Phase 22-24 で追加した data source / pipeline / schtask 健全性を チェック。

【検査項目】
1. data/cookies.json mtime (age < 14 日)
2. data/paddock_frames/ 蓄積数 (race × horse)
3. data/youtube_jra_live/ (今後の蓄積予定)
4. data/video_ai_features/ (YOLOv8 + gait 出力)
5. data/shadow_log/ (Phase 23 shadow 結果)
6. YouTube schtask 登録 状態
7. paddock_pipeline_log.json 確認
8. data/race_review_features.csv 鮮度
9. data/event_effect_features.csv 鮮度

Usage:
    python tools/check_video_sources.py
    python tools/check_video_sources.py --json
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


def file_age_days(path):
    if not os.path.exists(path):
        return None
    return (time.time() - os.path.getmtime(path)) / 86400


def status_for_age(age_days, warn_days, ng_days):
    if age_days is None:
        return 'MISSING'
    if age_days > ng_days:
        return 'NG'
    if age_days > warn_days:
        return 'WARN'
    return 'OK'


def count_dir(path, ext=None):
    if not os.path.exists(path):
        return 0
    total = 0
    for root, dirs, files in os.walk(path):
        if ext:
            files = [f for f in files if f.endswith(ext)]
        total += len(files)
    return total


def count_subdirs(path):
    if not os.path.exists(path):
        return 0
    return sum(1 for d in os.listdir(path)
               if os.path.isdir(os.path.join(path, d)))


def check_schtask(name):
    try:
        r = subprocess.run(['schtasks', '/query', '/tn', name, '/fo', 'list'],
                           capture_output=True, text=True, timeout=10,
                           encoding='cp932', errors='replace')
        return r.returncode == 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description='Phase 22-24 動画 / data source 鮮度 check')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    checks = []

    # 1. Cookie
    cookie_path = os.path.join(BASE_DIR, 'data', 'cookies.json')
    age = file_age_days(cookie_path)
    checks.append({
        'name': 'cookies.json mtime',
        'value': f'{age:.1f}d' if age is not None else 'MISSING',
        'status': status_for_age(age, warn_days=7, ng_days=14),
    })

    # 2. Paddock frames
    paddock_dir = os.path.join(BASE_DIR, 'data', 'paddock_frames')
    n_paddock_dirs = count_subdirs(paddock_dir)
    n_paddock_frames = count_dir(paddock_dir, '.jpg')
    checks.append({
        'name': 'paddock_frames dirs (race×horse)',
        'value': n_paddock_dirs,
        'status': 'OK' if n_paddock_dirs >= 1 else 'WARN',
    })
    checks.append({
        'name': 'paddock_frames total jpgs',
        'value': n_paddock_frames,
        'status': 'OK' if n_paddock_frames >= 20 else 'WARN',
    })

    # 3. YouTube live
    yt_dir = os.path.join(BASE_DIR, 'data', 'youtube_jra_live')
    n_yt = count_dir(yt_dir, '.mp4')
    checks.append({
        'name': 'youtube_jra_live mp4 count',
        'value': n_yt,
        'status': 'OK' if n_yt >= 0 else 'WARN',  # 0 でも今は OK (5/16+ で蓄積開始)
    })

    # 4. Video AI features
    feat_dir = os.path.join(BASE_DIR, 'data', 'video_ai_features')
    n_feats = count_subdirs(feat_dir)
    checks.append({
        'name': 'video_ai_features dirs',
        'value': n_feats,
        'status': 'OK' if n_feats >= 1 else 'WARN',
    })

    # 5. Shadow logs
    shadow_dir = os.path.join(BASE_DIR, 'data', 'shadow_log')
    n_shadow = count_dir(shadow_dir, '.json')
    checks.append({
        'name': 'shadow_log json count',
        'value': n_shadow,
        'status': 'OK' if n_shadow >= 0 else 'WARN',
    })

    # 6. YouTube schtask
    for t in ['Keiba-YouTubeLiveRecord-Sat', 'Keiba-YouTubeLiveRecord-Sun']:
        registered = check_schtask(t)
        checks.append({
            'name': f'schtask: {t}',
            'value': 'registered' if registered else 'not_registered',
            'status': 'OK' if registered else 'WARN',
        })

    # 7. Pipeline log
    pipeline_log = os.path.join(BASE_DIR, 'data', 'paddock_pipeline_log.json')
    age = file_age_days(pipeline_log)
    checks.append({
        'name': 'paddock_pipeline_log.json',
        'value': f'{age:.1f}d' if age is not None else 'MISSING',
        'status': 'OK' if age is not None and age < 7 else 'WARN',
    })

    # 8. Race review features
    rrf = os.path.join(BASE_DIR, 'data', 'race_review_features.csv')
    age = file_age_days(rrf)
    checks.append({
        'name': 'race_review_features.csv',
        'value': f'{age:.1f}d' if age is not None else 'MISSING',
        'status': status_for_age(age, warn_days=14, ng_days=30),
    })

    # 9. Event effects
    eef = os.path.join(BASE_DIR, 'data', 'event_effect_features.csv')
    age = file_age_days(eef)
    checks.append({
        'name': 'event_effect_features.csv',
        'value': f'{age:.1f}d' if age is not None else 'MISSING',
        'status': status_for_age(age, warn_days=14, ng_days=30),
    })

    # Aggregate
    ok = sum(1 for c in checks if c['status'] == 'OK')
    warn = sum(1 for c in checks if c['status'] == 'WARN')
    ng = sum(1 for c in checks if c['status'] == 'NG')
    missing = sum(1 for c in checks if c['status'] == 'MISSING')

    if args.json:
        print(json.dumps({
            'checks': checks,
            'summary': {'OK': ok, 'WARN': warn, 'NG': ng, 'MISSING': missing},
            'checked_at': datetime.now().isoformat(),
        }, indent=2, ensure_ascii=False))
    else:
        print('=== Phase 22-24 video / data source health check ===\n')
        for c in checks:
            marker = {'OK': '✓', 'WARN': '⚠', 'NG': '✗', 'MISSING': '?'}.get(c['status'], '?')
            print(f'  {marker} {c["status"]:<8} {c["name"]:<40} = {c["value"]}')
        print(f'\nSummary: OK={ok}, WARN={warn}, NG={ng}, MISSING={missing}')

    return 2 if ng > 0 else (1 if warn > 0 else 0)


if __name__ == '__main__':
    sys.exit(main())
