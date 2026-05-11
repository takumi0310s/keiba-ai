#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V21 video features unified pipeline (capture → YOLOv8 → gait → body condition).

paddock_pipeline.py の後に YOLOv8 + gait features + body condition を 一括実行し、
horse_id 単位で V21 features CSV を生成する 統合 runner。

【処理 chain】
1. paddock_pipeline.py で frame capture (skip 済なら そのまま)
2. video_ai_yolov8.py で bbox 検出
3. video_ai_gait_features.py で gait/motion 20 features
4. video_ai_body_condition.py で 馬体 condition 18 features
5. 全 horse_id × race_id を 1 CSV にまとめる (V21 学習用)

【V15 投資保護】 全 新規 file、 V15 model 不変

Usage:
    # 5/10 全 race × top 3 馬 自動 chain
    python tools/video_pipeline_unified.py 20260510 --top-n 3

    # 既存 frame で features 再抽出のみ (capture skip)
    python tools/video_pipeline_unified.py --features-only 20260411 --top-n 3

    # 動作 確認 (1 race のみ)
    python tools/video_pipeline_unified.py 20260411 --top-n 1 --max-races 1
"""
import argparse
import csv
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
FRAME_DIR = os.path.join(BASE_DIR, 'data', 'paddock_frames')
FEATURES_DIR = os.path.join(BASE_DIR, 'data', 'video_ai_features')
OUT_CSV = os.path.join(BASE_DIR, 'data', 'v21_video_features.csv')


def run_cmd(cmd, timeout=300):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding='utf-8', errors='replace')
        return r.returncode == 0, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return False, '', 'TIMEOUT'
    except Exception as e:
        return False, '', str(e)


def process_one(race_id, horse_id, fps, duration, skip_capture=False):
    """1 馬 chain: capture → YOLOv8 → gait → body condition."""
    frame_subdir = os.path.join(FRAME_DIR, f'{race_id}_{horse_id}')
    feats_subdir = os.path.join(FEATURES_DIR, f'{race_id}_{horse_id}')

    # 1. capture (paddock_video_capture.py)
    manifest = os.path.join(frame_subdir, 'manifest.json')
    has_frames = os.path.exists(manifest)
    if has_frames:
        try:
            m = json.load(open(manifest, 'r', encoding='utf-8'))
            n_frames = m.get('summary', {}).get('frames_saved', 0)
            has_frames = n_frames > 5
        except Exception:
            has_frames = False

    if not skip_capture and not has_frames:
        ok, _, _ = run_cmd([
            sys.executable, os.path.join(BASE_DIR, 'tools', 'paddock_video_capture.py'),
            str(horse_id), '--race-id', str(race_id),
            '--fps', str(fps), '--duration', str(duration),
        ], timeout=duration + 90)
        if not ok:
            return {'status': 'capture_failed', 'horse_id': horse_id, 'race_id': race_id}
    elif not has_frames:
        return {'status': 'no_frames_skip_capture', 'horse_id': horse_id, 'race_id': race_id}

    # 2. YOLOv8 bbox
    yolo_json = os.path.join(feats_subdir, 'yolov8_features.json')
    if not os.path.exists(yolo_json):
        ok, _, _ = run_cmd([
            sys.executable, os.path.join(BASE_DIR, 'tools', 'video_ai_yolov8.py'),
            frame_subdir, '--conf', '0.2',
        ], timeout=180)
        if not ok or not os.path.exists(yolo_json):
            return {'status': 'yolo_failed', 'horse_id': horse_id, 'race_id': race_id}

    # 3. gait features
    gait_json = os.path.join(feats_subdir, 'gait_features.json')
    if not os.path.exists(gait_json):
        ok, _, _ = run_cmd([
            sys.executable, os.path.join(BASE_DIR, 'tools', 'video_ai_gait_features.py'),
            feats_subdir,
        ], timeout=60)
        if not ok or not os.path.exists(gait_json):
            return {'status': 'gait_failed', 'horse_id': horse_id, 'race_id': race_id}

    # 4. body condition
    body_json = os.path.join(feats_subdir, 'body_condition_features.json')
    if not os.path.exists(body_json):
        ok, _, _ = run_cmd([
            sys.executable, os.path.join(BASE_DIR, 'tools', 'video_ai_body_condition.py'),
            frame_subdir,
        ], timeout=60)
        if not ok or not os.path.exists(body_json):
            return {'status': 'body_failed', 'horse_id': horse_id, 'race_id': race_id}

    # collect features
    try:
        gait_d = json.load(open(gait_json, 'r', encoding='utf-8')).get('features', {})
        body_d = json.load(open(body_json, 'r', encoding='utf-8')).get('aggregated', {})
        return {
            'status': 'OK',
            'race_id': race_id,
            'horse_id': horse_id,
            **{f'gait_{k}': v for k, v in gait_d.items()},
            **{f'body_{k}': v for k, v in body_d.items()},
        }
    except Exception as e:
        return {'status': 'merge_failed', 'horse_id': horse_id, 'race_id': race_id, 'msg': str(e)}


def main():
    ap = argparse.ArgumentParser(description='V21 video features unified pipeline')
    ap.add_argument('date', help='YYYYMMDD (daily_predictions ファイル参照)')
    ap.add_argument('--top-n', dest='top_n', type=int, default=3)
    ap.add_argument('--fps', type=int, default=3)
    ap.add_argument('--duration', type=int, default=30)
    ap.add_argument('--max-races', dest='max_races', type=int, default=None)
    ap.add_argument('--features-only', dest='features_only', action='store_true',
                    help='capture skip、 既存 frame で features 再抽出のみ')
    ap.add_argument('--sleep-between', dest='sleep_between', type=int, default=3)
    args = ap.parse_args()

    daily_csv = os.path.join(BASE_DIR, 'data', 'daily_predictions', f'{args.date}.csv')
    if not os.path.exists(daily_csv):
        print(f'[ERROR] not found: {daily_csv}')
        return 1

    # 1. paddock_pipeline.py で capture (or skip)
    if not args.features_only:
        print(f'[STEP 1/2] Running paddock_pipeline.py for {args.date}...')
        cmd = [
            sys.executable, os.path.join(BASE_DIR, 'tools', 'paddock_pipeline.py'),
            args.date, '--top-n', str(args.top_n),
            '--fps', str(args.fps), '--duration', str(args.duration),
            '--sleep-between', str(args.sleep_between),
        ]
        if args.max_races:
            cmd += ['--max-races', str(args.max_races)]
        run_cmd(cmd, timeout=args.duration * args.top_n * (args.max_races or 12) * 2)

    # 2. 各 race × horse の features 抽出
    print(f'[STEP 2/2] Running video AI features extraction...')
    races = []
    with open(daily_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('race_id'):
                races.append(row)
    if args.max_races:
        races = races[:args.max_races]

    all_features = []
    for race in races:
        race_id = race['race_id']
        # for top-N, need horse_ids - take from existing manifest if available
        # or from paddock_frames dir scan
        dir_pattern = os.path.join(FRAME_DIR, f'{race_id}_*')
        import glob
        candidate_dirs = glob.glob(dir_pattern)
        for cd in candidate_dirs[:args.top_n]:
            base = os.path.basename(cd)
            try:
                horse_id = base.split('_', 1)[1]
            except Exception:
                continue
            print(f'[{race_id}] horse_id={horse_id}')
            r = process_one(race_id, horse_id, args.fps, args.duration,
                             skip_capture=args.features_only)
            if r['status'] == 'OK':
                all_features.append(r)
                print(f'  OK: gait + body 抽出済')
            else:
                print(f'  {r["status"]}: {r.get("msg", "")}')

    if not all_features:
        print('[WARN] no features extracted')
        return 1

    # CSV 出力
    cols = sorted({k for r in all_features for k in r.keys() if k != 'status'})
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_features:
            row = {k: r.get(k) for k in cols}
            w.writerow(row)

    print(f'\n[OK] {len(all_features)} horses x {len(cols)-2} features')
    print(f'[OK] saved: {OUT_CSV}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
