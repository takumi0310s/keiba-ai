#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W1.6: YOLOv8 bbox tracking ベース gait / posture / motion features 算出.

DLC SuperAnimal などの専用 model なしで、 YOLOv8 が出す bbox の時系列変化から
gait / posture / motion 系特徴量を抽出する 簡易版。 V21 features の基盤。

入力: data/video_ai_features/{frame_dir}/yolov8_features.json (per-frame bbox)
出力: data/video_ai_features/{frame_dir}/gait_features.json (動画 1 本 ≒ 1 馬の特徴量)

【抽出 features】
- bbox aspect ratio (width/height): 立姿勢 vs gallop 姿勢
- bbox area shift: 距離変化 / camera zoom
- bbox center motion (cx velocity, cy velocity): 移動速度
- detection coverage: 何 % の frame で検出されたか
- conf 統計: avg / std / min / max
- area 統計: avg / std (体格 視認性)

Usage:
    python tools/video_ai_gait_features.py data/video_ai_features/202603010112_2022106229/
    python tools/video_ai_gait_features.py path/to/frame_dir/ --target-idx 0  # 多頭 race で n 番目馬 tracking
"""
import argparse
import json
import os
import sys
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mean(xs):
    return sum(xs) / max(1, len(xs))


def std(xs):
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return var ** 0.5


def diff(xs):
    return [xs[i+1] - xs[i] for i in range(len(xs) - 1)]


def extract_target_horse_track(per_frame, target_idx=0):
    """各 frame の horses[] から、 最大 bbox の馬を抽出 (single horse paddock 想定)。

    target_idx > 0 の場合は、 face_x (left-to-right) ソートで n 番目の馬を返す。
    """
    tracks = []
    for frame in per_frame:
        horses = frame.get('horses', [])
        if not horses:
            tracks.append(None)
            continue
        if target_idx == 0:
            # 最大 bbox の馬 (paddock の主役馬)
            h = max(horses, key=lambda x: x['area'])
        else:
            # 多頭の場合 cx でソート、 n 番目
            sorted_h = sorted(horses, key=lambda x: x['cx'])
            if target_idx < len(sorted_h):
                h = sorted_h[target_idx]
            else:
                h = sorted_h[0]
        tracks.append({
            'idx': frame['idx'],
            'cx': h['cx'],
            'cy': h['cy'],
            'w': h['w'],
            'h': h['h'],
            'area': h['area'],
            'conf': h['conf'],
            'aspect': h['w'] / max(1.0, h['h']),
        })
    return tracks


def compute_gait_features(tracks):
    valid = [t for t in tracks if t is not None]
    n_total = len(tracks)
    n_valid = len(valid)
    coverage = n_valid / max(1, n_total)

    if n_valid < 2:
        return {
            'n_frames': n_total,
            'coverage': coverage,
            'note': 'insufficient detection',
        }

    aspects = [t['aspect'] for t in valid]
    areas = [t['area'] for t in valid]
    cxs = [t['cx'] for t in valid]
    cys = [t['cy'] for t in valid]
    ws = [t['w'] for t in valid]
    hs = [t['h'] for t in valid]
    confs = [t['conf'] for t in valid]

    dcx = diff(cxs)
    dcy = diff(cys)
    darea = diff(areas)
    daspect = diff(aspects)

    motion_speed = [abs(dx) + abs(dy) for dx, dy in zip(dcx, dcy)]

    feats = {
        # 基本統計
        'n_frames': n_total,
        'n_valid': n_valid,
        'coverage': round(coverage, 3),

        # bbox geometry
        'aspect_mean': round(mean(aspects), 3),
        'aspect_std': round(std(aspects), 3),
        'aspect_min': round(min(aspects), 3),
        'aspect_max': round(max(aspects), 3),
        'aspect_range': round(max(aspects) - min(aspects), 3),

        # bbox size
        'area_mean': round(mean(areas), 1),
        'area_std': round(std(areas), 1),
        'w_mean': round(mean(ws), 1),
        'h_mean': round(mean(hs), 1),

        # 検出信頼度
        'conf_mean': round(mean(confs), 3),
        'conf_std': round(std(confs), 3),
        'conf_min': round(min(confs), 3),

        # 移動 (camera 静止前提)
        'motion_dcx_mean': round(mean(dcx), 2) if dcx else 0,
        'motion_dcy_mean': round(mean(dcy), 2) if dcy else 0,
        'motion_speed_mean': round(mean(motion_speed), 2) if motion_speed else 0,
        'motion_speed_std': round(std(motion_speed), 2) if motion_speed else 0,
        'motion_speed_max': round(max(motion_speed), 2) if motion_speed else 0,

        # area / aspect 変動 (gait 周期性 proxy)
        'area_change_mean': round(mean([abs(d) for d in darea]), 1) if darea else 0,
        'aspect_change_mean': round(mean([abs(d) for d in daspect]), 3) if daspect else 0,
    }
    return feats


def main():
    ap = argparse.ArgumentParser(description='YOLOv8 bbox → gait/posture/motion features')
    ap.add_argument('input_dir', help='video_ai_features dir (yolov8_features.json 含む)')
    ap.add_argument('--target-idx', dest='target_idx', type=int, default=0,
                    help='多頭時の対象馬 (0=最大 bbox = paddock default、 1+=cx ソートで n 番目)')
    args = ap.parse_args()

    feats_path = os.path.join(args.input_dir, 'yolov8_features.json')
    if not os.path.exists(feats_path):
        print(f'[ERROR] not found: {feats_path}')
        print('  -> tools/video_ai_yolov8.py を先に実行してください')
        return 1

    with open(feats_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    per_frame = data.get('per_frame', [])
    if not per_frame:
        print('[ERROR] empty per_frame')
        return 1

    tracks = extract_target_horse_track(per_frame, target_idx=args.target_idx)
    feats = compute_gait_features(tracks)

    out_path = os.path.join(args.input_dir, 'gait_features.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'features': feats, 'computed_at': datetime.now().isoformat(),
                   'target_idx': args.target_idx}, f, indent=2, ensure_ascii=False)

    print(f'[OK] features extracted: {out_path}')
    for k, v in feats.items():
        print(f'  {k}: {v}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
