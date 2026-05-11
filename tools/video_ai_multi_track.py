#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""multi-horse tracking PoC: race 動画で 各馬を frame 間で 連続 ID 紐付け.

YOLOv8 が出す per-frame bbox を、 IoU + 距離 ベース で frame 間 connect。
個別馬 ID ごとに gait_features を集計可能にする。

入力: data/video_ai_features/{frame_dir}/yolov8_features.json
出力: data/video_ai_features/{frame_dir}/tracks.json (per-track frame 系列)
      data/video_ai_features/{frame_dir}/per_track_gait.json (track ごと gait features)

Usage:
    python tools/video_ai_multi_track.py data/video_ai_features/202603010112_0/
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


def iou(box_a, box_b):
    """Intersection over Union."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    x1 = max(xa1, xb1); y1 = max(ya1, yb1)
    x2 = min(xa2, xb2); y2 = min(ya2, yb2)
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / max(1e-9, union)


def center_dist(h_a, h_b):
    return ((h_a['cx'] - h_b['cx']) ** 2 + (h_a['cy'] - h_b['cy']) ** 2) ** 0.5


def assign_tracks(per_frame, iou_threshold=0.3, dist_threshold=80, max_gap=3):
    """各 horse detection に track_id を assign.

    Greedy matching: 前 frame の active track と IoU 高い detection を紐付け。
    紐付かない detection は新 track として開始。
    """
    next_id = 0
    tracks = {}  # track_id → {last_frame_idx, last_box, last_horse, history: [(frame_idx, horse_dict)]}

    for frame in per_frame:
        idx = frame['idx']
        horses = frame.get('horses', [])
        used_horse = [False] * len(horses)

        # 既存 track と match
        active = [(tid, t) for tid, t in tracks.items()
                  if idx - t['last_frame_idx'] <= max_gap]
        # 各 active track について、 最良 horse を選ぶ
        for tid, t in active:
            best_score = -1
            best_j = -1
            for j, h in enumerate(horses):
                if used_horse[j]:
                    continue
                iou_v = iou(t['last_box'], h['bbox'])
                dist = center_dist(t['last_horse'], h)
                # score = IoU 重視、 距離 penalty
                score = iou_v - 0.001 * dist
                if iou_v >= iou_threshold or dist <= dist_threshold:
                    if score > best_score:
                        best_score = score
                        best_j = j
            if best_j >= 0:
                h = horses[best_j]
                used_horse[best_j] = True
                t['last_frame_idx'] = idx
                t['last_box'] = h['bbox']
                t['last_horse'] = h
                t['history'].append({'frame_idx': idx, **h})

        # 未紐付 detection → 新 track 開始
        for j, h in enumerate(horses):
            if used_horse[j]:
                continue
            tid = next_id; next_id += 1
            tracks[tid] = {
                'last_frame_idx': idx,
                'last_box': h['bbox'],
                'last_horse': h,
                'history': [{'frame_idx': idx, **h}],
            }

    return tracks


def compute_track_gait(history):
    n = len(history)
    if n < 2:
        return {'n_frames': n, 'note': 'insufficient'}
    cxs = [h['cx'] for h in history]
    cys = [h['cy'] for h in history]
    areas = [h['area'] for h in history]
    aspects = [h['w'] / max(1, h['h']) for h in history]
    confs = [h['conf'] for h in history]

    def mean(xs): return sum(xs) / len(xs)
    def std(xs):
        if len(xs) < 2: return 0.0
        mu = mean(xs); return (sum((x-mu)**2 for x in xs) / (len(xs)-1)) ** 0.5

    dcx = [cxs[i+1]-cxs[i] for i in range(n-1)]
    dcy = [cys[i+1]-cys[i] for i in range(n-1)]
    speeds = [abs(a)+abs(b) for a,b in zip(dcx, dcy)]

    return {
        'n_frames': n,
        'frame_idx_first': history[0]['frame_idx'],
        'frame_idx_last': history[-1]['frame_idx'],
        'cx_first': round(cxs[0], 1), 'cx_last': round(cxs[-1], 1),
        'cy_first': round(cys[0], 1), 'cy_last': round(cys[-1], 1),
        'area_mean': round(mean(areas), 1),
        'aspect_mean': round(mean(aspects), 3),
        'aspect_std': round(std(aspects), 3),
        'conf_mean': round(mean(confs), 3),
        'speed_mean': round(mean(speeds), 2),
        'speed_max': round(max(speeds), 2),
        'speed_std': round(std(speeds), 2),
        'total_dx': round(sum(dcx), 1),
        'total_dy': round(sum(dcy), 1),
    }


def main():
    ap = argparse.ArgumentParser(description='Multi-horse tracking PoC')
    ap.add_argument('input_dir', help='video_ai_features dir')
    ap.add_argument('--iou', type=float, default=0.3)
    ap.add_argument('--dist', type=int, default=80)
    ap.add_argument('--max-gap', dest='max_gap', type=int, default=3)
    ap.add_argument('--min-track-len', dest='min_track_len', type=int, default=3,
                    help='短すぎる track を捨てる threshold')
    args = ap.parse_args()

    feats_path = os.path.join(args.input_dir, 'yolov8_features.json')
    if not os.path.exists(feats_path):
        print(f'[ERROR] not found: {feats_path}')
        return 1
    data = json.load(open(feats_path, 'r', encoding='utf-8'))
    per_frame = data.get('per_frame', [])
    print(f'[INFO] {len(per_frame)} frames in input')

    tracks = assign_tracks(per_frame, iou_threshold=args.iou,
                            dist_threshold=args.dist, max_gap=args.max_gap)

    # filter short tracks
    valid_tracks = {tid: t for tid, t in tracks.items()
                    if len(t['history']) >= args.min_track_len}
    print(f'[INFO] {len(tracks)} raw tracks, {len(valid_tracks)} >= min_len {args.min_track_len}')

    # save
    out_tracks = {
        str(tid): {
            'n_frames': len(t['history']),
            'first_frame': t['history'][0]['frame_idx'],
            'last_frame': t['history'][-1]['frame_idx'],
            'history': t['history'],
        }
        for tid, t in valid_tracks.items()
    }
    per_track_gait = {
        str(tid): compute_track_gait(t['history'])
        for tid, t in valid_tracks.items()
    }

    json.dump(out_tracks, open(os.path.join(args.input_dir, 'tracks.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    json.dump(per_track_gait, open(os.path.join(args.input_dir, 'per_track_gait.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    print(f'[OK] saved: {args.input_dir}/tracks.json, per_track_gait.json')
    print('\n[Track summary]')
    for tid, g in sorted(per_track_gait.items(), key=lambda x: -x[1].get('n_frames', 0))[:10]:
        print(f'  track {tid}: frames={g["n_frames"]}, '
              f'speed_mean={g.get("speed_mean", "?")}, '
              f'area_mean={g.get("area_mean", "?")}, '
              f'dx={g.get("total_dx", "?")}, dy={g.get("total_dy", "?")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
