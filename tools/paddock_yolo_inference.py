"""paddock 動画 frame に YOLOv8 推論 で 馬体検出 + bbox + features 抽出.

入力: data/paddock_frames/{race_id_horse_id}/frame_*.jpg
出力: data/paddock_features/{race_id}.csv

抽出 features (per horse):
- pf_bbox_count_avg: 1 frame 中 馬体検出 数 (1 が理想、 0 か 2+ は noise)
- pf_bbox_size_avg: 馬体 bbox 面積 平均 (馬体 大き = 健康 sign)
- pf_bbox_aspect_avg: 横長度 (1.5-2.5 想定、 outlier 注意)
- pf_motion_amount: frame 間 bbox 位置 変動 (落ち着き 指標、 小 = 落ち着き OK)
- pf_motion_std: 動き の std (一定 動き か 不規則 か)
- pf_horse_confidence_avg: YOLOv8 confidence 平均 (高 = 検出 reliable)
- pf_frames_count: 利用 frame 数

V15 投資保護 完全。 paddock_archive (オフライン data、 規約遵守) のみ 利用。

usage:
    python tools/paddock_yolo_inference.py
    python tools/paddock_yolo_inference.py --max-dirs 10  # quick test
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# YOLOv8
try:
    from ultralytics import YOLO
except ImportError:
    print('[ERROR] ultralytics not installed. pip install ultralytics')
    sys.exit(1)

BASE = Path(__file__).resolve().parent.parent
PADDOCK_DIR = BASE / 'data' / 'paddock_frames'
OUT_DIR = BASE / 'data' / 'paddock_features'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# COCO class 17 = horse (YOLOv8 official)
HORSE_CLASS = 17


def analyze_dir(model: YOLO, frame_dir: Path) -> dict:
    """1 paddock dir (race_id_horse_id) を 解析."""
    frames = sorted(glob.glob(str(frame_dir / 'frame_*.jpg')) +
                    glob.glob(str(frame_dir / 'frame_*.png')))
    if not frames:
        return {}

    bbox_counts = []
    bbox_sizes = []
    bbox_aspects = []
    bbox_centers = []  # for motion calc
    confidences = []
    person_counts = []  # handler 検出

    for frame_path in frames:
        try:
            results = model(frame_path, classes=[HORSE_CLASS, 0],
                            verbose=False, device='cpu')
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                bbox_counts.append(0)
                person_counts.append(0)
                continue
            n_horse = 0
            n_person = 0
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == HORSE_CLASS:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w, h = x2 - x1, y2 - y1
                    bbox_sizes.append(w * h)
                    bbox_aspects.append(w / max(h, 1))
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    bbox_centers.append((float(cx), float(cy)))
                    confidences.append(float(box.conf[0]))
                    n_horse += 1
                elif cls == 0:  # person (handler)
                    n_person += 1
            bbox_counts.append(n_horse)
            person_counts.append(n_person)
        except Exception as e:
            print(f'  [WARN] {frame_path}: {e}')
            bbox_counts.append(0)

    if not bbox_sizes:
        return {
            'pf_frames_count': len(frames),
            'pf_bbox_count_avg': 0.0,
            'pf_bbox_size_avg': 0.0,
            'pf_bbox_aspect_avg': 0.0,
            'pf_motion_amount': 0.0,
            'pf_motion_std': 0.0,
            'pf_horse_confidence_avg': 0.0,
            'pf_detection_rate': 0.0,
        }

    # Motion calc: distance between consecutive centers
    if len(bbox_centers) > 1:
        diffs = [
            ((bbox_centers[i][0] - bbox_centers[i-1][0]) ** 2 +
             (bbox_centers[i][1] - bbox_centers[i-1][1]) ** 2) ** 0.5
            for i in range(1, len(bbox_centers))
        ]
        motion_amount = float(np.mean(diffs))
        motion_std = float(np.std(diffs))
    else:
        motion_amount = 0.0
        motion_std = 0.0

    detection_rate = sum(1 for c in bbox_counts if c > 0) / len(bbox_counts)

    return {
        'pf_frames_count': len(frames),
        'pf_bbox_count_avg': float(np.mean(bbox_counts)),
        'pf_bbox_size_avg': float(np.mean(bbox_sizes)),
        'pf_bbox_aspect_avg': float(np.mean(bbox_aspects)),
        'pf_motion_amount': motion_amount,
        'pf_motion_std': motion_std,
        'pf_horse_confidence_avg': float(np.mean(confidences)),
        'pf_detection_rate': float(detection_rate),
        'pf_person_count_avg': float(np.mean(person_counts)) if person_counts else 0,
    }


def parse_dir_name(name: str) -> tuple:
    """'202604010302_2023101087' → ('202604010302', '2023101087')"""
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return name, ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-dirs', type=int, default=None)
    ap.add_argument('--model', default='yolov8s.pt')
    ap.add_argument('--output', default=str(OUT_DIR / 'paddock_features_all.csv'))
    args = ap.parse_args()

    print(f'=== YOLOv8 paddock inference ===')
    print(f'model: {args.model}')

    dirs = sorted([d for d in PADDOCK_DIR.iterdir() if d.is_dir()])
    if args.max_dirs:
        dirs = dirs[:args.max_dirs]
    print(f'dirs to process: {len(dirs)}')

    print('Loading YOLOv8 model ...')
    model = YOLO(args.model)
    print('Model loaded.')

    rows = []
    for i, d in enumerate(dirs):
        race_id, horse_id = parse_dir_name(d.name)
        if not race_id or not horse_id:
            continue
        feats = analyze_dir(model, d)
        if not feats:
            continue
        feats['race_id'] = race_id
        feats['horse_id'] = horse_id
        rows.append(feats)
        if (i + 1) % 10 == 0:
            print(f'  processed {i+1}/{len(dirs)} dirs')

    if not rows:
        print('No frames processed.')
        return

    df = pd.DataFrame(rows)
    # Reorder
    cols = ['race_id', 'horse_id'] + [c for c in df.columns if c not in ('race_id', 'horse_id')]
    df = df[cols]

    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f'\nsaved: {args.output}')
    print(f'rows: {len(df)}')
    print(f'\nstats:')
    print(df.describe()[['pf_bbox_count_avg', 'pf_bbox_size_avg',
                          'pf_motion_amount', 'pf_horse_confidence_avg',
                          'pf_detection_rate']])

    # Detection rate distribution
    print(f'\ndetection_rate (frame で 馬検出 された 率):')
    print(f'  >=90%: {(df["pf_detection_rate"] >= 0.9).sum()} dirs')
    print(f'  >=70%: {(df["pf_detection_rate"] >= 0.7).sum()} dirs')
    print(f'  >=50%: {(df["pf_detection_rate"] >= 0.5).sum()} dirs')
    print(f'  <50%:  {(df["pf_detection_rate"] < 0.5).sum()} dirs')


if __name__ == '__main__':
    main()
