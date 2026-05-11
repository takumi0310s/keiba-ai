#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W1.4: YOLOv8 馬 bbox 検出 PoC.

paddock / race / oikiri などの frame 画像から馬 (COCO class id=17 "horse") を検出し、
bbox 座標 + confidence + frame 内 馬数 を抽出。 V21 video features の基盤。

【規約】 input frame は私的複製範囲、 output features は数値のみで配布可。

Usage:
    python tools/video_ai_yolov8.py data/paddock_frames/202603010112_2022106229/
    python tools/video_ai_yolov8.py data/race_video_frames/202603010112_0/ --conf 0.4
    python tools/video_ai_yolov8.py path/to/frame.jpg --save-annotated  # 描画付き出力

Output:
    data/video_ai_features/{frame_dir_name}/yolov8_features.json
        per-frame: [{idx, n_horses, horses: [{bbox, conf, w, h, cx, cy}]}]
    data/video_ai_features/{frame_dir_name}/yolov8_summary.json
        aggregated stats (avg horse count, avg conf, etc)
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
OUT_BASE = os.path.join(BASE_DIR, 'data', 'video_ai_features')
MODEL_PATH = os.path.join(BASE_DIR, 'yolov8n.pt')

COCO_HORSE_ID = 17  # COCO class index for 'horse'


def list_frames(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        return []
    return sorted([os.path.join(input_path, f) for f in os.listdir(input_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])


def detect(frames, conf_threshold=0.3, save_annotated=False, out_dir=None):
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    print(f'[INFO] YOLO model loaded: {MODEL_PATH}')
    print(f'[INFO] processing {len(frames)} frame(s), conf>={conf_threshold}')

    per_frame = []
    all_h_count = []
    all_h_conf = []
    all_h_areas = []

    for i, fp in enumerate(frames):
        try:
            results = model(fp, conf=conf_threshold, verbose=False, classes=[COCO_HORSE_ID])
        except Exception as e:
            print(f'[WARN] {fp}: {e}')
            per_frame.append({'idx': i, 'file': os.path.basename(fp), 'error': str(e)})
            continue

        r = results[0]
        horses = []
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if cls != COCO_HORSE_ID:
                    continue
                x1, y1, x2, y2 = xyxy
                w = x2 - x1
                h = y2 - y1
                horses.append({
                    'bbox': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    'conf': round(conf, 3),
                    'w': round(w, 1),
                    'h': round(h, 1),
                    'cx': round((x1 + x2) / 2, 1),
                    'cy': round((y1 + y2) / 2, 1),
                    'area': round(w * h, 1),
                })

        n = len(horses)
        all_h_count.append(n)
        for h in horses:
            all_h_conf.append(h['conf'])
            all_h_areas.append(h['area'])

        per_frame.append({
            'idx': i,
            'file': os.path.basename(fp),
            'n_horses': n,
            'horses': horses,
        })

        if save_annotated and out_dir and n > 0:
            annotated = r.plot()
            try:
                import cv2
                ann_fp = os.path.join(out_dir, f'ann_{os.path.basename(fp)}')
                cv2.imwrite(ann_fp, annotated)
            except ImportError:
                pass

    summary = {
        'n_frames': len(frames),
        'n_frames_with_horse': sum(1 for c in all_h_count if c > 0),
        'avg_horse_count': round(sum(all_h_count) / max(1, len(all_h_count)), 2),
        'max_horse_count': max(all_h_count) if all_h_count else 0,
        'avg_conf': round(sum(all_h_conf) / max(1, len(all_h_conf)), 3) if all_h_conf else 0,
        'avg_area': round(sum(all_h_areas) / max(1, len(all_h_areas)), 1) if all_h_areas else 0,
    }
    return per_frame, summary


def main():
    ap = argparse.ArgumentParser(description='YOLOv8 馬 bbox 検出 PoC')
    ap.add_argument('input', help='frame ファイル または ディレクトリ')
    ap.add_argument('--conf', type=float, default=0.3, help='conf threshold (default 0.3)')
    ap.add_argument('--save-annotated', dest='save_annotated', action='store_true',
                    help='bbox 描画版を ann_*.jpg として保存')
    ap.add_argument('--out-name', dest='out_name', default=None,
                    help='出力 dir 名 (default: 入力 dir basename)')
    args = ap.parse_args()

    frames = list_frames(args.input)
    if not frames:
        print(f'[ERROR] no frames in: {args.input}')
        return 1

    out_name = args.out_name or os.path.basename(os.path.normpath(args.input))
    out_dir = os.path.join(OUT_BASE, out_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f'[INFO] out_dir: {out_dir}')
    per_frame, summary = detect(frames, conf_threshold=args.conf,
                                save_annotated=args.save_annotated, out_dir=out_dir)

    with open(os.path.join(out_dir, 'yolov8_features.json'), 'w', encoding='utf-8') as f:
        json.dump({'per_frame': per_frame, 'detected_at': datetime.now().isoformat()},
                  f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, 'yolov8_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'[OK] frames={summary["n_frames"]}, '
          f'with_horse={summary["n_frames_with_horse"]}, '
          f'avg_h={summary["avg_horse_count"]}, '
          f'avg_conf={summary["avg_conf"]}, '
          f'avg_area={summary["avg_area"]}')
    print(f'[OK] saved: {out_dir}/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
