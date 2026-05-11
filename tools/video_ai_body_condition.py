#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C6: Body condition score from paddock frames (CNN embedding + 色/光沢統計).

paddock frame で 馬 bbox を crop → 色 / 光沢 / 体型統計 から condition proxy feature 化。
重量 CNN は heavy なので PoC では OpenCV + numpy 統計で 軽量実装。

【出力 features (8 個)】
- coat_brightness: 毛色 明度 (馬体の良さ proxy、 健康な馬は艶あり)
- coat_saturation: 色 鮮やかさ
- coat_contrast: コントラスト (光沢)
- body_aspect: 馬体 縦横比
- body_compactness: 馬体 充足度
- color_dominance_r/g/b: RGB 主要色
- condition_score: 上記の重み和 (0-1)

【V15 投資保護】 新規 features 算出 のみ、 V15 model 不変。

Usage:
    python tools/video_ai_body_condition.py data/paddock_frames/202603010112_2022106229/
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


def extract_horse_features(img_path, bbox):
    """1 frame の bbox 領域から 色 / 光沢 / 体型 features 抽出."""
    import numpy as np
    try:
        from PIL import Image
    except ImportError:
        print('[ERROR] Pillow not installed: pip install Pillow')
        return None

    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    crop = img.crop((x1, y1, x2, y2))
    arr = np.asarray(crop, dtype=np.float32) / 255.0

    # RGB stats (float() で numpy float32 → python float に変換、 JSON serializable)
    r_mean = float(arr[..., 0].mean())
    g_mean = float(arr[..., 1].mean())
    b_mean = float(arr[..., 2].mean())
    brightness = (r_mean + g_mean + b_mean) / 3
    # saturation (HSV S 簡易 = max-min)
    max_c = arr.max(axis=-1)
    min_c = arr.min(axis=-1)
    saturation = float((max_c - min_c).mean())
    # contrast = std of grayscale
    gray = arr.mean(axis=-1)
    contrast = float(gray.std())

    body_w = x2 - x1
    body_h = y2 - y1
    body_aspect = body_w / max(1, body_h)
    body_area = body_w * body_h
    body_compactness = body_area / max(1, w * h)

    # condition score (heuristic): bright + 高 contrast + saturation 適度
    condition_score = (
        0.35 * min(1.0, brightness / 0.5)        # 適度な明るさ
        + 0.30 * min(1.0, contrast / 0.3)         # 高 contrast = 光沢
        + 0.20 * min(1.0, saturation / 0.4)       # 程よい彩度
        + 0.15 * min(1.0, body_compactness * 10)  # frame 内 馬体率
    )

    return {
        'coat_brightness': round(brightness, 4),
        'coat_saturation': round(saturation, 4),
        'coat_contrast': round(contrast, 4),
        'body_aspect': round(body_aspect, 4),
        'body_compactness': round(body_compactness, 4),
        'color_r': round(r_mean, 4),
        'color_g': round(g_mean, 4),
        'color_b': round(b_mean, 4),
        'condition_score': round(condition_score, 4),
    }


def main():
    ap = argparse.ArgumentParser(description='Body condition score from paddock frames (C6)')
    ap.add_argument('frame_dir', help='paddock frame dir')
    ap.add_argument('--target-idx', dest='target_idx', type=int, default=0,
                    help='multi-horse の場合 cx ソートで n 番目馬')
    args = ap.parse_args()

    yolov8_path = os.path.join(BASE_DIR, 'data', 'video_ai_features',
                                os.path.basename(os.path.normpath(args.frame_dir)),
                                'yolov8_features.json')
    if not os.path.exists(yolov8_path):
        print(f'[ERROR] YOLOv8 features not found: {yolov8_path}')
        print('  -> python tools/video_ai_yolov8.py [frame_dir] を先に実行')
        return 1

    with open(yolov8_path, 'r', encoding='utf-8') as f:
        yolo_data = json.load(f)

    per_frame = yolo_data.get('per_frame', [])
    if not per_frame:
        print('[ERROR] empty per_frame in YOLOv8')
        return 1

    # frame ごとに 対象馬 bbox crop → features
    all_feats = []
    for f_info in per_frame:
        horses = f_info.get('horses', [])
        if not horses:
            continue
        if args.target_idx == 0:
            target = max(horses, key=lambda x: x['area'])
        else:
            sorted_h = sorted(horses, key=lambda x: x['cx'])
            target = sorted_h[args.target_idx if args.target_idx < len(sorted_h) else 0]

        img_path = os.path.join(args.frame_dir, f_info['file'])
        if not os.path.exists(img_path):
            continue
        bbox = target['bbox']
        feats = extract_horse_features(img_path, bbox)
        if feats:
            feats['frame_idx'] = f_info['idx']
            all_feats.append(feats)

    if not all_feats:
        print('[ERROR] no features extracted')
        return 1

    # aggregate across frames
    keys = [k for k in all_feats[0].keys() if k != 'frame_idx']
    agg = {}
    for k in keys:
        vals = [f[k] for f in all_feats]
        agg[f'{k}_mean'] = round(sum(vals) / len(vals), 4)
        if len(vals) > 1:
            mu = sum(vals) / len(vals)
            agg[f'{k}_std'] = round((sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5, 4)

    out_dir = os.path.dirname(yolov8_path)
    out_path = os.path.join(out_dir, 'body_condition_features.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'aggregated': agg,
            'per_frame': all_feats,
            'computed_at': datetime.now().isoformat(),
            'target_idx': args.target_idx,
        }, f, indent=2, ensure_ascii=False)

    print(f'[OK] body condition features extracted ({len(all_feats)} frames)')
    print(f'[OK] saved: {out_path}')
    print('\n[Aggregated]')
    for k, v in agg.items():
        print(f'  {k}: {v}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
