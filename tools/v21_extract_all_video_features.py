#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V21 features 一括 抽出 pipeline (paddock_frames/* → unified video features CSV).

既存 data/paddock_frames/{race_id}_{horse_id}/ の 全 dir に対し:
1. YOLOv8 bbox 抽出 (yolov8_features.json)
2. gait features 計算 (gait_features.json)
3. body condition features (body_condition_features.json)
4. 全 horse_id × race_id の features を 1 CSV に集約

【V15 投資保護】 read-only 抽出のみ、 V15 model 不変

Usage:
    python tools/v21_extract_all_video_features.py
    python tools/v21_extract_all_video_features.py --force  # 既存 features 再生成
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
FRAME_DIR = os.path.join(BASE_DIR, 'data', 'paddock_frames')
FEATURES_DIR = os.path.join(BASE_DIR, 'data', 'video_ai_features')
OUT_CSV = os.path.join(BASE_DIR, 'data', 'v21_video_features_all.csv')

GAIT_KEYS = [
    'coverage', 'aspect_mean', 'aspect_std', 'aspect_range',
    'area_mean', 'area_std', 'w_mean', 'h_mean',
    'conf_mean', 'conf_std', 'conf_min',
    'motion_dcx_mean', 'motion_dcy_mean',
    'motion_speed_mean', 'motion_speed_std', 'motion_speed_max',
    'area_change_mean', 'aspect_change_mean',
]

BODY_KEYS = [
    'coat_brightness_mean', 'coat_brightness_std',
    'coat_saturation_mean', 'coat_saturation_std',
    'coat_contrast_mean', 'coat_contrast_std',
    'body_aspect_mean', 'body_aspect_std',
    'body_compactness_mean', 'body_compactness_std',
    'color_r_mean', 'color_g_mean', 'color_b_mean',
    'condition_score_mean', 'condition_score_std',
]


def list_paddock_dirs():
    if not os.path.exists(FRAME_DIR):
        return []
    return [d for d in os.listdir(FRAME_DIR)
            if os.path.isdir(os.path.join(FRAME_DIR, d)) and '_' in d]


def parse_dir_name(name):
    """race_id_horse_id format."""
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[0], '_'.join(parts[1:])
    return None, None


def ensure_yolo(dir_name):
    """YOLOv8 features 未生成なら 生成."""
    feat_path = os.path.join(FEATURES_DIR, dir_name, 'yolov8_features.json')
    if os.path.exists(feat_path):
        return True
    frames_dir = os.path.join(FRAME_DIR, dir_name)
    cmd = [sys.executable, os.path.join(BASE_DIR, 'tools', 'video_ai_yolov8.py'),
           frames_dir, '--conf', '0.2']
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception:
        pass
    return os.path.exists(feat_path)


def ensure_gait(dir_name):
    feat_path = os.path.join(FEATURES_DIR, dir_name, 'gait_features.json')
    if os.path.exists(feat_path):
        return True
    feats_dir = os.path.join(FEATURES_DIR, dir_name)
    cmd = [sys.executable, os.path.join(BASE_DIR, 'tools', 'video_ai_gait_features.py'),
           feats_dir]
    try:
        subprocess.run(cmd, capture_output=True, timeout=60)
    except Exception:
        pass
    return os.path.exists(feat_path)


def ensure_body(dir_name):
    feat_path = os.path.join(FEATURES_DIR, dir_name, 'body_condition_features.json')
    if os.path.exists(feat_path):
        return True
    frames_dir = os.path.join(FRAME_DIR, dir_name)
    cmd = [sys.executable, os.path.join(BASE_DIR, 'tools', 'video_ai_body_condition.py'),
           frames_dir]
    try:
        subprocess.run(cmd, capture_output=True, timeout=60)
    except Exception:
        pass
    return os.path.exists(feat_path)


def load_features(dir_name):
    """gait + body features を1 dict にまとめる."""
    feats = {}
    gait_path = os.path.join(FEATURES_DIR, dir_name, 'gait_features.json')
    if os.path.exists(gait_path):
        try:
            d = json.load(open(gait_path, 'r', encoding='utf-8')).get('features', {})
            for k in GAIT_KEYS:
                if k in d:
                    feats[f'video_gait_{k}'] = d[k]
        except Exception:
            pass

    body_path = os.path.join(FEATURES_DIR, dir_name, 'body_condition_features.json')
    if os.path.exists(body_path):
        try:
            d = json.load(open(body_path, 'r', encoding='utf-8')).get('aggregated', {})
            for k in BODY_KEYS:
                if k in d:
                    feats[f'video_body_{k}'] = d[k]
        except Exception:
            pass
    return feats


def main():
    ap = argparse.ArgumentParser(description='V21 video features extraction')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    dirs = list_paddock_dirs()
    print(f'[INFO] {len(dirs)} paddock dirs found')

    if not dirs:
        print('[ERROR] no paddock dirs')
        return 1

    rows = []
    for i, d in enumerate(dirs, 1):
        race_id, horse_id = parse_dir_name(d)
        if not race_id or not horse_id:
            continue

        # ensure all features generated
        ensure_yolo(d)
        ensure_gait(d)
        ensure_body(d)

        feats = load_features(d)
        if not feats:
            print(f'  [SKIP] {d}: no features')
            continue

        row = {'race_id': race_id, 'horse_id': horse_id, **feats}
        rows.append(row)

        if i % 10 == 0 or i == len(dirs):
            print(f'  [{i}/{len(dirs)}] processed {len(rows)} dirs')

    if not rows:
        print('[ERROR] no features extracted')
        return 1

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\n[OK] saved: {OUT_CSV}')
    print(f'[OK] shape: {df.shape}')

    n_features = [c for c in df.columns if c.startswith('video_')]
    print(f'[OK] video features: {len(n_features)}')
    print(f'[OK] coverage:')
    for c in n_features[:5]:
        rate = df[c].notna().mean()
        print(f'  {c}: {rate*100:.1f}%')
    return 0


if __name__ == '__main__':
    sys.exit(main())
