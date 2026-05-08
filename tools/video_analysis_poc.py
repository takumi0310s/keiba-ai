"""Session #47 E: 動画解析 PoC (任意、 13:00 以降 動画公開).

5/9 (土) 重賞 (G1/G2/G3) のみ 動画解析 PoC。
今回 target: 京都新聞杯 (G2) の有力馬 3-5 頭。

PoC level (動作確認のみ):
- YOLOv8 で 馬体検出
- 簡易 features (stride length / 体格 / pose)

注意:
- 動画は 1 ファイル数十 MB、 storage 数百 MB に達する可能性
- 5/8 13:00 まで公開待ち (今は skeleton)
- 失敗しても他に影響なし (学習目的のみ)

Usage:
  python tools/video_analysis_poc.py --race-id 202608030611
  python tools/video_analysis_poc.py --check-deps  # ultralytics / opencv check のみ
"""
import os
import sys
import json
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V18_DIR = os.path.join(BASE_DIR, 'data', 'v18')
VIDEO_DIR = os.path.join(V18_DIR, 'videos')


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def check_deps():
    """ultralytics + opencv 動作 check。"""
    issues = []
    try:
        import ultralytics
        log(f"ultralytics: {ultralytics.__version__}")
    except ImportError:
        issues.append('ultralytics not installed (pip install ultralytics)')

    try:
        import cv2
        log(f"opencv: {cv2.__version__}")
    except ImportError:
        issues.append('opencv-python not installed')

    try:
        import torch
        log(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
    except ImportError:
        issues.append('torch not installed')

    if issues:
        log("Issues:")
        for i in issues:
            log(f"  - {i}")
        return False
    log("All deps OK")
    return True


def download_video(race_id, horse_id, out_path):
    """netkeiba から動画 download (placeholder)。

    実装は 5/8 13:00 動画公開後に。
    現在は skeleton のみ (URL pattern 不明、 公開後に確認)。
    """
    log(f"Download video: race={race_id} horse={horse_id} → {out_path}")
    log("  PLACEHOLDER: 5/8 13:00 公開後に URL pattern 確認 + 実装")
    return False


def detect_horse_yolov8(video_path):
    """YOLOv8 で動画 frame ごと 馬体検出 (PoC)。

    Returns: list of bbox per frame.
    """
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError:
        log("YOLO/cv2 not available, skip")
        return None

    log(f"YOLOv8 detect: {video_path}")
    model = YOLO('yolov8n.pt')  # nano = 最軽量
    results = model.predict(source=video_path, classes=[17],  # COCO horse class = 17
                            verbose=False, save=False, max_det=10)
    bboxes_per_frame = []
    for r in results:
        if r.boxes is not None:
            bboxes_per_frame.append(r.boxes.xyxy.cpu().numpy().tolist())
        else:
            bboxes_per_frame.append([])
    log(f"  {len(bboxes_per_frame)} frames processed")
    return bboxes_per_frame


def compute_simple_features(bboxes_per_frame):
    """bbox 系列 から 簡易 features 計算。

    - stride_length_proxy: bbox 中心 x 移動の周期性
    - body_size_score: bbox 面積 平均
    - pose_variability: bbox サイズ変動 (緊張度 proxy)
    """
    if not bboxes_per_frame:
        return {}

    centers_x = []
    areas = []
    for frame in bboxes_per_frame:
        if frame:
            x1, y1, x2, y2 = frame[0][:4]
            centers_x.append((x1 + x2) / 2)
            areas.append((x2 - x1) * (y2 - y1))

    if not centers_x:
        return {'note': 'no horse detected'}

    import numpy as np
    return {
        'frames_with_horse': len(centers_x),
        'body_size_score': float(np.mean(areas)),
        'pose_variability': float(np.std(areas) / (np.mean(areas) + 1e-9)),
        'stride_proxy': float(np.std(np.diff(centers_x)) if len(centers_x) > 1 else 0),
    }


def run_poc(race_id, horse_ids=None):
    log(f"=== Session #47 E: video PoC for race {race_id} ===")
    if not check_deps():
        log("deps NG, abort PoC")
        return None

    os.makedirs(VIDEO_DIR, exist_ok=True)

    if horse_ids is None:
        horse_ids = ['placeholder_1', 'placeholder_2', 'placeholder_3']

    results = {}
    for hid in horse_ids:
        out_path = os.path.join(VIDEO_DIR, f'{race_id}_{hid}.mp4')
        ok = download_video(race_id, hid, out_path)
        if not ok or not os.path.exists(out_path):
            results[hid] = {'error': 'video not available (公開待ち or scrape NG)'}
            continue

        bboxes = detect_horse_yolov8(out_path)
        if bboxes is None:
            results[hid] = {'error': 'YOLO failed'}
            continue
        feats = compute_simple_features(bboxes)
        results[hid] = feats

    out_json = os.path.join(V18_DIR, 'video_poc_5_9_majors.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'race_id': race_id,
            'horse_ids': horse_ids,
            'results': results,
            'timestamp': datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2)
    log(f"Saved: {out_json}")

    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--race-id', default='202608030611', help='京都新聞杯 race_id (placeholder)')
    p.add_argument('--horse-ids', nargs='+', default=None)
    p.add_argument('--check-deps', action='store_true')
    args = p.parse_args()

    if args.check_deps:
        check_deps()
    else:
        run_poc(args.race_id, args.horse_ids)
