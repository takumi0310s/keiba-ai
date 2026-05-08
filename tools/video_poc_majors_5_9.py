"""5/9 重賞 動画 PoC 実行 (Session #49 B、 dev/training-poc).

5/9 重賞 3R の 動画 PoC:
- 京都 11R 京都新聞杯 (G2)
- 東京 11R エプソムカップ (G3)
- 新潟 11R 駿風 S (OP)

実 動画 download は ユーザー manual (netkeiba Premium login + Cookie 必要)。
本 tool は:
1. data/v18/videos_5_9/ 配下に動画があれば YOLOv8 inference
2. 動画なし馬は静止画 fallback (sample_horse.jpg 等で simulate)
3. features 集計 + JSON 出力

usage:
  # 全動画 batch run
  python tools/video_poc_majors_5_9.py --all

  # 単一動画
  python tools/video_poc_majors_5_9.py --video data/v18/videos_5_9/keishin_horse1.mp4

  # PoC dry-run (sample image で simulate)
  python tools/video_poc_majors_5_9.py --simulate

V15 production 完全独立、 dev/training-poc 専用。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
# 既存 main の tools/video_poc/extract_frames_and_detect.py を流用 (Session #43 D)
# dev/video-poc の tools/video_pipeline/ は別 branch、 self-contained で実装

VIDEO_DIR = BASE / "data" / "v18" / "videos_5_9"
FEATURES_OUT = BASE / "data" / "v18" / "video_features"


def get_target_horses() -> dict:
    """5/9 重賞 注目馬 list (placeholder、 5/9 朝 daily_predict 後 確定)."""
    return {
        "京都新聞杯_G2": {
            "race_id_pattern": "20260508*11",  # Kyoto 11R
            "noted_horses": [
                # 5/9 朝確定後に top10 馬を入れる
                # 現在は placeholder
                {"umaban": "?", "horse_name": "(5/9 朝確定)"}
            ],
        },
        "エプソムC_G3": {
            "race_id_pattern": "20260505*11",  # Tokyo 11R
            "noted_horses": [{"umaban": "?", "horse_name": "(5/9 朝確定)"}],
        },
        "駿風_S_OP": {
            "race_id_pattern": "20260504*11",  # Niigata 11R
            "noted_horses": [{"umaban": "?", "horse_name": "(5/9 朝確定)"}],
        },
    }


def process_video_or_image(path: Path) -> dict:
    """動画 or 静止画 で YOLOv8 + features 抽出 (self-contained)."""
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError as e:
        return {"status": "ImportError", "error": str(e)[:80]}

    if not path.exists():
        return {"status": "missing", "path": str(path)}

    model = YOLO("yolov8n.pt")
    suffix = path.suffix.lower()

    horses_all = []
    if suffix in (".jpg", ".jpeg", ".png"):
        # 静止画
        results = model(str(path), conf=0.25, verbose=False, device="cpu")
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                cls = int(box.cls[0]) if box.cls is not None else -1
                if cls != 17: continue
                conf = float(box.conf[0]) if box.conf is not None else 0
                xyxy = box.xyxy[0].tolist() if box.xyxy is not None else [0,0,0,0]
                horses_all.append({"conf": round(conf, 4), "bbox": [round(x, 1) for x in xyxy]})
    elif suffix in (".mp4", ".mov", ".avi"):
        # 動画 5 fps、 max 50 frames
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {"status": "open_fail"}
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(src_fps / 5))
        i = 0
        extracted = 0
        while extracted < 50:
            ret, frame = cap.read()
            if not ret: break
            if i % interval == 0:
                r = model(frame, conf=0.25, verbose=False, device="cpu")
                for res in r:
                    if res.boxes is None: continue
                    for box in res.boxes:
                        cls = int(box.cls[0]) if box.cls is not None else -1
                        if cls != 17: continue
                        conf = float(box.conf[0]) if box.conf is not None else 0
                        xyxy = box.xyxy[0].tolist() if box.xyxy is not None else [0,0,0,0]
                        horses_all.append({"conf": round(conf, 4), "bbox": [round(x, 1) for x in xyxy], "frame": extracted})
                extracted += 1
            i += 1
        cap.release()
    else:
        return {"status": "unsupported", "path": str(path)}

    # features 集計
    if not horses_all:
        return {
            "status": "ok",
            "n_horses_detected": 0,
            "features": {
                "video_horse_detected": 0,
                "video_max_conf": 0.0,
                "video_avg_size_score": 0.0,
                "video_aspect_ratio": 0.0,
            },
        }

    import numpy as np
    confs = [h["conf"] for h in horses_all]
    sizes = []
    aspect_ratios = []
    for h in horses_all:
        bbox = h["bbox"]
        w = bbox[2] - bbox[0]
        ht = bbox[3] - bbox[1]
        if w > 0 and ht > 0:
            sizes.append(w * ht)
            aspect_ratios.append(w / ht)

    return {
        "status": "ok",
        "n_horses_detected": len(horses_all),
        "features": {
            "video_horse_detected": 1,
            "video_max_conf": round(max(confs), 4),
            "video_avg_size_score": round(min(1.0, np.mean(sizes) / 100000), 4) if sizes else 0,
            "video_aspect_ratio": round(np.mean(aspect_ratios), 3) if aspect_ratios else 0,
        },
    }


def run_simulation() -> dict:
    """sample image で PoC simulate."""
    print("=" * 60)
    print("Video PoC simulation (sample images)")
    print("=" * 60)

    samples = [
        BASE / "data" / "video_poc" / "zidane.jpg",
        BASE / "data" / "video_poc" / "bus.jpg",
        BASE / "data" / "video_poc" / "sample_horse.jpg",
    ]
    results = []
    for p in samples:
        if not p.exists(): continue
        print(f"\n  processing: {p.name}")
        r = process_video_or_image(p)
        r["file"] = p.name
        results.append(r)
        print(f"    features: {r.get('features')}")
    return {"mode": "simulate", "n_processed": len(results), "results": results}


def run_majors_batch() -> dict:
    """data/v18/videos_5_9/ 配下の全動画/画像を 一括処理."""
    print("=" * 60)
    print("Video PoC majors 5/9 batch")
    print("=" * 60)

    if not VIDEO_DIR.exists():
        print(f"  [WARN] {VIDEO_DIR} 不在")
        return {"mode": "batch", "status": "no_videos", "note": "ユーザー manual DL を 5/9 朝に実行"}

    files = sorted(VIDEO_DIR.iterdir())
    if not files:
        print(f"  [WARN] 動画 0、 simulate mode に切替")
        return run_simulation()

    results = []
    for f in files:
        if f.is_dir(): continue
        print(f"\n  processing: {f.name}")
        r = process_video_or_image(f)
        r["file"] = f.name
        results.append(r)
        print(f"    features: {r.get('features')}")

    return {"mode": "batch", "n_processed": len(results), "results": results}


def main():
    p = argparse.ArgumentParser(description="5/9 重賞 動画 PoC (Session #49 B)")
    p.add_argument("--all", action="store_true", help="data/v18/videos_5_9/ 配下 全 batch")
    p.add_argument("--video", default=None)
    p.add_argument("--simulate", action="store_true", help="sample image で simulate")
    p.add_argument("--out", default="data/v18/video_features_5_9_majors.json")
    args = p.parse_args()

    if args.simulate:
        result = run_simulation()
    elif args.all or args.video is None:
        result = run_majors_batch()
    else:
        path = Path(args.video)
        if not path.is_absolute():
            path = BASE / args.video
        result = {"mode": "single", "results": [process_video_or_image(path)]}

    out_path = BASE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")

    # target horses meta
    targets = get_target_horses()
    targets_path = BASE / "data" / "v18" / "video_features" / "5_9_majors_target_horses.json"
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    targets_path.write_text(json.dumps(targets, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  target horses placeholder: {targets_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
