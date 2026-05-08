"""YOLOv8 馬体検出 module (Session #48 C、 dev/video-poc).

Session #42 E で動作確認済 (138 ms CPU)。 本 module は pipeline 統合用 wrapper。

usage:
  from tools.video_pipeline.yolo_inference import detect_in_video, detect_in_image
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")


def detect_in_image(img_path: Path, conf_threshold: float = 0.25) -> dict:
    """単一画像で YOLOv8 馬体検出. Session #42 E flow."""
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError as e:
        return {"status": "ImportError", "error": str(e)}

    model = YOLO("yolov8n.pt")
    t0 = time.time()
    results = model(str(img_path), conf=conf_threshold, verbose=False, device="cpu")
    inference_ms = (time.time() - t0) * 1000

    horses = []
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            cls = int(box.cls[0]) if box.cls is not None else -1
            if cls != 17: continue  # COCO horse class
            conf = float(box.conf[0]) if box.conf is not None else 0
            xyxy = box.xyxy[0].tolist() if box.xyxy is not None else [0,0,0,0]
            horses.append({
                "confidence": round(conf, 4),
                "bbox": [round(x, 1) for x in xyxy],
            })

    return {
        "status": "ok",
        "inference_ms": round(inference_ms, 1),
        "n_horses": len(horses),
        "horses": horses,
    }


def detect_in_video(video_path: Path, fps: int = 5, max_frames: int = 50,
                    conf_threshold: float = 0.25) -> dict:
    """動画で YOLOv8 馬体検出 (frame 抽出 + 全 frame inference).

    Session #43 D の tools/video_poc/extract_frames_and_detect.py と同様。
    """
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        return {"status": "ImportError", "error": str(e)}

    if not video_path.exists():
        return {"status": "missing", "path": str(video_path)}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "open_fail"}

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(src_fps / fps))

    model = YOLO("yolov8n.pt")
    frame_results = []
    i = 0
    extracted = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if i % interval == 0:
            t0 = time.time()
            r = model(frame, conf=conf_threshold, verbose=False, device="cpu")
            inference_ms = (time.time() - t0) * 1000

            horses_in_frame = []
            for res in r:
                if res.boxes is None: continue
                for box in res.boxes:
                    cls = int(box.cls[0]) if box.cls is not None else -1
                    if cls != 17: continue
                    conf = float(box.conf[0]) if box.conf is not None else 0
                    xyxy = box.xyxy[0].tolist() if box.xyxy is not None else [0,0,0,0]
                    horses_in_frame.append({
                        "confidence": round(conf, 4),
                        "bbox": [round(x, 1) for x in xyxy],
                    })

            frame_results.append({
                "frame_idx": extracted,
                "n_horses": len(horses_in_frame),
                "horses": horses_in_frame,
                "inference_ms": round(inference_ms, 1),
            })
            extracted += 1
            if extracted >= max_frames: break
        i += 1
    cap.release()

    n_with_horse = sum(1 for fr in frame_results if fr["n_horses"] > 0)
    return {
        "status": "ok",
        "video": video_path.name,
        "n_frames_extracted": extracted,
        "n_frames_with_horse": n_with_horse,
        "horse_detection_rate_pct": round(n_with_horse / extracted * 100, 2) if extracted > 0 else 0,
        "frames": frame_results[:5],  # サンプル top 5
    }


def cli():
    p = argparse.ArgumentParser(description="YOLOv8 inference (Session #48 C)")
    p.add_argument("--image", default=None)
    p.add_argument("--video", default=None)
    args = p.parse_args()

    if args.image:
        r = detect_in_image(Path(args.image))
    elif args.video:
        r = detect_in_video(Path(args.video))
    else:
        print("[!] --image or --video 指定必須")
        return

    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
