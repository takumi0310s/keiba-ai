"""動画 → frame 抽出 → YOLOv8 検出 (Session #43 D).

Phase 4 PoC の拡張: 実 sample 動画から frame 抽出し、 YOLOv8 で各 frame の
馬体検出を試行。 検出率 / 平均 confidence / inference 速度 を測定。

input:
- 動画 file (mp4/mov、 ユーザー manual 配置 or YouTube 等から DL)

output:
- data/video_poc/frames/<video_id>/ (frame 静止画)
- data/video_poc/detections/<video_id>.json (各 frame の検出結果)
- data/video_poc/<video_id>_summary.json (集計)

usage:
  # 実 sample 動画
  python tools/video_poc/extract_frames_and_detect.py --video data/video_poc/sample_race.mp4

  # 静止画 sample で 検出のみ
  python tools/video_poc/extract_frames_and_detect.py --image data/video_poc/sample_horse.jpg

V15 production 完全不変 (新規 PoC)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")


def extract_frames(video_path: Path, out_dir: Path, fps: int = 5, max_frames: int = 100) -> list[Path]:
    """ffmpeg or OpenCV で frame 抽出."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
    except ImportError:
        print(f"[ERROR] opencv-python 未 install")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] failed to open {video_path}")
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  source video: {src_frames} frames @ {src_fps:.2f} fps")
    print(f"  target sampling: {fps} fps, max_frames={max_frames}")

    interval = max(1, int(src_fps / fps))
    extracted = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if i % interval == 0:
            out_path = out_dir / f"frame_{len(extracted):04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            extracted.append(out_path)
            if len(extracted) >= max_frames:
                break
        i += 1
    cap.release()
    print(f"  extracted: {len(extracted)} frames -> {out_dir}")
    return extracted


def detect_horses(frames: list[Path], conf_threshold: float = 0.25) -> list[dict]:
    """各 frame で YOLOv8 horse detection."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print(f"[ERROR] ultralytics 未 install")
        return []

    model = YOLO('yolov8n.pt')
    print(f"  model loaded")

    results = []
    t0 = time.time()
    for frame_path in frames:
        t1 = time.time()
        try:
            r = model(str(frame_path), conf=conf_threshold, verbose=False, device='cpu')
            inference_ms = (time.time() - t1) * 1000

            detections = []
            for res in r:
                if res.boxes is None: continue
                for box in res.boxes:
                    cls = int(box.cls[0]) if box.cls is not None else -1
                    conf = float(box.conf[0]) if box.conf is not None else 0
                    is_horse = cls == 17  # COCO horse class
                    detections.append({
                        "class_id": cls,
                        "is_horse": is_horse,
                        "confidence": round(conf, 4),
                    })
            horses = [d for d in detections if d["is_horse"]]
            results.append({
                "frame": frame_path.name,
                "inference_ms": round(inference_ms, 1),
                "n_detections": len(detections),
                "n_horses": len(horses),
                "max_horse_conf": max([h["confidence"] for h in horses], default=0),
            })
        except Exception as e:
            results.append({"frame": frame_path.name, "error": str(e)[:120]})

    elapsed = time.time() - t0
    print(f"  total inference: {elapsed:.1f}s, {len(frames)} frames, avg {elapsed/len(frames)*1000:.1f} ms/frame")
    return results


def summarize(results: list[dict]) -> dict:
    n_total = len(results)
    n_with_horse = sum(1 for r in results if r.get("n_horses", 0) > 0)
    confs = [r["max_horse_conf"] for r in results if r.get("n_horses", 0) > 0]
    return {
        "n_frames": n_total,
        "n_with_horse_detected": n_with_horse,
        "horse_detection_rate_pct": round(n_with_horse / n_total * 100, 2) if n_total > 0 else 0,
        "mean_max_horse_conf": round(sum(confs) / len(confs), 4) if confs else 0,
        "max_horse_conf": round(max(confs), 4) if confs else 0,
    }


def main():
    p = argparse.ArgumentParser(description="動画 → frame 抽出 → YOLOv8 検出 (Session #43 D)")
    p.add_argument("--video", default=None, help="動画 file path")
    p.add_argument("--image", default=None, help="静止画 file path (検出のみ、 frame 抽出 skip)")
    p.add_argument("--fps", type=int, default=5, help="抽出 fps")
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out-dir", default="data/video_poc")
    p.add_argument("--video-id", default=None, help="出力 dir 名 (default: video filename)")
    args = p.parse_args()

    out_root = BASE / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    if args.image:
        # 静止画 1 枚で検出
        print("=" * 60)
        print(f"YOLOv8 検出 (静止画 1 枚): {args.image}")
        print("=" * 60)
        img_path = Path(args.image)
        if not img_path.is_absolute():
            img_path = BASE / args.image
        results = detect_horses([img_path], conf_threshold=args.conf)
        summary = summarize(results)
    elif args.video:
        # 動画 → frame 抽出 → 検出
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = BASE / args.video
        if not video_path.exists():
            print(f"[ERROR] video not found: {video_path}")
            sys.exit(1)

        video_id = args.video_id or video_path.stem
        frames_dir = out_root / "frames" / video_id

        print("=" * 60)
        print(f"動画 → frame → YOLOv8 検出: {video_path.name}")
        print("=" * 60)
        frames = extract_frames(video_path, frames_dir, fps=args.fps, max_frames=args.max_frames)
        if not frames:
            print(f"[ERROR] no frames extracted")
            sys.exit(2)
        results = detect_horses(frames, conf_threshold=args.conf)
        summary = summarize(results)

        det_path = out_root / "detections" / f"{video_id}.json"
        det_path.parent.mkdir(parents=True, exist_ok=True)
        det_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  detections: {det_path.relative_to(BASE)}")
    else:
        print("[!] --video or --image を指定してください")
        sys.exit(1)

    print(f"\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    summary_path = out_root / f"summary_{args.video_id or (Path(args.video or args.image).stem)}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  summary: {summary_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
