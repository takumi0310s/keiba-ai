"""歩様 / 体格 features 抽出 (Session #52 B+C、 dev/training-poc).

動画 (or 静止画) → YOLOv8 馬体検出 → motion features 化:
- stride_length_mean (歩幅 推定、 bbox y 振動から)
- body_size_relative (同 R 内 比較、 placeholder では absolute)
- stability_score (bbox 中心 移動の variance)
- tension_score (推定、 静止 frame ratio から)

usage:
  # 動画 batch (data/v18/videos_5_9/ 配下)
  python tools/horse_motion_features.py --batch

  # 単一動画
  python tools/horse_motion_features.py --video data/v18/videos_5_9/race1/horse_a.mp4

V15 production 完全独立、 dev/training-poc 専用。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
VIDEO_DIR = BASE / "data" / "v18" / "videos_5_9"


def detect_bboxes_in_video(video_path: Path, max_frames: int = 30,
                            conf_threshold: float = 0.25) -> list[dict]:
    """動画 → 30 frame YOLOv8 inference → bbox 系列."""
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError:
        return []

    if not video_path.exists():
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    interval = max(1, total_frames // max_frames) if total_frames else 1

    model = YOLO("yolov8n.pt")
    bboxes_seq = []
    i = 0
    extracted = 0

    while extracted < max_frames:
        ret, frame = cap.read()
        if not ret: break
        if i % interval == 0:
            results = model(frame, conf=conf_threshold, verbose=False, device="cpu")
            for r in results:
                if r.boxes is None: continue
                for box in r.boxes:
                    cls = int(box.cls[0]) if box.cls is not None else -1
                    if cls != 17: continue
                    conf = float(box.conf[0]) if box.conf is not None else 0
                    xyxy = box.xyxy[0].tolist() if box.xyxy is not None else [0,0,0,0]
                    bboxes_seq.append({
                        "frame_idx": extracted,
                        "conf": conf,
                        "x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3],
                    })
            extracted += 1
        i += 1
    cap.release()
    return bboxes_seq


def compute_motion_features(bboxes_seq: list[dict]) -> dict:
    """bbox 時系列 → motion features 4 件."""
    if not bboxes_seq:
        return {
            "stride_length_mean": 0.0,
            "body_size_relative": 0.0,
            "stability_score": 0.0,
            "tension_score": 0.0,
            "n_bboxes": 0,
            "n_frames_with_horse": 0,
        }

    import numpy as np
    # 各 frame の代表 bbox (max conf) を取る
    by_frame = {}
    for b in bboxes_seq:
        f = b["frame_idx"]
        if f not in by_frame or b["conf"] > by_frame[f]["conf"]:
            by_frame[f] = b
    frames = sorted(by_frame.keys())

    centers_y = [(by_frame[f]["y1"] + by_frame[f]["y2"]) / 2 for f in frames]
    centers_x = [(by_frame[f]["x1"] + by_frame[f]["x2"]) / 2 for f in frames]
    sizes = [(by_frame[f]["x2"] - by_frame[f]["x1"]) * (by_frame[f]["y2"] - by_frame[f]["y1"]) for f in frames]

    # stride: y 振動 (y diff の std を 推定 stride として)
    stride = 0.0
    if len(centers_y) >= 3:
        y_diff = np.diff(centers_y)
        stride = float(np.std(y_diff))

    # body_size_relative (placeholder: absolute mean、 同 R 内 normalize は別 step)
    body_size = float(np.mean(sizes)) if sizes else 0.0
    body_size_rel = min(1.0, body_size / 100000)

    # stability: x/y 中心 std の逆数 (小 std = 安定)
    stab = 0.0
    if len(centers_x) >= 2:
        std_x = float(np.std(centers_x))
        std_y = float(np.std(centers_y))
        stab = round(1 / (1 + std_x / 100 + std_y / 100), 4)

    # tension: 静止 frame ratio (連続 frame で bbox 変化少 = 静止)
    tension = 0.0
    if len(centers_x) >= 2:
        small_moves = sum(1 for i in range(1, len(centers_x))
                          if abs(centers_x[i] - centers_x[i-1]) < 5
                          and abs(centers_y[i] - centers_y[i-1]) < 5)
        tension = round(small_moves / max(1, len(centers_x) - 1), 4)

    return {
        "stride_length_mean": round(stride, 2),
        "body_size_relative": round(body_size_rel, 4),
        "stability_score": round(stab, 4),
        "tension_score": round(tension, 4),
        "n_bboxes": len(bboxes_seq),
        "n_frames_with_horse": len(frames),
    }


def process_video(video_path: Path) -> dict:
    bboxes = detect_bboxes_in_video(video_path)
    feats = compute_motion_features(bboxes)
    return {
        "video": str(video_path.relative_to(BASE)) if video_path.is_relative_to(BASE) else str(video_path),
        "n_bboxes": len(bboxes),
        "features": feats,
    }


def batch_process(out_csv: Path) -> dict:
    """data/v18/videos_5_9/ 配下 全動画 batch 処理 → CSV."""
    if not VIDEO_DIR.exists():
        return {"status": "no_video_dir"}

    rows = []
    for race_dir in VIDEO_DIR.iterdir():
        if not race_dir.is_dir(): continue
        for f in race_dir.iterdir():
            if f.suffix.lower() in (".mp4", ".mov", ".avi"):
                print(f"  processing {f.name}")
                r = process_video(f)
                rows.append({
                    "race_id": race_dir.name,
                    "horse_id": f.stem,
                    **r["features"],
                })

    if not rows:
        return {"status": "no_videos"}

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return {"status": "ok", "n_videos": len(rows), "out": str(out_csv)}


def main():
    p = argparse.ArgumentParser(description="horse_motion_features (Session #52 B+C)")
    p.add_argument("--video", default=None)
    p.add_argument("--batch", action="store_true")
    p.add_argument("--out", default="data/v18/horse_motion_5_9.csv")
    args = p.parse_args()

    print("=" * 70)
    print("horse_motion_features (Session #52 B+C、 dev/training-poc)")
    print("=" * 70)

    if args.batch:
        out_csv = BASE / args.out
        result = batch_process(out_csv)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.video:
        path = Path(args.video)
        if not path.is_absolute():
            path = BASE / args.video
        result = process_video(path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # PoC simulate (sample image で test)
        sample = BASE / "data" / "video_poc" / "sample_horse.jpg"
        if sample.exists():
            print(f"[simulate] {sample.name} (静止画 sample)")
            # 静止画で simulate (1 frame)
            from ultralytics import YOLO
            import cv2
            model = YOLO("yolov8n.pt")
            results = model(str(sample), conf=0.25, verbose=False, device="cpu")
            bboxes_seq = []
            for r in results:
                if r.boxes is None: continue
                for box in r.boxes:
                    cls = int(box.cls[0]) if box.cls is not None else -1
                    if cls != 17: continue
                    conf = float(box.conf[0]) if box.conf is not None else 0
                    xyxy = box.xyxy[0].tolist() if box.xyxy is not None else [0,0,0,0]
                    bboxes_seq.append({"frame_idx": 0, "conf": conf,
                                       "x1": xyxy[0], "y1": xyxy[1],
                                       "x2": xyxy[2], "y2": xyxy[3]})
            feats = compute_motion_features(bboxes_seq)
            print(json.dumps({"simulate": True, "features": feats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
