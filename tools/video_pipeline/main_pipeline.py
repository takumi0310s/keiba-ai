"""動画解析 統合 pipeline (Session #48 C、 dev/video-poc).

1 race の全馬の動画 → features.

flow:
1. download.py: 動画 download
2. yolo_inference.py: 馬体検出
3. keypoint_extract.py: 歩様 keypoint (Phase 4 で実装)
4. features_aggregate.py: features 化
5. main_pipeline.py: 1 race の全馬 features 統合

usage:
  python tools/video_pipeline/main_pipeline.py --race-id 202605020411
  python tools/video_pipeline/main_pipeline.py --image data/video_poc/sample.jpg

V15 production 完全独立、 dev/video-poc 専用。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
sys.path.insert(0, str(BASE / "tools" / "video_pipeline"))

from yolo_inference import detect_in_image, detect_in_video
from features_aggregate import aggregate_video_features


def run_pipeline_image(img_path: Path) -> dict:
    """静止画 1 枚 で pipeline 実行."""
    print(f"[pipeline] Step 1: download (skipped、 既 image)")
    print(f"[pipeline] Step 2: YOLOv8 inference ({img_path.name})")
    yolo = detect_in_image(img_path)
    print(f"  result: status={yolo.get('status')}, n_horses={yolo.get('n_horses', 0)}")

    print(f"[pipeline] Step 3: keypoint extract (deferred、 Phase 4)")
    keypoint = {"status": "deferred"}

    print(f"[pipeline] Step 4: features aggregate")
    # YOLOv8 image inference は frames 形式と異なる、 直接 features 化
    yolo_for_agg = {
        "status": yolo.get("status"),
        "horse_detection_rate_pct": 100 if yolo.get("n_horses", 0) > 0 else 0,
        "frames": [{"horses": yolo.get("horses", [])}],
    }
    feats = aggregate_video_features(yolo_for_agg, keypoint)
    print(f"  features: {feats}")

    return {
        "input": str(img_path),
        "yolo": yolo,
        "keypoint": keypoint,
        "features": feats,
    }


def run_pipeline_video(video_path: Path) -> dict:
    """動画 で pipeline 実行."""
    print(f"[pipeline] Step 1: download (skipped、 既 video)")
    print(f"[pipeline] Step 2: YOLOv8 video inference ({video_path.name})")
    yolo = detect_in_video(video_path)
    print(f"  result: status={yolo.get('status')}, "
          f"detection_rate={yolo.get('horse_detection_rate_pct', 0)}%")

    print(f"[pipeline] Step 3: keypoint extract (deferred、 Phase 4)")
    keypoint = {"status": "deferred"}

    print(f"[pipeline] Step 4: features aggregate")
    feats = aggregate_video_features(yolo, keypoint)
    print(f"  features: {feats}")

    return {
        "input": str(video_path),
        "yolo": yolo,
        "keypoint": keypoint,
        "features": feats,
    }


def main():
    p = argparse.ArgumentParser(description="video pipeline main (Session #48 C)")
    p.add_argument("--image", default=None)
    p.add_argument("--video", default=None)
    p.add_argument("--race-id", default=None, help="将来: race の全馬 features 取得")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.image:
        img_path = Path(args.image)
        if not img_path.is_absolute():
            img_path = BASE / args.image
        result = run_pipeline_image(img_path)
    elif args.video:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = BASE / args.video
        result = run_pipeline_video(video_path)
    elif args.race_id:
        print(f"[pipeline] race_id mode (deferred、 Phase 4 で 全馬 動画自動 download + pipeline)")
        result = {
            "race_id": args.race_id,
            "status": "deferred",
            "design": "5/8 13:00 動画公開後、 Phase 4 で 全馬 download + 並列 pipeline",
        }
    else:
        print("[!] --image, --video, --race-id いずれか指定")
        return

    out_path = BASE / (args.out or "data/v18/session_48_video_pipeline_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
