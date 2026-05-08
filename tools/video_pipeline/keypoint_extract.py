"""歩様 keypoint 抽出 module (Session #48 C、 dev/video-poc).

DLC SuperAnimal-Quadruped (Session #39 G で技術調査済) で zero-shot 動物姿勢推定。
4 足動物 39 種 pretrained、 馬を含む。

注意: DLC は本 Session で install せず、 design + skeleton のみ。
Phase 4 (7-8 月) で deeplabcut install + 実 inference。

usage:
  from tools.video_pipeline.keypoint_extract import extract_keypoints
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")


def extract_keypoints(video_path: Path, fps: int = 5, max_frames: int = 50) -> dict:
    """DLC SuperAnimal で keypoint 抽出 (本 Session では design)."""
    try:
        import deeplabcut  # 未 install (Phase 4 で install)
    except ImportError:
        return {
            "status": "deferred",
            "reason": "deeplabcut 未 install (Phase 4 で pip install deeplabcut)",
            "design": {
                "model": "superanimal_quadruped (4 足動物 zero-shot、 馬含む 39 種)",
                "keypoints": [
                    "nose (鼻)", "ear_left (左耳)", "ear_right (右耳)",
                    "shoulder (肩)", "tail_base (尻尾)",
                    "front_left_hoof (前左蹄)", "front_right_hoof (前右蹄)",
                    "back_left_hoof (後左蹄)", "back_right_hoof (後右蹄)",
                ],
                "frames": fps * 12,  # 1 動画 ~60 sec
                "estimated_inference_ms_per_frame": 200,  # GPU、 CPU は 数倍
            },
        }

    return {"status": "design", "note": "実 inference は Phase 4 で"}


def cli():
    p = argparse.ArgumentParser(description="DLC keypoint extract (Session #48 C)")
    p.add_argument("--video", required=True)
    args = p.parse_args()

    r = extract_keypoints(Path(args.video))
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
