"""動画 features 集計 module (Session #48 C、 dev/video-poc).

YOLOv8 馬体検出 + DLC keypoint から features 化:
- video_horse_size_score (体格、 bbox 面積)
- video_pose_stability (frame 間 bbox 変動)
- video_aspect_ratio (standing horse 標準 1.3-2.5)
- video_stride_freq (歩幅頻度、 keypoint 蹄 y 座標 peaks)
- video_gait_symmetry (左右対称性)

usage:
  from tools.video_pipeline.features_aggregate import aggregate_video_features
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BASE = Path(r"C:/Users/takum/keiba-ai")


def aggregate_yolo_features(yolo_result: dict) -> dict:
    """YOLO 検出結果から features 化."""
    out = {
        "video_horse_size_score": 0.0,
        "video_pose_stability": 0.0,
        "video_aspect_ratio": 0.0,
        "video_horse_detection_rate": 0.0,
    }

    if yolo_result.get("status") != "ok":
        return out

    frames = yolo_result.get("frames", [])
    if not frames:
        # all frames not exposed (top 5 のみ in skeleton)
        out["video_horse_detection_rate"] = yolo_result.get("horse_detection_rate_pct", 0) / 100
        return out

    bbox_areas = []
    aspect_ratios = []
    centers_x = []
    centers_y = []

    for fr in frames:
        for h in fr.get("horses", []):
            bbox = h.get("bbox", [0, 0, 0, 0])
            w = bbox[2] - bbox[0]
            ht = bbox[3] - bbox[1]
            if w > 0 and ht > 0:
                bbox_areas.append(w * ht)
                aspect_ratios.append(w / ht)
                centers_x.append((bbox[0] + bbox[2]) / 2)
                centers_y.append((bbox[1] + bbox[3]) / 2)

    if bbox_areas:
        out["video_horse_size_score"] = round(min(1.0, np.mean(bbox_areas) / 100000), 4)
    if aspect_ratios:
        out["video_aspect_ratio"] = round(np.mean(aspect_ratios), 3)
    if centers_x and centers_y and len(centers_x) >= 2:
        # frame 間 中心 std (低 = 安定 pose)
        std_x = np.std(centers_x)
        std_y = np.std(centers_y)
        out["video_pose_stability"] = round(1 / (1 + std_x + std_y), 4)  # high = stable

    out["video_horse_detection_rate"] = yolo_result.get("horse_detection_rate_pct", 0) / 100
    return out


def aggregate_keypoint_features(keypoint_result: dict) -> dict:
    """DLC keypoint 結果から features 化 (本 Session では skeleton)."""
    out = {
        "video_stride_freq": 0.0,
        "video_gait_symmetry": 0.0,
        "video_head_bobbing_amp": 0.0,
        "video_ear_pos_y_mean": 0.0,
    }

    if keypoint_result.get("status") != "ok":
        return out

    # DLC inference 完了後の logic (Phase 4 で実装):
    # - 蹄 keypoint y 座標 peaks → stride frequency
    # - 左右蹄 phase correlation → gait symmetry
    # - 頭 keypoint y 軌跡 → head bobbing
    # - 耳 keypoint mean y → ear position

    return out


def aggregate_video_features(yolo_result: dict, keypoint_result: dict = None) -> dict:
    """YOLO + keypoint 統合 features."""
    yolo_feats = aggregate_yolo_features(yolo_result)
    kp_feats = aggregate_keypoint_features(keypoint_result) if keypoint_result else {}
    return {**yolo_feats, **kp_feats}


def cli():
    p = argparse.ArgumentParser(description="features aggregate (Session #48 C)")
    p.add_argument("--yolo-json", required=True)
    p.add_argument("--keypoint-json", default=None)
    args = p.parse_args()

    yolo = json.loads(Path(args.yolo_json).read_text(encoding="utf-8"))
    kp = json.loads(Path(args.keypoint_json).read_text(encoding="utf-8")) if args.keypoint_json else None

    feats = aggregate_video_features(yolo, kp)
    print(json.dumps(feats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
