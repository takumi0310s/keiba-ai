"""Phase C: パトロール YOLO data prep (skeleton).

5/16 (今日) は skeleton + comment のみ。 actual implementation は 5/18-5/24 PoC で実施。

★ V15 production 完全不変 ★
- このファイルは V15 / predict_core.py / daily_predict.py に一切影響しない。
- 動画なし R では 全 8 features = 0 fill (V21 model 側で扱う、 本 file は 計算 のみ)。

設計 doc:
- data/v21/phase_c_patrol_8_features_spec.md (8 features 詳細仕様)
- data/v21/phase_c_patrol_yolo_poc_plan.md (5/18-5/24 PoC plan)
- data/v21/yolov8_lfs_setup.md (LFS 準備)

usage (5/24 PoC 完成想定):
    python tools/v21/patrol_yolo_data_prep.py \
        --video data/v21/patrol_poc/202605030611.mp4 \
        --race-id 202605030611 \
        --surface-enc 0 \
        --out data/v21/patrol_poc/202605030611_patrol_features.csv

5/16 self-test:
    python tools/v21/patrol_yolo_data_prep.py --self-test
    → expected output: 8 features list 表示、 default 0 fill DataFrame 1 行返却
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ★ skeleton 段階 ★: pandas / numpy のみ import。 ultralytics / cv2 / supervision は
# 5/18+ 本実装時に追加。 import 順序: stdlib → 3rd party (skeleton では minimal)。
try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    print(f"[fatal] numpy / pandas missing: {exc}")
    sys.exit(1)

BASE = Path(r"C:/Users/takum/keiba-ai")
POC_DIR = BASE / "data" / "v21" / "patrol_poc"

# ---------------------------------------------------------------------------
# 8 features 定義 (spec doc 同期)
# ---------------------------------------------------------------------------
PATROL_FEATURES = [
    "patrol_dangerous_pos_count",     # int 0-5  危険位置 frame 数
    "patrol_blocked_frame_count",     # int 0-N  前方塞がれ frame 数
    "patrol_track_change_count",      # int 0-10 進路変更回数
    "patrol_jam_risk_score",          # float 0-1 周囲 5m 馬密度 平均
    "patrol_acceleration_late",       # float    終い加速度 (pix/frame²)
    "patrol_position_recovery",       # int -17〜+17  4 角→ゴール 位置回復
    "patrol_cornering_smoothness",    # float 0-1  4 角 trajectory 平滑度
    "patrol_kickback_dust_exposure",  # float 0-1  砂被り time 比率 (ダート のみ)
]

# default fill (動画なし R 用、 V21 model 側で 適用)
PATROL_DEFAULT_FILL = {
    "patrol_dangerous_pos_count": 0,
    "patrol_blocked_frame_count": 0,
    "patrol_track_change_count": 0,
    "patrol_jam_risk_score": 0.0,
    "patrol_acceleration_late": 0.0,
    "patrol_position_recovery": 0,
    "patrol_cornering_smoothness": 0.5,  # ★ 中央値 fill (0 だと 最悪 解釈される) ★
    "patrol_kickback_dust_exposure": 0.0,
}


# ---------------------------------------------------------------------------
# 5/18+ で実装: 動画 → frame 抽出
# ---------------------------------------------------------------------------
def extract_patrol_frames(video_path: Path, fps: int = 2, out_dir: Path | None = None) -> list[Path]:
    """ffmpeg で 動画 → frame 抽出 (2 fps、 1080p 想定).

    Args:
        video_path: 入力 mp4 (OBS 録画想定、 60-120 秒)
        fps: 抽出 fps (default 2 → 1 R 120-240 frame)
        out_dir: 出力 ディレクトリ (default = video_path と同階層 / <stem>/)

    Returns:
        抽出 frame path list

    5/18 実装 TODO:
        cmd = f"ffmpeg -i {video_path} -vf fps={fps} {out_dir}/frame_%04d.jpg"
        subprocess.run(cmd, check=True)
        return sorted(out_dir.glob("frame_*.jpg"))
    """
    raise NotImplementedError("5/18 PoC で本実装 (skeleton 段階)")


# ---------------------------------------------------------------------------
# 5/19+ で実装: YOLOv8 検出
# ---------------------------------------------------------------------------
def detect_horses(frame_path: Path, model: Any, conf: float = 0.3) -> list[dict]:
    """YOLOv8 で 1 frame の 馬 (COCO class 17) を 検出.

    Args:
        frame_path: 入力 jpg
        model: ultralytics YOLO instance (yolov8s.pt zero-shot start)
        conf: confidence threshold

    Returns:
        list[dict]: [{bbox: [x, y, w, h], conf: float, class_id: 17}, ...]

    5/19 実装 TODO:
        from ultralytics import YOLO  # 関数内 lazy import
        results = model(frame_path, conf=conf, classes=[17])  # horse class
        return [{
            "bbox": box.xywh[0].tolist(),
            "conf": float(box.conf[0]),
            "class_id": int(box.cls[0]),
        } for box in results[0].boxes]
    """
    raise NotImplementedError("5/19 PoC で本実装")


# ---------------------------------------------------------------------------
# 5/20+ で実装: ByteTrack で 馬個別 tracking
# ---------------------------------------------------------------------------
def track_horses(frame_detections: list[list[dict]]) -> dict[int, list[dict]]:
    """ByteTrack で frame 間 tracking、 horse_id 付き track 返却.

    Args:
        frame_detections: extract→detect の全 frame 検出 list

    Returns:
        dict[horse_id, list[{frame_idx, bbox, conf}]]

    5/20 実装 TODO:
        import supervision as sv  # ByteTrack 含む
        tracker = sv.ByteTrack()
        tracks = {}
        for frame_idx, dets in enumerate(frame_detections):
            sv_dets = sv.Detections(xyxy=..., confidence=..., class_id=...)
            tracked = tracker.update_with_detections(sv_dets)
            for tid, bbox in zip(tracked.tracker_id, tracked.xyxy):
                tracks.setdefault(int(tid), []).append({
                    "frame_idx": frame_idx,
                    "bbox": bbox.tolist(),
                })
        return tracks
    """
    raise NotImplementedError("5/20 PoC で本実装")


# ---------------------------------------------------------------------------
# 5/22-5/23 で実装: 8 features 計算 関数
# ---------------------------------------------------------------------------
def compute_dangerous_pos_count(track: list[dict], all_detections: list[list[dict]]) -> int:
    """危険位置 (最内 ≤ 5% / 最外 ≥ 95% lateral) frame 数.

    5/22 実装 TODO:
        count = 0
        for entry in track:
            frame_dets = all_detections[entry["frame_idx"]]
            x_centers = [d["bbox"][0] for d in frame_dets]
            x_min, x_max = min(x_centers), max(x_centers)
            target_x = entry["bbox"][0]
            normalized = (target_x - x_min) / (x_max - x_min + 1e-6)
            if normalized <= 0.05 or normalized >= 0.95:
                count += 1
        return count
    """
    raise NotImplementedError("5/22 PoC で本実装")


def compute_blocked_frame_count(track: list[dict], all_detections: list[list[dict]]) -> int:
    """前方 5m 以内 他馬 占有 frame 数.

    5/22 実装 TODO:
        # 進行方向 = frame 間 bbox center diff の符号
        # 5m = pixel 換算 (馬体長 ~150 pixel × 2.5 ~= 375 pixel)
        ...
    """
    raise NotImplementedError("5/22 PoC で本実装")


def compute_track_change_count(track: list[dict]) -> int:
    """lateral 急変 回数 (5 frame moving mean diff > 馬体幅 × 1.5).

    5/22 実装 TODO:
        x_series = [e["bbox"][0] for e in track]
        ma5 = np.convolve(x_series, np.ones(5)/5, mode='valid')
        diffs = np.abs(np.diff(ma5))
        threshold = 75  # pixel ~ 馬体幅 × 1.5
        return int((diffs > threshold).sum())
    """
    raise NotImplementedError("5/22 PoC で本実装")


def compute_jam_risk_score(track: list[dict], all_detections: list[list[dict]]) -> float:
    """周囲 5m 以内 他馬 密度 平均 (0-1 normalize).

    5/22 実装 TODO:
        radius_pixel = 375  # 5m 相当
        densities = []
        for entry in track:
            frame_dets = all_detections[entry["frame_idx"]]
            tx, ty = entry["bbox"][:2]
            n_near = sum(1 for d in frame_dets
                         if abs(d["bbox"][0]-tx) <= radius_pixel
                         and abs(d["bbox"][1]-ty) <= radius_pixel
                         and d is not entry)
            densities.append(n_near / 17.0)  # max 17 他馬
        return float(np.mean(densities)) if densities else 0.0
    """
    raise NotImplementedError("5/22 PoC で本実装")


def compute_acceleration_late(track: list[dict]) -> float:
    """終い (最終 25% 区間) の 加速度 mean (pix/frame²).

    5/23 実装 TODO:
        n = len(track)
        late_start = int(n * 0.75)
        x_late = [e["bbox"][0] for e in track[late_start:]]
        velocity = np.diff(x_late)
        acceleration = np.diff(velocity)
        return float(np.mean(acceleration)) if len(acceleration) > 0 else 0.0
    """
    raise NotImplementedError("5/23 PoC で本実装")


def compute_position_recovery(track: list[dict], all_detections: list[list[dict]]) -> int:
    """4 角 → ゴール 順位差 (+ なら 回復、 - なら 後退).

    5/23 実装 TODO:
        n_total = len(all_detections)
        corner4_frame = int(n_total * 0.65)  # ★ R 種別 で calibrate 必要 ★
        goal_frame = n_total - 5
        # 順位 = 横位置 (進行方向) 順
        ...
    """
    raise NotImplementedError("5/23 PoC で本実装")


def compute_cornering_smoothness(track: list[dict]) -> float:
    """4 角区間 (50-70%) trajectory 平滑度 (1 - 角度差 std / 90°).

    5/23 実装 TODO:
        n = len(track)
        c4_start, c4_end = int(n * 0.5), int(n * 0.7)
        track_c4 = track[c4_start:c4_end]
        # 連続 3 点 で 角度 diff 計算
        angles = []
        for i in range(1, len(track_c4)-1):
            p1, p2, p3 = [t["bbox"][:2] for t in track_c4[i-1:i+2]]
            v1 = np.array(p2) - np.array(p1)
            v2 = np.array(p3) - np.array(p2)
            cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
            angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        std_deg = float(np.std(angles)) if angles else 0.0
        return max(0.0, 1.0 - std_deg / 90.0)
    """
    raise NotImplementedError("5/23 PoC で本実装")


def compute_kickback_dust_exposure(
    track: list[dict], all_detections: list[list[dict]], surface_enc: int
) -> float:
    """砂被り time 比率 (ダート R のみ、 surface_enc==1 のとき計測).

    5/23 実装 TODO:
        if surface_enc != 1:  # 芝 / 障害 は 0 fill
            return 0.0
        # 前方 2m (~150 pixel) 内 他馬 frame 数 + 砂煙 segmentation
        # PoC 段階では 簡易版 = 前方 他馬 frame 数 のみ
        ...
    """
    raise NotImplementedError("5/23 PoC で本実装")


# ---------------------------------------------------------------------------
# 5/24 で実装: end-to-end pipeline
# ---------------------------------------------------------------------------
def compute_patrol_features_for_race(
    video_path: Path, race_id: str, surface_enc: int = 0
) -> pd.DataFrame:
    """1 動画 → 18 馬 × 8 features の DataFrame 返却 (end-to-end).

    5/24 実装 TODO:
        frames = extract_patrol_frames(video_path, fps=2)
        from ultralytics import YOLO
        model = YOLO(str(BASE / "yolov8s.pt"))
        all_dets = [detect_horses(f, model) for f in frames]
        tracks = track_horses(all_dets)
        rows = []
        for horse_id, track in tracks.items():
            row = {"race_id": race_id, "horse_track_id": horse_id}
            row["patrol_dangerous_pos_count"]  = compute_dangerous_pos_count(track, all_dets)
            row["patrol_blocked_frame_count"]  = compute_blocked_frame_count(track, all_dets)
            row["patrol_track_change_count"]   = compute_track_change_count(track)
            row["patrol_jam_risk_score"]       = compute_jam_risk_score(track, all_dets)
            row["patrol_acceleration_late"]    = compute_acceleration_late(track)
            row["patrol_position_recovery"]    = compute_position_recovery(track, all_dets)
            row["patrol_cornering_smoothness"] = compute_cornering_smoothness(track)
            row["patrol_kickback_dust_exposure"] = compute_kickback_dust_exposure(
                track, all_dets, surface_enc
            )
            rows.append(row)
        return pd.DataFrame(rows)
    """
    raise NotImplementedError("5/24 PoC で本実装")


# ---------------------------------------------------------------------------
# self-test (5/16 段階で動作確認可能)
# ---------------------------------------------------------------------------
def run_self_test() -> int:
    """skeleton self-test: 8 features 名 + default fill 表示、 default DataFrame 1 行返却."""
    print("=" * 60)
    print("Phase C パトロール YOLO data prep skeleton (5/16)")
    print("=" * 60)
    print(f"\n[1] PATROL_FEATURES ({len(PATROL_FEATURES)} 件):")
    for i, f in enumerate(PATROL_FEATURES, 1):
        default = PATROL_DEFAULT_FILL[f]
        print(f"  {i}. {f}  (default fill = {default})")

    print("\n[2] default-fill DataFrame (動画なし R 想定、 1 行):")
    df = pd.DataFrame([{**PATROL_DEFAULT_FILL, "race_id": "test_0000", "horse_track_id": 1}])
    print(df.to_string(index=False))

    print("\n[3] 5/18+ 実装予定 関数 一覧:")
    todos = [
        "extract_patrol_frames (5/18, ffmpeg)",
        "detect_horses (5/19, ultralytics YOLOv8)",
        "track_horses (5/20, supervision.ByteTrack)",
        "compute_dangerous_pos_count (5/22)",
        "compute_blocked_frame_count (5/22)",
        "compute_track_change_count (5/22)",
        "compute_jam_risk_score (5/22)",
        "compute_acceleration_late (5/23)",
        "compute_position_recovery (5/23)",
        "compute_cornering_smoothness (5/23)",
        "compute_kickback_dust_exposure (5/23)",
        "compute_patrol_features_for_race (5/24, end-to-end)",
    ]
    for t in todos:
        print(f"  - {t}")

    print("\n[4] V15 production 投資保護 確認:")
    print("  - V15 .pkl.gz / predict_core.py / daily_predict.py に一切影響なし")
    print("  - 本 file は V21 model 学習時のみ呼び出される (5/18+ PoC + 9/1+ 投入想定)")
    print("\n[OK] skeleton self-test PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase C パトロール YOLO data prep (skeleton)")
    parser.add_argument("--self-test", action="store_true", help="skeleton self-test 実行")
    parser.add_argument("--video", type=Path, help="入力 mp4 (5/18+ 本実装で使用)")
    parser.add_argument("--race-id", type=str, help="race_id (5/18+)")
    parser.add_argument("--surface-enc", type=int, default=0, help="0=芝, 1=ダート, 2=障害")
    parser.add_argument("--out", type=Path, help="出力 csv path")
    args = parser.parse_args(argv)

    if args.self_test or not args.video:
        return run_self_test()

    # 5/24 PoC 想定 (現状 NotImplementedError で halt)
    print(f"[info] 動画 {args.video} → features 抽出 (5/18+ PoC で本実装)")
    df = compute_patrol_features_for_race(args.video, args.race_id, args.surface_enc)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"[ok] saved: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
