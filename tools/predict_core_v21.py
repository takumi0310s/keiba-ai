#!/usr/bin/env python
"""V21 candidate predict core — Phase 16 動画 30 features skeleton.

Phase 16 (2026-05-10) で追加した動画 30 features の skeleton 実装。
本ファイルは V15 production / V18 candidate と完全に分離。
V15 投資保護: predict_core.py / V15 model file 不変。

V21 candidate features:
  V20 base (207) + 動画 (30) = ★ 237 features ★
  V20 base = V15 (150) + Phase 11 JRDB (15) + Phase 12 DataLab (17) + Phase 13 netkeiba (25)

動画 features 内訳 (30):
  B. パドック CNN     (12) — RV パドック動画 + ResNet50/EfficientNet
  C. パトロール YOLO   (8) — RV パトロール動画 + YOLOv8/v11 + DeepSORT
  D. 調教 keypoint   (10) — RV 調教映像 + MMPose/OpenPose 馬 17 keypoint

Live activation timing:
  - 本 skeleton: default fill のみ (model 未学習 = 中央値 fill)
  - 5/15-6/15 RV trial: 重賞 動画手動収集
  - 7/1-9/2: GPU (RTX 4070 Ti SUPER 16GB) で fine-tune + features 抽出 自動化
  - 9/2 V21 投入候補 GO/no-go 判定

GPU 環境 (Session #42 確認済):
  - PyTorch 2.11.0+cu126
  - CUDA 利用可 (NVIDIA RTX 4070 Ti SUPER 16GB)
  - ultralytics 8.4.47
  - opencv-python 4.13.0
  - torchvision NMS CUDA 課題: 7/1 着手前に修正
"""
from __future__ import annotations
import os
from typing import Dict, List, Any, Optional

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'v21')

# =========================================================================
# Feature registry — 動画 30 features
# =========================================================================

# B. パドック CNN (12 features)
PADDOCK_CNN_FEATURES: List[str] = [
    'video_paddock_body_score',         # 馬体充実度 (0-100)
    'video_paddock_sweat_level',        # 発汗 level (0-3、 0=無/1=微/2=多/3=異常)
    'video_paddock_tension_score',      # 興奮度 (0-100、 50 = 通常)
    'video_paddock_hoof_health',        # 蹄健全度 (0-1)
    'video_paddock_hindleg_drive',      # 後肢踏み込み 強度 (0-1)
    'video_paddock_calmness_score',     # 落ち着き (0-1)
    'video_paddock_coat_shine',         # 毛艶 (0-1)
    'video_paddock_ear_position',       # 耳位置 stress (0-1、 1=後ろ向き)
    'video_paddock_head_carriage',      # 頭の運び (0-1、 1=理想)
    'video_paddock_back_arc',           # 背中弓り (0-1、 0=異常)
    'video_paddock_walk_rhythm',        # 歩調 安定度 (0-1)
    'video_paddock_overall_condition',  # 総合 condition (0-1)
]

# C. パトロール YOLO (8 features)
PATROL_YOLO_FEATURES: List[str] = [
    'video_patrol_furi_count',          # 不利受け 回数 (0-5)
    'video_patrol_route_position',      # 進路順 (0=最内、 1=最外)
    'video_patrol_loss_seconds',        # 失速時間 (秒)
    'video_patrol_contact_severity',    # 接触強度 (0-1)
    'video_patrol_block_count',         # 進路カット 回数 (0-3)
    'video_patrol_pace_loss',           # ペース loss (秒)
    'video_patrol_position_change',     # 順位変動 (絶対値)
    'video_patrol_track_run_distance',  # 実走行距離 (m、 直線理論距離との差)
]

# D. 調教 keypoint (10 features)
CHOKYOU_KEYPOINT_FEATURES: List[str] = [
    'video_chokyou_oikiri_time',        # 追切タイム (秒)
    'video_chokyou_last1f_speed',       # 終い 1F speed (m/s)
    'video_chokyou_smoothness',         # 動き smoothness (0-1)
    'video_chokyou_form_score',         # フォーム score (0-1)
    'video_chokyou_stride_length',      # 歩幅 (m)
    'video_chokyou_gait_symmetry',      # 左右 step 対称性 (0-1)
    'video_chokyou_neck_drive',         # 頸の使い方 (0-1)
    'video_chokyou_back_flex',          # 背中柔軟性 (0-1)
    'video_chokyou_hindquarter_power',  # 後駆 power (0-1)
    'video_chokyou_finish_extension',   # 終い 伸び (0-1)
]

ALL_V21_VIDEO_FEATURES: List[str] = (
    PADDOCK_CNN_FEATURES + PATROL_YOLO_FEATURES + CHOKYOU_KEYPOINT_FEATURES
)

# Default fill (model 未学習 / 動画未配置時)
V21_VIDEO_DEFAULTS: Dict[str, float] = {
    # B. パドック (12) — 中央値 fill
    'video_paddock_body_score': 50.0,
    'video_paddock_sweat_level': 0.5,
    'video_paddock_tension_score': 50.0,
    'video_paddock_hoof_health': 0.5,
    'video_paddock_hindleg_drive': 0.5,
    'video_paddock_calmness_score': 0.5,
    'video_paddock_coat_shine': 0.5,
    'video_paddock_ear_position': 0.5,
    'video_paddock_head_carriage': 0.5,
    'video_paddock_back_arc': 0.5,
    'video_paddock_walk_rhythm': 0.5,
    'video_paddock_overall_condition': 0.5,
    # C. パトロール (8) — 0 fill (不利なし default)
    'video_patrol_furi_count': 0.0,
    'video_patrol_route_position': 0.5,
    'video_patrol_loss_seconds': 0.0,
    'video_patrol_contact_severity': 0.0,
    'video_patrol_block_count': 0.0,
    'video_patrol_pace_loss': 0.0,
    'video_patrol_position_change': 0.0,
    'video_patrol_track_run_distance': 0.0,
    # D. 調教 (10) — 中央値 fill
    'video_chokyou_oikiri_time': 51.5,
    'video_chokyou_last1f_speed': 14.0,
    'video_chokyou_smoothness': 0.5,
    'video_chokyou_form_score': 0.5,
    'video_chokyou_stride_length': 6.5,
    'video_chokyou_gait_symmetry': 0.5,
    'video_chokyou_neck_drive': 0.5,
    'video_chokyou_back_flex': 0.5,
    'video_chokyou_hindquarter_power': 0.5,
    'video_chokyou_finish_extension': 0.5,
}

# =========================================================================
# Skeleton fetcher — 7/1+ 学習後 切替
# =========================================================================


def _are_models_available() -> Dict[str, bool]:
    """V21 model files の存在 check."""
    return {
        'paddock_cnn': os.path.isfile(os.path.join(MODELS_DIR, 'paddock_cnn_efficientnet.pth')),
        'patrol_yolo': os.path.isfile(os.path.join(MODELS_DIR, 'patrol_yolo_v8m.pt')),
        'chokyou_keypoint': os.path.isfile(os.path.join(MODELS_DIR, 'chokyou_keypoint_mmpose.pth')),
    }


def fetch_paddock_features(video_path: Optional[str] = None) -> Dict[str, float]:
    """B. パドック CNN 12 features.

    Phase 16 skeleton: default fill のみ。
    7/1+ で ResNet50 / EfficientNet fine-tuned model 経由 frame 抽出 + inference。

    Pipeline (5/15-9/2 構築):
      1. video → opencv frame 抽出 (1 fps、 30 frames / 動画)
      2. ResNet50/EfficientNet (pretrained ImageNet → fine-tune)
      3. multi-head regression: body_score / sweat / tension / etc.
      4. frame 平均 → 1 動画 → 12 features
    """
    if video_path is None or not _are_models_available()['paddock_cnn']:
        return {f: V21_VIDEO_DEFAULTS[f] for f in PADDOCK_CNN_FEATURES}
    # TODO 7/1+: opencv で frame 抽出 + EfficientNet inference
    return {f: V21_VIDEO_DEFAULTS[f] for f in PADDOCK_CNN_FEATURES}


def fetch_patrol_features(video_path: Optional[str] = None) -> Dict[str, float]:
    """C. パトロール YOLO 8 features.

    Phase 16 skeleton: default fill のみ。
    7/1+ で YOLOv8/v11 + DeepSORT 経由 馬体追跡 + 不利検出。

    Pipeline:
      1. video → opencv frame 抽出 (10 fps)
      2. YOLOv8 / v11 で全馬 bbox 検出 + 馬番 OCR
      3. DeepSORT で馬群 tracking
      4. 不利検出 logic: 接触 / 進路カット / 失速 / pace_loss
      5. 1 動画 → 8 features
    """
    if video_path is None or not _are_models_available()['patrol_yolo']:
        return {f: V21_VIDEO_DEFAULTS[f] for f in PATROL_YOLO_FEATURES}
    # TODO 7/1+: ultralytics + DeepSORT inference
    return {f: V21_VIDEO_DEFAULTS[f] for f in PATROL_YOLO_FEATURES}


def fetch_chokyou_features(video_path: Optional[str] = None) -> Dict[str, float]:
    """D. 調教 keypoint 10 features.

    Phase 16 skeleton: default fill のみ。
    7/1+ で MMPose / DeepLabCut SuperAnimal HORSE-10 経由 馬 keypoint 検出。

    Pipeline:
      1. video → opencv frame 抽出 (30 fps)
      2. MMPose で馬 17 keypoint 検出
      3. 動き解析 (smoothness / stride / gait_symmetry / 4 駆 power)
      4. タイム計測 (oikiri / last1f speed)
      5. 1 動画 → 10 features
    """
    if video_path is None or not _are_models_available()['chokyou_keypoint']:
        return {f: V21_VIDEO_DEFAULTS[f] for f in CHOKYOU_KEYPOINT_FEATURES}
    # TODO 7/1+: mmpose / dlc inference
    return {f: V21_VIDEO_DEFAULTS[f] for f in CHOKYOU_KEYPOINT_FEATURES}


def fetch_all_v21_video_features(
    paddock_video: Optional[str] = None,
    patrol_video: Optional[str] = None,
    chokyou_video: Optional[str] = None,
) -> Dict[str, float]:
    """Phase 16 全 30 動画 features 一括取得."""
    out: Dict[str, float] = {}
    out.update(fetch_paddock_features(paddock_video))
    out.update(fetch_patrol_features(patrol_video))
    out.update(fetch_chokyou_features(chokyou_video))
    assert len(out) == 30, f"expected 30 features, got {len(out)}"
    return out


def get_v21_video_feature_names() -> List[str]:
    """Phase 16 で追加された 30 動画 features 名 list."""
    return list(ALL_V21_VIDEO_FEATURES)


def get_v21_video_defaults() -> Dict[str, float]:
    """Phase 16 30 features の default 値."""
    return dict(V21_VIDEO_DEFAULTS)


# =========================================================================
# Self test
# =========================================================================

if __name__ == '__main__':
    print(f"[predict_core_v21] Phase 16 動画 features: {len(ALL_V21_VIDEO_FEATURES)} 件")
    print(f"[predict_core_v21] V21 candidate: V20 (207) + 動画 (30) = 237 features")
    avail = _are_models_available()
    print(f"[predict_core_v21] models 利用可: {avail}")

    dummy = fetch_all_v21_video_features()
    print(f"[predict_core_v21] dummy fetch keys: {len(dummy)} 件")
    assert set(dummy.keys()) == set(ALL_V21_VIDEO_FEATURES), 'feature 名 不一致'
    print("[predict_core_v21] OK: 全 30 features default 取得 成功")

    print(f"  B. パドック CNN ({len(PADDOCK_CNN_FEATURES)}): {PADDOCK_CNN_FEATURES[:3]}...")
    print(f"  C. パトロール YOLO ({len(PATROL_YOLO_FEATURES)}): {PATROL_YOLO_FEATURES[:3]}...")
    print(f"  D. 調教 keypoint ({len(CHOKYOU_KEYPOINT_FEATURES)}): {CHOKYOU_KEYPOINT_FEATURES[:3]}...")
