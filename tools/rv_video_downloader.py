#!/usr/bin/env python
"""JRA レーシングビュアー (RV) 動画 metadata fetcher — Phase 16 skeleton.

Phase 16 (2026-05-10) で追加した skeleton。
本ファイルは V15 production と完全に分離。

★ 重要: 本 skeleton は 自動 download を実行しない ★
  - JRA-VAN RV 規約: 自動大量 DL は規約違反 / access ban risk
  - 個人視聴は OK、 個人録画 (iOS/Mac 画面録画) はグレーゾーン
  - 公式 API なし (Mpeg4 ストリーミング、 直 download 経路なし)
  - JV-Link は メタデータ + 数値 のみ、 動画 binary record は無し (Session #42 確認済)

Live activation timing:
  - 5/15-6/15 RV trial 期間: 個人視聴 + 手動録画 (重賞のみ)
  - 6/15-7/1: 動画 features 抽出 logic 確定
  - 7/1-9/2 V21 動画統合 学習 (Phase 4 plan v2 維持)
  - 9/2 V21 投入候補

本 skeleton は **動画 metadata 管理 + 手動配置 動画への path 解決** のみ実装。
動画 frame 抽出 / model inference は tools/predict_core_v21.py 側。

Sources:
- docs/JRA_VAN_RV_TRIAL_GUIDE.md
- docs/PHASE_4_VIDEO_REPLAN_v2.md
- data/v18/phase10_racing_viewer_full.md
"""
from __future__ import annotations
import os
import json
from typing import Dict, List, Optional

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')
VIDEO_DIR = os.path.join(DATA_DIR, 'rv_videos')
METADATA_FILE = os.path.join(DATA_DIR, 'rv_video_metadata.json')

# 動画 category (RV 提供)
VIDEO_CATEGORIES = (
    'paddock',     # パドック動画
    'patrol',      # パトロールビデオ
    'chokyou',     # 調教映像 (重賞 出走予定馬)
    'race',        # 過去レース映像 (2002+)
    'multicam',    # マルチカメラ (重賞)
)


def get_video_path(race_id: str, umaban: int, category: str) -> Optional[str]:
    """手動配置済 動画 file path を解決.

    動画 file 命名規則 (5/15+ trial で確定):
      data/rv_videos/<category>/<race_id>_u<umaban>.mp4
      例: data/rv_videos/paddock/202605020611_u01.mp4

    Args:
        race_id: JV race_id (12 桁)
        umaban: 馬番 (1-18)
        category: VIDEO_CATEGORIES のいずれか

    Returns:
        file path (存在時) or None (未配置時)
    """
    if category not in VIDEO_CATEGORIES:
        raise ValueError(f"unknown category: {category}, must be in {VIDEO_CATEGORIES}")
    fname = f"{race_id}_u{umaban:02d}.mp4"
    fpath = os.path.join(VIDEO_DIR, category, fname)
    return fpath if os.path.isfile(fpath) else None


def list_available_videos(race_id: Optional[str] = None) -> Dict[str, List[str]]:
    """配置済 動画 list を category 別 集計.

    Args:
        race_id: 指定時、 該当 race のみ filter

    Returns:
        {category: [video_path, ...]}
    """
    out: Dict[str, List[str]] = {cat: [] for cat in VIDEO_CATEGORIES}
    if not os.path.isdir(VIDEO_DIR):
        return out
    for cat in VIDEO_CATEGORIES:
        cat_dir = os.path.join(VIDEO_DIR, cat)
        if not os.path.isdir(cat_dir):
            continue
        for fname in os.listdir(cat_dir):
            if not fname.endswith('.mp4'):
                continue
            if race_id and not fname.startswith(race_id):
                continue
            out[cat].append(os.path.join(cat_dir, fname))
    return out


def save_metadata(race_id: str, umaban: int, category: str, info: Dict) -> None:
    """動画 metadata (撮影日時 / 重賞名 / 視聴済 flag 等) 保存.

    手動 trial 中の管理用。 公開 NG なので metadata only DB。
    """
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    if os.path.isfile(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            db = json.load(f)
    else:
        db = {}
    key = f"{race_id}_u{umaban:02d}_{category}"
    db[key] = info
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def get_trial_status() -> Dict[str, any]:
    """5/15-6/15 RV trial 進捗 status."""
    available = list_available_videos()
    total = sum(len(v) for v in available.values())
    return {
        'trial_period': '2026-05-15 to 2026-06-15',
        'video_categories': list(VIDEO_CATEGORIES),
        'total_videos_collected': total,
        'by_category': {cat: len(paths) for cat, paths in available.items()},
        'video_dir_exists': os.path.isdir(VIDEO_DIR),
        'note': '5/15+ で 重賞 trial 開始、 個人視聴 + 手動録画。 自動 DL は規約違反のため実装なし。',
    }


if __name__ == '__main__':
    print(f"[rv_video_downloader] Phase 16 skeleton")
    print(f"[rv_video_downloader] VIDEO_DIR: {VIDEO_DIR}")
    print(f"[rv_video_downloader] VIDEO_DIR exists: {os.path.isdir(VIDEO_DIR)}")
    status = get_trial_status()
    print(f"[rv_video_downloader] trial status:")
    for k, v in status.items():
        print(f"    {k}: {v}")
    print("[rv_video_downloader] OK: skeleton 動作確認、 自動 DL は実装なし (規約遵守)")
