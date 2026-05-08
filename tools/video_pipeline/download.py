"""動画 download module (Session #48 C、 dev/video-poc).

source 候補:
- netkeiba 調教動画 (Premium、 金曜 13:00 公開)
- JRA-VAN ネクスト パドック動画 (各 R 20-30 分前)

usage:
  from tools.video_pipeline.download import download_training_video
  download_training_video(race_id, horse_name, out_path)

V15 production 完全独立、 dev/video-poc 専用。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")


def download_training_video(race_id: str, horse_name: str = None,
                            out_dir: str = "data/video_poc/training") -> dict:
    """調教動画 download (netkeiba Premium 経由).

    本 Session では design + skeleton。 実 download は Phase 4 (7-8 月) で。
    """
    out_path = BASE / out_dir / f"{race_id}_{horse_name or 'all'}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return {
        "status": "deferred",
        "race_id": race_id,
        "horse_name": horse_name,
        "out_path": str(out_path),
        "design": {
            "step1": "netkeiba Premium login (Cookie 経由)",
            "step2": "race 調教ページ で 動画 URL 取得",
            "step3": "URL から mp4 download (yt-dlp or requests)",
            "step4": "out_path に save",
        },
        "phase4_plan": "7-8 月 PoC で 50 動画蓄積、 1 動画 50 MB 想定",
    }


def download_paddock_video(race_id: str, out_dir: str = "data/video_poc/paddock") -> dict:
    """パドック動画 download (JRA-VAN ネクスト or netkeiba)."""
    out_path = BASE / out_dir / f"{race_id}_paddock.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return {
        "status": "deferred",
        "race_id": race_id,
        "out_path": str(out_path),
        "design": {
            "primary": "JRA-VAN ネクスト 加入後 動画 API",
            "fallback": "netkeiba Premium パドック動画",
            "trigger": "各 R 25 分前 schtasks (5/16+)",
        },
    }


def cli():
    p = argparse.ArgumentParser(description="video download (Session #48 C)")
    p.add_argument("--race-id", required=True)
    p.add_argument("--horse-name", default=None)
    p.add_argument("--type", choices=["training", "paddock"], default="training")
    args = p.parse_args()

    if args.type == "training":
        r = download_training_video(args.race_id, args.horse_name)
    else:
        r = download_paddock_video(args.race_id)
    import json
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
