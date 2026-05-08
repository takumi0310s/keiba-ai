"""パドック画像解析 PoC (Session #47 G、 dev/sprint2).

Phase 4 動画解析 (Session #39-44 設計済) の 拡張: パドック静止画。
YOLOv8 で 馬体検出 + 簡易 score (体格 / pose) → features 化。

source 候補:
- netkeiba パドック画像 (Premium 会員)
- JRA-VAN ネクスト 静止画
- (本 Session は code design + sample image での 動作確認)

usage:
  # 静止画 1 枚で 検出 + score
  python tools/paddock_image_analyzer.py --image path/to/paddock.jpg

  # backtest plan (5/15 merge 後 production)

V15 production 完全独立、 dev/sprint2 のみ。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")


def detect_horse_in_image(img_path: Path, conf_threshold: float = 0.25) -> dict:
    """YOLOv8 で 馬体検出 + 体格 score."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"status": "ultralytics 未 install", "error": "pip install ultralytics"}

    model = YOLO("yolov8n.pt")
    t0 = time.time()
    results = model(str(img_path), conf=conf_threshold, verbose=False, device="cpu")
    inference_ms = (time.time() - t0) * 1000

    detections = []
    horses = []
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            cls = int(box.cls[0]) if box.cls is not None else -1
            conf = float(box.conf[0]) if box.conf is not None else 0
            xyxy = box.xyxy[0].tolist() if box.xyxy is not None else [0, 0, 0, 0]
            is_horse = (cls == 17)
            d = {"class_id": cls, "is_horse": is_horse,
                 "confidence": round(conf, 4), "bbox": xyxy}
            detections.append(d)
            if is_horse:
                # 体格 score = bbox 面積 / 画像面積
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                horses.append({
                    "confidence": round(conf, 4),
                    "bbox": [round(x, 1) for x in xyxy],
                    "bbox_area_px": round(w * h, 1),
                    "aspect_ratio": round(w / h if h > 0 else 0, 3),
                })

    return {
        "status": "ok",
        "inference_ms": round(inference_ms, 1),
        "n_detections": len(detections),
        "n_horses": len(horses),
        "horses": horses[:5],  # top 5
        "max_horse_conf": max([h["confidence"] for h in horses], default=0),
    }


def compute_paddock_features(detection: dict) -> dict:
    """検出結果から features 化."""
    horses = detection.get("horses", [])
    if not horses:
        return {
            "paddock_horse_detected": 0,
            "paddock_max_conf": 0.0,
            "paddock_body_size_score": 0.0,
            "paddock_aspect_ratio": 0.0,
            "paddock_pose_score": 0.0,
        }

    main = max(horses, key=lambda h: h["bbox_area_px"])
    return {
        "paddock_horse_detected": 1,
        "paddock_max_conf": main["confidence"],
        "paddock_body_size_score": min(1.0, main["bbox_area_px"] / 100000),  # normalize
        "paddock_aspect_ratio": main["aspect_ratio"],  # 1.5-2.5 が standing horse 標準
        "paddock_pose_score": 1.0 if 1.3 <= main["aspect_ratio"] <= 2.5 else 0.5,
    }


def cli():
    p = argparse.ArgumentParser(description="paddock_image_analyzer PoC (Session #47 G)")
    p.add_argument("--image", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    img_path = None
    if args.image:
        img_path = Path(args.image)
        if not img_path.is_absolute():
            img_path = BASE / args.image
    else:
        # default sample (Session #42 E download 済)
        candidates = [
            BASE / "data" / "video_poc" / "zidane.jpg",  # ない
            BASE / "data" / "video_poc" / "bus.jpg",     # ない
            BASE / "data" / "video_poc" / "sample_horse.jpg",
        ]
        for c in candidates:
            if c.exists():
                img_path = c
                break

    if not img_path or not img_path.exists():
        print("[!] image 不在、 --image でパス指定")
        print("    Phase 4 では netkeiba / JRA-VAN ネクスト パドック画像取得 plan")
        return

    print(f"=" * 60)
    print(f"paddock_image_analyzer PoC (Session #47 G)")
    print(f"target: {img_path.name}")
    print(f"=" * 60)

    result = detect_horse_in_image(img_path)
    print(f"\nDetection result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    features = compute_paddock_features(result)
    print(f"\nFeatures:")
    print(json.dumps(features, ensure_ascii=False, indent=2))

    out_path = BASE / (args.out or f"data/v18/sprint2_paddock_{img_path.stem}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "image": img_path.name,
        "detection": result,
        "features": features,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    cli()
