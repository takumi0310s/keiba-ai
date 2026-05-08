"""YOLOv8 馬体検出 PoC (Session #42 E).

Phase 4 動画解析 の最初の step として、 YOLOv8 (ultralytics 8.4) の COCO
pretrained model で 馬体検出 (class 17 = horse) を試行。

確認項目:
1. ultralytics import + YOLO model load
2. 馬画像 sample (ImageNet horse class 等) で 物体検出
3. 検出精度 (confidence score) 評価
4. CUDA / CPU 速度比較

usage:
  python tools/video_poc/test_yolo_horse_detection.py
  python tools/video_poc/test_yolo_horse_detection.py --img <image_path>
  python tools/video_poc/test_yolo_horse_detection.py --download-sample

V15 production 完全不変 (新規 PoC)。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
SAMPLE_DIR = BASE / "data" / "video_poc"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def download_sample_image(out_path: Path) -> bool:
    """ImageNet horse sample 画像をダウンロード (PoC test 用)."""
    # ultralytics 公式 サンプル (zidane.jpg は人、 bus.jpg はバス、 horses は無いため自前生成)
    # 代替: Wikimedia direct URL ではなく ultralytics の test image
    url = "https://github.com/ultralytics/assets/raw/main/im/zidane.jpg"  # COCO test、 馬は写ってないが detection 動作確認用
    print(f"[poc] download sample: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        print(f"  saved: {out_path} ({len(data)/1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"  [ERROR] download failed: {e}")
        return False


def detect_horse_yolo(img_path: Path, conf_threshold: float = 0.25) -> dict:
    """YOLOv8 で 馬体検出 (COCO class 17)."""
    from ultralytics import YOLO

    print(f"\n[poc] loading YOLOv8 nano model ...")
    t0 = time.time()
    # YOLOv8n は約 6 MB、 ultralytics 初回 download
    model = YOLO('yolov8n.pt')
    print(f"  load time: {time.time()-t0:.2f}s")

    print(f"\n[poc] running inference on {img_path.name} (device=cpu, torchvision CUDA bug 回避)")
    t1 = time.time()
    results = model(str(img_path), conf=conf_threshold, verbose=False, device='cpu')
    inference_time = time.time() - t1
    print(f"  inference time: {inference_time*1000:.1f} ms")

    # 検出結果 解析
    detections = []
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            cls = int(box.cls[0]) if box.cls is not None else -1
            conf = float(box.conf[0]) if box.conf is not None else 0
            xyxy = box.xyxy[0].tolist() if box.xyxy is not None else [0,0,0,0]
            class_name = model.names.get(cls, f"class_{cls}")
            detections.append({
                "class_id": cls,
                "class_name": class_name,
                "confidence": round(conf, 4),
                "bbox": [round(x, 1) for x in xyxy],
                "is_horse": cls == 17,
            })

    horses = [d for d in detections if d["is_horse"]]
    return {
        "image": str(img_path.name),
        "inference_ms": round(inference_time*1000, 1),
        "n_detections": len(detections),
        "n_horses": len(horses),
        "max_horse_conf": max([h["confidence"] for h in horses], default=0),
        "all_classes": list(set(d["class_name"] for d in detections)),
        "detections": detections[:20],  # top 20 for log
    }


def main():
    p = argparse.ArgumentParser(description="YOLOv8 馬体検出 PoC (Session #42 E)")
    p.add_argument("--img", default=None, help="image path (省略時 sample download)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--download-sample", action="store_true")
    args = p.parse_args()

    # arch + GPU 確認
    print("=" * 60)
    print("YOLOv8 馬体検出 PoC (Session #42 E)")
    print("=" * 60)
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  [WARN] PyTorch 未 install")

    try:
        import ultralytics
        print(f"  ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("  [ERROR] ultralytics 未 install。 pip install ultralytics")
        sys.exit(1)

    # 画像 入手
    if args.img:
        img_path = Path(args.img)
    else:
        img_path = SAMPLE_DIR / "sample_horse.jpg"
        if args.download_sample or not img_path.exists():
            ok = download_sample_image(img_path)
            if not ok:
                sys.exit(2)

    if not img_path.exists():
        print(f"[ERROR] image not found: {img_path}")
        sys.exit(3)
    print(f"\n  target: {img_path} ({img_path.stat().st_size/1024:.1f} KB)")

    # 推論実行
    result = detect_horse_yolo(img_path, args.conf)

    # 結果表示
    print(f"\n=== 検出結果 ===")
    print(f"  inference: {result['inference_ms']:.1f} ms")
    print(f"  total detections: {result['n_detections']}")
    print(f"  horse detections: {result['n_horses']}")
    print(f"  max horse conf: {result['max_horse_conf']:.4f}")
    print(f"  classes detected: {result['all_classes']}")
    print()
    print("  top detections (class, conf, bbox):")
    for d in result['detections'][:5]:
        mark = "★" if d['is_horse'] else "  "
        print(f"    {mark} {d['class_name']:15s} conf={d['confidence']:.4f}  bbox={d['bbox']}")

    # 保存
    import json
    out_json = SAMPLE_DIR / f"yolo_result_{img_path.stem}.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  result saved: {out_json.relative_to(BASE)}")

    if result['n_horses'] > 0:
        print(f"\n  [OK] YOLOv8 で 馬体検出 成功 (max conf {result['max_horse_conf']:.4f})")
    else:
        print(f"\n  [WARN] 馬が検出されませんでした (conf {args.conf} 以上)")


if __name__ == "__main__":
    main()
