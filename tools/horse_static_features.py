"""Session #63 C: 静止画 YOLOv8 features 抽出.

5/9 静止画 DL は Session #63 B で全失敗 (netkeiba server block) のため、
本 script は **defensive fallback** として実装:
  1. data/v18/static_5_9/<race>/<horse>.jpg を Glob
  2. ファイル存在馬 → YOLOv8 推論 → bbox features
  3. ファイル不在馬 → NaN

5/9 当日は全馬 NaN になる想定だが、 5/16+ で manual DL or server 復旧時に
直ちに利用可。

usage:
  python tools/horse_static_features.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
STATIC_DIR = BASE / "data" / "v18" / "static_5_9"
OUT_CSV = BASE / "data" / "v18" / "horse_static_features_5_9.csv"
OUT_DOC = BASE / "data" / "v18" / "session_63_yolo_features.md"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

YOLO_MODEL_PATH = BASE / "yolov8n.pt"
HORSE_CLASS_ID = 17  # COCO class 17 = horse


def extract_features(jpg_path: Path):
    """YOLOv8 推論で 1 枚から features 抽出."""
    try:
        from ultralytics import YOLO
        from PIL import Image
        import numpy as np
    except ImportError as e:
        return None, f"import_fail: {e}"

    try:
        model = _get_model()
        results = model(str(jpg_path), classes=[HORSE_CLASS_ID], verbose=False)
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            return None, "no_horse_detected"
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        idx = int(confs.argmax())
        x1, y1, x2, y2 = boxes[idx]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        # body_size_relative は 同 R 内 percentile 化を後段で適用。 ここでは bbox area
        bbox_area = float(w * h)
        pose_score = float(w / h)  # 縦横比、 立ち姿 (1 前後で安定)

        # coat_score: bbox 内の HSV S 平均
        img = Image.open(jpg_path).convert("HSV")
        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
        arr = np.array(crop)
        coat_score = float(arr[:, :, 1].mean()) / 255.0
        return {
            "bbox_area": bbox_area,
            "pose_score": pose_score,
            "coat_score": coat_score,
            "conf": float(confs[idx]),
        }, "ok"
    except Exception as e:
        return None, f"exception: {type(e).__name__}: {e}"


_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(str(YOLO_MODEL_PATH))
    return _MODEL


def main():
    rows = []
    n_total = 0
    n_ok = 0
    n_fail = 0
    n_none = 0

    if not STATIC_DIR.exists():
        print(f"[skip] {STATIC_DIR} not found — 全馬 NaN")
    else:
        for race_dir in sorted(STATIC_DIR.iterdir()):
            if not race_dir.is_dir():
                continue
            rid = race_dir.name
            for jpg in sorted(race_dir.glob("*.jpg")):
                hid = jpg.stem
                n_total += 1
                feats, status = extract_features(jpg)
                if feats:
                    n_ok += 1
                    rows.append({
                        "race_id": rid, "horse_id": hid,
                        "bbox_area": feats["bbox_area"],
                        "pose_score": feats["pose_score"],
                        "coat_score": feats["coat_score"],
                        "conf": feats["conf"],
                        "status": status,
                    })
                else:
                    n_fail += 1
                    rows.append({
                        "race_id": rid, "horse_id": hid,
                        "bbox_area": "", "pose_score": "", "coat_score": "",
                        "conf": "", "status": status,
                    })

    # body_size_relative は 同 race 内で percentile 化
    if rows:
        from collections import defaultdict
        by_race = defaultdict(list)
        for r in rows:
            if r["bbox_area"] != "":
                by_race[r["race_id"]].append((r["horse_id"], r["bbox_area"]))
        for rid, lst in by_race.items():
            lst.sort(key=lambda x: x[1])
            n = len(lst)
            pct = {hid: (i + 1) / n for i, (hid, _) in enumerate(lst)}
            for r in rows:
                if r["race_id"] == rid and r["horse_id"] in pct:
                    r["body_size_relative"] = pct[r["horse_id"]]
        for r in rows:
            r.setdefault("body_size_relative", "")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        cols = ["race_id", "horse_id", "body_size_relative",
                "bbox_area", "pose_score", "coat_score", "conf", "status"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    if n_total == 0:
        n_none = 1
    print(f"\n=== 静止画 features: total={n_total}, ok={n_ok}, fail={n_fail} ===")
    print(f"csv: {OUT_CSV.relative_to(BASE)} ({len(rows)} rows)")

    doc_lines = [
        "# Session #63 C: 静止画 YOLOv8 features 抽出 結果",
        "",
        f"対象: {STATIC_DIR}",
        f"画像数: {n_total}",
        f"OK: {n_ok}",
        f"FAIL: {n_fail}",
        "",
    ]
    if n_total == 0:
        doc_lines += [
            "## 静止画 0 枚 (Session #63 B 全 fail のため)",
            "",
            "→ 全馬 NaN、 数値のみで scoring に降格 (E 段階)",
            "→ csv は schema のみ (空)、 5/16+ で 静止画 入手次第 即運用可",
        ]
    else:
        doc_lines += [
            "## 出力",
            "- csv: data/v18/horse_static_features_5_9.csv",
            "- columns: race_id, horse_id, body_size_relative (同 R percentile),",
            "  bbox_area, pose_score, coat_score, conf, status",
        ]
    OUT_DOC.write_text("\n".join(doc_lines), encoding="utf-8")
    print(f"doc: {OUT_DOC.relative_to(BASE)}")


if __name__ == "__main__":
    main()
