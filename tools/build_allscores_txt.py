"""全馬スコア .txt 生成ツール。

data/allscores/YYYYMMDD/*.json (1レース1ファイル) から
全レース全馬の V15/V16 スコアを整形した .txt を生成する。

usage:
    python tools/build_allscores_txt.py [--date YYYYMMDD]
    from tools.build_allscores_txt import build_allscores_txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ALLSCORES_DIR = BASE_DIR / "data" / "allscores"


def build_allscores_txt(date_str: str | None = None) -> str | None:
    """指定日の全馬スコア .txt を生成。生成したファイルパスを返す。

    data/allscores/YYYYMMDD/*.json を読み込み、
    data/allscores/allscores_YYYYMMDD.txt に書き出す。
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    race_dir = ALLSCORES_DIR / date_str
    if not race_dir.exists():
        print(f"[build_allscores_txt] No data dir: {race_dir}")
        return None

    race_files = sorted(race_dir.glob("*.json"))
    if not race_files:
        print(f"[build_allscores_txt] No race JSON files in {race_dir}")
        return None

    lines: list[str] = []
    lines.append(f"全馬スコア {date_str}  (V15=本番, V16=参考/投票しない)")
    lines.append("=" * 72)

    races_loaded = 0
    for rf in race_files:
        try:
            with open(rf, encoding="utf-8") as f:
                rec = json.load(f)
        except Exception as e:
            print(f"  [WARN] {rf.name}: {e}")
            continue

        race_name = rec.get("race_name", rf.stem)
        course = rec.get("course", "")
        race_num = rec.get("race_num", "")
        distance = rec.get("distance", "")
        surface = rec.get("surface", "")
        condition = rec.get("condition", "")
        cond_key = rec.get("cond_key", "")
        num_horses = rec.get("num_horses", 0)

        header = (
            f"\n【{course} {race_num}R】{race_name}"
            f"  {surface}{distance}m {condition}  条件{cond_key}  {num_horses}頭"
        )
        lines.append(header)
        lines.append("-" * 72)

        horses = rec.get("horses", [])
        v15_scores = rec.get("v15_scores", {})
        v16_scores = rec.get("v16_scores", {})

        # V15 ランキング
        v15_ranked = sorted(
            [(str(k), float(v)) for k, v in v15_scores.items()],
            key=lambda x: x[1], reverse=True
        )
        # V16 ランキング
        v16_ranked = sorted(
            [(str(k), float(v)) for k, v in v16_scores.items()],
            key=lambda x: x[1], reverse=True
        )
        # 馬名マップ
        name_map = {str(h.get("馬番", h.get("horse_num", ""))): h.get("馬名", h.get("name", "")) for h in horses}

        lines.append("  ▼ V15 (本番スコア順)")
        for rank, (uma, score) in enumerate(v15_ranked, 1):
            name = name_map.get(uma, "")
            v16_sc = v16_scores.get(uma, v16_scores.get(int(uma) if uma.isdigit() else uma))
            v16_str = f"  V16:{v16_sc:.4f}" if v16_sc is not None else ""
            lines.append(f"  {rank:2d}位  {uma:>2}番 {name:<12} V15:{score:.4f}{v16_str}")

        if v16_scores:
            lines.append("  ▼ V16 (参考スコア順, 投票しない)")
            for rank, (uma, score) in enumerate(v16_ranked, 1):
                name = name_map.get(uma, "")
                v15_sc = v15_scores.get(uma, v15_scores.get(int(uma) if uma.isdigit() else uma))
                v15_str = f"  V15:{v15_sc:.4f}" if v15_sc is not None else ""
                lines.append(f"  {rank:2d}位  {uma:>2}番 {name:<12} V16:{score:.4f}{v15_str}")

        races_loaded += 1

    lines.append(f"\n{'=' * 72}")
    lines.append(f"計 {races_loaded}レース  生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    out_path = ALLSCORES_DIR / f"allscores_{date_str}.txt"
    ALLSCORES_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[build_allscores_txt] {races_loaded}R → {out_path}")
    return str(out_path)


def save_race_allscores(
    date_str: str,
    race_id: str,
    race_name: str,
    course: str,
    race_num,
    distance: int,
    surface: str,
    condition: str,
    cond_key: str,
    num_horses: int,
    horses: list[dict],
    v15_scores: dict,
    v16_scores: dict | None = None,
) -> None:
    """1レース分のスコアを data/allscores/YYYYMMDD/<race_id>.json に保存。"""
    race_dir = ALLSCORES_DIR / date_str
    race_dir.mkdir(parents=True, exist_ok=True)

    rec = {
        "race_id": str(race_id),
        "race_name": race_name,
        "course": course,
        "race_num": str(race_num),
        "distance": distance,
        "surface": surface,
        "condition": condition,
        "cond_key": cond_key,
        "num_horses": num_horses,
        "horses": [
            {"馬番": h.get("馬番", h.get("horse_num", "")),
             "馬名": h.get("馬名", h.get("name", ""))}
            for h in horses
        ],
        "v15_scores": {str(k): round(float(v), 4) for k, v in v15_scores.items()},
        "v16_scores": {str(k): round(float(v), 4) for k, v in (v16_scores or {}).items()},
    }

    out = race_dir / f"{race_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全馬スコア .txt 生成")
    parser.add_argument("--date", default=None, help="YYYYMMDD (default: today)")
    args = parser.parse_args()
    path = build_allscores_txt(args.date)
    if path:
        print(f"Generated: {path}")
    else:
        print("No data.")
        sys.exit(1)
