"""二段階予測 system (Session #48 B、 dev/two-stage).

- Stage 1 (朝 08:00、 現状 daily_predict 不変)
- Stage 2 (各 R 70 分前): 当日馬体重を反映して再予測

Stage 1 vs Stage 2 の差分を Discord 通知。

usage:
  # Stage 2 単独実行 (race 70 分前)
  python tools/two_stage_predict.py --race-id 202605020412

  # 全 R で Stage 2 比較 (5/15+ schtasks 想定)
  python tools/two_stage_predict.py --date 20260509

V15 production 完全独立、 dev/two-stage 専用。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")


def load_stage1_predictions(date: str) -> dict:
    """朝予測 (Stage 1) の結果を読み込み."""
    p = BASE / "data" / "daily_predictions" / f"{date}.csv"
    if not p.exists():
        return {"available": False, "path": str(p)}
    try:
        import pandas as pd
        df = pd.read_csv(p, dtype=str)
        out = {"available": True, "races": {}}
        for race_id, group in df.groupby("race_id"):
            out["races"][race_id] = {
                "n_horses": len(group),
                "race_name": group.iloc[0].get("race_name", "") if "race_name" in group.columns else "",
                "course": group.iloc[0].get("course", "") if "course" in group.columns else "",
                "race_num": group.iloc[0].get("race_num", "") if "race_num" in group.columns else "",
            }
        return out
    except Exception as e:
        return {"available": False, "error": str(e)[:120]}


def fetch_current_weights(race_id: str) -> dict:
    """各 R 70 分前の当日馬体重 取得 (本 Session では design のみ).

    実 production では:
    - JV-Link WF datatype (公式、 リアルタイム)
    - or netkeiba 出馬表 update (各 R 70 分前 反映)

    本 Session では deferred、 5/16+ 実装。
    """
    return {
        "status": "deferred",
        "design": {
            "primary": "JV-Link WF (公式、 5/16+ 実装)",
            "fallback": "netkeiba 出馬表 polling (70 分前)",
            "trigger": "schtasks 各 R 65 分前 (5/16 以降 admin 追加)",
        },
        "race_id": race_id,
    }


def stage2_predict(race_id: str, current_weights: dict = None) -> dict:
    """Stage 2 予測 (当日体重 反映).

    本 Session では design + skeleton。 実装は 5/16+ V18 trial 後。
    """
    return {
        "status": "deferred",
        "race_id": race_id,
        "design": {
            "step1": "predict_core で V15 features 構築 (Stage 1 と同様)",
            "step2": "race_day_weight_features.py で当日体重 features 計算",
            "step3": "V15 model に features 追加して predict",
            "step4": "Stage 1 との top1/top2/top3 差分計算",
            "step5": "差分 > threshold なら Discord 通知",
        },
        "expected_features_added": [
            "current_weight",
            "weight_change_kg",
            "weight_change_pct",
            "weight_vs_3r_avg",
            "weight_vs_same_dist_avg",
        ],
    }


def compare_stage1_stage2(s1: dict, s2: dict) -> dict:
    """Stage 1 vs Stage 2 の差分."""
    if not s1.get("available") or s2.get("status") == "deferred":
        return {"status": "deferred or partial"}
    # design only
    return {
        "status": "design",
        "logic": "top1 prob 変化、 top1 馬番 変化、 ranking shift 計算",
    }


def main():
    p = argparse.ArgumentParser(description="two_stage_predict (Session #48 B)")
    p.add_argument("--date", default=None)
    p.add_argument("--race-id", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    print("=" * 60)
    print("two_stage_predict (Session #48 B、 dev/two-stage)")
    print("=" * 60)

    if args.date:
        s1 = load_stage1_predictions(args.date)
        print(f"\n[Stage 1] date {args.date}:")
        print(f"  available: {s1.get('available')}")
        if s1.get("available"):
            print(f"  races: {len(s1['races'])}")

        # Stage 2 (deferred)
        if s1.get("available") and s1["races"]:
            sample_race = list(s1["races"].keys())[0]
            s2 = stage2_predict(sample_race)
            print(f"\n[Stage 2] sample race {sample_race}:")
            print(f"  status: {s2.get('status')}")
            print(f"  design: {s2.get('design')}")

            cmp = compare_stage1_stage2(s1, s2)
            print(f"\n[Compare]: {cmp.get('status')}")

    elif args.race_id:
        weights = fetch_current_weights(args.race_id)
        print(f"\n[fetch_current_weights] {weights}")

        s2 = stage2_predict(args.race_id, weights)
        print(f"\n[Stage 2] {s2.get('status')}")

    summary = {
        "date": args.date,
        "race_id": args.race_id,
        "stage1": s1 if args.date else None,
        "note": "Stage 2 production trigger は 5/16+ admin schtasks (各 R 65 分前)",
    }

    out_path = BASE / (args.out or "data/v18/session_48_two_stage_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
