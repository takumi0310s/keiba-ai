"""5/9 重賞 5 system 合議 v2 (Session #52 D、 dev/training-poc).

Session #49 C の simulate System 5 → 実動画 features に置換。

5 system:
1. V15 単独 (production baseline)
2. V15 + 拡張調教 (Sprint 2)
3. V15 + TM 公式調教 (TFJV)
4. V15 + 当日体重 (Stage 2 proxy)
5. ★ V15 + 動画 features (Session #52 B+C 実 PoC) ★

★ 投票なし、 verdict 用学習データ ★

usage:
  python tools/predict_majors_5system_5_9_v2.py --date 20260509

V15 production 完全独立、 dev/training-poc 専用。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")


MAJORS_5_9 = [
    {"course": "東京", "race_num": 11, "race_name": "エプソムカップ (G3)", "start": "15:45"},
    {"course": "京都", "race_num": 11, "race_name": "京都新聞杯 (G2)", "start": "15:30"},
    {"course": "新潟", "race_num": 11, "race_name": "駿風 S (OP)", "start": "15:20"},
]


def load_v15_prediction(race_id: str) -> dict:
    p = BASE / "data" / "daily_predictions" / "20260509.csv"
    if not p.exists():
        return {"system": 1, "name": "V15 単独", "status": "deferred",
                "reason": "5/9 daily_predictions 未生成"}
    try:
        df = pd.read_csv(p, dtype=str)
        race_pred = df[df["race_id"].astype(str) == str(race_id)]
        if len(race_pred) == 0:
            return {"system": 1, "status": "missing"}
        row = race_pred.iloc[0]
        return {
            "system": 1, "name": "V15 単独", "status": "ok",
            "top1": row.get("top1_num", "?"),
            "top2": row.get("top2_num", "?"),
            "top3": row.get("top3_num", "?"),
        }
    except Exception as e:
        return {"system": 1, "status": "error", "error": str(e)[:120]}


def system_2_extended(race_id: str) -> dict:
    return {"system": 2, "name": "V15 + 拡張調教", "status": "deferred"}


def system_3_tm(race_id: str) -> dict:
    return {"system": 3, "name": "V15 + TM 公式", "status": "deferred"}


def system_4_weight(race_id: str) -> dict:
    return {"system": 4, "name": "V15 + 当日体重", "status": "deferred"}


def system_5_video_real(race_id: str) -> dict:
    """System 5: 実動画 features (Session #52 B+C 実行 後)."""
    motion_csv = BASE / "data" / "v18" / "horse_motion_5_9.csv"
    if not motion_csv.exists():
        return {
            "system": 5, "name": "V15 + 動画 features (実 PoC)",
            "status": "deferred",
            "reason": "5/9 朝 動画 DL + horse_motion_features 実行後 利用可"
        }

    try:
        df = pd.read_csv(motion_csv, encoding="utf-8-sig")
        # race_id 一致 行
        race_data = df[df["race_id"].astype(str).str.contains(str(race_id), na=False)]
        if len(race_data) == 0:
            # placeholder match (race_name fragment)
            return {
                "system": 5, "name": "V15 + 動画 features",
                "status": "no_data", "n_horses_with_video": 0,
            }

        return {
            "system": 5, "name": "V15 + 動画 features (実 PoC)",
            "status": "ok",
            "n_horses_with_video": len(race_data),
            "features_summary": {
                "stride_mean": round(race_data["stride_length_mean"].mean(), 2),
                "body_size_relative_mean": round(race_data["body_size_relative"].mean(), 4),
                "stability_mean": round(race_data["stability_score"].mean(), 4),
                "tension_mean": round(race_data["tension_score"].mean(), 4),
            },
            "design": "V15 prediction に 動画 features を 補正係数として加算 (Phase 4 で本格)",
        }
    except Exception as e:
        return {"system": 5, "status": "error", "error": str(e)[:120]}


def consensus_top1(predictions: list[dict]) -> dict:
    top1s = [p.get("top1") for p in predictions
             if p.get("status") == "ok" and p.get("top1") not in (None, "?")]

    if not top1s:
        return {"consensus_top1": None, "confidence": "deferred (5/9 朝 全 system 動作後)"}

    counter = Counter(top1s)
    most_common, count = counter.most_common(1)[0]
    if count == 5:
        confidence = "★★ 高信頼 (5/5) ★★"
    elif count == 4:
        confidence = "★ 中高 (4/5)"
    elif count == 3:
        confidence = "中 (3/5)"
    elif count == 2:
        confidence = "低 (2/5)"
    else:
        confidence = "不一致"

    return {
        "consensus_top1": most_common,
        "agreement_count": count,
        "n_systems_responding": len(top1s),
        "confidence": confidence,
        "all_top1s": dict(counter),
    }


def main():
    p = argparse.ArgumentParser(description="5 system v2 (Session #52 D)")
    p.add_argument("--date", default="20260509")
    p.add_argument("--out", default="data/v18/predictions_majors_5system_5_9_FINAL.json")
    args = p.parse_args()

    print("=" * 70)
    print(f"5/9 重賞 5 system v2 (Session #52 D、 実動画統合)")
    print("=" * 70)
    print("★★ 投票なし、 verdict 用学習データ ★★")

    all_results = []

    for major in MAJORS_5_9:
        race_id = f"placeholder_{major['course']}_11R"
        print(f"\n--- {major['course']} 11R {major['race_name']} ({major['start']}) ---")

        s1 = load_v15_prediction(race_id)
        s2 = system_2_extended(race_id)
        s3 = system_3_tm(race_id)
        s4 = system_4_weight(race_id)
        s5 = system_5_video_real(major["race_name"])  # race_name fragment match

        cons = consensus_top1([s1, s2, s3, s4, s5])

        for s, name in [(s1, "V15"), (s2, "拡張"), (s3, "TM"),
                        (s4, "体重"), (s5, "動画 ★実 PoC")]:
            print(f"  System {s.get('system')} ({name}): {s.get('status')}")
        print(f"  合議: {cons.get('confidence')}")

        all_results.append({
            "race": major,
            "predictions": [s1, s2, s3, s4, s5],
            "consensus": cons,
        })

    out_path = BASE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "date": args.date,
        "version": "5system_v2_with_real_video_features",
        "majors": all_results,
        "note": "★★ 5/9 重賞 投票なし ★★ Session #52 B+C 実動画 PoC 反映",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")
    print("\n★ 5/9 重賞 投票なし、 12R 1勝のみ V15 案B改 ★")


if __name__ == "__main__":
    main()
