"""5/9 重賞 4 system 予測 (Session #48 D、 dev/training-poc).

5/9 (土) 重賞 3 R で 4 system 比較予測 (★ 投票なし、 学習用 ★):
- 東京 11R エプソムカップ (G3) 15:45
- 京都 11R 京都新聞杯 (G2) 15:30
- 新潟 11R 駿風 S (OP) 15:20

4 system:
1. V15 単独 (production current、 baseline)
2. V15 + 拡張調教 (Session #47 features)
3. V15 + TM 公式調教 (TFJV TM_DATA)
4. V15 + 全部 + 当日体重 (Stage 2) + パドック (Phase 4 candidate)

合議 logic:
- 4 system top1 が 一致 → 高信頼 ★
- 3 system top1 一致 → 中信頼
- 2 以下 → 低信頼

★★ 重賞 3 R は 投票しない、 verdict のみ ★★

usage:
  python tools/predict_majors_4system_5_9.py --date 20260509 --no-discord

V15 production 完全独立、 dev/training-poc 専用。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")


# 5/9 重賞 3 R (race_id pattern: 2026 + course_code + kai + nichi + race_num)
# 東京 = 05、 京都 = 08、 新潟 = 04
MAJORS_5_9 = [
    {"course": "東京", "race_num": 11, "race_name": "エプソムカップ (G3)", "start": "15:45"},
    {"course": "京都", "race_num": 11, "race_name": "京都新聞杯 (G2)", "start": "15:30"},
    {"course": "新潟", "race_num": 11, "race_name": "駿風 S (OP)", "start": "15:20"},
]


def predict_system_1_v15(race_id: str) -> dict:
    """System 1: V15 単独 (production current).

    本実装では既存 daily_predictions/ から読み込み。
    """
    p = BASE / "data" / "daily_predictions" / "20260509.csv"
    if not p.exists():
        return {"system": 1, "name": "V15 単独", "status": "deferred", "reason": "5/9 daily_predictions 未生成 (5/8 21:00 後)"}
    try:
        import pandas as pd
        df = pd.read_csv(p, dtype=str)
        race_pred = df[df["race_id"].astype(str) == str(race_id)]
        if len(race_pred) == 0:
            return {"system": 1, "name": "V15 単独", "status": "missing", "race_id": race_id}
        # top3 (top1_num / top2_num / top3_num)
        row = race_pred.iloc[0]
        return {
            "system": 1,
            "name": "V15 単独 (production)",
            "status": "ok",
            "race_id": race_id,
            "top1": row.get("top1_num", "?"),
            "top2": row.get("top2_num", "?"),
            "top3": row.get("top3_num", "?"),
        }
    except Exception as e:
        return {"system": 1, "name": "V15 単独", "status": "error", "error": str(e)[:120]}


def predict_system_2_extended_training(race_id: str) -> dict:
    """System 2: V15 + 拡張調教 (Session #47 features)."""
    return {
        "system": 2,
        "name": "V15 + 拡張調教 (Sprint 2 features)",
        "status": "deferred",
        "race_id": race_id,
        "design": "Sprint 2 horse_weight + race_interval + running_style features を merge",
        "expected_top1": "(System 1 と同じ or 微妙に異なる)",
    }


def predict_system_3_tm_official(race_id: str) -> dict:
    """System 3: V15 + TM 公式調教 (TFJV TM_DATA)."""
    return {
        "system": 3,
        "name": "V15 + TM 公式調教 (TFJV)",
        "status": "deferred",
        "race_id": race_id,
        "design": "TFJV TM_DATA から 各馬の公式調教タイムを features 化",
        "feature_count_added": "5-10 (距離別 best time、 ratings 等)",
    }


def predict_system_4_full(race_id: str) -> dict:
    """System 4: V15 + 全部 + Stage 2 当日体重 + パドック PoC."""
    return {
        "system": 4,
        "name": "V15 + 全部 + Stage 2 + パドック (Phase 4)",
        "status": "deferred",
        "race_id": race_id,
        "design": "Stage 2 (Session #48 B) + 動画 features (Session #48 C) 統合",
        "trigger": "各 R 65 分前 (5/16+ schtasks)、 5/9 では未稼働",
    }


def consensus_top1(predictions: list[dict]) -> dict:
    """4 system の top1 合議."""
    top1s = [p.get("top1") for p in predictions if p.get("status") == "ok" and p.get("top1") not in (None, "?")]
    if not top1s:
        return {"consensus_top1": None, "agreement_count": 0, "confidence": "low"}

    counter = Counter(top1s)
    most_common, count = counter.most_common(1)[0]
    if count == 4:
        confidence = "★ 高信頼 (4/4 一致)"
    elif count == 3:
        confidence = "中信頼 (3/4 一致)"
    elif count == 2:
        confidence = "低信頼 (2/4 一致)"
    else:
        confidence = "不一致 (各 system 異なる)"

    return {
        "consensus_top1": most_common,
        "agreement_count": count,
        "n_systems_predicted": len(top1s),
        "confidence": confidence,
        "all_top1s": dict(counter),
    }


def main():
    p = argparse.ArgumentParser(description="5/9 重賞 4 system 予測 (Session #48 D)")
    p.add_argument("--date", default="20260509")
    p.add_argument("--no-discord", action="store_true")
    p.add_argument("--out", default="data/v18/predictions_5_9_majors_4system.json")
    args = p.parse_args()

    print("=" * 70)
    print(f"5/9 重賞 4 system 予測 ({args.date})")
    print("=" * 70)
    print("★★ 投票なし、 verdict 用学習データ ★★")

    all_results = []

    for major in MAJORS_5_9:
        # race_id 構築 (実際は daily_predict が生成、 ここでは placeholder)
        # 5/9 East/Kyoto/Niigata の course_code は確認必要
        # placeholder format: 2026 + course (XX) + kai (01) + nichi (??) + 11
        race_id_placeholder = f"placeholder_{major['course']}_11R"

        print(f"\n--- {major['course']} 11R {major['race_name']} ({major['start']}) ---")

        s1 = predict_system_1_v15(race_id_placeholder)
        s2 = predict_system_2_extended_training(race_id_placeholder)
        s3 = predict_system_3_tm_official(race_id_placeholder)
        s4 = predict_system_4_full(race_id_placeholder)

        cons = consensus_top1([s1, s2, s3, s4])

        print(f"  System 1 (V15): {s1.get('status')} top1={s1.get('top1', 'N/A')}")
        print(f"  System 2 (拡張): {s2.get('status')}")
        print(f"  System 3 (TM): {s3.get('status')}")
        print(f"  System 4 (全部): {s4.get('status')}")
        print(f"  合議: {cons.get('confidence')} → top1 = {cons.get('consensus_top1')}")

        all_results.append({
            "race": major,
            "predictions": [s1, s2, s3, s4],
            "consensus": cons,
        })

    out_path = BASE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "date": args.date,
        "majors": all_results,
        "note": "★ 投票なし ★ verdict + 4 system 比較学習データ。 5/9 各 R 終了 5 分後 verdict 通知。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")
    print("\n★ 重賞 3 R は 投票なし、 verdict のみ (絶対遵守) ★")


if __name__ == "__main__":
    main()
