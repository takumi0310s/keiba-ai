"""5/9 重賞 5 system 合議予測 (Session #49 C、 dev/training-poc).

5 system:
1. V15 単独 (production baseline)
2. V15 + 拡張調教 (Sprint 2)
3. V15 + TM 公式調教 (TFJV)
4. V15 + 当日体重 (Stage 2 proxy、 Session #48 B)
5. V15 + 動画 features (Session #49 B) ★新★

合議:
- 5/5 一致 → ★ 高信頼 ★
- 4/5 → 中 高
- 3/5 → 中
- 2/5 → 低
- 1 一致以下 → 不信

★ 投票なし、 verdict 用学習データ ★

usage:
  python tools/predict_majors_5system_5_9.py --date 20260509

V15 production 完全独立、 dev/training-poc 専用。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")


MAJORS_5_9 = [
    {"course": "東京", "race_num": 11, "race_name": "エプソムカップ (G3)", "start": "15:45"},
    {"course": "京都", "race_num": 11, "race_name": "京都新聞杯 (G2)", "start": "15:30"},
    {"course": "新潟", "race_num": 11, "race_name": "駿風 S (OP)", "start": "15:20"},
]


def load_v15_prediction(race_id: str) -> dict:
    """System 1: V15 単独."""
    p = BASE / "data" / "daily_predictions" / "20260509.csv"
    if not p.exists():
        return {"system": 1, "name": "V15 単独", "status": "deferred",
                "reason": "5/9 daily_predictions 未生成 (5/9 朝 08:00 自動)"}
    try:
        import pandas as pd
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


def system_2_extended_training(race_id: str) -> dict:
    """System 2: V15 + 拡張調教 (Sprint 2 features)."""
    return {
        "system": 2, "name": "V15 + 拡張調教 (Sprint 2)",
        "status": "deferred",
        "design": "Sprint 2 horse_weight + race_interval + running_style merge"
    }


def system_3_tm_official(race_id: str) -> dict:
    """System 3: V15 + TM 公式調教 (TFJV TM_DATA)."""
    return {
        "system": 3, "name": "V15 + TM 公式調教",
        "status": "deferred",
        "design": "TFJV TM_DATA から 公式調教タイム features"
    }


def system_4_race_day_weight(race_id: str) -> dict:
    """System 4: V15 + 当日体重 (Stage 2 proxy)."""
    return {
        "system": 4, "name": "V15 + 当日体重 (Stage 2 proxy)",
        "status": "deferred",
        "design": "5/9 各 R 70 分前 当日体重 polling、 Session #48 B"
    }


def system_5_video_features(race_id: str) -> dict:
    """System 5: V15 + 動画 features (Session #49 B 結果反映)."""
    p = BASE / "data" / "v18" / "video_features_5_9_majors.json"
    if not p.exists():
        return {
            "system": 5, "name": "V15 + 動画 features",
            "status": "deferred",
            "reason": "video_features 未生成 (5/9 朝 動画 DL 後 batch run)"
        }
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        n = len(data.get("results", []))
        return {
            "system": 5, "name": "V15 + 動画 features",
            "status": "deferred" if n == 0 else "partial",
            "n_videos_processed": n,
            "design": "video features (size/conf/aspect) を V15 prediction に補正係数として加算"
        }
    except Exception as e:
        return {"system": 5, "status": "error", "error": str(e)[:120]}


def consensus_top1(predictions: list[dict]) -> dict:
    """5 system top1 合議."""
    top1s = [p.get("top1") for p in predictions
             if p.get("status") == "ok" and p.get("top1") not in (None, "?")]

    if not top1s:
        return {
            "consensus_top1": None, "agreement_count": 0,
            "n_systems_responding": 0, "confidence": "deferred (5/9 朝 全 system 動作後)"
        }

    counter = Counter(top1s)
    most_common, count = counter.most_common(1)[0]
    if count == 5:
        confidence = "★★ 高信頼 (5/5 一致) ★★"
    elif count == 4:
        confidence = "★ 中高信頼 (4/5)"
    elif count == 3:
        confidence = "中信頼 (3/5)"
    elif count == 2:
        confidence = "低信頼 (2/5)"
    else:
        confidence = "不一致"

    return {
        "consensus_top1": most_common,
        "agreement_count": count,
        "n_systems_responding": len(top1s),
        "confidence": confidence,
        "all_top1s": dict(counter),
    }


def recommendation(consensus: dict) -> dict:
    """合議結果から「もし買うなら」 推奨買い目 (★ 投票しない、 verdict 用 ★)."""
    if not consensus.get("consensus_top1"):
        return {"if_buying": "skip (合議 不一致 or deferred)"}

    confidence = consensus.get("confidence", "")
    n = consensus.get("agreement_count", 0)

    if n >= 4:
        return {
            "if_buying": "馬連 軸 1 頭流し (top1 - top2,3)",
            "expected_ev": ">= 1.5 (高信頼時)",
            "note": "★ ただし 5/9 重賞は 投票しない (絶対遵守) ★",
        }
    if n == 3:
        return {
            "if_buying": "3連複 BOX (top3、 8 通り)",
            "expected_ev": "1.0-1.5",
            "note": "★ 投票しない ★",
        }
    return {
        "if_buying": "skip (合議 confidence 低)",
        "note": "★ 投票しない ★",
    }


def main():
    p = argparse.ArgumentParser(description="5/9 重賞 5 system 合議 (Session #49 C)")
    p.add_argument("--date", default="20260509")
    p.add_argument("--out", default="data/v18/recommendations_5_9_majors.json")
    args = p.parse_args()

    print("=" * 70)
    print(f"5/9 重賞 5 system 合議 ({args.date})")
    print("=" * 70)
    print("★★ 投票なし、 verdict 用学習データ ★★")

    all_results = []

    for major in MAJORS_5_9:
        race_id_placeholder = f"placeholder_{major['course']}_11R"
        print(f"\n--- {major['course']} 11R {major['race_name']} ({major['start']}) ---")

        s1 = load_v15_prediction(race_id_placeholder)
        s2 = system_2_extended_training(race_id_placeholder)
        s3 = system_3_tm_official(race_id_placeholder)
        s4 = system_4_race_day_weight(race_id_placeholder)
        s5 = system_5_video_features(race_id_placeholder)

        cons = consensus_top1([s1, s2, s3, s4, s5])
        rec = recommendation(cons)

        for s, name_short in [(s1, "V15"), (s2, "拡張調教"), (s3, "TM"),
                              (s4, "当日体重"), (s5, "動画 features ★")]:
            print(f"  System {s.get('system')} ({name_short}): {s.get('status')} top1={s.get('top1', 'N/A')}")
        print(f"  合議: {cons.get('confidence')} → top1 = {cons.get('consensus_top1')}")
        print(f"  recommendation: {rec.get('if_buying')}")
        print(f"  ★ {rec.get('note', '投票しない')} ★")

        all_results.append({
            "race": major,
            "predictions": [s1, s2, s3, s4, s5],
            "consensus": cons,
            "recommendation": rec,
        })

    out_path = BASE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "date": args.date,
        "majors": all_results,
        "note": "★★ 5/9 重賞 投票なし、 verdict + 5 system 比較学習データ ★★",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")
    print("\n★ 5/9 重賞 投票なし (絶対遵守)、 12R 1勝のみ V15 案B改 max 2,100円 ★")


if __name__ == "__main__":
    main()
