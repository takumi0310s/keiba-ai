"""5/9 重賞 5 system v62 (Session #62 F、 realistic simulate motion 統合).

Session #60 v60 の simulate (全馬 V15 一致) を改善:
- System 5 を realistic simulate (V15 ranking + motion 監視) で動作
- consensus 集計を V15 + simulate System 5 の 2 source で再計算
- top1/top2/top3 を 確定して Discord 通知用 JSON 出力

★ 投票なし ★ (5/9 朝 投票は 12R 1勝 ¥2,100 案B改 strict のみ)

usage:
  python tools/predict_majors_5system_5_9_v62.py

V15 production 完全独立、 dev/training-poc 専用。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")


MAJORS = [
    {"course": "東京", "race_num": 11, "race_name": "エプソムカップ (G3)",
     "start": "15:45", "race_id": "202605020511"},
    {"course": "京都", "race_num": 11, "race_name": "京都新聞杯 (G2)",
     "start": "15:30", "race_id": "202608030511"},
    {"course": "新潟", "race_num": 11, "race_name": "駿風 S (OP)",
     "start": "15:20", "race_id": "202604010311"},
]


def load_v15(race_id: str) -> dict:
    p = BASE / "data" / "daily_predictions" / "20260509.csv"
    if not p.exists():
        return {"system": 1, "name": "V15 単独", "status": "no_csv"}
    df = pd.read_csv(p, dtype=str)
    row = df[df["race_id"].astype(str) == str(race_id)]
    if len(row) == 0:
        return {"system": 1, "status": "missing"}
    r = row.iloc[0]
    return {
        "system": 1, "name": "V15 単独 (production baseline)", "status": "ok",
        "top1_num": r.get("top1_num"), "top1_name": r.get("top1_name"),
        "top2_num": r.get("top2_num"), "top2_name": r.get("top2_name"),
        "top3_num": r.get("top3_num"), "top3_name": r.get("top3_name"),
        "top1_score": float(r.get("top1_score") or 0),
        "trio_bets": r.get("trio_bets", ""),
        "num_horses": int(r.get("num_horses") or 0),
    }


def system_5_realistic_simulate(race_id: str, motion_csv: Path) -> dict:
    """realistic simulate motion features を読み込み、 V15 一致仮定 + variance 評価。"""
    if not motion_csv.exists():
        return {"system": 5, "status": "no_motion_csv"}
    df = pd.read_csv(motion_csv, encoding="utf-8-sig")
    sub = df[df["race_id"].astype(str) == str(race_id)]
    if len(sub) == 0:
        return {"system": 5, "status": "no_data_for_race"}
    # 降順 stability で top3 推定 (high stability = 安定 = 好走見込)
    sub = sub.sort_values("stability_score", ascending=False).reset_index(drop=True)
    top3 = sub.head(3)
    return {
        "system": 5, "name": "V15 + 動画 features (realistic simulate)",
        "status": "simulate_realistic",
        "reason": "Session #62 D で動画 DL 全失敗 (server 400)、 V15 ranking 反映の simulate",
        "n_horses_with_features": len(sub),
        "top1_num": str(int(top3.iloc[0]["umaban"])) if len(top3) >= 1 else None,
        "top2_num": str(int(top3.iloc[1]["umaban"])) if len(top3) >= 2 else None,
        "top3_num": str(int(top3.iloc[2]["umaban"])) if len(top3) >= 3 else None,
        "features_summary": {
            "stride_mean": round(float(sub["stride_length_mean"].mean()), 3),
            "stability_mean": round(float(sub["stability_score"].mean()), 4),
            "tension_mean": round(float(sub["tension_score"].mean()), 4),
        },
    }


def consensus(predictions: list[dict]) -> dict:
    top1s = [p.get("top1_num") for p in predictions
             if p.get("status") in ("ok", "simulate_realistic") and p.get("top1_num")]
    if not top1s:
        return {"consensus_top1": None, "confidence": "no_data"}
    counter = Counter(top1s)
    most_common, count = counter.most_common(1)[0]
    conf = ("★★ 高 (4+/5)" if count >= 4
            else "★ 中 (3/5)" if count == 3
            else "中低 (2/5)" if count == 2
            else "不一致 (1/5)")
    return {"consensus_top1": most_common, "agreement_count": count,
            "n_responding": len(top1s), "confidence": conf,
            "all_top1s": dict(counter)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--motion", default="data/v18/horse_motion_5_9_REAL.csv")
    p.add_argument("--out", default="data/v18/predictions_majors_5system_5_9_v62.json")
    args = p.parse_args()

    print("=" * 70)
    print("5/9 重賞 5 system v62 (realistic simulate motion 統合)")
    print("=" * 70)
    print("★ 投票なし、 verdict 用 ★")

    motion_csv = BASE / args.motion

    all_results = []
    for major in MAJORS:
        rid = major["race_id"]
        print(f"\n--- {major['course']} 11R {major['race_name']} ({major['start']}) ---")
        s1 = load_v15(rid)
        s2 = {"system": 2, "name": "V15 + 拡張調教", "status": "deferred"}
        s3 = {"system": 3, "name": "V15 + TM 公式", "status": "deferred"}
        s4 = {"system": 4, "name": "V15 + 当日体重", "status": "deferred"}
        s5 = system_5_realistic_simulate(rid, motion_csv)

        cons = consensus([s1, s2, s3, s4, s5])
        for s, lbl in [(s1, "V15"), (s2, "拡張"), (s3, "TM"),
                        (s4, "体重"), (s5, "動画 simulate")]:
            print(f"  {lbl}: {s.get('status')} top1={s.get('top1_num') or '-'}")
        print(f"  合議: {cons['confidence']} (top1={cons['consensus_top1']})")

        all_results.append({"race": major,
                            "predictions": [s1, s2, s3, s4, s5],
                            "consensus": cons})

    out_path = BASE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "version": "5system_v62_realistic_simulate",
        "branch": "dev/training-poc",
        "session": "Session #62 F",
        "date": "20260509",
        "majors": all_results,
        "note": "Session #62 D 動画 DL 0/3 (server 400)、 realistic simulate で完成、 投票なし",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")
    print("\n★ 5/9 投票: 12R 1勝 ¥2,100 (案B改 strict) のみ ★")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
