"""5/9 重賞 5 system 合議 v60 (Session #60 C、 実 race_id 対応 + simulate motion).

Session #52 D の予測スクリプトは race_id placeholder で V15 prediction が見つからない
バグがあったため、 5/9 朝の本実行用に下記を本版で確定:

1. real race_id を使って data/daily_predictions/20260509.csv から V15 予測ロード
2. 動画 DL 失敗 (Session #60 B 確認) → simulate motion CSV 生成
3. System 2 / 3 / 4 は 5/9 時点 deferred (Phase 3 待ち)、 status のみ返す
4. System 5 は simulate 値で動作 (Phase 4 で本格化)
5. 合議 → 各重賞 top1/top2/top3、 verdict 用 JSON 出力

★★ 投票なし、 5/9 朝 12R 1勝 ¥2,100 (案B改) のみ ★★

usage:
  python tools/predict_majors_5system_5_9_v60.py
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")


MAJORS_5_9 = [
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
    try:
        df = pd.read_csv(p, dtype=str)
        race_pred = df[df["race_id"].astype(str) == str(race_id)]
        if len(race_pred) == 0:
            return {"system": 1, "name": "V15 単独", "status": "missing",
                    "race_id": race_id}
        row = race_pred.iloc[0]
        return {
            "system": 1, "name": "V15 単独 (production baseline)", "status": "ok",
            "top1_num": row.get("top1_num", "?"), "top1_name": row.get("top1_name", "?"),
            "top2_num": row.get("top2_num", "?"), "top2_name": row.get("top2_name", "?"),
            "top3_num": row.get("top3_num", "?"), "top3_name": row.get("top3_name", "?"),
            "top1_score": float(row.get("top1_score", 0) or 0),
            "trio_bets": row.get("trio_bets", ""),
        }
    except Exception as e:
        return {"system": 1, "status": "error", "error": str(e)[:120]}


def system_2(race_id: str) -> dict:
    return {"system": 2, "name": "V15 + 拡張調教",
            "status": "deferred", "reason": "Sprint 2 features 統合 待ち"}


def system_3(race_id: str) -> dict:
    return {"system": 3, "name": "V15 + TM 公式調教",
            "status": "deferred", "reason": "TFJV TCOV 統合 (Phase 3) 待ち"}


def system_4(race_id: str) -> dict:
    return {"system": 4, "name": "V15 + 当日体重 (Stage 2 proxy)",
            "status": "deferred", "reason": "9:30 morning weight check 結果 待ち"}


def system_5_simulate(race_id: str, v15_top1_num: str | None) -> dict:
    """Session #60 B で動画 DL 失敗 (HTTP 400) のため simulate モード。

    simulate logic: V15 top1 を base にして、 motion features 値を擬似生成。
    実 PoC ではない (Phase 4 で video DL 実 impl 後に置換)。
    """
    rng = random.Random(int(race_id))
    feats = {
        "stride_length_mean": round(2.5 + rng.random() * 0.5, 2),
        "body_size_relative": round(0.45 + rng.random() * 0.1, 4),
        "stability_score": round(0.85 + rng.random() * 0.1, 4),
        "tension_score": round(0.15 + rng.random() * 0.1, 4),
    }
    return {
        "system": 5, "name": "V15 + 動画 features (simulate)",
        "status": "simulate",
        "reason": "Session #60 B で動画 DL 全失敗 (HTTP 400)、 simulate 値使用",
        "features_simulated": feats,
        "predicted_top1_num": v15_top1_num,  # simulate では V15 と一致と仮定
        "design_note": "実 PoC は Phase 4 (Playwright + YOLOv8) で実装",
    }


def consensus(predictions: list[dict]) -> dict:
    top1s = [p.get("top1_num") or p.get("predicted_top1_num")
             for p in predictions
             if p.get("status") in ("ok", "simulate")
             and (p.get("top1_num") or p.get("predicted_top1_num")) not in (None, "?")]

    if not top1s:
        return {"consensus_top1": None, "confidence": "no_data"}

    counter = Counter(top1s)
    most_common, count = counter.most_common(1)[0]
    n_responding = len(top1s)
    if count >= 4:
        confidence = "★★ 高信頼 (4+/5)"
    elif count == 3:
        confidence = "★ 中 (3/5)"
    elif count == 2:
        confidence = "低 (2/5)"
    else:
        confidence = "不一致"

    return {
        "consensus_top1": most_common,
        "agreement_count": count,
        "n_systems_responding": n_responding,
        "confidence": confidence,
        "all_top1s": dict(counter),
    }


def write_simulate_motion_csv(out: Path) -> None:
    """動画 DL 失敗のため simulate motion CSV を生成 (downstream tool 互換)."""
    rows = []
    for m in MAJORS_5_9:
        rng = random.Random(int(m["race_id"]))
        # 各 race の top3 馬分 (V15 から 取得しても良いが、 PoC simulate のため固定 3 件)
        for i in range(3):
            rows.append({
                "race_id": m["race_id"],
                "horse_id": f"sim_{m['race_id']}_{i+1}",
                "stride_length_mean": round(2.4 + rng.random() * 0.6, 2),
                "body_size_relative": round(0.4 + rng.random() * 0.15, 4),
                "stability_score": round(0.8 + rng.random() * 0.15, 4),
                "tension_score": round(0.1 + rng.random() * 0.15, 4),
                "n_bboxes": 0,
                "n_frames_with_horse": 0,
                "source": "simulate_session60",
            })
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  simulate motion csv: {out} ({len(rows)} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="20260509")
    p.add_argument("--out", default="data/v18/predictions_majors_5system_5_9_FINAL.json")
    p.add_argument("--motion-out", default="data/v18/horse_motion_5_9.csv")
    args = p.parse_args()

    print("=" * 70)
    print(f"5/9 重賞 5 system v60 (Session #60 C、 simulate モード)")
    print("=" * 70)
    print("★★ 投票なし、 verdict 用学習データ ★★")

    # simulate motion csv (downstream互換)
    write_simulate_motion_csv(BASE / args.motion_out)

    all_results = []
    for major in MAJORS_5_9:
        race_id = major["race_id"]
        print(f"\n--- {major['course']} {major['race_num']}R "
              f"{major['race_name']} ({major['start']}) [race_id={race_id}] ---")

        s1 = load_v15(race_id)
        s2 = system_2(race_id)
        s3 = system_3(race_id)
        s4 = system_4(race_id)
        s5 = system_5_simulate(race_id, s1.get("top1_num"))

        cons = consensus([s1, s2, s3, s4, s5])

        for s, label in [(s1, "V15"), (s2, "拡張"), (s3, "TM"),
                          (s4, "体重"), (s5, "動画 simulate")]:
            top1 = s.get("top1_num") or s.get("predicted_top1_num") or "-"
            print(f"  System {s.get('system')} ({label}): {s.get('status')} top1={top1}")
        print(f"  合議: {cons.get('confidence')} (consensus_top1={cons.get('consensus_top1')})")

        all_results.append({
            "race": major,
            "predictions": [s1, s2, s3, s4, s5],
            "consensus": cons,
        })

    out_path = BASE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "date": args.date,
        "version": "5system_v60_simulate_motion",
        "branch": "dev/training-poc",
        "majors": all_results,
        "note": "Session #60 C - 動画 DL 失敗のため System 5 は simulate、 投票なし",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")
    print("\n★ 5/9 重賞 投票なし、 12R 1勝 ¥2,100 (案B改) のみ ★")


if __name__ == "__main__":
    main()
