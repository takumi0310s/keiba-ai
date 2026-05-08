"""5/10 朝 重賞 5 system verdict 拡張 (Session #49 D、 dev/training-poc).

5/9 重賞 3R 結果 + 5 system 予測 を比較:
- 各 system 単独の hit_rate / top1_rate
- 動画 features の貢献度
- 「もし重賞 buy したら ROI X.XX%」 算出
- 5 system 合議の高信頼 R での hit_rate

★ 5/9 重賞 投票なし、 verdict 学習データ ★

usage:
  python tools/verify_majors_5system_5_10.py --date 20260509

V15 production 完全独立、 dev/training-poc 専用。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")


def load_majors_predictions() -> dict:
    """Session #49 C 出力 (5 system 予測) を load."""
    p = BASE / "data" / "v18" / "recommendations_5_9_majors.json"
    if not p.exists():
        return {"available": False, "path": str(p)}
    return json.loads(p.read_text(encoding="utf-8"))


def load_actual_results(date: str) -> dict:
    """5/9 結果 (daily_results) を load."""
    p = BASE / "data" / "daily_results" / f"{date}.csv"
    if not p.exists():
        return {"available": False, "path": str(p)}
    try:
        df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
        # 11R 重賞 抽出
        df["race_num_int"] = pd.to_numeric(df.get("race_num"), errors="coerce")
        majors = df[df["race_num_int"] == 11]
        return {"available": True, "n_majors": len(majors), "df": majors}
    except Exception as e:
        return {"available": False, "error": str(e)[:120]}


def evaluate_system(predictions: list, actual: dict) -> dict:
    """各 system の hit_rate を計算 (5 system 個別)."""
    if not actual.get("available"):
        return {"status": "results_unavailable"}

    df = actual["df"]
    by_system = {}

    for sys_idx in range(1, 6):
        n_correct = 0
        n_total = 0
        for major in predictions.get("majors", []):
            cons = major.get("consensus", {})
            preds = major.get("predictions", [])
            if sys_idx > len(preds): continue
            sys_pred = preds[sys_idx - 1]
            if sys_pred.get("status") != "ok": continue
            top1 = sys_pred.get("top1")
            # actual top1 finish
            race_data = df[df.get("course") == major["race"]["course"]]
            if len(race_data) == 0: continue
            actual_top1 = race_data.iloc[0].get("top1_finish_num")
            n_total += 1
            if str(top1) == str(actual_top1):
                n_correct += 1
        by_system[f"system_{sys_idx}"] = {
            "n_total": n_total,
            "n_correct_top1": n_correct,
            "top1_rate_pct": round(n_correct / n_total * 100, 2) if n_total > 0 else 0,
        }

    return {"status": "ok", "by_system": by_system}


def consensus_evaluation(predictions: list, actual: dict) -> dict:
    """合議 高信頼時 の hit_rate."""
    if not actual.get("available"):
        return {"status": "deferred"}

    high_conf_results = []
    for major in predictions.get("majors", []):
        cons = major.get("consensus", {})
        n = cons.get("agreement_count", 0)
        if n >= 4:  # 高信頼 4-5/5
            high_conf_results.append({
                "race": major["race"]["race_name"],
                "consensus_top1": cons.get("consensus_top1"),
                "agreement": n,
                "actual_top1": "(5/10 朝に集計)",
            })

    return {
        "status": "design_ok" if not actual.get("available") else "ok",
        "n_high_confidence": len(high_conf_results),
        "details": high_conf_results,
    }


def estimate_roi_if_bought(predictions: list, actual: dict) -> dict:
    """「もし重賞 buy したら」 想定 ROI."""
    return {
        "status": "deferred",
        "design": {
            "scenario": "全 3 重賞 高信頼時 馬連 軸 1 頭流し 2 点 (200 円/R)",
            "max_loss": "600 円 (3 R × 200 円、 全外し時)",
            "expected_max_payout": "300-1500 円/R hit (G2/G3 馬連)",
            "calc": "5/10 朝 結果照合後 自動計算",
        },
        "note": "★ 5/9 重賞 投票なし、 計算のみ (絶対遵守) ★",
    }


def main():
    p = argparse.ArgumentParser(description="5/10 朝 重賞 5 system verdict (Session #49 D)")
    p.add_argument("--date", default="20260509")
    p.add_argument("--out", default="data/v18/session_49_verify_extended.json")
    args = p.parse_args()

    print("=" * 70)
    print(f"5/10 朝 重賞 5 system verdict ({args.date})")
    print("=" * 70)
    print("★ 5/9 重賞 投票なし、 verdict 学習データ ★")

    # load
    pred = load_majors_predictions()
    print(f"\n[Predictions] available: {bool(pred.get('majors'))}")
    if pred.get("majors"):
        print(f"  n_majors: {len(pred['majors'])}")

    actual = load_actual_results(args.date)
    print(f"\n[Actual results] available: {actual.get('available')}")
    if actual.get("available"):
        print(f"  n_majors in results: {actual.get('n_majors')}")

    # evaluate
    sys_eval = evaluate_system(pred, actual)
    cons_eval = consensus_evaluation(pred, actual)
    roi_est = estimate_roi_if_bought(pred, actual)

    print(f"\n[System evaluation] {sys_eval.get('status')}")
    if sys_eval.get("by_system"):
        for k, v in sys_eval["by_system"].items():
            print(f"  {k}: {v}")

    print(f"\n[Consensus high confidence] {cons_eval.get('status')}")
    print(f"  n_high_confidence: {cons_eval.get('n_high_confidence', 0)}")

    print(f"\n[Estimated ROI if bought] {roi_est.get('status')}")
    print(f"  scenario: {roi_est.get('design', {}).get('scenario', 'N/A')}")

    summary = {
        "date": args.date,
        "predictions_loaded": bool(pred.get("majors")),
        "actual_loaded": actual.get("available", False),
        "system_evaluation": sys_eval,
        "consensus_evaluation": cons_eval,
        "estimated_roi": roi_est,
        "note": "★ 5/9 重賞 投票なし、 verdict 用学習データ ★",
    }

    out_path = BASE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
