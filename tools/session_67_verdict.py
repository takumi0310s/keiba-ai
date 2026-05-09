"""Session #67 C: 全 R verdict 集計 (V15 朝予測 + 動画 v66 + Stage 2).

入力:
  data/results/20260509_results.csv (Session #67 A 出力)
  data/daily_predictions/20260509.csv (V15 朝予測)
  data/v18/horse_total_scores_5_9.csv (動画 v66 NO_TYB)
  data/v18/pre_race_predict_5_9_R*.json (Stage 2 取れた R)

出力:
  data/v18/system_comparison_5_9.csv
  data/v18/session_67_all_verdict.md
"""
from __future__ import annotations

import csv
import glob
import json
import re
import sys
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def load_v66_top1(scores_csv: Path) -> dict[str, int]:
    """horse_total_scores_5_9.csv → {race_id: top1_umaban}."""
    out = {}
    if not scores_csv.exists():
        return out
    import pandas as pd
    df = pd.read_csv(scores_csv, dtype=str)
    df["rank_in_race"] = df["rank_in_race"].astype(int)
    for rid, g in df.groupby("race_id"):
        top1 = g[g["rank_in_race"] == 1]
        if len(top1) >= 1:
            out[str(rid)] = int(top1.iloc[0]["umaban"])
    return out


def load_5system_top1(json_path: Path) -> dict[str, dict]:
    """predictions_majors_5system_5_9_FINAL.json → {race_id: {sys_n: top1}}."""
    out = {}
    if not json_path.exists():
        return out
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for m in data.get("majors", []):
        rid = str(m["race"]["race_id"])
        sys_top = {}
        for p in m["predictions"]:
            n = p.get("system")
            if p.get("status") == "ok":
                sys_top[n] = p.get("top1_num")
            elif p.get("status") == "simulate":
                sys_top[n] = p.get("predicted_top1_num")
        out[rid] = sys_top
    return out


def load_stage2(glob_pattern: str) -> dict[str, dict]:
    """pre_race_predict_5_9_R*.json → {race_id: {stage1_top1, stage2_top1, stage2_status}}."""
    out = {}
    for path in glob.glob(glob_pattern):
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        rid = str(d.get("race_id", ""))
        if not rid:
            # ファイル名から race_id 抽出 (R7_東京_202605020507.json)
            m = re.search(r"_(\d{12})\.json$", path)
            if m: rid = m.group(1)
        out[rid] = {
            "stage1_top1": d.get("stage1_top1") or d.get("morning_top1"),
            "stage2_top1": d.get("stage2_top1") or d.get("predict_top1"),
            "stage2_status": d.get("status", "unknown"),
        }
    return out


def main():
    results_csv = BASE / "data" / "results" / "20260509_results.csv"
    pred_csv = BASE / "data" / "daily_predictions" / "20260509.csv"
    scores_csv = BASE / "data" / "v18" / "horse_total_scores_5_9.csv"
    sys5_json = BASE / "data" / "v18" / "predictions_majors_5system_5_9_FINAL.json"
    stage2_glob = str(BASE / "data" / "v18" / "pre_race_predict_5_9_R*.json")

    import pandas as pd
    res_df = pd.read_csv(results_csv, dtype=str)
    pred_df = pd.read_csv(pred_csv, dtype=str)

    v66 = load_v66_top1(scores_csv)
    sys5 = load_5system_top1(sys5_json)
    stage2 = load_stage2(stage2_glob)

    out_csv = BASE / "data" / "v18" / "system_comparison_5_9.csv"
    fields = ["race_id", "course", "race_num", "race_name", "is_major", "is_12r",
              "finish_1", "finish_2", "finish_3",
              "v15_top1", "v15_top1_hit", "v15_top3_overlap", "v15_trio_hit",
              "v66_top1", "v66_top1_hit",
              "stage2_top1", "stage2_top1_hit", "stage2_status",
              "payout_trio", "v15_700yen_pnl"]

    rows = []
    sys_stats = {
        "v15_top1": {"hit": 0, "tot": 0},
        "v15_top3": {"hit": 0, "tot": 0},
        "v15_trio": {"hit": 0, "tot": 0},
        "v66_top1": {"hit": 0, "tot": 0},
        "stage2_top1": {"hit": 0, "tot": 0},
    }

    pred_by_id = {str(r["race_id"]): r for _, r in pred_df.iterrows()}

    for _, r in res_df.iterrows():
        rid = str(r["race_id"])
        if r["fetch_status"] != "ok":
            continue
        pr = pred_by_id.get(rid)
        if pr is None:
            continue

        try:
            f1 = int(r["finish_1"]); f2 = int(r["finish_2"]); f3 = int(r["finish_3"])
        except Exception:
            continue
        actual_set = {f1, f2, f3}

        try:
            v15_top1 = int(pr["top1_num"])
            v15_top2 = int(pr["top2_num"])
            v15_top3 = int(pr["top3_num"])
        except Exception:
            v15_top1 = v15_top2 = v15_top3 = 0
        v15_top3_set = {v15_top1, v15_top2, v15_top3}

        v15_top1_hit = (v15_top1 == f1)
        v15_top3_overlap = len(v15_top3_set & actual_set)
        # trio hit: V15 trio_bets に actual_set が含まれるか
        trio_bets_str = str(pr.get("trio_bets", ""))
        v15_trio_hit = False
        for b in trio_bets_str.split(";"):
            try:
                ns = {int(x) for x in b.strip().split("-")}
                if ns == actual_set:
                    v15_trio_hit = True
                    break
            except Exception:
                pass

        v66_top1 = v66.get(rid, 0)
        v66_top1_hit = (v66_top1 == f1) if v66_top1 else False

        st2 = stage2.get(rid, {})
        st2_top1 = st2.get("stage2_top1")
        try:
            st2_top1_int = int(st2_top1) if st2_top1 else 0
        except Exception:
            st2_top1_int = 0
        st2_top1_hit = (st2_top1_int == f1) if st2_top1_int else None

        try:
            payout_trio = int(r.get("payout_trio") or 0)
        except Exception:
            payout_trio = 0
        v15_700_pnl = (payout_trio - 700) if v15_trio_hit else -700

        is_major = pr.get("race_name", "") in ("京都新聞杯", "エプソムカップ", "駿風 S", "駿風S")
        is_12r = (str(pr.get("race_num", "")) == "12")

        # 統計加算
        sys_stats["v15_top1"]["tot"] += 1
        sys_stats["v15_top1"]["hit"] += int(v15_top1_hit)
        sys_stats["v15_top3"]["tot"] += 1
        sys_stats["v15_top3"]["hit"] += int(v15_top3_overlap >= 1)
        sys_stats["v15_trio"]["tot"] += 1
        sys_stats["v15_trio"]["hit"] += int(v15_trio_hit)
        if v66_top1:
            sys_stats["v66_top1"]["tot"] += 1
            sys_stats["v66_top1"]["hit"] += int(v66_top1_hit)
        if st2_top1_hit is not None:
            sys_stats["stage2_top1"]["tot"] += 1
            sys_stats["stage2_top1"]["hit"] += int(st2_top1_hit)

        rows.append({
            "race_id": rid,
            "course": r["course"],
            "race_num": r["race_num"],
            "race_name": r["race_name"],
            "is_major": int(bool(is_major)),
            "is_12r": int(is_12r),
            "finish_1": f1, "finish_2": f2, "finish_3": f3,
            "v15_top1": v15_top1, "v15_top1_hit": int(v15_top1_hit),
            "v15_top3_overlap": v15_top3_overlap,
            "v15_trio_hit": int(v15_trio_hit),
            "v66_top1": v66_top1, "v66_top1_hit": int(v66_top1_hit),
            "stage2_top1": st2_top1_int,
            "stage2_top1_hit": int(bool(st2_top1_hit)) if st2_top1_hit is not None else "",
            "stage2_status": st2.get("stage2_status", ""),
            "payout_trio": payout_trio,
            "v15_700yen_pnl": v15_700_pnl,
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Summary
    print("=" * 60)
    print(f"Total verdicts: {len(rows)}")
    for k, s in sys_stats.items():
        if s["tot"] > 0:
            rate = s["hit"] / s["tot"] * 100
            print(f"  {k}: {s['hit']}/{s['tot']} ({rate:.1f}%)")
        else:
            print(f"  {k}: 0/0 (-)")

    # 重賞 R + 12R hypothetical ROI
    n_inv = 0; n_hit = 0; tot_inv = 0; tot_payout = 0
    for r in rows:
        if r["is_major"] or r["is_12r"]:
            n_inv += 1
            tot_inv += 700
            if r["v15_trio_hit"]:
                n_hit += 1
                tot_payout += r["payout_trio"]
    print(f"\n重賞 + 12R 仮投資 (V15 三連複 7点):")
    print(f"  N: {n_inv}, hit: {n_hit}, 投資: ¥{tot_inv:,}, 払戻: ¥{tot_payout:,}")
    if tot_inv > 0:
        roi = tot_payout / tot_inv * 100
        print(f"  ROI: {roi:.1f}% (※ payout 値は HJC 簡易 parser のため正確性 LOW)")

    print(f"\nout: {out_csv.relative_to(BASE)}")

    # JSON summary も出力 (D 領域で再利用)
    summary_json = BASE / "data" / "v18" / "session_67_verdict_summary.json"
    summary_json.write_text(json.dumps({
        "total_verdicts": len(rows),
        "sys_stats": sys_stats,
        "majors_12r_n_inv": n_inv,
        "majors_12r_n_hit": n_hit,
        "majors_12r_tot_inv": tot_inv,
        "majors_12r_tot_payout": tot_payout,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
