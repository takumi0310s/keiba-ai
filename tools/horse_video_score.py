"""
Session #61 B: 重賞 3R 全馬 動画スコア + V15 統合 score 計算
input:
  - data/v18/horse_motion_5_9.csv (3 race × 3 頭 simulate)
  - data/v18/predictions_majors_5system_5_9_FINAL.json (race meta + V15 top3)
  - data/v18/predictions_5_9_all.json (V15 全馬 score)
output:
  - data/v18/horse_video_scores_5_9.csv (race_name 併記、 全馬 ranking)
  - data/v18/session_61_scoring_logic.md
"""
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
V18 = ROOT / "data" / "v18"

motion = pd.read_csv(V18 / "horse_motion_5_9.csv")
final = json.load(open(V18 / "predictions_majors_5system_5_9_FINAL.json", encoding="utf-8"))
allp = json.load(open(V18 / "predictions_5_9_all.json", encoding="utf-8"))

# 馬名 lookup: jrdb_kyi.csv の最終列 race_id, col6=umaban, col8=horse_name
NAME_LOOKUP = {}
TARGET_RIDS = {"202605020511", "202608030511", "202604010311"}
with open(ROOT / "data" / "jrdb_kyi.csv", encoding="utf-8", errors="ignore") as f:
    for line in f:
        cols = line.rstrip().split(",")
        if len(cols) < 9:
            continue
        rid = cols[-1]
        if rid in TARGET_RIDS:
            try:
                umaban = int(cols[5])
                name = cols[7]
                NAME_LOOKUP[(rid, umaban)] = name
            except Exception:
                pass

allp_by_id = {p["race_id"]: p for p in allp["predictions"]}

MAJORS = []
for m in final["majors"]:
    r = m["race"]
    rid = r["race_id"]
    grade = "G3" if "G3" in r["race_name"] else "G2" if "G2" in r["race_name"] else "OP"
    base = allp_by_id.get(rid, {})
    v15_scores = base.get("v15_scores", {})
    v15_top3 = base.get("v15_top3", [])
    sys_v15 = next((p for p in m["predictions"] if p["system"] == 1), {})
    MAJORS.append({
        "race_id": rid,
        "venue": r["course"],
        "race_num": r["race_num"],
        "race_name": r["race_name"],
        "grade": grade,
        "start": r["start"],
        "v15_scores": v15_scores,
        "v15_top3": v15_top3,
        "fallback_top3": {
            "1": (sys_v15.get("top1_num"), sys_v15.get("top1_name"), sys_v15.get("top1_score")),
            "2": (sys_v15.get("top2_num"), sys_v15.get("top2_name"), None),
            "3": (sys_v15.get("top3_num"), sys_v15.get("top3_name"), None),
        },
    })

motion_by_race = {rid: g for rid, g in motion.groupby("race_id")}

def race_pct(score, scores):
    if not scores:
        return 0.5
    vals = sorted(scores)
    rank = sum(1 for v in vals if v < score) + 0.5 * sum(1 for v in vals if v == score)
    return rank / len(vals)

rows = []
for r in MAJORS:
    rid = r["race_id"]
    scores_dict = r["v15_scores"]
    all_v = list(scores_dict.values())
    motion_g = motion_by_race.get(int(rid))
    motion_repr = None
    if motion_g is not None and len(motion_g) > 0:
        motion_repr = {
            "stride_mean": float(motion_g["stride_length_mean"].mean()),
            "body_mean": float(motion_g["body_size_relative"].mean()),
            "stab_mean": float(motion_g["stability_score"].mean()),
            "tens_mean": float(motion_g["tension_score"].mean()),
            "n": len(motion_g),
        }
    if scores_dict:
        for umaban, score in sorted(scores_dict.items(), key=lambda x: -x[1]):
            pct = race_pct(score, all_v)
            integ = pct
            name = NAME_LOOKUP.get((rid, int(umaban)))
            if not name:
                name = next((t["horse_name"] for t in r["v15_top3"] if str(t["umaban"]) == str(umaban)), None)
            rows.append({
                "race_id": rid,
                "venue": r["venue"],
                "race_no": r["race_num"],
                "race_name": r["race_name"],
                "race_grade": r["grade"],
                "race_start_time": r["start"],
                "umaban": int(umaban),
                "horse_name": name or "",
                "v15_score": round(score, 4),
                "v15_pct": round(pct, 3),
                "integrated_score": round(integ, 3),
                "source": "v15+motion_repr" if motion_repr else "v15",
                "confidence": "low",
                "motion_n": motion_repr["n"] if motion_repr else 0,
            })
    else:
        for k in ("1", "2", "3"):
            num, nm, sc = r["fallback_top3"][k]
            rows.append({
                "race_id": rid,
                "venue": r["venue"],
                "race_no": r["race_num"],
                "race_name": r["race_name"],
                "race_grade": r["grade"],
                "race_start_time": r["start"],
                "umaban": int(num) if num else 0,
                "horse_name": nm or "",
                "v15_score": round(sc, 4) if sc else None,
                "v15_pct": None,
                "integrated_score": None,
                "source": "fallback_top3",
                "confidence": "low",
                "motion_n": motion_repr["n"] if motion_repr else 0,
            })

df = pd.DataFrame(rows)
df["rank_in_race"] = df.groupby("race_id").cumcount() + 1
out_csv = V18 / "horse_video_scores_5_9.csv"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"saved: {out_csv} ({len(df)} rows)")

# logic md
md = ["# Session #61 B: scoring logic", "",
      "## input", "- horse_motion_5_9.csv (3 race × 3 頭 simulate)",
      "- predictions_majors_5system_5_9_FINAL.json (race meta + V15 top3)",
      "- predictions_5_9_all.json (V15 全馬 score)",
      "", "## scoring 方針",
      "- 動画 motion = simulate のみ × 3 頭 → 全馬 unique scoring 不可",
      "- 代替: V15 score を race 内 percentile 化 → integrated_score",
      "- motion 3 頭は race 代表値 (stride/body/stab/tens 平均) として補助情報",
      "- 東京 11R エプソム C は v15_scores 0 件 → fallback top3 のみ",
      "", "## 出力", f"- {out_csv.name} ({len(df)} rows、 race_name 併記)",
      "", "## 制約事項",
      "- 真の動画解析 score ではない (Session #60 動画 DL 失敗のため)",
      "- 全馬 score = V15 ベース。 motion features は補助",
      "- 次回: 動画 DL 経路修正後に true motion-based scoring へ",
      ""]
(V18 / "session_61_scoring_logic.md").write_text("\n".join(md), encoding="utf-8")
print(f"saved: {V18 / 'session_61_scoring_logic.md'}")
