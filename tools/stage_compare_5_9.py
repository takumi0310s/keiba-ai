"""5/9 朝 (Stage 1) vs 1h 前 (Stage 2) 予測 比較 framework (Session #65 D).

dev/two-stage 専用。 V15 production 完全独立 (read-only)。

入力:
  - 朝 (Stage 1): data/daily_predictions/20260509.csv
  - 1h 前 (Stage 2): data/v18/pre_race_predict_5_9_R*.json (Glob)
  - 実結果 (verdict): data/v18/verdicts_5_9_realtime.json (Session #61 産物、 placeholder)

CLI:
  python tools/stage_compare_5_9.py --summary    # 累積 metric
  python tools/stage_compare_5_9.py --by-race    # R 別 diff
  python tools/stage_compare_5_9.py --json       # JSON 形式

出力 (任意):
  data/v18/stage_compare_5_9_summary.json
  data/v18/stage_compare_5_9_summary.md
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tools"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DATE = "20260509"
DAILY_PRED = BASE / "data" / "daily_predictions" / f"{DATE}.csv"
STAGE2_GLOB = BASE / "data" / "v18" / "pre_race_predict_5_9_R*.json"
VERDICT_PATH = BASE / "data" / "v18" / "verdicts_5_9_realtime.json"
OUT_DIR = BASE / "data" / "v18"


def load_morning():
    import pandas as pd
    return pd.read_csv(DAILY_PRED, dtype=str)


def load_stage2_files() -> list[dict]:
    out = []
    for p in sorted(glob.glob(str(STAGE2_GLOB))):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            d["_path"] = p
            out.append(d)
        except Exception as e:
            print(f"[skip] {p}: {e}", file=sys.stderr)
    return out


def load_verdicts() -> list[dict]:
    """Session #61 realtime_5_9.py の verdict log。 placeholder OK."""
    if not VERDICT_PATH.exists():
        return []
    try:
        return json.loads(VERDICT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def compare_pair(morning_row, stage2_blob: dict) -> dict:
    """1 R の 朝 vs Stage 2 比較."""
    m_top1 = str(morning_row.get("top1_num", ""))
    m_top1_name = str(morning_row.get("top1_name", ""))
    m_top1_score = float(morning_row.get("top1_score", 0) or 0)
    m_top3 = {str(morning_row.get("top1_num", "")),
              str(morning_row.get("top2_num", "")),
              str(morning_row.get("top3_num", ""))}

    s2 = stage2_blob.get("stage2", {})
    s2_top3_list = s2.get("top3", []) or []
    if s2.get("error") or not s2_top3_list:
        return {
            "race_id": str(morning_row.get("race_id", "")),
            "course": str(morning_row.get("course", "")),
            "race_num": str(morning_row.get("race_num", "")),
            "race_name": str(morning_row.get("race_name", "")),
            "stage2_ok": False,
            "stage2_error": s2.get("error", "no_top3"),
            "morning_top1": m_top1,
            "morning_top1_name": m_top1_name,
            "morning_top1_score": m_top1_score,
        }

    s2_top1 = str(s2_top3_list[0]["umaban"])
    s2_top1_name = str(s2_top3_list[0]["name"])
    s2_top1_score = float(s2_top3_list[0]["score"])
    s2_top3 = {str(h["umaban"]) for h in s2_top3_list[:3]}

    overlap = len(m_top3 & s2_top3)
    return {
        "race_id": str(morning_row.get("race_id", "")),
        "course": str(morning_row.get("course", "")),
        "race_num": str(morning_row.get("race_num", "")),
        "race_name": str(morning_row.get("race_name", "")),
        "stage2_ok": True,
        "morning_top1": m_top1,
        "morning_top1_name": m_top1_name,
        "morning_top1_score": m_top1_score,
        "stage2_top1": s2_top1,
        "stage2_top1_name": s2_top1_name,
        "stage2_top1_score": s2_top1_score,
        "top1_changed": s2_top1 != m_top1,
        "top3_overlap": overlap,
        "score_diff": s2_top1_score - m_top1_score,
    }


def integrate_verdicts(rows: list[dict], verdicts: list[dict]) -> list[dict]:
    """Stage 2 比較行に 実結果 (5/10 朝 backfill 想定) を merge."""
    by_rid = {v.get("race_id", ""): v for v in verdicts}
    for r in rows:
        v = by_rid.get(r["race_id"])
        if v is None:
            r["actual_trio"] = None
            r["morning_in_trio"] = None
            r["stage2_in_trio"] = None
            continue
        actual = v.get("actual_trio") or []
        actual_set = {str(n) for n in actual}
        r["actual_trio"] = actual
        r["morning_in_trio"] = (r.get("morning_top1") in actual_set) if r.get("morning_top1") else None
        r["stage2_in_trio"] = (r.get("stage2_top1") in actual_set) if r.get("stage2_top1") else None
    return rows


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    n_ok = sum(1 for r in rows if r.get("stage2_ok"))
    n_changed = sum(1 for r in rows if r.get("top1_changed"))
    overlaps = [r.get("top3_overlap", 0) for r in rows if r.get("stage2_ok")]
    score_diffs = [r.get("score_diff", 0) for r in rows if r.get("stage2_ok")]
    morn_hits = [r.get("morning_in_trio") for r in rows if r.get("morning_in_trio") is not None]
    s2_hits = [r.get("stage2_in_trio") for r in rows if r.get("stage2_in_trio") is not None]

    return {
        "n_total": n,
        "n_stage2_ok": n_ok,
        "n_top1_changed": n_changed,
        "top1_change_rate": n_changed / max(n_ok, 1),
        "top3_overlap_mean": sum(overlaps) / max(len(overlaps), 1),
        "score_diff_mean": sum(score_diffs) / max(len(score_diffs), 1),
        "morning_top1_in_trio_rate": (sum(1 for x in morn_hits if x) / max(len(morn_hits), 1)
                                       if morn_hits else None),
        "stage2_top1_in_trio_rate": (sum(1 for x in s2_hits if x) / max(len(s2_hits), 1)
                                      if s2_hits else None),
        "n_with_verdict": len(morn_hits),
    }


def cmd_summary(args):
    df = load_morning()
    stage2_blobs = load_stage2_files()
    by_rid = {b["race_id"]: b for b in stage2_blobs}
    verdicts = load_verdicts()

    rows = []
    for _, m in df.iterrows():
        rid = str(m.get("race_id", ""))
        s2 = by_rid.get(rid)
        if s2 is None:
            continue
        rows.append(compare_pair(m, s2))
    rows = integrate_verdicts(rows, verdicts)
    agg = aggregate(rows)

    body_lines = [
        f"# Session #65 D: 朝 vs 1h 前 比較 summary ({DATE})",
        f"",
        f"updated: {datetime.now().isoformat()}",
        f"",
        f"## 累積 metrics",
        f"- 比較対象 R: {agg['n_total']}",
        f"- Stage 2 成功 R: {agg['n_stage2_ok']}",
        f"- top1 変更 R: {agg['n_top1_changed']} ({agg['top1_change_rate']*100:.1f}% of OK)",
        f"- top3 重複 mean: {agg['top3_overlap_mean']:.2f} / 3",
        f"- score 差 mean: {agg['score_diff_mean']:+.4f}",
        f"",
        f"## 実結果 と統合 (placeholder)",
        f"- verdict 取得 R: {agg['n_with_verdict']}",
    ]
    if agg.get("morning_top1_in_trio_rate") is not None:
        body_lines.append(
            f"- 朝 top1 が trio に入った率: {agg['morning_top1_in_trio_rate']*100:.1f}%")
        body_lines.append(
            f"- 1h 前 top1 が trio に入った率: {agg['stage2_top1_in_trio_rate']*100:.1f}%")
        diff = (agg['stage2_top1_in_trio_rate'] - agg['morning_top1_in_trio_rate']) * 100
        body_lines.append(f"- Stage 2 効果 (差): {diff:+.1f} pt")
    else:
        body_lines.append("- 実結果未取得 (5/10 朝 backfill 後 再実行)")

    body = "\n".join(body_lines)
    print(body)

    out_md = OUT_DIR / "stage_compare_5_9_summary.md"
    out_md.write_text(body, encoding="utf-8")
    out_json = OUT_DIR / "stage_compare_5_9_summary.json"
    out_json.write_text(json.dumps({"summary": agg, "rows": rows},
                                   ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nout: {out_md.relative_to(BASE)}")
    print(f"out: {out_json.relative_to(BASE)}")


def cmd_by_race(args):
    df = load_morning()
    stage2_blobs = load_stage2_files()
    by_rid = {b["race_id"]: b for b in stage2_blobs}
    verdicts = load_verdicts()

    print(f"{'race_id':<14} {'venue':<5} {'R':<3} {'morn_top1':<10} {'s2_top1':<10} {'chg':<5} {'overlap':<8} {'score_diff':<10}")
    rows = []
    for _, m in df.iterrows():
        rid = str(m.get("race_id", ""))
        s2 = by_rid.get(rid)
        if s2 is None:
            continue
        r = compare_pair(m, s2)
        rows.append(r)
        chg = "★" if r.get("top1_changed") else "-"
        if r.get("stage2_ok"):
            print(f"{r['race_id']:<14} {r['course']:<5} {r['race_num']:<3} "
                  f"{r['morning_top1']:<10} {r.get('stage2_top1','-'):<10} {chg:<5} "
                  f"{r.get('top3_overlap','-'):<8} {r.get('score_diff',0):+.3f}")
        else:
            print(f"{r['race_id']:<14} {r['course']:<5} {r['race_num']:<3} "
                  f"{r['morning_top1']:<10} ERR        -     -        -")

    rows = integrate_verdicts(rows, verdicts)
    if args.json:
        out = OUT_DIR / "stage_compare_5_9_by_race.json"
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\nout: {out.relative_to(BASE)}")


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=False)

    p_s = sp.add_parser("summary")
    p_b = sp.add_parser("by-race")
    p_b.add_argument("--json", action="store_true")

    p.add_argument("--summary", action="store_true")
    p.add_argument("--by-race", action="store_true")
    p.add_argument("--json", action="store_true")

    args = p.parse_args()
    if args.cmd == "summary" or args.summary:
        cmd_summary(args)
    elif args.cmd == "by-race" or args.by_race:
        cmd_by_race(args)
    else:
        cmd_summary(args)


if __name__ == "__main__":
    main()
