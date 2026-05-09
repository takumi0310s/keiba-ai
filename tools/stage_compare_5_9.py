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
            "stage2_error_kind": s2.get("error_kind"),  # Session #68 D: kind 集計用
            "stage2_diag": s2.get("diag", {}),           # Session #68 D: 診断情報 保存
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
    """Session #68 D: hit rate を 3 系統 で集計.

    系統:
      1. morning_only: 全 R で朝予測 top1 が trio 入った率 (= Stage 2 失敗 R も morning fallback)
      2. stage2_success_only: Stage 2 成功 R に限った top1 入賞率
      3. integrated: stage2 が成功した R は stage2 top1、 失敗 R は morning top1 を採用 (実運用方針)

    error_kind 別の集計も追加 (netkeiba_block / shutuba_empty / exception)。
    """
    n = len(rows)
    n_ok = sum(1 for r in rows if r.get("stage2_ok"))
    n_failed = n - n_ok
    n_changed = sum(1 for r in rows if r.get("top1_changed"))
    overlaps = [r.get("top3_overlap", 0) for r in rows if r.get("stage2_ok")]
    score_diffs = [r.get("score_diff", 0) for r in rows if r.get("stage2_ok")]

    # Session #68 D: 失敗 R の error_kind 別 count
    err_kinds = {}
    for r in rows:
        if not r.get("stage2_ok"):
            kind = r.get("stage2_error_kind") or _infer_kind(r.get("stage2_error", ""))
            err_kinds[kind] = err_kinds.get(kind, 0) + 1

    # 系統 1: morning_only — 全 R 対象、 morning_in_trio で集計
    morn_hits_all = [r.get("morning_in_trio") for r in rows if r.get("morning_in_trio") is not None]
    # 系統 2: stage2_success_only — Stage 2 成功 R のみ
    morn_hits_ok = [r.get("morning_in_trio") for r in rows
                    if r.get("stage2_ok") and r.get("morning_in_trio") is not None]
    s2_hits_ok = [r.get("stage2_in_trio") for r in rows
                  if r.get("stage2_ok") and r.get("stage2_in_trio") is not None]
    # 系統 3: integrated — Stage 2 成功 R は s2 top1、 失敗 R は morning top1
    integrated_hits = []
    for r in rows:
        if r.get("stage2_ok") and r.get("stage2_in_trio") is not None:
            integrated_hits.append(r["stage2_in_trio"])
        elif r.get("morning_in_trio") is not None:
            integrated_hits.append(r["morning_in_trio"])

    def _rate(lst):
        return (sum(1 for x in lst if x) / max(len(lst), 1)) if lst else None

    return {
        "n_total": n,
        "n_stage2_ok": n_ok,
        "n_stage2_failed": n_failed,
        "stage2_failure_rate": n_failed / max(n, 1),
        "stage2_error_kinds": err_kinds,
        "n_top1_changed": n_changed,
        "top1_change_rate": n_changed / max(n_ok, 1),
        "top3_overlap_mean": sum(overlaps) / max(len(overlaps), 1),
        "score_diff_mean": sum(score_diffs) / max(len(score_diffs), 1) if score_diffs else 0.0,
        # Session #68 D: 3 系統 hit rate
        "hit_rate_morning_only": _rate(morn_hits_all),
        "hit_rate_stage2_only_morning_ref": _rate(morn_hits_ok),
        "hit_rate_stage2_only_stage2_ref": _rate(s2_hits_ok),
        "hit_rate_integrated": _rate(integrated_hits),
        "n_with_verdict_total": len(morn_hits_all),
        "n_with_verdict_stage2_ok": len(morn_hits_ok),
        # 旧 key (互換性のため維持)
        "morning_top1_in_trio_rate": _rate(morn_hits_all),
        "stage2_top1_in_trio_rate": _rate(s2_hits_ok),
        "n_with_verdict": len(morn_hits_all),
    }


def _infer_kind(err: str) -> str:
    """旧 JSON (error_kind 無し) を error 文字列から逆引き."""
    if not err:
        return "no_error_msg"
    if "HTTP 400" in err or "server block" in err:
        return "netkeiba_block"
    if "returned None" in err:
        return "shutuba_empty"
    return "other"


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
        f"# Session #65 D + Session #68 D: 朝 vs 1h 前 比較 summary ({DATE})",
        f"",
        f"updated: {datetime.now().isoformat()}",
        f"",
        f"## 累積 metrics",
        f"- 比較対象 R: {agg['n_total']}",
        f"- Stage 2 成功 R: {agg['n_stage2_ok']}",
        f"- Stage 2 失敗 R: {agg['n_stage2_failed']} ({agg['stage2_failure_rate']*100:.1f}%)",
    ]
    err_kinds = agg.get("stage2_error_kinds", {})
    if err_kinds:
        body_lines.append(f"- 失敗 R の error_kind 内訳:")
        for kind, cnt in sorted(err_kinds.items(), key=lambda x: -x[1]):
            body_lines.append(f"  - {kind}: {cnt}")
    body_lines += [
        f"- top1 変更 R: {agg['n_top1_changed']} ({agg['top1_change_rate']*100:.1f}% of OK)",
        f"- top3 重複 mean: {agg['top3_overlap_mean']:.2f} / 3",
        f"- score 差 mean: {agg['score_diff_mean']:+.4f}",
        f"",
        f"## 実結果 と統合 (3 系統 hit rate)",
        f"- verdict 取得 R (全体): {agg['n_with_verdict_total']}",
        f"- verdict 取得 R (Stage 2 成功 のみ): {agg['n_with_verdict_stage2_ok']}",
    ]
    if agg.get("hit_rate_morning_only") is not None:
        body_lines.append("")
        body_lines.append("### 系統 1: 朝予測のみ (全 R)")
        body_lines.append(
            f"- 朝 top1 が trio 入り 率 = {agg['hit_rate_morning_only']*100:.1f}%")

        if agg.get("hit_rate_stage2_only_stage2_ref") is not None:
            body_lines.append("")
            body_lines.append("### 系統 2: Stage 2 成功 R のみ")
            body_lines.append(
                f"- 朝 top1 が trio 入り 率 = {agg['hit_rate_stage2_only_morning_ref']*100:.1f}% (参照)")
            body_lines.append(
                f"- Stage 2 top1 が trio 入り 率 = {agg['hit_rate_stage2_only_stage2_ref']*100:.1f}%")
            diff = (agg['hit_rate_stage2_only_stage2_ref'] - agg['hit_rate_stage2_only_morning_ref']) * 100
            body_lines.append(f"- Stage 2 効果 (差) = {diff:+.1f} pt")

        if agg.get("hit_rate_integrated") is not None:
            body_lines.append("")
            body_lines.append("### 系統 3: integrated (Stage 2 成功は s2、 失敗は morning fallback) ★実運用方針")
            body_lines.append(
                f"- integrated top1 が trio 入り 率 = {agg['hit_rate_integrated']*100:.1f}%")
            diff_int = (agg['hit_rate_integrated'] - agg['hit_rate_morning_only']) * 100
            body_lines.append(f"- 朝のみ vs integrated 差 = {diff_int:+.1f} pt")
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
