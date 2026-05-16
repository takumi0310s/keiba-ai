"""outcome record CLI - task after metrics の read-only 計算 + JSON 記録

V15 production 完全不変、 cumulative_results.csv は read のみ。

usage:
  python tools/outcome_record.py --task-id P0-2 \\
                                  --task-name "kyoto/chukyo strategy7 re-exclude" \\
                                  --phase P0 \\
                                  --before data/task_outcomes/baseline_v15.json \\
                                  --after-since 2026-05-18 \\
                                  --expected "+5pt ROI"

  # sanity check (baseline-only dummy run):
  python tools/outcome_record.py --task-id SANITY \\
                                  --task-name "sanity check" \\
                                  --before data/task_outcomes/baseline_v15.json \\
                                  --after-since 2026-05-16 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CUM_CSV = REPO / "data" / "cumulative_results.csv"
OUTCOMES_DIR = REPO / "data" / "task_outcomes"


def _load_cumulative() -> pd.DataFrame:
    """cumulative_results.csv を read-only で読む."""
    if not CUM_CSV.exists():
        raise FileNotFoundError(f"cumulative_results.csv not found: {CUM_CSV}")
    df = pd.read_csv(CUM_CSV, encoding="utf-8-sig")
    # date を datetime に
    if "date" in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """ROI / 的中率 / PnL / N を計算 (read-only)."""
    settled = df[df["status"] == "settled"].copy()
    n = len(settled)
    if n == 0:
        return {
            "n": 0,
            "roi": None,
            "hit_rate_top3": None,
            "trio_hit_rate": None,
            "inv": 0,
            "pay": 0,
            "pnl": 0,
            "roi_per_race": [],
        }
    inv = pd.to_numeric(settled["investment"], errors="coerce").fillna(0).sum()
    pay = pd.to_numeric(settled["actual_payout"], errors="coerce").fillna(0).sum()
    roi = (pay / inv * 100) if inv > 0 else None
    finish = pd.to_numeric(settled.get("top1_finish"), errors="coerce")
    hit3_n = int((finish <= 3).sum())
    trio_hit_n = int(pd.to_numeric(settled.get("trio_hit"), errors="coerce").fillna(0).sum())

    # per-race ROI for t-test
    inv_per = pd.to_numeric(settled["investment"], errors="coerce").fillna(0)
    pay_per = pd.to_numeric(settled["actual_payout"], errors="coerce").fillna(0)
    roi_per_race = np.where(inv_per > 0, pay_per / inv_per * 100, np.nan)
    roi_per_race = roi_per_race[~np.isnan(roi_per_race)]

    return {
        "n": n,
        "roi": round(float(roi), 4) if roi is not None else None,
        "hit_rate_top3": round(hit3_n / n, 4),
        "trio_hit_rate": round(trio_hit_n / n, 4),
        "inv": int(inv),
        "pay": int(pay),
        "pnl": int(pay - inv),
        "roi_per_race": roi_per_race.tolist(),
    }


def welch_t_test(before_arr, after_arr) -> dict:
    """Welch's t-test for ROI per race (unequal variance)."""
    try:
        from scipy import stats as sp_stats
    except ImportError:
        return {"error": "scipy not installed", "available": False}

    if len(before_arr) < 2 or len(after_arr) < 2:
        return {
            "available": False,
            "reason": f"insufficient sample (before={len(before_arr)}, after={len(after_arr)})",
        }
    t_stat, p_val = sp_stats.ttest_ind(after_arr, before_arr, equal_var=False)
    # 95% CI for mean diff via Welch
    mean_diff = float(np.mean(after_arr) - np.mean(before_arr))
    se = float(np.sqrt(np.var(after_arr, ddof=1) / len(after_arr) + np.var(before_arr, ddof=1) / len(before_arr)))
    # approx df via Welch-Satterthwaite
    n1, n2 = len(after_arr), len(before_arr)
    s1, s2 = np.var(after_arr, ddof=1), np.var(before_arr, ddof=1)
    if (s1 / n1 + s2 / n2) == 0:
        ci_low, ci_high = mean_diff, mean_diff
    else:
        df_w = (s1 / n1 + s2 / n2) ** 2 / ((s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1) + 1e-12)
        try:
            tcrit = sp_stats.t.ppf(0.975, df_w)
        except Exception:
            tcrit = 1.96
        ci_low = mean_diff - tcrit * se
        ci_high = mean_diff + tcrit * se
    return {
        "available": True,
        "type": "Welch's t-test (ROI per race, %)",
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_val), 6),
        "mean_diff": round(mean_diff, 4),
        "ci_95": [round(float(ci_low), 4), round(float(ci_high), 4)],
        "n_before": int(len(before_arr)),
        "n_after": int(len(after_arr)),
        "significant_at_0_05": bool(p_val < 0.05),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--task-name", required=True)
    ap.add_argument("--phase", default="P1")
    ap.add_argument("--before", required=True, help="baseline JSON path")
    ap.add_argument("--after-since", required=True, help="YYYY-MM-DD")
    ap.add_argument("--after-until", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--expected", default=None, help="expected delta note")
    ap.add_argument("--status", default="shadow_eval", help="pending/shadow_eval/accepted/rejected/rolled_back")
    ap.add_argument("--notes", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true", help="print only, do not write")
    args = ap.parse_args()

    # 1. baseline 読み込み
    before_path = Path(args.before)
    if not before_path.is_absolute():
        before_path = REPO / before_path
    if not before_path.exists():
        print(f"ERROR: baseline not found: {before_path}", file=sys.stderr)
        sys.exit(1)
    with open(before_path, encoding="utf-8") as f:
        baseline = json.load(f)

    # 2. cumulative 読み込み + after 期間 filter
    df = _load_cumulative()
    since = pd.to_datetime(args.after_since)
    until = pd.to_datetime(args.after_until) if args.after_until else pd.Timestamp.today().normalize()
    mask = (df["date_dt"] >= since) & (df["date_dt"] <= until)
    after_df = df[mask].copy()

    # 3. before metrics は baseline から
    before_roi_cands = baseline.get("metrics", {}).get("actual_roi", {}).get("candidate_values", {})
    before_roi = None
    # use 5/16 evening recompute as default before
    if "raw_cumulative_5_16_evening" in before_roi_cands:
        before_roi = before_roi_cands["raw_cumulative_5_16_evening"]["value"]
    elif "raw_cumulative_93_23" in before_roi_cands:
        before_roi = before_roi_cands["raw_cumulative_93_23"]["value"]
    before_metrics = {
        "roi": before_roi,
        "hit_rate_top3": baseline.get("metrics", {}).get("hit_rate_top3", {}).get("value"),
        "trio_hit_rate": baseline.get("metrics", {}).get("trio_hit_rate", {}).get("value"),
        "n": baseline.get("metrics", {}).get("n_settled"),
        "baseline_ref": str(before_path.relative_to(REPO)).replace("\\", "/"),
    }

    # baseline per-race ROI for t-test: recompute from entire cumulative (settled rows)
    full_metrics = compute_metrics(df)
    before_roi_arr = np.array(full_metrics["roi_per_race"])

    # 4. after metrics
    after_metrics_raw = compute_metrics(after_df)
    after_roi_arr = np.array(after_metrics_raw["roi_per_race"])

    after_metrics = {
        "roi": after_metrics_raw["roi"],
        "hit_rate_top3": after_metrics_raw["hit_rate_top3"],
        "trio_hit_rate": after_metrics_raw["trio_hit_rate"],
        "n": after_metrics_raw["n"],
        "inv": after_metrics_raw["inv"],
        "pay": after_metrics_raw["pay"],
        "pnl": after_metrics_raw["pnl"],
        "data_window": f"{args.after_since} - {until.strftime('%Y-%m-%d')}",
    }

    # 5. delta
    def _safe_delta(a, b):
        if a is None or b is None:
            return None
        return round(a - b, 4)

    delta = {
        "roi": _safe_delta(after_metrics["roi"], before_metrics["roi"]),
        "hit_rate_top3": _safe_delta(after_metrics["hit_rate_top3"], before_metrics["hit_rate_top3"]),
        "trio_hit_rate": _safe_delta(after_metrics["trio_hit_rate"], before_metrics["trio_hit_rate"]),
    }

    # 6. stat test
    stat_test = welch_t_test(before_roi_arr, after_roi_arr)

    # 7. assemble outcome
    outcome = {
        "task_id": args.task_id,
        "task_name": args.task_name,
        "phase": args.phase,
        "implemented_at": datetime.now().isoformat(timespec="seconds"),
        "before": before_metrics,
        "after": after_metrics,
        "delta": delta,
        "expected": {"description": args.expected} if args.expected else None,
        "statistical_test": stat_test,
        "status": args.status,
        "notes": args.notes,
    }

    # 8. write
    out_path = OUTCOMES_DIR / f"{args.task_id}.json"
    if args.dry_run:
        print("[DRY-RUN] would write to:", out_path)
    else:
        OUTCOMES_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(outcome, f, ensure_ascii=False, indent=2)
        print(f"WROTE: {out_path}")

    # 9. summary
    print("=" * 60)
    print(f"task: [{args.task_id}] {args.task_name} ({args.phase})")
    print(f"  before: roi={before_metrics['roi']}, hit3={before_metrics['hit_rate_top3']}, n={before_metrics['n']}")
    print(f"  after : roi={after_metrics['roi']}, hit3={after_metrics['hit_rate_top3']}, n={after_metrics['n']}")
    print(f"  delta : roi={delta['roi']}, hit3={delta['hit_rate_top3']}")
    if stat_test.get("available"):
        print(f"  test  : p={stat_test['p_value']}, CI95={stat_test['ci_95']}, sig={stat_test['significant_at_0_05']}")
    else:
        print(f"  test  : not available ({stat_test.get('reason', stat_test.get('error', 'unknown'))})")
    print(f"  status: {args.status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
