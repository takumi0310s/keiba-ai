"""
daily cumulative audit -- 真値 vs CLAUDE.md / baseline_v15.json drift 検出

毎日 21:00 (DailyResultsEvening 後) で 起動想定。
read-only audit、 V15 production 不変、 Discord 警告通知のみ。

source of truth: data/cumulative_results.csv (status=settled rows)
drift threshold:
  - ROI delta > 5pt vs baseline_v15.json adopted_value
  - PnL delta > 5000 yen vs baseline_v15.json adopted_pnl_yen
  - CLAUDE.md に旧 drift 値 (13,530 / 119.2%) が "drift" / "旧値" 注釈なしで残存

usage:
  python tools/daily_cumulative_audit.py
  python tools/daily_cumulative_audit.py --no-discord  # 通知抑止 (dry-run 推奨)

exit code:
  0 = no drift
  1 = drift detected
  2 = fatal error (csv missing 等)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Windows cp932 console で 絵文字 / yen sign 印字 → utf-8 化 (schtask redirect 対策)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[1]
CUM_CSV = REPO / "data" / "cumulative_results.csv"
BASELINE_JSON = REPO / "data" / "task_outcomes" / "baseline_v15.json"
CLAUDE_MD = REPO / "CLAUDE.md"
OUT_TRUTH = REPO / "data" / "cumulative_truth.json"
LOG_DIR = REPO / "logs"
OUT_LOG = LOG_DIR / "daily_cumulative_audit.log"

# drift threshold (5/16 制定、 baseline_v15.json adopted_value 基準)
ROI_DRIFT_PT = 5.0
PNL_DRIFT_YEN = 5000

# CLAUDE.md 内 旧値 (memory drift 標識)
LEGACY_VALUES = ["13,530", "119.2%"]
# 旧値 引用が allow される annotation (drift 系 docs / history 記述)
ALLOW_ANNOTATIONS = ["drift", "旧値", "memory_drift", "snapshot", "history"]


def compute_truth() -> dict:
    """cumulative_results.csv から真値を計算 (status=settled のみ)"""
    if not CUM_CSV.exists():
        raise FileNotFoundError(f"cumulative_results.csv not found: {CUM_CSV}")

    cum = pd.read_csv(CUM_CSV)
    settled = cum[cum["status"] == "settled"].copy()
    settled["inv"] = pd.to_numeric(settled["investment"], errors="coerce").fillna(0)
    settled["pay"] = pd.to_numeric(settled["actual_payout"], errors="coerce").fillna(0)
    total_inv = int(settled["inv"].sum())
    total_pay = int(settled["pay"].sum())
    pnl = total_pay - total_inv
    roi = (total_pay / total_inv * 100) if total_inv > 0 else 0.0

    # top1 finish <= 3 rate (top1_finish が数値の row のみ)
    settled["top1_finish_num"] = pd.to_numeric(settled["top1_finish"], errors="coerce")
    valid_top1 = settled.dropna(subset=["top1_finish_num"])
    hit3 = float((valid_top1["top1_finish_num"] <= 3).mean()) if len(valid_top1) > 0 else 0.0

    # trio_hit rate
    settled["trio_hit_num"] = pd.to_numeric(settled["trio_hit"], errors="coerce").fillna(0)
    trio_hit = float((settled["trio_hit_num"] == 1).mean()) if len(settled) > 0 else 0.0

    return {
        "calculation_date": datetime.now().isoformat(timespec="seconds"),
        "source": str(CUM_CSV.relative_to(REPO)).replace("\\", "/"),
        "n_settled": int(len(settled)),
        "n_top1_filled": int(len(valid_top1)),
        "total_inv": total_inv,
        "total_pay": total_pay,
        "pnl": pnl,
        "roi_pct": float(roi),
        "hit3_rate": hit3,
        "trio_hit_rate": trio_hit,
    }


def detect_drift_baseline(truth: dict) -> dict:
    """baseline_v15.json の adopted_value と truth を比較"""
    drift: dict = {}
    if not BASELINE_JSON.exists():
        return {"_warning": f"baseline_v15.json not found: {BASELINE_JSON}"}

    with open(BASELINE_JSON, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    metrics = baseline.get("metrics", {})
    adopted_roi = metrics.get("actual_roi", {}).get("adopted_value")
    adopted_pnl = metrics.get("cumulative_pnl_yen", {}).get("adopted_value")

    if adopted_roi is not None:
        delta_roi = truth["roi_pct"] - float(adopted_roi)
        if abs(delta_roi) > ROI_DRIFT_PT:
            drift["roi_drift_pt"] = round(delta_roi, 3)
            drift["roi_baseline"] = float(adopted_roi)
            drift["roi_truth"] = round(truth["roi_pct"], 3)

    if adopted_pnl is not None:
        delta_pnl = truth["pnl"] - int(adopted_pnl)
        if abs(delta_pnl) > PNL_DRIFT_YEN:
            drift["pnl_drift_yen"] = int(delta_pnl)
            drift["pnl_baseline"] = int(adopted_pnl)
            drift["pnl_truth"] = int(truth["pnl"])

    return drift


def detect_drift_claude_md() -> list:
    """CLAUDE.md 内に 旧値 (13,530 / 119.2%) が annotation なしで 残存しているか"""
    found: list = []
    if not CLAUDE_MD.exists():
        return found

    text = CLAUDE_MD.read_text(encoding="utf-8")
    for i, line in enumerate(text.split("\n"), 1):
        lower = line.lower()
        # 注釈付き引用は許容
        if any(tag in lower for tag in [a.lower() for a in ALLOW_ANNOTATIONS]):
            continue
        for v in LEGACY_VALUES:
            if v in line:
                found.append({"line": i, "value": v, "text": line[:140].strip()})
    return found


def send_discord(message: str) -> bool:
    """Discord webhook で 警告通知 (環境変数 hardcode 禁止)"""
    try:
        import requests  # type: ignore
    except ImportError:
        print("[WARN] requests not installed, skip discord")
        return False

    webhook = os.environ.get("DISCORD_WEBHOOK_UPDATES") or os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        print("[WARN] no DISCORD_WEBHOOK_UPDATES / DISCORD_WEBHOOK_URL configured")
        return False
    try:
        # discord content 上限 2000 chars
        content = message[:1900]
        resp = requests.post(webhook, json={"content": content}, timeout=10)
        return resp.status_code in (200, 204)
    except Exception as e:
        print(f"[WARN] webhook send failed: {e}")
        return False


def append_log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="daily cumulative audit (read-only)")
    parser.add_argument("--no-discord", action="store_true", help="Discord 通知抑止 (dry-run)")
    args = parser.parse_args()

    try:
        truth = compute_truth()
    except FileNotFoundError as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        append_log(f"FATAL: {e}")
        return 2

    # 真値を save (weekly_report 等から参照)
    OUT_TRUTH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TRUTH, "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2, ensure_ascii=False)

    # drift detect
    drift_baseline = detect_drift_baseline(truth)
    drift_claude = detect_drift_claude_md()

    summary = (
        f"[DailyCumulativeAudit] {truth['calculation_date'][:10]} "
        f"ROI {truth['roi_pct']:.2f}% / PnL {truth['pnl']:+,d}円 / "
        f"N={truth['n_settled']} / hit3 {truth['hit3_rate']:.1%}"
    )
    print(summary)

    msg_parts = [summary]
    has_drift = False

    if drift_baseline and "_warning" not in drift_baseline:
        has_drift = True
        msg_parts.append(f"baseline drift: {json.dumps(drift_baseline, ensure_ascii=False)}")
    elif "_warning" in drift_baseline:
        msg_parts.append(drift_baseline["_warning"])

    if drift_claude:
        has_drift = True
        msg_parts.append(f"CLAUDE.md 旧値残存 ({len(drift_claude)} 箇所):")
        for d in drift_claude[:5]:
            msg_parts.append(f"  L{d['line']} [{d['value']}] {d['text']}")

    full_msg = "\n".join(msg_parts)
    append_log(full_msg)

    if has_drift:
        warn_msg = f":warning: {full_msg}"
        print(warn_msg)
        if not args.no_discord:
            send_discord(warn_msg)
        return 1

    print("[OK] no drift detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
