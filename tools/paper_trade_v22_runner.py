"""Phase 20: paper_trade_engine_v22.py 拡張 runner.

Phase 17 の tools/paper_trade_engine_v22.py (5-model 戦略 関数 + console
runner) はそのまま維持し、 本 file で daily 運用向け wrapper を提供。

強化内容:
  - --date YYYYMMDD で 1 日のみ評価 (5/17 本番運用 ready)
  - --threshold-scan で V15 score threshold sweep (0.60/0.65/0.70/0.75/0.80)
  - per-race detail CSV (data/v22/paper_trade_detail_YYYYMMDD.csv)
  - per-model summary CSV (data/v22/paper_trade_summary_YYYYMMDD.csv)
  - rolling summary (data/v22/paper_trade_rolling.csv)
  - --notify で Discord 通知 (data/discord_dedup_state.json で dedup)
  - schtask-friendly: 例外時も exit 0 (V15 production を巻き込まない)
  - 案 B 改 strict (戦略 ⑦) は paper_trade_engine_v22.paper_v15 で既反映

V15 production: 完全 read-only。 predict_core / daily_predict / app.py 不変。

Usage:
  python tools/paper_trade_v22_runner.py                   # 履歴全体 比較
  python tools/paper_trade_v22_runner.py --date 20260517   # 1 日のみ
  python tools/paper_trade_v22_runner.py --notify          # Discord 通知
  python tools/paper_trade_v22_runner.py --threshold-scan  # threshold sweep
  python tools/paper_trade_v22_runner.py --rolling         # rolling 表示
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

BASE = Path(r"C:/Users/takum/keiba-ai")
sys.path.insert(0, str(BASE / "tools"))

OUT_DIR = BASE / "data" / "v22"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEDUP_PATH = BASE / "data" / "discord_dedup_state.json"
DEFAULT_BET = 700

try:
    from backtest_engine import load_history, RaceRecord  # noqa: F401
    from paper_trade_engine_v22 import (
        paper_v15, paper_v18, paper_v20, paper_v21, paper_v22,
    )
except Exception as e:  # noqa: BLE001
    print(f"[paper_v22_runner] FATAL import: {e}", file=sys.stderr)
    raise


# =====================================================================
# Per-race evaluation (detail capture)
# =====================================================================

def _evaluate_with_detail(
    races: list,
    strategy_fn: Callable,
    label: str,
) -> dict:
    n_total = len(races)
    n_bet = 0
    n_hit = 0
    total_inv = 0
    total_pay = 0
    detail: list[dict] = []

    for r in races:
        try:
            decision = strategy_fn(r) or {}
        except Exception as e:  # noqa: BLE001
            decision = {"bet": 0, "reason": f"strategy_err_{type(e).__name__}"}
        bet = int(decision.get("bet", 0) or 0)
        reason = str(decision.get("reason", ""))
        if bet == 0:
            detail.append({
                "race_id": r.race_id, "date": r.date, "course": r.course,
                "race_num": r.race_num, "model": label,
                "bet": 0, "reason": reason, "hit": "",
                "investment": 0, "payout": 0, "pnl": 0,
            })
            continue
        n_bet += 1
        total_inv += bet
        hit = bool(r.trio_hit)
        ratio = bet / DEFAULT_BET
        pay = int(r.payout * ratio) if hit else 0
        total_pay += pay
        if hit:
            n_hit += 1
        detail.append({
            "race_id": r.race_id, "date": r.date, "course": r.course,
            "race_num": r.race_num, "model": label,
            "bet": bet, "reason": reason, "hit": int(hit),
            "investment": bet, "payout": pay, "pnl": pay - bet,
        })

    return {
        "model": label,
        "n_total": n_total,
        "n_bet": n_bet,
        "n_hit": n_hit,
        "investment": total_inv,
        "payout": total_pay,
        "profit": total_pay - total_inv,
        "roi_pct": round(100.0 * total_pay / max(total_inv, 1), 2),
        "hit_rate": round(100.0 * n_hit / max(n_bet, 1), 2),
        "detail": detail,
    }


# =====================================================================
# V22 RL model loader (gracefully optional)
# =====================================================================

def _load_v22_model():
    path = BASE / "models" / "v22" / "v22_rl_ppo_phase17_initial.zip"
    if not path.exists():
        return None
    try:
        from stable_baselines3 import PPO
        return PPO.load(str(path))
    except Exception as e:  # noqa: BLE001
        print(f"[paper_v22_runner] V22 RL load failed: {e}", file=sys.stderr)
        return None


# =====================================================================
# Threshold scan
# =====================================================================

def _v15_at(threshold: float) -> Callable:
    """paper_v15 の score 閾値を threshold に置換した closure."""
    def fn(race):
        # paper_v15 既存ロジックを再現するため、 直接 import せず重複定義
        name = race.race_name
        is_grade = any(g in name for g in ("G1", "G2", "G3", "GⅠ", "GⅡ", "GⅢ"))
        is_special = "特別" in name and not is_grade
        if race.condition in ("E", "B"):
            return {"bet": 0, "reason": "cond_skip"}
        if is_special:
            return {"bet": 0, "reason": "tokubetsu_skip"}
        if race.morning_top1_score < threshold:
            return {"bet": 0, "reason": f"low_score_{threshold}"}
        return {"bet": DEFAULT_BET, "reason": f"V15@{threshold}"}
    return fn


def threshold_scan(races: list, thresholds: Iterable[float] = (0.60, 0.65, 0.70, 0.75, 0.80)) -> list[dict]:
    out = []
    for th in thresholds:
        s = _evaluate_with_detail(races, _v15_at(th), f"V15@{th}")
        s.pop("detail", None)
        out.append(s)
    return out


# =====================================================================
# Output helpers
# =====================================================================

DETAIL_FIELDS = [
    "race_id", "date", "course", "race_num", "model",
    "bet", "reason", "hit", "investment", "payout", "pnl",
]
SUMMARY_FIELDS = [
    "model", "n_total", "n_bet", "n_hit", "hit_rate",
    "investment", "payout", "profit", "roi_pct",
]


def _save_detail_csv(date_label: str, summaries: list[dict]) -> Path:
    path = OUT_DIR / f"paper_trade_detail_{date_label}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        w.writeheader()
        for s in summaries:
            for d in s.get("detail", []):
                w.writerow(d)
    return path


def _save_summary_csv(date_label: str, summaries: list[dict]) -> Path:
    path = OUT_DIR / f"paper_trade_summary_{date_label}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k) for k in SUMMARY_FIELDS})
    return path


def _update_rolling(date_label: str, summaries: list[dict]) -> Path:
    path = OUT_DIR / "paper_trade_rolling.csv"
    fieldnames = ["date"] + SUMMARY_FIELDS
    rows: list[dict] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("date") != date_label:
                    rows.append(r)
    for s in summaries:
        rows.append({"date": date_label, **{k: s.get(k) for k in SUMMARY_FIELDS}})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


# =====================================================================
# Discord notify (dedup)
# =====================================================================

def _notify_summary(summaries: list[dict], date_label: str) -> None:
    try:
        from notify import send_discord
    except Exception as e:  # noqa: BLE001
        print(f"[paper_v22_runner] notify import failed: {e}", file=sys.stderr)
        return

    seen: dict = {}
    if DEDUP_PATH.exists():
        try:
            seen = json.loads(DEDUP_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            seen = {}
    dedup_key = f"paper_v22_{date_label}"
    if dedup_key in seen:
        print(f"[paper_v22_runner] dedup skip: {dedup_key}")
        return

    lines = [f"## paper trade {date_label} (5-model)"]
    for s in summaries:
        lines.append(
            f"- {s['model']}: bet={s['n_bet']}/{s['n_total']} hit={s['n_hit']} "
            f"ROI={s['roi_pct']:.1f}% pnl={s['profit']:+}"
        )
    msg = "\n".join(lines)
    ok = send_discord(f"paper trade {date_label}", msg, color="green", channel="updates")
    print(f"[paper_v22_runner] discord: {'OK' if ok else 'SKIP'}")
    seen[dedup_key] = datetime.now().isoformat(timespec="seconds")
    DEDUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEDUP_PATH.write_text(json.dumps(seen, ensure_ascii=False, indent=2), encoding="utf-8")


# =====================================================================
# Main
# =====================================================================

def run(date: str | None, notify: bool, threshold_scan_mode: bool, rolling_mode: bool) -> int:
    if rolling_mode:
        path = OUT_DIR / "paper_trade_rolling.csv"
        if not path.exists():
            print("[paper_v22_runner] no rolling history yet")
            return 0
        print(path.read_text(encoding="utf-8-sig"))
        return 0

    races = load_history(start_date=date, end_date=date) if date else load_history()
    print(f"[paper_v22_runner] loaded {len(races)} races")
    if not races:
        print("[paper_v22_runner] no races to evaluate")
        return 0

    date_label = date or "all"

    if threshold_scan_mode:
        scans = threshold_scan(races)
        print("\n=== threshold scan (V15 + 戦略⑦) ===")
        print(f"{'model':<10} {'bet':>4} {'hit':>4} {'ROI%':>7} {'pnl':>8}")
        for s in scans:
            print(
                f"{s['model']:<10} {s['n_bet']:>4} {s['n_hit']:>4} "
                f"{s['roi_pct']:>6.1f}% {s['profit']:>+8}"
            )
        scan_path = OUT_DIR / f"paper_trade_threshold_scan_{date_label}.json"
        scan_path.write_text(json.dumps(scans, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {scan_path}")
        return 0

    v22_model = _load_v22_model()
    print(f"[paper_v22_runner] V22 RL model: {'loaded' if v22_model else 'unavailable'}")

    summaries: list[dict] = []
    summaries.append(_evaluate_with_detail(races, paper_v15, "V15"))
    summaries.append(_evaluate_with_detail(races, paper_v18, "V18 cand"))
    summaries.append(_evaluate_with_detail(races, paper_v20, "V20 cand"))
    summaries.append(_evaluate_with_detail(races, paper_v21, "V21 cand"))
    summaries.append(_evaluate_with_detail(
        races,
        lambda r: paper_v22(r, model=v22_model),
        "V22 RL",
    ))

    print(f"\n=== Paper trade 比較 ({date_label}、 5-model) ===")
    print(f"{'model':<10} {'bet':>4} {'hit':>4} {'inv':>8} {'pay':>8} "
          f"{'pnl':>8} {'ROI%':>7} {'hit%':>6}")
    print("-" * 80)
    for s in summaries:
        print(
            f"{s['model']:<10} {s['n_bet']:>4} {s['n_hit']:>4} "
            f"{s['investment']:>8} {s['payout']:>8} {s['profit']:>+8} "
            f"{s['roi_pct']:>6.1f}% {s['hit_rate']:>5.1f}%"
        )

    detail_path = _save_detail_csv(date_label, summaries)
    summary_path = _save_summary_csv(date_label, summaries)
    rolling_path = _update_rolling(date_label, summaries)
    print(f"\nsaved: {detail_path}")
    print(f"saved: {summary_path}")
    print(f"saved: {rolling_path}")

    if notify:
        _notify_summary(summaries, date_label)

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 20 paper_trade_v22 runner (5-model)")
    p.add_argument("--date", help="YYYYMMDD、 1 日のみ評価")
    p.add_argument("--notify", action="store_true", help="Discord 通知 (dedup 経由)")
    p.add_argument("--threshold-scan", action="store_true", help="V15 score threshold sweep")
    p.add_argument("--rolling", action="store_true", help="rolling summary 表示")
    args = p.parse_args()

    try:
        return run(args.date, args.notify, args.threshold_scan, args.rolling)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 0  # schtask-friendly


if __name__ == "__main__":
    sys.exit(main())
