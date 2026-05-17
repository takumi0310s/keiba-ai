"""
P0-5 Live Orchestrator main entry — 朝 8:30 fire (SAT/SUN)

★ V15 production 不変、 race -20/-15/-10 min で 各 race 順次処理 ★
★ G1 day blocklist 遵守、 戦略⑦案 C 適用 ★

usage:
  python tools/live_orchestrator_main.py [--date YYYYMMDD] [--mock] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "data" / "live_orchestrator_log"

# ★ G1 day blocklist (5/17 G1 day) ★
G1_DAY_BLOCKLIST = ["20260517"]


def log_event(date_str: str, event: dict) -> None:
    """log 出力 (data/live_orchestrator_log/{date}.log)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{date_str}.log"
    event["timestamp"] = datetime.now().isoformat()
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def is_g1_day_blocked(date_str: str) -> bool:
    return date_str in G1_DAY_BLOCKLIST


def load_today_races(date_str: str) -> list[dict]:
    """daily_predictions/{date}.csv から race list 取得."""
    import pandas as pd

    path = REPO / "data" / "daily_predictions" / f"{date_str}.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path, dtype={"race_id": str}, encoding="utf-8-sig")
    races: list[dict] = []
    for _, row in df.iterrows():
        races.append(
            {
                "race_id": str(row["race_id"]),
                "race_name": str(row.get("race_name", "")),
                "course": str(row.get("course", "")),
                "condition": str(row.get("condition", "")),
                "distance": row.get("distance"),
                "num_horses": row.get("num_horses"),
            }
        )
    return races


def process_race(
    race_id: str,
    date_str: str,
    race_meta: dict,
    mock: bool = True,
    dry_run: bool = True,
) -> dict:
    """1 race 処理 (-20/-15 min 順次)."""
    log_event(date_str, {"event": "process_race_start", "race_id": race_id})

    # tools/ を path に追加 (sibling import 対応)
    tools_path = str(REPO / "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)

    # 戦略⑦案 C skip check (recalc_15min から流用)
    try:
        from recalc_15min import is_strategy_7c_excluded
    except Exception as e:
        log_event(
            date_str,
            {"event": "import_fail", "race_id": race_id, "error": str(e)},
        )
        return {"status": "import_fail", "error": str(e)}

    skip, reason = is_strategy_7c_excluded(race_meta)
    if skip:
        log_event(
            date_str,
            {"event": "strategy_7c_skip", "race_id": race_id, "reason": reason},
        )
        return {"status": "skip", "reason": reason}

    # -20 min: live_data_fetcher
    try:
        from live_data_fetcher import fetch_pre_features

        fetched = fetch_pre_features(
            race_id, date_str=date_str, mock=mock, dry_run=dry_run
        )
        fetch_status = fetched.get("status") if isinstance(fetched, dict) else None
        # ★ fetch_pre_features は dry_run 時に status 未設定で features を返すため、
        #   明示 fail (blocked/fail) のみ NG 判定する ★
        if fetch_status in ("blocked", "fail"):
            log_event(
                date_str,
                {"event": "fetch_fail", "race_id": race_id, "fetched": fetch_status},
            )
            return {"status": "fetch_fail", "fetched": fetched}
    except Exception as e:
        log_event(
            date_str,
            {"event": "fetch_exception", "race_id": race_id, "error": str(e)},
        )
        return {"status": "fetch_exception", "error": str(e)}

    # -15 min: recalc_15min
    try:
        from recalc_15min import run_recalc

        recalc = run_recalc(
            race_id, date_str=date_str, mock=mock, dry_run=dry_run
        )
        recalc_status = recalc.get("status") if isinstance(recalc, dict) else None
        log_event(
            date_str,
            {
                "event": "recalc_done",
                "race_id": race_id,
                "status": recalc_status,
            },
        )
        return {"status": "ok", "recalc_status": recalc_status}
    except Exception as e:
        log_event(
            date_str,
            {"event": "recalc_exception", "race_id": race_id, "error": str(e)},
        )
        return {"status": "recalc_exception", "error": str(e)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    date_str = args.date or datetime.now().strftime("%Y%m%d")
    log_event(
        date_str,
        {
            "event": "orchestrator_start",
            "mock": args.mock,
            "dry_run": args.dry_run,
        },
    )

    # G1 day blocklist (★ 5/17 強制 skip ★)
    if is_g1_day_blocked(date_str):
        log_event(date_str, {"event": "g1_day_blocked"})
        print(f"[INFO] {date_str} = G1 day blocked, skipping")
        return 0

    # daily_predictions check
    races = load_today_races(date_str)
    if not races:
        log_event(date_str, {"event": "no_races"})
        print(f"[WARN] {date_str} = no races (daily_predictions not found)")
        return 0

    log_event(date_str, {"event": "races_loaded", "n": len(races)})

    # 全 R 順次処理 (★ mock=True / dry_run=True で 実 fetch なし ★)
    results = {"ok": 0, "skip": 0, "fail": 0}
    for race in races:
        result = process_race(
            race["race_id"],
            date_str,
            race,
            mock=args.mock,
            dry_run=args.dry_run,
        )
        status = result.get("status")
        if status == "ok":
            results["ok"] += 1
        elif status == "skip":
            results["skip"] += 1
        else:
            results["fail"] += 1

    log_event(date_str, {"event": "orchestrator_done", "results": results})
    print(f"[OK] orchestrator done: {results}")

    n_total = len(races)
    if n_total > 0 and results["fail"] >= n_total * 0.5:
        return 2  # 半分以上 fail = critical
    if results["fail"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
