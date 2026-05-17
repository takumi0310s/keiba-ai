"""P0-5 順6: -15 min live data fetcher (★ dry-run / mock のみ、 production fetch は 5/24+ ★).

Phase 0-5 (-15 min 再計算) の data fetch layer。 3 source の skeleton:

  - JV-Link O1 (連続オッズ、 主軸) — JV-Link COM、 32-bit Python 必須
  - JV-Link TCOV (馬場 リアルタイム、 補助) — JV-Link COM
  - netkeiba 直前 odds (既存 fallback、 規約 audit PASS)

★ 重要 ★:
  - 本 file は G1 day (5/17 ヴィクトリアマイル) 中の作成。
  - G1 day blocklist で 実 fetch を強制拒否 (mock のみ可)。
  - JV-Link / netkeiba の **real fetch は 5/24 (Sat) 以降** に別 implementation で着手。
  - production deploy 前に user 判断 + 規約 audit + IP ban guard 必須。
  - rate limit 厳守 (1.5 sec inter-request = 40 req/min < 60 上限)。

V15 production 完全不変 (本 file は新規 tool、 predict_core / app.py 触らない)。

usage (mock dry-run のみ、 5/17 中):
  python tools/live_data_fetcher.py --race-id 202605020611 --mock --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIVE_DATA_DIR = REPO / "data" / "live_pre_features"

# rate limit (★ <= 60 req/min ★)
RATE_LIMIT_SEC = 1.5  # 1.5 sec inter-request = 40 req/min
JVLINK_FETCH_TIMEOUT = 30
HTTP_TIMEOUT = 15

# ★ G1 day blocklist — 該当日中は 実 fetch 永久禁止 ★
# (mock dry-run は OK。 real fetch は --mock False かつ blocklist 外 の日のみ)
G1_DAY_BLOCKLIST = [
    "20260517",  # ヴィクトリアマイル
]


def is_safe_to_fetch(now: datetime | None = None) -> tuple[bool, str]:
    """G1 day 中 / 規約違反 fetch 防止 guard.

    Args:
        now: 現在時刻 (test 用に注入可、 default は datetime.now())

    Returns:
        (safe, reason): safe=True なら fetch 可、 False なら拒否 + reason
    """
    if now is None:
        now = datetime.now()
    today_str = now.strftime("%Y%m%d")
    if today_str in G1_DAY_BLOCKLIST:
        return False, "g1_day_blocklist"
    return True, "ok"


def fetch_jvlink_o1(race_id: str, dry_run: bool = True) -> dict:
    """JV-Link O1 (連続オッズ) fetch.

    ★ dry_run=True 時 mock data 返す、 JVOpen 一切呼ばない ★
    Real fetch は 5/24+ 別 implementation で (32-bit Python venv + win32com)。

    Args:
        race_id: e.g. '202605020611'
        dry_run: True で mock data (★ 5/17 中 強制 ★)

    Returns:
        dict with odds_snapshot / tansho_odds / fukusho_odds / popularity
    """
    if dry_run:
        return {
            "source": "jvlink_o1_mock",
            "race_id": str(race_id),
            "fetched_at": datetime.now().isoformat(),
            "odds_snapshot": {
                # mock: 5 頭 odds (単勝 × 100 倍 想定)
                1: 8.0,
                2: 12.0,
                3: 4.0,
                4: 20.0,
                5: 1.5,
            },
            "tansho_odds": {1: 8.0, 2: 12.0, 3: 4.0, 4: 20.0, 5: 1.5},
            "fukusho_odds": {1: 2.5, 2: 3.0, 3: 1.8, 4: 4.5, 5: 1.2},
            "popularity": {5: 1, 3: 2, 1: 3, 2: 4, 4: 5},
        }

    # ★ Real fetch は 5/24+、 ここで JVOpen 呼ばない ★
    raise NotImplementedError(
        "Real JV-Link O1 fetch is reserved for 5/24+ implementation. "
        "Use dry_run=True (--mock) for now."
    )


def fetch_jvlink_tcov(race_id: str, dry_run: bool = True) -> dict:
    """JV-Link TCOV (馬場 リアルタイム) fetch.

    ★ dry_run=True 時 mock data ★

    Returns:
        dict with baba_condition / moisture_rate / cushion_value
    """
    if dry_run:
        return {
            "source": "jvlink_tcov_mock",
            "race_id": str(race_id),
            "fetched_at": datetime.now().isoformat(),
            "baba_condition": "良",
            "moisture_rate": 8.5,
            "cushion_value": 9.2,
        }
    raise NotImplementedError(
        "Real JV-Link TCOV fetch is reserved for 5/24+ implementation."
    )


def fetch_netkeiba_pre_odds(race_id: str, dry_run: bool = True) -> dict:
    """netkeiba 直前情報 fetch (fallback).

    ★ G1 day 中 IP ban risk、 dry_run=True 強制 ★

    Returns:
        dict with horse_weights / weight_diff
    """
    if dry_run:
        return {
            "source": "netkeiba_pre_mock",
            "race_id": str(race_id),
            "fetched_at": datetime.now().isoformat(),
            "horse_weights": {1: 488, 2: 502, 3: 460, 4: 478, 5: 490},
            "weight_diff": {1: 0, 2: 4, 3: -2, 4: 12, 5: -3},
        }
    raise NotImplementedError(
        "Real netkeiba fetch is reserved for 5/24+ + IP ban safety review."
    )


def merge_sources(
    o1_data: dict | None,
    tcov_data: dict | None,
    netkeiba_data: dict | None,
) -> dict:
    """3 source の data を merge、 unified features に変換.

    Args:
        o1_data / tcov_data / netkeiba_data: 各 source の dict or None

    Returns:
        dict with sources_used + merged fields
    """
    result: dict = {
        "merge_timestamp": datetime.now().isoformat(),
        "sources_used": [],
    }

    if o1_data:
        result["sources_used"].append("jvlink_o1")
        result["odds_snapshot"] = o1_data.get("odds_snapshot", {})
        result["tansho_odds"] = o1_data.get("tansho_odds", {})
        result["fukusho_odds"] = o1_data.get("fukusho_odds", {})
        result["popularity"] = o1_data.get("popularity", {})

    if tcov_data:
        result["sources_used"].append("jvlink_tcov")
        result["baba_condition"] = tcov_data.get("baba_condition")
        result["moisture_rate"] = tcov_data.get("moisture_rate")
        result["cushion_value"] = tcov_data.get("cushion_value")

    if netkeiba_data:
        result["sources_used"].append("netkeiba_pre")
        result["horse_weights"] = netkeiba_data.get("horse_weights", {})
        result["weight_diff"] = netkeiba_data.get("weight_diff", {})

    return result


def fetch_pre_features(
    race_id: str,
    date_str: str | None = None,
    dry_run: bool = True,
    mock: bool = True,
) -> dict:
    """-15 min features 取得 main entry point.

    Args:
        race_id: e.g. '202605020611'
        date_str: YYYYMMDD (default today)
        dry_run: True で file 書き込み skip + mock data
        mock: True 強制 mock (★ G1 day default True ★)

    Returns:
        dict with merged features OR status='blocked'/'fail' on error.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    # ★ safety guard ★
    if not mock:
        safe, reason = is_safe_to_fetch()
        if not safe:
            return {
                "status": "blocked",
                "reason": reason,
                "recommendation": "use --mock for dry-run",
            }

    # fetch each source (★ dry_run + mock 強制 ★)
    o1_data: dict | None = None
    tcov_data: dict | None = None
    netkeiba_data: dict | None = None
    effective_dry = dry_run or mock

    try:
        o1_data = fetch_jvlink_o1(race_id, dry_run=effective_dry)
        time.sleep(RATE_LIMIT_SEC)
    except Exception as e:
        print(f"[WARN] jvlink_o1 fetch fail: {e}", file=sys.stderr)

    try:
        tcov_data = fetch_jvlink_tcov(race_id, dry_run=effective_dry)
        time.sleep(RATE_LIMIT_SEC)
    except Exception as e:
        print(f"[WARN] jvlink_tcov fetch fail: {e}", file=sys.stderr)

    try:
        netkeiba_data = fetch_netkeiba_pre_odds(race_id, dry_run=effective_dry)
    except Exception as e:
        print(f"[WARN] netkeiba fetch fail: {e}", file=sys.stderr)

    if not any([o1_data, tcov_data, netkeiba_data]):
        return {"status": "fail", "reason": "all sources failed"}

    merged = merge_sources(o1_data, tcov_data, netkeiba_data)

    # save (dry_run=False 時のみ。 mock=True でも file 書き込みは OK だが
    #  test の決定性のため dry_run=True なら skip)
    if not dry_run:
        out_dir = LIVE_DATA_DIR / date_str
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{race_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        merged["output_file"] = str(out_file)

    merged["status"] = "ok"
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--race-id", required=True)
    ap.add_argument("--date", default=None, help="YYYYMMDD (default today)")
    ap.add_argument(
        "--mock",
        action="store_true",
        help="★ G1 day default、 mock data only (実 fetch しない) ★",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="実 fetch しない (mock 返す) + file 書き込み skip",
    )
    args = ap.parse_args()

    # ★ G1 day blocklist + 5/17 中 mock 強制 ★
    safe, reason = is_safe_to_fetch()
    forced_mock = (not safe) or args.mock

    result = fetch_pre_features(
        args.race_id,
        args.date,
        dry_run=args.dry_run or forced_mock,
        mock=forced_mock,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
