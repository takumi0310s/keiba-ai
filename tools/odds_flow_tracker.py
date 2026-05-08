"""オッズ flow tracker (Session #45 D、 dev/sprint1).

直前 10 分のオッズ変動を track し、 「隠れた人気変化」 を検出。
真の市場予測 vs 表面オッズ の差分 features を生成。

機能:
1. 1 分毎に netkeiba から 単勝オッズを取得 (production 統合時)
2. 5 分前 / 10 分前 と比較
3. odds 急落 (人気急上昇) / 急騰 (人気急落) を flag
4. 各馬の 「flow score」 を 計算

usage:
  from tools.odds_flow_tracker import OddsFlowTracker
  tracker = OddsFlowTracker()
  tracker.snapshot(race_id, timestamp, odds_dict)  # 1 分毎
  features = tracker.compute_features(race_id, current_time)
  # → {umaban: {'flow_score': float, 'odds_change_5min': float, ...}}

V15 production 完全独立 (新規 module、 dev/sprint1 branch のみ)。
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

BASE = Path(r"C:/Users/takum/keiba-ai")
SNAPSHOT_DIR = BASE / "data" / "odds_flow_snapshots"


class OddsFlowTracker:
    """In-memory + disk persistent オッズ snapshot tracker."""

    def __init__(self, snapshot_dir: Optional[Path] = None):
        self.snapshot_dir = snapshot_dir or SNAPSHOT_DIR
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.cache = defaultdict(list)  # race_id → [(timestamp, odds_dict), ...]

    def snapshot(self, race_id: str, timestamp: datetime, odds_dict: dict) -> None:
        """単一 race の snapshot 記録 (timestamp + odds_dict)."""
        entry = {
            "timestamp": timestamp.isoformat(),
            "odds": odds_dict,
        }
        self.cache[race_id].append(entry)

        # disk persist (1 race per file)
        out = self.snapshot_dir / f"{race_id}.jsonl"
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_history(self, race_id: str) -> list:
        """disk から race 全 snapshot 読み込み."""
        path = self.snapshot_dir / f"{race_id}.jsonl"
        if not path.exists():
            return []
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    out.append(json.loads(line))
                except: continue
        return out

    def compute_features(self, race_id: str, current_time: datetime,
                         lookback_min: int = 10) -> dict:
        """各馬の flow features を計算.

        Returns:
            {umaban: {
                "current_odds": float,
                "odds_5min_ago": float,
                "odds_10min_ago": float,
                "odds_change_5min_pct": float,  # (current - 5min_ago) / 5min_ago * 100
                "odds_change_10min_pct": float,
                "flow_score": float,  # -1 (人気急落) ~ +1 (人気急上昇)
            }}
        """
        history = self.load_history(race_id)
        if len(history) < 2:
            return {}

        # parse timestamps
        for h in history:
            h["dt"] = datetime.fromisoformat(h["timestamp"])

        # 最新 snapshot (current)
        history.sort(key=lambda x: x["dt"])
        current = history[-1]

        # 5 / 10 分前 snapshot (最も近い)
        target_5min = current["dt"] - timedelta(minutes=5)
        target_10min = current["dt"] - timedelta(minutes=10)

        def find_closest(target):
            best = None
            best_diff = None
            for h in history:
                diff = abs((h["dt"] - target).total_seconds())
                if best_diff is None or diff < best_diff:
                    best = h
                    best_diff = diff
            # tolerance ±2 min
            if best_diff is not None and best_diff <= 120:
                return best
            return None

        snap_5min = find_closest(target_5min)
        snap_10min = find_closest(target_10min)

        out = {}
        for umaban, current_odds in current["odds"].items():
            try:
                co = float(current_odds)
            except: continue
            entry = {"current_odds": co}
            o5 = None; o10 = None
            if snap_5min and umaban in snap_5min["odds"]:
                try: o5 = float(snap_5min["odds"][umaban])
                except: pass
            if snap_10min and umaban in snap_10min["odds"]:
                try: o10 = float(snap_10min["odds"][umaban])
                except: pass

            entry["odds_5min_ago"] = o5
            entry["odds_10min_ago"] = o10

            # 5 min change (%)
            if o5 and o5 > 0:
                change_5 = (co - o5) / o5 * 100
                entry["odds_change_5min_pct"] = round(change_5, 2)
            else:
                entry["odds_change_5min_pct"] = 0.0

            # 10 min change (%)
            if o10 and o10 > 0:
                change_10 = (co - o10) / o10 * 100
                entry["odds_change_10min_pct"] = round(change_10, 2)
            else:
                entry["odds_change_10min_pct"] = 0.0

            # flow score: 急落 = +1 (人気急上昇)、 急騰 = -1 (人気急落)
            # change < -10% (10% 以上下落 = 人気急上昇) → +1
            # change > +10% (10% 以上上昇 = 人気急落) → -1
            change = entry["odds_change_5min_pct"]
            if change <= -10:
                entry["flow_score"] = 1.0
            elif change >= 10:
                entry["flow_score"] = -1.0
            else:
                entry["flow_score"] = -change / 10  # linear, e.g. -5% → +0.5

            out[umaban] = entry

        return out


def cli():
    p = argparse.ArgumentParser(description="odds flow tracker")
    p.add_argument("--race-id", required=True)
    p.add_argument("--simulate", action="store_true", help="simulation で sample data 生成")
    args = p.parse_args()

    tracker = OddsFlowTracker()

    if args.simulate:
        # 10 分前 → 5 分前 → 現在 の 3 snapshot 生成
        now = datetime.now()
        odds_10min = {"01": 5.0, "02": 8.0, "03": 12.0, "04": 20.0, "05": 50.0}
        odds_5min  = {"01": 4.5, "02": 8.0, "03": 11.0, "04": 22.0, "05": 50.0}
        odds_now   = {"01": 4.0, "02": 9.0, "03": 10.0, "04": 22.0, "05": 50.0}
        tracker.snapshot(args.race_id, now - timedelta(minutes=10), odds_10min)
        tracker.snapshot(args.race_id, now - timedelta(minutes=5), odds_5min)
        tracker.snapshot(args.race_id, now, odds_now)

    features = tracker.compute_features(args.race_id, datetime.now())
    print(json.dumps(features, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
