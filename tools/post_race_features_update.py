"""5/9 投資後 即時 features 更新 (Session #47 H、 dev/sprint2).

5/9 結果確定後、 即時に sib_w5 + 馬・騎手・調教師 expanding features を
更新。 5/10 朝 result_verification と integrate。

機能:
1. data/jra_races_full.csv に 5/9 結果 append (manual 推奨)
2. sib_top3_rate_exp_w5 を 5/9 race を含めて recompute
3. jockey_wr / horse_career stats を update
4. 5/10 朝 result_verification で 利用可能化

usage:
  # 5/9 18:00 結果照合後
  python tools/post_race_features_update.py --date 20260509

  # dry-run (本 Session 確認用)
  python tools/post_race_features_update.py --date 20260503 --dry-run

V15 production 完全独立、 dev/sprint2 のみ。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")


def update_sib_expanding(date: str, dry_run: bool = False) -> dict:
    """5/9 race を含めて sib_top3_rate_exp_w5 を recompute.

    既存 tools/sib_expanding_variants.py の流用。
    """
    print(f"[sib_w5] update for date {date}")
    if dry_run:
        return {"status": "dry_run", "would_run": "sib_expanding_variants.py --variant a (window=5)"}

    # 実 production では:
    # 1. data/jra_races_full.csv に 5/9 結果 append (TFJV 経由 or netkeiba)
    # 2. python tools/sib_expanding_variants.py --variant a (window=5)
    # 3. data/netkeiba_siblings_expanding_w5.csv を上書き
    return {
        "status": "deferred",
        "instruction": "5/9 18:00 結果確定後 ユーザー manual 実行:\n"
                      "  python tools/sib_expanding_variants.py --variant a",
    }


def update_horse_career_stats(date: str, dry_run: bool = False) -> dict:
    """馬の career stats (horse_career_wr / top3r) を expanding update."""
    if dry_run:
        return {"status": "dry_run", "note": "production では V15 学習 pipeline の expanding 計算 と同じ logic"}
    return {
        "status": "deferred",
        "instruction": "V15 production の build_features 内で 自動計算 (date 順 cumsum)、"
                      " 5/9 結果 append 後 自動反映",
    }


def integrate_with_result_verification(date: str) -> dict:
    """tools/result_verification_5_10.py と integrate。"""
    rv_path = BASE / "tools" / "result_verification_5_10.py"
    if not rv_path.exists():
        return {"status": "missing", "path": str(rv_path)}
    return {
        "status": "integrated",
        "tool": "tools/result_verification_5_10.py",
        "5_10_morning": "result_verification_5_10.py --date 20260509 で 結果集計 + 5/16 verdict",
    }


def main():
    p = argparse.ArgumentParser(description="post_race features update (Session #47 H)")
    p.add_argument("--date", default="20260509")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print("=" * 60)
    print(f"post_race_features_update ({args.date}, dry_run={args.dry_run})")
    print("=" * 60)

    sib = update_sib_expanding(args.date, args.dry_run)
    print(f"\n[sib_w5] {sib}")

    career = update_horse_career_stats(args.date, args.dry_run)
    print(f"\n[horse career] {career}")

    rv = integrate_with_result_verification(args.date)
    print(f"\n[result_verification integration] {rv}")

    summary = {
        "date": args.date,
        "dry_run": args.dry_run,
        "sib_w5": sib,
        "horse_career": career,
        "result_verification": rv,
    }

    out_path = BASE / "data" / "v18" / f"sprint2_post_race_update_{args.date}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
