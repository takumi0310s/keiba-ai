"""JV-Link backfill 計画 (Session #40 D1).

5/24+ JRA-VAN 加入後の bulk fetch 計画を、 既存 data inventory から逆算。
- data/jra_races_full.csv の年別 row 数
- 既存 data の不足箇所 (gap)
- JV-Link で取得する datatype × 期間
- 推定 fetch 時間 / data 量

usage:
  python tools/jvlink_backfill_plan.py
  python tools/jvlink_backfill_plan.py --out data/v18/jvlink_backfill_plan.json

V15 production 完全不変 (read-only inventory)。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")


def inventory_jra_races() -> dict:
    p = BASE / "data" / "jra_races_full.csv"
    if not p.exists():
        return {"exists": False, "note": "missing"}
    df = pd.read_csv(p, usecols=["year", "month"], low_memory=False)
    df["year_int"] = pd.to_numeric(df["year"], errors="coerce")
    df["year_full"] = df["year_int"].apply(
        lambda y: 2000 + int(y) if pd.notna(y) and int(y) <= 30
        else (1900 + int(y) if pd.notna(y) else None)
    )
    by_year = df.groupby("year_full").size().sort_index().to_dict()
    by_year = {int(k): int(v) for k, v in by_year.items() if pd.notna(k)}
    return {
        "exists": True,
        "total_rows": int(len(df)),
        "by_year": by_year,
        "min_year": min(by_year) if by_year else None,
        "max_year": max(by_year) if by_year else None,
    }


def inventory_jrdb_kyi() -> dict:
    p = BASE / "data" / "jrdb_kyi.csv"
    if not p.exists():
        return {"exists": False}
    return {
        "exists": True,
        "size_mb": round(p.stat().st_size / 1024 / 1024, 1),
        "mtime": datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
    }


def inventory_jra_payouts() -> dict:
    p = BASE / "data" / "jra_payouts.csv"
    if not p.exists():
        return {"exists": False}
    df = pd.read_csv(p, dtype={"race_date": str}, usecols=["race_date"])
    return {
        "exists": True,
        "rows": int(len(df)),
        "min_date": str(df["race_date"].min()),
        "max_date": str(df["race_date"].max()),  # 4/6 停止確認用
    }


def inventory_blood() -> dict:
    p = BASE / "data" / "blood_full.csv"
    if not p.exists():
        return {"exists": False}
    df = pd.read_csv(p, usecols=["horse_id"], dtype={"horse_id": str})
    return {
        "exists": True,
        "horse_count": int(df["horse_id"].nunique()),
    }


def inventory_speed_index() -> dict:
    p = BASE / "data" / "netkeiba_speed_index.csv"
    if not p.exists():
        return {"exists": False}
    df = pd.read_csv(p, usecols=["race_id"], dtype={"race_id": str})
    df["year"] = df["race_id"].str[:4]
    by_year = df.groupby("year").size().to_dict()
    return {
        "exists": True,
        "rows": int(len(df)),
        "by_year": {str(k): int(v) for k, v in by_year.items()},
    }


def jvlink_backfill_targets() -> list[dict]:
    """JV-Link で取得すべき datatype × 期間 list."""
    today = datetime.date.today()
    one_year_ago = today - datetime.timedelta(days=365)
    five_year_ago = today - datetime.timedelta(days=365 * 5)
    return [
        {
            "datatype": "RACE",
            "purpose": "公式 race 詳細 (override jra_races_full.csv)",
            "from": "20100101",
            "to": today.strftime("%Y%m%d"),
            "estimated_records": 800_000,
            "priority": 1,
            "phase": "Phase 3 後半 (6/9-13)",
        },
        {
            "datatype": "HR",
            "purpose": "公式 払戻 (jra_payouts.csv 4/6 停止 解消)",
            "from": "20180101",  # 既存と同期間
            "to": today.strftime("%Y%m%d"),
            "estimated_records": 30_000,
            "priority": 1,
            "phase": "Phase 3 前半 (5/24-31)",
        },
        {
            "datatype": "O1",  # 単複オッズ
            "purpose": "公式 オッズ (paci_* 自前算出 + 当日リアルタイム)",
            "from": one_year_ago.strftime("%Y%m%d"),
            "to": today.strftime("%Y%m%d"),
            "estimated_records": 100_000,
            "priority": 2,
            "phase": "Phase 3 前半 (5/27-)",
        },
        {
            "datatype": "TCOV",
            "purpose": "公式 調教 (netkeiba 調教と整合チェック / 補完)",
            "from": one_year_ago.strftime("%Y%m%d"),
            "to": today.strftime("%Y%m%d"),
            "estimated_records": 200_000,
            "priority": 3,
            "phase": "Phase 3 後半 (6/9-13)",
        },
        {
            "datatype": "WOOD",
            "purpose": "公式 木馬場 (netkeiba と整合チェック / 補完)",
            "from": one_year_ago.strftime("%Y%m%d"),
            "to": today.strftime("%Y%m%d"),
            "estimated_records": 50_000,
            "priority": 3,
            "phase": "Phase 3 後半 (6/9-13)",
        },
        {
            "datatype": "BLOD",
            "purpose": "公式 血統 (blood_full.csv override)",
            "from": "20100101",
            "to": today.strftime("%Y%m%d"),
            "estimated_records": 80_000,
            "priority": 2,
            "phase": "Phase 3 後半 (6/14-20)",
        },
        {
            "datatype": "WF",
            "purpose": "馬体重 (Pattern B 当日情報)",
            "from": one_year_ago.strftime("%Y%m%d"),
            "to": today.strftime("%Y%m%d"),
            "estimated_records": 100_000,
            "priority": 2,
            "phase": "Phase 3 後半 (6/14-20)",
        },
    ]


def main():
    p = argparse.ArgumentParser(description="JV-Link backfill 計画")
    p.add_argument("--out", default="data/v18/jvlink_backfill_plan.json")
    args = p.parse_args()

    plan = {
        "generated": datetime.datetime.now().isoformat(),
        "inventory": {
            "jra_races_full": inventory_jra_races(),
            "jrdb_kyi": inventory_jrdb_kyi(),
            "jra_payouts": inventory_jra_payouts(),
            "blood": inventory_blood(),
            "netkeiba_speed_index": inventory_speed_index(),
        },
        "jvlink_targets": jvlink_backfill_targets(),
    }

    print("=" * 70)
    print("JV-Link backfill 計画")
    print("=" * 70)
    inv = plan["inventory"]
    print(f"  jra_races_full: rows={inv['jra_races_full'].get('total_rows', 'N/A'):,}, "
          f"年範囲 {inv['jra_races_full'].get('min_year')}-{inv['jra_races_full'].get('max_year')}")
    print(f"  jra_payouts: rows={inv['jra_payouts'].get('rows', 'N/A')}, "
          f"max_date={inv['jra_payouts'].get('max_date', 'N/A')}")
    print(f"  blood: {inv['blood'].get('horse_count', 'N/A'):,} horses")
    print(f"  speed_index: {inv['netkeiba_speed_index'].get('rows', 'N/A'):,} rows")

    print("\nJV-Link backfill targets (priority 順):")
    for t in sorted(plan["jvlink_targets"], key=lambda x: x["priority"]):
        print(f"  P{t['priority']} {t['datatype']:6s} ({t['from']}-{t['to']}) "
              f"~{t['estimated_records']:,} records  [{t['phase']}]")
        print(f"          {t['purpose']}")

    out_path = BASE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
