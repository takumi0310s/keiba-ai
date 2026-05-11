"""30 year backtest data collection skeleton (Phase 22 / Session #86).

TFJV (C:/TFJV) 配下の 14 datatype を 1995-2026 全期間 parse する skeleton。
Session #44 で実装した tools/tfjv_parser.py を import して活用、 年×datatype 単位で
parquet (or csv) に書き出す。

設計 doc: docs/BACKTEST_30_YEAR_DESIGN.md (Session #84)

★ V15 production 完全不変 (read-only collector、 出力は data/backtest_30year/)。

usage:
  # DRY-RUN: 存在 file 列挙 + 容量 見積り のみ
  python tools/backtest_30year_collect.py --dry-run

  # SE 1995-2009 (V15 学習 data 2010+ と重複しない 15 年) を parquet で
  python tools/backtest_30year_collect.py --year-from 1995 --year-to 2009 \
      --datatype SE --output data/backtest_30year/

  # 全 datatype 全期間 (要 100 GB 以上 disk)
  python tools/backtest_30year_collect.py --year-from 1995 --year-to 2026 \
      --datatype RA,SE,HR,H1,UM,WF,SK,DM,HC,WH,HN,TR,BR,BT \
      --output data/backtest_30year/
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

# 既存 parser を活用
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from tools.tfjv_parser import (
        PARSERS as TFJV_PARSERS,
        iter_records,
        iter_directory,
    )
except Exception:  # pragma: no cover  (fallback for ad-hoc run)
    from tfjv_parser import (  # type: ignore
        PARSERS as TFJV_PARSERS,
        iter_records,
        iter_directory,
    )


BASE = Path(r"C:/Users/takum/keiba-ai")
TFJV_ROOT = Path(r"C:/TFJV")
DEFAULT_OUT = BASE / "data" / "backtest_30year"


# ===== datatype → TFJV directory mapping =====
# (14 datatypes spec). Session #84 BACKTEST_30_YEAR_DESIGN 準拠。
# 実在しない datatype は skeleton で raw_dir=None としておき、 dry-run で warn する。
DATATYPE_DIR = {
    # core (Session #44 で parser 動作確認済)
    "RA": "RA_DATA",   # race info  (TFJV では SE 内に混在の場合あり、 兼用 SE_DATA)
    "SE": "SE_DATA",   # 馬毎レース情報
    "HR": "HR_DATA",   # 払戻         (TFJV では SE 内 / HY 内 mix)
    "H1": "HY_DATA",   # 単複オッズ
    "UM": "UM_DATA",   # 馬個体
    "WF": "W5_DATA",   # WIN5
    # 拡張 (skeleton、 parser TODO)
    "SK": "ES_DATA",   # ES = 速報 SE (旧称 SK_DATA)、 一部環境では SK
    "DM": "DE_DATA",   # 競走馬除外 DE
    "HC": "CK_DATA",   # CK = check / 出走確定 (HC 兼用)
    "WH": "BS_DATA",   # WH = 騎乗者変更 / 馬名変更 (BS_DATA 旧 BS)
    "HN": "BY_DATA",   # 馬名 (BY)
    "TR": "TM_DATA",   # TM = タイム
    "BR": "BR_DATA",   # 血統
    "BT": "BT_DATA",   # 血統登録 (旧 BR と差別化)
}

# ===== 年×datatype 別 推定容量 (GB)  Session #84 試算ベース =====
# raw 圧縮済 .DAT (TFJV) の年あたり 平均 size。 30 年 合計の概算。
# (Glob 実測値 from worktree investigation 5/11)
SIZE_GB_PER_YEAR = {
    "RA": 0.05,   # 50 MB/year (RA は SE/ES 混在で控えめ)
    "SE": 0.06,   # 60 MB/year (1.9 GB / 30 年 = 63 MB)
    "HR": 0.02,
    "H1": 0.07,   # 2.0 GB / 30 年 = 67 MB
    "UM": 0.02,   # 497 MB / 90 年 ≒ 5 MB、 30 年で 150 MB
    "WF": 0.005,  # 7 MB / 15 年 = 0.5 MB
    "SK": 0.02,
    "DM": 0.01,
    "HC": 0.03,   # 657 MB / 23 年 = 28 MB
    "WH": 0.005,
    "HN": 0.005,
    "TR": 0.003,
    "BR": 0.002,
    "BT": 0.005,
}

# parquet 化で features 200+ × 215 万 rows = 50-100 GB (Session #84 試算)
# 本 collect 段階では raw record の dict → parquet/csv で ~3 GB 想定


def datatype_dir(dtype: str) -> Path | None:
    """datatype → TFJV subdir."""
    name = DATATYPE_DIR.get(dtype)
    if not name:
        return None
    p = TFJV_ROOT / name
    return p if p.exists() else None


def list_year_files(dtype: str, year: int) -> list[Path]:
    """指定 datatype × year に該当する .DAT file 列挙."""
    root = datatype_dir(dtype)
    if not root:
        return []
    yroot = root / str(year)
    if not yroot.exists():
        return []
    return sorted(yroot.glob("*.DAT"))


def estimate_capacity(datatypes: list[str], yfrom: int, yto: int) -> dict:
    """容量 見積り (GB) を計算."""
    years = yto - yfrom + 1
    per_type: dict[str, dict] = {}
    total_gb = 0.0
    for dt in datatypes:
        gb_per_year = SIZE_GB_PER_YEAR.get(dt, 0.01)
        sub = gb_per_year * years
        per_type[dt] = {
            "years": years,
            "gb_per_year": gb_per_year,
            "estimated_gb": round(sub, 3),
        }
        total_gb += sub
    return {
        "year_from": yfrom,
        "year_to": yto,
        "datatypes": list(datatypes),
        "total_years": years,
        "per_datatype": per_type,
        "estimated_raw_gb": round(total_gb, 2),
        "estimated_parquet_gb": round(total_gb * 1.5, 2),   # parquet expand ~1.5x
        "estimated_features_gb": round(total_gb * 20, 2),    # +features 50-100 GB
    }


def dry_run(datatypes: list[str], yfrom: int, yto: int) -> dict:
    """存在 file 列挙 + 容量 見積り のみ (parse しない)."""
    report = {
        "mode": "dry_run",
        "year_from": yfrom,
        "year_to": yto,
        "datatypes": datatypes,
        "file_summary": {},
        "missing_years": {},
    }
    grand_total_files = 0
    for dt in datatypes:
        root = datatype_dir(dt)
        if not root:
            report["file_summary"][dt] = {"status": "MISSING_DIR", "root": None}
            continue
        per_year = {}
        missing = []
        total_files = 0
        for y in range(yfrom, yto + 1):
            files = list_year_files(dt, y)
            per_year[y] = len(files)
            total_files += len(files)
            if len(files) == 0:
                missing.append(y)
        report["file_summary"][dt] = {
            "status": "OK" if total_files > 0 else "EMPTY",
            "root": str(root),
            "total_files": total_files,
            "year_breakdown": per_year,
        }
        if missing:
            report["missing_years"][dt] = missing
        grand_total_files += total_files
    report["grand_total_files"] = grand_total_files
    report["capacity_estimate"] = estimate_capacity(datatypes, yfrom, yto)
    return report


def collect_year(dtype: str, year: int, out_dir: Path,
                 format: str = "parquet", max_records: int | None = None) -> dict:
    """1 年分 1 datatype を抽出して file へ書き出し.

    output: {out_dir}/{year}/{dtype}.parquet (or .csv.gz)
    """
    files = list_year_files(dtype, year)
    if not files:
        return {"datatype": dtype, "year": year, "status": "no_files", "n": 0}

    year_dir = out_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    ext = "parquet" if format == "parquet" else "csv.gz"
    out_path = year_dir / f"{dtype}.{ext}"

    records: list[dict] = []
    n = 0
    t0 = time.time()
    for f in files:
        for rec in iter_records(f, target_type=dtype):
            rec["_year"] = year
            rec["_datatype"] = dtype
            records.append(rec)
            n += 1
            if max_records and n >= max_records:
                break
        if max_records and n >= max_records:
            break

    # write
    if format == "parquet":
        try:
            import pandas as pd
            df = pd.DataFrame(records)
            df.to_parquet(out_path, index=False)
        except ImportError:
            # fallback to csv.gz
            out_path = out_path.with_suffix("").with_suffix(".csv.gz")
            _write_csv_gz(records, out_path)
    else:
        _write_csv_gz(records, out_path)

    return {
        "datatype": dtype,
        "year": year,
        "status": "ok",
        "n": n,
        "elapsed": round(time.time() - t0, 2),
        "out": str(out_path),
    }


def _write_csv_gz(records: list[dict], out_path: Path) -> None:
    if not records:
        return
    columns = list(records[0].keys())
    with gzip.open(out_path, "wt", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def collect_range(datatypes: list[str], yfrom: int, yto: int,
                  out_dir: Path, format: str = "parquet",
                  max_records: int | None = None) -> dict:
    """全 year × 全 datatype を順次 collect."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    t0 = time.time()
    for y in range(yfrom, yto + 1):
        for dt in datatypes:
            r = collect_year(dt, y, out_dir, format=format,
                             max_records=max_records)
            results.append(r)
            if r["status"] == "ok":
                print(f"  [{y} {dt}] n={r['n']:,} elapsed={r['elapsed']}s -> {r['out']}")
            elif r["status"] == "no_files":
                print(f"  [{y} {dt}] (no files)")
    return {
        "summary_total_elapsed": round(time.time() - t0, 1),
        "results": results,
    }


def parse_datatypes(s: str) -> list[str]:
    out = []
    for tok in s.split(","):
        t = tok.strip().upper()
        if not t:
            continue
        if t not in DATATYPE_DIR:
            print(f"[!] unknown datatype: {t} (skip). known: {sorted(DATATYPE_DIR)}",
                  file=sys.stderr)
            continue
        out.append(t)
    return out


def cli():
    p = argparse.ArgumentParser(
        description="30 year backtest data collector (Phase 22 skeleton)")
    p.add_argument("--year-from", type=int, default=1995)
    p.add_argument("--year-to", type=int, default=2026)
    p.add_argument("--datatype", default="RA,SE,HR,H1,UM,WF",
                   help="comma-separated, e.g. RA,SE,HR")
    p.add_argument("--output", default=str(DEFAULT_OUT))
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    p.add_argument("--dry-run", action="store_true",
                   help="存在 file 列挙 + 容量 見積り のみ (parse しない)")
    p.add_argument("--max-records", type=int, default=None,
                   help="datatype × year ごとの 上限 (test 用)")
    p.add_argument("--report-out", default=None,
                   help="json レポート path (dry-run / 本実行とも)")
    args = p.parse_args()

    dtypes = parse_datatypes(args.datatype)
    if not dtypes:
        print("[!] valid datatype が無いので abort", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = BASE / args.output

    if args.dry_run:
        report = dry_run(dtypes, args.year_from, args.year_to)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.report_out:
            Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.report_out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"  report -> {args.report_out}")
        return

    print(f"=== collect_range datatypes={dtypes} years={args.year_from}-{args.year_to} ===")
    print(f"  output: {out_dir}")
    print(f"  format: {args.format}")
    cap = estimate_capacity(dtypes, args.year_from, args.year_to)
    print(f"  estimated raw: {cap['estimated_raw_gb']} GB / "
          f"parquet: {cap['estimated_parquet_gb']} GB")

    summary = collect_range(
        dtypes, args.year_from, args.year_to, out_dir,
        format=args.format, max_records=args.max_records)
    print(f"\n=== done: {summary['summary_total_elapsed']}s ===")
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"  report -> {args.report_out}")


if __name__ == "__main__":
    cli()
