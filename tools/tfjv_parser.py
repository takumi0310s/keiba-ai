"""TARGET frontier JV (C:/TFJV) binary .DAT parser (Session #44 B).

JV-Data 仕様準拠の binary record を直接 parse。 既存 tools/extract_jvdata.py は
TARGET GUI export (TXT/keiba_data.csv) を入力としていたが、 本 parser は **直接
binary を読む** ため GUI export 不要。

主要 record type:
  RA: レース詳細 (race info)、 SE_DATA / ES_DATA / DE_DATA
  SE: 馬毎レース情報、 同上
  HR: 払戻金、 SE_DATA
  H1: 単複オッズ、 HY_DATA
  H6: 三連単オッズ、 HY_DATA
  UM: 馬個体、 UM_DATA
  KS: 騎手、 TFJ_KISI.DAT
  WF: WIN5、 W5_DATA

usage:
  # 1 ファイル parse (test)
  python tools/tfjv_parser.py --file C:/TFJV/SE_DATA/2025/SH202511.DAT

  # ディレクトリ全 file parse → CSV
  python tools/tfjv_parser.py --dir C:/TFJV/SE_DATA/2025 --type RA --out data/tfjv/RA_2025.csv

  # 全 SE_DATA から HR 払戻 のみ抽出
  python tools/tfjv_parser.py --root C:/TFJV/SE_DATA --type HR --out data/tfjv/HR_all.csv

V15 production 完全不変 (read-only parser、 出力は data/tfjv/ 別 dir)。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

BASE = Path(r"C:/Users/takum/keiba-ai")
TFJV_ROOT = Path(r"C:/TFJV")


# ===== JV-Data record field offsets (簡易、 主要 fields のみ) =====
# 仕様: https://jra-van.jp/dlb/manual/recordlayout/
# 各 datatype の record は CRLF 区切り、 内部 ASCII + Shift-JIS 漢字混在

RA_FIELD_OFFSETS = {
    # offset (start, length)
    "record_type":  (0,  2),
    "data_kbn":     (2,  1),
    "year":         (3,  4),
    "month_day":    (7,  4),    # MMDD
    "year_create":  (11, 4),    # 作成 年
    "create_md":    (15, 4),
    "course_code":  (19, 2),
    "kai":          (21, 2),
    "nichi":        (23, 2),
    "race_num":     (25, 2),
    # 後段は yobi (28)、 youbi、 開催情報 etc.、 spec full ref
    "youbi_code":   (27, 1),
    "race_name":    (28, 60),   # Shift-JIS、 60 bytes (30 chars)
    # ... (略、 race_name_short、 race_class、 surface、 distance 等 後段で追加)
}

SE_FIELD_OFFSETS = {
    "record_type":  (0,  2),
    "data_kbn":     (2,  1),
    "year":         (3,  4),
    "month_day":    (7,  4),
    "year_create":  (11, 4),
    "create_md":    (15, 4),
    "course_code":  (19, 2),
    "kai":          (21, 2),
    "nichi":        (23, 2),
    "race_num":     (25, 2),
    "wakuban":      (27, 1),
    "umaban":       (28, 2),
    "horse_id":     (30, 10),   # 血統登録番号 10 chars
    "horse_name":   (40, 36),   # Shift-JIS、 36 bytes (18 chars)
    # ... 性別、 年齢、 騎手、 騎手 id、 着順 等
}

HR_FIELD_OFFSETS = {
    "record_type":  (0,  2),
    "data_kbn":     (2,  1),
    "year":         (3,  4),
    "month_day":    (7,  4),
    "year_create":  (11, 4),
    "create_md":    (15, 4),
    "course_code":  (19, 2),
    "kai":          (21, 2),
    "nichi":        (23, 2),
    "race_num":     (25, 2),
    # 各 払戻種別 の offset は spec full ref (単勝 / 複勝 / 枠連 / 馬連 / 馬単 / ワイド / 三連複 / 三連単)
    # 簡易 parse で raw 全文を保存、 後段で詳細 parser
    "raw_payouts":  (27, -1),    # 残り全部
}

H1_FIELD_OFFSETS = {  # 単複オッズ
    "record_type":  (0,  2),
    "data_kbn":     (2,  1),
    "year":         (3,  4),
    "month_day":    (7,  4),
    "year_create":  (11, 4),
    "create_md":    (15, 4),
    "course_code":  (19, 2),
    "kai":          (21, 2),
    "nichi":        (23, 2),
    "race_num":     (25, 2),
    "raw_odds":     (27, -1),
}

UM_FIELD_OFFSETS = {  # 馬個体
    "record_type":  (0,  2),
    "data_kbn":     (2,  1),
    "horse_id":     (3, 10),
    "horse_name":   (13, 36),   # Shift-JIS
    # ... 性別、 父、 母、 母父、 生年月日 等
}

WF_FIELD_OFFSETS = {  # WIN5
    "record_type":  (0,  2),
    "data_kbn":     (2,  1),
    "year":         (3,  4),
    "month_day":    (7,  4),
    "year_create":  (11, 4),
    "create_md":    (15, 4),
    "raw_w5":       (19, -1),
}

PARSERS = {
    "RA": RA_FIELD_OFFSETS,
    "SE": SE_FIELD_OFFSETS,
    "HR": HR_FIELD_OFFSETS,
    "H1": H1_FIELD_OFFSETS,
    "UM": UM_FIELD_OFFSETS,
    "WF": WF_FIELD_OFFSETS,
}


def parse_record(raw: bytes, schema: dict, encoding: str = "shift_jis") -> dict:
    """raw bytes と offsets schema で record を dict に変換."""
    out = {}
    for name, (start, length) in schema.items():
        if length == -1:
            chunk = raw[start:]
        else:
            chunk = raw[start:start+length]
        try:
            if name in ("horse_name", "race_name") or name.endswith("_kanji"):
                # Shift-JIS 漢字 fields
                val = chunk.decode(encoding, errors="replace").rstrip("\x00 ")
            else:
                val = chunk.decode("ascii", errors="replace")
        except Exception:
            val = chunk.hex()
        out[name] = val.strip()
    return out


def iter_records(file_path: Path, target_type: str | None = None,
                 encoding: str = "shift_jis"):
    """1 .DAT ファイルから 各 record (CRLF 区切り) を yield."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
    except Exception as e:
        print(f"[!] read error {file_path}: {e}", file=sys.stderr)
        return

    # CRLF 区切り
    records = data.split(b"\r\n")
    for raw in records:
        if not raw or len(raw) < 4:
            continue
        try:
            rtype = raw[:2].decode("ascii", errors="replace").strip()
        except Exception:
            continue
        if target_type and rtype != target_type:
            continue
        schema = PARSERS.get(rtype)
        if not schema:
            # 未知 type、 raw のみ
            yield {"record_type": rtype, "raw_size": len(raw),
                   "raw_head": raw[:50].decode("ascii", errors="replace")}
            continue
        parsed = parse_record(raw, schema, encoding=encoding)
        parsed["_file"] = file_path.name
        yield parsed


def iter_directory(dir_path: Path, target_type: str | None = None):
    """ディレクトリ配下の全 .DAT ファイルを iterate."""
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.upper().endswith(".DAT"):
                yield from iter_records(Path(root) / f, target_type)


def to_csv(records: Iterable[dict], out_path: Path,
           max_records: int | None = None, batch_size: int = 10000):
    """records を CSV に書き出し (encoding=utf-8-sig、 Excel 互換)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    columns = None
    f = None
    writer = None
    try:
        for rec in records:
            if columns is None:
                columns = list(rec.keys())
                f = open(out_path, "w", encoding="utf-8-sig", newline="")
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
                writer.writeheader()
            writer.writerow(rec)
            n += 1
            if max_records and n >= max_records:
                break
            if n % batch_size == 0:
                print(f"  written {n:,} records ...")
    finally:
        if f:
            f.close()
    return n


def cli():
    p = argparse.ArgumentParser(description="TFJV binary parser (Session #44 B)")
    p.add_argument("--file", help="1 .DAT ファイル parse (test 用)")
    p.add_argument("--dir", help="ディレクトリ配下 一括 parse")
    p.add_argument("--root", default=str(TFJV_ROOT), help="TFJV root (default C:/TFJV)")
    p.add_argument("--type", default=None,
                   choices=list(PARSERS.keys()) + [None],
                   help="record type filter (RA/SE/HR/H1/UM/WF)")
    p.add_argument("--out", default=None, help="出力 CSV path")
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--list-only", action="store_true", help="先頭 5 records 表示のみ")
    args = p.parse_args()

    t0 = time.time()
    n = 0

    if args.file:
        path = Path(args.file)
        recs = list(iter_records(path, args.type))
        if args.list_only:
            print(f"=== {path.name} (records {len(recs)}) ===")
            for i, r in enumerate(recs[:5]):
                print(f"  [{i}] {r}")
            return
        if args.out:
            out_path = BASE / args.out if not Path(args.out).is_absolute() else Path(args.out)
            n = to_csv(iter(recs), out_path, max_records=args.max_records)
            print(f"  written {n} records -> {out_path}")
        else:
            print(f"=== {path.name}: {len(recs)} records ===")
            for r in recs[:3]:
                print(r)

    elif args.dir:
        dir_path = Path(args.dir)
        records = iter_directory(dir_path, args.type)
        if args.list_only:
            for i, r in enumerate(records):
                if i >= 5: break
                print(f"  [{i}] {r}")
            return
        if args.out:
            out_path = BASE / args.out if not Path(args.out).is_absolute() else Path(args.out)
            n = to_csv(records, out_path, max_records=args.max_records)
            print(f"  written {n} records -> {out_path}")
    else:
        print("[!] --file or --dir 指定必須")
        sys.exit(1)

    print(f"  elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    cli()
