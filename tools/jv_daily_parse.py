# -*- coding: utf-8 -*-
"""JV-Link 日次パース (2026-08-12 第1弾-2)。

data/jvlink/daily/*.dat (jv_daily_fetch.ps1 の生保存・UTF-8 1行1レコード) を
record type で振り分けて CSV へ append+dedup。冪等 (processed.json で処理済管理)。

パース精度の方針 (raw-first):
  - SE: jvlink_parser.parse_se (バイトオフセット実測検証済) の full parse
  - HR/O1-O6/WC/RA/UM: メタ列 (日付/場/R等) + raw blob 保持
    → .dat も恒久保存のため、後からパーサ強化時に全量再パース可能
  - 行→bytes は line.encode('cp932') (JV は SJIS 固定長。unicode→cp932 で原バイト列復元)
"""
from __future__ import annotations
import glob, json, os, sys
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "tools"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

from jvlink_parser import JVLinkParser  # noqa: E402

DAILY = os.path.join(BASE, "data", "jvlink", "daily")
OUT = os.path.join(BASE, "data", "jvlink")
PROC = os.path.join(DAILY, "processed.json")
P = JVLinkParser() if hasattr(JVLinkParser, "__call__") else None

# record_type → (出力csv, dedupキー)
ROUTES = {
    "SE": ("jv_se.csv", ["_event_date", "course_code", "race_num", "umaban"]),
    "HR": ("jv_hr.csv", ["_event_date", "course_code", "race_num"]),
    "RA": ("jv_ra.csv", ["_event_date", "course_code", "race_num"]),
    "UM": ("jv_um.csv", ["horse_id"]),
    "WC": ("jv_wc.csv", ["_event_date", "horse_id", "raw_wood"]),
    "O1": ("jv_o1.csv", ["_event_date", "course_code", "race_num"]),
    "O2": ("jv_o2.csv", ["_event_date", "course_code", "race_num"]),
    "O3": ("jv_o3.csv", ["_event_date", "course_code", "race_num"]),
    "O4": ("jv_o4.csv", ["_event_date", "course_code", "race_num"]),
    "O5": ("jv_o5.csv", ["_event_date", "course_code", "race_num"]),
    "O6": ("jv_o6.csv", ["_event_date", "course_code", "race_num"]),
    # HC = 坂路調教 (SLOP dataspec の実レコード型。WC=ウッドチップ)
    "HC": ("jv_hc.csv", ["train_date", "blood_num", "train_time"]),
}


def parse_hc_local(raw: bytes) -> dict:
    """HC 坂路調教 (58B固定長・2026-08-12 実レコードから経験的に確定)。
    [0:2]HC [2]区分 [3:11]作成日 [11]トレセン(0/1) [12:20]調教年月日
    [20:24]時刻HHMM [24:34]血統登録番号 [34:58]タイム部(4F計+ラップ、詳細は
    jv_training_features 側で分解・検証)"""
    a = lambda s, e: raw[s:e].decode("ascii", errors="replace")
    return {
        "record_type": "HC",
        "make_date": a(3, 11),
        "train_center": a(11, 12),
        "train_date": a(12, 20),
        "train_time": a(20, 24),
        "blood_num": a(24, 34),
        "time_block": a(34, 58),
        "_event_date": a(12, 20),
    }

# レース系レコードの開催日は [11:19] (parse_hr/o1 の _event_date=[3:11] は作成日で
# 2開催日が dedup 衝突するため上書きする)
_RACE_EVENT_AT_11 = {"HR", "RA", "O1", "O2", "O3", "O4", "O5", "O6"}


def get_parser():
    try:
        return JVLinkParser()
    except TypeError:
        return JVLinkParser.__new__(JVLinkParser)


def parse_line(parser, line):
    line = line.lstrip("﻿")  # StreamWriter の BOM が先頭レコードに付く対策
    rt = line[:2]
    try:
        raw = line.encode("cp932", errors="replace")
    except Exception:
        return rt, None
    if rt == "HC":
        fn = parse_hc_local  # jvlink_parser.parse_hc は誤スライス(horse_idがヘッダ跨ぎ)のため必ずローカル
    else:
        fn = getattr(parser, f"parse_{rt.lower()}", None)
    if fn is None:
        if rt.startswith("O") and len(rt) == 2:
            fn = getattr(parser, "parse_o1", None)  # O2-O6 も同メタ構造 → 流用しblob保持
        if fn is None:
            return rt, None
    try:
        d = fn(raw)
        d["record_type"] = rt
        if rt in _RACE_EVENT_AT_11 and len(raw) > 26:
            ev = raw[11:19].decode("ascii", errors="replace")
            if ev.isdigit():
                d["_event_date"] = ev
                d["course_code"] = raw[19:21].decode("ascii", errors="replace")
                d["race_num"] = raw[25:27].decode("ascii", errors="replace")
        return rt, d
    except Exception:
        return rt, None


def main():
    parser = get_parser()
    done = set()
    if os.path.exists(PROC):
        try:
            done = set(json.load(open(PROC, encoding="utf-8")))
        except Exception:
            pass
    files = [f for f in sorted(glob.glob(os.path.join(DAILY, "*.dat")))
             if os.path.basename(f) not in done]
    if not files:
        print("[jv_parse] 新規 .dat なし")
        return 0
    buckets: dict[str, list] = {}
    counts: dict[str, int] = {}
    for fp in files:
        for line in open(fp, encoding="utf-8", errors="replace"):
            line = line.rstrip("\r\n")
            if len(line) < 3:
                continue
            rt, d = parse_line(parser, line)
            counts[rt] = counts.get(rt, 0) + 1
            if d is not None and rt in ROUTES:
                buckets.setdefault(rt, []).append(d)
    print(f"[jv_parse] files={len(files)} 種別内訳: "
          + " ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    for rt, rows in buckets.items():
        out_csv, keys = ROUTES[rt]
        new = pd.DataFrame(rows)
        # blob 列は raw_* 名。dedup キーに無い列欠損は許容
        path = os.path.join(OUT, out_csv)
        if os.path.exists(path):
            old = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
            new = pd.concat([old, new.astype(str)], ignore_index=True)
        kcols = [k for k in keys if k in new.columns]
        if kcols:
            new = new.drop_duplicates(subset=kcols, keep="last")
        new.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  {out_csv}: +{len(rows)} → 計 {len(new):,}")
    done |= {os.path.basename(f) for f in files}
    json.dump(sorted(done), open(PROC, "w", encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
