#!/usr/bin/env python
"""アメダス 1 分粒度 scraper (Phase 22 Agent C, 2026-05-11).

★ 重大な実調査結果 ★
気象庁 Bosai (`https://www.jma.go.jp/bosai/amedas/`) は **10 分粒度 が最細**。
公式 1 分粒度 endpoint は **公開されていない**。

裏付け:
  - /bosai/amedas/data/map/YYYYMMDDHHMMSS.json は 10分 step (= sun10m / precipitation10m)
  - /bosai/amedas/data/point/{stno}/YYYYMMDD_HH.json は 10分 step key (091000 / 092000 / ...)
  - data.jma.go.jp/obd/stats/etrn/ も `１０分ごとの値を表示` 明記
  - /bosai/amedas_h1m/ 系 endpoint (1分用の仮設 path) は 404
  - /bosai/amedas/data/point/{stno}/YYYYMMDDHHMM.json (1分粒度仮設) も 404

→ 1分 粒度 を 公的 source で 取得するには:
   a) 気象庁 JMA 有償 サービス (AMeDAS リアルタイム 1分値配信)
   b) 個別 観測所 の 別 API (現状 公開 source 未発見)

このスクリプトは **10 分粒度 が現状の最善** とした上で 取得 & 1分相当 への
最近傍補間 path を提供する。 補間粒度の透明化のため source 列に
"jma_bosai_10min" を明示し、 future 1分 source 発見時 に 差し替え可。

Usage:
    python tools/scrape_amedas_1min.py --dryrun --course 中山
    python tools/scrape_amedas_1min.py --course 東京 --date 20260511
"""
import argparse
import io
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import requests

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# 競馬場 → アメダス観測所 (scrape_weather.py と同期)
COURSE_TO_AMEDAS = {
    "札幌": "14163",
    "函館": "23232",
    "福島": "36127",
    "新潟": "54232",
    "東京": "44132",   # 府中
    "中山": "45147",   # 船橋
    "中京": "51106",   # 名古屋
    "京都": "61286",
    "阪神": "63518",   # 宝塚
    "小倉": "82182",   # 北九州
}

BASE = "https://www.jma.go.jp/bosai/amedas/data"


def fetch_point_10min(stno: str, dt: datetime) -> dict:
    """指定観測所 の YYYYMMDD_HH.json (10分粒度) を取得.

    Args:
        stno: 観測所コード (5桁)
        dt: 取得対象 datetime (時間単位)

    Returns:
        dict: {YYYYMMDDHHMM00: {temp/humidity/wind/precipitation10m/...}, ...}
              keys は HH 内の 10分 刻み (例 09:00, 09:10, 09:20, ..., 09:50)
    """
    yyyymmdd = dt.strftime("%Y%m%d")
    hh = dt.strftime("%H")
    url = f"{BASE}/point/{stno}/{yyyymmdd}_{hh}.json"
    r = requests.get(url, headers=HEADERS, timeout=10)
    if r.status_code != 200:
        return {}
    return r.json()


def probe_1min_endpoints(stno: str, dt: datetime) -> dict:
    """1 分粒度 endpoint の存在 を 系統的 に 確認 (DRY-RUN)."""
    yyyymmdd = dt.strftime("%Y%m%d")
    hh = dt.strftime("%H")
    yyyymmddhhmm = dt.strftime("%Y%m%d%H%M")
    candidates = [
        # 仮定 endpoint
        f"{BASE}/point/{stno}/{yyyymmddhhmm}.json",
        f"https://www.jma.go.jp/bosai/amedas_h1m/data/point/{stno}/{yyyymmdd}_{hh}.json",
        f"https://www.jma.go.jp/bosai/amedas_1min/data/point/{stno}/{yyyymmdd}_{hh}.json",
        f"{BASE}/point_1min/{stno}/{yyyymmdd}_{hh}.json",
        f"{BASE}/map_1min/{yyyymmddhhmm}00.json",
    ]
    out = {}
    for url in candidates:
        try:
            r = requests.get(url, headers=HEADERS, timeout=5)
            out[url] = r.status_code
        except Exception as e:
            out[url] = f"ERR:{e}"
    return out


def expand_10min_to_1min(records: dict) -> list:
    """10 分粒度 dict を 1 分粒度 list に 最近傍補間.

    output: [{timestamp, temp, humidity, wind, wind_dir, precip_1min,
              source, granularity_real}]
    """
    rows = []
    keys = sorted(records.keys())
    if not keys:
        return rows

    # 各 10 分 record を 10 行 (1 分単位) に複製
    for k in keys:
        try:
            dt = datetime.strptime(k, "%Y%m%d%H%M%S")
        except ValueError:
            continue
        obs = records[k]
        temp = _get_first(obs, "temp")
        humidity = _get_first(obs, "humidity")
        wind = _get_first(obs, "wind")
        wind_dir = _get_first(obs, "windDirection")
        precip_10m = _get_first(obs, "precipitation10m")
        # 10 分降水 を 1 分相当 に 等分 (粗 approx, 真の 1 分降水 ではない)
        precip_1min = (precip_10m / 10.0) if precip_10m is not None else None

        for i in range(10):
            ts = dt + timedelta(minutes=i)
            rows.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "temp": temp,
                "humidity": humidity,
                "wind": wind,
                "wind_dir": wind_dir,
                "precipitation_1min_est": precip_1min,
                "source": "jma_bosai_10min",
                "granularity_real": "10min_interpolated_to_1min",
            })
    return rows


def _get_first(obs: dict, key: str):
    v = obs.get(key)
    if isinstance(v, list) and v:
        return v[0]
    return v


def save_csv(rows: list, path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(_csv_val(r.get(c)) for c in cols) + "\n")


def _csv_val(v) -> str:
    if v is None:
        return ""
    return str(v)


def dryrun(course: str = "中山") -> None:
    stno = COURSE_TO_AMEDAS.get(course)
    if not stno:
        print(f"unknown course: {course}")
        return
    # 1 時間前 の data を 対象 に (現在時刻 直近 は data 揃っていない 可能性)
    dt = datetime.now() - timedelta(hours=1)
    dt = dt.replace(minute=0, second=0, microsecond=0)

    print("=" * 70)
    print(f"  AMeDAS 1-min scraper — DRY-RUN")
    print(f"  course={course}  stno={stno}  hour={dt}")
    print("=" * 70)

    # Step 1: 既存 10 分 endpoint
    print("\n[STEP 1] 既存 10 分粒度 endpoint (公式)")
    rec10 = fetch_point_10min(stno, dt)
    print(f"  /point/{stno}/{dt:%Y%m%d}_{dt:%H}.json -> {len(rec10)} keys")
    if rec10:
        sample_key = sorted(rec10.keys())[0]
        print(f"  sample key: {sample_key}")
        print(f"  fields: {list(rec10[sample_key].keys())[:10]}")

    # Step 2: 1 分粒度 endpoint 候補 を probe
    print("\n[STEP 2] 1 分粒度 endpoint 候補 を 系統 probe")
    probe = probe_1min_endpoints(stno, dt)
    has_1min = False
    for url, st in probe.items():
        marker = " ★" if st == 200 else ""
        print(f"  HTTP {st}{marker}  {url}")
        if st == 200:
            has_1min = True

    print("\n[STEP 3] 結論")
    if has_1min:
        print("  → 1 分 endpoint 発見、 後で 実装 へ 切替")
    else:
        print("  → 1 分 endpoint 全て 404、 公式 1 分 は 公開されて いない 確定")
        print("  → 現状 path: 10 分粒度 を 取得 + 1 分 に 補間 (source 明示)")

    # Step 4: 1 分相当 への 補間
    print("\n[STEP 4] 10 分 → 1 分 補間 sample (現状の 最善)")
    rows = expand_10min_to_1min(rec10)
    print(f"  expanded {len(rows)} 1-min rows from {len(rec10)} 10-min rec")
    if rows:
        print(f"  first: {rows[0]}")
        print(f"  last:  {rows[-1]}")


def run_save(course: str, date_str: Optional[str] = None) -> None:
    stno = COURSE_TO_AMEDAS.get(course)
    if not stno:
        print(f"unknown course: {course}")
        return
    if date_str:
        dt0 = datetime.strptime(date_str, "%Y%m%d")
    else:
        dt0 = datetime.now() - timedelta(days=1)
        dt0 = dt0.replace(hour=0, minute=0, second=0, microsecond=0)

    # 1 日 分 (24 時間) を 取得
    all_rec = {}
    for h in range(24):
        dt = dt0 + timedelta(hours=h)
        rec = fetch_point_10min(stno, dt)
        all_rec.update(rec)

    rows = expand_10min_to_1min(all_rec)
    out = f"data/amedas_1min_{dt0:%Y%m%d}_{course}.csv"
    save_csv(rows, out)
    print(f"saved {len(rows)} rows -> {out}")


def main():
    p = argparse.ArgumentParser(description="AMeDAS 1-min scraper (with 10-min interp)")
    p.add_argument("--dryrun", action="store_true")
    p.add_argument("--course", default="中山")
    p.add_argument("--date", default=None, help="YYYYMMDD (default: yesterday)")
    args = p.parse_args()

    if args.dryrun:
        dryrun(course=args.course)
    else:
        run_save(course=args.course, date_str=args.date)


if __name__ == "__main__":
    main()
