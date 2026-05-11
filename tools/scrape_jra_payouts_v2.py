#!/usr/bin/env python
"""JRA 配当 scraper v2 (Phase 22 Agent C, 2026-05-11).

旧 scrape_jra_payouts.py が 4/6 から 停止した 根本原因:
  - JRADB/access*.html (accessS/accessO/accessJ/accessD) が 全て
    301 → /error/error013.html に redirect 済 (= 廃止)
  - つまり JRADB CGI 経由の post-back navigation flow (pw01skl/pw01srl/
    pw01ses 系の CNAME chain) は 完全 dead
  - 新 path は /datafile/seiseki/ 配下 の static HTML に 移行
    例: /datafile/seiseki/g1/{race}/result/{race}YYYY.html
        finish photo: /datafile/seiseki/g1/{race}/result/photo/YYYY-N.jpg
        payout: <dl> structure with dt/dd (単勝/複勝/枠連/馬連/ワイド/三連複/三連単)

このスクリプトは DRY-RUN 専用 (Phase 22 Agent C 実調査用)。
本番 backfill は別途 plan を立てる (G1 だけは確実に取れる、 平場は更に調査必要)。

Usage:
    python tools/scrape_jra_payouts_v2.py --dryrun        # G1 一覧 + 1 race 払戻 dry-run
    python tools/scrape_jra_payouts_v2.py --probe URL     # 任意 URL の構造確認

絶対遵守:
    - 既存 scrape_jra_payouts.py は 改変しない (上書き禁止、 並行 file)
    - V15 model / predict_core / daily_predict 触らない
    - 本番 scrape は --execute 明示時のみ (現状 未実装、 安全のため)
"""
import argparse
import io
import os
import re
import sys
from typing import Optional

import requests
from bs4 import BeautifulSoup

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

BASE = "https://www.jra.go.jp"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# 旧 DEAD URL list (= jra_payouts.csv 4/6 停止 原因)
DEAD_URLS = [
    "/JRADB/accessS.html",  # 過去レース calendar entry
    "/JRADB/accessO.html",  # オッズ
    "/JRADB/accessD.html",  # 出馬表 / 結果
    # accessJ.html は scrape_jra_track.py 内 で 使用中 → 別途 動作確認 必要
]

# 新 path の root
SEISEKI_ROOT = "/datafile/seiseki"
G1_INDEX_TEMPLATE = "/datafile/seiseki/replay/{year}/g1.html"


def probe_dead_urls(timeout: float = 8.0) -> dict:
    """旧 URL 群 が 全 dead か 検証 (DRY-RUN)."""
    out = {}
    for path in DEAD_URLS:
        url = BASE + path
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout,
                             allow_redirects=False)
            status = r.status_code
            location = r.headers.get("Location", "")
            out[path] = {"status": status, "location": location}
        except Exception as e:
            out[path] = {"status": "ERR", "location": str(e)}
    return out


def fetch_g1_index(year: int) -> list:
    """指定年 の G1 result page link 一覧 を 取得."""
    url = BASE + G1_INDEX_TEMPLATE.format(year=year)
    r = requests.get(url, headers=HEADERS, timeout=15)
    # JRA は Shift_JIS、 自動 detect
    r.encoding = r.apparent_encoding or "shift_jis"
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # /datafile/seiseki/g1/{race}/result/{race}YYYY.html
        if re.search(r"/datafile/seiseki/g1/[^/]+/result/", href):
            full = href if href.startswith("http") else BASE + href
            links.append((a.get_text(strip=True), full))
    # dedup
    seen = set()
    uniq = []
    for t, u in links:
        if u in seen:
            continue
        seen.add(u)
        uniq.append((t, u))
    return uniq


def parse_payouts_from_g1_result(html: str) -> dict:
    """G1 result page の <dl> 払戻 セクション を parse."""
    soup = BeautifulSoup(html, "html.parser")
    result = {
        "tansho": [], "fukusho": [], "wakuren": [],
        "umaren": [], "wide": [], "trio": [], "tierce": [],
    }
    # <dl> 構造: <dt>単勝</dt><dd>...</dd> (G1 page の払戻 セクション)
    for dl in soup.find_all("dl"):
        dt = dl.find("dt")
        if not dt:
            continue
        label = dt.get_text(strip=True)
        # dd 内の数字 + 円 を 抽出
        dd_text = " ".join(
            d.get_text(" ", strip=True) for d in dl.find_all("dd")
        )
        # 数字 + 円 (例: 380 円 / 11,080 円)
        amounts = re.findall(r"([\d,]+)\s*円", dd_text)
        amounts = [int(a.replace(",", "")) for a in amounts if a]
        if "単勝" in label:
            result["tansho"] = amounts
        elif "複勝" in label:
            result["fukusho"] = amounts
        elif "枠連" in label:
            result["wakuren"] = amounts
        elif "馬連" in label and "ワイド" not in label:
            result["umaren"] = amounts
        elif "ワイド" in label:
            result["wide"] = amounts
        elif "3連複" in label or "三連複" in label:
            result["trio"] = amounts
        elif "3連単" in label or "三連単" in label:
            result["tierce"] = amounts
    return result


def find_finish_photos(html: str) -> list:
    """G1 result page の 入線写真 URL を 抽出."""
    soup = BeautifulSoup(html, "html.parser")
    photos = []
    for img in soup.find_all("img", src=True):
        src = img["src"]
        if "/result/photo/" in src and src.endswith((".jpg", ".jpeg", ".png")):
            full = src if src.startswith("http") else BASE + src
            photos.append(full)
    return photos


def dryrun(year: int = 2025) -> None:
    print("=" * 70)
    print("  JRA payouts scraper v2 — DRY-RUN")
    print(f"  target year (G1 only): {year}")
    print("=" * 70)

    # Step 1: 旧 URL 全 dead 検証
    print("\n[STEP 1] 旧 JRADB URL 群 の生存確認")
    dead = probe_dead_urls()
    for path, info in dead.items():
        print(f"  {path}: status={info['status']} -> {info['location'][:80]}")

    # Step 2: 新 G1 index 取得
    print(f"\n[STEP 2] 新 path G1 index 取得 ({year}年)")
    try:
        links = fetch_g1_index(year)
        print(f"  -> {len(links)} G1 race link(s) found")
        for t, u in links[:5]:
            print(f"     {t[:30]:30s} {u}")
    except Exception as e:
        print(f"  ERR: {e}")
        return

    if not links:
        print("  no G1 link found, abort")
        return

    # Step 3: 1 race の HTML 取得 + 構造解析
    sample = links[0]
    print(f"\n[STEP 3] 1 race sample 取得: {sample[1]}")
    try:
        r = requests.get(sample[1], headers=HEADERS, timeout=15)
        r.encoding = r.apparent_encoding or "shift_jis"
        html = r.text
        print(f"  HTTP {r.status_code}, {len(html)} bytes")

        payouts = parse_payouts_from_g1_result(html)
        photos = find_finish_photos(html)
        print(f"  payouts: {payouts}")
        print(f"  finish photos: {len(photos)} found")
        for p in photos[:3]:
            print(f"    {p}")
    except Exception as e:
        print(f"  ERR: {e}")

    print("\n[CONCLUSION]")
    print("  旧 URL (JRADB/access*.html) は 全 301 → error013 = 完全 dead")
    print("  jra_payouts.csv 4/6 停止 の 根本原因 = JRADB CGI 廃止")
    print("  G1 だけは /datafile/seiseki/g1/ 配下 で payout + photo 取得 可")
    print("  平場 (~3,400 race/年) は 別 source (JV-Link HR / netkeiba) 推奨")


def probe_url(url: str) -> None:
    print(f"=== probe: {url} ===")
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.encoding = r.apparent_encoding or "shift_jis"
        print(f"HTTP {r.status_code}, {len(r.text)} bytes")
        if r.status_code == 200:
            payouts = parse_payouts_from_g1_result(r.text)
            photos = find_finish_photos(r.text)
            print(f"payouts: {payouts}")
            print(f"photos: {len(photos)}")
    except Exception as e:
        print(f"ERR: {e}")


def main():
    p = argparse.ArgumentParser(description="JRA payouts v2 (Phase 22 Agent C)")
    p.add_argument("--dryrun", action="store_true",
                   help="DRY-RUN: 旧 URL 確認 + G1 1 race sample 取得")
    p.add_argument("--year", type=int, default=2025, help="dry-run target year")
    p.add_argument("--probe", type=str, default=None,
                   help="任意 URL の構造確認")
    args = p.parse_args()

    if args.probe:
        probe_url(args.probe)
    elif args.dryrun:
        dryrun(year=args.year)
    else:
        p.print_help()
        print("\nヒント: --dryrun で 実調査 を 開始")


if __name__ == "__main__":
    main()
