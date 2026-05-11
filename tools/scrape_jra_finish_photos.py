#!/usr/bin/env python
"""JRA 公式 入線写真 (ゴール写真) scraper (Phase 22 Agent C, 2026-05-11).

★ 実調査結果 ★
JRA 公式 (jra.go.jp) で 入線写真 を 確認できた 場所:
  /datafile/seiseki/g1/{race}/result/{race}{year}.html
    例: /datafile/seiseki/g1/takamatsu/result/takamatsu2025.html

  写真 URL pattern:
    /datafile/seiseki/g1/{race}/result/photo/{year}-{N}.jpg
    例: /datafile/seiseki/g1/takamatsu/result/photo/2025-1.jpg
        /datafile/seiseki/g1/takamatsu/result/photo/2025-2.jpg
        ...

平場 (G1 以外) は JRA 公式 では 入線写真 が 公開されて いない 模様。
(他 source 候補: netkeiba "result/photo" ページ、 各場 official、 SNS)

利用上の注意:
  - JRA 公式 (官庁) source、 個人 利用 範囲 で 取得 OK
  - 再 distribute は 規約確認 必要
  - 大量 download 禁止 (rate limit + 礼儀)
  - data/jra_finish_photos/ は .gitignore に追加

Usage:
    python tools/scrape_jra_finish_photos.py --dryrun
    python tools/scrape_jra_finish_photos.py --probe-url URL
    python tools/scrape_jra_finish_photos.py --year 2025 --download  # 実 DL
"""
import argparse
import io
import os
import re
import sys
import time
from typing import Optional
from urllib.parse import urlparse

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

OUT_DIR = "data/jra_finish_photos"
G1_INDEX_TEMPLATE = "/datafile/seiseki/replay/{year}/g1.html"


def fetch_g1_index(year: int) -> list:
    """{year}年 の G1 result page link 一覧."""
    url = BASE + G1_INDEX_TEMPLATE.format(year=year)
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.encoding = r.apparent_encoding or "shift_jis"
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r"/datafile/seiseki/g1/[^/]+/result/", href):
            full = href if href.startswith("http") else BASE + href
            links.append((a.get_text(strip=True), full))
    seen = set()
    uniq = []
    for t, u in links:
        if u in seen:
            continue
        seen.add(u)
        uniq.append((t, u))
    return uniq


def extract_finish_photo_urls(html: str) -> list:
    """G1 result page HTML から 入線写真 URL を抽出."""
    soup = BeautifulSoup(html, "html.parser")
    photos = []
    for img in soup.find_all("img", src=True):
        src = img["src"]
        if "/result/photo/" in src and src.lower().endswith((".jpg", ".jpeg", ".png")):
            full = src if src.startswith("http") else BASE + src
            photos.append(full)
    # 重複除外 + 順序保持
    seen = set()
    uniq = []
    for p in photos:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def race_id_from_url(url: str) -> str:
    """page URL から race_id 相当 ({race}{year}) を抽出.

    例: .../g1/takamatsu/result/takamatsu2025.html -> takamatsu2025
    """
    m = re.search(r"/g1/([^/]+)/result/([^/]+)\.html", url)
    if m:
        return m.group(2)
    return os.path.basename(urlparse(url).path).rsplit(".", 1)[0]


def download_photo(url: str, dest_dir: str, dryrun: bool = True) -> Optional[str]:
    """photo を DL (dryrun=True なら 取得せず URL のみ return)."""
    os.makedirs(dest_dir, exist_ok=True)
    fname = os.path.basename(urlparse(url).path)
    dest = os.path.join(dest_dir, fname)
    if dryrun:
        return dest  # 仮 path
    if os.path.exists(dest):
        return dest
    r = requests.get(url, headers=HEADERS, timeout=20, stream=True)
    if r.status_code != 200:
        return None
    with open(dest, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    time.sleep(1.0)
    return dest


def dryrun(year: int = 2025) -> None:
    print("=" * 70)
    print(f"  JRA finish-photo scraper — DRY-RUN  ({year}年 G1)")
    print("=" * 70)

    # Step 1: G1 result page 一覧
    print("\n[STEP 1] G1 result page 一覧")
    links = fetch_g1_index(year)
    print(f"  -> {len(links)} pages found")
    for t, u in links[:5]:
        print(f"     {t[:30]:30s} {u}")

    if not links:
        print("  no pages, abort")
        return

    # Step 2: 1 sample page から 写真 URL 抽出
    sample = links[0]
    print(f"\n[STEP 2] sample page から 写真 URL 抽出")
    print(f"  page: {sample[1]}")
    try:
        r = requests.get(sample[1], headers=HEADERS, timeout=15)
        r.encoding = r.apparent_encoding or "shift_jis"
        photos = extract_finish_photo_urls(r.text)
        print(f"  HTTP {r.status_code}, {len(photos)} photo URL(s)")
        for p in photos:
            print(f"     {p}")
    except Exception as e:
        print(f"  ERR: {e}")
        return

    # Step 3: 実 DL は skip (dryrun)
    if photos:
        race_id = race_id_from_url(sample[1])
        dest_dir = os.path.join(OUT_DIR, race_id)
        print(f"\n[STEP 3] DL skeleton (DRY-RUN: no actual file write)")
        for p in photos[:2]:
            dest = download_photo(p, dest_dir, dryrun=True)
            print(f"  would save: {p}\n          -> {dest}")

    print("\n[CONCLUSION]")
    print("  URL pattern 確定:")
    print("    page : /datafile/seiseki/g1/{race}/result/{race}{year}.html")
    print("    photo: /datafile/seiseki/g1/{race}/result/photo/{year}-N.jpg")
    print("  G1 のみ 取得可、 平場 は 別 source (netkeiba 等) 検討")


def probe_url(url: str) -> None:
    print(f"=== probe: {url} ===")
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.encoding = r.apparent_encoding or "shift_jis"
    print(f"HTTP {r.status_code}, {len(r.text)} bytes")
    if r.status_code == 200:
        photos = extract_finish_photo_urls(r.text)
        print(f"photos: {len(photos)}")
        for p in photos[:10]:
            print(f"  {p}")


def run_download(year: int) -> None:
    links = fetch_g1_index(year)
    print(f"found {len(links)} G1 page(s) for {year}")
    total = 0
    for label, page_url in links:
        try:
            r = requests.get(page_url, headers=HEADERS, timeout=15)
            r.encoding = r.apparent_encoding or "shift_jis"
            photos = extract_finish_photo_urls(r.text)
            race_id = race_id_from_url(page_url)
            dest_dir = os.path.join(OUT_DIR, race_id)
            for p in photos:
                dest = download_photo(p, dest_dir, dryrun=False)
                if dest:
                    total += 1
                    print(f"  saved: {dest}")
            time.sleep(2.0)
        except Exception as e:
            print(f"  ERR for {page_url}: {e}")
    print(f"total saved: {total}")


def main():
    p = argparse.ArgumentParser(description="JRA finish-photo scraper")
    p.add_argument("--dryrun", action="store_true")
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--probe-url", default=None)
    p.add_argument("--download", action="store_true",
                   help="実 DL (要 dryrun 確認後)")
    args = p.parse_args()

    if args.probe_url:
        probe_url(args.probe_url)
    elif args.download:
        run_download(args.year)
    elif args.dryrun:
        dryrun(year=args.year)
    else:
        p.print_help()
        print("\nヒント: --dryrun で 実調査")


if __name__ == "__main__":
    main()
