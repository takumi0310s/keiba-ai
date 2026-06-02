#!/usr/bin/env python3
"""JRDB PACI 修復 scraper (Phase 26 / Task #13、 5/12 実装).

CLAUDE.md 既知 bug: `data/jrdb_paci.csv` が 4/4 で更新停止。

原因調査:
  - `data/jrdb/raw/Paci/PACI*.zip` 自体は 5/3 (mtime) まで取得済
  - `data/jrdb/extracted/Paci/KYI*.txt` も 5/9 まで存在
  - **しかし `tools/parse_jrdb.py` を再実行していないため
    `data/jrdb_paci.csv` 自体が更新されていない**
  - `tools/daily_jrdb_kyi.bat` は scrape_jrdb (LZH 個別取得) を呼ぶが
    PACI 一括取得 + parse_jrdb 再生成 step が無い → 5/3 以降止まっていた

本 module は 2 ステップ で 修復:
  1. JRDB から最新 PACI*.zip を取得 (`tools/download_jrdb.py` の関数 再利用)
  2. ZIP 内 KYI 抽出 → `tools/parse_jrdb.py` 再実行で
     `data/jrdb_paci.csv` 全件 再生成

★ 既存 file は 触らない。 `predict_core.py` / `daily_predict.py` / V15 model 不変。

usage:
  python tools/scrape_jrdb_paci.py                        # 最新 PACI 取得 + parse
  python tools/scrape_jrdb_paci.py --dry-run              # 取得対象 list のみ
  python tools/scrape_jrdb_paci.py --years 2026           # 2026 のみ
  python tools/scrape_jrdb_paci.py --skip-download        # download skip, parse のみ
  python tools/scrape_jrdb_paci.py --skip-parse           # download のみ、 parse skip
  python tools/scrape_jrdb_paci.py --since 20260403       # 指定日以降のみ取得
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import time
import warnings
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

# ★ import時は stdout を再ラップしない (daily_paci_refresh 等から import されると
#   二重ラップで silent_runner のログリダイレクト下に "I/O operation on closed file"
#   を起こすため)。 単体実行(__main__)時のみ utf-8 ラップする。
if sys.platform == "win32" and __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "jrdb" / "raw" / "Paci"
EXTRACT_DIR = BASE_DIR / "data" / "jrdb" / "extracted" / "Paci"
TARGET_CSV = BASE_DIR / "data" / "jrdb_paci.csv"

JRDB_BASE = "http://www.jrdb.com/member"
PACI_INDEX_URL = f"{JRDB_BASE}/datazip/Paci/index.html"


def load_credentials() -> tuple[str, str]:
    """Load JRDB_ID / JRDB_PASSWORD from .env."""
    env_path = BASE_DIR / ".env"
    jrdb_id = jrdb_pw = ""
    if not env_path.exists():
        return "", ""
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("JRDB_ID="):
                jrdb_id = line.split("=", 1)[1].strip('"').strip("'")
            elif line.startswith("JRDB_PASSWORD="):
                jrdb_pw = line.split("=", 1)[1].strip('"').strip("'")
    return jrdb_id, jrdb_pw


def get_paci_index(session: requests.Session, auth: HTTPBasicAuth,
                   years: list[int]) -> list[tuple[str, str]]:
    """Fetch PACI index page and list all matching files.

    Returns: list of (filename, href) tuples like
             ("PACI260509.zip", "PACI260509.zip")
    """
    from bs4 import BeautifulSoup

    resp = session.get(PACI_INDEX_URL, auth=auth, timeout=30, verify=True)
    resp.encoding = "shift_jis"
    if resp.status_code != 200:
        print(f"[!] PACI index fetch failed: status={resp.status_code}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    items: list[tuple[str, str]] = []
    yy_set = {str(y)[2:] for y in years}
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        href = a["href"]
        if not text.startswith("PACI") or ".zip" not in text:
            continue
        # PACI260509.zip -> yy = "26"
        yy = text[4:6]
        if yy in yy_set:
            items.append((text, href))
    return items


def download_zip(session: requests.Session, auth: HTTPBasicAuth,
                 url: str, dest: Path, force: bool = False) -> tuple[bool, str]:
    """Download a single ZIP file.

    force=True のとき、 既存ファイルがあっても再ダウンロードする
    (★ 当日/直近の PACI は部分版がキャッシュされていることがあるため強制取得 ★)。
    force=False でも、 サーバの content-length が既存サイズより大きければ再取得
    (部分版→完全版 への自動更新)。
    """
    if dest.exists() and dest.stat().st_size > 100 and not force:
        # 既存があっても、 サーバ側が大きい(=完全版に更新された)なら再DL
        try:
            h = session.head(url, auth=auth, timeout=30, verify=True,
                             allow_redirects=True)
            srv = int(h.headers.get("content-length", 0))
            loc = dest.stat().st_size
            if srv > 0 and srv > loc * 1.05:  # サーバが5%超大きい=完全版に更新
                print(f"  [update] {dest.name}: local {loc:,}B < server {srv:,}B → 再DL")
            else:
                return True, "skip"
        except Exception:
            return True, "skip"  # HEAD失敗時は従来通りskip(安全側)

    for attempt in range(3):
        try:
            resp = session.get(url, auth=auth, timeout=60, verify=True, stream=True)
            if resp.status_code == 404:
                return False, "404"
            if resp.status_code == 401:
                return False, "401-auth"
            if resp.status_code != 200:
                if attempt < 2:
                    time.sleep(3)
                    continue
                return False, f"http-{resp.status_code}"

            content_len = int(resp.headers.get("content-length", 0))
            if content_len < 100:
                return False, "empty"

            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True, f"{dest.stat().st_size:,}B"
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
                continue
            return False, str(e)

    return False, "max_retries"


def extract_zip(zip_path: Path, extract_to: Path) -> int:
    """Extract a ZIP file. Returns number of entries extracted."""
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
            return len(zf.namelist())
    except (zipfile.BadZipFile, OSError):
        return -1
    except Exception:
        return -1


def filename_to_date(filename: str) -> str | None:
    """PACI260509.zip -> '20260509'."""
    if not filename.startswith("PACI") or not filename.endswith(".zip"):
        return None
    yymmdd = filename[4:10]
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return None
    return f"20{yymmdd}"


def filter_by_since(items: list[tuple[str, str]],
                    since_yyyymmdd: str | None) -> list[tuple[str, str]]:
    """Keep only files dated >= since_yyyymmdd."""
    if not since_yyyymmdd:
        return items
    out = []
    for filename, href in items:
        date_str = filename_to_date(filename)
        if date_str and date_str >= since_yyyymmdd:
            out.append((filename, href))
    return out


def step_download(years: list[int], since: str | None,
                  dry_run: bool) -> dict:
    """Step 1: Download PACI*.zip files from JRDB."""
    print("=" * 60)
    print(" Step 1: PACI ZIP download")
    print("=" * 60)

    jrdb_id, jrdb_pw = load_credentials()
    if not (jrdb_id and jrdb_pw):
        print("[!] .env に JRDB_ID / JRDB_PASSWORD が無い")
        return {"status": "no-credentials", "downloaded": 0, "skipped": 0, "failed": 0}

    auth = HTTPBasicAuth(jrdb_id, jrdb_pw)
    session = requests.Session()
    session.headers.update({"User-Agent":
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) JRDB-PACI-Scraper"})

    print(f"  JRDB login: ID={jrdb_id[:4]}****  years={years}  since={since or 'none'}")
    items = get_paci_index(session, auth, years)
    items = filter_by_since(items, since)
    print(f"  index: {len(items)} PACI*.zip files matched")

    if dry_run:
        # 既存 file との diff
        existing = {p.name for p in RAW_DIR.glob("PACI*.zip")}
        to_dl = [it for it in items if it[0] not in existing]
        print(f"  DRY RUN: {len(items)} found, {len(to_dl)} new (not yet downloaded)")
        for fn, _ in to_dl[:30]:
            print(f"    NEW: {fn}")
        if len(to_dl) > 30:
            print(f"    ... and {len(to_dl) - 30} more")
        return {"status": "dry-run", "found": len(items), "new": len(to_dl)}

    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "extracted": 0}
    # ★ 直近(今日〜3日前)の PACI は部分版キャッシュを信用せず強制再DL ★
    recent_cut = (datetime.now() - timedelta(days=3)).strftime("%Y%m%d")
    for i, (filename, href) in enumerate(items, 1):
        url = f"{JRDB_BASE}/datazip/Paci/{href}"
        dest = RAW_DIR / filename
        fdate = filename_to_date(filename)
        force = bool(fdate and fdate >= recent_cut)  # 直近=強制再DL
        ok, msg = download_zip(session, auth, url, dest, force=force)
        if ok:
            if msg == "skip":
                stats["skipped"] += 1
            else:
                stats["downloaded"] += 1
                n = extract_zip(dest, EXTRACT_DIR)
                if n > 0:
                    stats["extracted"] += n
        else:
            stats["failed"] += 1
            print(f"  [!] {filename} FAILED ({msg})")

        if i % 20 == 0:
            print(f"  [{i}/{len(items)}] DL={stats['downloaded']} "
                  f"Skip={stats['skipped']} Fail={stats['failed']}")
        time.sleep(0.3)

    print(f"  Final: DL={stats['downloaded']} Skip={stats['skipped']} "
          f"Fail={stats['failed']} Extracted={stats['extracted']} files")
    stats["status"] = "ok"
    return stats


def step_parse(dry_run: bool) -> dict:
    """Step 2: Run parse_jrdb.py to regenerate data/jrdb_paci.csv."""
    print()
    print("=" * 60)
    print(" Step 2: parse_jrdb.py → data/jrdb_paci.csv")
    print("=" * 60)

    parse_script = BASE_DIR / "tools" / "parse_jrdb.py"
    if not parse_script.exists():
        print(f"[!] {parse_script} not found")
        return {"status": "missing-parse-script"}

    if dry_run:
        print("  DRY RUN: would run parse_jrdb.py")
        before_mtime = TARGET_CSV.stat().st_mtime if TARGET_CSV.exists() else None
        print(f"  current jrdb_paci.csv: "
              f"{datetime.fromtimestamp(before_mtime).isoformat() if before_mtime else 'missing'}")
        return {"status": "dry-run"}

    before_mtime = TARGET_CSV.stat().st_mtime if TARGET_CSV.exists() else 0
    before_size = TARGET_CSV.stat().st_size if TARGET_CSV.exists() else 0
    print(f"  before: {TARGET_CSV.name} "
          f"mtime={datetime.fromtimestamp(before_mtime).isoformat() if before_mtime else 'none'} "
          f"size={before_size:,}B")

    cmd = [sys.executable, str(parse_script)]
    print(f"  run: {' '.join(cmd)}")
    t0 = time.time()
    try:
        # capture limited tail of output for log
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True,
                              text=True, timeout=1800, encoding="utf-8",
                              errors="replace")
        elapsed = time.time() - t0
        print(f"  parse_jrdb.py rc={proc.returncode} elapsed={elapsed:.1f}s")
        if proc.returncode != 0:
            print("  --- stderr tail ---")
            for line in (proc.stderr or "").splitlines()[-15:]:
                print(f"    {line}")
            return {"status": "parse-error", "rc": proc.returncode}
        # show KYI summary tail
        out = proc.stdout or ""
        lines = out.splitlines()
        in_kyi = False
        for ln in lines:
            if "KYI" in ln and "(Paci" in ln:
                in_kyi = True
            if in_kyi:
                print(f"    {ln}")
            if in_kyi and ln.startswith("---") and "KYI" not in ln:
                break
    except subprocess.TimeoutExpired:
        return {"status": "parse-timeout"}

    after_mtime = TARGET_CSV.stat().st_mtime if TARGET_CSV.exists() else 0
    after_size = TARGET_CSV.stat().st_size if TARGET_CSV.exists() else 0
    print(f"  after:  {TARGET_CSV.name} "
          f"mtime={datetime.fromtimestamp(after_mtime).isoformat() if after_mtime else 'none'} "
          f"size={after_size:,}B (diff {after_size - before_size:+,}B)")

    return {
        "status": "ok",
        "elapsed_s": elapsed,
        "before_size": before_size,
        "after_size": after_size,
        "size_diff": after_size - before_size,
    }


def main():
    p = argparse.ArgumentParser(
        description="JRDB PACI 修復 scraper (Task #13、 jrdb_paci.csv 4/4 stop 修復)"
    )
    p.add_argument("--years", nargs="+", type=int,
                   default=[datetime.now().year - 1, datetime.now().year],
                   help="target years (default: current and previous year)")
    p.add_argument("--since", default=None,
                   help="only fetch files dated YYYYMMDD or later")
    p.add_argument("--dry-run", action="store_true",
                   help="show what would be done, no actual download/parse")
    p.add_argument("--skip-download", action="store_true",
                   help="skip step 1 (download), run only parse")
    p.add_argument("--skip-parse", action="store_true",
                   help="skip step 2 (parse), run only download")
    args = p.parse_args()

    print(f"JRDB PACI 修復 scraper  start={datetime.now().isoformat(timespec='seconds')}")
    print(f"  TARGET_CSV    = {TARGET_CSV}")
    print(f"  RAW_DIR       = {RAW_DIR}")
    print(f"  EXTRACT_DIR   = {EXTRACT_DIR}")
    print(f"  args          = years={args.years} since={args.since} "
          f"dry_run={args.dry_run}")
    print()

    summary: dict = {}
    if not args.skip_download:
        summary["download"] = step_download(args.years, args.since, args.dry_run)
    else:
        print("  -- step 1 (download) skipped via --skip-download")

    if not args.skip_parse:
        summary["parse"] = step_parse(args.dry_run)
    else:
        print("  -- step 2 (parse) skipped via --skip-parse")

    print()
    print("=" * 60)
    print(f" SUMMARY  end={datetime.now().isoformat(timespec='seconds')}")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
