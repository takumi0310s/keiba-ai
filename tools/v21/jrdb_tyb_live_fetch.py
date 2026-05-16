"""
JRDB TYB (直前情報) live fetch — 5/9 から fetch 停止 した TYB を 復旧 + 当日分 取得

V15 production 完全不変。 download_jrdb.py の URL pattern を踏襲、 Tyb のみ daily fetch。
朝 (06:00 or 当日朝 schtask 起動時) で 当日分 TYB.zip を ダウンロード → 解凍 → CSV merge。

usage:
  python tools/v21/jrdb_tyb_live_fetch.py                  # 当日朝 06:00 想定
  python tools/v21/jrdb_tyb_live_fetch.py --date 20260516  # 特定日付
  python tools/v21/jrdb_tyb_live_fetch.py --rebuild-csv    # extracted → jrdb_tyb.csv 再構築

source: https://www.jrdb.com/member/datazip/Tyb/index.html
auth: .env JRDB_ID + JRDB_PASSWORD
"""
from __future__ import annotations
import argparse
import io
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.auth import HTTPBasicAuth

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "data" / "jrdb" / "raw" / "Tyb"
EXTRACT_DIR = REPO / "data" / "jrdb" / "extracted" / "Tyb"
JRDB_BASE = "http://www.jrdb.com/member"


def load_credentials() -> tuple[str, str]:
    env_path = REPO / ".env"
    jrdb_id = jrdb_pw = ""
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("JRDB_ID="):
                jrdb_id = line.split("=", 1)[1].strip("\"'")
            elif line.startswith("JRDB_PASSWORD="):
                jrdb_pw = line.split("=", 1)[1].strip("\"'")
    if not jrdb_id or not jrdb_pw:
        raise RuntimeError(".env に JRDB_ID / JRDB_PASSWORD 未設定")
    return jrdb_id, jrdb_pw


def fetch_index(session: requests.Session, auth) -> list[tuple[str, str]]:
    """Tyb index page を 取得、 (filename, href) list を返す."""
    url = f"{JRDB_BASE}/datazip/Tyb/index.html"
    resp = session.get(url, auth=auth, timeout=30)
    resp.encoding = "shift_jis"
    soup = BeautifulSoup(resp.text, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        href = a["href"]
        if "TYB" in text and ".zip" in text.lower():
            out.append((text, href))
    return out


def download_file(session, url, dest_path: Path, auth) -> tuple[bool, str]:
    if dest_path.exists() and dest_path.stat().st_size > 100:
        return True, "skip"
    for attempt in range(3):
        try:
            resp = session.get(url, auth=auth, timeout=60, stream=True)
            if resp.status_code == 404:
                return False, "404"
            if resp.status_code == 401:
                return False, "401"
            if resp.status_code != 200:
                if attempt < 2:
                    time.sleep(3)
                    continue
                return False, f"http_{resp.status_code}"
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return True, f"{dest_path.stat().st_size:,}B"
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
                continue
            return False, str(e)
    return False, "max_retries"


def extract_zip(zip_path: Path, extract_to: Path) -> int:
    extract_to.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
            return len(zf.namelist())
    except Exception:
        return -1


def fetch_recent_tyb(date_str: str | None = None, max_days_back: int = 30) -> dict:
    """index page から TYB ファイル一覧 取得、 未取得 (or date 指定) 分を ダウンロード."""
    jrdb_id, jrdb_pw = load_credentials()
    auth = HTTPBasicAuth(jrdb_id, jrdb_pw)
    session = requests.Session()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] fetching Tyb index...")
    files = fetch_index(session, auth)
    print(f"[INFO] index lists {len(files)} files")

    # date 指定があれば その date のみ
    target_files = []
    if date_str:
        yy = date_str[2:4]
        mmdd = date_str[4:8]
        prefix = f"TYB{yy}{mmdd}"
        target_files = [(fn, href) for fn, href in files if fn.startswith(prefix)]
    else:
        # 最近 max_days_back 日分の files を 抽出 (filename ベース)
        target_files = [(fn, href) for fn, href in files if "TYB26" in fn]

    print(f"[INFO] target: {len(target_files)} files")

    summary = {"target": len(target_files), "downloaded": 0, "skipped": 0, "failed": 0, "extracted": 0}
    new_files = []
    for fn, href in target_files:
        dest = RAW_DIR / fn
        file_url = f"{JRDB_BASE}/datazip/Tyb/{href}"
        ok, msg = download_file(session, file_url, dest, auth)
        if ok:
            if msg == "skip":
                summary["skipped"] += 1
            else:
                summary["downloaded"] += 1
                n = extract_zip(dest, EXTRACT_DIR)
                if n > 0:
                    summary["extracted"] += n
                new_files.append(fn)
                print(f"  [+] {fn} ({msg})")
        else:
            summary["failed"] += 1
            print(f"  [!] {fn} FAILED ({msg})")
        time.sleep(0.4)

    return {"summary": summary, "new_files": new_files}


def rebuild_tyb_csv() -> dict:
    """extracted/Tyb/*.txt を 全 parse して data/jrdb_tyb.csv 再構築."""
    # 既存 parser を使う (parse_jrdb.py 等)。 ここでは ★ skeleton として呼び出し ★
    # 実 parse は 既存 download_parse_jrdb_batch2.py / parse_jrdb_extended.py が担当
    print("[INFO] rebuild_tyb_csv は既存 parser に委譲推奨:")
    print("       python tools/parse_jrdb_extended.py")
    print("       python tools/build_jrdb_v2_csv.py")
    return {"status": "delegated_to_existing_parser"}


def main():
    ap = argparse.ArgumentParser(description="JRDB Tyb 直前情報 fetch")
    ap.add_argument("--date", help="YYYYMMDD 指定 (省略時は 直近 max_days_back 日分 fetch)")
    ap.add_argument("--max-days-back", type=int, default=30, help="日数 (default 30)")
    ap.add_argument("--rebuild-csv", action="store_true", help="extracted から jrdb_tyb.csv 再構築")
    args = ap.parse_args()

    if args.rebuild_csv:
        result = rebuild_tyb_csv()
        print(result)
        return 0

    result = fetch_recent_tyb(date_str=args.date, max_days_back=args.max_days_back)
    print(f"\n=== summary ===")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")
    if result["new_files"]:
        print(f"\n  new files: {len(result['new_files'])}")
        for fn in result["new_files"][-10:]:
            print(f"    + {fn}")
        print(f"\n  ★ 次 step ★: python tools/parse_jrdb_extended.py で CSV 再構築")
    return 0


if __name__ == "__main__":
    sys.exit(main())
