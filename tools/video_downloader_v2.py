"""動画 downloader v2 (Session #62 C、 dev/training-poc).

Session #60 B で yt-dlp generic + HTTP 400 で全失敗。
本 v2 は Playwright (real Chromium) で page を実行 → HLS (m3u8) URL 抽出
→ Playwright bundled ffmpeg で merge する framework。

server 復旧後 (race.netkeiba.com 200 戻ったら) 1 行で動く設計。

usage:
  # 単一 race
  python tools/video_downloader_v2.py --race-id 202608030511

  # 全重賞 (5/9 majors)
  python tools/video_downloader_v2.py --majors

  # netkeiba reachability check のみ
  python tools/video_downloader_v2.py --probe

V15 production 完全独立、 dev/training-poc 専用。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
VIDEO_DIR = BASE / "data" / "v18" / "videos_5_9"
COOKIES_NETSCAPE = VIDEO_DIR / "cookies.txt"
ERR_LOG = BASE / "data" / "v18" / "video_dl_errors_5_9.log"
FFMPEG_BIN = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright" / "ffmpeg-1011" / "ffmpeg-win64.exe"

MAJORS_5_9 = [
    {"name": "京都新聞杯_G2", "race_id": "202608030511"},
    {"name": "エプソムC_G3", "race_id": "202605020511"},
    {"name": "駿風S_OP", "race_id": "202604010311"},
]


def log_err(msg: str) -> None:
    ERR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ERR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def probe_netkeiba() -> dict:
    """netkeiba 各 subdomain の reachability を Playwright でチェック."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"status": "no_playwright"}

    targets = [
        "https://www.netkeiba.com/",
        "https://race.netkeiba.com/",
        "https://db.netkeiba.com/",
        "https://race.sp.netkeiba.com/",
    ]
    results = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()
        for u in targets:
            try:
                r = page.goto(u, wait_until="domcontentloaded", timeout=15000)
                results[u] = r.status if r else None
            except Exception as e:
                results[u] = f"EXC:{type(e).__name__}"
        browser.close()
    return results


def fetch_page_with_video(race_id: str, retries: int = 3, sleep_sec: int = 5) -> dict:
    """Playwright で movie.html を開いて page HTML + console URLs を取得.

    cookies は data/v18/videos_5_9/cookies.txt (Netscape) があれば import.
    Returns: {status, html, video_urls (m3u8 candidates), error}
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"status": "no_playwright"}

    movie_url = f"https://race.netkeiba.com/race/movie.html?race_id={race_id}"

    last_err = None
    for attempt in range(retries):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                ctx = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/130.0.0.0 Safari/537.36",
                    locale="ja-JP",
                )
                # import Netscape cookies (best-effort)
                cookies_loaded = 0
                if COOKIES_NETSCAPE.exists():
                    cookies = []
                    for line in COOKIES_NETSCAPE.read_text(encoding="utf-8").splitlines():
                        if line.startswith("#") or not line.strip():
                            continue
                        parts = line.split("\t")
                        if len(parts) < 7:
                            continue
                        cookies.append({
                            "domain": parts[0],
                            "path": parts[2],
                            "secure": parts[3] == "TRUE",
                            "expires": int(parts[4]) if parts[4].isdigit() else -1,
                            "name": parts[5],
                            "value": parts[6],
                        })
                    if cookies:
                        try:
                            ctx.add_cookies(cookies)
                            cookies_loaded = len(cookies)
                        except Exception:
                            pass

                # capture all .m3u8 / .mp4 URLs from network
                video_urls = []
                page = ctx.new_page()

                def on_request(req):
                    u = req.url
                    if any(u.lower().endswith(ext) or ext in u.lower()
                           for ext in (".m3u8", ".mp4", ".mpd")):
                        video_urls.append(u)

                page.on("request", on_request)

                resp = page.goto(movie_url, wait_until="domcontentloaded", timeout=20000)
                status = resp.status if resp else None
                if status == 200:
                    # wait a bit for video player to load
                    page.wait_for_timeout(3000)
                    html = page.content()
                    # also regex-scan HTML for m3u8
                    m3u8_in_html = re.findall(r'https?://[^\s"\']+\.m3u8[^\s"\']*', html)
                    video_urls.extend(m3u8_in_html)
                    browser.close()
                    return {
                        "status": "ok",
                        "http_status": status,
                        "cookies_loaded": cookies_loaded,
                        "html_len": len(html),
                        "video_urls": list(dict.fromkeys(video_urls)),  # dedup, preserve order
                    }
                else:
                    last_err = f"http_{status}"
                    browser.close()
                    if attempt < retries - 1:
                        time.sleep(sleep_sec)
                    continue

        except Exception as e:
            last_err = f"exc:{type(e).__name__}:{str(e)[:80]}"
            log_err(f"race_id={race_id} attempt={attempt+1} {last_err}")
            if attempt < retries - 1:
                time.sleep(sleep_sec)

    return {"status": "fail", "error": last_err, "video_urls": []}


def download_via_ffmpeg(url: str, out_path: Path, headers: dict | None = None) -> dict:
    """Playwright bundled ffmpeg で m3u8/mp4 を merge.

    Returns: {status, size_kb, error}
    """
    if not FFMPEG_BIN.exists():
        return {"status": "no_ffmpeg", "ffmpeg_bin": str(FFMPEG_BIN)}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(FFMPEG_BIN), "-y", "-i", url]
    if headers:
        hdr_str = "\r\n".join(f"{k}: {v}" for k, v in headers.items()) + "\r\n"
        cmd[1:1] = ["-headers", hdr_str]
    cmd += ["-c", "copy", "-bsf:a", "aac_adtstoasc", str(out_path)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode == 0 and out_path.exists():
            return {
                "status": "ok",
                "size_kb": round(out_path.stat().st_size / 1024, 1),
            }
        return {
            "status": "ffmpeg_fail",
            "returncode": proc.returncode,
            "stderr": (proc.stderr or "")[-300:],
        }
    except Exception as e:
        return {"status": "exc", "error": str(e)[:200]}


def attempt_race(race_id: str, name: str = "") -> dict:
    """1 race の page → video URL 抽出 → DL 一連."""
    print(f"\n  [{name}] race_id={race_id}")

    page_r = fetch_page_with_video(race_id)
    print(f"    page: {page_r.get('status')} (http={page_r.get('http_status')}, "
          f"video_urls={len(page_r.get('video_urls', []))})")

    if page_r.get("status") != "ok" or not page_r.get("video_urls"):
        return {"race_id": race_id, "name": name, "status": "no_video_url",
                "page_result": page_r}

    out_path = VIDEO_DIR / name / "main.mp4"
    # try first m3u8/mp4 url
    url = page_r["video_urls"][0]
    headers = {"Referer": f"https://race.netkeiba.com/race/movie.html?race_id={race_id}",
               "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    dl_r = download_via_ffmpeg(url, out_path, headers)
    print(f"    ffmpeg: {dl_r.get('status')} size_kb={dl_r.get('size_kb')}")

    return {
        "race_id": race_id, "name": name,
        "status": dl_r.get("status"),
        "out": str(out_path) if dl_r.get("status") == "ok" else None,
        "page_result": page_r,
        "dl_result": dl_r,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--race-id")
    p.add_argument("--majors", action="store_true")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--out", default="data/v18/session_62_dl_results.json")
    args = p.parse_args()

    print("=" * 70)
    print("video_downloader v2 (Session #62 C、 Playwright + ffmpeg)")
    print(f"start: {datetime.now()}")
    print(f"ffmpeg: {FFMPEG_BIN} (exists={FFMPEG_BIN.exists()})")
    print("=" * 70)

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    if args.probe:
        result = {"probe": probe_netkeiba()}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.majors:
        results = []
        for major in MAJORS_5_9:
            r = attempt_race(major["race_id"], major["name"])
            results.append(r)
            time.sleep(5)  # rate limit
        result = {"majors": results,
                  "summary": {
                      "total": len(results),
                      "ok": sum(1 for r in results if r.get("status") == "ok"),
                  }}
        print(f"\n=== summary: {result['summary']['ok']}/{result['summary']['total']} ok ===")
    elif args.race_id:
        result = {"single": attempt_race(args.race_id)}
    else:
        p.print_help()
        return 1

    out_path = BASE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
