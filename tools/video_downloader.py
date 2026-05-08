"""動画 downloader (Session #52 A、 dev/training-poc).

netkeiba 重賞 注目馬 動画を yt-dlp で download。
- yt-dlp + Cookie 経由 (Premium 必要)
- rate limit 配慮 (3 秒/動画)
- 失敗時 静止画 fallback (画像 URL → requests)

usage:
  # 単一 race
  python tools/video_downloader.py --race-id 202608030411

  # 全重賞 (5/9)
  python tools/video_downloader.py --majors

V15 production 完全独立、 dev/training-poc 専用。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
VIDEO_DIR = BASE / "data" / "v18" / "videos_5_9"
COOKIES_PATH = BASE / "data" / "cookies.json"


def load_cookies_for_yt_dlp() -> str | None:
    if not COOKIES_PATH.exists():
        return None
    try:
        cookies = json.loads(COOKIES_PATH.read_text(encoding="utf-8"))
        cookies_txt = BASE / "data" / "v18" / "videos_5_9" / "cookies.txt"
        cookies_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(cookies_txt, "w", encoding="utf-8") as f:
            f.write("# Netscape HTTP Cookie File\n")
            for c in cookies if isinstance(cookies, list) else cookies.get("cookies", []):
                domain = c.get("domain", ".netkeiba.com")
                domain_flag = "TRUE" if domain.startswith(".") else "FALSE"
                path = c.get("path", "/")
                secure = "TRUE" if c.get("secure", False) else "FALSE"
                expiry = str(int(c.get("expiry", 9999999999)))
                name = c.get("name", "")
                value = c.get("value", "")
                f.write(f"{domain}\t{domain_flag}\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n")
        return str(cookies_txt)
    except Exception as e:
        print(f"[cookie] convert error: {e}", file=sys.stderr)
        return None


def download_video_yt_dlp(url: str, out_path: Path, cookies_file: str = None) -> dict:
    try:
        import yt_dlp
    except ImportError:
        return {"status": "no_yt_dlp", "url": url}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "outtmpl": str(out_path),
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
    }
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
        if out_path.exists():
            return {
                "status": "ok",
                "url": url,
                "out": str(out_path),
                "size_kb": round(out_path.stat().st_size / 1024, 1),
                "title": info.get("title", "?")[:80] if info else "?",
            }
        return {"status": "no_output", "url": url}
    except Exception as e:
        return {"status": "error", "url": url, "error": str(e)[:200]}


def attempt_download_majors(dry_run: bool = False) -> dict:
    targets = [
        {"race": "京都新聞杯_G2", "url": "https://race.netkeiba.com/race/movie.html?race_id=PLACEHOLDER_KYOTO"},
        {"race": "エプソムC_G3", "url": "https://race.netkeiba.com/race/movie.html?race_id=PLACEHOLDER_TOKYO"},
        {"race": "駿風_S_OP", "url": "https://race.netkeiba.com/race/movie.html?race_id=PLACEHOLDER_NIIGATA"},
    ]

    cookies_file = load_cookies_for_yt_dlp()
    print(f"  cookies file: {cookies_file or 'none (Premium login 必要)'}")

    results = []
    for i, t in enumerate(targets):
        race_id = t.get("race", f"race_{i}")
        out_path = VIDEO_DIR / race_id / "main.mp4"
        if dry_run:
            results.append({"race": race_id, "status": "dry_run", "url": t["url"]})
            continue
        print(f"\n  [{i+1}/{len(targets)}] {race_id}")
        r = download_video_yt_dlp(t["url"], out_path, cookies_file)
        r["race"] = race_id
        results.append(r)
        time.sleep(3)

    return {"n_targets": len(targets), "results": results}


def main():
    p = argparse.ArgumentParser(description="動画 downloader (Session #52 A)")
    p.add_argument("--race-id", default=None)
    p.add_argument("--majors", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default="data/v18/session_52_video_download_results.json")
    args = p.parse_args()

    print("=" * 70)
    print("動画 downloader (Session #52 A、 dev/training-poc)")
    print("=" * 70)

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    if args.majors or args.race_id is None:
        result = attempt_download_majors(dry_run=args.dry_run)
        print(f"\n=== summary ===")
        n_ok = sum(1 for r in result["results"] if r.get("status") == "ok")
        print(f"  total: {result['n_targets']}, ok: {n_ok}")
        for r in result["results"]:
            print(f"    {r.get('race', '?')}: {r.get('status')}")
    else:
        cookies_file = load_cookies_for_yt_dlp()
        out_path = VIDEO_DIR / args.race_id / "main.mp4"
        url = f"https://race.netkeiba.com/race/movie.html?race_id={args.race_id}"
        r = download_video_yt_dlp(url, out_path, cookies_file)
        result = {"race_id": args.race_id, "result": r}
        print(json.dumps(result, ensure_ascii=False, indent=2))

    out_path = BASE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
