"""Session #63 B: 5/9 重賞 + 12R 全出走馬 静止画 download.

netkeiba 出馬表 page から各馬の馬個体写真を取得。
失敗時は skip (NaN 化)、 rate limit 3 秒/画像。

usage:
  python tools/paddock_image_downloader.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

BASE = Path(r"C:/Users/takum/keiba-ai")
OUT_BASE = BASE / "data" / "v18" / "static_5_9"
DOC_OUT = BASE / "data" / "v18" / "session_63_image_dl.md"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
}

TARGETS = [
    {"race_id": "202608030511", "course": "京都", "race_num": 11, "race_name": "京都新聞杯", "grade": "G2"},
    {"race_id": "202608030512", "course": "京都", "race_num": 12, "race_name": "4歳以上2勝クラス", "grade": "-"},
    {"race_id": "202605020511", "course": "東京", "race_num": 11, "race_name": "エプソムC", "grade": "G3"},
    {"race_id": "202605020512", "course": "東京", "race_num": 12, "race_name": "4歳以上2勝クラス", "grade": "-"},
    {"race_id": "202604010311", "course": "新潟", "race_num": 11, "race_name": "駿風 S", "grade": "OP"},
    {"race_id": "202604010312", "course": "新潟", "race_num": 12, "race_name": "4歳以上1勝クラス", "grade": "-"},
]


def _load_cookies():
    """data/cookies.json を Selenium 形式 → requests 形式 dict に変換."""
    cookie_path = BASE / "data" / "cookies.json"
    if not cookie_path.exists():
        return {}
    try:
        data = json.loads(cookie_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {c["name"]: c["value"] for c in data if "name" in c and "value" in c}
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def fetch_shutuba_html(race_id: str, cookies: dict) -> str | None:
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, cookies=cookies, timeout=15)
        resp.encoding = "EUC-JP"
        if resp.status_code == 200:
            return resp.text
    except Exception as e:
        print(f"  [fetch error] {e}")
    return None


def extract_horse_list(html: str) -> list[dict]:
    """馬個体 ID + 馬名 + 馬番 抽出."""
    horses = []
    seen_ids = set()
    # /horse/2022104999/ pattern
    for m in re.finditer(r'/horse/(\d{10})/?"', html):
        hid = m.group(1)
        if hid in seen_ids:
            continue
        seen_ids.add(hid)
        horses.append({"horse_id": hid})
    # 馬番 + 馬名 association rough — 順序維持
    return horses


def fetch_horse_image(horse_id: str, out_path: Path, cookies: dict) -> tuple[bool, str]:
    """馬個体 page から馬体写真 (db.netkeiba.com /horse/<id>/) を試行."""
    page_url = f"https://db.netkeiba.com/horse/{horse_id}/"
    try:
        resp = requests.get(page_url, headers=HEADERS, cookies=cookies, timeout=15)
        resp.encoding = "EUC-JP"
        if resp.status_code != 200:
            return False, f"page status {resp.status_code}"
        # img src で /horse_photo/ or /horse_image/ 系を探索
        m = re.search(r'<img[^>]+src="([^"]+horse[^"]+\.(?:jpg|jpeg|png))"', resp.text)
        if not m:
            m = re.search(r'<img[^>]+src="(https?://[^"]+/horse[^"]+\.(?:jpg|jpeg|png))"', resp.text)
        if not m:
            return False, "no horse image found"
        img_url = m.group(1)
        if img_url.startswith("/"):
            img_url = "https://db.netkeiba.com" + img_url
        img_resp = requests.get(img_url, headers=HEADERS, cookies=cookies, timeout=15)
        if img_resp.status_code != 200:
            return False, f"img status {img_resp.status_code}"
        if len(img_resp.content) < 500:
            return False, f"img too small {len(img_resp.content)}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(img_resp.content)
        return True, f"{len(img_resp.content)} bytes"
    except Exception as e:
        return False, f"exception {type(e).__name__}: {e}"


def main():
    cookies = _load_cookies()
    print(f"cookies loaded: {len(cookies)} keys")

    log_lines = ["# Session #63 B: 静止画 DL 結果", ""]
    log_lines.append(f"対象: {len(TARGETS)} R (重賞 3 + 12R 3)")
    log_lines.append("")

    total_ok = 0
    total_fail = 0
    summary = []

    for t in TARGETS:
        rid = t["race_id"]
        print(f"\n=== {t['course']} R{t['race_num']} {t['race_name']} ({rid}) ===")
        html = fetch_shutuba_html(rid, cookies)
        if not html:
            print("  shutuba fetch failed")
            log_lines.append(f"## {t['course']} R{t['race_num']} {t['race_name']}")
            log_lines.append("- shutuba fetch FAIL")
            log_lines.append("")
            summary.append({"race_id": rid, "ok": 0, "fail": 0, "horses": 0, "status": "shutuba_fail"})
            continue

        horses = extract_horse_list(html)
        print(f"  horses found: {len(horses)}")
        log_lines.append(f"## {t['course']} R{t['race_num']} {t['race_name']} (grade {t['grade']})")
        log_lines.append(f"- horses found: {len(horses)}")

        ok_count = 0
        fail_count = 0
        for h in horses:
            hid = h["horse_id"]
            out = OUT_BASE / rid / f"{hid}.jpg"
            if out.exists() and out.stat().st_size > 500:
                ok_count += 1
                continue
            ok, msg = fetch_horse_image(hid, out, cookies)
            if ok:
                ok_count += 1
                print(f"    {hid} OK ({msg})")
            else:
                fail_count += 1
                print(f"    {hid} FAIL ({msg})")
            time.sleep(3)

        log_lines.append(f"- OK: {ok_count}, FAIL: {fail_count}")
        log_lines.append("")
        total_ok += ok_count
        total_fail += fail_count
        summary.append({"race_id": rid, "ok": ok_count, "fail": fail_count,
                        "horses": len(horses), "status": "ok"})

    log_lines.append("---")
    log_lines.append(f"## 合計: OK={total_ok}, FAIL={total_fail}")
    log_lines.append("")
    log_lines.append("## 詳細")
    for s in summary:
        log_lines.append(f"- {s['race_id']}: {s['status']} OK={s['ok']} FAIL={s['fail']} horses={s['horses']}")

    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    DOC_OUT.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n=== 合計: OK={total_ok}, FAIL={total_fail} ===")
    print(f"doc: {DOC_OUT.relative_to(BASE)}")


if __name__ == "__main__":
    main()
