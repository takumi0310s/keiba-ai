# -*- coding: utf-8 -*-
"""
SCI (類似馬場) 分析ツール データ取得スクリプト  — research/ruiji 専用
- 認証: POST / に aikotoba → sci_gate Cookie
- 索引: /<date>/meta.json
- 本体: /an/<date>/<quote(f without .html)>.json
- 制約: 全 HTTP リクエスト間 2 秒スリープ厳守 / raw/<date>/ へ保存 / 再開可能

既存 V15 運用系には一切触れない。data/ には書き込まない。
"""
import sys, os, time, json, urllib.parse, io
import requests

BASE = "https://sci-ruiji-me-tt7e8scz.pages.dev"
AIKOTOBA = "chronogenesis-8519"
INTERVAL = 2.1  # 秒 (>=2s 厳守、僅かに上乗せ)
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

# rj_pace.json で確認済みの公開日 20 件 (2026-06-06〜08-09 の週末)
DATES = [
    "20260606", "20260607", "20260613", "20260614",
    "20260620", "20260621", "20260627", "20260628",
    "20260704", "20260705", "20260711", "20260712",
    "20260718", "20260719", "20260725", "20260726",
    "20260801", "20260802", "20260808", "20260809",
]

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ruiji-research/1.0"

def log(msg):
    print(msg, flush=True)

_last = [0.0]
def wait():
    """前回リクエストから INTERVAL 秒空ける。"""
    dt = time.time() - _last[0]
    if dt < INTERVAL:
        time.sleep(INTERVAL - dt)
    _last[0] = time.time()

def is_json(resp):
    return "application/json" in resp.headers.get("Content-Type", "")

def auth(sess):
    wait()
    r = sess.post(BASE + "/", data={"aikotoba": AIKOTOBA},
                  allow_redirects=False, timeout=30)
    if "sci_gate" not in sess.cookies.get_dict():
        raise SystemExit(f"認証失敗: status={r.status_code} cookies={sess.cookies.get_dict()}")
    log(f"[auth] OK  sci_gate 取得 (status {r.status_code})")

def fetch_json(sess, path):
    wait()
    r = sess.get(BASE + path, timeout=30)
    return r

def main():
    os.makedirs(RAW, exist_ok=True)
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    auth(sess)

    grand_races = 0
    grand_saved = 0
    grand_skip_existing = 0
    grand_empty_f = 0
    grand_fail = 0

    for di, date in enumerate(DATES, 1):
        ddir = os.path.join(RAW, date)
        os.makedirs(ddir, exist_ok=True)
        # --- meta.json ---
        meta_path = os.path.join(ddir, "meta.json")
        if os.path.exists(meta_path) and os.path.getsize(meta_path) > 0:
            meta = json.load(io.open(meta_path, encoding="utf-8"))
        else:
            r = fetch_json(sess, f"/{date}/meta.json")
            if not is_json(r):
                log(f"[{di}/{len(DATES)}] {date}  meta.json 取得不可 (content-type={r.headers.get('Content-Type')}) — スキップ")
                continue
            meta = r.json()
            io.open(meta_path, "w", encoding="utf-8").write(
                json.dumps(meta, ensure_ascii=False))
        races = meta.get("races", [])
        # --- 各レース JSON ---
        saved = skip_exist = empty_f = fail = 0
        for rc in races:
            f = rc.get("f", "")
            if not f:
                empty_f += 1
                continue
            stem = f[:-5] if f.endswith(".html") else f  # remove .html
            out = os.path.join(ddir, stem + ".json")
            if os.path.exists(out) and os.path.getsize(out) > 0:
                skip_exist += 1
                continue
            enc = urllib.parse.quote(stem, safe="")
            r = fetch_json(sess, f"/an/{date}/{enc}.json")
            if r.status_code == 200 and is_json(r):
                io.open(out, "w", encoding="utf-8").write(
                    r.text)
                saved += 1
            else:
                log(f"    ! {date} {rc.get('v')}{rc.get('r')}R fail "
                    f"status={r.status_code} ct={r.headers.get('Content-Type')}")
                fail += 1
        grand_races += len(races); grand_saved += saved
        grand_skip_existing += skip_exist; grand_empty_f += empty_f; grand_fail += fail
        log(f"[{di}/{len(DATES)}] {date}  races={len(races)}  "
            f"saved={saved} 既存={skip_exist} f空={empty_f} fail={fail}")

    log("=" * 56)
    log(f"完了: 日付{len(DATES)}  総レース{grand_races}  "
        f"新規保存{grand_saved}  既存{grand_skip_existing}  "
        f"f空(障害等){grand_empty_f}  失敗{grand_fail}")

if __name__ == "__main__":
    main()
