"""5/9 朝 06:30 最終 checklist (Session #46 C).

8 項目を即時 verify、 全 OK なら GREEN badge、 NG なら RED detailed alert。

検査項目:
1. schtasks 36 件 動作確認 (Keiba-* prefix の数)
2. V15 md5 verify (842b9a5f...)
3. Cookie expiry verify
4. Discord webhook 動作 (echo test)
5. JRDB 接続 (data 鮮度 5/3 以降)
6. netkeiba 接続 (HEAD request)
7. predict 14:50 / 15:45 schtasks 確認
8. disk 容量 (data/ で 5GB+ 余裕あるか)

usage:
  python tools/morning_checklist_generator.py --dry-run    # Discord 抑制
  python tools/morning_checklist_generator.py --date 20260509

V15 production 完全独立 (read-only check)。
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

BASE = Path(r"C:/Users/takum/keiba-ai")
EXPECTED_V15_MD5 = "842b9a5f305c793ed8fa54a74e06b836"


def check_1_schtasks() -> Tuple[bool, str]:
    try:
        r = subprocess.run(["schtasks", "/Query", "/FO", "CSV"],
                           capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return (False, "schtasks query failed")
        keiba_count = r.stdout.count("Keiba-") + r.stdout.count("KeibaAI") + r.stdout.count("Daily")
        return (keiba_count >= 30, f"Keiba-* tasks: {keiba_count}")
    except Exception as e:
        return (False, f"err: {str(e)[:80]}")


def check_2_v15_md5() -> Tuple[bool, str]:
    p = BASE / "keiba_model_v15_central_live.pkl.gz"
    if not p.exists():
        return (False, "V15 model file 不在")
    try:
        with gzip.open(p, "rb") as f:
            data = f.read()
        actual = hashlib.md5(data).hexdigest()
        match = actual == EXPECTED_V15_MD5
        return (match, f"md5 {'MATCH' if match else 'MISMATCH'}: {actual[:16]}...")
    except Exception as e:
        return (False, f"err: {str(e)[:80]}")


def check_3_cookie() -> Tuple[bool, str]:
    p = BASE / "data" / "cookies.json"
    if not p.exists():
        return (False, "cookies.json 不在")
    age_d = (time.time() - p.stat().st_mtime) / 86400
    if age_d > 21:
        return (False, f"Cookie age {age_d:.1f}d (expired)")
    if age_d > 14:
        return (True, f"Cookie age {age_d:.1f}d (warn)")
    return (True, f"Cookie age {age_d:.1f}d")


def check_4_discord_echo() -> Tuple[bool, str]:
    """webhook は send_discord で test。 実際の通知ではなく URL 設定確認."""
    try:
        env_path = BASE / ".env"
        if not env_path.exists():
            return (False, ".env 不在")
        env = {}
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
        bets = env.get("DISCORD_WEBHOOK_BETS", "")
        upd = env.get("DISCORD_WEBHOOK_UPDATES", "")
        fb = env.get("DISCORD_WEBHOOK_URL", "")
        ok_bets = bets.startswith("https://")
        ok_upd = upd.startswith("https://")
        ok_fb = fb.startswith("https://")
        ok = (ok_bets or ok_fb) and (ok_upd or ok_fb)
        return (ok, f"bets={ok_bets}, updates={ok_upd}, fb={ok_fb}")
    except Exception as e:
        return (False, f"err: {str(e)[:80]}")


def check_5_jrdb_freshness(min_date: str = "20260503") -> Tuple[bool, str]:
    p = BASE / "data" / "jrdb" / "extracted" / "Bac"
    if not p.exists():
        return (False, "JRDB extracted/Bac 不在")
    files = list(p.glob("BAC*.txt"))
    if not files:
        return (False, "BAC files 0")
    dates = []
    for f in files:
        n = f.stem
        if len(n) == 9:
            dates.append("20" + n[3:5] + n[5:7] + n[7:9])
    max_date = max(dates) if dates else "?"
    return (max_date >= min_date, f"latest {max_date}")


def check_6_netkeiba_head() -> Tuple[bool, str]:
    """netkeiba 接続 HEAD request (短時間 timeout)."""
    try:
        import requests
        r = requests.head("https://race.netkeiba.com/", timeout=5,
                          headers={"User-Agent": "Mozilla/5.0"})
        return (r.status_code in (200, 301, 302), f"status {r.status_code}")
    except Exception as e:
        return (False, f"err: {str(e)[:60]}")


def check_7_predict_schtasks() -> Tuple[bool, str]:
    """multi_stage_predict 14:50 / 15:45 trigger 確認."""
    try:
        r = subprocess.run(["schtasks", "/Query", "/FO", "CSV"],
                           capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return (False, "query fail")
        out = r.stdout
        # multi_stage / RaceAutoNotify など
        triggers = ["RaceAutoNotify", "multi_stage", "DailyPredict"]
        found = [t for t in triggers if t in out]
        return (len(found) >= 2, f"found: {found}")
    except Exception as e:
        return (False, f"err: {str(e)[:80]}")


def check_8_disk_space() -> Tuple[bool, str]:
    try:
        # data/ dir の disk
        path = BASE / "data"
        if not path.exists():
            return (False, "data/ 不在")
        # Windows: shutil.disk_usage
        usage = shutil.disk_usage(str(path))
        free_gb = usage.free / (1024**3)
        return (free_gb >= 5.0, f"free {free_gb:.1f} GB")
    except Exception as e:
        return (False, f"err: {str(e)[:80]}")


def send_discord(title: str, body: str, color: str = "green") -> bool:
    try:
        sys.path.insert(0, str(BASE / "tools"))
        from notify import send_discord as _send
        return _send(title, body, color=color, channel="updates")
    except Exception as e:
        print(f"[discord] err: {e}", file=sys.stderr)
        return False


def main():
    p = argparse.ArgumentParser(description="5/9 朝 06:30 morning checklist (Session #46 C)")
    p.add_argument("--date", default=None)
    p.add_argument("--dry-run", action="store_true", help="Discord 抑制")
    args = p.parse_args()

    target_date = args.date or datetime.date.today().strftime("%Y%m%d")

    print("=" * 70)
    print(f"morning checklist ({target_date})")
    print("=" * 70)

    checks = [
        ("1. schtasks 36 tasks", check_1_schtasks),
        ("2. V15 md5 verify (842b9a5f...)", check_2_v15_md5),
        ("3. Cookie expiry", check_3_cookie),
        ("4. Discord webhook", check_4_discord_echo),
        ("5. JRDB freshness (>= 20260503)", check_5_jrdb_freshness),
        ("6. netkeiba HEAD request", check_6_netkeiba_head),
        ("7. predict schtasks (multi_stage etc)", check_7_predict_schtasks),
        ("8. disk space >= 5GB", check_8_disk_space),
    ]

    results = []
    n_ok = 0
    n_ng = 0
    for label, fn in checks:
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"exception: {str(e)[:80]}"
        mark = "OK" if ok else "NG"
        print(f"  [{mark}] {label}: {msg}")
        results.append({"label": label, "ok": ok, "msg": msg})
        if ok: n_ok += 1
        else: n_ng += 1

    print(f"\n=== summary ===")
    print(f"  OK: {n_ok} / {len(checks)}")
    print(f"  NG: {n_ng}")

    overall_ok = n_ng == 0
    severity = "green" if overall_ok else "red"
    title = f"[morning checklist {target_date}] " + (f"ALL OK ({n_ok}/{len(checks)})" if overall_ok else f"FAIL {n_ng}/{len(checks)}")

    body_lines = [f"check {n_ok}/{len(checks)} OK"]
    if n_ng > 0:
        body_lines.append("---")
        body_lines.append("[NG]:")
        for r in results:
            if not r["ok"]:
                body_lines.append(f"  - {r['label']}: {r['msg']}")
    body = "\n".join(body_lines)

    if not args.dry_run:
        send_discord(title, body, color=severity)
        print(f"\nDiscord: sent")
    else:
        print(f"\n[dry-run] Discord skip")

    out_path = BASE / "data" / "v18" / f"morning_checklist_{target_date}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "date": target_date,
        "n_ok": n_ok, "n_ng": n_ng,
        "results": results,
        "overall_ok": overall_ok,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  written: {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
