"""5/8 21:00 自動 pre-race check (Session #46 A).

5/9 (土) の出馬表確定後、 race_name / 発走時刻 / 頭数 / 11R 重賞警告 +
Cookie expiry 確認を 自動化。 異常あれば Discord WARN、 正常なら OK 通知。

注意: 既存 tools/pre_race_check.py は 別目的 (実戦前 8 項目 check)、
本 file は別名で隔離 (Session #46 A 専用)。

usage:
  # 今すぐ実行 (5/9 のレース情報事前確認)
  python tools/pre_race_check_5_9.py --date 20260509

  # schtasks 想定
  schtasks /Create /TN "Keiba-PreRaceCheck_2100" \
      /TR "python C:/Users/takum/keiba-ai/tools/pre_race_check_5_9.py" \
      /SC WEEKLY /D SAT /ST 21:00 /F

V15 production 完全独立 (read-only)。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")


def fetch_race_calendar(date: str) -> dict:
    """date (YYYYMMDD) の開催情報を取得 (read-only).

    候補 1: data/jra_calendar.csv
    候補 2: data/daily_predictions/<date>.csv (既生成なら)
    """
    out = {"available": False, "races": [], "courses": [], "source": ""}

    cal_path = BASE / "data" / "jra_calendar.csv"
    if cal_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(cal_path, dtype=str)
            sub = df[df.get('date', '').astype(str) == date] if 'date' in df.columns else df.head(0)
            if len(sub) > 0:
                out["available"] = True
                out["source"] = "data/jra_calendar.csv"
                out["courses"] = sorted(sub.get('course', pd.Series()).dropna().unique().tolist())
        except Exception as e:
            out["error_calendar"] = str(e)[:120]

    pred_path = BASE / "data" / "daily_predictions" / f"{date}.csv"
    if pred_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(pred_path, dtype=str)
            races = []
            for _, r in df.iterrows():
                rid = str(r.get('race_id', ''))
                if not rid: continue
                races.append({
                    "race_id": rid,
                    "course": str(r.get('course', '')),
                    "race_num": str(r.get('race_num', '')),
                    "race_name": str(r.get('race_name', '')),
                    "num_horses": str(r.get('num_horses', '')),
                })
            out["races"] = races
            out["available"] = True
            out["source"] = "data/daily_predictions/"
        except Exception as e:
            out["error_pred"] = str(e)[:120]

    return out


def detect_grade_races(races: list) -> list:
    """11R / 12R で 重賞 (G1/G2/G3/L) を検出."""
    flagged = []
    for r in races:
        race_num = r.get("race_num", "")
        race_name = r.get("race_name", "")
        try:
            rn = int(race_num)
            if rn not in (11, 12):
                continue
        except Exception:
            continue
        for grade in ["G1", "G2", "G3", "(GⅠ)", "(GⅡ)", "(GⅢ)",
                      "GⅠ", "GⅡ", "GⅢ", "(L)", "リステッド"]:
            if grade in race_name:
                flagged.append({
                    "race_id": r.get("race_id"),
                    "course": r.get("course"),
                    "race_num": race_num,
                    "race_name": race_name,
                    "grade": grade,
                    "warning": "★ 11R/12R 重賞 検出: 案B改は 1勝のみ → 投票絶対 NG",
                })
                break
    return flagged


def detect_1sho_races(races: list) -> list:
    """12R で 1勝クラス を検出."""
    found = []
    for r in races:
        race_num = r.get("race_num", "")
        race_name = r.get("race_name", "")
        if "1勝" in race_name:
            found.append({
                "race_id": r.get("race_id"),
                "course": r.get("course"),
                "race_num": race_num,
                "race_name": race_name,
                "num_horses": r.get("num_horses"),
            })
    return found


def check_cookie_freshness() -> dict:
    cookies_path = BASE / "data" / "cookies.json"
    if not cookies_path.exists():
        return {"status": "missing", "warning": "data/cookies.json 不在"}
    age_sec = time.time() - cookies_path.stat().st_mtime
    age_d = age_sec / 86400
    if age_d > 21:
        return {"status": "expired", "age_days": round(age_d, 1),
                "warning": f"Cookie age {age_d:.1f} days、 ★ refresh 必須 ★"}
    if age_d > 14:
        return {"status": "warn", "age_days": round(age_d, 1),
                "warning": f"Cookie age {age_d:.1f} days、 refresh 推奨"}
    return {"status": "ok", "age_days": round(age_d, 1)}


def send_discord(title: str, body: str, color: str = "green") -> bool:
    try:
        sys.path.insert(0, str(BASE / "tools"))
        from notify import send_discord as _send
        return _send(title, body, color=color, channel="updates")
    except Exception as e:
        print(f"[discord] send error: {e}", file=sys.stderr)
        return False


def main():
    p = argparse.ArgumentParser(description="5/8 21:00 pre-race check (Session #46 A)")
    p.add_argument("--date", default=None)
    p.add_argument("--no-discord", action="store_true")
    p.add_argument("--retries", type=int, default=3)
    args = p.parse_args()

    target_date = args.date or (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y%m%d")

    print("=" * 70)
    print(f"pre-race check ({target_date})")
    print("=" * 70)

    print("\n[Step 1] race_name 取得 ...")
    cal = None
    for attempt in range(args.retries):
        cal = fetch_race_calendar(target_date)
        if cal.get("available"):
            break
        print(f"  retry {attempt+1}/{args.retries} ...")
        time.sleep(2)

    if not cal.get("available"):
        print(f"  [WARN] race info 取得失敗、 5/8 21:00 後に手動確認推奨")

    races = cal.get("races", [])
    courses = cal.get("courses", [])
    print(f"  races: {len(races)}, courses: {courses}, source: {cal.get('source', '?')}")

    print("\n[Step 2] 重賞検出 (11R/12R) ...")
    flagged = detect_grade_races(races)
    if flagged:
        print(f"  ★ 重賞 検出: {len(flagged)} R ★")
        for f in flagged:
            print(f"    - {f['course']} {f['race_num']}R {f['race_name']} ({f['grade']})")
    else:
        print(f"  重賞 0")

    print("\n[Step 3] 12R 1勝クラス 検出 ...")
    candidates = detect_1sho_races(races)
    cand_12r = [c for c in candidates if c["race_num"] == "12"]
    if cand_12r:
        print(f"  12R 1勝クラス候補: {len(cand_12r)} R")
        for c in cand_12r[:5]:
            print(f"    - {c['course']} {c['race_num']}R {c['race_name']} ({c['num_horses']}頭)")
    else:
        print(f"  1勝 R 検出 0")

    print("\n[Step 4] Cookie freshness ...")
    cookie = check_cookie_freshness()
    print(f"  status: {cookie.get('status')}")
    if cookie.get("warning"):
        print(f"  {cookie['warning']}")

    n_flagged = len(flagged)
    overall_ok = (cal.get("available") and cookie["status"] in ("ok", "warn"))

    severity = "green" if (overall_ok and n_flagged == 0) else ("yellow" if overall_ok else "red")
    title = f"[pre-race check {target_date}] " + ("OK" if severity == "green" else
                                                  ("WARN" if severity == "yellow" else "FAIL"))

    body_lines = [
        f"target: {target_date}",
        f"race info: {'OK' if cal.get('available') else 'FAIL'} (source: {cal.get('source', 'N/A')})",
        f"  total races: {len(races)} ({len(courses)} courses)",
        f"重賞検出 (11R/12R): {n_flagged}",
    ]
    if flagged:
        body_lines.append("  ★ 重賞検出 (案B改 投票NG) ★:")
        for f in flagged[:3]:
            body_lines.append(f"    - {f['course']} {f['race_num']}R {f['race_name']}")
    body_lines.append(f"12R 1勝クラス候補: {len(cand_12r)}")
    body_lines.append(f"Cookie status: {cookie.get('status')} ({cookie.get('age_days', '?')}d)")
    body = "\n".join(body_lines)

    print(f"\n=== summary ===")
    print(body)

    if not args.no_discord:
        ok = send_discord(title, body, color=severity)
        print(f"\nDiscord: {'sent' if ok else 'FAIL'}")

    out_path = BASE / "data" / "v18" / f"pre_race_check_{target_date}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "target_date": target_date,
        "calendar": cal,
        "flagged_grade": flagged,
        "candidates_12r_1sho": cand_12r,
        "cookie": cookie,
        "severity": severity,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
