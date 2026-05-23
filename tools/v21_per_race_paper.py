#!/usr/bin/env python3
"""
tools/v21_per_race_paper.py
V21 paper per-race prediction scheduler.

Fires 17 min before each race, fetches TYB (shadow observe), runs V21 (paper),
sends Discord.  V15 production completely unchanged.  V21 is paper/comparison only.

Usage:
    python tools/v21_per_race_paper.py [--date YYYYMMDD]
    python tools/v21_per_race_paper.py --date 20260524

Design:
- Reads today's race list from netkeiba (same as race_auto_notify)
- Falls back to daily_predictions/{date}.csv if netkeiba fetch fails
- Schedules threading.Timer for each race at (start - 17 min)
- Timer fires: fetch TYB shadow → predict V21 → filter → Discord → log
- All exceptions inside fire_race_notify are swallowed (scheduler keeps running)
- V15 model is NEVER loaded by this script
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths & bootstrap
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env", override=False)
except ImportError:
    pass

# Windows console encoding guard: avoid cp932 UnicodeEncodeError in print()
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

V21_MODEL_PATH = REPO / "models" / "v21_candidate.pkl.gz"
PAPER_LOG_DIR = REPO / "data" / "v21_paper_log"
NOTIFY_BEFORE_MIN = 17  # fire 17 min before race start (within 15-20 min window)

DISCORD_WEBHOOK: str = (
    os.environ.get("DISCORD_WEBHOOK_BETS")
    or os.environ.get("DISCORD_WEBHOOK_V21_PAPER")
    or os.environ.get("DISCORD_WEBHOOK_UPDATES")
    or ""
)

COURSE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_v21_model() -> Optional[dict]:
    """Load V21 candidate model. Returns None gracefully if file missing."""
    if not V21_MODEL_PATH.exists():
        print(f"[V21] model not found at {V21_MODEL_PATH}")
        return None
    try:
        with gzip.open(str(V21_MODEL_PATH), "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "model" in data:
            print(f"[V21] model loaded OK ({V21_MODEL_PATH.name})")
            return data
        # Some pickles store the model directly
        print(f"[V21] model loaded (raw, wrapping) ({V21_MODEL_PATH.name})")
        return {"model": data}
    except Exception as e:
        print(f"[V21] model load failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Race schedule
# ---------------------------------------------------------------------------
def _fetch_races_netkeiba(date_str: str) -> list[dict]:
    """Fetch race list from netkeiba. Returns [] on failure."""
    try:
        import requests
        from bs4 import BeautifulSoup

        session = requests.Session()
        session.headers.update(HEADERS)
        url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
        resp = session.get(url, timeout=15)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")

        races: list[dict] = []
        seen: set[str] = set()
        for a in soup.find_all("a", href=True):
            m = re.search(r"race_id=(\d{12})", a["href"])
            if not m:
                continue
            race_id = m.group(1)
            if race_id in seen:
                continue
            seen.add(race_id)
            parent = a.find_parent("li") or a.find_parent("div")
            text = parent.get_text(strip=True) if parent else a.get_text(strip=True)
            tm = re.search(r"(\d{1,2}:\d{2})", text)
            start_time = tm.group(1) if tm else ""
            rn = re.search(r"(\d{1,2})R", text)
            race_num = int(rn.group(1)) if rn else 0
            course = COURSE_MAP.get(race_id[4:6], "?")
            races.append({
                "race_id": race_id,
                "course": course,
                "race_num": race_num,
                "start_time": start_time,
            })
        return sorted(races, key=lambda r: r.get("start_time", ""))
    except Exception as e:
        print(f"[schedule] netkeiba fetch failed: {e}")
        return []


def _fetch_races_csv(date_str: str) -> list[dict]:
    """Fallback: read daily_predictions/{date}.csv for race list."""
    csv_path = REPO / "data" / "daily_predictions" / f"{date_str}.csv"
    if not csv_path.exists():
        return []
    try:
        import csv as _csv

        races: list[dict] = []
        seen: set[str] = set()
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                race_id = str(row.get("race_id", "")).strip()
                if not race_id or race_id in seen:
                    continue
                seen.add(race_id)
                # CSV has no start_time col; derive from race_id later
                races.append({
                    "race_id": race_id,
                    "course": str(row.get("course", COURSE_MAP.get(race_id[4:6], "?"))),
                    "race_num": int(row.get("race_num", 0) or 0),
                    "start_time": "",  # unknown; will be skipped unless filled
                })
        return races
    except Exception as e:
        print(f"[schedule] CSV fallback failed: {e}")
        return []


def load_race_schedule(date_str: str) -> list[dict]:
    """Return list of race dicts with start_dt populated. Skips races with unknown time."""
    races = _fetch_races_netkeiba(date_str)
    if not races:
        print("[schedule] Falling back to daily_predictions CSV")
        races = _fetch_races_csv(date_str)

    result: list[dict] = []
    for r in races:
        start_time = r.get("start_time", "")
        if not start_time or ":" not in start_time:
            print(f"  {r['race_id']}: no start_time - skip scheduling")
            continue
        try:
            base = datetime.strptime(date_str, "%Y%m%d")
            h, m = start_time.split(":")
            start_dt = base.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
            r["start_dt"] = start_dt
            result.append(r)
        except Exception as e:
            print(f"  {r['race_id']}: parse error ({e}) - skip")
    return result


# ---------------------------------------------------------------------------
# TYB fetch
# ---------------------------------------------------------------------------
def fetch_tyb_for_race(race_id: str, start_time_str: str) -> Optional[dict]:
    """
    Fetch TYB via tyb_shadow_fetcher.fetch_tyb_shadow.
    Returns None if disabled, not available, or any error.
    """
    try:
        from tyb_shadow_fetcher import fetch_tyb_shadow, TYB_SHADOW_ENABLED
        if not TYB_SHADOW_ENABLED:
            return None
        result = fetch_tyb_shadow(race_id, start_time_str, enabled=True)
        return result
    except Exception as e:
        print(f"    [TYB] fetch error: {e}")
        return None


# ---------------------------------------------------------------------------
# Strategy filter (mirrors v21_paper_predict._passes_strategy7c)
# ---------------------------------------------------------------------------
def _passes_strategy_filter(race_name: str, course: str, cond_key: str, distance: int) -> tuple[bool, str]:
    """Return (passes, reason). Mirrors 戦略⑦案C + C4."""
    race_name_str = str(race_name)
    is_graded = any(g in race_name_str for g in ["G1", "G2", "G3", "GⅠ", "GⅡ", "GⅢ"])
    is_listed = any(s in race_name_str for s in ["L)", "(L)", "OP)", "(OP)"])
    is_open_tokubetsu = any(s in race_name_str for s in ["杯", "賞", "ステークス", "カップ", "ハンデ"])

    if "特別" in race_name_str and not (is_graded or is_listed or is_open_tokubetsu):
        return False, "strategy_7_06_tokubetsu"
    if course == "京都" and not (is_graded or is_listed):
        return False, "strategy_7_kyoto"
    if cond_key == "E":
        return False, "strategy_7_cond_E"
    if cond_key == "B":
        return False, "strategy_7_cond_B"
    if cond_key == "X" and not (is_graded or is_listed):
        return False, "strategy_7_cond_X"
    # C4: Cond-A 1600-1800m
    if cond_key == "A" and 1600 <= distance <= 1800:
        return False, "strategy_c4_condA_1600_1800"
    return True, ""


# ---------------------------------------------------------------------------
# V21 prediction
# ---------------------------------------------------------------------------
def predict_v21(race_id: str, v21_model: dict, tyb_data: Optional[dict]) -> Optional[dict]:
    """
    Run V21 prediction for one race using predict_core feature builder.
    V21 model dict is passed in — V15 load_models() is NOT called.
    If TYB is unavailable, TYB columns default to 0 (model handles gracefully).
    Returns result dict or None on failure.
    """
    try:
        from predict_core import (
            parse_shutuba,
            fetch_realtime_odds_full,
            save_odds_base,
            classify_race_condition,
            generate_trio_bets,
            generate_umaren_bets,
            build_features,
            predict_race,
            fetch_jra_and_weather,
            get_horse_stats,
            apply_horse_stats,
            set_horse_defaults,
        )
    except Exception as e:
        print(f"    [V21] predict_core import failed: {e}")
        return None

    try:
        race_name, horses, horse_ids, rinfo = parse_shutuba(race_id)
        if not horses:
            print(f"    [V21] no horse data")
            return None
        if rinfo.get("surface") == "障":
            print(f"    [V21] obstacle race - skip")
            return None

        distance = rinfo.get("distance", 0)
        if distance <= 1000:
            print(f"    [V21] distance <=1000m - skip")
            return None

        num_horses = len(horses)

        # odds
        try:
            odds_full = fetch_realtime_odds_full(race_id)
            odds_dict = {u: v["odds"] for u, v in odds_full.items()}
            if odds_full:
                save_odds_base(race_id, odds_full)
            time.sleep(0.5)
        except Exception:
            odds_dict = {}

        # jra + weather
        jra_info, weather_info = {}, {}
        try:
            jra_info, weather_info = fetch_jra_and_weather(rinfo.get("course", ""))
        except Exception:
            pass

        # horse stats
        for i, (horse, hid) in enumerate(zip(horses, horse_ids)):
            if hid:
                try:
                    stats = get_horse_stats(hid, rinfo["distance"], rinfo["surface"], rinfo.get("course", ""))
                    apply_horse_stats(horse, stats, rinfo)
                except Exception:
                    set_horse_defaults(horse)
            else:
                set_horse_defaults(horse)
            if i < num_horses - 1:
                time.sleep(0.2)

        # features
        odds_available = bool(odds_dict and any(v > 0 for v in odds_dict.values()))
        df = build_features(horses, rinfo, v21_model, race_id=race_id,
                            odds_dict=odds_dict, jra_track_info=jra_info,
                            weather_info=weather_info)
        if df is None or len(df) == 0:
            print(f"    [V21] feature build failed")
            return None

        # JRDB merge (best effort; TYB features injected here if available)
        try:
            from jrdb_features import merge_jrdb_predict_features
            df = merge_jrdb_predict_features(df, race_id)
        except Exception:
            pass

        # TYB column injection (if tyb_data available)
        if tyb_data is not None:
            try:
                _inject_tyb_features(df, tyb_data)
                print(f"    [V21] TYB features injected ({len(tyb_data)} horses)")
            except Exception as e:
                print(f"    [V21] TYB inject failed (non-fatal): {e}")

        df = predict_race(df, v21_model, odds_available, race_info=rinfo)
        df = df.sort_values("スコア", ascending=False).reset_index(drop=True)

        cond_key, cond_profile = classify_race_condition(rinfo, num_horses)
        bet_type = cond_profile.get("bet_type", "trio")

        if bet_type == "umaren":
            bets = generate_umaren_bets(df)
        else:
            bets = generate_trio_bets(df)

        top5 = [int(df.iloc[i]["馬番"]) for i in range(min(5, len(df)))]
        top5_names = [str(df.iloc[i].get("馬名", "?")) for i in range(min(5, len(df)))]
        top1 = top5[0] if top5 else 0
        top1_name = top5_names[0] if top5_names else "?"
        top1_score = float(df.iloc[0].get("スコア", 0)) if len(df) > 0 else 0.0

        passes, skip_reason = _passes_strategy_filter(
            race_name, rinfo.get("course", ""), cond_key, distance
        )

        return {
            "race_id": race_id,
            "race_name": race_name,
            "rinfo": rinfo,
            "cond_key": cond_key,
            "bet_type": bet_type,
            "bets": bets,
            "top5": top5,
            "top5_names": top5_names,
            "top1": top1,
            "top1_name": top1_name,
            "top1_score": top1_score,
            "distance": distance,
            "num_horses": num_horses,
            "strategy_pass": passes,
            "strategy_skip_reason": skip_reason,
            "formation": bets,
            "tyb_injected": tyb_data is not None,
        }
    except Exception as e:
        print(f"    [V21] prediction error: {e}")
        return None


def _inject_tyb_features(df, tyb_data: dict) -> None:
    """
    Inject TYB observe data into the feature DataFrame.
    tyb_data: dict keyed by umaban (int or str) → dict of TYB fields.
    TYB feature cols used by V21: padock_idx, sogo_idx, odds_idx, idm, etc.
    """
    import pandas as pd

    TYB_FEATURE_COLS = {
        "tyb_padock_idx": "padock_idx",
        "tyb_sogo_idx": "sogo_idx",
        "tyb_odds_idx": "odds_idx",
        "tyb_idm": "idm",
        "tyb_jockey_idx": "jockey_idx",
        "tyb_info_idx": "info_idx",
        "tyb_bagu_change": "bagu_change",
        "tyb_ashimoto": "ashimoto",
        "tyb_cancel_flag": "cancel_flag",
        "tyb_batai_code": "batai_code",
    }

    horses_by_uma: dict[int, dict] = {}
    if isinstance(tyb_data, dict):
        # If keyed by umaban directly
        for k, v in tyb_data.items():
            try:
                horses_by_uma[int(k)] = v
            except (ValueError, TypeError):
                pass
        # If it's a nested dict with 'horses' key
        if "horses" in tyb_data and isinstance(tyb_data["horses"], dict):
            for k, v in tyb_data["horses"].items():
                try:
                    horses_by_uma[int(k)] = v
                except (ValueError, TypeError):
                    pass

    if not horses_by_uma:
        return

    for feat_col, tyb_field in TYB_FEATURE_COLS.items():
        if feat_col not in df.columns:
            df[feat_col] = 0.0
        for idx in df.index:
            uma = int(df.at[idx, "馬番"]) if "馬番" in df.columns else -1
            if uma in horses_by_uma:
                val = horses_by_uma[uma].get(tyb_field)
                if val is not None:
                    try:
                        df.at[idx, feat_col] = float(val)
                    except (ValueError, TypeError):
                        pass


# ---------------------------------------------------------------------------
# Discord message builder
# ---------------------------------------------------------------------------
def build_discord_message(
    race_id: str,
    pred: dict,
    tyb_data: Optional[dict],
    race_info: dict,
) -> str:
    """Build [V21 paper] Discord message. Always includes paper warning."""
    race_name = pred.get("race_name") or "?"
    rinfo = pred.get("rinfo", {})
    course = rinfo.get("course", race_info.get("course", "?"))
    race_num = rinfo.get("race_num", race_info.get("race_num", "?"))
    surface = rinfo.get("surface", "?")
    condition = rinfo.get("condition", "?")
    distance = pred.get("distance", 0)
    num_horses = pred.get("num_horses", 0)
    cond_key = pred.get("cond_key", "?")
    bet_type = pred.get("bet_type", "trio")
    bets = pred.get("bets", [])
    top5 = pred.get("top5", [])
    top1 = pred.get("top1", "?")
    top1_name = pred.get("top1_name", "?")
    top1_score = pred.get("top1_score", 0.0)
    strategy_pass = pred.get("strategy_pass", False)
    tyb_injected = pred.get("tyb_injected", False)

    # Formation lines
    if not strategy_pass:
        skip_reason = pred.get("strategy_skip_reason", "filter")
        bet_section = f"(戦略フィルタ除外: {skip_reason})\n"
    elif bet_type == "umaren" and bets:
        bet_section = "馬連フォーメーション 2点 (V21 paper)\n"
        for b in bets:
            bet_section += f"  {' - '.join(str(x) for x in b)}\n"
    elif bets:
        n1 = top5[0] if top5 else "?"
        second = top5[1:3] if len(top5) >= 3 else top5[1:]
        third = top5[1:] if len(top5) >= 2 else []
        second_str = ", ".join(str(x) for x in second) if second else "?"
        third_str = ", ".join(str(x) for x in third) if third else "?"
        bet_section = (
            f"三連複フォーメーション {len(bets)}点 (V21 paper)\n"
            f"1列目: {n1}\n"
            f"2列目: {second_str}\n"
            f"3列目: {third_str}\n"
        )
    else:
        bet_section = "(買い目なし)\n"

    tyb_status = "TYB取得済 (直前データ反映)" if tyb_injected else "TYB未取得 (V15同等スコア)"

    msg = (
        f"🚫🚫🚫【V21 paper — 投票禁止】🚫🚫🚫\n"
        f"⚠ V15 との比較用。絶対に投票しないでください。\n"
        f"⚠ 実際の投票は V15 (上の買い目) のみ使用。\n"
        f"─────────────────────────\n"
        f"{course}{race_num}R {race_name}\n"
        f"   {surface}{distance}m {condition} 条件{cond_key} ({num_horses}頭)\n"
        f"V21候補モデル WF AUC 0.8696 / V15+TYB10 features\n\n"
        f"{bet_section}\n"
        f"軸: {top1}({top1_name}) スコア{top1_score:.4f}\n"
        f"Top5: {' → '.join(str(x) for x in top5)}\n"
        f"TYB: {tyb_status}\n"
        f"─────────────────────────\n"
        f"🚫 これはpaper予測。V15買い目のみで投票してください 🚫"
    )
    return msg


# ---------------------------------------------------------------------------
# Discord sender
# ---------------------------------------------------------------------------
def send_discord(message: str) -> bool:
    """POST message to DISCORD_WEBHOOK. Returns True on success."""
    if not DISCORD_WEBHOOK:
        print("[Discord] V21 paper webhook not configured - skip send")
        return False
    try:
        import requests
        resp = requests.post(DISCORD_WEBHOOK, json={"content": message}, timeout=10)
        ok = resp.status_code in (200, 204)
        if not ok:
            print(f"[Discord] send failed: HTTP {resp.status_code}")
        return ok
    except Exception as e:
        print(f"[Discord] send error: {e}")
        return False


# ---------------------------------------------------------------------------
# Paper log
# ---------------------------------------------------------------------------
def record_paper_log(
    race_id: str,
    date_str: str,
    pred: Optional[dict],
    tyb_data: Optional[dict],
) -> None:
    """Save prediction to data/v21_paper_log/{date}/{race_id}.json."""
    try:
        log_dir = PAPER_LOG_DIR / date_str
        log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "race_id": race_id,
            "timestamp": datetime.now().isoformat(),
            "v21_top5": pred["top5"] if pred else None,
            "v21_top5_names": pred["top5_names"] if pred else None,
            "v21_formation": pred["formation"] if pred else None,
            "v21_strategy_pass": pred["strategy_pass"] if pred else None,
            "v21_strategy_skip_reason": pred["strategy_skip_reason"] if pred else None,
            "cond_key": pred["cond_key"] if pred else None,
            "bet_type": pred["bet_type"] if pred else None,
            "distance": pred.get("distance") if pred else None,
            "num_horses": pred.get("num_horses") if pred else None,
            "race_name": str(pred.get("race_name", "")) if pred else None,
            "course": pred["rinfo"].get("course", "") if pred and pred.get("rinfo") else None,
            "tyb_injected": pred.get("tyb_injected", False) if pred else False,
            "tyb_data_keys": list(tyb_data.keys()) if isinstance(tyb_data, dict) else None,
            "prediction_failed": pred is None,
        }
        out_file = log_dir / f"{race_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)
        print(f"    [V21 log] saved: {out_file}")
    except Exception as e:
        print(f"    [V21 log] write failed: {e}")


# ---------------------------------------------------------------------------
# Timer: fire per-race
# ---------------------------------------------------------------------------
def fire_race_notify(race_info: dict, v21_model: dict) -> None:
    """
    Called by threading.Timer at race_time - NOTIFY_BEFORE_MIN.
    Fetch TYB → predict V21 → filter → send Discord → log.
    All exceptions are swallowed so the scheduler keeps running.
    """
    race_id = race_info.get("race_id", "?")
    course = race_info.get("course", "?")
    race_num = race_info.get("race_num", "?")
    start_time_str = ""
    try:
        start_dt = race_info.get("start_dt")
        if start_dt:
            start_time_str = start_dt.strftime("%H%M")
    except Exception:
        pass

    print(f"\n  >>> [V21 paper] {course}{race_num}R ({race_id}) @ {datetime.now().strftime('%H:%M')}")

    date_str = datetime.now().strftime("%Y%m%d")

    try:
        # 1. Fetch TYB
        tyb_data: Optional[dict] = None
        try:
            tyb_data = fetch_tyb_for_race(race_id, start_time_str)
            if tyb_data is not None:
                print(f"    [TYB] fetched OK")
            else:
                print(f"    [TYB] not available (disabled or failed)")
        except Exception as e:
            print(f"    [TYB] error (non-fatal): {e}")

        # 2. Predict V21
        pred: Optional[dict] = None
        try:
            pred = predict_v21(race_id, v21_model, tyb_data)
        except Exception as e:
            print(f"    [V21] predict error (non-fatal): {e}")

        if pred is None:
            print(f"    [V21] prediction failed - logging and continuing")
            record_paper_log(race_id, date_str, None, tyb_data)
            return

        print(f"    [V21] top1={pred['top1']}({pred['top1_name']}) "
              f"cond={pred['cond_key']} pass={pred['strategy_pass']}")

        # 3. Build message and send Discord (regardless of strategy pass — always log, only notify if pass)
        try:
            msg = build_discord_message(race_id, pred, tyb_data, race_info)
        except Exception as e:
            print(f"    [Discord] message build error: {e}")
            msg = (
                f"🚫🚫🚫【V21 paper — 投票禁止】🚫🚫🚫\n"
                f"⚠ 絶対に投票しないでください。実際の投票は V15 のみ。\n"
                f"{course}{race_num}R ({race_id})\n"
                f"[メッセージ生成エラー: {e}]\n"
                f"🚫 これはpaper予測。V15買い目のみで投票してください 🚫"
            )

        # 4. Send only if strategy passes
        if pred.get("strategy_pass"):
            try:
                ok = send_discord(msg)
                print(f"    [Discord] V21 paper sent: {ok}")
            except Exception as e:
                print(f"    [Discord] send error (non-fatal): {e}")
        else:
            reason = pred.get("strategy_skip_reason", "filter")
            print(f"    [V21] strategy filter → Discord skip (reason: {reason})")

        # 5. Always record paper log
        record_paper_log(race_id, date_str, pred, tyb_data)

    except Exception as e:
        # Top-level swallow — scheduler must never crash
        print(f"    [V21] UNHANDLED error in fire_race_notify (swallowed): {e}")
        try:
            record_paper_log(race_id, date_str, None, None)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Schedule per-race timer
# ---------------------------------------------------------------------------
def schedule_race_notify(race_info: dict, v21_model: dict) -> Optional[threading.Timer]:
    """Schedule threading.Timer to fire NOTIFY_BEFORE_MIN before race start."""
    start_dt: Optional[datetime] = race_info.get("start_dt")
    if start_dt is None:
        return None

    notify_dt = start_dt - timedelta(minutes=NOTIFY_BEFORE_MIN)
    now = datetime.now()
    delay_sec = (notify_dt - now).total_seconds()

    if delay_sec < -60:
        return None  # already past

    delay_sec = max(delay_sec, 0)
    t = threading.Timer(delay_sec, fire_race_notify, args=(race_info, v21_model))
    t.daemon = True
    t.start()
    return t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Entry point.
    Usage: python tools/v21_per_race_paper.py [--date YYYYMMDD]
    """
    parser = argparse.ArgumentParser(
        description="V21 paper per-race prediction scheduler (fires 17min before each race)"
    )
    parser.add_argument("--date", type=str, default="", help="Date YYYYMMDD (default: today)")
    args = parser.parse_args()

    date_str = args.date.strip() or datetime.now().strftime("%Y%m%d")

    print("=" * 60)
    print(f"  V21 Per-Race Paper Scheduler: {date_str}")
    print(f"  Fires: {NOTIFY_BEFORE_MIN} min before each race")
    print(f"  V21 model: {V21_MODEL_PATH}")
    print(f"  Discord: {'configured' if DISCORD_WEBHOOK else 'NOT CONFIGURED'}")
    print("=" * 60)

    # Load V21 model once (not V15)
    v21_model = load_v21_model()
    if v21_model is None:
        print("[ERROR] V21 model not found at models/v21_candidate.pkl.gz - exiting")
        sys.exit(1)

    # Load race schedule
    print("\n  Fetching race schedule...")
    races = load_race_schedule(date_str)
    print(f"  Found: {len(races)} races with start_time")

    if not races:
        print("  No races found (or no races with known start time).")
        return

    # Schedule timers
    print("\n  Scheduling V21 paper timers...")
    timers: list[threading.Timer] = []
    now = datetime.now()

    for race in races:
        start_dt: datetime = race["start_dt"]
        notify_dt = start_dt - timedelta(minutes=NOTIFY_BEFORE_MIN)
        delay_sec = (notify_dt - now).total_seconds()

        if delay_sec < -60:
            print(f"  {race['course']}{race['race_num']}R {race.get('start_time','?')}: already passed - skip")
            continue

        delay_sec = max(delay_sec, 0)
        t = threading.Timer(delay_sec, fire_race_notify, args=(race, v21_model))
        t.daemon = True
        t.start()
        timers.append(t)
        print(
            f"  {race['course']}{race['race_num']}R {race.get('start_time','?')}: "
            f"V21 paper at {notify_dt.strftime('%H:%M')} "
            f"(in {int(delay_sec // 60)}min)"
        )

    active = len(timers)
    print(f"\n  Active V21 paper timers: {active}")

    if active == 0:
        print("  All races already passed.")
        return

    print(f"  Waiting for races... (Ctrl+C to stop)")
    print(f"  Log dir: {PAPER_LOG_DIR / date_str}")
    print()

    try:
        while any(t.is_alive() for t in timers):
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n  V21 per-race paper stopped by user.")
    finally:
        for t in timers:
            t.cancel()

    print("\n  All V21 paper timers fired. Exiting.")


if __name__ == "__main__":
    main()
