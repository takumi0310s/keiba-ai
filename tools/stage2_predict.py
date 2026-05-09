"""5/9 Stage 2 (1h 前) 個別 R 予測 + 朝予測 (Stage 1) との差分 Discord 通知 (Session #65 B).

dev/two-stage 専用、 V15 production 完全独立 (read-only):
  - tools/predict_one_race.py (V15 ensemble、 関数のみ呼ぶ)
  - tools/race_day_weight_features.py (Session #48 B、 features 計算)
  - tools/notify.py (Discord)

絶対遵守:
  - V15 model file 触らない
  - daily_predict.py / race_auto_notify.py を 絶対 trigger しない
  - Session #61 schtasks 9 件不変
  - Stage 2 予測は 学習用、 投票推奨ではない
  - kill-switch (data/v18/pre_race_predict.kill) で即停止可

注: 既存 tools/pre_race_predict.py は 「前夜予測 (daily_predict ラッパー)」
で別目的。 本 file は別名で隔離。

CLI:
  python tools/stage2_predict.py --race-id 202604010312     # 単一 R
  python tools/stage2_predict.py --check-next-1h            # 次 1h の全 R
  python tools/stage2_predict.py --no-discord               # dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tools"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DATE = "20260509"
DAILY_PRED = BASE / "data" / "daily_predictions" / f"{DATE}.csv"
OUT_DIR = BASE / "data" / "v18"
CACHE_PATH = OUT_DIR / "pre_race_predict_cache_5_9.json"
KILL_SWITCH = OUT_DIR / "pre_race_predict.kill"

# Session #65 A 静的 schedule (anchor base、 11R 実時刻 + JRA 標準 interval)
RACE_START_TIMES = {
    # 京都
    "202608030501": "09:55", "202608030502": "10:25", "202608030503": "10:55",
    "202608030504": "11:30", "202608030505": "12:00", "202608030506": "12:35",
    "202608030507": "13:30", "202608030508": "14:00", "202608030509": "14:25",
    "202608030510": "14:55", "202608030511": "15:30", "202608030512": "16:00",
    # 東京
    "202605020501": "10:10", "202605020502": "10:40", "202605020503": "11:15",
    "202605020504": "11:50", "202605020505": "12:25", "202605020506": "13:00",
    "202605020507": "13:45", "202605020508": "14:15", "202605020509": "14:45",
    "202605020510": "15:15", "202605020511": "15:45", "202605020512": "16:25",
    # 新潟 (1R/4R 不在)
    "202604010302": "10:30", "202604010303": "11:00",
    "202604010305": "12:00", "202604010306": "12:30", "202604010307": "13:00",
    "202604010308": "13:30", "202604010309": "14:00", "202604010310": "14:30",
    "202604010311": "15:20", "202604010312": "16:10",
}


def load_morning_predictions():
    import pandas as pd
    return pd.read_csv(DAILY_PRED, dtype=str)


def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def get_morning_row(race_id: str):
    df = load_morning_predictions()
    sub = df[df["race_id"].astype(str) == race_id]
    if len(sub) == 0:
        return None
    return sub.iloc[0]


def parse_start_time(race_id: str, today: datetime | None = None) -> datetime | None:
    hhmm = RACE_START_TIMES.get(race_id)
    if not hhmm:
        return None
    today = today or datetime.now()
    h, m = map(int, hhmm.split(":"))
    return today.replace(hour=h, minute=m, second=0, microsecond=0)


def races_in_next_window(window_min: int = 60, now: datetime | None = None) -> list[str]:
    now = now or datetime.now()
    end = now + timedelta(minutes=window_min)
    out = []
    for race_id in RACE_START_TIMES:
        st = parse_start_time(race_id, now)
        if st is None:
            continue
        if now <= st <= end:
            out.append(race_id)
    out.sort(key=lambda rid: parse_start_time(rid, now))
    return out


def predict_stage2(race_id: str) -> dict:
    """V15 ensemble で再予測。 戻り値: {race_name, rinfo, top3, n_horses, error}"""
    try:
        import predict_one_race as por
        ret = por.predict_one_race(race_id)
        if ret is None:
            return {"error": "predict_one_race returned None"}
        result, race_name, rinfo = ret
        # スコア降順 sort 済 (predict_one_race 末尾)
        top3 = []
        for _, row in result.head(3).iterrows():
            top3.append({
                "umaban": str(row.get("馬番", "?")),
                "name": str(row.get("馬名", "?")),
                "score": float(row.get("スコア", 0) or 0),
            })
        return {
            "race_name": race_name,
            "rinfo": dict(rinfo) if rinfo else {},
            "top3": top3,
            "n_horses": len(result),
            "error": None,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


def diff_top1(morning_top1: str, stage2_top1: str) -> bool:
    return str(morning_top1) != str(stage2_top1)


def build_message(race_id: str, morning_row, stage2: dict) -> tuple[str, str, str]:
    """戻り値: (title, body, color)"""
    course = str(morning_row.get("course", "?"))
    rn = str(morning_row.get("race_num", "?"))
    rname = str(morning_row.get("race_name", "?"))
    start = RACE_START_TIMES.get(race_id, "?")
    morning_top1 = str(morning_row.get("top1_num", ""))
    morning_top1_name = str(morning_row.get("top1_name", ""))
    morning_top2 = str(morning_row.get("top2_num", ""))
    morning_top2_name = str(morning_row.get("top2_name", ""))
    morning_top3 = str(morning_row.get("top3_num", ""))
    morning_top3_name = str(morning_row.get("top3_name", ""))
    morning_score = float(morning_row.get("top1_score", 0) or 0)

    title = f"R{rn} {course} 1h 前予測 (Stage 2)"

    if stage2.get("error"):
        body = (
            f"## {title}\n"
            f"発走: {start} / レース: {rname}\n\n"
            f"### Stage 2 予測 失敗\n"
            f"error: `{stage2['error']}`\n\n"
            f"### 朝予測 (Stage 1) top3\n"
            f"1. {morning_top1} {morning_top1_name} (score={morning_score:.3f})\n"
            f"2. {morning_top2} {morning_top2_name}\n"
            f"3. {morning_top3} {morning_top3_name}\n\n"
            f"★ V15 投票方針 (絶対遵守) ★\n"
            f"- 新潟 12R 4歳以上1勝 ¥700 のみ (案B改 strict)\n"
            f"- 11R 重賞 投票しない / Stage 2 は学習用、 投票推奨ではない\n"
        )
        return title, body, "yellow"

    s2 = stage2["top3"]
    new_top1 = s2[0]["umaban"] if s2 else "?"
    new_top1_name = s2[0]["name"] if s2 else "?"
    new_score = s2[0]["score"] if s2 else 0.0
    score_diff = new_score - morning_score
    top1_changed = diff_top1(morning_top1, new_top1)

    lines = [
        f"## {title}",
        f"発走: {start} / {rname}",
        "",
        f"### 朝予測 (Stage 1) top3",
        f"1. {morning_top1} {morning_top1_name} (score={morning_score:.3f})",
        f"2. {morning_top2} {morning_top2_name}",
        f"3. {morning_top3} {morning_top3_name}",
        "",
        f"### 1h 前予測 (Stage 2) top3",
    ]
    for i, h in enumerate(s2, 1):
        marker = " ★ NEW" if (i == 1 and top1_changed) else ""
        lines.append(f"{i}. {h['umaban']} {h['name']} (score={h['score']:.3f}){marker}")
    lines.append(f"confidence diff: {score_diff:+.3f}")
    lines.append("")
    lines.append("### 差分 alert")
    if top1_changed:
        lines.append(f"- ★ top1 変更: {morning_top1} {morning_top1_name} → {new_top1} {new_top1_name}")
    else:
        lines.append(f"- top1 不変: {new_top1} {new_top1_name}")
    lines.append("")
    lines.append("### 信頼度")
    lines.append(f"- 当日体重: predict_core build_features 経由 (取得可不可は log 参照)")
    lines.append(f"- オッズ: predict_one_race 内 fetch_realtime_odds_full (失敗時 朝オッズ)")
    lines.append("")
    lines.append("★ V15 投票方針 (絶対遵守) ★")
    lines.append("- 新潟 12R 4歳以上1勝 ¥700 のみ (案B改 strict)")
    lines.append("- 11R 重賞 投票しない (verdict 用)")
    lines.append("- Stage 2 は学習用、 投票推奨ではない")
    lines.append("- 累計 +13,530 円 死守")

    color = "yellow" if top1_changed else "blue"
    return title, "\n".join(lines), color


def send_discord(title: str, body: str, color: str, dedup_key: str) -> bool:
    try:
        from notify import send_discord as _send
        return _send(title, body, color=color, channel="bets")
    except Exception as e:
        print(f"[discord error] {e}", file=sys.stderr)
        return False


def predict_one(race_id: str, force: bool = False, no_discord: bool = False) -> dict:
    cache = load_cache()
    if not force and race_id in cache:
        print(f"[skip dedup] {race_id} already predicted at {cache[race_id]}")
        return {"race_id": race_id, "skipped": "dedup"}

    morning = get_morning_row(race_id)
    if morning is None:
        print(f"[skip] {race_id} not in daily_predictions")
        return {"race_id": race_id, "skipped": "not_in_morning"}

    print(f"=== Stage 2 predict {race_id} ({morning.get('course')} R{morning.get('race_num')}) ===")
    stage2 = predict_stage2(race_id)
    title, body, color = build_message(race_id, morning, stage2)
    print(body)

    out_path = OUT_DIR / f"pre_race_predict_5_9_R{morning.get('race_num')}_{morning.get('course')}_{race_id}.json"
    out_path.write_text(json.dumps({
        "race_id": race_id,
        "morning": {
            "top1_num": str(morning.get("top1_num", "")),
            "top1_name": str(morning.get("top1_name", "")),
            "top1_score": float(morning.get("top1_score", 0) or 0),
            "top2_num": str(morning.get("top2_num", "")),
            "top2_name": str(morning.get("top2_name", "")),
            "top3_num": str(morning.get("top3_num", "")),
            "top3_name": str(morning.get("top3_name", "")),
        },
        "stage2": stage2,
        "ts": datetime.now().isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out: {out_path.relative_to(BASE)}")

    if not no_discord:
        ok = send_discord(title, body, color, dedup_key=f"pre_race_{race_id}")
        print(f"[discord] {'sent' if ok else 'FAIL'}")

    cache[race_id] = datetime.now().isoformat()
    save_cache(cache)
    return {"race_id": race_id, "predicted": True, "title": title}


def cmd_check_next_1h(args):
    if KILL_SWITCH.exists():
        print(f"[kill-switch] {KILL_SWITCH} → no-op exit")
        return
    rids = races_in_next_window(window_min=60)
    print(f"[check_next_1h] window=60min, candidates={len(rids)}: {rids}")
    for rid in rids:
        try:
            predict_one(rid, force=False, no_discord=args.no_discord)
        except Exception as e:
            print(f"[error] {rid}: {e}")
        time.sleep(2)  # rate limit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--race-id", default=None)
    p.add_argument("--check-next-1h", action="store_true")
    p.add_argument("--no-discord", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="dedup cache を無視して再予測")
    args = p.parse_args()

    if KILL_SWITCH.exists():
        print(f"[kill-switch] {KILL_SWITCH} → no-op exit")
        return

    if args.check_next_1h:
        cmd_check_next_1h(args)
        return
    if args.race_id:
        predict_one(args.race_id, force=args.force, no_discord=args.no_discord)
        return
    p.error("--race-id か --check-next-1h を指定してください")


if __name__ == "__main__":
    main()
