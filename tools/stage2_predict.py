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


def _probe_netkeiba(race_id: str) -> dict:
    """Session #68 C: parse_shutuba 失敗時の診断 (status_code / len / Cookie 有無)."""
    import requests
    try:
        import predict_core as pc
        h = pc._get_headers_with_cookie()
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        r = requests.get(url, headers=h, timeout=10)
        text = r.text or ""
        return {
            "status_code": r.status_code,
            "response_len": len(text),
            "has_horse_list": "HorseList" in text,
            "cookie_present": "Cookie" in h,
            "url": url,
        }
    except Exception as e:
        return {"probe_error": f"{type(e).__name__}: {e}"}


def predict_stage2(race_id: str) -> dict:
    """V15 ensemble で再予測。
    戻り値: {race_name, rinfo, top3, n_horses, error, error_kind, diag}
    Session #68 C: error_kind を分類 (netkeiba_block / shutuba_empty / exception / None)。
    """
    try:
        import predict_one_race as por
        ret = por.predict_one_race(race_id)
        if ret is None:
            diag = _probe_netkeiba(race_id)
            print(f"[stage2-trace] predict_one_race returned None, diag={diag}")
            err = "predict_one_race returned None"
            kind = "shutuba_empty"
            if diag.get("status_code") == 400:
                err = f"netkeiba HTTP 400 (server block) / len={diag.get('response_len')}"
                kind = "netkeiba_block"
            elif diag.get("status_code") and diag.get("status_code") != 200:
                err = f"netkeiba HTTP {diag['status_code']} / len={diag.get('response_len')}"
                kind = "netkeiba_other"
            return {"error": err, "error_kind": kind, "diag": diag}
        result, race_name, rinfo = ret
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
            "error_kind": None,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": f"{type(e).__name__}: {e}",
            "error_kind": "exception",
            "diag": {"trace": traceback.format_exc()[-500:]},
        }


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
        kind = stage2.get("error_kind") or "unknown"
        diag = stage2.get("diag") or {}
        diag_str = ""
        if kind == "netkeiba_block":
            diag_str = f"  - netkeiba HTTP {diag.get('status_code')} (server block 想定、 Session #62/63 既知)\n"
        elif kind == "netkeiba_other":
            diag_str = f"  - netkeiba HTTP {diag.get('status_code')} / len={diag.get('response_len')}\n"
        elif kind == "shutuba_empty":
            diag_str = f"  - shutuba page 空 (HorseList not found、 status={diag.get('status_code', '?')})\n"
        elif kind == "exception":
            diag_str = f"  - 例外発生 (trace tail 確認)\n"

        body = (
            f"## {title} — Stage 1 fallback 採用\n"
            f"発走: {start} / レース: {rname}\n\n"
            f"### Stage 2 失敗 ({kind})\n"
            f"error: `{stage2['error']}`\n"
            f"{diag_str}"
            f"\n### 採用予測 = 朝予測 (Stage 1) top3 ★\n"
            f"1. {morning_top1} {morning_top1_name} (score={morning_score:.3f})\n"
            f"2. {morning_top2} {morning_top2_name}\n"
            f"3. {morning_top3} {morning_top3_name}\n\n"
            f"※ Stage 2 取得不可のため、 朝予測 (Stage 1) を本 R の最終予測として扱う。\n"
            f"※ retry: 次 fire (30 分後) で自動再試行 (cache に書込み無し)\n\n"
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

    # Session #68 C: 失敗時は cache に書き込まない (= 次 fire で再試行可)
    if stage2.get("error") is None:
        cache[race_id] = datetime.now().isoformat()
        save_cache(cache)
        return {"race_id": race_id, "predicted": True, "title": title}
    else:
        print(f"[stage2-trace] skip cache write (error_kind={stage2.get('error_kind')}) → next fire で再試行")
        return {"race_id": race_id, "predicted": False,
                "error": stage2.get("error"),
                "error_kind": stage2.get("error_kind"),
                "title": title}


def cmd_check_next_1h(args):
    if KILL_SWITCH.exists():
        print(f"[kill-switch] {KILL_SWITCH} → no-op exit")
        return

    rids = races_in_next_window(window_min=60)
    print(f"[check_next_1h] window=60min, candidates={len(rids)}: {rids}")

    # Session #68 C: fire 起動時 1 回 netkeiba probe → block 検知時 全 R skip
    if rids:
        diag = _probe_netkeiba(rids[0])
        print(f"[probe] netkeiba: status={diag.get('status_code')} len={diag.get('response_len')} cookie={diag.get('cookie_present')}")
        if diag.get("status_code") == 400 and not args.no_discord and not args.skip_block_alert:
            try:
                from notify import send_discord as _send
                _send(
                    "Stage 2 fire skip (netkeiba block 検知)",
                    f"## Stage 2 fire skip\n"
                    f"netkeiba HTTP 400 (server block) を検知 (probe race_id={rids[0]})。\n"
                    f"今 fire の Stage 2 予測は全 R skip、 次 30 分後の fire で再試行。\n\n"
                    f"※ 朝予測 (Stage 1) が最終予測として有効。\n"
                    f"※ block 解除されれば 自動復活。",
                    color="yellow",
                    channel="bets",
                )
            except Exception as e:
                print(f"[discord block-alert error] {e}")
            print(f"[probe] block 検知 → 全 R skip")
            return

    for rid in rids:
        try:
            predict_one(rid, force=args.force, no_discord=args.no_discord)
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
    p.add_argument("--skip-block-alert", action="store_true",
                   help="netkeiba block 検知時 Discord 警告を送らない (test 用)")
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
