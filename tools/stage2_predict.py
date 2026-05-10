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

# Session #78: 5/9 hardcode 撤廃。 DATE は CLI --date で override、 default = 当日。
DATE = datetime.now().strftime("%Y%m%d")
DAILY_PRED = BASE / "data" / "daily_predictions" / f"{DATE}.csv"
DAILY_PRED_FULL_DIR = BASE / "data" / "daily_predictions_full"  # Session #71 生成 (Session #72 で参照)
OUT_DIR = BASE / "data" / "v18"
def _date_short(d: str) -> str:
    return f"{int(d[4:6])}_{int(d[6:8])}"
CACHE_PATH = OUT_DIR / f"pre_race_predict_cache_{_date_short(DATE)}.json"
KILL_SWITCH = OUT_DIR / "pre_race_predict.kill"

# Session #72 C: Discord body 上限 (実 2000 char、 安全 margin で 1700 以下に table 抑制)
DISCORD_BODY_SAFE_LIMIT = 1700
HORSE_NAME_TRUNC = 14

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
    # ========== 5/10 (Phase 4 Session 追加、 5/9 same time slot 仮定) ==========
    # 京都 5/10 (kaiji 3 day 6)
    "202608030601": "09:55", "202608030602": "10:25", "202608030603": "10:55",
    "202608030604": "11:30", "202608030605": "12:00", "202608030606": "12:35",
    "202608030607": "13:30", "202608030608": "14:00", "202608030609": "14:25",
    "202608030610": "14:55", "202608030611": "15:30", "202608030612": "16:00",
    # 東京 5/10 (kaiji 2 day 6)
    "202605020601": "10:10", "202605020602": "10:40", "202605020603": "11:15",
    "202605020604": "11:50", "202605020605": "12:25", "202605020606": "13:00",
    "202605020607": "13:45", "202605020608": "14:15", "202605020609": "14:45",
    "202605020610": "15:15", "202605020611": "15:45", "202605020612": "16:25",
    # 新潟 5/10 (kaiji 1 day 4、 R4 不在)
    "202604010401": "10:00", "202604010402": "10:30", "202604010403": "11:00",
    "202604010405": "12:00", "202604010406": "12:30", "202604010407": "13:00",
    "202604010408": "13:30", "202604010409": "14:00", "202604010410": "14:30",
    "202604010411": "15:20", "202604010412": "16:10",
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


def load_full_predictions(race_id: str, date: str = DATE) -> list[dict] | None:
    """Session #72 C: data/daily_predictions_full/{date}.csv から race_id の全馬 row を返す.

    schema (Session #71 生成想定):
      race_id, course, race_num, race_name, num_horses, distance, surface, track_condition,
      horse_rank, umaban, horse_name, score, odds

    file 不在 / race_id 不在 / 行数 0 → None (top3 fallback へ)。
    """
    import pandas as pd
    csv_path = DAILY_PRED_FULL_DIR / f"{date}.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        print(f"[full-pred] read error: {e}")
        return None
    if "race_id" not in df.columns:
        return None
    sub = df[df["race_id"].astype(str) == race_id].copy()
    if len(sub) == 0:
        return None
    # Phase 5 (5/10): Session #71 daily_predictions_full schema 名称対応
    # CSV 実列: V15_score / rank_in_race / horse_num
    # build_horse_table 期待: score / horse_rank / umaban
    rename_map = {}
    if "V15_score" in sub.columns and "score" not in sub.columns:
        rename_map["V15_score"] = "score"
    if "rank_in_race" in sub.columns and "horse_rank" not in sub.columns:
        rename_map["rank_in_race"] = "horse_rank"
    if "horse_num" in sub.columns and "umaban" not in sub.columns:
        rename_map["horse_num"] = "umaban"
    if rename_map:
        sub = sub.rename(columns=rename_map)
    # numeric coerce
    for c in ("score", "odds"):
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    if "horse_rank" in sub.columns:
        sub["horse_rank"] = pd.to_numeric(sub["horse_rank"], errors="coerce")
        sub = sub.sort_values("horse_rank", na_position="last")
    elif "score" in sub.columns:
        sub = sub.sort_values("score", ascending=False, na_position="last")
    return sub.to_dict(orient="records")


def build_horse_table(rows: list[dict],
                      max_chars: int = DISCORD_BODY_SAFE_LIMIT) -> tuple[str, int]:
    """Session #72 C: 全馬 V15 score 順 markdown table を構築.

    rows: [{horse_rank, umaban, horse_name, score, odds}, ...] (sorted)
    戻り値: (table_str, n_truncated)
    """
    header = ("| 順 | 馬番 | 馬名 | V15 score | 単勝オッズ |\n"
              "|----|----|--------|------|------|\n")
    lines = [header.rstrip("\n")]
    body_lines: list[str] = []

    def _format_row(r: dict, rank: int) -> str:
        umaban = r.get("umaban", "?")
        name = (r.get("horse_name") or "?")[:HORSE_NAME_TRUNC]
        score = r.get("score")
        odds = r.get("odds")
        try:
            score_str = f"{float(score):.3f}" if score not in (None, "") else "-"
        except (TypeError, ValueError):
            score_str = "-"
        try:
            odds_str = f"{float(odds):.1f}" if odds not in (None, "") else "-"
        except (TypeError, ValueError):
            odds_str = "-"
        return f"| {rank} | {umaban} | {name} | {score_str} | {odds_str} |"

    n_truncated = 0
    cur = "\n".join(lines) + "\n"
    for i, r in enumerate(rows, 1):
        line = _format_row(r, i)
        if len(cur) + len(line) + 1 > max_chars:
            n_truncated = len(rows) - i + 1
            break
        body_lines.append(line)
        cur = cur + line + "\n"

    table = "\n".join(lines + body_lines)
    if n_truncated > 0:
        table += f"\n... (以下 {n_truncated} 頭省略、 size 上限のため)"
    return table, n_truncated


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


def build_message_all_horses(race_id: str, morning_row, stage2: dict,
                             full_rows: list[dict]) -> tuple[str, str, str]:
    """Session #72 C: 全馬 V15 score 順 通知 (Session #71 連携、 daily_predictions_full 利用).

    full_rows: load_full_predictions(race_id) の結果 (sorted by horse_rank)。
    title 例: "5/10 R12 新潟 4歳以上1勝クラス (16:10) 全馬 V15 score"
    body: meta + 全馬 markdown table + Stage 2 status + 投票方針
    """
    course = str(morning_row.get("course", "?"))
    rn = str(morning_row.get("race_num", "?"))
    rname = str(morning_row.get("race_name", "?"))
    start = RACE_START_TIMES.get(race_id, "?")
    distance = morning_row.get("distance", "?")
    surface = morning_row.get("surface", "?")
    cond = morning_row.get("track_condition", "?")
    n_horses = len(full_rows)

    title = f"R{rn} {course} {rname} ({start} 発走) 全馬 V15 score"

    table_str, n_trunc = build_horse_table(full_rows)

    s2_status_lines: list[str] = []
    if stage2.get("error"):
        kind = stage2.get("error_kind") or "unknown"
        diag = stage2.get("diag") or {}
        s2_status_lines.append(f"### Stage 2 状況 (失敗: {kind})")
        if kind == "netkeiba_block":
            s2_status_lines.append(f"- netkeiba HTTP {diag.get('status_code')} (server block 想定) → Stage 1 (朝予測) 採用")
        elif kind == "netkeiba_other":
            s2_status_lines.append(f"- netkeiba HTTP {diag.get('status_code')} → Stage 1 採用")
        elif kind == "shutuba_empty":
            s2_status_lines.append(f"- shutuba page 空 → Stage 1 採用")
        elif kind == "exception":
            s2_status_lines.append(f"- 例外発生 → Stage 1 採用")
        s2_status_lines.append("- 次 fire (30 分後) で再試行")
    else:
        s2_status_lines.append(f"### Stage 2 状況 (成功)")
        s2_status_lines.append(f"- 当日体重統合: 取得済 / オッズ 反映済")
        s2_top3 = stage2.get("top3") or []
        if s2_top3:
            morn_top1 = str(morning_row.get("top1_num", ""))
            new_top1 = str(s2_top3[0].get("umaban", ""))
            if new_top1 != morn_top1 and new_top1:
                s2_status_lines.append(f"- ★ Stage 2 top1 変更: {morn_top1} → {new_top1}")
            else:
                s2_status_lines.append(f"- top1 不変")

    body_parts = [
        f"## R{rn} {course} {rname} ({start} 発走)",
        f"出走: {n_horses} 頭 / コース: {course} {surface}{distance}m / 馬場: {cond}",
        "",
        f"### 全馬 V15 score 順 (Stage 1 = 朝予測 8:00)",
        table_str,
        "",
    ]
    body_parts += s2_status_lines + [
        "",
        "### V15 投票方針 (絶対遵守)",
        "- Stage 2 通知は学習用、 投票推奨ではない",
        "- 投票は朝予測 (Stage 1) の trio_bets に従う",
    ]
    body = "\n".join(body_parts)
    color = "blue" if not stage2.get("error") else "yellow"
    return title, body, color


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

    # Session #72 C: 全馬 score available なら新通知 (table)、 不在なら旧 top3 通知
    full_rows = load_full_predictions(race_id, DATE)
    if full_rows:
        print(f"[full-pred] {len(full_rows)} 馬 取得 → 全馬 score 順 table 通知")
        title, body, color = build_message_all_horses(race_id, morning, stage2, full_rows)
    else:
        print(f"[full-pred] daily_predictions_full 不在 → top3 通知 fallback (旧 logic)")
        title, body, color = build_message(race_id, morning, stage2)
    print(body)

    out_path = OUT_DIR / f"pre_race_predict_{_date_short(DATE)}_R{morning.get('race_num')}_{morning.get('course')}_{race_id}.json"
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
    p.add_argument("--date", default=None,
                   help="Session #78: 対象日 YYYYMMDD (default = 当日)")
    p.add_argument("--dry-run", action="store_true",
                   help="Session #78: import + path 検証のみ (predict 実行せず exit)")
    args = p.parse_args()

    # Session #78: --date 指定時は module 定数 を上書き
    if args.date:
        global DATE, DAILY_PRED, CACHE_PATH
        DATE = args.date
        DAILY_PRED = BASE / "data" / "daily_predictions" / f"{DATE}.csv"
        CACHE_PATH = OUT_DIR / f"pre_race_predict_cache_{_date_short(DATE)}.json"
        print(f"[date] override DATE={DATE}")

    if args.dry_run:
        print(f"[dry-run] DATE={DATE}")
        print(f"[dry-run] DAILY_PRED={DAILY_PRED} exists={DAILY_PRED.exists()}")
        print(f"[dry-run] CACHE_PATH={CACHE_PATH}")
        print(f"[dry-run] KILL_SWITCH={KILL_SWITCH} exists={KILL_SWITCH.exists()}")
        print(f"[dry-run] OK")
        return

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
