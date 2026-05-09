"""Session #63 F: Discord 5 通 (重賞 3 + 12R 1 + 完了 1).

dedup logic: data/discord_dedup_state.json の dedup_key で 重複防止。

usage:
  python tools/notify_session_63.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")
sys.path.insert(0, str(BASE / "tools"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from notify import send_discord  # noqa: E402

DEDUP_STATE = BASE / "data" / "discord_dedup_state.json"
SCORES_CSV = BASE / "data" / "v18" / "horse_total_scores_5_9.csv"
DAILY_PRED = BASE / "data" / "daily_predictions" / "20260509.csv"

RACE_GROUPS = {
    "kyoto_g2":   ["202608030511"],
    "epsom_g3":   ["202605020511"],
    "sprint_op":  ["202604010311"],
    "r12_class":  ["202608030512", "202604010312", "202605020512"],
}

RACE_LABEL = {
    "202608030511": ("京都", 11, "京都新聞杯", "G2", "15:30"),
    "202608030512": ("京都", 12, "4歳以上2勝クラス", "-", "16:00"),
    "202605020511": ("東京", 11, "エプソムカップ", "G3", "15:45"),
    "202605020512": ("東京", 12, "4歳以上2勝クラス", "-", "16:25"),
    "202604010311": ("新潟", 11, "駿風 S", "OP", "15:20"),
    "202604010312": ("新潟", 12, "4歳以上1勝クラス (★V15 ¥700★)", "-", "16:10"),
}


def _load_dedup() -> dict:
    if DEDUP_STATE.exists():
        try:
            return json.loads(DEDUP_STATE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_dedup(state: dict):
    DEDUP_STATE.parent.mkdir(parents=True, exist_ok=True)
    DEDUP_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                           encoding="utf-8")


def send(title: str, body: str, color: str, dedup_key: str,
         channel: str = "bets") -> bool:
    state = _load_dedup()
    if state.get(dedup_key):
        print(f"[skip dedup] {dedup_key}")
        return True
    ok = send_discord(title, body, color=color, channel=channel)
    if ok:
        state[dedup_key] = datetime.now().isoformat()
        _save_dedup(state)
    return ok


def build_race_body(scores: pd.DataFrame, daily: pd.DataFrame, rid: str) -> tuple[str, str]:
    course, rn, rname, grade, start = RACE_LABEL[rid]
    sub = scores[scores["race_id"].astype(str) == rid].sort_values("rank_in_race")
    if len(sub) == 0:
        return f"{course} R{rn} {rname}", "(score data 不在)"
    n = len(sub)
    dpred = daily[daily["race_id"].astype(str) == rid]
    v15_top1 = (dpred["top1_num"].iloc[0], dpred["top1_name"].iloc[0]) if len(dpred) else ("?", "?")
    trio_bets = dpred["trio_bets"].iloc[0] if len(dpred) else ""

    lines = [
        f"## 5/9 {course} R{rn} {rname} ({grade}) — 全馬総合スコア (動画代替★)",
        f"発走 {start}、 {n} 頭",
        f"V15 top1: {v15_top1[0]} {v15_top1[1]}",
        "",
        f"### Top 5 (統合スコア順、 重み: training 0.30 / IDM 0.25 / 激走 0.20 / 厩舎 0.15 / 人気 0.10)",
    ]
    for i, (_, r) in enumerate(sub.head(5).iterrows(), 1):
        stars = "⭐" * min(5, max(1, int(r["integrated_score"] * 5)))
        lines.append(
            f"{i}. **{r['umaban']} {r['horse_name']}** "
            f"score {r['integrated_score']:.3f} {stars}"
        )
        lines.append(
            f"   training {r.get('training_idx', 0):.1f} / "
            f"IDM {r.get('idm_score', 0):.1f} / "
            f"激走 {r.get('gekiso_idx', 0):.0f} / "
            f"厩舎 {r.get('stable_idx', 0):.1f}"
        )
    lines += [
        "",
        f"### 全馬 ranking (順位:馬番)",
        ", ".join(f"{r['rank_in_race']}:{r['umaban']}" for _, r in sub.iterrows()),
        "",
        f"### V15 三連複 7点 (verdict 比較用)",
        trio_bets[:120] + ("..." if len(trio_bets) > 120 else ""),
        "",
        f"※ TYB paddock_idx 13:00 後 取得可、 重み再調整で再 push 可",
    ]
    return f"5/9 {course} R{rn} {rname} ({grade}) 全馬総合スコア", "\n".join(lines)


def build_r12_body(scores: pd.DataFrame, daily: pd.DataFrame) -> tuple[str, str]:
    lines = [
        "## 5/9 12R クラス 全馬総合スコア (V15 投資対象 + 観戦)",
        "",
    ]
    for rid in RACE_GROUPS["r12_class"]:
        course, rn, rname, grade, start = RACE_LABEL[rid]
        sub = scores[scores["race_id"].astype(str) == rid].sort_values("rank_in_race")
        if len(sub) == 0: continue
        vote = "★ V15 ¥700 ★" if rid == "202604010312" else "× 案B改 除外 (2勝)"
        lines.append(f"### {course} R{rn} {rname} ({start}) — {vote}")
        for i, (_, r) in enumerate(sub.head(3).iterrows(), 1):
            lines.append(
                f"{i}. {r['umaban']} {r['horse_name']} score {r['integrated_score']:.3f}"
            )
        lines.append("")
    lines += [
        "★ V15 投票方針 (絶対遵守) ★",
        "- 新潟 12R 4歳以上1勝のみ ¥700 (案B改 strict)",
        "- 京都/東京 12R 2勝は **投票しない** (verdict / 観戦)",
        "- 累計 +13,530 円 死守",
    ]
    return "5/9 12R クラス 全馬総合スコア (V15 ¥700 + 観戦)", "\n".join(lines)


def build_completion_body() -> tuple[str, str]:
    body = """## Session #63 完了 (動画代替 全馬総合スコア)

- 動画 DL は server block 確定 (Session #62)
- 静止画 DL も server block 確定 (Session #63 B、 netkeiba HTTP 400 全 page)
- 代替: **JRDB 数値 features (KYI 90/90)** で全馬総合スコア + ranking 構築
- 全 R / 全馬 cover (重賞 3 + 12R 3 = 90 馬)

### features 内訳
- training_idx 0.30 / idm 0.25 / gekiso (paddock 代替) 0.20 / stable 0.15 / ninki 0.10
- 静止画 features (YOLOv8 body/pose/coat) は schema 完成、 5/16+ で復活時即運用可
- TYB paddock_idx は 13:00+ publish 後 retry で重み再調整可 (paddock 0.30 へ)

### 詳細
- data/v18/horse_total_evaluation_5_9.md (各 race Top5 + 全馬 + 妙味/警戒)
- data/v18/horse_global_ranking_5_9.md (race またぎ Top10)
- data/v18/horse_total_scores_5_9.csv

### 含意
- 5/9 重賞 + 12R 全馬 verdict 可 (ROI 「もし投票してたら」 計算可)
- Phase 4 動画 plan の **代替 完成**、 V20 (6/8) features 統合候補
- 9 月 V21 投入は 動画 dependency なしで可能

★ V15 投票方針 (絶対遵守) ★
- 新潟 12R 4歳以上1勝のみ ¥700 (案B改 strict)
- 11R 重賞 投票しない (verdict 用)
- 京都/東京 12R 2勝 投票しない (案B改 除外)
- 累計 +13,530円 死守
"""
    return "Session #63 完了 (動画代替 全馬総合スコア)", body


def main():
    scores = pd.read_csv(SCORES_CSV, dtype=str, keep_default_na=False)
    for c in ["paddock_idx", "training_idx", "idm_score", "stable_idx",
              "ninki_idx", "gekiso_idx", "weight_diff",
              "integrated_score", "rank_in_race"]:
        if c in scores.columns:
            scores[c] = pd.to_numeric(scores[c], errors="coerce")
    daily = pd.read_csv(DAILY_PRED, dtype=str)

    # 1. 京都新聞杯 G2
    title, body = build_race_body(scores, daily, "202608030511")
    send(title, body, "blue", "session63_kyoto_g2")

    # 2. エプソム G3
    title, body = build_race_body(scores, daily, "202605020511")
    send(title, body, "blue", "session63_epsom_g3")

    # 3. 駿風 S OP
    title, body = build_race_body(scores, daily, "202604010311")
    send(title, body, "blue", "session63_sprint_op")

    # 4. 12R クラス まとめ
    title, body = build_r12_body(scores, daily)
    send(title, body, "yellow", "session63_r12_class")

    # 5. 完了通知
    title, body = build_completion_body()
    send(title, body, "green", "session63_complete", channel="updates")

    print("\n=== Session #63 Discord 5 通 完了 ===")


if __name__ == "__main__":
    main()
