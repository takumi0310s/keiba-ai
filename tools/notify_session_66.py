"""Session #66 C: Discord 5 通 (paddock 統合版 全馬総合スコア).

Session #63 F (v63) の Discord 5 通 を paddock 統合版 (v66) で再送。
dedup_key を session66_* で 重複防止 (Session #63 と区別)。

usage:
  python tools/notify_session_66.py
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
SCORES_CSV = BASE / "data" / "v18" / "horse_total_scores_5_9_v66.csv"
DAILY_PRED = BASE / "data" / "daily_predictions" / "20260509.csv"

RACE_LABEL = {
    "202608030511": ("京都", 11, "京都新聞杯", "G2", "15:30"),
    "202608030512": ("京都", 12, "4歳以上2勝クラス", "-", "16:00"),
    "202605020511": ("東京", 11, "エプソムカップ", "G3", "15:45"),
    "202605020512": ("東京", 12, "4歳以上2勝クラス", "-", "16:25"),
    "202604010311": ("新潟", 11, "駿風 S", "OP", "15:20"),
    "202604010312": ("新潟", 12, "4歳以上1勝クラス (★V15 ¥700★)", "-", "16:10"),
}

R12_RACES = ["202608030512", "202604010312", "202605020512"]


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
        print(f"[sent] {dedup_key}")
    else:
        print(f"[fail] {dedup_key}")
    return ok


def detect_mode(scores: pd.DataFrame) -> tuple[bool, int, str, str]:
    """returns (has_tyb, n_tyb, weights_str, mode_label)."""
    paddock = pd.to_numeric(scores.get("paddock_idx"), errors="coerce")
    n_tyb = int(paddock.notna().sum())
    has_tyb = n_tyb > 0
    if has_tyb:
        ws = "paddock 0.30 / training 0.20 / IDM 0.20 / 激走 0.15 / 厩舎 0.10 / 人気 0.05"
        label = "WITH_TYB (paddock 統合)"
    else:
        ws = "training 0.30 / IDM 0.25 / 激走 0.20 / 厩舎 0.15 / 人気 0.10"
        label = "NO_TYB fallback"
    return has_tyb, n_tyb, ws, label


def build_race_body(scores: pd.DataFrame, daily: pd.DataFrame, rid: str,
                    has_tyb: bool, weights_str: str) -> tuple[str, str]:
    course, rn, rname, grade, start = RACE_LABEL[rid]
    sub = scores[scores["race_id"].astype(str) == rid].sort_values("rank_in_race")
    if len(sub) == 0:
        return f"{course} R{rn} {rname}", "(score data 不在)"
    n = len(sub)
    dpred = daily[daily["race_id"].astype(str) == rid]
    v15_top1 = (dpred["top1_num"].iloc[0], dpred["top1_name"].iloc[0]) if len(dpred) else ("?", "?")
    trio_bets = dpred["trio_bets"].iloc[0] if len(dpred) else ""

    badge = "★paddock 統合版★" if has_tyb else "★NO_TYB fallback★"
    lines = [
        f"## 5/9 {course} R{rn} {rname} ({grade}) — 全馬総合スコア ({badge})",
        f"発走 {start}、 {n} 頭",
        f"V15 top1: {v15_top1[0]} {v15_top1[1]}",
        "",
        f"### Top 5 (統合スコア順、 重み: {weights_str})",
    ]
    for i, (_, r) in enumerate(sub.head(5).iterrows(), 1):
        stars = "⭐" * min(5, max(1, int(r["integrated_score"] * 5)))
        lines.append(
            f"{i}. **{r['umaban']} {r['horse_name']}** "
            f"score {r['integrated_score']:.3f} {stars}"
        )
        if has_tyb:
            paddock = pd.to_numeric(r.get("paddock_idx"), errors="coerce")
            paddock_str = f"{paddock:.1f}" if pd.notna(paddock) else "n/a"
            lines.append(
                f"   パドック {paddock_str} / "
                f"training {r.get('training_idx', 0):.1f} / "
                f"IDM {r.get('idm_score', 0):.1f} / "
                f"激走 {r.get('gekiso_idx', 0):.0f}"
            )
        else:
            lines.append(
                f"   training {r.get('training_idx', 0):.1f} / "
                f"IDM {r.get('idm_score', 0):.1f} / "
                f"激走 {r.get('gekiso_idx', 0):.0f} / "
                f"厩舎 {r.get('stable_idx', 0):.1f}"
            )
    lines += [
        "",
        "### 全馬 ranking (順位:馬番)",
        ", ".join(f"{r['rank_in_race']}:{r['umaban']}" for _, r in sub.iterrows()),
        "",
        f"### V15 三連複 7点 (verdict 比較用)",
        trio_bets[:120] + ("..." if len(trio_bets) > 120 else ""),
    ]
    return f"5/9 {course} R{rn} {rname} ({grade}) 全馬総合スコア (v66)", "\n".join(lines)


def build_r12_body(scores: pd.DataFrame, daily: pd.DataFrame,
                   has_tyb: bool, weights_str: str) -> tuple[str, str]:
    badge = "★paddock 統合版★" if has_tyb else "★NO_TYB fallback★"
    lines = [
        f"## 5/9 12R クラス 全馬総合スコア (V15 投資対象 + 観戦) ({badge})",
        f"重み: {weights_str}",
        "",
    ]
    for rid in R12_RACES:
        course, rn, rname, grade, start = RACE_LABEL[rid]
        sub = scores[scores["race_id"].astype(str) == rid].sort_values("rank_in_race")
        if len(sub) == 0:
            continue
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
    return "5/9 12R クラス 全馬総合スコア (v66、 V15 ¥700 + 観戦)", "\n".join(lines)


def build_completion_body(has_tyb: bool, n_tyb: int, weights_str: str,
                          mode_label: str) -> tuple[str, str]:
    body = f"""## Session #66 完了 (★paddock 統合版★)

- TYB 取得: {'OK ' + str(n_tyb) + ' 馬' if has_tyb else 'NO (publish 待ち)、 NO_TYB fallback 維持'}
- 重み mode: {mode_label}
- 重み: {weights_str}
- Session #63 v63 (NO_TYB) → v66 ({'WITH_TYB' if has_tyb else 'fallback'}) 切替
- 全 R / 全馬 cover (重賞 3 + 12R 3 = 90 馬)

### 出力
- data/v18/horse_total_scores_5_9_v66.csv
- data/v18/horse_total_evaluation_5_9_v66.md
- data/v18/horse_global_ranking_5_9_v66.md

### 含意
- {('paddock 主軸 0.30 で TYB feed をフル活用、 verdict 精度 向上' if has_tyb else 'TYB publish 後 再 push で paddock 統合 完成')}
- V15 投資保護 完全 (本 score は verdict 用、 投票しない)
- V20 / V21 features 統合 候補 確定

★ V15 投票方針 (絶対遵守) ★
- 新潟 12R 4歳以上1勝のみ ¥700 (案B改 strict、 V15 単独)
- 11R 重賞 投票しない (verdict 用)
- 京都/東京 12R 2勝 投票しない (案B改 除外)
- 累計 +13,530円 死守
"""
    return f"Session #66 完了 ({'paddock 統合' if has_tyb else 'NO_TYB fallback 維持'})", body


def main():
    if not SCORES_CSV.exists():
        print(f"[ERROR] {SCORES_CSV} not found — run horse_total_score_v66.py first")
        sys.exit(1)

    scores = pd.read_csv(SCORES_CSV, dtype=str, keep_default_na=False)
    for c in ["paddock_idx", "training_idx", "idm_score", "stable_idx",
              "ninki_idx", "gekiso_idx", "weight_diff",
              "integrated_score", "rank_in_race"]:
        if c in scores.columns:
            scores[c] = pd.to_numeric(scores[c], errors="coerce")
    daily = pd.read_csv(DAILY_PRED, dtype=str)

    has_tyb, n_tyb, weights_str, mode_label = detect_mode(scores)
    print(f"mode: {mode_label} ({n_tyb} 馬 TYB あり)")

    suffix = "_tyb" if has_tyb else "_notyb"

    # 1. 京都新聞杯 G2
    title, body = build_race_body(scores, daily, "202608030511", has_tyb, weights_str)
    send(title, body, "blue", f"session66_kyoto_g2{suffix}")

    # 2. エプソム G3
    title, body = build_race_body(scores, daily, "202605020511", has_tyb, weights_str)
    send(title, body, "blue", f"session66_epsom_g3{suffix}")

    # 3. 駿風 S OP
    title, body = build_race_body(scores, daily, "202604010311", has_tyb, weights_str)
    send(title, body, "blue", f"session66_sprint_op{suffix}")

    # 4. 12R クラス
    title, body = build_r12_body(scores, daily, has_tyb, weights_str)
    send(title, body, "yellow", f"session66_r12_class{suffix}")

    # 5. 完了通知
    title, body = build_completion_body(has_tyb, n_tyb, weights_str, mode_label)
    send(title, body, "green", f"session66_complete{suffix}", channel="updates")

    print("\n=== Session #66 Discord 5 通 完了 ===")


if __name__ == "__main__":
    main()
