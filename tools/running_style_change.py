"""脚色変化 features (Session #47 C、 dev/sprint2).

過去 5 R の 4 角通過順位 (pass4) 系列から 脚色変化 pattern 検出。

脚色定義 (4 角通過順位 / 出走頭数):
- 逃げ: pass4 == 1 / 1 のみ
- 先行: pass4 ∈ [2, 4] / 1-4 (上位 30%)
- 差し: pass4 ∈ [5, 12] / 中位
- 追込: pass4 上位 70% 以下

features:
- running_style_current (推定脚色 0-3)
- style_change_pattern_5r (過去 5R の最頻 + 直近の 差分)
- style_jockey_change_corr (騎手乗替時の脚色変化)

V15 production 完全独立、 dev/sprint2 のみ。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")


def style_from_pass4(pass4: int, num_horses: int = 16) -> int:
    """4 角通過順位から脚色 code (0=逃げ, 1=先行, 2=差し, 3=追込)."""
    if pass4 == 1:
        return 0  # 逃げ
    ratio = pass4 / max(num_horses, 1)
    if ratio <= 0.3:
        return 1  # 先行
    if ratio <= 0.7:
        return 2  # 差し
    return 3  # 追込


def compute_style_features(history_df: pd.DataFrame, horse_id: str,
                           current_jockey: str = None) -> dict:
    """過去 5 R の 脚色 sequence + 変化 pattern."""
    h = history_df[history_df["horse_id"].astype(str) == str(horse_id)].copy()
    h = h.sort_values("date_s", ascending=False).head(5)

    out = {
        "running_style_recent_5r_mode": -1,
        "running_style_recent_3r_mean": -1.0,
        "style_change_count": 0,
        "style_jockey_change_match": 0,
        "history_n": len(h),
    }

    if len(h) == 0:
        return out

    h["pass4_num"] = pd.to_numeric(h.get("pass4"), errors="coerce")
    h["num_horses_num"] = pd.to_numeric(h.get("num_horses"), errors="coerce").fillna(16)
    valid = h.dropna(subset=["pass4_num"])
    if len(valid) == 0:
        return out

    styles = valid.apply(lambda r: style_from_pass4(int(r["pass4_num"]),
                                                     int(r["num_horses_num"])), axis=1).tolist()

    out["history_n"] = len(styles)
    if not styles:
        return out

    # mode (最頻)
    from collections import Counter
    mode_counter = Counter(styles)
    out["running_style_recent_5r_mode"] = mode_counter.most_common(1)[0][0]

    # mean (直近 3 走)
    if len(styles) >= 3:
        out["running_style_recent_3r_mean"] = round(np.mean(styles[:3]), 2)
    elif len(styles) >= 1:
        out["running_style_recent_3r_mean"] = round(np.mean(styles), 2)

    # 変化 count (直近 5 走で異なる style 数)
    out["style_change_count"] = len(set(styles))

    # 騎手変更との相関 (簡易: 騎手列があれば)
    if current_jockey and "jockey" in h.columns and len(valid) > 0:
        prev_jockey = valid.iloc[0].get("jockey")
        if prev_jockey != current_jockey and len(styles) >= 2:
            # 騎手変わった → 脚色変化があるか
            if styles[0] != styles[1]:
                out["style_jockey_change_match"] = 1

    return out


def backtest_distribution() -> dict:
    """jra_races_full から 脚色分布 + top3 率 を確認."""
    p = BASE / "data" / "jra_races_full.csv"
    if not p.exists():
        return {"available": False}

    df = pd.read_csv(p, usecols=["pass4", "num_horses", "finish"], low_memory=False)
    df["pass4_num"] = pd.to_numeric(df["pass4"], errors="coerce")
    df["num_horses_num"] = pd.to_numeric(df["num_horses"], errors="coerce")
    df["finish_num"] = pd.to_numeric(df["finish"], errors="coerce")
    df = df.dropna(subset=["pass4_num", "num_horses_num", "finish_num"])
    df = df[df["pass4_num"] > 0]
    df["style"] = df.apply(lambda r: style_from_pass4(int(r["pass4_num"]),
                                                       int(r["num_horses_num"])), axis=1)
    df["top3"] = (df["finish_num"] <= 3).astype(int)

    by_style = {}
    style_names = ["逃げ", "先行", "差し", "追込"]
    for code, name in enumerate(style_names):
        sub = df[df["style"] == code]
        by_style[name] = {
            "code": code,
            "n": len(sub),
            "top3_rate": round(sub["top3"].mean() * 100, 2) if len(sub) > 0 else 0,
        }

    return {
        "available": True,
        "n_total": int(len(df)),
        "by_style": by_style,
    }


def cli():
    p = argparse.ArgumentParser(description="running_style_change (Session #47 C)")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--out", default="data/v18/sprint2_running_style_backtest.json")
    args = p.parse_args()

    if args.backtest:
        result = backtest_distribution()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        out_path = BASE / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  written: {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    cli()
