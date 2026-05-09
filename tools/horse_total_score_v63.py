"""Session #63 E: 全馬統合スコア + ranking 構築.

C (静止画 features) + D (JRDB 数値 features) を merge、 同 R 内 percentile 正規化、
重み付き和で integrated_score (0-1) と rank_in_race を算出。

5/9 当日は静止画 0 枚 (B 全 fail) のため、 数値 features (KYI) のみで scoring。
TYB 未publish (13:00+ 取得可能) のため paddock_idx は NaN、 gekiso_idx で代替。

usage:
  python tools/horse_total_score_v63.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

BASE = Path(r"C:/Users/takum/keiba-ai")
NUMERIC_CSV = BASE / "data" / "v18" / "horse_numeric_features_5_9.csv"
STATIC_CSV = BASE / "data" / "v18" / "horse_static_features_5_9.csv"
DAILY_PRED = BASE / "data" / "daily_predictions" / "20260509.csv"

OUT_CSV = BASE / "data" / "v18" / "horse_total_scores_5_9.csv"
OUT_DETAIL = BASE / "data" / "v18" / "horse_total_evaluation_5_9.md"
OUT_GLOBAL = BASE / "data" / "v18" / "horse_global_ranking_5_9.md"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 重み (TYB なし版 — 5/9 11:XX 時点)
WEIGHTS_NO_TYB = {
    "training_idx": 0.30,
    "idm_score":    0.25,
    "gekiso_idx":   0.20,
    "stable_idx":   0.15,
    "ninki_idx":    0.10,
}
# 重み (TYB あり版、 13:00+ 再実行時)
WEIGHTS_WITH_TYB = {
    "paddock_idx":  0.30,
    "training_idx": 0.20,
    "idm_score":    0.15,
    "gekiso_idx":   0.10,
    "stable_idx":   0.10,
    "ninki_idx":    0.05,
    "weight_diff":  0.10,
}

RACE_META = {
    "202608030511": {"course": "京都", "race_num": 11, "race_name": "京都新聞杯", "grade": "G2", "start": "15:30", "vote": "× verdict"},
    "202608030512": {"course": "京都", "race_num": 12, "race_name": "4歳以上2勝クラス", "grade": "-", "start": "16:00", "vote": "× 2勝 (案B改 除外)"},
    "202604010311": {"course": "新潟", "race_num": 11, "race_name": "駿風 S",       "grade": "OP", "start": "15:20", "vote": "× verdict"},
    "202604010312": {"course": "新潟", "race_num": 12, "race_name": "4歳以上1勝クラス", "grade": "-", "start": "16:10", "vote": "★ V15 ¥700 ★"},
    "202605020511": {"course": "東京", "race_num": 11, "race_name": "エプソムカップ", "grade": "G3", "start": "15:45", "vote": "× verdict"},
    "202605020512": {"course": "東京", "race_num": 12, "race_name": "4歳以上2勝クラス", "grade": "-", "start": "16:25", "vote": "× 2勝 (案B改 除外)"},
}


def percentile_normalize(series: pd.Series) -> pd.Series:
    """同 R 内 0-1 percentile 正規化 (rank / N、 値大きい方が良い)."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=series.index)  # 全 NaN は中央値 0.5
    rank = s.rank(method="min", na_option="bottom") - 1
    n = max(1, len(s) - 1)
    pct = rank / n
    pct = pct.where(s.notna(), 0.5)  # NaN は 0.5
    return pct


def main():
    if not NUMERIC_CSV.exists():
        print(f"[ERROR] {NUMERIC_CSV} not found — run horse_numeric_features.py first")
        sys.exit(1)

    num = pd.read_csv(NUMERIC_CSV, dtype=str, keep_default_na=False)
    daily = pd.read_csv(DAILY_PRED, dtype=str)

    # 数値変換
    for c in ["paddock_idx", "training_idx", "idm_score", "stable_idx",
              "ninki_idx", "gekiso_idx", "weight_diff", "tansho_odds"]:
        if c in num.columns:
            num[c] = pd.to_numeric(num[c], errors="coerce")

    # paddock_idx 取得状況で重み切替
    has_tyb = num["paddock_idx"].notna().sum() > 0
    weights = WEIGHTS_WITH_TYB if has_tyb else WEIGHTS_NO_TYB
    print(f"weights mode: {'WITH_TYB' if has_tyb else 'NO_TYB'} ({list(weights.keys())})")

    # 同 R percentile
    out_rows = []
    for rid, sub in num.groupby("race_id"):
        sub = sub.copy().sort_values("umaban")
        norm_cols = {}
        for feat in weights.keys():
            if feat in sub.columns:
                norm_cols[feat] = percentile_normalize(sub[feat])
            else:
                norm_cols[feat] = pd.Series(0.5, index=sub.index)

        score = pd.Series(0.0, index=sub.index)
        wsum = sum(weights.values())
        for feat, w in weights.items():
            score = score + (w / wsum) * norm_cols[feat]

        sub["integrated_score"] = score.values
        sub["rank_in_race"] = sub["integrated_score"].rank(ascending=False, method="min").astype(int)

        # confidence
        # 静止画 csv 空 → mid (数値のみ)
        sub["confidence"] = "mid" if not has_tyb else "high"

        out_rows.append(sub)

    result = pd.concat(out_rows, ignore_index=True)
    result = result.sort_values(["race_id", "rank_in_race"])

    # 出力 CSV
    cols_keep = ["race_id", "course", "race_num", "race_name", "umaban", "horse_name",
                 "paddock_idx", "training_idx", "idm_score", "stable_idx", "ninki_idx",
                 "gekiso_idx", "weight_diff", "tansho_odds",
                 "integrated_score", "rank_in_race", "confidence"]
    cols_keep = [c for c in cols_keep if c in result.columns]
    result[cols_keep].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"csv: {OUT_CSV.relative_to(BASE)} ({len(result)} rows)")

    # 詳細レポート (各 race ごと)
    daily_lookup = {str(r["race_id"]): r for _, r in daily.iterrows()}

    detail_lines = ["# Session #63 E: 5/9 全馬総合スコア + ranking (★動画 DL 不可 → 静止画+JRDB 代替★)",
                    "",
                    f"**作成**: 2026-05-09 11:XX (Session #63 E、 dev/training-poc)",
                    f"**重み mode**: {'WITH_TYB' if has_tyb else 'NO_TYB (13:00+ 再実行で WITH_TYB 化)'}",
                    f"**confidence**: {'high' if has_tyb else 'mid'} (全馬同 confidence)",
                    "",
                    "## 重み一覧",
                    ""]
    for feat, w in weights.items():
        detail_lines.append(f"- {feat}: {w}")
    detail_lines += ["", "---", ""]

    for rid in RACE_META:
        meta = RACE_META[rid]
        sub = result[result["race_id"] == rid].copy()
        if len(sub) == 0:
            continue
        n_horses = len(sub)
        dpred = daily_lookup.get(rid, {})
        v15_top1 = dpred.get("top1_num", "?")
        v15_top1_name = dpred.get("top1_name", "?")
        trio_bets = dpred.get("trio_bets", "")

        detail_lines.append(f"## 5/9 {meta['course']} R{meta['race_num']} {meta['race_name']} ({meta['grade']})")
        detail_lines.append(f"発走 {meta['start']}、 {n_horses} 頭、 投票方針: {meta['vote']}")
        detail_lines.append(f"V15 top1: {v15_top1} {v15_top1_name}")
        detail_lines.append("")
        detail_lines.append(f"### Top 5 (統合スコア順)")
        for i, (_, row) in enumerate(sub.head(5).iterrows(), 1):
            stars = "⭐" * min(5, max(1, int(row["integrated_score"] * 5)))
            detail_lines.append(
                f"{i}. {row['umaban']} {row['horse_name']}: "
                f"score={row['integrated_score']:.3f} {stars}"
            )
            paddock = row.get("paddock_idx", "")
            paddock_str = f"{paddock:.1f}" if pd.notna(paddock) else "n/a"
            detail_lines.append(
                f"   - パドック指数: {paddock_str} / "
                f"調教指数: {row.get('training_idx', 0):.1f} / "
                f"IDM: {row.get('idm_score', 0):.1f}"
            )
            detail_lines.append(
                f"   - 激走: {row.get('gekiso_idx', 0):.0f} / "
                f"厩舎: {row.get('stable_idx', 0):.1f} / "
                f"人気: {row.get('ninki_idx', 0):.0f} (信頼度: {row['confidence']})"
            )
        detail_lines.append("")

        # 全馬リスト
        detail_lines.append(f"### 全馬リスト ({n_horses} 頭)")
        detail_lines.append("| 順位 | 馬番 | 馬名 | 統合 | 調教 | IDM | 激走 | 厩舎 | 人気 |")
        detail_lines.append("|------|------|------|------|------|-----|------|------|------|")
        for _, row in sub.iterrows():
            detail_lines.append(
                f"| {row['rank_in_race']} | {row['umaban']} | {row['horse_name'][:14]} | "
                f"{row['integrated_score']:.3f} | "
                f"{row.get('training_idx', 0):.1f} | "
                f"{row.get('idm_score', 0):.1f} | "
                f"{row.get('gekiso_idx', 0):.0f} | "
                f"{row.get('stable_idx', 0):.1f} | "
                f"{row.get('ninki_idx', 0):.0f} |"
            )
        detail_lines.append("")

        # 妙味馬 / 凡走警戒
        sub_sorted_ninki = sub.sort_values("ninki_idx", ascending=False)
        if len(sub_sorted_ninki) >= 3:
            top_ninki = set(sub_sorted_ninki.head(3)["umaban"].astype(str).tolist())
            bot_ninki = set(sub_sorted_ninki.tail(5)["umaban"].astype(str).tolist())
            top_score = sub.head(5)
            myomi = [r for _, r in top_score.iterrows()
                     if str(r["umaban"]) in bot_ninki]
            warn = [r for _, r in sub_sorted_ninki.head(3).iterrows()
                    if r["rank_in_race"] > 5]
            detail_lines.append("### 妙味馬 / 凡走警戒")
            if myomi:
                for r in myomi:
                    detail_lines.append(
                        f"- 妙味: {r['umaban']} {r['horse_name']} "
                        f"(統合 rank={r['rank_in_race']}、 人気指数 低位)"
                    )
            if warn:
                for r in warn:
                    detail_lines.append(
                        f"- 警戒: {r['umaban']} {r['horse_name']} "
                        f"(人気指数 {r.get('ninki_idx', 0):.0f}、 統合 rank={r['rank_in_race']})"
                    )
            if not myomi and not warn:
                detail_lines.append("- (顕著な乖離なし)")
            detail_lines.append("")

        # 推奨買い目 (PoC、 投票しない)
        top3 = sub.head(3)["umaban"].astype(int).tolist()
        if len(top3) == 3:
            others = sub.iloc[3:6]["umaban"].astype(int).tolist()
            recos = []
            base = [top3[0]]
            second = [top3[1], top3[2]]
            third = [top3[1], top3[2]] + others[:3]
            third = list(dict.fromkeys(third))[:5]
            for s in second:
                for t in third:
                    bet = sorted({base[0], s, t})
                    if len(bet) == 3 and bet not in recos:
                        recos.append(bet)
            recos = recos[:7]
            detail_lines.append(f"### 推奨買い目 (PoC、 投票しない、 verdict 用)")
            detail_lines.append("- 三連複 7点 (統合 top1 軸 - top2,3 - top2-6):")
            for b in recos:
                detail_lines.append(f"  - {'-'.join(str(x) for x in b)}")
            detail_lines.append(f"- V15 三連複 7点: {trio_bets}")
            detail_lines.append("")

        detail_lines.append("---")
        detail_lines.append("")

    detail_lines.append("## V15 投資保護 確認")
    detail_lines.append("- 本 score は **PoC、 verdict 用、 投票しない**")
    detail_lines.append("- 5/9 投票: 新潟 12R 4歳以上1勝のみ ¥700 (案B改 strict、 V15 単独)")
    detail_lines.append("- 重賞 + 京都/東京 12R 2勝 投票しない (案B改 除外 / verdict)")

    OUT_DETAIL.write_text("\n".join(detail_lines), encoding="utf-8")
    print(f"detail: {OUT_DETAIL.relative_to(BASE)}")

    # Global ranking (race またぎ Top 10)
    glb = result.sort_values("integrated_score", ascending=False).head(10)
    g_lines = ["# Session #63 E: 5/9 全馬統合 ranking (race またぎ Top 10)",
               "",
               f"**作成**: 2026-05-09 11:XX (Session #63 E、 dev/training-poc)",
               f"**重み mode**: {'WITH_TYB' if has_tyb else 'NO_TYB'}",
               "",
               "## Top 10 (race またぎ、 同 R 内 percentile による正規化済)",
               "| Rank | race | umaban | horse_name | score | rank_in_race | training | IDM | gekiso |",
               "|------|------|--------|------------|-------|--------------|----------|-----|--------|"]
    for i, (_, r) in enumerate(glb.iterrows(), 1):
        meta = RACE_META.get(r["race_id"], {})
        race_label = f"{meta.get('course','?')}R{meta.get('race_num','?')}"
        g_lines.append(
            f"| {i} | {race_label} | {r['umaban']} | {r['horse_name'][:14]} | "
            f"{r['integrated_score']:.3f} | {r['rank_in_race']} | "
            f"{r.get('training_idx', 0):.1f} | "
            f"{r.get('idm_score', 0):.1f} | "
            f"{r.get('gekiso_idx', 0):.0f} |"
        )
    g_lines += ["", "## レース別 内訳"]
    for rid in RACE_META:
        meta = RACE_META[rid]
        sub = result[result["race_id"] == rid].head(3)
        if len(sub) == 0: continue
        g_lines.append(f"\n### {meta['course']} R{meta['race_num']} {meta['race_name']}")
        for _, r in sub.iterrows():
            g_lines.append(
                f"- {r['rank_in_race']}位: {r['umaban']} {r['horse_name']} "
                f"(score {r['integrated_score']:.3f})"
            )
    g_lines += ["",
                "## 統計",
                f"- 全 race: {len(RACE_META)}",
                f"- 全馬: {len(result)}",
                f"- 平均 integrated_score: {result['integrated_score'].mean():.3f}",
                f"- max integrated_score: {result['integrated_score'].max():.3f}",
                f"- min integrated_score: {result['integrated_score'].min():.3f}",
                "",
                "## V15 投票方針 (絶対遵守)",
                "- 新潟 12R 4歳以上1勝のみ ¥700 (案B改 strict)",
                "- 重賞 + 京都/東京 12R 2勝 投票しない (verdict / 観戦)"]
    OUT_GLOBAL.write_text("\n".join(g_lines), encoding="utf-8")
    print(f"global: {OUT_GLOBAL.relative_to(BASE)}")


if __name__ == "__main__":
    main()
