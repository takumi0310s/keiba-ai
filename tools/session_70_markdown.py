"""Session #70 C: 5月 11R/12R 重賞除外 全頭スコア markdown.

source は data/v18/session_70_filtered_races.csv (B 出力) のみ。
LEAK 防止: V15 model.predict() 不使用。
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")


WEEKDAY_JP = ["月", "火", "水", "木", "金", "土", "日"]


def class_label(race_name: str) -> str:
    s = str(race_name)
    if "1勝" in s: return "1勝"
    if "2勝" in s: return "2勝"
    if "3勝" in s: return "3勝"
    if "未勝利" in s or "新馬" in s: return "未勝利/新馬"
    if "(L)" in s or "リステッド" in s: return "L"
    if "(GⅠ)" in s or "G1" in s: return "G1"  # filter で弾けてるはずだが念のため
    if "(GⅡ)" in s or "G2" in s: return "G2"
    if "(GⅢ)" in s or "G3" in s: return "G3"
    return "OP/特別"


def fmt_date(date_str: str) -> str:
    s = str(date_str)
    y = int(s[:4]); m = int(s[4:6]); d = int(s[6:8])
    import datetime
    wd = WEEKDAY_JP[datetime.date(y, m, d).weekday()]
    return f"{m}/{d} ({wd})"


def main():
    df = pd.read_csv(BASE / "data" / "v18" / "session_70_filtered_races.csv",
                     encoding="utf-8-sig")
    df["date"] = df["date"].astype(str).str.replace(".0", "", regex=False)

    out = []
    out.append("# 5月 11R/12R (重賞除外) 全頭 V15 production saved score 一覧")
    out.append("")
    out.append("**期間**: 2026-05-01 〜 2026-05-09")
    out.append(f"**対象 R**: {len(df)} 件")
    out.append("**重賞除外**: G1 (天皇賞春) / G2 (京王杯SC, 京都新聞杯) / G3 (ユニコーンS, エプソムC) — 5 件")
    out.append("**source**: production_saved_score (data/cumulative_results.csv 5/2-5/3, data/daily_predictions/20260509.csv 5/9)")
    out.append("**🚨 LEAK 防止**: V15 model.predict() 不使用、 predict_core / daily_predict 一切実行せず、 read-only access のみ")
    out.append("")
    out.append("---")
    out.append("")
    out.append("## ⚠ source 制約")
    out.append("")
    out.append("- **5/2, 5/3** の `data/cumulative_results.csv` は CLAUDE.md 既知問題により `top1_num/score` 列が NaN (95%欠損)。 `top1_finish` (V15 top1 馬の実際の着順) と `trio_bets / trio_payout / profit` は populated。 **score 値 そのものは production save 失敗** → 着順 + 投票結果 のみ記載可能。")
    out.append("- **5/9** は `data/daily_predictions/20260509.csv` に top1/top2/top3 の score + 馬番 + trio_bets が production saved。 4 着以下の馬の score は production csv に未保存 (full v15_scores JSON は別 branch dev/training-poc にあるが本 session では干渉防止のため参照しない)。")
    out.append("- → 「全頭スコア」 は **production saved の範囲では top1/top2/top3 が最大**。 4 着以下は本 audit では空欄。")
    out.append("")
    out.append("---")
    out.append("")

    # group by date
    for date_str in sorted(df["date"].unique()):
        out.append(f"## {fmt_date(date_str)} の 11R/12R (重賞除外)")
        out.append("")
        sub = df[df["date"] == date_str].sort_values(["course", "race_num"])
        for _, r in sub.iterrows():
            course = r["course"]
            rn = int(r["race_num"])
            rname = r["race_name"]
            cls = class_label(rname)
            out.append(f"### {course} {rn}R {rname} ({cls})")

            # メタ
            num_horses = r.get("num_horses")
            distance = r.get("distance")
            surface = r.get("surface")
            condition = r.get("condition") if not pd.isna(r.get("condition")) else r.get("track_condition", "")
            meta = []
            if not pd.isna(num_horses): meta.append(f"出走 {int(float(num_horses))} 頭")
            if not pd.isna(distance): meta.append(f"{surface} {int(float(distance))}m" if not pd.isna(surface) else f"{int(float(distance))}m")
            if not pd.isna(condition) and str(condition).strip(): meta.append(f"馬場 {condition}")
            if meta:
                out.append("- " + " / ".join(meta))

            # V15 production saved score (top1/2/3)
            out.append("")
            out.append("#### V15 production saved score (★ no leak、 production csv 由来 ★)")
            out.append("")
            top1n = r.get("top1_num")
            top1name = r.get("top1_name")
            top1s = r.get("top1_score")
            top2n = r.get("top2_num")
            top2name = r.get("top2_name") if "top2_name" in r else None
            top3n = r.get("top3_num")
            top3name = r.get("top3_name") if "top3_name" in r else None

            if pd.isna(top1s):
                out.append("> ⚠ cumulative_results.csv の top1_num/score 欠損 (CLAUDE.md 既知)。 production saved score 値は不明。")
                out.append("> trio_bets / trio_result / payout / profit は populated。")
            else:
                out.append("| 順位 | 馬番 | 馬名 | V15 score |")
                out.append("|---|---|---|---|")
                if not pd.isna(top1n):
                    out.append(f"| 1 | {int(float(top1n))} | {top1name or '-'} | {float(top1s):.4f} |")
                if not pd.isna(top2n):
                    s2 = "-"
                    out.append(f"| 2 | {int(float(top2n))} | {top2name or '-'} | {s2} |")
                if not pd.isna(top3n):
                    s3 = "-"
                    out.append(f"| 3 | {int(float(top3n))} | {top3name or '-'} | {s3} |")
                out.append("")
                out.append("> 注: top2/top3 の score は daily_predictions csv に列がないため `-`。 馬番 + 馬名は populated。")

            # 結果 (5/2-5/3 cumulative)
            out.append("")
            out.append("#### 実結果 / V15 投票結果 (production)")
            out.append("")
            top1_finish = r.get("top1_finish")
            top2_finish = r.get("top2_finish")
            top3_finish = r.get("top3_finish")
            trio_result = r.get("trio_result")
            trio_hit = r.get("trio_hit")
            trio_payout = r.get("trio_payout")
            actual_payout = r.get("actual_payout")
            profit = r.get("profit")
            investment = r.get("investment")
            trio_bets = r.get("trio_bets")

            if not pd.isna(top1_finish):
                out.append(f"- V15 top1 → {int(float(top1_finish))} 着")
            if not pd.isna(top2_finish):
                out.append(f"- V15 top2 → {int(float(top2_finish))} 着")
            if not pd.isna(top3_finish):
                out.append(f"- V15 top3 → {int(float(top3_finish))} 着")
            if not pd.isna(trio_result):
                out.append(f"- 1-2-3 着 (trio_result): `{trio_result}`")
            if not pd.isna(trio_bets):
                out.append(f"- V15 trio_bets: `{trio_bets}`")
            if not pd.isna(trio_hit):
                hit_str = "✅ HIT" if float(trio_hit) > 0 else "❌ miss"
                out.append(f"- trio 判定: {hit_str}")
            if not pd.isna(trio_payout):
                out.append(f"- trio_payout: ¥{int(float(trio_payout)):,}")
            if not pd.isna(actual_payout) and not pd.isna(investment):
                out.append(f"- 投資 ¥{int(float(investment)):,} → 払戻 ¥{int(float(actual_payout)):,}")
            if not pd.isna(profit):
                p = int(float(profit))
                sign = "+" if p >= 0 else ""
                out.append(f"- 損益: {sign}¥{p:,}")

            # 5/9 (daily_predictions) で trio_bets / profit が NaN なら 5/9 投票対象判定
            if pd.isna(trio_hit) and date_str == "20260509":
                if rn == 12 and "1勝" in rname:
                    out.append(f"- ★ 5/9 V15 案B改 投票対象 ★")
                    out.append(f"  - 軸: {int(float(top1n))} {top1name}")
                    out.append(f"  - trio_bets (saved): `{trio_bets}`")
                    out.append(f"  - 結果: 11 ハイクオリティ → 3着、 三連複 7点 全 miss、 損益 -¥700 (Session #67 確定)")
                else:
                    out.append("- 案B改 strict 除外 (12R 1勝以外、 投票なし)")

            out.append("")
        out.append("---")
        out.append("")

    out_path = BASE / "data" / "v18" / "may_filtered_horse_scores.md"
    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"written: {out_path.relative_to(BASE)}")
    print(f"size: {out_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
