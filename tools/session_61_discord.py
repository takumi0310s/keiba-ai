"""Session #61 D: 全馬 動画スコア + レース名 を Discord 別通知 (updates channel)。
Session #60 とは別 message として送信。
"""
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from notify import send_discord  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V18 = ROOT / "data" / "v18"
df = pd.read_csv(V18 / "horse_video_scores_5_9.csv")

# 各 race ごと description に詰める (1 message)
parts = []
for rid, g in df.groupby("race_id", sort=False):
    venue = g["venue"].iloc[0]
    rno = int(g["race_no"].iloc[0])
    rname = g["race_name"].iloc[0]
    grade = g["race_grade"].iloc[0]
    start = g["race_start_time"].iloc[0]
    parts.append(f"**{venue} {rno}R {rname} [{grade}] {start}**")
    show = g.head(5) if g["v15_pct"].notna().any() else g
    for _, r in show.iterrows():
        v15 = f"{r['v15_score']:.3f}" if pd.notna(r["v15_score"]) else "—"
        pct = f"{r['v15_pct']:.2f}" if pd.notna(r["v15_pct"]) else "—"
        parts.append(f"  {int(r['rank_in_race'])}. [{int(r['umaban'])}] {r['horse_name']} ({v15}/p{pct})")
    parts.append("")

# 統合 top10
top = df[df["v15_pct"].notna()].sort_values("v15_pct", ascending=False).head(10)
parts.append("**統合 ranking top10 (V15 percentile)**")
for i, (_, r) in enumerate(top.iterrows(), 1):
    parts.append(
        f"  {i}. {r['venue']}{int(r['race_no'])}R [{int(r['umaban'])}]{r['horse_name']} "
        f"({r['race_name']}) p{r['v15_pct']:.2f}"
    )

intro = (
    "重賞 3R × 全馬 (35 頭) 解析 + 出走レース名。\n"
    "動画 DL は #60 B で全失敗、 motion 3 頭は simulate 代表値。\n"
    "全馬 score = V15 percentile ベース。詳細: horse_video_scores_5_9.csv\n\n"
)
msg = intro + "\n".join(parts)
msg = msg[:1990]
title = "Session #61: 5/9 重賞 全馬 動画スコア (別通知)"

ok = send_discord(title, msg, color="blue", channel="updates")
print("Discord:", "OK" if ok else "SKIP/FAIL")
