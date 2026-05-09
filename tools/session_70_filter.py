"""Session #70 B: 5月 11R/12R (重賞除外) 絞り込み.

LEAK 完全防止: V15 model.predict() を一切呼ばない。
data/cumulative_results.csv (5/2, 5/3) + data/daily_predictions/20260509.csv のみ
read-only で参照し、 11R/12R AND not stakes (G1/G2/G3) で filter。

出力: data/v18/session_70_filtered_races.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")

# G1/G2/G3 マーカー (race_name に含まれていれば確実)
STAKES_PATTERNS = [
    "G1", "G2", "G3",
    "GⅠ", "GⅡ", "GⅢ",
    "(G1)", "(G2)", "(G3)",
    "(GⅠ)", "(GⅡ)", "(GⅢ)",
    "GI ", "GII ", "GIII ",
]

# 5月の重賞 (race_name に grade marker が無いケースの manual list)
# 期間 5/2-5/9 の G1/G2/G3 のみを enumerate (リステッド L は含めない=keep する設計)
MAY_STAKES_NAMES = {
    # 5/2
    "京王杯SC",      # G2 (東京 11R)
    "京王杯スプリングカップ",
    "ユニコーンS",   # G3 (京都 11R)
    "ユニコーンステークス",
    # 5/3
    "天皇賞(春)",    # G1 (京都 11R)
    "天皇賞・春",
    "天皇賞春",
    # 5/9
    "京都新聞杯",    # G2 (京都 11R)
    "エプソムC",     # G3 (東京 11R)
    "エプソムカップ",
}


def is_stakes(race_name: str) -> bool:
    s = str(race_name)
    if any(p in s for p in STAKES_PATTERNS):
        return True
    return any(name in s for name in MAY_STAKES_NAMES)


def load_cumulative_may() -> pd.DataFrame:
    df = pd.read_csv(BASE / "data" / "cumulative_results.csv", encoding="utf-8-sig")
    df["date"] = df["date"].astype(str).str.split(".").str[0]  # "20260502.0" -> "20260502"
    df = df[df["date"].astype(str).str.startswith("202605")].copy()
    df = df[df["race_num"].isin([11, 12])].copy()
    df["source"] = "production_saved_score (cumulative_results.csv)"
    # column 名を daily_predictions に揃える
    keep = [
        "date", "race_id", "course", "race_num", "race_name",
        "num_horses", "distance", "surface", "condition", "track_condition",
        "top1_num", "top1_name", "top1_score", "top2_num", "top3_num",
        "top1_finish", "top2_finish", "top3_finish",
        "trio_bets", "trio_result", "trio_hit", "trio_payout",
        "actual_payout", "investment", "profit",
        "source",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def load_daily_predictions_5_9() -> pd.DataFrame:
    df = pd.read_csv(BASE / "data" / "daily_predictions" / "20260509.csv", dtype=str)
    df = df[df["race_num"].astype(int).isin([11, 12])].copy()
    df["date"] = "20260509"
    df["source"] = "production_saved_score (daily_predictions/20260509.csv)"
    # numeric 化 (top1_score)
    df["top1_score"] = pd.to_numeric(df["top1_score"], errors="coerce")
    return df


def main():
    cum = load_cumulative_may()
    dp9 = load_daily_predictions_5_9()

    print(f"cumulative (5/2, 5/3) 11R/12R: {len(cum)}")
    print(f"daily_predictions (5/9) 11R/12R: {len(dp9)}")

    # 重賞 filter
    cum_kept = cum[~cum["race_name"].apply(is_stakes)].copy()
    dp9_kept = dp9[~dp9["race_name"].apply(is_stakes)].copy()

    print(f"\n重賞除外後:")
    print(f"  cumulative: {len(cum_kept)} (除外 {len(cum) - len(cum_kept)} 件)")
    print(f"  daily_predictions: {len(dp9_kept)} (除外 {len(dp9) - len(dp9_kept)} 件)")

    # 除外 race を doc 用に表示
    excluded_cum = cum[cum["race_name"].apply(is_stakes)][["date", "course", "race_num", "race_name"]]
    excluded_dp9 = dp9[dp9["race_name"].apply(is_stakes)][["date", "course", "race_num", "race_name"]]
    print("\n除外された重賞:")
    for _, r in pd.concat([excluded_cum, excluded_dp9]).iterrows():
        print(f"  {r['date']} {r['course']} R{r['race_num']} {r['race_name']}")

    # 統合 (column union)
    combined = pd.concat([cum_kept, dp9_kept], ignore_index=True, sort=False)
    out_path = BASE / "data" / "v18" / "session_70_filtered_races.csv"
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nwritten: {out_path.relative_to(BASE)} ({len(combined)} R)")
    print(f"\n--- preview ---")
    print(combined[["date", "course", "race_num", "race_name", "num_horses",
                    "top1_num", "top1_name", "top1_score", "source"]].to_string(index=False))


if __name__ == "__main__":
    main()
