"""P0-2 拡張 戦略⑦ 全 worst 除外設計 分析スクリプト (read-only)

Goal:
- 場 (8) x 条件 (6) ROI grid
- Welch t-test + bootstrap 95% CI per cell
- 4 strategies (A/B/C/D) comparison
- オッズ帯 x 場 x 条件 interaction
- 推奨案 + 5/18+ 実装 prompt

Output: stdout JSON for embedding in markdown
Source: data/cumulative_results.csv (status=settled)
       race_id 4-6桁 で 場 code 復元 (mojibake 列回避)

NEVER modifies any file other than docs/P0_2_EXTENSION_DESIGN_2026_05_16.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("C:/Users/takum/keiba-ai")
CSV = ROOT / "data/cumulative_results.csv"

# JRA 場 code (race_id 5-6 桁)
JRA_COURSE_MAP = {
    "01": "Sapporo",
    "02": "Hakodate",
    "03": "Fukushima",
    "04": "Niigata",
    "05": "Tokyo",
    "06": "Nakayama",
    "07": "Chukyo",
    "08": "Kyoto",
    "09": "Hanshin",
    "10": "Kokura",
}

# race_id format: YYYYCCKKKKNN (12桁) OR YYYYCCXXNN (10桁)
# CC = course code at positions [4:6]


def load_data():
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    df = df[df["status"] == "settled"].copy()
    # restrict to JRA (drop NAR rows whose course code is not in JRA_COURSE_MAP)
    df["race_id_str"] = df["race_id"].astype(str)
    df["course_code"] = df["race_id_str"].str[4:6]
    df["course_en"] = df["course_code"].map(JRA_COURSE_MAP)
    df_jra = df[df["course_en"].notna()].copy()
    # ensure numeric
    for col in ["investment", "actual_payout", "trio_hit", "umaren_hit"]:
        if col in df_jra.columns:
            df_jra[col] = pd.to_numeric(df_jra[col], errors="coerce")
    df_jra["pnl"] = df_jra["actual_payout"].fillna(0) - df_jra["investment"].fillna(0)
    df_jra["hit"] = (df_jra["actual_payout"].fillna(0) > 0).astype(int)
    # condition cleanup: drop NAR_X (already excluded via course filter)
    df_jra["cond_clean"] = df_jra["condition"].astype(str).str.strip()
    return df_jra


def bootstrap_roi_ci(returns_per_yen, n_iter=10000, seed=42):
    """returns_per_yen = payout / investment per race.
    ROI = mean(payout) / mean(inv) * 100"""
    if len(returns_per_yen) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(returns_per_yen)
    idx = rng.integers(0, n, size=(n_iter, n))
    sample = returns_per_yen[idx]  # n_iter x n
    means = sample.mean(axis=1) * 100
    return (np.percentile(means, 2.5), np.percentile(means, 97.5))


def welch_t_test_one_sample(values, mu0):
    """Test H0: mean(values) == mu0 (Welch-like; here only one sample so std-error t-test)"""
    from scipy import stats

    if len(values) < 3:
        return (np.nan, np.nan, np.nan)
    t_stat, p_val = stats.ttest_1samp(values, mu0)
    return (float(t_stat), float(p_val), float(values.mean()))


def cell_stats(sub, baseline_per_yen=1.0133):
    """sub: DataFrame for a single cell. Returns dict of metrics.
    baseline_per_yen = 1.0133 (ROI 101.33%)
    """
    n = len(sub)
    inv = float(sub["investment"].sum())
    pay = float(sub["actual_payout"].sum())
    pnl = pay - inv
    roi_pct = (pay / inv * 100) if inv > 0 else np.nan
    hit_pct = float(sub["hit"].mean() * 100) if n > 0 else np.nan
    # per-race ratio (payout/investment); investment is 700 for all
    rpy = (sub["actual_payout"].fillna(0) / sub["investment"].replace(0, np.nan)).dropna().values
    ci_low, ci_high = bootstrap_roi_ci(rpy) if len(rpy) >= 3 else (np.nan, np.nan)
    t_stat, p_val, mean_rpy = welch_t_test_one_sample(rpy, baseline_per_yen) if len(rpy) >= 3 else (np.nan, np.nan, np.nan)
    return {
        "n": n,
        "inv": inv,
        "pay": pay,
        "pnl": pnl,
        "roi_pct": roi_pct,
        "hit_pct": hit_pct,
        "ci_low_pct": ci_low,
        "ci_high_pct": ci_high,
        "t_stat": t_stat,
        "p_val": p_val,
    }


def build_grid(df):
    courses = ["Tokyo", "Nakayama", "Hanshin", "Kyoto", "Chukyo", "Fukushima", "Niigata", "Kokura"]
    conditions = ["A", "B", "C", "D", "E", "X"]
    rows = []
    for c in courses:
        for cond in conditions:
            sub = df[(df["course_en"] == c) & (df["cond_clean"] == cond)]
            s = cell_stats(sub)
            rows.append({"course": c, "condition": cond, **s})
    return pd.DataFrame(rows)


def project_strategy(df, exclude_courses=None, exclude_conds=None):
    """Project ROI/PnL when excluding given courses/conditions."""
    keep = df.copy()
    if exclude_courses:
        keep = keep[~keep["course_en"].isin(exclude_courses)]
    if exclude_conds:
        keep = keep[~keep["cond_clean"].isin(exclude_conds)]
    excl = df[~df.index.isin(keep.index)]
    return {
        "n_keep": len(keep),
        "n_excl": len(excl),
        "inv_keep": float(keep["investment"].sum()),
        "pay_keep": float(keep["actual_payout"].sum()),
        "pnl_keep": float((keep["actual_payout"] - keep["investment"]).sum()),
        "roi_keep_pct": float(keep["actual_payout"].sum() / keep["investment"].sum() * 100) if keep["investment"].sum() > 0 else np.nan,
        "inv_excl": float(excl["investment"].sum()),
        "pay_excl": float(excl["actual_payout"].sum()),
        "pnl_excl": float((excl["actual_payout"] - excl["investment"]).sum()),
    }


def odds_band_interaction(df):
    """top1_score の代用として、 actual_payout/700 = oikiri 配当倍 を proxy として odds band 化。
    実 odds がない (top1_odds 欠) ため、 hit 時の payout/700 で配当帯を作成。
    miss は 'miss' bucket とする。
    """
    df2 = df.copy()
    df2["return_x"] = df2["actual_payout"].fillna(0) / 700.0
    def band(x):
        if x == 0:
            return "miss"
        if x < 3:
            return "1-3x"
        if x < 7:
            return "3-7x"
        if x < 15:
            return "7-15x"
        if x < 50:
            return "15-50x"
        return "50x+"
    df2["band"] = df2["return_x"].apply(band)
    grp = df2.groupby(["course_en", "cond_clean", "band"]).agg(
        n=("race_id", "count"),
        inv=("investment", "sum"),
        pay=("actual_payout", "sum"),
    ).reset_index()
    grp["pnl"] = grp["pay"] - grp["inv"]
    grp["roi_pct"] = grp["pay"] / grp["inv"] * 100
    return grp


def main():
    df = load_data()
    summary = {
        "n_total_settled_jra": len(df),
        "n_excluded_nar": "see comment",  # NAR removed via course filter
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "baseline": {
            "n": int(len(df)),
            "inv": float(df["investment"].sum()),
            "pay": float(df["actual_payout"].sum()),
            "pnl": float((df["actual_payout"] - df["investment"]).sum()),
            "roi_pct": float(df["actual_payout"].sum() / df["investment"].sum() * 100),
        },
    }
    grid = build_grid(df)
    grid_records = grid.to_dict(orient="records")

    # Strategy comparisons
    strategies = {
        "A_master": {"courses": ["Kyoto", "Chukyo"], "conds": []},
        "B_with_TY_NY": {"courses": ["Kyoto", "Chukyo", "Tokyo", "Nakayama"], "conds": []},
        "C_with_X": {"courses": ["Kyoto", "Chukyo"], "conds": ["X"]},
        "D_all_worst": {"courses": ["Kyoto", "Chukyo", "Tokyo", "Nakayama"], "conds": ["X"]},
        "E_X_only": {"courses": [], "conds": ["X"]},
        "F_TY_NY_only": {"courses": ["Tokyo", "Nakayama"], "conds": []},
    }
    proj = {}
    for k, v in strategies.items():
        proj[k] = project_strategy(df, exclude_courses=v["courses"], exclude_conds=v["conds"])
        proj[k]["spec"] = v

    # Odds-band interaction
    ob = odds_band_interaction(df)
    # Top worst interactions by PnL (N >= 5)
    worst_ob = ob[ob["n"] >= 5].sort_values("pnl").head(15).to_dict(orient="records")

    # Sort grid by p_val for ranking
    grid_sig = grid[grid["n"] >= 10].copy()
    grid_sig = grid_sig.sort_values("p_val").to_dict(orient="records")

    out = {
        "summary": summary,
        "grid": grid_records,
        "grid_significant": grid_sig,
        "projections": proj,
        "worst_odds_band_interactions": worst_ob,
    }
    print(json.dumps(out, indent=2, default=lambda x: None if pd.isna(x) else float(x)))


if __name__ == "__main__":
    main()
