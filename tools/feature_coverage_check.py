"""特徴量カバレッジ測定 (4/19予測データ対象)

merge_jrdb_predict_features を 4/19 の各レースで実行し、
各 jrdb_* 特徴量の「デフォルト値以外の率」を測定する。

カバレッジ = (default以外の値 / 全頭) で算出。デフォルト値はバグの可能性が
高い欠損補填なので、これを除外したカバレッジを真の取得率とみなす。

Usage:
    python tools/feature_coverage_check.py --date 20260419
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")
DATA = BASE / "data"
sys.path.insert(0, str(BASE / "tools"))
sys.path.insert(0, str(BASE))

warnings.filterwarnings("ignore")


def load_predictions(ymd: str) -> pd.DataFrame:
    p = DATA / "daily_predictions" / f"{ymd}.csv"
    df = pd.read_csv(p, encoding="utf-8-sig", dtype=str)
    df["num_horses"] = pd.to_numeric(df["num_horses"], errors="coerce").fillna(0).astype(int)
    return df


def categorize_feature(name: str) -> str:
    if name.startswith("jrdb_prev_"):
        return "jrdb_sed_prev"
    if name.startswith("jrdb_kta_"):
        return "jrdb_kta"
    if name.startswith("jrdb_ze_"):
        return "jrdb_ze"
    if name.startswith("jrdb_oikiri") or name.startswith("jrdb_ten_time") or name.startswith("jrdb_shimai"):
        return "jrdb_cha"
    if name.startswith("jrdb_cid_") or name.startswith("jrdb_ls_idx"):
        return "jrdb_jo"
    if "_baba_" in name or name == "jrdb_tb_homestr_inner":
        return "jrdb_kab_sr"
    if name.startswith("jrdb_paddock") or name.startswith("jrdb_odds_idx") or "live" in name or "demeanor" in name or "body_code" in name:
        return "jrdb_tyb"
    if name.startswith("jrdb_dam_") or name.startswith("jrdb_bms_"):
        return "jrdb_blood"
    if name.startswith("jrdb_heavy_apt_skb") or name.startswith("jrdb_anshin") or name.startswith("jrdb_run_stage"):
        return "jrdb_skb"
    return "jrdb_kyi_basic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from jrdb_features import (
        merge_jrdb_predict_features,
        JRDB_DEFAULTS, PACI_TIER_A_DEFAULTS,
    )

    preds = load_predictions(args.date)
    n_races = len(preds)
    expected_total_horses = int(preds["num_horses"].sum())
    print(f"[Feature] target_date={args.date} races={n_races} horses={expected_total_horses}")

    # collect feature value lists
    feat_values: dict[str, list] = defaultdict(list)
    for _, row in preds.iterrows():
        rid = row["race_id"]
        n = int(row["num_horses"])
        horses = pd.DataFrame({
            "horse_num": list(range(1, n + 1)),
            "馬名": [f"h{i}" for i in range(1, n + 1)],
        })
        try:
            out = merge_jrdb_predict_features(horses, rid)
        except Exception as e:
            print(f"  [skip] race {rid}: {e}")
            continue
        for col in out.columns:
            if col.startswith("jrdb_"):
                feat_values[col].extend(out[col].tolist())

    rows = []
    for col, vals in feat_values.items():
        s = pd.Series(vals)
        notna_count = s.notna().sum()
        notna_rate = notna_count / len(s) if len(s) else 0
        default = JRDB_DEFAULTS.get(col, None)
        if default is None:
            non_default_rate = notna_rate
            default_used_rate = 0.0
        else:
            non_default = (s.notna() & (s != default))
            non_default_rate = non_default.sum() / len(s) if len(s) else 0
            default_used_rate = ((s == default).sum() / len(s)) if len(s) else 0
        category = categorize_feature(col)
        if non_default_rate >= 0.80:
            status = "OK"
        elif non_default_rate >= 0.50:
            status = "REVIEW"
        else:
            status = "LOW"
        rows.append({
            "feature": col,
            "category": category,
            "default": default,
            "non_default_rate": non_default_rate,
            "default_used_rate": default_used_rate,
            "notna_rate": notna_rate,
            "n": len(s),
            "status": status,
        })

    df = pd.DataFrame(rows).sort_values(["status", "non_default_rate"], ascending=[True, True])

    # category-level summary
    cat = df.groupby("category").agg(
        mean_cov=("non_default_rate", "mean"),
        min_cov=("non_default_rate", "min"),
        n_features=("feature", "count"),
    ).round(3).sort_values("mean_cov")

    # markdown report
    lines = []
    lines.append(f"# 特徴量カバレッジレポート ({args.date})")
    lines.append("")
    lines.append(f"対象: {n_races}レース / {expected_total_horses}頭")
    lines.append("")
    lines.append("## カテゴリ別平均カバレッジ")
    lines.append("")
    lines.append("| category | mean_cov | min_cov | n_features |")
    lines.append("|---|---|---|---|")
    for c, r in cat.iterrows():
        lines.append(f"| {c} | {r['mean_cov']*100:.1f}% | {r['min_cov']*100:.1f}% | {int(r['n_features'])} |")
    lines.append("")
    lines.append("## 特徴量別 (carverage<80% を抜粋、悪い順)")
    lines.append("")
    lines.append("| feature | category | non_default | default_used | n | status |")
    lines.append("|---|---|---|---|---|---|")
    low = df[df["status"] != "OK"]
    for _, r in low.iterrows():
        lines.append(
            f"| {r['feature']} | {r['category']} | {r['non_default_rate']*100:.1f}% | "
            f"{r['default_used_rate']*100:.1f}% | {r['n']} | {r['status']} |"
        )
    lines.append("")
    lines.append("## 全特徴量")
    lines.append("")
    lines.append("| feature | category | non_default | default_used | n | status |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['feature']} | {r['category']} | {r['non_default_rate']*100:.1f}% | "
            f"{r['default_used_rate']*100:.1f}% | {r['n']} | {r['status']} |"
        )

    out = args.out or f"report/feature_coverage_{args.date}.md"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print(f"[Feature] wrote {out}")

    # JSON snapshot for diff
    snap = out.replace(".md", ".json")
    df.to_json(snap, orient="records", force_ascii=False, indent=2)
    print(f"[Feature] snapshot {snap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
