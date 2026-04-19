"""データソース別・年別カバレッジレポート.

v16 再学習可否判定のため、各データソースの年別取得状況を集計する。

分母:
- jra_races_full.csv の年別総行数 (horse_id × race_id のペア数)

分子 (各ソース):
- race_id × umaban (or horse_id) のマッチ数

出力:
- report/v16_coverage_20260419.tsv  (TSV)
- report/v16_coverage_20260419.md   (Markdown)
"""
from __future__ import annotations

import os
import sys
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
REPORT_DIR = os.path.join(BASE, "report")
os.makedirs(REPORT_DIR, exist_ok=True)


def _year_from_race_id(rid) -> int | None:
    try:
        s = str(int(rid))
    except Exception:
        try:
            s = str(rid)
        except Exception:
            return None
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return None


def load_csv(path: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """CSV をロード。BOM 対応。存在しなければ空 DF."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
    except Exception as e:
        print(f"  [WARN] {path}: {e}", file=sys.stderr)
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e2:
            print(f"  [ERR] {path}: {e2}", file=sys.stderr)
            return pd.DataFrame()
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    return df


def _get_race_col(df: pd.DataFrame) -> str | None:
    """race_id カラム名を解決 (JRDB は nk_race_id を使う)."""
    for c in ("race_id", "nk_race_id"):
        if c in df.columns:
            return c
    return None


def count_by_year(df: pd.DataFrame) -> dict[int, int]:
    """race_id から年を抽出して件数集計."""
    col = _get_race_col(df)
    if col is None:
        return {}
    yrs = df[col].map(_year_from_race_id).dropna().astype(int)
    return yrs.value_counts().to_dict()


def count_unique_race_by_year(df: pd.DataFrame) -> dict[int, int]:
    """unique race_id 件数を年別集計."""
    col = _get_race_col(df)
    if col is None:
        return {}
    uniq = df[[col]].drop_duplicates()
    uniq["year"] = uniq[col].map(_year_from_race_id)
    uniq = uniq.dropna()
    uniq["year"] = uniq["year"].astype(int)
    return uniq["year"].value_counts().to_dict()


def main() -> int:
    years = list(range(2020, 2026))

    # --- 分母: jra_races_full ---
    print("[BASE] jra_races_full.csv を読み込み...")
    base = load_csv(os.path.join(DATA, "jra_races_full.csv"),
                    usecols=["year", "race_id", "horse_id", "umaban"])
    # jra_races_full は year が 2桁 (15,20,25 etc) → 4桁 (2015,2020,2025) に変換
    if not base.empty:
        base["year4"] = base["year"].astype(int).apply(lambda y: 2000 + y if y < 100 else y)
        base_by_year = base["year4"].value_counts().to_dict()
        base_races_by_year = base.drop_duplicates("race_id").groupby("year4").size().to_dict()
    else:
        base_by_year = {}
        base_races_by_year = {}

    # --- 各ソース ---
    sources = {
        # key: (path, horse-level bool, display name)
        "training_eval":    ("netkeiba_training_eval.csv",    True,  "調教評価"),
        "master_index":     ("netkeiba_master_index.csv",     True,  "マスターインデックス"),
        "upset":            ("netkeiba_upset_level.csv",      False, "波乱度 (race-level)"),
        "training_times":   ("netkeiba_training_times.csv",   True,  "調教タイム"),
        "speed_index":      ("netkeiba_speed_index.csv",      True,  "タイム指数"),
        "stable_comments":  ("netkeiba_stable_comments.csv",  True,  "厩舎コメント"),
        "race_review":      ("netkeiba_race_review.csv",      True,  "レース短評"),
        "shinba_eval":      ("netkeiba_shinba_eval.csv",      True,  "新馬評価"),
        "jrdb_kyi":         ("jrdb_kyi.csv",                  True,  "JRDB KYI"),
        "jrdb_sed":         ("jrdb_sed.csv",                  True,  "JRDB SED"),
        "jrdb_tyb":         ("jrdb_tyb.csv",                  True,  "JRDB TYB"),
        "jrdb_cyb":         ("jrdb_cyb.csv",                  True,  "JRDB CYB"),
        "jrdb_joa":         ("jrdb_joa.csv",                  True,  "JRDB JOA"),
    }

    rows = []
    for key, (fname, horse_level, disp) in sources.items():
        path = os.path.join(DATA, fname)
        df = load_csv(path)
        if df.empty:
            print(f"  [SKIP] {key}: not found or empty")
            for y in years:
                rows.append({"source": key, "year": y, "count": 0, "race_count": 0,
                             "coverage_horse": 0.0, "coverage_race": 0.0, "display": disp})
            continue
        by_year = count_by_year(df)
        race_by_year = count_unique_race_by_year(df)
        for y in years:
            count = int(by_year.get(y, 0))
            race_count = int(race_by_year.get(y, 0))
            base_horse = int(base_by_year.get(y, 0))
            base_race = int(base_races_by_year.get(y, 0))
            cov_h = 100.0 * count / base_horse if base_horse > 0 else 0.0
            cov_r = 100.0 * race_count / base_race if base_race > 0 else 0.0
            rows.append({
                "source": key,
                "year": y,
                "count": count,
                "race_count": race_count,
                "coverage_horse": round(cov_h, 1),
                "coverage_race": round(cov_r, 1),
                "display": disp,
            })
        total = sum(by_year.get(y, 0) for y in years)
        print(f"  [OK] {key}: total {total:,} rows across {years[0]}-{years[-1]}")

    out = pd.DataFrame(rows)
    tsv_path = os.path.join(REPORT_DIR, "v16_coverage_20260419.tsv")
    out.to_csv(tsv_path, sep="\t", index=False)
    print(f"\nTSV saved: {tsv_path}")

    # --- Markdown 出力 ---
    md_lines = []
    md_lines.append("# v16 データカバレッジレポート")
    md_lines.append(f"- 作成: 2026-04-19")
    md_lines.append(f"- 分母 (年別出走頭数 / レース数):")
    md_lines.append("")
    md_lines.append("| 年 | 出走頭数 (horse_id × race_id) | レース数 |")
    md_lines.append("|:---:|---:|---:|")
    for y in years:
        md_lines.append(f"| {y} | {int(base_by_year.get(y, 0)):,} | {int(base_races_by_year.get(y, 0)):,} |")
    md_lines.append("")

    md_lines.append("## ソース別 × 年別 カバレッジ (horse-level %)")
    md_lines.append("")
    header = "| source | " + " | ".join(str(y) for y in years) + " | 総件数 |"
    md_lines.append(header)
    md_lines.append("|:---|" + "|".join(["---:"] * len(years)) + "|---:|")
    for key in sources:
        disp = sources[key][2]
        cells = []
        total = 0
        for y in years:
            r = next((rr for rr in rows if rr["source"] == key and rr["year"] == y), None)
            if r:
                cov = r["coverage_horse"]
                cnt = r["count"]
                total += cnt
                cells.append(f"{cov:.1f}%")
            else:
                cells.append("0.0%")
        md_lines.append(f"| {disp} ({key}) | " + " | ".join(cells) + f" | {total:,} |")
    md_lines.append("")

    md_lines.append("## race-level カバレッジ (unique race_id %)")
    md_lines.append("")
    md_lines.append(header)
    md_lines.append("|:---|" + "|".join(["---:"] * len(years)) + "|---:|")
    for key in sources:
        disp = sources[key][2]
        cells = []
        total = 0
        for y in years:
            r = next((rr for rr in rows if rr["source"] == key and rr["year"] == y), None)
            if r:
                cov = r["coverage_race"]
                cnt = r["race_count"]
                total += cnt
                cells.append(f"{cov:.1f}%")
            else:
                cells.append("0.0%")
        md_lines.append(f"| {disp} ({key}) | " + " | ".join(cells) + f" | {total:,} |")
    md_lines.append("")

    # v16 Trigger 判定
    md_lines.append("## v16 Trigger 判定")
    md_lines.append("")
    md_lines.append("| 条件 | 閾値 | 最低カバレッジ (2020-2025) | 判定 |")
    md_lines.append("|:---|:---:|:---:|:---:|")
    triggers = []
    for key, threshold, label in [
        ("training_eval", 40.0, "training_eval >= 40%"),
        ("master_index",  30.0, "master_index >= 30%"),
    ]:
        covs = [r["coverage_horse"] for r in rows if r["source"] == key]
        minc = min(covs) if covs else 0.0
        ok = minc >= threshold
        triggers.append(ok)
        md_lines.append(f"| {label} | {threshold}% | {minc:.1f}% | {'✅ OK' if ok else '❌ NG'} |")
    all_ok = all(triggers)
    md_lines.append("")
    md_lines.append(f"### 総合判定: {'✅ v16 学習可能' if all_ok else '❌ v16 学習不可 (データ不足)'}")
    md_lines.append("")

    md_path = os.path.join(REPORT_DIR, "v16_coverage_20260419.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"MD saved: {md_path}")

    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
