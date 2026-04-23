"""JRDB結合率の詳細マッピング (4/19予測データ対象)

各JRDBファイル(KYI/SED/TYB/CYB/JOA/KAB/KTA/CHA/KKA/JO/SRB)で
- 結合成功率 (期待頭数に対する実際の取得行数)
- 未結合内訳の自動分類:
    新馬戦/地方転籍/取消・除外/キー不一致(バグ可能性)/日付ミスマッチ/その他

Usage:
    python tools/jrdb_coverage_detailed.py --date 20260419
    python tools/jrdb_coverage_detailed.py --date 20260419 --out report/jrdb_coverage_detailed_20260423.md
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"C:/Users/takum/keiba-ai")
DATA = BASE / "data"

# 対象JRDBファイル定義
# (file_basename, race_id_col, umaban_col, encoding, label, prev_data)
#   prev_data=True なら blood_num 経由結合 (前走履歴)、=False なら当日結合
JRDB_DEFS = [
    ("jrdb_kyi.csv", "nk_race_id", "馬番",  False, "KYI(基本指数)",   None),
    ("jrdb_sed.csv", "race_id",    "umaban", True,  "SED(前走成績)",   "blood_num"),
    ("jrdb_tyb.csv", "nk_race_id", "馬番",  False, "TYB(当日情報)",   None),
    ("jrdb_cyb.csv", "race_id",    "umaban", False, "CYB(調教詳細)",   None),
    ("jrdb_joa.csv", "race_id",    "umaban", False, "JOA(オッズ)",     None),
    ("jrdb_kab.csv", "race_id",    None,     False, "KAB(開催)",       None),
    ("jrdb_kta.csv", "race_id",    "blood_num", False, "KTA(展開予想)", "blood_num"),
    ("jrdb_cha.csv", "race_id",    "umaban", False, "CHA(追切)",       None),
    ("jrdb_kka.csv", "race_id",    "umaban", False, "KKA(脚質)",       None),
    ("jrdb_jo.csv",  "race_id",    "umaban", False, "JO(CID/LS)",     None),
]


def load_predictions(ymd: str) -> pd.DataFrame:
    p = DATA / "daily_predictions" / f"{ymd}.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, encoding="utf-8-sig", dtype=str)
    df["num_horses"] = pd.to_numeric(df["num_horses"], errors="coerce").fillna(0).astype(int)
    return df


def load_jrdb(filename: str) -> pd.DataFrame | None:
    p = DATA / filename
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, encoding="utf-8-sig", dtype=str, low_memory=False)
        return df
    except Exception as e:
        print(f"[WARN] {filename}: {e}")
        return None


def normalize_rid(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.zfill(12)


def measure_one(jdf: pd.DataFrame, rid_col: str, target_rids: set[str],
                target_horses_per_race: dict[str, int]) -> dict:
    """1ファイルの結合状況を測定"""
    if jdf is None:
        return {"available": False}

    if rid_col not in jdf.columns:
        # fallback: try alternate
        for alt in ("race_id", "nk_race_id", "jra_race_id"):
            if alt in jdf.columns:
                rid_col = alt
                break
        else:
            return {"available": True, "rid_col_missing": True}

    rids = normalize_rid(jdf[rid_col])
    matched = jdf[rids.isin(target_rids)].copy()
    matched["_rid"] = normalize_rid(matched[rid_col])

    per_race = matched.groupby("_rid").size().to_dict()

    expected_races = len(target_rids)
    matched_races = sum(1 for r in target_rids if r in per_race)
    expected_horses = sum(target_horses_per_race.values())
    matched_horses = sum(min(per_race.get(r, 0), target_horses_per_race[r])
                         for r in target_rids)

    # 未結合の race_id 一覧
    unmatched_races = sorted([r for r in target_rids if r not in per_race])

    return {
        "available": True,
        "rid_col_missing": False,
        "rid_col_used": rid_col,
        "expected_races": expected_races,
        "matched_races": matched_races,
        "race_match_rate": matched_races / expected_races if expected_races else 0,
        "expected_horses": expected_horses,
        "matched_horses": matched_horses,
        "horse_match_rate": matched_horses / expected_horses if expected_horses else 0,
        "unmatched_races": unmatched_races,
        "raw_rows_in_target": len(matched),
    }


def classify_unmatched(unmatched_rids: list[str], preds: pd.DataFrame) -> dict:
    """未結合 race_id の分類 (race_name から推定)"""
    rec = {"shinba": [], "key_mismatch_candidate": [], "other": []}
    name_map = preds.set_index("race_id")["race_name"].to_dict()
    for rid in unmatched_rids:
        nm = str(name_map.get(rid, ""))
        if "新馬" in nm:
            rec["shinba"].append(rid)
        else:
            # KYI 自体は当日 race_id で結合できているかが基準
            # 他 JRDB ファイルが未結合 → キー不一致候補
            rec["key_mismatch_candidate"].append(rid)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    preds = load_predictions(args.date)
    target_rids_norm = set(normalize_rid(preds["race_id"]).tolist())
    target_horses_per_race = (
        preds.assign(_rid=normalize_rid(preds["race_id"]))
             .set_index("_rid")["num_horses"].astype(int).to_dict()
    )
    total_expected_horses = sum(target_horses_per_race.values())

    print(f"[Coverage] target_date={args.date}  races={len(target_rids_norm)}  expected_horses={total_expected_horses}")

    rows = []
    detailed = {}
    for fname, rid_col, _uma, _prev, label, _meta in JRDB_DEFS:
        jdf = load_jrdb(fname)
        m = measure_one(jdf, rid_col, target_rids_norm, target_horses_per_race)
        m["file"] = fname
        m["label"] = label
        rows.append(m)
        if m.get("available") and not m.get("rid_col_missing"):
            detailed[fname] = classify_unmatched(m["unmatched_races"], preds)

    # Markdown report
    lines = []
    lines.append(f"# JRDB結合率詳細レポート ({args.date})")
    lines.append("")
    lines.append(f"対象: {len(target_rids_norm)}レース / 期待頭数 {total_expected_horses}頭")
    lines.append("")
    lines.append("## 結合率サマリー")
    lines.append("")
    lines.append("| ファイル | ラベル | 利用可 | レース結合率 | 頭数結合率 | 未結合レース | 備考 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        if not r.get("available"):
            lines.append(f"| {r['file']} | {r['label']} | NO | - | - | - | ファイル不在 |")
            continue
        if r.get("rid_col_missing"):
            lines.append(f"| {r['file']} | {r['label']} | YES | - | - | - | rid列なし |")
            continue
        lines.append(
            f"| {r['file']} | {r['label']} | YES | "
            f"{r['matched_races']}/{r['expected_races']} ({r['race_match_rate']*100:.1f}%) | "
            f"{r['matched_horses']}/{r['expected_horses']} ({r['horse_match_rate']*100:.1f}%) | "
            f"{len(r['unmatched_races'])} | rid={r['rid_col_used']} |"
        )
    lines.append("")
    lines.append("## 未結合レースの分類")
    lines.append("")
    for fname, cls in detailed.items():
        if not (cls["shinba"] or cls["key_mismatch_candidate"] or cls["other"]):
            continue
        lines.append(f"### {fname}")
        lines.append(f"- 新馬戦(正当): {len(cls['shinba'])}件")
        lines.append(f"- キー不一致(要調査): {len(cls['key_mismatch_candidate'])}件")
        if cls["key_mismatch_candidate"][:5]:
            lines.append(f"  - 例: {cls['key_mismatch_candidate'][:5]}")
        lines.append(f"- その他: {len(cls['other'])}件")
        lines.append("")

    out = args.out or f"report/jrdb_coverage_detailed_{args.date}.md"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print(f"[Coverage] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
