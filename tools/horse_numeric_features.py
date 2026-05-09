"""Session #63 D: JRDB / JV-Link 数値 features 統合.

5/9 全 R 出走馬の paddock_idx (TYB) / training_idx + idm (KYI) /
weight_diff (TYB) を 該当 race_id でフィルタして CSV 化。

5/9 JRDB feed 未配信 (latest TYB 5/2 / KYI 5/3) の場合、 該当 race_id 行は
存在しないので **全馬 NaN**。 doc に明記。

usage:
  python tools/horse_numeric_features.py
"""
from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
TYB_DIR = BASE / "data" / "jrdb" / "extracted" / "Tyb"
KYI_DIR = BASE / "data" / "jrdb" / "extracted" / "Kyi"
CYB_DIR = BASE / "data" / "jrdb" / "extracted" / "Cyb"
OUT_CSV = BASE / "data" / "v18" / "horse_numeric_features_5_9.csv"
OUT_DOC = BASE / "data" / "v18" / "session_63_numeric_features.md"

sys.path.insert(0, str(BASE / "tools"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TARGET_RACE_IDS = {
    "202608030511", "202608030512",
    "202605020511", "202605020512",
    "202604010311", "202604010312",
}


def parse_tyb_lines():
    """全 TYB ファイルから 5/9 該当 race の row 抽出 → dict[(rid,umaban)] -> features."""
    from parse_jrdb import parse_tyb_line
    out = {}
    files = sorted(glob.glob(str(TYB_DIR / "TYB*.txt")))
    for fp in files:
        try:
            with open(fp, "rb") as f:
                for line in f:
                    line = line.rstrip(b"\r\n")
                    if len(line) < 60:
                        continue
                    try:
                        row = parse_tyb_line(line)
                    except Exception:
                        continue
                    rid = row.get("race_id")
                    if rid not in TARGET_RACE_IDS:
                        continue
                    uma = row.get("umaban")
                    if uma is None:
                        continue
                    out[(rid, int(uma))] = {
                        "paddock_idx": row.get("padock_idx"),
                        "padock_mark": row.get("padock_mark"),
                        "weight_diff": row.get("weight_diff"),
                        "horse_weight": row.get("horse_weight"),
                        "tansho_odds": row.get("tansho_odds"),
                        "fukusho_odds": row.get("fukusho_odds"),
                        "tyb_source": Path(fp).name,
                    }
        except Exception as e:
            print(f"[TYB skip] {fp}: {e}")
    return out


def parse_kyi_lines():
    """全 KYI ファイルから 5/9 該当 race の row 抽出."""
    from parse_jrdb import KYI_COLUMNS, _field, _safe_int, _safe_float, _build_race_id
    out = {}
    files = sorted(glob.glob(str(KYI_DIR / "KYI*.txt")))
    for fp in files:
        try:
            with open(fp, "rb") as f:
                for line in f:
                    line = line.rstrip(b"\r\n")
                    if len(line) < 100:
                        continue
                    try:
                        row = {}
                        for name, s, l in KYI_COLUMNS:
                            row[name] = _field(line, s, l)
                        rid = _build_race_id(row["basho_code"], row["year"],
                                             row["kai"], row["nichi"], row["race_num"])
                        if rid not in TARGET_RACE_IDS:
                            continue
                        uma = _safe_int(row["umaban"])
                        if uma is None:
                            continue
                        out[(rid, int(uma))] = {
                            "horse_name": row.get("horse_name", "").strip(),
                            "idm_score": _safe_float(row.get("idm")),
                            "training_idx": _safe_float(row.get("train_idx")),
                            "stable_idx": _safe_float(row.get("stable_idx")),
                            "ninki_idx": _safe_float(row.get("ninki_idx")),
                            "gekiso_idx": _safe_float(row.get("gekiso_idx")),
                            "kyi_source": Path(fp).name,
                        }
                    except Exception:
                        continue
        except Exception as e:
            print(f"[KYI skip] {fp}: {e}")
    return out


def parse_cyb_lines():
    """CYB から train_eval / train_mark."""
    from parse_jrdb import _field, _safe_int, _build_race_id
    out = {}
    CYB_COLUMNS = [
        ('basho_code', 1, 2), ('year', 3, 2), ('kai', 5, 1), ('nichi', 6, 1),
        ('race_num', 7, 2), ('umaban', 9, 2),
        ('train_type', 11, 1), ('train_baba', 13, 1),
        ('train_mark', 14, 1), ('train_amount', 15, 1),
        ('train_eval', 63, 1),
    ]
    files = sorted(glob.glob(str(CYB_DIR / "CYB*.txt")))
    for fp in files:
        try:
            with open(fp, "rb") as f:
                for line in f:
                    line = line.rstrip(b"\r\n")
                    if len(line) < 60:
                        continue
                    try:
                        row = {}
                        for name, s, l in CYB_COLUMNS:
                            row[name] = _field(line, s, l)
                        rid = _build_race_id(row["basho_code"], row["year"],
                                             row["kai"], row["nichi"], row["race_num"])
                        if rid not in TARGET_RACE_IDS:
                            continue
                        uma = _safe_int(row["umaban"])
                        if uma is None:
                            continue
                        out[(rid, int(uma))] = {
                            "train_eval": row.get("train_eval", "").strip(),
                            "train_mark": row.get("train_mark", "").strip(),
                            "cyb_source": Path(fp).name,
                        }
                    except Exception:
                        continue
        except Exception as e:
            print(f"[CYB skip] {fp}: {e}")
    return out


def main():
    print("=== TYB parse ===")
    tyb = parse_tyb_lines()
    print(f"  {len(tyb)} rows for 5/9 races")
    print("=== KYI parse ===")
    kyi = parse_kyi_lines()
    print(f"  {len(kyi)} rows for 5/9 races")
    print("=== CYB parse ===")
    cyb = parse_cyb_lines()
    print(f"  {len(cyb)} rows for 5/9 races")

    # 全馬 列挙: daily_predictions の race_id × umaban (1..num_horses)
    import pandas as pd
    df = pd.read_csv(BASE / "data" / "daily_predictions" / "20260509.csv", dtype=str)
    target = df[df["race_id"].isin(TARGET_RACE_IDS)].copy()

    rows = []
    for _, r in target.iterrows():
        rid = str(r["race_id"])
        try:
            n = int(str(r.get("num_horses", "0")) or 0)
        except Exception:
            n = 0
        for uma in range(1, n + 1):
            tyb_row = tyb.get((rid, uma), {})
            kyi_row = kyi.get((rid, uma), {})
            cyb_row = cyb.get((rid, uma), {})
            rows.append({
                "race_id": rid,
                "course": r.get("course", ""),
                "race_num": r.get("race_num", ""),
                "race_name": r.get("race_name", ""),
                "umaban": uma,
                "horse_name": kyi_row.get("horse_name", ""),
                "paddock_idx": tyb_row.get("paddock_idx") or "",
                "padock_mark": tyb_row.get("padock_mark") or "",
                "weight_diff": tyb_row.get("weight_diff") if tyb_row.get("weight_diff") not in (None, "") else "",
                "horse_weight": tyb_row.get("horse_weight") or "",
                "training_idx": kyi_row.get("training_idx") or "",
                "idm_score": kyi_row.get("idm_score") or "",
                "stable_idx": kyi_row.get("stable_idx") or "",
                "ninki_idx": kyi_row.get("ninki_idx") or "",
                "gekiso_idx": kyi_row.get("gekiso_idx") or "",
                "train_eval": cyb_row.get("train_eval") or "",
                "train_mark": cyb_row.get("train_mark") or "",
                "tansho_odds": tyb_row.get("tansho_odds") or "",
                "tyb_source": tyb_row.get("tyb_source", ""),
                "kyi_source": kyi_row.get("kyi_source", ""),
                "cyb_source": cyb_row.get("cyb_source", ""),
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["race_id", "course", "race_num", "race_name", "umaban", "horse_name",
            "paddock_idx", "padock_mark", "weight_diff", "horse_weight",
            "training_idx", "idm_score", "stable_idx", "ninki_idx", "gekiso_idx",
            "train_eval", "train_mark", "tansho_odds",
            "tyb_source", "kyi_source", "cyb_source"]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    n_tyb = sum(1 for r in rows if r["paddock_idx"] != "")
    n_kyi = sum(1 for r in rows if r["idm_score"] != "")
    n_cyb = sum(1 for r in rows if r["train_eval"] != "")

    print(f"\n=== 全 {len(rows)} 馬: TYB={n_tyb} KYI={n_kyi} CYB={n_cyb} ===")
    print(f"csv: {OUT_CSV.relative_to(BASE)}")

    doc_lines = [
        "# Session #63 D: JRDB / JV-Link 数値 features 統合 結果",
        "",
        f"対象: 5/9 重賞 3 + 12R 3 = 6 R / {len(rows)} 馬",
        "",
        "## カバレッジ",
        f"- TYB (パドック指数): {n_tyb}/{len(rows)} 馬 ({n_tyb/max(len(rows),1)*100:.1f}%)",
        f"- KYI (IDM/training_idx): {n_kyi}/{len(rows)} 馬 ({n_kyi/max(len(rows),1)*100:.1f}%)",
        f"- CYB (train_eval/mark): {n_cyb}/{len(rows)} 馬 ({n_cyb/max(len(rows),1)*100:.1f}%)",
        "",
        "## JRDB feed 状況",
        f"- TYB latest available: {sorted(set(r['tyb_source'] for r in rows if r['tyb_source']))}",
        f"- KYI latest available: {sorted(set(r['kyi_source'] for r in rows if r['kyi_source']))}",
        f"- CYB latest available: {sorted(set(r['cyb_source'] for r in rows if r['cyb_source']))}",
        "",
        "## JV-Link 当日体重 (SE)",
        "- 5/9 当日 取得 skip (時間制約、 JRDB TYB の weight_diff で代替)",
        "",
        "## 出力",
        f"- csv: {OUT_CSV.name}",
        "- columns: race_id, course, race_num, race_name, umaban, horse_name,",
        "  paddock_idx, padock_mark, weight_diff, horse_weight,",
        "  training_idx, idm_score, stable_idx, ninki_idx, gekiso_idx,",
        "  train_eval, train_mark, tansho_odds, *_source",
    ]
    OUT_DOC.write_text("\n".join(doc_lines), encoding="utf-8")
    print(f"doc: {OUT_DOC.relative_to(BASE)}")


if __name__ == "__main__":
    main()
