"""premium_cache_to_csv — weekly_premium_cache JSON を CSV へ転換 (修復 + 恒久統合)

bug 真因 (Session #27):
- daily_premium_scrape.py は cache JSON にのみ書き込む
- 各種 CSV (netkeiba_speed_index.csv 等) は別 script (scrape_speed_index.py 等) で更新
- 別 script は --year デフォルト 2025 で 2026 race を対象にしない
- 結果: cache に存在しても CSV には 0 行追加

本 script は cache JSON を走査して各 CSV に append する。 race_id+umaban で dedupe。

Usage:
    python tools/premium_cache_to_csv.py                    # 全 cache 走査
    python tools/premium_cache_to_csv.py --date 20260502    # 特定日のみ
    python tools/premium_cache_to_csv.py --since 20260501   # 5/1 以降のみ
    python tools/premium_cache_to_csv.py --dry-run          # CSV 書き込みなし
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
CACHE_DIR = BASE / "data/weekly_premium_cache"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# CSV ターゲット定義
SPEED_INDEX_CSV = BASE / "data/netkeiba_speed_index.csv"
SPEED_INDEX_HEADER = [
    "race_id", "umaban", "horse_name", "sex_age", "weight_carry", "jockey",
    "index_max", "index_avg5", "index_dist", "index_course",
    "index_run1", "index_run2", "index_run3",
    "odds", "popularity",
]

TRAINING_CSV = BASE / "data/netkeiba_training_eval.csv"
TRAINING_HEADER = [
    "race_id", "umaban", "horse_name", "wood_best_4f", "wood_best_3f",
    "wood_count_2w", "sakaro_best_4f", "sakaro_best_3f", "sakaro_count_2w",
    "time_1f_last", "training_intensity",
]

STABLE_COMMENT_CSV = BASE / "data/netkeiba_stable_comments.csv"
STABLE_COMMENT_HEADER = [
    "race_id", "umaban", "horse_name", "comment", "comment_score",
]


def load_existing_keys(csv_path: Path) -> set[tuple[str, str]]:
    """既存 CSV から (race_id, umaban) の集合を返す。 dedupe 用。"""
    keys = set()
    if not csv_path.exists():
        return keys
    try:
        with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = (row.get("race_id") or "").strip()
                u = (row.get("umaban") or "").strip()
                if rid and u:
                    keys.add((rid, u))
    except Exception as e:
        print(f"  [WARN] {csv_path.name} 既存 key 読込失敗: {e}")
    return keys


def append_csv(csv_path: Path, header: list[str], rows: list[list]) -> int:
    """CSV に append (header 既存なら省略)。 row 数 return"""
    if not rows:
        return 0
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8-sig", newline="") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        for row in rows:
            # CSV-safe 文字列化
            row_str = []
            for v in row:
                s = "" if v is None else str(v)
                # comma 含む文字は ; へ置換 (元の scrape_speed_index と同じ約束)
                s = s.replace(",", "；")
                row_str.append(s)
            f.write(",".join(row_str) + "\n")
    return len(rows)


def process_cache_dir(date_str: str, dry_run: bool = False) -> dict:
    """1 日分の cache を 3 CSV へ転換"""
    cache_path = CACHE_DIR / date_str / "premium_cache.json"
    if not cache_path.exists():
        return {"date": date_str, "status": "no_cache"}

    try:
        with cache_path.open("r", encoding="utf-8") as f:
            cache = json.load(f)
    except Exception as e:
        return {"date": date_str, "status": f"json_load_err: {e}"}

    # dedupe key
    si_existing = load_existing_keys(SPEED_INDEX_CSV)
    tr_existing = load_existing_keys(TRAINING_CSV)
    sc_existing = load_existing_keys(STABLE_COMMENT_CSV)

    si_rows: list[list] = []
    tr_rows: list[list] = []
    sc_rows: list[list] = []

    for race_id, race_data in cache.items():
        # speed_index
        si = race_data.get("speed_index", {}) or {}
        for umaban, val in si.items():
            key = (str(race_id), str(umaban))
            if key in si_existing:
                continue
            si_existing.add(key)
            si_rows.append([
                race_id, umaban,
                val.get("horse_name", ""), "", "", "",  # sex_age/weight_carry/jockey 未保存
                val.get("index_max", 0), val.get("index_avg5", 0),
                val.get("index_dist", 0), val.get("index_course", 0),
                val.get("index_run1", 0), val.get("index_run2", 0), val.get("index_run3", 0),
                "", "",  # odds/popularity 未保存
            ])

        # training (cache 実スキーマ: course/time_4f/time_3f/time_1f/intensity/rank/
        # evaluation/review/date/is_sakaro/is_wood/laps — 6/12 Fable統合で実態に合わせ修正。
        # 旧実装は wood_best_4f 等の存在しないキーを読み、2026年分が全列空のゾンビ行になっていた)
        tr = race_data.get("training", {}) or {}
        for umaban, val in tr.items():
            if not isinstance(val, dict):
                continue
            key = (str(race_id), str(umaban))
            if key in tr_existing:
                continue
            tr_existing.add(key)
            laps = val.get("laps") or []
            if isinstance(laps, list) and laps:
                time_raw = "".join(str(x) for x in laps)
            else:
                t4, t3, t1 = val.get("time_4f", ""), val.get("time_3f", ""), val.get("time_1f", "")
                time_raw = f"{t4}({t3})({t1})" if str(t4).strip() else ""
            tr_rows.append([
                race_id, umaban,
                val.get("horse_name", ""),
                val.get("review", ""),          # prev_review
                val.get("date", ""),            # training_date
                val.get("course", ""),          # training_course
                val.get("condition", ""),       # training_condition (cache 未収集なら空)
                val.get("rider", ""),           # training_rider (同上)
                time_raw,                       # training_time_raw
                val.get("position", ""),        # training_position (同上)
                val.get("intensity", ""),       # training_intensity
                val.get("evaluation", ""),      # training_move
                val.get("rank", ""),            # training_rank
            ])

        # stable_comments
        cm = race_data.get("comments", {}) or {}
        for umaban, val in cm.items():
            if not isinstance(val, dict):
                # 旧形式: 文字列のみ
                comment_str = str(val) if val else ""
                if not comment_str:
                    continue
                key = (str(race_id), str(umaban))
                if key in sc_existing:
                    continue
                sc_existing.add(key)
                sc_rows.append([race_id, umaban, "", comment_str, ""])
                continue
            key = (str(race_id), str(umaban))
            if key in sc_existing:
                continue
            sc_existing.add(key)
            sc_rows.append([
                race_id, umaban,
                val.get("horse_name", ""),
                val.get("comment", "") or val.get("text", ""),
                val.get("comment_score", "") or val.get("score", ""),
            ])

    if dry_run:
        return {
            "date": date_str, "status": "dry_run",
            "speed_index_new": len(si_rows),
            "training_new": len(tr_rows),
            "stable_comments_new": len(sc_rows),
        }

    n_si = append_csv(SPEED_INDEX_CSV, SPEED_INDEX_HEADER, si_rows)
    n_tr = append_csv(TRAINING_CSV, TRAINING_HEADER, tr_rows)
    n_sc = append_csv(STABLE_COMMENT_CSV, STABLE_COMMENT_HEADER, sc_rows)
    return {
        "date": date_str, "status": "ok",
        "speed_index_new": n_si,
        "training_new": n_tr,
        "stable_comments_new": n_sc,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--since", type=str, default=None,
                   help="YYYYMMDD 以降のみ処理 (default: 全 cache)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not CACHE_DIR.exists():
        print(f"[NG] cache dir 不在: {CACHE_DIR}")
        return 1

    if args.date:
        targets = [args.date]
    else:
        all_dates = sorted([d.name for d in CACHE_DIR.iterdir() if d.is_dir() and len(d.name) == 8 and d.name.isdigit()])
        if args.since:
            targets = [d for d in all_dates if d >= args.since]
        else:
            targets = all_dates

    print(f"=== premium_cache_to_csv ({'DRY RUN' if args.dry_run else 'APPLY'}) ===")
    print(f"対象 cache 日数: {len(targets)}")

    total = {"speed_index_new": 0, "training_new": 0, "stable_comments_new": 0}
    summary = []
    for d in targets:
        r = process_cache_dir(d, dry_run=args.dry_run)
        summary.append(r)
        if r.get("status") == "ok" or r.get("status") == "dry_run":
            for k in total:
                total[k] += r.get(k, 0)
        # 進捗
        print(f"  {d}: si={r.get('speed_index_new', 0)} tr={r.get('training_new', 0)} sc={r.get('stable_comments_new', 0)} status={r.get('status')}")

    print()
    print(f"=== 合計 (新規追加) ===")
    print(f"  speed_index:     +{total['speed_index_new']} 行")
    print(f"  training_eval:   +{total['training_new']} 行")
    print(f"  stable_comments: +{total['stable_comments_new']} 行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
