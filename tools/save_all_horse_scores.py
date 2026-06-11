"""Session #71: 全頭分 V15 production score を別 csv に並行保存.

既存 daily_predict.py / predict_core.py / app.py / V15 model 一切 触らず、
predict_core から関数を import して同一 logic で再 inference、 全頭 score を
data/daily_predictions_full/{date}.csv に書き出す。

CLI:
  python tools/save_all_horse_scores.py --date 20260510
  python tools/save_all_horse_scores.py --date today
  python tools/save_all_horse_scores.py --date 20260510 --race-id 202604010312
  python tools/save_all_horse_scores.py --date 20260510 --dry-run

★ Session #64 spam 教訓 ★:
  daily_predict.py / race_auto_notify.py を絶対 trigger しない (subprocess も NG)。
  ワンショット exit、 loop 不可。

★ Session #70 LEAK 防止 ★:
  source = "v15_production_full" を全 row に明記。
  これは「production saved score の全頭版」 (top3 cut なし)、
  daily_predict.py が 8:00 に 同一 model + 同一 features で計算したものと等価。
  過去 R に対する retroactive inference (LEAK) ではない。

★ kill-switch ★:
  data/v18/save_all_horse_scores.kill が存在すれば即時 no-op exit。
"""
from __future__ import annotations

import os

# Windows console close + OpenMP 安全設定 (daily_predict.py と揃える)
if os.name == "nt":
    os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import re
import sys
import time
import traceback
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "tools"))

from jrdb_features import merge_jrdb_once
from predict_core import (
    apply_horse_stats,
    build_features,
    fetch_jra_and_weather,
    fetch_realtime_odds_full,
    fetch_result_odds,
    get_horse_stats,
    is_race_started,
    load_models,
    parse_shutuba,
    predict_race,
    save_odds_base,
    set_horse_defaults,
)

PREDICTIONS_DIR = os.path.join(BASE_DIR, "data", "daily_predictions")
PREDICTIONS_FULL_DIR = os.path.join(BASE_DIR, "data", "daily_predictions_full")
KILL_SWITCH = os.path.join(BASE_DIR, "data", "v18", "save_all_horse_scores.kill")

# 出力 CSV column 順 (Session #70 LEAK 防止思想と整合)
CSV_COLUMNS = [
    "date", "race_id", "course", "race_num", "race_name", "num_horses",
    "distance", "surface", "condition",
    "horse_num", "horse_name", "horse_id",
    "V15_score", "rank_in_race", "popularity", "odds",
    "source",
]


def get_race_ids_from_existing_csv(date_str: str) -> list[str]:
    """既存 daily_predictions/{date}.csv から race_id を取得.
    daily_predict.py 完了後に走る前提。 ground-truth の R 一覧に従う。"""
    path = os.path.join(PREDICTIONS_DIR, f"{date_str}.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id": str})
        return df["race_id"].astype(str).tolist()
    except Exception as e:
        # retry 1 回 (write 中の可能性)
        print(f"[WARN] csv read fail ({e})、 30s sleep + retry")
        time.sleep(30)
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id": str})
            return df["race_id"].astype(str).tolist()
        except Exception as e2:
            print(f"[ERROR] csv read fail (2回目): {e2}")
            return []


def predict_one_race_full(race_id: str, model_data: dict,
                          jra_weather_cache: dict) -> tuple:
    """1 R の全頭 inference. daily_predict.py の race loop と同等 logic、
    但し top3 cut しない。 (sorted_df, race_name, race_info) を返す。
    エラー時は (None, error_msg)."""
    # parse_shutuba
    try:
        race_name, horses, horse_ids, race_info = parse_shutuba(race_id)
    except Exception as e:
        return None, f"parse_shutuba fail: {e}"

    if not horses:
        return None, "馬データなし"
    if race_info.get("surface") == "障":
        return None, "障害除外"

    num_horses = len(horses)

    # オッズ取得 (daily_predict.py と同じ)
    race_started = is_race_started(race_id)
    pop_dict = {}
    if race_started:
        try:
            odds_dict, pop_dict = fetch_result_odds(race_id)
        except Exception:
            odds_dict, pop_dict = {}, {}
    else:
        try:
            odds_full = fetch_realtime_odds_full(race_id)
            odds_dict = {u: v["odds"] for u, v in odds_full.items()}
        except Exception:
            odds_dict = {}

    odds_available = len(odds_dict) > 0
    if odds_available:
        for horse in horses:
            umaban = horse.get("馬番", 0)
            if umaban in odds_dict:
                horse["単勝オッズ"] = odds_dict[umaban]
            if umaban in pop_dict:
                horse["人気順位"] = pop_dict[umaban]
    if not odds_available:
        if any(h.get("単勝オッズ", 0) > 0 for h in horses):
            odds_available = True

    # JRA 馬場・天候 (cache)
    course_name = race_info.get("course", "")
    jra_info, weather_info = {}, {}
    if model_data.get("is_live") and course_name:
        if course_name not in jra_weather_cache:
            try:
                jra_info, weather_info = fetch_jra_and_weather(course_name)
            except Exception:
                jra_info, weather_info = {}, {}
            jra_weather_cache[course_name] = (jra_info, weather_info)
        else:
            jra_info, weather_info = jra_weather_cache[course_name]

    # 各馬の過去成績取得
    for i, (horse, hid) in enumerate(zip(horses, horse_ids)):
        if hid:
            try:
                stats = get_horse_stats(
                    hid, race_info["distance"], race_info["surface"], course_name
                )
                apply_horse_stats(horse, stats, race_info)
            except Exception:
                set_horse_defaults(horse)
        else:
            set_horse_defaults(horse)
        if i < num_horses - 1:
            time.sleep(0.3)

    # 特徴量構築 + 予測
    df = build_features(
        horses, race_info, model_data, race_id=race_id,
        odds_dict=odds_dict, jra_track_info=jra_info, weather_info=weather_info,
    )
    # ★二重マージ禁止: build_features 内で適用済みのため merge_jrdb_once でガード (6/11 Fable sweep)★
    try:
        df = merge_jrdb_once(df, race_id)
    except Exception as e:
        print(f"  [JRDB] merge skip: {e}")

    df = predict_race(df, model_data, odds_available, race_info=race_info)

    # 全頭 sort + rank 付与
    sorted_df = df.sort_values("スコア", ascending=False).reset_index(drop=True)
    sorted_df["rank_in_race"] = range(1, len(sorted_df) + 1)

    return (sorted_df, race_name, race_info), None


def df_row_to_csv_dict(row, race_id, race_name, race_info, num_horses, date_str):
    """sorted_df の 1 row を CSV row dict に変換."""
    rn_match = re.search(r"(\d+)", str(race_info.get("race_num", "0")))
    race_num_int = int(rn_match.group(1)) if rn_match else 0
    return {
        "date": date_str,
        "race_id": race_id,
        "course": race_info.get("course", ""),
        "race_num": race_num_int,
        "race_name": race_name,
        "num_horses": num_horses,
        "distance": race_info.get("distance", 0),
        "surface": race_info.get("surface", ""),
        "condition": race_info.get("condition", ""),
        "horse_num": int(row.get("馬番", 0) or 0),
        "horse_name": str(row.get("馬名", "")),
        "horse_id": str(row.get("horse_id", "") or ""),
        "V15_score": float(row.get("スコア", 0) or 0),
        "rank_in_race": int(row.get("rank_in_race", 0) or 0),
        "popularity": int(row.get("人気順位", 0) or 0),
        "odds": float(row.get("単勝オッズ", 0) or 0),
        "source": "v15_production_full",
    }


def main():
    # kill-switch 最優先
    if os.path.exists(KILL_SWITCH):
        print(f"[save_all_horse_scores] kill-switch active ({KILL_SWITCH}) → no-op exit")
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="today", help="YYYYMMDD or 'today'")
    ap.add_argument("--race-id", default=None, help="single race for test")
    ap.add_argument("--dry-run", action="store_true", help="csv 保存スキップ")
    args = ap.parse_args()

    date_str = (
        datetime.now().strftime("%Y%m%d") if args.date == "today" else args.date
    )
    print("=" * 60)
    print(f"save_all_horse_scores - {date_str} (dry_run={args.dry_run})")
    print("=" * 60)

    # model 1 回 load (daily_predict.py 同様)
    print("\n[STEP 0] V15 model load...")
    try:
        model_data = load_models()
    except Exception as e:
        print(f"[ERROR] model load fail: {e}")
        traceback.print_exc()
        sys.exit(1)
    if model_data.get("model") is None:
        print("[ERROR] model load fail (None)")
        sys.exit(1)
    print(f"  loaded: is_live={model_data.get('is_live')}")

    # race_id list
    if args.race_id:
        race_ids = [args.race_id]
    else:
        race_ids = get_race_ids_from_existing_csv(date_str)

    if not race_ids:
        print(f"[INFO] {date_str} の race 一覧見つからず "
              f"(daily_predict.py 未完了 or 非開催)")
        return

    print(f"\n[STEP 1] 対象 R: {len(race_ids)}")

    # 各 R 推論
    all_rows = []
    failed = []
    jra_weather_cache: dict = {}

    t0 = time.time()
    for idx, race_id in enumerate(race_ids):
        print(f"\n[{idx+1}/{len(race_ids)}] race_id={race_id}")
        try:
            result, err = predict_one_race_full(race_id, model_data, jra_weather_cache)
            if result is None:
                print(f"  [SKIP] {err}")
                failed.append({"race_id": race_id, "reason": err})
                continue
            sorted_df, race_name, race_info = result
            num_horses = len(sorted_df)
            for _, row in sorted_df.iterrows():
                all_rows.append(df_row_to_csv_dict(
                    row, race_id, race_name, race_info, num_horses, date_str
                ))
            print(f"  [OK] {race_name} {num_horses} 頭")
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            failed.append({"race_id": race_id, "reason": str(e)})

    elapsed = time.time() - t0
    print(f"\n[STEP 2] inference 完了 ({elapsed:.1f}s, "
          f"{len(race_ids)} R, {len(all_rows)} row, {len(failed)} fail)")

    # CSV 保存
    if not all_rows:
        print("[WARN] 0 row、 csv 保存スキップ")
        sys.exit(2)

    out_df = pd.DataFrame(all_rows, columns=CSV_COLUMNS)

    if args.dry_run:
        print(f"\n[DRY RUN] {len(out_df)} row、 保存スキップ")
        print(out_df.head(10).to_string(index=False))
        return

    os.makedirs(PREDICTIONS_FULL_DIR, exist_ok=True)
    out_path = os.path.join(PREDICTIONS_FULL_DIR, f"{date_str}.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] {len(out_df)} row 保存: {out_path}")

    if failed:
        print(f"\n失敗 R: {len(failed)}")
        for f in failed:
            print(f"  - {f['race_id']}: {f['reason']}")


if __name__ == "__main__":
    main()
