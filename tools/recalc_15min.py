"""
P0-5 順5: -15 min 再計算 orchestrator

朝 8:00 V15 prob (daily_predictions/{date}.csv) を load、
-15 min features (live_pre_features/{date}/{race_id}.json) と結合、
calibrator_overlay 経由で 補正 ranking、
戦略⑦案 C 適用 (京都 / 条件 X skip)、
data/recalc_15min/{date}/{race_id}.json に出力。

★ V15 production 投票判断 影響なし、 shadow output only ★
★ Discord 通知は discord_recalc_notify.py (#updates 専用) に delegate ★

usage:
  python tools/recalc_15min.py --race-id 202605020611 [--date 20260517] [--mock] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def load_morning_prediction(race_id: str, date_str: str) -> list[dict] | None:
    """daily_predictions/{date}.csv から race_id の朝 8:00 ranking load.

    daily_predictions_full/{date}.csv (全頭、 V15_score 列) があれば優先。
    なければ daily_predictions/{date}.csv の top1 のみで fallback。
    """
    race_id_str = str(race_id)

    # 1) 全頭 fullファイル優先
    full_path = REPO / "data" / "daily_predictions_full" / f"{date_str}.csv"
    if full_path.exists():
        full_df = pd.read_csv(full_path, dtype={"race_id": str}, encoding="utf-8-sig")
        race_full = full_df[full_df["race_id"].astype(str) == race_id_str]
        if len(race_full) > 0:
            # column 名は V15_score (大文字)、 v15_prob に rename
            score_col = "V15_score" if "V15_score" in race_full.columns else "v15_score"
            if score_col in race_full.columns:
                return (
                    race_full[["horse_num", score_col]]
                    .rename(columns={score_col: "v15_prob"})
                    .to_dict("records")
                )

    # 2) top1 のみ fallback
    path = REPO / "data" / "daily_predictions" / f"{date_str}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"race_id": str}, encoding="utf-8-sig")
    race_rows = df[df["race_id"].astype(str) == race_id_str]
    if len(race_rows) == 0:
        return None

    row = race_rows.iloc[0]
    return [{"horse_num": int(row["top1_num"]), "v15_prob": float(row["top1_score"])}]


def load_15min_features(race_id: str, date_str: str) -> dict | None:
    """data/live_pre_features/{date}/{race_id}.json から -15 min features load.

    ★ 5/17 G1 day は file 存在しない、 mock で動作 ★
    """
    path = REPO / "data" / "live_pre_features" / date_str / f"{race_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_race_meta(race_id: str, date_str: str) -> dict | None:
    """race meta (course / condition / race_name) load (戦略⑦案 C 用)."""
    race_id_str = str(race_id)
    path = REPO / "data" / "daily_predictions" / f"{date_str}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"race_id": str}, encoding="utf-8-sig")
    race_rows = df[df["race_id"].astype(str) == race_id_str]
    if len(race_rows) == 0:
        return None
    row = race_rows.iloc[0]
    return {
        "race_id": race_id_str,
        "race_name": str(row.get("race_name", "")),
        "course": str(row.get("course", "")),
        "condition": str(row.get("condition", "")),
        "distance": row.get("distance"),
        "num_horses": row.get("num_horses"),
    }


def is_strategy_7c_excluded(race_meta: dict) -> tuple[bool, str | None]:
    """戦略⑦案 C で skip 対象か (race_auto_notify.py と整合).

    京都 / 条件 X の Graded race 保護 logic も継承。
    """
    course = race_meta.get("course", "") or ""
    condition = race_meta.get("condition", "") or ""
    race_name = race_meta.get("race_name", "") or ""

    is_graded = any(g in race_name for g in ["G1", "G2", "G3", "GⅠ", "GⅡ", "GⅢ"])
    is_listed = any(s in race_name for s in ["L)", "(L)", "OP)", "(OP)"])

    if course == "京都" and not (is_graded or is_listed):
        return True, "strategy_7_kyoto_p0_2_5_17"
    if condition == "X" and not (is_graded or is_listed):
        return True, "strategy_7_cond_X_p0_2_5_17"
    return False, None


def _json_default(o: Any) -> Any:
    """JSON serializer for numpy / pandas types."""
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def run_recalc(
    race_id: str,
    date_str: str | None = None,
    mock: bool = False,
    dry_run: bool = False,
) -> dict:
    """-15 min recalc full pipeline.

    Returns:
        dict with 'status', 'race_id', 'original_ranking', 'recalc_ranking',
        'notification_result', etc.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    result: dict[str, Any] = {
        "race_id": str(race_id),
        "date_str": date_str,
        "started_at": datetime.now().isoformat(),
    }

    # 1. 朝 8:00 prediction load
    morning = load_morning_prediction(race_id, date_str)
    if morning is None:
        if mock:
            morning = [
                {"horse_num": 5, "v15_prob": 0.65},
                {"horse_num": 3, "v15_prob": 0.55},
                {"horse_num": 1, "v15_prob": 0.48},
                {"horse_num": 7, "v15_prob": 0.42},
                {"horse_num": 4, "v15_prob": 0.38},
            ]
        else:
            result["status"] = "fallback"
            result["reason"] = "no_morning_prediction"
            result["fallback_to"] = "V15_morning_8am"
            return result

    result["original_ranking"] = morning

    # 2. race meta load (戦略⑦案 C 用)
    race_meta = load_race_meta(race_id, date_str)
    if race_meta is None:
        if mock:
            race_meta = {
                "race_id": str(race_id),
                "race_name": "mock race",
                "course": "東京",
                "condition": "A",
            }
        else:
            result["status"] = "fallback"
            result["reason"] = "no_race_meta"
            return result

    result["race_meta"] = race_meta

    # 3. 戦略⑦案 C skip check
    skip, skip_reason = is_strategy_7c_excluded(race_meta)
    if skip:
        result["status"] = "strategy_7c_skip"
        result["skip_reason"] = skip_reason
        return result

    # 4. -15 min features load (mock or real)
    pre_features = load_15min_features(race_id, date_str)
    if pre_features is None and mock:
        pre_features = {
            "odds_snapshot": {
                # mock: 5番 急変 (朝 3.0 → -15 min 1.5、 implied 過剰)
                5: 1.5,
                3: 4.0,
                1: 8.0,
                7: 12.0,
                4: 20.0,
            },
            "weight_diff": {
                5: -3,  # 適度な絞り = positive
                3: 0,
                1: 12,  # 馬体重 大増 = penalty
                7: -2,
                4: 0,
            },
        }

    result["pre_features_available"] = pre_features is not None

    # 5. calibrator_overlay 適用 (★ V15 prob 不変、 delta only ★)
    try:
        from tools.calibrator_overlay import overlay_race_ranking
    except ImportError:
        # path 通っていない場合 fallback
        sys.path.insert(0, str(REPO / "tools"))
        from calibrator_overlay import overlay_race_ranking  # type: ignore

    df = pd.DataFrame(morning)
    if pre_features:
        odds_map = pre_features.get("odds_snapshot", {}) or {}
        weight_map = pre_features.get("weight_diff", {}) or {}
        # JSON key は文字列の場合あり、 horse_num int / str 両対応
        def _lookup(m: dict, hn: Any) -> Any:
            if hn in m:
                return m[hn]
            try:
                if str(hn) in m:
                    return m[str(hn)]
                if int(hn) in m:
                    return m[int(hn)]
            except (ValueError, TypeError):
                pass
            return None

        df["odds_15min"] = df["horse_num"].map(lambda hn: _lookup(odds_map, hn))
        df["weight_diff"] = df["horse_num"].map(lambda hn: _lookup(weight_map, hn))

    recalc_df = overlay_race_ranking(df)
    recalc_ranking = recalc_df.sort_values("new_rank").to_dict("records")
    result["recalc_ranking"] = recalc_ranking

    # 6. Discord 通知 (★ #updates 専用、 V15 投票判断影響なし ★)
    try:
        from tools.discord_recalc_notify import notify_recalc
    except ImportError:
        sys.path.insert(0, str(REPO / "tools"))
        from discord_recalc_notify import notify_recalc  # type: ignore

    original_list = [
        (int(r["horse_num"]), float(r["v15_prob"])) for r in morning
    ]
    recalc_list = [
        (int(r["horse_num"]), float(r["corrected_prob"])) for r in recalc_ranking
    ]

    notif = notify_recalc(
        str(race_id),
        original_list,
        recalc_list,
        race_meta,
        dry_run=dry_run or mock,
    )
    result["notification_result"] = notif

    result["status"] = "ok"

    # 7. output 保存 (status set 後に書き込み)
    if not dry_run:
        out_dir = REPO / "data" / "recalc_15min" / date_str
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{race_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=_json_default)
        result["output_file"] = str(out_file)

    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="P0-5 順5: -15 min recalc orchestrator (V15 不変、 shadow eval)"
    )
    ap.add_argument("--race-id", required=True)
    ap.add_argument("--date", default=None, help="YYYYMMDD (default today)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    result = run_recalc(
        args.race_id, date_str=args.date, mock=args.mock, dry_run=args.dry_run
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
