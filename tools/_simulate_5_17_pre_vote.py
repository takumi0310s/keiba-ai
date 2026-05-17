"""投票前-1: 5/17 ヴィクトリアマイル G1 day 直前 全 R simulation (paper only).

★ V15 production 不変、 Discord 実発火なし、 G1 day blocklist 遵守 ★

各 R で:
1. V15 朝 8:00 ranking (daily_predictions_full/20260517.csv) load
2. 朝 8:00 odds (odds_base_20260517.csv) ± 5% noise で dummy -15 min odds 生成
3. weight_diff = neutral (0) で mock
4. calibrator_overlay 適用 → simulated -15 min ranking
5. 戦略⑦案 C 判定 (京都/条件 X、 G1/G2/G3/L 保護)
6. 順位変動 severity 集計 (none / minor / major / critical)
7. data/recalc_15min_simulation/20260517/{race_id}.json に paper output

NO live fetch (G1_DAY_BLOCKLIST), NO Discord, NO V15 modification.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from calibrator_overlay import overlay_race_ranking  # noqa: E402

DATE_STR = "20260517"
NOISE_PCT = 0.05  # ± 5% noise for dummy -15 min odds
RNG_SEED = 42

OUT_DIR = REPO / "data" / "recalc_15min_simulation" / DATE_STR
OUT_DIR.mkdir(parents=True, exist_ok=True)


def is_strategy_7c_excluded(race_meta: dict) -> tuple[bool, str | None]:
    course = race_meta.get("course", "") or ""
    condition = race_meta.get("condition", "") or ""
    race_name = race_meta.get("race_name", "") or ""
    num_horses = race_meta.get("num_horses", 16) or 16
    distance = race_meta.get("distance", 1600) or 1600

    is_graded = any(g in race_name for g in ["G1", "G2", "G3", "GⅠ", "GⅡ", "GⅢ"])
    is_listed = any(s in race_name for s in ["L)", "(L)", "OP)", "(OP)"])

    # 京都 / 条件 X は graded/listed 保護
    if course == "京都" and not (is_graded or is_listed):
        return True, "kyoto_non_graded"
    if condition == "X" and not (is_graded or is_listed):
        return True, "cond_X_non_graded"
    # 1000m 以下 (条件 D 内の 1000m 以下)
    if isinstance(distance, (int, float)) and distance <= 1000:
        return True, "dist_le_1000m"
    # 条件 E
    if condition == "E":
        return True, "cond_E"
    # 条件 B
    if condition == "B":
        return True, "cond_B"
    # 06_平場特別 (race_name に "特別" 含み、 重賞でない)
    if "特別" in race_name and not (is_graded or is_listed):
        return True, "non_graded_tokubetsu"
    return False, None


def classify_severity(rank_changes: list[int]) -> str:
    """順位変動 severity 分類.

    none: 0 件
    minor: top1 不変、 top2-top5 内で 1-2 件入替
    major: top1 入替、 または top3 で 2 件以上入替
    critical: top1 入替 かつ top3 で 2 件以上入替
    """
    if not rank_changes:
        return "none"
    top1_changed = rank_changes[0] != 0
    top3_swaps = sum(1 for rc in rank_changes[:3] if rc != 0)
    if top1_changed and top3_swaps >= 2:
        return "critical"
    if top1_changed:
        return "major"
    if top3_swaps >= 2:
        return "major"
    if any(rc != 0 for rc in rank_changes[:5]):
        return "minor"
    return "none"


def main() -> int:
    # 1. load daily_predictions_full (V15 朝 8:00、 全頭)
    full_path = REPO / "data" / "daily_predictions_full" / f"{DATE_STR}.csv"
    summary_path = REPO / "data" / "daily_predictions" / f"{DATE_STR}.csv"
    odds_path = REPO / "data" / f"odds_base_{DATE_STR}.csv"

    if not full_path.exists():
        print(f"[ERR] {full_path} not found")
        return 1

    full_df = pd.read_csv(full_path, dtype={"race_id": str}, encoding="utf-8-sig")
    summary_df = pd.read_csv(summary_path, dtype={"race_id": str}, encoding="utf-8-sig")
    odds_df = pd.read_csv(odds_path, dtype={"race_id": str}, encoding="utf-8-sig") if odds_path.exists() else None

    # 2. dummy -15 min odds 生成 (朝 8:00 odds ± 5% noise)
    rng = np.random.default_rng(RNG_SEED)

    race_ids = sorted(full_df["race_id"].unique().tolist())
    print(f"[INFO] {len(race_ids)} races detected for {DATE_STR}")

    all_results = []
    skip_reasons: dict[str, int] = {}
    severity_counts: dict[str, int] = {"none": 0, "minor": 0, "major": 0, "critical": 0}
    n_voted_after_7c = 0

    for race_id in race_ids:
        race_horses = full_df[full_df["race_id"] == race_id].copy()
        if len(race_horses) == 0:
            continue

        # race meta from summary
        meta_row = summary_df[summary_df["race_id"] == race_id]
        if len(meta_row) == 0:
            continue
        m = meta_row.iloc[0]
        race_meta = {
            "race_id": race_id,
            "race_name": str(m["race_name"]),
            "course": str(m["course"]),
            "condition": str(m["condition"]),
            "num_horses": int(m["num_horses"]),
            "distance": int(m["distance"]),
            "race_num": int(m["race_num"]),
        }

        # 戦略⑦案 C 判定
        skip, reason = is_strategy_7c_excluded(race_meta)
        if skip:
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        else:
            n_voted_after_7c += 1

        # 朝 8:00 ranking (top5)
        race_sorted = race_horses.sort_values("V15_score", ascending=False).reset_index(drop=True)
        morning_top5 = race_sorted.head(5)[["horse_num", "horse_name", "V15_score", "odds"]].to_dict("records")

        # dummy -15 min odds で simulated -15 min ranking
        df_for_overlay = race_sorted[["horse_num", "V15_score", "odds"]].rename(
            columns={"V15_score": "v15_prob"}
        ).copy()

        # apply noise to odds
        df_for_overlay["odds_15min"] = df_for_overlay["odds"].apply(
            lambda o: float(o) * (1.0 + rng.uniform(-NOISE_PCT, NOISE_PCT)) if pd.notna(o) and o > 1.0 else None
        )
        df_for_overlay["weight_diff"] = 0  # neutral mock

        recalc_df = overlay_race_ranking(df_for_overlay)
        recalc_sorted = recalc_df.sort_values("new_rank")
        recalc_top5 = recalc_sorted.head(5)[["horse_num", "v15_prob", "corrected_prob"]].to_dict("records")

        # 順位変動 計算
        morning_horse_order = [int(h["horse_num"]) for h in morning_top5]
        recalc_horse_order = [int(h["horse_num"]) for h in recalc_top5]

        # top-k 入替: 各順位で異なるか
        rank_changes = []
        for i in range(min(5, len(morning_horse_order), len(recalc_horse_order))):
            rank_changes.append(1 if morning_horse_order[i] != recalc_horse_order[i] else 0)

        severity = classify_severity(rank_changes)
        severity_counts[severity] += 1

        race_result = {
            "race_id": race_id,
            "race_meta": race_meta,
            "strategy_7c": {"skipped": skip, "reason": reason},
            "morning_top5": morning_top5,
            "simulated_15min_top5": recalc_top5,
            "rank_changes_per_position": rank_changes,
            "severity": severity,
            "note": "dummy -15 min odds = 朝 8:00 odds ± 5% noise (paper simulation only)",
        }
        all_results.append(race_result)

        # paper output 保存
        out_file = OUT_DIR / f"{race_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(race_result, f, indent=2, ensure_ascii=False, default=lambda o: float(o) if hasattr(o, "item") else str(o))

    # summary
    summary = {
        "date": DATE_STR,
        "total_races": len(race_ids),
        "voted_after_strategy_7c": n_voted_after_7c,
        "skip_reasons": skip_reasons,
        "severity_counts": severity_counts,
        "noise_pct": NOISE_PCT,
        "rng_seed": RNG_SEED,
        "generated_at": datetime.now().isoformat(),
        "note": "paper simulation only, V15 production 不変、 Discord 実発火なし、 G1 day blocked",
    }
    summary_file = OUT_DIR / "_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Victoria Mile focus (東京 11R)
    vm_id = "202605020811"
    vm_result = next((r for r in all_results if r["race_id"] == vm_id), None)
    if vm_result:
        print("\n" + "=" * 60)
        print("VICTORIA MILE (Tokyo 11R) focus:")
        print("=" * 60)
        print(json.dumps(vm_result, indent=2, ensure_ascii=False, default=lambda o: float(o) if hasattr(o, "item") else str(o)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
