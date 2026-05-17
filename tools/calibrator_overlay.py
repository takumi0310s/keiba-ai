"""
P0-5 順4: post-hoc calibrator overlay (-15 min recalc 用)

V15 prob は不変、 別 layer で odds_shift + weight_diff 等を微補正。
既存 calibrator_v15_pilot_v2.pkl (Isotonic、 315 sample retrain) framework 流用。

★ V15 production 投票判断 影響なし、 shadow eval 用 ★

Author: P0-5 順4 (commit 親集中、 本 file は新規)
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ORIG_CALIBRATOR = REPO / "data" / "calibrator_v15_pilot_v2.pkl"  # 315 sample retrain
OVERLAY_OUT = REPO / "data" / "v21" / "calibrator_overlay_v1.pkl"

# 補正 max delta cap (大変動防止)
DELTA_CAP = 0.10

# odds-based shift の weight (V15 prob と implied prob の divergence への係数)
ODDS_WEIGHT = 0.3

# 馬体重 補正 閾値
WEIGHT_LARGE = 15  # ±15kg 以上 = penalty -0.05
WEIGHT_MEDIUM = 10  # ±10〜14kg = penalty -0.02
WEIGHT_SQUEEZE_LOW = 2  # 適度減 2〜5kg = +0.01
WEIGHT_SQUEEZE_HIGH = 5
WEIGHT_PENALTY_LARGE = -0.05
WEIGHT_PENALTY_MEDIUM = -0.02
WEIGHT_BONUS_SQUEEZE = 0.01

# 除外 features (T4 LEAK 監査済、 commit 04cbfcd3)
# `*` suffix = prefix match
LEAK_FEATURES = {
    "paci_info_idx",      # +0.4139、 odds-related SUSPECT
    "jrdb_odds_idx",      # +0.4214 (TYB、 永久放棄)
    "jrdb_paddock_idx",   # post-race confirmed (TYB)
    "skb_kishi_code_*",   # V15.1 SKB leak (Session #38)
}


def is_safe_feature(name: str) -> bool:
    """LEAK 監査済 + safe feature 判定.

    Args:
        name: feature 名

    Returns:
        True = safe (overlay 投入 OK), False = leak / 除外
    """
    if not isinstance(name, str):
        return False
    for entry in LEAK_FEATURES:
        if entry.endswith("*"):
            if name.startswith(entry[:-1]):
                return False
        elif name == entry:
            return False
    return True


def compute_odds_shift(v15_prob: float, odds_15min: float | None) -> float:
    """V15 prob と -15 min direct odds の divergence で 補正量算出.

    Args:
        v15_prob: V15 朝 8:00 prob (0-1)
        odds_15min: -15 min direct odds (e.g. 3.5)

    Returns:
        delta (clipped at ±DELTA_CAP)
    """
    if odds_15min is None or pd.isna(odds_15min) or odds_15min <= 1.0:
        return 0.0

    # implied prob from odds (rough estimate、 controle margin は無視)
    implied = 1.0 / float(odds_15min)

    # V15 prob と implied の差 (positive = market expects higher than V15)
    raw_delta = (implied - float(v15_prob)) * ODDS_WEIGHT

    return float(np.clip(raw_delta, -DELTA_CAP, DELTA_CAP))


def compute_weight_shift(weight_diff: float | None) -> float:
    """馬体重 増減 で 補正量算出.

    Args:
        weight_diff: 馬体重 変化 (-20 〜 +20 kg、 通常 ±10 まで)

    Returns:
        delta (clipped)
    """
    if weight_diff is None or pd.isna(weight_diff):
        return 0.0

    abs_diff = abs(float(weight_diff))
    if abs_diff >= WEIGHT_LARGE:
        return WEIGHT_PENALTY_LARGE  # -0.05
    elif abs_diff >= WEIGHT_MEDIUM:
        return WEIGHT_PENALTY_MEDIUM  # -0.02
    elif WEIGHT_SQUEEZE_LOW <= abs_diff <= WEIGHT_SQUEEZE_HIGH and float(weight_diff) < 0:
        # 「適度な絞り」 = 体重 減 のみ positive (+0.01)
        return WEIGHT_BONUS_SQUEEZE
    else:
        return 0.0


def apply_overlay(
    v15_prob: float | np.ndarray,
    odds_15min: float | None = None,
    weight_diff: float | None = None,
) -> float | np.ndarray:
    """post-hoc calibrator overlay 適用.

    V15 prob は不変、 別 layer で odds_15min + weight_diff の delta を加算。

    Args:
        v15_prob: V15 朝 8:00 prob (single value or array)
        odds_15min: -15 min direct odds (optional)
        weight_diff: 馬体重 変化 (optional)

    Returns:
        corrected_prob (V15 prob ± overlay、 cap ±DELTA_CAP, bounds [0.001, 0.999])
    """
    scalar = not isinstance(v15_prob, np.ndarray)
    arr = np.asarray([v15_prob], dtype=float) if scalar else v15_prob.astype(float)

    out = np.empty_like(arr)
    for i, p in enumerate(arr):
        delta_odds = compute_odds_shift(p, odds_15min) if odds_15min is not None else 0.0
        delta_weight = compute_weight_shift(weight_diff) if weight_diff is not None else 0.0
        total_delta = float(np.clip(delta_odds + delta_weight, -DELTA_CAP, DELTA_CAP))
        out[i] = float(np.clip(p + total_delta, 0.001, 0.999))

    if scalar:
        return float(out[0])
    return out


def overlay_race_ranking(race_horses_df: pd.DataFrame) -> pd.DataFrame:
    """1 race の全頭に overlay を 適用、 新 ranking 返す.

    Args:
        race_horses_df: DataFrame with columns:
            - 'horse_num' (required)
            - 'v15_prob'  (required)
            - 'odds_15min' (optional)
            - 'weight_diff' (optional)

    Returns:
        DataFrame with added 'corrected_prob' + 'new_rank' columns.
        Original V15 prob 不変。
    """
    df = race_horses_df.copy()
    if "v15_prob" not in df.columns:
        raise ValueError("race_horses_df must have 'v15_prob' column")

    def _apply(row):
        return apply_overlay(
            v15_prob=row["v15_prob"],
            odds_15min=row.get("odds_15min") if "odds_15min" in df.columns else None,
            weight_diff=row.get("weight_diff") if "weight_diff" in df.columns else None,
        )

    df["corrected_prob"] = df.apply(_apply, axis=1)
    df["new_rank"] = df["corrected_prob"].rank(ascending=False, method="min").astype(int)
    return df


def load_existing_calibrator() -> dict[str, Any] | None:
    """既存 calibrator_v15_pilot_v2.pkl framework load (★ read-only ★).

    Returns:
        dict (isotonic + platt + metrics) or None if not exists.
    """
    if not ORIG_CALIBRATOR.exists():
        return None
    with open(ORIG_CALIBRATOR, "rb") as f:
        return pickle.load(f)


def build_overlay_state() -> dict[str, Any]:
    """overlay state dict 構築 (param + meta).

    Returns:
        state dict (version, thresholds, leak features list)
    """
    return {
        "version": "overlay_v1",
        "delta_cap": DELTA_CAP,
        "odds_weight": ODDS_WEIGHT,
        "leak_features": sorted(LEAK_FEATURES),
        "weight_diff_thresholds": {
            "large_kg": WEIGHT_LARGE,
            "medium_kg": WEIGHT_MEDIUM,
            "squeeze_low_kg": WEIGHT_SQUEEZE_LOW,
            "squeeze_high_kg": WEIGHT_SQUEEZE_HIGH,
            "penalty_large": WEIGHT_PENALTY_LARGE,
            "penalty_medium": WEIGHT_PENALTY_MEDIUM,
            "bonus_squeeze": WEIGHT_BONUS_SQUEEZE,
        },
        "note": "V15 prob 不変、 別 layer overlay。 V15 production 投票判断 影響なし。",
    }


def save_overlay_state(state: dict[str, Any], path: Path | None = None) -> Path:
    """overlay state 保存 (新規 pkl、 既存 calibrator は touch せず)."""
    target = path if path is not None else OVERLAY_OUT
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        pickle.dump(state, f)
    return target


def simulate_on_history(date_str: str, dry_run: bool = True) -> dict[str, Any]:
    """過去 race で overlay 適用 simulation (★ read-only ★).

    daily_predictions/{date}.csv + odds_base_{date}.csv で
    overlay 適用後の ranking 変動 を 計測。

    Args:
        date_str: YYYYMMDD
        dry_run: True なら 何も save しない

    Returns:
        result dict (n_races, n_ranking_change, mean_delta, ...)
    """
    daily_pred = REPO / "data" / "daily_predictions" / f"{date_str}.csv"
    odds_base = REPO / "data" / f"odds_base_{date_str}.csv"
    cumulative = REPO / "data" / "cumulative_results.csv"

    if not daily_pred.exists():
        return {"error": f"no daily_predictions for {date_str}", "date": date_str}

    pred_df = pd.read_csv(daily_pred)

    odds_df = None
    if odds_base.exists():
        try:
            odds_df = pd.read_csv(odds_base)
        except Exception as e:
            odds_df = None

    # cumulative results join (for AUC measurement post-hoc)
    cum_df = None
    if cumulative.exists():
        try:
            cum_df = pd.read_csv(cumulative, low_memory=False)
            cum_df = cum_df[cum_df.get("date").astype(str) == str(date_str)] if "date" in cum_df.columns else None
        except Exception:
            cum_df = None

    n_races = len(pred_df) if pred_df is not None else 0
    n_ranking_change = 0
    deltas = []

    for _, race in pred_df.iterrows():
        race_id = race.get("race_id")
        if pd.isna(race_id):
            continue
        race_id_str = str(int(race_id)) if isinstance(race_id, (int, float)) else str(race_id)

        # top1 prob と corrected prob
        v15_prob = race.get("top1_score")
        if pd.isna(v15_prob):
            continue

        # 仮想 -15 min odds: odds_base から top1 horse num の odds を取得
        odds_15min = None
        top1_num = race.get("top1_num")
        if odds_df is not None and not pd.isna(top1_num):
            try:
                top1_num_int = int(top1_num)
                race_odds = odds_df[
                    (odds_df["race_id"].astype(str) == race_id_str)
                    & (odds_df["horse_num"] == top1_num_int)
                ]
                if len(race_odds) > 0:
                    odds_15min = float(race_odds.iloc[0]["odds"])
            except Exception:
                odds_15min = None

        corrected = apply_overlay(float(v15_prob), odds_15min=odds_15min, weight_diff=None)
        delta = corrected - float(v15_prob)
        deltas.append(delta)

        if abs(delta) > 1e-6:
            n_ranking_change += 1

    result = {
        "date": date_str,
        "n_races": n_races,
        "n_with_delta": n_ranking_change,
        "mean_abs_delta": float(np.mean(np.abs(deltas))) if deltas else 0.0,
        "max_abs_delta": float(np.max(np.abs(deltas))) if deltas else 0.0,
        "delta_cap_respected": bool(all(abs(d) <= DELTA_CAP + 1e-9 for d in deltas)),
        "dry_run": dry_run,
        "has_odds_base": odds_df is not None,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P0-5 calibrator_overlay (V15 不変、 shadow eval 用)")
    ap.add_argument("--simulate", help="YYYYMMDD で 過去 race simulation")
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--build-state", action="store_true", help="overlay state pkl 構築")
    args = ap.parse_args(argv)

    if args.build_state:
        state = build_overlay_state()
        out = save_overlay_state(state)
        print(f"[OK] saved {out}")
        print(f"     keys: {list(state.keys())}")
        return 0

    if args.simulate:
        result = simulate_on_history(args.simulate, dry_run=args.dry_run)
        for k, v in result.items():
            print(f"  {k}: {v}")
        return 0

    print("usage: --simulate YYYYMMDD or --build-state")
    return 0


if __name__ == "__main__":
    sys.exit(main())
