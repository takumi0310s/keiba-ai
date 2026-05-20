"""V15 confidence score filter (paper only, ★ 最推奨 ★)

★ paper only — 実 cash は V15 のみ ★

confidence score 計算:
  confidence = top1_prob × (1 - top2_prob) × num_horses_factor

  num_horses_factor:
    ≤7頭 (条件E): 1.2 (少頭数は diff が出やすい)
    8-14頭 (標準): 1.0
    15頭+ (多頭数): 0.8 (混戦リスク)

  threshold grid: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
  - threshold 以下の confidence は skip

主要関数:
  compute_confidence(top1_prob, top2_prob, num_horses) -> float
  filter_by_confidence(confidence, threshold) -> bool (True=投票、False=skip)
  get_paper_formation(predictions_list, threshold, race_meta) -> str or None
    → None の場合は skip
  backtest_confidence(csv_path, thresholds) -> DataFrame
    → threshold ごとの n, hit3_rate, roi_pct, skip_rate
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_confidence(
    top1_prob: float,
    top2_prob: float,
    num_horses: int,
) -> float:
    """Compute confidence score for a race.

    Parameters
    ----------
    top1_prob:
        V15 predicted probability for top-1 horse (0~1).
    top2_prob:
        V15 predicted probability for top-2 horse (0~1).
    num_horses:
        Number of horses in the race.

    Returns
    -------
    float
        Confidence score in [0, 1].
    """
    if num_horses <= 7:
        factor = 1.2
    elif num_horses <= 14:
        factor = 1.0
    else:
        factor = 0.8

    raw = top1_prob * (1.0 - top2_prob) * factor
    # Clamp to [0, 1]
    return float(min(max(raw, 0.0), 1.0))


def filter_by_confidence(confidence: float, threshold: float) -> bool:
    """Return True (invest) when confidence > threshold, else False (skip).

    Parameters
    ----------
    confidence:
        Output of :func:`compute_confidence`.
    threshold:
        Minimum confidence to accept (exclusive).

    Returns
    -------
    bool
        True = 投票, False = skip.
    """
    return confidence > threshold


def get_paper_formation(
    predictions_list: list[dict],
    threshold: float,
    race_meta: dict,
) -> Optional[str]:
    """Return formation string or None (skip).

    Parameters
    ----------
    predictions_list:
        List of dicts with at least keys 'horse_num', 'prob'.
        Sorted descending by prob is assumed.
    threshold:
        Confidence threshold.
    race_meta:
        Dict with at least 'num_horses' (int) and optionally 'condition' (str).

    Returns
    -------
    str or None
        Formation string (e.g. "1-3-5; 1-3-7; ...") or None if skipped.
    """
    if len(predictions_list) < 2:
        return None

    top1_prob = float(predictions_list[0].get("prob", 0.0))
    top2_prob = float(predictions_list[1].get("prob", 0.0))
    num_horses = int(race_meta.get("num_horses", 14))

    conf = compute_confidence(top1_prob, top2_prob, num_horses)
    if not filter_by_confidence(conf, threshold):
        return None

    # Build simple formation placeholder using top horse numbers
    sorted_preds = sorted(predictions_list, key=lambda x: -x.get("prob", 0))
    top_nums = [str(int(p["horse_num"])) for p in sorted_preds[:6]]
    if len(top_nums) < 3:
        return None

    condition = race_meta.get("condition", "A")
    if condition == "E" and len(top_nums) >= 3:
        # umaren 2点
        return f"{top_nums[0]}-{top_nums[1]}; {top_nums[0]}-{top_nums[2]}"

    # trio 7点 フォーメーション
    lines = []
    col1 = top_nums[0]
    col2 = top_nums[1:3]
    col3 = top_nums[1:6]
    for b in col2:
        for c in col3:
            nums = sorted([int(col1), int(b), int(c)])
            combo = f"{nums[0]}-{nums[1]}-{nums[2]}"
            if combo not in lines and nums[0] != nums[1] and nums[1] != nums[2]:
                lines.append(combo)
    return "; ".join(lines[:7]) if lines else None


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest_confidence(
    csv_path: str,
    thresholds: list[float],
) -> pd.DataFrame:
    """Backtest confidence filter on cumulative_results.csv.

    Parameters
    ----------
    csv_path:
        Path to ``data/cumulative_results.csv``.
    thresholds:
        List of threshold values to evaluate.

    Returns
    -------
    pd.DataFrame
        Columns: threshold, split, n, hit3_rate, roi_pct, skip_rate,
                 total_payout, total_invest.
    """
    df = pd.read_csv(csv_path)

    # Normalise numeric columns
    df["date"] = pd.to_numeric(df["date"], errors="coerce")
    df["top1_score"] = pd.to_numeric(df["top1_score"], errors="coerce")
    df["top2_score"] = pd.to_numeric(df["top2_score"], errors="coerce")
    df["num_horses"] = pd.to_numeric(df["num_horses"], errors="coerce")
    df["actual_payout"] = pd.to_numeric(df["actual_payout"], errors="coerce").fillna(0)
    df["investment"] = pd.to_numeric(df["investment"], errors="coerce").fillna(700)
    df["trio_hit"] = pd.to_numeric(df["trio_hit"], errors="coerce").fillna(0)

    # Keep only rows with top1_score available and settled status
    scored = df[df["top1_score"].notna()].copy()

    # Infer num_horses from condition when missing
    def infer_horses(row: pd.Series) -> float:
        if pd.notna(row["num_horses"]):
            return row["num_horses"]
        cond = str(row.get("condition", ""))
        if cond == "E":
            return 6.0
        elif cond == "C" or cond == "X":
            return 16.0
        else:
            return 12.0

    scored["num_horses_eff"] = scored.apply(infer_horses, axis=1)

    # Compute confidence; use 0 for missing top2_score (conservative)
    scored["top2_score_eff"] = scored["top2_score"].fillna(0.0)
    scored["confidence"] = scored.apply(
        lambda r: compute_confidence(
            float(r["top1_score"]),
            float(r["top2_score_eff"]),
            int(r["num_horses_eff"]),
        ),
        axis=1,
    )

    # hold-out split: train=4/1-5/3, test=5/4-5/17
    TRAIN_END = 20260503
    TEST_START = 20260504
    TEST_END = 20260517

    train_mask = scored["date"] <= TRAIN_END
    test_mask = (scored["date"] >= TEST_START) & (scored["date"] <= TEST_END)

    total_n = len(scored)
    rows = []

    for thr in thresholds:
        for split_name, mask in [("train", train_mask), ("test", test_mask), ("all", pd.Series([True] * len(scored), index=scored.index))]:
            subset = scored[mask].copy()
            if subset.empty:
                rows.append(
                    dict(
                        threshold=thr,
                        split=split_name,
                        n=0,
                        hit3_rate=float("nan"),
                        roi_pct=float("nan"),
                        skip_rate=float("nan"),
                        total_payout=0,
                        total_invest=0,
                    )
                )
                continue

            accepted = subset[subset["confidence"] > thr]
            n_total_split = len(subset)
            n_accepted = len(accepted)
            skip_rate = 1.0 - (n_accepted / n_total_split) if n_total_split > 0 else float("nan")

            if n_accepted == 0:
                rows.append(
                    dict(
                        threshold=thr,
                        split=split_name,
                        n=0,
                        hit3_rate=float("nan"),
                        roi_pct=float("nan"),
                        skip_rate=skip_rate,
                        total_payout=0,
                        total_invest=0,
                    )
                )
                continue

            total_payout = accepted["actual_payout"].sum()
            total_invest = accepted["investment"].sum()
            roi_pct = (total_payout / total_invest * 100) if total_invest > 0 else float("nan")
            hit3_rate = (accepted["trio_hit"] > 0).mean() * 100

            rows.append(
                dict(
                    threshold=thr,
                    split=split_name,
                    n=n_accepted,
                    hit3_rate=round(hit3_rate, 1),
                    roi_pct=round(roi_pct, 1),
                    skip_rate=round(skip_rate * 100, 1),
                    total_payout=int(total_payout),
                    total_invest=int(total_invest),
                )
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = Path(__file__).parent.parent / "data" / "cumulative_results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    result = backtest_confidence(str(csv_path), thresholds)

    # Display pivot: train vs test side by side
    print("=" * 70)
    print("Confidence Score Filter Backtest")
    print("hold-out: train=4/1-5/3  |  test=5/4-5/17")
    print("=" * 70)

    for split in ["train", "test", "all"]:
        sub = result[result["split"] == split].reset_index(drop=True)
        print(f"\n[{split}]")
        print(
            f"{'thr':>5}  {'n':>5}  {'hit3%':>7}  {'ROI%':>7}  {'skip%':>7}"
        )
        print("-" * 40)
        for _, row in sub.iterrows():
            n = int(row["n"])
            hit = f"{row['hit3_rate']:.1f}" if n > 0 else " ---"
            roi = f"{row['roi_pct']:.1f}" if n > 0 else " ---"
            skip = f"{row['skip_rate']:.1f}" if not pd.isna(row['skip_rate']) else " ---"
            print(f"{row['threshold']:>5.2f}  {n:>5}  {hit:>7}  {roi:>7}  {skip:>7}")

    # Find best threshold on hold-out test ROI
    test_df = result[result["split"] == "test"].dropna(subset=["roi_pct"])
    if not test_df.empty:
        best_row = test_df.loc[test_df["roi_pct"].idxmax()]
        print()
        print("=" * 70)
        print(
            f"[REC] Recommended threshold (hold-out ROI max): {best_row['threshold']:.2f}"
        )
        print(
            f"  -> test n={int(best_row['n'])}, "
            f"hit3={best_row['hit3_rate']:.1f}%, "
            f"ROI={best_row['roi_pct']:.1f}%, "
            f"skip={best_row['skip_rate']:.1f}%"
        )
        # Warn if train-test gap > 20pt
        train_df = result[(result["split"] == "train") & (result["threshold"] == best_row["threshold"])]
        if not train_df.empty and not pd.isna(train_df.iloc[0]["roi_pct"]):
            gap = abs(best_row["roi_pct"] - train_df.iloc[0]["roi_pct"])
            flag = " gap OK (saiyou kouho)" if gap <= 20 else " [!] gap > 20pt (overfitting risk)"
            print(f"  train-test ROI gap: {gap:.1f}pt{flag}")
        print("=" * 70)
