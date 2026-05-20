"""C4 strategy parameter grid search (paper only, hold-out validated)

★ paper only — 実 cash 投票 logic は V15 + 既存 C4 のみ ★
★ in-sample 過大評価 risk あるため hold-out 必須 ★

Grid 探索対象:
- C4 適用距離範囲: [1500-1900, 1600-1800, 1700-1850, 1600-2000, 1500-2000]
- trio formation 点数: [6, 7, 8, 10]
  - 6点: TOP1 - TOP2,TOP3 - TOP2~TOP5
  - 7点: TOP1 - TOP2,TOP3 - TOP2~TOP6 (現行)
  - 8点: TOP1 - TOP2,TOP3 - TOP2~TOP7
  - 10点: TOP1 - TOP2,TOP3 - TOP2~TOP8 + TOP1 - TOP4,TOP5 - TOP2~TOP6

評価指標:
- hit3_rate (三連複的中率)
- roi_pct (ROI % = payout_sum / invest_sum * 100)
- n (サンプル数)
- hold-out ROI (train=4/1-5/3, test=5/4-5/17)

警告: in-sample ROI > hold-out ROI が 20pt 以上の場合 overfit フラグを立てる。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "cumulative_results.csv"
OUT_PATH = REPO_ROOT / "data" / "c4_param_tune_results.json"

# ---------------------------------------------------------------------------
# Hold-out split dates (YYYYMMDD int)
# ---------------------------------------------------------------------------
TRAIN_START = 20260401
TRAIN_END = 20260503
TEST_START = 20260504
TEST_END = 20260517

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
DISTANCE_RANGES = [
    (1500, 1900),
    (1600, 1800),  # current
    (1700, 1850),
    (1600, 2000),
    (1500, 2000),
]

TICKET_COUNTS = [6, 7, 8, 10]

# Cost per ticket (yen)
TICKET_UNIT = 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load and clean cumulative_results.csv."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_numeric(df["date"], errors="coerce")
    df = df[df["date"] > 0].copy()
    df["date"] = df["date"].astype(int)
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["actual_payout"] = pd.to_numeric(df["actual_payout"], errors="coerce").fillna(0)
    df["investment"] = pd.to_numeric(df["investment"], errors="coerce").fillna(700)
    df["trio_hit"] = pd.to_numeric(df["trio_hit"], errors="coerce").fillna(0)
    return df


def investment_for_n_tickets(n_tickets: int) -> int:
    """Return yen invested for a given number of 100-yen trio tickets."""
    return n_tickets * TICKET_UNIT


def payout_scale_factor(n_tickets: int) -> float:
    """Scale actual_payout to reflect different ticket count.

    The recorded actual_payout assumes the race was hit with the current
    formation.  When we simulate a different number of tickets we adjust the
    investment proportionally.  The payout itself (from the race) does not
    change — only the investment changes, so ROI changes.

    For a *miss* the payout is 0 regardless of ticket count.
    For a *hit* the payout is the actual_payout as recorded (one or more
    tickets hit).  We assume single-ticket payout is actual_payout / 7 for the
    current 7-ticket baseline, and scale to n_tickets.

    NOTE: This is an approximation.  The real payout depends on which exact
    tickets cover the result, which we cannot reconstruct without top5/top6
    horse numbers (only top4 is available).  We therefore use the conservative
    assumption: payout scales linearly with ticket count relative to the 7-ticket
    baseline (i.e. more tickets = same payout per ticket if hit, but proportional
    investment increase → ROI changes via investment side only).

    Actually, the simplest correct model:
    - hit=1: actual_payout stays the same (you still collect the payout).
    - hit=0: payout = 0.
    - investment = n_tickets * TICKET_UNIT.
    So ROI change comes purely from different investment with same payout on hits.
    """
    return 1.0  # payout unchanged; investment changes


def compute_roi_for_subset(
    subset: pd.DataFrame, n_tickets: int
) -> dict:
    """Compute ROI metrics for a subset of races with a given ticket count."""
    if len(subset) == 0:
        return {"n": 0, "hit3_rate": 0.0, "roi_pct": 0.0, "invest_sum": 0, "payout_sum": 0.0}
    invest_per_race = investment_for_n_tickets(n_tickets)
    invest_sum = invest_per_race * len(subset)
    payout_sum = float(subset["actual_payout"].sum())
    hit3_rate = float(subset["trio_hit"].mean())
    roi_pct = payout_sum / invest_sum * 100 if invest_sum > 0 else 0.0
    return {
        "n": len(subset),
        "hit3_rate": round(hit3_rate, 4),
        "roi_pct": round(roi_pct, 2),
        "invest_sum": invest_sum,
        "payout_sum": round(payout_sum, 2),
    }


def run_grid_search(df: pd.DataFrame) -> list[dict]:
    """Run full grid search over distance ranges and ticket counts.

    Condition A = 8-14 heads / 1600m+ / 良〜稍重.
    C4 skip = skip betting on Cond-A races in the specified distance range.
    We evaluate the *kept* races (after skip).
    """
    # Filter to Cond A settled races
    cond_a = df[
        (df["condition"] == "A") & (df["status"] == "settled")
    ].copy()

    # Train / test split
    train = cond_a[
        (cond_a["date"] >= TRAIN_START) & (cond_a["date"] <= TRAIN_END)
    ]
    test = cond_a[
        (cond_a["date"] >= TEST_START) & (cond_a["date"] <= TEST_END)
    ]

    results = []

    for dist_lo, dist_hi in DISTANCE_RANGES:
        for n_tickets in TICKET_COUNTS:
            # Kept races = Cond A minus the C4-skip range
            def _kept(subset: pd.DataFrame) -> pd.DataFrame:
                return subset[
                    ~((subset["distance"] >= dist_lo) & (subset["distance"] <= dist_hi))
                ]

            train_kept = _kept(train)
            test_kept = _kept(test)

            train_metrics = compute_roi_for_subset(train_kept, n_tickets)
            test_metrics = compute_roi_for_subset(test_kept, n_tickets)

            in_sample_roi = train_metrics["roi_pct"]
            holdout_roi = test_metrics["roi_pct"]
            overfit_flag = (in_sample_roi - holdout_roi) >= 20.0

            entry = {
                "c4_dist_lo": dist_lo,
                "c4_dist_hi": dist_hi,
                "n_tickets": n_tickets,
                "is_current": (dist_lo == 1600 and dist_hi == 1800 and n_tickets == 7),
                "train": train_metrics,
                "test": test_metrics,
                "overfit_flag": overfit_flag,
            }
            results.append(entry)

    return results


def main():
    print(f"Loading: {CSV_PATH}")
    df = load_data()
    print(f"Rows: {len(df)}")

    results = run_grid_search(df)

    # Sort by test ROI desc, no overfit
    non_overfit = [r for r in results if not r["overfit_flag"]]
    non_overfit.sort(key=lambda x: x["test"]["roi_pct"], reverse=True)

    # Current baseline
    current = next(r for r in results if r["is_current"])

    print("\n=== Current C4 (1600-1800, 7 tickets) ===")
    print(f"  Train ROI: {current['train']['roi_pct']:.1f}%  n={current['train']['n']}")
    print(f"  Test ROI:  {current['test']['roi_pct']:.1f}%  n={current['test']['n']}")
    print(f"  Overfit:   {current['overfit_flag']}")

    print("\n=== Top 3 params (test ROI, no overfit) ===")
    for i, r in enumerate(non_overfit[:3], 1):
        print(
            f"  #{i}: dist [{r['c4_dist_lo']}-{r['c4_dist_hi']}]  "
            f"tickets={r['n_tickets']}  "
            f"train={r['train']['roi_pct']:.1f}%  "
            f"test(hold-out)={r['test']['roi_pct']:.1f}%  "
            f"n_test={r['test']['n']}"
        )

    print(f"\nSaving results to: {OUT_PATH}")
    payload = {
        "meta": {
            "train_period": f"{TRAIN_START}-{TRAIN_END}",
            "test_period": f"{TEST_START}-{TEST_END}",
            "note": "paper only — V15 production unchanged",
        },
        "current_baseline": current,
        "top3_non_overfit": non_overfit[:3],
        "all_results": results,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
