"""Paper trade engine for V15 vs V18 vs V20 parallel evaluation.

V15 production prediction is the only real-money path. V18 / V20 are paper-only:
predictions are computed in shadow mode and recorded for ROI comparison.

Outputs:
  data/paper_trade/v15_YYYYMMDD.csv  (mirror of daily_predictions/YYYYMMDD.csv)
  data/paper_trade/v18_YYYYMMDD.csv  (V18 sib_w5 shadow predictions)
  data/paper_trade/v20_YYYYMMDD.csv  (V20 PoC shadow predictions; placeholder until V20 4-model trained)
  data/paper_trade/summary_YYYYMMDD.csv (V15 vs V18 vs V20 hit/ROI per race)
  data/paper_trade/summary_rolling.csv (cumulative across dates)

V15 production is never modified by this script. The engine is read-only with
respect to predict_core / daily_predict / app.py.

Usage:
    python tools/paper_trade_engine.py --date 20260510
    python tools/paper_trade_engine.py --date 20260510 --models v15,v18
    python tools/paper_trade_engine.py --rolling
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_DIR = BASE_DIR / "data" / "daily_predictions"
RESULTS_DIR = BASE_DIR / "data" / "daily_results"
PAPER_DIR = BASE_DIR / "data" / "paper_trade"
V18_MODEL_DIR = BASE_DIR / "data" / "v18" / "v18v19_sib_exp_w5"

INVESTMENT_PER_RACE = 700  # 円, V15 baseline


def ensure_dirs():
    PAPER_DIR.mkdir(parents=True, exist_ok=True)


def load_v15_predictions(date_str: str) -> pd.DataFrame:
    path = PRED_DIR / f"{date_str}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id": str})
    df["model"] = "v15"
    return df


def load_v18_shadow_predictions(date_str: str) -> pd.DataFrame:
    """V18 shadow predictions for the date.

    Source 1: data/v18/v18v19_sib_exp_w5/v18_oos_YYYYMMDD.csv (per-race retro output)
    Source 2: data/paper_trade/v18_YYYYMMDD.csv (already-recorded shadow)
    Returns empty DataFrame if neither exists.
    """
    candidates = [
        PAPER_DIR / f"v18_{date_str}.csv",
        V18_MODEL_DIR / f"v18_oos_{date_str}.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id": str})
            df["model"] = "v18"
            return df
    return pd.DataFrame()


def load_v20_shadow_predictions(date_str: str) -> pd.DataFrame:
    """V20 shadow predictions for the date.

    Until the 4-model ensemble is trained, returns empty (placeholder).
    """
    path = PAPER_DIR / f"v20_{date_str}.csv"
    if path.exists():
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id": str})
        df["model"] = "v20"
        return df
    return pd.DataFrame()


def load_results(date_str: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{date_str}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id": str})


def _parse_trio_bets(bets_str: str):
    if not isinstance(bets_str, str) or not bets_str.strip():
        return []
    out = []
    for combo in bets_str.split(";"):
        nums = [s.strip() for s in combo.strip().split("-") if s.strip()]
        if len(nums) == 3:
            try:
                out.append(tuple(sorted(int(n) for n in nums)))
            except ValueError:
                continue
    return out


def _trio_hit(bets, finish_top3):
    if not bets or not finish_top3 or len(finish_top3) < 3:
        return False
    target = tuple(sorted(int(n) for n in finish_top3[:3]))
    return target in bets


def _parse_trio_result(s: str) -> tuple:
    """daily_results.csv の trio_result (例: '3-7-11') を sorted tuple に変換。"""
    if not isinstance(s, str) or not s.strip():
        return ()
    nums = []
    for tok in s.replace(",", "-").split("-"):
        tok = tok.strip()
        if not tok:
            continue
        try:
            nums.append(int(tok))
        except ValueError:
            return ()
    return tuple(sorted(nums)) if len(nums) == 3 else ()


def evaluate_predictions(pred_df: pd.DataFrame, results_df: pd.DataFrame, model: str = "v15") -> pd.DataFrame:
    """予測 df を results df と突合し hit / payout / pnl を付与。

    V15 は daily_results 既存 trio_hit / trio_payout を信頼 (production 一致)。
    V18 / V20 は trio_result を真値とし、 paper bets で再判定。
    """
    if pred_df.empty:
        return pd.DataFrame()
    if results_df.empty:
        out = pred_df.copy()
        out["hit"] = pd.NA
        out["payout"] = pd.NA
        out["pnl"] = pd.NA
        return out

    res = results_df.set_index("race_id") if "race_id" in results_df.columns else pd.DataFrame()
    rows = []
    for _, p in pred_df.iterrows():
        rid = str(p.get("race_id", ""))
        bets = _parse_trio_bets(p.get("trio_bets", ""))
        bet_type = p.get("bet_type", "trio")
        invest = float(p.get("investment", INVESTMENT_PER_RACE))

        hit = pd.NA
        payout = 0.0
        if rid in res.index:
            r = res.loc[rid]
            settled = str(r.get("status", "")).lower() == "settled"
            trio_result = _parse_trio_result(str(r.get("trio_result", ""))) if "trio_result" in r.index else ()

            if model == "v15" and "trio_hit" in r.index and pd.notna(r["trio_hit"]):
                hit = bool(int(r["trio_hit"]))
                if hit and "trio_payout" in r.index and pd.notna(r["trio_payout"]):
                    raw_payout = float(r["trio_payout"])
                    payout = raw_payout * (invest / max(len(bets), 1) / 100.0) if bets else raw_payout
                if not settled and not hit and not trio_result:
                    hit = pd.NA
            elif bet_type == "trio":
                if not trio_result:
                    hit = pd.NA
                else:
                    hit = trio_result in bets
                    if hit and "trio_payout" in r.index and pd.notna(r["trio_payout"]):
                        raw_payout = float(r["trio_payout"])
                        payout = raw_payout * (invest / max(len(bets), 1) / 100.0)
            elif bet_type == "umaren":
                top2 = tuple(sorted(trio_result[:2])) if trio_result else ()
                ubets = [tuple(sorted(b[:2])) for b in bets if len(b) >= 2]
                hit = top2 in ubets if top2 else pd.NA
                if hit is True and "umaren_payout" in r.index and pd.notna(r["umaren_payout"]):
                    raw_payout = float(r["umaren_payout"])
                    payout = raw_payout * (invest / max(len(ubets), 1) / 100.0)

        if hit is True:
            pnl = payout - invest
        elif hit is False:
            pnl = -invest
        else:
            pnl = pd.NA
        rows.append({
            **p.to_dict(),
            "hit": hit,
            "payout": payout if hit is True else 0.0,
            "pnl": pnl,
        })
    return pd.DataFrame(rows)


def summarize(eval_df: pd.DataFrame, model: str) -> dict:
    if eval_df.empty:
        return {"model": model, "n_races": 0, "n_hits": 0, "hit_rate": None,
                "investment": 0, "payout": 0, "pnl": 0, "roi_pct": None}
    settled = eval_df.dropna(subset=["hit"])
    n = len(settled)
    if n == 0:
        return {"model": model, "n_races": len(eval_df), "n_hits": 0, "hit_rate": None,
                "investment": float(eval_df["investment"].sum()) if "investment" in eval_df else 0,
                "payout": 0, "pnl": 0, "roi_pct": None}
    hits = int(settled["hit"].astype(bool).sum())
    invest = float(settled["investment"].sum()) if "investment" in settled else n * INVESTMENT_PER_RACE
    payout = float(settled["payout"].sum())
    pnl = payout - invest
    roi = (payout / invest * 100.0) if invest > 0 else None
    return {
        "model": model,
        "n_races": n,
        "n_hits": hits,
        "hit_rate": hits / n if n else None,
        "investment": invest,
        "payout": payout,
        "pnl": pnl,
        "roi_pct": roi,
    }


def run_for_date(date_str: str, models=("v15", "v18", "v20")):
    ensure_dirs()
    results = load_results(date_str)
    summaries = []
    per_model_eval = {}

    if "v15" in models:
        v15 = load_v15_predictions(date_str)
        ev15 = evaluate_predictions(v15, results, model="v15")
        if not ev15.empty:
            ev15.to_csv(PAPER_DIR / f"v15_{date_str}.csv", index=False, encoding="utf-8-sig")
        per_model_eval["v15"] = ev15
        summaries.append(summarize(ev15, "v15"))

    if "v18" in models:
        v18 = load_v18_shadow_predictions(date_str)
        ev18 = evaluate_predictions(v18, results, model="v18")
        if not ev18.empty:
            ev18.to_csv(PAPER_DIR / f"v18_eval_{date_str}.csv", index=False, encoding="utf-8-sig")
        per_model_eval["v18"] = ev18
        summaries.append(summarize(ev18, "v18"))

    if "v20" in models:
        v20 = load_v20_shadow_predictions(date_str)
        ev20 = evaluate_predictions(v20, results, model="v20")
        if not ev20.empty:
            ev20.to_csv(PAPER_DIR / f"v20_eval_{date_str}.csv", index=False, encoding="utf-8-sig")
        per_model_eval["v20"] = ev20
        summaries.append(summarize(ev20, "v20"))

    summary_df = pd.DataFrame(summaries)
    summary_df["date"] = date_str
    summary_df.to_csv(PAPER_DIR / f"summary_{date_str}.csv", index=False, encoding="utf-8-sig")

    rolling_path = PAPER_DIR / "summary_rolling.csv"
    if rolling_path.exists():
        rolling = pd.read_csv(rolling_path, encoding="utf-8-sig")
        rolling = rolling[rolling["date"] != date_str]
        rolling = pd.concat([rolling, summary_df], ignore_index=True)
    else:
        rolling = summary_df.copy()
    rolling.to_csv(rolling_path, index=False, encoding="utf-8-sig")

    return summary_df, per_model_eval


def aggregate_rolling():
    rolling_path = PAPER_DIR / "summary_rolling.csv"
    if not rolling_path.exists():
        return pd.DataFrame()
    rolling = pd.read_csv(rolling_path, encoding="utf-8-sig")
    grouped = rolling.groupby("model").agg(
        n_dates=("date", "nunique"),
        n_races=("n_races", "sum"),
        n_hits=("n_hits", "sum"),
        investment=("investment", "sum"),
        payout=("payout", "sum"),
        pnl=("pnl", "sum"),
    ).reset_index()
    grouped["hit_rate"] = grouped["n_hits"] / grouped["n_races"].replace(0, pd.NA)
    grouped["roi_pct"] = grouped["payout"] / grouped["investment"].replace(0, pd.NA) * 100.0
    return grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--models", default="v15,v18,v20")
    ap.add_argument("--rolling", action="store_true")
    args = ap.parse_args()

    if args.rolling:
        agg = aggregate_rolling()
        if agg.empty:
            print("[INFO] no rolling history yet")
        else:
            print(agg.to_string(index=False))
        return

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    summary_df, _ = run_for_date(args.date, models=models)
    print(f"=== paper_trade summary for {args.date} ===")
    if summary_df.empty:
        print("[INFO] nothing to summarize")
    else:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
