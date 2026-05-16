"""
V15 calibrator retrain from joined daily_predictions/ + cumulative_results.csv

bottleneck: 既存 calibrator_v15_pilot.pkl は 21 sample のみで isotonic 飽和。
            cumulative_results.csv top1_score は 95% 欠損 (542/563)。
            だが daily_predictions/YYYYMMDD.csv には top1_score 完備。

解決: race_id 単位で join + (top1_score, top1_finish<=3) を train pairs に。
出力: data/calibrator_v15_pilot_v2.pkl (元 file は touch しない)

V15 production 完全不変。 calibrator file は新規 (production が読むのは _pilot.pkl のまま)。
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


REPO = Path(__file__).resolve().parents[2]
DAILY_PRED_DIR = REPO / "data" / "daily_predictions"
DAILY_RES_DIR = REPO / "data" / "daily_results"
OUT_PKL = REPO / "data" / "calibrator_v15_pilot_v2.pkl"
OUT_REPORT = REPO / "data" / "v21" / "calibrator_v15_retrain_report.json"
ORIG_PKL = REPO / "data" / "calibrator_v15_pilot.pkl"


def load_train_pairs() -> pd.DataFrame:
    """daily_predictions (V15 top1_score) × daily_results (V15 top1_finish) で (top1_score, label) 構築"""
    pred_frames = []
    for csv in sorted(DAILY_PRED_DIR.glob("2026*.csv")):
        if "nar" in csv.name or "prerace" in csv.name:
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        df["race_id"] = df["race_id"].astype(str)
        df = df[["race_id", "top1_num", "top1_score"]].copy()
        df["src_date"] = csv.stem
        pred_frames.append(df)
    pred = pd.concat(pred_frames, ignore_index=True)
    print(f"[INFO] daily_predictions: {len(pred)} races from {pred['src_date'].nunique()} dates")

    res_frames = []
    for csv in sorted(DAILY_RES_DIR.glob("2026*.csv")):
        if "nar" in csv.name or "payouts" in csv.name:
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if "top1_finish" not in df.columns:
            print(f"[SKIP] {csv.name}: no top1_finish column")
            continue
        df["race_id"] = df["race_id"].astype(str)
        cols = ["race_id", "top1_finish"] + (["status"] if "status" in df.columns else [])
        df = df[cols].copy()
        if "status" not in df.columns:
            df["status"] = "settled"
        res_frames.append(df)
    res = pd.concat(res_frames, ignore_index=True)
    res = res[res["status"] == "settled"].copy()
    res["top1_finish_num"] = pd.to_numeric(res["top1_finish"], errors="coerce")
    res["label"] = (res["top1_finish_num"] <= 3).astype(int)
    print(f"[INFO] daily_results settled: {len(res)} races")

    merged = pred.merge(res, on="race_id", how="inner")
    merged["top1_score"] = pd.to_numeric(merged["top1_score"], errors="coerce")
    merged = merged.dropna(subset=["top1_score", "top1_finish_num"])
    print(f"[INFO] merged (pred × res): {len(merged)} clean pairs")

    return merged[["race_id", "src_date", "top1_score", "label", "top1_finish_num"]]


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    n = len(p)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(e)


def main():
    pairs = load_train_pairs()
    n = len(pairs)
    print(f"[INFO] train pairs: {n}")
    if n < 30:
        print(f"[WARN] sample too small (n={n}), abort")
        return

    x = pairs["top1_score"].to_numpy(dtype=float)
    y = pairs["label"].to_numpy(dtype=int)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x, y)
    platt = LogisticRegression(max_iter=1000)
    platt.fit(x.reshape(-1, 1), y)

    iso_p = iso.predict(x)
    platt_p = platt.predict_proba(x.reshape(-1, 1))[:, 1]

    metrics = {
        "before": {"brier": brier(x, y), "ece": ece(x, y)},
        "after_iso": {"brier": brier(iso_p, y), "ece": ece(iso_p, y)},
        "after_platt": {"brier": brier(platt_p, y), "ece": ece(platt_p, y)},
    }

    test_pts = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.70, 0.95])
    test_iso = iso.predict(test_pts).tolist()
    test_platt = platt.predict_proba(test_pts.reshape(-1, 1))[:, 1].tolist()

    out = {
        "isotonic": iso,
        "platt": platt,
        "metrics": metrics,
        "n_samples": int(n),
        "trained_at": pd.Timestamp.now().isoformat(),
        "source": "daily_predictions × cumulative_results inner-join",
        "label": "top1_finish <= 3",
    }
    with open(OUT_PKL, "wb") as f:
        pickle.dump(out, f)
    print(f"[INFO] saved: {OUT_PKL}")

    report = {
        "n_samples": int(n),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "pos_rate": float(y.mean()),
        "top1_score_stats": {
            "min": float(x.min()),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "std": float(x.std()),
        },
        "metrics": metrics,
        "test_points": [float(p) for p in test_pts],
        "iso_predictions": test_iso,
        "platt_predictions": test_platt,
        "trained_at": out["trained_at"],
        "date_coverage": sorted(pairs["src_date"].unique().tolist()),
    }
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[INFO] report: {OUT_REPORT}")

    if ORIG_PKL.exists():
        with open(ORIG_PKL, "rb") as f:
            orig = pickle.load(f)
        print("\n=== compare vs original calibrator ===")
        print(f"  orig n_samples: {orig.get('n_samples', 'N/A')}")
        print(f"  v2   n_samples: {n}")
        print(f"  orig iso(0.30) = {orig['isotonic'].predict(np.array([0.30]))[0]:.4f}")
        print(f"  v2   iso(0.30) = {iso.predict(np.array([0.30]))[0]:.4f}")
        print(f"  orig iso(0.20) = {orig['isotonic'].predict(np.array([0.20]))[0]:.4f}")
        print(f"  v2   iso(0.20) = {iso.predict(np.array([0.20]))[0]:.4f}")
        print(f"  orig iso(0.15) = {orig['isotonic'].predict(np.array([0.15]))[0]:.4f}")
        print(f"  v2   iso(0.15) = {iso.predict(np.array([0.15]))[0]:.4f}")

    print(f"\nSummary: n={n}, pos_rate={y.mean():.3f}, brier_iso={metrics['after_iso']['brier']:.4f}")


if __name__ == "__main__":
    main()
