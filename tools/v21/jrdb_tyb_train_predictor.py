"""
JRDB TYB top3 predictor 学習 + 保存

V15 top1_score + TYB 17 features → top3 hit binary を logistic regression で 学習。
strategy_layer_v2 の calibrator 代替 (or 補強) として 利用可能。

V15 production 完全不変。 新規 model file は data/tyb_top3_predictor.pkl。
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]
DAILY_PRED_DIR = REPO / "data" / "daily_predictions"
DAILY_RES_DIR = REPO / "data" / "daily_results"
TYB_CSV = REPO / "data" / "jrdb_tyb.csv"
OUT_PKL = REPO / "data" / "tyb_top3_predictor.pkl"
OUT_JSON = REPO / "data" / "v21" / "tyb_top3_predictor_report.json"

FEATURE_COLS = ['top1_score',
                'idm', 'jockey_idx', 'info_idx', 'odds_idx', 'padock_idx', 'sogo_idx',
                'bagu_change', 'ashimoto', 'cancel_flag',
                'tansho_odds', 'fukusho_odds', 'horse_weight', 'weight_diff',
                'padock_mark', 'kehai_code', 'baba_code']


def load_data() -> pd.DataFrame:
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
        pred_frames.append(df)
    pred = pd.concat(pred_frames, ignore_index=True)

    res_frames = []
    for csv in sorted(DAILY_RES_DIR.glob("2026*.csv")):
        if "nar" in csv.name or "payouts" in csv.name:
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if "top1_finish" not in df.columns:
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

    pr = pred.merge(res, on="race_id", how="inner")
    pr["top1_score"] = pd.to_numeric(pr["top1_score"], errors="coerce")
    pr["top1_num"] = pd.to_numeric(pr["top1_num"], errors="coerce")
    pr = pr.dropna(subset=["top1_score", "top1_num", "top1_finish_num"])

    tyb = pd.read_csv(TYB_CSV, dtype={"race_id": str})
    tyb["umaban"] = pd.to_numeric(tyb["umaban"], errors="coerce")

    full = pr.merge(
        tyb,
        left_on=["race_id", "top1_num"],
        right_on=["race_id", "umaban"],
        how="inner",
    )
    print(f"[INFO] joined: {len(full)} rows, {full['label'].sum()} pos, {len(full) - full['label'].sum()} neg")
    return full


def main():
    df = load_data()
    if len(df) < 100:
        print(f"[ERROR] sample too small (n={len(df)})、 abort")
        return 1

    y = df["label"].values.astype(int)
    X_df = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    medians = X_df.median(numeric_only=True)
    X_df = X_df.fillna(medians).fillna(0.0)
    X = X_df.values.astype(float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 5-fold CV AUC
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    aucs_full = []
    aucs_v15_only = []
    for tr, te in kf.split(Xs):
        # V15 + TYB
        m = LogisticRegression(max_iter=2000)
        m.fit(Xs[tr], y[tr])
        p = m.predict_proba(Xs[te])[:, 1]
        aucs_full.append(roc_auc_score(y[te], p))

        # V15 only (top1_score column index 0)
        m2 = LogisticRegression(max_iter=2000)
        m2.fit(Xs[tr, 0:1], y[tr])
        p2 = m2.predict_proba(Xs[te, 0:1])[:, 1]
        aucs_v15_only.append(roc_auc_score(y[te], p2))

    auc_full = float(np.mean(aucs_full))
    auc_v15 = float(np.mean(aucs_v15_only))
    print(f"[INFO] V15 only 5CV AUC: {auc_v15:.4f}")
    print(f"[INFO] V15+TYB 5CV AUC:  {auc_full:.4f} (delta {auc_full - auc_v15:+.4f})")

    # full fit + save
    final_model = LogisticRegression(max_iter=2000)
    final_model.fit(Xs, y)
    pred_full = final_model.predict_proba(Xs)[:, 1]
    train_auc = roc_auc_score(y, pred_full)
    print(f"[INFO] train AUC: {train_auc:.4f} (over-fit risk if >> CV)")

    out = {
        "model": final_model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "feature_medians": {k: float(v) for k, v in medians.items()},
        "auc_v15_only_5cv": auc_v15,
        "auc_v15_plus_tyb_5cv": auc_full,
        "delta_auc": auc_full - auc_v15,
        "n_samples": int(len(df)),
        "pos_rate": float(y.mean()),
        "trained_at": pd.Timestamp.now().isoformat(),
    }
    with open(OUT_PKL, "wb") as f:
        pickle.dump(out, f)
    print(f"[INFO] saved: {OUT_PKL}")

    report = {
        "n_samples": int(len(df)),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "pos_rate": float(y.mean()),
        "auc_v15_only_5cv": auc_v15,
        "auc_v15_plus_tyb_5cv": auc_full,
        "delta_auc": auc_full - auc_v15,
        "train_auc": float(train_auc),
        "feature_cols": FEATURE_COLS,
        "lr_coefficients": {k: float(c) for k, c in zip(FEATURE_COLS, final_model.coef_[0])},
        "trained_at": out["trained_at"],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[INFO] report: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
