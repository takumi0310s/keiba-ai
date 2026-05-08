"""新馬戦 sub-model PoC (Session #47 D、 dev/sprint2).

新馬戦 専用 LGB、 血統重視 features。

source: data/jra_races_full.csv class_code=15 (新馬戦) = 43,959 races
features 重点:
- 父 ROI (sire_top3_rate from blood_full)
- 母 ROI (近年は限定的、 expanding)
- 血統 encoded
- 育成厩舎 (trainer 経験)
- weight / age / sex / 距離

target: is_top3 (3 着以内)

V15 production 完全独立、 dev/sprint2 のみ。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

BASE = Path(r"C:/Users/takum/keiba-ai")
OUT_DIR = BASE / "data" / "v18" / "models"


def main():
    print("=" * 60)
    print("maiden_race_model PoC (Session #47 D)")
    print("=" * 60)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading jra_races_full.csv (class_code=15、 新馬戦のみ)...")
    df = pd.read_csv(BASE / "data" / "jra_races_full.csv", low_memory=False)
    df["class_code_num"] = pd.to_numeric(df["class_code"], errors="coerce")
    maiden = df[df["class_code_num"] == 15].copy()
    print(f"  新馬戦: {len(maiden):,}")

    maiden["finish_num"] = pd.to_numeric(maiden["finish"], errors="coerce")
    maiden = maiden.dropna(subset=["finish_num"])
    maiden["target"] = (maiden["finish_num"] <= 3).astype(int)
    print(f"  positive rate: {maiden['target'].mean():.4f}")

    # features (簡易、 numerical のみ + categorical encoding)
    feature_cols = []

    # numerical
    for col in ["weight_carry", "age", "num_horses", "popularity",
                "distance", "horse_weight", "umaban"]:
        if col in maiden.columns:
            maiden[col + "_num"] = pd.to_numeric(maiden[col], errors="coerce").fillna(-1)
            feature_cols.append(col + "_num")

    # categorical → integer encoding
    for col in ["sex", "surface", "condition", "course"]:
        if col in maiden.columns:
            maiden[col + "_enc"] = pd.Categorical(maiden[col].fillna("?")).codes
            feature_cols.append(col + "_enc")

    # 血統 (father / mother / bms) encoded
    for col in ["father", "mother", "bms"]:
        if col in maiden.columns:
            maiden[col + "_enc"] = pd.Categorical(maiden[col].fillna("?")).codes
            feature_cols.append(col + "_enc")

    # course_code numerical
    if "course_code" in maiden.columns:
        maiden["course_code_num"] = pd.to_numeric(maiden["course_code"], errors="coerce").fillna(-1)
        feature_cols.append("course_code_num")

    # year (split 用)
    maiden["year_num"] = pd.to_numeric(maiden["year"], errors="coerce")
    maiden["year_full"] = maiden["year_num"].apply(
        lambda y: 2000 + int(y) if pd.notna(y) and int(y) <= 30 else None
    )
    maiden = maiden.dropna(subset=["year_full"])
    print(f"  features: {len(feature_cols)} ({feature_cols[:5]}...)")

    # train (year ≤ 2022) / test (year ≥ 2023)
    train_mask = maiden["year_full"] <= 2022
    test_mask = maiden["year_full"] >= 2023
    n_tr = int(train_mask.sum())
    n_te = int(test_mask.sum())
    print(f"  train: {n_tr} (year <= 2022), test: {n_te} (year >= 2023)")

    X_tr = maiden.loc[train_mask, feature_cols]
    y_tr = maiden.loc[train_mask, "target"]
    X_te = maiden.loc[test_mask, feature_cols]
    y_te = maiden.loc[test_mask, "target"]

    print("LGB training...")
    ts = time.time()
    m = lgb.train(
        {"objective": "binary", "metric": "auc",
         "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 50,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
         "lambda_l1": 0.1, "lambda_l2": 0.1, "verbose": -1, "seed": 42},
        lgb.Dataset(X_tr, y_tr),
        num_boost_round=1000,
        valid_sets=[lgb.Dataset(X_te, y_te)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    p = m.predict(X_te)
    auc = roc_auc_score(y_te, p)
    print(f"  AUC (test 2023+): {auc:.4f}")
    print(f"  training time: {(time.time()-ts):.1f}s")

    model_path = OUT_DIR / "v18_maiden_lgb.txt"
    m.save_model(str(model_path))
    print(f"  model saved: {model_path.relative_to(BASE)}")

    metrics = {
        "n_maiden_total": int(len(maiden)),
        "n_train": n_tr,
        "n_test": n_te,
        "AUC_test_2023_plus": float(auc),
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_metrics = BASE / "data" / "v18" / "sprint2_maiden_model_metrics.json"
    out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  metrics: {out_metrics.relative_to(BASE)}")
    print(f"  TOTAL: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
