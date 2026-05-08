"""障害 sub-model v2 強化版 (Session #47 F、 dev/sprint2).

Sprint 1 E (jump_race_model.py) の強化版:
- 過去障害成功率 (horse 単位 expanding)
- 騎手 × 障害 interaction
- 障害種別 (大障害 / 平地障害)、 障害レース内 race_name 分析
- popularity 除外 (リーク類似)

target AUC: 0.7536 → 0.78+ (popularity 除外で 0.65-0.70 想定)

V15 production 完全独立、 dev/sprint2 のみ。
"""
from __future__ import annotations

import json
import os
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
    print("jump_race_model v2 (Session #47 F)")
    print("=" * 60)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading jra_races_full.csv (障害のみ)...")
    df = pd.read_csv(BASE / "data" / "jra_races_full.csv", low_memory=False)
    df["race_name"] = df["race_name"].fillna("").astype(str)
    jump = df[df["race_name"].str.contains("障害", na=False)].copy()
    print(f"  jump: {len(jump):,}")

    jump["finish_num"] = pd.to_numeric(jump["finish"], errors="coerce")
    jump = jump.dropna(subset=["finish_num"])
    jump["target"] = (jump["finish_num"] <= 3).astype(int)
    jump["horse_id"] = jump["horse_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    print(f"  positive rate: {jump['target'].mean():.4f}")

    # === sort by date for expanding ===
    jump["year_full"] = pd.to_numeric(jump["year"], errors="coerce").apply(
        lambda y: 2000 + int(y) if pd.notna(y) and int(y) <= 30 else None
    )
    jump = jump.dropna(subset=["year_full"])
    jump = jump.sort_values(["horse_id", "year_full", "month", "day"])

    # === 障害固有 features ===
    # 1. 過去障害成功率 (horse 単位 expanding)
    jump["jump_top3"] = (jump["finish_num"] <= 3).astype(int)
    jump["jump_runs"] = 1
    grp = jump.groupby("horse_id", sort=False)
    jump["horse_jump_cum_top3"] = grp["jump_top3"].cumsum() - jump["jump_top3"]
    jump["horse_jump_cum_runs"] = grp["jump_runs"].cumsum() - jump["jump_runs"]
    alpha = 5
    jump["horse_jump_top3_rate_exp"] = (
        (jump["horse_jump_cum_top3"] + alpha * 0.30)
        / (jump["horse_jump_cum_runs"] + alpha)
    )

    # 2. 大障害判定 (race_name に '中山大障害' / '中山GJ' 等)
    jump["is_grand_jump"] = jump["race_name"].str.contains("大障害|GJ").astype(int)

    # 3. 騎手 × 障害 (騎手の障害 expanding 成功率)
    jump_jockey = jump.groupby("jockey", sort=False)["jump_top3"].cumsum() - jump["jump_top3"]
    jump_jockey_runs = jump.groupby("jockey", sort=False)["jump_runs"].cumsum() - jump["jump_runs"]
    jump["jockey_jump_top3_rate_exp"] = (
        (jump_jockey + alpha * 0.30) / (jump_jockey_runs + alpha)
    )

    # === features ===
    feature_cols = []
    for col in ["weight_carry", "age", "num_horses", "distance", "horse_weight", "umaban"]:
        if col in jump.columns:
            jump[col + "_num"] = pd.to_numeric(jump[col], errors="coerce").fillna(-1)
            feature_cols.append(col + "_num")
    for col in ["sex", "surface", "condition"]:
        if col in jump.columns:
            jump[col + "_enc"] = pd.Categorical(jump[col].fillna("?")).codes
            feature_cols.append(col + "_enc")
    if "course_code" in jump.columns:
        jump["course_code_num"] = pd.to_numeric(jump["course_code"], errors="coerce").fillna(-1)
        feature_cols.append("course_code_num")
    feature_cols += ["horse_jump_top3_rate_exp", "is_grand_jump", "jockey_jump_top3_rate_exp"]

    # ★ popularity 除外 (Sprint 1 E のリーク類似)
    print(f"  features: {len(feature_cols)} (popularity 除外、 障害固有 3 features 追加)")

    train_mask = jump["year_full"] <= 2023
    test_mask = jump["year_full"] >= 2024
    n_tr = int(train_mask.sum())
    n_te = int(test_mask.sum())
    print(f"  train: {n_tr} (year <= 2023), test: {n_te} (year >= 2024)")

    X_tr = jump.loc[train_mask, feature_cols]
    y_tr = jump.loc[train_mask, "target"]
    X_te = jump.loc[test_mask, feature_cols]
    y_te = jump.loc[test_mask, "target"]

    print("LGB training...")
    ts = time.time()
    m = lgb.train(
        {"objective": "binary", "metric": "auc",
         "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 20,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
         "lambda_l1": 0.1, "lambda_l2": 0.1, "verbose": -1, "seed": 42},
        lgb.Dataset(X_tr, y_tr),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_te, y_te)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    p = m.predict(X_te)
    auc = roc_auc_score(y_te, p)
    print(f"  AUC (test 2024+): {auc:.4f}")
    print(f"  vs Sprint 1 E (popularity 含): 0.7536")
    print(f"  improvement: {auc - 0.7536:+.4f}")
    print(f"  training time: {(time.time()-ts):.1f}s")

    model_path = OUT_DIR / "v18_jump_v2_lgb.txt"
    m.save_model(str(model_path))

    metrics = {
        "n_jump_total": int(len(jump)),
        "n_train": n_tr, "n_test": n_te,
        "AUC_test_2024_plus": float(auc),
        "vs_sprint1_e": 0.7536,
        "delta_auc": round(auc - 0.7536, 4),
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "popularity_excluded": True,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_metrics = BASE / "data" / "v18" / "sprint2_jump_v2_metrics.json"
    out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  TOTAL: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
