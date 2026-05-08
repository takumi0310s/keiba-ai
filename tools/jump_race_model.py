"""障害レース sub-model PoC (Session #45 E、 dev/sprint1).

現状: 障害は除外、 投票しない。
改善: TFJV 既存 jra_races_full.csv の障害 14,257 records を base に LGB 学習。

V15 学習 data から障害のみ抽出 → 簡易 features → LGB single fold

input:
- data/jra_races_full.csv (532K rows、 障害 14,257)

output:
- data/v18/models/v18_jump_lgb.txt (障害専用 model)
- data/v18/sprint1_jump_model_metrics.json

V15 production 完全独立。
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
from sklearn.metrics import roc_auc_score, log_loss

BASE = Path(r"C:/Users/takum/keiba-ai")
OUT_DIR = BASE / "data" / "v18" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("障害 sub-model PoC (Session #45 E)")
    print("=" * 60)
    t0 = time.time()

    # data load
    print("Loading jra_races_full.csv (障害 のみ抽出)...")
    df = pd.read_csv(BASE / "data" / "jra_races_full.csv", low_memory=False)
    df['race_name'] = df['race_name'].fillna('').astype(str)
    jump = df[df['race_name'].str.contains('障害', na=False)].copy()
    print(f"  jump races: {len(jump):,}")

    # target: is_top3 (3 着以内)
    jump['finish_num'] = pd.to_numeric(jump['finish'], errors='coerce')
    jump = jump.dropna(subset=['finish_num'])
    jump['target'] = (jump['finish_num'] <= 3).astype(int)
    print(f"  positive rate: {jump['target'].mean():.4f}")

    # 簡易 features
    feature_cols = []
    # numeric: weight_carry / age / num_horses / popularity / distance / horse_weight
    for col in ['weight_carry', 'age', 'num_horses', 'popularity', 'distance', 'horse_weight']:
        if col in jump.columns:
            jump[col + '_num'] = pd.to_numeric(jump[col], errors='coerce').fillna(-1)
            feature_cols.append(col + '_num')
    # categorical encoding (簡易)
    for col in ['sex', 'surface', 'condition']:
        if col in jump.columns:
            jump[col + '_enc'] = pd.Categorical(jump[col].fillna('?')).codes
            feature_cols.append(col + '_enc')
    # 障害固有: course_code (障害適性、 中山 / 京都 / 阪神 / 小倉 / 新潟 / 中京)
    if 'course_code' in jump.columns:
        jump['course_code_num'] = pd.to_numeric(jump['course_code'], errors='coerce').fillna(-1)
        feature_cols.append('course_code_num')
    # year
    if 'year' in jump.columns:
        jump['_y'] = 2000 + pd.to_numeric(jump['year'], errors='coerce').fillna(0).astype(int)
    print(f"  features: {len(feature_cols)} ({feature_cols})")

    # train (year ≤ 23) / test (year >= 24)
    train_mask = jump['_y'] <= 2023
    test_mask = jump['_y'] >= 2024
    n_tr = train_mask.sum()
    n_te = test_mask.sum()
    print(f"  train: {n_tr} (year <= 2023), test: {n_te} (year >= 2024)")

    X_tr = jump.loc[train_mask, feature_cols]
    y_tr = jump.loc[train_mask, 'target']
    X_te = jump.loc[test_mask, feature_cols]
    y_te = jump.loc[test_mask, 'target']

    # LGB train
    print("LGB training...")
    ts = time.time()
    m = lgb.train(
        {'objective': 'binary', 'metric': 'auc',
         'learning_rate': 0.05, 'num_leaves': 63,
         'min_data_in_leaf': 20,
         'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
         'lambda_l1': 0.1, 'lambda_l2': 0.1,
         'verbose': -1, 'seed': 42},
        lgb.Dataset(X_tr, y_tr),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_te, y_te)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    p_te = m.predict(X_te)
    auc = roc_auc_score(y_te, p_te)
    print(f"  AUC (test 2024+): {auc:.4f}")
    print(f"  training time: {(time.time()-ts):.1f}s")

    # save model
    model_path = OUT_DIR / "v18_jump_lgb.txt"
    m.save_model(str(model_path))
    print(f"  model saved: {model_path.relative_to(BASE)}")

    # metrics
    metrics = {
        "n_jump_total": int(len(jump)),
        "n_train": int(n_tr),
        "n_test": int(n_te),
        "AUC_test_2024_plus": float(auc),
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_metrics = BASE / "data" / "v18" / "sprint1_jump_model_metrics.json"
    out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  metrics: {out_metrics.relative_to(BASE)}")
    print(f"\n  TOTAL elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
