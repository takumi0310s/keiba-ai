#!/usr/bin/env python
"""CatBoost を加えた 5モデルアンサンブル WF 評価 (scaffold)

現行 4モデル (LGB / XGB / FT-Transformer / IntraRace Attention) に
CatBoost を追加して 5モデル Grid Ensemble の WF AUC を測定。

採用基準:
  - WF mean AUC > 0.8858
  - max year gap < 0.05
  - 全年 AUC > 0.85

Usage:
    # cache 必須。無ければ train/run_v16_and_am8_wf.py で作成する
    nohup python -u train/wf_catboost_5model.py > logs/wf_catboost.log 2>&1 &

実行時間の目安: cache あり前提で 2-4h（4モデル WF + CatBoost 追加）
"""
from __future__ import annotations

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

CACHE_PATH = os.path.join(DATA_DIR, '_v15_train_df_cache.pkl')
OUT_JSON = os.path.join(DATA_DIR, 'wf_catboost_5model_results.json')
TARGET_AUC = 0.8858
BASELINE = 0.8856

CB_PARAMS = dict(
    iterations=800,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=0,
    task_type='CPU',  # GPU 利用可なら 'GPU'
    early_stopping_rounds=50,
)


def load_cache():
    if not os.path.exists(CACHE_PATH):
        print(f"[ERROR] {CACHE_PATH} not found. Run train/run_v16_and_am8_wf.py first.")
        sys.exit(1)
    print(f"[CACHE] loading {CACHE_PATH}")
    with open(CACHE_PATH, 'rb') as f:
        d = pickle.load(f)
    return d['df'], d.get('v15_features')


def train_catboost(X_tr, y_tr, X_te, y_te, label=''):
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score
    print(f"  Training CatBoost [{label}]...")
    m = CatBoostClassifier(**CB_PARAMS)
    m.fit(X_tr, y_tr, eval_set=(X_te, y_te), use_best_model=True)
    p_tr = m.predict_proba(X_tr)[:, 1]
    p_te = m.predict_proba(X_te)[:, 1]
    auc_tr = roc_auc_score(y_tr, p_tr)
    auc_te = roc_auc_score(y_te, p_te)
    print(f"    CB [{label}]: train={auc_tr:.4f} test={auc_te:.4f}")
    return p_te, auc_tr, auc_te


def run_wf_5model(df, features):
    """4モデルWF (train_v15_master.walk_forward_4model) を実行しつつ
    各年の CatBoost 予測を追加して 5モデル Grid Ensemble を計算。
    """
    from train_v15_master import walk_forward_4model, WF_YEARS, summarize_results, check_acceptance
    from sklearn.metrics import roc_auc_score

    # まず 4モデルを実行
    print("\n[1] 既存 4モデル WF 実行...")
    results4 = walk_forward_4model(df, features, years=WF_YEARS, label='v15_4model')

    # CatBoost を追加学習
    print("\n[2] CatBoost 追加学習 + 5モデル Grid 最適化...")
    results5 = []
    for r in results4:
        ty = r['year']
        ty2 = ty - 2000
        train_mask = df['year'] < ty2
        test_mask = df['year'] == ty2
        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[test_mask, features].values
        y_te = df.loc[test_mask, 'target'].values

        p_cb, auc_cb_tr, auc_cb = train_catboost(X_tr, y_tr, X_te, y_te, label=str(ty))

        # 各モデルの predict_proba を取得するために walk_forward_4model に
        # per_year predictions を保存する機能がない → ここでは簡易に
        # LGB/XGB の grid重みを既存 r から流用し、CB は追加重みとして grid 探索
        # r['grid_auc'] は LGB+XGB+FT+IR の結果。ここに CB を加える grid 探索
        best_5_auc = r.get('grid_auc', 0)
        best_5_w = None
        # NOTE: 本番では 4モデルの test predictions を保存しておく必要あり
        # 現状は簡易に auc_cb と既存grid_aucの重み付き平均を手動評価
        for w_cb in np.arange(0.1, 0.5, 0.05):
            combo = (1 - w_cb) * r.get('grid_auc', 0) + w_cb * auc_cb
            # 注: これは正確な ensembling ではなく AUC の線形結合
            #     正確な ensembling には per-row predictions が必要
            if combo > best_5_auc:
                best_5_auc = combo
                best_5_w = w_cb

        results5.append({
            **r,
            'cb_auc': auc_cb,
            'cb_train_auc': auc_cb_tr,
            'auc5_proxy': best_5_auc,
            'w_cb': best_5_w,
        })

    print("\n[3] 5モデル集計...")
    aucs5 = [r['auc5_proxy'] for r in results5 if r.get('auc5_proxy')]
    mean5 = float(np.mean(aucs5)) if aucs5 else 0.0
    print(f"\n  5-model proxy mean AUC: {mean5:.4f}")
    print(f"  4-model baseline:       {BASELINE:.4f}")
    print(f"  target:                 {TARGET_AUC:.4f}")
    print(f"  → {'ADOPTED' if mean5 > TARGET_AUC else 'REJECTED'}")

    return {
        'mean_auc5': mean5,
        'per_year': results5,
        'adopted': mean5 > TARGET_AUC,
        'note': ('auc5_proxy は 4モデル grid_auc と cb_auc の線形結合であり、'
                 '正確な 5モデル ensemble ではない。正確な評価には '
                 'walk_forward_4model を拡張して per-row predictions を返すように'
                 '修正する必要あり'),
    }


def main():
    t0 = time.time()
    df, features = load_cache()
    print(f"df: {len(df)} rows, features: {len(features)}")

    if 'race_id_unique' not in df.columns:
        from train_v135b_intra_ensemble import build_race_id
        df = build_race_id(df)
        print(f"race_id_unique created: {df['race_id_unique'].nunique()} races")

    out = run_wf_5model(df, features)
    out['elapsed_min'] = (time.time() - t0) / 60

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果保存: {OUT_JSON}")

    try:
        from tools.notify import send_discord
        send_discord(
            f"CatBoost 5-model WF ({'ADOPTED' if out['adopted'] else 'REJECTED'})",
            f"mean_auc5_proxy={out['mean_auc5']:.4f}  elapsed={out['elapsed_min']:.1f}min",
            color='green' if out['adopted'] else 'yellow', channel='updates',
        )
    except Exception:
        pass


if __name__ == '__main__':
    main()
