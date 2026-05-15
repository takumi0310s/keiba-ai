"""V15 + V22 enhanced top 100 stacking test (fold 25 = 2025 data、 OOS for V15).

V15 model は 2025 前で 学習済 → 2025 race に predict は OOS。
V22 enhanced top 100 fold 25 OOS predictions と stacking で V15 越え 試行。

stacking approach:
1. V15 model load + 2025 race に predict
2. V22 top 100 fold 25 prediction (既存 WF から)
3. 単純 average ensemble (V15 weight 0.5-0.95 探索)
4. AUC + ROI 比較

V15 投資保護 完全。
"""
from __future__ import annotations

import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'train'))

V15_MODEL = BASE / 'keiba_model_v15_central_live.pkl.gz'
V15_CACHE = BASE / 'data' / '_v15_optuna_df_cache.pkl.gz'

print('=== V15 + V22 Stacking Test (fold 25 = 2025) ===')

# 1. V15 model load
print('Loading V15 model ...')
with gzip.open(V15_MODEL, 'rb') as f:
    v15 = pickle.load(f)
v15_lgb = v15['model']
v15_xgb = v15['xgb_model']
v15_features = v15['features']
print(f'  V15 features: {len(v15_features)}')
print(f'  V15 version: {v15.get("version")}')
print(f'  V15 AUC (train): {v15.get("auc")}')

# 2. V15 cache load
print('Loading V15 cache ...')
with gzip.open(V15_CACHE, 'rb') as f:
    d = pickle.load(f)
df = d['df']
df['target'] = (df['finish'] <= 3).astype(int)

# 3. fold 25 (2025) subset
df25 = df[df['year'] == 25].copy()
print(f'  fold 25 (2025) rows: {len(df25):,}')

# 4. V15 predict on 2025 (OOS since V15 trained before 2025)
print('V15 predict on 2025 ...')
v15_feat_avail = [f for f in v15_features if f in df25.columns]
missing_v15 = [f for f in v15_features if f not in df25.columns]
print(f'  V15 features available: {len(v15_feat_avail)}/{len(v15_features)} (missing: {missing_v15[:5]}...)')

# Use available subset (fill missing with 0)
X25 = df25[v15_feat_avail].fillna(0).astype(np.float32).values
# If missing features, pad with 0 to match V15 features count
if len(missing_v15) > 0:
    pad = np.zeros((len(df25), len(missing_v15)), dtype=np.float32)
    X25_full = np.hstack([X25, pad])
    print(f'  padded {len(missing_v15)} missing features with 0')
else:
    X25_full = X25

# rearrange columns to match V15 features order
X25_dict = {f: 0.0 for f in v15_features}
v15_indices = {f: i for i, f in enumerate(v15_features)}
X25_in_order = np.zeros((len(df25), len(v15_features)), dtype=np.float32)
for j, f in enumerate(v15_feat_avail):
    if f in v15_indices:
        X25_in_order[:, v15_indices[f]] = X25[:, j]

# V15 predict (LGB + XGB ensemble)
p_v15_lgb = v15_lgb.predict(X25_in_order)
print(f'  V15 LGB pred: shape={p_v15_lgb.shape}, mean={p_v15_lgb.mean():.4f}')

import xgboost as xgb
dmatrix = xgb.DMatrix(X25_in_order)
p_v15_xgb = v15_xgb.predict(dmatrix)
print(f'  V15 XGB pred: shape={p_v15_xgb.shape}, mean={p_v15_xgb.mean():.4f}')

ens_w = v15.get('ensemble_weights', {'lgb': 0.5, 'xgb': 0.5})
print(f'  V15 ensemble weights: {ens_w}')
p_v15 = ens_w.get('lgb', 0.5) * p_v15_lgb + ens_w.get('xgb', 0.5) * p_v15_xgb

# AUC
y25 = df25['target'].values
auc_v15 = roc_auc_score(y25, p_v15)
print(f'\n=== V15 OOS AUC on 2025: {auc_v15:.4f} ===')

# 5. V22 top 100 fold 25 prediction を JSON から load
print('\nLoading V22 top 100 fold 25 predictions ...')
v22_files = sorted(Path(BASE / 'models' / 'v22_enhanced_top100').glob('top100_wf_summary_*.json'))
print(f'  v22 summary files: {[f.name for f in v22_files]}')

if not v22_files:
    print('  ★ V22 top 100 summary not found ★')
    sys.exit(0)

with open(v22_files[-1], 'r', encoding='utf-8') as f:
    v22sum = json.load(f)
v22_fold25 = next((r for r in v22sum['results'] if r['fold'] == '25-25'), None)
if v22_fold25:
    print(f'  V22 fold 25: LGB={v22_fold25["auc_lgb"]:.4f}, XGB={v22_fold25["auc_xgb"]:.4f}, '
          f'FT={v22_fold25["auc_ft"]:.4f}, IR={v22_fold25["auc_ir"]:.4f}, '
          f'Grid={v22_fold25["auc_grid"]:.4f}')

# V22 raw predictions per race は JSON にない → re-run しないと unable
# 代替: V15 OOS 自体 を 確認 し、 V15 が 2025 で どの AUC か honest 報告
print(f'\n=== Comparison ===')
print(f'  V15 OOS AUC on 2025 (re-run): {auc_v15:.4f}')
print(f'  V22 top100 fold25 Grid AUC:    {v22_fold25["auc_grid"]:.4f}')
print(f'  V15 baseline (mean WF):        0.8939')
print(f'  V22 base (mean WF):            0.8800')
print(f'')
print(f'  V15 specifically on 2025:      {auc_v15:.4f}')
print(f'  V22 enhanced on 2025:          {v22_fold25["auc_grid"]:.4f}')

if auc_v15 > v22_fold25["auc_grid"]:
    print(f'\n★ V15 wins on 2025 fold ({auc_v15:.4f} > {v22_fold25["auc_grid"]:.4f}) ★')
else:
    print(f'\n★ V22 wins on 2025 fold ({v22_fold25["auc_grid"]:.4f} > {auc_v15:.4f}) ★')

# Per-fold stacking 試行 不可 (V22 raw predictions per race が 必要 だが JSON 内 ない)
print('\n★ Note: 完全 stacking には V22 raw predictions per race が必要、 次回 WF 時 保存予定 ★')
