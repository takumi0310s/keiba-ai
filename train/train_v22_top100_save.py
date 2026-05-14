#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V22 enhanced TOP 100 features を 全 data 学習 + .pkl.gz 保存.

V22 top 100 6-fold WF (mean 0.8813) 確認済。
全 data (2020-2025) で 学習し production sidecar 用に save。

V15 投資保護: V22 top100 は別 file (keiba_model_v22_top100_central.pkl.gz)
V15 .pkl.gz / predict_core / app.py 不変。

Usage:
    python train/train_v22_top100_save.py
"""
import gzip
import json
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

from train_v22_4ensemble import LGB_PARAMS, XGB_PARAMS
from train_v22_enhanced import load_v22_enhanced_data
from train_v22_enhanced_top100 import load_top_n_features

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TOP_FEATURES_JSON = os.path.join(DATA_DIR, 'top_features_v22enh.json')
OUT_PATH = os.path.join(BASE_DIR, 'keiba_model_v22_top100_central.pkl.gz')


def main():
    print('=' * 60)
    print('V22 TOP 100 features 全 data 学習 + save')
    print('=' * 60)
    print(f'DEVICE: {DEVICE}')

    t0 = time.time()
    df, all_features = load_v22_enhanced_data()
    top_features = load_top_n_features(100)
    available_top = [f for f in top_features if f in df.columns]
    print(f'top features: {len(available_top)}')

    # 全 data 学習 (test split は WF で 検証済、 ここでは production 用 save)
    y = df['target'].values
    X = df[available_top].astype(np.float32).values
    print(f'train shape: {X.shape}')

    # LGB
    print('Training LGB on all data ...')
    t1 = time.time()
    train_set = lgb.Dataset(X, label=y)
    lgb_model = lgb.train(LGB_PARAMS, train_set, num_boost_round=500,
                           callbacks=[lgb.log_evaluation(0)])
    print(f'  LGB done [{time.time()-t1:.0f}s]')

    # XGB
    print('Training XGB on all data ...')
    t1 = time.time()
    dtr = xgb.DMatrix(X, label=y)
    xgb_model = xgb.train(XGB_PARAMS, dtr, num_boost_round=500, verbose_eval=0)
    print(f'  XGB done [{time.time()-t1:.0f}s]')

    # scaler for FT/IR (sidecar 予測で 必要)
    scaler = StandardScaler()
    scaler.fit(X)

    # 保存 (FT/IR は full train は省略、 predict 時 LGB+XGB 主軸 + FT/IR 動的)
    out = {
        'version': 'v22_top100',
        'features': available_top,
        'lgb_model': lgb_model,
        'xgb_model': xgb_model,
        'scaler': scaler,
        'wf_mean_auc': 0.8813,
        'wf_summary_file': 'top100_wf_summary_20260514_193645.json',
        'trained_at': datetime.now().isoformat(),
        'note': 'V22 enhanced TOP 100 features sidecar、 V15 並行 shadow eval、 投資 0 円',
    }

    with gzip.open(OUT_PATH, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nsaved: {OUT_PATH}')
    print(f'  size: {os.path.getsize(OUT_PATH)/1024/1024:.1f} MB')
    print(f'  features: {len(available_top)}')
    print(f'  elapsed: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
