"""paddock 動画 features の expanding/aggregation module.

YOLOv8 inference 結果 (data/paddock_features/paddock_features_all.csv) を
V20+/V22 学習 用 format に変換。 per-horse stats + race-relative features 出力。

V15 .pkl.gz / predict_core / app.py 完全不変。

usage:
    python train/features_paddock_video.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / 'data'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=str(DATA_DIR / 'paddock_features' / 'paddock_features_all.csv'))
    ap.add_argument('--output', default=str(DATA_DIR / 'features_paddock_video.csv'))
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f'[INFO] paddock features 未生成: {args.input}')
        print('実行: python tools/paddock_yolo_inference.py')
        return

    df = pd.read_csv(args.input)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)
    print(f'paddock features: {len(df)} rows')

    # race-relative features (race 内 相対 順位/比較)
    df_out = df.copy()

    # 各 race 内 で 各 features の rank + std normalize
    for col in ['pf_bbox_size_avg', 'pf_motion_amount', 'pf_horse_confidence_avg',
                'pf_detection_rate']:
        if col not in df.columns:
            continue
        # race 内 rank (1 が 最大)
        df_out[f'{col}_rank'] = df.groupby('race_id')[col].rank(ascending=False, method='min')
        # race 内 mean からの 偏差 (positive = race 平均より大)
        df_out[f'{col}_zscore'] = df.groupby('race_id')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6))

    # paddock 評価 score (合成)
    if all(c in df_out.columns for c in ['pf_horse_confidence_avg', 'pf_detection_rate']):
        # 大型 horse + 高 confidence + 安定検出 = 馬体 良好
        df_out['paddock_health_score'] = (
            df_out['pf_horse_confidence_avg'] * df_out['pf_detection_rate']
        )
        # 低 motion = 落ち着き OK
        if 'pf_motion_amount' in df_out.columns:
            df_out['paddock_calm_score'] = 1.0 / (1.0 + df_out['pf_motion_amount'] / 50)

    print(f'output features: {len(df_out.columns)}')
    print(f'cols: {df_out.columns.tolist()[:15]}')
    df_out.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f'saved: {args.output}')


if __name__ == '__main__':
    main()
