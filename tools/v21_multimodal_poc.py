#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A4: V21 Multimodal (tabular + video AI features) PoC.

V15 tabular (4-ensemble) + Phase 22 動画 AI features (gait/bbox 20 features) を
late fusion で combine し、 V21 候補 PoC を作る。

【V15 投資保護】 V15 model 一切触らず、 stacking layer のみ V21 として 新規構築。
予測時は V15 出力 + video features を input にした meta-learner (LGB) を予測 layer に追加。

【限界】 video features は paddock 1 馬分しかないため、 本実装は **architecture demo + sample inference のみ**。
本格学習は paddock 全レース 全頭 frame 抽出 + 数千 race 集積後。

Usage:
    # demo: synthetic multimodal stacking demo
    python tools/v21_multimodal_poc.py demo

    # 実 video features を使う case (Phase 22 で 出力済)
    python tools/v21_multimodal_poc.py predict --gait data/video_ai_features/202603010112_2022106229/gait_features.json --v15-pred 0.65
"""
import argparse
import json
import os
import sys
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Video features (Phase 22 で gait_features.json から取れる 20 features)
VIDEO_FEATURE_KEYS = [
    'coverage', 'aspect_mean', 'aspect_std', 'aspect_range',
    'area_mean', 'area_std', 'w_mean', 'h_mean',
    'conf_mean', 'conf_std', 'conf_min',
    'motion_dcx_mean', 'motion_dcy_mean',
    'motion_speed_mean', 'motion_speed_std', 'motion_speed_max',
    'area_change_mean', 'aspect_change_mean',
]


def load_video_features(gait_path):
    with open(gait_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    feats = d.get('features', {})
    return {k: feats.get(k, 0.0) for k in VIDEO_FEATURE_KEYS}


def cmd_demo(args):
    """V15 tabular pred + synthetic video features で stacking demo."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    np.random.seed(42)
    n = 2000

    # V15 tabular pred (somewhat calibrated, AUC ~0.89)
    p_true = np.random.beta(2, 5, n)
    y_true = (np.random.rand(n) < p_true).astype(int)
    v15_pred = np.clip(p_true * 0.9 + np.random.normal(0, 0.08, n), 0.001, 0.999)

    # Synthetic video features (gait + posture)
    # 真の馬で異なる パターン (gait_speed が high な馬は good performer 寄り)
    video_signal = p_true + np.random.normal(0, 0.15, n)
    video_feats = np.column_stack([
        video_signal,                          # gait speed signal
        np.random.normal(0.7, 0.1, n),          # conf
        np.random.normal(1.3, 0.2, n),          # aspect
        np.random.normal(15, 8, n),             # motion_speed
        np.random.normal(146000, 30000, n),     # area
    ])

    # Train meta-learner on first half, evaluate on second half
    split = n // 2
    X_train = np.column_stack([v15_pred[:split], video_feats[:split]])
    X_test = np.column_stack([v15_pred[split:], video_feats[split:]])
    y_train, y_test = y_true[:split], y_true[split:]

    # Meta-learner: LR (lightweight) - alternative LGB
    meta = LogisticRegression(max_iter=1000)
    meta.fit(X_train, y_train)
    v21_pred = meta.predict_proba(X_test)[:, 1]

    # AUC 比較
    from sklearn.metrics import roc_auc_score, brier_score_loss
    auc_v15 = roc_auc_score(y_test, v15_pred[split:])
    auc_v21 = roc_auc_score(y_test, v21_pred)
    brier_v15 = brier_score_loss(y_test, v15_pred[split:])
    brier_v21 = brier_score_loss(y_test, v21_pred)
    print(f'[V15 baseline ] AUC={auc_v15:.4f}, Brier={brier_v15:.4f}')
    print(f'[V21 multimodal] AUC={auc_v21:.4f}, Brier={brier_v21:.4f}')
    print(f'[DELTA        ] AUC={auc_v21 - auc_v15:+.4f}, Brier={brier_v15 - brier_v21:+.4f}')

    # Feature importance
    print(f'\n[META coefficient]')
    feature_names = ['v15_pred', 'gait_signal', 'conf', 'aspect', 'motion_speed', 'area']
    for name, coef in zip(feature_names, meta.coef_[0]):
        print(f'  {name}: {coef:+.4f}')
    print(f'  intercept: {meta.intercept_[0]:+.4f}')
    return 0


def cmd_predict(args):
    """実 video features と V15 pred から V21 stacking 予測 (sample)."""
    v_feats = load_video_features(args.gait)
    print(f'[INFO] video features ({len(v_feats)}):')
    for k, v in v_feats.items():
        print(f'  {k}: {v}')
    print(f'[INFO] V15 pred: {args.v15_pred}')

    # Stub meta-learner: weighted sum (実 学習 model load は本実装で)
    # gait coverage / conf / aspect の signal を加重
    boost = 0.0
    if v_feats['coverage'] >= 0.9:
        boost += 0.02 * v_feats['conf_mean']  # 高 conf = +
    if 1.2 <= v_feats['aspect_mean'] <= 1.4:
        boost += 0.01  # 標準姿勢
    if v_feats['motion_speed_std'] < 30:
        boost += 0.01  # 安定 walk

    v21_pred = max(0.001, min(0.999, args.v15_pred + boost))
    print(f'\n[V21 multimodal sample inference] {v21_pred:.4f} (V15 base {args.v15_pred} + video boost {boost:+.4f})')
    return 0


def main():
    ap = argparse.ArgumentParser(description='V21 Multimodal stacking PoC (A4)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('demo', help='Synthetic data で stacking demo')

    pred = sub.add_parser('predict', help='Real video features で V21 sample inference')
    pred.add_argument('--gait', required=True, help='gait_features.json path')
    pred.add_argument('--v15-pred', type=float, required=True, help='V15 model output (0-1)')

    args = ap.parse_args()
    if args.cmd == 'demo':
        return cmd_demo(args)
    elif args.cmd == 'predict':
        return cmd_predict(args)
    return 1


if __name__ == '__main__':
    sys.exit(main())
