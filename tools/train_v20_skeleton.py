#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V20 training script skeleton (5/24+ 投入 用、 Phase 25 plan).

V15 4-ensemble (LGB+XGB+FT+IntraRace) を baseline に、 Phase 22-23 で確認した
新 features を加えて V20 学習。 class_down が #1 signal (verify 済) であることを活用。

【V15 投資保護】 V15 model file (.pkl.gz) 一切 触らず、 別 file (v20_*) として 保存。
1 ヶ月並行運用 (V15 + V20 GUI 比較) 後に V15 archive 判定。

【追加 features】
- jockey_change, trainer_change, class_change (上下)
- jockey/trainer/class_down/class_up の expanding top3 rate
- 動画 features (paddock 蓄積 後): gait + body_condition (Phase 22-24 で抽出済)

【LEAK 厳禁 (Session #38 確定)】
- skb_* 全 10 features 完全除外 (V15.1 POST-RACE LEAK 確定済)
- sib_*_exp 修正版 のみ (Session #39 C で実装済)

Usage:
    # dry-run (script 構造確認のみ、 実 学習はしない)
    python tools/train_v20_skeleton.py --dry-run

    # 実 学習 (時間長、 user task)
    python tools/train_v20_skeleton.py --year-from 2020 --year-to 2025
"""
import argparse
import os
import sys

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Phase 22-23 で 確認した features (signal 強度 検証済)
V20_ADDITIONAL_FEATURES = [
    # 強 signal (verify 済)
    'class_down',                   # 67,372 importance (#1)
    'class_down_top3_rate_exp',     # 4,486 importance
    'jockey_change',                # 7,047
    'jockey_change_top3_rate_exp',  # 8,178
    'class_up',                      # 65
    'class_up_top3_rate_exp',       # 3,242
    'trainer_change',                # 3,231
    'trainer_change_top3_rate_exp', # 2,554
    'class_change',                  # 846
    # 弱 signal だが残す (interaction 期待)
    'rmk_delay', 'rmk_trouble',
    # 動画 features (paddock 蓄積後)
    'gait_aspect_mean', 'gait_aspect_std', 'gait_motion_speed_mean',
    'gait_motion_speed_std', 'gait_motion_speed_max',
    'gait_area_mean', 'gait_conf_mean', 'gait_coverage',
    'body_condition_score_mean', 'body_coat_brightness_mean',
    'body_coat_contrast_mean', 'body_body_compactness_mean',
]

# Session #38 確定 LEAK features (除外必須)
V20_LEAK_FEATURES = [
    'skb_kishi_code_1', 'skb_kishi_code_2', 'skb_kishi_code_3',
    'skb_baba_code_1',  'skb_baba_code_2',  'skb_baba_code_3',
    'skb_kyaku_code_1', 'skb_kyaku_code_2', 'skb_kyaku_code_3',
    'skb_turf_hoof',
    # V15 LEAK_FEATURES_A
    'odds_log', 'horse_weight', 'condition_enc',
    'weight_change', 'weight_change_abs', 'weight_cat',
    'weight_cat_dist', 'cond_surface',
]


def cmd_dry_run(args):
    """V20 学習 構造 + features list 表示のみ."""
    print('=== V20 training skeleton (dry-run) ===\n')
    print(f'対象期間: {args.year_from}-{args.year_to}')
    print(f'\n[Phase 22-23 新 features ({len(V20_ADDITIONAL_FEATURES)})]')
    for f in V20_ADDITIONAL_FEATURES:
        print(f'  + {f}')
    print(f'\n[除外 LEAK features ({len(V20_LEAK_FEATURES)})]')
    for f in V20_LEAK_FEATURES:
        print(f'  - {f}')
    print('\n[architecture]')
    print('  - LGB + XGB + FT-Transformer + IntraRace Attention (V15 と同じ 4-ensemble)')
    print('  - sib_*_exp 修正版 (Session #39 C 実装済)')
    print('  - 学習 target: top3 (V15 と同じ)')
    print('  - WF 6-fold (2020-2025)')
    print('  - 出力: keiba_model_v20_central.pkl.gz, keiba_model_v20_central_live.pkl.gz')
    print('  - V15 model file は完全 freeze (.pkl.gz 不変)')
    print('\n[判定基準 (Phase 25 plan)]')
    print('  - WF AUC ≥ 0.880 (V15 0.8939 比 ±)')
    print('  - 全年 AUC ≥ 0.85 (gap < 0.05)')
    print('  - 実 ROI 全条件 ≥ 100%')
    print('  - LIVE retro winner_top1 ≥ 30%')
    print('  - LEAK 監査 PASS')
    print('\n→ GO なら 5/24 開催 V20 段階投入、 NO-GO なら V15 案 B 改 単独継続')
    return 0


def cmd_train(args):
    """実 学習 (skeleton、 V15 投資保護 のため 完全 train code は user 実装).

    本 script は train/train_v135b_intra_ensemble.py を参考に、 V20 用 拡張 features と
    LEAK 除外 を適用した学習 を実装する 場所のみ準備。
    """
    print('=== V20 実 学習 (要 train/v20_*.py 拡張) ===')
    print('[REFERENCE] train/train_v135b_intra_ensemble.py の構造を 維持しつつ:')
    print('  1. merge_v15_1_features に V20_ADDITIONAL_FEATURES 追加')
    print('  2. V20_LEAK_FEATURES を 必ず drop')
    print('  3. 学習 output を keiba_model_v20_*.pkl.gz (新 file 名)')
    print('  4. V15 model file は touch しない')
    print('\n本 skeleton は dry-run のみ完成、 実 学習 は user が train/ で 実装 推奨')
    return 0


def main():
    ap = argparse.ArgumentParser(description='V20 training skeleton')
    ap.add_argument('--year-from', dest='year_from', type=int, default=2020)
    ap.add_argument('--year-to', dest='year_to', type=int, default=2025)
    ap.add_argument('--dry-run', dest='dry_run', action='store_true', default=True)
    args = ap.parse_args()

    if args.dry_run:
        return cmd_dry_run(args)
    else:
        return cmd_train(args)


if __name__ == '__main__':
    sys.exit(main())
