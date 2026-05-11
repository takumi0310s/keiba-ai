#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V20 trio 7-point formation simulation (V15 戦略⑦ format match).

V15 戦略⑦ trio 7 点 = TOP1 軸 - TOP2,3 - TOP2-6 で hit 率 44% (条件 A).
V20 model で 同じ formation を 適用、 hit 率 + ROI 算出。

【V15 投資保護】 V20 model 評価のみ、 V15 model 不変
"""
import argparse
import gzip
import os
import pickle
import sys
from itertools import combinations

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    import pandas as pd
    import xgboost as xgb

    # Load V20 model
    model_path = os.path.join(BASE_DIR, 'data', 'v20_lgb_xgb_models.pkl.gz')
    with gzip.open(model_path, 'rb') as f:
        m = pickle.load(f)
    feature_cols = m['feature_cols']

    # 2025 test data
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'v20_training_data_full.csv'),
                      encoding='utf-8', low_memory=False)
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df_25 = df[df['year'] == 25].copy()

    for c in ['surface', 'condition', 'course', 'class_code', 'father',
                'bms', 'sex', 'coat_color', 'location']:
        if c in df_25.columns:
            df_25[c] = df_25[c].astype('category').cat.codes

    if 'horse_num' in df_25.columns:
        df_25['horse_num'] = pd.to_numeric(df_25['horse_num'], errors='coerce')
    if 'umaban' in df_25.columns:
        df_25['umaban'] = pd.to_numeric(df_25['umaban'], errors='coerce')

    X = df_25[feature_cols].fillna(-1)
    lgb_p = m['lgb'].predict(X)
    xgb_p = m['xgb'].predict(xgb.DMatrix(X))
    df_25['v20_pred'] = 0.5 * lgb_p + 0.5 * xgb_p

    # race_id 8 digit
    df_25['race_id_str'] = df_25['race_id'].astype(str).str.zfill(10)
    df_25['race_id_8'] = df_25['race_id_str'].str[:8]
    print(f'[INFO] 2025 races: {df_25["race_id_8"].nunique():,}')

    # 各 race で V20 ranking → trio 7 点 formation
    # V15 形式: TOP1 軸 - TOP2,3 - TOP2-6 = pairs (top1-2-3, top1-2-4, ..., top1-3-6)
    hits = 0
    n_race = 0
    total_bet = 0
    total_pnl = 0
    AVG_PAYOUT = 4500  # 戦略⑦ ROI 140% suggest 平均 trio 4500 yen
    BET_PER_POINT = 100  # 7点 × 100円 = 700円

    hit_by_cond = {}
    for rid, grp in df_25.groupby('race_id_8'):
        n_horses = len(grp)
        if n_horses < 6:
            continue
        sorted_grp = grp.sort_values('v20_pred', ascending=False)
        top_horses = sorted_grp['horse_id'].iloc[:6].tolist()
        # V15 形式 7 点 formation: TOP1 軸 - TOP2,3 - TOP2-6
        formation = [
            tuple(sorted([top_horses[0], top_horses[1], top_horses[2]])),
            tuple(sorted([top_horses[0], top_horses[1], top_horses[3]])),
            tuple(sorted([top_horses[0], top_horses[1], top_horses[4]])),
            tuple(sorted([top_horses[0], top_horses[1], top_horses[5]])),
            tuple(sorted([top_horses[0], top_horses[2], top_horses[3]])),
            tuple(sorted([top_horses[0], top_horses[2], top_horses[4]])),
            tuple(sorted([top_horses[0], top_horses[2], top_horses[5]])),
        ]

        # 実 top3 馬
        actual_top3 = tuple(sorted(grp[grp['top3'] == 1]['horse_id'].tolist()))

        # hit: actual_top3 が formation に含まれるか
        hit = actual_top3 in formation
        n_race += 1
        total_bet += BET_PER_POINT * 7  # 700円
        if hit:
            hits += 1
            total_pnl += AVG_PAYOUT - 700  # net
        else:
            total_pnl -= 700

    hit_rate = hits / max(1, n_race)
    roi = (total_pnl / max(1, total_bet) + 1) * 100

    print(f'\n=== V20 trio 7-pt formation simulation (2025) ===')
    print(f'  N race: {n_race:,}')
    print(f'  hit: {hits:,} ({hit_rate*100:.1f}%)')
    print(f'  total bet: ¥{total_bet:,}')
    print(f'  total PnL: ¥{total_pnl:+,}')
    print(f'  ROI: {roi:.1f}%')
    print(f'  ※ AVG_PAYOUT={AVG_PAYOUT} 仮定')

    print(f'\n=== 比較 ===')
    print(f'  V15 戦略⑦ (実 4-ensemble、 124 features): 140%+ ROI')
    print(f'  V20 (2-ensemble、 75 features): {roi:.1f}% ROI')
    print(f'  V15 baseline AUC 0.8939 vs V20 0.8376 (-0.056)')

    # AUC vs hit rate proxy
    print(f'\n  V15 hit rate 想定: 44% (条件 A 平均)')
    print(f'  V20 hit rate 本 sim: {hit_rate*100:.1f}%')
    print(f'  V20 改善 path: FT-Transformer + IntraRace + V15 features 全 → hit 35-45% / ROI 110-150%')

    return 0


if __name__ == '__main__':
    sys.exit(main())
