#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V20 predictions で 実 ROI simulation (V15 戦略⑦ 風 trio bet 比較).

v20_lgb_xgb_models.pkl.gz の 2025 fold predictions を用い、 V15 戦略⑦ 風 bet sizing で
ROI 算出。

【V15 投資保護】 V20 evaluation のみ、 V15 model 不変

Usage:
    python tools/v20_eval_roi.py
"""
import argparse
import gzip
import json
import os
import pickle
import sys

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    import pandas as pd
    import numpy as np
    import xgboost as xgb

    # V20 model load
    model_path = os.path.join(BASE_DIR, 'data', 'v20_lgb_xgb_models.pkl.gz')
    with gzip.open(model_path, 'rb') as f:
        m = pickle.load(f)
    feature_cols = m['feature_cols']
    print(f'[INFO] V20 model loaded: trained_year={m["trained_year"]}, {len(feature_cols)} features')

    # Test data: 2025
    data_path = os.path.join(BASE_DIR, 'data', 'v20_training_data_full.csv')
    df = pd.read_csv(data_path, encoding='utf-8', low_memory=False)
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['win'] = (df['finish'] == 1).astype(int)
    df_25 = df[df['year'] == 25]
    print(f'[INFO] 2025 test: {len(df_25):,} horses')

    # Encode categoricals (same as training)
    for c in ['surface', 'condition', 'course', 'class_code', 'father',
                'bms', 'sex', 'coat_color', 'location']:
        if c in df_25.columns:
            df_25[c] = df_25[c].astype('category').cat.codes

    if 'horse_num' in df_25.columns:
        df_25['horse_num'] = pd.to_numeric(df_25['horse_num'], errors='coerce')
    if 'umaban' in df_25.columns:
        df_25['umaban'] = pd.to_numeric(df_25['umaban'], errors='coerce')

    # Predict
    X = df_25[feature_cols].fillna(-1)
    lgb_pred = m['lgb'].predict(X)
    dtest = xgb.DMatrix(X)
    xgb_pred = m['xgb'].predict(dtest)
    ens_pred = 0.5 * lgb_pred + 0.5 * xgb_pred
    df_25['v20_pred'] = ens_pred

    # race ごとに top1-3 pick + 簡易 trio bet
    print(f'\n=== V20 race-level trio simulation (2025) ===')
    # race_id key (full Year/MM/DD format に変換)
    race_groups = df_25.groupby('race_id')

    # 戦略⑦ exclusion
    def classify_cond(row):
        nh = row.get('num_horses', 0)
        d = row.get('distance', 0)
        c = row.get('condition', 0)
        heavy = c >= 2  # encoded
        if nh <= 7: return 'E'
        if d <= 1400: return 'D'
        if 8 <= nh <= 14 and d >= 1600 and not heavy: return 'A'
        if 8 <= nh <= 14 and d >= 1600 and heavy: return 'B'
        if nh >= 15 and d >= 1600 and not heavy: return 'C'
        return 'X'

    total_bet = 0
    total_pnl = 0
    n_bet = 0
    n_hit = 0

    # 簡易 trio: top1 軸 + top2,3 - top2-6 (V15 戦略⑦ 同形式)
    POP_ODDS = {1: 3.0, 2: 5.5, 3: 8.0, 4: 11.0, 5: 14.5,
                6: 19.0, 7: 24.0, 8: 29.0, 9: 35.0, 10: 42.0}
    # 簡易 trio payout estimate: avg ~3000-5000 円
    AVG_TRIO_PAYOUT = 4000

    for rid, grp in race_groups:
        if len(grp) < 3:
            continue
        cond = classify_cond(grp.iloc[0])
        # 戦略⑦ exclude
        if cond in ['B', 'E']:
            continue
        # V20 pred ranking
        sorted_grp = grp.sort_values('v20_pred', ascending=False)
        top3_pred = sorted_grp.iloc[:3]
        # 簡易 trio: top1 軸、 7 点 想定
        actual_top3 = set(grp[grp['top3'] == 1]['horse_id'].tolist())
        predicted_top1 = top3_pred.iloc[0]['horse_id']
        predicted_top3 = set(top3_pred['horse_id'].tolist())
        # hit: 予測 top3 がそのまま 実 top3 と一致 (簡易 全的中 想定)
        hit = (predicted_top3 == actual_top3)
        bet = 700
        total_bet += bet
        n_bet += 1
        if hit:
            total_pnl += AVG_TRIO_PAYOUT - bet
            n_hit += 1
        else:
            total_pnl -= bet

    hit_rate = n_hit / max(1, n_bet)
    roi = (total_pnl / max(1, total_bet) + 1) * 100
    print(f'  N race: {n_bet:,}')
    print(f'  hit: {n_hit:,} ({hit_rate*100:.1f}%)')
    print(f'  total bet: ¥{total_bet:,}')
    print(f'  total PnL: ¥{total_pnl:+,}')
    print(f'  ROI: {roi:.1f}%')

    # 比較
    print(f'\n=== 比較 ===')
    print(f'  V15 戦略⑦ (実 2026 平均): 140%+ 想定')
    print(f'  V20 simulation (本 backtest): {roi:.1f}%')
    print(f'  ※ 簡易 trio simulation (実 V15 strict trio formation 7 点 と差異あり)')
    print(f'  ※ AVG payout {AVG_TRIO_PAYOUT} 仮定、 実際 race ごと 大きく 異なる')

    # Save
    out = os.path.join(BASE_DIR, 'data', 'v20_roi_eval.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({
            'test_year': 2025,
            'n_race': int(n_bet),
            'n_hit': int(n_hit),
            'hit_rate': float(hit_rate),
            'total_bet': int(total_bet),
            'total_pnl': int(total_pnl),
            'roi': float(roi),
            'assumed_payout': AVG_TRIO_PAYOUT,
        }, f, indent=2, ensure_ascii=False)
    print(f'\n[OK] saved: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
