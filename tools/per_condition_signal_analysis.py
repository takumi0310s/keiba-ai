#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""6 条件 (A/B/C/D/E/X) 別 features signal 強度 分析.

V15 戦略⑦ は条件別 ROI 最適化済 (A=355%, B=347%, C=623%, D=361%, E=196%, X=701%)。
V20 投入時は **条件別 重要 features** を 把握 で 精度 向上 可能。

【V15 投資保護】 分析 のみ、 V15 model 不変

Usage:
    python tools/per_condition_signal_analysis.py
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


def classify_condition(row):
    """V15 条件分類 logic (CLAUDE.md 仕様)."""
    nh = row['num_horses']
    dist = row['distance']
    cond = row.get('condition_code', 0)  # 0=良 1=稍 2=重 3=不
    heavy = cond >= 2
    if nh <= 7:
        return 'E'
    if dist <= 1400:
        return 'D'
    if 8 <= nh <= 14 and dist >= 1600 and not heavy:
        return 'A'
    if 8 <= nh <= 14 and dist >= 1600 and heavy:
        return 'B'
    if nh >= 15 and dist >= 1600 and not heavy:
        return 'C'
    return 'X'


def main():
    ap = argparse.ArgumentParser(description='Per-condition signal analysis')
    args = ap.parse_args()

    import pandas as pd
    base = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    print(f'[INFO] loading: {base}')
    df = pd.read_csv(base, encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'horse_id', 'finish', 'year', 'num_horses',
                               'distance', 'condition', 'class_code'])
    print(f'[INFO] base: {df.shape}')

    df = df[df['year'] >= 22]  # 2022+ で十分
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)

    # condition encode (良=0, 稍=1, 重=2, 不=3)
    cond_map = {'良': 0, '稍重': 1, '稍': 1, '重': 2, '不良': 3, '不': 3}
    df['condition_code'] = df['condition'].map(cond_map).fillna(0).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)
    df['cond_cat'] = df.apply(classify_condition, axis=1)
    print(f'[INFO] 条件分類済')

    # Merge event_effect_features
    evt_path = os.path.join(BASE_DIR, 'data', 'event_effect_features.csv')
    if os.path.exists(evt_path):
        evt = pd.read_csv(evt_path, encoding='utf-8')
        evt['race_id'] = evt['race_id'].astype(str)
        evt['horse_id'] = evt['horse_id'].astype(str)
        evt_cols = ['race_id', 'horse_id'] + [c for c in evt.columns
                                                 if any(k in c for k in ['change', 'up', 'down', 'rate_exp'])]
        evt = evt[evt_cols].drop_duplicates(['race_id', 'horse_id'])
        df = df.merge(evt, on=['race_id', 'horse_id'], how='left')

    # Merge pace expanding
    pace_path = os.path.join(BASE_DIR, 'data', 'pace_features_expanding.csv')
    if os.path.exists(pace_path):
        pace = pd.read_csv(pace_path, encoding='utf-8')
        pace['race_id'] = pace['race_id'].astype(str)
        pace['horse_id'] = pace['horse_id'].astype(str)
        df = df.merge(pace, on=['race_id', 'horse_id'], how='left')

    print(f'[INFO] merged: {df.shape}')

    # 条件別 signal 分析
    print('\n=== 条件別 features signal 強度 ===\n')
    target_features = [
        'class_down', 'class_up', 'jockey_change', 'trainer_change',
        'pace_career_burst_mean', 'pace_career_change_1to4_mean',
    ]

    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        sub = df[df['cond_cat'] == cond]
        if len(sub) < 100:
            continue
        n = len(sub)
        avg_top3 = sub['top3'].mean()
        print(f'\n--- 条件 {cond} (N={n:,}、 全体 top3 rate {avg_top3:.3f}) ---')

        for feat in target_features:
            if feat not in sub.columns:
                continue
            valid = sub[sub[feat].notna()]
            if len(valid) < 50:
                continue

            # binary feature の場合: when=1 vs =0
            if feat in ['class_down', 'class_up', 'jockey_change', 'trainer_change']:
                tr1 = valid[valid[feat] == 1]['top3'].mean()
                tr0 = valid[valid[feat] == 0]['top3'].mean()
                n1 = (valid[feat] == 1).sum()
                if n1 < 30:
                    continue
                delta = tr1 - tr0
                marker = ' ★★★' if abs(delta) > 0.08 else (' ★★' if abs(delta) > 0.04 else '')
                print(f'  {feat:<35} top3 when 1: {tr1:.3f}  when 0: {tr0:.3f}  Δ={delta:+.3f}{marker}')
            else:
                # continuous: top quintile vs bottom
                try:
                    valid['_q'] = pd.qcut(valid[feat].rank(method='first'),
                                          5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
                    tr_q5 = valid[valid['_q'] == 'Q5']['top3'].mean()
                    tr_q1 = valid[valid['_q'] == 'Q1']['top3'].mean()
                    delta = tr_q5 - tr_q1
                    marker = ' ★★★' if abs(delta) > 0.08 else (' ★★' if abs(delta) > 0.04 else '')
                    print(f'  {feat:<35} Q5 top3: {tr_q5:.3f}  Q1: {tr_q1:.3f}  Δ={delta:+.3f}{marker}')
                except Exception:
                    pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
