#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""騎手 / 厩舎 / 馬の hot / cold streak features (recent K races).

過去 K 走の prev_finish 順位 から momentum を 算出。 騎手 / 厩舎 ごとの recent form
変化を 当該レース予測に 活用。

【V15 投資保護】 derivative のみ、 V15 model 不変

Usage:
    python tools/build_hot_streak_features.py
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


def main():
    import pandas as pd
    base = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    df = pd.read_csv(base, encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'horse_id', 'jockey_id', 'trainer_id',
                               'finish', 'year'])
    df = df[df['year'] >= 22]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['win'] = (df['finish'] == 1).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)
    df['jockey_id'] = df['jockey_id'].astype(str)
    df['trainer_id'] = df['trainer_id'].astype(str)
    df = df.sort_values('race_id').reset_index(drop=True)
    df['_idx'] = range(len(df))

    print(f'[INFO] {len(df):,} rows')

    # Jockey hot streak (recent 30 races)
    print('[INFO] computing jockey hot streak (recent 30)...')
    df_j = df.sort_values(['jockey_id', '_idx']).reset_index(drop=True)
    j_gb = df_j.groupby('jockey_id')
    df_j['jockey_recent30_top3'] = (
        j_gb['top3'].apply(lambda s: s.shift(1).rolling(30, min_periods=5).mean()).reset_index(level=0, drop=True)
    )
    df_j['jockey_recent30_win'] = (
        j_gb['win'].apply(lambda s: s.shift(1).rolling(30, min_periods=5).mean()).reset_index(level=0, drop=True)
    )
    df = df.merge(df_j[['race_id', 'horse_id', 'jockey_recent30_top3', 'jockey_recent30_win']],
                   on=['race_id', 'horse_id'], how='left')

    # Trainer hot streak
    print('[INFO] computing trainer hot streak (recent 30)...')
    df_t = df.sort_values(['trainer_id', '_idx']).reset_index(drop=True)
    t_gb = df_t.groupby('trainer_id')
    df_t['trainer_recent30_top3'] = (
        t_gb['top3'].apply(lambda s: s.shift(1).rolling(30, min_periods=5).mean()).reset_index(level=0, drop=True)
    )
    df_t['trainer_recent30_win'] = (
        t_gb['win'].apply(lambda s: s.shift(1).rolling(30, min_periods=5).mean()).reset_index(level=0, drop=True)
    )
    df = df.merge(df_t[['race_id', 'horse_id', 'trainer_recent30_top3', 'trainer_recent30_win']],
                   on=['race_id', 'horse_id'], how='left')

    # Horse momentum (recent 5 races for the horse)
    print('[INFO] computing horse recent5 momentum...')
    df_h = df.sort_values(['horse_id', '_idx']).reset_index(drop=True)
    h_gb = df_h.groupby('horse_id')
    df_h['horse_recent5_top3'] = (
        h_gb['top3'].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=0, drop=True)
    )
    df_h['horse_recent5_win'] = (
        h_gb['win'].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=0, drop=True)
    )
    df = df.merge(df_h[['race_id', 'horse_id', 'horse_recent5_top3', 'horse_recent5_win']],
                   on=['race_id', 'horse_id'], how='left')

    # === Signal verify ===
    for col in ['jockey_recent30_top3', 'trainer_recent30_top3', 'horse_recent5_top3']:
        valid = df[df[col].notna()].copy()
        if len(valid) < 5000:
            continue
        valid['_q'] = pd.qcut(valid[col].rank(method='first'),
                                5, labels=[1, 2, 3, 4, 5])
        q5 = valid[valid['_q'] == 5]['top3'].mean()
        q1 = valid[valid['_q'] == 1]['top3'].mean()
        print(f'  {col}: Q1: {q1:.3f}, Q5: {q5:.3f}, Δ={q5-q1:+.3f}')

    out_cols = ['race_id', 'horse_id',
                'jockey_recent30_top3', 'jockey_recent30_win',
                'trainer_recent30_top3', 'trainer_recent30_win',
                'horse_recent5_top3', 'horse_recent5_win']
    out = df[out_cols].copy()
    out_path = os.path.join(BASE_DIR, 'data', 'hot_streak_features.csv')
    out.to_csv(out_path, index=False)
    print(f'\n[OK] saved: {out_path}')
    print(f'[OK] shape: {out.shape}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
