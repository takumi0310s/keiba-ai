#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V21 学習 data builder: V15 tabular + video AI + remarks + event effects を 1 CSV に merge.

V21 model 学習に使う統合 features csv を生成。 race_id + horse_id をキーに merge。

【merge 対象】
1. V15 base features (data/jra_races_full.csv または既存 features cache)
2. data/v21_video_features.csv (gait 20 + body 18 = 38 features)
3. data/race_review_features.csv (remarks 9 features)
4. data/event_effect_features.csv (events 14 features)
出力: data/v21_training_features.csv

【V15 投資保護】 train/ V15 関連 file 触らず、 新規 CSV のみ生成。

Usage:
    python tools/v21_training_data_builder.py
    python tools/v21_training_data_builder.py --base data/jra_races_full.csv
    python tools/v21_training_data_builder.py --year-from 2024 --year-to 2026
"""
import argparse
import os
import sys
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASE = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
OUT_PATH = os.path.join(BASE_DIR, 'data', 'v21_training_features.csv')


def main():
    ap = argparse.ArgumentParser(description='V21 training data builder')
    ap.add_argument('--base', default=DEFAULT_BASE, help='base race CSV')
    ap.add_argument('--video', default=os.path.join(BASE_DIR, 'data', 'v21_video_features.csv'))
    ap.add_argument('--remarks', default=os.path.join(BASE_DIR, 'data', 'race_review_features.csv'))
    ap.add_argument('--events', default=os.path.join(BASE_DIR, 'data', 'event_effect_features.csv'))
    ap.add_argument('--year-from', dest='year_from', type=int, default=None)
    ap.add_argument('--year-to', dest='year_to', type=int, default=None)
    ap.add_argument('--out', default=OUT_PATH)
    args = ap.parse_args()

    import pandas as pd

    print(f'[INFO] loading base: {args.base}')
    df = pd.read_csv(args.base, encoding='utf-8', low_memory=False)
    print(f'  shape: {df.shape}')

    # year filter (use 'year' column if available, else parse race_id)
    if args.year_from is not None:
        if 'year' in df.columns:
            # year は 15-26 形式 (15 = 2015, 26 = 2026) or 4 桁 (2015-2026)
            year_max = df['year'].max()
            if year_max < 100:  # 2 digit year
                yf = args.year_from - 2000
                yt = args.year_to - 2000 if args.year_to else None
            else:
                yf = args.year_from
                yt = args.year_to
            df = df[df['year'] >= yf]
            if yt:
                df = df[df['year'] <= yt]
        elif 'race_id' in df.columns:
            df['_year'] = df['race_id'].astype(str).str[:4].astype(int)
            df = df[df['_year'] >= args.year_from]
            if args.year_to:
                df = df[df['_year'] <= args.year_to]
            df = df.drop(columns=['_year'])
        print(f'  after year filter: {df.shape}')

    # key check
    must = ['race_id', 'horse_id']
    missing = [c for c in must if c not in df.columns]
    if missing:
        print(f'[ERROR] base missing: {missing}')
        return 1
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)

    # video features
    n_video = 0
    if os.path.exists(args.video):
        v = pd.read_csv(args.video, encoding='utf-8')
        v['race_id'] = v['race_id'].astype(str)
        v['horse_id'] = v['horse_id'].astype(str)
        # drop status / fileのみ keep features
        v_cols = ['race_id', 'horse_id'] + [c for c in v.columns
                                              if c.startswith(('gait_', 'body_'))]
        v = v[v_cols]
        df = df.merge(v, on=['race_id', 'horse_id'], how='left', suffixes=('', '_video'))
        n_video = v.shape[1] - 2
        print(f'[merge] video features: +{n_video} cols ({len(v)} rows merged)')

    # remarks
    n_rmk = 0
    if os.path.exists(args.remarks):
        r = pd.read_csv(args.remarks, encoding='utf-8')
        r['race_id'] = r['race_id'].astype(str)
        # remarks uses umaban + horse_name (not horse_id) - need to join by race_id + umaban
        # Get umaban from base if available
        if 'horse_num' in df.columns or 'umaban' in df.columns:
            base_uma_col = 'umaban' if 'umaban' in df.columns else 'horse_num'
            r_cols = ['race_id', 'umaban'] + [c for c in r.columns if c.startswith('rmk_')]
            r = r[r_cols]
            df['_umaban_key'] = df[base_uma_col].astype(float).astype('Int64')
            r['_umaban_key'] = r['umaban'].astype(float).astype('Int64')
            df = df.merge(r.drop(columns=['umaban']), left_on=['race_id', '_umaban_key'],
                           right_on=['race_id', '_umaban_key'], how='left')
            df = df.drop(columns=['_umaban_key'])
            n_rmk = len([c for c in df.columns if c.startswith('rmk_')])
            print(f'[merge] remarks features: +{n_rmk} cols')
        else:
            print('[WARN] cannot merge remarks: base has no umaban column')

    # events
    n_evt = 0
    if os.path.exists(args.events):
        e = pd.read_csv(args.events, encoding='utf-8')
        e['race_id'] = e['race_id'].astype(str)
        e['horse_id'] = e['horse_id'].astype(str)
        # finish / top3 は base に既存の可能性 → suffix で区別
        evt_cols = ['race_id', 'horse_id'] + [c for c in e.columns
                                                 if any(k in c for k in ['change', '_up', '_down',
                                                                            '_rate_exp'])]
        e = e[evt_cols].drop_duplicates(['race_id', 'horse_id'])
        df = df.merge(e, on=['race_id', 'horse_id'], how='left')
        n_evt = e.shape[1] - 2
        print(f'[merge] events features: +{n_evt} cols')

    # output
    df.to_csv(args.out, index=False)
    print(f'\n[OK] V21 training features saved: {args.out}')
    print(f'[OK] final shape: {df.shape}')
    print(f'  - V15 base: {df.shape[1] - n_video - n_rmk - n_evt} cols')
    print(f'  - video AI: +{n_video}')
    print(f'  - remarks:  +{n_rmk}')
    print(f'  - events:   +{n_evt}')

    # 非 null 統計
    if n_video > 0:
        with_video = df.filter(like='gait_').notna().any(axis=1).sum()
        print(f'  - rows with video features: {with_video} ({with_video/len(df)*100:.2f}%)')
    if n_rmk > 0:
        with_rmk = df.filter(like='rmk_').notna().any(axis=1).sum()
        print(f'  - rows with remarks features: {with_rmk} ({with_rmk/len(df)*100:.1f}%)')
    if n_evt > 0:
        with_evt = (df.filter(like='_change').notna().any(axis=1) |
                     df.filter(like='_rate_exp').notna().any(axis=1)).sum()
        print(f'  - rows with event features: {with_evt} ({with_evt/len(df)*100:.1f}%)')

    return 0


if __name__ == '__main__':
    sys.exit(main())
