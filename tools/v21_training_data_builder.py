#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V21 training data builder (V20 training data + video features merge).

v20_training_data_full.csv (101 cols) + v21_video_features_all.csv (33 video features) を
horse_id + race_id 単位で merge し、 V21 学習 data 生成。

注: race_id format mismatch:
- v20 = TFJV format (10 digit、 umaban 込み)
- v21_video = netkeiba format (12 digit)
→ horse_id only で merge、 race_id matching は 諦め (current 制約)

【V15 投資保護】 既存 csv 不変、 V21 新 csv 出力のみ

Usage:
    python tools/v21_training_data_builder.py
    python tools/v21_training_data_builder.py --year-from 2020 --year-to 2026
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
    ap = argparse.ArgumentParser(description='V21 training data builder')
    ap.add_argument('--year-from', type=int, default=2020)
    ap.add_argument('--year-to', type=int, default=2026)
    ap.add_argument('--out', default=os.path.join(BASE_DIR, 'data', 'v21_training_data_full.csv'))
    args = ap.parse_args()

    import pandas as pd

    # V20 base data load
    v20_path = os.path.join(BASE_DIR, 'data', 'v20_training_data_full.csv')
    print(f'[INFO] loading V20 base: {v20_path}')
    df = pd.read_csv(v20_path, encoding='utf-8', low_memory=False)
    yf = args.year_from - 2000 if args.year_from >= 2000 else args.year_from
    yt = args.year_to - 2000 if args.year_to >= 2000 else args.year_to
    df = df[(df['year'] >= yf) & (df['year'] <= yt)]
    print(f'  V20 shape: {df.shape}')

    # horse_id mapping: V20 (TFJV 8-digit) → V21 (netkeiba 10-digit)
    # rule: prepend "20" to 8-digit zero-padded V20 horse_id
    # horse_id is loaded as float64 (due to NaN); convert via Int64 to strip .0
    df['horse_id_int'] = pd.to_numeric(df['horse_id'], errors='coerce').astype('Int64')
    df = df[df['horse_id_int'].notna()].copy()
    df['horse_id_str'] = df['horse_id_int'].astype(str)
    df['horse_id_v21'] = '20' + df['horse_id_str'].str.zfill(8)

    # Video features load
    v21_path = os.path.join(BASE_DIR, 'data', 'v21_video_features_all.csv')
    if not os.path.exists(v21_path):
        print(f'[ERROR] {v21_path} not found, run v21_extract_all_video_features.py first')
        return 1
    video = pd.read_csv(v21_path, encoding='utf-8')
    print(f'  Video features shape: {video.shape}')
    video['horse_id_v21'] = video['horse_id'].astype(str)
    video_cols = [c for c in video.columns if c.startswith('video_')]
    print(f'  Video features: {len(video_cols)}')

    # merge by netkeiba-format horse_id
    video_dedup = video.drop_duplicates('horse_id_v21', keep='last')[
        ['horse_id_v21'] + video_cols]
    print(f'\n[INFO] merging via netkeiba 10-digit horse_id...')
    df = df.merge(video_dedup, on='horse_id_v21', how='left')

    print(f'\n[OK] V21 merged shape: {df.shape}')
    print(f'[OK] coverage (video features):')
    for c in video_cols[:5]:
        rate = df[c].notna().mean()
        print(f'  {c}: {rate*100:.2f}%')

    # Save
    df.to_csv(args.out, index=False)
    print(f'\n[OK] saved: {args.out}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
