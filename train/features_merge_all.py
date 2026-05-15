"""features 統合 module: 全 features_*.csv を V22 学習用 1 file に merge.

入力 (全 5/13 PM 作成):
- data/features_free.csv (38 cols、 round 1)
- data/features_jrdb_ukc.csv (13 cols、 round 1)
- data/features_stable_comment_sentiment.csv (10 cols、 round 1)
- data/features_race_review_sentiment.csv (10 cols、 round 1)
- data/features_track_lap.csv (17 cols、 round 2)
- data/features_jrdb_jo.csv (22 cols、 round 2)
- data/features_horse_advanced.csv (18 cols、 round 2)
- data/features_venue.csv (15 cols、 round 2)

出力:
- data/features_merged_all.csv (race_id + horse_id + umaban + 全 features)

V20+/V22 学習で V15 cache に LEFT JOIN する 用。
LEAK-free 保証 (元 csv が expanding window で計算済)。

usage:
    python train/features_merge_all.py [--start-year 20]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

FEATURE_FILES = {
    # 全 race_id が jra_races_full 10桁 format (matches V15 cache)
    'free': 'features_free.csv',
    'jrdb_ukc': 'features_jrdb_ukc.csv',
    'track_lap': 'features_track_lap.csv',
    'jrdb_jo': 'features_jrdb_jo.csv',
    'horse_advanced': 'features_horse_advanced.csv',
    'venue': 'features_venue.csv',
    # 5/15 追加 (V15 越え 試行 用)
    # 'master_idx': 'features_master_index_expanding.csv',  # race_id 12桁 nk、 format mismatch で 除外
    'nk_ai': 'features_netkeiba_ai.csv',
    'nk_extra': 'features_netkeiba_extra.csv',
    # sentiment 系 (netkeiba 12桁 race_id) は format 不一致のため 別 module で merge
}

# merge key 統一
KEY_COLS = ['race_id', 'horse_id', 'umaban']


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'race_id' in df.columns:
        df['race_id'] = df['race_id'].astype(str)
    if 'horse_id' in df.columns:
        df['horse_id'] = pd.to_numeric(df['horse_id'], errors='coerce').astype('Int64').astype(str)
    if 'umaban' in df.columns:
        df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(0).astype(int)
    return df


def merge_all(data_dir: Path) -> pd.DataFrame:
    """全 features csv を read + merge。"""
    merged = None
    used = []
    total_added_cols = 0
    for tag, fname in FEATURE_FILES.items():
        fp = data_dir / fname
        if not fp.exists():
            print(f'  [SKIP] {fname} not found')
            continue
        df = pd.read_csv(fp, encoding='utf-8-sig', low_memory=False)
        n_rows = len(df)
        n_cols = len(df.columns)
        df = normalize_keys(df)

        # key cols 不足 check
        available_keys = [k for k in KEY_COLS if k in df.columns]
        if len(available_keys) < 2:
            print(f'  [SKIP] {fname}: only keys {available_keys} (need at least 2)')
            continue

        # merge key 選定: 全 cols あれば 3 つ、 なければ race_id + umaban
        if 'horse_id' in df.columns and 'umaban' in df.columns:
            merge_keys = ['race_id', 'umaban']  # horse_id は redundant、 整合性 issue 多い
        else:
            merge_keys = available_keys[:2]

        # 重複 key 除外
        df = df.drop_duplicates(merge_keys, keep='last')

        # feature cols 抽出 (key 以外)
        feat_cols = [c for c in df.columns if c not in KEY_COLS
                     and c not in ('race_date', 'horse_name', 'finish', 'score', 'review_score')]
        df_sub = df[merge_keys + feat_cols].copy()

        # tag prefix 付与 (column 名衝突回避)
        rename_map = {c: f'{tag}__{c}' for c in feat_cols}
        df_sub = df_sub.rename(columns=rename_map)
        prefixed_cols = list(rename_map.values())

        print(f'  [LOAD] {fname}: {n_rows:,} rows × {n_cols} cols → {len(prefixed_cols)} prefixed cols (key: {merge_keys})')

        if merged is None:
            merged = df_sub
        else:
            merged = merged.merge(df_sub, on=merge_keys, how='outer')
        used.append((tag, len(prefixed_cols)))
        total_added_cols += len(prefixed_cols)

    print(f'  total: {len(merged):,} rows × {len(merged.columns)} cols ({total_added_cols} feature cols)')

    # NaN fill (新 feature 多くは expanding/sentiment、 NaN → 0 か中性値)
    for c in merged.columns:
        if c in KEY_COLS:
            continue
        # 平均値で fill (safer than 0)
        if merged[c].dtype.kind in 'iufb':
            mean = merged[c].mean()
            merged[c] = merged[c].fillna(mean if not pd.isna(mean) else 0)

    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=str(BASE / 'data'))
    ap.add_argument('--output', default=str(BASE / 'data' / 'features_merged_all.csv'))
    args = ap.parse_args()

    print(f"[features_merge_all] reading from {args.data_dir}")
    df = merge_all(Path(args.data_dir))

    print(f"[features_merge_all] writing {args.output} ({len(df):,} rows × {len(df.columns)} cols)")
    df.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(f"\nsample (head 3, key cols):")
    print(df[KEY_COLS].head(3) if all(k in df.columns for k in KEY_COLS) else df.head(3))


if __name__ == '__main__':
    main()
