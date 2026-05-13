"""sentiment features を jrdb_kyi race_id mapping で V22 format に変換.

入力:
- data/features_stable_comment_sentiment.csv (race_id = 12桁 netkeiba)
- data/features_race_review_sentiment.csv (race_id = 12桁 netkeiba)
- data/jrdb_kyi.csv (jra_race_id 10桁 ↔ nk_race_id 12桁 21,141 unique)

出力:
- data/features_sentiment_merged.csv (race_id = 10桁 jra format)

V22 enhanced 再 retrain で 使う 用。 今回 retrain は 含まず (時間制約)、 次 retrain (V20 真の構築 5/24+) で 使う。

usage:
    python train/features_merge_sentiment.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent


def build_id_mapping(kyi_csv: str) -> dict:
    """jrdb_kyi.csv から nk_race_id → jra_race_id mapping を構築."""
    df = pd.read_csv(kyi_csv, encoding='utf-8-sig', usecols=['jra_race_id', 'nk_race_id'],
                     dtype=str, low_memory=False)
    df = df.drop_duplicates('nk_race_id', keep='last')
    return dict(zip(df['nk_race_id'], df['jra_race_id']))


def merge_sentiment(stable_csv: str, review_csv: str, mapping: dict) -> pd.DataFrame:
    out_dfs = []
    for tag, fp in [('stable', stable_csv), ('review', review_csv)]:
        if not os.path.exists(fp):
            print(f'  [SKIP] {fp}')
            continue
        df = pd.read_csv(fp, encoding='utf-8-sig', low_memory=False)
        df['race_id'] = df['race_id'].astype(str)
        # race_id を 12桁 nk → 10桁 jra に変換
        df['jra_race_id'] = df['race_id'].map(mapping)
        matched = df['jra_race_id'].notna().sum()
        print(f'  [{tag}] {fp}: {len(df):,} rows, mapped {matched:,} ({matched/len(df)*100:.1f}%)')
        df = df.dropna(subset=['jra_race_id'])
        df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(0).astype(int)
        # tag prefix 付け
        feat_cols = [c for c in df.columns if c not in
                     ('race_id', 'jra_race_id', 'umaban', 'race_date', 'horse_name',
                      'finish', 'score', 'review_score')]
        rename = {c: f'sentiment_{tag}__{c}' for c in feat_cols}
        df = df.rename(columns=rename)
        df_sub = df[['jra_race_id', 'umaban'] + list(rename.values())].rename(
            columns={'jra_race_id': 'race_id'})
        df_sub = df_sub.drop_duplicates(['race_id', 'umaban'], keep='last')
        out_dfs.append(df_sub)
    if not out_dfs:
        return pd.DataFrame()

    merged = out_dfs[0]
    for d in out_dfs[1:]:
        merged = merged.merge(d, on=['race_id', 'umaban'], how='outer')

    # NaN fill 0
    feat_cols = [c for c in merged.columns if c not in ('race_id', 'umaban')]
    merged[feat_cols] = merged[feat_cols].fillna(0)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stable',
                    default=str(BASE / 'data' / 'features_stable_comment_sentiment.csv'))
    ap.add_argument('--review',
                    default=str(BASE / 'data' / 'features_race_review_sentiment.csv'))
    ap.add_argument('--kyi', default=str(BASE / 'data' / 'jrdb_kyi.csv'))
    ap.add_argument('--output', default=str(BASE / 'data' / 'features_sentiment_merged.csv'))
    args = ap.parse_args()

    print(f"[sentiment_merge] building id mapping from {args.kyi}")
    mapping = build_id_mapping(args.kyi)
    print(f"  mapping entries: {len(mapping):,}")

    df = merge_sentiment(args.stable, args.review, mapping)
    if df.empty:
        print('[sentiment_merge] no data merged.')
        return

    print(f"[sentiment_merge] writing {args.output} ({len(df):,} rows × {len(df.columns)} cols)")
    df.to_csv(args.output, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
