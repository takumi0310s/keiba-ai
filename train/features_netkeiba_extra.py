"""netkeiba マスター 未活用 4 source (追加発見) から features 抽出.

V15 で 完全未活用 (本日 second batch 発見):
- netkeiba_training_eval.csv (302K rows) - 調教評価詳細 (rank/course/intensity)
- netkeiba_upset_level.csv (36K rows) - 番狂わせ レベル + 人気馬 信頼度
- netkeiba_track_index.csv (20K rows) - 馬場 index + comment

★ 注意 ★:
- netkeiba_master_index.csv (139K) は finish_order 含む = post-race、 expanding 化 必要のため別途
- 本 module は LEAK-free な race-level / pre-race features のみ

V15 .pkl.gz / predict_core / app.py 完全不変。

usage:
    python train/features_netkeiba_extra.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / 'data'


def build_training_eval_features(fp: Path) -> pd.DataFrame:
    """調教評価詳細 features."""
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp, low_memory=False)
    df['race_id'] = df['race_id'].astype(str)
    df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(0).astype(int)

    # training_rank: A/B/C/D → 4/3/2/1
    rank_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'a': 4, 'b': 3, 'c': 2, 'd': 1}
    df['te_rank_score'] = df.get('training_rank', '').astype(str).str.strip().map(rank_map).fillna(0).astype(int)

    # training_intensity: 重い/普通/軽い → 数値化 (keyword)
    intensity_text = df.get('training_intensity', '').astype(str)
    df['te_intensity_heavy'] = intensity_text.str.contains('重|強', na=False).astype(int)
    df['te_intensity_light'] = intensity_text.str.contains('軽|易', na=False).astype(int)

    # training_position: 1-多 (調教 通過順位)
    df['te_position'] = pd.to_numeric(df.get('training_position', 0), errors='coerce').fillna(0).astype(int)

    # training_move: 抜群 / 良い / 普通 / 不振 etc.
    move_text = df.get('training_move', '').astype(str)
    df['te_move_excellent'] = move_text.str.contains('抜群|良', na=False).astype(int)
    df['te_move_poor'] = move_text.str.contains('不振|物足り|平凡', na=False).astype(int)
    df['te_move_smooth'] = move_text.str.contains('滑ら|スムーズ', na=False).astype(int)

    # training_course: 美浦坂路 / 栗東坂路 / 美浦CW / 栗東CW etc. → ☆ encoded category
    course_text = df.get('training_course', '').astype(str)
    df['te_course_sakaro'] = course_text.str.contains('坂路', na=False).astype(int)
    df['te_course_wood'] = course_text.str.contains('CW|W', na=False).astype(int)
    df['te_course_pool'] = course_text.str.contains('プ', na=False).astype(int)

    # training_condition: 良好 / 不良 / etc.
    cond_text = df.get('training_condition', '').astype(str)
    df['te_cond_good'] = cond_text.str.contains('良|好', na=False).astype(int)
    df['te_cond_bad'] = cond_text.str.contains('悪|不', na=False).astype(int)

    out_cols = ['race_id', 'umaban',
                'te_rank_score', 'te_intensity_heavy', 'te_intensity_light',
                'te_position', 'te_move_excellent', 'te_move_poor', 'te_move_smooth',
                'te_course_sakaro', 'te_course_wood', 'te_course_pool',
                'te_cond_good', 'te_cond_bad']
    return df[out_cols].drop_duplicates(['race_id', 'umaban'], keep='last')


def build_upset_features(fp: Path) -> pd.DataFrame:
    """番狂わせ レベル (race-level)."""
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp, low_memory=False)
    df['race_id'] = df['race_id'].astype(str)
    df['upset_level_raw'] = pd.to_numeric(df.get('upset_level', 0), errors='coerce').fillna(0).astype(int)
    df['top_pop_reliability'] = pd.to_numeric(df.get('top_popularity_reliability'),
                                              errors='coerce').fillna(0.5)
    out = df[['race_id', 'upset_level_raw', 'top_pop_reliability']].copy()
    return out.drop_duplicates('race_id', keep='last')


def build_track_index_features(fp: Path) -> pd.DataFrame:
    """馬場 index (race-level)."""
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp, low_memory=False)
    df['race_id'] = df['race_id'].astype(str)
    df['track_index_nk'] = pd.to_numeric(df.get('track_index'), errors='coerce').fillna(0)

    # track_comment から keyword 抽出
    comment_text = df.get('track_comment', '').astype(str)
    df['tk_comment_inner_advantage'] = comment_text.str.contains('内[^外]*[有伸]', na=False).astype(int)
    df['tk_comment_outer_advantage'] = comment_text.str.contains('外[^内]*[有伸]', na=False).astype(int)
    df['tk_comment_renovated'] = comment_text.str.contains('改修|新装|刷新', na=False).astype(int)
    df['tk_comment_heavy'] = comment_text.str.contains('重|含水', na=False).astype(int)
    df['tk_comment_good'] = comment_text.str.contains('良好|良状態', na=False).astype(int)

    return df[['race_id', 'track_index_nk',
               'tk_comment_inner_advantage', 'tk_comment_outer_advantage',
               'tk_comment_renovated', 'tk_comment_heavy', 'tk_comment_good']].drop_duplicates('race_id', keep='last')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(DATA_DIR / 'features_netkeiba_extra.csv'))
    args = ap.parse_args()

    print('=== features_netkeiba_extra (V15 未活用 second batch) ===')

    df_te = build_training_eval_features(DATA_DIR / 'netkeiba_training_eval.csv')
    print(f'  training_eval: {len(df_te):,} unique (race_id+umaban)')

    df_up = build_upset_features(DATA_DIR / 'netkeiba_upset_level.csv')
    print(f'  upset_level (race-level): {len(df_up):,} unique')

    df_tk = build_track_index_features(DATA_DIR / 'netkeiba_track_index.csv')
    print(f'  track_index (race-level): {len(df_tk):,} unique')

    # merge
    if not df_te.empty:
        merged = df_te
    else:
        merged = pd.DataFrame(columns=['race_id', 'umaban'])
    if not df_up.empty:
        merged = merged.merge(df_up, on='race_id', how='left')
    if not df_tk.empty:
        merged = merged.merge(df_tk, on='race_id', how='left')

    # fillna
    for c in merged.columns:
        if c in ('race_id', 'umaban'):
            continue
        if merged[c].dtype.kind in 'iufb':
            merged[c] = merged[c].fillna(0)

    print(f'\n[merged] {len(merged):,} rows × {len(merged.columns)} cols')
    merged.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f'saved: {args.out}')


if __name__ == '__main__':
    main()
