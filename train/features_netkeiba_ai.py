"""netkeiba マスター 未活用 AI 予想 data から features 抽出.

完全 未活用 4 source (V15 で 一切利用なし):
- netkeiba_ai_position.csv (67,952 rows) - 馬の予想通過位置 X/Y% + 信頼色
- netkeiba_race_analysis.csv (53,301 rows) - 各馬 個別 comment + score + eval
- netkeiba_ana_best.csv (41,653 rows) - 本命/上昇度/穴 カテゴリ分類
- netkeiba_ai_opinion.csv (4,929 rows) - pace prediction (H/M/S) + opinion text

これは ★ netkeiba AI 予想 を 我々の model にmeta-feature として stacking ★ 効果。

V15 .pkl.gz / predict_core / app.py 完全不変。
V20+/V22 学習で merge する 用。

usage:
    python train/features_netkeiba_ai.py
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / 'data'


def build_ai_position_features(fp: Path) -> pd.DataFrame:
    """AI position: 予想通過位置 X/Y% + color信頼度."""
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp, low_memory=False)
    df['race_id'] = df['race_id'].astype(str)
    df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(0).astype(int)
    df['ai_pos_left_pct'] = pd.to_numeric(df.get('position_left_pct'), errors='coerce').fillna(50)
    df['ai_pos_top_pct'] = pd.to_numeric(df.get('position_top_pct'), errors='coerce').fillna(50)
    # color_class encode: Color01-Color04
    df['ai_pos_color'] = df.get('color_class', 'Color00').astype(str).str.extract(r'(\d+)', expand=False)
    df['ai_pos_color'] = pd.to_numeric(df['ai_pos_color'], errors='coerce').fillna(0).astype(int)
    # 派生 features
    df['ai_pos_distance_to_center'] = np.sqrt(
        (df['ai_pos_left_pct'] - 50) ** 2 + (df['ai_pos_top_pct'] - 50) ** 2
    )
    df['ai_pos_is_top'] = (df['ai_pos_top_pct'] < 30).astype(int)  # 上位 (= 先頭付近)
    df['ai_pos_is_outer'] = (df['ai_pos_left_pct'] > 70).astype(int)  # 外側
    return df[['race_id', 'umaban',
               'ai_pos_left_pct', 'ai_pos_top_pct', 'ai_pos_color',
               'ai_pos_distance_to_center', 'ai_pos_is_top', 'ai_pos_is_outer']].drop_duplicates(['race_id', 'umaban'], keep='last')


def build_race_analysis_features(fp: Path) -> pd.DataFrame:
    """race_analysis: 各馬 score + evaluation."""
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp, low_memory=False)
    df['race_id'] = df['race_id'].astype(str)
    df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(0).astype(int)
    df['ai_analysis_score'] = pd.to_numeric(df.get('score'), errors='coerce').fillna(0)
    # evaluation は category (◎○▲△...), one-hot 化省略、 存在 flag のみ
    df['ai_analysis_has_eval'] = df.get('evaluation', '').astype(str).str.strip().astype(bool).astype(int)
    # comment は LLM 不要、 keyword count (簡易)
    comment_col = df.get('comment', pd.Series([''] * len(df))).astype(str)
    df['ai_analysis_comment_len'] = comment_col.str.len().fillna(0)
    positive_kw = ['好調', '絶好', '万全', '充実', '上昇', '楽勝']
    negative_kw = ['不安', '心配', '疲れ', '苦戦', '物足り']
    df['ai_analysis_pos_kw'] = comment_col.apply(
        lambda s: sum(s.count(kw) for kw in positive_kw))
    df['ai_analysis_neg_kw'] = comment_col.apply(
        lambda s: sum(s.count(kw) for kw in negative_kw))
    df['ai_analysis_net_score'] = df['ai_analysis_score'] + df['ai_analysis_pos_kw'] - df['ai_analysis_neg_kw']
    return df[['race_id', 'umaban',
               'ai_analysis_score', 'ai_analysis_has_eval',
               'ai_analysis_comment_len', 'ai_analysis_pos_kw',
               'ai_analysis_neg_kw', 'ai_analysis_net_score']].drop_duplicates(['race_id', 'umaban'], keep='last')


def build_ana_best_features(fp: Path) -> pd.DataFrame:
    """ana_best: race-level の category × horses → per-horse flag."""
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp, low_memory=False)
    df['race_id'] = df['race_id'].astype(str)
    # horses は 連結 string "2タイセイ3コッコ9エミジヤイミーン..."
    # 単純な flag だけ抽出: category 別 race-level flag
    # category: 本命 / 上昇度 / 穴 ...
    df['cat_label'] = df.get('category', '').astype(str)
    # race-level pivot
    out_records = []
    for rid, grp in df.groupby('race_id'):
        record = {'race_id': rid}
        for _, row in grp.iterrows():
            cat = row['cat_label']
            horses_str = str(row.get('horses', ''))
            # safe encode: each category as race-level flag (horse identify は別工程)
            if '本命' in cat or 'honmei' in cat.lower():
                record['ai_anabest_has_honmei'] = 1
            elif '上昇' in cat:
                record['ai_anabest_has_rise'] = 1
            elif '穴' in cat:
                record['ai_anabest_has_ana'] = 1
        out_records.append(record)
    out = pd.DataFrame(out_records)
    for c in ('ai_anabest_has_honmei', 'ai_anabest_has_rise', 'ai_anabest_has_ana'):
        if c not in out.columns:
            out[c] = 0
        out[c] = out[c].fillna(0).astype(int)
    return out[['race_id', 'ai_anabest_has_honmei', 'ai_anabest_has_rise', 'ai_anabest_has_ana']].drop_duplicates('race_id', keep='last')


def build_ai_opinion_features(fp: Path) -> pd.DataFrame:
    """ai_opinion: pace prediction (H/M/S) + opinion sentiment."""
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp, low_memory=False)
    df['race_id'] = df['race_id'].astype(str)
    pace_map = {'H': 2, 'M': 1, 'S': 0, '': 1, 'nan': 1}
    df['ai_opinion_pace'] = df.get('pace', '').astype(str).str.strip().map(pace_map).fillna(1).astype(int)
    text_col = df.get('opinion_text', pd.Series([''] * len(df))).astype(str)
    df['ai_opinion_text_len'] = text_col.str.len().fillna(0)
    df['ai_opinion_mentions_pace'] = text_col.str.contains('ペース', na=False).astype(int)
    df['ai_opinion_mentions_pos'] = text_col.str.contains('上昇|好調|絶好|有力', na=False).astype(int)
    df['ai_opinion_mentions_neg'] = text_col.str.contains('不安|心配|不利', na=False).astype(int)
    return df[['race_id', 'ai_opinion_pace', 'ai_opinion_text_len',
               'ai_opinion_mentions_pace', 'ai_opinion_mentions_pos',
               'ai_opinion_mentions_neg']].drop_duplicates('race_id', keep='last')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(DATA_DIR / 'features_netkeiba_ai.csv'))
    args = ap.parse_args()

    print('=== features_netkeiba_ai (V15 未活用 4 source 統合) ===')

    df_pos = build_ai_position_features(DATA_DIR / 'netkeiba_ai_position.csv')
    print(f'  ai_position: {len(df_pos):,} unique (race_id+umaban)')

    df_ana = build_race_analysis_features(DATA_DIR / 'netkeiba_race_analysis.csv')
    print(f'  race_analysis: {len(df_ana):,} unique')

    df_best = build_ana_best_features(DATA_DIR / 'netkeiba_ana_best.csv')
    print(f'  ana_best (race-level): {len(df_best):,} unique race_id')

    df_op = build_ai_opinion_features(DATA_DIR / 'netkeiba_ai_opinion.csv')
    print(f'  ai_opinion (race-level): {len(df_op):,} unique race_id')

    # merge: race+umaban level (pos + ana) + race-level (best + opinion)
    if not df_pos.empty:
        merged = df_pos
    else:
        merged = pd.DataFrame(columns=['race_id', 'umaban'])
    if not df_ana.empty:
        merged = merged.merge(df_ana, on=['race_id', 'umaban'], how='outer')
    if not df_best.empty:
        merged = merged.merge(df_best, on='race_id', how='left')
    if not df_op.empty:
        merged = merged.merge(df_op, on='race_id', how='left')

    # fillna
    for c in merged.columns:
        if c in ('race_id', 'umaban'):
            continue
        if merged[c].dtype.kind in 'iufb':
            merged[c] = merged[c].fillna(0)

    print(f'\n[merged] {len(merged):,} rows × {len(merged.columns)} cols')
    merged.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f'saved: {args.out}')

    print('\nsample:')
    print(merged.head(3).to_string())


if __name__ == '__main__':
    main()
