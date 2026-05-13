"""厩舎コメント / レース短評 簡易 sentiment 数値化.

netkeiba_stable_comments.csv / netkeiba_race_review.csv から
keyword-based 簡易 sentiment score を算出。 LLM は不要、 license free。

V20+/V22 学習で merge。 V15 production / predict_core 不変。

scoring rules (Bayesian smoothing 不要、 単純 dictionary):
- positive: +1 each (最高/絶好調/万全/順調/堂々/楽勝/上昇/絶賛 等)
- negative: -1 each (不安/落鉄/疲れ/失速/重い/不調/休み明け 等)
- neutral 中性: 0 (普通/可もなく/まずまず)
- 連投 (体重マイナス/...) は重み -2

usage:
    python train/features_sentiment.py
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent

POSITIVE_KW = [
    '絶好調', '最高', '万全', '順調', '堂々', '楽勝', '上昇',
    '余裕', '完璧', '充実', '好調', '気合', '勢い', '抜群',
    '輝き', '威圧', '伸び', '快走', 'ベスト', '充分', '充足',
    '集中', '冷静', '完成', '充電', '張り', '映え',
]
NEGATIVE_KW = [
    '不安', '疲れ', '失速', '重い', '不調', '出遅れ', '落鉄',
    '苦戦', '物足り', '心配', '怪我', '違和感', '休み明け',
    '太め', '緩い', '硬い', '辛い', '苦しい', '伸びない',
    '足元', '掻く', '気性', '荒い', '物見', '掛か', '張り過ぎ',
]
NEUTRAL_KW = ['普通', '可もなく', 'まずまず', '無難', '平凡']

CRITICAL_NEG = ['取消', '出走除外', '骨折', '故障', '休養', '放牧']


def compute_score(text: str) -> dict:
    """text → score dict.

    Returns:
        positive_count, negative_count, neutral_count, critical_neg,
        net_score (pos - neg - 2*crit)
    """
    if not isinstance(text, str) or not text:
        return dict(positive_count=0, negative_count=0,
                    neutral_count=0, critical_neg=0, net_score=0.0)
    t = text
    pos = sum(t.count(kw) for kw in POSITIVE_KW)
    neg = sum(t.count(kw) for kw in NEGATIVE_KW)
    neu = sum(t.count(kw) for kw in NEUTRAL_KW)
    crit = sum(t.count(kw) for kw in CRITICAL_NEG)
    return dict(positive_count=pos, negative_count=neg,
                neutral_count=neu, critical_neg=crit,
                net_score=float(pos - neg - 2 * crit))


def build_stable_comment_features(csv_path: str) -> pd.DataFrame:
    """厩舎コメント (stable_comments) → score."""
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)
    if 'comment' not in df.columns:
        for c in ('text', 'content', 'message'):
            if c in df.columns:
                df = df.rename(columns={c: 'comment'})
                break
    if 'comment' not in df.columns:
        print(f"[WARN] no comment column in {csv_path}, cols: {df.columns.tolist()[:8]}")
        return pd.DataFrame()
    print(f"[stable_comment] {csv_path}: {len(df):,} rows")
    scores = df['comment'].apply(compute_score)
    sc_df = pd.DataFrame(list(scores))
    sc_df.columns = ['stable_comment_' + c for c in sc_df.columns]
    out = pd.concat([df.drop(columns=['comment']), sc_df], axis=1)
    return out


def build_race_review_features(csv_path: str) -> pd.DataFrame:
    """レース短評 / 備考 (race_review) → 前走 short_eval score."""
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)
    text_col = None
    for c in ('remarks', 'review', 'short_review', 'biko', '備考', 'comment', 'text'):
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        print(f"[WARN] no text column in {csv_path}, cols: {df.columns.tolist()[:10]}")
        return pd.DataFrame()
    print(f"[race_review] {csv_path}: {len(df):,} rows (text col: {text_col})")
    scores = df[text_col].apply(compute_score)
    sc_df = pd.DataFrame(list(scores))
    sc_df.columns = ['race_review_' + c for c in sc_df.columns]
    out = pd.concat([df.drop(columns=[text_col]), sc_df], axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stable',
                    default=str(BASE / 'data' / 'netkeiba_stable_comments.csv'))
    ap.add_argument('--review',
                    default=str(BASE / 'data' / 'netkeiba_race_review.csv'))
    ap.add_argument('--out-dir', default=str(BASE / 'data'))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    sc = build_stable_comment_features(args.stable)
    if not sc.empty:
        p = out_dir / 'features_stable_comment_sentiment.csv'
        sc.to_csv(p, index=False, encoding='utf-8-sig')
        print(f"  wrote {p} ({len(sc):,} rows × {len(sc.columns)} cols)")

    rv = build_race_review_features(args.review)
    if not rv.empty:
        p = out_dir / 'features_race_review_sentiment.csv'
        rv.to_csv(p, index=False, encoding='utf-8-sig')
        print(f"  wrote {p} ({len(rv):,} rows × {len(rv.columns)} cols)")


if __name__ == '__main__':
    main()
