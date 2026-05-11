#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C7 (修正): race_review.csv の remarks (短い tag) を categorical feature 化.

netkeiba_race_review.csv は 277K 行、 73K が non-empty remarks。 unique は 37 種類の
**短い tag** (出遅れ / S不利 / 4角膨れる / 向正面接触 等)。 NLP embedding 不要、
カテゴリ flag / グルーピング が最適。

【V15 投資保護】 既存 train/ V15 関連 file 触らず、 新規 features csv 生成のみ。
V20/V21 学習時に merge_v15_1_features に追加可能な形式。

【設計】 37 unique remarks → 5 グループ + 個別 binary flag に encoding:
1. delay: 出遅れ
2. trouble: S不利, 向正面不利, 直線不利, 4角不利 etc
3. yore: ヨレル系 (向正面ヨレル, 直線ヨレル)
4. fukure: 膨れる系 (1角〜4角膨れる)
5. contact: 接触系 (向正面接触)
6. demote: 降着系 (1位入線降着, 2位入線降着)

加えて全 unique value を one-hot で残す option。

Usage:
    python tools/build_remarks_features.py
    python tools/build_remarks_features.py --one-hot
    python tools/build_remarks_features.py --inspect  # 全 unique remarks 一覧
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
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'netkeiba_race_review.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'race_review_features.csv')

# Group mapping
REMARK_GROUPS = {
    'delay': ['出遅れ', '出遅', 'スタートやや遅い'],
    'trouble': ['S不利', '不利', '向正面不利', '直線不利', '4角不利', '3角不利',
                '2角不利', '1角不利', 'スタート不利'],
    'yore': ['ヨレル', '向正面ヨレル', '直線ヨレル', '4角ヨレル'],
    'fukure': ['膨れる', '1角膨れる', '2角膨れる', '3角膨れる', '4角膨れる'],
    'contact': ['接触', '向正面接触', '直線接触', '4角接触', '3角接触', '2角接触', '1角接触'],
    'demote': ['1位入線降着', '2位入線降着', '降着'],
    'late_pace': ['スローペース', '後方', '差し届かず'],
    'fast_pace': ['ハイペース', 'ハナ争い'],
}


def classify_remark(remark):
    """remark → group flag dict."""
    if not remark or not isinstance(remark, str):
        return {g: 0 for g in REMARK_GROUPS}
    flags = {}
    for group, keywords in REMARK_GROUPS.items():
        flags[group] = int(any(kw in remark for kw in keywords))
    return flags


def main():
    ap = argparse.ArgumentParser(description='Race review remarks → categorical features (C7)')
    ap.add_argument('--one-hot', action='store_true', help='全 unique remarks one-hot 化 (37 cols)')
    ap.add_argument('--inspect', action='store_true', help='unique remarks 一覧 表示')
    args = ap.parse_args()

    import pandas as pd
    print(f'[INFO] reading: {INPUT_PATH}')
    df = pd.read_csv(INPUT_PATH, encoding='utf-8')
    print(f'[INFO] {len(df)} rows, {df["remarks"].notna().sum()} non-null remarks')

    if args.inspect:
        unique = df['remarks'].dropna().astype(str).value_counts()
        print(f'[INSPECT] {len(unique)} unique remarks:')
        for r, c in unique.items():
            print(f'  {c:>6}  {r}')
        return 0

    # group flags 生成
    df['remarks_filled'] = df['remarks'].fillna('')
    group_df = pd.DataFrame([classify_remark(r) for r in df['remarks_filled']])
    group_df.columns = [f'rmk_{c}' for c in group_df.columns]

    # any remark flag
    group_df['rmk_any'] = (df['remarks_filled'].str.len() > 0).astype(int)

    result = pd.concat([
        df[['race_id', 'umaban', 'horse_name', 'finish', 'review_score']].reset_index(drop=True),
        group_df.reset_index(drop=True),
    ], axis=1)

    if args.one_hot:
        unique_remarks = df['remarks_filled'].value_counts()
        # 上位 30 件を one-hot
        top_remarks = unique_remarks[unique_remarks.index != ''].head(30).index.tolist()
        for r in top_remarks:
            col = f'rmk_oh_{r}'
            result[col] = (df['remarks_filled'] == r).astype(int)
        print(f'[INFO] one-hot: {len(top_remarks)} top remarks')

    result.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
    print(f'[OK] saved: {OUTPUT_PATH}')
    print(f'[OK] shape: {result.shape}, cols: {list(result.columns)[:15]}...')

    # 統計
    for col in [c for c in result.columns if c.startswith('rmk_')]:
        rate = result[col].mean()
        n = result[col].sum()
        if n > 0:
            print(f'  {col}: {n:>6} rows ({rate*100:.1f}%)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
