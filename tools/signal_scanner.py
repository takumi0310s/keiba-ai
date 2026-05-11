#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comprehensive signal scanner: 全 候補 features を 自動 scan して +5pt 以上の signal 列挙.

V20 候補 features 14 件は既 verify 済。 他に未発見の strong signal がないか
網羅的 grid search (binary + continuous quintile)。

【V15 投資保護】 検索のみ、 V15 model 不変

Usage:
    python tools/signal_scanner.py
    python tools/signal_scanner.py --threshold 0.05  # 5pt 以上 表示
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
    ap = argparse.ArgumentParser(description='Signal scanner')
    ap.add_argument('--threshold', type=float, default=0.05,
                    help='top3 rate delta threshold (default 0.05 = 5pt)')
    args = ap.parse_args()

    import pandas as pd
    base = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    df = pd.read_csv(base, encoding='utf-8', low_memory=False)
    df = df[df['year'] >= 22]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['win'] = (df['finish'] == 1).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)
    base_top3 = df['top3'].mean()
    print(f'[INFO] {len(df):,} rows、 baseline top3 rate: {base_top3:.3f}')

    # 既存 base columns で binary / categorical signal scan
    binary_candidates = []
    for col in df.columns:
        if df[col].dtype not in ('int64', 'float64'):
            continue
        if df[col].isna().sum() > len(df) * 0.5:
            continue
        u = df[col].dropna().unique()
        if len(u) == 2 and set(u) <= {0, 1}:
            binary_candidates.append(col)

    print(f'[INFO] binary candidates: {len(binary_candidates)}')

    # 数値型で 5 quintile 化
    numeric_candidates = []
    for col in df.columns:
        if df[col].dtype not in ('int64', 'float64'):
            continue
        if col in ('finish', 'top3', 'win', 'top1_finish'):
            continue
        if df[col].notna().sum() < 10000:
            continue
        if df[col].nunique() < 10:
            continue
        numeric_candidates.append(col)

    print(f'[INFO] numeric candidates: {len(numeric_candidates)}')

    found = []

    # binary
    for col in binary_candidates:
        if df[col].sum() < 1000:
            continue
        tr1 = df[df[col] == 1]['top3'].mean()
        tr0 = df[df[col] == 0]['top3'].mean()
        delta = tr1 - tr0
        if abs(delta) >= args.threshold:
            found.append({
                'feature': col,
                'type': 'binary',
                'n_pos': int(df[col].sum()),
                'top3_when_1': round(tr1, 3),
                'top3_when_0': round(tr0, 3),
                'delta': round(delta, 3),
            })

    # numeric quintile
    for col in numeric_candidates:
        valid = df[df[col].notna()].copy()
        if len(valid) < 1000:
            continue
        try:
            valid['_q'] = pd.qcut(valid[col].rank(method='first'),
                                    5, labels=[1, 2, 3, 4, 5])
            tr5 = valid[valid['_q'] == 5]['top3'].mean()
            tr1 = valid[valid['_q'] == 1]['top3'].mean()
            delta = tr5 - tr1
            if abs(delta) >= args.threshold:
                found.append({
                    'feature': col,
                    'type': 'quintile',
                    'n': len(valid),
                    'top3_Q5': round(tr5, 3),
                    'top3_Q1': round(tr1, 3),
                    'delta': round(delta, 3),
                })
        except Exception:
            pass

    # sort by abs(delta)
    found_sorted = sorted(found, key=lambda x: -abs(x['delta']))

    print(f'\n=== Signal scanner 結果 ({len(found)} features、 threshold ≥ {args.threshold:.3f}) ===')
    for f in found_sorted[:30]:
        if f['type'] == 'binary':
            print(f'  {f["delta"]:+.3f}  [bin ] {f["feature"]:<30} n_pos={f["n_pos"]:>6,}, '
                  f'top3 when 1: {f["top3_when_1"]} when 0: {f["top3_when_0"]}')
        else:
            print(f'  {f["delta"]:+.3f}  [quin] {f["feature"]:<30} n={f["n"]:>6,}, '
                  f'Q5: {f["top3_Q5"]} Q1: {f["top3_Q1"]}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
