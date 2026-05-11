#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V15 backtest / cumulative_results から calibration 用 (pred, label) CSV 抽出.

calibrate_confidence.py の fit に渡せる形式に変換。
複数 source 対応:
1. data/cumulative_results.csv (top1_score / top1_finish) - 限定的 (21 rows)
2. train/train_v135b_intra_ensemble.py の WF 出力 (要追加実装)
3. 既存 actual_roi_v135b.json (aggregate only、 per-race 無)

【現状】 source 1 のみ動作。 source 2 は train script 改造 必要 (V15 投資保護 慎重)。

Usage:
    # cumulative から抽出
    python tools/extract_calibration_data.py cumulative
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


def cmd_cumulative(args):
    import pandas as pd
    path = os.path.join(BASE_DIR, 'data', 'cumulative_results.csv')
    if not os.path.exists(path):
        print(f'[ERROR] not found: {path}')
        return 1
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['pred'] = pd.to_numeric(df['top1_score'], errors='coerce')
    df['top1_finish_int'] = pd.to_numeric(df['top1_finish'], errors='coerce')
    df['label'] = (df['top1_finish_int'] <= 3).astype(int)

    valid = df[df['pred'].notna() & df['top1_finish_int'].notna()].copy()
    print(f'[INFO] valid (pred, label) pairs: {len(valid)} / total {len(df)} ({len(valid)/len(df)*100:.1f}%)')

    if len(valid) < 30:
        print('[WARN] sample too small (<30) for reliable calibration. ')
        print('       train/train_v135b_intra_ensemble.py で per-fold prediction CSV 出力 改造 推奨')

    if len(valid) == 0:
        return 1

    out_path = args.out
    valid[['race_id', 'pred', 'label']].to_csv(out_path, index=False)
    print(f'[OK] saved: {out_path} ({len(valid)} rows)')

    # stats
    print(f'\n[stats]')
    print(f'  pred range: [{valid["pred"].min():.3f}, {valid["pred"].max():.3f}]')
    print(f'  label==1 rate: {valid["label"].mean():.3f}')
    return 0


def main():
    ap = argparse.ArgumentParser(description='Extract (pred,label) for calibration')
    sub = ap.add_subparsers(dest='cmd', required=True)
    cum = sub.add_parser('cumulative', help='cumulative_results.csv から抽出')
    cum.add_argument('--out', default=os.path.join(BASE_DIR, 'data', 'calibration_train.csv'))
    args = ap.parse_args()
    if args.cmd == 'cumulative':
        return cmd_cumulative(args)
    return 1


if __name__ == '__main__':
    sys.exit(main())
