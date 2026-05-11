#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""daily_predictions × daily_results から calibration 用 (pred, label) を大量生成.

extract_calibration_data.py は cumulative_results.csv 使用で 21 sample 限定。
本 script は daily_predictions/*.csv × daily_results/*.csv の全 history を 使い、
真の calibration 用 dataset を生成。 期待: 数百 race 以上のサンプル。

【V15 投資保護】 既存 csv の読み込みのみ、 production 一切 不変

Usage:
    python tools/build_calibration_from_daily.py
    python tools/build_calibration_from_daily.py --from 20260301 --to 20260510
    python tools/build_calibration_from_daily.py --fit  # 自動 calibrator fit まで
"""
import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(BASE_DIR, 'data', 'daily_predictions')
RESULT_DIR = os.path.join(BASE_DIR, 'data', 'daily_results')
OUT_PATH = os.path.join(BASE_DIR, 'data', 'calibration_full.csv')


def main():
    ap = argparse.ArgumentParser(description='daily 両 csv から calibration data')
    ap.add_argument('--from', dest='from_date', default=None)
    ap.add_argument('--to', dest='to_date', default=None)
    ap.add_argument('--out', default=OUT_PATH)
    ap.add_argument('--fit', action='store_true', help='生成後 自動 calibrator fit')
    args = ap.parse_args()

    import pandas as pd

    # Gather daily_predictions
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, '*.csv')))
    print(f'[INFO] {len(pred_files)} prediction files found')

    pairs = []
    for fp in pred_files:
        date = os.path.basename(fp).replace('.csv', '')
        if not date.isdigit():
            continue
        if args.from_date and date < args.from_date:
            continue
        if args.to_date and date > args.to_date:
            continue

        result_path = os.path.join(RESULT_DIR, f'{date}.csv')
        if not os.path.exists(result_path):
            continue

        try:
            pred_df = pd.read_csv(fp, encoding='utf-8-sig')
            res_df = pd.read_csv(result_path, encoding='utf-8-sig')
        except Exception as e:
            print(f'  [WARN] {date}: {e}')
            continue

        # match by race_id
        pred_df['race_id'] = pred_df['race_id'].astype(str)
        res_df['race_id'] = res_df['race_id'].astype(str)

        # res should have outcome info (top1_finish? trio_hit?)
        if 'top1_finish' not in res_df.columns:
            continue

        merged = pred_df.merge(res_df[['race_id', 'top1_finish']], on='race_id', how='inner')
        if 'top1_score' not in merged.columns:
            continue
        merged['pred'] = pd.to_numeric(merged['top1_score'], errors='coerce')
        merged['top1_finish_int'] = pd.to_numeric(merged['top1_finish'], errors='coerce')
        merged['label'] = (merged['top1_finish_int'] <= 3).astype(int)
        merged = merged.dropna(subset=['pred', 'top1_finish_int'])
        merged['date'] = date
        pairs.append(merged[['race_id', 'date', 'pred', 'label']])

    if not pairs:
        print('[ERROR] no valid (pred, label) pairs found')
        return 1
    df = pd.concat(pairs, ignore_index=True)
    df = df.drop_duplicates(['race_id'])
    print(f'[INFO] total unique (pred, label) pairs: {len(df)}')

    df.to_csv(args.out, index=False)
    print(f'[OK] saved: {args.out}')

    # stats
    print(f'\n[stats]')
    print(f'  pred range: [{df["pred"].min():.3f}, {df["pred"].max():.3f}]')
    print(f'  pred mean: {df["pred"].mean():.3f}')
    print(f'  label==1 rate: {df["label"].mean():.3f}')

    if args.fit and len(df) >= 30:
        print(f'\n[CALIBRATOR FIT]')
        out_pkl = args.out.replace('.csv', '_calibrator.pkl')
        r = subprocess.run([sys.executable,
                              os.path.join(BASE_DIR, 'tools', 'calibrate_confidence.py'),
                              'fit', '--input', args.out, '--out', out_pkl],
                            capture_output=True, text=True, timeout=120,
                            encoding='utf-8', errors='replace')
        print(r.stdout[-500:])
        if r.returncode == 0:
            print(f'\n[OK] calibrator: {out_pkl}')
            # copy to default calibrator path
            default_cal = os.path.join(BASE_DIR, 'data', 'calibrator_v15.pkl')
            import shutil
            shutil.copy(out_pkl, default_cal)
            print(f'[OK] copied to default: {default_cal}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
