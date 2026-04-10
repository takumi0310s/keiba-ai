#!/usr/bin/env python
"""JRDB KYI CSVを生データから再構築

新フィールド（入厩何走目/入厩年月日/入厩何日前/放牧先）を追加した
KYI_FIELDSでlzhファイルを再パースし、jrdb_kyi.csvを再生成する。

Usage:
    python tools/rebuild_jrdb_kyi.py
"""

import os
import sys
import time
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'jrdb_raw', 'kyi')

from scrape_jrdb import (
    extract_lzh, parse_fixed_length, KYI_FIELDS,
    jrdb_to_jra_race_id, jrdb_to_netkeiba_race_id,
)


def rebuild():
    start = time.time()
    print("=" * 60)
    print("  JRDB KYI CSV 再構築")
    print("=" * 60)

    if not os.path.isdir(RAW_DIR):
        print(f"ERROR: Raw data directory not found: {RAW_DIR}")
        sys.exit(1)

    lzh_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.lzh')])
    print(f"  LZH files: {len(lzh_files)}")

    all_dfs = []
    errors = 0

    for i, fname in enumerate(lzh_files):
        fpath = os.path.join(RAW_DIR, fname)
        try:
            with open(fpath, 'rb') as f:
                lzh_bytes = f.read()

            extracted = extract_lzh(lzh_bytes)
            for inner_name, data_bytes in extracted.items():
                if inner_name.upper().startswith('KYI'):
                    df = parse_fixed_length(data_bytes, KYI_FIELDS)
                    df['jra_race_id'] = df.apply(jrdb_to_jra_race_id, axis=1)
                    df['nk_race_id'] = df.apply(jrdb_to_netkeiba_race_id, axis=1)
                    if '馬名' in df.columns:
                        df['馬名'] = df['馬名'].str.strip()
                    if '放牧先' in df.columns:
                        df['放牧先'] = df['放牧先'].str.strip()
                    if '入厩年月日' in df.columns:
                        df['入厩年月日'] = df['入厩年月日'].str.strip()
                    all_dfs.append(df)

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(lzh_files)}] parsed {fname}")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR: {fname}: {e}")

    if not all_dfs:
        print("ERROR: No data parsed")
        sys.exit(1)

    print(f"\n  Concatenating {len(all_dfs)} DataFrames...")
    result = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate
    key_cols = ['jra_race_id', '馬番']
    if all(c in result.columns for c in key_cols):
        before = len(result)
        result = result.drop_duplicates(subset=key_cols, keep='last')
        print(f"  Deduplicated: {before} → {len(result)}")

    # Save
    output_path = os.path.join(DATA_DIR, 'jrdb_kyi.csv')
    result.to_csv(output_path, index=False, encoding='utf-8-sig')

    elapsed = time.time() - start
    print(f"\n  Saved: {output_path}")
    print(f"  Rows: {len(result)}")
    print(f"  Columns: {len(result.columns)}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.1f}s")

    # Verify new columns
    new_cols = ['入厩何走目', '入厩年月日', '入厩何日前', '放牧先']
    for col in new_cols:
        if col in result.columns:
            non_empty = result[col].dropna()
            if col in ['入厩何走目', '入厩何日前']:
                non_empty = non_empty[non_empty != 0]
            else:
                non_empty = non_empty[non_empty != '']
            coverage = len(non_empty) / len(result) * 100
            print(f"  {col}: {len(non_empty)} non-empty ({coverage:.1f}%)")
        else:
            print(f"  {col}: NOT IN OUTPUT!")


if __name__ == '__main__':
    rebuild()
