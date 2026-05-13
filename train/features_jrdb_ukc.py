"""JRDB UKC (馬基本データ) から expanding feature 抽出.

V20+/V22 学習 で merge。 V15 production / predict_core 不変。

UKC merge key: blood_num (jra_races_full.horse_id と 同一)

抽出 features:
- owner_code → owner_code_top3r_jrdb (expanding alpha=30, JRA races_full の owner と独立 source)
- birthplace → birthplace_top3r (expanding alpha=50、 産地ごと top3 率)
- father_code → father_code_top3r (種牡馬 code expanding wr)
- bms_code → bms_code_top3r (母父 code expanding wr)
- father_birth_year, mother_birth_year, bms_birth_year (族系 世代 マーカー)

usage:
    python train/features_jrdb_ukc.py [--start-year 20]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent


def _expanding_bayes_wr(df: pd.DataFrame, group_col: str, label_col: str,
                       alpha: int = 30, base_rate: float = 0.25) -> pd.Series:
    """expanding window で Bayesian smoothing 勝率 (当該 row 除外)."""
    df_sorted = df.sort_values([group_col, 'year', 'month', 'day', 'race_num'])
    grp = df_sorted.groupby(group_col)
    cumsum = grp[label_col].cumsum()
    cumcnt = grp.cumcount() + 1
    prev_sum = cumsum - df_sorted[label_col]
    prev_cnt = cumcnt - 1
    wr = (prev_sum + alpha * base_rate) / (prev_cnt + alpha)
    wr.index = df_sorted.index
    return wr.reindex(df.index).clip(0, 1)


def build_ukc_features(races_csv: str, ukc_csv: str, start_year: int = 20) -> pd.DataFrame:
    """jra_races_full + UKC merge → owner_code / birthplace 等 expanding wr."""
    print(f"[ukc_features] reading {races_csv}...")
    df = pd.read_csv(races_csv, encoding='utf-8-sig', low_memory=False,
                     usecols=['year', 'month', 'day', 'race_num', 'race_id',
                              'horse_id', 'umaban', 'finish'])
    df = df[df['year'] >= start_year].copy()
    print(f"  races: {len(df):,} rows (year >= {start_year})")

    print(f"[ukc_features] reading {ukc_csv}...")
    ukc = pd.read_csv(ukc_csv, encoding='utf-8-sig', low_memory=False,
                      dtype={'blood_num': str, 'owner_code': str,
                             'father_code': str, 'bms_code': str})
    print(f"  UKC: {len(ukc):,} rows")

    # merge: horse_id == blood_num
    df['_hid'] = pd.to_numeric(df['horse_id'], errors='coerce').astype('Int64').astype(str)
    ukc['_bnum'] = ukc['blood_num'].astype(str).str.strip()
    # UKC は last data_date を 保つ
    ukc = ukc.sort_values('data_date').drop_duplicates('_bnum', keep='last')
    df = df.merge(
        ukc[['_bnum', 'owner_code', 'birthplace', 'father_code', 'bms_code',
             'father_birth_year', 'mother_birth_year', 'bms_birth_year']],
        left_on='_hid', right_on='_bnum', how='left',
    ).drop(columns=['_hid', '_bnum'])
    matched = df['owner_code'].notna().sum()
    print(f"  matched UKC: {matched:,}/{len(df):,} ({matched/len(df)*100:.1f}%)")

    # fill na with sentinel '0'
    for c in ('owner_code', 'birthplace', 'father_code', 'bms_code'):
        df[c] = df[c].fillna('0').astype(str)
    for c in ('father_birth_year', 'mother_birth_year', 'bms_birth_year'):
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(2000).astype(int)

    # labels
    df['_is_top3'] = (df['finish'] <= 3).astype(int)
    df['_is_win'] = (df['finish'] == 1).astype(int)

    # expanding wr
    print("  computing expanding wr ...")
    df['owner_code_top3r_jrdb'] = _expanding_bayes_wr(df, 'owner_code', '_is_top3', 30, 0.25)
    df['owner_code_wr_jrdb'] = _expanding_bayes_wr(df, 'owner_code', '_is_win', 30, 0.08)
    df['birthplace_top3r'] = _expanding_bayes_wr(df, 'birthplace', '_is_top3', 50, 0.25)
    df['birthplace_wr'] = _expanding_bayes_wr(df, 'birthplace', '_is_win', 50, 0.08)
    df['father_code_top3r'] = _expanding_bayes_wr(df, 'father_code', '_is_top3', 50, 0.25)
    df['bms_code_top3r'] = _expanding_bayes_wr(df, 'bms_code', '_is_top3', 50, 0.25)

    # birth year (族系 cohort)
    df['father_birth_year_ohash'] = (df['father_birth_year'] % 50).astype(int)
    df['bms_birth_year_ohash'] = (df['bms_birth_year'] % 50).astype(int)
    df['age_gap_father'] = (df['year'] + 2000 - df['father_birth_year']).clip(0, 50)
    df['age_gap_bms'] = (df['year'] + 2000 - df['bms_birth_year']).clip(0, 60)

    df = df.drop(columns=['_is_top3', '_is_win'])

    out_cols = ['race_id', 'horse_id', 'umaban',
                'owner_code_top3r_jrdb', 'owner_code_wr_jrdb',
                'birthplace_top3r', 'birthplace_wr',
                'father_code_top3r', 'bms_code_top3r',
                'father_birth_year_ohash', 'bms_birth_year_ohash',
                'age_gap_father', 'age_gap_bms']
    return df[out_cols].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--races', default=str(BASE / 'data' / 'jra_races_full.csv'))
    ap.add_argument('--ukc', default=str(BASE / 'data' / 'jrdb_ukc.csv'))
    ap.add_argument('--output', default=str(BASE / 'data' / 'features_jrdb_ukc.csv'))
    ap.add_argument('--start-year', type=int, default=20)
    args = ap.parse_args()

    out = build_ukc_features(args.races, args.ukc, start_year=args.start_year)
    print(f"[ukc_features] writing {args.output} ({len(out):,} rows × {len(out.columns)} cols)")
    out.to_csv(args.output, index=False, encoding='utf-8-sig')

    print("\nsample (head 3):")
    print(out.head(3).to_string())
    print(f"\nnull rates:")
    for c in out.columns[3:]:
        null = out[c].isnull().mean()
        if null > 0.001:
            print(f"  {c}: {null*100:.1f}% null")


if __name__ == '__main__':
    main()
