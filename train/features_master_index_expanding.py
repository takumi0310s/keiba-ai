"""netkeiba_master_index expanding 化.

master_index.csv は finish_order + 5 indices を含む post-race data。
expanding window で 馬の **過去 indices 平均** を計算 (current race 除外、 LEAK-free)。

抽出 features (5 個、 各 horse の 過去 indices 累積 平均):
- master_idx_prev_avg_time
- master_idx_prev_avg_master
- master_idx_prev_avg_start
- master_idx_prev_avg_chase
- master_idx_prev_avg_agari

merge: race_id + umaban → jra_races_full で horse_id 取得 → 馬 ごと 時系列順 expanding

V15 .pkl.gz / predict_core / app.py 完全不変。

usage:
    python train/features_master_index_expanding.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / 'data'

INDEX_COLS = ['time_index', 'master_index', 'start_index', 'chase_index', 'agari_index']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(DATA_DIR / 'features_master_index_expanding.csv'))
    args = ap.parse_args()

    print('=== master_index expanding 化 (LEAK-free) ===')

    # 1. master_index 読込
    mi_path = DATA_DIR / 'netkeiba_master_index.csv'
    mi = pd.read_csv(mi_path, low_memory=False)
    mi['race_id'] = mi['race_id'].astype(str)
    mi['umaban'] = pd.to_numeric(mi['umaban'], errors='coerce').fillna(0).astype(int)
    print(f'  master_index: {len(mi):,} rows')

    # 2. jra_races_full と merge で horse_id 取得 (race_id + umaban で)
    rf_path = DATA_DIR / 'jra_races_full.csv'
    print(f'  loading jra_races_full ...')
    rf = pd.read_csv(rf_path, encoding='utf-8-sig', low_memory=False,
                     usecols=['year', 'month', 'day', 'race_num', 'race_id',
                              'horse_id', 'umaban', 'course', 'kai', 'nichi'],
                     dtype={'race_id': str})
    print(f'  jra_races_full: {len(rf):,} rows')

    # netkeiba race_id (12桁) を 直接 build from jra_races_full
    COURSE_CODE_MAP = {
        '札幌': '01', '函館': '02', '福島': '03', '新潟': '04', '東京': '05',
        '中山': '06', '中京': '07', '京都': '08', '阪神': '09', '小倉': '10',
    }
    rf['_course_code'] = rf['course'].map(COURSE_CODE_MAP).fillna('00')
    rf['nk_race_id'] = (
        '20' + rf['year'].astype(int).astype(str).str.zfill(2)
        + rf['_course_code']
        + rf['kai'].astype(int).astype(str).str.zfill(2)
        + rf['nichi'].astype(int).astype(str).str.zfill(2)
        + rf['race_num'].astype(int).astype(str).str.zfill(2)
    )
    rf_with_nk = rf.copy()
    print(f'  nk_race_id built: {len(rf_with_nk):,}')

    # 3. master_index 行 と rf を merge (race_id=nk_race_id + umaban → horse_id)
    rf_with_nk['umaban'] = pd.to_numeric(rf_with_nk['umaban'], errors='coerce').fillna(0).astype(int)
    rf_with_nk['horse_id'] = pd.to_numeric(rf_with_nk['horse_id'], errors='coerce').fillna(0).astype('Int64').astype(str)
    merged = mi.merge(rf_with_nk[['nk_race_id', 'umaban', 'horse_id', 'year', 'month', 'day', 'race_num']],
                      left_on=['race_id', 'umaban'],
                      right_on=['nk_race_id', 'umaban'],
                      how='left')
    matched = merged['horse_id'].notna().sum()
    print(f'  master_index → horse_id mapped: {matched:,}/{len(merged):,}')

    # 4. drop unmapped
    merged = merged.dropna(subset=['horse_id', 'year']).copy()
    merged['year'] = merged['year'].astype(int)
    for c in INDEX_COLS:
        merged[c] = pd.to_numeric(merged[c], errors='coerce').fillna(0)

    # 5. sort by horse + time
    merged = merged.sort_values(['horse_id', 'year', 'month', 'day', 'race_num']).reset_index(drop=True)

    # 6. expanding mean (当該 race 除外 = shift(1) で 計算)
    print('  computing expanding means ...')
    grp = merged.groupby('horse_id')
    for c in INDEX_COLS:
        # shift(1) で 前 race までの 累積、 expanding mean
        merged[f'master_idx_prev_avg_{c.replace("_index", "")}'] = (
            grp[c].shift(1).expanding().mean().reset_index(level=0, drop=True)
        )

    # 7. fillna with overall mean
    out_cols = ['race_id', 'umaban', 'horse_id'] + [
        f'master_idx_prev_avg_{c.replace("_index", "")}' for c in INDEX_COLS
    ]
    out = merged[out_cols].copy()
    for c in out.columns[3:]:
        out[c] = out[c].fillna(out[c].mean())

    print(f'\n[output] {len(out):,} rows × {len(out.columns)} cols')
    out.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f'saved: {args.out}')

    print('\nsample:')
    print(out.head(3).to_string())
    print(f'\nstats:')
    for c in out.columns[3:]:
        print(f'  {c}: mean={out[c].mean():.1f}, std={out[c].std():.1f}')


if __name__ == '__main__':
    main()
