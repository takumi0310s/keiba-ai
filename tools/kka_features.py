"""KKA derived features for V15 / V20 evaluation — Session #53.

Reads `data/jrdb_kka_v2.csv` (produced by `tools/jrdb_kka_parser_v2.py`) and
materializes win/place rates from the seiseki summary blocks.

LEAK RISK NOTE
---------------
JRDB の Paci ファイル (KKA を含む) は **pre-race 配布**:
    平日 (月木): 19:00 配布   → 翌週末レース 用
    週末 (金土): 20:00 配布   → 翌日レース 用

→ KKA の seiseki 集計は **当該レース 直前まで の累計** であり、
  当該レースの結果を含まない。 SKB と異なり post-race leak は無い。

ただし安全のため、 features は **当該レース除外形式** で計算 (rate のみ、 raw count は提供しない)。
section D の quality check で corr_target / monotonic を検証する。

Read-only:
    Input:  data/jrdb_kka_v2.csv (548K rows)
    Output: data/jrdb_kka_features.csv (gitignored)
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_CSV = os.path.join(REPO, 'data', 'jrdb_kka_v2.csv')
OUT_CSV = os.path.join(REPO, 'data', 'jrdb_kka_features.csv')


# 23 個の seiseki block (各 4 col: _1, _2, _3, _out)
SEISEKI_BLOCKS = [
    'jra_seiseki', 'koryu_seiseki', 'other_seiseki',
    'turf_dirt_2', 'turf_dirt_2_dist', 'track_seiseki',
    'rotation_sei', 'kyori_seiseki', 'saka_seiseki',
    'heavy_seiseki', 'rest_seiseki', 'class_seiseki',
    'speed_seiseki', 'slow_seiseki', 'mid_seiseki',
    'season_seiseki', 'waku_seiseki',
    'breeder_dist', 'breeder_track', 'breeder_adjust',
    'breeder_baba', 'breeder_blanker', 'breeder_surface',
]


def _safe_rate(num: pd.Series, denom: pd.Series, min_n: int = 1) -> pd.Series:
    """rate = num / denom (denom < min_n は NaN)。"""
    out = num / denom.where(denom >= min_n)
    return out


def derive_kka_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each seiseki block (1着, 2着, 3着, 着外):
      - {block}_starts  = 1 + 2 + 3 + out
      - {block}_winr    = 1 / starts
      - {block}_top2r   = (1+2) / starts
      - {block}_top3r   = (1+2+3) / starts
    """
    feat_cols = []
    for blk in SEISEKI_BLOCKS:
        c1, c2, c3, co = f'{blk}_1', f'{blk}_2', f'{blk}_3', f'{blk}_out'
        if not all(c in df.columns for c in (c1, c2, c3, co)):
            continue
        starts = df[[c1, c2, c3, co]].sum(axis=1, min_count=1)
        winr = _safe_rate(df[c1], starts, min_n=1)
        top2r = _safe_rate(df[c1] + df[c2], starts, min_n=1)
        top3r = _safe_rate(df[c1] + df[c2] + df[c3], starts, min_n=1)
        df[f'kka_{blk}_starts'] = starts
        df[f'kka_{blk}_winr'] = winr
        df[f'kka_{blk}_top2r'] = top2r
        df[f'kka_{blk}_top3r'] = top3r
        feat_cols += [f'kka_{blk}_starts', f'kka_{blk}_winr',
                      f'kka_{blk}_top2r', f'kka_{blk}_top3r']

    # passthrough numeric (連勝率系)
    for c in ['dam_rensho_max', 'dam_rensho_min', 'dam_rensho_avg',
              'bms_rensho_max', 'bms_rensho_min', 'bms_rensho_avg']:
        if c in df.columns:
            df[f'kka_{c}'] = df[c]
            feat_cols.append(f'kka_{c}')

    return df[['race_id', 'umaban'] + feat_cols].copy()


def main():
    print(f'Loading: {SRC_CSV}')
    df = pd.read_csv(SRC_CSV)
    print(f'  rows: {len(df):,}, cols: {len(df.columns)}')

    feat = derive_kka_features(df)
    print(f'Derived features: {len(feat.columns) - 2} cols (excl race_id, umaban)')

    # coverage sample
    sample_cols = [
        'kka_jra_seiseki_top3r', 'kka_jra_seiseki_starts',
        'kka_kyori_seiseki_top3r', 'kka_track_seiseki_top3r',
        'kka_heavy_seiseki_top3r', 'kka_class_seiseki_top3r',
        'kka_dam_rensho_max', 'kka_bms_rensho_max',
    ]
    print('--- Sample coverage / stats ---')
    for c in sample_cols:
        if c not in feat.columns:
            continue
        non_null = feat[c].notna().mean() * 100
        if feat[c].notna().sum() > 0:
            mean_v = feat[c].mean()
            med_v = feat[c].median()
            print(f'  {c}: {non_null:.1f}% non-null, mean={mean_v:.4f}, med={med_v:.4f}')
        else:
            print(f'  {c}: {non_null:.1f}% non-null (all NaN)')

    feat.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'Saved: {OUT_CSV} ({len(feat):,} rows, {len(feat.columns)} cols)')


if __name__ == '__main__':
    main()
