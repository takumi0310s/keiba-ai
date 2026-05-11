#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""horse_id mapper: TFJV 8-digit ⇔ netkeiba 10-digit.

verified rule (2026-05-12):
- V20 (TFJV) horse_id: 8-digit, first 2 digit = year (e.g., '19102173' = 2019, seq 102173)
- V21 (netkeiba) horse_id: 10-digit, first 4 digit = year (e.g., '2022106229' = 2022, seq 106229)
- conversion: TFJV→netkeiba = '20' + zfill(8) of TFJV horse_id

Usage:
    from tools.horse_id_mapper import tfjv_to_netkeiba, netkeiba_to_tfjv
    tfjv_to_netkeiba('19102173')  # → '2019102173'
    netkeiba_to_tfjv('2019102173')  # → '19102173'

verified: 22/29 V21 paddock horse_id matched V20 base (rest = 2024 産 未 V20 収録).
"""
import pandas as pd


def tfjv_to_netkeiba(horse_id):
    """TFJV 8-digit → netkeiba 10-digit.

    accepts: int, str, or pd.Series. NaN safe.
    """
    if isinstance(horse_id, pd.Series):
        ids = pd.to_numeric(horse_id, errors='coerce').astype('Int64').astype(str)
        ids = ids.where(ids != '<NA>', None)
        return ids.apply(lambda x: '20' + x.zfill(8) if x is not None else None)
    if pd.isna(horse_id):
        return None
    try:
        n = int(float(horse_id))
        return '20' + str(n).zfill(8)
    except (ValueError, TypeError):
        return None


def netkeiba_to_tfjv(horse_id):
    """netkeiba 10-digit → TFJV 8-digit (year prefix strip)."""
    if isinstance(horse_id, pd.Series):
        ids = horse_id.astype(str)
        return ids.apply(lambda x: x[2:] if (x and len(x) >= 10 and x.startswith('20')) else None)
    if pd.isna(horse_id):
        return None
    s = str(horse_id)
    if len(s) >= 10 and s.startswith('20'):
        return s[2:]
    return None


def verify():
    """Verification by paddock dirs."""
    import os
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v20 = pd.read_csv(os.path.join(BASE, 'data', 'v20_training_data_full.csv'),
                      usecols=['horse_id', 'year'], dtype={'horse_id': str})
    v20['horse_id_nk'] = tfjv_to_netkeiba(v20['horse_id'])

    v21 = pd.read_csv(os.path.join(BASE, 'data', 'v21_video_features_all.csv'))
    v21['horse_id_str'] = v21['horse_id'].astype(str)

    matched = set(v20['horse_id_nk'].dropna()) & set(v21['horse_id_str'])
    print(f'V20 unique horses: {v20["horse_id_nk"].nunique()}')
    print(f'V21 paddock horses: {v21["horse_id_str"].nunique()}')
    print(f'matched: {len(matched)}')
    print(f'sample: {list(matched)[:5]}')


if __name__ == '__main__':
    verify()
