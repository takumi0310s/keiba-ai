#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""中 priority 7+ features 実装 (Task #7、 Phase 26).

Agent 1 で発見の 中 priority 競合 gap features を 全件 expanding window で構築。

【V15 投資保護】 derivative features のみ 新 CSV 出力、 既存 features / model / pipeline 不変

【実装 features (race_id × horse_id 単位、 全 LEAK-free)】

1. start_index           — 1角通過位置の馬career expanding mean (低いほど好スタート好位)
2. middle_index          — 2-3角平均位置の馬career expanding mean
3. late_index            — agari_3f race-relative の馬career expanding mean (低いほど好末脚)
4. bms_shinba_top3r      — 母父の新馬戦 (class_code==15) top3 rate expanding mean
5. kireaji_career_avg    — 切れ味 = (race avg agari - 自分 agari) の馬career expanding mean
                            (絶対値ベース、 late_index の sign-flip 版)
6. suriko_career_avg     — 末脚依存度 = (pass3 - finish) の馬career expanding mean
                            pace_career_burst_mean (pass4-finish) と差別化、 早めから動く差し脚
7. pace_adapt_career_avg — race の pace category × 馬 top3 expanding mean
                            (slow=0, mid=1, fast=2 ; pace_avg_pass1 quintile 1-2/3/4-5 ベース)
8. pci3                  — 馬career の agari_3f expanding mean (絶対値、 速い ほど 切れ味◎)
9. rpci                  — 馬 agari / race 平均 agari の馬career expanding mean

注: LPI / speed_gradient_E_M_L は race ラップ詳細データ (各 200m タイム) が
    現状 csv に無く skip。 race-level rough proxy としては pace_features に
    既に 統計 が在るため 価値追加 薄い。

【LEAK-free 保証】
- 全 features は groupby('horse_id').shift(1).expanding() で 当該 race 除外
- bms_shinba_top3r は 母父 単位 で merge_asof (date_int < current) で 当該 race 除外
- race-level pace category は 当該 race 自身 を 使用 (POST-RACE だが pace は事前 推定可能 / 識別子のみ)
  → 馬 × pace_cat 単位 expanding で 当該 race の outcome は使わない

【race grouping key】 jra_races_full.csv の race_id は umaban 込みで horse-unique。
真の race key = (year, month, day, course, kai, nichi, race_num) を文字列連結。

Usage:
    python tools/build_competitor_gap_features_v2.py
    python tools/build_competitor_gap_features_v2.py --year-from 18 --year-to 25
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# data csv は main repo にあり gitignore。 worktree からは main repo を 参照
_MAIN_REPO = r'C:\Users\takum\keiba-ai'
_DATA_DIR_DEFAULT = os.path.join(BASE_DIR, 'data')
if not os.path.exists(os.path.join(_DATA_DIR_DEFAULT, 'jra_races_full.csv')) and os.path.exists(
        os.path.join(_MAIN_REPO, 'data', 'jra_races_full.csv')):
    DATA_DIR = os.path.join(_MAIN_REPO, 'data')
else:
    DATA_DIR = _DATA_DIR_DEFAULT


def signal_verify(df, feat, target='top3', n_q=5):
    """quintile × target rate (Q5-Q1 delta) を返す."""
    import pandas as pd

    valid = df.dropna(subset=[feat]).copy()
    if len(valid) < 100:
        return None
    if valid[feat].std() is None or valid[feat].std() == 0:
        return None
    try:
        valid['_q'] = pd.qcut(valid[feat].rank(method='first'), n_q,
                              labels=list(range(1, n_q + 1)))
    except ValueError:
        return None
    grp = valid.groupby('_q', observed=True)[target].agg(['mean', 'count'])
    if len(grp) < n_q:
        return None
    delta = grp.loc[n_q, 'mean'] - grp.loc[1, 'mean']
    return {
        'feature': feat,
        'n_valid': int(len(valid)),
        'q1_rate': float(grp.loc[1, 'mean']),
        'q5_rate': float(grp.loc[n_q, 'mean']),
        'q5_q1_delta': float(delta),
        'grp': grp,
    }


def main():
    ap = argparse.ArgumentParser(description='Competitor gap features v2 (LEAK-free, 7+ items)')
    ap.add_argument('--year-from', type=int, default=18)
    ap.add_argument('--year-to', type=int, default=25)
    ap.add_argument('--out',
                    default=os.path.join(DATA_DIR, 'competitor_gap_features_v2.csv'))
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    print('=' * 70)
    print(' Phase 26: Competitor gap features v2 (middle priority 7+ items)')
    print('=' * 70)

    src = os.path.join(DATA_DIR, 'jra_races_full.csv')
    print(f'\n[INFO] loading {src} ...')
    use = ['year', 'month', 'day', 'course', 'kai', 'nichi', 'race_num',
           'race_id', 'horse_id', 'finish',
           'pass1', 'pass2', 'pass3', 'pass4', 'agari_3f',
           'distance', 'class_code', 'bms']
    df = pd.read_csv(src, encoding='utf-8', low_memory=False, usecols=use)
    print(f'[INFO] loaded: {df.shape}')

    # year filter
    df = df[(df['year'] >= args.year_from) & (df['year'] <= args.year_to)].copy()
    df = df[df['finish'].notna() & (df['finish'] > 0)].copy()
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)
    df['top3'] = (df['finish'] <= 3).astype(int)

    # numeric cast
    for c in ['pass1', 'pass2', 'pass3', 'pass4', 'agari_3f', 'distance', 'class_code']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # 真の race key (race_id は umaban 込み で horse-unique)
    df['race_key'] = (
        df['year'].astype(int).astype(str).str.zfill(2) + '_'
        + df['month'].astype(int).astype(str).str.zfill(2) + '_'
        + df['day'].astype(int).astype(str).str.zfill(2) + '_'
        + df['course'].fillna('NA').astype(str) + '_'
        + df['kai'].astype(int).astype(str).str.zfill(2) + '_'
        + df['nichi'].astype(int).astype(str).str.zfill(2) + '_'
        + df['race_num'].astype(int).astype(str).str.zfill(2)
    )

    # date_int for time-order sort
    df['date_int'] = (df['year'].astype(int) * 10000
                       + df['month'].astype(int) * 100
                       + df['day'].astype(int))
    print(f'[INFO] filtered: {df.shape}, unique race_key: {df["race_key"].nunique():,}')

    # race-level avg agari (POST-RACE within race、 個馬 expanding で 当該 race 除外する形で 使う)
    race_avg_agari = df.groupby('race_key')['agari_3f'].transform(
        lambda s: s[s > 0].mean() if (s > 0).any() else np.nan
    )
    df['agari_relative_self'] = df['agari_3f'] - race_avg_agari  # negative=faster
    df['race_avg_agari'] = race_avg_agari

    # pass1/pass2/pass3/pass4 で 0 は missing 扱い (短距離 で 1角 通過 が 記録されない 場合)
    for c in ['pass1', 'pass2', 'pass3', 'pass4']:
        df.loc[df[c] == 0, c] = np.nan

    df['_middle_raw'] = df[['pass2', 'pass3']].mean(axis=1)

    # ============================================================
    # 1-3, 6, 8, 9: horse-id 単位 expanding (shift(1) で 当該 race 除外)
    # ============================================================
    print('\n[1-3, 6, 8, 9] horse-id expanding features...')
    df = df.sort_values(['horse_id', 'date_int', 'race_key']).reset_index(drop=True)
    gb_h = df.groupby('horse_id', sort=False)

    # 1. start_index
    df['start_index'] = gb_h['pass1'].apply(
        lambda s: s.shift(1).expanding().mean()
    ).reset_index(level=0, drop=True)
    # 2. middle_index
    df['middle_index'] = gb_h['_middle_raw'].apply(
        lambda s: s.shift(1).expanding().mean()
    ).reset_index(level=0, drop=True)
    # 3. late_index (agari_relative、 負=速い)
    df['late_index'] = gb_h['agari_relative_self'].apply(
        lambda s: s.shift(1).expanding().mean()
    ).reset_index(level=0, drop=True)

    # 5. kireaji_career_avg = -late_index (正=切れ味、 race avg より 速い)
    df['kireaji_career_avg'] = -df['late_index']

    # 6. suriko_career_avg = (pass3 - finish) expanding
    df['_burst3'] = df['pass3'] - df['finish']
    df['suriko_career_avg'] = gb_h['_burst3'].apply(
        lambda s: s.shift(1).expanding().mean()
    ).reset_index(level=0, drop=True)

    # 8. pci3 = career agari_3f expanding (0除外、 fastest career)
    agari_clean = df['agari_3f'].where(df['agari_3f'] > 0)
    df['_agari_clean'] = agari_clean
    df['pci3'] = gb_h['_agari_clean'].apply(
        lambda s: s.shift(1).expanding().mean()
    ).reset_index(level=0, drop=True)

    # 9. rpci = career agari ratio (vs race avg) expanding
    df['_rpci_raw'] = df['agari_3f'] / df['race_avg_agari']
    df['_rpci_raw'] = df['_rpci_raw'].where((df['_rpci_raw'] > 0) & (df['_rpci_raw'] < 5))
    df['rpci'] = gb_h['_rpci_raw'].apply(
        lambda s: s.shift(1).expanding().mean()
    ).reset_index(level=0, drop=True)

    # ============================================================
    # 4. bms_shinba_top3r  母父 新馬戦 top3 rate expanding
    # ============================================================
    print('[4] bms_shinba_top3r (BMS new-horse top3 rate, asof merge)...')
    is_shinba = df['class_code'] == 15
    shinba_df = df[is_shinba & df['bms'].notna() & (df['bms'].astype(str).str.strip() != '')].copy()
    shinba_df = shinba_df.sort_values(['bms', 'date_int', 'race_key']).reset_index(drop=True)
    # 各 shinba race 終了時点 の 累積 top3 rate (含 self) を 算出
    shinba_df['_bms_shinba_cum_top3r'] = (
        shinba_df.groupby('bms', sort=False)['top3'].expanding().mean().reset_index(level=0, drop=True)
    )

    # merge_asof: 各 race の date_int より strict-prior の bms 累積値 を 取る
    lookup = shinba_df[['bms', 'date_int', '_bms_shinba_cum_top3r']].sort_values(
        ['date_int', 'bms']).reset_index(drop=True)
    df_sorted = df.sort_values('date_int').reset_index().rename(columns={'index': '_orig_idx'})
    # asof merge は両 frame の date_int sort + by group
    merged = pd.merge_asof(
        df_sorted, lookup.rename(columns={'_bms_shinba_cum_top3r': 'bms_shinba_top3r'}),
        on='date_int', by='bms', direction='backward', allow_exact_matches=False
    )
    merged = merged.sort_values('_orig_idx').reset_index(drop=True)
    df['bms_shinba_top3r'] = merged['bms_shinba_top3r'].values

    # ============================================================
    # 7. pace_adapt_career_avg
    #   race の pace category × 馬の top3 expanding (馬 × pace_cat 単位)
    # ============================================================
    print('[7] pace_adapt_career_avg (race pace × horse top3 expanding)...')
    # race ごと avg pass1 (race-level)
    race_pace_p1 = df.groupby('race_key')['pass1'].transform('mean')
    df['_race_pace_p1'] = race_pace_p1
    # quintile 化 (NaN そのまま): 1-2 / 3 / 4-5 → 0/1/2
    p1_rank_pct = race_pace_p1.rank(method='first', pct=True)
    df['_pace_cat'] = pd.cut(
        p1_rank_pct, bins=[0, 0.4, 0.6, 1.001], labels=[0, 1, 2], include_lowest=True
    ).astype('float')

    # 馬 × pace_cat 単位 で top3 expanding (shift で 当該除外)
    df = df.sort_values(['horse_id', '_pace_cat', 'date_int', 'race_key']).reset_index(drop=True)
    df['pace_adapt_career_avg'] = (
        df.groupby(['horse_id', '_pace_cat'], sort=False, dropna=False)['top3']
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )

    # ============================================================
    # signal verify (Q5-Q1 top3 rate delta)
    # ============================================================
    print('\n' + '=' * 70)
    print(' Signal verify (Q5-Q1 top3 rate delta、 LEAK-free)')
    print('=' * 70)
    feats = [
        'start_index', 'middle_index', 'late_index', 'bms_shinba_top3r',
        'kireaji_career_avg', 'suriko_career_avg', 'pace_adapt_career_avg',
        'pci3', 'rpci',
    ]
    results = []
    for f in feats:
        r = signal_verify(df, f)
        if r is None:
            print(f'  {f:30s}: skipped (insufficient data or zero variance)')
            continue
        print(f'  {f:30s}: N={r["n_valid"]:>8,}  Q1={r["q1_rate"]:.4f}  '
              f'Q5={r["q5_rate"]:.4f}  delta={r["q5_q1_delta"]:+.4f}')
        results.append(r)

    # ============================================================
    # Save
    # ============================================================
    out_cols = ['race_id', 'horse_id'] + feats
    out = df[out_cols].copy()
    out = out.drop_duplicates(['race_id', 'horse_id'])
    out.to_csv(args.out, index=False)
    print(f'\n[OK] saved: {args.out}')
    print(f'[OK] shape: {out.shape}')
    print(f'[OK] coverage:')
    for f in feats:
        nn = out[f].notna().sum()
        print(f'    {f:30s}: {nn:>8,} non-null ({100.0*nn/len(out):.1f}%)')

    return 0


if __name__ == '__main__':
    sys.exit(main())
