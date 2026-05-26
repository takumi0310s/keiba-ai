"""Build full BLOD features from V15 cache sire_enc/bms_enc (expanding window).

JV-Link BLOD HN/SK data uses blood_num format incompatible with JRDB horse_id.
This script instead builds features directly from V15 cache race history using
sire_enc and bms_enc (already in cache), with surface and distance segmentation.

Features built (all expanding window, leak-free, cumsum-before-current-race):
  sire_blod_top3r_exp     : sire offspring top3 rate (all races)
  sire_blod_turf_wr_exp   : sire offspring turf win rate
  sire_blod_dirt_wr_exp   : sire offspring dirt win rate
  sire_blod_sprint_wr_exp : sire offspring sprint (<=1400m) win rate
  sire_blod_middle_wr_exp : sire offspring middle (1500-2000m) win rate
  sire_blod_long_wr_exp   : sire offspring long (>2000m) win rate
  bms_blod_{same 6}       : same 6 features for broodmare sire (bms_enc)

Output: data/v20/blod_full_features_v15cache.parquet
  Index: original V15 cache index

Usage:
  python train/build_blod_full_features.py
"""
from __future__ import annotations
import sys, os, time, pickle
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PKL = os.path.join(BASE, 'data', '_v15_train_df_cache.pkl')
OUT_DIR   = os.path.join(BASE, 'data', 'v20')
OUT_FILE  = os.path.join(OUT_DIR, 'blod_full_features_v15cache.parquet')
os.makedirs(OUT_DIR, exist_ok=True)


def build_expanding_features(df: pd.DataFrame, group_col: str, label: str) -> pd.DataFrame:
    """Build expanding surface/distance win-rate features for group_col.

    Returns DataFrame indexed by _orig_idx with 6 feature columns.
    """
    feat_cols = [
        f'{label}_top3r_exp',
        f'{label}_turf_wr_exp',
        f'{label}_dirt_wr_exp',
        f'{label}_sprint_wr_exp',
        f'{label}_middle_wr_exp',
        f'{label}_long_wr_exp',
    ]
    result = pd.DataFrame(np.nan, index=df['_orig_idx'], columns=feat_cols, dtype=np.float32)

    # Only rows with known group_col (enc != 100 = unknown sire)
    sub = df[df[group_col].notna() & (df[group_col] != 100)].copy()
    sub = sub.sort_values(['date_num', '_orig_idx'])

    # Precompute binary flags
    sub['_is_top3']   = (sub['finish'] <= 3).astype(np.int8)
    sub['_is_win']    = (sub['finish'] == 1).astype(np.int8)
    sub['_is_turf']   = (sub['surface_enc'] == 1).astype(np.int8)
    sub['_is_dirt']   = (sub['surface_enc'] == 0).astype(np.int8)  # V15: 0=dirt, 1=turf
    sub['_is_sprint'] = (sub['distance'] <= 1400).astype(np.int8)
    sub['_is_middle'] = ((sub['distance'] >= 1500) & (sub['distance'] <= 2000)).astype(np.int8)
    sub['_is_long']   = (sub['distance'] > 2000).astype(np.int8)

    # Build daily aggregate per (group_enc, date_num) for each category
    agg_cols = {
        'top3': ('_is_top3', 'sum'),
        'n_all': ('_is_top3', 'count'),
        'turf_win': ('_is_win', lambda x: (x * sub.loc[x.index, '_is_turf']).sum()),
        'n_turf':   ('_is_turf', 'sum'),
        'dirt_win': ('_is_win', lambda x: (x * sub.loc[x.index, '_is_dirt']).sum()),
        'n_dirt':   ('_is_dirt', 'sum'),
        'spr_win':  ('_is_win', lambda x: (x * sub.loc[x.index, '_is_sprint']).sum()),
        'n_spr':    ('_is_sprint', 'sum'),
        'mid_win':  ('_is_win', lambda x: (x * sub.loc[x.index, '_is_middle']).sum()),
        'n_mid':    ('_is_middle', 'sum'),
        'lng_win':  ('_is_win', lambda x: (x * sub.loc[x.index, '_is_long']).sum()),
        'n_lng':    ('_is_long', 'sum'),
    }

    # Simpler: build separate daily tables and merge
    daily_base = sub.groupby([group_col, 'date_num']).agg(
        top3=('_is_top3', 'sum'),
        n_all=('_is_top3', 'count'),
    ).reset_index().sort_values([group_col, 'date_num'])

    # expanding shift-by-1 for top3
    daily_base['cum_top3'] = daily_base.groupby(group_col)['top3'].transform(
        lambda x: x.cumsum().shift(1).fillna(0))
    daily_base['cum_n'] = daily_base.groupby(group_col)['n_all'].transform(
        lambda x: x.cumsum().shift(1).fillna(0))
    daily_base['top3r_exp'] = (daily_base['cum_top3'] / daily_base['cum_n'].clip(lower=1)
                               ).where(daily_base['cum_n'] > 0)

    def daily_wr(flag_col, win_col):
        """Build daily win-rate expanding feature for a subset."""
        sub2 = sub[[group_col, 'date_num', flag_col, win_col]].copy()
        sub2['_counted_win'] = sub2[flag_col] * sub2[win_col]
        d = sub2.groupby([group_col, 'date_num']).agg(
            wins=('_counted_win', 'sum'),
            total=(flag_col, 'sum'),
        ).reset_index().sort_values([group_col, 'date_num'])
        d['cum_wins'] = d.groupby(group_col)['wins'].transform(
            lambda x: x.cumsum().shift(1).fillna(0))
        d['cum_n'] = d.groupby(group_col)['total'].transform(
            lambda x: x.cumsum().shift(1).fillna(0))
        d['wr_exp'] = (d['cum_wins'] / d['cum_n'].clip(lower=1)).where(d['cum_n'] > 0)
        return d[[group_col, 'date_num', 'wr_exp']]

    daily_turf   = daily_wr('_is_turf',   '_is_win')
    daily_dirt   = daily_wr('_is_dirt',   '_is_win')
    daily_sprint = daily_wr('_is_sprint', '_is_win')
    daily_middle = daily_wr('_is_middle', '_is_win')
    daily_long   = daily_wr('_is_long',   '_is_win')

    # merge_asof to join each feature back to main df
    df_key = sub[['_orig_idx', group_col, 'date_num']].sort_values('date_num').copy()

    def asof_join(daily, feat_name):
        d_sorted = daily.sort_values('date_num')
        merged = pd.merge_asof(
            df_key, d_sorted, on='date_num', by=group_col, direction='backward')
        merged_idx = merged.set_index('_orig_idx')['wr_exp']
        result[feat_name] = merged_idx.reindex(result.index).astype(np.float32)

    # top3r via merge_asof
    d_base_sorted = daily_base[[group_col, 'date_num', 'top3r_exp']].sort_values('date_num')
    merged_base = pd.merge_asof(df_key, d_base_sorted, on='date_num', by=group_col, direction='backward')
    result[f'{label}_top3r_exp'] = merged_base.set_index('_orig_idx')['top3r_exp'].reindex(result.index).astype(np.float32)

    asof_join(daily_turf,   f'{label}_turf_wr_exp')
    asof_join(daily_dirt,   f'{label}_dirt_wr_exp')
    asof_join(daily_sprint, f'{label}_sprint_wr_exp')
    asof_join(daily_middle, f'{label}_middle_wr_exp')
    asof_join(daily_long,   f'{label}_long_wr_exp')

    return result


def main():
    # ===== Load V15 cache =====
    print("Loading V15 cache ...")
    t0 = time.time()
    with open(CACHE_PKL, 'rb') as f:
        d = pickle.load(f)
    df = d['df'].copy()
    df['_orig_idx'] = df.index
    df['_y'] = df['year'].apply(lambda y: 2000 + int(y) if int(y) <= 30 else 1900 + int(y))
    print(f"  shape={df.shape}, elapsed={time.time()-t0:.0f}s")

    # Derive date_num (YYYYMMDD integer) for sorting
    if 'date_num' not in df.columns:
        df['date_num'] = df['_y'] * 10000
    if 'surface_enc' not in df.columns or 'distance' not in df.columns:
        print("[ERROR] surface_enc or distance not in V15 cache"); sys.exit(1)
    df['surface_enc'] = df['surface_enc'].fillna(0)
    df['distance']    = df['distance'].fillna(0)

    # ===== Build sire features =====
    print("\nBuilding sire expanding features (sire_enc) ...")
    t0 = time.time()
    sire_feats = build_expanding_features(df, 'sire_enc', 'sire_blod')
    print(f"  elapsed: {time.time()-t0:.0f}s")
    for col in sire_feats.columns:
        fill = sire_feats[col].notna().mean()
        print(f"  {col}: fill={fill:.1%}")

    # ===== Build BMS features =====
    print("\nBuilding BMS expanding features (bms_enc) ...")
    t0 = time.time()
    bms_feats = build_expanding_features(df, 'bms_enc', 'bms_blod')
    print(f"  elapsed: {time.time()-t0:.0f}s")
    for col in bms_feats.columns:
        fill = bms_feats[col].notna().mean()
        print(f"  {col}: fill={fill:.1%}")

    # ===== Assemble & save =====
    result_df = pd.concat([sire_feats, bms_feats], axis=1)
    result_df.index.name = None

    result_df.to_parquet(OUT_FILE)
    size_mb = os.path.getsize(OUT_FILE) / 1e6
    print(f"\n[SAVED] {OUT_FILE} ({size_mb:.1f} MB)")
    print(f"\nNext: python train/train_v20_base.py  (retrain V20 with BLOD full features)")


if __name__ == '__main__':
    main()
