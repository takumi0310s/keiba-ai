"""Build horse × course × dist_cat expanding top3 rate feature.

Feature:
  horse_course_top3r_exp: Bayesian-smoothed top3 rate of each horse at
                          a specific (course_enc, dist_cat) combination,
                          computed as expanding cumsum BEFORE the current race.
  horse_course_n_exp:     Number of prior races at this course/dist (raw count).

Output: data/v20/horse_course_features_v15cache.parquet

Usage:
  python train/build_horse_course_features.py
"""
from __future__ import annotations
import sys, os, time, pickle
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PKL = os.path.join(BASE, 'data', '_v15_train_df_cache.pkl')
OUT_DIR   = os.path.join(BASE, 'data', 'v20')
OUT_FILE  = os.path.join(OUT_DIR, 'horse_course_features_v15cache.parquet')
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA = 3.0    # Bayesian smoothing strength
PRIOR = 0.25   # prior top3 rate (~25% average)


def main():
    print("Loading V15 cache ...")
    t0 = time.time()
    with open(CACHE_PKL, 'rb') as f:
        d = pickle.load(f)
    df = d['df'].copy()
    df['_orig_idx'] = df.index
    print(f"  shape={df.shape}, elapsed={time.time()-t0:.0f}s")

    # ===== Derive helper columns =====
    df['_y'] = df['year'].apply(lambda y: 2000 + int(y) if int(y) <= 30 else 1900 + int(y))
    if 'date_num' not in df.columns:
        df['date_num'] = df['_y'] * 10000

    # dist_cat: prefer existing column, else compute
    if 'dist_cat' in df.columns:
        df['_dist_cat'] = df['dist_cat'].fillna(2).astype(int)
    else:
        df['_dist_cat'] = pd.cut(
            df['distance'].fillna(1600),
            bins=[0, 1200, 1400, 1600, 2000, 99999],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)

    df['_is_top3'] = (df['finish'] <= 3).astype(np.int8)

    # ===== Sort for expanding window =====
    df_s = df.sort_values(['horse_id', 'date_num', '_orig_idx']).copy()

    print("Computing horse × course_enc × dist_cat expanding top3r ...")
    t0 = time.time()

    # Group key
    grp_keys = ['horse_id', 'course_enc', '_dist_cat']
    g = df_s.groupby(grp_keys, sort=False)

    # cumcount() = 0 for first occurrence = number of PRIOR races in this group
    df_s['_cum_n'] = g.cumcount()

    # cumsum of is_top3 shifted by 1 = top3 count BEFORE current race
    df_s['_cum_top3'] = g['_is_top3'].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )

    # Bayesian smoothing: set NaN when _cum_n == 0 (no prior info)
    has_prior = df_s['_cum_n'] > 0
    df_s['horse_course_top3r_exp'] = np.where(
        has_prior,
        (df_s['_cum_top3'] + ALPHA * PRIOR) / (df_s['_cum_n'] + ALPHA),
        np.nan
    ).astype(np.float32)

    df_s['horse_course_n_exp'] = df_s['_cum_n'].astype(np.float32)

    elapsed = time.time() - t0
    print(f"  elapsed: {elapsed:.1f}s")

    # ===== Stats =====
    fill = df_s['horse_course_top3r_exp'].notna().mean()
    print(f"  fill rate (has prior at this course/dist): {fill:.1%}")

    has_data = df_s[df_s['horse_course_top3r_exp'].notna()]
    print(f"  mean top3r (smoothed): {has_data['horse_course_top3r_exp'].mean():.3f}")
    print(f"  std:                   {has_data['horse_course_top3r_exp'].std():.3f}")
    print(f"  median _cum_n (when >0): {df_s.loc[has_prior, '_cum_n'].median():.1f}")

    # Coverage by year
    print("\nCoverage by year:")
    for yr in sorted(df_s['_y'].unique()):
        sub = df_s[df_s['_y'] == yr]
        f = sub['horse_course_top3r_exp'].notna().mean()
        print(f"  {yr}: {f:.1%}")

    # ===== Save =====
    result = (
        df_s.set_index('_orig_idx')[['horse_course_top3r_exp', 'horse_course_n_exp']]
        .sort_index()
    )
    result.to_parquet(OUT_FILE)
    size_mb = os.path.getsize(OUT_FILE) / 1e6
    print(f"\n[SAVED] {OUT_FILE} ({size_mb:.1f} MB)")
    print(f"  non-null: {result['horse_course_top3r_exp'].notna().sum():,}")
    print("\nNext: python train/train_v21_clean.py")


if __name__ == '__main__':
    main()
