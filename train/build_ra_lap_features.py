"""Build prev_race lap features from JV-Link RA records.

Extends data/v20/lap_features_v15cache.parquet by filling 2015-2019 gaps
using JV-Link RA record lap times (ra_laps_20150101_20191231.tsv).

Input:
  data/v20/ra_laps_20150101_20191231.tsv
  data/v20/lap_features_v15cache.parquet   (existing, netkeiba 2021-2025)
  data/_v15_train_df_cache.pkl

Output:
  data/v20/lap_features_v15cache.parquet   (updated, RA fills NaN gaps)

Usage:
  python train/build_ra_lap_features.py
"""
from __future__ import annotations
import sys, os, time, pickle
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RA_TSV    = os.path.join(BASE, 'data', 'v20', 'ra_laps_20150101_20191231.tsv')
CACHE_PKL = os.path.join(BASE, 'data', '_v15_train_df_cache.pkl')
LAP_PARQ  = os.path.join(BASE, 'data', 'v20', 'lap_features_v15cache.parquet')

os.makedirs(os.path.join(BASE, 'data', 'v20'), exist_ok=True)


def ra_to_jrdb_internal(race_date: str, course_code: str, kai: str, nichi: str, race_num: str) -> str:
    """Convert RA record fields to JRDB-internal 8-char race_id_str."""
    yy = str(race_date)[2:4]
    course = str(course_code).zfill(2)
    try:
        kai_int   = int(kai)
        nichi_int = int(nichi)
    except (ValueError, TypeError):
        return ''
    race  = str(race_num).zfill(2)
    kai_c   = str(kai_int) if kai_int <= 9 else format(kai_int, 'X')[-1]
    nichi_c = format(nichi_int, 'X')[-1] if nichi_int <= 15 else 'F'
    return f"{course}{yy}{kai_c}{nichi_c}{race}"


def main():
    # ===== 1. Load RA lap TSV =====
    print("Loading RA lap TSV ...")
    if not os.path.exists(RA_TSV):
        print(f"[ERROR] {RA_TSV} not found.")
        print("  Run: C:\\Windows\\SysWOW64\\WindowsPowerShell\\v1.0\\powershell.exe"
              " -File tools\\jv_ra_lap_full_fetch.ps1")
        sys.exit(1)

    t0 = time.time()
    ra = pd.read_csv(RA_TSV, sep='\t', encoding='utf-8', dtype=str, on_bad_lines='skip')
    print(f"  rows: {len(ra):,}, elapsed: {time.time()-t0:.1f}s")

    # Parse lap_times → first3f / last3f (seconds)
    def compute_laps(lap_str: str):
        if not isinstance(lap_str, str) or not lap_str:
            return np.nan, np.nan
        try:
            laps = [int(x) for x in lap_str.split(',')]
        except ValueError:
            return np.nan, np.nan
        if len(laps) < 3:
            return np.nan, np.nan
        return sum(laps[:3]) / 10.0, sum(laps[-3:]) / 10.0

    pairs = [compute_laps(s) for s in ra.get('lap_times', pd.Series(dtype=str))]
    ra['first3f'] = [p[0] for p in pairs]
    ra['last3f']  = [p[1] for p in pairs]
    ra = ra.dropna(subset=['first3f', 'last3f'])
    ra['first3f'] = ra['first3f'].astype(float)
    ra['last3f']  = ra['last3f'].astype(float)

    # Build JRDB race_id lookup
    ra['jrdb_id'] = ra.apply(
        lambda r: ra_to_jrdb_internal(
            r.get('race_date', ''), r.get('course_code', ''),
            r.get('kai', ''),       r.get('nichi', ''),
            r.get('race_num', '')), axis=1)
    ra = ra[ra['jrdb_id'] != '']
    ra_lookup = ra.drop_duplicates('jrdb_id').set_index('jrdb_id')[['first3f', 'last3f']].copy()
    print(f"  RA lookup races: {len(ra_lookup):,}")
    if len(ra_lookup) > 0:
        print(f"  first3f range: {ra_lookup['first3f'].min():.1f} – {ra_lookup['first3f'].max():.1f}")
        print(f"  last3f  range: {ra_lookup['last3f'].min():.1f} – {ra_lookup['last3f'].max():.1f}")

    # ===== 2. Load V15 cache =====
    print("\nLoading V15 cache ...")
    t0 = time.time()
    with open(CACHE_PKL, 'rb') as f:
        d = pickle.load(f)
    df = d['df'].copy()
    print(f"  shape={df.shape}, elapsed={time.time()-t0:.0f}s")

    df['_orig_idx']  = df.index
    df['race_id_str'] = df['race_id_str'].astype(str)
    df['horse_id']    = df['horse_id'].astype(str)

    # date_num for chronological sort
    if 'date_num' not in df.columns:
        df['_y_tmp'] = df['year'].apply(lambda y: 2000 + int(y) if int(y) <= 30 else 1900 + int(y))
        df['date_num'] = df['_y_tmp'] * 10000
        df.drop(columns=['_y_tmp'], inplace=True)

    df_sorted = df.sort_values(['horse_id', 'date_num']).copy()

    # prev_race_id_str per horse (shift within sorted group)
    df_sorted['prev_race_id_str_computed'] = (
        df_sorted.groupby('horse_id')['race_id_str'].shift(1)
    )
    n_with_prev = df_sorted['prev_race_id_str_computed'].notna().sum()
    print(f"  prev race available: {n_with_prev:,} / {len(df_sorted):,}")

    # ===== 3. Vectorized join with RA lookup =====
    print("\nJoining RA laps ...")
    t0 = time.time()

    pid_series = df_sorted['prev_race_id_str_computed']
    first3f_arr = pid_series.map(lambda p: ra_lookup.at[p, 'first3f'] if (pd.notna(p) and p in ra_lookup.index) else np.nan).values
    last3f_arr  = pid_series.map(lambda p: ra_lookup.at[p, 'last3f']  if (pd.notna(p) and p in ra_lookup.index) else np.nan).values

    df_sorted['ra_first3f']   = first3f_arr
    df_sorted['ra_last3f']    = last3f_arr
    df_sorted['ra_pace_diff'] = first3f_arr - last3f_arr

    fill_ra = (~np.isnan(first3f_arr)).mean()
    print(f"  RA fill rate: {fill_ra:.1%} ({(~np.isnan(first3f_arr)).sum():,}/{len(df_sorted):,})")
    print(f"  elapsed: {time.time()-t0:.1f}s")

    # ===== 4. Load existing lap parquet + merge =====
    result_cols = ['prev_race_first3f_real', 'prev_race_last3f_real', 'prev_race_pace_diff_real']
    if os.path.exists(LAP_PARQ):
        print("\nLoading existing lap parquet ...")
        existing = pd.read_parquet(LAP_PARQ)
        pre_fill = existing['prev_race_first3f_real'].notna().mean()
        print(f"  existing fill before: {pre_fill:.1%}")
    else:
        print("\nNo existing lap parquet — creating fresh.")
        existing = pd.DataFrame(index=df.index, columns=result_cols, dtype=np.float64)

    # Build RA result parquet indexed by V15 cache index
    ra_result = df_sorted[['_orig_idx', 'ra_first3f', 'ra_last3f', 'ra_pace_diff']].set_index('_orig_idx').sort_index()
    ra_result.columns = result_cols

    # Merge: keep existing non-NaN, fill NaN from RA
    merged = existing.copy()
    for col in result_cols:
        missing_mask = merged[col].isna()
        fill_vals    = ra_result[col].reindex(merged.index)
        merged.loc[missing_mask, col] = fill_vals[missing_mask]

    post_fill = merged['prev_race_first3f_real'].notna().mean()
    added     = post_fill - pre_fill if os.path.exists(LAP_PARQ) else post_fill
    print(f"  merged fill after : {post_fill:.1%}  (+{added:.1%} from RA)")

    # ===== 5. Coverage by year =====
    _y = df['year'].apply(lambda y: 2000 + int(y) if int(y) <= 30 else 1900 + int(y))
    merged['_y'] = _y
    print("\nCoverage by year (after merge):")
    for yr in sorted(merged['_y'].dropna().unique()):
        sub = merged[merged['_y'] == int(yr)]
        filled = sub['prev_race_first3f_real'].notna().sum()
        total  = len(sub)
        print(f"  {int(yr)}: {filled}/{total} ({filled/total*100:.0f}%)")
    merged.drop(columns=['_y'], inplace=True)

    # ===== 6. Save =====
    merged.to_parquet(LAP_PARQ)
    size_mb = os.path.getsize(LAP_PARQ) / 1e6
    print(f"\n[SAVED] {LAP_PARQ} ({size_mb:.1f} MB)")
    print(f"  total non-null: {merged['prev_race_first3f_real'].notna().sum():,}")
    print("\nNext: python train/train_v20_base.py  (retrain V20 with extended lap features)")


if __name__ == '__main__':
    main()
