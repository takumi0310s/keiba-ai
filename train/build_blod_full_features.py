"""Build full BLOD features from JV-Link HN/SK records.

Input:
  data/v20/blod_hn.tsv  -- HN records (繁殖馬: blood_num, sex, sire_blood_num, dam_blood_num)
  data/v20/blod_sk.tsv  -- SK records (産駒: sire_blood_num → horse_id)

Features built (all expanding window, leak-free):
  sire_turf_wr_exp     : sire offspring turf win rate (cumsum / races, before current race)
  sire_dirt_wr_exp     : sire offspring dirt win rate
  sire_sprint_wr_exp   : sire offspring sprint (<=1400m) win rate
  sire_middle_wr_exp   : sire offspring middle (1500-2000m) win rate
  sire_long_wr_exp     : sire offspring long (>=2100m) win rate
  bms_turf_wr_exp      : broodmare sire turf win rate (expanding)
  bms_top3r_exp        : broodmare sire top3 rate (expanding)

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
HN_TSV    = os.path.join(BASE, 'data', 'v20', 'blod_hn.tsv')
SK_TSV    = os.path.join(BASE, 'data', 'v20', 'blod_sk.tsv')
CACHE_PKL = os.path.join(BASE, 'data', '_v15_train_df_cache.pkl')
OUT_DIR   = os.path.join(BASE, 'data', 'v20')
OUT_FILE  = os.path.join(OUT_DIR, 'blod_full_features_v15cache.parquet')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # ===== 1. Load HN (血統マスタ) =====
    print("Loading HN records ...")
    if not os.path.exists(HN_TSV):
        print(f"[ERROR] {HN_TSV} not found.")
        print("  Run: C:\\Windows\\SysWOW64\\WindowsPowerShell\\v1.0\\powershell.exe -File tools\\jv_blod_full_fetch.ps1")
        sys.exit(1)
    hn = pd.read_csv(HN_TSV, sep='\t', encoding='utf-8', dtype=str).fillna('')
    print(f"  HN: {len(hn):,} rows, columns={list(hn.columns)}")

    # Build blood_num → (sex, sire_blood_num, dam_blood_num) lookup
    hn_lookup = hn.set_index('blood_num')[['sex', 'sire_blood_num', 'dam_blood_num']].to_dict('index')

    # ===== 2. Load SK (産駒登録) =====
    print("Loading SK records ...")
    if not os.path.exists(SK_TSV):
        print(f"[ERROR] {SK_TSV} not found.")
        sys.exit(1)
    sk = pd.read_csv(SK_TSV, sep='\t', encoding='utf-8', dtype=str).fillna('')
    sk = sk[sk['offspring_horse_id'] != ''].copy()
    sk['horse_id'] = sk['offspring_horse_id'].str.strip().str.zfill(10)
    print(f"  SK: {len(sk):,} rows (with horse_id)")

    # ===== 3. Build sire/bms mapping: horse_id → sire_blood_num, bms_blood_num =====
    print("Building sire/bms mapping ...")
    sk_map = sk.set_index('horse_id')['sire_blood_num'].to_dict()

    def get_bms(horse_bn: str) -> str:
        """Get broodmare sire blood_num from horse's dam's sire."""
        info = hn_lookup.get(horse_bn, {})
        dam_bn = info.get('dam_blood_num', '')
        if not dam_bn:
            return ''
        dam_info = hn_lookup.get(dam_bn, {})
        return dam_info.get('sire_blood_num', '')

    # ===== 4. Load V15 cache =====
    print("\nLoading V15 cache ...")
    t0 = time.time()
    with open(CACHE_PKL, 'rb') as f:
        d = pickle.load(f)
    df = d['df'].copy()
    df['_orig_idx'] = df.index
    print(f"  shape={df.shape}, elapsed={time.time()-t0:.0f}s")

    # horse_id in V15: 8-char, SK horse_id: 10-char (leading zeros)
    df['horse_id_10'] = df['horse_id'].astype(str).str.zfill(10)

    # Map each horse to sire/bms blood_num
    df['sire_bn']   = df['horse_id_10'].map(sk_map).fillna('')
    df['bms_bn']    = df['sire_bn'].apply(lambda bn: get_bms(bn) if bn else '')
    fill_sire = (df['sire_bn'] != '').mean()
    fill_bms  = (df['bms_bn'] != '').mean()
    print(f"  sire fill: {fill_sire:.1%}, bms fill: {fill_bms:.1%}")

    # ===== 5. Surface/distance columns from V15 =====
    df['_y'] = df['year'].apply(lambda y: 2000 + int(y) if int(y) <= 30 else 1900 + int(y))
    df['date_num'] = df['_y'] * 10000 + df.get('month', pd.Series(0, index=df.index)).fillna(0).astype(int) * 100

    # Use race_id_str date component if available
    if 'race_id_str' in df.columns:
        df['date_num'] = df['race_id_str'].astype(str).str[:8].apply(
            lambda s: int(s) if s.isdigit() else 0)

    surface_col  = 'surface_enc' if 'surface_enc' in df.columns else None
    distance_col = 'distance'    if 'distance'    in df.columns else None
    finish_col   = 'finish'      if 'finish'      in df.columns else None

    # ===== 6. Build expanding features per sire / bms =====
    print("\nBuilding expanding features ...")

    feats_list = []

    def make_expanding(group_col, label):
        """Generic expanding window feature builder for a blood group column."""
        sub = df[df[group_col] != ''].copy()
        sub = sub.sort_values(['date_num', '_orig_idx'])

        results = {}

        for bn, g in sub.groupby(group_col, sort=False):
            g = g.sort_values(['date_num', '_orig_idx'])
            y    = (g[finish_col] <= 3).astype(int).values if finish_col else None
            win  = (g[finish_col] == 1).astype(int).values if finish_col else None
            surf = g[surface_col].values if surface_col else None
            dist = g[distance_col].values if distance_col else None

            n = len(g)
            top3r_exp = np.zeros(n, dtype=np.float32)
            turf_wr_exp = np.zeros(n, dtype=np.float32)
            dirt_wr_exp = np.zeros(n, dtype=np.float32)
            sprint_wr_exp = np.zeros(n, dtype=np.float32)
            middle_wr_exp = np.zeros(n, dtype=np.float32)
            long_wr_exp   = np.zeros(n, dtype=np.float32)

            cum_top3 = 0; cum_races = 0
            cum_turf_win = 0; cum_turf_races = 0
            cum_dirt_win = 0; cum_dirt_races = 0
            cum_spr_win  = 0; cum_spr_races  = 0
            cum_mid_win  = 0; cum_mid_races  = 0
            cum_long_win = 0; cum_long_races = 0

            for i in range(n):
                # Expanding (cumsum BEFORE current race)
                if cum_races > 0:
                    top3r_exp[i] = cum_top3 / cum_races
                if cum_turf_races > 0:
                    turf_wr_exp[i] = cum_turf_win / cum_turf_races
                if cum_dirt_races > 0:
                    dirt_wr_exp[i] = cum_dirt_win / cum_dirt_races
                if cum_spr_races > 0:
                    sprint_wr_exp[i] = cum_spr_win / cum_spr_races
                if cum_mid_races > 0:
                    middle_wr_exp[i] = cum_mid_win / cum_mid_races
                if cum_long_races > 0:
                    long_wr_exp[i] = cum_long_win / cum_long_races

                # Update counters with current race
                if y is not None:
                    cum_top3 += y[i]; cum_races += 1
                if surf is not None:
                    s = int(surf[i]) if not np.isnan(surf[i]) else -1
                    if s == 1:  # turf
                        cum_turf_win += (win[i] if win is not None else 0); cum_turf_races += 1
                    elif s == 2:  # dirt
                        cum_dirt_win += (win[i] if win is not None else 0); cum_dirt_races += 1
                if dist is not None:
                    d_m = int(dist[i]) if not np.isnan(dist[i]) else 0
                    w = win[i] if win is not None else 0
                    if d_m <= 1400:
                        cum_spr_win += w; cum_spr_races += 1
                    elif d_m <= 2000:
                        cum_mid_win += w; cum_mid_races += 1
                    elif d_m > 2000:
                        cum_long_win += w; cum_long_races += 1

            for j, idx in enumerate(g['_orig_idx'].values):
                results[idx] = {
                    f'{label}_top3r_exp':    float(top3r_exp[j]),
                    f'{label}_turf_wr_exp':  float(turf_wr_exp[j]),
                    f'{label}_dirt_wr_exp':  float(dirt_wr_exp[j]),
                    f'{label}_sprint_wr_exp':float(sprint_wr_exp[j]),
                    f'{label}_middle_wr_exp':float(middle_wr_exp[j]),
                    f'{label}_long_wr_exp':  float(long_wr_exp[j]),
                }
        return results

    print("  Building sire expanding features ...")
    sire_results = make_expanding('sire_bn', 'sire_blod')
    print(f"  Sire: {len(sire_results):,} rows")

    print("  Building bms expanding features ...")
    bms_results  = make_expanding('bms_bn', 'bms_blod')
    print(f"  BMS:  {len(bms_results):,} rows")

    # ===== 7. Assemble output parquet =====
    print("\nAssembling output ...")
    all_feat_cols = [
        'sire_blod_top3r_exp', 'sire_blod_turf_wr_exp', 'sire_blod_dirt_wr_exp',
        'sire_blod_sprint_wr_exp', 'sire_blod_middle_wr_exp', 'sire_blod_long_wr_exp',
        'bms_blod_top3r_exp', 'bms_blod_turf_wr_exp', 'bms_blod_dirt_wr_exp',
        'bms_blod_sprint_wr_exp', 'bms_blod_middle_wr_exp', 'bms_blod_long_wr_exp',
    ]
    result_df = pd.DataFrame(index=df.index, columns=all_feat_cols, dtype=np.float32)

    for idx, vals in sire_results.items():
        for k, v in vals.items():
            if k in result_df.columns:
                result_df.loc[idx, k] = v
    for idx, vals in bms_results.items():
        for k, v in vals.items():
            if k in result_df.columns:
                result_df.loc[idx, k] = v

    for col in all_feat_cols:
        fill = result_df[col].notna().mean()
        print(f"  {col}: fill={fill:.1%}")

    result_df.to_parquet(OUT_FILE)
    size_mb = os.path.getsize(OUT_FILE) / 1e6
    print(f"\n[SAVED] {OUT_FILE} ({size_mb:.1f} MB)")
    print(f"\nNext: python train/train_v20_base.py  (retrain V20 with BLOD full features)")


if __name__ == '__main__':
    main()
