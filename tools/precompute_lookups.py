"""
jra_races_full.csvからdaily_predict.py用の特徴量ルックアップテーブルを事前生成する。

Usage:
    python tools/precompute_lookups.py

Output:
    data/feature_lookups.pkl

含まれるルックアップ:
    - sire_surface_wr: {(father, surface_enc): win_rate}
    - sire_dist_wr: {(father, dist_cat): win_rate}
    - bms_surface_wr: {(bms, surface_enc): win_rate}
    - trainer_top3: {trainer: top3_rate}
    - jockey_surface_wr: {(jockey, surface_enc): win_rate}
    - frame_course_dist_wr: {(bracket_pos, course_enc, dist_cat): win_rate}
    - horse_stats: {horse_id: {career_races, career_wr, career_top3r, ...}}
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "jra_races_full.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "feature_lookups.pkl")

ALPHA_SIRE = 50
ALPHA_HORSE = 5
ALPHA_JOCKEY = 10
ALPHA_FRAME = 100
ALPHA_TRAINER = 30


def bayesian_rate(wins, total, alpha, prior=0.3):
    """ベイジアンスムージング"""
    return (wins + alpha * prior) / (total + alpha)


def main():
    print("Loading jra_races_full.csv...")
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig', low_memory=False)
    print(f"  {len(df):,} rows loaded")

    # Date column
    df['race_date'] = pd.to_datetime(
        df['year'].astype(str).str.zfill(2).apply(lambda x: ('20' if int(x) < 50 else '19') + x)
        + df['month'].astype(str).str.zfill(2)
        + df['day'].astype(str).str.zfill(2),
        format='%Y%m%d', errors='coerce'
    )

    # Surface encoding
    df['surface_enc'] = df['surface'].map({'芝': 0, 'ダ': 1}).fillna(1).astype(int)

    # Distance category
    df['dist_cat'] = pd.cut(df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999],
                            labels=[0, 1, 2, 3, 4]).astype(float).fillna(2).astype(int)

    # Course encoding
    COURSE_MAP = {'札幌': 0, '函館': 1, '福島': 2, '新潟': 3, '東京': 4,
                  '中山': 5, '中京': 6, '京都': 7, '阪神': 8, '小倉': 9}
    df['course_enc'] = df['course'].map(COURSE_MAP).fillna(4).astype(int)

    # Bracket position
    df['bracket_pos'] = pd.cut(df['umaban'], bins=[0, 3, 6, 18],
                               labels=[0, 1, 2]).astype(float).fillna(1).astype(int)

    # Win/place flags
    df['is_win'] = (df['finish'] == 1).astype(int)
    df['is_top3'] = (df['finish'] <= 3).astype(int)

    lookups = {}

    # === 1. Sire surface win rate ===
    print("Computing sire_surface_wr...")
    sire_surf = df.groupby(['father', 'surface_enc']).agg(
        wins=('is_win', 'sum'), total=('is_win', 'count')
    ).reset_index()
    sire_surf['wr'] = sire_surf.apply(
        lambda r: bayesian_rate(r['wins'], r['total'], ALPHA_SIRE, 0.1), axis=1
    )
    lookups['sire_surface_wr'] = {
        (row['father'], row['surface_enc']): row['wr']
        for _, row in sire_surf.iterrows()
    }
    print(f"  {len(lookups['sire_surface_wr']):,} entries")

    # === 2. Sire distance win rate ===
    print("Computing sire_dist_wr...")
    sire_dist = df.groupby(['father', 'dist_cat']).agg(
        wins=('is_win', 'sum'), total=('is_win', 'count')
    ).reset_index()
    sire_dist['wr'] = sire_dist.apply(
        lambda r: bayesian_rate(r['wins'], r['total'], ALPHA_SIRE, 0.1), axis=1
    )
    lookups['sire_dist_wr'] = {
        (row['father'], row['dist_cat']): row['wr']
        for _, row in sire_dist.iterrows()
    }
    print(f"  {len(lookups['sire_dist_wr']):,} entries")

    # === 3. BMS surface win rate ===
    print("Computing bms_surface_wr...")
    bms_surf = df.groupby(['bms', 'surface_enc']).agg(
        wins=('is_win', 'sum'), total=('is_win', 'count')
    ).reset_index()
    bms_surf['wr'] = bms_surf.apply(
        lambda r: bayesian_rate(r['wins'], r['total'], ALPHA_SIRE, 0.1), axis=1
    )
    lookups['bms_surface_wr'] = {
        (row['bms'], row['surface_enc']): row['wr']
        for _, row in bms_surf.iterrows()
    }
    print(f"  {len(lookups['bms_surface_wr']):,} entries")

    # === 4. Trainer top3 rate ===
    print("Computing trainer_top3...")
    trainer = df.groupby('trainer').agg(
        top3=('is_top3', 'sum'), total=('is_top3', 'count')
    ).reset_index()
    trainer['rate'] = trainer.apply(
        lambda r: bayesian_rate(r['top3'], r['total'], ALPHA_TRAINER, 0.25), axis=1
    )
    lookups['trainer_top3'] = {
        row['trainer']: row['rate'] for _, row in trainer.iterrows()
    }
    print(f"  {len(lookups['trainer_top3']):,} entries")

    # === 5. Jockey surface win rate ===
    print("Computing jockey_surface_wr...")
    jockey_surf = df.groupby(['jockey', 'surface_enc']).agg(
        wins=('is_win', 'sum'), total=('is_win', 'count')
    ).reset_index()
    jockey_surf['wr'] = jockey_surf.apply(
        lambda r: bayesian_rate(r['wins'], r['total'], ALPHA_JOCKEY, 0.05), axis=1
    )
    lookups['jockey_surface_wr'] = {
        (row['jockey'], row['surface_enc']): row['wr']
        for _, row in jockey_surf.iterrows()
    }
    print(f"  {len(lookups['jockey_surface_wr']):,} entries")

    # === 6. Frame × Course × Distance win rate ===
    print("Computing frame_course_dist_wr...")
    fcd = df.groupby(['bracket_pos', 'course_enc', 'dist_cat']).agg(
        wins=('is_win', 'sum'), total=('is_win', 'count')
    ).reset_index()
    fcd['wr'] = fcd.apply(
        lambda r: bayesian_rate(r['wins'], r['total'], ALPHA_FRAME, 0.1), axis=1
    )
    lookups['frame_course_dist_wr'] = {
        (row['bracket_pos'], row['course_enc'], row['dist_cat']): row['wr']
        for _, row in fcd.iterrows()
    }
    print(f"  {len(lookups['frame_course_dist_wr']):,} entries")

    # === 7. Horse career stats (expanding window, latest snapshot) ===
    print("Computing horse_stats...")
    df_sorted = df.sort_values('race_date')
    horse_groups = df_sorted.groupby('horse_id')

    horse_stats = {}
    for hid, grp in horse_groups:
        n = len(grp)
        wins = grp['is_win'].sum()
        top3 = grp['is_top3'].sum()
        career_wr = bayesian_rate(wins, n, ALPHA_HORSE, 0.1)
        career_top3r = bayesian_rate(top3, n, ALPHA_HORSE, 0.3)

        # Distance-specific top3 rate
        dist_top3 = {}
        for dc, dg in grp.groupby('dist_cat'):
            dt3 = dg['is_top3'].sum()
            dist_top3[int(dc)] = bayesian_rate(dt3, len(dg), ALPHA_HORSE, 0.3)

        # Surface-specific top3 rate
        surf_top3 = {}
        for se, sg in grp.groupby('surface_enc'):
            st3 = sg['is_top3'].sum()
            surf_top3[int(se)] = bayesian_rate(st3, len(sg), ALPHA_HORSE, 0.3)

        # Last race details for dist_change, prev_prize, prev_agari_relative
        last = grp.iloc[-1]
        last2 = grp.iloc[-2] if n >= 2 else last

        horse_stats[int(hid)] = {
            'career_races': n,
            'career_wr': career_wr,
            'career_top3r': career_top3r,
            'dist_top3': dist_top3,
            'surf_top3': surf_top3,
            'last_distance': int(last['distance']),
            'last_prize': float(last['prize']) if pd.notna(last['prize']) else 0,
            'last_agari': float(last['agari_3f']) if pd.notna(last.get('agari_3f')) else 35.5,
            'last_training_4f': float(last['training_4f']) if pd.notna(last.get('training_4f')) else 0,
        }

    lookups['horse_stats'] = horse_stats
    print(f"  {len(horse_stats):,} horses")

    # === 8. Race-level agari stats (for prev_agari_relative) ===
    print("Computing race_avg_agari...")
    race_agari = df.groupby('race_id')['agari_3f'].mean().to_dict()
    lookups['race_avg_agari'] = race_agari
    print(f"  {len(race_agari):,} races")

    # === 9. Training time global mean ===
    valid_training = df[df['training_4f'] > 0]['training_4f']
    lookups['training_mean'] = float(valid_training.mean()) if len(valid_training) > 0 else 52.0
    print(f"  Training mean: {lookups['training_mean']:.1f}s")

    # === Save ===
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(lookups, f)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\nSaved to {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print("Done!")


if __name__ == '__main__':
    main()
