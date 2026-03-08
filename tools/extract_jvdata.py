#!/usr/bin/env python
"""TARGET Frontier JV データ抽出スクリプト
C:\TFJV から中央データを抽出し、7つのCSVに変換する。

① data/jra_races_full.csv - レース成績全期間
② data/training_times.csv - 調教タイム（坂路・ウッドチップ）
③ data/odds_history.csv - オッズ履歴
④ data/jockey_history_full.csv - 騎手長期成績
⑤ data/trainer_history_full.csv - 調教師長期成績
⑥ data/blood_full.csv - 血統データ
⑦ data/horse_history_full.csv - 馬毎レース情報
"""
import pandas as pd
import numpy as np
import os
import sys
import csv
from collections import defaultdict

# Paths
TFJV_DIR = 'C:/TFJV'
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
KEIBA_DATA = os.path.join(TFJV_DIR, 'TXT', 'keiba_data.csv')
TARGET_ODDS = os.path.join(DATA_DIR, 'target_odds.csv')
SAKARO_CSV = os.path.join(TFJV_DIR, 'TXT', 'target_sakaro.csv')
WOOD_CSV = os.path.join(TFJV_DIR, 'TXT', 'target_wood.csv')

# Correct 52-column header mapping (confirmed from target_seiseki.csv header)
COLUMNS = [
    'year', 'month', 'day', 'kai', 'course', 'nichi', 'race_num',
    'race_name', 'class_code', 'surface', 'course_code', 'distance',
    'condition', 'horse_name', 'sex', 'age', 'jockey', 'weight_carry',
    'num_horses', 'umaban', 'finish', 'finish2', 'abnormal_code',
    'time_margin', 'popularity', 'run_time', 'run_time_x10', 'empty',
    'pass1', 'pass2', 'pass3', 'pass4', 'agari_3f', 'horse_weight',
    'trainer', 'location', 'prize', 'horse_id', 'jockey_id', 'trainer_id',
    'race_id', 'owner', 'breeder', 'father', 'mother', 'bms',
    'coat_color', 'birthday', 'tansho_odds', 'mark1', 'mark2', 'training_4f'
]


def load_race_csv(path, label=""):
    """Load a race result CSV (no header, 52 cols, cp932)."""
    print(f"  Loading {label}: {path}")
    df = pd.read_csv(path, encoding='cp932', header=None, low_memory=False,
                     dtype=str, on_bad_lines='skip')
    if df.shape[1] >= 52:
        df = df.iloc[:, :52]
    df.columns = COLUMNS[:df.shape[1]]
    print(f"    {len(df)} rows loaded")
    return df


def extract_races_full():
    """① レース成績全期間: Merge keiba_data.csv + target_odds.csv → jra_races_full.csv"""
    print("\n=== ① Extracting jra_races_full.csv ===")

    dfs = []

    # Load target_odds.csv (current training data, 2010-2025)
    if os.path.exists(TARGET_ODDS):
        df1 = load_race_csv(TARGET_ODDS, "target_odds.csv")
        dfs.append(df1)

    # Load keiba_data.csv (TARGET export, 2015-2025)
    if os.path.exists(KEIBA_DATA):
        df2 = load_race_csv(KEIBA_DATA, "keiba_data.csv")
        dfs.append(df2)

    if not dfs:
        print("  ERROR: No race data found!")
        return None

    # Merge and deduplicate
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(df)} rows")

    # Create dedup key: race_id + umaban
    df['dedup_key'] = df['race_id'].astype(str).str.strip() + '_' + df['umaban'].astype(str).str.strip()

    # Keep first occurrence (target_odds has more years, so it comes first)
    before = len(df)
    df = df.drop_duplicates(subset='dedup_key', keep='first')
    print(f"  After dedup: {len(df)} rows (removed {before - len(df)} duplicates)")

    df = df.drop(columns=['dedup_key'])

    # Sort by date
    df['year_num'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    df['month_num'] = pd.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
    df['day_num'] = pd.to_numeric(df['day'], errors='coerce').fillna(0).astype(int)
    df = df.sort_values(['year_num', 'month_num', 'day_num', 'race_num', 'umaban'])
    df = df.drop(columns=['year_num', 'month_num', 'day_num'])

    # Save with header
    out_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {out_path} ({len(df)} rows)")

    return df


def extract_training_times():
    """② 調教タイム（坂路・ウッドチップ）→ training_times.csv"""
    print("\n=== ② Extracting training_times.csv ===")

    dfs = []

    # 坂路
    if os.path.exists(SAKARO_CSV):
        print(f"  Loading sakaro: {SAKARO_CSV}")
        sakaro = pd.read_csv(SAKARO_CSV, encoding='cp932', dtype=str)
        print(f"    {len(sakaro)} rows, cols: {list(sakaro.columns)}")
        # Normalize
        sakaro_norm = pd.DataFrame({
            'training_type': 'sakaro',
            'location': sakaro.iloc[:, 0],       # 場所
            'date': sakaro.iloc[:, 1],            # 年月日
            'horse_name': sakaro.iloc[:, 4],      # 馬名
            'sex': sakaro.iloc[:, 6],             # 性別
            'age': sakaro.iloc[:, 7],             # 年齢
            'trainer': sakaro.iloc[:, 9],         # 調教師
            'time_4f': sakaro.iloc[:, 10],        # Time1 (=4F)
            'time_3f': sakaro.iloc[:, 11],        # Time2 (=3F)
            'time_2f': sakaro.iloc[:, 12],        # Time3 (=2F)
            'time_1f': sakaro.iloc[:, 13],        # Time4 (=1F)
            'lap4': sakaro.iloc[:, 14],
            'lap3': sakaro.iloc[:, 15],
            'lap2': sakaro.iloc[:, 16],
            'lap1': sakaro.iloc[:, 17],
            'horse_id': '',                       # 坂路にはhorse_idがない
        })
        dfs.append(sakaro_norm)

    # ウッドチップ
    if os.path.exists(WOOD_CSV):
        print(f"  Loading wood: {WOOD_CSV}")
        wood = pd.read_csv(WOOD_CSV, encoding='cp932', dtype=str)
        print(f"    {len(wood)} rows, cols: {list(wood.columns[:10])}")
        # Normalize
        wood_norm = pd.DataFrame({
            'training_type': 'wood',
            'location': wood.iloc[:, 0],          # 場所
            'date': wood.iloc[:, 3],              # 年月日
            'horse_name': wood.iloc[:, 6],        # 馬名
            'sex': wood.iloc[:, 8],               # 性別
            'age': wood.iloc[:, 9],               # 年齢
            'trainer': wood.iloc[:, 11],          # 調教師
            'time_4f': wood.iloc[:, 18],          # 4F
            'time_3f': wood.iloc[:, 19],          # 3F
            'time_2f': wood.iloc[:, 20],          # 2F
            'time_1f': wood.iloc[:, 21],          # 1F
            'lap4': wood.iloc[:, 27],
            'lap3': wood.iloc[:, 28],
            'lap2': wood.iloc[:, 29],
            'lap1': wood.iloc[:, 30],
            'horse_id': wood.iloc[:, 31],         # 血統登録番号
        })
        dfs.append(wood_norm)

    if not dfs:
        print("  ERROR: No training time data found!")
        return None

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(df)} rows")

    out_path = os.path.join(DATA_DIR, 'training_times.csv')
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {out_path} ({len(df)} rows)")

    return df


def extract_odds_history(race_df):
    """③ オッズ履歴 → odds_history.csv"""
    print("\n=== ③ Extracting odds_history.csv ===")

    if race_df is None:
        print("  ERROR: No race data!")
        return None

    # Extract odds from race data (tansho_odds column)
    odds_df = race_df[['race_id', 'horse_name', 'umaban', 'horse_id',
                        'tansho_odds', 'popularity', 'year', 'month', 'day',
                        'course', 'race_num', 'finish']].copy()

    odds_df['tansho_odds'] = pd.to_numeric(odds_df['tansho_odds'], errors='coerce')
    odds_df = odds_df[odds_df['tansho_odds'].notna() & (odds_df['tansho_odds'] > 0)]

    print(f"  Odds records: {len(odds_df)}")

    out_path = os.path.join(DATA_DIR, 'odds_history.csv')
    odds_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {out_path} ({len(odds_df)} rows)")

    return odds_df


def extract_jockey_history(race_df):
    """④ 騎手長期成績 → jockey_history_full.csv"""
    print("\n=== ④ Extracting jockey_history_full.csv ===")

    if race_df is None:
        return None

    df = race_df.copy()
    df['finish_n'] = pd.to_numeric(df['finish'], errors='coerce')
    df['prize_n'] = pd.to_numeric(df['prize'], errors='coerce').fillna(0)
    df = df[df['finish_n'].notna() & (df['finish_n'] >= 1)]

    df['is_win'] = (df['finish_n'] == 1).astype(int)
    df['is_top3'] = (df['finish_n'] <= 3).astype(int)

    # Overall stats
    stats = df.groupby(['jockey_id', 'jockey']).agg(
        total_races=('is_win', 'count'),
        wins=('is_win', 'sum'),
        top3=('is_top3', 'sum'),
        total_prize=('prize_n', 'sum'),
    ).reset_index()
    stats['win_rate'] = stats['wins'] / stats['total_races'].clip(1)
    stats['top3_rate'] = stats['top3'] / stats['total_races'].clip(1)

    # Surface-specific stats
    for surf_name, surf_val in [('turf', '芝'), ('dirt', 'ダ')]:
        mask = df['surface'].astype(str).str.contains(surf_val, na=False)
        surf_stats = df[mask].groupby('jockey_id').agg(
            **{f'{surf_name}_races': ('is_win', 'count'),
               f'{surf_name}_wins': ('is_win', 'sum'),
               f'{surf_name}_top3': ('is_top3', 'sum')}
        ).reset_index()
        surf_stats[f'{surf_name}_wr'] = surf_stats[f'{surf_name}_wins'] / surf_stats[f'{surf_name}_races'].clip(1)
        stats = stats.merge(surf_stats, on='jockey_id', how='left')

    # Distance category stats
    df['distance_n'] = pd.to_numeric(df['distance'], errors='coerce').fillna(1600)
    for dist_name, lo, hi in [('sprint', 0, 1400), ('mile', 1401, 1800),
                               ('middle', 1801, 2200), ('long', 2201, 9999)]:
        mask = (df['distance_n'] >= lo) & (df['distance_n'] <= hi)
        d_stats = df[mask].groupby('jockey_id').agg(
            **{f'{dist_name}_races': ('is_win', 'count'),
               f'{dist_name}_wins': ('is_win', 'sum')}
        ).reset_index()
        d_stats[f'{dist_name}_wr'] = d_stats[f'{dist_name}_wins'] / d_stats[f'{dist_name}_races'].clip(1)
        stats = stats.merge(d_stats, on='jockey_id', how='left')

    stats = stats.fillna(0)
    print(f"  Jockey records: {len(stats)}")

    out_path = os.path.join(DATA_DIR, 'jockey_history_full.csv')
    stats.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {out_path} ({len(stats)} rows)")

    return stats


def extract_trainer_history(race_df):
    """⑤ 調教師長期成績 → trainer_history_full.csv"""
    print("\n=== ⑤ Extracting trainer_history_full.csv ===")

    if race_df is None:
        return None

    df = race_df.copy()
    df['finish_n'] = pd.to_numeric(df['finish'], errors='coerce')
    df['prize_n'] = pd.to_numeric(df['prize'], errors='coerce').fillna(0)
    df = df[df['finish_n'].notna() & (df['finish_n'] >= 1)]

    df['is_win'] = (df['finish_n'] == 1).astype(int)
    df['is_top3'] = (df['finish_n'] <= 3).astype(int)

    # Overall stats
    stats = df.groupby(['trainer_id', 'trainer']).agg(
        total_races=('is_win', 'count'),
        wins=('is_win', 'sum'),
        top3=('is_top3', 'sum'),
        total_prize=('prize_n', 'sum'),
    ).reset_index()
    stats['win_rate'] = stats['wins'] / stats['total_races'].clip(1)
    stats['top3_rate'] = stats['top3'] / stats['total_races'].clip(1)

    # Surface-specific stats
    for surf_name, surf_val in [('turf', '芝'), ('dirt', 'ダ')]:
        mask = df['surface'].astype(str).str.contains(surf_val, na=False)
        surf_stats = df[mask].groupby('trainer_id').agg(
            **{f'{surf_name}_races': ('is_win', 'count'),
               f'{surf_name}_wins': ('is_win', 'sum'),
               f'{surf_name}_top3': ('is_top3', 'sum')}
        ).reset_index()
        surf_stats[f'{surf_name}_wr'] = surf_stats[f'{surf_name}_wins'] / surf_stats[f'{surf_name}_races'].clip(1)
        stats = stats.merge(surf_stats, on='trainer_id', how='left')

    stats = stats.fillna(0)
    print(f"  Trainer records: {len(stats)}")

    out_path = os.path.join(DATA_DIR, 'trainer_history_full.csv')
    stats.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {out_path} ({len(stats)} rows)")

    return stats


def extract_blood_data(race_df):
    """⑥ 血統データ → blood_full.csv"""
    print("\n=== ⑥ Extracting blood_full.csv ===")

    if race_df is None:
        return None

    df = race_df.copy()
    df['finish_n'] = pd.to_numeric(df['finish'], errors='coerce')
    df = df[df['finish_n'].notna() & (df['finish_n'] >= 1)]
    df['is_win'] = (df['finish_n'] == 1).astype(int)
    df['is_top3'] = (df['finish_n'] <= 3).astype(int)

    # Unique horse-blood records
    blood = df.drop_duplicates(subset='horse_id', keep='last')[
        ['horse_id', 'horse_name', 'father', 'mother', 'bms',
         'coat_color', 'birthday', 'sex']
    ].copy()

    # Sire performance stats (progeny aggregate)
    sire_stats = df.groupby('father').agg(
        sire_progeny_races=('is_win', 'count'),
        sire_progeny_wins=('is_win', 'sum'),
        sire_progeny_top3=('is_top3', 'sum'),
    ).reset_index()
    sire_stats['sire_wr'] = sire_stats['sire_progeny_wins'] / sire_stats['sire_progeny_races'].clip(1)
    sire_stats['sire_top3r'] = sire_stats['sire_progeny_top3'] / sire_stats['sire_progeny_races'].clip(1)

    # Sire surface performance
    df['distance_n'] = pd.to_numeric(df['distance'], errors='coerce').fillna(1600)
    for surf_name, surf_val in [('turf', '芝'), ('dirt', 'ダ')]:
        mask = df['surface'].astype(str).str.contains(surf_val, na=False)
        ss = df[mask].groupby('father').agg(
            **{f'sire_{surf_name}_races': ('is_win', 'count'),
               f'sire_{surf_name}_wins': ('is_win', 'sum')}
        ).reset_index()
        ss[f'sire_{surf_name}_wr'] = ss[f'sire_{surf_name}_wins'] / ss[f'sire_{surf_name}_races'].clip(1)
        sire_stats = sire_stats.merge(ss, on='father', how='left')

    # Sire distance performance
    for dist_name, lo, hi in [('sprint', 0, 1400), ('mile', 1401, 1800),
                               ('middle', 1801, 2200), ('long', 2201, 9999)]:
        mask = (df['distance_n'] >= lo) & (df['distance_n'] <= hi)
        ds = df[mask].groupby('father').agg(
            **{f'sire_{dist_name}_races': ('is_win', 'count'),
               f'sire_{dist_name}_wins': ('is_win', 'sum')}
        ).reset_index()
        ds[f'sire_{dist_name}_wr'] = ds[f'sire_{dist_name}_wins'] / ds[f'sire_{dist_name}_races'].clip(1)
        sire_stats = sire_stats.merge(ds, on='father', how='left')

    # BMS performance stats
    bms_stats = df.groupby('bms').agg(
        bms_progeny_races=('is_win', 'count'),
        bms_progeny_wins=('is_win', 'sum'),
        bms_progeny_top3=('is_top3', 'sum'),
    ).reset_index()
    bms_stats['bms_wr'] = bms_stats['bms_progeny_wins'] / bms_stats['bms_progeny_races'].clip(1)
    bms_stats['bms_top3r'] = bms_stats['bms_progeny_top3'] / bms_stats['bms_progeny_races'].clip(1)

    # Merge sire/bms stats into blood data
    blood = blood.merge(sire_stats, on='father', how='left')
    blood = blood.merge(bms_stats, on='bms', how='left')
    blood = blood.fillna(0)

    print(f"  Blood records: {len(blood)}")
    print(f"  Unique sires: {blood['father'].nunique()}")
    print(f"  Unique BMS: {blood['bms'].nunique()}")

    out_path = os.path.join(DATA_DIR, 'blood_full.csv')
    blood.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {out_path} ({len(blood)} rows)")

    return blood


def extract_horse_history(race_df):
    """⑦ 馬毎レース情報 → horse_history_full.csv"""
    print("\n=== ⑦ Extracting horse_history_full.csv ===")

    if race_df is None:
        return None

    df = race_df.copy()
    df['finish_n'] = pd.to_numeric(df['finish'], errors='coerce')
    df['prize_n'] = pd.to_numeric(df['prize'], errors='coerce').fillna(0)
    df['agari_n'] = pd.to_numeric(df['agari_3f'], errors='coerce')
    df['distance_n'] = pd.to_numeric(df['distance'], errors='coerce').fillna(1600)
    df = df[df['finish_n'].notna() & (df['finish_n'] >= 1)]

    df['is_win'] = (df['finish_n'] == 1).astype(int)
    df['is_top3'] = (df['finish_n'] <= 3).astype(int)

    # Per-horse stats
    stats = df.groupby(['horse_id', 'horse_name']).agg(
        total_races=('is_win', 'count'),
        wins=('is_win', 'sum'),
        top3=('is_top3', 'sum'),
        total_prize=('prize_n', 'sum'),
        avg_finish=('finish_n', 'mean'),
        best_finish=('finish_n', 'min'),
        avg_agari=('agari_n', 'mean'),
        best_agari=('agari_n', 'min'),
    ).reset_index()
    stats['win_rate'] = stats['wins'] / stats['total_races'].clip(1)
    stats['top3_rate'] = stats['top3'] / stats['total_races'].clip(1)

    # Surface preference
    for surf_name, surf_val in [('turf', '芝'), ('dirt', 'ダ')]:
        mask = df['surface'].astype(str).str.contains(surf_val, na=False)
        ss = df[mask].groupby('horse_id').agg(
            **{f'{surf_name}_races': ('is_win', 'count'),
               f'{surf_name}_wins': ('is_win', 'sum'),
               f'{surf_name}_top3': ('is_top3', 'sum')}
        ).reset_index()
        ss[f'{surf_name}_wr'] = ss[f'{surf_name}_wins'] / ss[f'{surf_name}_races'].clip(1)
        ss[f'{surf_name}_top3r'] = ss[f'{surf_name}_top3'] / ss[f'{surf_name}_races'].clip(1)
        stats = stats.merge(ss, on='horse_id', how='left')

    # Distance preference
    for dist_name, lo, hi in [('sprint', 0, 1400), ('mile', 1401, 1800),
                               ('middle', 1801, 2200), ('long', 2201, 9999)]:
        mask = (df['distance_n'] >= lo) & (df['distance_n'] <= hi)
        ds = df[mask].groupby('horse_id').agg(
            **{f'{dist_name}_races': ('is_win', 'count'),
               f'{dist_name}_wins': ('is_win', 'sum')}
        ).reset_index()
        ds[f'{dist_name}_wr'] = ds[f'{dist_name}_wins'] / ds[f'{dist_name}_races'].clip(1)
        stats = stats.merge(ds, on='horse_id', how='left')

    # Course preference
    course_stats = df.groupby(['horse_id', 'course']).agg(
        course_races=('is_win', 'count'),
        course_wins=('is_win', 'sum'),
    ).reset_index()
    course_stats['course_wr'] = course_stats['course_wins'] / course_stats['course_races'].clip(1)

    # Best course for each horse
    best_course = course_stats.sort_values('course_wr', ascending=False).drop_duplicates('horse_id')
    best_course = best_course[['horse_id', 'course']].rename(columns={'course': 'best_course'})
    stats = stats.merge(best_course, on='horse_id', how='left')

    # Blood info
    blood_info = df.drop_duplicates(subset='horse_id', keep='last')[
        ['horse_id', 'father', 'mother', 'bms', 'sex', 'birthday']
    ]
    stats = stats.merge(blood_info, on='horse_id', how='left')

    stats = stats.fillna(0)
    print(f"  Horse records: {len(stats)}")

    out_path = os.path.join(DATA_DIR, 'horse_history_full.csv')
    stats.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {out_path} ({len(stats)} rows)")

    return stats


def main():
    print("=" * 60)
    print("  TARGET Frontier JV Data Extraction")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ① Race results
    race_df = extract_races_full()

    # ② Training times
    training_df = extract_training_times()

    # ③ Odds history
    odds_df = extract_odds_history(race_df)

    # ④ Jockey history
    jockey_df = extract_jockey_history(race_df)

    # ⑤ Trainer history
    trainer_df = extract_trainer_history(race_df)

    # ⑥ Blood data
    blood_df = extract_blood_data(race_df)

    # ⑦ Horse history
    horse_df = extract_horse_history(race_df)

    # Summary
    print("\n" + "=" * 60)
    print("  EXTRACTION SUMMARY")
    print("=" * 60)
    results = {
        'jra_races_full.csv': race_df,
        'training_times.csv': training_df,
        'odds_history.csv': odds_df,
        'jockey_history_full.csv': jockey_df,
        'trainer_history_full.csv': trainer_df,
        'blood_full.csv': blood_df,
        'horse_history_full.csv': horse_df,
    }
    for name, df in results.items():
        rows = len(df) if df is not None else 0
        print(f"  {name:30s} {rows:>10,} rows")

    print("\nExtraction complete!")


if __name__ == '__main__':
    main()
