"""track_bias + race_lap → race-level features + 馬番 interaction.

netkeiba_track_bias.csv: track_index (-15..+15、 内有利/外有利)
netkeiba_race_lap.csv: pace_first_half, pace_second_half (race-level)

抽出 features:
- track_index (race 単位)
- track_bias_inner_advantage (track_index 反転 == 内有利 flag)
- track_bias_outer_advantage
- pace_first_half, pace_second_half (前走の race lap merge)
- pace_diff (second - first)
- horse_num_x_bias (馬番 × track_index 交互作用)
- bracket_x_bias (枠番 × track_index)
- lap_volatility (lap times の標準偏差、 race 全体 動き)

V20+/V22 学習用、 V15 不変。

usage:
    python train/features_track_lap.py [--start-year 20]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent


def _lap_volatility(s: str) -> float:
    """'12.4 - 11.3 - 12.4' → std."""
    if not isinstance(s, str):
        return 0.0
    try:
        nums = [float(x.strip()) for x in s.split('-') if x.strip()]
        if len(nums) < 2:
            return 0.0
        return float(np.std(nums))
    except Exception:
        return 0.0


def _lap_first_half(s: str) -> float:
    """ラップから前半 3F"""
    if not isinstance(s, str):
        return 0.0
    try:
        nums = [float(x.strip()) for x in s.split('-') if x.strip()]
        if len(nums) < 3:
            return 0.0
        return float(sum(nums[:3]))
    except Exception:
        return 0.0


def _lap_last_3f(s: str) -> float:
    """ラップから上がり 3F"""
    if not isinstance(s, str):
        return 0.0
    try:
        nums = [float(x.strip()) for x in s.split('-') if x.strip()]
        if len(nums) < 3:
            return 0.0
        return float(sum(nums[-3:]))
    except Exception:
        return 0.0


COURSE_CODE_MAP = {
    '札幌': '01', '函館': '02', '福島': '03', '新潟': '04', '東京': '05',
    '中山': '06', '中京': '07', '京都': '08', '阪神': '09', '小倉': '10',
}


def build_track_lap_features(races_csv: str, bias_csv: str, lap_csv: str,
                             mapping_csv: str,
                             start_year: int = 20) -> pd.DataFrame:
    print(f"[track_lap] reading {races_csv} ...")
    df = pd.read_csv(races_csv, encoding='utf-8-sig', low_memory=False,
                     usecols=['year', 'month', 'day', 'race_num', 'race_id',
                              'course', 'kai', 'nichi',
                              'horse_id', 'umaban', 'finish'],
                     dtype={'race_id': str})
    df = df[df['year'] >= start_year].copy()
    print(f"  races: {len(df):,} rows")

    # netkeiba race_id 直接構築 (12桁): "20" + year + course_code + kai + nichi + race_num
    df['_course_code'] = df['course'].map(COURSE_CODE_MAP).fillna('00')
    df['_nk_id'] = (
        '20' + df['year'].astype(int).astype(str).str.zfill(2)
        + df['_course_code']
        + df['kai'].astype(int).astype(str).str.zfill(2)
        + df['nichi'].astype(int).astype(str).str.zfill(2)
        + df['race_num'].astype(int).astype(str).str.zfill(2)
    )
    print(f"  nk_race_id sample: {df['_nk_id'].head(3).tolist()}")

    print(f"[track_lap] reading {bias_csv} ...")
    bias = pd.read_csv(bias_csv, encoding='utf-8-sig', low_memory=False,
                       dtype={'race_id': str})
    bias['track_index'] = pd.to_numeric(bias['track_index'], errors='coerce').fillna(0)
    bias['track_bias_inner_advantage'] = (bias['track_index'] <= -5).astype(int)
    bias['track_bias_outer_advantage'] = (bias['track_index'] >= 5).astype(int)
    bias = bias[['race_id', 'track_index',
                 'track_bias_inner_advantage', 'track_bias_outer_advantage']]
    bias = bias.drop_duplicates('race_id', keep='last')
    print(f"  bias: {len(bias):,} rows (dedup)")

    print(f"[track_lap] reading {lap_csv} ...")
    lap = pd.read_csv(lap_csv, encoding='utf-8-sig', low_memory=False,
                      dtype={'race_id': str})
    lap['pace_first_half'] = pd.to_numeric(lap['pace_first_half'], errors='coerce').fillna(36.0)
    lap['pace_second_half'] = pd.to_numeric(lap['pace_second_half'], errors='coerce').fillna(36.0)
    lap['pace_diff_race'] = lap['pace_second_half'] - lap['pace_first_half']
    lap['lap_volatility'] = lap['lap_times'].apply(_lap_volatility)
    lap['lap_first_3f_race'] = lap['lap_times'].apply(_lap_first_half)
    lap['lap_last_3f_race'] = lap['lap_times'].apply(_lap_last_3f)
    lap = lap[['race_id', 'pace_first_half', 'pace_second_half',
               'pace_diff_race', 'lap_volatility', 'lap_first_3f_race', 'lap_last_3f_race']]
    lap = lap.drop_duplicates('race_id', keep='last')
    print(f"  lap: {len(lap):,} rows (dedup)")

    df = df.merge(bias.rename(columns={'race_id': '_nk_id'}), on='_nk_id', how='left')
    matched = df['track_index'].notna().sum()
    print(f"  bias matched: {matched:,}/{len(df):,} ({matched/len(df)*100:.1f}%)")

    df = df.merge(lap.rename(columns={'race_id': '_nk_id'}), on='_nk_id', how='left')
    matched = df['pace_first_half'].notna().sum()
    print(f"  lap matched: {matched:,}/{len(df):,} ({matched/len(df)*100:.1f}%)")

    # fillna
    df['track_index'] = df['track_index'].fillna(0)
    df['track_bias_inner_advantage'] = df['track_bias_inner_advantage'].fillna(0).astype(int)
    df['track_bias_outer_advantage'] = df['track_bias_outer_advantage'].fillna(0).astype(int)
    df['pace_first_half'] = df['pace_first_half'].fillna(36.0)
    df['pace_second_half'] = df['pace_second_half'].fillna(36.0)
    df['pace_diff_race'] = df['pace_diff_race'].fillna(0)
    df['lap_volatility'] = df['lap_volatility'].fillna(0)
    df['lap_first_3f_race'] = df['lap_first_3f_race'].fillna(36.0)
    df['lap_last_3f_race'] = df['lap_last_3f_race'].fillna(36.0)

    # 馬番 × track_index 交互作用
    df['umaban_num'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(8).astype(int)
    df['horse_num_x_bias'] = df['umaban_num'] * df['track_index']
    # 中央 umaban (5-12) は bias 影響 小、 1-4 or 13+ は 大
    df['is_inner_umaban'] = (df['umaban_num'] <= 4).astype(int)
    df['is_outer_umaban'] = (df['umaban_num'] >= 13).astype(int)
    df['umaban_x_inner_advantage'] = df['is_inner_umaban'] * df['track_bias_inner_advantage']
    df['umaban_x_outer_advantage'] = df['is_outer_umaban'] * df['track_bias_outer_advantage']

    out_cols = ['race_id', 'horse_id', 'umaban',
                'track_index',
                'track_bias_inner_advantage', 'track_bias_outer_advantage',
                'pace_first_half', 'pace_second_half', 'pace_diff_race',
                'lap_volatility', 'lap_first_3f_race', 'lap_last_3f_race',
                'horse_num_x_bias',
                'is_inner_umaban', 'is_outer_umaban',
                'umaban_x_inner_advantage', 'umaban_x_outer_advantage']
    return df[out_cols].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--races', default=str(BASE / 'data' / 'jra_races_full.csv'))
    ap.add_argument('--bias', default=str(BASE / 'data' / 'netkeiba_track_bias.csv'))
    ap.add_argument('--lap', default=str(BASE / 'data' / 'netkeiba_race_lap.csv'))
    ap.add_argument('--mapping', default=str(BASE / 'data' / 'race_id_mapping.csv'))
    ap.add_argument('--output', default=str(BASE / 'data' / 'features_track_lap.csv'))
    ap.add_argument('--start-year', type=int, default=20)
    args = ap.parse_args()

    out = build_track_lap_features(args.races, args.bias, args.lap, args.mapping,
                                    start_year=args.start_year)
    print(f"[track_lap] writing {args.output} ({len(out):,} rows × {len(out.columns)} cols)")
    out.to_csv(args.output, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
