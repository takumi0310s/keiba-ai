"""馬 高度 features: 重賞 / streak / jockey-horse combo / 同 race 連投.

既存 jra_races_full.csv のみ で 計算 (expanding LEAK-free)。

抽出 features:
- horse_grade_win_count (G1/G2/G3 勝ち数 expanding)
- horse_grade_top3_count (G1-G3 入着 数)
- horse_open_win_count (OP・特別 勝ち数)
- horse_winning_streak (連勝 streak、 直近 連勝 数)
- horse_losing_streak (連敗 数)
- horse_total_prize_career (累計 賞金 cumsum、 ※当該レース除外)
- jockey_horse_combo_count (馬と騎手 過去 騎乗 回数)
- jockey_horse_combo_top3r (combo 入着率、 Bayesian)
- trainer_horse_combo_count (馬と調教師 関係、 入厩期間 推定)
- same_race_attempts (同 race_name 連投 回数 — G1 等で 同 race 挑戦 履歴)
- horse_recent_prize_3r (直近 3 走 賞金 sum)
- horse_pop_avg_3r (直近 3 走 人気 平均)
- horse_jrace_count (生涯 出走 回数 expanding)
- horse_recent_top1_rate_5r (直近 5 走 1着率)

usage:
    python train/features_horse_advanced.py [--start-year 20]
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent


def _class_to_grade(c) -> int:
    """class_code → grade 番号.
    1=G1, 2=G2, 3=G3, 4=OP特別, 5=L 等。 0=平場/未勝利。
    JRA class_code 標準 mapping (近似)。
    """
    try:
        v = int(c)
    except Exception:
        return 0
    return v


def _is_grade1_2_3(c) -> int:
    v = _class_to_grade(c)
    # 1=G1, 2=G2, 3=G3 (重賞)
    return 1 if v in (1, 2, 3) else 0


def _is_grade1(c) -> int:
    return 1 if _class_to_grade(c) == 1 else 0


def _winning_streak(s: pd.Series) -> pd.Series:
    """連続して1着続いた数 (現在まで)。 win 列を 0/1 で受け取り."""
    streak = []
    cur = 0
    for v in s.tolist():
        if v == 1:
            cur += 1
        else:
            cur = 0
        streak.append(cur)
    return pd.Series(streak, index=s.index)


def _losing_streak(s: pd.Series) -> pd.Series:
    """連続して 1 着でなかった数 (現在まで)。"""
    streak = []
    cur = 0
    for v in s.tolist():
        if v == 0:
            cur += 1
        else:
            cur = 0
        streak.append(cur)
    return pd.Series(streak, index=s.index)


def build_horse_advanced(races_csv: str, start_year: int = 20) -> pd.DataFrame:
    print(f"[horse_advanced] reading {races_csv} ...")
    df = pd.read_csv(races_csv, encoding='utf-8-sig', low_memory=False,
                     usecols=['year', 'month', 'day', 'race_num', 'race_id',
                              'horse_id', 'jockey_id', 'trainer_id', 'umaban',
                              'finish', 'class_code', 'prize', 'race_name', 'popularity'],
                     dtype={'race_id': str})
    df = df[df['year'] >= start_year].copy()
    print(f"  races: {len(df):,} rows")

    # sort: 馬 ごと 時系列順
    df = df.sort_values(['horse_id', 'year', 'month', 'day', 'race_num']).reset_index(drop=True)

    # 重賞 flag
    df['_is_grade'] = df['class_code'].apply(_is_grade1_2_3)
    df['_is_g1'] = df['class_code'].apply(_is_grade1)
    df['_is_win'] = (df['finish'] == 1).astype(int)
    df['_is_top3'] = (df['finish'] <= 3).astype(int)
    df['_grade_win'] = df['_is_grade'] * df['_is_win']
    df['_grade_top3'] = df['_is_grade'] * df['_is_top3']
    df['_g1_win'] = df['_is_g1'] * df['_is_win']
    df['_prize_num'] = pd.to_numeric(df['prize'], errors='coerce').fillna(0)
    df['_pop_num'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(99)

    grp_horse = df.groupby('horse_id')

    print("  computing horse cumulative features ...")
    # 当該 行 除外 = cumsum().shift(1)
    df['horse_grade_win_count'] = grp_horse['_grade_win'].cumsum().shift(1).fillna(0).astype(int)
    df['horse_grade_top3_count'] = grp_horse['_grade_top3'].cumsum().shift(1).fillna(0).astype(int)
    df['horse_g1_win_count'] = grp_horse['_g1_win'].cumsum().shift(1).fillna(0).astype(int)
    df['horse_total_prize_career'] = grp_horse['_prize_num'].cumsum().shift(1).fillna(0)
    df['horse_jrace_count'] = grp_horse.cumcount()

    # streak (前走以前 base、 当該 行 除外)
    print("  computing streaks ...")
    df['_prev_win'] = grp_horse['_is_win'].shift(1).fillna(0).astype(int)
    # 直接 iteration で streak 計算 (高速 + index 維持)
    horse_arr = df['horse_id'].values
    prev_win_arr = df['_prev_win'].values
    win_streak = np.zeros(len(df), dtype=np.int32)
    lose_streak = np.zeros(len(df), dtype=np.int32)
    cur_win = 0
    cur_lose = 0
    prev_h = None
    for i in range(len(df)):
        h = horse_arr[i]
        if h != prev_h:
            cur_win = 0
            cur_lose = 0
            prev_h = h
        if prev_win_arr[i] == 1:
            cur_win += 1
            cur_lose = 0
        else:
            cur_lose += 1
            cur_win = 0
        win_streak[i] = cur_win
        lose_streak[i] = cur_lose
    df['horse_winning_streak'] = win_streak
    df['horse_losing_streak'] = lose_streak

    # 直近 3 走 / 5 走 集計
    print("  rolling features ...")
    df['horse_recent_prize_3r'] = grp_horse['_prize_num'].shift(1).rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    df['horse_pop_avg_3r'] = grp_horse['_pop_num'].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df['horse_recent_top1_rate_5r'] = grp_horse['_is_win'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)

    # jockey-horse combo (馬 + 騎手 ペアの 過去 出走数 + top3r)
    print("  computing jockey-horse combo ...")
    df['_jh_pair'] = df['horse_id'].astype(str) + '_' + df['jockey_id'].astype(str)
    grp_jh = df.groupby('_jh_pair')
    df['jockey_horse_combo_count'] = grp_jh.cumcount()  # 当該 row 以前 の count
    cumsum_top3 = grp_jh['_is_top3'].cumsum().shift(1).fillna(0)
    df['jockey_horse_combo_top3'] = cumsum_top3.astype(int)
    # Bayesian (alpha=5)
    df['jockey_horse_combo_top3r'] = (cumsum_top3 + 5 * 0.25) / (df['jockey_horse_combo_count'] + 5)
    df['jockey_horse_combo_top3r'] = df['jockey_horse_combo_top3r'].clip(0, 1)

    # trainer-horse 期間 (近似: 入厩期間)
    print("  computing trainer-horse combo ...")
    df['_th_pair'] = df['horse_id'].astype(str) + '_' + df['trainer_id'].astype(str)
    grp_th = df.groupby('_th_pair')
    df['trainer_horse_combo_count'] = grp_th.cumcount()

    # 同 race_name 連投 (例: 同じ G1 へ 何回 挑戦)
    print("  same race_name attempts ...")
    df['_hn_pair'] = df['horse_id'].astype(str) + '_' + df['race_name'].astype(str)
    grp_hn = df.groupby('_hn_pair')
    df['same_race_attempts'] = grp_hn.cumcount()

    # cleanup
    df = df.drop(columns=['_is_grade', '_is_g1', '_is_win', '_is_top3',
                          '_grade_win', '_grade_top3', '_g1_win',
                          '_prize_num', '_pop_num', '_prev_win',
                          '_jh_pair', '_th_pair', '_hn_pair'])

    out_cols = ['race_id', 'horse_id', 'umaban',
                'horse_grade_win_count', 'horse_grade_top3_count',
                'horse_g1_win_count', 'horse_total_prize_career',
                'horse_jrace_count',
                'horse_winning_streak', 'horse_losing_streak',
                'horse_recent_prize_3r', 'horse_pop_avg_3r',
                'horse_recent_top1_rate_5r',
                'jockey_horse_combo_count', 'jockey_horse_combo_top3',
                'jockey_horse_combo_top3r',
                'trainer_horse_combo_count', 'same_race_attempts']
    return df[out_cols].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--races', default=str(BASE / 'data' / 'jra_races_full.csv'))
    ap.add_argument('--output', default=str(BASE / 'data' / 'features_horse_advanced.csv'))
    ap.add_argument('--start-year', type=int, default=20)
    args = ap.parse_args()

    out = build_horse_advanced(args.races, start_year=args.start_year)
    print(f"[horse_advanced] writing {args.output} ({len(out):,} rows × {len(out.columns)} cols)")
    out.to_csv(args.output, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
