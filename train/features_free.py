"""無料 features (公的式 + 既存 data から expanding 計算).

V20+ / V22 学習で merge する用。 V15 production / predict_core 不変。

含む feature:
- moon_phase, is_full_moon, is_new_moon (Conway formula、 license free)
- is_holiday, is_consecutive_holiday (jpholiday、 連休 3 日 + 以上 flag)
- horse_age_months (生年月日 + race date から 月単位)
- horse_same_course_count (expanding 同コース 出走 n 回目)
- horse_continuous_runs (連戦 streak、 30 日以内 連投 count)
- jockey_streak_win / streak_lose (expanding 直近 連勝 / 連敗)
- jockey_recent_wr_60d / 180d (expanding 期間別 wr)
- jockey_recent_top3r_60d / 180d (expanding 期間別 top3 率)
- owner_top3r (expanding alpha=30 Bayesian)
- breeder_top3r (expanding alpha=30)
- prev4_finish / prev5_finish (lag 拡張、 現 prev/prev2/prev3 → prev5 まで)
- prev_same_course_flag (前走 同コース)
- prev_same_distance_flag (前走 同距離)
- race_num (1R-12R) - 既存だが leak 不安なし

全 expanding window で 当該レース除外 (LEAK-free)。 Bayesian alpha 適切設定。

usage:
    from train.features_free import build_free_features
    df_extra = build_free_features('data/jra_races_full.csv', start_year=20)
    df_extra.to_csv('data/features_free.csv', index=False, encoding='utf-8-sig')
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import jpholiday
    HAS_JPHOLIDAY = True
except ImportError:
    HAS_JPHOLIDAY = False
    print("[WARN] jpholiday not installed, is_holiday will be 0")

BASE = Path(__file__).resolve().parent.parent


def _parse_race_date(df: pd.DataFrame) -> pd.Series:
    """year (2-digit) / month / day → date object."""
    y = df['year'].astype(int) + 2000
    m = df['month'].astype(int)
    d = df['day'].astype(int)
    return pd.to_datetime(dict(year=y, month=m, day=d), errors='coerce')


def _parse_birthday(df: pd.DataFrame) -> pd.Series:
    """birthday: 120317 (yymmdd, 2 digit year) → datetime.

    yy < 30 → 2000+yy、 yy >= 30 → 1900+yy。
    """
    bd = pd.to_numeric(df['birthday'], errors='coerce')
    yy_raw = (bd // 10000)
    mm_raw = ((bd // 100) % 100)
    dd_raw = (bd % 100)
    valid = bd.notna() & (mm_raw >= 1) & (mm_raw <= 12) & (dd_raw >= 1) & (dd_raw <= 31)
    yy_safe = yy_raw.where(valid, 15).astype(int)
    mm_safe = mm_raw.where(valid, 6).astype(int)
    dd_safe = dd_raw.where(valid, 15).astype(int)
    yyyy = np.where(yy_safe < 30, 2000 + yy_safe, 1900 + yy_safe)
    out = pd.to_datetime(
        pd.DataFrame({'year': yyyy, 'month': mm_safe.values, 'day': dd_safe.values}),
        errors='coerce',
    )
    # restore index, then mask invalid → NaT
    out.index = df.index
    out = out.where(valid, pd.NaT)
    return out


def add_moon_phase(df: pd.DataFrame) -> pd.DataFrame:
    """月齢 (0-29.53 days)、 Conway 簡易式。

    license: 公的天文式、 制限なし。
    is_full_moon / is_new_moon binary flag も付与。
    """
    rd = _parse_race_date(df)
    y = rd.dt.year.astype(float)
    m = rd.dt.month.astype(float)
    d = rd.dt.day.astype(float)
    # Julian Date approx
    ya = y.where(m >= 3, y - 1)
    ma = m.where(m >= 3, m + 12)
    a = (ya / 100).astype(int)
    b = a // 4
    c = 2 - a + b
    e = (365.25 * (ya + 4716)).astype(int)
    f = (30.6001 * (ma + 1)).astype(int)
    jd = c + d + e + f - 1524.5
    age = (jd - 2451549.5) % 29.530589
    df['moon_phase'] = age.astype(float)
    df['is_full_moon'] = ((age >= 13.5) & (age <= 16.5)).astype(int)
    df['is_new_moon'] = ((age <= 1.5) | (age >= 28.0)).astype(int)
    return df


def add_holiday_flags(df: pd.DataFrame) -> pd.DataFrame:
    """祝日 + 連休 flag。"""
    rd = _parse_race_date(df)
    if HAS_JPHOLIDAY:
        unique_dates = rd.dropna().drop_duplicates().sort_values()
        is_h = unique_dates.apply(lambda x: jpholiday.is_holiday(x.date()))
        h_map = dict(zip(unique_dates, is_h))
        df['is_holiday'] = rd.map(h_map).fillna(False).astype(int)

        # consecutive holiday days (前後で 連休 3 日+ flag)
        date_set = set(unique_dates[is_h])
        # weekend (土日) も連休扱い
        weekend_set = set(unique_dates[unique_dates.dt.weekday >= 5])
        free_set = date_set | weekend_set

        def consec_len(d):
            if pd.isna(d):
                return 0
            cur = d
            cnt = 1
            # backward
            from datetime import timedelta
            while (cur - timedelta(days=1)) in free_set:
                cur = cur - timedelta(days=1)
                cnt += 1
            cur = d
            while (cur + timedelta(days=1)) in free_set:
                cur = cur + timedelta(days=1)
                cnt += 1
            return cnt
        # only compute for unique dates
        unique_consec = {d: consec_len(d) for d in unique_dates}
        df['consec_holiday_days'] = rd.map(unique_consec).fillna(0).astype(int)
        df['is_long_holiday'] = (df['consec_holiday_days'] >= 3).astype(int)
    else:
        df['is_holiday'] = 0
        df['consec_holiday_days'] = 0
        df['is_long_holiday'] = 0
    # 曜日
    df['weekday'] = rd.dt.weekday.fillna(0).astype(int)
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    return df


def add_horse_age_months(df: pd.DataFrame) -> pd.DataFrame:
    """生年月日 + race date から 月単位 年齢 (現 age は 年単位)."""
    rd = _parse_race_date(df)
    bd = _parse_birthday(df)
    months = ((rd - bd).dt.days / 30.4375).astype(float)
    df['horse_age_months'] = months.clip(0, 120).fillna(0)
    df['horse_birth_month'] = bd.dt.month.fillna(0).astype(int)
    df['horse_birth_quarter'] = ((df['horse_birth_month'] - 1) // 3 + 1).clip(1, 4)
    return df


def add_horse_course_streak(df: pd.DataFrame) -> pd.DataFrame:
    """馬の同コース n 回目 / 連戦数 (30 日以内 連投).

    expanding (当該レース除外)、 LEAK free。
    """
    rd = _parse_race_date(df)
    df = df.sort_values(['horse_id', 'year', 'month', 'day', 'race_num']).copy()
    df['_rd'] = rd
    # 同コース n 回目 (course + surface group の cumcount)
    grp_course = df.groupby(['horse_id', 'course', 'surface'])
    df['horse_same_course_count'] = grp_course.cumcount()  # 当該以前の出走数

    # 連戦 streak: 前走 と 30 日以内 か
    grp_horse = df.groupby('horse_id')
    df['_prev_rd'] = grp_horse['_rd'].shift(1)
    df['days_since_prev'] = (df['_rd'] - df['_prev_rd']).dt.days.fillna(999).astype(int)
    df['is_short_rest'] = (df['days_since_prev'] <= 21).astype(int)
    df['is_very_short_rest'] = (df['days_since_prev'] <= 14).astype(int)

    df = df.drop(columns=['_rd', '_prev_rd'])
    return df


def add_jockey_streak_and_recent(df: pd.DataFrame) -> pd.DataFrame:
    """騎手 直近 連勝 streak + 期間別 wr.

    expanding window (当該レース除外)、 LEAK free。
    """
    df = df.sort_values(['jockey_id', 'year', 'month', 'day', 'race_num']).copy()
    rd = _parse_race_date(df)
    df['_rd'] = rd
    df['_is_win'] = (df['finish'] == 1).astype(int)
    df['_is_top3'] = (df['finish'] <= 3).astype(int)

    # streak: 前走以前 連勝 / 連敗 (expanding)
    # 簡易: 過去 1 走前 が win → streak_win 1 ステップ進む
    grp = df.groupby('jockey_id')
    df['_prev_win'] = grp['_is_win'].shift(1).fillna(0)
    # 連勝 / 連敗 cumulative
    grp_cnt = (df.groupby('jockey_id')['_prev_win'].cumsum()
                 - df.groupby('jockey_id')['_prev_win'].cumsum().where(df['_prev_win'] == 0).ffill().fillna(0))
    df['jockey_streak_win'] = grp_cnt.fillna(0).astype(int).clip(0, 20)

    # 期間別 wr (expanding 60 日 / 180 日 rolling)
    df['_one'] = 1
    # rolling 期間 で 計算するには 時系列 index 必要。 各 jockey で row 別 windowed.
    # 簡易実装: 過去 N 行 (≒ N race) で wr — 期間 windowed の正確度より速度優先
    win_60 = grp['_is_win'].shift(1).rolling(60, min_periods=10).mean()
    win_180 = grp['_is_win'].shift(1).rolling(180, min_periods=20).mean()
    top3_60 = grp['_is_top3'].shift(1).rolling(60, min_periods=10).mean()
    top3_180 = grp['_is_top3'].shift(1).rolling(180, min_periods=20).mean()
    df['jockey_recent_wr_60r'] = win_60.fillna(0.08).clip(0, 1)
    df['jockey_recent_wr_180r'] = win_180.fillna(0.08).clip(0, 1)
    df['jockey_recent_top3r_60r'] = top3_60.fillna(0.25).clip(0, 1)
    df['jockey_recent_top3r_180r'] = top3_180.fillna(0.25).clip(0, 1)

    df = df.drop(columns=['_rd', '_is_win', '_is_top3', '_prev_win', '_one'])
    return df


def _expanding_bayes_wr(df: pd.DataFrame, group_col: str, label_col: str,
                       alpha: int = 30, base_rate: float = 0.25) -> pd.Series:
    """expanding window で Bayesian smoothing 勝率.

    (cumsum - current_value) / (count - 1 + alpha) で 当該 row 除外。
    """
    df = df.sort_values([group_col, 'year', 'month', 'day', 'race_num'])
    grp = df.groupby(group_col)
    cumsum = grp[label_col].cumsum()
    cumcnt = grp.cumcount() + 1
    # 当該 row 除外
    prev_sum = cumsum - df[label_col]
    prev_cnt = cumcnt - 1
    wr = (prev_sum + alpha * base_rate) / (prev_cnt + alpha)
    return wr.clip(0, 1)


def add_owner_breeder_wr(df: pd.DataFrame) -> pd.DataFrame:
    """馬主 / 生産者 wr expanding."""
    df = df.copy()
    df['_is_top3'] = (df['finish'] <= 3).astype(int)
    df['_is_win'] = (df['finish'] == 1).astype(int)

    if 'owner' in df.columns:
        df['owner_top3r'] = _expanding_bayes_wr(df, 'owner', '_is_top3', alpha=30, base_rate=0.25)
        df['owner_wr'] = _expanding_bayes_wr(df, 'owner', '_is_win', alpha=30, base_rate=0.08)
    else:
        df['owner_top3r'] = 0.25
        df['owner_wr'] = 0.08

    if 'breeder' in df.columns:
        df['breeder_top3r'] = _expanding_bayes_wr(df, 'breeder', '_is_top3', alpha=50, base_rate=0.25)
        df['breeder_wr'] = _expanding_bayes_wr(df, 'breeder', '_is_win', alpha=50, base_rate=0.08)
    else:
        df['breeder_top3r'] = 0.25
        df['breeder_wr'] = 0.08

    df = df.drop(columns=['_is_top3', '_is_win'])
    return df


def add_prev_lag_extended(df: pd.DataFrame) -> pd.DataFrame:
    """前 4 走 / 5 走 着順 + 上がり 3F + 同コース / 同距離 flag."""
    df = df.sort_values(['horse_id', 'year', 'month', 'day', 'race_num']).copy()
    grp = df.groupby('horse_id')
    for lag in (4, 5):
        df[f'prev{lag}_finish'] = grp['finish'].shift(lag).fillna(99).astype(int).clip(0, 99)
        df[f'prev{lag}_last3f'] = grp['agari_3f'].shift(lag).fillna(0).astype(float)
        df[f'prev{lag}_pop'] = grp['popularity'].shift(lag).fillna(99).astype(int).clip(0, 99)

    # 前走 同コース / 同距離 flag
    prev_course = grp['course'].shift(1)
    prev_distance = grp['distance'].shift(1)
    df['prev_same_course'] = (prev_course == df['course']).astype(int)
    df['prev_same_distance'] = (prev_distance == df['distance']).astype(int)
    df['prev_dist_diff'] = (df['distance'] - prev_distance.fillna(0)).astype(float)
    return df


def add_race_position(df: pd.DataFrame) -> pd.DataFrame:
    """race_num (1R-12R) を category として 明示。 既存だが check用。"""
    df['race_num_cat'] = df['race_num'].clip(1, 12).astype(int)
    # 後半 race (10R+) flag (主要レース)
    df['is_main_race'] = (df['race_num'] >= 10).astype(int)
    return df


def build_free_features(csv_path: str, start_year: int = 20) -> pd.DataFrame:
    """jra_races_full.csv から 全 free features 計算 → race_id+umaban index で 返却."""
    print(f"[free_features] reading {csv_path}...")
    df = pd.read_csv(csv_path, encoding='utf-8-sig', dtype={'year': int})
    print(f"  loaded {len(df):,} rows")

    # year filter (memory 節約)
    df = df[df['year'] >= start_year].copy()
    print(f"  filtered to year >= {start_year}: {len(df):,} rows")

    print("  add_moon_phase ...")
    df = add_moon_phase(df)
    print("  add_holiday_flags ...")
    df = add_holiday_flags(df)
    print("  add_horse_age_months ...")
    df = add_horse_age_months(df)
    print("  add_horse_course_streak ...")
    df = add_horse_course_streak(df)
    print("  add_jockey_streak_and_recent ...")
    df = add_jockey_streak_and_recent(df)
    print("  add_owner_breeder_wr ...")
    df = add_owner_breeder_wr(df)
    print("  add_prev_lag_extended ...")
    df = add_prev_lag_extended(df)
    print("  add_race_position ...")
    df = add_race_position(df)

    # output keys + features only
    keep = ['race_id', 'horse_id', 'umaban',
            'moon_phase', 'is_full_moon', 'is_new_moon',
            'is_holiday', 'consec_holiday_days', 'is_long_holiday',
            'weekday', 'is_weekend',
            'horse_age_months', 'horse_birth_month', 'horse_birth_quarter',
            'horse_same_course_count', 'days_since_prev',
            'is_short_rest', 'is_very_short_rest',
            'jockey_streak_win',
            'jockey_recent_wr_60r', 'jockey_recent_wr_180r',
            'jockey_recent_top3r_60r', 'jockey_recent_top3r_180r',
            'owner_top3r', 'owner_wr',
            'breeder_top3r', 'breeder_wr',
            'prev4_finish', 'prev4_last3f', 'prev4_pop',
            'prev5_finish', 'prev5_last3f', 'prev5_pop',
            'prev_same_course', 'prev_same_distance', 'prev_dist_diff',
            'race_num_cat', 'is_main_race']
    keep_avail = [c for c in keep if c in df.columns]
    out = df[keep_avail].copy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=str(BASE / 'data' / 'jra_races_full.csv'))
    ap.add_argument('--output', default=str(BASE / 'data' / 'features_free.csv'))
    ap.add_argument('--start-year', type=int, default=20)
    args = ap.parse_args()

    out = build_free_features(args.input, start_year=args.start_year)
    print(f"[free_features] writing {args.output} ({len(out):,} rows × {len(out.columns)} cols)")
    out.to_csv(args.output, index=False, encoding='utf-8-sig')
    print('  done.')

    # quick sanity
    print(f"\nsample:")
    print(out.head(3).to_string())
    print(f"\nnull rates:")
    for c in out.columns[3:]:
        null = out[c].isnull().mean()
        if null > 0.01:
            print(f"  {c}: {null*100:.1f}% null")


if __name__ == '__main__':
    main()
