"""JRDB JO (情報データ) から 未利用 16 column 抽出.

merge key: race_id (jrdb 形式 = jra_race_id) + umaban
jrdb_jo.csv は 302K rows、 既加入 source。

抽出 features:
- cid_soten_idx, cid_sara_idx, cid_idx, cid (4 件、 CID 系 評価)
- ls_idx (LS 指標)、 ls_eval (LS 評価)
- yoso_odds (予想オッズ、 LIVE 前 推定値)
- gaisha_bb, gaisha_bb_wr, gaisha_bb_rensho (外舎 / 外厩 統計)
- breeder_bb, breeder_bb_wr, breeder_bb_rensho (生産者 統計)
- soten_odds (騒天 オッズ、 投票 trend)
- em (em 評価)

V20+/V22 学習用、 V15 不変。

usage:
    python train/features_jrdb_jo.py [--start-year 20]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent


def build_jo_features(races_csv: str, jo_csv: str, start_year: int = 20) -> pd.DataFrame:
    print(f"[jrdb_jo] reading {races_csv} ...")
    df = pd.read_csv(races_csv, encoding='utf-8-sig', low_memory=False,
                     usecols=['year', 'race_id', 'horse_id', 'umaban'],
                     dtype={'race_id': str})
    df = df[df['year'] >= start_year].copy()
    print(f"  races: {len(df):,} rows")

    print(f"[jrdb_jo] reading {jo_csv} ...")
    jo = pd.read_csv(jo_csv, encoding='utf-8-sig', low_memory=False,
                     dtype={'race_id': str, 'umaban': str})
    print(f"  jrdb_jo: {len(jo):,} rows")

    # race_id format conversion: jrdb_jo.race_id は 12桁 nk format。
    # jra_races_full の race_id を 12桁 nk に 直接 build (features_track_lap と同じ)
    COURSE_CODE_MAP = {
        '札幌': '01', '函館': '02', '福島': '03', '新潟': '04', '東京': '05',
        '中山': '06', '中京': '07', '京都': '08', '阪神': '09', '小倉': '10',
    }
    full = pd.read_csv(races_csv, encoding='utf-8-sig', low_memory=False,
                       usecols=['year', 'race_id', 'umaban', 'course', 'kai', 'nichi', 'race_num'],
                       dtype={'race_id': str})
    full = full[full['year'] >= start_year].copy()
    full['_course_code'] = full['course'].map(COURSE_CODE_MAP).fillna('00')
    full['_nk_id'] = (
        '20' + full['year'].astype(int).astype(str).str.zfill(2)
        + full['_course_code']
        + full['kai'].astype(int).astype(str).str.zfill(2)
        + full['nichi'].astype(int).astype(str).str.zfill(2)
        + full['race_num'].astype(int).astype(str).str.zfill(2)
    )
    # df は (year >= start_year) フィルタ済、 同じ filter で full と join
    df = df.merge(full[['race_id', 'umaban', '_nk_id']], on=['race_id', 'umaban'], how='left')

    # jrdb_jo merge keys: race_id (nk) + umaban
    feature_cols = ['cid_soten_idx', 'cid_sara_idx', 'cid_idx', 'cid',
                    'ls_idx', 'ls_eval', 'yoso_odds', 'em',
                    'gaisha_bb', 'gaisha_bb_wr', 'gaisha_bb_rensho',
                    'breeder_bb', 'breeder_bb_wr', 'breeder_bb_rensho',
                    'soten_odds']
    jo['umaban'] = jo['umaban'].astype(str)
    # umaban のzero-pad 揃え
    jo['umaban'] = jo['umaban'].str.zfill(2)
    df['umaban_str'] = pd.to_numeric(df['umaban'], errors='coerce').fillna(0).astype(int).astype(str).str.zfill(2)

    jo_sub = jo[['race_id', 'umaban'] + feature_cols].copy()
    jo_sub = jo_sub.drop_duplicates(['race_id', 'umaban'], keep='last')

    df = df.merge(jo_sub.rename(columns={'race_id': '_nk_id'}),
                  left_on=['_nk_id', 'umaban_str'],
                  right_on=['_nk_id', 'umaban'], how='left', suffixes=('', '_jo'))

    matched = df['cid_idx'].notna().sum()
    print(f"  matched: {matched:,}/{len(df):,} ({matched/len(df)*100:.1f}%)")

    # fill na
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 派生
    df['cid_evaluation'] = df['cid_soten_idx'] - df['cid_sara_idx']  # CID 当日 vs 直前 差
    df['ls_strength'] = df['ls_idx']  # LS 指標
    df['gaisha_strength'] = df['gaisha_bb_wr'].clip(0, 1) * 100  # 外舎 強さ score
    df['breeder_strength'] = df['breeder_bb_wr'].clip(0, 1) * 100

    out_cols = ['race_id', 'horse_id', 'umaban'] + feature_cols + [
        'cid_evaluation', 'ls_strength', 'gaisha_strength', 'breeder_strength']
    return df[out_cols].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--races', default=str(BASE / 'data' / 'jra_races_full.csv'))
    ap.add_argument('--jo', default=str(BASE / 'data' / 'jrdb_jo.csv'))
    ap.add_argument('--output', default=str(BASE / 'data' / 'features_jrdb_jo.csv'))
    ap.add_argument('--start-year', type=int, default=20)
    args = ap.parse_args()

    out = build_jo_features(args.races, args.jo, start_year=args.start_year)
    print(f"[jrdb_jo] writing {args.output} ({len(out):,} rows × {len(out.columns)} cols)")
    out.to_csv(args.output, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
