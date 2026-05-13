"""JRA 10 場 + 主要 地方場 hardcoded 地理・コース特徴.

公的データ (Google Maps / 国土地理院 / JRA 公式) より。 商標 不使用、 数値 のみ。

抽出 features:
- venue_elevation_m (海抜 m)
- venue_lat / venue_lon (緯度 / 経度)
- venue_temp_base_apr (4 月平均気温 = race 多い 月の基準)
- venue_humidity_jul (7 月平均湿度)
- track_circumference_m (右回りコース 1 周距離)
- track_homestretch_m (直線 距離)
- track_curve_type (1=きつい / 2=普通 / 3=ゆるい)
- track_grade_diff_m (高低差 m)
- is_right_turn (右回り=1、 左回り=0)
- venue_humid_factor (湿潤性 indicator)
- is_seaside (海近 = 風強)

V20+/V22 学習用、 V15 不変。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent

# JRA 10 場 + 主要 NAR 地方場 の geographic + course detail
# source: JRA 公式 / 国土地理院 / 各 場 公式
VENUE_DATA = {
    '札幌':  dict(lat=43.0667, lon=141.3833, elev=15,  temp_apr=7.5,  humid_jul=78,
                 circ=1640, home=266, curve=2, grade=0.7, right=1, humid_factor=2, seaside=1),
    '函館':  dict(lat=41.8167, lon=140.7500, elev=4,   temp_apr=7.0,  humid_jul=82,
                 circ=1626, home=263, curve=1, grade=3.5, right=1, humid_factor=2, seaside=1),
    '福島':  dict(lat=37.7833, lon=140.4500, elev=68,  temp_apr=11.0, humid_jul=76,
                 circ=1600, home=292, curve=1, grade=1.9, right=1, humid_factor=1, seaside=0),
    '新潟':  dict(lat=37.9167, lon=139.0500, elev=8,   temp_apr=10.8, humid_jul=76,
                 circ=2223, home=659, curve=3, grade=0.8, right=0, humid_factor=2, seaside=1),
    '東京':  dict(lat=35.6500, lon=139.4500, elev=40,  temp_apr=13.5, humid_jul=76,
                 circ=2083, home=525, curve=3, grade=2.7, right=0, humid_factor=1, seaside=0),
    '中山':  dict(lat=35.7100, lon=140.0167, elev=14,  temp_apr=13.0, humid_jul=78,
                 circ=1840, home=310, curve=1, grade=5.3, right=1, humid_factor=1, seaside=0),
    '中京':  dict(lat=35.1167, lon=136.8667, elev=10,  temp_apr=13.2, humid_jul=72,
                 circ=1705, home=412, curve=2, grade=2.4, right=0, humid_factor=1, seaside=0),
    '京都':  dict(lat=35.0500, lon=135.7500, elev=70,  temp_apr=13.5, humid_jul=70,
                 circ=1782, home=403, curve=2, grade=4.3, right=1, humid_factor=1, seaside=0),
    '阪神':  dict(lat=34.7833, lon=135.3667, elev=4,   temp_apr=14.0, humid_jul=72,
                 circ=1689, home=473, curve=2, grade=2.4, right=1, humid_factor=1, seaside=1),
    '小倉':  dict(lat=33.8833, lon=130.8833, elev=12,  temp_apr=14.5, humid_jul=78,
                 circ=1615, home=293, curve=1, grade=3.0, right=1, humid_factor=2, seaside=1),
    # 主要 NAR
    '川崎':  dict(lat=35.5333, lon=139.7000, elev=3,   temp_apr=14.0, humid_jul=78,
                 circ=1200, home=300, curve=1, grade=0.0, right=0, humid_factor=2, seaside=1),
    '大井':  dict(lat=35.5833, lon=139.7500, elev=2,   temp_apr=14.0, humid_jul=78,
                 circ=1600, home=386, curve=1, grade=0.0, right=0, humid_factor=2, seaside=1),
    '船橋':  dict(lat=35.7000, lon=140.0000, elev=3,   temp_apr=13.5, humid_jul=78,
                 circ=1400, home=308, curve=1, grade=0.0, right=0, humid_factor=2, seaside=1),
    '浦和':  dict(lat=35.8500, lon=139.6500, elev=10,  temp_apr=13.0, humid_jul=76,
                 circ=1200, home=220, curve=1, grade=0.0, right=0, humid_factor=1, seaside=0),
    '園田':  dict(lat=34.7333, lon=135.4167, elev=4,   temp_apr=14.0, humid_jul=72,
                 circ=1051, home=213, curve=1, grade=0.0, right=1, humid_factor=2, seaside=1),
    '門別':  dict(lat=42.4500, lon=142.2833, elev=20,  temp_apr=4.8,  humid_jul=82,
                 circ=1600, home=330, curve=1, grade=0.0, right=0, humid_factor=2, seaside=1),
    '盛岡':  dict(lat=39.7000, lon=141.1833, elev=200, temp_apr=8.0,  humid_jul=78,
                 circ=1600, home=305, curve=1, grade=4.4, right=0, humid_factor=1, seaside=0),
    '水沢':  dict(lat=39.1500, lon=141.1333, elev=46,  temp_apr=9.5,  humid_jul=78,
                 circ=1200, home=245, curve=1, grade=0.0, right=0, humid_factor=1, seaside=0),
    '名古屋':dict(lat=35.1500, lon=137.0000, elev=10,  temp_apr=13.2, humid_jul=72,
                 circ=1100, home=240, curve=1, grade=0.0, right=0, humid_factor=1, seaside=0),
    '金沢':  dict(lat=36.6000, lon=136.6500, elev=10,  temp_apr=11.5, humid_jul=74,
                 circ=1200, home=236, curve=1, grade=0.0, right=0, humid_factor=2, seaside=1),
    '高知':  dict(lat=33.5667, lon=133.5333, elev=10,  temp_apr=14.5, humid_jul=78,
                 circ=1100, home=200, curve=1, grade=0.0, right=0, humid_factor=2, seaside=1),
    '佐賀':  dict(lat=33.2667, lon=130.3000, elev=8,   temp_apr=14.0, humid_jul=80,
                 circ=1100, home=220, curve=1, grade=0.0, right=0, humid_factor=2, seaside=0),
    '笠松':  dict(lat=35.3833, lon=136.7667, elev=11,  temp_apr=13.0, humid_jul=72,
                 circ=1100, home=201, curve=1, grade=0.0, right=0, humid_factor=1, seaside=0),
}


def build_venue_features(races_csv: str, start_year: int = 20) -> pd.DataFrame:
    print(f"[venue] reading {races_csv} ...")
    df = pd.read_csv(races_csv, encoding='utf-8-sig', low_memory=False,
                     usecols=['year', 'race_id', 'horse_id', 'umaban', 'course'],
                     dtype={'race_id': str})
    df = df[df['year'] >= start_year].copy()
    print(f"  races: {len(df):,} rows")

    # 各 column を venue table から look up
    out = df[['race_id', 'horse_id', 'umaban']].copy()
    out['venue_lat'] = df['course'].map(lambda c: VENUE_DATA.get(c, {}).get('lat', 35.0))
    out['venue_lon'] = df['course'].map(lambda c: VENUE_DATA.get(c, {}).get('lon', 138.0))
    out['venue_elevation_m'] = df['course'].map(lambda c: VENUE_DATA.get(c, {}).get('elev', 30))
    out['venue_temp_base_apr'] = df['course'].map(lambda c: VENUE_DATA.get(c, {}).get('temp_apr', 12.0))
    out['venue_humidity_jul'] = df['course'].map(lambda c: VENUE_DATA.get(c, {}).get('humid_jul', 75))
    out['track_circumference_m'] = df['course'].map(lambda c: VENUE_DATA.get(c, {}).get('circ', 1700))
    out['track_homestretch_m'] = df['course'].map(lambda c: VENUE_DATA.get(c, {}).get('home', 300))
    out['track_curve_type'] = df['course'].map(lambda c: VENUE_DATA.get(c, {}).get('curve', 2))
    out['track_grade_diff_m'] = df['course'].map(lambda c: VENUE_DATA.get(c, {}).get('grade', 2.0))
    out['is_right_turn'] = df['course'].map(lambda c: VENUE_DATA.get(c, {}).get('right', 0))
    out['venue_humid_factor'] = df['course'].map(lambda c: VENUE_DATA.get(c, {}).get('humid_factor', 1))
    out['is_seaside'] = df['course'].map(lambda c: VENUE_DATA.get(c, {}).get('seaside', 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--races', default=str(BASE / 'data' / 'jra_races_full.csv'))
    ap.add_argument('--output', default=str(BASE / 'data' / 'features_venue.csv'))
    ap.add_argument('--start-year', type=int, default=20)
    args = ap.parse_args()

    out = build_venue_features(args.races, start_year=args.start_year)
    print(f"[venue] writing {args.output} ({len(out):,} rows × {len(out.columns)} cols)")
    out.to_csv(args.output, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
