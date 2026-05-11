#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Race ID mapping utility (TFJV ↔ netkeiba).

jra_races_full.csv の race_id は TFJV 形式 (10 digit、 例: 0615110101)、
daily_predictions / paddock などは netkeiba 形式 (12 digit、 例: 202605020611)。
両 format を 相互変換 + mapping table 生成。

【TFJV format 解析 (推定)】
- 0/1 (prefix)
- C (course code、 6=中山、 8=東京 等)
- YY (年 下 2 digit)
- KK (開催回)
- DD (日)
- RR (race番号)

【netkeiba format】
YYYY (年) + CC (course) + KK (開催) + DD (日) + RR (race番号)

【V15 投資保護】 mapping utility のみ

Usage:
    python tools/race_id_mapper.py --build-mapping  # 全 race の mapping table 生成
    python tools/race_id_mapper.py --tfjv 0726110101  # → netkeiba 推定
    python tools/race_id_mapper.py --netkeiba 202605020611  # → TFJV 推定
"""
import argparse
import os
import sys

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_mapping():
    """jra_races_full.csv から TFJV ↔ netkeiba mapping 生成 (推定)."""
    import pandas as pd
    base = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    df = pd.read_csv(base, encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'year', 'month', 'day', 'kai', 'nichi',
                               'race_num', 'course'])
    df['race_id_str'] = df['race_id'].astype(str)
    df = df[df['year'] >= 22]  # 直近データのみ

    # course → netkeiba course code mapping (推定)
    # 一般的: 札幌=01, 函館=02, 福島=03, 新潟=04, 東京=05, 中山=06, 中京=07, 京都=08, 阪神=09, 小倉=10
    course_map = {
        '札幌': '01', '函館': '02', '福島': '03', '新潟': '04', '東京': '05',
        '中山': '06', '中京': '07', '京都': '08', '阪神': '09', '小倉': '10',
    }

    # netkeiba race_id 推定: YYYY + CC + KK + DD + RR
    def to_netkeiba(row):
        year_full = 2000 + int(row['year'])
        course_cd = course_map.get(row['course'])
        if not course_cd:
            return None
        kai = str(int(row['kai'])).zfill(2)
        nichi = str(int(row['nichi'])).zfill(2)
        rno = str(int(row['race_num'])).zfill(2)
        return f'{year_full}{course_cd}{kai}{nichi}{rno}'

    df['netkeiba_race_id'] = df.apply(to_netkeiba, axis=1)
    df = df.dropna(subset=['netkeiba_race_id']).drop_duplicates('race_id_str')

    out_path = os.path.join(BASE_DIR, 'data', 'race_id_mapping.csv')
    out = df[['race_id_str', 'netkeiba_race_id', 'year', 'month', 'day',
               'course', 'race_num']].copy()
    out.columns = ['tfjv_race_id', 'netkeiba_race_id', 'year', 'month',
                    'day', 'course', 'race_num']
    out.to_csv(out_path, index=False, encoding='utf-8')
    print(f'[OK] mapping saved: {out_path}')
    print(f'[OK] {len(out):,} unique race mappings')
    print(f'\nsample:')
    print(out.head(10).to_string())
    return 0


def lookup_tfjv(tfjv_id):
    import pandas as pd
    path = os.path.join(BASE_DIR, 'data', 'race_id_mapping.csv')
    if not os.path.exists(path):
        print('[ERROR] mapping table 未生成、 --build-mapping を 先に実行')
        return 1
    df = pd.read_csv(path, encoding='utf-8',
                      dtype={'tfjv_race_id': str, 'netkeiba_race_id': str})
    match = df[df['tfjv_race_id'] == tfjv_id]
    if match.empty:
        print(f'[NOT FOUND] tfjv {tfjv_id}')
        return 1
    print(match.to_string())
    return 0


def lookup_netkeiba(nk_id):
    import pandas as pd
    path = os.path.join(BASE_DIR, 'data', 'race_id_mapping.csv')
    if not os.path.exists(path):
        print('[ERROR] mapping table 未生成、 --build-mapping を 先に実行')
        return 1
    df = pd.read_csv(path, encoding='utf-8',
                      dtype={'tfjv_race_id': str, 'netkeiba_race_id': str})
    match = df[df['netkeiba_race_id'] == nk_id]
    if match.empty:
        print(f'[NOT FOUND] netkeiba {nk_id}')
        return 1
    print(match.to_string())
    return 0


def main():
    ap = argparse.ArgumentParser(description='Race ID mapper')
    ap.add_argument('--build-mapping', dest='build', action='store_true')
    ap.add_argument('--tfjv', help='TFJV race_id → netkeiba')
    ap.add_argument('--netkeiba', help='netkeiba race_id → TFJV')
    args = ap.parse_args()

    if args.build: return build_mapping()
    if args.tfjv: return lookup_tfjv(args.tfjv)
    if args.netkeiba: return lookup_netkeiba(args.netkeiba)
    ap.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
