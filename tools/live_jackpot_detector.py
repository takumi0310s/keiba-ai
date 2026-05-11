#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Live Jackpot pattern detector: 当日 race で 4-way pattern 該当馬を識別.

5/17 開催で V15 通知に **加えて** Jackpot pattern 該当馬を 別 alert 通知。
V15 production は完全 不変、 補完 information のみ提供。

【4-way Jackpot pattern】
- class_down = 1 (前走から降級)
- horse_recent5_top3 >= 0.6 (直近 5 走 top3 率 60%+)
- jockey_recent30_top3 >= 0.30 (騎手 直近 30 走 top3 率 30%+)
- jockey_change = 0 (前走と同 騎手)

→ 実証 top3 rate **64.8%**、 単勝 ROI **184%**

【V15 投資保護】 V15 通知に追加 / 上書きせず、 別 alert のみ

Usage:
    python tools/live_jackpot_detector.py 20260517
    python tools/live_jackpot_detector.py 20260411 --verbose
"""
import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fetch_horse_data_for_race(race_id, cookies):
    """1 race の shutuba から 馬情報 取得 (horse_id, jockey_id, 前走 info)."""
    import requests
    url = f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}'
    try:
        r = requests.get(url, cookies=cookies, timeout=15,
                          headers={'User-Agent': 'Mozilla/5.0'})
        r.encoding = 'euc-jp'
        return r.text
    except Exception:
        return None


def check_jackpot(horse_features):
    """horse_features dict が 4-way pattern 満たすかチェック."""
    return (
        horse_features.get('class_down') == 1 and
        horse_features.get('horse_recent5_top3', 0) >= 0.6 and
        horse_features.get('jockey_recent30_top3', 0) >= 0.30 and
        horse_features.get('jockey_change') == 0
    )


def main():
    ap = argparse.ArgumentParser(description='Live Jackpot pattern detector')
    ap.add_argument('date', help='YYYYMMDD')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    import pandas as pd

    # daily_predictions 読み込み (V15 出力)
    daily_path = os.path.join(BASE_DIR, 'data', 'daily_predictions', f'{args.date}.csv')
    if not os.path.exists(daily_path):
        print(f'[INFO] daily_predictions/{args.date}.csv 未生成')
        print('     → 当日 daily_predict.py 実行後に jackpot detector 実行')
        return 1

    df_pred = pd.read_csv(daily_path, encoding='utf-8-sig')
    race_ids = df_pred['race_id'].astype(str).unique().tolist()
    print(f'[INFO] {len(race_ids)} races for {args.date}')

    # 既存 features 読み込み
    hs_path = os.path.join(BASE_DIR, 'data', 'hot_streak_features.csv')
    ev_path = os.path.join(BASE_DIR, 'data', 'event_effect_features.csv')

    if not all(os.path.exists(p) for p in [hs_path, ev_path]):
        print('[ERROR] hot_streak / event_effect features 未生成')
        return 1

    hs = pd.read_csv(hs_path, encoding='utf-8', dtype={'race_id': str, 'horse_id': str})
    ev = pd.read_csv(ev_path, encoding='utf-8', dtype={'race_id': str, 'horse_id': str})

    # 既存 race_id で merge (当該開催 race_id は features 未含 → past data から approx)
    # 実運用では 当該 race 朝までの hot_streak を 再計算する必要
    # → 今夜は demo として 既存 features の data で過去 race_id で check
    print('\n=== Jackpot pattern 該当 馬 ===')
    print('(features は 既存 csv ベース、 当該 race_id の features が無い場合 SKIP)')

    found_count = 0
    for race_id in race_ids:
        hs_race = hs[hs['race_id'] == race_id]
        ev_race = ev[ev['race_id'] == race_id]
        if hs_race.empty or ev_race.empty:
            continue
        merged = hs_race.merge(ev_race, on=['race_id', 'horse_id'], how='inner', suffixes=('', '_d'))
        for _, row in merged.iterrows():
            feats = {
                'class_down': row.get('class_down', 0),
                'horse_recent5_top3': row.get('horse_recent5_top3', 0),
                'jockey_recent30_top3': row.get('jockey_recent30_top3', 0),
                'jockey_change': row.get('jockey_change', 1),
            }
            if check_jackpot(feats):
                found_count += 1
                print(f'  ★ JACKPOT: race {race_id} horse_id {row["horse_id"]}')
                if args.verbose:
                    for k, v in feats.items():
                        print(f'      {k}: {v}')

    print(f'\n[SUMMARY] Jackpot 該当 馬: {found_count} 件')

    if found_count == 0:
        print('  ※ 当日 race_id の features 未生成、 or 該当 馬 該当 なし')
        print('  ※ 実運用には 朝 V15 通知 と同時に features を 再生成 する仕組みが必要')

    # 推奨 action
    print('\n[5/17 開催 推奨 action]')
    print('1. V15 daily_predict で 通常 通知 (戦略⑦ 適用)')
    print('2. 当日 朝 hot_streak / event 系 features を 再生成')
    print('3. Jackpot 該当馬 が出たら 別 alert (700 円 → 1500 円 増額 検討)')
    print('   ※ ただし V15 投資保護下 / 統合は 5/24+ 慎重判定')

    return 0


if __name__ == '__main__':
    sys.exit(main())
