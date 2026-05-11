#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""コース別 Jackpot pattern 安定性 検証.

中山 / 東京 / 京都 / 阪神 / 中京 / 福島 / 新潟 / 札幌 / 函館 / 小倉 で 一貫 効くか。
京都 は data 問題 で V15 戦略⑦ 除外中、 Jackpot pattern では改善あるか確認。

【V15 投資保護】 分析のみ、 V15 model 不変

Usage:
    python tools/course_jackpot_stability.py
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

POP_ODDS_WIN = {1: 3.0, 2: 5.5, 3: 8.0, 4: 11.0, 5: 14.5,
                6: 19.0, 7: 24.0, 8: 29.0, 9: 35.0, 10: 42.0,
                11: 50.0, 12: 60.0, 13: 70.0, 14: 80.0, 15: 95.0,
                16: 110.0, 17: 130.0, 18: 150.0}


def main():
    import pandas as pd

    base = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    df = pd.read_csv(base, encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'horse_id', 'finish', 'year', 'course',
                               'popularity'])
    df = df[df['year'] >= 22]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['win'] = (df['finish'] == 1).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)

    for fname in ['event_effect_features.csv', 'hot_streak_features.csv']:
        path = os.path.join(BASE_DIR, 'data', fname)
        sub = pd.read_csv(path, encoding='utf-8', dtype={'race_id': str, 'horse_id': str})
        df = df.merge(sub, on=['race_id', 'horse_id'], how='left', suffixes=('', '_d'))
        df = df.drop(columns=[c for c in df.columns if c.endswith('_d')])

    df = df.dropna(subset=['horse_recent5_top3', 'jockey_recent30_top3',
                             'class_down', 'jockey_change', 'popularity'])
    df['horse_q5'] = (pd.qcut(df['horse_recent5_top3'].rank(method='first'),
                                 5, labels=[1, 2, 3, 4, 5]) == 5).astype(int)
    df['jockey_q5'] = (pd.qcut(df['jockey_recent30_top3'].rank(method='first'),
                                  5, labels=[1, 2, 3, 4, 5]) == 5).astype(int)

    df['is_jackpot'] = (
        (df['class_down'] == 1) & (df['horse_q5'] == 1) &
        (df['jockey_q5'] == 1) & (df['jockey_change'] == 0)
    ).astype(int)

    df['win_odds'] = df['popularity'].astype(int).map(POP_ODDS_WIN).fillna(50.0)
    df['win_pnl'] = df.apply(lambda r: 100 * (r['win_odds'] - 1) if r['win'] else -100, axis=1)

    print('=== コース別 Jackpot pattern 安定性 ===\n')
    print(f'{"course":<8} {"all_n":>8} {"jpot_n":>7} {"jpot%":>7} {"jpot_top3":>10} {"jpot_win":>10} {"jpot_ROI":>9} {"base_top3":>10} {"base_ROI":>9}')
    print('-' * 100)

    course_stats = []
    for course, sub in df.groupby('course'):
        if len(sub) < 1000:
            continue
        jpot = sub[sub['is_jackpot'] == 1]
        if len(jpot) < 20:
            continue
        jpot_inv = len(jpot) * 100
        base_inv = len(sub) * 100
        jpot_roi = (jpot['win_pnl'].sum() / jpot_inv + 1) * 100
        base_roi = (sub['win_pnl'].sum() / base_inv + 1) * 100
        course_stats.append({
            'course': course, 'n_all': len(sub), 'n_jpot': len(jpot),
            'pct': len(jpot)/len(sub)*100, 'jpot_top3': jpot['top3'].mean(),
            'jpot_win': jpot['win'].mean(), 'jpot_roi': jpot_roi,
            'base_top3': sub['top3'].mean(), 'base_roi': base_roi,
        })

    course_stats.sort(key=lambda x: -x['jpot_roi'])
    for c in course_stats:
        print(f'  {c["course"]:<8} {c["n_all"]:>8,} {c["n_jpot"]:>7,} {c["pct"]:>6.2f}% '
              f'{c["jpot_top3"]:>10.3f} {c["jpot_win"]:>10.3f} '
              f'{c["jpot_roi"]:>8.1f}% {c["base_top3"]:>10.3f} {c["base_roi"]:>8.1f}%')

    print('\n[特記事項]')
    # 京都 注目
    kyoto = [c for c in course_stats if c['course'] == '京都']
    if kyoto:
        k = kyoto[0]
        print(f'  京都: V15 戦略⑦ 除外中 (base ROI {k["base_roi"]:.1f}%)、 '
              f'Jackpot ROI {k["jpot_roi"]:.1f}% で 復活 可能性')

    # 最 ROI / 最 ROI 低 course
    if course_stats:
        best = course_stats[0]
        worst = course_stats[-1]
        print(f'  ベスト: {best["course"]} jpot ROI {best["jpot_roi"]:.1f}%')
        print(f'  ワースト: {worst["course"]} jpot ROI {worst["jpot_roi"]:.1f}%')

    return 0


if __name__ == '__main__':
    sys.exit(main())
