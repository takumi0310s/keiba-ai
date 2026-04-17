"""当日の買い目を 1頭 - 2頭 - 5頭 フォーメーション形式で表示。

Usage:
    python tools/show_bets.py                  # 今日
    python tools/show_bets.py --date 20260412
    python tools/show_bets.py --course 中山
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(BASE, 'data', 'daily_predictions')


def _parse_trio(s):
    """`4-6-9; 4-9-15; ...` から登場する全馬番セットを返す。"""
    out = []
    for combo in (s or '').split(';'):
        combo = combo.strip()
        if not combo:
            continue
        try:
            out.append(tuple(int(x) for x in combo.split('-')))
        except ValueError:
            continue
    return out


def compute_formation(row, pair_odds=None):
    """予測CSV行から1-2-5フォーメーション情報を返す。

    Parameters:
        row: dict-like (pandas Series も可)
        pair_odds: {(n1, n2): odds} 馬連オッズ (省略可)

    Returns dict:
        bet_type: 'trio' | 'umaren'
        axis: int             # 軸馬番 (top1_num)
        second_col: list[int] # 2列目 (2頭、三連複時)
        third_col: list[int]  # 3列目 (5頭、三連複時)
        umaren_pairs: list[tuple[int, int, float|None]]
                               # [(相手馬番, 金額, オッズ)] 馬連時のみ
        n_points: int         # 合計点数
        investment: int       # 合計金額

    馬連の金額配分:
        - pair_odds が与えられ両ペアのオッズ取得済み:
            オッズ高い側 = 400円 / 低い側 = 300円 (配当期待大に厚張り)
        - オッズ未取得フォールバック: TOP2=400, TOP3=300
    """
    bet_type = (row.get('bet_type') or '').lower()
    try:
        top1 = int(row.get('top1_num') or 0)
        top2 = int(row.get('top2_num') or 0)
        top3 = int(row.get('top3_num') or 0)
        nh = int(row.get('num_horses') or 0)
        investment = int(row.get('investment') or 0)
    except (ValueError, TypeError):
        return None

    second_col = sorted({top2, top3} - {top1})

    if bet_type == 'umaren':
        key_t2 = tuple(sorted([top1, top2])) if top2 and top2 != top1 else None
        key_t3 = tuple(sorted([top1, top3])) if top3 and top3 != top1 else None
        odds_t2 = (pair_odds or {}).get(key_t2) if key_t2 else None
        odds_t3 = (pair_odds or {}).get(key_t3) if key_t3 else None

        # オッズ連動: 高い方=400円, 低い方=300円
        if odds_t2 and odds_t3 and odds_t2 > 0 and odds_t3 > 0:
            if odds_t2 >= odds_t3:
                amt_t2, amt_t3 = 400, 300
            else:
                amt_t2, amt_t3 = 300, 400
        else:
            # フォールバック: TOP2=400, TOP3=300
            amt_t2, amt_t3 = 400, 300

        pairs = []
        if top2 and top2 != top1:
            pairs.append((top2, amt_t2, odds_t2))
        if top3 and top3 != top1:
            pairs.append((top3, amt_t3, odds_t3))

        return {
            'bet_type': 'umaren',
            'axis': top1,
            'second_col': second_col,
            'third_col': [],
            'umaren_pairs': pairs,
            'n_points': len(pairs),
            'investment': investment,
        }

    # trio: 1-2-5フォーメーション
    trio_combos = _parse_trio(row.get('trio_bets', ''))
    all_nums = set()
    for c in trio_combos:
        all_nums.update(c)
    third_col = sorted(n for n in all_nums if n != top1)
    # 必ず5頭にする（足りなければtop2/top3, 過剰なら先頭5）
    while len(third_col) < 5 and len(third_col) < (nh if nh else 18):
        for n in (top2, top3):
            if n and n != top1 and n not in third_col:
                third_col.append(n)
                break
        else:
            break
    third_col = sorted(set(third_col))[:5]

    return {
        'bet_type': 'trio',
        'axis': top1,
        'second_col': second_col,
        'third_col': third_col,
        'umaren_pairs': [],
        'n_points': len(trio_combos),
        'investment': investment,
    }


def show_row(row):
    rid = row.get('race_id', '')
    course = row.get('course', '')
    rno = row.get('race_num', '')
    rname = row.get('race_name', '')
    cond = row.get('condition', '')
    nh = row.get('num_horses', '')
    dist = row.get('distance', '')
    surf = row.get('surface', '')
    tc = row.get('track_condition', '')

    form = compute_formation(row)
    if form is None:
        return

    bet_type = form['bet_type']
    top1 = form['axis']
    second_col = form['second_col']
    third_col = form['third_col']
    trio_combos = _parse_trio(row.get('trio_bets', ''))

    print(f"\n=== {course}{rno}R {rname} ({cond} {surf}{dist}m {tc} {nh}頭) ===")
    print(f"  race_id={rid}  券種={bet_type}")
    if bet_type == 'umaren':
        for partner, amt, odds in form['umaren_pairs']:
            odds_txt = f" (オッズ{odds:.1f}倍)" if odds else ''
            print(f"  馬連 {top1}-{partner}{odds_txt}: {amt}円")
        print(f"  合計: {form['investment']}円")
        return
    print(f"  三連複フォーメーション 1-2-5")
    print(f"    1列目(軸): {top1}")
    print(f"    2列目: {', '.join(map(str, second_col))}")
    print(f"    3列目: {', '.join(map(str, third_col))}  (※軸を除外、5頭固定)")
    print(f"    点数: {form['n_points']}点 × 100円 = {form['investment']}円")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=datetime.now().strftime('%Y%m%d'))
    ap.add_argument('--course', default=None, help='会場でフィルタ（例: 中山）')
    args = ap.parse_args()

    path = os.path.join(PRED_DIR, f'{args.date}.csv')
    if not os.path.exists(path):
        print(f'[ERROR] 予測ファイルなし: {path}')
        sys.exit(1)

    with open(path, 'r', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        rows = list(rd)

    if args.course:
        rows = [r for r in rows if r.get('course') == args.course]

    print(f"=== {args.date} 買い目一覧 ({len(rows)}レース) ===")
    for row in rows:
        show_row(row)


if __name__ == '__main__':
    main()
