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
    bet_type = (row.get('bet_type') or '').lower()

    try:
        top1 = int(row.get('top1_num') or 0)
        top2 = int(row.get('top2_num') or 0)
        top3 = int(row.get('top3_num') or 0)
    except ValueError:
        return

    trio_combos = _parse_trio(row.get('trio_bets', ''))
    all_nums = set()
    for c in trio_combos:
        all_nums.update(c)
    third_col = sorted(n for n in all_nums if n != top1)
    # 必ず5頭にする（足りなければtop2/top3, 過剰なら先頭5）
    while len(third_col) < 5 and len(third_col) < (int(nh) if nh else 18):
        for n in (top2, top3):
            if n and n != top1 and n not in third_col:
                third_col.append(n)
                break
        else:
            break
    third_col = sorted(set(third_col))[:5]

    second_col = sorted({top2, top3} - {top1})

    print(f"\n=== {course}{rno}R {rname} ({cond} {surf}{dist}m {tc} {nh}頭) ===")
    print(f"  race_id={rid}  券種={bet_type}")
    if bet_type == 'umaren':
        print(f"  馬連: 軸 {top1} - 相手 {', '.join(map(str, second_col))}")
        return
    print(f"  三連複フォーメーション 1-2-5")
    print(f"    1列目(軸): {top1}")
    print(f"    2列目: {', '.join(map(str, second_col))}")
    print(f"    3列目: {', '.join(map(str, third_col))}  (※軸を除外、5頭固定)")
    n_pts = len(second_col) * len([x for x in third_col if x not in second_col]) + \
            (len(second_col) * (len(second_col) - 1) // 2 if all(s in third_col for s in second_col) else 0)
    print(f"    点数(参考): trio_bets列={len(trio_combos)}点")


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
