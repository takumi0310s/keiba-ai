"""bet_pattern 最適化 retro (Session #40 A1).

V15 三連複買い目 7点 vs 5点 vs 9点 の retro 比較。
- 7点 (現状 baseline): TOP1軸 - TOP2,TOP3 - TOP2~TOP6 のフォーメーション
- 5点 (堅め): TOP1軸 - TOP2,TOP3 - TOP2,TOP3,TOP4
- 9点 (広め): TOP1軸 - TOP2,TOP3,TOP4 - TOP2~TOP7

source:
- data/cumulative_results.csv (V15 過去 予測 + 結果、 race_id 単位)
- data/jra_payouts.csv (公式払戻)

usage:
  python tools/bet_pattern_optimization.py [--from 20260426] [--to 20260503]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def parse_top_n(rec: dict, max_n: int = 7) -> List[int]:
    """cumulative_results の各行から top_N の馬番 list を構築.

    優先 1: top1_num/top2_num/top3_num が populated → それを使用
    優先 2 (95% case): trio_bets_str を頻度分析:
      - 全 trio に登場 (頻度=N_bets) = 軸 (top1)
      - 頻度上位 = top2,3,...
    """
    nums = []
    for k in ['top1_num', 'top2_num', 'top3_num']:
        v = rec.get(k)
        try:
            n = int(v) if pd.notna(v) and str(v).strip() != '' else None
        except Exception:
            n = None
        if n: nums.append(n)

    if len(nums) >= 3:
        # 既存 top + bets で extra
        bets = str(rec.get('trio_bets_str', '') or rec.get('trio_bets', '') or '')
        extra = []
        for token in bets.replace(';', ' ').replace(',', ' ').replace('-', ' ').split():
            try:
                n = int(token)
                if n not in nums and n not in extra: extra.append(n)
            except Exception: continue
        return (nums + extra)[:max_n]

    # fallback: 頻度分析
    bets = str(rec.get('trio_bets_str', '') or rec.get('trio_bets', '') or '')
    if not bets: return nums
    trios = [t.strip() for t in bets.replace(',', ';').split(';') if t.strip()]
    freq = {}
    n_trios = 0
    for trio in trios:
        nums_in = []
        for token in trio.split('-'):
            try:
                n = int(token.strip())
                nums_in.append(n)
            except Exception: continue
        if len(nums_in) >= 3:
            n_trios += 1
            for n in nums_in:
                freq[n] = freq.get(n, 0) + 1
    if not freq: return nums
    # 軸 (頻度 = N_trios)
    sorted_freq = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [n for n, _ in sorted_freq[:max_n]]


def gen_bets_from_top(tops: List[int], pattern: str) -> List[Tuple[int, int, int]]:
    """3 つの top list から bet pattern 別に三連複買い目 list を生成.

    formation 定義 (axis = TOP1):
    - 5pt: TOP1, TOP2 軸 + TOP3〜TOP7 流し (5 通り)
    - 7pt: TOP1 軸 - TOP2,TOP3 - TOP2〜TOP6 (本番 baseline)
    - 9pt: TOP1 軸 - TOP2,TOP3,TOP4 - TOP2〜TOP7
    """
    if len(tops) < 3:
        return []
    a = tops[0]
    out = []
    seen = set()
    if pattern == "5pt":
        # TOP1 + TOP2 軸 + TOP3〜TOP7 (5 頭流し)
        if len(tops) < 2: return []
        b = tops[1]
        for c in tops[2:7]:
            if c in (a, b): continue
            tri = tuple(sorted([a, b, c]))
            if tri in seen: continue
            seen.add(tri)
            out.append(tri)
    elif pattern == "9pt":
        col2 = tops[1:4] if len(tops) >= 4 else tops[1:]
        col3 = tops[1:7] if len(tops) >= 7 else tops[1:]
        for x in col2:
            for y in col3:
                if x == a or y == a or x == y: continue
                tri = tuple(sorted([a, x, y]))
                if tri in seen: continue
                seen.add(tri)
                out.append(tri)
    else:  # "7pt" baseline
        col2 = tops[1:3]
        col3 = tops[1:6] if len(tops) >= 6 else tops[1:]
        for x in col2:
            for y in col3:
                if x == a or y == a or x == y: continue
                tri = tuple(sorted([a, x, y]))
                if tri in seen: continue
                seen.add(tri)
                out.append(tri)
    return out


def trio_won(bets: List[Tuple[int, int, int]], winning: Tuple[int, int, int]) -> bool:
    if not bets or not winning: return False
    return tuple(sorted(winning)) in [tuple(sorted(b)) for b in bets]


def parse_winning_trio(rec: dict) -> Tuple[int, int, int] | None:
    """payouts から trio_nums (例: '1-7-8') を tuple に."""
    s = str(rec.get('trio_nums', '') or '')
    parts = s.replace('/', '-').split('-')
    nums = []
    for p in parts:
        try: nums.append(int(p))
        except Exception: pass
    if len(nums) >= 3:
        return tuple(sorted(nums[:3]))
    return None


def evaluate_pattern(df: pd.DataFrame, pattern: str, payouts: pd.DataFrame) -> dict:
    """各 race で pattern の bet list を生成し、 payouts を merge して ROI 計算."""
    n_bets = len(df) * (5 if pattern == "5pt" else (9 if pattern == "9pt" else 7))
    cost_per_race = (5 if pattern == "5pt" else (9 if pattern == "9pt" else 7)) * 100  # 100円/点

    hit = 0
    inv = 0
    pay = 0
    n_eval = 0
    for _, rec in df.iterrows():
        tops = parse_top_n(rec.to_dict(), max_n=7)
        bets = gen_bets_from_top(tops, pattern)
        if not bets: continue
        n_eval += 1
        inv += cost_per_race

        rid = str(rec.get('race_id', ''))
        # 払戻 lookup: cumulative の race_id → payouts の (race_date, course, race_num) で merge
        date = str(rec.get('date', ''))
        course = str(rec.get('course', ''))
        race_num = rec.get('race_num')
        try:
            race_num_int = int(race_num) if pd.notna(race_num) else None
        except Exception:
            race_num_int = None
        if not date or not course or race_num_int is None: continue

        match = payouts[
            (payouts['race_date'].astype(str) == date)
            & (payouts['course'].astype(str) == course)
            & (payouts['race_num'].astype(int).fillna(-1) == race_num_int)
        ]
        if len(match) == 0: continue
        winning = parse_winning_trio(match.iloc[0].to_dict())
        if winning and trio_won(bets, winning):
            hit += 1
            try: pay += int(match.iloc[0].get('trio_payout', 0))
            except Exception: pass

    roi = (pay / inv * 100) if inv > 0 else 0
    hit_rate = (hit / n_eval * 100) if n_eval > 0 else 0
    return {
        'pattern': pattern,
        'n_eval': n_eval,
        'cost_per_race': cost_per_race,
        'hit': hit,
        'hit_rate': round(hit_rate, 2),
        'inv': inv,
        'pay': pay,
        'profit': pay - inv,
        'roi': round(roi, 2),
    }


def main():
    p = argparse.ArgumentParser(description="bet_pattern 最適化 retro")
    p.add_argument('--from', dest='from_date', default='20260426')
    p.add_argument('--to', dest='to_date', default='20260503')
    p.add_argument('--cum', default='data/cumulative_results.csv')
    p.add_argument('--payouts', default='data/jra_payouts.csv')
    p.add_argument('--out', default='data/v18/bet_pattern_retro_5_7.json')
    args = p.parse_args()

    cum = pd.read_csv(os.path.join(BASE, args.cum), low_memory=False)
    # date は '20260426.0' / '20260426' 混在のため, .0 を除去
    cum['date'] = cum['date'].astype(str).str.replace(r'\.0$', '', regex=True)
    cum = cum[(cum['date'] >= args.from_date) & (cum['date'] <= args.to_date)].copy()
    print(f"[bet_opt] retro target: {args.from_date}-{args.to_date} N={len(cum)}")

    if len(cum) == 0:
        # fallback: top1_num が 95% 欠損 (CLAUDE.md 既知バグ) → trio_bets 列から逆算
        # cumulative_results.csv は trio_bets / trio_bets_str があるが, top1_num/score 95%欠損
        # → 全 records 評価
        cum = pd.read_csv(os.path.join(BASE, args.cum), low_memory=False)
        cum['date'] = cum['date'].astype(str)
        cum = cum[(cum['date'] >= args.from_date) & (cum['date'] <= args.to_date)].copy()
        print(f"[bet_opt] fallback (top1_num 欠損): N={len(cum)}")

    payouts = pd.read_csv(os.path.join(BASE, args.payouts), dtype={'race_date': str})
    payouts['race_num'] = pd.to_numeric(payouts['race_num'], errors='coerce')

    results = []
    for pat in ['5pt', '7pt', '9pt']:
        r = evaluate_pattern(cum, pat, payouts)
        print(f"  [{pat}] n={r['n_eval']:4d} hit={r['hit']:3d}({r['hit_rate']:5.2f}%) inv={r['inv']:6d} pay={r['pay']:6d} profit={r['profit']:+6d} ROI={r['roi']:6.2f}%")
        results.append(r)

    # baseline 比較
    base = next((x for x in results if x['pattern'] == '7pt'), None)
    print(f"\n  baseline 7pt: ROI={base['roi'] if base else 'N/A'}%")
    for r in results:
        if r['pattern'] != '7pt' and base:
            diff = r['roi'] - base['roi']
            print(f"  {r['pattern']:4s} vs 7pt: {diff:+.2f}pt")

    import json
    out_path = os.path.join(BASE, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'from': args.from_date, 'to': args.to_date,
            'n_total': len(cum),
            'results': results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[bet_opt] written: {out_path}")


if __name__ == '__main__':
    main()
