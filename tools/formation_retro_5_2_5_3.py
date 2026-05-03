"""5/2-5/3 daily_predictions の trio_bets formation を拡張シミュレーション.

現在の formation (7点):
  Top1-Top2-Top3, Top1-Top2-Top4, Top1-Top2-Top5, Top1-Top2-Top6
  Top1-Top3-Top4, Top1-Top3-Top5, Top1-Top3-Top6

候補 formation (拡張):
  V15 baseline (7点)
  +T14: 上記 + Top1-Top4-Top5 (8点)
  +T14_T15_T16: 上記 + Top1-Top4-Top5, Top1-Top4-Top6, Top1-Top5-Top6 (10点)
  +T2-Top4-5_etc: Top2-Top3-Top4 series 追加 (軸不変パターンを増やす)
  Box6: Top1-6 box (20点)
  Box5: Top1-5 box (10点)
"""
import sys, os, io, itertools
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE_DIR)

DATES = ['20260425', '20260426', '20260502', '20260503']  # healthy 4日


def parse_trio_str(s):
    if not isinstance(s, str): return None
    try:
        return tuple(sorted([int(x) for x in s.split('-') if x.strip()]))
    except: return None


def parse_bets(s):
    if not isinstance(s, str): return []
    out = []
    for b in s.split(';'):
        try:
            t = tuple(sorted([int(x) for x in b.strip().split('-') if x.strip()]))
            if len(t) == 3:
                out.append(t)
        except: pass
    return out


def reconstruct_top16(top1, top2, top3, bets):
    """既存 7点 trio_bets から Top4-6 を逆算"""
    counter = {}
    for tb in bets:
        for n in tb:
            counter[n] = counter.get(n, 0) + 1
    rest = sorted([n for n in counter if n not in (top1, top2, top3)], key=lambda x: -counter.get(x, 0))
    while len(rest) < 3: rest.append(0)
    return [top1, top2, top3] + rest[:3]


# --- Formation generators ---

def fmt_v15_baseline(top16):
    """V15 既存 7点"""
    n1,n2,n3 = top16[0:3]
    bets = []
    for x in top16[2:6]:
        if x and x not in (n1,n2): bets.append(tuple(sorted([n1,n2,x])))
    for x in top16[3:6]:
        if x and x not in (n1,n3): bets.append(tuple(sorted([n1,n3,x])))
    return list(set(bets))


def fmt_plus_top45(top16):
    """V15 + Top1-Top4-Top5 (1点追加, 8点)"""
    base = fmt_v15_baseline(top16)
    n1, t4, t5 = top16[0], top16[3], top16[4]
    if t4 and t5 and t4 != n1 and t5 != n1:
        base.append(tuple(sorted([n1, t4, t5])))
    return list(set(base))


def fmt_plus_t14_t15_t16(top16):
    """V15 + Top1-Top4-Top5, Top1-Top4-Top6, Top1-Top5-Top6 (3点追加, 10点)"""
    base = fmt_v15_baseline(top16)
    n1, t4, t5, t6 = top16[0], top16[3], top16[4], top16[5]
    for pair in [(t4,t5), (t4,t6), (t5,t6)]:
        a,b = pair
        if a and b and a != n1 and b != n1 and a != b:
            base.append(tuple(sorted([n1, a, b])))
    return list(set(base))


def fmt_box5(top16):
    """Top1-5 BOX (10点)"""
    top5 = [n for n in top16[:5] if n]
    return list(set(tuple(sorted(c)) for c in itertools.combinations(top5, 3)))


def fmt_box6(top16):
    """Top1-6 BOX (20点)"""
    top6 = [n for n in top16[:6] if n]
    return list(set(tuple(sorted(c)) for c in itertools.combinations(top6, 3)))


def fmt_no_axis_t2_t3_t4(top16):
    """V15 + Top2-Top3-Top4 (軸非依存追加 = 8点)"""
    base = fmt_v15_baseline(top16)
    t2,t3,t4 = top16[1], top16[2], top16[3]
    if t2 and t3 and t4 and len({t2,t3,t4}) == 3:
        base.append(tuple(sorted([t2,t3,t4])))
    return list(set(base))


def fmt_2axis(top16):
    """軸2頭 (Top1, Top2) - 流し (3頭目: Top3-6) = 4点"""
    n1,n2 = top16[0], top16[1]
    base = []
    for x in top16[2:6]:
        if x and x not in (n1,n2):
            base.append(tuple(sorted([n1,n2,x])))
    return list(set(base))


def fmt_axis1_t2_5_t3_5(top16):
    """V15 拡張: Top1-Top2-{T3,4,5,6} + Top1-Top3-{T4,5,6} + Top1-Top4-{T5,6} = 9点"""
    base = fmt_plus_top45(top16)
    n1, t4, t6 = top16[0], top16[3], top16[5]
    if t4 and t6 and t4 != n1 and t6 != n1:
        base.append(tuple(sorted([n1, t4, t6])))
    return list(set(base))


FORMATIONS = {
    'V15_baseline_7': (fmt_v15_baseline, 100),  # name, fn, bet_per_point
    'V15+T1-T4-T5_8': (fmt_plus_top45, 100),
    'V15+T14_T15_T16_10': (fmt_plus_t14_t15_t16, 100),
    'V15+T1-T4-T5+T1-T4-T6_9': (fmt_axis1_t2_5_t3_5, 100),
    'V15+T2-T3-T4_8': (fmt_no_axis_t2_t3_t4, 100),
    'Box5_10': (fmt_box5, 100),
    'Box6_20': (fmt_box6, 100),
    '2axis_4': (fmt_2axis, 175),  # 4点で 700円固定なら per_pt 175
}


def main():
    all_rows = []
    for d in DATES:
        pred = pd.read_csv(f'data/daily_predictions/{d}.csv', dtype={'race_id':str})
        pred = pred.drop_duplicates('race_id', keep='last')
        res = pd.read_csv(f'data/daily_results/{d}.csv', encoding='utf-8-sig', dtype={'race_id':str})
        res = res.drop_duplicates('race_id', keep='last')
        res = res[res['status']=='settled']

        for _, r in res.iterrows():
            rid = r['race_id']
            p = pred[pred['race_id']==rid]
            if len(p)==0: continue
            pp=p.iloc[0]
            try:
                top1 = int(pp['top1_num']); top2=int(pp['top2_num']); top3=int(pp['top3_num'])
            except: continue
            bets_v15 = parse_bets(pp.get('trio_bets',''))
            top16 = reconstruct_top16(top1, top2, top3, bets_v15)
            wt = parse_trio_str(r.get('trio_result',''))
            trio_pay = float(r.get('trio_payout', 0) or 0)
            invest_orig = int(r.get('investment', 0) or 0)
            race_num = pd.to_numeric(r.get('race_num', 0), errors='coerce')

            for fn_name, (fn, bet_per_pt) in FORMATIONS.items():
                bets_new = fn(top16)
                n_bets = len(bets_new)
                inv = n_bets * bet_per_pt
                # hit?
                hit = 1 if (wt and wt in bets_new) else 0
                pay = (trio_pay / 100.0) * bet_per_pt if hit else 0
                all_rows.append({
                    'date': d, 'race_id': rid, 'race_num': int(race_num) if pd.notna(race_num) else 0,
                    'formation': fn_name, 'n_bets': n_bets, 'inv': inv, 'pay': pay, 'hit': hit,
                })

    df = pd.DataFrame(all_rows)
    print('=== Formation 比較 (healthy 4日, settled races) ===')
    print(f'{"formation":40s}  n_R   tot_inv    tot_pay    profit     ROI    hit_rate')
    summ = []
    for fn_name in FORMATIONS:
        sub = df[df['formation']==fn_name]
        n_r = len(sub)
        inv = sub['inv'].sum(); pay = sub['pay'].sum()
        profit = pay - inv
        roi = pay/inv*100 if inv else 0
        hits = sub['hit'].sum()
        hit_rate = hits/n_r*100 if n_r else 0
        avg_n_bets = sub['n_bets'].mean()
        print(f'{fn_name:40s}  {n_r:3d}  {int(inv):8,d}  {int(pay):9,d}  {int(profit):+8,d}  {roi:6.1f}%  {hit_rate:5.1f}% (avg {avg_n_bets:.1f}点)')
        summ.append({'fn': fn_name, 'n_r': n_r, 'inv': int(inv), 'pay': int(pay),
                      'profit': int(profit), 'roi': float(roi), 'hits': int(hits),
                      'hit_rate': float(hit_rate), 'avg_n_bets': float(avg_n_bets)})

    print('\n=== 11R/12R のみ (主要レース) ===')
    df_1112 = df[df['race_num'].isin([11,12])]
    for fn_name in FORMATIONS:
        sub = df_1112[df_1112['formation']==fn_name]
        n_r = len(sub)
        inv = sub['inv'].sum(); pay = sub['pay'].sum()
        profit = pay - inv
        roi = pay/inv*100 if inv else 0
        hits = sub['hit'].sum()
        hit_rate = hits/n_r*100 if n_r else 0
        print(f'{fn_name:40s}  {n_r:3d}  {int(inv):8,d}  {int(pay):9,d}  {int(profit):+8,d}  {roi:6.1f}%  {hit_rate:5.1f}%')

    # 5/9 推奨フォーメーション選定: ROI 最大 かつ point 数妥当
    df.to_csv('data/v18/formation_retro_5_2_5_3.csv', index=False, encoding='utf-8-sig')
    print('\nSaved data/v18/formation_retro_5_2_5_3.csv')

    # save summary
    pd.DataFrame(summ).to_csv('data/v18/formation_retro_summary.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
