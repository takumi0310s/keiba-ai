"""3/14 全36レース振り返り分析 + 結果CSV再保存"""
import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
import re
import time
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

def get_course_name(race_id):
    code = str(race_id)[4:6]
    return {'06':'中山','07':'中京','09':'阪神'}.get(code, '?')

def fetch_finish_order(race_id):
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    text = resp.content.decode('EUC-JP', errors='replace')
    soup = BeautifulSoup(text, 'html.parser')

    result_table = soup.find("table", class_="RaceTable01")
    if not result_table:
        return {}

    finish_data = {}
    for tr in result_table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        finish_text = tds[0].get_text(strip=True)
        if not finish_text.isdigit():
            continue
        finish = int(finish_text)
        umaban = 0
        for td in tds[1:4]:
            t = td.get_text(strip=True)
            if t.isdigit() and 1 <= int(t) <= 18:
                umaban = int(t)
                break
        if umaban == 0:
            for td in tds:
                cls = " ".join(td.get("class", []))
                if "Umaban" in cls or "Num" in cls:
                    t = td.get_text(strip=True)
                    if t.isdigit() and 1 <= int(t) <= 18:
                        umaban = int(t)
                        break
        if umaban > 0:
            name_tag = tr.select_one("span.Horse_Name a, a[href*='/horse/']")
            name = name_tag.get_text(strip=True)[:8] if name_tag else ''
            finish_data[umaban] = {'finish': finish, 'name': name}

    return finish_data

def main():
    df_pred = pd.read_csv(os.path.join(BASE_DIR, 'data/daily_predictions/20260314.csv'), encoding='utf-8-sig')
    with open(os.path.join(BASE_DIR, 'data/daily_results/20260314_payouts.json'), 'r') as f:
        payouts_json = json.load(f)

    print("Fetching detailed results...")
    all_results = {}
    for _, row in df_pred.iterrows():
        rid = str(row['race_id'])
        all_results[rid] = fetch_finish_order(rid)
        time.sleep(0.5)

    print(f"\n{'='*90}")
    print("  2026/03/14 全36レース AI予測 vs 実績")
    print(f"{'='*90}")

    print(f"\n{'場':>3} {'R':>2} {'条件':>2} {'馬場':>5} {'頭':>2} | {'TOP1(予測1位)':>20} | {'TOP2':>8} | {'TOP3':>8} | {'実三連複':>10} {'配当':>8}")
    print("-" * 105)

    cond_stats = {}
    total_inv = 0
    total_pay = 0
    near_miss_detail = []
    hit_detail = []

    for _, row in df_pred.iterrows():
        rid = str(row['race_id'])
        course = get_course_name(rid)
        rnum = row['race_num']
        cond = row['condition']
        surface = row['surface']
        distance = row['distance']
        num_h = row['num_horses']

        t1n = int(row['top1_num'])
        t2n = int(row['top2_num'])
        t3n = int(row['top3_num'])

        fd = all_results.get(rid, {})
        p = payouts_json.get(rid, {})
        trio_pay = p.get('trio', 0)

        t1f = fd.get(t1n, {}).get('finish', '-')
        t2f = fd.get(t2n, {}).get('finish', '-')
        t3f = fd.get(t3n, {}).get('finish', '-')

        top3_actual = sorted([n for n, d in fd.items() if d['finish'] <= 3])
        trio_bets = row.get('trio_bets', '')
        hit = False
        if trio_bets and len(top3_actual) >= 3:
            actual_set = set(top3_actual[:3])
            for bet in trio_bets.split("; "):
                nums = set(int(n) for n in bet.split("-"))
                if nums == actual_set:
                    hit = True
                    break

        payout = trio_pay if hit else 0
        total_inv += 700
        total_pay += payout

        if cond not in cond_stats:
            cond_stats[cond] = {'races':0,'hits':0,'inv':0,'pay':0,'t1_in3':0,'near':0}
        cond_stats[cond]['races'] += 1
        cond_stats[cond]['inv'] += 700
        cond_stats[cond]['pay'] += payout
        if hit:
            cond_stats[cond]['hits'] += 1
            hit_detail.append({
                'course': course, 'rnum': rnum, 'cond': cond,
                'trio': '-'.join(str(n) for n in top3_actual[:3]),
                'payout': trio_pay, 'score': row['top1_score'],
            })
        if isinstance(t1f, int) and t1f <= 3:
            cond_stats[cond]['t1_in3'] += 1

        near = []
        fourth_horse = ''
        for label, num, finish in [('T1', t1n, t1f), ('T2', t2n, t2f), ('T3', t3n, t3f)]:
            if isinstance(finish, int) and finish == 4:
                near.append(label)
                # Get name of 3rd place horse to compare
                third_num = [n for n, d in fd.items() if d['finish'] == 3]
                third_name = fd.get(third_num[0], {}).get('name', '?') if third_num else '?'
                fourth_horse = fd.get(num, {}).get('name', f'{num}番')
        if near:
            cond_stats[cond]['near'] += 1
            near_miss_detail.append({
                'course': course, 'rnum': rnum, 'cond': cond,
                'near': near, 'trio_pay': trio_pay,
                'surface': surface, 'distance': distance,
                'fourth_horse': fourth_horse,
            })

        mark = " HIT!" if hit else (" *4着" if near else "")
        trio_str = '-'.join(str(n) for n in top3_actual[:3]) if len(top3_actual) >= 3 else '?'
        t1_display = f"{t1n:>2}番 {str(row['top1_name'])[:5]:<5} {t1f:>2}着"
        print(f"{course:>3} {rnum:>2} {cond:>2} {surface}{distance:>4}m {num_h:>2} | {t1_display} | {t2n:>2}番{t2f:>3}着 | {t3n:>2}番{t3f:>3}着 | {trio_str:>10} {trio_pay:>7,}円{mark}")

    print(f"\n{'='*90}")
    print(f"  全体: {sum(s['hits'] for s in cond_stats.values())}/{sum(s['races'] for s in cond_stats.values())}的中")
    print(f"  投資: {total_inv:,}円  払戻: {total_pay:,}円  収支: {total_pay-total_inv:+,}円  ROI: {total_pay/total_inv*100:.1f}%")

    print(f"\n## 条件別サマリー + バックテスト比較")
    bt_roi = {'A': 205.3, 'C': 285.6, 'D': 136.0}
    bt_hit = {'A': 44.5, 'C': 33.7, 'D': 27.0}
    print(f"{'条件':>4} {'N':>3} {'的中':>3} {'的中率':>6} {'ROI':>6} {'BT ROI':>7} {'BT的中':>6} {'TOP1入線':>8} {'4着惜':>5}")
    print("-" * 65)
    for c in ['A','B','C','D','E','X']:
        s = cond_stats.get(c)
        if not s: continue
        roi = s['pay']/s['inv']*100 if s['inv'] > 0 else 0
        hr = s['hits']/s['races']*100 if s['races'] > 0 else 0
        br = bt_roi.get(c, '-')
        bh = bt_hit.get(c, '-')
        print(f"  {c:>2} {s['races']:>3} {s['hits']:>3} {hr:>5.1f}% {roi:>5.1f}% {str(br):>6}% {str(bh):>5}% {s['t1_in3']:>2}/{s['races']:<5} {s['near']:>4}")

    print(f"\n## 的中レース詳細")
    for h in hit_detail:
        print(f"  {h['course']} {h['rnum']}R (条件{h['cond']}): 三連複 {h['trio']} = {h['payout']:,}円  TOP1スコア: {h['score']:.4f}")

    print(f"\n## 惜しい外れ (4着惜敗) 詳細")
    for n in near_miss_detail:
        print(f"  {n['course']} {n['rnum']}R (条件{n['cond']}) {n['surface']}{n['distance']}m: {','.join(n['near'])}が4着 | 三連複配当: {n['trio_pay']:,}円 | 4着馬: {n['fourth_horse']}")

    # Save corrected CSV
    rows_out = []
    for _, row in df_pred.iterrows():
        rid = str(row['race_id'])
        fd = all_results.get(rid, {})
        p = payouts_json.get(rid, {})

        t1n = int(row['top1_num'])
        t2n = int(row['top2_num'])
        t3n = int(row['top3_num'])

        t1f = fd.get(t1n, {}).get('finish')
        t2f = fd.get(t2n, {}).get('finish')
        t3f = fd.get(t3n, {}).get('finish')

        top3_actual = sorted([n for n, d in fd.items() if d['finish'] <= 3])
        trio_bets = row.get('trio_bets', '')
        hit = False
        if trio_bets and len(top3_actual) >= 3:
            actual_set = set(top3_actual[:3])
            for bet in trio_bets.split("; "):
                nums = set(int(n) for n in bet.split("-"))
                if nums == actual_set:
                    hit = True
                    break

        trio_payout = p.get('trio', 0) if hit else 0

        rows_out.append({
            'race_id': rid, 'course': get_course_name(rid), 'race_num': row['race_num'],
            'race_name': row.get('race_name', ''), 'condition': row['condition'],
            'num_horses': row['num_horses'], 'distance': row['distance'],
            'surface': row['surface'], 'track_condition': row.get('track_condition', ''),
            'top1_num': t1n, 'top1_name': row['top1_name'], 'top1_score': row['top1_score'],
            'top2_num': t2n, 'top3_num': t3n,
            'top1_finish': t1f if t1f else '-', 'top2_finish': t2f if t2f else '-',
            'top3_finish': t3f if t3f else '-',
            'trio_bets': trio_bets,
            'trio_result': '-'.join(str(n) for n in top3_actual[:3]) if len(top3_actual) >= 3 else '',
            'bet_type': row.get('bet_type', 'trio'), 'trio_hit': 1 if hit else 0,
            'trio_payout': p.get('trio', 0), 'umaren_payout': p.get('umaren', 0),
            'actual_payout': trio_payout, 'investment': 700,
            'profit': trio_payout - 700, 'status': 'settled',
        })

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(os.path.join(BASE_DIR, 'data/daily_results/20260314.csv'), index=False, encoding='utf-8-sig')
    df_out['date'] = '20260314'
    df_out.to_csv(os.path.join(BASE_DIR, 'data/cumulative_results.csv'), index=False, encoding='utf-8-sig')
    print(f"\n  CSV保存完了")

if __name__ == '__main__':
    main()
