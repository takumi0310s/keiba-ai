"""3/14結果の再取得・再判定スクリプト"""
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def fetch_result_debug(race_id):
    """結果取得（デバッグ版：詳細ログ付き）"""
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"    [ERROR] 取得失敗: {e}")
        return None, None, None

    # 着順テーブル
    result_table = soup.find("table", class_="RaceTable01")
    if not result_table:
        result_table = soup.find("table", class_="race_table_01")
    if not result_table:
        print(f"    [ERROR] テーブルなし")
        return None, None, None

    finish_order = {}
    for row in result_table.find_all("tr"):
        tds = row.find_all("td")
        if len(tds) < 4:
            continue
        finish_text = tds[0].get_text(strip=True)
        if not finish_text.isdigit():
            continue
        finish = int(finish_text)

        # 馬番: class="Num Txt_C" のtdから取得
        umaban = 0
        # 方法1: class名に"Txt_C"を含むtdを探す（結果ページの馬番列）
        for td in tds:
            cls = " ".join(td.get("class", []))
            if "Txt_C" in cls:
                t = td.get_text(strip=True)
                if t.isdigit() and 1 <= int(t) <= 18:
                    umaban = int(t)
                    break
        # 方法2: class名に"Umaban"を含むtdを探す（出馬表ページ）
        if umaban == 0:
            for td in tds:
                cls = " ".join(td.get("class", []))
                if "Umaban" in cls:
                    t = td.get_text(strip=True)
                    if t.isdigit() and 1 <= int(t) <= 18:
                        umaban = int(t)
                        break
        # 方法3: tds[2]（枠番tds[1]をスキップ）
        if umaban == 0 and len(tds) >= 3:
            t = tds[2].get_text(strip=True)
            if t.isdigit() and 1 <= int(t) <= 18:
                umaban = int(t)
        if umaban > 0:
            finish_order[umaban] = finish

    # TOP3
    top3 = sorted([(f, n) for n, f in finish_order.items() if f <= 3])
    trio_nums = sorted([n for _, n in top3[:3]]) if len(top3) >= 3 else None

    # 払戻金
    payouts = {'trio': 0, 'umaren': 0}

    # すべてのthタグから券種を探す
    for pt in soup.find_all("table"):
        for row in pt.find_all("tr"):
            th = row.find("th")
            if not th:
                continue
            th_text = th.get_text(strip=True)
            th_hex = th_text.encode('utf-8').hex()

            tds = row.find_all("td")
            payout_vals = []
            for td in tds:
                txt = td.get_text(strip=True)
                for m in re.finditer(r'([\d,]+)円', txt):
                    payout_vals.append(int(m.group(1).replace(',', '')))
                # 円がない場合、カンマ付き数値を探す
                if not payout_vals:
                    for m in re.finditer(r'^([\d,]+)$', txt):
                        val = int(m.group(1).replace(',', ''))
                        if val >= 100:
                            payout_vals.append(val)

            if not payout_vals:
                continue

            # 三連複: "三連複" or 文字化け "3連複"
            if '三連複' in th_text or '3連複' in th_text or '33e980a3e8a487' in th_hex:
                payouts['trio'] = payout_vals[0]
            elif '馬連' in th_text and '三' not in th_text and '単' not in th_text and '3' not in th_text:
                payouts['umaren'] = payout_vals[0]
            elif 'e9a6ace980a3' in th_hex and 'e9a6ace58d98' not in th_hex and '33' not in th_hex:
                payouts['umaren'] = payout_vals[0]

    return finish_order, trio_nums, payouts


def main():
    print("=" * 60)
    print("  3/14 結果再取得・再判定")
    print("=" * 60)

    # 予測CSV読み込み
    pred_path = os.path.join(BASE_DIR, "data", "daily_predictions", "20260314.csv")
    df_pred = pd.read_csv(pred_path, encoding='utf-8-sig')
    print(f"\n予測レース数: {len(df_pred)}")

    results = []
    hits = []
    mismatches = []

    for idx, row in df_pred.iterrows():
        race_id = str(row['race_id'])
        course = row.get('course', '')
        race_num = row.get('race_num', 0)
        trio_bets_str = row.get('trio_bets', '')
        bet_type = row.get('bet_type', 'trio')
        condition = row.get('condition', '')

        print(f"\n[{idx+1:2d}] {course} {race_num}R (ID={race_id})")

        finish_order, trio_nums, payouts = fetch_result_debug(race_id)
        time.sleep(0.8)

        if finish_order is None:
            print(f"    結果取得失敗")
            results.append({**row.to_dict(),
                           'top1_finish': '-', 'top2_finish': '-', 'top3_finish': '-',
                           'trio_result': '', 'trio_hit': 0, 'trio_payout': 0,
                           'umaren_payout': 0, 'actual_payout': 0,
                           'investment': 700, 'profit': -700, 'status': 'pending'})
            continue

        trio_result_str = '-'.join(str(n) for n in trio_nums) if trio_nums else ''
        print(f"    着順: {dict(sorted(finish_order.items(), key=lambda x: x[1]))}")
        print(f"    三連複結果: {trio_result_str}")
        print(f"    配当: trio={payouts['trio']}, umaren={payouts['umaren']}")

        # TOP3 着順
        top1_num = int(row.get('top1_num', 0))
        top2_num = int(row.get('top2_num', 0))
        top3_num = int(row.get('top3_num', 0))
        top1_finish = finish_order.get(top1_num, '-')
        top2_finish = finish_order.get(top2_num, '-')
        top3_finish = finish_order.get(top3_num, '-')
        print(f"    AI TOP3着順: {top1_num}番={top1_finish}着, {top2_num}番={top2_finish}着, {top3_num}番={top3_finish}着")

        # 的中判定
        trio_hit = False
        hit_combo = None
        umaren_hit = False
        actual_payout = 0

        if bet_type == 'trio' and trio_nums:
            actual_set = set(trio_nums)
            bets = trio_bets_str.split("; ")
            print(f"    買い目7点: {bets}")
            print(f"    実際の三連複: {actual_set}")
            for bet in bets:
                bet_nums = [int(n) for n in bet.split("-")]
                if set(bet_nums) == actual_set:
                    trio_hit = True
                    hit_combo = bet
                    actual_payout = payouts['trio']
                    break
        elif bet_type == 'umaren':
            top2_actual = set(n for n, f in finish_order.items() if f <= 2)
            umaren_bets = [set([top1_num, top2_num]), set([top1_num, top3_num])]
            for ub in umaren_bets:
                if ub == top2_actual:
                    umaren_hit = True
                    actual_payout = payouts['umaren']
                    break

        if trio_hit:
            print(f"    >>> HIT! 的中: {hit_combo} 配当={actual_payout}円 <<<")
            hits.append(f"{course} {race_num}R: {hit_combo} = {actual_payout}円")
        elif umaren_hit:
            print(f"    >>> HIT! 馬連的中 配当={actual_payout}円 <<<")
            hits.append(f"{course} {race_num}R: umaren = {actual_payout}円")
        else:
            print(f"    miss")

        profit = actual_payout - 700
        result_row = {
            'race_id': race_id, 'course': course, 'race_num': race_num,
            'race_name': row.get('race_name', ''), 'condition': condition,
            'num_horses': row.get('num_horses', 0),
            'distance': row.get('distance', 0),
            'surface': row.get('surface', ''),
            'track_condition': row.get('track_condition', ''),
            'top1_num': top1_num,
            'top1_name': row.get('top1_name', ''),
            'top1_score': row.get('top1_score', 0),
            'top2_num': top2_num,
            'top3_num': top3_num,
            'top1_finish': top1_finish,
            'top2_finish': top2_finish,
            'top3_finish': top3_finish,
            'trio_bets': trio_bets_str,
            'trio_result': trio_result_str,
            'bet_type': bet_type,
            'trio_hit': 1 if trio_hit else 0,
            'trio_payout': payouts['trio'] if trio_hit else 0,
            'umaren_payout': payouts['umaren'] if umaren_hit else 0,
            'actual_payout': actual_payout,
            'investment': 700,
            'profit': profit,
            'status': 'settled',
        }
        results.append(result_row)

    # サマリー
    print(f"\n{'=' * 60}")
    print(f"  再判定結果サマリー")
    print(f"{'=' * 60}")
    print(f"\n  的中レース ({len(hits)}件):")
    for h in hits:
        print(f"    {h}")

    settled = [r for r in results if r.get('status') == 'settled']
    total_inv = len(settled) * 700
    total_payout = sum(r.get('actual_payout', 0) for r in settled)
    total_profit = total_payout - total_inv
    hit_count = sum(1 for r in settled if r.get('trio_hit') == 1 or r.get('umaren_payout', 0) > 0)
    roi = total_payout / total_inv * 100 if total_inv > 0 else 0

    print(f"\n  レース数: {len(settled)}")
    print(f"  的中: {hit_count}/{len(settled)}")
    print(f"  投資: {total_inv:,}円")
    print(f"  払戻: {total_payout:,}円")
    print(f"  収支: {'+' if total_profit >= 0 else ''}{total_profit:,}円")
    print(f"  ROI: {roi:.1f}%")

    # CSV保存
    out_path = os.path.join(BASE_DIR, "data", "daily_results", "20260314.csv")
    df_out = pd.DataFrame(results)
    df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n  保存: {out_path}")

    # 累積結果も更新
    cumul_path = os.path.join(BASE_DIR, "data", "cumulative_results.csv")
    df_out['date'] = '20260314'
    if os.path.exists(cumul_path):
        df_cumul = pd.read_csv(cumul_path, encoding='utf-8-sig')
        df_cumul = df_cumul[df_cumul['date'] != '20260314']
        df_cumul = pd.concat([df_cumul, df_out], ignore_index=True)
    else:
        df_cumul = df_out
    df_cumul.to_csv(cumul_path, index=False, encoding='utf-8-sig')
    print(f"  累積更新: {cumul_path}")


if __name__ == '__main__':
    main()
