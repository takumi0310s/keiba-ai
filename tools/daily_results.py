"""
毎夕の結果照合スクリプト
当日の予測結果と実際のレース結果を照合し、的中判定・ROI計算を行う。

Usage:
    python tools/daily_results.py                  # 今日の結果
    python tools/daily_results.py --date 20260315  # 日付指定
"""
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import os
import sys
import json
import argparse
import time
from datetime import datetime

# === パス設定 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# === 定数 ===
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
INVESTMENT_PER_RACE = 700


def fetch_race_result(race_id):
    """netkeibaからレース結果を取得

    Returns:
        dict: {
            'finish_order': {馬番: 着順, ...},
            'payouts': {'trio': int, 'umaren': int, 'wide': int, 'tansho': int},
            'trio_nums': [n1, n2, n3] or None,  # 三連複の着順馬番
        }
    """
    result = {
        'finish_order': {},
        'payouts': {'trio': 0, 'umaren': 0, 'wide': 0, 'tansho': 0},
        'trio_nums': None,
    }
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"  [ERROR] 結果ページ取得失敗: {e}")
        return None

    # 着順テーブルの解析
    result_table = soup.find("table", class_="RaceTable01")
    if not result_table:
        # 旧レイアウトも試す
        result_table = soup.find("table", class_="race_table_01")
    if not result_table:
        print(f"  [WARN] 結果テーブルが見つかりません（レース未確定の可能性）")
        return None

    top3_nums = []
    for row in result_table.find_all("tr"):
        tds = row.find_all("td")
        if len(tds) < 4:
            continue
        # 着順（1列目）
        finish_text = tds[0].get_text(strip=True)
        if not finish_text.isdigit():
            continue
        finish = int(finish_text)
        # 馬番（3列目程度）
        umaban = 0
        for td in tds[1:4]:
            t = td.get_text(strip=True)
            if t.isdigit() and 1 <= int(t) <= 18:
                umaban = int(t)
                break
        if umaban == 0:
            # クラスからUmabanを探す
            for td in tds:
                cls = " ".join(td.get("class", []))
                if "Umaban" in cls or "Num" in cls:
                    t = td.get_text(strip=True)
                    if t.isdigit() and 1 <= int(t) <= 18:
                        umaban = int(t)
                        break
        if umaban > 0:
            result['finish_order'][umaban] = finish
            if finish <= 3:
                top3_nums.append((finish, umaban))

    top3_nums.sort()
    if len(top3_nums) >= 3:
        result['trio_nums'] = sorted([n for _, n in top3_nums[:3]])

    # 払戻金テーブルの解析
    payout_tables = soup.find_all("table", class_="Payout_Detail_Table")
    if not payout_tables:
        payout_tables = soup.find_all("table", class_="pay_table_01")

    for pt in payout_tables:
        for row in pt.find_all("tr"):
            tds = row.find_all("td")
            th = row.find("th")
            if not th:
                continue
            bet_type_text = th.get_text(strip=True)

            # 各種配当を取得
            payout_val = 0
            for td in tds:
                t = td.get_text(strip=True).replace(',', '').replace('円', '').replace('¥', '')
                try:
                    v = int(t)
                    if v >= 100:
                        payout_val = v
                        break
                except:
                    # 複数配当がある場合（ワイドなど）
                    nums = re.findall(r'[\d,]+', td.get_text(strip=True).replace(',', ''))
                    for n in nums:
                        try:
                            v = int(n)
                            if v >= 100:
                                payout_val = v
                                break
                        except:
                            continue
                    if payout_val > 0:
                        break

            if '単勝' in bet_type_text:
                result['payouts']['tansho'] = payout_val
            elif '馬連' in bet_type_text and '三' not in bet_type_text:
                result['payouts']['umaren'] = payout_val
            elif '三連複' in bet_type_text:
                result['payouts']['trio'] = payout_val
            elif 'ワイド' in bet_type_text:
                # ワイドは最初の配当のみ取得（簡易）
                if result['payouts']['wide'] == 0:
                    result['payouts']['wide'] = payout_val

    # 払戻テーブルが見つからない場合の別解析
    if result['payouts']['trio'] == 0 and result['payouts']['tansho'] == 0:
        # ページ全体からパターンマッチ
        all_text = soup.get_text()
        trio_match = re.search(r'三連複[^\d]*?([\d,]+)\s*円', all_text)
        if trio_match:
            result['payouts']['trio'] = int(trio_match.group(1).replace(',', ''))
        umaren_match = re.search(r'馬連[^\d]*?([\d,]+)\s*円', all_text)
        if umaren_match:
            result['payouts']['umaren'] = int(umaren_match.group(1).replace(',', ''))
        tansho_match = re.search(r'単勝[^\d]*?([\d,]+)\s*円', all_text)
        if tansho_match:
            result['payouts']['tansho'] = int(tansho_match.group(1).replace(',', ''))

    return result


def check_trio_hit(trio_bets_str, actual_trio_nums):
    """三連複的中判定

    Args:
        trio_bets_str: "1-2-3; 1-2-4; ..." 形式
        actual_trio_nums: [n1, n2, n3] ソート済み

    Returns:
        (hit: bool, hit_combo: str or None)
    """
    if not trio_bets_str or not actual_trio_nums:
        return False, None

    actual_set = set(actual_trio_nums)
    bets = trio_bets_str.split("; ")
    for bet in bets:
        nums = [int(n) for n in bet.split("-")]
        if set(nums) == actual_set:
            return True, bet
    return False, None


def run_daily_results(date_str):
    """指定日の結果照合"""
    print(f"{'=' * 60}")
    print(f"KEIBA AI 結果照合 - {date_str}")
    print(f"{'=' * 60}")

    # 予測CSVロード
    pred_path = os.path.join(BASE_DIR, "data", "daily_predictions", f"{date_str}.csv")
    if not os.path.exists(pred_path):
        print(f"[ERROR] 予測ファイルが見つかりません: {pred_path}")
        print(f"  先にdaily_predict.pyを実行してください")
        return

    df_pred = pd.read_csv(pred_path, encoding='utf-8-sig')
    print(f"\n予測レース数: {len(df_pred)}")

    results = []
    for idx, row in df_pred.iterrows():
        race_id = str(row['race_id'])
        course = row.get('course', '')
        race_num = row.get('race_num', 0)
        race_name = row.get('race_name', '')
        condition = row.get('condition', '')
        trio_bets_str = row.get('trio_bets', '')
        bet_type = row.get('bet_type', 'trio')
        investment = row.get('investment', INVESTMENT_PER_RACE)

        print(f"\n[{idx+1}/{len(df_pred)}] {course} {race_num}R {race_name} (ID={race_id})")

        # 結果取得
        race_result = fetch_race_result(race_id)
        time.sleep(1.0)  # 負荷軽減

        if race_result is None:
            print(f"  結果未確定")
            results.append({
                'race_id': race_id, 'course': course, 'race_num': race_num,
                'race_name': race_name, 'condition': condition,
                'trio_hit': None, 'trio_payout': 0,
                'umaren_hit': None, 'umaren_payout': 0,
                'investment': investment, 'profit': -investment,
                'status': 'pending',
            })
            continue

        finish_order = race_result['finish_order']
        payouts = race_result['payouts']
        trio_nums = race_result['trio_nums']

        # 的中判定
        trio_hit = False
        trio_payout = 0
        hit_combo = None
        umaren_hit = False
        umaren_payout = 0

        if bet_type == 'trio' and trio_nums:
            trio_hit, hit_combo = check_trio_hit(trio_bets_str, trio_nums)
            if trio_hit:
                trio_payout = payouts.get('trio', 0)
        elif bet_type == 'umaren':
            # 馬連判定（TOP1-TOP2, TOP1-TOP3）
            top1_num = row.get('top1_num', 0)
            top2_num = row.get('top2_num', 0)
            top3_num = row.get('top3_num', 0)
            top2_actual = set(n for n, f in finish_order.items() if f <= 2)
            umaren_bets = [
                set([top1_num, top2_num]),
                set([top1_num, top3_num]),
            ]
            for ub in umaren_bets:
                if ub.issubset(top2_actual) and len(ub) == 2:
                    umaren_hit = True
                    umaren_payout = payouts.get('umaren', 0)
                    break

        actual_payout = trio_payout if trio_hit else (umaren_payout if umaren_hit else 0)
        profit = actual_payout - investment

        # 上位3頭の着順
        top1_finish = finish_order.get(row.get('top1_num', 0), '-')
        top2_finish = finish_order.get(row.get('top2_num', 0), '-')
        top3_finish = finish_order.get(row.get('top3_num', 0), '-')

        result_row = {
            'race_id': race_id, 'course': course, 'race_num': race_num,
            'race_name': race_name, 'condition': condition,
            'trio_hit': 1 if trio_hit else 0,
            'trio_payout': trio_payout,
            'umaren_hit': 1 if umaren_hit else 0,
            'umaren_payout': umaren_payout,
            'investment': investment,
            'profit': profit,
            'status': 'settled',
            'top1_finish': top1_finish,
            'top2_finish': top2_finish,
            'top3_finish': top3_finish,
            'trio_result': '-'.join(str(n) for n in trio_nums) if trio_nums else '',
        }
        results.append(result_row)

        # コンソール出力
        hit_mark = "HIT!" if (trio_hit or umaren_hit) else "miss"
        payout_disp = f"払戻 {actual_payout:,}円" if actual_payout > 0 else ""
        print(f"  結果: {hit_mark} {payout_disp}")
        print(f"  三連複結果: {result_row['trio_result']}")
        print(f"  AI TOP3 着順: {top1_finish}着/{top2_finish}着/{top3_finish}着")
        if trio_hit:
            print(f"  的中組合せ: {hit_combo}")

    # 結果CSV保存
    if results:
        out_dir = os.path.join(BASE_DIR, "data", "daily_results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{date_str}.csv")
        df_results = pd.DataFrame(results)
        df_results.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n保存先: {out_path}")

        # 累積結果に追記
        cumul_path = os.path.join(BASE_DIR, "data", "cumulative_results.csv")
        df_results['date'] = date_str
        if os.path.exists(cumul_path):
            df_cumul = pd.read_csv(cumul_path, encoding='utf-8-sig')
            # 同日の結果は上書き
            df_cumul = df_cumul[df_cumul['date'] != date_str]
            df_cumul = pd.concat([df_cumul, df_results], ignore_index=True)
        else:
            df_cumul = df_results
        df_cumul.to_csv(cumul_path, index=False, encoding='utf-8-sig')
        print(f"累積結果更新: {cumul_path}")

        # サマリー出力
        settled = [r for r in results if r.get('status') == 'settled']
        pending = [r for r in results if r.get('status') == 'pending']
        if settled:
            total_inv = sum(r['investment'] for r in settled)
            total_payout = sum(r.get('trio_payout', 0) + r.get('umaren_payout', 0) for r in settled)
            total_profit = total_payout - total_inv
            hit_count = sum(1 for r in settled if r.get('trio_hit') == 1 or r.get('umaren_hit') == 1)
            roi = (total_payout / total_inv * 100) if total_inv > 0 else 0

            print(f"\n{'=' * 60}")
            print(f"  日次サマリー: {date_str}")
            print(f"{'=' * 60}")
            print(f"  対象レース: {len(settled)}R (未確定: {len(pending)}R)")
            print(f"  的中: {hit_count}/{len(settled)} ({hit_count/len(settled)*100:.1f}%)")
            print(f"  投資: {total_inv:,}円")
            print(f"  払戻: {total_payout:,}円")
            profit_sign = '+' if total_profit >= 0 else ''
            print(f"  収支: {profit_sign}{total_profit:,}円")
            print(f"  ROI: {roi:.1f}%")

            # 条件別
            cond_stats = {}
            for r in settled:
                c = r.get('condition', 'X')
                if c not in cond_stats:
                    cond_stats[c] = {'count': 0, 'hit': 0, 'inv': 0, 'pay': 0}
                cond_stats[c]['count'] += 1
                if r.get('trio_hit') == 1 or r.get('umaren_hit') == 1:
                    cond_stats[c]['hit'] += 1
                cond_stats[c]['inv'] += r['investment']
                cond_stats[c]['pay'] += r.get('trio_payout', 0) + r.get('umaren_payout', 0)

            print(f"\n  条件別:")
            for c in sorted(cond_stats.keys()):
                s = cond_stats[c]
                c_roi = (s['pay'] / s['inv'] * 100) if s['inv'] > 0 else 0
                print(f"    {c}: {s['hit']}/{s['count']}的中 ROI {c_roi:.1f}%")

            print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KEIBA AI 結果照合")
    parser.add_argument("--date", type=str, default=None,
                        help="照合日 YYYYMMDD (デフォルト: 今日)")
    args = parser.parse_args()

    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime("%Y%m%d")

    try:
        datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        print(f"[ERROR] 日付形式が不正です: {date_str} (YYYYMMDD)")
        sys.exit(1)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] daily_results.py 開始")
    run_daily_results(date_str)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] daily_results.py 終了")
