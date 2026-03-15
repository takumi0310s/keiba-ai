"""
毎夕の結果照合スクリプト
当日の予測結果と実際のレース結果を照合し、的中判定・ROI計算を行う。

データソース:
  --source db  : Streamlit DB (keiba_predictions.db) から読み込み（デフォルト）
  --source csv : daily_predict CSV (data/daily_predictions/) から読み込み

結果出力:
  - data/daily_results/{date}.csv        — 日別結果
  - data/cumulative_results.csv          — 累積結果
  - data/track_record.csv               — Streamlit Cloud同期用（git管理）

Usage:
    python tools/daily_results.py                  # 今日の結果（DB参照）
    python tools/daily_results.py --date 20260315  # 日付指定
    python tools/daily_results.py --source csv     # CSV参照
"""
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import os
import sys
import json
import sqlite3
import argparse
import time
from datetime import datetime

# === パス設定 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# === 定数 ===
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
INVESTMENT_PER_RACE = 700
STREAMLIT_DB_PATHS = [
    os.path.join(BASE_DIR, "keiba_predictions.db"),    # app.pyのメインDB
    os.path.join(BASE_DIR, "keiba_race_results.db"),   # 予備DB
]
TRACK_RECORD_CSV = os.path.join(BASE_DIR, "data", "track_record.csv")


def fetch_race_result(race_id):
    """netkeibaからレース結果を取得"""
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

    result_table = soup.find("table", class_="RaceTable01")
    if not result_table:
        result_table = soup.find("table", class_="race_table_01")
    if not result_table:
        print(f"  [WARN] 結果テーブルが見つかりません（レース未確定の可能性）")
        return None

    top3_nums = []
    for row in result_table.find_all("tr"):
        tds = row.find_all("td")
        if len(tds) < 4:
            continue
        finish_text = tds[0].get_text(strip=True)
        if not finish_text.isdigit():
            continue
        finish = int(finish_text)
        umaban = 0
        for td in tds:
            cls = " ".join(td.get("class", []))
            if "Txt_C" in cls:
                t = td.get_text(strip=True)
                if t.isdigit() and 1 <= int(t) <= 18:
                    umaban = int(t)
                    break
        if umaban == 0:
            for td in tds:
                cls = " ".join(td.get("class", []))
                if "Umaban" in cls:
                    t = td.get_text(strip=True)
                    if t.isdigit() and 1 <= int(t) <= 18:
                        umaban = int(t)
                        break
        if umaban == 0 and len(tds) >= 3:
            t = tds[2].get_text(strip=True)
            if t.isdigit() and 1 <= int(t) <= 18:
                umaban = int(t)
        if umaban > 0:
            result['finish_order'][umaban] = finish
            if finish <= 3:
                top3_nums.append((finish, umaban))

    top3_nums.sort()
    if len(top3_nums) >= 3:
        result['trio_nums'] = sorted([n for _, n in top3_nums[:3]])

    # 払戻金テーブルの解析
    BET_TYPE_HEX = {
        'tansho':  'e58d98e58b9d',
        'umaren':  'e9a6ace980a3',
        'umatan':  'e9a6ace58d98',
        'wide':    'e383afe382a4e38389',
        'trio_g':  '33e980a3e8a487',
        'tierce_g':'33e980a3e58d98',
    }

    payout_tables = soup.find_all("table", class_="Payout_Detail_Table")
    if not payout_tables:
        payout_tables = soup.find_all("table", class_="pay_table_01")

    for pt in payout_tables:
        for row in pt.find_all("tr"):
            th = row.find("th")
            if not th:
                continue
            th_text = th.get_text(strip=True)
            th_hex = th_text.encode('utf-8').hex()

            tds = row.find_all("td")
            payout_vals = []
            for td in tds:
                for m in re.finditer(r'([\d,]+)円', td.get_text(strip=True)):
                    payout_vals.append(int(m.group(1).replace(',', '')))

            if not payout_vals:
                continue
            payout_val = payout_vals[0]

            if '単勝' in th_text or (BET_TYPE_HEX['tansho'] in th_hex and '連' not in th_text and 'e980a3' not in th_hex):
                result['payouts']['tansho'] = payout_val
            elif '三連複' in th_text or BET_TYPE_HEX['trio_g'] in th_hex:
                result['payouts']['trio'] = payout_val
            elif '馬連' in th_text and '三' not in th_text and '単' not in th_text:
                result['payouts']['umaren'] = payout_val
            elif BET_TYPE_HEX['umaren'] in th_hex and BET_TYPE_HEX['umatan'] not in th_hex and '33' not in th_hex:
                result['payouts']['umaren'] = payout_val
            elif 'ワイド' in th_text or BET_TYPE_HEX['wide'] in th_hex:
                if result['payouts']['wide'] == 0:
                    result['payouts']['wide'] = payout_val

    return result


def check_trio_hit(trio_bets_str, actual_trio_nums):
    """三連複的中判定（文字列形式 "1-2-3; 1-2-4; ..."）"""
    if not trio_bets_str or not actual_trio_nums:
        return False, None

    actual_set = set(actual_trio_nums)
    bets = trio_bets_str.split("; ")
    for bet in bets:
        nums = [int(n) for n in bet.split("-")]
        if set(nums) == actual_set:
            return True, bet
    return False, None


# ===== データソース読み込み =====

def load_predictions_from_db(date_str):
    """Streamlit DB から予測データを読み込む"""
    race_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    for db_path in STREAMLIT_DB_PATHS:
        if not os.path.exists(db_path):
            continue
        result = _try_load_from_db(db_path, race_date)
        if result:
            db_name = os.path.basename(db_path)
            print(f"  [DB] {db_name} から {len(result)}レース分の予測を読み込み")
            return result

    return None


def _try_load_from_db(db_path, race_date):
    """指定DBから予測データを読み込む"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT rr.race_id, rr.race_name, rr.bet_condition, rr.bet_type,
                   rr.trio_bets, rr.umaren_bets, rr.num_horses, rr.buy_recommended
            FROM race_results rr
            WHERE rr.predicted_at LIKE ?
            ORDER BY rr.race_id
        """, (f"{race_date}%",))
        rr_rows = c.fetchall()

        if not rr_rows:
            conn.close()
            return None

        c.execute("""
            SELECT race_id, horse_num, ai_rank
            FROM predictions
            WHERE race_date = ? AND ai_rank <= 6
            ORDER BY race_id, ai_rank
        """, (race_date,))
        pred_rows = c.fetchall()
        conn.close()

        top_nums = {}
        for pr in pred_rows:
            rid = str(pr['race_id'])
            if rid not in top_nums:
                top_nums[rid] = {}
            top_nums[rid][pr['ai_rank']] = pr['horse_num']

        predictions = []
        for rr in rr_rows:
            rid = str(rr['race_id'])
            trio_bets_raw = rr['trio_bets']
            umaren_bets_raw = rr['umaren_bets']
            bet_type = rr['bet_type'] or 'trio'
            condition = rr['bet_condition'] or ''

            if bet_type == 'trio' and trio_bets_raw:
                bets_list = json.loads(trio_bets_raw)
                trio_bets_str = '; '.join(
                    '-'.join(str(n) for n in sorted(b)) for b in bets_list
                )
            elif bet_type == 'umaren' and umaren_bets_raw:
                bets_list = json.loads(umaren_bets_raw)
                trio_bets_str = '; '.join(
                    '-'.join(str(n) for n in sorted(b)) for b in bets_list
                )
            else:
                trio_bets_str = ''

            race_name = rr['race_name'] or ''
            course = re.sub(r'\d+R$', '', race_name)
            race_num_m = re.search(r'(\d+)R$', race_name)
            race_num = int(race_num_m.group(1)) if race_num_m else 0

            tops = top_nums.get(rid, {})

            predictions.append({
                'race_id': rid,
                'course': course,
                'race_num': race_num,
                'race_name': race_name,
                'condition': condition,
                'trio_bets': trio_bets_str,
                'bet_type': bet_type,
                'investment': INVESTMENT_PER_RACE,
                'top1_num': tops.get(1, 0),
                'top2_num': tops.get(2, 0),
                'top3_num': tops.get(3, 0),
            })

        return predictions

    except Exception as e:
        print(f"  [WARN] DB読み込みエラー ({os.path.basename(db_path)}): {e}")
        return None


def load_predictions_from_csv(date_str):
    """daily_predict CSVから予測データを読み込む"""
    pred_path = os.path.join(BASE_DIR, "data", "daily_predictions", f"{date_str}.csv")
    if not os.path.exists(pred_path):
        return None

    df_pred = pd.read_csv(pred_path, encoding='utf-8-sig')
    predictions = []
    for _, row in df_pred.iterrows():
        predictions.append({
            'race_id': str(row['race_id']),
            'course': row.get('course', ''),
            'race_num': row.get('race_num', 0),
            'race_name': row.get('race_name', ''),
            'condition': row.get('condition', ''),
            'trio_bets': row.get('trio_bets', ''),
            'bet_type': row.get('bet_type', 'trio'),
            'investment': row.get('investment', INVESTMENT_PER_RACE),
            'top1_num': row.get('top1_num', 0),
            'top2_num': row.get('top2_num', 0),
            'top3_num': row.get('top3_num', 0),
        })
    return predictions


# ===== 結果同期用CSV書き出し =====

def save_track_record_csv(results, date_str):
    """track_record.csv に結果を追記（Streamlit Cloud同期用）

    このCSVはgit管理されるため、pushすればStreamlit Cloudでも参照可能。
    app.pyのTRACK RECORDページがこのCSVを読めるようにする。
    """
    settled = [r for r in results if r.get('status') == 'settled']
    if not settled:
        return

    rows = []
    for r in settled:
        rows.append({
            'date': date_str,
            'race_id': r['race_id'],
            'course': r['course'],
            'race_num': r['race_num'],
            'race_name': r['race_name'],
            'condition': r['condition'],
            'bet_type': r.get('bet_type', 'trio'),
            'trio_bets': r.get('trio_bets_str', ''),
            'trio_result': r.get('trio_result', ''),
            'hit': 1 if r.get('trio_hit') == 1 or r.get('umaren_hit') == 1 else 0,
            'payout': r.get('trio_payout', 0) + r.get('umaren_payout', 0),
            'investment': r['investment'],
            'profit': r['profit'],
        })

    df_new = pd.DataFrame(rows)

    if os.path.exists(TRACK_RECORD_CSV):
        df_existing = pd.read_csv(TRACK_RECORD_CSV, encoding='utf-8-sig')
        # 同日分は上書き
        df_existing = df_existing[df_existing['date'] != date_str]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(TRACK_RECORD_CSV, index=False, encoding='utf-8-sig')
    print(f"TRACK RECORD CSV更新: {TRACK_RECORD_CSV}")


# ===== メイン処理 =====

def run_daily_results(date_str, source='db'):
    """指定日の結果照合"""
    print(f"{'=' * 60}")
    print(f"KEIBA AI 結果照合 - {date_str}")
    print(f"{'=' * 60}")

    # 予測データ読み込み
    predictions = None
    pred_source = ''

    if source == 'db':
        predictions = load_predictions_from_db(date_str)
        if predictions:
            pred_source = 'Streamlit DB'
        else:
            print(f"[ERROR] Streamlit DBに {date_str} の予測データがありません")
            for p in STREAMLIT_DB_PATHS:
                exists = "存在" if os.path.exists(p) else "なし"
                print(f"  {os.path.basename(p)}: {exists}")
            print(f"  Streamlitで予測を実行してください")
            return
    elif source == 'csv':
        predictions = load_predictions_from_csv(date_str)
        if predictions:
            pred_source = 'daily_predict CSV'
        else:
            pred_path = os.path.join(BASE_DIR, "data", "daily_predictions", f"{date_str}.csv")
            print(f"[ERROR] 予測CSVが見つかりません: {pred_path}")
            print(f"  daily_predict.pyを実行してください")
            return

    print(f"\n予測ソース: {pred_source}")
    print(f"予測レース数: {len(predictions)}")

    results = []
    for idx, row in enumerate(predictions):
        race_id = row['race_id']
        course = row['course']
        race_num = row['race_num']
        race_name = row['race_name']
        condition = row['condition']
        trio_bets_str = row['trio_bets']
        bet_type = row['bet_type']
        investment = row['investment']

        print(f"\n[{idx+1}/{len(predictions)}] {course} {race_num}R {race_name} (ID={race_id})")

        # 結果取得
        race_result = fetch_race_result(race_id)
        time.sleep(1.0)

        if race_result is None:
            print(f"  結果未確定")
            results.append({
                'race_id': race_id, 'course': course, 'race_num': race_num,
                'race_name': race_name, 'condition': condition,
                'trio_hit': None, 'trio_payout': 0,
                'umaren_hit': None, 'umaren_payout': 0,
                'investment': investment, 'profit': -investment,
                'status': 'pending', 'bet_type': bet_type,
                'trio_bets_str': trio_bets_str,
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

        top1_finish = finish_order.get(row.get('top1_num', 0), '-')
        top2_finish = finish_order.get(row.get('top2_num', 0), '-')
        top3_finish = finish_order.get(row.get('top3_num', 0), '-')

        result_row = {
            'race_id': race_id, 'course': course, 'race_num': race_num,
            'race_name': race_name, 'condition': condition,
            'bet_type': bet_type,
            'trio_bets_str': trio_bets_str,
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

        hit_mark = "HIT!" if (trio_hit or umaren_hit) else "miss"
        payout_disp = f"払戻 {actual_payout:,}円" if actual_payout > 0 else ""
        print(f"  結果: {hit_mark} {payout_disp}")
        print(f"  三連複結果: {result_row['trio_result']}")
        print(f"  AI TOP3 着順: {top1_finish}着/{top2_finish}着/{top3_finish}着")
        if trio_hit:
            print(f"  的中組合せ: {hit_combo}")

    # 結果保存
    if results:
        # 日別CSV
        out_dir = os.path.join(BASE_DIR, "data", "daily_results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{date_str}.csv")
        df_results = pd.DataFrame(results)
        df_results.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n保存先: {out_path}")

        # 累積CSV
        cumul_path = os.path.join(BASE_DIR, "data", "cumulative_results.csv")
        df_results['date'] = date_str
        if os.path.exists(cumul_path):
            df_cumul = pd.read_csv(cumul_path, encoding='utf-8-sig')
            df_cumul = df_cumul[df_cumul['date'] != date_str]
            df_cumul = pd.concat([df_cumul, df_results], ignore_index=True)
        else:
            df_cumul = df_results
        df_cumul.to_csv(cumul_path, index=False, encoding='utf-8-sig')
        print(f"累積結果更新: {cumul_path}")

        # track_record.csv（Streamlit Cloud同期用）
        save_track_record_csv(results, date_str)

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
            print(f"  予測ソース: {pred_source}")
            print(f"  対象レース: {len(settled)}R (未確定: {len(pending)}R)")
            print(f"  的中: {hit_count}/{len(settled)} ({hit_count/len(settled)*100:.1f}%)")
            print(f"  投資: {total_inv:,}円")
            print(f"  払戻: {total_payout:,}円")
            profit_sign = '+' if total_profit >= 0 else ''
            print(f"  収支: {profit_sign}{total_profit:,}円")
            print(f"  ROI: {roi:.1f}%")

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
    parser.add_argument("--source", type=str, default="db",
                        choices=["db", "csv"],
                        help="予測ソース: db=Streamlit DB（デフォルト）, csv=daily_predict CSV")
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
    run_daily_results(date_str, source=args.source)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] daily_results.py 終了")
