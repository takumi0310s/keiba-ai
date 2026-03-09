"""
結果照合システム
predictions_log.csvの未確定レースの結果をnetkeibaから取得し、
的中判定・ROI計算を行う。

Usage:
    python check_results.py              # 未確定レースの結果を照合
    python check_results.py --summary    # 成績サマリー表示
    python check_results.py --force      # 全レース再照合
"""
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import json
import time
import os
from datetime import datetime

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
LOG_PATH = "data/predictions_log.csv"
RESULTS_PATH = "data/results_log.csv"


def scrape_race_result(race_id, is_nar=False):
    """netkeibaからレース結果（着順・配当）を取得"""
    if is_nar or len(str(race_id)) == 12:
        url = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"
    else:
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "EUC-JP"
    except Exception as e:
        print(f"  [ERROR] {race_id}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # 着順取得
    result_table = soup.select_one("table.RaceTable01, table.race_table_01")
    if not result_table:
        return None

    finish_order = {}  # {馬番: 着順}
    rows = result_table.select("tr")
    for row in rows[1:]:  # ヘッダースキップ
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        try:
            finish = cells[0].get_text(strip=True)
            if not finish.isdigit():
                continue
            finish = int(finish)
            # 馬番は通常2番目か3番目のセル
            umaban = None
            for i in range(1, min(4, len(cells))):
                t = cells[i].get_text(strip=True)
                if t.isdigit() and 1 <= int(t) <= 18:
                    umaban = int(t)
                    break
            if umaban:
                finish_order[umaban] = finish
        except (ValueError, IndexError):
            continue

    if not finish_order:
        return None

    # 上位3着を取得
    top3 = sorted(finish_order.items(), key=lambda x: x[1])[:3]
    top3_nums = [num for num, _ in top3]

    # 配当取得
    payouts = {}
    payout_tables = soup.select("table.Payout_Detail_Table, table.pay_table_01")
    if not payout_tables:
        payout_tables = soup.find_all("table", class_=re.compile(r"[Pp]ay"))

    for table in payout_tables:
        rows = table.find_all("tr")
        for row in rows:
            th = row.find("th")
            if not th:
                continue
            bet_type = th.get_text(strip=True)
            tds = row.find_all("td")
            if len(tds) < 2:
                continue

            nums_text = tds[0].get_text(strip=True)
            payout_text = tds[1].get_text(strip=True)
            payout_val = int(re.sub(r'[,円\s]', '', payout_text)) if re.search(r'\d', payout_text) else 0
            nums = [int(n) for n in re.findall(r'\d+', nums_text)]

            key = None
            if '単勝' in bet_type:
                key = 'tansho'
            elif '複勝' in bet_type:
                key = 'fukusho'
            elif '馬連' in bet_type and '三' not in bet_type:
                key = 'umaren'
            elif 'ワイド' in bet_type:
                key = 'wide'
            elif '三連複' in bet_type:
                key = 'trio'
            elif '三連単' in bet_type:
                key = 'tierce'

            if key:
                if key not in payouts:
                    payouts[key] = []
                payouts[key].append({'nums': nums, 'payout': payout_val})

    # PaybackWrap 別パターン
    if not payouts:
        wrap = soup.select_one("div.PaybackWrap, div.Result_Pay_Back")
        if wrap:
            for tbl in wrap.find_all("table"):
                for row in tbl.find_all("tr"):
                    cells = row.find_all(["th", "td"])
                    if len(cells) >= 3:
                        bet_type = cells[0].get_text(strip=True)
                        nums_text = cells[1].get_text(strip=True)
                        payout_text = cells[2].get_text(strip=True)
                        payout_val = int(re.sub(r'[,円\s]', '', payout_text)) if re.search(r'\d', payout_text) else 0
                        nums = [int(n) for n in re.findall(r'\d+', nums_text)]

                        key = None
                        if '単勝' in bet_type:
                            key = 'tansho'
                        elif '複勝' in bet_type:
                            key = 'fukusho'
                        elif '馬連' in bet_type and '三' not in bet_type:
                            key = 'umaren'
                        elif 'ワイド' in bet_type:
                            key = 'wide'
                        elif '三連複' in bet_type:
                            key = 'trio'
                        elif '三連単' in bet_type:
                            key = 'tierce'

                        if key:
                            if key not in payouts:
                                payouts[key] = []
                            payouts[key].append({'nums': nums, 'payout': payout_val})

    return {
        'finish_order': finish_order,
        'top3': top3_nums,
        'payouts': payouts,
    }


def check_hit(bets, top3, bet_type, payouts):
    """的中判定と配当計算
    Returns: (hit: bool, payout: int)
    """
    if not bets or not top3:
        return False, 0

    bets_list = json.loads(bets) if isinstance(bets, str) else bets
    top3_set = set(top3)
    top2_set = set(top3[:2])

    if bet_type == 'trio':
        # 三連複: 3頭の組み合わせが一致
        for bet in bets_list:
            if set(bet) == top3_set:
                # 配当取得
                if payouts and 'trio' in payouts:
                    return True, payouts['trio'][0]['payout']
                return True, 0
        return False, 0

    elif bet_type == 'umaren':
        # 馬連: 1-2着の組み合わせ
        for bet in bets_list:
            if set(bet) == top2_set:
                if payouts and 'umaren' in payouts:
                    return True, payouts['umaren'][0]['payout']
                return True, 0
        return False, 0

    elif bet_type == 'wide':
        # ワイド: TOP3内の2頭の組み合わせ（複数的中あり）
        total_payout = 0
        hit = False
        for bet in bets_list:
            bet_set = set(bet)
            if bet_set.issubset(top3_set):
                hit = True
                if payouts and 'wide' in payouts:
                    for wp in payouts['wide']:
                        if set(wp['nums']) == bet_set:
                            total_payout += wp['payout']
                            break
        return hit, total_payout

    return False, 0


def check_pending_results(force=False):
    """未確定レースの結果を照合"""
    if not os.path.exists(LOG_PATH):
        print("予測ログファイルが見つかりません。")
        print("先にpredict_and_log.pyで予測を実行してください。")
        return

    log_df = pd.read_csv(LOG_PATH)

    if force:
        pending = log_df
    else:
        pending = log_df[log_df['result_status'] == 'pending']

    if len(pending) == 0:
        print("未確定レースはありません。")
        return

    print(f"未確定レース: {len(pending)}件")
    print()

    updated = 0
    for idx in pending.index:
        row = log_df.loc[idx]
        race_id = str(row['race_id'])
        is_nar = bool(row.get('is_nar', 0))

        print(f"  [{updated+1}/{len(pending)}] {race_id} {row.get('race_name','')} ... ", end="")

        result = scrape_race_result(race_id, is_nar=is_nar)
        if result is None:
            print("結果未確定")
            continue

        top3 = result['top3']
        payouts = result['payouts']
        bet_type = row['bet_type']
        bets = row['bets']
        investment = int(row.get('investment', 700))

        hit, payout = check_hit(bets, top3, bet_type, payouts)

        roi = (payout / investment * 100) if investment > 0 and payout > 0 else 0

        log_df.at[idx, 'result_status'] = 'settled'
        log_df.at[idx, 'actual_top3'] = json.dumps(top3)
        log_df.at[idx, 'hit'] = 1 if hit else 0
        log_df.at[idx, 'payout'] = payout
        log_df.at[idx, 'roi'] = round(roi, 1)

        status = "✓ 的中" if hit else "✗ 不的中"
        payout_str = f" +{payout}円" if payout > 0 else ""
        print(f"{status}{payout_str}  着順: {top3}")

        updated += 1
        time.sleep(1.5)

    # 保存
    log_df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
    print(f"\n{updated}件の結果を更新しました。")

    # results_log.csvにも保存
    settled = log_df[log_df['result_status'] == 'settled']
    if len(settled) > 0:
        settled.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")


def show_summary():
    """成績サマリーを表示"""
    if not os.path.exists(LOG_PATH):
        print("予測ログファイルが見つかりません。")
        return

    log_df = pd.read_csv(LOG_PATH)
    settled = log_df[log_df['result_status'] == 'settled']
    pending = log_df[log_df['result_status'] == 'pending']

    print("=" * 60)
    print("実運用テスト 成績サマリー")
    print("=" * 60)
    print(f"総予測数: {len(log_df)}")
    print(f"確定済み: {len(settled)}")
    print(f"未確定:   {len(pending)}")
    print()

    if len(settled) == 0:
        print("確定済みのレースがありません。")
        return

    # 全体成績
    total_invest = settled['investment'].sum()
    total_payout = settled['payout'].sum()
    total_hits = (settled['hit'] == 1).sum()
    total_roi = total_payout / total_invest * 100 if total_invest > 0 else 0

    print(f"【全体成績】")
    print(f"  的中: {total_hits}/{len(settled)} ({total_hits/len(settled)*100:.1f}%)")
    print(f"  投資: {total_invest:,}円")
    print(f"  回収: {total_payout:,}円")
    print(f"  収支: {total_payout - total_invest:+,}円")
    print(f"  ROI:  {total_roi:.1f}%")
    print()

    # 条件別
    print(f"【条件別成績】")
    print(f"{'条件':>4} {'N':>4} {'的中':>4} {'的中率':>7} {'投資':>10} {'回収':>10} {'ROI':>8}")
    print("-" * 55)
    for cond in sorted(settled['cond_key'].unique()):
        sub = settled[settled['cond_key'] == cond]
        n = len(sub)
        hits = (sub['hit'] == 1).sum()
        invest = sub['investment'].sum()
        payout = sub['payout'].sum()
        roi = payout / invest * 100 if invest > 0 else 0
        print(f"{cond:>4} {n:>4} {hits:>4} {hits/n*100:>6.1f}% {invest:>9,}円 {payout:>9,}円 {roi:>7.1f}%")
    print()

    # 買い目タイプ別
    print(f"【買い目タイプ別】")
    for bt in sorted(settled['bet_type'].unique()):
        sub = settled[settled['bet_type'] == bt]
        n = len(sub)
        hits = (sub['hit'] == 1).sum()
        invest = sub['investment'].sum()
        payout = sub['payout'].sum()
        roi = payout / invest * 100 if invest > 0 else 0
        print(f"  {bt}: {hits}/{n} ({hits/n*100:.1f}%) ROI {roi:.1f}%")
    print()

    # 日別成績
    if 'predicted_at' in settled.columns:
        settled_copy = settled.copy()
        settled_copy['date'] = pd.to_datetime(settled_copy['predicted_at']).dt.date
        print(f"【日別成績】")
        print(f"{'日付':>12} {'N':>4} {'的中':>4} {'的中率':>7} {'収支':>10} {'ROI':>8}")
        print("-" * 55)
        for date in sorted(settled_copy['date'].unique()):
            sub = settled_copy[settled_copy['date'] == date]
            n = len(sub)
            hits = (sub['hit'] == 1).sum()
            invest = sub['investment'].sum()
            payout = sub['payout'].sum()
            roi = payout / invest * 100 if invest > 0 else 0
            print(f"{str(date):>12} {n:>4} {hits:>4} {hits/n*100:>6.1f}% {payout-invest:>+9,}円 {roi:>7.1f}%")
        print()

    # 中央 vs 地方
    for label, is_nar_val in [('中央(JRA)', 0), ('地方(NAR)', 1)]:
        sub = settled[settled['is_nar'] == is_nar_val]
        if len(sub) > 0:
            n = len(sub)
            hits = (sub['hit'] == 1).sum()
            invest = sub['investment'].sum()
            payout = sub['payout'].sum()
            roi = payout / invest * 100 if invest > 0 else 0
            print(f"  {label}: {hits}/{n} ({hits/n*100:.1f}%) ROI {roi:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="結果照合システム")
    parser.add_argument("--summary", action="store_true", help="成績サマリー表示")
    parser.add_argument("--force", action="store_true", help="全レース再照合")
    args = parser.parse_args()

    if args.summary:
        show_summary()
    else:
        check_pending_results(force=args.force)
        print()
        show_summary()
