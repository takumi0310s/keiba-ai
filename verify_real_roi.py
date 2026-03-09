"""
実配当ROI検証スクリプト
netkeibaから過去レースの実配当を取得し、バックテストの推定ROIと比較する。
"""
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import time
import json
import os
from datetime import datetime

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

PAYOUT_CACHE_PATH = "data/payout_cache.json"


def load_payout_cache():
    if os.path.exists(PAYOUT_CACHE_PATH):
        with open(PAYOUT_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_payout_cache(cache):
    os.makedirs("data", exist_ok=True)
    with open(PAYOUT_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def scrape_race_payouts(race_id):
    """netkeibaからレースの払戻金を取得する。
    Returns: dict with keys 'tansho', 'fukusho', 'umaren', 'wide', 'umatan', 'trio', 'tierce'
    各値は [(馬番組, 払戻金), ...] のリスト
    """
    # race_idの長さで中央/地方判定
    is_nar = len(str(race_id)) == 12 and str(race_id)[:2] not in ('20',)
    # netkeibaのrace_id形式判定
    rid = str(race_id)
    if len(rid) <= 10:
        # 中央: 10桁以下
        url = f"https://race.netkeiba.com/race/result.html?race_id={rid}"
    else:
        url = f"https://nar.netkeiba.com/race/result.html?race_id={rid}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "EUC-JP"
    except Exception as e:
        print(f"  [ERROR] {race_id}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    payouts = {}

    # 払戻テーブルを探す
    payout_tables = soup.select("table.Payout_Detail_Table, table.pay_table_01")
    if not payout_tables:
        # 結果ページの別パターン
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

            # 馬番と払戻金を抽出
            nums_text = tds[0].get_text(strip=True)
            payout_text = tds[1].get_text(strip=True)

            # 払戻金を数値化 (カンマ、円を除去)
            payout_val = int(re.sub(r'[,円\s]', '', payout_text)) if re.search(r'\d', payout_text) else 0

            # 馬番を解析
            nums = re.findall(r'\d+', nums_text)

            key = None
            if '単勝' in bet_type:
                key = 'tansho'
            elif '複勝' in bet_type:
                key = 'fukusho'
            elif '馬連' in bet_type and '三' not in bet_type:
                key = 'umaren'
            elif 'ワイド' in bet_type:
                key = 'wide'
            elif '馬単' in bet_type:
                key = 'umatan'
            elif '三連複' in bet_type:
                key = 'trio'
            elif '三連単' in bet_type:
                key = 'tierce'

            if key:
                if key not in payouts:
                    payouts[key] = []
                payouts[key].append({
                    'nums': [int(n) for n in nums],
                    'payout': payout_val
                })

    # 別パターン: PaybackWrap内のテーブル
    if not payouts:
        wrap = soup.select_one("div.PaybackWrap, div.Result_Pay_Back")
        if wrap:
            for tbl in wrap.find_all("table"):
                rows = tbl.find_all("tr")
                for row in rows:
                    cells = row.find_all(["th", "td"])
                    if len(cells) >= 3:
                        bet_type = cells[0].get_text(strip=True)
                        nums_text = cells[1].get_text(strip=True)
                        payout_text = cells[2].get_text(strip=True)

                        payout_val = int(re.sub(r'[,円\s]', '', payout_text)) if re.search(r'\d', payout_text) else 0
                        nums = re.findall(r'\d+', nums_text)

                        key = None
                        if '単勝' in bet_type:
                            key = 'tansho'
                        elif '複勝' in bet_type:
                            key = 'fukusho'
                        elif '馬連' in bet_type and '三' not in bet_type:
                            key = 'umaren'
                        elif 'ワイド' in bet_type:
                            key = 'wide'
                        elif '馬単' in bet_type:
                            key = 'umatan'
                        elif '三連複' in bet_type:
                            key = 'trio'
                        elif '三連単' in bet_type:
                            key = 'tierce'

                        if key:
                            if key not in payouts:
                                payouts[key] = []
                            payouts[key].append({
                                'nums': [int(n) for n in nums],
                                'payout': payout_val
                            })

    return payouts if payouts else None


def build_netkeiba_race_id(sim_race_id, year):
    """simulation_results_jra.csvのrace_idからnetkeibaのrace_idを構築"""
    rid = str(sim_race_id)
    # simulation_results_jra.csvのrace_idは独自形式
    # netkeibaは YYYY + 場所2桁 + 回2桁 + 日2桁 + レース番号2桁 = 12桁
    # 例: 202406050811 = 2024年 06場(中山) 05回 08日 11R
    # sim_race_idは場所+回+日+レース番号 の短縮形の可能性
    # 年を補完して12桁にする
    if len(rid) == 8:
        return str(year) + rid
    elif len(rid) == 10:
        return rid
    elif len(rid) == 12:
        return rid
    return str(year) + rid.zfill(8)


def check_trio_hit(ai_top3, actual_top3, trio_bets_7point):
    """三連複7点の的中判定
    ai_top3: AI予測TOP3の馬番リスト
    actual_top3: 実際の上位3頭の馬番セット
    trio_bets_7point: 7点の三連複買い目 [[n1,n2,n3], ...]
    """
    actual_set = set(actual_top3)
    for bet in trio_bets_7point:
        if set(bet) == actual_set:
            return True
    return False


def verify_roi_with_real_payouts(sample_size=200, year_range=(2023, 2025)):
    """バックテスト結果に対して実配当で照合する"""
    sim_df = pd.read_csv("data/simulation_results_jra.csv")
    sim_df = sim_df[(sim_df['year'] >= year_range[0]) & (sim_df['year'] <= year_range[1])]

    # 的中レースのみ対象 (配当取得が意味あるのは的中時のみだが、
    # 不的中レースも含めてROI計算するため全体をサンプリング)
    # 条件別に均等サンプリング
    sampled = []
    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        cond_df = sim_df[sim_df['cond_key'] == cond]
        n = min(len(cond_df), sample_size // 6)
        if n > 0:
            sampled.append(cond_df.sample(n=n, random_state=42))
    sampled_df = pd.concat(sampled).reset_index(drop=True)

    print(f"サンプルサイズ: {len(sampled_df)}レース")
    print(f"期間: {year_range[0]}-{year_range[1]}")
    print()

    # キャッシュ読み込み
    cache = load_payout_cache()
    results = []
    new_fetches = 0

    for idx, row in sampled_df.iterrows():
        race_id_nk = build_netkeiba_race_id(row['race_id'], row['year'])
        cond = row['cond_key']

        if race_id_nk in cache:
            payouts = cache[race_id_nk]
        else:
            print(f"  [{idx+1}/{len(sampled_df)}] Fetching {race_id_nk} (cond={cond})...")
            payouts = scrape_race_payouts(race_id_nk)
            if payouts:
                cache[race_id_nk] = payouts
                new_fetches += 1
            else:
                cache[race_id_nk] = {}
                new_fetches += 1

            # 保存 & スリープ
            if new_fetches % 10 == 0:
                save_payout_cache(cache)
            time.sleep(1.5)

        # 実配当の計算
        trio_hit = row['trio_hit']
        trio_est_return = row['trio_return']

        real_trio_payout = 0
        real_umaren_payout = 0
        real_wide_payout = 0

        if payouts and isinstance(payouts, dict):
            # 三連複の実配当
            if trio_hit and 'trio' in payouts:
                for p in payouts['trio']:
                    real_trio_payout = p['payout']  # 100円あたり

            # 馬連
            if row.get('umaren_hit') and 'umaren' in payouts:
                for p in payouts['umaren']:
                    real_umaren_payout = p['payout']

            # ワイド (複数的中の可能性あり)
            if row.get('wide_hit') and 'wide' in payouts:
                for p in payouts['wide']:
                    real_wide_payout += p['payout']

        results.append({
            'race_id': race_id_nk,
            'year': row['year'],
            'cond_key': cond,
            'trio_hit': trio_hit,
            'trio_invest': row['trio_invest'],
            'trio_est_return': trio_est_return,
            'trio_real_payout_per100': real_trio_payout,
            'trio_real_return': real_trio_payout * 7 if trio_hit else 0,  # 7点買い
            'umaren_hit': row.get('umaren_hit', False),
            'umaren_est_return': row.get('umaren_return', 0),
            'umaren_real_payout_per100': real_umaren_payout,
            'wide_hit': row.get('wide_hit', False),
            'wide_est_return': row.get('wide_return', 0),
            'wide_real_payout_per100': real_wide_payout,
        })

    # キャッシュ保存
    save_payout_cache(cache)

    # 結果分析
    result_df = pd.DataFrame(results)
    result_df.to_csv("data/roi_verification.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("実配当ROI検証結果")
    print("=" * 70)

    comparison = []
    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        sub = result_df[result_df['cond_key'] == cond]
        if len(sub) == 0:
            continue
        n = len(sub)
        hits = sub['trio_hit'].sum()
        hit_rate = hits / n * 100 if n > 0 else 0
        est_invest = sub['trio_invest'].sum()
        est_return = sub['trio_est_return'].sum()
        est_roi = est_return / est_invest * 100 if est_invest > 0 else 0

        # 実配当ROI (trio_real_payout_per100は100円あたり)
        # 7点買い=700円投資。的中時は payout_per100 円が100円あたりの払戻
        # 実際の払戻 = payout_per100 / 100 * 100 = payout_per100 円 (100円購入時)
        # 7点中1点的中: 投資700円, 回収=payout_per100円
        real_invest = n * 700
        real_return = sub['trio_real_payout_per100'].sum()  # 100円あたりの合計
        real_roi = real_return / real_invest * 100 if real_invest > 0 else 0

        # 推定ROIも同じ基準で再計算
        est_roi_recalc = est_return / est_invest * 100 if est_invest > 0 else 0

        comparison.append({
            'condition': cond,
            'N': n,
            'hits': int(hits),
            'hit_rate': round(hit_rate, 1),
            'est_roi': round(est_roi_recalc, 1),
            'real_roi': round(real_roi, 1),
            'diff': round(real_roi - est_roi_recalc, 1),
        })

        print(f"\n条件{cond} (N={n}, 的中={int(hits)}, 的中率={hit_rate:.1f}%):")
        print(f"  推定ROI: {est_roi_recalc:.1f}%")
        print(f"  実配当ROI: {real_roi:.1f}%")
        print(f"  差分: {real_roi - est_roi_recalc:+.1f}%")

    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv("data/roi_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n\n=== 比較表 ===")
    print(f"{'条件':>4} {'N':>5} {'的中':>4} {'的中率':>7} {'推定ROI':>9} {'実ROI':>9} {'差分':>8}")
    print("-" * 50)
    for _, r in comp_df.iterrows():
        print(f"{r['condition']:>4} {r['N']:>5} {r['hits']:>4} {r['hit_rate']:>6.1f}% {r['est_roi']:>8.1f}% {r['real_roi']:>8.1f}% {r['diff']:>+7.1f}%")

    # 全体
    total_n = comp_df['N'].sum()
    total_hits = comp_df['hits'].sum()
    total_est = result_df['trio_est_return'].sum()
    total_real = result_df['trio_real_payout_per100'].sum()
    total_invest = total_n * 700
    print("-" * 50)
    print(f"{'全体':>4} {total_n:>5} {total_hits:>4} {total_hits/total_n*100:>6.1f}% {total_est/total_invest*100:>8.1f}% {total_real/total_invest*100:>8.1f}% {(total_real-total_est)/total_invest*100:>+7.1f}%")

    return comp_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="実配当ROI検証")
    parser.add_argument("--sample", type=int, default=120, help="サンプルサイズ (default: 120)")
    parser.add_argument("--year-start", type=int, default=2023)
    parser.add_argument("--year-end", type=int, default=2025)
    args = parser.parse_args()

    verify_roi_with_real_payouts(
        sample_size=args.sample,
        year_range=(args.year_start, args.year_end)
    )
