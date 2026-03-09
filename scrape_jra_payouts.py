#!/usr/bin/env python
"""中央競馬 配当データスクレイパー

netkeibaのレース結果ページから配当データ（三連複・馬連・ワイド・単勝・複勝）を取得。
2018年〜2025年のバックテスト期間に対応。

Usage:
    python scrape_jra_payouts.py              # 未取得分を取得
    python scrape_jra_payouts.py --status     # 進捗確認
    python scrape_jra_payouts.py --year 2024  # 特定年のみ
    python scrape_jra_payouts.py --reset      # 進捗リセット

出力:
    data/jra_payouts.csv         - 配当データ
    data/jra_payout_progress.json - 進捗情報
"""
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import json
import time
import os
import random
import sys
from datetime import datetime

# === Configuration ===
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RACES_PATH = os.path.join(DATA_DIR, 'jra_races_full.csv')
OUTPUT_PATH = os.path.join(DATA_DIR, 'jra_payouts.csv')
PROGRESS_PATH = os.path.join(DATA_DIR, 'jra_payout_progress.json')

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

CSV_HEADER = [
    'nk_race_id', 'tansho_nums', 'tansho_payout',
    'fukusho_nums', 'fukusho_payouts',
    'umaren_nums', 'umaren_payout',
    'wide_nums', 'wide_payouts',
    'trio_nums', 'trio_payout',
    'tierce_nums', 'tierce_payout',
]

BATCH_SIZE = 100  # CSVに追記する間隔
MAX_RETRIES = 3
MIN_DELAY = 3.0
MAX_DELAY = 6.0


def get_race_ids(year_from=2018, year_to=2025):
    """jra_races_full.csvからnetkeiba形式のレースIDリストを生成"""
    df = pd.read_csv(RACES_PATH, encoding='utf-8-sig', dtype=str,
                     usecols=['race_id', 'year', 'kai', 'nichi', 'race_num'])
    df['year_full'] = df['year'].astype(int) + 2000
    df['course_code'] = df['race_id'].str[:2]

    # netkeiba race_id: 20{YY}{CC}{KK}{NN}{RR}
    df['nk_race_id'] = ('20' + df['year'] + df['course_code']
                         + df['kai'].str.zfill(2) + df['nichi'].str.zfill(2)
                         + df['race_num'].str.zfill(2))

    mask = (df['year_full'] >= year_from) & (df['year_full'] <= year_to)
    race_ids = sorted(df.loc[mask, 'nk_race_id'].unique().tolist())
    return race_ids


def load_progress():
    """進捗ファイルを読み込み"""
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'completed': [],
        'failed': [],
        'total': 0,
        'started_at': None,
        'last_updated': None,
    }


def save_progress(progress):
    """進捗ファイルを保存"""
    progress['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(PROGRESS_PATH, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def scrape_payout(race_id, session):
    """netkeibaレース結果ページから配当データを取得"""
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    headers = {"User-Agent": random.choice(USER_AGENTS)}

    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, headers=headers, timeout=15)
            if resp.status_code == 403:
                wait = 30 * (attempt + 1)
                print(f"    403 Forbidden, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None  # レースが存在しない
            resp.raise_for_status()
            resp.encoding = 'EUC-JP'
            break
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(10)
                continue
            print(f"    ERROR {race_id}: {e}")
            return None
    else:
        return None

    soup = BeautifulSoup(resp.text, 'html.parser')
    result = {'nk_race_id': race_id}

    # 配当テーブルを探す
    payout_data = _parse_payouts(soup)
    if not payout_data:
        return None

    result.update(payout_data)
    return result


def _parse_payouts(soup):
    """配当テーブルからデータを抽出"""
    data = {
        'tansho_nums': '', 'tansho_payout': 0,
        'fukusho_nums': '', 'fukusho_payouts': '',
        'umaren_nums': '', 'umaren_payout': 0,
        'wide_nums': '', 'wide_payouts': '',
        'trio_nums': '', 'trio_payout': 0,
        'tierce_nums': '', 'tierce_payout': 0,
    }

    found = False

    # パターン1: Payout_Detail_Table
    tables = soup.select("table.Payout_Detail_Table, table.pay_table_01")
    if not tables:
        tables = soup.find_all("table", class_=re.compile(r"[Pp]ay"))

    for table in tables:
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

            # 配当金額を抽出
            payout_val = _parse_payout_value(payout_text)
            nums = re.findall(r'\d+', nums_text)

            if '単勝' in bet_type:
                data['tansho_nums'] = '-'.join(nums)
                data['tansho_payout'] = payout_val
                found = True
            elif '複勝' in bet_type:
                # 複勝は複数行ある場合がある
                if data['fukusho_nums']:
                    data['fukusho_nums'] += '/' + '-'.join(nums)
                    data['fukusho_payouts'] += '/' + str(payout_val)
                else:
                    data['fukusho_nums'] = '-'.join(nums)
                    data['fukusho_payouts'] = str(payout_val)
            elif '馬連' in bet_type and '三' not in bet_type:
                data['umaren_nums'] = '-'.join(nums)
                data['umaren_payout'] = payout_val
                found = True
            elif 'ワイド' in bet_type:
                if data['wide_nums']:
                    data['wide_nums'] += '/' + '-'.join(nums)
                    data['wide_payouts'] += '/' + str(payout_val)
                else:
                    data['wide_nums'] = '-'.join(nums)
                    data['wide_payouts'] = str(payout_val)
            elif '三連複' in bet_type:
                data['trio_nums'] = '-'.join(nums)
                data['trio_payout'] = payout_val
                found = True
            elif '三連単' in bet_type:
                data['tierce_nums'] = '-'.join(nums)
                data['tierce_payout'] = payout_val

    # パターン2: PaybackWrap
    if not found:
        wrap = soup.select_one("div.PaybackWrap, div.Result_Pay_Back")
        if wrap:
            for tbl in wrap.find_all("table"):
                for row in tbl.find_all("tr"):
                    cells = row.find_all(["th", "td"])
                    if len(cells) >= 3:
                        bet_type = cells[0].get_text(strip=True)
                        nums_text = cells[1].get_text(strip=True)
                        payout_text = cells[2].get_text(strip=True)
                        payout_val = _parse_payout_value(payout_text)
                        nums = re.findall(r'\d+', nums_text)

                        if '単勝' in bet_type:
                            data['tansho_nums'] = '-'.join(nums)
                            data['tansho_payout'] = payout_val
                            found = True
                        elif '複勝' in bet_type:
                            if data['fukusho_nums']:
                                data['fukusho_nums'] += '/' + '-'.join(nums)
                                data['fukusho_payouts'] += '/' + str(payout_val)
                            else:
                                data['fukusho_nums'] = '-'.join(nums)
                                data['fukusho_payouts'] = str(payout_val)
                        elif '馬連' in bet_type and '三' not in bet_type:
                            data['umaren_nums'] = '-'.join(nums)
                            data['umaren_payout'] = payout_val
                            found = True
                        elif 'ワイド' in bet_type:
                            if data['wide_nums']:
                                data['wide_nums'] += '/' + '-'.join(nums)
                                data['wide_payouts'] += '/' + str(payout_val)
                            else:
                                data['wide_nums'] = '-'.join(nums)
                                data['wide_payouts'] = str(payout_val)
                        elif '三連複' in bet_type:
                            data['trio_nums'] = '-'.join(nums)
                            data['trio_payout'] = payout_val
                            found = True
                        elif '三連単' in bet_type:
                            data['tierce_nums'] = '-'.join(nums)
                            data['tierce_payout'] = payout_val

    return data if found else None


def _parse_payout_value(text):
    """配当テキストから金額を抽出"""
    # "1,234円" → 1234, "12,340" → 12340
    cleaned = re.sub(r'[,円\s]', '', text)
    match = re.search(r'(\d+)', cleaned)
    if match:
        return int(match.group(1))
    return 0


def append_to_csv(records):
    """CSVにバッチ追記"""
    if not records:
        return

    df = pd.DataFrame(records, columns=CSV_HEADER)
    file_exists = os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 0

    df.to_csv(OUTPUT_PATH, mode='a', header=not file_exists,
              index=False, encoding='utf-8-sig')


def run_scraper(year_from=2018, year_to=2025):
    """メインスクレイピング処理"""
    print(f"{'='*60}")
    print(f"  JRA配当データスクレイパー")
    print(f"  期間: {year_from}-{year_to}")
    print(f"  開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # レースIDリスト生成
    all_race_ids = get_race_ids(year_from, year_to)
    print(f"  全レース数: {len(all_race_ids)}")

    # 進捗ロード
    progress = load_progress()
    completed = set(progress.get('completed', []))
    failed = set(progress.get('failed', []))

    # 既存CSV確認
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH, encoding='utf-8-sig', dtype=str, usecols=['nk_race_id'])
        completed.update(existing['nk_race_id'].tolist())

    remaining = [r for r in all_race_ids if r not in completed and r not in failed]
    print(f"  取得済み: {len(completed)}")
    print(f"  残り: {len(remaining)}")

    if not remaining:
        print("  全レース取得済みです。")
        return

    if not progress.get('started_at'):
        progress['started_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    progress['total'] = len(all_race_ids)
    save_progress(progress)

    session = requests.Session()
    batch = []
    batch_count = 0
    error_streak = 0

    for i, race_id in enumerate(remaining):
        # 進捗表示
        total_done = len(completed) + i
        pct = total_done / len(all_race_ids) * 100
        if i % 10 == 0:
            print(f"\r  [{total_done}/{len(all_race_ids)}] {pct:.1f}% - {race_id}", end="", flush=True)

        result = scrape_payout(race_id, session)

        if result:
            row = [result.get(col, '') for col in CSV_HEADER]
            batch.append(row)
            completed.add(race_id)
            error_streak = 0
        else:
            failed.add(race_id)
            error_streak += 1

        # バッチ保存
        batch_count += 1
        if batch_count >= BATCH_SIZE:
            append_to_csv(batch)
            batch = []
            batch_count = 0
            progress['completed'] = list(completed)
            progress['failed'] = list(failed)
            save_progress(progress)
            print(f"\n  Saved batch. Total: {len(completed)}/{len(all_race_ids)}")

        # 連続エラー対策
        if error_streak >= 10:
            print(f"\n  連続エラー{error_streak}件。60秒待機...")
            time.sleep(60)
            error_streak = 0

        # リクエスト間隔
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    # 残りを保存
    if batch:
        append_to_csv(batch)

    progress['completed'] = list(completed)
    progress['failed'] = list(failed)
    save_progress(progress)

    print(f"\n\n{'='*60}")
    print(f"  完了: {len(completed)}/{len(all_race_ids)} ({len(completed)/len(all_race_ids)*100:.1f}%)")
    print(f"  失敗: {len(failed)}")
    print(f"  出力: {OUTPUT_PATH}")
    print(f"{'='*60}")


def show_status():
    """進捗状況を表示"""
    progress = load_progress()
    completed = len(progress.get('completed', []))
    failed = len(progress.get('failed', []))
    total = progress.get('total', 0)

    print(f"{'='*40}")
    print(f"  JRA配当スクレイパー 進捗")
    print(f"{'='*40}")
    print(f"  取得済み: {completed}/{total} ({completed/total*100:.1f}%)" if total > 0 else "  未開始")
    print(f"  失敗: {failed}")
    print(f"  開始: {progress.get('started_at', '-')}")
    print(f"  最終: {progress.get('last_updated', '-')}")

    if os.path.exists(OUTPUT_PATH):
        df = pd.read_csv(OUTPUT_PATH, encoding='utf-8-sig', dtype=str)
        print(f"  CSVレコード: {len(df)}")
        trio_valid = df['trio_payout'].astype(float).gt(0).sum()
        print(f"  三連複配当あり: {trio_valid}")
    print(f"{'='*40}")

    # 推定残り時間
    if completed > 0 and total > completed:
        remaining = total - completed
        avg_time = 4.5  # 平均4.5秒/レース
        eta_min = remaining * avg_time / 60
        print(f"  推定残り時間: {eta_min:.0f}分 ({eta_min/60:.1f}時間)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='JRA配当データスクレイパー')
    parser.add_argument('--status', action='store_true', help='進捗確認')
    parser.add_argument('--year', type=int, help='特定年のみ取得')
    parser.add_argument('--reset', action='store_true', help='進捗リセット')
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.reset:
        if os.path.exists(PROGRESS_PATH):
            os.remove(PROGRESS_PATH)
            print("進捗をリセットしました。")
        if os.path.exists(OUTPUT_PATH):
            os.remove(OUTPUT_PATH)
            print("CSVを削除しました。")
    elif args.year:
        run_scraper(year_from=args.year, year_to=args.year)
    else:
        run_scraper()
