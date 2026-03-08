#!/usr/bin/env python
"""netkeibaから騎手・調教師の年間成績(2020〜現在)をスクレイピング

出力:
  data/jockey_history.csv   - 騎手別年間成績
  data/trainer_history.csv  - 調教師別年間成績

Usage:
  python scrape_jockey_trainer.py
  python scrape_jockey_trainer.py --start-year 2020 --end-year 2025
"""
import pandas as pd
import numpy as np
import os
import sys
import time
import argparse
import requests
from bs4 import BeautifulSoup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
JOCKEY_OUTPUT = os.path.join(DATA_DIR, 'jockey_history.csv')
TRAINER_OUTPUT = os.path.join(DATA_DIR, 'trainer_history.csv')

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}

# netkeibaの騎手・調教師一覧URL
JOCKEY_LIST_URL = "https://db.netkeiba.com/jockey/result/?pid=jockey_list&year={year}&page={page}"
TRAINER_LIST_URL = "https://db.netkeiba.com/trainer/result/?pid=trainer_list&year={year}&page={page}"

# 個別騎手・調教師の年間成績URL
JOCKEY_RESULT_URL = "https://db.netkeiba.com/jockey/result/{jockey_id}/"
TRAINER_RESULT_URL = "https://db.netkeiba.com/trainer/result/{trainer_id}/"

# 騎手リーディングURL（年別）
JOCKEY_LEADING_URL = "https://db.netkeiba.com/jockey/leading/?year={year}&page={page}"
TRAINER_LEADING_URL = "https://db.netkeiba.com/trainer/leading/?year={year}&page={page}"


def fetch_page(url, max_retries=3):
    """ページ取得（リトライ付き）"""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                resp.encoding = 'euc-jp'
                return resp.text
            elif resp.status_code == 403 or resp.status_code == 400:
                print(f"  Blocked ({resp.status_code}): {url}")
                return None
            else:
                print(f"  HTTP {resp.status_code}: {url} (retry {attempt+1})")
                time.sleep(3)
        except requests.exceptions.RequestException as e:
            print(f"  Error: {e} (retry {attempt+1})")
            time.sleep(3)
    return None


def scrape_leading_page(url):
    """リーディング一覧ページから成績データを取得"""
    html = fetch_page(url)
    if not html:
        return []

    soup = BeautifulSoup(html, 'html.parser')
    results = []

    table = soup.find('table', class_='nk_tb_common')
    if not table:
        # 別のテーブルクラスを試行
        tables = soup.find_all('table')
        for t in tables:
            if t.find('th') and ('勝率' in t.get_text() or '1着' in t.get_text()):
                table = t
                break

    if not table:
        return []

    rows = table.find_all('tr')
    for row in rows[1:]:  # ヘッダーをスキップ
        cells = row.find_all('td')
        if len(cells) < 5:
            continue

        # 名前とIDのリンクを取得
        name_cell = cells[1] if len(cells) > 1 else cells[0]
        link = name_cell.find('a')
        if not link:
            continue

        href = link.get('href', '')
        name = link.get_text(strip=True)

        # IDを抽出（/jockey/XXXXX/ or /trainer/XXXXX/）
        person_id = ''
        parts = href.strip('/').split('/')
        for p in parts:
            if p.isdigit() or (len(p) == 5 and p.replace('0', '').isdigit()):
                person_id = p
                break

        if not person_id:
            continue

        # 成績データを取得
        try:
            data = {
                'id': person_id,
                'name': name,
            }

            # カラム位置は年度/ページによって変わりうるので柔軟に取得
            texts = [c.get_text(strip=True) for c in cells]

            # 一般的なフォーマット: 順位, 名前, 1着, 2着, 3着, 着外, 騎乗数, 勝率, 連対率, 複勝率, 賞金
            for idx, txt in enumerate(texts):
                txt_clean = txt.replace(',', '').replace('%', '')
                if idx == 0 and txt_clean.isdigit():
                    data['rank'] = int(txt_clean)

            # 数値カラムを順に取得
            num_values = []
            for txt in texts[2:]:
                txt_clean = txt.replace(',', '').replace('%', '').replace('万', '')
                try:
                    num_values.append(float(txt_clean))
                except ValueError:
                    continue

            if len(num_values) >= 7:
                data['wins'] = int(num_values[0])
                data['seconds'] = int(num_values[1])
                data['thirds'] = int(num_values[2])
                data['unplaced'] = int(num_values[3])
                data['rides'] = int(num_values[0] + num_values[1] + num_values[2] + num_values[3])
                data['win_rate'] = num_values[4] if num_values[4] <= 1 else num_values[4] / 100
                data['place_rate'] = num_values[5] if num_values[5] <= 1 else num_values[5] / 100
                data['show_rate'] = num_values[6] if num_values[6] <= 1 else num_values[6] / 100
                if len(num_values) >= 8:
                    data['prize'] = num_values[7]
                results.append(data)

        except Exception as e:
            continue

    return results


def scrape_leading_all_pages(base_url, year, max_pages=10):
    """全ページのリーディングデータを取得"""
    all_results = []
    for page in range(1, max_pages + 1):
        url = base_url.format(year=year, page=page)
        results = scrape_leading_page(url)
        if not results:
            break
        all_results.extend(results)
        print(f"    Page {page}: {len(results)} entries")
        time.sleep(2)

    return all_results


def scrape_jockey_history(start_year, end_year):
    """騎手年間成績を取得"""
    print("\n" + "=" * 60)
    print(f"  騎手年間成績スクレイピング ({start_year}-{end_year})")
    print("=" * 60)

    all_data = []
    for year in range(start_year, end_year + 1):
        print(f"\n  {year}年:")
        results = scrape_leading_all_pages(JOCKEY_LEADING_URL, year)
        if not results:
            print(f"    データ取得失敗（ブロックされた可能性あり）")
            continue

        for r in results:
            r['year'] = year

        all_data.extend(results)
        print(f"    合計: {len(results)} 騎手")
        time.sleep(3)

    if all_data:
        df = pd.DataFrame(all_data)
        # 整理
        cols = ['year', 'id', 'name', 'rank', 'rides', 'wins', 'seconds', 'thirds',
                'unplaced', 'win_rate', 'place_rate', 'show_rate', 'prize']
        cols = [c for c in cols if c in df.columns]
        df = df[cols].sort_values(['year', 'rank']).reset_index(drop=True)
        df.to_csv(JOCKEY_OUTPUT, index=False, encoding='utf-8-sig')
        print(f"\n  Saved: {JOCKEY_OUTPUT}")
        print(f"  Total: {len(df)} records ({df['id'].nunique()} unique jockeys)")
        return df
    else:
        print("\n  データを取得できませんでした。")
        return None


def scrape_trainer_history(start_year, end_year):
    """調教師年間成績を取得"""
    print("\n" + "=" * 60)
    print(f"  調教師年間成績スクレイピング ({start_year}-{end_year})")
    print("=" * 60)

    all_data = []
    for year in range(start_year, end_year + 1):
        print(f"\n  {year}年:")
        results = scrape_leading_all_pages(TRAINER_LEADING_URL, year)
        if not results:
            print(f"    データ取得失敗（ブロックされた可能性あり）")
            continue

        for r in results:
            r['year'] = year

        all_data.extend(results)
        print(f"    合計: {len(results)} 調教師")
        time.sleep(3)

    if all_data:
        df = pd.DataFrame(all_data)
        cols = ['year', 'id', 'name', 'rank', 'rides', 'wins', 'seconds', 'thirds',
                'unplaced', 'win_rate', 'place_rate', 'show_rate', 'prize']
        cols = [c for c in cols if c in df.columns]
        df = df[cols].sort_values(['year', 'rank']).reset_index(drop=True)
        df.to_csv(TRAINER_OUTPUT, index=False, encoding='utf-8-sig')
        print(f"\n  Saved: {TRAINER_OUTPUT}")
        print(f"  Total: {len(df)} records ({df['id'].nunique()} unique trainers)")
        return df
    else:
        print("\n  データを取得できませんでした。")
        return None


def main():
    parser = argparse.ArgumentParser(description='netkeiba 騎手・調教師成績スクレイピング')
    parser.add_argument('--start-year', type=int, default=2020, help='開始年 (default: 2020)')
    parser.add_argument('--end-year', type=int, default=2025, help='終了年 (default: 2025)')
    parser.add_argument('--jockey-only', action='store_true', help='騎手のみ')
    parser.add_argument('--trainer-only', action='store_true', help='調教師のみ')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("  netkeiba 騎手・調教師 年間成績スクレイピング")
    print(f"  期間: {args.start_year} - {args.end_year}")
    print("=" * 60)
    print("\n  NOTE: db.netkeiba.comがブロックしている場合、データ取得に失敗します。")
    print("  その場合は時間をおいて再実行してください。\n")

    if not args.trainer_only:
        jockey_df = scrape_jockey_history(args.start_year, args.end_year)

    if not args.jockey_only:
        trainer_df = scrape_trainer_history(args.start_year, args.end_year)

    print("\n" + "=" * 60)
    print("  完了")
    print("=" * 60)


if __name__ == '__main__':
    main()
