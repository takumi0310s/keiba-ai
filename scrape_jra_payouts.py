#!/usr/bin/env python
"""中央競馬 配当データスクレイパー (JRA公式DB版)

JRA公式サイト(jra.go.jp/JRADB)から配当データを取得。
ナビゲーションフロー: カレンダー → 開催日 → 全レース一覧（配当込み）

1開催日分（最大36レース）を1リクエストで取得可能。
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
import requests
from bs4 import BeautifulSoup
import re
import json
import time
import os
import sys
import io
from datetime import datetime

# UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# === Configuration ===
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUTPUT_PATH = os.path.join(DATA_DIR, 'jra_payouts.csv')
PROGRESS_PATH = os.path.join(DATA_DIR, 'jra_payout_progress.json')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

JRA_DB_URL = 'https://www.jra.go.jp/JRADB/accessS.html'

# 開催日あたりの待機時間（秒）
MIN_DELAY = 2.0
MAX_DELAY = 4.0

CSV_HEADER = [
    'race_date', 'course', 'kai', 'nichi', 'race_num',
    'tansho_nums', 'tansho_payout',
    'fukusho_nums', 'fukusho_payouts',
    'umaren_nums', 'umaren_payout',
    'wide_nums', 'wide_payouts',
    'trio_nums', 'trio_payout',
    'tierce_nums', 'tierce_payout',
]

# 競馬場コード → 名前
COURSE_NAMES = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
    '05': '東京', '06': '中山', '07': '中京', '08': '京都',
    '09': '阪神', '10': '小倉',
}


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_progress(progress):
    progress['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(PROGRESS_PATH, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def append_to_csv(rows):
    """CSVにバッチ追記"""
    if not rows:
        return
    write_header = not os.path.exists(OUTPUT_PATH)
    with open(OUTPUT_PATH, 'a', encoding='utf-8-sig', newline='') as f:
        if write_header:
            f.write(','.join(CSV_HEADER) + '\n')
        for row in rows:
            f.write(','.join(str(v) for v in row) + '\n')


def get_calendar_links(session, year, month_label):
    """JRA DBカレンダーから開催日リンクを取得

    Args:
        session: requests.Session
        year: 対象年 (int)
        month_label: 月ラベル（例: '202403'）→ pw01skl10YYYYMM/XX

    Returns:
        list of (label, cname): 開催日ごとのリンク
    """
    # まず過去レース結果ページにアクセス
    r = session.post(JRA_DB_URL, data={'CNAME': 'pw01skl00999999/B3'},
                     headers=HEADERS, timeout=15)
    r.encoding = 'shift_jis'

    # 対象年のカレンダーへ移動
    # 現在年から目的年まで「前年」を辿る
    current_year = datetime.now().year
    years_back = current_year - year

    for _ in range(years_back):
        soup = BeautifulSoup(r.text, 'html.parser')
        prev_link = None
        for elem in soup.find_all(onclick=True):
            if elem.get_text(strip=True) == '前年':
                oc = elem['onclick']
                m = re.search(r"doAction\(['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]\)", oc)
                if m:
                    prev_link = m.group(2)
                    break
        if not prev_link:
            print(f"  Error: Cannot navigate to year {year}")
            return []
        r = session.post(JRA_DB_URL, data={'CNAME': prev_link},
                         headers=HEADERS, timeout=15)
        r.encoding = 'shift_jis'
        time.sleep(0.5)

    # 月ページへ移動（カレンダーから月を選択）
    soup = BeautifulSoup(r.text, 'html.parser')

    # 開催月リンクを取得
    month_links = []
    for elem in soup.find_all(onclick=True):
        oc = elem['onclick']
        t = elem.get_text(strip=True)
        m = re.search(r"doAction\(['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]\)", oc)
        if m and 'pw01skl10' in m.group(2):
            month_links.append((t, m.group(2)))

    # 全月から開催日を収集
    all_race_days = []
    visited_months = set()

    for month_text, month_cname in month_links:
        # 月CNAMEからYYYYMM抽出
        m = re.search(r'pw01skl10(\d{6})', month_cname)
        if not m:
            continue
        ym = m.group(1)
        if ym[:4] != str(year):
            continue
        if ym in visited_months:
            continue
        visited_months.add(ym)

        r = session.post(JRA_DB_URL, data={'CNAME': month_cname},
                         headers=HEADERS, timeout=15)
        r.encoding = 'shift_jis'
        soup = BeautifulSoup(r.text, 'html.parser')
        time.sleep(0.3)

        # 開催日リンクを取得 (pw01srl = race day list)
        for elem in soup.find_all(onclick=True):
            oc = elem['onclick']
            t = elem.get_text(strip=True)
            m2 = re.search(r"doAction\(['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]\)", oc)
            if m2 and 'pw01srl10' in m2.group(2):
                all_race_days.append((t, m2.group(2)))

    return all_race_days


def get_all_race_day_links(session, year):
    """指定年の全開催日リンクを取得"""
    print(f"  {year}年のカレンダーを取得中...")

    # 過去レース結果ページ → 年カレンダー
    r = session.post(JRA_DB_URL, data={'CNAME': 'pw01skl00999999/B3'},
                     headers=HEADERS, timeout=15)
    r.encoding = 'shift_jis'

    current_year = datetime.now().year
    years_back = current_year - year

    for i in range(years_back):
        soup = BeautifulSoup(r.text, 'html.parser')
        prev_link = None
        for elem in soup.find_all(onclick=True):
            if elem.get_text(strip=True) == '前年':
                oc = elem['onclick']
                m = re.search(r"doAction\(['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]\)", oc)
                if m:
                    prev_link = m.group(2)
                    break
        if not prev_link:
            print(f"  Error: Cannot navigate to year {year}")
            return []
        r = session.post(JRA_DB_URL, data={'CNAME': prev_link},
                         headers=HEADERS, timeout=15)
        r.encoding = 'shift_jis'
        time.sleep(0.5)

    # 年カレンダーページから全月＋全開催日リンクを収集
    soup = BeautifulSoup(r.text, 'html.parser')
    all_race_days = []
    seen_cnames = set()

    # 現在表示月の開催日
    for elem in soup.find_all(onclick=True):
        oc = elem['onclick']
        t = elem.get_text(strip=True)
        m = re.search(r"doAction\(['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]\)", oc)
        if m and 'pw01srl10' in m.group(2):
            cname = m.group(2)
            if cname not in seen_cnames:
                all_race_days.append((t, cname))
                seen_cnames.add(cname)

    # 月リンクを取得して各月を巡回
    month_links = []
    for elem in soup.find_all(onclick=True):
        oc = elem['onclick']
        m = re.search(r"doAction\(['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]\)", oc)
        if m and 'pw01skl10' in m.group(2):
            cname = m.group(2)
            ym = re.search(r'pw01skl10(\d{6})', cname)
            if ym and ym.group(1)[:4] == str(year):
                month_links.append(cname)

    # 各月ページを巡回
    for mcname in month_links:
        r = session.post(JRA_DB_URL, data={'CNAME': mcname},
                         headers=HEADERS, timeout=15)
        r.encoding = 'shift_jis'
        soup = BeautifulSoup(r.text, 'html.parser')
        time.sleep(0.3)

        for elem in soup.find_all(onclick=True):
            oc = elem['onclick']
            t = elem.get_text(strip=True)
            m = re.search(r"doAction\(['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]\)", oc)
            if m and 'pw01srl10' in m.group(2):
                cname = m.group(2)
                if cname not in seen_cnames:
                    all_race_days.append((t, cname))
                    seen_cnames.add(cname)

    print(f"  {year}年: {len(all_race_days)}開催日")
    return all_race_days


def scrape_race_day(session, day_cname, day_label):
    """開催日の全レース配当を取得

    Args:
        session: requests.Session
        day_cname: 開催日のCNAME（pw01srl10...）
        day_label: 表示ラベル（例: '2回中山3日'）

    Returns:
        list of dict: 各レースの配当データ
    """
    # pw01srl → pw01ses に変換（全レース一覧ページ）
    ses_cname = day_cname.replace('pw01srl', 'pw01ses')

    # チェックサムの再計算（srl→sesでは変わる可能性）
    # まず開催日ページにアクセスしてses リンクを取得
    r = session.post(JRA_DB_URL, data={'CNAME': day_cname},
                     headers=HEADERS, timeout=15)
    r.encoding = 'shift_jis'
    soup = BeautifulSoup(r.text, 'html.parser')

    # "全てのレースを表示" リンクを探す
    ses_link = None
    for elem in soup.find_all(onclick=True):
        oc = elem['onclick']
        m = re.search(r"doAction\(['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]\)", oc)
        if m and 'pw01ses' in m.group(2):
            ses_link = m.group(2)
            break

    if not ses_link:
        print(f"    {day_label}: ses link not found")
        return []

    # 全レース一覧ページを取得
    r = session.post(JRA_DB_URL, data={'CNAME': ses_link},
                     headers=HEADERS, timeout=15)
    r.encoding = 'shift_jis'
    soup = BeautifulSoup(r.text, 'html.parser')

    # CNAMEから開催情報を抽出
    # pw01ses10CCYYYYKKNNYYYYMMDD/XX
    m = re.search(r'pw01ses10(\d{2})(\d{4})(\d{2})(\d{2})(\d{8})', ses_link)
    if not m:
        return []
    course_code = m.group(1)
    year = m.group(2)
    kai = m.group(3)
    nichi = m.group(4)
    race_date = m.group(5)
    course_name = COURSE_NAMES.get(course_code, course_code)

    # 配当データ抽出
    refunds = soup.find_all('div', class_='refund_area')
    if not refunds:
        # class属性が完全一致しない場合、正規表現で検索
        refunds = soup.find_all('div', class_=re.compile(r'refund_area'))

    results = []
    for race_idx, div in enumerate(refunds):
        race_num = race_idx + 1
        data = parse_refund_div(div)
        data['race_date'] = race_date
        data['course'] = course_name
        data['kai'] = kai
        data['nichi'] = nichi
        data['race_num'] = str(race_num).zfill(2)
        results.append(data)

    return results


def parse_refund_div(div):
    """refund_area divから配当データを抽出"""
    data = {
        'tansho_nums': '', 'tansho_payout': 0,
        'fukusho_nums': '', 'fukusho_payouts': '',
        'umaren_nums': '', 'umaren_payout': 0,
        'wide_nums': '', 'wide_payouts': '',
        'trio_nums': '', 'trio_payout': 0,
        'tierce_nums': '', 'tierce_payout': 0,
    }

    for dl in div.find_all('dl'):
        dt = dl.find('dt')
        if not dt:
            continue
        bet_type = dt.get_text(strip=True)
        dd = dl.find('dd')
        if not dd:
            continue

        lines = dd.find_all('div', class_='line')
        entries = []
        for line in lines:
            num_div = line.find('div', class_='num')
            yen_div = line.find('div', class_='yen')
            if num_div and yen_div:
                nums = num_div.get_text(strip=True)
                # 配当金額: カンマ・円を除去
                yen_text = yen_div.get_text(strip=True)
                yen_val = re.sub(r'[,円\s]', '', yen_text)
                try:
                    yen_val = int(yen_val)
                except ValueError:
                    yen_val = 0
                entries.append((nums, yen_val))

        if not entries:
            continue

        if '単勝' in bet_type:
            data['tansho_nums'] = entries[0][0]
            data['tansho_payout'] = entries[0][1]
        elif '複勝' in bet_type:
            data['fukusho_nums'] = '/'.join(e[0] for e in entries)
            data['fukusho_payouts'] = '/'.join(str(e[1]) for e in entries)
        elif '馬連' in bet_type:
            data['umaren_nums'] = entries[0][0]
            data['umaren_payout'] = entries[0][1]
        elif 'ワイド' in bet_type:
            data['wide_nums'] = '/'.join(e[0] for e in entries)
            data['wide_payouts'] = '/'.join(str(e[1]) for e in entries)
        elif '3連複' in bet_type or '三連複' in bet_type:
            data['trio_nums'] = entries[0][0]
            data['trio_payout'] = entries[0][1]
        elif '3連単' in bet_type or '三連単' in bet_type:
            data['tierce_nums'] = entries[0][0]
            data['tierce_payout'] = entries[0][1]

    return data


def run_scraper(year_from=2018, year_to=2025):
    """メインスクレイピング処理"""
    print(f"\n{'='*60}")
    print(f"  JRA配当スクレイパー (JRA公式DB版)")
    print(f"  対象: {year_from}年〜{year_to}年")
    print(f"{'='*60}")

    os.makedirs(DATA_DIR, exist_ok=True)

    progress = load_progress()
    completed_days = set(progress.get('completed_days', []))
    failed_days = set(progress.get('failed_days', []))

    if not progress.get('started_at'):
        progress['started_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    session = requests.Session()
    total_races = 0
    total_days = 0

    for year in range(year_from, year_to + 1):
        print(f"\n  === {year}年 ===")

        # 開催日リンクを取得
        race_days = get_all_race_day_links(session, year)
        time.sleep(1.0)

        for day_label, day_cname in race_days:
            # 既に取得済みの開催日はスキップ
            day_key = f"{year}_{day_cname}"
            if day_key in completed_days:
                continue

            total_days += 1

            try:
                results = scrape_race_day(session, day_cname, day_label)

                if results:
                    rows = []
                    for data in results:
                        row = [data.get(col, '') for col in CSV_HEADER]
                        rows.append(row)
                    append_to_csv(rows)
                    total_races += len(results)
                    completed_days.add(day_key)
                    print(f"    {day_label}: {len(results)}R取得 (累計 {total_races}R)")
                else:
                    print(f"    {day_label}: データなし")
                    failed_days.add(day_key)

            except Exception as e:
                print(f"    {day_label}: ERROR - {e}")
                failed_days.add(day_key)

            # 進捗保存（10開催日ごと）
            if total_days % 10 == 0:
                progress['completed_days'] = list(completed_days)
                progress['failed_days'] = list(failed_days)
                progress['total_races'] = total_races
                save_progress(progress)

            # リクエスト間隔
            time.sleep(2.0 + 0.5 * len(results) if results else 1.0)

    # 最終保存
    progress['completed_days'] = list(completed_days)
    progress['failed_days'] = list(failed_days)
    progress['total_races'] = total_races
    save_progress(progress)

    print(f"\n{'='*60}")
    print(f"  完了: {len(completed_days)}開催日, {total_races}レース")
    print(f"  失敗: {len(failed_days)}")
    print(f"  出力: {OUTPUT_PATH}")
    print(f"{'='*60}")


def show_status():
    """進捗状況を表示"""
    progress = load_progress()
    completed = len(progress.get('completed_days', []))
    failed = len(progress.get('failed_days', []))
    total_races = progress.get('total_races', 0)

    print(f"{'='*40}")
    print(f"  JRA配当スクレイパー 進捗")
    print(f"{'='*40}")
    print(f"  開催日: {completed}日完了, {failed}日失敗")
    print(f"  レース: {total_races}件")
    print(f"  開始: {progress.get('started_at', '-')}")
    print(f"  最終: {progress.get('last_updated', '-')}")

    if os.path.exists(OUTPUT_PATH):
        df = pd.read_csv(OUTPUT_PATH, encoding='utf-8-sig', dtype=str)
        print(f"  CSVレコード: {len(df)}")
        trio_valid = df['trio_payout'].astype(float).gt(0).sum()
        print(f"  三連複配当あり: {trio_valid}")

        # 年別集計
        if 'race_date' in df.columns and len(df) > 0:
            df['year'] = df['race_date'].str[:4]
            print(f"\n  年別:")
            for y, g in df.groupby('year'):
                print(f"    {y}: {len(g)}レース")
    print(f"{'='*40}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='JRA配当データスクレイパー (JRA公式DB版)')
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
