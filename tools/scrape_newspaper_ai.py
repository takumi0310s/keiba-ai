#!/usr/bin/env python
"""
newspaper.html + shutuba.html からAI展開予測・波乱度・予測タイムを取得するスクレイパー

取得データ:
1. AI展開予測（レースラップ予測 200m刻み）
2. 各馬AI予測タイム（前半3F/後半3F）×4パターン
3. 各馬AI予測ポジション（スタート後/3コーナー/4コーナー座標）
4. 波乱度（Lv 1-5）+ 上位人気信頼度
5. AI見解テキスト + 推奨馬
6. ポジション別複勝率ヒートマップ
7. 推定ポジション有利馬

リクエスト間隔: 16秒（並行スクレイピング考慮）
"""

import os
import sys
import re
import json
import time
import csv
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DELAY = 16

def _load_cookie():
    env_path = os.path.join(BASE_DIR, '.env')
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('NETKEIBA_COOKIE='):
                val = line[len('NETKEIBA_COOKIE='):]
                return val.strip('"').strip("'")
    return None

def _make_session():
    import requests
    cookie_str = _load_cookie()
    if not cookie_str:
        return None
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://race.netkeiba.com/',
    })
    for part in cookie_str.split(';'):
        part = part.strip()
        if '=' in part:
            key, val = part.split('=', 1)
            session.cookies.set(key.strip(), val.strip())
    return session

def scrape_newspaper(session, race_id):
    """newspaper.htmlからAI展開予測データを取得"""
    from bs4 import BeautifulSoup

    url = f"https://race.netkeiba.com/race/newspaper.html?race_id={race_id}"
    resp = session.get(url, timeout=30)
    # newspaper.html is mainly UTF-8 with some EUC-JP in JS comments
    html = resp.content.decode('utf-8', errors='ignore')
    soup = BeautifulSoup(html, 'html.parser')

    result = {'race_id': race_id}

    # 1. AI予測ラップタイム (NoCheckData = 考慮なしバージョン)
    haron_table = soup.find('table', class_='Race_HaronTime')
    if haron_table:
        rows = haron_table.find_all('tr', class_=lambda c: c and 'HaronTime' in c and 'NoCheckData' in c)
        if rows:
            laps = []
            for row in rows:
                tds = row.find_all('td')
                for td in tds:
                    # Each td has "cumulative<br>lap" format → get_text produces "12.312.3"
                    parts = td.get_text(separator='|', strip=True).split('|')
                    if len(parts) >= 2:
                        laps.append({'cum': parts[0], 'lap': parts[1]})
                    elif len(parts) == 1:
                        # Try regex split: "12.312.3" → ["12.3", "12.3"]
                        nums = re.findall(r'\d+:\d+\.\d+|\d+\.\d+', parts[0])
                        if len(nums) >= 2:
                            laps.append({'cum': nums[0], 'lap': nums[1]})
            result['predicted_laps'] = laps

    # 2. 各馬AI予測タイム（前半3F/後半3F）- NoCheckData版
    predict_table = soup.find('table', class_=lambda x: x and 'PredictRap_Table' in x and 'NoCheckData' in x)
    if predict_table:
        horse_times = []
        for tr in predict_table.find_all('tr', class_='HorseList'):
            tds = tr.find_all('td')
            if len(tds) >= 4:
                horse_num = tds[0].get_text(strip=True)
                horse_name = tds[1].get_text(strip=True)
                first_3f = tds[2].get_text(strip=True)
                last_3f = tds[3].get_text(strip=True)
                horse_times.append({
                    'horse_num': int(horse_num) if horse_num.isdigit() else horse_num,
                    'horse_name': horse_name,
                    'ai_first_3f': float(first_3f) if first_3f else None,
                    'ai_last_3f': float(last_3f) if last_3f else None,
                })
        result['ai_horse_times'] = horse_times

    # 3. 各馬ポジション予測（スタート後）
    develop_img = soup.find('div', class_=lambda x: x and 'DevelopImg01' in str(x))
    if develop_img:
        positions = []
        for horse_span in develop_img.find_all('span', class_='HorseIcon'):
            style = horse_span.get('style', '')
            horse_id = horse_span.get('id', '')
            waku_span = horse_span.find('span', class_='Waku')
            name_span = horse_span.find('span', class_='HorseName')

            top_match = re.search(r'top:\s*([-\d.]+)%', style)
            left_match = re.search(r'left:\s*([-\d.]+)%', style)

            if waku_span and name_span:
                positions.append({
                    'horse_num': int(waku_span.get_text(strip=True)) if waku_span.get_text(strip=True).isdigit() else waku_span.get_text(strip=True),
                    'horse_name': name_span.get_text(strip=True),
                    'position_top': float(top_match.group(1)) if top_match else None,
                    'position_left': float(left_match.group(1)) if left_match else None,
                })
        result['ai_start_positions'] = positions

    # 4. AI見解テキスト
    opinion_section = soup.find('section', class_='DevelopOpinionArea')
    if opinion_section:
        no_check = opinion_section.find('dd', class_='NoCheckData')
        if no_check:
            result['ai_opinion'] = no_check.get_text(strip=True)

    # 5. ポジション有利馬
    pickup = soup.find('div', class_='PositionPickupHorseWrap')
    if pickup:
        advantage_horses = []
        for li in pickup.find_all('li'):
            waku = li.find('span', class_=re.compile(r'Waku\d'))
            link = li.find('a')
            if waku and link:
                advantage_horses.append({
                    'horse_num': int(waku.get_text(strip=True)) if waku.get_text(strip=True).isdigit() else waku.get_text(strip=True),
                    'horse_name': link.get_text(strip=True),
                })
        result['position_advantage_horses'] = advantage_horses

    # 6. ポジション別複勝率ヒートマップ
    heatmap = soup.find('div', class_='Position_HeatMap')
    if heatmap:
        cells = []
        lis = heatmap.find_all('li')
        positions_label = ['front', 'middle', 'rear']
        for i, li in enumerate(lis):
            divs = li.find_all('div')
            pos_label = positions_label[i] if i < len(positions_label) else f'pos{i}'
            tracks = ['inner', 'center', 'outer']
            for j, div in enumerate(divs):
                span = div.find('span')
                if span:
                    val = span.get_text(strip=True).replace('%', '')
                    track_label = tracks[j] if j < len(tracks) else f'track{j}'
                    cells.append({
                        'position': pos_label,
                        'track': track_label,
                        'win_rate': float(val) if val.replace('.', '').isdigit() else 0,
                    })
        result['position_heatmap'] = cells

    # 7. トラックバイアス
    bias_div = soup.find('div', class_='DevelopBiasTxt')
    if bias_div:
        bias_texts = [span.get_text(strip=True) for span in bias_div.find_all('span')]
        result['track_bias'] = bias_texts  # [inner, center, outer]

    # 8. AnaBest (能力/上昇度)
    ana_tables = soup.find_all('table', class_='AnaBestTable')
    for t in ana_tables:
        th = t.find('th')
        if th:
            label = th.get_text(strip=True)
            td = t.find('td')
            if td:
                horses = []
                for span in td.find_all('span', class_=re.compile(r'Waku\d')):
                    num = span.get_text(strip=True)
                    if num.isdigit():
                        horses.append(int(num))
                if label == '能力':
                    result['ana_best_ability'] = horses
                elif label == '上昇度':
                    result['ana_best_growth'] = horses

    return result

def scrape_shutuba_upset(session, race_id):
    """shutuba.htmlから波乱度・AIレース相性度を取得"""
    from bs4 import BeautifulSoup

    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = session.get(url, timeout=30)
    html = resp.content.decode('utf-8', errors='ignore')
    soup = BeautifulSoup(html, 'html.parser')

    result = {'race_id': race_id}

    # 波乱度 (Lv 1-5) - JS renders 'active' class, so parse from JS source
    upset_match = re.search(r"var\s+upsetLevel\s*=\s*parseInt\('(\d+)'\)", html)
    if upset_match:
        result['upset_level'] = int(upset_match.group(1))
    else:
        # Fallback: check for active class (Playwright-rendered HTML)
        upset_list = soup.find('ul', class_='Upset_Level_List')
        if upset_list:
            active = upset_list.find('li', class_='active')
            if active:
                result['upset_level'] = int(active.get_text(strip=True))

    # 上位人気信頼度
    turmoil = soup.find('td', class_='Turmoil_Score')
    if turmoil:
        p = turmoil.find('p')
        if p:
            span = p.find('span')
            if span:
                val = span.get_text(strip=True).replace('%', '')
                try:
                    result['top_popularity_reliability'] = float(val)
                except ValueError:
                    pass

    return result


def scrape_race_list(session, race_ids, output_dir=None):
    """複数レースのデータを一括取得"""
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, 'data')

    all_newspaper = []
    all_upset = []

    for i, race_id in enumerate(race_ids):
        print(f"[{i+1}/{len(race_ids)}] {race_id}")

        try:
            np_data = scrape_newspaper(session, race_id)
            all_newspaper.append(np_data)
            print(f"  newspaper: laps={len(np_data.get('predicted_laps', []))}, "
                  f"horse_times={len(np_data.get('ai_horse_times', []))}, "
                  f"positions={len(np_data.get('ai_start_positions', []))}")
        except Exception as e:
            print(f"  newspaper ERROR: {e}")
            all_newspaper.append({'race_id': race_id, 'error': str(e)})

        time.sleep(DELAY)

        try:
            up_data = scrape_shutuba_upset(session, race_id)
            all_upset.append(up_data)
            print(f"  shutuba: upset_lv={up_data.get('upset_level')}, "
                  f"reliability={up_data.get('top_popularity_reliability')}")
        except Exception as e:
            print(f"  shutuba ERROR: {e}")
            all_upset.append({'race_id': race_id, 'error': str(e)})

        time.sleep(DELAY)

    # Save newspaper AI data (JSON for complex nested data)
    np_path = os.path.join(output_dir, 'netkeiba_newspaper_ai.json')
    with open(np_path, 'w', encoding='utf-8') as f:
        json.dump(all_newspaper, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {np_path} ({len(all_newspaper)} races)")

    # Save upset/wave level as CSV (flat data)
    upset_path = os.path.join(output_dir, 'netkeiba_upset_level.csv')
    with open(upset_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['race_id', 'upset_level', 'top_popularity_reliability'])
        writer.writeheader()
        for d in all_upset:
            if 'error' not in d:
                writer.writerow({
                    'race_id': d['race_id'],
                    'upset_level': d.get('upset_level', ''),
                    'top_popularity_reliability': d.get('top_popularity_reliability', ''),
                })
    print(f"Saved: {upset_path} ({len([d for d in all_upset if 'error' not in d])} races)")

    # Also save horse-level AI prediction times as CSV
    times_path = os.path.join(output_dir, 'netkeiba_ai_predict_times.csv')
    with open(times_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['race_id', 'horse_num', 'horse_name', 'ai_first_3f', 'ai_last_3f'])
        writer.writeheader()
        for np_data in all_newspaper:
            for ht in np_data.get('ai_horse_times', []):
                writer.writerow({
                    'race_id': np_data['race_id'],
                    'horse_num': ht['horse_num'],
                    'horse_name': ht['horse_name'],
                    'ai_first_3f': ht.get('ai_first_3f', ''),
                    'ai_last_3f': ht.get('ai_last_3f', ''),
                })
    print(f"Saved: {times_path}")

    return all_newspaper, all_upset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--race_id', type=str, default=None, help='Single race ID')
    parser.add_argument('--date', type=str, default=None, help='Date YYYYMMDD to scrape all races')
    args = parser.parse_args()

    session = _make_session()
    if not session:
        print("ERROR: Cookie not found. Run: python tools/refresh_cookie.py")
        sys.exit(1)

    if args.race_id:
        race_ids = [args.race_id]
    else:
        # Default: test with a sample race
        race_ids = ['202605010811']

    scrape_race_list(session, race_ids)

if __name__ == '__main__':
    main()
