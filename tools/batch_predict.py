"""
一括予測CLIスクリプト
指定日のJRA全レースを一括予測し、条件・買い目・推奨度を一覧表示する。

Usage:
    python tools/batch_predict.py                     # 今日
    python tools/batch_predict.py --date 20260321     # 日付指定
    python tools/batch_predict.py --date 20260321 --recommended  # 推奨のみ
"""
import pandas as pd
import numpy as np
import pickle
import json
import requests
from bs4 import BeautifulSoup
import re
import time
import os
import sys
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
INVESTMENT_PER_RACE = 700

COURSE_MAP = {
    '札幌':0,'函館':1,'福島':2,'新潟':3,'東京':4,'中山':5,'中京':6,'京都':7,'阪神':8,'小倉':9,
}
COND_MAP = {'良':0,'稍':1,'稍重':1,'重':2,'不':3,'不良':3}

CONDITION_PROFILES = {
    'A': {'bet_type':'trio','label':'8-14頭/1600m+/良~稍','investment':700,'recommended':True},
    'B': {'bet_type':'trio','label':'8-14頭/1600m+/重~不良','investment':700,'recommended':True},
    'C': {'bet_type':'trio','label':'15頭+/1600m+/良~稍','investment':700,'recommended':True},
    'D': {'bet_type':'trio','label':'1200-1400m','investment':700,'recommended':True},
    'E': {'bet_type':'umaren','label':'7頭以下','investment':700,'recommended':True},
    'X': {'bet_type':'trio','label':'15頭+/重~不良','investment':700,'recommended':True},
}


def classify_condition(num_horses, distance, condition_enc):
    heavy = condition_enc >= 2
    if num_horses <= 7:
        return 'E'
    if distance <= 1400:
        return 'D'
    if 8 <= num_horses <= 14 and distance >= 1600 and not heavy:
        return 'A'
    if 8 <= num_horses <= 14 and distance >= 1600 and heavy:
        return 'B'
    if num_horses >= 15 and distance >= 1600 and not heavy:
        return 'C'
    return 'X'


def fetch_race_list(date_str):
    """指定日のレース一覧を取得"""
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')
        races = []
        for dl in soup.select('dl.RaceList_DataList'):
            course_elem = dl.find_previous('p', class_='RaceList_DataHeader_Title')
            course = ''
            if course_elem:
                course_text = course_elem.get_text(strip=True)
                for c in COURSE_MAP:
                    if c in course_text:
                        course = c
                        break

            for dd in dl.select('dd'):
                a = dd.select_one('a')
                if a and a.get('href'):
                    m = re.search(r'race_id=(\d{12})', a['href'])
                    if m:
                        race_id = m.group(1)
                        race_num_match = re.search(r'(\d+)R', dd.get_text())
                        race_num = race_num_match.group(0) if race_num_match else ''
                        race_name = a.get_text(strip=True).replace(race_num, '').strip()
                        races.append({
                            'race_id': race_id,
                            'course': course,
                            'race_num': race_num,
                            'race_name': race_name or f'{course}{race_num}',
                        })
        return races
    except Exception as e:
        print(f"[ERROR] Race list fetch failed: {e}")
        return []


def fetch_race_info(race_id):
    """レースの基本情報を取得（距離・馬場・頭数・出走馬）"""
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = 'EUC-JP'
        soup = BeautifulSoup(resp.text, 'html.parser')

        # レース情報
        info = {'distance': 1600, 'surface': '芝', 'condition': '良', 'course': ''}
        race_data = soup.select_one('div.RaceData01') or soup.select_one('span.RaceData01')
        if race_data:
            text = race_data.get_text()
            dm = re.search(r'(\d{3,4})m', text)
            if dm:
                info['distance'] = int(dm.group(1))
            if '障' in text:
                info['surface'] = '障'
            elif 'ダ' in text:
                info['surface'] = 'ダ'
            else:
                info['surface'] = '芝'
            for c in ['不良', '重', '稍重', '稍', '良']:
                if c in text:
                    info['condition'] = c
                    break

        # 出走馬数
        horses = soup.select('tr.HorseList')
        info['num_horses'] = len(horses)

        # TOP馬名（簡易）
        horse_names = []
        for h in horses[:6]:
            name_elem = h.select_one('span.HorseName a')
            if name_elem:
                horse_names.append(name_elem.get_text(strip=True))
        info['horse_names'] = horse_names

        return info
    except Exception as e:
        return {'distance': 1600, 'surface': '芝', 'condition': '良', 'num_horses': 14, 'horse_names': []}


def calc_recommendation_score(cond_key, distance, num_horses):
    """推奨度スコア (0-100)"""
    # 基本スコア: 条件別期待ROIベース
    roi_map = {'C': 90, 'X': 88, 'B': 82, 'A': 75, 'D': 55, 'E': 50}
    score = roi_map.get(cond_key, 40)

    # 1000m以下は非推奨
    if cond_key == 'D' and distance <= 1000:
        score = 10

    # 頭数補正: 多すぎも少なすぎもNG
    if 10 <= num_horses <= 16:
        score += 5
    elif num_horses < 6:
        score -= 10

    return min(100, max(0, score))


def main():
    parser = argparse.ArgumentParser(description="KEIBA AI Batch Predict")
    parser.add_argument("--date", type=str, default=None, help="YYYYMMDD")
    parser.add_argument("--recommended", action='store_true', help="Show recommended only")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y%m%d")
    date_disp = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"

    print(f"{'=' * 70}")
    print(f"  KEIBA AI BATCH PREDICT: {date_disp}")
    print(f"{'=' * 70}")

    # Fetch race list
    print(f"\nFetching race list for {date_str}...")
    races = fetch_race_list(date_str)
    if not races:
        print("[ERROR] No races found. Check the date.")
        return

    print(f"  Found {len(races)} races")

    # Get info for each race
    results = []
    for i, race in enumerate(races):
        rid = race['race_id']
        course = race.get('course', '')
        rnum = race.get('race_num', '')
        rname = race.get('race_name', '')

        print(f"  [{i+1}/{len(races)}] {course} {rnum} {rname[:20]}...", end='', flush=True)

        info = fetch_race_info(rid)
        num_h = info['num_horses']
        distance = info['distance']
        surface = info['surface']
        condition = info['condition']
        cond_enc = COND_MAP.get(condition, 0)

        # Skip obstacle races
        if surface == '障':
            print(" (skip: obstacle)")
            continue

        cond_key = classify_condition(num_h, distance, cond_enc)
        profile = CONDITION_PROFILES.get(cond_key, {})
        bet_type = profile.get('bet_type', 'trio')
        bet_label = '三連複7点' if bet_type == 'trio' else '馬連2点'
        inv = INVESTMENT_PER_RACE
        rec_score = calc_recommendation_score(cond_key, distance, num_h)

        is_recommended = rec_score >= 50
        is_1000m_warn = cond_key == 'D' and distance <= 1000

        results.append({
            'race_id': rid,
            'course': course,
            'race_num': rnum,
            'race_name': rname,
            'distance': distance,
            'surface': surface,
            'condition': condition,
            'num_horses': num_h,
            'cond_key': cond_key,
            'bet_type': bet_label,
            'investment': inv,
            'rec_score': rec_score,
            'recommended': is_recommended,
            'warning': '1000m以下:非推奨' if is_1000m_warn else '',
        })
        print(f" [{cond_key}] {surface}{distance}m {condition} {num_h}頭 Rec:{rec_score}")
        time.sleep(0.5)

    # Sort by recommendation score (descending)
    results.sort(key=lambda x: x['rec_score'], reverse=True)

    # Filter if requested
    if args.recommended:
        results = [r for r in results if r['recommended']]

    # Display results
    print(f"\n{'=' * 70}")
    print(f"  PREDICTION SUMMARY ({len(results)} races)")
    print(f"{'=' * 70}")
    print(f"  {'#':>2} {'Rec':>3} {'Course':<5} {'Race':>4} {'Name':<14} {'S/D':>6} {'H':>2} {'Cond':>4} {'Bet':<10} {'Inv':>6}")
    print(f"  {'-' * 68}")

    total_inv = 0
    for i, r in enumerate(results):
        rec_mark = '*' if r['rec_score'] >= 75 else '+' if r['rec_score'] >= 50 else '-'
        sd = f"{r['surface']}{r['distance']}m"
        name = r['race_name'][:13]
        warn = f" {r['warning']}" if r['warning'] else ''
        print(f"  {i+1:>2} [{rec_mark}] {r['course']:<5} {r['race_num']:>4} {name:<14} {sd:>6} {r['num_horses']:>2} {r['cond_key']:>4} {r['bet_type']:<10} {r['investment']:>5}円{warn}")
        total_inv += r['investment']

    print(f"  {'-' * 68}")
    print(f"  Total investment: {total_inv:,}円 ({len(results)} races)")

    # Save CSV
    out_dir = os.path.join(BASE_DIR, 'data', 'batch_predictions')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{date_str}_batch.csv')
    pd.DataFrame(results).to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n  Saved: {out_path}")

    # Summary by condition
    print(f"\n  Condition distribution:")
    cond_counts = {}
    for r in results:
        ck = r['cond_key']
        cond_counts[ck] = cond_counts.get(ck, 0) + 1
    for ck in sorted(cond_counts.keys()):
        print(f"    {ck}: {cond_counts[ck]} races")


if __name__ == '__main__':
    main()
