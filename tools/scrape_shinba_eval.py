#!/usr/bin/env python3
"""
netkeibaの新馬戦レース新聞ページから評価データを取得
- 厩舎コメント + 厩舎評価マーク (A/B/C)
- 調教コメント + 調教ランク (A/B/C/D)
- コメントスコア化 (-3〜+3)
"""

import sys
import io
import os
import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_CSV = os.path.join(DATA_DIR, 'netkeiba_shinba_eval.csv')

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, '.env'))
except ImportError:
    pass

_cookie = os.getenv('NETKEIBA_COOKIE', '')
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
if _cookie:
    HEADERS['Cookie'] = _cookie

REQUEST_INTERVAL = 5  # seconds

# ===== コメントスコア化 =====

POSITIVE_WORDS = [
    ('抜群', 3), ('文句なし', 3), ('申し分', 3), ('完璧', 3),
    ('素質高', 3), ('能力高', 3), ('即戦力', 3),
    ('上々', 2), ('良化', 2), ('好調', 2), ('順調', 2), ('いい動き', 2),
    ('バネ', 2), ('切れ', 2), ('しっかり', 2), ('力強', 2),
    ('成長', 2), ('好感触', 2), ('期待', 2), ('楽しみ', 2),
    ('センス', 2), ('素軽', 2), ('伸びやか', 2),
    ('まずまず', 1), ('悪くな', 1), ('水準', 1), ('真面目', 1),
    ('前向き', 1), ('問題な', 1), ('堅実', 1), ('安定', 1),
    ('プラス', 1), ('乗り込め', 1),
]

NEGATIVE_WORDS = [
    ('難しい', -1), ('課題', -1), ('使いつつ', -1), ('緩さ', -1),
    ('もう一歩', -1), ('ひと息', -1), ('一息', -1), ('物足り', -1),
    ('反応.*甘', -1), ('甘くなる', -1), ('重さ', -1),
    ('不安', -2), ('動き切れ', -2), ('心配', -2), ('厳しい', -2),
    ('伸び一息', -2), ('追って甘', -2), ('力不足', -2),
    ('どうか', -1), ('つかみ切れ', -1), ('微妙', -1),
]


def score_comment(text):
    """コメントからスコアを算出 (-3 ~ +3)"""
    if not text:
        return 0
    score = 0
    for word, val in POSITIVE_WORDS:
        if re.search(word, text):
            score += val
    for word, val in NEGATIVE_WORDS:
        if re.search(word, text):
            score += val
    return max(-3, min(3, score))


# ===== race_id変換 =====

def jv_to_netkeiba(jv_rid, nichi_num):
    """TARGET JV race_id -> netkeiba race_id
    JV format: CCYYKNRRHH (nichi may be hex A=10,B=11.. when >=10)
    netkeiba format: YYYYCCKKNNNRR (12 digits, all decimal)
    nichi_num: numeric nichi value from CSV column (avoids hex parsing)
    """
    s = str(jv_rid).zfill(10)
    course = s[0:2]       # 2-digit course code (same as netkeiba)
    year = '20' + s[2:4]  # 4-digit year
    kai = s[4:5]          # 1-digit kai
    race_num = s[6:8]     # 2-digit race_num
    return f'{year}{course}{int(kai):02d}{int(nichi_num):02d}{race_num}'


def get_shinba_race_ids():
    """jra_races_full.csvから新馬レースのnetkeiba race_idを取得"""
    csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
    df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False,
                     usecols=['year', 'month', 'day', 'course', 'kai', 'nichi',
                              'race_num', 'class_code', 'race_id', 'distance',
                              'num_horses'])
    shinba = df[(df['class_code'] == 15) & (df['year'].isin([24, 25]))]

    seen = set()
    races = []
    for _, r in shinba.drop_duplicates('race_id').iterrows():
        nk_id = jv_to_netkeiba(r['race_id'], r['nichi'])
        if nk_id not in seen:
            seen.add(nk_id)
            race_date = f"20{int(r['year']):02d}-{int(r['month']):02d}-{int(r['day']):02d}"
            races.append({
                'nk_race_id': nk_id,
                'race_date': race_date,
                'race_num': int(r['race_num']),
                'distance': int(r['distance']),
            })
    return races


# ===== スクレイピング =====

def scrape_newspaper(race_id):
    """newspaper.htmlから厩舎コメント+評価、調教コメント+ランクを取得"""
    url = f'https://race.netkeiba.com/race/newspaper.html?race_id={race_id}'
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code == 403:
        print(f'  403 Forbidden, skipping')
        return 'blocked'
    if resp.status_code != 200:
        print(f'  HTTP {resp.status_code}, skipping')
        return None  # page not available (400 etc.), not a block

    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, 'html.parser')

    # 厩舎コメント取得
    stable_data = {}  # umaban -> {comment, eval_mark}
    sc = soup.find('table', class_='Stable_Comment')
    if sc:
        for row in sc.find_all('tr')[1:]:
            cells = row.find_all('td')
            if len(cells) >= 5:
                umaban = cells[1].get_text(strip=True)
                name = cells[2].get_text(strip=True)
                comment = cells[3].get_text(strip=True)
                hyoka = cells[4]
                icon = hyoka.find('span', class_=True)
                mark = ''
                if icon:
                    for c in icon.get('class', []):
                        if c.startswith('Icon_Mark_'):
                            mark_num = c.replace('Icon_Mark_', '')
                            mark_map = {'01': 'A', '02': 'B', '03': 'C'}
                            mark = mark_map.get(mark_num, mark_num)
                stable_data[umaban] = {
                    'horse_name': name,
                    'stable_comment': comment,
                    'stable_eval': mark,
                }

    # 調教データ取得
    training_data = {}  # umaban -> {review, rank, critic}
    ot = soup.find('table', class_='OikiriTable')
    if ot:
        current_umaban = None
        for row in ot.find_all('tr')[1:]:
            cells = row.find_all('td')
            if not cells:
                continue
            cls0 = ' '.join(cells[0].get('class', []))
            if 'Waku' in cls0 and 'WakuBan' not in cls0:
                # Horse row
                if len(cells) >= 3:
                    current_umaban = cells[1].get_text(strip=True)
                    review = cells[3].get_text(strip=True) if len(cells) > 3 else ''
                    training_data[current_umaban] = {
                        'training_review': review,
                        'training_rank': '',
                        'training_critic': '',
                    }
            elif current_umaban:
                # Training detail row
                for cell in cells:
                    cls = ' '.join(cell.get('class', []))
                    if 'Rank_' in cls:
                        rank = ''
                        for c in cell.get('class', []):
                            if c.startswith('Rank_'):
                                rank = c.replace('Rank_', '')
                        if current_umaban in training_data:
                            training_data[current_umaban]['training_rank'] = rank
                    elif 'Training_Critic' in cls:
                        critic = cell.get_text(strip=True)
                        if current_umaban in training_data:
                            training_data[current_umaban]['training_critic'] = critic

    # マージ
    results = []
    all_umabans = set(list(stable_data.keys()) + list(training_data.keys()))
    for umaban in sorted(all_umabans, key=lambda x: int(x) if x.isdigit() else 999):
        sd = stable_data.get(umaban, {})
        td = training_data.get(umaban, {})
        horse_name = sd.get('horse_name', '') or td.get('horse_name', '')
        stable_comment = sd.get('stable_comment', '')
        training_review = td.get('training_review', '')

        # コメントスコア
        stable_score = score_comment(stable_comment)
        training_score = score_comment(training_review)
        combined_score = max(-3, min(3, stable_score + training_score))

        results.append({
            'race_id': race_id,
            'umaban': umaban,
            'horse_name': horse_name,
            'stable_eval': sd.get('stable_eval', ''),
            'training_rank': td.get('training_rank', ''),
            'stable_comment': stable_comment,
            'training_review': training_review,
            'training_critic': td.get('training_critic', ''),
            'stable_score': stable_score,
            'training_score': training_score,
            'comment_score': combined_score,
        })

    return results


# ===== メイン =====

def main():
    print(f'新馬評価スクレイピング開始: {datetime.now()}')

    # 既存データ読み込み
    existing_ids = set()
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV, encoding='utf-8')
        existing_ids = set(existing_df['race_id'].astype(str).unique())
        print(f'既存データ: {len(existing_ids)} races, {len(existing_df)} rows')

    # 新馬レースID取得
    races = get_shinba_race_ids()
    print(f'新馬レース総数 (2024-2025): {len(races)}')

    # 未取得分のみ
    todo = [r for r in races if r['nk_race_id'] not in existing_ids]
    print(f'未取得: {len(todo)} races')

    if not todo:
        print('全レース取得済み')
        return

    all_rows = []
    success = 0
    fail = 0
    skip_403 = 0

    for i, race in enumerate(todo):
        rid = race['nk_race_id']
        print(f'[{i+1}/{len(todo)}] {rid} ({race["race_date"]} R{race["race_num"]}) ', end='')

        try:
            result = scrape_newspaper(rid)
            if result == 'blocked':
                fail += 1
                skip_403 += 1
                if skip_403 >= 10:
                    print('\n10回連続403 → 中断')
                    break
            elif result is None:
                # HTTP 400 etc. - page doesn't exist, not a block
                fail += 1
                skip_403 = 0
            elif len(result) == 0:
                print(f'  データなし')
                fail += 1
                skip_403 = 0
            else:
                # race_dateを追加
                for row in result:
                    row['race_date'] = race['race_date']
                    row['distance'] = race['distance']
                all_rows.extend(result)
                success += 1
                skip_403 = 0
                print(f'  {len(result)}頭')
        except Exception as e:
            print(f'  ERROR: {e}')
            fail += 1

        # 定期保存 (50レースごと)
        if all_rows and (success % 50 == 0):
            _save(all_rows, existing_ids)
            print(f'  --- 中間保存: {len(all_rows)} rows ---')

        time.sleep(REQUEST_INTERVAL)

    # 最終保存
    if all_rows:
        _save(all_rows, existing_ids)

    print(f'\n=== 完了 ===')
    print(f'成功: {success}, 失敗: {fail}')
    print(f'取得行数: {len(all_rows)}')

    # 最終レポート
    if os.path.exists(OUTPUT_CSV):
        final_df = pd.read_csv(OUTPUT_CSV, encoding='utf-8')
        total_races = final_df['race_id'].nunique()
        total_horses = len(final_df)
        eval_coverage = (final_df['stable_eval'] != '').sum() / len(final_df) * 100
        rank_coverage = (final_df['training_rank'] != '').sum() / len(final_df) * 100
        print(f'\n=== 最終レポート ===')
        print(f'レース数: {total_races}')
        print(f'馬数: {total_horses}')
        print(f'厩舎評価カバレッジ: {eval_coverage:.1f}%')
        print(f'調教ランクカバレッジ: {rank_coverage:.1f}%')
        print(f'評価分布: {final_df["stable_eval"].value_counts().to_dict()}')
        print(f'ランク分布: {final_df["training_rank"].value_counts().to_dict()}')
        print(f'スコア分布: {final_df["comment_score"].value_counts().sort_index().to_dict()}')


def _save(new_rows, existing_ids):
    """既存データとマージして保存"""
    new_df = pd.DataFrame(new_rows)
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV, encoding='utf-8')
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['race_id', 'umaban'], keep='last')
    else:
        combined = new_df

    cols = ['race_id', 'race_date', 'umaban', 'horse_name', 'distance',
            'stable_eval', 'training_rank', 'stable_comment', 'training_review',
            'training_critic', 'stable_score', 'training_score', 'comment_score']
    for c in cols:
        if c not in combined.columns:
            combined[c] = ''
    combined = combined[cols]
    combined.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')


if __name__ == '__main__':
    main()
