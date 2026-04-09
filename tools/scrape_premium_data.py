#!/usr/bin/env python
"""netkeiba Premium 一括データ取得スクリプト

Cookie認証で以下を取得:
1. 調教タイム（oikiriページ）→ data/netkeiba_training_times.csv
2. 厩舎コメント → data/netkeiba_stable_comments.csv
3. レース傾向（data_listページ）→ data/netkeiba_race_tendency.csv

Usage:
    python tools/scrape_premium_data.py                    # 2025年全レース
    python tools/scrape_premium_data.py --year 2024        # 指定年
    python tools/scrape_premium_data.py --year 2025 --month 1  # 指定月
"""
import pandas as pd
import numpy as np
import requests
import re
import os
import sys
import io
import json
import time
import argparse
from datetime import datetime
from bs4 import BeautifulSoup

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DELAY_MIN = 3
DELAY_MAX = 5
RETRY_DELAY = 30
MAX_RETRIES = 3

# Output files
TRAINING_CSV = os.path.join(DATA_DIR, 'netkeiba_training_times.csv')
COMMENT_CSV = os.path.join(DATA_DIR, 'netkeiba_stable_comments.csv')
TENDENCY_CSV = os.path.join(DATA_DIR, 'netkeiba_race_tendency.csv')

TRAINING_HEADER = [
    'race_id', 'race_date', 'umaban', 'horse_name', 'course', 'condition',
    'rider', 'time_6f', 'time_5f', 'time_4f', 'time_3f', 'time_1f',
    'intensity', 'rank', 'evaluation', 'training_date',
]
COMMENT_HEADER = ['race_id', 'race_date', 'umaban', 'horse_name', 'comment', 'score']
TENDENCY_HEADER = ['race_id', 'race_date', 'category', 'value']


def _load_session():
    env_path = os.path.join(BASE_DIR, '.env')
    if not os.path.exists(env_path):
        print("ERROR: .env not found")
        return None
    cookie_str = ''
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('NETKEIBA_COOKIE='):
                cookie_str = line.strip().split('=', 1)[1].strip('"').strip("'")
    if not cookie_str:
        print("ERROR: NETKEIBA_COOKIE not set in .env")
        return None
    session = requests.Session()
    session.headers.update(HEADERS)
    for part in cookie_str.split(';'):
        part = part.strip()
        if '=' in part:
            k, v = part.split('=', 1)
            session.cookies.set(k.strip(), v.strip())
    return session


def _get(session, url, retry=0):
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code in (400, 403, 429, 500, 502, 503):
            if retry < MAX_RETRIES:
                print(f"  {resp.status_code} - waiting {RETRY_DELAY}s (retry {retry+1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
                return _get(session, url, retry + 1)
            print(f"  FAIL {resp.status_code} after {MAX_RETRIES} retries: {url}")
            return None
        if resp.status_code == 404:
            return None
        resp.encoding = 'EUC-JP'
        return resp
    except Exception as e:
        if retry < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return _get(session, url, retry + 1)
        return None


def _append_csv(path, rows, header):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', encoding='utf-8-sig', newline='') as f:
        if write_header:
            f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(v).replace(',', '；') for v in row) + '\n')


def _target_to_netkeiba(target_rid):
    """Convert TARGET JV race_id (10-digit) to netkeiba format (12-digit).

    TARGET: CC(2)+YY(2)+K(1)+N(1)+RR(2)+UU(2)
    netkeiba: YYYY(4)+CC(2)+KK(2)+NN(2)+RR(2)
    N can be hex: A=10, B=11, C=12
    """
    rid = str(target_rid).zfill(10)
    cc = rid[0:2]
    yy = rid[2:4]
    k = rid[4:5]
    n = rid[5:6]
    rr = rid[6:8]
    n_map = {'A': '10', 'B': '11', 'C': '12'}
    n_dec = n_map.get(n, n.zfill(2))
    return f"20{yy}{cc}{k.zfill(2)}{n_dec}{rr}"


def get_race_ids_for_year(year):
    """jra_races_full.csvからnetkeibaフォーマットのレースIDを生成。"""
    csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
    df = pd.read_csv(csv_path, encoding='utf-8-sig',
                      usecols=['year', 'race_id', 'month'],
                      dtype=str, low_memory=False)
    df['year_int'] = pd.to_numeric(df['year'], errors='coerce')
    yr2 = year % 100
    df = df[df['year_int'] == yr2]

    nk_ids = set()
    for rid in df['race_id'].dropna().unique():
        nk_ids.add(_target_to_netkeiba(rid))

    print(f"  Year {year}: {len(nk_ids)} unique netkeiba race_ids")
    return sorted(nk_ids)


def _detect_course(text):
    """テキストからコース種別を判定する。"""
    if not text:
        return ''
    if '坂' in text:
        return '坂路'
    if re.search(r'[ＣC][ＷW]|ウッド', text):
        return 'CW'
    if re.search(r'[ＤD][ＰP]|ポリ', text):
        return 'ポリトラック'
    if re.match(r'^[美栗]?[ＥＢＡＤＣEBADCニ]$', text.strip()):
        return 'CW'
    if re.search(r'[美栗][ＷW]', text):
        return 'CW'
    if re.search(r'[美栗][ＥＢＡＤＣEBADCニ]', text):
        return 'CW'
    if re.search(r'芝', text):
        return '芝'
    if re.search(r'ダ[ートー]', text) or text.strip() == 'ダ':
        return 'ダート'
    return ''


_INT_MAP = {'一杯': '一杯', '強め': '強め', '馬也': '馬なり', '馬ナリ': '馬なり',
            'G前': '強め', '直一杯': '一杯', '仕掛': '強め', '末強め': '強め',
            '馬なり': '馬なり'}


def _extract_rank(row):
    """行からRank (A/B/C/D) を抽出する。"""
    for td in row.find_all("td"):
        for cls in td.get("class", []):
            if cls.startswith("Rank_"):
                rank_text = cls.replace("Rank_", "")
                if rank_text in ('A', 'B', 'C', 'D'):
                    return rank_text
    return ""


def _parse_time_text(time_text):
    """TrainingTimeData cell の生テキストから4F/3F/1Fを抽出する。"""
    t4f, t3f, t1f = 0.0, 0.0, 0.0
    if not time_text or '----' in time_text:
        return t4f, t3f, t1f
    nums = re.findall(r'(\d{2}\.\d)', time_text)
    if not nums:
        return t4f, t3f, t1f
    vals = [float(x) for x in nums]
    candidates_4f = [v for v in vals if 35 < v < 70]
    candidates_3f = [v for v in vals if 25 < v < 50]
    candidates_1f = [v for v in vals if 10 < v < 20]
    if candidates_4f:
        t4f = candidates_4f[0]
    if candidates_3f:
        for v in candidates_3f:
            if t4f == 0 or v < t4f:
                t3f = v
                break
    if candidates_1f:
        t1f = candidates_1f[-1]
    return t4f, t3f, t1f


def _parse_time_list(ul_elem):
    """TrainingTimeDataList ul要素から4F/3F/1Fを抽出する。"""
    t4f, t3f, t1f = 0.0, 0.0, 0.0
    lis = ul_elem.find_all('li')
    times = []
    for li in lis:
        t = li.get_text(strip=True)
        m = re.match(r'(\d+\.?\d*)', t)
        times.append(float(m.group(1)) if m else 0.0)
    if len(times) >= 5:
        if times[0] > 0:  # CW: 6F,5F,4F,3F,1F
            t4f, t3f, t1f = times[2], times[3], times[4]
        else:  # 坂路: -,4F,3F,2F,1F
            t4f, t3f, t1f = times[1], times[2], times[4]
    elif len(times) == 4:
        t4f, t3f, t1f = times[0], times[1], times[3]
    return t4f, t3f, t1f


def scrape_oikiri(session, race_id, race_date=''):
    """Scrape training times from oikiri page.

    Supports two HTML patterns:
      Pattern A: single row (13 cells) — horse info + training detail in one row
      Pattern B: two-row pair — 5 cells (horse) + 9 cells (detail)
    Uses same logic as scrape_training.py for robustness.
    """
    url = f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}"
    resp = _get(session, url)
    if resp is None:
        return None  # HTTP failure

    soup = BeautifulSoup(resp.text, 'html.parser')

    # Find table: try OikiriAllWrapper first, then OikiriTable regex
    wrapper = soup.find('div', class_='OikiriAllWrapper')
    if wrapper:
        table = wrapper.find('table')
    else:
        table = soup.find('table', class_=re.compile(r'Oikiri'))
    if not table:
        return []

    time_lists = soup.find_all('ul', class_='TrainingTimeDataList')

    results = []
    horse_order = []  # track horse encounter order for TrainingTimeDataList alignment
    pending_umaban = None
    pending_horse_name = ''

    for row in table.find_all('tr'):
        td_umaban = row.select_one('td.Umaban')
        cells = row.find_all('td')

        if td_umaban:
            try:
                umaban = int(td_umaban.get_text(strip=True))
            except (ValueError, TypeError):
                continue

            horse_div = row.select_one('div.Horse_Name')
            horse_name = horse_div.get_text(strip=True) if horse_div else ''

            if len(cells) >= 10:
                # Pattern A: single row with all data
                full_text = row.get_text(strip=True)

                rank = _extract_rank(row)

                # Course
                course = ''
                for td in cells:
                    ct = td.get_text(strip=True)
                    c = _detect_course(ct)
                    if c and len(ct) <= 12:
                        course = c
                        break

                # Condition
                cond = ''
                for c in ['良', '稍', '重', '不']:
                    if c in full_text:
                        cond = c
                        break

                # Rider
                rider = ''
                td_rider = row.select_one('td.Jockey') or row.select_one('td.Rider')
                if td_rider:
                    rider = td_rider.get_text(strip=True)[:10]

                # Intensity
                intensity = ''
                td_load = row.select_one('td.TrainingLoad')
                if td_load:
                    load_text = td_load.get_text(strip=True)
                    for key, val in _INT_MAP.items():
                        if key in load_text:
                            intensity = val
                            break
                if not intensity:
                    for key, val in _INT_MAP.items():
                        if key in full_text:
                            intensity = val
                            break

                # Evaluation
                td_critic = row.select_one('td.Training_Critic')
                evaluation = td_critic.get_text(strip=True) if td_critic else ''

                # Training date
                training_date = ''
                td_day = row.select_one('td.Training_Day')
                if td_day:
                    training_date = td_day.get_text(strip=True)
                if not training_date:
                    m = re.search(r'(\d{4})/(\d{1,2})/(\d{1,2})', full_text)
                    if m:
                        training_date = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"

                # Times from TrainingTimeData cell
                t4f = t3f = t1f = 0.0
                td_time = row.select_one('td.TrainingTimeData')
                if td_time:
                    t4f, t3f, t1f = _parse_time_text(td_time.get_text(strip=True))

                results.append([
                    race_id, race_date, umaban, horse_name, course, cond,
                    '', '', t4f, t3f, t1f,  # no 6f/5f from cell parse
                    rider, intensity, rank, evaluation, training_date,
                ])
                horse_order.append(len(results) - 1)
                pending_umaban = None
            else:
                # Pattern B: horse row only, wait for detail row
                pending_umaban = umaban
                pending_horse_name = horse_name

        elif pending_umaban is not None:
            # Pattern B: detail row
            full_text = row.get_text(strip=True)
            rank = _extract_rank(row)

            course = ''
            for td in cells:
                ct = td.get_text(strip=True)
                c = _detect_course(ct)
                if c and len(ct) <= 12:
                    course = c
                    break
            if not course:
                course = _detect_course(full_text)

            cond = ''
            for c in ['良', '稍', '重', '不']:
                if c in full_text:
                    cond = c
                    break

            rider = ''
            td_rider = row.select_one('td.Jockey') or row.select_one('td.Rider')
            if td_rider:
                rider = td_rider.get_text(strip=True)[:10]

            intensity = ''
            td_load = row.select_one('td.TrainingLoad')
            if td_load:
                load_text = td_load.get_text(strip=True)
                for key, val in _INT_MAP.items():
                    if key in load_text:
                        intensity = val
                        break
            if not intensity:
                for key, val in _INT_MAP.items():
                    if key in full_text:
                        intensity = val
                        break

            td_critic = row.select_one('td.Training_Critic')
            evaluation = td_critic.get_text(strip=True) if td_critic else ''

            training_date = ''
            td_day = row.select_one('td.Training_Day')
            if td_day:
                training_date = td_day.get_text(strip=True)
            if not training_date:
                m = re.search(r'(\d{4})/(\d{1,2})/(\d{1,2})', full_text)
                if m:
                    training_date = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"

            t4f = t3f = t1f = 0.0
            td_time = row.select_one('td.TrainingTimeData')
            if td_time:
                t4f, t3f, t1f = _parse_time_text(td_time.get_text(strip=True))

            results.append([
                race_id, race_date, pending_umaban, pending_horse_name, course, cond,
                '', '', t4f, t3f, t1f,
                rider, intensity, rank, evaluation, training_date,
            ])
            horse_order.append(len(results) - 1)
            pending_umaban = None

    # Fill times from TrainingTimeDataList for horses missing 4F
    for ti, res_idx in enumerate(horse_order):
        if ti >= len(time_lists):
            break
        row_data = results[res_idx]
        # row_data[8] = t4f (index after race_id, race_date, umaban, horse_name, course, cond, t6f_placeholder, t5f_placeholder)
        if row_data[8] == 0.0:  # t4f
            t4f, t3f, t1f = _parse_time_list(time_lists[ti])
            if t4f > 0:
                row_data[8] = t4f
            if t3f > 0 and row_data[9] == 0.0:
                row_data[9] = t3f
            if t1f > 0 and row_data[10] == 0.0:
                row_data[10] = t1f

    # Fix column order to match TRAINING_HEADER:
    # race_id, race_date, umaban, horse_name, course, condition,
    # rider, time_6f, time_5f, time_4f, time_3f, time_1f,
    # intensity, rank, evaluation, training_date
    fixed = []
    for r in results:
        # Current: race_id, race_date, umaban, horse_name, course, cond, '', '', t4f, t3f, t1f, rider, intensity, rank, evaluation, training_date
        fixed.append([
            r[0], r[1], r[2], r[3], r[4], r[5],  # race_id..condition
            r[11],  # rider
            r[6], r[7], r[8], r[9], r[10],  # time_6f, time_5f, time_4f, time_3f, time_1f
            r[12], r[13], r[14], r[15],  # intensity, rank, evaluation, training_date
        ])

    return fixed


def scrape_comment(session, race_id, race_date=''):
    """Scrape stable comments."""
    url = f"https://race.netkeiba.com/race/comment.html?race_id={race_id}"
    resp = _get(session, url)
    if resp is None:
        return None  # HTTP failure
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', class_=re.compile(r'Stable_Comment|Comment_Table'))
    if not table:
        return []

    from scrape_training import _score_comment

    results = []
    for row in table.find_all('tr'):
        tds = row.find_all('td')
        if len(tds) < 4:
            continue
        try:
            umaban = int(tds[1].get_text(strip=True))
        except (ValueError, TypeError):
            continue
        horse_name = tds[2].get_text(strip=True)[:20]
        comment = tds[3].get_text(strip=True)
        score = _score_comment(comment)
        results.append([race_id, race_date, umaban, horse_name, comment, score])

    return results


def scrape_tendency(session, race_id, race_date=''):
    """Scrape race tendency data from data_list page."""
    url = f"https://race.netkeiba.com/race/data_list.html?race_id={race_id}"
    resp = _get(session, url)
    if resp is None:
        return []
    soup = BeautifulSoup(resp.text, 'html.parser')

    results = []
    for table in soup.find_all('table', class_='RaceCommon_Table'):
        tds = table.find_all(['th', 'td'])
        if len(tds) >= 2:
            category = tds[0].get_text(strip=True)[:30]
            value = tds[1].get_text(strip=True)[:60]
            results.append([race_id, race_date, category, value])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--month', type=int, default=0, help='0=all months')
    parser.add_argument('--limit', type=int, default=0, help='Max races (0=unlimited)')
    args = parser.parse_args()

    print("=" * 60)
    print(f"  netkeiba Premium Data Scraper")
    print(f"  Year: {args.year}, Month: {args.month or 'all'}")
    print("=" * 60)

    session = _load_session()
    if session is None:
        return

    # Get race IDs
    race_ids = get_race_ids_for_year(args.year)
    if args.month > 0:
        csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
        df = pd.read_csv(csv_path, encoding='utf-8-sig',
                          usecols=['year', 'month', 'race_id'], dtype=str, low_memory=False)
        yr2 = args.year % 100
        df['year_int'] = pd.to_numeric(df['year'], errors='coerce')
        df['month_int'] = pd.to_numeric(df['month'], errors='coerce')
        month_rids = set()
        for rid in df[(df['year_int'] == yr2) & (df['month_int'] == args.month)]['race_id'].dropna().unique():
            nk_id = _target_to_netkeiba(rid)
            month_rids.add(nk_id)
        race_ids = sorted(set(race_ids) & month_rids)
        print(f"  Filtered to month {args.month}: {len(race_ids)} races")

    # Deduplicate: only 12-digit race IDs
    race_ids = sorted(set(rid for rid in race_ids if len(str(rid)) >= 10))

    # Check what we already have
    existing_training = set()
    if os.path.exists(TRAINING_CSV):
        try:
            edf = pd.read_csv(TRAINING_CSV, encoding='utf-8-sig', usecols=['race_id'], dtype=str)
            existing_training = set(edf['race_id'].unique())
        except Exception:
            pass
    existing_comment = set()
    if os.path.exists(COMMENT_CSV):
        try:
            edf = pd.read_csv(COMMENT_CSV, encoding='utf-8-sig', usecols=['race_id'], dtype=str)
            existing_comment = set(edf['race_id'].unique())
        except Exception:
            pass

    new_training_ids = [rid for rid in race_ids if str(rid) not in existing_training]
    new_comment_ids = [rid for rid in race_ids if str(rid) not in existing_comment]

    # Apply limit AFTER filtering existing (not before, to avoid 0-new-rows bug)
    if args.limit > 0:
        new_training_ids = new_training_ids[:args.limit]
        new_comment_ids = new_comment_ids[:args.limit]

    print(f"  Total race IDs: {len(race_ids)}")
    print(f"  New for training: {len(new_training_ids)}")
    print(f"  New for comments: {len(new_comment_ids)}")

    # Scrape all new race IDs
    all_ids = sorted(set(new_training_ids + new_comment_ids))
    total = len(all_ids)
    stats = {'training_races': 0, 'training_rows': 0, 'comment_races': 0, 'comment_rows': 0,
             'tendency_races': 0, 'errors': 0, 'http_errors': 0}
    consecutive_empty = 0

    for i, race_id in enumerate(all_ids):
        rid = str(race_id)
        pct = (i + 1) / total * 100
        print(f"\r  [{i+1}/{total} {pct:.0f}%] {rid}", end='', flush=True)

        got_data = False
        http_fail = False
        try:
            # Training
            if rid not in existing_training:
                rows = scrape_oikiri(session, rid)
                if rows is None:
                    http_fail = True
                elif rows:
                    _append_csv(TRAINING_CSV, rows, TRAINING_HEADER)
                    stats['training_races'] += 1
                    stats['training_rows'] += len(rows)
                    got_data = True

            # Comments
            if rid not in existing_comment:
                rows = scrape_comment(session, rid)
                if rows is None:
                    http_fail = True
                elif rows:
                    _append_csv(COMMENT_CSV, rows, COMMENT_HEADER)
                    stats['comment_races'] += 1
                    stats['comment_rows'] += len(rows)
                    got_data = True

            # Tendency (only first 100 races to avoid excessive requests)
            if i < 100:
                rows = scrape_tendency(session, rid)
                if rows:
                    _append_csv(TENDENCY_CSV, rows, TENDENCY_HEADER)
                    stats['tendency_races'] += 1
                    got_data = True

        except Exception as e:
            stats['errors'] += 1
            http_fail = True

        if http_fail:
            consecutive_empty += 1
        else:
            consecutive_empty = 0

        # Early abort: only on consecutive HTTP failures (not empty content)
        if consecutive_empty >= 10 and i >= 10:
            stats['http_errors'] = consecutive_empty
            print(f"\n  ABORT: {consecutive_empty} consecutive HTTP failures. Server may be down.")
            break

        # Rate limiting
        delay = DELAY_MIN + np.random.random() * (DELAY_MAX - DELAY_MIN)
        time.sleep(delay)

        # Progress every 50 races
        if (i + 1) % 50 == 0:
            print(f"\n  Progress: training={stats['training_races']}R/{stats['training_rows']}rows, "
                  f"comments={stats['comment_races']}R/{stats['comment_rows']}rows, errors={stats['errors']}")

    # Summary
    print(f"\n\n{'=' * 60}")
    print(f"  SCRAPING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Training: {stats['training_races']} races, {stats['training_rows']} rows")
    print(f"  Comments: {stats['comment_races']} races, {stats['comment_rows']} rows")
    print(f"  Tendency: {stats['tendency_races']} races")
    print(f"  Errors: {stats['errors']}")

    for path, name in [(TRAINING_CSV, 'Training'), (COMMENT_CSV, 'Comments'), (TENDENCY_CSV, 'Tendency')]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            try:
                n = len(pd.read_csv(path, encoding='utf-8-sig'))
            except Exception:
                n = '?'
            print(f"  {name}: {path} ({size/1024:.0f}KB, {n} rows)")

    try:
        from notify import send_discord
        send_discord("Premium Data取得完了",
                     f"調教: {stats['training_races']}R/{stats['training_rows']}行\n"
                     f"コメント: {stats['comment_races']}R/{stats['comment_rows']}行\nエラー: {stats['errors']}",
                     color="green" if stats['errors'] == 0 else "yellow")
    except Exception:
        pass

    return stats


if __name__ == '__main__':
    main()
