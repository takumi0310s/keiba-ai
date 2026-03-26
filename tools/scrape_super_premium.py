#!/usr/bin/env python
"""netkeiba スーパープレミアム限定データ取得スクリプト

取得データ:
1. AI展開予測（各馬のポジション%）→ data/netkeiba_ai_position.csv
2. 馬場指数・馬場コメント → data/netkeiba_track_index.csv
3. 調教評価（A/B/C＋短評＋ラップ）→ data/netkeiba_training_eval.csv
4. AI見解・推奨馬 → data/netkeiba_ai_opinion.csv
5. 能力/上昇度ピックアップ → data/netkeiba_ana_best.csv

Usage:
    python tools/scrape_super_premium.py                    # 2025年全レース
    python tools/scrape_super_premium.py --year 2024        # 指定年
    python tools/scrape_super_premium.py --year 2025 --month 6  # 指定月
    python tools/scrape_super_premium.py --limit 10         # テスト10件
    python tools/scrape_super_premium.py --source result    # resultページのみ
    python tools/scrape_super_premium.py --source newspaper # newspaperのみ
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
DELAY_MIN = 5
DELAY_MAX = 10
RETRY_DELAY = 30
MAX_RETRIES = 3

# Output files
POSITION_CSV = os.path.join(DATA_DIR, 'netkeiba_ai_position.csv')
TRACK_INDEX_CSV = os.path.join(DATA_DIR, 'netkeiba_track_index.csv')
TRAINING_EVAL_CSV = os.path.join(DATA_DIR, 'netkeiba_training_eval.csv')
AI_OPINION_CSV = os.path.join(DATA_DIR, 'netkeiba_ai_opinion.csv')
ANA_BEST_CSV = os.path.join(DATA_DIR, 'netkeiba_ana_best.csv')

POSITION_HEADER = ['race_id', 'umaban', 'horse_name', 'position_left_pct', 'position_top_pct', 'color_class']
TRACK_INDEX_HEADER = ['race_id', 'track_index', 'track_comment']
TRAINING_EVAL_HEADER = [
    'race_id', 'umaban', 'horse_name', 'prev_review',
    'training_date', 'training_course', 'training_condition',
    'training_rider', 'training_time_raw', 'training_position',
    'training_intensity', 'training_move', 'training_rank',
]
AI_OPINION_HEADER = ['race_id', 'pace', 'opinion_text']
ANA_BEST_HEADER = ['race_id', 'category', 'horses']


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


def _get(session, url, encoding='utf-8', retry=0):
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code in (403, 429):
            if retry < MAX_RETRIES:
                print(f"  {resp.status_code} - waiting {RETRY_DELAY}s (retry {retry+1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
                return _get(session, url, encoding, retry + 1)
            return None
        if resp.status_code == 404:
            return None
        resp.encoding = encoding
        return resp
    except Exception as e:
        if retry < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return _get(session, url, encoding, retry + 1)
        return None


def _escape_csv(val):
    """Escape value for CSV output."""
    s = str(val).replace(',', '；').replace('\n', ' ').replace('\r', '')
    return s


def _append_csv(path, rows, header):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', encoding='utf-8-sig', newline='') as f:
        if write_header:
            f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(_escape_csv(v) for v in row) + '\n')


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
        rid = str(rid).zfill(10)
        cc = rid[0:2]
        yy = rid[2:4]
        k = rid[4:5]
        n = rid[5:6]
        rr = rid[6:8]
        nk_id = f"20{yy}{cc}{k.zfill(2)}{n.zfill(2)}{rr}"
        nk_ids.add(nk_id)

    return sorted(nk_ids)


def _parse_training_detail(tr, race_id, umaban, horse_name, prev_review):
    """Parse training detail from a single TR element."""
    tds = tr.find_all(['td', 'th'])

    date_td = tr.find('td', class_='Training_Day')
    t_date = date_td.get_text(strip=True)[:16] if date_td else ''

    time_td = tr.find('td', class_='TrainingTimeData')
    t_time = time_td.get_text(strip=True)[:50] if time_td else ''

    load_td = tr.find('td', class_='TrainingLoad')
    t_intensity = load_td.get_text(strip=True)[:10] if load_td else ''

    critic_td = tr.find('td', class_='Training_Critic')
    t_move = critic_td.get_text(strip=True)[:20] if critic_td else ''

    t_rank = ''
    rank_td = tr.find('td', class_=re.compile(r'Rank_[A-D]'))
    if rank_td:
        for cls in rank_td.get('class', []):
            if cls.startswith('Rank_'):
                t_rank = cls.replace('Rank_', '')[:1]
                break
        if not t_rank:
            t_rank = rank_td.get_text(strip=True)[:5]

    # Course and condition - find by position after date
    t_course = t_cond = t_rider = t_pos = ''
    all_td_texts = [td.get_text(strip=True) for td in tds]
    date_idx = -1
    for idx_td, txt in enumerate(all_td_texts):
        if re.match(r'\d{4}/', txt):
            date_idx = idx_td
            break
    if date_idx >= 0 and date_idx + 3 < len(all_td_texts):
        t_course = all_td_texts[date_idx + 1][:10]
        t_cond = all_td_texts[date_idx + 2][:5]
        t_rider = all_td_texts[date_idx + 3][:10]

    # Position (number after training time)
    time_idx = -1
    for idx_td, td in enumerate(tds):
        if 'TrainingTimeData' in str(td.get('class', [])):
            time_idx = idx_td
            break
    if time_idx >= 0 and time_idx + 1 < len(tds):
        pos_text = tds[time_idx + 1].get_text(strip=True)
        if pos_text.isdigit():
            t_pos = pos_text

    return [
        race_id, umaban, horse_name, prev_review,
        t_date, t_course, t_cond, t_rider, t_time,
        t_pos, t_intensity, t_move, t_rank,
    ]


def scrape_newspaper(session, race_id):
    """Scrape newspaper page for AI position prediction, AI opinion, training eval, ana best."""
    url = f"https://race.netkeiba.com/race/newspaper.html?race_id={race_id}"
    resp = _get(session, url, encoding='utf-8')
    if resp is None:
        return {}, {}, [], []

    soup = BeautifulSoup(resp.text, 'html.parser')

    # 1. AI展開予測 - horse position predictions
    positions = []
    block2 = soup.find('div', class_=lambda c: c and 'AiTenkaiBlock02' in str(c)
                        and ('Dirt' in str(c) or 'Turf' in str(c) or 'AntiClockwise' in str(c) or 'Clockwise' in str(c)))
    if not block2:
        # Fallback: any AiTenkaiBlock02
        block2 = soup.find('div', class_='AiTenkaiBlock02')

    if block2:
        for icon in block2.find_all('span', class_='HorseIcon'):
            style = icon.get('style', '')
            text = icon.get_text(strip=True)
            # Parse "9ダブル" -> umaban=9
            m = re.match(r'(\d+)', text)
            if not m:
                continue
            umaban = int(m.group(1))
            horse_name = text[len(m.group(1)):]

            # Parse position from style: top:XX%;left:YY%
            left_m = re.search(r'left:\s*([\d.]+)%', style)
            top_m = re.search(r'top:\s*([\d.-]+)%', style)
            left_pct = float(left_m.group(1)) if left_m else 0.0
            top_pct = float(top_m.group(1)) if top_m else 0.0

            # Color class (pace style)
            color_cls = ''
            for c in icon.get('class', []):
                if c.startswith('Color'):
                    color_cls = c
                    break

            positions.append([race_id, umaban, horse_name, left_pct, top_pct, color_cls])

    # 2. AI見解テキスト + ペース
    ai_data = {}
    pace_div = soup.find('div', class_=lambda c: c and 'Pace_' in str(c))
    if pace_div:
        pace_text = pace_div.get_text(strip=True)
        ai_data['pace'] = pace_text  # H, M, S

    opinion_section = soup.find('section', class_='DevelopOpinionArea')
    if opinion_section:
        # Get all paragraph text
        paras = []
        for p in opinion_section.find_all('p'):
            txt = p.get_text(strip=True)
            if txt and len(txt) > 5:
                paras.append(txt)
        ai_data['opinion'] = ' '.join(paras)

    # If no DevelopOpinionArea, try finding AI勝負診断 text
    if 'opinion' not in ai_data:
        for div in soup.find_all('div', class_=re.compile('Develop')):
            txt = div.get_text(strip=True)
            if len(txt) > 20 and 'マスターコース' not in txt:
                ai_data['opinion'] = txt[:500]
                break

    # 3. 調教評価 (OikiriTable on newspaper page)
    # Two formats:
    #   A) 1-row per horse (未勝利 etc.): All data in one TR with 10+ tds
    #   B) 2-row per horse (GI etc.): Row1=馬名+前走短評(4 tds), Row2=調教詳細(10 tds)
    # Detect by checking if the TR has Training_Day td.
    training_evals = []
    oikiri = soup.find('table', class_='OikiriTable')
    if oikiri:
        all_rows = oikiri.find_all('tr', class_=re.compile(r'OikiriDataHead|HorseList'))

        i = 0
        while i < len(all_rows):
            tr = all_rows[i]
            tds = tr.find_all(['td', 'th'])

            # Check if this row has training data (date td)
            has_training = tr.find('td', class_='Training_Day') is not None

            # Format B: 2-row structure - this row has horse name only
            if not has_training:
                umaban_td = tr.find('td', class_='Umaban')
                if not umaban_td:
                    i += 1
                    continue
                try:
                    umaban = int(umaban_td.get_text(strip=True))
                except (ValueError, TypeError):
                    i += 1
                    continue

                horse_td = tr.find('td', class_=re.compile('Horse_Info'))
                horse_name = ''
                prev_review = ''
                if horse_td:
                    full_text = horse_td.get_text(strip=True)
                    m_name = re.match(r'(.+?)前走(.*)', full_text)
                    if m_name:
                        horse_name = m_name.group(1)[:20]
                        prev_review = m_name.group(2)[:100]
                    else:
                        horse_name = full_text[:20]

                # Next row should have training details
                if i + 1 < len(all_rows):
                    detail_tr = all_rows[i + 1]
                    if detail_tr.find('td', class_='Training_Day'):
                        result = _parse_training_detail(detail_tr, race_id, umaban, horse_name, prev_review)
                        if result:
                            training_evals.append(result)
                        i += 2
                        continue
                i += 1
                continue

            # Format A: 1-row structure - all data in this row
            umaban_td = tr.find('td', class_='Umaban')
            if umaban_td:
                try:
                    umaban = int(umaban_td.get_text(strip=True))
                except (ValueError, TypeError):
                    i += 1
                    continue
            else:
                try:
                    umaban = int(tds[1].get_text(strip=True))
                except (ValueError, TypeError):
                    i += 1
                    continue

            horse_td = tr.find('td', class_=re.compile('Horse_Info'))
            horse_name = ''
            prev_review = ''
            if horse_td:
                full_text = horse_td.get_text(strip=True)
                m_name = re.match(r'(.+?)前走(.*)', full_text)
                if m_name:
                    horse_name = m_name.group(1)[:20]
                    prev_review = m_name.group(2)[:100]
                else:
                    horse_name = full_text[:20]

            result = _parse_training_detail(tr, race_id, umaban, horse_name, prev_review)
            if result:
                training_evals.append(result)
            i += 1

    # 4. 能力/上昇度ピックアップ (AnaBestTable)
    ana_bests = []
    for t in soup.find_all('table', class_='AnaBestTable'):
        th = t.find('th')
        if th:
            category = th.get_text(strip=True)[:20]
            td = t.find('td')
            if td:
                horses = td.get_text(strip=True)[:100]
                ana_bests.append([race_id, category, horses])

    return positions, ai_data, training_evals, ana_bests


def scrape_result(session, race_id):
    """Scrape result page for track index and track comment."""
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    resp = _get(session, url, encoding='euc-jp')
    if resp is None:
        return {}

    soup = BeautifulSoup(resp.text, 'html.parser')

    track_data = {}

    # 馬場指数 and 馬場コメント are in separate table rows
    for table in soup.find_all('table', class_='RaceCommon_Table'):
        for tr in table.find_all('tr'):
            ths = tr.find_all('th')
            tds = tr.find_all('td')
            if ths and tds:
                key = ths[0].get_text(strip=True)
                val = tds[0].get_text(strip=True)
                if '馬場指数' in key:
                    try:
                        track_data['track_index'] = int(val)
                    except (ValueError, TypeError):
                        track_data['track_index'] = val
                elif '馬場コメント' in key:
                    track_data['track_comment'] = val[:200]

    return track_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--month', type=int, default=0, help='0=all months')
    parser.add_argument('--limit', type=int, default=0, help='Max races (0=unlimited)')
    parser.add_argument('--source', type=str, default='both',
                        choices=['both', 'newspaper', 'result'],
                        help='Which pages to scrape')
    args = parser.parse_args()

    print("=" * 60)
    print(f"  netkeiba Super Premium Data Scraper")
    print(f"  Year: {args.year}, Month: {args.month or 'all'}")
    print(f"  Source: {args.source}")
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
            rid = str(rid).zfill(10)
            nk_id = f"20{rid[2:4]}{rid[0:2]}{rid[4:5].zfill(2)}{rid[5:6].zfill(2)}{rid[6:8]}"
            month_rids.add(nk_id)
        race_ids = sorted(set(race_ids) & month_rids)
        print(f"  Filtered to month {args.month}: {len(race_ids)} races")

    race_ids = sorted(set(rid for rid in race_ids if len(str(rid)) >= 10))

    if args.limit > 0:
        race_ids = race_ids[:args.limit]

    # Check existing data
    existing_pos = set()
    if os.path.exists(POSITION_CSV):
        try:
            edf = pd.read_csv(POSITION_CSV, encoding='utf-8-sig', usecols=['race_id'], dtype=str)
            existing_pos = set(edf['race_id'].unique())
        except Exception:
            pass

    existing_track = set()
    if os.path.exists(TRACK_INDEX_CSV):
        try:
            edf = pd.read_csv(TRACK_INDEX_CSV, encoding='utf-8-sig', usecols=['race_id'], dtype=str)
            existing_track = set(edf['race_id'].unique())
        except Exception:
            pass

    existing_training = set()
    if os.path.exists(TRAINING_EVAL_CSV):
        try:
            edf = pd.read_csv(TRAINING_EVAL_CSV, encoding='utf-8-sig', usecols=['race_id'], dtype=str)
            existing_training = set(edf['race_id'].unique())
        except Exception:
            pass

    existing_opinion = set()
    if os.path.exists(AI_OPINION_CSV):
        try:
            edf = pd.read_csv(AI_OPINION_CSV, encoding='utf-8-sig', usecols=['race_id'], dtype=str)
            existing_opinion = set(edf['race_id'].unique())
        except Exception:
            pass

    # Filter to new races only
    if args.source == 'newspaper':
        new_ids = [rid for rid in race_ids if str(rid) not in existing_pos or str(rid) not in existing_training]
    elif args.source == 'result':
        new_ids = [rid for rid in race_ids if str(rid) not in existing_track]
    else:
        new_ids = [rid for rid in race_ids
                   if str(rid) not in existing_pos
                   or str(rid) not in existing_track
                   or str(rid) not in existing_training]

    total = len(new_ids)
    print(f"  Total race IDs: {len(race_ids)}")
    print(f"  Already scraped (position): {len(existing_pos)}")
    print(f"  Already scraped (track): {len(existing_track)}")
    print(f"  Already scraped (training): {len(existing_training)}")
    print(f"  New to scrape: {total}")

    if total == 0:
        print("  Nothing new to scrape.")
        return

    stats = {
        'position_races': 0, 'position_rows': 0,
        'track_races': 0,
        'training_races': 0, 'training_rows': 0,
        'opinion_races': 0,
        'ana_best_races': 0,
        'errors': 0,
    }

    for i, race_id in enumerate(new_ids):
        rid = str(race_id)
        pct = (i + 1) / total * 100
        print(f"\r  [{i+1}/{total} {pct:.0f}%] {rid}", end='', flush=True)

        try:
            # Newspaper page
            if args.source in ('both', 'newspaper'):
                if rid not in existing_pos or rid not in existing_training or rid not in existing_opinion:
                    positions, ai_data, training_evals, ana_bests = scrape_newspaper(session, rid)

                    if positions and rid not in existing_pos:
                        _append_csv(POSITION_CSV, positions, POSITION_HEADER)
                        stats['position_races'] += 1
                        stats['position_rows'] += len(positions)

                    if ai_data and rid not in existing_opinion:
                        pace = ai_data.get('pace', '')
                        opinion = ai_data.get('opinion', '')
                        if pace or opinion:
                            _append_csv(AI_OPINION_CSV, [[rid, pace, opinion]], AI_OPINION_HEADER)
                            stats['opinion_races'] += 1

                    if training_evals and rid not in existing_training:
                        _append_csv(TRAINING_EVAL_CSV, training_evals, TRAINING_EVAL_HEADER)
                        stats['training_races'] += 1
                        stats['training_rows'] += len(training_evals)

                    if ana_bests:
                        _append_csv(ANA_BEST_CSV, ana_bests, ANA_BEST_HEADER)
                        stats['ana_best_races'] += 1

                    # Rate limiting between pages
                    delay = DELAY_MIN + np.random.random() * (DELAY_MAX - DELAY_MIN)
                    time.sleep(delay)

            # Result page
            if args.source in ('both', 'result'):
                if rid not in existing_track:
                    track_data = scrape_result(session, rid)

                    if track_data:
                        track_index = track_data.get('track_index', '')
                        track_comment = track_data.get('track_comment', '')
                        _append_csv(TRACK_INDEX_CSV,
                                    [[rid, track_index, track_comment]],
                                    TRACK_INDEX_HEADER)
                        stats['track_races'] += 1

                    delay = DELAY_MIN + np.random.random() * (DELAY_MAX - DELAY_MIN)
                    time.sleep(delay)

        except Exception as e:
            stats['errors'] += 1
            print(f"\n  ERROR on {rid}: {e}")

        # Progress every 50 races
        if (i + 1) % 50 == 0:
            print(f"\n  Progress: positions={stats['position_races']}R/{stats['position_rows']}rows, "
                  f"track={stats['track_races']}R, training={stats['training_races']}R, "
                  f"opinions={stats['opinion_races']}R, errors={stats['errors']}")

    # Summary
    print(f"\n\n{'=' * 60}")
    print(f"  SCRAPING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  AI Position:     {stats['position_races']} races, {stats['position_rows']} rows")
    print(f"  Track Index:     {stats['track_races']} races")
    print(f"  Training Eval:   {stats['training_races']} races, {stats['training_rows']} rows")
    print(f"  AI Opinion:      {stats['opinion_races']} races")
    print(f"  Ana Best:        {stats['ana_best_races']} races")
    print(f"  Errors:          {stats['errors']}")

    for path, name in [
        (POSITION_CSV, 'AI Position'),
        (TRACK_INDEX_CSV, 'Track Index'),
        (TRAINING_EVAL_CSV, 'Training Eval'),
        (AI_OPINION_CSV, 'AI Opinion'),
        (ANA_BEST_CSV, 'Ana Best'),
    ]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            try:
                n = len(pd.read_csv(path, encoding='utf-8-sig'))
            except Exception:
                n = '?'
            print(f"  {name}: {path} ({size/1024:.0f}KB, {n} rows)")

    return stats


if __name__ == '__main__':
    main()
