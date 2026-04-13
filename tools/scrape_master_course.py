#!/usr/bin/env python
"""netkeiba Master Course データ調査・取得スクリプト

Master Course (マスターコース) 加入者向けのデータを調査・スクレイピングする。
race.netkeiba.com が HTTP 400 を返す場合（2026-04-04以降のサーバー障害）は
復旧後に再実行すること。

データターゲット:
  1. 個別ラップ (individual lap data)
  2. 3分解指数 (start/chase/finish index)
  3. トラックバイアス (track bias)
  4. ペース予測AI (pace prediction)
  5. 波乱度 (upset level)

Usage:
    # 調査モード: 1レースの全データソースを探索しHTMLを保存
    python tools/scrape_master_course.py --investigate --race_id 202605020511

    # 個別ソースのスクレイピング
    python tools/scrape_master_course.py --source laps --year 2025 --limit 10
    python tools/scrape_master_course.py --source index --year 2024 --limit 100
    python tools/scrape_master_course.py --source race_laps --year 2020-2026
    python tools/scrape_master_course.py --source bias --year 2025
    python tools/scrape_master_course.py --source pace --year 2025
    python tools/scrape_master_course.py --source upset --year 2025

    # 全ソース一括
    python tools/scrape_master_course.py --source all --year 2020-2026

    # 単一レース
    python tools/scrape_master_course.py --source laps --race_id 202605020511
    python tools/scrape_master_course.py --source index --race_id 202505010101
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
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DEBUG_DIR = os.path.join(DATA_DIR, '_debug')

HEADERS_HTTP = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Conservative rate limiting
DELAY_SECONDS = 10
RETRY_DELAY = 60
MAX_RETRIES = 3

# ---------- Output files ----------
CSV_INDIVIDUAL_LAP = os.path.join(DATA_DIR, 'netkeiba_individual_lap.csv')
CSV_MASTER_INDEX = os.path.join(DATA_DIR, 'netkeiba_master_index_mc.csv')  # 7-field schema (scrape_master_index.pyと衝突するため分離)
CSV_RACE_LAPS = os.path.join(DATA_DIR, 'netkeiba_race_laps.csv')
CSV_TRACK_BIAS = os.path.join(DATA_DIR, 'netkeiba_track_bias.csv')
CSV_PACE_PREDICTION = os.path.join(DATA_DIR, 'netkeiba_pace_prediction.csv')
CSV_UPSET_LEVEL = os.path.join(DATA_DIR, 'netkeiba_upset_level.csv')

# ---------- CSV headers ----------
HEADER_LAP = [
    'race_id', 'umaban', 'horse_name',
    'lap_1f', 'lap_2f', 'lap_3f', 'lap_4f', 'lap_5f',
    'lap_6f', 'lap_7f', 'lap_8f', 'lap_9f', 'lap_10f',
    'lap_11f', 'lap_12f', 'lap_13f', 'lap_14f', 'lap_15f',
    'lap_16f', 'lap_17f', 'lap_18f',
    'total_laps', 'last_3f_individual',
]

HEADER_MASTER_INDEX = [
    'race_id', 'umaban', 'horse_name',
    'master_total', 'master_start', 'master_chase', 'master_finish',
]

HEADER_RACE_LAPS = [
    'race_id',
    'furlong_1', 'furlong_2', 'furlong_3', 'furlong_4', 'furlong_5',
    'furlong_6', 'furlong_7', 'furlong_8', 'furlong_9', 'furlong_10',
    'furlong_11', 'furlong_12', 'furlong_13', 'furlong_14', 'furlong_15',
    'furlong_16', 'furlong_17', 'furlong_18',
    'total_furlongs', 'first_3f', 'last_3f',
]

HEADER_TRACK_BIAS = [
    'race_id', 'race_date', 'course', 'surface', 'distance',
    'bias_type', 'bias_description', 'inner_outer', 'front_rear',
]

HEADER_PACE_PREDICTION = [
    'race_id', 'umaban', 'horse_name',
    'predicted_position_3f', 'predicted_position_4c',
    'predicted_pace', 'pace_label',
]

HEADER_UPSET_LEVEL = [
    'race_id', 'race_date', 'course', 'race_num',
    'upset_level', 'upset_label', 'confidence_favorite',
]

# ---------- URL templates ----------
# Each data type has multiple URL candidates to probe
URL_CANDIDATES = {
    'laps': [
        'https://race.netkeiba.com/race/lap.html?race_id={race_id}',
        'https://race.netkeiba.com/race/result.html?race_id={race_id}',
        'https://db.netkeiba.com/race/{race_id}/',
    ],
    'index': [
        'https://race.netkeiba.com/race/speed.html?race_id={race_id}',
        'https://race.netkeiba.com/race/master_speed.html?race_id={race_id}',
        'https://race.netkeiba.com/race/speed_detail.html?race_id={race_id}',
    ],
    'bias': [
        'https://race.netkeiba.com/race/track_bias.html?race_id={race_id}',
        'https://race.netkeiba.com/race/course.html?race_id={race_id}',
        'https://race.netkeiba.com/race/data_list.html?race_id={race_id}',
        'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}',
    ],
    'pace': [
        'https://race.netkeiba.com/race/yosou.html?race_id={race_id}',
        'https://race.netkeiba.com/race/ai.html?race_id={race_id}',
        'https://race.netkeiba.com/race/pace.html?race_id={race_id}',
        'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}',
    ],
    'upset': [
        'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}',
        'https://race.netkeiba.com/race/odds.html?race_id={race_id}',
        'https://race.netkeiba.com/race/yosou.html?race_id={race_id}',
    ],
}

# Keywords to search for in HTML to identify data sections
KEYWORDS = {
    'laps': ['個別ラップ', 'LapTime', 'lap_time', 'Lap_Table', 'RapTime',
             'ラップタイム', 'IndividualLap', 'HaronTime', '通過順位'],
    'index': ['start_index', 'chase_index', 'finish_index', '3分解',
              'スタート指数', '追走指数', '上がり指数', 'SpeedIndex',
              'MasterIndex', 'master_index', 'sk__start', 'sk__chase', 'sk__finish'],
    'bias': ['TrackBias', 'track_bias', 'トラックバイアス', 'コースバイアス',
             'バイアス', 'inner', 'outer', '内有利', '外有利', '前有利', '差し有利'],
    'pace': ['PacePrediction', 'ペース予測', 'AI予測', 'ai_pace',
             'predicted_pace', '隊列予想', '展開予想', 'PaceIcon'],
    'upset': ['波乱度', 'UpsetLevel', 'upset_level', '荒れ', '堅い',
              '波乱指数', 'VolatilityIndex', 'volatility'],
}


# ===================================================================
#  Session / HTTP utilities
# ===================================================================

def _load_session():
    """Load requests session with netkeiba cookie from .env."""
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
    session.headers.update(HEADERS_HTTP)
    for part in cookie_str.split(';'):
        part = part.strip()
        if '=' in part:
            k, v = part.split('=', 1)
            session.cookies.set(k.strip(), v.strip())
    return session


def _get(session, url, retry=0):
    """GET with retry logic. Returns response or None."""
    try:
        resp = session.get(url, timeout=20)
        if resp.status_code in (400, 403, 429, 500, 502, 503):
            if retry < MAX_RETRIES:
                print(f"\n  HTTP {resp.status_code} - retry {retry+1}/{MAX_RETRIES} "
                      f"(waiting {RETRY_DELAY}s)")
                time.sleep(RETRY_DELAY)
                return _get(session, url, retry + 1)
            print(f"\n  FAIL {resp.status_code} after {MAX_RETRIES} retries: {url}")
            return None
        if resp.status_code == 404:
            return None
        resp.encoding = 'EUC-JP'
        return resp
    except requests.exceptions.Timeout:
        if retry < MAX_RETRIES:
            print(f"\n  Timeout - retry {retry+1}/{MAX_RETRIES}")
            time.sleep(RETRY_DELAY)
            return _get(session, url, retry + 1)
        return None
    except Exception as e:
        if retry < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return _get(session, url, retry + 1)
        print(f"\n  Exception: {e}")
        return None


def _append_csv(path, rows, header):
    """Append rows to CSV, writing header if file is new/empty."""
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', encoding='utf-8-sig', newline='') as f:
        if write_header:
            f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(v).replace(',', '；') for v in row) + '\n')


def _save_debug_html(race_id, page_type, url, html_text):
    """Save raw HTML to data/_debug/ for analysis."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    safe_name = url.split('?')[0].split('/')[-1].replace('.html', '')
    fname = f"{race_id}_{page_type}_{safe_name}.html"
    fpath = os.path.join(DEBUG_DIR, fname)
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(html_text)
    return fpath


# ===================================================================
#  Race ID utilities
# ===================================================================

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


def get_race_ids(year):
    """Get netkeiba race_ids from jra_races_full.csv for a given year."""
    csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        return []
    df = pd.read_csv(csv_path, encoding='utf-8-sig',
                     usecols=['year', 'race_id'], dtype=str, low_memory=False)
    yr2 = year % 100
    df['year_int'] = pd.to_numeric(df['year'], errors='coerce')
    df = df[df['year_int'] == yr2]
    nk_ids = set()
    for rid in df['race_id'].dropna().unique():
        nk_ids.add(_target_to_netkeiba(rid))
    print(f"  Year {year}: {len(nk_ids)} unique netkeiba race_ids")
    return sorted(nk_ids)


def _load_existing_race_ids(csv_path):
    """Load already-scraped race_ids from a CSV."""
    existing = set()
    if os.path.exists(csv_path):
        try:
            edf = pd.read_csv(csv_path, encoding='utf-8-sig',
                              usecols=['race_id'], dtype=str)
            existing = set(edf['race_id'].unique())
        except Exception:
            pass
    return existing


# ===================================================================
#  Probe / Investigate
# ===================================================================

def probe_page(session, race_id, page_type, url):
    """Fetch a page and return (success, status_code, soup, raw_html, findings).

    findings is a dict of keyword matches found in the HTML.
    """
    full_url = url.format(race_id=race_id)
    findings = {
        'url': full_url,
        'status': None,
        'has_content': False,
        'keyword_hits': [],
        'tables_found': 0,
        'table_classes': [],
        'interesting_divs': [],
        'data_sections': [],
    }

    resp = _get(session, full_url)
    if resp is None:
        findings['status'] = 'FAIL (no response)'
        return False, None, '', findings

    findings['status'] = resp.status_code
    html = resp.text
    soup = BeautifulSoup(html, 'html.parser')

    # Check if page has meaningful content (not just error/redirect)
    body_text = soup.get_text(strip=True) if soup.body else ''
    findings['has_content'] = len(body_text) > 200

    # Search for keywords
    html_lower = html.lower()
    for kw in KEYWORDS.get(page_type, []):
        if kw.lower() in html_lower:
            findings['keyword_hits'].append(kw)

    # Find tables
    tables = soup.find_all('table')
    findings['tables_found'] = len(tables)
    for t in tables[:10]:
        cls = t.get('class', [])
        tid = t.get('id', '')
        label = f"class={cls}" if cls else ""
        if tid:
            label += f" id={tid}"
        if label:
            findings['table_classes'].append(label.strip())

    # Find interesting divs with data-related classes/ids
    for div in soup.find_all(['div', 'section'], limit=50):
        cls_str = ' '.join(div.get('class', []))
        div_id = div.get('id', '')
        combined = f"{cls_str} {div_id}".lower()
        data_keywords = ['lap', 'index', 'bias', 'pace', 'upset', 'speed',
                         'master', 'premium', 'data', 'result', 'rap']
        for dk in data_keywords:
            if dk in combined:
                desc = f"<{div.name} class='{cls_str}' id='{div_id}'>"
                if desc not in findings['interesting_divs']:
                    findings['interesting_divs'].append(desc)
                break

    # Find data-like sections (spans/divs with numeric-heavy content)
    for elem in soup.find_all(['span', 'td'], class_=True, limit=100):
        cls_list = elem.get('class', [])
        for cls in cls_list:
            cls_l = cls.lower()
            if any(k in cls_l for k in ['lap', 'index', 'bias', 'pace',
                                         'speed', 'master', 'rap', 'haron']):
                section = f"{cls}: {elem.get_text(strip=True)[:60]}"
                if section not in findings['data_sections']:
                    findings['data_sections'].append(section)

    return True, soup, html, findings


def investigate_all(session, race_id):
    """Probe all data sources for one race and report what's available.

    Saves raw HTML to data/_debug/ and prints a comprehensive report.
    """
    print("=" * 70)
    print(f"  INVESTIGATION REPORT: race_id = {race_id}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    report = {}
    os.makedirs(DEBUG_DIR, exist_ok=True)

    for page_type, urls in URL_CANDIDATES.items():
        print(f"\n--- [{page_type.upper()}] ---")
        report[page_type] = []

        for url_template in urls:
            url_display = url_template.format(race_id=race_id)
            print(f"\n  Probing: {url_display}")

            success, soup, html, findings = probe_page(
                session, race_id, page_type, url_template
            )

            # Save HTML for debugging
            if success and html:
                fpath = _save_debug_html(race_id, page_type, url_template, html)
                findings['debug_file'] = fpath
                print(f"  Saved HTML: {os.path.basename(fpath)}")

            # Print findings
            print(f"  Status: {findings['status']}")
            print(f"  Has content: {findings['has_content']}")
            if findings['keyword_hits']:
                print(f"  KEYWORD HITS: {findings['keyword_hits']}")
            print(f"  Tables: {findings['tables_found']}")
            if findings['table_classes']:
                for tc in findings['table_classes'][:5]:
                    print(f"    - {tc}")
            if findings['interesting_divs']:
                print(f"  Interesting elements:")
                for d in findings['interesting_divs'][:8]:
                    print(f"    - {d}")
            if findings['data_sections']:
                print(f"  Data sections:")
                for ds in findings['data_sections'][:10]:
                    print(f"    - {ds}")

            report[page_type].append(findings)

            # Rate limit between probes
            time.sleep(DELAY_SECONDS)

    # Save report JSON
    report_path = os.path.join(DEBUG_DIR, f"{race_id}_investigation.json")
    # Convert to serializable
    serializable = {}
    for k, v_list in report.items():
        serializable[k] = []
        for f in v_list:
            sf = {sk: sv for sk, sv in f.items() if sk != 'soup'}
            serializable[k].append(sf)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\n\nFull report saved: {report_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for page_type, findings_list in report.items():
        has_data = any(f.get('keyword_hits') for f in findings_list)
        has_tables = any(f.get('tables_found', 0) > 0 for f in findings_list)
        status = "FOUND" if has_data else ("TABLES" if has_tables else "NOT FOUND")
        print(f"  {page_type:10s}: {status}")
        if has_data:
            for f in findings_list:
                if f.get('keyword_hits'):
                    print(f"             URL: {f['url']}")
                    print(f"             Keywords: {f['keyword_hits']}")

    return report


# ===================================================================
#  Scraper: Individual Laps
# ===================================================================

def scrape_individual_laps(session, race_id):
    """Parse individual lap data from race result or lap page.

    Tries multiple URL patterns:
    1. race.netkeiba.com/race/lap.html (dedicated lap page, may be master-only)
    2. db.netkeiba.com/race/ (result page, has lap summary and sometimes per-horse laps)
    """
    rows = []

    # Try dedicated lap page first
    url1 = f"https://race.netkeiba.com/race/lap.html?race_id={race_id}"
    resp = _get(session, url1)
    if resp and resp.status_code == 200:
        soup = BeautifulSoup(resp.text, 'html.parser')
        rows = _parse_lap_page(soup, race_id)
        if rows:
            return rows

    time.sleep(DELAY_SECONDS)

    # Fallback: db.netkeiba.com result page
    url2 = f"https://db.netkeiba.com/race/{race_id}/"
    resp = _get(session, url2)
    if resp and resp.status_code == 200:
        soup = BeautifulSoup(resp.text, 'html.parser')
        rows = _parse_db_result_laps(soup, race_id)

    return rows


def _parse_lap_page(soup, race_id):
    """Parse the dedicated lap.html page for per-horse lap times."""
    rows = []

    # Look for lap data tables
    # Pattern 1: Table with class containing 'Lap' or 'lap'
    table = soup.find('table', class_=re.compile(r'[Ll]ap|[Rr]ap|[Hh]aron'))
    if not table:
        # Pattern 2: Any table with lap-like headers
        for t in soup.find_all('table'):
            header_text = t.get_text()
            if any(kw in header_text for kw in ['1F', '2F', '3F', 'ラップ', 'Lap']):
                table = t
                break

    if not table:
        return rows

    for tr in table.find_all('tr'):
        tds = tr.find_all('td')
        if len(tds) < 3:
            continue

        # Try to extract umaban and horse_name
        umaban = ''
        horse_name = ''
        laps = []

        for i, td in enumerate(tds):
            text = td.get_text(strip=True)
            # First numeric cell is likely umaban
            if not umaban and re.match(r'^\d{1,2}$', text):
                umaban = text
                continue
            # Horse name: non-numeric, longer text
            if not horse_name and umaban and not re.match(r'^[\d.]+$', text) and len(text) > 1:
                horse_name = text
                continue
            # Lap times: numeric with decimal
            m = re.match(r'^(\d{1,2}\.\d)$', text)
            if m:
                laps.append(float(m.group(1)))

        if umaban and laps:
            row = [race_id, umaban, horse_name]
            # Pad laps to 18 slots
            for j in range(18):
                row.append(laps[j] if j < len(laps) else '')
            row.append(len(laps))  # total_laps
            # last_3f_individual: sum of last 3 laps
            if len(laps) >= 3:
                row.append(round(sum(laps[-3:]), 1))
            else:
                row.append('')
            rows.append(row)

    return rows


def _parse_db_result_laps(soup, race_id):
    """Parse lap data from db.netkeiba.com result page.

    The result page typically has:
    - Race-level lap times in a Lap_Table or similar
    - Per-horse 'last 3F' in the result table (already in jra_races_full.csv)
    - Sometimes per-horse passing positions that can be correlated
    """
    rows = []

    # Look for the race-level lap table
    # db.netkeiba has a section with race laps and corner passing
    rap_section = soup.find('table', {'summary': re.compile(r'ラップ|タイム', re.I)})
    if not rap_section:
        rap_section = soup.find('table', class_=re.compile(r'[Rr]ace_lap|[Ll]ap'))

    # Also try the result table for per-horse last3f
    result_table = soup.find('table', class_=re.compile(r'race_table_01|Result'))
    if not result_table:
        result_table = soup.find('table', {'summary': re.compile(r'レース結果', re.I)})

    if result_table:
        for tr in result_table.find_all('tr'):
            tds = tr.find_all('td')
            if len(tds) < 8:
                continue
            # Typical result table columns:
            # 着順, 枠, 馬番, 馬名, 性齢, 斤量, 騎手, タイム, 着差, 人気, 単勝, 上がり3F, ...
            try:
                umaban_text = tds[2].get_text(strip=True) if len(tds) > 2 else ''
                horse_td = tds[3] if len(tds) > 3 else None
                horse_name = ''
                if horse_td:
                    a_tag = horse_td.find('a')
                    if a_tag:
                        horse_name = a_tag.get_text(strip=True)
                    else:
                        horse_name = horse_td.get_text(strip=True)

                if not re.match(r'^\d{1,2}$', umaban_text):
                    continue

                # Look for last3f (上がり3F) - usually around column index 11-12
                last3f = ''
                for td in tds[8:]:
                    text = td.get_text(strip=True)
                    m = re.match(r'^(\d{2}\.\d)$', text)
                    if m and 30.0 < float(m.group(1)) < 45.0:
                        last3f = m.group(1)
                        break

                # Extract passing position data if available
                pass_text = ''
                for td in tds:
                    text = td.get_text(strip=True)
                    if re.match(r'^\d{1,2}-\d{1,2}', text):
                        pass_text = text
                        break

                # Build row with available data
                # No individual per-furlong laps from result page, just last3f
                row = [race_id, umaban_text, horse_name]
                # 18 empty lap slots
                for _ in range(18):
                    row.append('')
                row.append(0)  # total_laps
                row.append(last3f)  # last_3f_individual
                rows.append(row)
            except (IndexError, ValueError):
                continue

    return rows


# ===================================================================
#  Scraper: Master Index (3-component speed index)
# ===================================================================

def scrape_master_index(session, race_id):
    """Parse 3-component master speed index from db.netkeiba.com result page.

    Source: https://db.netkeiba.com/race/{race_id}/
    The result table (class="race_table_01 nk_tb_common") contains per-horse:
      - TimeIndexMasterCell01 cells (4 per horse, hidden by default):
        [0] = master_total (総合指数M)
        [1] = master_start (スタート指数)
        [2] = master_chase (追走指数)
        [3] = master_finish (上がり指数)

    Requires netkeiba premium (master course) cookie for data to be populated.

    Returns list of rows: [race_id, umaban, horse_name,
                           master_total, master_start, master_chase, master_finish]
    """
    url = f"https://db.netkeiba.com/race/{race_id}/"
    resp = _get(session, url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')
    rows = []

    # Find the main result table
    result_table = soup.find('table', class_=re.compile(r'race_table_01'))
    if not result_table:
        result_table = soup.find('table', {'summary': re.compile(r'レース結果', re.I)})
    if not result_table:
        return rows

    for tr in result_table.find_all('tr'):
        tds = tr.find_all('td')
        if len(tds) < 8:
            continue

        # Extract umaban (column index 2: 着順=0, 枠番=1, 馬番=2)
        try:
            umaban_text = tds[2].get_text(strip=True)
            if not re.match(r'^\d{1,2}$', umaban_text):
                continue
            umaban = int(umaban_text)
        except (IndexError, ValueError):
            continue

        # Extract horse name (column index 3)
        horse_name = ''
        try:
            horse_td = tds[3]
            a_tag = horse_td.find('a')
            if a_tag:
                horse_name = a_tag.get_text(strip=True)
            else:
                horse_name = horse_td.get_text(strip=True)
        except (IndexError, AttributeError):
            pass

        # Extract TimeIndexMasterCell01 values (4 cells per horse)
        master_cells = tr.find_all('td', class_=re.compile(r'TimeIndexMasterCell01'))
        if len(master_cells) < 4:
            # No master data for this horse (premium not available or no data)
            continue

        def _extract_index(td_elem):
            """Extract integer index value from a TimeIndexMasterCell01 td."""
            text = td_elem.get_text(strip=True)
            if not text:
                return ''
            m = re.match(r'^(\d+)$', text)
            return int(m.group(1)) if m else ''

        master_total = _extract_index(master_cells[0])
        master_start = _extract_index(master_cells[1])
        master_chase = _extract_index(master_cells[2])
        master_finish = _extract_index(master_cells[3])

        # Skip if all values are empty (premium not activated)
        if master_total == '' and master_start == '' and master_chase == '' and master_finish == '':
            continue

        rows.append([
            race_id, umaban, horse_name,
            master_total, master_start, master_chase, master_finish,
        ])

    return rows


def bulk_scrape_master_index(session, years, limit=0, delay=5):
    """Bulk scrape master index data for specified years.

    Args:
        session: requests session with cookies
        years: list of years (e.g. [2020, 2021, ..., 2026])
        limit: max races to scrape (0 = unlimited)
        delay: seconds between requests (default 5)
    """
    print("=" * 60)
    print(f"  Bulk Scrape: Master Index (3-component)")
    print(f"  Years: {years}")
    print(f"  Output: {CSV_MASTER_INDEX}")
    print(f"  Delay: {delay}s")
    print("=" * 60)

    all_race_ids = []
    for year in years:
        ids = get_race_ids(year)
        all_race_ids.extend(ids)

    all_race_ids = sorted(set(all_race_ids))
    print(f"  Total race_ids: {len(all_race_ids)}")

    existing = _load_existing_race_ids(CSV_MASTER_INDEX)
    race_ids = [rid for rid in all_race_ids if str(rid) not in existing]
    print(f"  Already scraped: {len(existing)}")
    print(f"  Remaining: {len(race_ids)}")

    if limit > 0:
        race_ids = race_ids[:limit]
        print(f"  Limited to: {limit}")

    if not race_ids:
        print("  Nothing to scrape.")
        return {'races': 0, 'rows': 0, 'errors': 0, 'empty': 0}

    total = len(race_ids)
    stats = {'races': 0, 'rows': 0, 'errors': 0, 'empty': 0}
    start_time = time.time()

    for i, race_id in enumerate(race_ids):
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
        print(f"\r  [{i+1}/{total} {(i+1)/total*100:.0f}%] {race_id} "
              f"({stats['races']}ok/{stats['empty']}empty/{stats['errors']}err "
              f"ETA:{eta/60:.0f}m)", end='', flush=True)

        try:
            result_rows = scrape_master_index(session, race_id)
            if result_rows:
                _append_csv(CSV_MASTER_INDEX, result_rows, HEADER_MASTER_INDEX)
                stats['races'] += 1
                stats['rows'] += len(result_rows)
            else:
                stats['empty'] += 1
        except Exception as e:
            stats['errors'] += 1
            if stats['errors'] <= 5:
                print(f"\n  Error on {race_id}: {e}")
            elif stats['errors'] == 6:
                print("\n  (Suppressing further error messages)")

        time.sleep(delay)

        if (i + 1) % 50 == 0:
            print(f"\n  Progress: {stats['races']} races / "
                  f"{stats['rows']} rows / {stats['errors']} errors / "
                  f"{stats['empty']} empty")

        if stats['errors'] > 10 and stats['races'] == 0 and stats['empty'] == 0:
            print("\n\n  ABORT: Too many errors with no successful scrapes.")
            break

    elapsed_total = time.time() - start_time
    print(f"\n\n{'=' * 60}")
    print(f"  COMPLETE: Master Index (3-component)")
    print(f"  Races scraped: {stats['races']}")
    print(f"  Rows collected: {stats['rows']}")
    print(f"  Empty (no data): {stats['empty']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Time: {elapsed_total/60:.1f} minutes")
    if os.path.exists(CSV_MASTER_INDEX):
        try:
            total_rows = len(pd.read_csv(CSV_MASTER_INDEX, encoding='utf-8-sig'))
            print(f"  CSV total rows: {total_rows}")
        except Exception:
            pass
    print("=" * 60)
    return stats


def scrape_race_laps(session, race_id):
    """Parse race-level lap times from db.netkeiba.com result page.

    Source: https://db.netkeiba.com/race/{race_id}/
    Looks for <table class="result_table_02" summary="ラップタイム">
    which contains:
      - ラップ row: "12.7 - 11.5 - 12.3 - ..." (per-furlong lap times)
      - ペース row: "12.7 - 24.2 - 36.5 - ... (first_3f-last_3f)"

    Returns list with single row:
        [race_id, f1, f2, ..., f18, total_furlongs, first_3f, last_3f]
    """
    url = f"https://db.netkeiba.com/race/{race_id}/"
    resp = _get(session, url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')

    # Find the lap time table
    lap_table = soup.find('table', {'summary': re.compile(r'ラップタイム', re.I)})
    if not lap_table:
        # Fallback: look for table with caption containing ラップ
        for t in soup.find_all('table'):
            caption = t.find('caption')
            if caption and 'ラップ' in caption.get_text():
                lap_table = t
                break
    if not lap_table:
        return []

    laps = []
    first_3f = ''
    last_3f = ''

    for tr in lap_table.find_all('tr'):
        th = tr.find('th')
        td = tr.find('td')
        if not th or not td:
            continue

        th_text = th.get_text(strip=True)
        td_text = td.get_text(strip=True)

        if 'ラップ' in th_text:
            # Parse "12.7 - 11.5 - 12.3 - 12.6 - 12.5 - 12.5 - 12.3"
            parts = re.findall(r'(\d+\.\d)', td_text)
            laps = [float(p) for p in parts]

        elif 'ペース' in th_text:
            # Parse "(36.5-37.3)" at the end for first_3f and last_3f
            m = re.search(r'\((\d+\.\d)\s*[-\u2013\u2212]\s*(\d+\.\d)\)', td_text)
            if m:
                first_3f = float(m.group(1))
                last_3f = float(m.group(2))

    if not laps:
        return []

    # Build row: race_id, furlong_1..furlong_18, total_furlongs, first_3f, last_3f
    row = [race_id]
    for j in range(18):
        row.append(laps[j] if j < len(laps) else '')
    row.append(len(laps))  # total_furlongs

    # If first_3f/last_3f not parsed from ペース row, compute from laps
    if not first_3f and len(laps) >= 3:
        first_3f = round(sum(laps[:3]), 1)
    if not last_3f and len(laps) >= 3:
        last_3f = round(sum(laps[-3:]), 1)

    row.append(first_3f)
    row.append(last_3f)

    return [row]


def bulk_scrape_race_laps(session, years, limit=0, delay=5):
    """Bulk scrape race-level lap times for specified years.

    Args:
        session: requests session with cookies
        years: list of years
        limit: max races to scrape (0 = unlimited)
        delay: seconds between requests (default 5)
    """
    print("=" * 60)
    print(f"  Bulk Scrape: Race Laps")
    print(f"  Years: {years}")
    print(f"  Output: {CSV_RACE_LAPS}")
    print(f"  Delay: {delay}s")
    print("=" * 60)

    all_race_ids = []
    for year in years:
        ids = get_race_ids(year)
        all_race_ids.extend(ids)

    all_race_ids = sorted(set(all_race_ids))
    print(f"  Total race_ids: {len(all_race_ids)}")

    existing = _load_existing_race_ids(CSV_RACE_LAPS)
    race_ids = [rid for rid in all_race_ids if str(rid) not in existing]
    print(f"  Already scraped: {len(existing)}")
    print(f"  Remaining: {len(race_ids)}")

    if limit > 0:
        race_ids = race_ids[:limit]
        print(f"  Limited to: {limit}")

    if not race_ids:
        print("  Nothing to scrape.")
        return {'races': 0, 'rows': 0, 'errors': 0, 'empty': 0}

    total = len(race_ids)
    stats = {'races': 0, 'rows': 0, 'errors': 0, 'empty': 0}
    start_time = time.time()

    for i, race_id in enumerate(race_ids):
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
        print(f"\r  [{i+1}/{total} {(i+1)/total*100:.0f}%] {race_id} "
              f"({stats['races']}ok/{stats['empty']}empty/{stats['errors']}err "
              f"ETA:{eta/60:.0f}m)", end='', flush=True)

        try:
            result_rows = scrape_race_laps(session, race_id)
            if result_rows:
                _append_csv(CSV_RACE_LAPS, result_rows, HEADER_RACE_LAPS)
                stats['races'] += 1
                stats['rows'] += len(result_rows)
            else:
                stats['empty'] += 1
        except Exception as e:
            stats['errors'] += 1
            if stats['errors'] <= 5:
                print(f"\n  Error on {race_id}: {e}")
            elif stats['errors'] == 6:
                print("\n  (Suppressing further error messages)")

        time.sleep(delay)

        if (i + 1) % 50 == 0:
            print(f"\n  Progress: {stats['races']} races / "
                  f"{stats['rows']} rows / {stats['errors']} errors / "
                  f"{stats['empty']} empty")

        if stats['errors'] > 10 and stats['races'] == 0 and stats['empty'] == 0:
            print("\n\n  ABORT: Too many errors with no successful scrapes.")
            break

    elapsed_total = time.time() - start_time
    print(f"\n\n{'=' * 60}")
    print(f"  COMPLETE: Race Laps")
    print(f"  Races scraped: {stats['races']}")
    print(f"  Rows collected: {stats['rows']}")
    print(f"  Empty (no data): {stats['empty']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Time: {elapsed_total/60:.1f} minutes")
    if os.path.exists(CSV_RACE_LAPS):
        try:
            total_rows = len(pd.read_csv(CSV_RACE_LAPS, encoding='utf-8-sig'))
            print(f"  CSV total rows: {total_rows}")
        except Exception:
            pass
    print("=" * 60)
    return stats


# ===================================================================
#  Scraper: Track Bias
# ===================================================================

def scrape_track_bias(session, race_id):
    """Parse track bias data from various pages.

    Track bias may appear in:
    - Dedicated track_bias.html page
    - data_list.html (race tendency page)
    - shutuba.html (within course info section)
    """
    rows = []

    # Try dedicated page
    for url_template in URL_CANDIDATES['bias']:
        url = url_template.format(race_id=race_id)
        resp = _get(session, url)
        if resp is None:
            time.sleep(DELAY_SECONDS)
            continue

        soup = BeautifulSoup(resp.text, 'html.parser')
        html_text = resp.text.lower()

        # Look for bias keywords
        has_bias = any(kw.lower() in html_text for kw in KEYWORDS['bias'])
        if not has_bias:
            time.sleep(DELAY_SECONDS)
            continue

        # Extract race metadata
        race_date = ''
        course = ''
        surface = ''
        distance = ''

        # Try to get race info from header
        race_info = soup.find(class_=re.compile(r'RaceData|Race_Info|racedata'))
        if race_info:
            info_text = race_info.get_text()
            # Surface/distance
            m = re.search(r'(芝|ダ[ートー]*)\s*(\d{3,4})m', info_text)
            if m:
                surface = m.group(1)
                distance = m.group(2)
            # Course name
            m2 = re.search(r'(東京|中山|阪神|京都|中京|小倉|新潟|札幌|函館|福島)', info_text)
            if m2:
                course = m2.group(1)

        # Look for bias indicators in various formats
        bias_elements = []

        # Pattern 1: Explicit bias sections
        for elem in soup.find_all(['div', 'span', 'p', 'td'],
                                   class_=re.compile(r'[Bb]ias|バイアス', re.I)):
            bias_elements.append(elem.get_text(strip=True))

        # Pattern 2: Inner/outer advantage text
        for text_elem in soup.find_all(string=re.compile(r'内有利|外有利|前有利|差し有利|フラット')):
            bias_elements.append(text_elem.strip())

        # Pattern 3: Look for structured bias data in tables
        for table in soup.find_all('table'):
            table_text = table.get_text()
            if any(kw in table_text for kw in ['バイアス', 'bias', '内枠', '外枠', '脚質']):
                for tr in table.find_all('tr'):
                    row_text = tr.get_text(strip=True)
                    if row_text and len(row_text) < 200:
                        bias_elements.append(row_text)

        # Determine bias summary
        inner_outer = ''
        front_rear = ''
        full_text = ' '.join(bias_elements)
        if '内有利' in full_text:
            inner_outer = 'inner'
        elif '外有利' in full_text:
            inner_outer = 'outer'
        elif 'フラット' in full_text:
            inner_outer = 'flat'

        if '前有利' in full_text or '逃げ有利' in full_text:
            front_rear = 'front'
        elif '差し有利' in full_text or '追い込み' in full_text:
            front_rear = 'closer'

        if bias_elements:
            bias_desc = ' | '.join(bias_elements[:5])
            rows.append([
                race_id, race_date, course, surface, distance,
                'scraped', bias_desc[:200], inner_outer, front_rear,
            ])
            break  # Found data, no need to try more URLs

        time.sleep(DELAY_SECONDS)

    return rows


# ===================================================================
#  Scraper: Pace Prediction AI
# ===================================================================

def scrape_pace_prediction(session, race_id):
    """Parse AI pace prediction from yosou/pace pages.

    Look for predicted running positions and pace classification.
    """
    rows = []

    for url_template in URL_CANDIDATES['pace']:
        url = url_template.format(race_id=race_id)
        resp = _get(session, url)
        if resp is None:
            time.sleep(DELAY_SECONDS)
            continue

        soup = BeautifulSoup(resp.text, 'html.parser')
        html_text = resp.text.lower()

        has_pace = any(kw.lower() in html_text for kw in KEYWORDS['pace'])
        if not has_pace:
            time.sleep(DELAY_SECONDS)
            continue

        # Pattern 1: Pace icon/label in shutuba page
        # Look for pace indicators per horse
        pace_label_overall = ''
        pace_section = soup.find(class_=re.compile(r'[Pp]ace|展開'))
        if pace_section:
            pace_text = pace_section.get_text(strip=True)
            if 'ハイ' in pace_text or 'H' in pace_text:
                pace_label_overall = 'H'
            elif 'スロー' in pace_text or 'S' in pace_text:
                pace_label_overall = 'S'
            elif 'ミドル' in pace_text or 'M' in pace_text:
                pace_label_overall = 'M'

        # Pattern 2: Per-horse predicted positions
        # Look in shutuba table or dedicated prediction section
        for table in soup.find_all('table'):
            table_classes = ' '.join(table.get('class', []))
            if not any(kw in table_classes.lower() for kw in
                       ['shutuba', 'result', 'race', 'yosou', 'pace']):
                if not any(kw in table.get_text()[:200].lower()
                           for kw in ['馬番', '馬名', '予想', '展開']):
                    continue

            for tr in table.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) < 3:
                    continue

                umaban = ''
                horse_name = ''
                pred_3f = ''
                pred_4c = ''
                pred_pace = ''

                for td in tds:
                    text = td.get_text(strip=True)
                    cls_str = ' '.join(td.get('class', []))

                    # Umaban
                    if not umaban and re.match(r'^\d{1,2}$', text):
                        umaban = text
                    # Horse name
                    elif not horse_name and umaban and len(text) > 1 and not re.match(r'^[\d.]+$', text):
                        a_tag = td.find('a')
                        if a_tag:
                            horse_name = a_tag.get_text(strip=True)
                        elif len(text) <= 20:
                            horse_name = text
                    # Pace position keywords in class
                    if 'pace' in cls_str.lower() or '展開' in cls_str:
                        pred_pace = text

                if umaban and (horse_name or pred_pace):
                    rows.append([
                        race_id, umaban, horse_name,
                        pred_3f, pred_4c, pred_pace,
                        pace_label_overall,
                    ])

            if rows:
                break

        if rows:
            break  # Found data
        time.sleep(DELAY_SECONDS)

    return rows


# ===================================================================
#  Scraper: Upset Level (波乱度)
# ===================================================================

def scrape_upset_level(session, race_id):
    """Parse upset level / volatility indicator.

    This may appear in shutuba page, odds page, or yosou page
    as a visual indicator or numeric value.
    """
    rows = []

    for url_template in URL_CANDIDATES['upset']:
        url = url_template.format(race_id=race_id)
        resp = _get(session, url)
        if resp is None:
            time.sleep(DELAY_SECONDS)
            continue

        soup = BeautifulSoup(resp.text, 'html.parser')
        html_text = resp.text

        has_upset = any(kw.lower() in html_text.lower() for kw in KEYWORDS['upset'])
        if not has_upset:
            time.sleep(DELAY_SECONDS)
            continue

        # Extract race metadata
        race_date = ''
        course = ''
        race_num = ''

        # Try extracting from race_id: YYYYCCKKNNRR
        if len(race_id) == 12:
            race_num = str(int(race_id[10:12]))

        # Look for upset indicators
        upset_level = ''
        upset_label = ''
        confidence_fav = ''

        # Pattern 1: Explicit upset level section
        for elem in soup.find_all(['div', 'span', 'p', 'td'],
                                   class_=re.compile(r'[Uu]pset|波乱|荒れ|堅')):
            text = elem.get_text(strip=True)
            if text:
                upset_label = text[:50]
                # Try to extract numeric level
                m = re.search(r'(\d+)', text)
                if m:
                    upset_level = m.group(1)

        # Pattern 2: Look for visual indicators (stars, bars, icons)
        for elem in soup.find_all(['img', 'span', 'div']):
            alt = elem.get('alt', '')
            title = elem.get('title', '')
            combined = f"{alt} {title}".lower()
            if '波乱' in combined or '荒れ' in combined or '堅い' in combined:
                upset_label = combined.strip()[:50]
                # Count stars or extract level
                stars = combined.count('★') or combined.count('star')
                if stars:
                    upset_level = str(stars)

        # Pattern 3: Look in text for expressions
        body_text = soup.get_text()
        for pattern in [r'波乱度[：:\s]*(\d+)', r'荒れ度[：:\s]*(\d+)',
                        r'堅い度[：:\s]*(\d+)', r'配当期待[：:\s]*(\d+)']:
            m = re.search(pattern, body_text)
            if m:
                upset_level = m.group(1)
                break

        # Pattern 4: Favorite confidence (1番人気信頼度)
        m = re.search(r'1番人気.*?(\d{1,3})%', body_text)
        if m:
            confidence_fav = m.group(1)

        if upset_level or upset_label or confidence_fav:
            rows.append([
                race_id, race_date, course, race_num,
                upset_level, upset_label, confidence_fav,
            ])
            break

        time.sleep(DELAY_SECONDS)

    return rows


# ===================================================================
#  Bulk scrape
# ===================================================================

SOURCE_MAP = {
    'laps': {
        'func': scrape_individual_laps,
        'csv': CSV_INDIVIDUAL_LAP,
        'header': HEADER_LAP,
        'label': 'Individual Laps',
    },
    'index': {
        'func': scrape_master_index,
        'csv': CSV_MASTER_INDEX,
        'header': HEADER_MASTER_INDEX,
        'label': 'Master Index (3-component)',
    },
    'race_laps': {
        'func': scrape_race_laps,
        'csv': CSV_RACE_LAPS,
        'header': HEADER_RACE_LAPS,
        'label': 'Race Laps (per-furlong)',
    },
    'bias': {
        'func': scrape_track_bias,
        'csv': CSV_TRACK_BIAS,
        'header': HEADER_TRACK_BIAS,
        'label': 'Track Bias',
    },
    'pace': {
        'func': scrape_pace_prediction,
        'csv': CSV_PACE_PREDICTION,
        'header': HEADER_PACE_PREDICTION,
        'label': 'Pace Prediction',
    },
    'upset': {
        'func': scrape_upset_level,
        'csv': CSV_UPSET_LEVEL,
        'header': HEADER_UPSET_LEVEL,
        'label': 'Upset Level',
    },
}


def bulk_scrape(session, source, years, limit=0):
    """Bulk scrape one data type for specified years.

    Args:
        session: requests session with cookies
        source: one of 'laps', 'index', 'bias', 'pace', 'upset'
        years: list of years to scrape
        limit: max races to scrape (0 = unlimited)
    """
    cfg = SOURCE_MAP.get(source)
    if not cfg:
        print(f"ERROR: Unknown source '{source}'. "
              f"Available: {list(SOURCE_MAP.keys())}")
        return

    print("=" * 60)
    print(f"  Bulk Scrape: {cfg['label']}")
    print(f"  Years: {years}")
    print(f"  Output: {cfg['csv']}")
    print("=" * 60)

    # Gather all race_ids for the specified years
    all_race_ids = []
    for year in years:
        ids = get_race_ids(year)
        all_race_ids.extend(ids)

    # Remove duplicates and sort
    all_race_ids = sorted(set(all_race_ids))
    print(f"  Total race_ids: {len(all_race_ids)}")

    # Skip already scraped
    existing = _load_existing_race_ids(cfg['csv'])
    race_ids = [rid for rid in all_race_ids if str(rid) not in existing]
    print(f"  Already scraped: {len(existing)}")
    print(f"  Remaining: {len(race_ids)}")

    if limit > 0:
        race_ids = race_ids[:limit]
        print(f"  Limited to: {limit}")

    if not race_ids:
        print("  Nothing to scrape.")
        return

    total = len(race_ids)
    stats = {'races': 0, 'rows': 0, 'errors': 0, 'empty': 0}
    start_time = time.time()

    for i, race_id in enumerate(race_ids):
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
        print(f"\r  [{i+1}/{total} {(i+1)/total*100:.0f}%] {race_id} "
              f"({stats['races']}ok/{stats['empty']}empty/{stats['errors']}err "
              f"ETA:{eta/60:.0f}m)", end='', flush=True)

        try:
            result_rows = cfg['func'](session, race_id)
            if result_rows:
                _append_csv(cfg['csv'], result_rows, cfg['header'])
                stats['races'] += 1
                stats['rows'] += len(result_rows)
            else:
                stats['empty'] += 1
        except Exception as e:
            stats['errors'] += 1
            if stats['errors'] <= 5:
                print(f"\n  Error on {race_id}: {e}")
            elif stats['errors'] == 6:
                print("\n  (Suppressing further error messages)")

        # Rate limit
        time.sleep(DELAY_SECONDS)

        # Progress report every 50 races
        if (i + 1) % 50 == 0:
            print(f"\n  Progress: {stats['races']} races / "
                  f"{stats['rows']} rows / {stats['errors']} errors / "
                  f"{stats['empty']} empty")

        # Abort on too many consecutive errors
        if stats['errors'] > 10 and stats['races'] == 0 and stats['empty'] == 0:
            print("\n\n  ABORT: Too many errors with no successful scrapes.")
            print("  This may indicate the server is down or authentication failed.")
            print("  Try --investigate first to check page availability.")
            break

    elapsed_total = time.time() - start_time

    print(f"\n\n{'=' * 60}")
    print(f"  COMPLETE: {cfg['label']}")
    print(f"  Races scraped: {stats['races']}")
    print(f"  Rows collected: {stats['rows']}")
    print(f"  Empty (no data): {stats['empty']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Time: {elapsed_total/60:.1f} minutes")

    if os.path.exists(cfg['csv']):
        try:
            total_rows = len(pd.read_csv(cfg['csv'], encoding='utf-8-sig'))
            print(f"  CSV total rows: {total_rows}")
        except Exception:
            pass

    print("=" * 60)

    return stats


# ===================================================================
#  CLI
# ===================================================================

def parse_year_range(year_str):
    """Parse year string like '2025' or '2020-2026' into list of years."""
    if '-' in year_str:
        parts = year_str.split('-')
        start = int(parts[0])
        end = int(parts[1])
        return list(range(start, end + 1))
    return [int(year_str)]


def main():
    parser = argparse.ArgumentParser(
        description="netkeiba Master Course Data Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/scrape_master_course.py --investigate --race_id 202605020511
  python tools/scrape_master_course.py --source laps --year 2025 --limit 10
  python tools/scrape_master_course.py --source index --year 2024 --limit 100
  python tools/scrape_master_course.py --source race_laps --year 2020-2026
  python tools/scrape_master_course.py --source all --year 2020-2026
        """,
    )
    parser.add_argument('--investigate', action='store_true',
                        help='Investigation mode: probe all URLs for one race')
    parser.add_argument('--source', type=str, default='',
                        choices=['laps', 'index', 'race_laps', 'bias', 'pace', 'upset', 'all', ''],
                        help='Data source to scrape')
    parser.add_argument('--year', type=str, default='2025',
                        help='Year or year range (e.g., 2025 or 2020-2026)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Max races to scrape (0=unlimited)')
    parser.add_argument('--race_id', type=str, default='',
                        help='Single race_id to scrape/investigate')

    args = parser.parse_args()

    print("=" * 60)
    print("  netkeiba Master Course Data Scraper")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load session
    session = _load_session()
    if session is None:
        sys.exit(1)

    # Investigation mode
    if args.investigate:
        if not args.race_id:
            print("ERROR: --investigate requires --race_id")
            sys.exit(1)
        investigate_all(session, args.race_id)
        return

    # Single race_id mode
    if args.race_id and args.source:
        source = args.source
        if source == 'all':
            sources = list(SOURCE_MAP.keys())
        else:
            sources = [source]

        for src in sources:
            cfg = SOURCE_MAP[src]
            print(f"\n--- Scraping {cfg['label']} for {args.race_id} ---")
            try:
                result_rows = cfg['func'](session, args.race_id)
                if result_rows:
                    _append_csv(cfg['csv'], result_rows, cfg['header'])
                    print(f"  OK: {len(result_rows)} rows -> {cfg['csv']}")
                    for row in result_rows[:3]:
                        print(f"    {row[:5]}...")
                else:
                    print("  No data found.")
            except Exception as e:
                print(f"  Error: {e}")
            time.sleep(DELAY_SECONDS)
        return

    # Bulk scrape mode
    if not args.source:
        print("ERROR: --source is required (laps/index/race_laps/bias/pace/upset/all)")
        print("  Or use --investigate --race_id XXXX for investigation mode")
        sys.exit(1)

    years = parse_year_range(args.year)

    if args.source == 'all':
        sources = list(SOURCE_MAP.keys())
    else:
        sources = [args.source]

    all_stats = {}
    for src in sources:
        stats = bulk_scrape(session, src, years, limit=args.limit)
        all_stats[src] = stats
        if len(sources) > 1:
            print()  # Blank line between sources

    # Final summary for multi-source
    if len(sources) > 1:
        print("\n" + "=" * 60)
        print("  OVERALL SUMMARY")
        print("=" * 60)
        for src, stats in all_stats.items():
            if stats:
                print(f"  {src:8s}: {stats['races']} races, "
                      f"{stats['rows']} rows, {stats['errors']} errors")
            else:
                print(f"  {src:8s}: skipped/failed")
        print("=" * 60)

    # Discord notification
    try:
        sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
        from notify_done import send as notify_send
        total_rows = sum(s.get('rows', 0) for s in all_stats.values() if s)
        total_errors = sum(s.get('errors', 0) for s in all_stats.values() if s)
        notify_send(
            "Master Course取得完了",
            f"ソース: {', '.join(sources)}\n"
            f"年: {args.year}\n"
            f"行数: {total_rows} / エラー: {total_errors}",
            color="green" if total_errors == 0 else "yellow",
        )
    except Exception:
        pass


if __name__ == '__main__':
    from tools.scraper_guard import check_scraping_allowed; check_scraping_allowed()
    main()
