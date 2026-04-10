#!/usr/bin/env python
"""
残存データ9項目の取得可否調査スクリプト
Playwrightでnetkeibaにログイン状態でアクセスし、HTML構造を確認する。
リクエスト間隔15秒以上（並行スクレイピングとの干渉回避）
"""

import os
import sys
import json
import time
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DELAY = 16  # 15秒以上
RACE_ID = "202605010811"  # サンプルレースID

def _load_cookie():
    env_path = os.path.join(BASE_DIR, '.env')
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('NETKEIBA_COOKIE='):
                val = line[len('NETKEIBA_COOKIE='):]
                return val.strip('"').strip("'")
    return None

def _setup_browser_context(playwright, cookie_str):
    """Launch browser and set cookies."""
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
    # Set cookies from string
    cookies = []
    for part in cookie_str.split(';'):
        part = part.strip()
        if '=' in part:
            key, val = part.split('=', 1)
            cookies.append({
                'name': key.strip(),
                'value': val.strip(),
                'domain': '.netkeiba.com',
                'path': '/',
            })
    context.add_cookies(cookies)
    return browser, context

def save_html_debug(html, name):
    """Save HTML for debugging."""
    debug_dir = os.path.join(BASE_DIR, 'data', '_debug')
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, f'{name}.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  [DEBUG] HTML saved: {path}")

def investigate_data_analysis(page, results):
    """1. データ分析43種"""
    print("\n=== 1. データ分析43種 ===")
    url = f"https://race.netkeiba.com/race/data_list.html?race_id={RACE_ID}"
    print(f"  URL: {url}")

    try:
        page.goto(url, wait_until='domcontentloaded', timeout=30000)
        page.wait_for_timeout(3000)  # JS rendering wait

        html = page.content()
        save_html_debug(html, '01_data_analysis')

        # Check for login redirect
        if 'プレミアムサービス案内' in html or '会員登録' in html:
            results['1_data_analysis'] = {
                'status': 'FAILED',
                'reason': 'プレミアム会員限定ページ、Cookie認証不可',
                'url': url
            }
            return

        # Check for data tables
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Look for tab structure (枠番別、距離別、etc.)
        tabs = soup.find_all(['li', 'a', 'div'], class_=re.compile(r'Tab|tab|DataList'))
        tables = soup.find_all('table')

        # Look for specific data sections
        sections_found = []
        for text_pattern in ['枠番別', '距離別', '馬場別', '脚質別', '騎手別', '調教師別', '種牡馬別']:
            if text_pattern in html:
                sections_found.append(text_pattern)

        # Try clicking tabs to load dynamic content
        tab_elements = page.query_selector_all('.DataListMenu li, .RaceDataMenu li, [class*="tab"] li, [class*="Tab"] li')
        tab_texts = []
        for tab in tab_elements:
            txt = tab.inner_text().strip()
            if txt:
                tab_texts.append(txt)

        # Check for DataList content
        data_tables = soup.find_all('table', class_=re.compile(r'Data|data|Nk'))

        results['1_data_analysis'] = {
            'status': 'SUCCESS' if (tables or sections_found) else 'PARTIAL',
            'url': url,
            'tables_found': len(tables),
            'sections_found': sections_found,
            'tab_texts': tab_texts[:20],
            'html_length': len(html),
            'has_data': bool(tables and len(tables) > 0),
        }

        if tables:
            # Parse first table structure
            first_table = tables[0]
            headers = [th.get_text(strip=True) for th in first_table.find_all('th')]
            rows = first_table.find_all('tr')
            results['1_data_analysis']['first_table_headers'] = headers[:20]
            results['1_data_analysis']['first_table_rows'] = len(rows)

        print(f"  Tables: {len(tables)}, Sections: {sections_found}")
        print(f"  Tabs: {tab_texts[:10]}")

    except Exception as e:
        results['1_data_analysis'] = {'status': 'ERROR', 'reason': str(e), 'url': url}
        print(f"  ERROR: {e}")

def investigate_furi_check(page, results):
    """2. 不利馬チェック"""
    print("\n=== 2. 不利馬チェック ===")

    # Try multiple possible URLs
    urls_to_try = [
        f"https://race.netkeiba.com/race/furi.html?race_id={RACE_ID}",
        f"https://race.netkeiba.com/race/furi_check.html?race_id={RACE_ID}",
        f"https://race.netkeiba.com/race/movie.html?race_id={RACE_ID}",  # sometimes combined
    ]

    # Also check if it's embedded in result page
    result_url = f"https://race.netkeiba.com/race/result.html?race_id={RACE_ID}"
    urls_to_try.append(result_url)

    found = False
    for url in urls_to_try:
        print(f"  Trying: {url}")
        try:
            page.goto(url, wait_until='domcontentloaded', timeout=15000)
            page.wait_for_timeout(2000)
            html = page.content()

            if '404' in page.url or 'Not Found' in html[:500]:
                print(f"    → 404")
                time.sleep(DELAY)
                continue

            if '不利' in html or 'furi' in html.lower() or 'Furi' in html:
                save_html_debug(html, '02_furi_check')
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                tables = soup.find_all('table')
                furi_divs = soup.find_all(['div', 'span', 'td'], string=re.compile(r'不利'))

                results['2_furi_check'] = {
                    'status': 'SUCCESS',
                    'url': url,
                    'tables_found': len(tables),
                    'furi_elements': len(furi_divs),
                    'html_length': len(html),
                }
                found = True
                print(f"    → Found! tables={len(tables)}, furi_elements={len(furi_divs)}")
                break

            time.sleep(DELAY)
        except Exception as e:
            print(f"    → Error: {e}")
            time.sleep(DELAY)
            continue

    if not found:
        # Check the race result page for 不利 info in race review
        print("  Checking result page for 不利 data...")
        try:
            page.goto(result_url, wait_until='domcontentloaded', timeout=15000)
            page.wait_for_timeout(2000)
            html = page.content()
            save_html_debug(html, '02_furi_result_page')

            has_furi = '不利' in html
            results['2_furi_check'] = {
                'status': 'NOT_FOUND',
                'reason': '専用ページなし。レース結果ページの備考欄に「不利」情報あり' if has_furi else '専用ページなし、不利情報も見つからず',
                'urls_tried': urls_to_try,
                'has_furi_in_result': has_furi,
            }
        except Exception as e:
            results['2_furi_check'] = {'status': 'ERROR', 'reason': str(e)}

def investigate_newspaper(page, results):
    """3. 競馬新聞"""
    print("\n=== 3. 競馬新聞 newspaper ===")
    url = f"https://race.netkeiba.com/race/newspaper.html?race_id={RACE_ID}"
    print(f"  URL: {url}")

    try:
        page.goto(url, wait_until='domcontentloaded', timeout=30000)
        page.wait_for_timeout(3000)

        html = page.content()
        save_html_debug(html, '03_newspaper')

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Check if premium-restricted
        is_premium = 'プレミアムサービス案内' in html

        # Find all data fields unique to newspaper
        tables = soup.find_all('table')

        # Look for specific newspaper data
        data_fields = {}
        for field in ['印', '予想', 'AI予想', '展開', 'コメント', '追切', '厩舎',
                       '調教', 'ランク', '指数', 'タイム指数', 'スピード指数']:
            if field in html:
                data_fields[field] = True

        # Find newspaper-specific elements
        newspaper_cells = soup.find_all(['td', 'div'], class_=re.compile(r'Newspaper|newspaper|Mark|Comment'))

        # Look for horse entries
        horse_entries = soup.find_all(['tr', 'div'], class_=re.compile(r'HorseList|Horse'))

        results['3_newspaper'] = {
            'status': 'SUCCESS' if tables else 'PARTIAL',
            'url': url,
            'is_premium': is_premium,
            'tables_found': len(tables),
            'data_fields_found': data_fields,
            'newspaper_cells': len(newspaper_cells),
            'horse_entries': len(horse_entries),
            'html_length': len(html),
        }

        print(f"  Tables: {len(tables)}, Fields: {list(data_fields.keys())}")
        print(f"  Premium: {is_premium}, Cells: {len(newspaper_cells)}")

    except Exception as e:
        results['3_newspaper'] = {'status': 'ERROR', 'reason': str(e), 'url': url}
        print(f"  ERROR: {e}")

def investigate_pace_prediction(page, results):
    """4. 展開予測AI"""
    print("\n=== 4. 展開予測AI ===")

    urls_to_try = [
        f"https://race.netkeiba.com/race/prediction.html?race_id={RACE_ID}",
        f"https://race.netkeiba.com/race/shutuba.html?race_id={RACE_ID}",  # may have embedded AI
        f"https://race.netkeiba.com/race/pace.html?race_id={RACE_ID}",
        f"https://race.netkeiba.com/race/tenkai.html?race_id={RACE_ID}",
    ]

    for url in urls_to_try:
        print(f"  Trying: {url}")
        try:
            page.goto(url, wait_until='domcontentloaded', timeout=15000)
            page.wait_for_timeout(2000)
            html = page.content()

            if '404' in page.url and 'Not Found' in html[:1000]:
                print(f"    → 404")
                time.sleep(DELAY)
                continue

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            # Check for AI/展開/ペース content
            has_ai = any(x in html for x in ['AI予想', 'AI展開', '展開予測', 'ペース予測', 'RacePace', 'PacePredict'])
            has_tenkai = '展開' in html and ('予測' in html or 'AI' in html)

            if has_ai or has_tenkai or 'shutuba' in url:
                save_html_debug(html, f'04_pace_{url.split("/")[-1].split("?")[0]}')

                # Look for pace-related elements
                pace_elements = soup.find_all(['div', 'span', 'td'],
                    class_=re.compile(r'Pace|pace|Tenkai|tenkai|Position|AI'))

                if 'shutuba' in url:
                    # Look for AI prediction section in shutuba page
                    ai_sections = soup.find_all(['div', 'section'],
                        class_=re.compile(r'AI|Prediction|Race_Pace|RacePace'))

                    results['4_pace_prediction'] = {
                        'status': 'FOUND_IN_SHUTUBA' if ai_sections else 'CHECK_SHUTUBA',
                        'url': url,
                        'ai_sections': len(ai_sections),
                        'has_ai_text': has_ai,
                        'html_length': len(html),
                    }
                else:
                    results['4_pace_prediction'] = {
                        'status': 'SUCCESS',
                        'url': url,
                        'pace_elements': len(pace_elements),
                        'has_ai': has_ai,
                        'html_length': len(html),
                    }
                break

            time.sleep(DELAY)
        except Exception as e:
            print(f"    → Error: {e}")
            time.sleep(DELAY)
            continue

    if '4_pace_prediction' not in results:
        results['4_pace_prediction'] = {
            'status': 'NOT_FOUND',
            'reason': '専用URLが見つからない。出馬表ページに埋め込みの可能性',
            'urls_tried': urls_to_try,
        }

def investigate_upset_level(page, results):
    """5. 波乱度予測"""
    print("\n=== 5. 波乱度予測 ===")

    urls_to_try = [
        f"https://race.netkeiba.com/race/upset.html?race_id={RACE_ID}",
        f"https://race.netkeiba.com/race/haran.html?race_id={RACE_ID}",
    ]

    # Also check shutuba page for embedded upset level
    shutuba_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={RACE_ID}"

    for url in urls_to_try:
        print(f"  Trying: {url}")
        try:
            page.goto(url, wait_until='domcontentloaded', timeout=15000)
            page.wait_for_timeout(2000)
            html = page.content()

            if '波乱' in html or 'upset' in html.lower() or '荒れ' in html:
                save_html_debug(html, '05_upset')
                results['5_upset_level'] = {
                    'status': 'SUCCESS',
                    'url': url,
                    'html_length': len(html),
                }
                print(f"    → Found!")
                return

            time.sleep(DELAY)
        except Exception as e:
            print(f"    → Error: {e}")
            time.sleep(DELAY)
            continue

    # Check shutuba page
    print(f"  Checking shutuba for embedded upset level...")
    try:
        page.goto(shutuba_url, wait_until='domcontentloaded', timeout=30000)
        page.wait_for_timeout(3000)
        html = page.content()
        save_html_debug(html, '05_upset_shutuba')

        has_upset = any(x in html for x in ['波乱度', '荒れ度', 'UpsetLevel', 'upset_level', 'HaranLevel'])

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        upset_elements = soup.find_all(['div', 'span'], string=re.compile(r'波乱|荒れ'))
        level_elements = soup.find_all(['div', 'span'], class_=re.compile(r'Level|Lv|level'))

        results['5_upset_level'] = {
            'status': 'FOUND_IN_SHUTUBA' if has_upset else 'NOT_FOUND',
            'url': shutuba_url,
            'has_upset_text': has_upset,
            'upset_elements': len(upset_elements),
            'level_elements': len(level_elements),
            'reason': '出馬表ページに埋め込みの可能性' if not has_upset else None,
        }
    except Exception as e:
        results['5_upset_level'] = {'status': 'ERROR', 'reason': str(e)}

def investigate_ev_simulation(page, results):
    """6. 期待値シミュレーション"""
    print("\n=== 6. 期待値シミュレーション ===")

    urls_to_try = [
        f"https://race.netkeiba.com/race/ev.html?race_id={RACE_ID}",
        f"https://race.netkeiba.com/race/simulation.html?race_id={RACE_ID}",
        f"https://race.netkeiba.com/race/expectation.html?race_id={RACE_ID}",
        f"https://race.netkeiba.com/race/kitaichi.html?race_id={RACE_ID}",
    ]

    for url in urls_to_try:
        print(f"  Trying: {url}")
        try:
            page.goto(url, wait_until='domcontentloaded', timeout=15000)
            page.wait_for_timeout(2000)
            html = page.content()

            if '期待値' in html or 'EV' in html or 'シミュレーション' in html:
                save_html_debug(html, '06_ev_simulation')
                results['6_ev_simulation'] = {
                    'status': 'SUCCESS',
                    'url': url,
                    'html_length': len(html),
                }
                print(f"    → Found!")
                return

            time.sleep(DELAY)
        except Exception as e:
            print(f"    → Error: {e}")
            time.sleep(DELAY)

    # Check shutuba page
    shutuba_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={RACE_ID}"
    print(f"  Checking shutuba for embedded EV...")
    try:
        # Reuse previously saved shutuba if available
        page.goto(shutuba_url, wait_until='domcontentloaded', timeout=30000)
        page.wait_for_timeout(3000)
        html = page.content()

        has_ev = any(x in html for x in ['期待値', 'ExpectedValue', 'EV_Simulation', 'SimResult'])

        results['6_ev_simulation'] = {
            'status': 'FOUND_IN_SHUTUBA' if has_ev else 'NOT_FOUND',
            'url': shutuba_url,
            'has_ev_text': has_ev,
            'reason': '専用ページなし' if not has_ev else None,
        }
    except Exception as e:
        results['6_ev_simulation'] = {'status': 'ERROR', 'reason': str(e)}

def investigate_running_data(page, results):
    """7. 走行データ"""
    print("\n=== 7. 走行データ ===")

    urls_to_try = [
        f"https://race.netkeiba.com/race/running.html?race_id={RACE_ID}",
        f"https://race.netkeiba.com/race/sokou.html?race_id={RACE_ID}",
        f"https://race.netkeiba.com/race/race_data.html?race_id={RACE_ID}",
        f"https://race.netkeiba.com/race/analyze.html?race_id={RACE_ID}",
    ]

    for url in urls_to_try:
        print(f"  Trying: {url}")
        try:
            page.goto(url, wait_until='domcontentloaded', timeout=15000)
            page.wait_for_timeout(2000)
            html = page.content()

            # Check for running/walking data content
            has_data = any(x in html for x in ['走行', '走破', 'ラップ', 'Running', 'LapTime',
                                                'コーナー通過', '上がり3F', 'セクション'])
            is_error = '404' in page.url or len(html) < 500

            if has_data and not is_error:
                save_html_debug(html, f'07_running_{url.split("/")[-1].split("?")[0]}')

                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                tables = soup.find_all('table')

                results['7_running_data'] = {
                    'status': 'SUCCESS',
                    'url': url,
                    'tables_found': len(tables),
                    'html_length': len(html),
                }
                print(f"    → Found! tables={len(tables)}")
                return

            time.sleep(DELAY)
        except Exception as e:
            print(f"    → Error: {e}")
            time.sleep(DELAY)

    # Check result page for race lap data
    result_url = f"https://race.netkeiba.com/race/result.html?race_id={RACE_ID}"
    print(f"  Checking result page for running data...")
    try:
        page.goto(result_url, wait_until='domcontentloaded', timeout=15000)
        page.wait_for_timeout(2000)
        html = page.content()
        save_html_debug(html, '07_running_result')

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Look for lap time / corner position data
        lap_data = soup.find_all(['div', 'table'], class_=re.compile(r'Lap|Corner|Race_Lap|RapTable'))

        results['7_running_data'] = {
            'status': 'IN_RESULT_PAGE' if lap_data else 'NOT_FOUND',
            'url': result_url,
            'lap_elements': len(lap_data),
            'reason': 'レース結果ページにラップ/コーナーデータあり' if lap_data else '専用ページなし',
        }
    except Exception as e:
        results['7_running_data'] = {'status': 'ERROR', 'reason': str(e)}

def main():
    from playwright.sync_api import sync_playwright

    cookie_str = _load_cookie()
    if not cookie_str:
        print("ERROR: NETKEIBA_COOKIE not found in .env")
        sys.exit(1)

    results = {}

    with sync_playwright() as p:
        browser, context = _setup_browser_context(p, cookie_str)
        page = context.new_page()

        # Verify cookie works
        print("=== Cookie Verification ===")
        page.goto('https://race.netkeiba.com/race/oikiri.html?race_id=202605010811',
                   wait_until='domcontentloaded', timeout=30000)
        page.wait_for_timeout(2000)
        html = page.content()
        is_premium = 'プレミアムサービス案内' not in html and len(html) > 5000
        print(f"  Premium access: {'OK' if is_premium else 'FAILED'} (html_length={len(html)})")

        if not is_premium:
            print("WARNING: Premium cookie may be expired. Proceeding anyway...")

        time.sleep(DELAY)

        # 1. Data Analysis
        investigate_data_analysis(page, results)
        time.sleep(DELAY)

        # 2. Furi Check
        investigate_furi_check(page, results)
        time.sleep(DELAY)

        # 3. Newspaper
        investigate_newspaper(page, results)
        time.sleep(DELAY)

        # 4. Pace Prediction
        investigate_pace_prediction(page, results)
        time.sleep(DELAY)

        # 5. Upset Level
        investigate_upset_level(page, results)
        time.sleep(DELAY)

        # 6. EV Simulation
        investigate_ev_simulation(page, results)
        time.sleep(DELAY)

        # 7. Running Data
        investigate_running_data(page, results)

        browser.close()

    # Save results
    output_path = os.path.join(BASE_DIR, 'data', 'remaining_data_investigation.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n\n=== RESULTS SAVED: {output_path} ===")
    for key, val in results.items():
        print(f"  {key}: {val.get('status', 'UNKNOWN')}")

    return results

if __name__ == '__main__':
    main()
