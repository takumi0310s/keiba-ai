#!/usr/bin/env python
"""今週末レース限定データ一括取得スクレイパー

「当週のみ公開」のデータを金曜〜土曜朝に事前取得:
  1. newspaper.html → AI予測タイム・展開予測・見解
  2. shutuba.html   → 波乱度(Lv1-5)・上位人気信頼度
  3. data_list.html  → データ分析43種（枠順・脚質・騎手・調教師・種牡馬別着度数）

保存先:
  data/netkeiba_newspaper_ai_thisweek.csv   (馬別AI予測タイム)
  data/netkeiba_newspaper_ai_thisweek.json  (レース別全データ)
  data/netkeiba_upset_level_thisweek.csv    (波乱度)
  data/netkeiba_data_analysis_thisweek.csv  (データ分析)

Usage:
    python tools/scrape_weekend_thisweek.py                    # 明日+明後日(土日)
    python tools/scrape_weekend_thisweek.py --date 20260411    # 指定日から2日間
    python tools/scrape_weekend_thisweek.py --dry-run          # 取得計画のみ表示
"""
import os
import sys
import io
import re
import json
import time
import csv
import random
import argparse
from datetime import datetime, timedelta

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

DELAY = 15  # リクエスト間隔(秒)
BAN_WAIT = 600  # IPバン時待機(秒)

# 出力ファイル
OUT_NEWSPAPER_JSON = os.path.join(DATA_DIR, 'netkeiba_newspaper_ai_thisweek.json')
OUT_NEWSPAPER_CSV = os.path.join(DATA_DIR, 'netkeiba_newspaper_ai_thisweek.csv')
OUT_UPSET_CSV = os.path.join(DATA_DIR, 'netkeiba_upset_level_thisweek.csv')
OUT_ANALYSIS_CSV = os.path.join(DATA_DIR, 'netkeiba_data_analysis_thisweek.csv')


def _load_session():
    import requests
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
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://race.netkeiba.com/',
    })
    for part in cookie_str.split(';'):
        part = part.strip()
        if '=' in part:
            key, val = part.split('=', 1)
            session.cookies.set(key.strip(), val.strip())
    return session


def get_race_ids_for_dates(session, dates):
    """複数日のレースIDを取得"""
    import requests as _req
    all_ids = []
    for d in dates:
        url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={d}"
        try:
            resp = session.get(url, timeout=15)
            resp.encoding = 'EUC-JP'
            ids = sorted(set(re.findall(r'race_id=(\d{12})', resp.text)))
            print(f"  {d}: {len(ids)} races")
            all_ids.extend([(rid, d) for rid in ids])
        except Exception as e:
            print(f"  {d}: ERROR {e}")
        time.sleep(2)
    return all_ids


def scrape_newspaper(session, race_id):
    """newspaper.htmlからAI展開予測データを取得（既存scrape_newspaper_ai.pyと同一ロジック）"""
    from bs4 import BeautifulSoup

    url = f"https://race.netkeiba.com/race/newspaper.html?race_id={race_id}"
    resp = session.get(url, timeout=30)
    if resp.status_code in (403, 429):
        raise ConnectionError(f"HTTP {resp.status_code}")
    html = resp.content.decode('utf-8', errors='ignore')
    soup = BeautifulSoup(html, 'html.parser')

    result = {'race_id': race_id}

    # 1. AI予測ラップタイム (NoCheckData)
    haron_table = soup.find('table', class_='Race_HaronTime')
    if haron_table:
        rows = haron_table.find_all('tr', class_=lambda c: c and 'HaronTime' in c and 'NoCheckData' in c)
        laps = []
        for row in rows:
            for td in row.find_all('td'):
                parts = td.get_text(separator='|', strip=True).split('|')
                if len(parts) >= 2:
                    laps.append({'cum': parts[0], 'lap': parts[1]})
                elif len(parts) == 1:
                    nums = re.findall(r'\d+:\d+\.\d+|\d+\.\d+', parts[0])
                    if len(nums) >= 2:
                        laps.append({'cum': nums[0], 'lap': nums[1]})
        result['predicted_laps'] = laps

    # 2. 各馬AI予測タイム（前半3F/後半3F）
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

    # 3. AI見解テキスト
    opinion_section = soup.find('section', class_='DevelopOpinionArea')
    if opinion_section:
        no_check = opinion_section.find('dd', class_='NoCheckData')
        if no_check:
            result['ai_opinion'] = no_check.get_text(strip=True)

    # 4. ポジション有利馬
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

    # 5. ポジション別複勝率ヒートマップ
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
                        'position': pos_label, 'track': track_label,
                        'win_rate': float(val) if val.replace('.', '').isdigit() else 0,
                    })
        result['position_heatmap'] = cells

    # 6. トラックバイアス
    bias_div = soup.find('div', class_='DevelopBiasTxt')
    if bias_div:
        result['track_bias'] = [span.get_text(strip=True) for span in bias_div.find_all('span')]

    # 7. AnaBest
    for t in soup.find_all('table', class_='AnaBestTable'):
        th = t.find('th')
        if th:
            label = th.get_text(strip=True)
            td = t.find('td')
            if td:
                horses = [int(s.get_text(strip=True)) for s in td.find_all('span', class_=re.compile(r'Waku\d'))
                          if s.get_text(strip=True).isdigit()]
                if label == '能力':
                    result['ana_best_ability'] = horses
                elif label == '上昇度':
                    result['ana_best_growth'] = horses

    return result


def scrape_upset(session, race_id):
    """shutuba.htmlから波乱度・上位人気信頼度を取得"""
    from bs4 import BeautifulSoup

    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = session.get(url, timeout=30)
    if resp.status_code in (403, 429):
        raise ConnectionError(f"HTTP {resp.status_code}")
    html = resp.content.decode('utf-8', errors='ignore')

    result = {'race_id': race_id}

    # 波乱度 from JS variable
    m = re.search(r"var\s+upsetLevel\s*=\s*parseInt\('(\d+)'\)", html)
    if m:
        result['upset_level'] = int(m.group(1))

    # 上位人気信頼度
    soup = BeautifulSoup(html, 'html.parser')
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


def scrape_data_analysis(session, race_id):
    """data_list.htmlからデータ分析テーブルを取得"""
    from bs4 import BeautifulSoup

    url = f"https://race.netkeiba.com/race/data_list.html?race_id={race_id}"
    resp = session.get(url, timeout=15)
    if resp.status_code in (403, 429):
        raise ConnectionError(f"HTTP {resp.status_code}")
    resp.encoding = 'euc-jp'

    soup = BeautifulSoup(resp.text, 'html.parser')
    results = []

    for table in soup.find_all('table', class_='RaceCommon_Table'):
        for tr in table.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if len(cells) >= 2:
                results.append({
                    'race_id': race_id,
                    'category': cells[0],
                    'value': cells[1],
                })

    # レースメタデータ
    race_data1 = soup.find('div', class_='RaceData01')
    race_data2 = soup.find('div', class_='RaceData02')
    if race_data1:
        results.append({'race_id': race_id, 'category': 'race_info',
                        'value': race_data1.get_text(strip=True)})
    if race_data2:
        results.append({'race_id': race_id, 'category': 'race_detail',
                        'value': race_data2.get_text(strip=True)})

    return results


def save_newspaper_csv(all_newspaper):
    """newspaper AI予測タイムをCSV保存"""
    with open(OUT_NEWSPAPER_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'race_id', 'horse_num', 'horse_name', 'ai_first_3f', 'ai_last_3f'])
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


def save_newspaper_json(all_newspaper):
    """newspaper全データをJSON保存（ネスト構造のため）"""
    with open(OUT_NEWSPAPER_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_newspaper, f, ensure_ascii=False, indent=2)


def save_upset_csv(all_upset):
    """波乱度をCSV保存"""
    with open(OUT_UPSET_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['race_id', 'upset_level', 'top_popularity_reliability'])
        writer.writeheader()
        for d in all_upset:
            if 'error' not in d:
                writer.writerow({
                    'race_id': d['race_id'],
                    'upset_level': d.get('upset_level', ''),
                    'top_popularity_reliability': d.get('top_popularity_reliability', ''),
                })


def save_analysis_csv(all_analysis):
    """データ分析をCSV保存"""
    import pandas as pd
    df = pd.DataFrame(all_analysis)
    df.to_csv(OUT_ANALYSIS_CSV, index=False, encoding='utf-8-sig')


def main():
    parser = argparse.ArgumentParser(description='今週末限定データ一括取得')
    parser.add_argument('--date', type=str, default='',
                        help='開始日 YYYYMMDD (デフォルト: 明日)')
    parser.add_argument('--dry-run', action='store_true', help='取得計画のみ表示')
    args = parser.parse_args()

    # デフォルト: 明日から2日間（金曜実行→土日取得）
    if args.date:
        base = datetime.strptime(args.date, '%Y%m%d')
    else:
        base = datetime.now() + timedelta(days=1)

    dates = []
    for delta in range(2):
        d = (base + timedelta(days=delta)).strftime('%Y%m%d')
        dates.append(d)

    print("=" * 60)
    print(f"  週末限定データ取得")
    print(f"  対象日: {', '.join(dates)}")
    print(f"  リクエスト間隔: {DELAY}秒")
    print("=" * 60)

    session = _load_session()
    if not session:
        sys.exit(1)

    # 1. レースID取得
    print("\n[1/4] レースID取得...")
    race_list = get_race_ids_for_dates(session, dates)
    race_ids = [rid for rid, _ in race_list]
    print(f"  合計: {len(race_ids)} races")

    if not race_ids:
        print("  レースなし。終了。")
        return

    # リクエスト数見積り: newspaper + upset + data_analysis = 3回/レース
    total_requests = len(race_ids) * 3
    est_minutes = total_requests * DELAY / 60
    print(f"  リクエスト数: {total_requests} (約{est_minutes:.0f}分)")

    if args.dry_run:
        print("\n  [DRY RUN] 取得は実行しません")
        for rid, d in race_list:
            print(f"    {d} {rid}")
        return

    # 2. 全レースデータ取得
    all_newspaper = []
    all_upset = []
    all_analysis = []
    consecutive_bans = 0
    start_time = time.time()

    for i, (race_id, date_str) in enumerate(race_list):
        progress = f"[{i+1}/{len(race_list)}] {race_id} ({date_str})"

        # 2a. newspaper.html
        try:
            np_data = scrape_newspaper(session, race_id)
            all_newspaper.append(np_data)
            n_times = len(np_data.get('ai_horse_times', []))
            opinion = 'Y' if np_data.get('ai_opinion') else 'N'
            print(f"  {progress} newspaper: {n_times}馬 opinion={opinion}", end='')
            consecutive_bans = 0
        except ConnectionError:
            consecutive_bans += 1
            print(f"  {progress} newspaper: BAN ({consecutive_bans})", end='')
            if consecutive_bans >= 3:
                print(f"\n  !!! 3連続BAN。{BAN_WAIT//60}分待機...")
                time.sleep(BAN_WAIT)
                session = _load_session()
                consecutive_bans = 0
        except Exception as e:
            print(f"  {progress} newspaper: ERR {e}", end='')
            all_newspaper.append({'race_id': race_id, 'error': str(e)})

        time.sleep(DELAY)

        # 2b. shutuba.html (波乱度)
        try:
            up_data = scrape_upset(session, race_id)
            all_upset.append(up_data)
            lv = up_data.get('upset_level', '?')
            rel = up_data.get('top_popularity_reliability', '?')
            print(f" | upset=Lv{lv} ({rel}%)", end='')
        except ConnectionError:
            print(f" | upset=BAN", end='')
        except Exception as e:
            print(f" | upset=ERR", end='')
            all_upset.append({'race_id': race_id, 'error': str(e)})

        time.sleep(DELAY)

        # 2c. data_list.html (データ分析)
        try:
            analysis = scrape_data_analysis(session, race_id)
            all_analysis.extend(analysis)
            print(f" | analysis={len(analysis)}rows")
        except ConnectionError:
            print(f" | analysis=BAN")
        except Exception as e:
            print(f" | analysis=ERR")

        time.sleep(DELAY)

        # 10レースごとに中間保存
        if (i + 1) % 10 == 0:
            save_newspaper_json(all_newspaper)
            save_newspaper_csv(all_newspaper)
            save_upset_csv(all_upset)
            if all_analysis:
                save_analysis_csv(all_analysis)
            print(f"  --- 中間保存 ({i+1}/{len(race_list)}) ---")

    # 3. 最終保存
    print(f"\n[2/4] データ保存...")
    save_newspaper_json(all_newspaper)
    save_newspaper_csv(all_newspaper)
    save_upset_csv(all_upset)
    if all_analysis:
        save_analysis_csv(all_analysis)

    # 蓄積版CSVにも追記（波乱度）
    _append_to_master_upset(all_upset)
    # 蓄積版CSVにも追記（データ分析）
    _append_to_master_analysis(all_analysis)

    elapsed = time.time() - start_time
    n_np_ok = len([d for d in all_newspaper if 'error' not in d])
    n_up_ok = len([d for d in all_upset if 'error' not in d])

    print(f"\n{'=' * 60}")
    print(f"  COMPLETE ({elapsed/60:.1f}分)")
    print(f"  newspaper: {n_np_ok}/{len(race_list)} races")
    print(f"  upset:     {n_up_ok}/{len(race_list)} races")
    print(f"  analysis:  {len(all_analysis)} rows")
    print(f"  保存先:")
    print(f"    {OUT_NEWSPAPER_CSV}")
    print(f"    {OUT_NEWSPAPER_JSON}")
    print(f"    {OUT_UPSET_CSV}")
    print(f"    {OUT_ANALYSIS_CSV}")
    print(f"{'=' * 60}")

    # Discord通知
    try:
        from notify import send_discord
        send_discord("週末データ取得完了",
                     f"newspaper: {n_np_ok}R / upset: {n_up_ok}R / analysis: {len(all_analysis)}rows\n"
                     f"対象: {', '.join(dates)} ({elapsed/60:.1f}分)",
                     color="green")
    except Exception:
        pass


def _append_to_master_upset(all_upset):
    """蓄積版netkeiba_upset_level.csvに追記"""
    master = os.path.join(DATA_DIR, 'netkeiba_upset_level.csv')
    existing_ids = set()
    if os.path.exists(master):
        try:
            import pandas as pd
            df = pd.read_csv(master, dtype=str)
            existing_ids = set(df['race_id'].values)
        except Exception:
            pass

    new_rows = [d for d in all_upset if 'error' not in d and d['race_id'] not in existing_ids]
    if not new_rows:
        return

    file_exists = os.path.exists(master) and os.path.getsize(master) > 0
    with open(master, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['race_id', 'upset_level', 'top_popularity_reliability'])
        if not file_exists:
            writer.writeheader()
        for d in new_rows:
            writer.writerow({
                'race_id': d['race_id'],
                'upset_level': d.get('upset_level', ''),
                'top_popularity_reliability': d.get('top_popularity_reliability', ''),
            })
    print(f"  蓄積版upset: +{len(new_rows)}行 → {master}")


def _append_to_master_analysis(all_analysis):
    """蓄積版netkeiba_data_analysis.csvに追記"""
    if not all_analysis:
        return
    master = os.path.join(DATA_DIR, 'netkeiba_data_analysis.csv')
    existing_ids = set()
    if os.path.exists(master):
        try:
            import pandas as pd
            df = pd.read_csv(master, dtype=str)
            existing_ids = set(df['race_id'].astype(str).unique())
        except Exception:
            pass

    new_rows = [r for r in all_analysis if str(r['race_id']) not in existing_ids]
    if not new_rows:
        return

    import pandas as pd
    df_new = pd.DataFrame(new_rows)
    if os.path.exists(master):
        df_existing = pd.read_csv(master)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(master, index=False, encoding='utf-8-sig')
    print(f"  蓄積版analysis: +{len(new_rows)}行 → {master}")


# --- データ読込ユーティリティ（predict_core / app.py用） ---

def load_thisweek_upset():
    """今週の波乱度データを読み込み → {race_id: {upset_level, reliability}}"""
    if not os.path.exists(OUT_UPSET_CSV):
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(OUT_UPSET_CSV, dtype={'race_id': str})
        result = {}
        for _, row in df.iterrows():
            result[str(row['race_id'])] = {
                'upset_level': int(row['upset_level']) if pd.notna(row.get('upset_level')) else None,
                'top_popularity_reliability': float(row['top_popularity_reliability'])
                    if pd.notna(row.get('top_popularity_reliability')) else None,
            }
        return result
    except Exception:
        return {}


def load_thisweek_newspaper():
    """今週のnewspaper AIデータを読み込み → {race_id: {ai_horse_times, ai_opinion, ...}}"""
    if not os.path.exists(OUT_NEWSPAPER_JSON):
        return {}
    try:
        with open(OUT_NEWSPAPER_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        result = {}
        for item in data:
            rid = str(item.get('race_id', ''))
            if rid:
                result[rid] = item
        return result
    except Exception:
        return {}


def load_thisweek_analysis():
    """今週のデータ分析を読み込み → {race_id: [{category, value}, ...]}"""
    if not os.path.exists(OUT_ANALYSIS_CSV):
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(OUT_ANALYSIS_CSV, dtype={'race_id': str})
        result = {}
        for rid, group in df.groupby('race_id'):
            result[str(rid)] = group[['category', 'value']].to_dict('records')
        return result
    except Exception:
        return {}


if __name__ == '__main__':
    from tools.scraper_guard import check_scraping_allowed; check_scraping_allowed()
    main()
