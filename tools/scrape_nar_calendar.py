"""NAR 当日 レーシング・カレンダー scraper. 13:00 自動発火 想定.

当日開催される NAR レースの (場・レース番号・発走時刻・race_name) 一覧を軽量取得する。
shutuba スクレイプ (16:30) より 3.5h 早く実行され、当日開催の有無確認・遅延予知に使う。

URL:
  list: https://nar.netkeiba.com/top/race_list_sub.html?kaisai_date=YYYYMMDD

出力:
  data/nar_calendar_YYYYMMDD.csv
  columns: race_id, course, course_code, race_num, race_name, race_time, distance, surface

note:
  shutuba と違い 1 race ずつ詳細を fetch する 重い処理は避ける。
  race_list_sub.html 1 リクエストで取れる情報のみ parse。
  発走時刻は race_list 内 .RaceList_Itemtime / 類似 selector を試す。
  該当 selector 不在の race_id は race_time='' で記録 (fallback あり)。
  race_id 構造: YYYYCCKKDDRR (CC=場code, RR=race_num)。
"""
from __future__ import annotations

import os, sys, argparse, csv, re, time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import requests
from bs4 import BeautifulSoup
from datetime import datetime

BASE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE)
sys.path.insert(0, os.path.join(BASE, 'tools'))

from scrape_nar_all import (
    NAR_TRACKS, create_session, safe_get, nav_sleep,
)

OUT_HEADER = [
    'race_id', 'course', 'course_code', 'race_num',
    'race_name', 'race_time', 'distance', 'surface',
]


def fetch_calendar(session, date):
    """race_list_sub.html を取得して race rows を返す."""
    url = f'https://nar.netkeiba.com/top/race_list_sub.html?kaisai_date={date}'
    r = safe_get(session, url)
    if r is None:
        return None, 'timeout'
    if r.status_code != 200:
        return None, f'http_{r.status_code}'

    try:
        # race_list_sub.html は utf-8 (shutuba は euc-jp、混在に注意)
        # apparent_encoding で判定して decode
        enc = r.apparent_encoding or 'utf-8'
        try:
            text = r.content.decode(enc)
        except (UnicodeDecodeError, LookupError):
            # fallback: utf-8 → euc-jp の順
            try:
                text = r.content.decode('utf-8')
            except UnicodeDecodeError:
                text = r.content.decode('euc-jp', errors='replace')
        soup = BeautifulSoup(text, 'html.parser')
    except Exception as e:
        return None, f'parse_err: {e}'

    rows = []
    seen = set()

    # race_id を持つ a タグを全列挙
    for a in soup.select('a[href*="race_id="]'):
        href = a.get('href', '')
        m = re.search(r'race_id=(\d+)', href)
        if not m:
            continue
        race_id = m.group(1)
        if race_id in seen:
            continue
        seen.add(race_id)

        # course / race_num を race_id から逆算
        course_code = race_id[4:6] if len(race_id) >= 6 else ''
        course = NAR_TRACKS.get(int(course_code), '') if course_code.isdigit() else ''
        race_num = race_id[-2:] if len(race_id) >= 2 else ''
        try:
            race_num = str(int(race_num))
        except ValueError:
            pass

        # 発走時刻 / レース名 / 距離 を周辺から取得
        race_time, race_name, distance, surface = '', '', 0, ''

        # 親要素を 3 段階遡って情報源を探す (RaceList_Item / dl / li 等の構造ばらつきに対応)
        parent = a
        block_text = a.get_text(' ', strip=True)
        for _ in range(3):
            parent = parent.parent
            if parent is None:
                break
            block_text = parent.get_text(' ', strip=True)
            if 'm' in block_text and re.search(r'\d{1,2}:\d{2}', block_text):
                break

        # 発走時刻
        tm = re.search(r'(\d{1,2}:\d{2})', block_text)
        if tm:
            race_time = tm.group(1)

        # 距離・表面
        dm = re.search(r'([ダ芝障])\s*(\d{3,4})\s*m', block_text)
        if dm:
            surface = dm.group(1)
            distance = int(dm.group(2))

        # race_name: a タグ内 の RaceName class があれば優先
        rn_el = a.select_one('.RaceList_ItemTitle, .RaceName')
        if rn_el:
            race_name = rn_el.get_text(strip=True)
        if not race_name:
            # link 直接 text を fallback (短い場合のみ)
            txt = a.get_text(strip=True)
            if 0 < len(txt) < 60 and not re.match(r'^\d+R$', txt):
                race_name = txt

        rows.append({
            'race_id': race_id, 'course': course, 'course_code': course_code,
            'race_num': race_num, 'race_name': race_name, 'race_time': race_time,
            'distance': distance, 'surface': surface,
        })

    return rows, 'ok'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=None, help='YYYYMMDD (default 今日)')
    parser.add_argument('--output', default=None,
                        help='出力 CSV (default data/nar_calendar_DATE.csv)')
    args = parser.parse_args()

    date = args.date or datetime.now().strftime('%Y%m%d')
    out_path = args.output or f'data/nar_calendar_{date}.csv'

    print(f"=== scrape_nar_calendar {date} ===")
    print(f"  output: {out_path}")

    session = create_session()
    nav_sleep()

    rows, status = fetch_calendar(session, date)
    if rows is None:
        print(f"  FAIL: {status}")
        # 失敗時も空 CSV を生成 (downstream の存在チェック対策)
        with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
            csv.DictWriter(f, fieldnames=OUT_HEADER).writeheader()
        sys.exit(1)

    if not rows:
        print(f"  → no NAR races on {date} (empty calendar)")
    else:
        # 場・レース番号 で sort
        rows.sort(key=lambda r: (r['course_code'], r['race_num'].zfill(2)))
        # 場別件数 表示
        from collections import Counter
        ct = Counter(r['course'] for r in rows if r['course'])
        for c, n in sorted(ct.items()):
            print(f"  {c}: {n} races")
        print(f"  total: {len(rows)} races")

    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=OUT_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] {out_path}")


if __name__ == '__main__':
    main()
