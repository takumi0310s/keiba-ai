"""NAR 当日 出馬表 (shutuba) + 前夜オッズ scraper.

predict_nar.py --date YYYYMMDD で読まれる data/nar_today_shutuba_YYYYMMDD.csv を生成。

URL:
  list:    https://nar.netkeiba.com/top/race_list_sub.html?kaisai_date=YYYYMMDD
  shutuba: https://nar.netkeiba.com/race/shutuba.html?race_id=...

table:
  table.RaceTable01.ShutubaTable
  header: 枠 馬番 印 馬名 性齢 斤量 厩舎 騎手 馬体重(増減) 予想オッズ 人気 ...
  data row: 15 cells、[1]=馬番 [3]=馬名 [4]=性齢 [5]=斤量 [7]=騎手 [8]=馬体重 [9]=odds [10]=pop_rank

note:
  予想オッズ (発走前) を採用、live 確定オッズ refresh は別 task (NarLiveOddsRefresh)。
  scrape_nar_all.py の create_session/safe_get/race_sleep を流用 (重複回避)。
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

# scrape_nar_all から helper を import (URL access policy 統一)
from scrape_nar_all import (
    NAR_TRACKS, USER_AGENTS, create_session, safe_get,
    race_sleep, nav_sleep, get_race_ids, _executor,
)

CONDITION_MAP = {'良': 0, '稍重': 1, '稍': 1, '重': 2, '不良': 3, '不': 3}

# NAR 騎手 名前先頭につく地区 prefix (jockey_stats は prefix なしで記録)
NAR_PREFECTURES = (
    '北海道', '神奈川',  # 3 chars
    '岩手', '埼玉', '千葉', '東京', '石川', '岐阜', '愛知', '兵庫', '高知', '佐賀',  # 2 chars
)


def strip_jockey_prefix(name: str) -> str:
    """NAR jockey の "北海道小国博行" → "小国博行" 形式に正規化."""
    if not name: return name
    for pref in NAR_PREFECTURES:
        if name.startswith(pref):
            return name[len(pref):]
    return name

OUT_HEADER = [
    'race_id', 'race_name', 'race_date', 'course', 'course_code', 'distance',
    'surface', 'surface_enc', 'condition', 'condition_enc', 'weather',
    'class_info', 'num_horses', 'race_time',
    'bracket', 'horse_num', 'horse_name', 'sex_age', 'weight_carry',
    'jockey', 'trainer', 'horse_weight', 'horse_weight_change',
    'odds', 'pop_rank',
]


def parse_shutuba(session, race_id, date_hint=''):
    """1 レース分の shutuba を取得して horse rows を返す."""
    url = f'https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}'
    r = safe_get(session, url)
    if r is None:
        return None, 'timeout'
    if r.status_code != 200:
        return None, f'http_{r.status_code}'

    try:
        text = r.content.decode('euc-jp', errors='replace')
        soup = BeautifulSoup(text, 'html.parser')
        title = soup.title.string if soup.title else ''
        if 'エラー' in title or 'Error' in title:
            return None, 'error_page'

        # race meta
        rn_el = soup.select_one('.RaceName')
        race_name = rn_el.get_text(strip=True) if rn_el else ''

        rd1_el = soup.select_one('.RaceData01')
        rd1 = rd1_el.get_text(strip=True) if rd1_el else ''
        rd2_el = soup.select_one('.RaceData02')
        rd2 = rd2_el.get_text(' ', strip=True) if rd2_el else ''

        distance, surface, condition, weather = 0, '', '', ''
        race_time = ''
        tm = re.search(r'(\d{1,2}:\d{2})', rd1)
        if tm: race_time = tm.group(1)
        dm = re.search(r'([ダ芝障])(\d+)m', rd1)
        if dm: surface, distance = dm.group(1), int(dm.group(2))
        cm = re.search(r'馬場:(\S+)', rd1)
        if cm: condition = cm.group(1)
        wm = re.search(r'天候:(\S+)', rd1)
        if wm: weather = wm.group(1)

        course, course_code = '', race_id[4:6] if len(race_id) >= 6 else ''
        for code, name in NAR_TRACKS.items():
            if name in rd2:
                course, course_code = name, str(code)
                break

        clm = re.search(r'(C\d|B\d|A\d|2勝|3勝|サラ系\S+)', rd2)
        class_info = clm.group(1) if clm else ''

        race_date = date_hint
        if not race_date:
            tm2 = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', title)
            if tm2:
                race_date = f'{tm2.group(1)}{int(tm2.group(2)):02d}{int(tm2.group(3)):02d}'

        # surface_enc / condition_enc
        surface_enc = 0 if '芝' in surface else (2 if '障' in surface else 1)
        condition_enc = CONDITION_MAP.get(condition, 0)

        # shutuba table
        t = soup.select_one('table.RaceTable01.ShutubaTable')
        if not t:
            return None, 'no_table'

        all_rows = t.find_all('tr')
        if len(all_rows) < 2:
            return None, 'no_rows'

        # data 行は cells >= 14 (header は 14 列)
        horse_rows = []
        for row in all_rows:
            cells = row.find_all(['td'])
            if len(cells) < 10:
                continue  # header / グループ行 skip
            t = [c.get_text(strip=True) for c in cells]
            n = len(t)
            # 動的 layout: 馬番 in cell[1], 馬名 in cell[3]
            try:
                horse_num = t[1] if len(t) > 1 else ''
                # validate: horse_num should be 1-2 digit number
                if not re.match(r'^\d{1,2}$', horse_num):
                    continue
            except Exception:
                continue
            bracket = t[0]
            horse_name = t[3] if n > 3 else ''
            sex_age = t[4] if n > 4 else ''
            weight_carry = t[5] if n > 5 else ''
            # NAR 場合: [6] 厩舎 / [7] 騎手 (header: 厩舎/騎手 順だが check)
            trainer = t[6] if n > 6 else ''
            jockey_raw = t[7] if n > 7 else ''
            jockey = strip_jockey_prefix(jockey_raw)
            # pre-race shutuba:
            #   [8] = 予想オッズ range text (例 "54.110") — 馬体重 は当日 announced
            #   [9] = 予想オッズ (single)
            #   [10] = 人気
            # post-race の場合は [8] が "馬体重(増減)" (例 "490(+2)") に変わる場合あり → 両対応
            hw_raw = t[8] if n > 8 else ''
            horse_weight, horse_weight_change = '', ''
            hwm = re.match(r'^(\d{3})\(([+-]?\d+)\)$', hw_raw)  # strict match: 馬体重 形式
            if hwm:
                horse_weight, horse_weight_change = hwm.group(1), hwm.group(2)
            # else: horse_weight は空 (pre-race で発表前) — predict_nar.py で default 480 fill
            odds = t[9] if n > 9 else ''
            pop_rank = t[10] if n > 10 else ''

            horse_rows.append({
                'race_id': race_id, 'race_name': race_name, 'race_date': race_date,
                'course': course, 'course_code': course_code, 'distance': distance,
                'surface': surface, 'surface_enc': surface_enc,
                'condition': condition, 'condition_enc': condition_enc, 'weather': weather,
                'class_info': class_info, 'num_horses': 0,  # filled later
                'race_time': race_time,
                'bracket': bracket, 'horse_num': horse_num, 'horse_name': horse_name,
                'sex_age': sex_age, 'weight_carry': weight_carry,
                'jockey': jockey, 'trainer': trainer,
                'horse_weight': horse_weight, 'horse_weight_change': horse_weight_change,
                'odds': odds, 'pop_rank': pop_rank,
            })

        # num_horses = 行数
        nh = len(horse_rows)
        for h in horse_rows:
            h['num_horses'] = nh

        return horse_rows, 'ok'

    except Exception as e:
        return None, f'parse_err: {e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=None, help='YYYYMMDD (default 今日)')
    parser.add_argument('--tracks', default=None,
                        help='場 名 カンマ区切り (例: "船橋,大井") - default 全 NAR 場')
    parser.add_argument('--output', default=None, help='出力 CSV (default data/nar_today_shutuba_DATE.csv)')
    parser.add_argument('--limit', type=int, default=0, help='最大 race 件数 (テスト用、0 = 制限なし)')
    args = parser.parse_args()

    date = args.date or datetime.now().strftime('%Y%m%d')
    out_path = args.output or f'data/nar_today_shutuba_{date}.csv'

    track_filter = None
    if args.tracks:
        track_filter = set(args.tracks.split(','))

    print(f"=== scrape_nar_today {date} ===")
    print(f"  tracks: {track_filter or 'ALL'}")
    print(f"  output: {out_path}")
    print(f"  limit:  {args.limit or 'none'}")

    session = create_session()
    nav_sleep()

    race_ids = get_race_ids(session, date)
    print(f"  race_ids: {len(race_ids)}")
    if not race_ids:
        print("  → no NAR races today、空 csv 生成 で skip")
        # 空 CSV
        with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
            csv.DictWriter(f, fieldnames=OUT_HEADER).writeheader()
        return

    # filter by track
    if track_filter:
        # race_id [4:6] で 場 code 推定
        # NAR_TRACKS は code → name、reverse map
        name_to_code = {v: str(k) for k, v in NAR_TRACKS.items()}
        wanted_codes = {name_to_code[t] for t in track_filter if t in name_to_code}
        race_ids = [rid for rid in race_ids if rid[4:6] in wanted_codes]
        print(f"  filtered to {len(race_ids)} race_ids by track")

    if args.limit > 0:
        race_ids = race_ids[: args.limit]

    all_rows = []
    fail_count = 0
    t_start = time.time()
    for i, rid in enumerate(race_ids, 1):
        print(f"  [{i}/{len(race_ids)}] {rid} ...", end='', flush=True)
        rows, status = parse_shutuba(session, rid, date_hint=date)
        if rows:
            all_rows.extend(rows)
            print(f" OK ({len(rows)}h, {status})")
        else:
            fail_count += 1
            print(f" FAIL ({status})")
        if i < len(race_ids):
            race_sleep()

    elapsed = time.time() - t_start
    print(f"\nelapsed: {elapsed/60:.1f} min, total horses: {len(all_rows)}, failed races: {fail_count}")

    # write CSV
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=OUT_HEADER)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"[OK] {out_path}")


if __name__ == '__main__':
    main()
