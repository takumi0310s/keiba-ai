#!/usr/bin/env python
"""netkeiba 専門家印 / みんなの印 collect scraper (Phase 22 Agent B)

netkeiba スーパープレミアム + マスターコース cookie が前提。

★ 実調査済み URL & 構造 (2026-05-11 probe で確認):

  1) https://race.netkeiba.com/yoso/yoso_pro_opinion_list.html?race_id=XXX
     → 専門家 (=newspaper / yosoka) 毎の TM 印 list (本命/対抗/単穴/連下/星/押え)
     構造:
       <table class="YosoShirushiTable01">  # 1 table per expert
         <thead><tr><th>専門家名</th></tr></thead>
         <tbody>
           <tr>
             <th class="Mark_Pro"><span class="Icon_Shirushi Icon_Honmei"></span></th>
             <td><span class="Num WakuN">{umaban}</span>
                 <a href="/horse/yyyy...">{horse_name}</a>
                 <span class="fwN">({pop}人気)</span></td>
           </tr> ...
       </table>
     Icon_Shirushi suffix → mark:
       Icon_Honmei  = ◎ (本命, 1 per expert)
       Icon_Taikou  = ○ (対抗)
       Icon_Kurosan = ▲ (単穴, 黒三角)
       Icon_Hoshi   = ☆ (星 / 注)
       Icon_Osae    = △ (連下 / 押え)

  2) https://race.netkeiba.com/yoso/mark_list.html?race_id=XXX
     → みんなの印 集計。 prediction count が 0 のレースもあるため低 yield。
     現状 server-side ではグレー (XHR 経由)。 当面 集計値は API か空欄でスキップ。

★ 出力 schema (race_id × umaban で 1 行、 専門家集計):
  race_id, horse_id, umaban, horse_name,
  tm_honmei,   # ◎ の数 (= 専門家数のうち本命)
  tm_taikou,   # ○ の数
  tm_kurosan,  # ▲ の数
  tm_hoshi,    # ☆ の数
  tm_osae,     # △ の数
  tm_marked,   # ◎○▲☆△ どれか付いた専門家数
  tm_score,    # 重みづけスコア (◎5 ○3 ▲2 ☆1 △0.5)
  n_experts,   # 専門家総数 (列数)
  fetched_at

★ DRY-RUN mode 必須。 規約遵守: 私的利用、 配布 NG。

Usage:
    python tools/bulk_scrape_expert_marks.py --probe 202506050811  # 有馬記念で確認
    python tools/bulk_scrape_expert_marks.py --year-from 2024 --year-to 2026 --dry-run
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')

OUTPUT_CSV = os.path.join(DATA_DIR, 'netkeiba_expert_marks.csv')
COOKIES_JSON = os.path.join(DATA_DIR, 'cookies.json')

CSV_HEADER = [
    'race_id', 'horse_id', 'umaban', 'horse_name',
    'tm_honmei', 'tm_taikou', 'tm_kurosan', 'tm_hoshi', 'tm_osae',
    'tm_marked', 'tm_score', 'n_experts',
    'fetched_at',
]

HEADERS_BASE = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
    'Referer': 'https://race.netkeiba.com/',
}

DELAY_MIN = 3.0
DELAY_MAX = 5.0
RETRY_DELAY = 30
MAX_RETRIES = 3

# Icon_Shirushi suffix → (japanese, weight)
ICON_MAP = {
    'Icon_Honmei':  ('honmei',  5.0),
    'Icon_Taikou':  ('taikou',  3.0),
    'Icon_Kurosan': ('kurosan', 2.0),
    'Icon_Hoshi':   ('hoshi',   1.0),
    'Icon_Osae':    ('osae',    0.5),
}


# ============ Cookie / session ============

def _load_cookies_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            arr = json.load(f)
        return {c['name']: c['value'] for c in arr if 'name' in c and 'value' in c}
    except Exception as e:
        print(f"  WARN: cookies.json load failed: {e}")
        return {}


def _load_env_cookie():
    env_path = os.path.join(BASE_DIR, '.env')
    if not os.path.exists(env_path):
        return ''
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('NETKEIBA_COOKIE='):
                return line.strip().split('=', 1)[1].strip('"').strip("'")
    return ''


def _build_session():
    session = requests.Session()
    session.headers.update(HEADERS_BASE)
    cj = _load_cookies_json(COOKIES_JSON)
    for k, v in cj.items():
        session.cookies.set(k, v, domain='.netkeiba.com')
    env_cookie = _load_env_cookie()
    if env_cookie:
        for part in env_cookie.split(';'):
            part = part.strip()
            if '=' not in part:
                continue
            k, v = part.split('=', 1)
            k, v = k.strip(), v.strip()
            if k not in session.cookies:
                session.cookies.set(k, v, domain='.netkeiba.com')
    print(f"  cookies loaded: {len(session.cookies)} (json={len(cj)}, env present={bool(env_cookie)})")
    return session


def _get(session, url, retry=0, encoding='EUC-JP'):
    try:
        resp = session.get(url, timeout=20)
        if resp.status_code in (403, 429):
            if retry < MAX_RETRIES:
                wait = RETRY_DELAY * (retry + 1)
                print(f"\n  {resp.status_code} - waiting {wait}s (retry {retry+1}/{MAX_RETRIES})")
                time.sleep(wait)
                return _get(session, url, retry + 1, encoding)
            return None
        if resp.status_code != 200:
            return None
        resp.encoding = encoding
        return resp
    except Exception:
        if retry < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return _get(session, url, retry + 1, encoding)
        return None


# ============ Parser ============

def _classify_icon(span):
    """Icon_Shirushi span → ('honmei'/'taikou'/.../None, weight)"""
    cls = span.get('class', []) or []
    for c in cls:
        if c in ICON_MAP:
            return ICON_MAP[c]
    return (None, 0.0)


def parse_pro_opinion_list(html):
    """yoso_pro_opinion_list.html を parse して
    {umaban: {'horse_id', 'horse_name', counts(dict), score(float)}} を返す。
    n_experts は別途 count される。
    """
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table', class_='YosoShirushiTable01')
    n_experts = 0
    horses = defaultdict(lambda: {
        'horse_id': '', 'horse_name': '',
        'honmei': 0, 'taikou': 0, 'kurosan': 0, 'hoshi': 0, 'osae': 0,
        'marked': 0, 'score': 0.0,
    })

    for table in tables:
        # Each table = one expert. (Some empty tables exist for layout)
        rows = table.find_all('tr')
        # Expect at least 1 row with a mark
        had_mark = False
        for tr in rows:
            th = tr.find('th', class_='Mark_Pro')
            if not th:
                continue
            span = th.find('span', class_='Icon_Shirushi')
            if not span:
                continue
            mark_name, weight = _classify_icon(span)
            if not mark_name:
                continue
            # Get horse cell
            td = tr.find('td')
            if not td:
                continue
            num_span = td.find('span', class_='Num')
            if not num_span:
                continue
            try:
                umaban = int(re.sub(r'[^\d]', '', num_span.get_text(strip=True)) or 0)
            except ValueError:
                continue
            if umaban == 0:
                continue
            a = td.find('a', href=re.compile(r'/horse/'))
            horse_name = ''
            horse_id = ''
            if a:
                horse_name = a.get_text(strip=True)[:30]
                m = re.search(r'/horse/(\w+)', a['href'])
                if m:
                    horse_id = m.group(1)
            h = horses[umaban]
            h['horse_id'] = h['horse_id'] or horse_id
            h['horse_name'] = h['horse_name'] or horse_name
            h[mark_name] += 1
            h['marked'] += 1
            h['score'] += weight
            had_mark = True
        if had_mark:
            n_experts += 1

    return horses, n_experts


# ============ Scrape (1 race) ============

def scrape_one_race(session, race_id, verbose=False):
    fetched_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    url = f"https://race.netkeiba.com/yoso/yoso_pro_opinion_list.html?race_id={race_id}"
    resp = _get(session, url)
    if resp is None:
        if verbose:
            print(f"    [TM] {url} -> HTTP fail")
        return []
    horses, n_experts = parse_pro_opinion_list(resp.text)
    if verbose:
        print(f"    [TM] {url}")
        print(f"      -> {len(horses)} horses, {n_experts} experts")

    out_rows = []
    for u in sorted(horses.keys()):
        h = horses[u]
        out_rows.append([
            race_id, h['horse_id'], u, h['horse_name'],
            h['honmei'], h['taikou'], h['kurosan'], h['hoshi'], h['osae'],
            h['marked'], round(h['score'], 2), n_experts,
            fetched_at,
        ])
    return out_rows


# ============ CSV ============

def _append_csv(path, rows):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', encoding='utf-8-sig', newline='') as f:
        if write_header:
            f.write(','.join(CSV_HEADER) + '\n')
        for row in rows:
            f.write(','.join(str(v).replace(',', '；') for v in row) + '\n')


def load_existing_race_ids():
    if not os.path.exists(OUTPUT_CSV):
        return set()
    try:
        df = pd.read_csv(OUTPUT_CSV, encoding='utf-8-sig', usecols=['race_id'], dtype=str)
        return set(df['race_id'].unique())
    except Exception:
        return set()


def _target_to_netkeiba(target_rid):
    rid = str(target_rid).zfill(10)
    cc, yy, k, n, rr = rid[0:2], rid[2:4], rid[4:5], rid[5:6], rid[6:8]
    n_map = {'A': '10', 'B': '11', 'C': '12'}
    n_dec = n_map.get(n, n.zfill(2))
    return f"20{yy}{cc}{k.zfill(2)}{n_dec}{rr}"


def build_task_list(year_from, year_to):
    csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
    races = pd.read_csv(csv_path, encoding='utf-8-sig',
                        dtype=str, low_memory=False,
                        usecols=['race_id', 'year', 'class_code'])
    races['year_int'] = pd.to_numeric(races['year'], errors='coerce')
    races['nk_id'] = races['race_id'].apply(
        lambda x: _target_to_netkeiba(x) if pd.notna(x) else '')
    existing = load_existing_race_ids()
    tasks = []
    for year in range(year_from, year_to + 1):
        yr2 = year % 100
        yr_races = races[races['year_int'] == yr2]
        total = yr_races['nk_id'].nunique()
        new_ids = sorted(set(yr_races['nk_id'].unique()) - existing)
        tasks.extend(new_ids)
        already = total - len(new_ids)
        print(f"  {year}: {len(new_ids)} new / {total} target ({already} already done)")
    return tasks


# ============ Commands ============

def cmd_probe(args):
    rid = args.probe
    print(f"\n  PROBE: race_id={rid}")
    session = _build_session()
    rows = scrape_one_race(session, rid, verbose=True)
    print(f"\n  TOTAL rows extracted: {len(rows)}")
    for r in rows[:5]:
        print('    ', r)
    if rows:
        existing = load_existing_race_ids()
        if rid in existing:
            print(f"  NOTE: race_id {rid} already in csv, skipping write")
        else:
            _append_csv(OUTPUT_CSV, rows)
            print(f"  appended {len(rows)} rows to {OUTPUT_CSV}")
    return 0


def cmd_run(args):
    print("=" * 60)
    print(f"  Expert Marks Bulk Scraper (Pro TM marks)")
    print(f"  Years: {args.year_from}-{args.year_to}")
    print(f"  Dry-run: {args.dry_run}")
    print("=" * 60)
    session = _build_session() if not args.dry_run else None
    existing = load_existing_race_ids()
    print(f"  Existing: {len(existing)} unique race_ids\n")
    tasks = build_task_list(args.year_from, args.year_to)
    if args.limit > 0:
        tasks = tasks[:args.limit]
    total = len(tasks)
    print(f"\n  Total to scrape: {total} races")
    if total == 0:
        print("  Nothing to do.")
        return 0
    avg_delay = (DELAY_MIN + DELAY_MAX) / 2
    est_min = total * avg_delay / 60
    print(f"  Estimated time: ~{est_min:.0f} min ({est_min/60:.1f} h)\n")
    if args.dry_run:
        print("  [DRY-RUN] No HTTP request issued. Sample task race_ids:")
        for r in tasks[:10]:
            print(f"    {r}")
        return 0

    stats = {'scraped': 0, 'rows': 0, 'empty': 0, 'errors': 0}
    start = time.time()
    for i, rid in enumerate(tasks):
        pct = (i + 1) / total * 100
        elapsed = time.time() - start
        rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
        eta_min = (total - i - 1) / rate if rate > 0 else 0
        print(f"\r  [{i+1}/{total} {pct:.1f}%] {rid} | {rate:.1f}R/min | ETA {eta_min:.0f}min | "
              f"OK:{stats['scraped']} empty:{stats['empty']} err:{stats['errors']}",
              end='', flush=True)
        try:
            rows = scrape_one_race(session, rid, verbose=False)
            if rows:
                _append_csv(OUTPUT_CSV, rows)
                stats['scraped'] += 1
                stats['rows'] += len(rows)
            else:
                stats['empty'] += 1
        except Exception as e:
            stats['errors'] += 1
            if stats['errors'] <= 5:
                print(f"\n  ERROR: {e}")
        delay = DELAY_MIN + np.random.random() * (DELAY_MAX - DELAY_MIN)
        time.sleep(delay)

    elapsed_min = (time.time() - start) / 60
    print(f"\n\n{'=' * 60}")
    print(f"  COMPLETE ({elapsed_min:.1f} min)")
    print(f"  Scraped: {stats['scraped']} R, {stats['rows']} rows")
    print(f"  Empty: {stats['empty']}, Errors: {stats['errors']}")
    print("=" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year-from', type=int, default=2024)
    parser.add_argument('--year-to', type=int, default=2026)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--probe', type=str, default='', help='1 race_id だけテスト')
    args = parser.parse_args()
    if args.probe:
        return cmd_probe(args)
    return cmd_run(args)


if __name__ == '__main__':
    try:
        from tools.scraper_guard import check_scraping_allowed
        if '--probe' not in sys.argv and '--dry-run' not in sys.argv:
            check_scraping_allowed()
    except Exception:
        pass
    sys.exit(main() or 0)
