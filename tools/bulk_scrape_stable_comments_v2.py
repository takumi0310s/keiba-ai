#!/usr/bin/env python
"""厩舎コメント拡張 scraper v2 (Phase 22 Agent B)

既存 data/netkeiba_stable_comments.csv (~7,500 race_ids, WF coverage ~30%) を
拡張するための後継 scraper。

主な改良点 (vs bulk_scrape_comments.py):
  1. data/cookies.json (Playwright形式) と .env NETKEIBA_COOKIE の両対応
  2. 出力先を別 csv (netkeiba_stable_comments_v2.csv) で安全に分離
  3. 既存 v1 csv の race_id と新規 v2 csv 両方を resume 対象に
  4. DRY-RUN モード (--dry-run): 取得対象件数のみ表示、scrape はしない
  5. --year-from / --year-to で柔軟な期間指定
  6. SCRAPER-GUARD は OPERATIONAL_CALLERS 外なので 通常 window 内で実行

netkeiba 規約遵守: 私的利用範囲のみ。配布 NG。 rate limit 2-4 秒/req。

Usage:
    # 1 race_id でプローブ (DRY-RUN ではなく実行 1 件のみ)
    python tools/bulk_scrape_stable_comments_v2.py --probe 202608030611

    # 2024-2026 高クラスのみ DRY-RUN
    python tools/bulk_scrape_stable_comments_v2.py --year-from 2024 --year-to 2026 --dry-run

    # 本番 (例: 2024-2026 高クラス、上限なし)
    python tools/bulk_scrape_stable_comments_v2.py --year-from 2024 --year-to 2026
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
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

# v1 (既存) と v2 (新規) を両方読んで重複 skip
V1_CSV = os.path.join(DATA_DIR, 'netkeiba_stable_comments.csv')
V2_CSV = os.path.join(DATA_DIR, 'netkeiba_stable_comments_v2.csv')
COOKIES_JSON = os.path.join(DATA_DIR, 'cookies.json')

# v1 と同じ schema にして merge 互換性を確保
CSV_HEADER = ['race_id', 'race_date', 'umaban', 'horse_name', 'comment', 'score']

HEADERS_BASE = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
}

DELAY_MIN = 2.0
DELAY_MAX = 4.0
RETRY_DELAY = 30
MAX_RETRIES = 3

# netkeiba ではコメントは中〜上クラスで充実。低クラス (新馬/未勝利) は ~0
HIGH_CLASSES = {'15', '43', '67', '115', '131', '163', '179', '195'}


# ============ Cookie / session ============

def _load_cookies_json(path):
    """Playwright export 形式の cookies.json を dict 化"""
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
    """data/cookies.json (優先) + .env NETKEIBA_COOKIE を merge"""
    session = requests.Session()
    session.headers.update(HEADERS_BASE)

    # 1) cookies.json
    cj = _load_cookies_json(COOKIES_JSON)
    for k, v in cj.items():
        session.cookies.set(k, v, domain='.netkeiba.com')

    # 2) .env (既存値を上書きせず追加)
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


# ============ HTTP ============

def _get(session, url, retry=0):
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code in (403, 429):
            if retry < MAX_RETRIES:
                wait = RETRY_DELAY * (retry + 1)
                print(f"\n  {resp.status_code} - waiting {wait}s (retry {retry+1}/{MAX_RETRIES})")
                time.sleep(wait)
                return _get(session, url, retry + 1)
            return None
        if resp.status_code != 200:
            return None
        resp.encoding = 'EUC-JP'
        return resp
    except Exception:
        if retry < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return _get(session, url, retry + 1)
        return None


# ============ score (v1 と同じロジック) ============

def _score_comment(text):
    if not text:
        return 0
    pos3 = ['絶好調', '抜群の動き', '破格', '文句なし']
    pos2 = ['好調', '上昇', '好仕上がり', '万全', '充実', '態勢整', '上積み']
    pos1 = ['順調', '落ち着き', 'まずまず', '元気', '状態面問題な', '上向き', 'いい動き',
            '力は出せる', '態勢は整', 'しっかり']
    neg1 = ['平凡', '変わり身', '現状維持', '微妙', '物足りな']
    neg2 = ['不安', '太め', '細め', '久々', '叩き台', '上積み必要', 'ピリッとし']
    neg3 = ['故障', '出来落ち', '復帰戦', '大幅プラス', '大幅マイナス']

    for w in pos3:
        if w in text:
            return 3
    for w in neg3:
        if w in text:
            return -3
    score = 0
    for w in pos2:
        if w in text:
            score = max(score, 2)
    for w in neg2:
        if w in text:
            score = min(score, -2)
    if score != 0:
        return score
    for w in pos1:
        if w in text:
            return 1
    for w in neg1:
        if w in text:
            return -1
    return 0


# ============ Scrape ============

def scrape_comment(session, race_id):
    """1 race の comment.html から行 list を返す"""
    url = f"https://race.netkeiba.com/race/comment.html?race_id={race_id}"
    resp = _get(session, url)
    if resp is None:
        return []
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', class_=re.compile(r'Stable_Comment|Comment_Table'))
    if not table:
        return []

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
        results.append([race_id, '', umaban, horse_name, comment, score])
    return results


# ============ CSV ============

def _append_csv(path, rows):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', encoding='utf-8-sig', newline='') as f:
        if write_header:
            f.write(','.join(CSV_HEADER) + '\n')
        for row in rows:
            f.write(','.join(str(v).replace(',', '；') for v in row) + '\n')


def load_existing_race_ids():
    """v1 + v2 両方の race_id を union"""
    ids = set()
    for path in (V1_CSV, V2_CSV):
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, encoding='utf-8-sig', usecols=['race_id'], dtype=str)
            ids |= set(df['race_id'].unique())
        except Exception:
            continue
    return ids


def _target_to_netkeiba(target_rid):
    rid = str(target_rid).zfill(10)
    cc, yy, k, n, rr = rid[0:2], rid[2:4], rid[4:5], rid[5:6], rid[6:8]
    n_map = {'A': '10', 'B': '11', 'C': '12'}
    n_dec = n_map.get(n, n.zfill(2))
    return f"20{yy}{cc}{k.zfill(2)}{n_dec}{rr}"


def build_task_list(year_from, year_to, all_classes=False):
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
        if not all_classes:
            yr_races = yr_races[yr_races['class_code'].isin(HIGH_CLASSES)]
        total = yr_races['nk_id'].nunique()
        new_ids = sorted(set(yr_races['nk_id'].unique()) - existing)
        tasks.extend(new_ids)
        already = total - len(new_ids)
        print(f"  {year}: {len(new_ids)} new / {total} target ({already} already done)")
    return tasks


# ============ main ============

def cmd_probe(args):
    """1 race_id だけ叩いて構造を確認"""
    session = _build_session()
    rid = args.probe
    print(f"\n  PROBE: race_id={rid}")
    rows = scrape_comment(session, rid)
    print(f"  -> {len(rows)} rows")
    for r in rows[:5]:
        print('    ', r)
    if rows:
        # 重複 skip しつつ v2 csv に保存
        existing = load_existing_race_ids()
        if rid in existing:
            print(f"  NOTE: race_id {rid} already in v1/v2 csv, skipping write")
        else:
            _append_csv(V2_CSV, rows)
            print(f"  appended {len(rows)} rows to {V2_CSV}")
    return 0


def cmd_run(args):
    print("=" * 60)
    print(f"  Stable Comment Bulk Scraper v2")
    print(f"  Years: {args.year_from}-{args.year_to}")
    print(f"  Classes: {'ALL' if args.all_classes else 'HIGH'}")
    print(f"  Dry-run: {args.dry_run}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    session = _build_session() if not args.dry_run else None

    existing = load_existing_race_ids()
    print(f"  Existing (v1+v2 merged): {len(existing)} unique race_ids\n")

    tasks = build_task_list(args.year_from, args.year_to, args.all_classes)
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
        print("  [DRY-RUN] No HTTP request issued. Showing first 10 task race_ids:")
        for r in tasks[:10]:
            print(f"    {r}")
        return 0

    stats = {'scraped': 0, 'rows': 0, 'empty': 0, 'errors': 0}
    start_time = time.time()

    for i, race_id in enumerate(tasks):
        pct = (i + 1) / total * 100
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
        eta_min = (total - i - 1) / rate if rate > 0 else 0
        print(f"\r  [{i+1}/{total} {pct:.1f}%] {race_id} | "
              f"{rate:.1f}R/min | ETA {eta_min:.0f}min | "
              f"OK:{stats['scraped']} empty:{stats['empty']} err:{stats['errors']}",
              end='', flush=True)
        try:
            rows = scrape_comment(session, race_id)
            if rows:
                _append_csv(V2_CSV, rows)
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

    elapsed_min = (time.time() - start_time) / 60
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
    parser.add_argument('--all-classes', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--probe', type=str, default='', help='1 race_id だけテスト')
    args = parser.parse_args()

    if args.probe:
        return cmd_probe(args)
    return cmd_run(args)


if __name__ == '__main__':
    # SCRAPER-GUARD は probe / dry-run も含む通常 path で適用
    # (本番タスクとして登録する際は scraper_guard を caller="bulk_scrape_v2" で wrap)
    try:
        from tools.scraper_guard import check_scraping_allowed
        # probe / dry-run はガード bypass (実 HTTP は 1 件 or 0 件)
        if '--probe' not in sys.argv and '--dry-run' not in sys.argv:
            check_scraping_allowed()
    except Exception:
        pass
    sys.exit(main() or 0)
