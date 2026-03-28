#!/usr/bin/env python
"""毎朝AM3:00自動実行: 今週末レースのプレミアムデータ事前取得

取得データ:
  - 調教タイム（最終追切 4F/3F/1F + ランク + 強度）
  - タイム指数（max, avg5, dist, course, run1-3）
  - 厩舎コメント（スコア付き）

キャッシュ: data/weekly_premium_cache/{date}/premium_cache.json
予測時にキャッシュがあればスクレイピング不要で高速化

Usage:
    python tools/daily_premium_scrape.py              # 今日+明日のレース
    python tools/daily_premium_scrape.py --date 20260328
    python tools/daily_premium_scrape.py --dry-run    # リクエストなしで計画表示
"""
import os
import sys
import io
import re
import json
import time
import argparse
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

if sys.platform == 'win32' and getattr(sys.stdout, 'encoding', '') != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'weekly_premium_cache')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DELAY = 5


def get_race_ids_for_date(date_str):
    """netkeibaの開催情報から当日のrace_idリストを取得"""
    try:
        from scrape_training import _make_session
        session = _make_session()
    except Exception:
        session = None
    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)

    race_ids = []
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
    try:
        resp = session.get(url, timeout=15)
        resp.encoding = 'EUC-JP'
        for m in re.finditer(r'race_id=(\d{12})', resp.text):
            race_ids.append(m.group(1))
    except Exception as e:
        print(f"  WARNING: race_list取得失敗: {e}")
    return sorted(set(race_ids))


def get_weekend_race_ids(base_date_str):
    """今日〜2日後までのレースIDを全取得"""
    base = datetime.strptime(base_date_str, '%Y%m%d')
    all_ids = []
    for delta in range(3):  # today, tomorrow, day after
        d = (base + timedelta(days=delta)).strftime('%Y%m%d')
        ids = get_race_ids_for_date(d)
        if ids:
            print(f"  {d}: {len(ids)} races")
            all_ids.extend(ids)
        time.sleep(1)
    return sorted(set(all_ids))


def _check_shinba(race_id):
    """レース名に「新馬」を含むか簡易チェック（shutuba.htmlのタイトルで判定）"""
    try:
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')
        title = soup.find('title')
        if title and '新馬' in title.text:
            return True
        race_name = soup.find(class_='RaceName')
        if race_name and '新馬' in race_name.get_text():
            return True
    except Exception:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(description="週末レースのプレミアムデータ事前取得")
    parser.add_argument('--date', type=str, default='')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime('%Y%m%d')
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 60)
    print(f"  Premium Data Pre-fetch: {date_str}")
    print("=" * 60)

    # Get race IDs for this weekend
    print("\n  Fetching race list...")
    race_ids = get_weekend_race_ids(date_str)
    print(f"  Total: {len(race_ids)} races")

    if not race_ids:
        print("  No races found. Exiting.")
        try:
            from notify import send_discord
            send_discord("Premium Pre-fetch", f"{date_str}: レースなし", color="yellow")
        except Exception:
            pass
        return

    # Check cache
    cache_path = os.path.join(CACHE_DIR, date_str)
    cache_file = os.path.join(cache_path, 'premium_cache.json')
    existing_cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            existing_cache = json.load(f)
    new_ids = [rid for rid in race_ids if rid not in existing_cache]
    print(f"  Cached: {len(existing_cache)}, New: {len(new_ids)}")

    if not new_ids:
        print("  All races already cached.")
        return

    if args.dry_run:
        print(f"  [DRY RUN] Would fetch {len(new_ids)} races")
        return

    # Import scrapers
    try:
        from scrape_training import fetch_training_times, fetch_stable_comments
        from scrape_speed_index import scrape_speed_index, _load_session
        si_session = _load_session()
    except ImportError as e:
        print(f"  Import error: {e}")
        return

    # Import shinba eval scraper
    try:
        from scrape_shinba_eval import scrape_newspaper as scrape_shinba_newspaper
    except ImportError:
        scrape_shinba_newspaper = None

    # Note: race review (備考) is scraped by daily_results.py after races finish,
    # not by pre-fetch (horses haven't raced yet)

    # Scrape each race
    os.makedirs(cache_path, exist_ok=True)
    all_data = dict(existing_cache)
    n_training = n_si = n_comment = n_shinba = n_err = 0

    for i, race_id in enumerate(new_ids):
        print(f"\r  [{i+1}/{len(new_ids)}] {race_id}", end='', flush=True)

        race_data = {'training': {}, 'speed_index': {}, 'comments': {}}

        try:
            training = fetch_training_times(race_id)
            if training:
                race_data['training'] = {str(k): v for k, v in training.items()}
                n_training += 1
        except Exception:
            pass

        if si_session:
            try:
                si_rows = scrape_speed_index(si_session, race_id)
                if si_rows:
                    for row in si_rows:
                        race_data['speed_index'][str(row[1])] = {
                            'horse_name': row[2],
                            'index_max': row[6], 'index_avg5': row[7],
                            'index_dist': row[8], 'index_course': row[9],
                            'index_run1': row[10], 'index_run2': row[11], 'index_run3': row[12],
                        }
                    n_si += 1
            except Exception:
                pass

        try:
            comments = fetch_stable_comments(race_id)
            if comments:
                race_data['comments'] = {str(k): v for k, v in comments.items()}
                n_comment += 1
        except Exception:
            pass

        # 新馬評価: レース名に「新馬」を含むか確認してから取得
        if scrape_shinba_newspaper:
            try:
                _is_shinba = _check_shinba(race_id)
                if _is_shinba:
                    shinba_rows = scrape_shinba_newspaper(race_id)
                    if shinba_rows and shinba_rows != 'blocked' and len(shinba_rows) > 0:
                        shinba_dict = {}
                        for row in shinba_rows:
                            shinba_dict[str(row['umaban'])] = {
                                'horse_name': row['horse_name'],
                                'stable_eval': row['stable_eval'],
                                'training_rank': row['training_rank'],
                                'stable_comment': row['stable_comment'],
                                'training_review': row['training_review'],
                                'comment_score': row['comment_score'],
                            }
                        race_data['shinba_eval'] = shinba_dict
                        n_shinba += 1
            except Exception:
                pass

        all_data[race_id] = race_data
        time.sleep(DELAY)

        # Save incrementally every 10 races
        if (i + 1) % 10 == 0:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2, default=str)

    # Final save
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n\n{'=' * 60}")
    print(f"  COMPLETE")
    print(f"  Training: {n_training}/{len(new_ids)}")
    print(f"  Speed Index: {n_si}/{len(new_ids)}")
    print(f"  Comments: {n_comment}/{len(new_ids)}")
    print(f"  Shinba Eval: {n_shinba}/{len(new_ids)}")
    print(f"  Cache: {cache_file}")
    print(f"{'=' * 60}")

    try:
        from notify import send_discord
        send_discord("Premium Pre-fetch完了",
                     f"{date_str}: {len(new_ids)}R取得\n"
                     f"調教: {n_training} / 指数: {n_si} / コメント: {n_comment}"
                     + (f" / 新馬: {n_shinba}" if n_shinba else ""),
                     color="green")
    except Exception:
        pass


if __name__ == '__main__':
    main()
