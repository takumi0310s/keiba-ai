#!/usr/bin/env python
"""毎朝AM3:00自動実行: 今週末レースのプレミアムデータ事前取得

取得データ:
  - 調教タイム（最終追切 4F/3F/1F + ランク + 強度）
  - タイム指数（max, avg5, dist, course, run1-3）
  - 厩舎コメント（スコア付き）
  - JRDB前日データ（KYI/CYB/ZED/JOA/KAB → IDM・調教指数・激走指数等）

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


def fetch_jrdb_pre_race(date_str, dry_run=False):
    """JRDB前日データ（KYI/CYB/ZED/JOA/KAB）をダウンロード・パース・CSV追記。

    Args:
        date_str: YYYYMMDD形式の日付
        dry_run: Trueならダウンロードしない

    Returns:
        dict: {type: row_count} 取得結果サマリー
    """
    try:
        from scrape_jrdb import fetch_and_parse, save_csv
    except ImportError:
        print("  [WARN] scrape_jrdb import failed, skipping JRDB")
        return {}

    # YYYYMMDD → YYMMDD（JRDBファイル名形式）
    jrdb_date = date_str[2:]  # '20260405' → '260405'
    types = ['KYI', 'CYB', 'ZED', 'JOA', 'KAB']
    results = {}

    print(f"\n  --- JRDB前日データ取得 ({jrdb_date}) ---")

    if dry_run:
        print(f"  [DRY RUN] Would fetch: {', '.join(types)}")
        return {}

    for ft in types:
        try:
            df = fetch_and_parse(ft, jrdb_date)
            if df is not None and len(df) > 0:
                save_csv(df, ft, append=True)
                results[ft] = len(df)
                print(f"  {ft}: {len(df)} records saved")
            else:
                print(f"  {ft}: データなし")
                results[ft] = 0
        except Exception as e:
            print(f"  {ft}: ERROR - {e}")
            results[ft] = -1
        time.sleep(2)

    # Extended JRDB data (CHA/JO/KTA/KKA)
    try:
        import subprocess
        # ★6/12 Fable統合: KTA/KKA の日次直接フェッチを先行実行 (直近8日)。
        # 旧経路はアーカイブindex依存(KTA)・extracted非空スキップ(KKA)で新規DLゼロになり
        # KTA 4/5・KKA 5/3 の内容停止を起こした (FABLE_SWEEP_LOG.md Phase A/B)★
        from datetime import datetime as _dt, timedelta as _td
        _start = (_dt.now() - _td(days=8)).strftime('%Y%m%d')
        subprocess.run([sys.executable, os.path.join(BASE_DIR, 'tools', 'jrdb_daily_fix_fetch.py'),
                        '--types', 'kta', 'kka', '--start', _start], timeout=300)
        subprocess.run([sys.executable, os.path.join(BASE_DIR, 'tools', 'download_parse_jrdb_batch2.py'), '--types', 'cha', 'kta'], timeout=180)
        subprocess.run([sys.executable, os.path.join(BASE_DIR, 'tools', 'download_parse_jrdb_extra.py'), '--types', 'jo', 'kka'], timeout=180)
        print(f"  [JRDB] Extended data (CHA/JO/KTA/KKA) updated")
    except Exception as e:
        print(f"  [WARN] Extended JRDB fetch failed: {e}")

    return results


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

    # --- JRDB前日データ取得（KYI/CYB/ZED/JOA/KAB） ---
    # JRDBはレース単位ではなく開催日単位なので、各対象日で1回ずつ取得
    jrdb_results = {}
    base = datetime.strptime(date_str, '%Y%m%d')
    jrdb_dates_done = set()
    for delta in range(3):
        d = (base + timedelta(days=delta)).strftime('%Y%m%d')
        if d not in jrdb_dates_done:
            jrdb_dates_done.add(d)
            r = fetch_jrdb_pre_race(d, dry_run=args.dry_run)
            for k, v in r.items():
                jrdb_results[k] = jrdb_results.get(k, 0) + max(v, 0)

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

    # Session #27 修正: cache JSON → CSV 自動転換 (bug 修正)
    # cache に書き込むだけでは CSV 反映されない bug の恒久対策
    csv_to_csv_result = {"speed_index_new": 0, "training_new": 0, "stable_comments_new": 0}
    try:
        from premium_cache_to_csv import process_cache_dir
        csv_to_csv_result = process_cache_dir(date_str, dry_run=False)
        print(f"  [CSV append] speed_index +{csv_to_csv_result.get('speed_index_new', 0)}, "
              f"training +{csv_to_csv_result.get('training_new', 0)}, "
              f"comments +{csv_to_csv_result.get('stable_comments_new', 0)}")
    except Exception as e:
        print(f"  [WARN] cache→CSV append 失敗: {e}")

    print(f"\n\n{'=' * 60}")
    print(f"  COMPLETE")
    print(f"  Training: {n_training}/{len(new_ids)}")
    print(f"  Speed Index: {n_si}/{len(new_ids)}")
    print(f"  Comments: {n_comment}/{len(new_ids)}")
    print(f"  Shinba Eval: {n_shinba}/{len(new_ids)}")
    print(f"  CSV append: si+{csv_to_csv_result.get('speed_index_new', 0)} / "
          f"tr+{csv_to_csv_result.get('training_new', 0)} / "
          f"sc+{csv_to_csv_result.get('stable_comments_new', 0)}")
    if jrdb_results:
        jrdb_str = ', '.join(f"{k}:{v}" for k, v in jrdb_results.items() if v > 0)
        print(f"  JRDB: {jrdb_str or 'なし'}")
    print(f"  Cache: {cache_file}")
    print(f"{'=' * 60}")

    # --- 週末限定データ取得（波乱度・AI予測・データ分析）---
    # 金曜または木曜の場合のみ実行（出馬表確定後〜レース前）
    now = datetime.now()
    is_friday_run = now.weekday() == 4  # 金曜=4
    is_thursday_run = now.weekday() == 3  # 木曜=3 (翌日金曜分)
    if is_friday_run or is_thursday_run or args.date:
        print(f"\n  --- 週末限定データ取得 (newspaper/upset/analysis) ---")
        try:
            import subprocess
            # 明日から2日間（金曜→土日）
            tw_base = base + timedelta(days=1) if not args.date else base
            tw_date = tw_base.strftime('%Y%m%d')
            cmd = [sys.executable, os.path.join(BASE_DIR, 'tools', 'scrape_weekend_thisweek.py'),
                   '--date', tw_date]
            if args.dry_run:
                cmd.append('--dry-run')
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                                    encoding='utf-8', errors='replace')
            if result.stdout:
                for line in result.stdout.strip().split('\n')[-5:]:
                    print(f"    {line}")
            if result.returncode != 0 and result.stderr:
                print(f"    [WARN] thisweek scrape stderr: {result.stderr[-200:]}")
        except Exception as e:
            print(f"    [WARN] thisweek scrape failed: {e}")

    try:
        from notify import send_discord
        jrdb_msg = ""
        if jrdb_results:
            jrdb_parts = [f"{k}:{v}" for k, v in jrdb_results.items() if v > 0]
            if jrdb_parts:
                jrdb_msg = f"\nJRDB: {', '.join(jrdb_parts)}"
        send_discord("Premium Pre-fetch完了",
                     f"{date_str}: {len(new_ids)}R取得\n"
                     f"調教: {n_training} / 指数: {n_si} / コメント: {n_comment}"
                     + (f" / 新馬: {n_shinba}" if n_shinba else "")
                     + jrdb_msg,
                     color="green")
    except Exception:
        pass


if __name__ == '__main__':
    # SCRAPER-GUARD: 土日でも 03:00-05:59 の早朝スロットは許可 (caller="daily_premium_scrape")。
    # それ以外の時間帯 (土日 06:00+ 等) は即終了し、次回起動を妨げない。
    from tools.scraper_guard import check_scraping_allowed
    check_scraping_allowed(caller="daily_premium_scrape", mode="exit")
    main()
