"""NAR 結果 (result) scraper. 当日終了 後 21:30 自動発火 想定.

predict_nar.py の retro 評価 + ROI 計算 用 data/nar_results_YYYYMMDD.csv を生成。

URL:
  result: https://nar.netkeiba.com/race/result.html?race_id=...

使い方:
  python tools/scrape_nar_results.py [--date YYYYMMDD] [--tracks 船橋,大井] [--output ...]

内部:
  scrape_nar_all.py の parse_race() を流用。
  当日 race_ids を get_race_ids で取得 → 各 race の結果を 1件ずつ scrape。
"""
from __future__ import annotations

import os, sys, argparse, csv, time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from datetime import datetime, timedelta

BASE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE)
sys.path.insert(0, os.path.join(BASE, 'tools'))

# scrape_nar_all から流用
from scrape_nar_all import (
    NAR_TRACKS, create_session, race_sleep, nav_sleep,
    get_race_ids, parse_race, CSV_HEADER,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=None,
                        help='YYYYMMDD (default 昨日)。NAR 結果は当日 21時頃 公開、22時以降に取得が安全')
    parser.add_argument('--tracks', default=None,
                        help='場 名 カンマ区切り (default 全場)')
    parser.add_argument('--output', default=None,
                        help='出力 CSV (default data/nar_results_DATE.csv)')
    parser.add_argument('--limit', type=int, default=0, help='テスト用 最大 race 件数')
    args = parser.parse_args()

    if args.date:
        date = args.date
    else:
        date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    out_path = args.output or f'data/nar_results_{date}.csv'

    track_filter = None
    if args.tracks:
        track_filter = set(args.tracks.split(','))

    print(f"=== scrape_nar_results {date} ===")
    print(f"  tracks: {track_filter or 'ALL'}")
    print(f"  output: {out_path}")

    session = create_session()
    nav_sleep()

    race_ids = get_race_ids(session, date)
    print(f"  race_ids: {len(race_ids)}")
    if not race_ids:
        print(f"  → no NAR races on {date}、空 CSV 生成")
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            csv.DictWriter(f, fieldnames=CSV_HEADER).writeheader()
        return

    # filter by track
    if track_filter:
        name_to_code = {v: str(k) for k, v in NAR_TRACKS.items()}
        wanted_codes = {name_to_code[t] for t in track_filter if t in name_to_code}
        race_ids = [rid for rid in race_ids if rid[4:6] in wanted_codes]
        print(f"  filtered to {len(race_ids)} race_ids")

    if args.limit > 0:
        race_ids = race_ids[: args.limit]

    all_rows = []
    fail_count = 0
    t_start = time.time()
    for i, rid in enumerate(race_ids, 1):
        print(f"  [{i}/{len(race_ids)}] {rid} ...", end='', flush=True)
        rows, status = parse_race(session, rid, date_hint=date)
        if rows:
            all_rows.extend(rows)
            print(f" OK ({len(rows)}h)")
        else:
            fail_count += 1
            print(f" FAIL ({status})")
        if i < len(race_ids):
            race_sleep()

    elapsed = time.time() - t_start
    print(f"\nelapsed: {elapsed/60:.1f} min, horses: {len(all_rows)}, failed races: {fail_count}")

    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"[OK] {out_path}")


if __name__ == '__main__':
    main()
