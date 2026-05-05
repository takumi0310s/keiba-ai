"""NAR 確定オッズ refresh scraper. 19:00 自動発火 想定 (発走 30分前 帯).

shutuba (16:30 取得 = 予想オッズ) → 19:00 で 確定オッズ に更新する。
発走時刻が 19:00 以降の race のみ refresh する (発走済み race の odds は不要)。

URL:
  shutuba: https://nar.netkeiba.com/race/shutuba.html?race_id=...
  (table.RaceTable01.ShutubaTable の cell[9] = odds, cell[10] = pop_rank)

入力:
  data/nar_today_shutuba_YYYYMMDD.csv (16:30 で 生成済み 前提)

出力:
  data/nar_live_odds_YYYYMMDD.csv (新規)
    columns: race_id, horse_num, horse_name, odds, pop_rank, fetched_at
  + 元 shutuba CSV の odds/pop_rank 列を in-place 更新 (--update-shutuba option)

note:
  shutuba の parse_shutuba をそのまま流用して 1 race ずつ refetch する。
  発走 5分以内 の race は skip (すでに 締め切り)。
  shutuba CSV 不在時は all-races mode で動作 (race_list から race_ids を取り直す)。
"""
from __future__ import annotations

import os, sys, argparse, csv, re, time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from datetime import datetime, timedelta

BASE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE)
sys.path.insert(0, os.path.join(BASE, 'tools'))

from scrape_nar_all import (
    NAR_TRACKS, create_session, race_sleep, nav_sleep, get_race_ids,
)
from scrape_nar_today import parse_shutuba

OUT_HEADER = ['race_id', 'horse_num', 'horse_name', 'odds', 'pop_rank', 'fetched_at']


def parse_hhmm(s):
    """'17:50' → datetime.time, 不明は None."""
    if not s:
        return None
    m = re.match(r'^(\d{1,2}):(\d{2})$', s.strip())
    if not m:
        return None
    try:
        return datetime.strptime(s.strip(), '%H:%M').time()
    except ValueError:
        return None


def load_shutuba_targets(shutuba_path, now_dt, skip_window_min=5):
    """shutuba CSV から refresh 対象 race_ids を抽出.

    返却: list of race_id (発走 skip_window_min 分前まで未締切 のもの)
    + race_time map (race_id -> race_time str)
    """
    if not os.path.exists(shutuba_path):
        return None, {}, 'no_shutuba_csv'

    targets = []
    rt_map = {}
    seen = set()

    with open(shutuba_path, 'r', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        for row in rd:
            rid = row.get('race_id', '')
            if not rid or rid in seen:
                continue
            seen.add(rid)

            rt = row.get('race_time', '').strip()
            rt_map[rid] = rt
            t = parse_hhmm(rt)
            # 発走時刻不明 → 念のため refresh 対象に含める
            if t is None:
                targets.append(rid)
                continue
            # 発走済み (or skip_window_min 以内) → skip
            race_dt = datetime.combine(now_dt.date(), t)
            if race_dt + timedelta(minutes=skip_window_min) < now_dt:
                continue
            targets.append(rid)

    return targets, rt_map, 'ok'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=None, help='YYYYMMDD (default 今日)')
    parser.add_argument('--tracks', default=None,
                        help='場 名 カンマ区切り (例: "船橋,大井")')
    parser.add_argument('--output', default=None,
                        help='出力 CSV (default data/nar_live_odds_DATE.csv)')
    parser.add_argument('--shutuba', default=None,
                        help='入力 shutuba CSV (default data/nar_today_shutuba_DATE.csv)')
    parser.add_argument('--update-shutuba', action='store_true',
                        help='元 shutuba CSV の odds/pop_rank 列を in-place 上書き')
    parser.add_argument('--skip-min', type=int, default=5,
                        help='発走 N 分前以内の race は skip (default 5)')
    parser.add_argument('--limit', type=int, default=0, help='テスト用 最大 race 件数')
    parser.add_argument('--ignore-time', action='store_true',
                        help='発走時刻 filter を無効化 (全 race を refresh)')
    args = parser.parse_args()

    date = args.date or datetime.now().strftime('%Y%m%d')
    shutuba_path = args.shutuba or f'data/nar_today_shutuba_{date}.csv'
    out_path = args.output or f'data/nar_live_odds_{date}.csv'

    track_filter = None
    if args.tracks:
        track_filter = set(args.tracks.split(','))

    print(f"=== scrape_nar_live_odds {date} ===")
    print(f"  shutuba: {shutuba_path}")
    print(f"  output:  {out_path}")
    print(f"  tracks:  {track_filter or 'ALL'}")
    print(f"  skip if race start within {args.skip_min} min")

    now_dt = datetime.now()
    targets, rt_map, status = load_shutuba_targets(
        shutuba_path, now_dt, skip_window_min=args.skip_min
    )

    session = create_session()
    nav_sleep()

    # shutuba CSV がない場合は race_list から fallback
    if status == 'no_shutuba_csv':
        print(f"  [WARN] shutuba CSV 不在 → race_list fallback")
        targets = get_race_ids(session, date)
        rt_map = {rid: '' for rid in targets}

    if args.ignore_time:
        # ignore-time 指定なら shutuba 全 race を再対象
        if os.path.exists(shutuba_path):
            with open(shutuba_path, 'r', encoding='utf-8-sig') as f:
                rd = csv.DictReader(f)
                all_ids = []
                seen = set()
                for row in rd:
                    rid = row.get('race_id', '')
                    if rid and rid not in seen:
                        seen.add(rid)
                        all_ids.append(rid)
                        rt_map.setdefault(rid, row.get('race_time', ''))
            targets = all_ids

    # track filter
    if track_filter:
        name_to_code = {v: str(k) for k, v in NAR_TRACKS.items()}
        wanted_codes = {name_to_code[t] for t in track_filter if t in name_to_code}
        targets = [rid for rid in targets if rid[4:6] in wanted_codes]

    if args.limit > 0:
        targets = targets[: args.limit]

    print(f"  refresh targets: {len(targets)} races")
    if not targets:
        print("  → no targets、空 CSV 生成 で skip")
        with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
            csv.DictWriter(f, fieldnames=OUT_HEADER).writeheader()
        return

    # 各 race の odds を refetch
    odds_rows = []
    fetched_at = now_dt.strftime('%Y-%m-%d %H:%M:%S')
    fail_count = 0
    t_start = time.time()

    for i, rid in enumerate(targets, 1):
        rt = rt_map.get(rid, '')
        print(f"  [{i}/{len(targets)}] {rid} (race_time={rt or '?'}) ...",
              end='', flush=True)
        rows, st = parse_shutuba(session, rid, date_hint=date)
        if rows:
            for h in rows:
                odds_rows.append({
                    'race_id': rid,
                    'horse_num': h.get('horse_num', ''),
                    'horse_name': h.get('horse_name', ''),
                    'odds': h.get('odds', ''),
                    'pop_rank': h.get('pop_rank', ''),
                    'fetched_at': fetched_at,
                })
            print(f" OK ({len(rows)}h)")
        else:
            fail_count += 1
            print(f" FAIL ({st})")
        if i < len(targets):
            race_sleep()

    elapsed = time.time() - t_start
    print(f"\nelapsed: {elapsed/60:.1f} min, "
          f"horses: {len(odds_rows)}, failed races: {fail_count}")

    # write live_odds CSV
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=OUT_HEADER)
        w.writeheader()
        for r in odds_rows:
            w.writerow(r)
    print(f"[OK] {out_path}")

    # update shutuba in-place if requested
    if args.update_shutuba and os.path.exists(shutuba_path) and odds_rows:
        # build lookup (race_id, horse_num) → (odds, pop_rank)
        lookup = {(r['race_id'], r['horse_num']): (r['odds'], r['pop_rank'])
                  for r in odds_rows}

        with open(shutuba_path, 'r', encoding='utf-8-sig') as f:
            rd = csv.DictReader(f)
            fields = rd.fieldnames or []
            existing = list(rd)

        updated = 0
        for row in existing:
            key = (row.get('race_id', ''), row.get('horse_num', ''))
            if key in lookup:
                new_odds, new_pop = lookup[key]
                if new_odds:
                    row['odds'] = new_odds
                if new_pop:
                    row['pop_rank'] = new_pop
                updated += 1

        with open(shutuba_path, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in existing:
                w.writerow(r)
        print(f"[OK] updated {updated} rows in {shutuba_path}")


if __name__ == '__main__':
    main()
