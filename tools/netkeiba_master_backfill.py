#!/usr/bin/env python
"""netkeiba マスター 過去 backfill — Phase 18 B (controlled).

★★★ 重要 ★★★
Phase 13 元規約「scope 限定: 当日開催 R のみ、 過去 backfill しない」を Phase 18
で限定的に拡張する。 BAN risk が高いため デフォルトは何もせず、 全フラグを
明示的に指定した時のみ動作する。

Safety gates (全 ON が必須):
  1. --i-accept-tos-risk         明示同意
  2. --max-races N               1 セッション上限 (デフォルト 100、 max 5000)
  3. --max-daily-quota N          24h 内累計上限 (デフォルト 1000)
  4. data/netkeiba_master/.disabled が無い (kill switch OFF)
  5. .env に NETKEIBA_COOKIE 設定済

Rate limit: 既存 netkeiba_master_scraper の 3 sec interval を継承 (12 sec / R 想定)。

Resume / checkpoint:
  data/netkeiba_master/backfill_progress.csv に処理済 race_id を append。
  --resume で skip 再開可能。

Usage:
  # 100 R だけ試行 (1 R 当たり 12 sec → 約 20 分)
  python tools/netkeiba_master_backfill.py \\
      --i-accept-tos-risk --max-races 100 \\
      --start-date 20260103 --end-date 20260104

  # 1 日 1000 R 上限で 5 年 backfill (16,000 R / 1000 = 16 日)
  python tools/netkeiba_master_backfill.py \\
      --i-accept-tos-risk --max-races 5000 --max-daily-quota 1000 \\
      --start-date 20210101 --end-date 20260510 --resume

  # kill switch (即停止)
  python tools/netkeiba_master_scraper.py --disable

★ V15 投資保護 ★
- predict_core.py / daily_predict.py / app.py / V15 model 完全不変
- 取得 data は data/netkeiba_master/ 配下のみ (V15 prediction には影響しない)
- BAN 顕在化時は kill switch + Cookie refresh 待機

過去 24 年 80,000 R full backfill は本 script で実行しない方針。
直近 5 年 16,000 R を 16 日かけて段階取得が現実的上限。
"""
from __future__ import annotations
import os
import sys
import csv
import gzip
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Set

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / 'tools'))

from netkeiba_master_scraper import (  # noqa: E402
    fetch_master_features, is_disabled, _make_session, MASTER_DIR,
)

PROGRESS_PATH = Path(MASTER_DIR) / 'backfill_progress.csv'
QUOTA_PATH = Path(MASTER_DIR) / 'backfill_quota.json'

DEFAULT_MAX_RACES = 100
HARD_MAX_RACES = 5000
DEFAULT_DAILY_QUOTA = 1000


def _load_progress() -> Set[str]:
    if not PROGRESS_PATH.exists():
        return set()
    seen: Set[str] = set()
    with PROGRESS_PATH.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                seen.add(row[0])
    return seen


def _append_progress(race_id: str, status: str, note: str = ''):
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not PROGRESS_PATH.exists()
    with PROGRESS_PATH.open('a', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['race_id', 'fetched_at', 'status', 'note'])
        w.writerow([race_id, datetime.now().isoformat(timespec='seconds'), status, note])


def _load_quota() -> dict:
    if not QUOTA_PATH.exists():
        return {'date': '', 'count': 0}
    try:
        return json.loads(QUOTA_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {'date': '', 'count': 0}


def _save_quota(date: str, count: int):
    QUOTA_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUOTA_PATH.write_text(json.dumps({'date': date, 'count': count}), encoding='utf-8')


def _check_quota(daily_quota: int) -> tuple[bool, int]:
    today = datetime.now().strftime('%Y%m%d')
    q = _load_quota()
    if q.get('date') != today:
        return True, 0
    return q.get('count', 0) < daily_quota, q.get('count', 0)


def _bump_quota():
    today = datetime.now().strftime('%Y%m%d')
    q = _load_quota()
    if q.get('date') != today:
        q = {'date': today, 'count': 0}
    q['count'] = int(q.get('count', 0)) + 1
    _save_quota(q['date'], q['count'])


def discover_race_ids(start_date: str, end_date: str) -> List[str]:
    """既存 jra_races_full.csv 等から start_date〜end_date の race_id を列挙する。

    本 helper は外部データに依存しない placeholder。 実装は Phase 18 B 残作業:
    1) data/jra_races_full.csv から race_date in [start, end] の race_id 抽出
    2) または daily_predictions/*.csv から再構築
    呼び出し側 (CLI) で `--race-ids-file path` 経由で list 注入も可能。
    """
    csv_path = BASE_DIR / 'data' / 'jra_races_full.csv'
    if not csv_path.exists():
        return []
    import pandas as pd
    try:
        df = pd.read_csv(
            csv_path,
            usecols=['year', 'month', 'day', 'race_id'],
            dtype={'race_id': str, 'year': str, 'month': str, 'day': str},
        )
    except Exception:
        return []
    df = df.dropna(subset=['race_id', 'year', 'month', 'day'])
    yyyy = ('20' + df['year'].astype(str).str.zfill(2)).where(
        df['year'].astype(str).str.len() <= 2, df['year'].astype(str)
    )
    df['race_date'] = (
        yyyy + df['month'].astype(str).str.zfill(2) + df['day'].astype(str).str.zfill(2)
    )
    mask = (df['race_date'] >= start_date) & (df['race_date'] <= end_date)
    return sorted(df.loc[mask, 'race_id'].astype(str).unique().tolist())


def _save_race_bundle(race_id: str, payload: dict):
    out_dir = Path(MASTER_DIR) / 'backfill' / race_id[:4]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{race_id}.json.gz'
    with gzip.open(out_path, 'wt', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=None)


def run_backfill(
    race_ids: Iterable[str],
    max_races: int,
    daily_quota: int,
    umaban_per_race: int = 1,
    resume: bool = False,
) -> dict:
    if is_disabled():
        return {'error': 'kill_switch_active', 'processed': 0}

    session = _make_session()
    if session is None:
        return {'error': 'cookie_missing', 'processed': 0}

    seen = _load_progress() if resume else set()
    processed = 0
    skipped = 0
    failed = 0
    quota_ok, current_quota = _check_quota(daily_quota)
    if not quota_ok:
        return {
            'error': 'daily_quota_exhausted',
            'current_quota': current_quota,
            'limit': daily_quota,
            'processed': 0,
        }

    for race_id in race_ids:
        if processed >= max_races:
            break
        quota_ok, current_quota = _check_quota(daily_quota)
        if not quota_ok:
            print(f"[backfill] daily quota {daily_quota} reached ({current_quota}); stopping")
            break
        if race_id in seen:
            skipped += 1
            continue
        if is_disabled():
            print('[backfill] kill switch enabled mid-run; stopping')
            break
        try:
            bundle = fetch_master_features(race_id, umaban_per_race, session=session)
            payload = {
                'race_id': race_id,
                'umaban': umaban_per_race,
                'fetched_at': bundle.fetched_at,
                'fetch_status': bundle.fetch_status,
                'features': bundle.features,
            }
            _save_race_bundle(race_id, payload)
            ok = any(v == 'ok' for v in bundle.fetch_status.values())
            _append_progress(race_id, 'ok' if ok else 'partial')
            _bump_quota()
            processed += 1
            if processed % 10 == 0:
                print(f"[backfill] {processed}/{max_races} (quota {current_quota+1}/{daily_quota})")
        except Exception as e:
            failed += 1
            _append_progress(race_id, 'error', str(e)[:160])
            print(f"[backfill] {race_id} fail: {e}", file=sys.stderr)
        time.sleep(0.05)

    return {
        'processed': processed,
        'skipped': skipped,
        'failed': failed,
        'final_quota_today': _load_quota().get('count', 0),
    }


def _cli():
    p = argparse.ArgumentParser(description='netkeiba マスター 過去 backfill (Phase 18 B controlled)')
    p.add_argument('--i-accept-tos-risk', action='store_true', help='明示同意 (必須)')
    p.add_argument('--start-date', help='YYYYMMDD')
    p.add_argument('--end-date', help='YYYYMMDD')
    p.add_argument('--race-ids-file', help='race_id 一覧 (1 行 1 ID)')
    p.add_argument('--max-races', type=int, default=DEFAULT_MAX_RACES,
                   help=f'1 セッション上限 (デフォルト {DEFAULT_MAX_RACES}、 max {HARD_MAX_RACES})')
    p.add_argument('--max-daily-quota', type=int, default=DEFAULT_DAILY_QUOTA,
                   help=f'24h 累計上限 (デフォルト {DEFAULT_DAILY_QUOTA})')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='race_id 列挙のみ')
    args = p.parse_args()

    if not args.i_accept_tos_risk:
        print('refusing: --i-accept-tos-risk が必要 (Phase 18 B safety gate)', file=sys.stderr)
        sys.exit(2)

    if args.max_races > HARD_MAX_RACES:
        print(f'refusing: --max-races {args.max_races} > hard cap {HARD_MAX_RACES}', file=sys.stderr)
        sys.exit(2)

    if args.race_ids_file:
        race_ids = [
            line.strip()
            for line in Path(args.race_ids_file).read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.strip().startswith('#')
        ]
    elif args.start_date and args.end_date:
        race_ids = discover_race_ids(args.start_date, args.end_date)
    else:
        print('refusing: --start-date/--end-date または --race-ids-file が必要', file=sys.stderr)
        sys.exit(2)

    print(f'[backfill] candidate race_ids: {len(race_ids)}')
    if args.dry_run:
        for rid in race_ids[:20]:
            print(rid)
        return

    res = run_backfill(
        race_ids,
        max_races=args.max_races,
        daily_quota=args.max_daily_quota,
        resume=args.resume,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    _cli()
