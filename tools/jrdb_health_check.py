#!/usr/bin/env python
"""JRDB データ健全性チェック (保険スクリプト)

AM3:00 の DailyPremiumScrape が JRDB ファイルを取りこぼした場合に備え、
土日の AM7:30 に起動して主要ファイルの鮮度・サイズを検証し、不足があれば
自動で再取得を試みる。それでも駄目なら Discord で手動介入を求める。

Usage:
    python tools/jrdb_health_check.py                      # 今日の状態を確認し必要なら再取得
    python tools/jrdb_health_check.py --date 20260419      # 指定日
    python tools/jrdb_health_check.py --date 2026-04-19    # YYYY-MM-DD 形式も可
    python tools/jrdb_health_check.py --dry-run            # 検証のみ (再取得しない)

制約:
    - predict_core.py / daily_predict.py / jrdb_features.py を一切変更しない
    - 既存の scrape_jrdb.py / download_parse_jrdb_batch2.py / download_parse_jrdb_extra.py を
      そのまま呼び出すだけ
"""
import os
import sys
import io
import csv
import argparse
import subprocess
from datetime import datetime

if sys.platform == 'win32' and getattr(sys.stdout, 'encoding', '') != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(BASE_DIR, 'tools')
DATA_DIR = os.path.join(BASE_DIR, 'data')
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, TOOLS_DIR)

# 予測に必須の JRDB CSV 群。パスはいずれも data/ 配下。
# (type, csv_path, min_size_bytes, re-scrape cmd argv)
REQUIRED = [
    ('KYI', os.path.join(DATA_DIR, 'jrdb_kyi.csv'), 1_000_000,
     [sys.executable, os.path.join(TOOLS_DIR, 'scrape_jrdb.py'), '--type', 'KYI', '--force']),
    ('KAB', os.path.join(DATA_DIR, 'jrdb_kab.csv'), 100_000,
     [sys.executable, os.path.join(TOOLS_DIR, 'scrape_jrdb.py'), '--type', 'KAB', '--force']),
    ('KTA', os.path.join(DATA_DIR, 'jrdb_kta.csv'), 1_000_000,
     [sys.executable, os.path.join(TOOLS_DIR, 'download_parse_jrdb_batch2.py'), '--types', 'kta']),
    ('CHA', os.path.join(DATA_DIR, 'jrdb_cha.csv'), 1_000_000,
     [sys.executable, os.path.join(TOOLS_DIR, 'download_parse_jrdb_batch2.py'), '--types', 'cha']),
    ('KKA', os.path.join(DATA_DIR, 'jrdb_kka.csv'), 1_000_000,
     [sys.executable, os.path.join(TOOLS_DIR, 'download_parse_jrdb_extra.py'), '--types', 'kka']),
    ('JO',  os.path.join(DATA_DIR, 'jrdb_jo.csv'),  1_000_000,
     [sys.executable, os.path.join(TOOLS_DIR, 'download_parse_jrdb_extra.py'), '--types', 'jo']),
]


def normalize_date(s):
    s = (s or '').replace('-', '').replace('/', '')
    if not s:
        s = datetime.now().strftime('%Y%m%d')
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f'Invalid date: {s!r} (want YYYYMMDD)')
    return s


def check_file(name, path, min_size, date_str):
    """Return (ok: bool, reason: str)."""
    if not os.path.exists(path):
        return False, 'missing'
    size = os.path.getsize(path)
    if size < min_size:
        return False, f'too small ({size:,}B < {min_size:,}B)'
    try:
        with open(path, 'rb') as f:
            first = f.readline()
        if len(first) < 10:
            return False, 'empty header'
    except Exception as e:
        return False, f'read error: {e}'

    target = datetime.strptime(date_str, '%Y%m%d')
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
    except Exception as e:
        return False, f'mtime error: {e}'
    if mtime < target:
        return False, f'stale (mtime {mtime:%Y-%m-%d %H:%M})'

    # KAB には yyyymmdd 列があるので当日のレコードが含まれるか厳密検証
    if name == 'KAB':
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                hdr = next(reader, None)
                if not hdr:
                    return False, 'no header'
                try:
                    idx = hdr.index('yyyymmdd')
                except ValueError:
                    idx = None
                if idx is None:
                    return True, f'ok ({size:,}B mtime {mtime:%m/%d %H:%M})'
                for row in reader:
                    if len(row) > idx and row[idx] == date_str:
                        return True, f'ok (yyyymmdd={date_str}, {size:,}B)'
            return False, f'no row for yyyymmdd={date_str}'
        except Exception as e:
            return False, f'csv scan error: {e}'

    return True, f'ok ({size:,}B mtime {mtime:%m/%d %H:%M})'


def check_all(date_str):
    return [(name,) + check_file(name, path, min_size, date_str)
            for name, path, min_size, _ in REQUIRED]


def format_report(results, date_str, phase=''):
    lines = [f'JRDB Health {date_str}' + (f' [{phase}]' if phase else '')]
    for name, ok, reason in results:
        mark = 'OK ' if ok else 'NG '
        lines.append(f'  {mark} {name}: {reason}')
    ok = sum(1 for _, o, _ in results if o)
    lines.append(f'  -> {ok}/{len(results)} healthy')
    return '\n'.join(lines)


def run_rescrape(failed_names, date_str):
    seen = set()
    cmds = []
    for name, _, _, cmd in REQUIRED:
        if name not in failed_names:
            continue
        # scrape_jrdb.py は --date が使える。batch2/extra は未対応なので素通し。
        cmd_full = list(cmd)
        if any('scrape_jrdb.py' in c for c in cmd_full):
            cmd_full.extend(['--date', date_str])
        key = tuple(cmd_full)
        if key in seen:
            continue
        seen.add(key)
        cmds.append((name, cmd_full))

    for name, cmd in cmds:
        print(f'\n--- Re-scrape {name}: {" ".join(cmd)}')
        try:
            r = subprocess.run(cmd, cwd=BASE_DIR, timeout=900,
                               capture_output=True, text=True,
                               encoding='utf-8', errors='replace')
            if r.returncode == 0:
                tail = (r.stdout or '').strip().splitlines()[-3:]
                for line in tail:
                    print(f'   {line}')
                print(f'   [rc=0]')
            else:
                print(f'   [rc={r.returncode}]')
                if r.stderr:
                    print(f'   stderr: {r.stderr[-400:]}')
        except subprocess.TimeoutExpired:
            print(f'   TIMEOUT (900s)')
        except Exception as e:
            print(f'   ERROR {e}')


def notify(results, date_str, color, title_suffix=''):
    try:
        from notify import send_discord
    except Exception as e:
        print(f'[WARN] notify import failed: {e}')
        return
    ok = sum(1 for _, o, _ in results if o)
    total = len(results)
    title = f'JRDB Health {date_str} {ok}/{total}'
    if title_suffix:
        title += f' {title_suffix}'
    body = format_report(results, date_str)
    send_discord(title, body, color=color, channel='updates')


def main():
    p = argparse.ArgumentParser(description='JRDB data health check with optional re-scrape')
    p.add_argument('--date', default='', help='YYYYMMDD or YYYY-MM-DD (default: today)')
    p.add_argument('--dry-run', action='store_true', help='Check only, no re-scrape')
    args = p.parse_args()
    date_str = normalize_date(args.date)

    print('=' * 60)
    print(f'  JRDB Health Check: {date_str}' + (' [DRY-RUN]' if args.dry_run else ''))
    print('=' * 60)

    results = check_all(date_str)
    print()
    print(format_report(results, date_str, phase='initial'))
    failed = [n for n, ok, _ in results if not ok]

    if not failed:
        notify(results, date_str, color='green')
        print('\n[OK] All files healthy.')
        return 0

    if args.dry_run:
        notify(results, date_str, color='yellow', title_suffix='[DRY-RUN]')
        print(f'\n[DRY-RUN] Would re-scrape: {", ".join(failed)}')
        return 1

    notify(results, date_str, color='red', title_suffix='-> 再取得開始')
    run_rescrape(failed, date_str)

    results2 = check_all(date_str)
    print()
    print(format_report(results2, date_str, phase='post re-scrape'))
    failed2 = [n for n, ok, _ in results2 if not ok]

    if not failed2:
        notify(results2, date_str, color='green', title_suffix='回復')
        print('\n[RECOVERED]')
        return 0

    # Still failing — escalate
    notify(results2, date_str, color='red', title_suffix='⚠ 手動介入必要')
    print('\n[CRITICAL] Manual intervention required.')
    return 2


if __name__ == '__main__':
    sys.exit(main())
