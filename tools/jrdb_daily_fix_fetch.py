#!/usr/bin/env python3
"""JRDB 日次直接フェッチ (2026-06-12 Fable 統合タスク Phase B)。

背景 (停止根因):
 - KTA (4/5停止): download_parse_jrdb_batch2 が datazip 年度アーカイブの index 走査のみで
   member/data/Kta/ の日次 lzh を見ない → アーカイブ更新が止まると新規 0 件
 - KKA (5/3停止): download_parse_jrdb_extra が「extracted 非空なら何もしない」スキップ
 - SR/SRB: SED アーカイブ同梱 (SRB) + 年度 SRB アーカイブ依存で同型

対処: member/data/<Type>/<TYPE>yymmdd.lzh を日付直指定で取得し、既存の
EXTRACT_DIR に txt を置く (CSV 再構築は既存パーサ job がそのまま行う = 保存スキーマ不変)。
ローカル raw (lzh) があればネットに行かない。404 = 非開催/未公開としてスキップ。

usage:
  python tools/jrdb_daily_fix_fetch.py --types kta kka --start 20260401 --end 20260612
"""
from __future__ import annotations
import argparse, os, sys, time
from datetime import datetime, timedelta
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'tools'))
from scrape_jrdb import extract_lzh, JRDB_ID, JRDB_PASSWORD

RAW_DIR = os.path.join(BASE, 'data', 'jrdb', 'raw')
EXTRACT_DIR = os.path.join(BASE, 'data', 'jrdb', 'extracted')
JRDB_BASE = 'http://www.jrdb.com/member'

# type → (URLディレクトリ, ファイル接頭辞, raw/extract サブディレクトリ)
TYPES = {
    'kta': ('Kta', 'KTA', 'Kta'),
    'kab': ('Kab', 'KAB', 'Kab'),   # 2026-08-12 KAB日次復活 (当日朝の馬場差/含水系)
    'kka': ('Kka', 'KKA', 'Kka'),
    'srb': ('Sed', 'SRB', 'Srb'),   # SRB は SED アーカイブ同梱 (SED lzh 内に SRByymmdd.txt)
    'skb': ('Skb', 'SKB', 'Skb'),
    'cyb': ('Cyb', 'CYB', 'Cyb'),
    'tyb': ('Tyb', 'TYB', 'Tyb'),
}


def daterange(start: str, end: str):
    d = datetime.strptime(start, '%Y%m%d')
    e = datetime.strptime(end, '%Y%m%d')
    while d <= e:
        yield d.strftime('%y%m%d')
        d += timedelta(days=1)


def fetch_type(t: str, start: str, end: str, sleep: float = 0.4) -> tuple[int, int, int]:
    import requests
    urldir, prefix, sub = TYPES[t]
    raw = os.path.join(RAW_DIR, sub)
    ext = os.path.join(EXTRACT_DIR, sub)
    os.makedirs(raw, exist_ok=True)
    os.makedirs(ext, exist_ok=True)
    auth = (JRDB_ID, JRDB_PASSWORD)
    n_local = n_dl = n_404 = 0
    for ymd in daterange(start, end):
        txt = os.path.join(ext, f'{prefix}{ymd}.txt')
        if os.path.exists(txt):
            continue  # 既に展開済み
        lzh_path = os.path.join(raw, f'{prefix}{ymd}.lzh')
        data = None
        if os.path.exists(lzh_path) and os.path.getsize(lzh_path) > 100:
            data = open(lzh_path, 'rb').read()  # ローカル raw 優先 (再取得しない)
            n_local += 1
        else:
            url = f'{JRDB_BASE}/data/{urldir}/{prefix}{ymd}.lzh'
            try:
                r = requests.get(url, auth=auth, timeout=20)
            except Exception as e:
                print(f'  [ERR] {prefix}{ymd}: {e}')
                time.sleep(sleep)
                continue
            time.sleep(sleep)
            if r.status_code == 404:
                n_404 += 1
                continue
            if r.status_code != 200 or len(r.content) < 100:
                print(f'  [WARN] {prefix}{ymd}: HTTP {r.status_code} len={len(r.content)}')
                continue
            data = r.content
            open(lzh_path, 'wb').write(data)
            n_dl += 1
        try:
            for fname, content in extract_lzh(data).items():
                if fname.upper().startswith(prefix):
                    open(os.path.join(ext, os.path.basename(fname)), 'wb').write(content)
        except Exception as e:
            print(f'  [WARN] {prefix}{ymd} 展開失敗: {e}')
    print(f'  {t.upper()}: local={n_local} downloaded={n_dl} 404(非開催/未公開)={n_404}')
    return n_local, n_dl, n_404


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--types', nargs='+', required=True, choices=list(TYPES))
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', default=datetime.now().strftime('%Y%m%d'))
    a = ap.parse_args()
    for t in a.types:
        print(f'--- {t.upper()} 直接フェッチ {a.start}..{a.end} ---')
        fetch_type(t, a.start, a.end)


if __name__ == '__main__':
    main()
