#!/usr/bin/env python3
"""TYB/SKB/SRB 単独再構築 (2026-06-12 Fable統合タスク。parse_jrdb.py main は全種別を
一括再構築して稼働中CSV(paci等)を巻き込むため、種別単独のドライバを用意)。
EXTRACT_DIR の全 txt からフル再構築 = 保存スキーマ不変・追記事故なし。
SRB はローカル data/jrdb_raw/sed/*.lzh に同梱の SRB*.txt を先に展開してから再構築。

usage: python tools/fable_rebuild_type_20260612.py --types tyb skb srb
"""
from __future__ import annotations
import argparse, os, sys, glob
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'tools'))
EXTRACT = os.path.join(BASE, 'data', 'jrdb', 'extracted')
DATA = os.path.join(BASE, 'data')

TYB_KEY_COLS = ['race_id', 'umaban', 'idm', 'jockey_idx', 'info_idx', 'odds_idx', 'padock_idx',
                'sogo_idx', 'bagu_change', 'ashimoto', 'cancel_flag', 'tansho_odds', 'fukusho_odds',
                'horse_weight', 'weight_diff', 'weight_carry', 'odds_mark', 'padock_mark',
                'sogo_mark', 'batai_code', 'kehai_code', 'jockey_code', 'jockey_name',
                'baba_code', 'weather_code', 'start_time']


def rebuild(name, subdir, prefix, parse_line, min_len, key_cols, out_csv, dedup):
    files = sorted(glob.glob(os.path.join(EXTRACT, subdir, f'{prefix}*.txt')))
    rows = []; errs = 0
    for fp in files:
        try:
            with open(fp, 'rb') as f:
                for lb in f:
                    lb = lb.rstrip(b'\r\n')
                    if len(lb) < min_len:
                        continue
                    try:
                        rows.append(parse_line(lb))
                    except Exception:
                        errs += 1
        except Exception:
            errs += 1
    print(f'{name}: {len(files)} files -> {len(rows):,} rows ({errs} errors)')
    if not rows:
        print(f'  [SKIP] {name} 行ゼロ — 書き込みしない (全滅ガード)')
        return
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=dedup, keep='last')
    avail = [c for c in key_cols if c in df.columns] if key_cols else list(df.columns)
    out = os.path.join(DATA, out_csv)
    # 全滅ガード: 既存より50%以上減る再構築は拒否
    if os.path.exists(out):
        n_old = sum(1 for _ in open(out, encoding='utf-8-sig', errors='replace')) - 1
        if len(df) < n_old * 0.5:
            raise RuntimeError(f'{name} 全滅ガード: 再構築 {len(df)} < 既存 {n_old} の50%')
    df[avail].to_csv(out, index=False, encoding='utf-8-sig')
    print(f'  Saved: {out} ({len(df):,} rows, {len(avail)} cols)')


def extract_srb_from_local_sed():
    """ローカル data/jrdb_raw/sed/*.lzh から SRB*.txt を EXTRACT/Srb に展開 (ネット不要)。"""
    from scrape_jrdb import extract_lzh
    srb_dir = os.path.join(EXTRACT, 'Srb')
    os.makedirs(srb_dir, exist_ok=True)
    n = 0
    for lzh in sorted(glob.glob(os.path.join(BASE, 'data', 'jrdb_raw', 'sed', 'SED*.lzh'))):
        ymd = os.path.basename(lzh)[3:9]
        dest = os.path.join(srb_dir, f'SRB{ymd}.txt')
        if os.path.exists(dest):
            continue
        try:
            for fname, content in extract_lzh(open(lzh, 'rb').read()).items():
                if fname.upper().startswith('SRB'):
                    open(os.path.join(srb_dir, os.path.basename(fname)), 'wb').write(content)
                    n += 1
        except Exception as e:
            print(f'  [WARN] {lzh}: {e}')
    print(f'SRB: local SED lzh から {n} txt 展開')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--types', nargs='+', required=True, choices=['tyb', 'skb', 'srb'])
    a = ap.parse_args()
    if 'tyb' in a.types:
        from parse_jrdb import parse_tyb_line
        rebuild('TYB', 'Tyb', 'TYB', parse_tyb_line, 95, TYB_KEY_COLS, 'jrdb_tyb.csv',
                ['race_id', 'umaban'])
    if 'skb' in a.types:
        from parse_jrdb_missing import parse_skb_line
        rebuild('SKB', 'Skb', 'SKB', parse_skb_line, 100, None, 'jrdb_skb.csv',
                ['race_id', 'umaban'])
    if 'srb' in a.types:
        extract_srb_from_local_sed()
        from download_parse_jrdb_batch2 import parse_srb_line
        rebuild('SRB', 'Srb', 'SRB', parse_srb_line, 100, None, 'jrdb_srb.csv', ['race_id'])


if __name__ == '__main__':
    main()
