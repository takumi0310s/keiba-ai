# -*- coding: utf-8 -*-
"""item3: JRDB CSV 種別単独フル再構築ドライバ (kyi/sed/cyb/kta/kka)。

fable_rebuild_type_20260612.py と同型の安全パターン:
  data/jrdb/extracted/<Type>/ 全txt → parse → dedup → ★現行CSVヘッダに列整合★ → 原子的置換。
tyb/skb/srb は既存 fable_rebuild を使用 (本ドライバ対象外)。

安全装置:
  - 現行CSVヘッダと新DFの列を照合。現行列が新DFに無ければ ABORT (スキーマ保護)
  - 新行数 < 現行行数 × floor 比 なら ABORT (履歴縮小保護。kyi は2020年フロアで再構築)
  - .csv.new に書いてから os.replace (書きかけ露出防止)

usage: python research/v15r/rebuild_driver.py --types kyi sed cyb kta kka
"""
from __future__ import annotations
import argparse, glob, os, sys, time

BASE = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "tools"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

EXTRACT = os.path.join(BASE, "data", "jrdb", "extracted")
DATA = os.path.join(BASE, "data")

# batch2/extra は module-level で sys.stdout を新 TextIOWrapper に差し替える。
# 旧ラッパが GC されると underlying buffer ごと close され以後の print が全滅する
# (fable_rebuild SRB クラッシュと同根)。旧 stdout への強参照を保持して防ぐ。
_stdout_keep = [sys.stdout]
from parse_jrdb import parse_sed_line, parse_kyi_line, _field, _build_race_id, _safe_int  # noqa: E402
_stdout_keep.append(sys.stdout)
from download_parse_jrdb_batch2 import parse_kta_line  # noqa: E402
_stdout_keep.append(sys.stdout)
from download_parse_jrdb_extra import parse_kka_line  # noqa: E402
_stdout_keep.append(sys.stdout)

# parse_jrdb.py main の CYB インライン仕様を忠実複製 (830-890行)
CYB_COLUMNS = [
    ('basho_code', 1, 2), ('year', 3, 2), ('kai', 5, 1), ('nichi', 6, 1),
    ('race_num', 7, 2), ('umaban', 9, 2), ('train_type', 11, 1),
    ('train_course_type', 12, 1), ('train_baba', 13, 1), ('train_mark', 14, 1),
    ('train_amount', 15, 1), ('train_change', 16, 1), ('train_comment', 17, 40),
    ('comment_year', 57, 3), ('comment_date', 60, 3), ('train_eval', 63, 1),
    ('train_course', 64, 2),
]


def parse_cyb_line(line_bytes):
    row = {}
    for name, s, l in CYB_COLUMNS:
        row[name] = _field(line_bytes, s, l)
    row['race_id'] = _build_race_id(row['basho_code'], row['year'], row['kai'],
                                    row['nichi'], row['race_num'])
    row['umaban'] = _safe_int(row['umaban'])
    row['train_comment'] = row['train_comment'].strip()
    for c in ['basho_code', 'year', 'kai', 'nichi', 'race_num']:
        del row[c]
    return row


def parse_line_files(pattern, parse_func, min_len, label):
    rows, errors = [], 0
    files = sorted(glob.glob(pattern))
    print(f"  {label}: {len(files)} files")
    for fp in files:
        try:
            with open(fp, 'rb') as f:
                for lb in f:
                    lb = lb.rstrip(b'\r\n')
                    if len(lb) < min_len:
                        continue
                    try:
                        rows.append(parse_func(lb))
                    except Exception:
                        errors += 1
        except Exception:
            errors += 1
    print(f"  parsed {len(rows):,} rows ({errors} errors)")
    return pd.DataFrame(rows) if rows else None


def rebuild_kyi():
    """KYI: 固定長フルファイルパース (rebuild_jrdb_kyi.py 複製、ソース=extracted txt、2020年フロア)。"""
    from scrape_jrdb import (parse_fixed_length, KYI_FIELDS,
                             jrdb_to_jra_race_id, jrdb_to_netkeiba_race_id)
    files = sorted(glob.glob(os.path.join(EXTRACT, 'Kyi', 'KYI*.txt')))
    files = [f for f in files if os.path.basename(f)[3:5] >= '20']  # 2020+ (現行CSVと同フロア)
    print(f"  KYI: {len(files)} files (2020+)")
    dfs = []
    for i, fp in enumerate(files):
        try:
            with open(fp, 'rb') as f:
                df = parse_fixed_length(f.read(), KYI_FIELDS)
            df['jra_race_id'] = df.apply(jrdb_to_jra_race_id, axis=1)
            df['nk_race_id'] = df.apply(jrdb_to_netkeiba_race_id, axis=1)
            for c in ['馬名', '放牧先', '入厩年月日']:
                if c in df.columns:
                    df[c] = df[c].str.strip()
            dfs.append(df)
        except Exception as e:
            print(f"    ERROR {os.path.basename(fp)}: {e}")
        if (i + 1) % 200 == 0:
            print(f"    [{i+1}/{len(files)}]")
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['jra_race_id', '馬番'], keep='last')
    return df


SPECS = {
    'sed': dict(pattern=os.path.join(EXTRACT, 'Sed', 'SED*.txt'), func=parse_sed_line,
                min_len=370, dedup=['race_id', 'umaban'], out='jrdb_sed.csv', floor=1.0),
    'cyb': dict(pattern=os.path.join(EXTRACT, 'Cyb', 'CYB*.txt'), func=parse_cyb_line,
                min_len=60, dedup=['race_id', 'umaban'], out='jrdb_cyb.csv', floor=1.0),
    'kta': dict(pattern=os.path.join(EXTRACT, 'Kta', 'KTA*.txt'), func=parse_kta_line,
                min_len=340, dedup=['race_id', 'blood_num'], out='jrdb_kta.csv', floor=1.0),
    'kka': dict(pattern=os.path.join(EXTRACT, 'Kka', 'KKA*.txt'), func=parse_kka_line,
                min_len=280, dedup=['race_id', 'umaban'], out='jrdb_kka.csv', floor=1.0),
    # jrdb_paci.csv の源 = Paciバンドル内 KYI (前日版)。extracted/Kyi の同日KYIと
    # byte-identical を確認済 (2026-08-10) → Kyi ディレクトリからフル再構築。
    'paci': dict(pattern=os.path.join(EXTRACT, 'Kyi', 'KYI*.txt'), func=parse_kyi_line,
                 min_len=400, dedup=['race_id', 'umaban'], out='jrdb_paci.csv', floor=1.0),
}


def align_and_write(df, out_csv, floor_ratio):
    out_path = os.path.join(DATA, out_csv)
    cur_header = pd.read_csv(out_path, nrows=0, encoding='utf-8-sig').columns.tolist()
    missing = [c for c in cur_header if c not in df.columns]
    if missing:
        print(f"  ABORT: 新DFに現行列が欠落 {missing[:8]} → スキーマ保護のため書き込み中止")
        return False
    cur_rows = sum(1 for _ in open(out_path, encoding='utf-8-sig', errors='replace')) - 1
    if len(df) < cur_rows * floor_ratio:
        print(f"  ABORT: 新 {len(df):,} 行 < 現行 {cur_rows:,} × {floor_ratio} → 履歴縮小保護")
        return False
    df = df[cur_header]
    tmp = out_path + '.new'
    df.to_csv(tmp, index=False, encoding='utf-8-sig')
    os.replace(tmp, out_path)
    print(f"  Saved: {out_csv} {cur_rows:,} → {len(df):,} rows (+{len(df)-cur_rows:,})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--types', nargs='+', required=True,
                    choices=['kyi', 'sed', 'cyb', 'kta', 'kka', 'paci'])
    args = ap.parse_args()
    t0 = time.time()
    ok = True
    for t in args.types:
        print(f"\n=== {t.upper()} ===")
        if t == 'kyi':
            df = rebuild_kyi()
        else:
            s = SPECS[t]
            df = parse_line_files(s['pattern'], s['func'], s['min_len'], t.upper())
            if df is not None:
                df = df.drop_duplicates(subset=s['dedup'], keep='last')
        if df is None or len(df) == 0:
            print(f"  ABORT: {t} パース結果 0 行"); ok = False; continue
        out = 'jrdb_kyi.csv' if t == 'kyi' else SPECS[t]['out']
        if not align_and_write(df, out, 1.0):
            ok = False
    print(f"\nDone in {time.time()-t0:.0f}s  ({'ALL OK' if ok else 'SOME ABORTED'})")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
