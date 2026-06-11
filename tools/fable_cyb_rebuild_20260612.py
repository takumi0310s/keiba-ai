#!/usr/bin/env python3
"""jrdb_cyb.csv 再構築 (2026-06-12 Fable統合タスク)。
旧状態: scrape_jrdb の日次 parse_cyb (別スキーマ) と parse_jrdb main の bulk スキーマが
混在し 4/19 以降 列ズレのゴミ 350 行のみ。 → parse_jrdb main 内の CYB ロジック (bulk 正)
を単独実行で全量再構築 (EXTRACT/Paci + EXTRACT/Cyb の両方の CYB*.txt)。consumer は現状ゼロ。
"""
from __future__ import annotations
import os, sys, glob
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'tools'))
from parse_jrdb import _field, _build_race_id, _safe_int

EXTRACT = os.path.join(BASE, 'data', 'jrdb', 'extracted')
OUT = os.path.join(BASE, 'data', 'jrdb_cyb.csv')

CYB_COLUMNS = [
    ('basho_code', 1, 2), ('year', 3, 2), ('kai', 5, 1), ('nichi', 6, 1),
    ('race_num', 7, 2), ('umaban', 9, 2), ('train_type', 11, 1),
    ('train_course_type', 12, 1), ('train_baba', 13, 1), ('train_mark', 14, 1),
    ('train_amount', 15, 1), ('train_change', 16, 1), ('train_comment', 17, 40),
    ('comment_year', 57, 3), ('comment_date', 60, 3), ('train_eval', 63, 1),
    ('train_course', 64, 2),
]


def main():
    files = sorted(set(glob.glob(os.path.join(EXTRACT, 'Paci', 'CYB*.txt'))
                       + glob.glob(os.path.join(EXTRACT, 'Cyb', 'CYB*.txt'))))
    rows, errs = [], 0
    for fp in files:
        try:
            with open(fp, 'rb') as f:
                for lb in f:
                    lb = lb.rstrip(b'\r\n')
                    if len(lb) < 60:
                        continue
                    try:
                        row = {n: _field(lb, s, l) for n, s, l in CYB_COLUMNS}
                        row['race_id'] = _build_race_id(row['basho_code'], row['year'],
                                                        row['kai'], row['nichi'], row['race_num'])
                        row['umaban'] = _safe_int(row['umaban'])
                        row['train_comment'] = row['train_comment'].strip()
                        for c in ['basho_code', 'year', 'kai', 'nichi', 'race_num']:
                            del row[c]
                        rows.append(row)
                    except Exception:
                        errs += 1
        except Exception:
            errs += 1
    print(f'CYB: {len(files)} files -> {len(rows):,} rows ({errs} errors)')
    if not rows:
        print('行ゼロ — 書き込みしない')
        return
    df = pd.DataFrame(rows).drop_duplicates(subset=['race_id', 'umaban'], keep='last')
    df.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f'Saved: {OUT} ({len(df):,} rows)')


if __name__ == '__main__':
    main()
