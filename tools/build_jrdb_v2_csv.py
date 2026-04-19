"""jrdb_raw キャッシュから SED/TYB/CYB を再パース → 英語カラム v2 CSV を生成。

既存 data/jrdb_{sed,tyb,cyb}.csv は一切上書きしない。
data/jrdb_{sed,tyb,cyb}_v2.csv を新規生成する。

Usage:
    python tools/build_jrdb_v2_csv.py                 # SED/TYB/CYB 全部
    python tools/build_jrdb_v2_csv.py --types SED TYB # 指定のみ
    python tools/build_jrdb_v2_csv.py --years 2024 2025 2026  # 年指定 (SED用)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import warnings

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

from tools.scrape_jrdb import fetch_and_parse, DATA_DIR  # noqa: E402
from tools.jrdb_column_mapping import rename_jp_to_en  # noqa: E402

RAW_DIR = os.path.join(DATA_DIR, 'jrdb_raw')

V2_PATHS = {
    'SED': os.path.join(DATA_DIR, 'jrdb_sed_v2.csv'),
    'TYB': os.path.join(DATA_DIR, 'jrdb_tyb_v2.csv'),
    'CYB': os.path.join(DATA_DIR, 'jrdb_cyb_v2.csv'),
}


def _cache_dates(jrdb_type: str, years: list[str] | None = None) -> list[str]:
    """data/jrdb_raw/{type}/ から YYMMDD 日付を収集。years=['26','25',...] でフィルタ可"""
    subdir = os.path.join(RAW_DIR, jrdb_type.lower())
    if not os.path.isdir(subdir):
        return []
    dates: list[str] = []
    pat = re.compile(rf"{jrdb_type}(\d{{6}})\.lzh", re.IGNORECASE)
    for f in os.listdir(subdir):
        m = pat.match(f)
        if not m:
            continue
        d = m.group(1)
        if years is None or d[:2] in years:
            dates.append(d)
    return sorted(set(dates))


def build_one(jrdb_type: str, years: list[str] | None = None) -> pd.DataFrame:
    dates = _cache_dates(jrdb_type, years=years)
    print(f"=== {jrdb_type}: {len(dates)} cache dates "
          f"(years={years or 'all'}) ===")
    dfs: list[pd.DataFrame] = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for d in dates:
            df = fetch_and_parse(jrdb_type, d)
            if df is not None and len(df):
                dfs.append(df)
    if not dfs:
        print(f"{jrdb_type}: no data")
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True)
    # dedup (日本語カラムで)
    if 'jra_race_id' in result.columns and '馬番' in result.columns:
        before = len(result)
        result = result.drop_duplicates(subset=['jra_race_id', '馬番'], keep='last')
        print(f"{jrdb_type} dedup: {before} -> {len(result)}")
    # 英語カラムにリネーム
    result_en = rename_jp_to_en(result, jrdb_type)
    return result_en


def save(df: pd.DataFrame, jrdb_type: str) -> None:
    out = V2_PATHS[jrdb_type]
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"saved: {out}  shape={df.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--types', nargs='*', default=['SED', 'TYB', 'CYB'],
                    help='作成対象 (default: SED TYB CYB)')
    ap.add_argument('--years', nargs='*', default=None,
                    help='2桁年 (default: all)。例: --years 24 25 26')
    args = ap.parse_args()

    for t in args.types:
        t_u = t.upper()
        if t_u not in V2_PATHS:
            print(f"[WARN] skip unknown type: {t}")
            continue
        df = build_one(t_u, years=args.years)
        if len(df):
            save(df, t_u)


if __name__ == '__main__':
    main()
