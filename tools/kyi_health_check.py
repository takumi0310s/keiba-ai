#!/usr/bin/env python3
"""KYI族特徴の健全性チェック (2026-06-11 仕込み・検証側・本番不変)。
daily_predict の二重マージ修正(merge_jrdb_once)後の実地確認用:
最新(または指定日)の v15_feat_dump を読み、
 - 二重マージ衝突列(jrdb_*_x)の有無
 - KYI族デフォルト率(jrdb_idm==50 / running_style==0 / dist_apt==0)
を data/paper_s2b/kyi_check_{date}.json に記録して表示する。
paper_trade_s2b predict からも自動呼び出し(土曜朝に自動ログ)。
判定: _x列なし かつ デフォルト率<90% → OK(修正が効いている)。
"""
from __future__ import annotations
import os, sys, glob, json
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMP = os.path.join(BASE, 'data', 'v15_feat_dump')
OUT_DIR = os.path.join(BASE, 'data', 'paper_s2b')
DEFAULTS = {'jrdb_idm': 50.0, 'jrdb_running_style': 0, 'jrdb_dist_apt': 0}


def check_date(date: str) -> dict:
    pqs = sorted(glob.glob(os.path.join(DUMP, date, '*.parquet')))
    res = {'date': date, 'n_races': 0, 'x_col_races': 0, 'default_rate': None, 'verdict': 'NO_DATA'}
    tot = hit = 0
    for pq in pqs:
        try:
            df = pd.read_parquet(pq)
        except Exception:
            continue
        res['n_races'] += 1
        if any(str(c).endswith('_x') for c in df.columns):
            res['x_col_races'] += 1
        for c, dv in DEFAULTS.items():
            if c in df.columns:
                v = pd.to_numeric(df[c], errors='coerce').fillna(dv)
                tot += len(v); hit += int((v == dv).sum())
    if res['n_races'] == 0:
        return res
    res['default_rate'] = round(hit / tot, 4) if tot else None
    bad = res['x_col_races'] > 0 or (res['default_rate'] is not None and res['default_rate'] >= 0.9)
    res['verdict'] = 'NG_DEGRADED' if bad else 'OK'
    return res


GUARD_WHITELIST = {
    'jrdb_features.py',          # 定義元
    'predict_core.py',           # merge#1 正規(build_features内)
    'predict_core_v18.py',       # v18クローン内正規
    'fable_dpfix_verify.py',     # 意図的に旧経路を再現する検証ツール
    'fable_sweep_phase0_verify.py',
    'kyi_health_check.py',
    'feature_coverage_check.py', # build_features を経ない直マージ(診断)
    'predict_dryrun_compare.py', # 同上
}


def source_guard_scan(verbose: bool = True) -> list:
    """コードレベル全経路検査 (2026-06-11 Fable sweep Phase 0)。
    build_features(merge#1内包) と merge_jrdb_predict_features(df 直呼び の両方を含む
    .py = 二重マージ再発の疑い。新規経路が増えても検出できるよう毎回全走査する。"""
    import re
    bad = []
    for root, dirs, files in os.walk(BASE):
        dirs[:] = [d for d in dirs if d not in ('.git', '.claude', 'archive', '__pycache__', 'node_modules')]
        for fn in files:
            if not fn.endswith('.py') or fn in GUARD_WHITELIST:
                continue
            p = os.path.join(root, fn)
            try:
                src = open(p, encoding='utf-8', errors='replace').read()
            except Exception:
                continue
            # 代入形 `df = merge_jrdb_predict_features(df, ...)` のみ二重マージ疑い
            # (ガード関数内の `return merge_jrdb_predict_features(...)` は除外)
            if 'build_features(' in src and re.search(r'=\s*merge_jrdb_predict_features\(\s*(df|horses_df)', src):
                bad.append(os.path.relpath(p, BASE))
    if verbose:
        if bad:
            print(f"[kyi_health] ★二重マージ疑い {len(bad)} 件★: " + ', '.join(bad))
            print("  → merge_jrdb_once (jrdb_features) ガードに置換のこと (docs/SESSION_LEAK_AUDIT_S2B.md §7.6)")
        else:
            print("[kyi_health] コード全走査: 二重マージ疑いゼロ (build_features後の直マージなし)")
    return bad


def run(date: str | None = None, save: bool = True) -> dict:
    if not date:
        dirs = sorted(glob.glob(os.path.join(DUMP, '*')))
        date = os.path.basename(dirs[-1]) if dirs else ''
    res = check_date(date)
    dr = '-' if res['default_rate'] is None else '%.1f%%' % (res['default_rate'] * 100)
    line = (f"[kyi_health] {res['date']}: races={res['n_races']} 衝突_x列あり={res['x_col_races']}R "
            f"KYIデフォルト率={dr} → {res['verdict']}")
    print(line)
    if res['verdict'] == 'NG_DEGRADED':
        print("  ★二重マージ劣化の再発疑い: tools/daily_predict.py の merge_jrdb_once ガードと"
              " predict_core.build_features 内マージの状態を確認★ (docs/SESSION_LEAK_AUDIT_S2B.md §7.6)")
    if save and res['n_races'] > 0:
        os.makedirs(OUT_DIR, exist_ok=True)
        json.dump(res, open(os.path.join(OUT_DIR, f'kyi_check_{res["date"]}.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
    return res


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default='', help='YYYYMMDD (省略時=最新ダンプ)')
    a = ap.parse_args()
    run(a.date or None)
    source_guard_scan()
