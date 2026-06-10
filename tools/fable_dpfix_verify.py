#!/usr/bin/env python3
"""daily_predict 二重マージ修正の検証(読み取り+シミュレーションのみ)。
6/6-7ダンプから merge#1 直後相当の df を復元(素列←_x、_x/_y除去)し、
修正後経路(merge_jrdb_once=ガードでスキップ)を通して:
 1) merge_jrdb_once が no-op であること(衝突列が生まれない)
 2) KYI族デフォルト率: 旧経路(劣化) vs 新経路(修正後)
 3) 新経路スコア = predict_core直(復元df)採点と一致
"""
from __future__ import annotations
import os, sys, glob, warnings
warnings.filterwarnings('ignore')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath('tools'))
sys.path.insert(0, os.path.abspath('.'))
from predict_core import load_models, predict_race
from daily_predict import merge_jrdb_once

KYI_CHECK = ['jrdb_idm', 'jrdb_running_style', 'jrdb_dist_apt', 'jrdb_hoof_code', 'jrdb_ranch_rank']
DEFAULTS = {'jrdb_idm': 50.0, 'jrdb_running_style': 0, 'jrdb_dist_apt': 0, 'jrdb_hoof_code': 0, 'jrdb_ranch_rank': 0}


def default_rate(df):
    tot = hit = 0
    for c, dv in DEFAULTS.items():
        if c in df.columns:
            v = pd.to_numeric(df[c], errors='coerce').fillna(dv)
            tot += len(v); hit += int((v == dv).sum())
    return hit / tot if tot else float('nan')


def main():
    md = load_models()
    n = ok_noop = ok_score = 0
    dr_old = []; dr_new = []
    for date in ['20260606', '20260607']:
        for pq in sorted(glob.glob(f'data/v15_feat_dump/{date}/*.parquet')):
            try:
                dump = pd.read_parquet(pq)
            except Exception:
                continue
            if 'スコア' not in dump.columns or len(dump) < 6:
                continue
            n += 1
            dr_old.append(default_rate(dump))  # 旧経路の素列(劣化)
            # merge#1直後相当を復元
            m1 = dump.copy()
            xcols = [c for c in m1.columns if c.endswith('_x') and c[:-2] in m1.columns]
            for c in xcols:
                m1[c[:-2]] = m1[c]
            m1 = m1.drop(columns=[c for c in m1.columns if c.endswith('_x') or c.endswith('_y')])
            # 修正後経路: merge_jrdb_once → no-op 確認
            rid = str(m1['race_id'].iloc[0]) if 'race_id' in m1.columns else '000000000000'
            out = merge_jrdb_once(m1.copy(), rid)
            noop = (list(out.columns) == list(m1.columns)) and not any(c.endswith('_x') for c in out.columns)
            ok_noop += noop
            dr_new.append(default_rate(out))
            # スコア: 修正後経路 == predict_core直(復元df)
            s_new = pd.to_numeric(predict_race(out, md)['スコア'], errors='coerce').values
            s_ref = pd.to_numeric(predict_race(m1.copy(), md)['スコア'], errors='coerce').values
            ok_score += bool(np.nanmax(np.abs(s_new - s_ref)) < 1e-12)
    print(f"対象 {n}R")
    print(f"1) merge_jrdb_once no-op(衝突なし): {ok_noop}/{n}")
    print(f"2) KYI族デフォルト率: 旧経路(劣化)={np.mean(dr_old)*100:.1f}% → 修正後={np.mean(dr_new)*100:.1f}%")
    print(f"3) 修正後スコア = predict_core直と一致: {ok_score}/{n}")


if __name__ == '__main__':
    main()
