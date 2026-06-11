#!/usr/bin/env python3
"""Fable sweep Phase 0 検証(読み取り+シミュレーションのみ・本番不変)。
per-race 系(race_auto_notify / predict_one_race(_v3) / paper系 / save_all_horse_scores)の
二重マージ修正 = jrdb_features.merge_jrdb_once 共通ガードの判別テスト。

6/6-7 の v15_feat_dump から merge#1 直後相当の df を復元し:
 1) 旧経路の再現: merge_jrdb_predict_features を復元dfに再適用 → 衝突(_x/_y)+KYI族デフォルト化が**再現する**こと(=障害の実証)
 2) 新経路: merge_jrdb_once → no-op(列不変)であること
 3) 新経路スコア = predict_core直(復元df)と完全一致(<1e-12)
 4) 参考: 劣化スコア vs 復元スコアの per-race top1/top3 変化数(6/7影響量の記録)
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
from jrdb_features import merge_jrdb_once, merge_jrdb_predict_features

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
    n = ok_old_repro = ok_noop = ok_score = 0
    top1_chg = top3_chg = 0
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
            # merge#1直後相当を復元(素列←_x、_x/_y除去)
            m1 = dump.copy()
            xcols = [c for c in m1.columns if c.endswith('_x') and c[:-2] in m1.columns]
            for c in xcols:
                m1[c[:-2]] = m1[c]
            m1 = m1.drop(columns=[c for c in m1.columns if c.endswith('_x') or c.endswith('_y')])
            rid = str(m1['race_id'].iloc[0]) if 'race_id' in m1.columns else '000000000000'
            # 1) 旧経路の再現(障害の実証): 再マージ → 衝突 or デフォルト化
            try:
                old = merge_jrdb_predict_features(m1.copy(), rid)
                collided = any(str(c).endswith('_x') for c in old.columns)
                degraded = default_rate(old) > default_rate(m1) + 0.3
                ok_old_repro += int(collided or degraded)
                dr_old.append(default_rate(old))
            except Exception:
                pass
            # 2) 新経路: merge_jrdb_once → no-op
            out = merge_jrdb_once(m1.copy(), rid)
            noop = (list(out.columns) == list(m1.columns)) and not any(str(c).endswith('_x') for c in out.columns)
            ok_noop += int(noop)
            dr_new.append(default_rate(out))
            # 3) スコア完全一致
            d_new = predict_race(out, md)
            d_ref = predict_race(m1.copy(), md)
            s_new = pd.to_numeric(d_new['スコア'], errors='coerce').values
            s_ref = pd.to_numeric(d_ref['スコア'], errors='coerce').values
            ok_score += int(bool(np.nanmax(np.abs(s_new - s_ref)) < 1e-12))
            # 4) 参考: 劣化(ダンプ素列) vs 復元 の top 変化 (fable_dpfix_discriminate と同方式)
            try:
                d_deg = predict_race(dump.copy(), md)
                o_deg = d_deg.sort_values('スコア', ascending=False)['馬番'].astype(int).tolist()
                o_ref = d_ref.sort_values('スコア', ascending=False)['馬番'].astype(int).tolist()
                top1_chg += int(o_deg[0] != o_ref[0])
                top3_chg += int(set(o_deg[:3]) != set(o_ref[:3]))
            except Exception:
                pass
    print(f"対象 {n}R (6/6-6/7)")
    print(f"1) 旧経路の劣化再現(障害実証): {ok_old_repro}/{n}  旧KYIデフォルト率={np.mean(dr_old)*100:.1f}%")
    print(f"2) merge_jrdb_once no-op:      {ok_noop}/{n}  新KYIデフォルト率={np.mean(dr_new)*100:.1f}%")
    print(f"3) 新経路スコア=predict_core直: {ok_score}/{n}")
    print(f"4) 参考(6/6-7影響量): top1変化={top1_chg}R / top3変化={top3_chg}R")
    # 劣化再現は KYI データ自体が無い R では起きない(docs §7.6 の 41/43 と同値)ため n-2 を許容
    ok = (ok_old_repro >= n - 2) and (ok_noop == n) and (ok_score == n)
    print("VERDICT:", "PASS" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
