#!/usr/bin/env python3
"""Fable sweep Phase 1: 特徴族デフォルト率マトリクス (読み取り専用)。
6/6-7 の v15_feat_dump を merge#1 直後相当に復元し、全特徴族
(KYI/PACI/ZE/OZ/TYB/netkeiba系) のデフォルト率・_x/_y衝突列を計測。
全予測経路は predict_core.build_features に収束する(Phase 0 でガード統一済)ため、
この値 = 修正後の全経路の特徴健全性ベースライン。
"""
from __future__ import annotations
import os, sys, glob, json, warnings
warnings.filterwarnings('ignore')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

# 特徴族の代表列とデフォルト値 (jrdb_features.JRDB_DEFAULTS / predict_core 既定に基づく)
FAMILIES = {
    'KYI':      {'jrdb_idm': 50.0, 'jrdb_running_style': 0, 'jrdb_dist_apt': 0,
                 'jrdb_hoof_code': 0, 'jrdb_ranch_rank': 0},
    'PACI':     {'paci_jockey_exp_wr': 14.5, 'paci_jockey_exp_3rd': 21.9, 'paci_ninki_idx': 159.0},
    'ZE':       {'jrdb_ze_idm_avg': 37.0, 'jrdb_ze_ten_avg': -15.0, 'jrdb_ze_agari_avg': -12.0},
    'OZ':       {'jrdb_oz_win_odds': 0, 'odds_change_rate': 0},
    'TYB':      {'jrdb_paddock_idx': 50.0, 'jrdb_odds_idx': 50.0},
    'SED前走':  {'jrdb_prev_idm': 50.0, 'jrdb_prev_pace_idx': 50.0},
    'netkeiba': {'speed_index': 0, 'wood_best_4f_filled': 0, 'stable_comment_score': 0},
}


def main():
    rows = []
    n = 0
    xy_races = 0
    acc = {fam: [0, 0] for fam in FAMILIES}  # fam -> [default_hits, total]
    col_presence = {}
    for date in ['20260606', '20260607']:
        for pq in sorted(glob.glob(f'data/v15_feat_dump/{date}/*.parquet')):
            try:
                dump = pd.read_parquet(pq)
            except Exception:
                continue
            if len(dump) < 6:
                continue
            n += 1
            # merge#1 直後相当に復元
            m1 = dump.copy()
            for c in [c for c in m1.columns if c.endswith('_x') and c[:-2] in m1.columns]:
                m1[c[:-2]] = m1[c]
            if any(str(c).endswith('_x') for c in dump.columns):
                xy_races += 1
            for fam, cols in FAMILIES.items():
                for c, dv in cols.items():
                    if c in m1.columns:
                        v = pd.to_numeric(m1[c], errors='coerce').fillna(dv)
                        acc[fam][0] += int((v == dv).sum())
                        acc[fam][1] += len(v)
                        col_presence[c] = col_presence.get(c, 0) + 1
                    else:
                        col_presence[c] = col_presence.get(c, 0)
    print(f"対象 {n}R (6/6-6/7、復元=修正後経路相当)。旧ダンプの_x衝突あり {xy_races}R")
    print(f"{'特徴族':10s} {'デフォルト率':>10s}  代表列の存在R数")
    res = {}
    for fam, (hit, tot) in acc.items():
        rate = hit / tot if tot else float('nan')
        pres = {c: col_presence.get(c, 0) for c in FAMILIES[fam]}
        res[fam] = {'default_rate': round(rate, 4) if tot else None, 'presence': pres}
        print(f"{fam:10s} {rate*100 if tot else float('nan'):9.1f}%  {pres}")
    json.dump({'n_races': n, 'xy_races_old_dump': xy_races, 'families': res},
              open('data/fable_sweep_phase1_matrix.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print('-> data/fable_sweep_phase1_matrix.json')


if __name__ == '__main__':
    main()
