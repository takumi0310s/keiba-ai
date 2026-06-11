#!/usr/bin/env python3
"""Fable sweep Phase 4 S1 定量化 (検証専用・本番不変)。
S1 = odds_change_rate / pop_rank_change / odds_sharp_drop が学習時に確定オッズ・確定人気を
使用 (train_v134_odds_change.py:165-199)。leak-free v2 の V15 真値 (単勝105.0%等) は
この3特徴込み → 3特徴を除外した V15_noS1 を同一 WF パイプラインで再学習し寄与を定量化。
(s2b は ODDS_REMOVE で当該3特徴を元々除外済 = s2b 真値は非汚染)
"""
from __future__ import annotations
import os, sys, gzip, pickle, time, json
if sys.platform == "win32": sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath('tools'))
from v16_anaba_s2_eval import build_features
from v16_leakfree_roi_grid import load_payouts, make_oof, S_tan, S_trio4, S_umaren_t3box

S1 = ['odds_change_rate', 'pop_rank_change', 'odds_sharp_drop']
DATA = 'data'


def roi_of(ev, scol, pay, fn):
    ret = stake = hit = n = 0
    for k, g in ev.groupby('_rk'):
        if k not in pay or len(g) < 5: continue
        o = [int(x) for x in g.sort_values(scol, ascending=False)['horse_num'].tolist()]
        r, pts = fn(o, pay[k]); n += 1; ret += r; stake += 100 * pts; hit += (r > 0)
    return (ret/stake if stake else 0, hit/n if n else 0, n)


def main():
    t0 = time.time()
    obj = pickle.load(gzip.open(os.path.join(DATA, '_v15_optuna_df_cache_leakfree_v2.pkl.gz'), 'rb'))
    df = obj['df']; v15 = obj['features']
    if 'target' not in df.columns: df['target'] = (df['finish'] <= 3).astype(int)
    df = build_features(df)
    df['_rk'] = [f"{int(a)}_{b}_{int(c)}_{int(e)}_{int(f)}" for a, b, c, e, f in
                 zip(df['date_num'], df['course'], df['kai'], df['nichi'], df['race_num'])]
    models = {'V15': v15, 'V15_noS1': [f for f in v15 if f not in S1]}
    for f in set().union(*models.values()):
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    pay = load_payouts(); y = pd.to_numeric(df['year'], errors='coerce')
    print('WF学習 (V15 / V15_noS1)...', flush=True)
    res = {}
    for name, feats in models.items():
        feats = [f for f in feats if f in df.columns]
        s, auc = make_oof(df, feats); df[f's_{name.lower()}'] = s
        ev = df[y >= 23]
        res[name] = {'nf': len(feats), 'auc': auc,
                     'tan': roi_of(ev, f's_{name.lower()}', pay, S_tan),
                     't4': roi_of(ev, f's_{name.lower()}', pay, S_trio4),
                     'um': roi_of(ev, f's_{name.lower()}', pay, S_umaren_t3box)}
        print(f"  {name}: AUC={auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
    print("\n=== S1 (確定オッズ派生3特徴) の寄与 (leak-free v2, 2023-25 WF) ===")
    print(f"{'model':10s}{'feat':>5s}{'AUC':>8s}{'単勝':>8s}{'三連複t4':>9s}{'馬連box':>9s}{'N':>7s}")
    for n, r in res.items():
        print(f"{n:10s}{r['nf']:5d}{r['auc']:8.4f}{r['tan'][0]*100:7.1f}%{r['t4'][0]*100:8.1f}%{r['um'][0]*100:8.1f}%{r['tan'][2]:7d}")
    d = res['V15']; e = res['V15_noS1']
    print(f"\nS1寄与: AUC {d['auc']-e['auc']:+.4f} / 単勝 {(d['tan'][0]-e['tan'][0])*100:+.1f}pt / "
          f"三連複t4 {(d['t4'][0]-e['t4'][0])*100:+.1f}pt / 馬連 {(d['um'][0]-e['um'][0])*100:+.1f}pt")
    json.dump({k: {'nf': v['nf'], 'auc': v['auc'],
                   'tan': v['tan'], 't4': v['t4'], 'um': v['um']} for k, v in res.items()},
              open('data/fable_v15_noS1_eval.json', 'w', encoding='utf-8'), indent=1)
    print('-> data/fable_v15_noS1_eval.json')


if __name__ == '__main__':
    main()
