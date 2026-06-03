#!/usr/bin/env python3
"""大規模①: 89.9%カバレッジ leak-free v2 で V15/s2b/V24/V24b 完全再評価。
v1(85.8%)との比較で結論(s2b最良・V15真値~106%)の頑健性を確定。検証専用・本番不変。"""
from __future__ import annotations
import os, sys, gzip, pickle, time
if sys.platform == "win32": sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.abspath('tools'))
from v16_anaba_s2_eval import build_features, ODDS_REMOVE, PROXY_FAMILY, RAW_REPLACE, NEW
from v16_leakfree_roi_grid import load_payouts, make_oof, S_tan, S_trio4, S_umaren_t3box, S_fuku1
EXTRA = ['paci_goal_rank', 'paci_goal_diff', 'paci_dochu_rank']
V24_REMOVE = ['prev_race_last3f','prev_race_first3f','prev_race_pace_diff','prev_odds_log','pci','sire_shinba_top3r','has_training','odds_sharp_drop','is_nar','gaisha_rank','course_renovated','jrdb_prev_interference','stable_comment_score','has_wood_training','jrdb_ls_idx']
V24B_REMOVE = ['prev_race_last3f','prev_race_first3f','prev_race_pace_diff','prev_odds_log','pci','sire_shinba_top3r','has_training','odds_sharp_drop','is_nar','gaisha_rank']
DATA = 'data'


def roi_of(ev, scol, pay, fn):
    ret = stake = hit = n = 0
    for k, g in ev.groupby('_rk'):
        if k not in pay or len(g) < 5: continue
        o = [int(x) for x in g.sort_values(scol, ascending=False)['horse_num'].tolist()]
        r, pts = fn(o, pay[k]); n += 1; ret += r; stake += 100 * pts; hit += (r > 0)
    return (ret/stake if stake else 0, hit/n if n else 0, n)


def anaba(ev, scol, market='s_v15', topk=6):
    hit = h = nr = 0; sp = []
    for k, g in ev.groupby('_rk'):
        if len(g) < 4: continue
        nr += 1; s = g[scol].values; m = g[market].values; t = g['target'].values
        anti = [i for i in set(np.argsort(-s)[:topk]) if i not in set(np.argsort(-m)[:topk])]
        for i in anti: h += 1; hit += int(t[i])
        if len(s) >= 3:
            r, _ = spearmanr(s, m)
            if not np.isnan(r): sp.append(r)
    return (hit/h if h else float('nan'), float(np.mean(sp)), nr)


def main():
    t0 = time.time()
    obj = pickle.load(gzip.open(os.path.join(DATA, '_v15_optuna_df_cache_leakfree_v2.pkl.gz'), 'rb'))
    df = obj['df']; v15 = obj['features']
    if 'target' not in df.columns: df['target'] = (df['finish'] <= 3).astype(int)
    df = build_features(df)
    df['pop'] = pd.to_numeric(df['oz_base_pop_rank'], errors='coerce').fillna(df['num_horses_val'])
    df['_rk'] = [f"{int(a)}_{b}_{int(c)}_{int(e)}_{int(f)}" for a, b, c, e, f in
                 zip(df['date_num'], df['course'], df['kai'], df['nichi'], df['race_num'])]
    v16 = [f for f in v15 if f not in ODDS_REMOVE]
    models = {'V15': v15, 's2b': [f for f in v16 if f not in (PROXY_FAMILY+EXTRA+RAW_REPLACE)]+NEW,
              'V24': [f for f in v15 if f not in V24_REMOVE], 'V24b': [f for f in v15 if f not in V24B_REMOVE]}
    for f in set().union(*models.values()): df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    pay = load_payouts(); y = pd.to_numeric(df['year'], errors='coerce')
    print(f"leak-free v2 (89.9%cov)。 WF学習...", flush=True)
    res = {}
    for name, feats in models.items():
        feats = [f for f in feats if f in df.columns]
        s, auc = make_oof(df, feats); df[f's_{name.lower()}'] = s
        ev = df[y >= 23]
        tan = roi_of(ev, f's_{name.lower()}', pay, S_tan)
        t4 = roi_of(ev, f's_{name.lower()}', pay, S_trio4)
        um = roi_of(ev, f's_{name.lower()}', pay, S_umaren_t3box)
        res[name] = {'auc': auc, 'tan': tan, 't4': t4, 'um': um, 'nf': len(feats)}
        print(f"  {name:6s} done AUC={auc:.4f}", flush=True)
    # spearman/反市場 は s_v15 基準
    df['s_v15'] = df['s_v15']  # alias exists (V15 -> s_v15)
    for name in models:
        ev = df[y >= 23]
        hr, sp, nr = anaba(ev, f's_{name.lower()}', market='s_v15')
        res[name].update({'hit': hr, 'sp': sp, 'nr': nr})

    print("\n=== leak-free v2 (89.9%cov, 2023-25 WF) 横並び ===")
    print(f"{'model':6s}{'feat':>5s}{'AUC':>8s}{'単勝':>8s}{'三連複t4':>9s}{'馬連box':>9s}{'反市場':>8s}{'spvs V15':>9s}{'N':>7s}")
    for n, r in res.items():
        print(f"{n:6s}{r['nf']:5d}{r['auc']:8.4f}{r['tan'][0]*100:7.1f}%{r['t4'][0]*100:8.1f}%{r['um'][0]*100:8.1f}%{r['hit']*100:7.1f}%{r['sp']:9.4f}{r['t4'][2]:7d}")
    print("\n=== v1(85.8%) 参照: V15 単勝106.2/三連複155.7 | s2b 111.6/194.3 | V24 106.8/156.6 | V24b 106.1/161.1 ===")
    sb = res['s2b']
    beat = [n for n, r in res.items() if n not in ('s2b',) and (r['tan'][0] > sb['tan'][0] and r['t4'][0] > sb['t4'][0])]
    print(f"=== s2b を 単勝&三連複 両方で超える過去/V15モデル: {beat if beat else 'なし(s2b最良)'} ===")
    import json
    json.dump({n: {'auc': r['auc'], 'tan': r['tan'][0], 't4': r['t4'][0], 'um': r['um'][0], 'hit': r['hit'], 'sp': r['sp'], 'n': r['t4'][2]} for n, r in res.items()},
              open(os.path.join(DATA, 'v16_pastmodels_leakfree_v2.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
