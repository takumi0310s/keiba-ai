#!/usr/bin/env python3
"""Fable統合タスク Phase E: 全馬評価の精度測定 (読み取り専用・leak-free v2・2023-25 WF)。

指標 (モデル毎):
 - AUC (target=複勝圏)
 - top1 の 1着率 / 馬券内率
 - 馬券内3頭の top4/6/8 収納率
 - レース内 Spearman(score, finish) 平均
市場ベンチマーク: 前日人気順位 (oz_base_pop_rank) を予測とみなした同指標 → 各モデルの市場超え幅。
頭数別 (≤8 / 9-12 / 13-15 / 16+) の top1馬券内率も併記。
"""
from __future__ import annotations
import os, sys, gzip, pickle, json, time
if sys.platform == "win32": sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.abspath('tools'))
from v16_anaba_s2_eval import build_features, ODDS_REMOVE, PROXY_FAMILY, RAW_REPLACE, NEW
from v16_leakfree_roi_grid import make_oof

EXTRA = ['paci_goal_rank', 'paci_goal_diff', 'paci_dochu_rank']


def fast_auc(y, x):
    m = ~(np.isnan(x) | np.isnan(y))
    y, x = y[m], x[m]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = pd.Series(x).rank().values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def race_metrics(ev, scol):
    """レース単位指標。score 降順で評価。"""
    top1_win = top1_top3 = 0
    cont = {4: 0, 6: 0, 8: 0}
    sp = []
    n = 0
    by_size = {}
    for k, g in ev.groupby('_rk'):
        if len(g) < 5:
            continue
        n += 1
        g = g.sort_values(scol, ascending=False)
        fin = pd.to_numeric(g['finish'], errors='coerce').values
        top1_win += int(fin[0] == 1)
        t3 = int(fin[0] <= 3)
        top1_top3 += t3
        podium_pos = np.where(fin <= 3)[0]  # score順位の中で馬券内馬の位置
        for kk in (4, 6, 8):
            cont[kk] += int(len(podium_pos) >= 3 and (podium_pos < kk).sum() >= 3)
        s = g[scol].values
        m = ~np.isnan(fin)
        if m.sum() >= 3:
            r, _ = spearmanr(-s[m], fin[m])
            if not np.isnan(r):
                sp.append(r)
        nh = len(g)
        band = '≤8' if nh <= 8 else ('9-12' if nh <= 12 else ('13-15' if nh <= 15 else '16+'))
        b = by_size.setdefault(band, [0, 0])
        b[0] += t3; b[1] += 1
    return {'n': n, 'top1_win': top1_win / n, 'top1_top3': top1_top3 / n,
            'contain4': cont[4] / n, 'contain6': cont[6] / n, 'contain8': cont[8] / n,
            'spearman': float(np.mean(sp)),
            'by_size': {k: round(v[0] / v[1], 4) for k, v in sorted(by_size.items())}}


def main():
    t0 = time.time()
    obj = pickle.load(gzip.open('data/_v15_optuna_df_cache_leakfree_v2.pkl.gz', 'rb'))
    df, v15 = obj['df'], obj['features']
    if 'target' not in df.columns:
        df['target'] = (df['finish'] <= 3).astype(int)
    df = build_features(df)
    df['_rk'] = [f"{int(a)}_{b}_{int(c)}_{int(e)}_{int(f)}" for a, b, c, e, f in
                 zip(df['date_num'], df['course'], df['kai'], df['nichi'], df['race_num'])]
    v16 = [f for f in v15 if f not in ODDS_REMOVE]
    models = {'V15': v15,
              's2b': [f for f in v16 if f not in (PROXY_FAMILY + EXTRA + RAW_REPLACE)] + NEW}
    feats_all = set().union(*models.values()) | {'oz_base_pop_rank', 'num_horses_val'}
    for f in feats_all:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    y = pd.to_numeric(df['year'], errors='coerce')
    res = {}
    for name, feats in models.items():
        feats = [f for f in feats if f in df.columns]
        s, auc = make_oof(df, feats)
        df[f's_{name}'] = s
        ev = df[y >= 23]
        r = race_metrics(ev, f's_{name}')
        r['auc'] = auc; r['nf'] = len(feats)
        res[name] = r
        print(f'{name} done ({time.time()-t0:.0f}s)', flush=True)
    # 市場ベンチマーク: 前日人気 (oz_base_pop_rank 小=人気) → score = -pop
    ev = df[y >= 23].copy()
    ev['s_market'] = -ev['oz_base_pop_rank']
    rm = race_metrics(ev, 's_market')
    rm['auc'] = fast_auc(ev['target'].values.astype(float), ev['s_market'].values)
    rm['nf'] = 1
    res['market(前日人気)'] = rm

    print('\n=== 全馬評価 精度 (leak-free v2, 2023-25, N=%d R) ===' % res['V15']['n'])
    hdr = f"{'model':16s}{'AUC':>8s}{'top1勝率':>9s}{'top1馬券内':>10s}{'top4収納':>9s}{'top6収納':>9s}{'top8収納':>9s}{'Spearman':>9s}"
    print(hdr)
    for n, r in res.items():
        print(f"{n:16s}{r['auc']:8.4f}{r['top1_win']*100:8.1f}%{r['top1_top3']*100:9.1f}%"
              f"{r['contain4']*100:8.1f}%{r['contain6']*100:8.1f}%{r['contain8']*100:8.1f}%{r['spearman']:9.4f}")
    mk = res['market(前日人気)']
    print('\n=== 市場対比 (pt) ===')
    for n in ('V15', 's2b'):
        r = res[n]
        print(f"{n}: top1勝率 {(r['top1_win']-mk['top1_win'])*100:+.1f}pt / 馬券内 {(r['top1_top3']-mk['top1_top3'])*100:+.1f}pt / "
              f"top6収納 {(r['contain6']-mk['contain6'])*100:+.1f}pt / AUC {r['auc']-mk['auc']:+.4f}")
    print('\n=== 頭数別 top1馬券内率 ===')
    for n, r in res.items():
        print(f"{n:16s} {r['by_size']}")
    json.dump(res, open('data/fable_accuracy_eval.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=float)
    print('-> data/fable_accuracy_eval.json')


if __name__ == '__main__':
    main()
