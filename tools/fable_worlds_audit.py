#!/usr/bin/env python3
"""Fable独立監査: ワールズ型(地力高×近走好調×人気低)の定量定義 + leak-free v2検証。
検証専用・本番不変・投票未使用。閾値は2020-22分布から導出し2023-25で評価(look-ahead排除)。
比較は同一人気順位で層別(人気効果を除いた純粋edge) + race-clusterブートストラップ95%CI。
"""
from __future__ import annotations
import os, sys, gzip, pickle, json, time
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.abspath('tools'))
from v16_leakfree_roi_grid import load_payouts

DATA = 'data'
RNG = np.random.default_rng(42)
N_BOOT = 2000


def main():
    t0 = time.time()
    path = os.path.join(DATA, '_v15_optuna_df_cache_leakfree_v2.pkl.gz')
    obj = pickle.load(gzip.open(path, 'rb'))
    df = obj['df']
    if 'target' not in df.columns:
        df['target'] = (df['finish'] <= 3).astype(int)
    y = pd.to_numeric(df['year'], errors='coerce')
    df['year_n'] = y
    df['pop'] = pd.to_numeric(df['oz_base_pop_rank'], errors='coerce')
    df['_rk'] = [f"{int(a)}_{b}_{int(c)}_{int(e)}_{int(f)}" for a, b, c, e, f in
                 zip(df['date_num'], df['course'], df['kai'], df['nichi'], df['race_num'])]
    for c in ['jrdb_ze_idm_avg', 'avg_finish_3r', 'top3_count_3r', 'num_horses_val', 'finish', 'horse_num']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # ===== 独立リーク検査 =====
    print("=== 独立リーク検査 (v2 cache) ===")
    m = df['jrdb_ze_idm_avg'].notna() & df['finish'].notna()
    corr = np.corrcoef(df.loc[m, 'jrdb_ze_idm_avg'], df.loc[m, 'finish'])[0, 1]
    print(f"corr(ze_idm_avg, finish) = {corr:+.4f}  (リーク版は強負相関 / leak-free期待 ≈ -0.20)")
    first = df['prev_finish'].isna()  # 初出走proxy
    nan_rate_first = df.loc[first, 'jrdb_ze_idm_avg'].isna().mean()
    nan_rate_rest = df.loc[~first, 'jrdb_ze_idm_avg'].isna().mean()
    print(f"ze NaN率: 初出走(prev_finish欠損)={nan_rate_first:.3f} vs 経験馬={nan_rate_rest:.3f} (初出走≈1.0ならexpanding正常)")
    ev_all = df[df['year_n'] >= 23]
    pop1 = ev_all[ev_all['pop'] == 1]
    print(f"pop健全性: 前日人気1の勝率={(pop1['finish'] == 1).mean():.3f} (市場効率なら~0.30-0.33)、複勝率={(pop1['finish'] <= 3).mean():.3f}")

    # ===== ワールズ型 定義(閾値=2020-22分布から導出) =====
    # universe: 過去走あり(ze/avg_finish_3r非欠損)・pop有・8頭以上
    df['ability_pct'] = df.groupby('_rk')['jrdb_ze_idm_avg'].rank(pct=True)
    df['pop_pct'] = df.groupby('_rk')['pop'].rank(pct=True)
    uni = (df['jrdb_ze_idm_avg'].notna() & df['avg_finish_3r'].notna()
           & df['pop'].notna() & (df['num_horses_val'] >= 8))
    pre = df[uni & (df['year_n'] >= 20) & (df['year_n'] <= 22)]
    form_q1 = pre['avg_finish_3r'].quantile(0.25)
    form_med = pre['avg_finish_3r'].quantile(0.50)
    print(f"\n=== 閾値導出 (2020-22, N={len(pre)}) ===")
    print(f"avg_finish_3r: Q1={form_q1:.2f} / median={form_med:.2f}")
    print(f"定義(base): 地力=ze_idm_avgレース内上位25% AND 近走=avg_finish_3r<=Q1({form_q1:.2f}) AND 人気=レース内下位50%")

    defs = {
        'base':  lambda d: (d['ability_pct'] >= 0.75) & (d['avg_finish_3r'] <= form_q1) & (d['pop_pct'] >= 0.5),
        'alt1':  lambda d: (d['ability_pct'] >= 2/3) & (d['avg_finish_3r'] <= form_med) & (d['pop'] >= 7),
        'alt2':  lambda d: (d['ability_pct'] >= 0.75) & (d['top3_count_3r'] >= 1) & (d['pop_pct'] >= 0.5),
    }

    # ===== 払戻 per-horse =====
    pay = load_payouts()
    ev = df[uni & (df['year_n'] >= 23)].copy()
    tan_ret = np.zeros(len(ev)); fuk_ret = np.zeros(len(ev))
    rks = ev['_rk'].values; hns = ev['horse_num'].values
    miss = 0
    for i, (k, hn) in enumerate(zip(rks, hns)):
        pm = pay.get(k)
        if pm is None:
            miss += 1; tan_ret[i] = np.nan; fuk_ret[i] = np.nan; continue
        tan_ret[i] = pm['tan'][1] if pm['tan'][0] == int(hn) else 0
        fuk_ret[i] = pm['fuk'].get(int(hn), 0)
    ev['tan_ret'] = tan_ret; ev['fuk_ret'] = fuk_ret
    ev = ev[ev['tan_ret'].notna()]
    print(f"\n評価馬数(2023-25, universe, 払戻あり) = {len(ev)} (払戻欠損で除外 {miss})")

    results = {}
    for name, fn in defs.items():
        ev['W'] = fn(ev).astype(int)
        r = eval_def(ev, name)
        results[name] = r

    json.dump(results, open(os.path.join(DATA, 'fable_worlds_audit.json'), 'w'), ensure_ascii=False, indent=2, default=float)
    print(f"\nDONE in {time.time()-t0:.0f}s -> data/fable_worlds_audit.json")


def std_diff(pop_arr, w_arr, val_arr):
    """同人気順位層別の標準化差: Σ w_b * (mean_W,b - mean_nonW,b), w_b = N_W,b / N_W。
    人気順位bにW/非W両方が存在する層のみ使用。"""
    bands = {}
    for b in np.unique(pop_arr):
        mw = (pop_arr == b) & (w_arr == 1); mn = (pop_arr == b) & (w_arr == 0)
        if mw.sum() >= 1 and mn.sum() >= 1:
            bands[int(b)] = (mw.sum(), val_arr[mw].mean(), val_arr[mn].mean())
    tot = sum(v[0] for v in bands.values())
    if tot == 0:
        return np.nan, bands
    d = sum(v[0] * (v[1] - v[2]) for v in bands.values()) / tot
    return d, bands


def eval_def(ev, name):
    print(f"\n{'='*70}\n=== 定義 [{name}] ===")
    W = ev[ev['W'] == 1]
    nW = ev[ev['W'] == 0]
    print(f"該当: {len(W)}頭 / {len(ev)} ({len(W)/len(ev)*100:.1f}%)  レース数={W['_rk'].nunique()}  平均pop={W['pop'].mean():.1f}")
    out = {'n_W': int(len(W)), 'n_races': int(W['_rk'].nunique()), 'mean_pop': float(W['pop'].mean())}

    pop_a = ev['pop'].values.astype(int); w_a = ev['W'].values
    metrics = {'fukusho_rate': ev['target'].values.astype(float),
               'tan_roi': ev['tan_ret'].values / 100.0,
               'fuku_roi': ev['fuk_ret'].values / 100.0}
    # 点推定
    for mn, va in metrics.items():
        d, bands = std_diff(pop_a, w_a, va)
        wmean = va[w_a == 1].mean(); nmean_raw = va[w_a == 0].mean()
        out[mn] = {'W_raw': float(wmean), 'nonW_raw': float(nmean_raw), 'std_diff': float(d)}
        if mn == 'fukusho_rate':
            print(f"  複勝率: W={wmean*100:.1f}% 非W(生)={nmean_raw*100:.1f}%  同人気帯標準化差={d*100:+.2f}pt")
        else:
            print(f"  {mn}: W={wmean*100:.1f}% 非W(生)={nmean_raw*100:.1f}%  同人気帯標準化差={d*100:+.2f}pt")

    # race-cluster bootstrap CI (標準化差)
    rk_codes, rk_idx = pd.factorize(ev['_rk'])
    nrk = len(rk_idx)
    order = np.argsort(rk_codes, kind='stable')
    sorted_codes = rk_codes[order]
    starts = np.searchsorted(sorted_codes, np.arange(nrk))
    ends = np.searchsorted(sorted_codes, np.arange(nrk), side='right')
    boot = {mn: [] for mn in metrics}
    for it in range(N_BOOT):
        rs = RNG.integers(0, nrk, nrk)
        idx = np.concatenate([order[starts[r]:ends[r]] for r in rs])
        pa = pop_a[idx]; wa = w_a[idx]
        for mn, va in metrics.items():
            d, _ = std_diff(pa, wa, va[idx])
            boot[mn].append(d)
    for mn in metrics:
        lo, hi = np.percentile(boot[mn], [2.5, 97.5])
        out[mn]['ci95'] = [float(lo), float(hi)]
        sig = '有意' if (lo > 0 or hi < 0) else '★0跨ぐ=edge未証明★'
        print(f"  {mn} 標準化差 95%CI [{lo*100:+.2f}, {hi*100:+.2f}]pt -> {sig}")

    # 年別
    out['yearly'] = {}
    for yy in [23, 24, 25]:
        msk = ev['year_n'].values == yy
        yout = {}
        for mn, va in metrics.items():
            d, _ = std_diff(pop_a[msk], w_a[msk], va[msk])
            yout[mn] = float(d)
        nw = int(((ev['W'] == 1) & msk).sum())
        yout['n_W'] = nw
        out['yearly'][f'20{yy}'] = yout
        print(f"  20{yy}: N_W={nw} 複勝率差={yout['fukusho_rate']*100:+.2f}pt 単勝ROI差={yout['tan_roi']*100:+.1f}pt 複勝ROI差={yout['fuku_roi']*100:+.1f}pt")

    # 戦略視点: W のみ買い (絶対ROI + CI, race-cluster)
    wmask = w_a == 1
    tan = ev['tan_ret'].values; fuk = ev['fuk_ret'].values
    troi = tan[wmask].mean() / 100; froi = fuk[wmask].mean() / 100
    bt, bf = [], []
    for it in range(N_BOOT):
        rs = RNG.integers(0, nrk, nrk)
        idx = np.concatenate([order[starts[r]:ends[r]] for r in rs])
        wm = w_a[idx] == 1
        if wm.sum() == 0: continue
        bt.append(tan[idx][wm].mean() / 100); bf.append(fuk[idx][wm].mean() / 100)
    tlo, thi = np.percentile(bt, [2.5, 97.5]); flo, fhi = np.percentile(bf, [2.5, 97.5])
    out['strategy'] = {'tan_roi': float(troi), 'tan_ci': [float(tlo), float(thi)],
                       'fuku_roi': float(froi), 'fuku_ci': [float(flo), float(fhi)],
                       'bets_per_year': float(len(W) / 3)}
    print(f"  [戦略] W全買い 単勝ROI={troi*100:.1f}% CI[{tlo*100:.1f},{thi*100:.1f}] 複勝ROI={froi*100:.1f}% CI[{flo*100:.1f},{fhi*100:.1f}] 年間ベット≈{len(W)/3:.0f}")
    return out


if __name__ == '__main__':
    main()
