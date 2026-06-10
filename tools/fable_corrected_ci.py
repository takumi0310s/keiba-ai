#!/usr/bin/env python3
"""Fable監査① 採択: s2b vs V15 ペア差ROIの補正後再CI(検証専用・本番不変・leak-free v2のみ)。
補正3段:
 (1) jackpot処理: 2023 ¥728,220 1点を (a)該当レース除外 (b)winsorize(的中配当の99%tile cap) の2通り
 (2) 同日相関: i.i.d. race bootstrap → 日付クラスタbootstrap(同日レースを束で再抽出)
 (3) 多重比較: 検証グリッドの実セル数 m に対する Bonferroni(α=0.05/m)信頼水準でCI
★補正で数字が悪化してもそのまま記録する(糊塗禁止)★
OOFは v2 cache から決定論的に再生成(seed42/420rounds=公表値と同一パイプライン)し、
公表値(単勝 s2b111.3/V15 105.0・三連複t4 207.3/154.4)との一致で整合検証する。
"""
from __future__ import annotations
import os, sys, gzip, pickle, json, time, itertools
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb

sys.path.insert(0, os.path.abspath('tools'))
from v16_anaba_s2_eval import build_features, ODDS_REMOVE, PROXY_FAMILY, RAW_REPLACE, NEW
from v16_leakfree_roi_grid import load_payouts, LGB_P, XGB_P
EXTRA = ['paci_goal_rank', 'paci_goal_diff', 'paci_dochu_rank']
DATA = 'data'
CACHE = os.path.join(DATA, '_v15_optuna_df_cache_leakfree_v2.pkl.gz')
PERRACE = os.path.join(DATA, '_fable_perrace_returns.parquet')
RNG = np.random.default_rng(20260611)
N_BOOT = 100_000  # Bonferroni極端分位の安定化

# ===== 多重比較セル数 m の実カウント(leak-free評価で目視比較された (モデル×券種×選択) セル) =====
# v16_leakfree_roi_grid: STRATS12種 × 3モデル(v15/s2b/mkt)            = 36
#   + 自信度選択 5水準 × 2モデル × 3券種(単勝/複勝top1/三連複t4)        = 30
#   + 人気帯選択 4帯 × 2モデル × 3券種(単勝/複勝top1/三連複t3)          = 24
# v16_pastmodels_leakfree_v2: 4モデル × 3券種(単勝/t4/馬連box)          = 12
M_CELLS = 36 + 30 + 24 + 12  # = 102 (v2基準。v1でも同型グリッド実行=重複再実行は別途感度で確認)


def oof(df, feats, y):
    s = pd.Series(index=df.index, dtype=float)
    for ty in [23, 24, 25]:
        tr = y < ty; te = y == ty
        ml = lgb.train(LGB_P, lgb.Dataset(df.loc[tr, feats].values, label=df.loc[tr, 'target'].values), num_boost_round=420)
        mx = xgb.train(XGB_P, xgb.DMatrix(df.loc[tr, feats].values, label=df.loc[tr, 'target'].values), num_boost_round=420)
        s.loc[te] = 0.5 * ml.predict(df.loc[te, feats].values) + 0.5 * mx.predict(xgb.DMatrix(df.loc[te, feats].values))
        print(f"    fold 20{ty} done", flush=True)
    return s


def per_race(ev, pay, score):
    rows = []
    for k, g in ev.groupby('_rk'):
        if k not in pay or len(g) < 5: continue
        o = [int(x) for x in g.sort_values(score, ascending=False)['horse_num'].tolist()]
        pm = pay[k]
        tan = pm['tan'][1] if pm['tan'][0] == o[0] else 0
        t4 = 0
        if pm['trio']:
            won = pm['trio'][0]
            if any(frozenset(c) == won for c in itertools.combinations(o[:4], 3)):
                t4 = pm['trio'][1]
        rows.append((int(str(k).split('_')[0]), k, tan, t4))
    return pd.DataFrame(rows, columns=['date', 'rk', 'tan', 't4'])


def cluster_boot_diff(rs, rv, nd, pts, levels):
    """日付クラスタbootstrap。rs/rv=per-date払戻合計, nd=per-date レース数。
    levels=[(label, alpha)] で複数信頼水準のCIを返す。"""
    D = len(rs)
    base = (rs.sum() - rv.sum()) / (100 * pts * nd.sum())
    samp = np.empty(N_BOOT)
    chunk = 5000
    for s0 in range(0, N_BOOT, chunk):
        n = min(chunk, N_BOOT - s0)
        idx = RNG.integers(0, D, (n, D))
        samp[s0:s0+n] = (rs[idx].sum(1) - rv[idx].sum(1)) / (100 * pts * nd[idx].sum(1))
    out = {'base': float(base)}
    for label, alpha in levels:
        lo, hi = np.percentile(samp, [100*alpha/2, 100*(1-alpha/2)])
        out[label] = [float(lo), float(hi)]
    return out


def cluster_boot_abs(rr, nd, pts, levels):
    D = len(rr)
    base = rr.sum() / (100 * pts * nd.sum())
    samp = np.empty(N_BOOT)
    chunk = 5000
    for s0 in range(0, N_BOOT, chunk):
        n = min(chunk, N_BOOT - s0)
        idx = RNG.integers(0, D, (n, D))
        samp[s0:s0+n] = rr[idx].sum(1) / (100 * pts * nd[idx].sum(1))
    out = {'base': float(base)}
    for label, alpha in levels:
        lo, hi = np.percentile(samp, [100*alpha/2, 100*(1-alpha/2)])
        out[label] = [float(lo), float(hi)]
    return out


def date_sums(pr, col):
    g = pr.groupby('date')
    return g[f'{col}_s2b'].sum().values.astype(float), g[f'{col}_v15'].sum().values.astype(float), g.size().values.astype(float)


def main():
    t0 = time.time()
    if os.path.exists(PERRACE):
        pr = pd.read_parquet(PERRACE)
        print(f"per-race returns 再利用: {PERRACE} ({len(pr)}R)")
    else:
        assert os.path.exists(CACHE), "leak-free v2 cache が無い"
        obj = pickle.load(gzip.open(CACHE, 'rb')); df = obj['df']; v15 = obj['features']
        if 'target' not in df.columns: df['target'] = (df['finish'] <= 3).astype(int)
        df = build_features(df)
        df['_rk'] = [f"{int(a)}_{b}_{int(c)}_{int(e)}_{int(f)}" for a, b, c, e, f in
                     zip(df['date_num'], df['course'], df['kai'], df['nichi'], df['race_num'])]
        v16 = [f for f in v15 if f not in ODDS_REMOVE]
        s2b = [f for f in v16 if f not in (PROXY_FAMILY + EXTRA + RAW_REPLACE)] + NEW
        for f in set(v15) | set(s2b): df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
        y = pd.to_numeric(df['year'], errors='coerce')
        corr = np.corrcoef(df['jrdb_ze_idm_avg'], df['finish'])[0, 1]
        print(f"leak-free v2。 ze corr(finish)={corr:+.3f}。 WF OOF再生成(決定論)...", flush=True)
        print("  [s2b]"); df['s_s2b'] = oof(df, s2b, y)
        print("  [v15]"); df['s_v15'] = oof(df, v15, y)
        pay = load_payouts(); ev = df[y >= 23]
        a = per_race(ev, pay, 's_s2b').rename(columns={'tan': 'tan_s2b', 't4': 't4_s2b'})
        b = per_race(ev, pay, 's_v15').rename(columns={'tan': 'tan_v15', 't4': 't4_v15'})
        pr = a.merge(b[['rk', 'tan_v15', 't4_v15']], on='rk')
        pr.to_parquet(PERRACE)
        print(f"per-race returns 保存: {PERRACE} ({len(pr)}R)")

    N = len(pr)
    # ===== 整合検証(公表値と一致するか) =====
    base = {c: pr[c].sum() / (100 * p * N) for c, p in
            [('tan_s2b', 1), ('tan_v15', 1), ('t4_s2b', 4), ('t4_v15', 4)]}
    print(f"\n=== 整合検証(公表値: 単勝 s2b111.3/V15 105.0・t4 s2b207.3/V15 154.4) N={N} ===")
    print(f"  単勝 s2b={base['tan_s2b']*100:.1f}% V15={base['tan_v15']*100:.1f}% | t4 s2b={base['t4_s2b']*100:.1f}% V15={base['t4_v15']*100:.1f}%")

    # ===== jackpot 特定 =====
    jmax = pr['t4_s2b'].max()
    jrace = pr.loc[pr['t4_s2b'].idxmax()]
    print(f"\njackpot: {jrace['rk']} s2b t4払戻 ¥{int(jmax):,}")

    # winsorize cap: 両モデルプールの的中配当 99%tile
    hits_pool_t4 = np.concatenate([pr.loc[pr['t4_s2b'] > 0, 't4_s2b'].values, pr.loc[pr['t4_v15'] > 0, 't4_v15'].values])
    cap_t4 = float(np.percentile(hits_pool_t4, 99))
    hits_pool_tan = np.concatenate([pr.loc[pr['tan_s2b'] > 0, 'tan_s2b'].values, pr.loc[pr['tan_v15'] > 0, 'tan_v15'].values])
    cap_tan = float(np.percentile(hits_pool_tan, 99))
    print(f"winsorize cap(的中配当プール99%tile): t4=¥{cap_t4:,.0f} 単勝=¥{cap_tan:,.0f}")

    variants = {}
    variants['raw'] = pr.copy()
    variants['excl_jackpot'] = pr[pr['rk'] != jrace['rk']].copy()
    w = pr.copy()
    for c in ['t4_s2b', 't4_v15']: w[c] = w[c].clip(upper=cap_t4)
    for c in ['tan_s2b', 'tan_v15']: w[c] = w[c].clip(upper=cap_tan)
    variants['winsorize'] = w

    alpha_bonf = 0.05 / M_CELLS
    levels = [('ci95', 0.05), (f'ci_bonf(m={M_CELLS})', alpha_bonf)]
    print(f"\n多重比較: m={M_CELLS}セル → Bonferroni α={alpha_bonf:.5f} → {100*(1-alpha_bonf):.3f}%CI")
    print(f"bootstrap: 日付クラスタ(D={pr['date'].nunique()}日) × n={N_BOOT}")

    out = {'N': int(N), 'm_cells': M_CELLS, 'jackpot_rk': str(jrace['rk']), 'jackpot_pay': int(jmax),
           'cap_t4': cap_t4, 'cap_tan': cap_tan, 'n_dates': int(pr['date'].nunique()), 'results': {}}
    for vname, vpr in variants.items():
        print(f"\n===== variant [{vname}] (N={len(vpr)}) =====")
        vout = {}
        for bet, pts in [('tan', 1), ('t4', 4)]:
            rs, rv, nd = date_sums(vpr, bet)
            d = cluster_boot_diff(rs, rv, nd, pts, levels)
            asb = cluster_boot_abs(rs, nd, pts, levels)
            av15 = cluster_boot_abs(rv, nd, pts, levels)
            vout[bet] = {'diff': d, 's2b': asb, 'v15': av15}
            lab = '単勝' if bet == 'tan' else '三連複t4box'
            print(f" {lab}: s2b={asb['base']*100:.1f}% V15={av15['base']*100:.1f}%")
            print(f"   ペア差={d['base']*100:+.1f}pt  95%CI[{d['ci95'][0]*100:+.1f},{d['ci95'][1]*100:+.1f}]"
                  f"  Bonferroni{100*(1-alpha_bonf):.3f}%CI[{d[levels[1][0]][0]*100:+.1f},{d[levels[1][0]][1]*100:+.1f}]")
            for label, _ in levels:
                lo, hi = d[label]
                print(f"     {label}: {'0を跨がない=有意' if (lo > 0 or hi < 0) else '★0を跨ぐ=有意でない★'}")
        out['results'][vname] = vout

    # ===== 6/17 判定向け: paper必要N(検出力試算) =====
    print("\n===== 6/17判定向け: paper必要N試算(winsorize後ペア差を真値と仮定、片側α=0.05・検出力80%) =====")
    wpr = variants['winsorize']
    need = {}
    for bet, pts in [('tan', 1), ('t4', 4)]:
        di = (wpr[f'{bet}_s2b'] - wpr[f'{bet}_v15']).values / (100 * pts)  # per-race ROI diff
        delta = di.mean(); sd = di.std()
        n_req = ((1.645 + 0.8416) * sd / delta) ** 2 if delta > 0 else float('inf')
        need[bet] = {'delta_pt': float(delta * 100), 'sd_pt': float(sd * 100), 'n_required': float(n_req)}
        print(f"  {'単勝' if bet=='tan' else '三連複t4'}: ペア差/R={delta*100:+.2f}pt sd={sd*100:.0f}pt → 必要N≈{n_req:,.0f}R")
    out['paper_n_required'] = need

    json.dump(out, open(os.path.join(DATA, 'fable_corrected_ci.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s -> data/fable_corrected_ci.json")


if __name__ == '__main__':
    main()
