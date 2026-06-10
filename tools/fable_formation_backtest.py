#!/usr/bin/env python3
"""買い目構造(三連複/三連単フォーメーション)backtest — 戦略層検証・本番不変・leak-free v2のみ。
★設計は2020-22(OOF)・評価は2023-25(OOF)。look-ahead禁止★
★selection対策: 構造数 m=6 を事前固定(下記 STRUCTURES)。対F0ペア差CIは同日クラスタ
  bootstrap + Bonferroni(α=0.05/6) — Fable監査①と同手法★
★jackpot敏感性: winsorize(的中配当の99%tileキャップ)床を併記★

構造候補(m=6・固定・狙い):
 F0 現行7点    : 三連複F top1軸 × {top2,top3} × {top2..top6}     — 基準(本番と同形)
 F1 ながし10点 : 三連複 top1軸ながし 相手top2-6 (C(5,2)=10)      — 軸信頼・相手を切らない
 F2 二軸F      : 三連複F {top1,top2} × {top1..4} × {top1..6}      — 軸を2頭に分散
 F3 切り強化5点: 三連複F top1軸 × {top2,top3} × {top2..top5}      — 3列目top5まで(コスト減)
 F4 三連単F    : 1着{top1,2} × 2着{top1..4} × 3着{top1..6}        — 順序を当てに行く(控除率27.5%)
 F5 box4点     : 三連複 top4 BOX (C(4,3)=4)                        — 軸固定をやめ少点数
"""
from __future__ import annotations
import os, sys, gzip, pickle, json, time, itertools
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.abspath('tools'))
DATA = 'data'
CACHE = os.path.join(DATA, '_v15_optuna_df_cache_leakfree_v2.pkl.gz')
OOF_PATH = os.path.join(DATA, '_fable_v15_oof_full.parquet')
RNG = np.random.default_rng(20260612)
N_BOOT = 100_000
M_STRUCT = 6
ALPHA_BONF = 0.05 / M_STRUCT


def to_i(x):
    try: return int(float(x))
    except Exception: return None


def load_payouts_full():
    """jra_payouts.csv 2020-2025 (trio + tierce)。"""
    p = pd.read_csv(os.path.join(DATA, 'jra_payouts.csv'), low_memory=False)
    p = p[(p['race_date'] >= 20200101) & (p['race_date'] <= 20251231)]
    out = {}
    for _, r in p.iterrows():
        key = f"{int(r['race_date'])}_{r['course']}_{int(r['kai'])}_{int(r['nichi'])}_{int(r['race_num'])}"
        def pset(nums, pay):
            try:
                a = str(nums).replace(' ', '').split('-')
                if len(a) == 3 and all(to_i(x) is not None for x in a):
                    return (frozenset(to_i(x) for x in a), to_i(pay) or 0)
            except Exception: pass
            return None
        def pord(nums, pay):
            try:
                a = str(nums).replace(' ', '').split('-')
                if len(a) == 3 and all(to_i(x) is not None for x in a):
                    return (tuple(to_i(x) for x in a), to_i(pay) or 0)
            except Exception: pass
            return None
        out[key] = {'trio': pset(r['trio_nums'], r['trio_payout']),
                    'tierce': pord(r['tierce_nums'], r['tierce_payout'])}
    return out


def make_oof_full(df, feats):
    """WF OOF 2020-25 (fold=ty, train=y<ty)。Fable①と同一パイプライン(seed42/420r)。"""
    import lightgbm as lgb, xgboost as xgb
    LGB_P = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 63, 'learning_rate': 0.05,
             'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
             'min_child_samples': 50, 'verbose': -1, 'seed': 42}
    XGB_P = {'objective': 'binary:logistic', 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
             'colsample_bytree': 0.8, 'min_child_weight': 50, 'seed': 42, 'tree_method': 'hist', 'verbosity': 0}
    y = pd.to_numeric(df['year'], errors='coerce')
    s = pd.Series(index=df.index, dtype=float)
    for ty in [20, 21, 22, 23, 24, 25]:
        tr = y < ty; te = y == ty
        ml = lgb.train(LGB_P, lgb.Dataset(df.loc[tr, feats].values, label=df.loc[tr, 'target'].values), num_boost_round=420)
        mx = xgb.train(XGB_P, xgb.DMatrix(df.loc[tr, feats].values, label=df.loc[tr, 'target'].values), num_boost_round=420)
        s.loc[te] = 0.5 * ml.predict(df.loc[te, feats].values) + 0.5 * mx.predict(xgb.DMatrix(df.loc[te, feats].values))
        print(f"  fold 20{ty} done", flush=True)
    return s


# ===== 構造定義 (order=モデル順位順の馬番リスト) =====
def trio_form(order, axes_idx, col2_idx, col3_idx):
    bets = set()
    for a in axes_idx:
        for b in col2_idx:
            for c in col3_idx:
                s = frozenset((order[a], order[b], order[c]))
                if len(s) == 3:
                    bets.add(s)
    return bets

def F0(o): return trio_form(o, [0], [1, 2], [1, 2, 3, 4, 5])           # 現行7点
def F1(o): return {frozenset((o[0], a, b)) for a, b in itertools.combinations(o[1:6], 2)}  # ながし10点
def F2(o): return trio_form(o, [0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4, 5])  # 二軸F
def F3(o): return trio_form(o, [0], [1, 2], [1, 2, 3, 4])              # 切り強化5点
def F4(o):  # 三連単F (順序付き)
    bets = set()
    for a in o[:2]:
        for b in o[:4]:
            for c in o[:6]:
                if len({a, b, c}) == 3:
                    bets.add((a, b, c))
    return bets
def F5(o): return {frozenset(c) for c in itertools.combinations(o[:4], 3)}  # box4点

STRUCTURES = [('F0_現行7点', F0, 'trio'), ('F1_ながし10点', F1, 'trio'), ('F2_二軸F', F2, 'trio'),
              ('F3_切り強化5点', F3, 'trio'), ('F4_三連単F', F4, 'tierce'), ('F5_box4点', F5, 'trio')]


def cluster_boot_diff_roi(rf, sf, r0, s0, n_boot=N_BOOT):
    """日付クラスタbootstrap: ROI差 = sum(rf)/sum(sf) − sum(r0)/sum(s0)。per-date合計配列を入力。"""
    D = len(rf)
    base = rf.sum() / sf.sum() - r0.sum() / s0.sum()
    samp = np.empty(n_boot)
    for st in range(0, n_boot, 5000):
        n = min(5000, n_boot - st)
        idx = RNG.integers(0, D, (n, D))
        samp[st:st+n] = rf[idx].sum(1) / sf[idx].sum(1) - r0[idx].sum(1) / s0[idx].sum(1)
    ci95 = np.percentile(samp, [2.5, 97.5])
    cib = np.percentile(samp, [100 * ALPHA_BONF / 2, 100 * (1 - ALPHA_BONF / 2)])
    return float(base), [float(x) for x in ci95], [float(x) for x in cib]


def max_dd(net):
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    return float((peak - cum).max())


def main():
    t0 = time.time()
    obj = pickle.load(gzip.open(CACHE, 'rb')); df = obj['df']; v15 = obj['features']
    if 'target' not in df.columns:
        df['target'] = (df['finish'] <= 3).astype(int)
    df['_rk'] = [f"{int(a)}_{b}_{int(c)}_{int(e)}_{int(f)}" for a, b, c, e, f in
                 zip(df['date_num'], df['course'], df['kai'], df['nichi'], df['race_num'])]
    for f in v15:
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    y = pd.to_numeric(df['year'], errors='coerce')

    if os.path.exists(OOF_PATH):
        oof = pd.read_parquet(OOF_PATH)
        df = df.merge(oof, on=['_rk', 'horse_num'], how='left', suffixes=('', '_oof'))
        df['s_v15'] = df['s_v15_oof'] if 's_v15_oof' in df.columns else df['s_v15']
        print(f"OOF再利用: {OOF_PATH}")
    else:
        print("WF OOF 2020-25 生成(leak-free v2・決定論)...", flush=True)
        df['s_v15'] = make_oof_full(df, v15)
        df[df['s_v15'].notna()][['_rk', 'horse_num', 's_v15']].to_parquet(OOF_PATH)
        print(f"OOF保存: {OOF_PATH}")

    pay = load_payouts_full()
    print(f"payouts: {len(pay)} races (2020-25)")

    # レース表(順位列)。評価=6頭以上 & trio/tierce払戻あり。
    # ★設計期(2020-22)は jra_payouts.csv に当該年の配当が無い(現状2018+2023-26のみ)ため
    #   配当不要のカバレッジ統計のみで設計する(構造は本スクリプトで事前固定済・変更しない)★
    def race_orders(sub, need_pay):
        recs = {}
        for k, g in sub.groupby('_rk'):
            if len(g) < 6 or g['s_v15'].isna().any():
                continue
            pm = pay.get(k)
            if need_pay and (pm is None or pm['trio'] is None or pm['tierce'] is None):
                continue
            gg = g.sort_values('s_v15', ascending=False)
            recs[k] = {'order': [int(x) for x in gg['horse_num'].tolist()],
                       'podium': set(int(h) for h, fin in zip(gg['horse_num'], gg['finish']) if fin <= 3),
                       'winner': next((int(h) for h, fin in zip(gg['horse_num'], gg['finish']) if fin == 1), None),
                       'pm': pm, 'date': int(str(k).split('_')[0])}
        return recs

    rec_design = race_orders(df[(y >= 20) & (y <= 22)], need_pay=False)
    rec_eval = race_orders(df[y >= 23], need_pay=True)
    print(f"母集団: 設計2020-22={len(rec_design)}R(配当なし=カバレッジのみ) 評価2023-25={len(rec_eval)}R")

    # ===== ■0 土台統計 =====
    out = {'m_struct': M_STRUCT, 'alpha_bonf': ALPHA_BONF}
    print("\n=== ■0 土台統計(カバレッジ) ===")
    for lab, recs in [('設計2020-22', rec_design), ('評価2023-25', rec_eval)]:
        n = len(recs)
        if n == 0:
            print(f" [{lab}] N=0 (skip)"); continue
        cov = {K: 0 for K in [4, 5, 6, 8]}
        c_top1_pod = c_top1_win = c_win_top2 = 0
        pod_payout_capt = {k: 0 for k in [4, 5, 6, 8]}
        tot_payout = 0
        for r in recs.values():
            o = r['order']; pod = r['podium']
            for K in [4, 5, 6, 8]:
                if pod <= set(o[:K]):
                    cov[K] += 1
            c_top1_pod += o[0] in pod
            c_top1_win += o[0] == r['winner']
            c_win_top2 += r['winner'] in o[:2]
            if r['pm'] and r['pm']['trio']:
                tp = r['pm']['trio'][1]
                tot_payout += tp
                for K in [4, 5, 6, 8]:
                    if r['pm']['trio'][0] <= set(o[:K]):
                        pod_payout_capt[K] += tp
        s = {f'top{K}内全収納': cov[K] / n for K in [4, 5, 6, 8]}
        s.update({'top1馬券内': c_top1_pod / n, 'top1勝利': c_top1_win / n, '勝馬top2内': c_win_top2 / n})
        if tot_payout:
            s.update({f'払戻金捕捉率top{K}': pod_payout_capt[K] / tot_payout for K in [4, 5, 6, 8]})
        out[f'coverage_{lab}'] = s
        print(f" [{lab}] N={n}")
        print(f"  馬券内3頭全収納率: top4={s['top4内全収納']*100:.1f}% top5={s['top5内全収納']*100:.1f}% top6={s['top6内全収納']*100:.1f}% top8={s['top8内全収納']*100:.1f}%")
        print(f"  top1馬券内={s['top1馬券内']*100:.1f}% top1勝利={s['top1勝利']*100:.1f}% 勝馬がtop2内={s['勝馬top2内']*100:.1f}%")
        line = f"  ★切り危険★ 馬券内にtop7以下が混入する率={100-s['top6内全収納']*100:.1f}%"
        if tot_payout:
            line += f" / trio払戻金のtopK捕捉率: top4={s['払戻金捕捉率top4']*100:.1f}% top6={s['払戻金捕捉率top6']*100:.1f}% top8={s['払戻金捕捉率top8']*100:.1f}%"
        print(line)

    # ===== ■2 評価(2023-25) =====
    print(f"\n=== ■2 構造別評価(2023-25, N={len(rec_eval)}R, 100円/点) ===")
    keys = sorted(rec_eval.keys(), key=lambda k: rec_eval[k]['date'])
    rows = {}
    for name, fn, btype in STRUCTURES:
        ret = np.zeros(len(keys)); stake = np.zeros(len(keys)); hit = np.zeros(len(keys), bool)
        for i, k in enumerate(keys):
            r = rec_eval[k]; bets = fn(r['order'])
            stake[i] = 100 * len(bets)
            won = r['pm'][btype]
            if won and won[0] in bets:
                ret[i] = won[1]; hit[i] = True
        rows[name] = {'ret': ret, 'stake': stake, 'hit': hit, 'btype': btype}

    # winsorize床: 券種ごとに的中配当プールの99%tileキャップ
    caps = {}
    for btype in ['trio', 'tierce']:
        pool = np.concatenate([rows[n]['ret'][rows[n]['ret'] > 0] for n, _, bt in STRUCTURES if bt == btype]) if any(bt == btype for _, _, bt in STRUCTURES) else np.array([0])
        caps[btype] = float(np.percentile(pool, 99)) if len(pool) else 0
    print(f"winsorize cap: trio=¥{caps['trio']:,.0f} tierce=¥{caps['tierce']:,.0f}")

    dates = np.array([rec_eval[k]['date'] for k in keys])
    udates, dinv = np.unique(dates, return_inverse=True)
    def per_date(v):
        return np.bincount(dinv, weights=v, minlength=len(udates))

    r0 = rows['F0_現行7点']
    print(f"\n{'構造':16s}{'点数':>5s}{'ROI':>8s}{'wins床':>8s}{'的中率':>8s}{'maxDD':>10s}  年別ROI(23/24/25)")
    for name, fn, btype in STRUCTURES:
        r = rows[name]
        roi = r['ret'].sum() / r['stake'].sum()
        wret = np.minimum(r['ret'], caps[btype])
        wroi = wret.sum() / r['stake'].sum()
        hitr = r['hit'].mean()
        dd = max_dd(r['ret'] - r['stake'])
        yr_roi = []
        for yy in [2023, 2024, 2025]:
            m = (dates // 10000) == yy
            yr_roi.append(r['ret'][m].sum() / r['stake'][m].sum() if r['stake'][m].sum() else 0)
        pts = r['stake'].mean() / 100
        res = {'roi': float(roi), 'wroi': float(wroi), 'hit': float(hitr), 'pts': float(pts),
               'maxdd': dd, 'yearly': [float(x) for x in yr_roi]}
        if name != 'F0_現行7点':
            base, ci95, cib = cluster_boot_diff_roi(per_date(r['ret']), per_date(r['stake']),
                                                    per_date(r0['ret']), per_date(r0['stake']))
            wbase, wci95, wcib = cluster_boot_diff_roi(per_date(wret), per_date(r['stake']),
                                                       per_date(np.minimum(r0['ret'], caps['trio'])), per_date(r0['stake']))
            res.update({'diff': base, 'diff_ci95': ci95, 'diff_cibonf': cib,
                        'wdiff': wbase, 'wdiff_ci95': wci95, 'wdiff_cibonf': wcib})
        rows[name]['res'] = res
        print(f"{name:16s}{pts:5.1f}{roi*100:7.1f}%{wroi*100:7.1f}%{hitr*100:7.1f}%{dd:10,.0f}  " +
              "/".join(f"{x*100:.0f}%" for x in yr_roi))
        if name != 'F0_現行7点':
            sig = lambda ci: '生存' if ci[0] > 0 else ('負け確' if ci[1] < 0 else '0跨ぐ')
            print(f"  対F0差 raw {res['diff']*100:+.1f}pt 95%[{ci95[0]*100:+.1f},{ci95[1]*100:+.1f}] Bonf[{cib[0]*100:+.1f},{cib[1]*100:+.1f}]={sig(cib)}"
                  f" | winsorize {res['wdiff']*100:+.1f}pt Bonf[{wcib[0]*100:+.1f},{wcib[1]*100:+.1f}]={sig(wcib)}")

    # 三連単の必要N(①方式)
    r4 = rows['F4_三連単F']
    di = r4['ret'] / r4['stake'] - r0['ret'] / r0['stake']
    delta, sd = float(di.mean()), float(di.std())
    n_req = ((1.645 + 0.8416) * sd / delta) ** 2 if delta > 0 else float('inf')
    print(f"\n三連単F4: 控除率27.5%(三連複25%)・的中N={int(r4['hit'].sum())} / paperペア差確証 必要N≈{n_req:,.0f}R" if np.isfinite(n_req)
          else f"\n三連単F4: 対F0差が非正(δ={delta*100:+.2f}pt) → ペア差検証不能(必要N=∞)")
    out['tierce_paper_n'] = None if not np.isfinite(n_req) else float(n_req)

    out['structures'] = {n: rows[n]['res'] for n, _, _ in STRUCTURES}
    json.dump(out, open(os.path.join(DATA, 'fable_formation_backtest.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2, default=float)
    print(f"\nDONE in {time.time()-t0:.0f}s -> data/fable_formation_backtest.json")


if __name__ == '__main__':
    main()
