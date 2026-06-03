#!/usr/bin/env python3
"""ステップ5-3: leak-free cache で V15 vs s2b の ROI総当たり(券種×頭数×レース選択×軍資金)。
★必ず _v15_optuna_df_cache_leakfree.pkl.gz を使用★(リーク版cacheは使わない)。
予測=発走前情報のみ(WF OOF)。配当=JRA公式払戻(jra_payouts.csv, 正キー)でROI計算のみ。本番不変・投票未使用。
"""
from __future__ import annotations
import os, sys, gzip, pickle, json, time
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.abspath('tools'))
from v16_anaba_s2_eval import build_features, ODDS_REMOVE, PROXY_FAMILY, RAW_REPLACE, NEW
EXTRA = ['paci_goal_rank', 'paci_goal_diff', 'paci_dochu_rank']
DATA = 'data'
LGB_P = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 63, 'learning_rate': 0.05,
         'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50, 'verbose': -1, 'seed': 42}
XGB_P = {'objective': 'binary:logistic', 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
         'colsample_bytree': 0.8, 'min_child_weight': 50, 'seed': 42, 'tree_method': 'hist', 'verbosity': 0}


def to_i(x):
    try: return int(float(x))
    except Exception: return None


def load_payouts():
    p = pd.read_csv(os.path.join(DATA, 'jra_payouts.csv'), low_memory=False)
    p = p[(p['race_date'] >= 20230101) & (p['race_date'] <= 20251231)]
    out = {}
    for _, r in p.iterrows():
        key = f"{int(r['race_date'])}_{r['course']}_{int(r['kai'])}_{int(r['nichi'])}_{int(r['race_num'])}"
        fuk = {}
        try:
            for n, pay in zip(str(r['fukusho_nums']).replace(' ', '').split('/'), str(r['fukusho_payouts']).replace(' ', '').split('/')):
                if to_i(n) is not None: fuk[to_i(n)] = to_i(pay) or 0
        except Exception: pass
        wide = {}
        try:
            for nn, pay in zip(str(r['wide_nums']).replace(' ', '').split('/'), str(r['wide_payouts']).replace(' ', '').split('/')):
                a = nn.split('-')
                if len(a) == 2 and all(to_i(x) is not None for x in a): wide[frozenset(to_i(x) for x in a)] = to_i(pay) or 0
        except Exception: pass
        def pset(nums, pay):
            try:
                a = str(nums).replace(' ', '').split('-')
                if all(to_i(x) is not None for x in a): return (frozenset(to_i(x) for x in a), to_i(pay) or 0)
            except Exception: pass
            return None
        def pord(nums, pay):  # 三連単=順序付き(1着-2着-3着)
            try:
                a = str(nums).replace(' ', '').split('-')
                if len(a) == 3 and all(to_i(x) is not None for x in a): return (tuple(to_i(x) for x in a), to_i(pay) or 0)
            except Exception: pass
            return None
        out[key] = {'tan': (to_i(r['tansho_nums']), to_i(r['tansho_payout']) or 0), 'fuk': fuk, 'wide': wide,
                    'umaren': pset(r['umaren_nums'], r['umaren_payout']), 'trio': pset(r['trio_nums'], r['trio_payout']),
                    'tierce': pord(r['tierce_nums'], r['tierce_payout'])}
    return out


def make_oof(df, feats):
    y = pd.to_numeric(df['year'], errors='coerce')
    rows = []; aucs = []
    for ty in [23, 24, 25]:
        tr = y < ty; te = y == ty
        ml = lgb.train(LGB_P, lgb.Dataset(df.loc[tr, feats].values, label=df.loc[tr, 'target'].values), num_boost_round=420)
        mx = xgb.train(XGB_P, xgb.DMatrix(df.loc[tr, feats].values, label=df.loc[tr, 'target'].values), num_boost_round=420)
        p = 0.5 * ml.predict(df.loc[te, feats].values) + 0.5 * mx.predict(xgb.DMatrix(df.loc[te, feats].values))
        aucs.append(roc_auc_score(df.loc[te, 'target'].values, p))
        rows.append(pd.Series(p, index=df.index[te]))
    s = pd.concat(rows)
    return s.reindex(df.index), float(np.mean(aucs))


def race_table(df, score_col):
    """各レース: モデル順位top6 horse_num, 自信度(top1-top2 score), top1のpop。"""
    recs = {}
    for k, g in df.groupby('_rk'):
        if len(g) < 5: continue
        gg = g.sort_values(score_col, ascending=False)
        order = [int(x) for x in gg['horse_num'].tolist()]
        sc = gg[score_col].tolist()
        recs[k] = {'order': order, 'conf': sc[0] - sc[1], 'top1_pop': int(gg.iloc[0]['pop'])}
    return recs


def roi(recs, pay, keys, strat):
    """strat(order, pm)->(return, points). keys=対象レース。"""
    ret = stake = hit = n = 0
    for k in keys:
        if k not in pay or k not in recs: continue
        r, pts = strat(recs[k]['order'], pay[k])
        n += 1; ret += r; stake += 100 * pts; hit += (r > 0)
    return (ret / stake if stake else 0, hit / n if n else 0, n)


# ===== 券種 strategies: (order, pm)->(return, points) =====
def S_tan(o, pm): return (pm['tan'][1] if pm['tan'][0] == o[0] else 0, 1)
def S_fuku1(o, pm): return (pm['fuk'].get(o[0], 0), 1)
def S_fuku3(o, pm): return (sum(pm['fuk'].get(o[i], 0) for i in range(3)), 3)
def S_wide_t3(o, pm):
    import itertools; r = 0
    for a, b in itertools.combinations(o[:3], 2): r += pm['wide'].get(frozenset((a, b)), 0)
    return (r, 3)
def S_umaren_t2(o, pm): return ((pm['umaren'][1] if pm['umaren'] and pm['umaren'][0] == frozenset(o[:2]) else 0), 1)
def S_umaren_t3box(o, pm):
    import itertools; r = 0
    for a, b in itertools.combinations(o[:3], 2):
        if pm['umaren'] and pm['umaren'][0] == frozenset((a, b)): r += pm['umaren'][1]
    return (r, 3)
def _trio_box(o, pm, k):
    import itertools
    if not pm['trio']: return (0, 0)
    combos = list(itertools.combinations(o[:k], 3)); won = pm['trio'][0]
    r = pm['trio'][1] if any(frozenset(c) == won for c in combos) else 0
    return (r, len(combos))
def S_trio3(o, pm): return _trio_box(o, pm, 3)
def S_trio4(o, pm): return _trio_box(o, pm, 4)
def S_trio5(o, pm): return _trio_box(o, pm, 5)
def S_trio_form(o, pm):
    # 本番形: top1軸 × {top2,top3} × {top2..top6} (7点)
    if not pm['trio']: return (0, 0)
    n1 = o[0]; bets = set()
    for s in o[1:3]:
        for t in o[1:6]:
            c = frozenset((n1, s, t))
            if len(c) == 3: bets.add(c)
    r = pm['trio'][1] if pm['trio'][0] in bets else 0
    return (r, len(bets))

def S_tierce3(o, pm):  # 三連単 top3 完全順序 (1点)
    if not pm['tierce']: return (0, 0)
    return (pm['tierce'][1] if pm['tierce'][0] == (o[0], o[1], o[2]) else 0, 1)
def S_tierce_125(o, pm):  # 三連単 formation 1-2-5: 1着=top1, 2着∈top2-3, 3着∈top2-6
    if not pm['tierce']: return (0, 0)
    import itertools
    second = o[1:3]; third = o[1:6]; bets = set()
    for b in second:
        for c in third:
            if len({o[0], b, c}) == 3: bets.add((o[0], b, c))
    return (pm['tierce'][1] if pm['tierce'][0] in bets else 0, len(bets))

STRATS = [('単勝', S_tan), ('複勝top1', S_fuku1), ('複勝top3', S_fuku3), ('ワイドtop3box', S_wide_t3),
          ('馬連top2', S_umaren_t2), ('馬連top3box', S_umaren_t3box), ('三連複top3box', S_trio3),
          ('三連複top4box', S_trio4), ('三連複top5box', S_trio5), ('三連複formation7', S_trio_form),
          ('三連単top3(順序)', S_tierce3), ('三連単form1-2-5', S_tierce_125)]


def main():
    t0 = time.time()
    path = os.path.join(DATA, '_v15_optuna_df_cache_leakfree.pkl.gz')
    assert os.path.exists(path), "leak-free cache が無い"
    obj = pickle.load(gzip.open(path, 'rb')); df = obj['df']; v15 = obj['features']
    if 'target' not in df.columns: df['target'] = (df['finish'] <= 3).astype(int)
    df = build_features(df)
    df['pop'] = pd.to_numeric(df['oz_base_pop_rank'], errors='coerce').fillna(df['num_horses_val'])
    df['_rk'] = [f"{int(a)}_{b}_{int(c)}_{int(e)}_{int(f)}" for a, b, c, e, f in
                 zip(df['date_num'], df['course'], df['kai'], df['nichi'], df['race_num'])]
    v16 = [f for f in v15 if f not in ODDS_REMOVE]
    s2b = [f for f in v16 if f not in (PROXY_FAMILY + EXTRA + RAW_REPLACE)] + NEW
    for f in set(v15) | set(s2b): df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    print(f"leak-free cache 使用。 ze_idm_avg corr(finish)={np.corrcoef(df['jrdb_ze_idm_avg'],df['finish'])[0,1]:+.3f}(≈-0.20=leak-free確認)")

    print("\nWF学習(leak-free)...", flush=True)
    df['s_v15'], a15 = make_oof(df, v15)
    df['s_s2b'], asb = make_oof(df, s2b)
    df['s_mkt'] = -df['pop']
    print(f"  WF AUC: V15={a15:.4f}  s2b={asb:.4f}")
    pay = load_payouts()
    ev = df[pd.to_numeric(df['year'], errors='coerce') >= 23].copy()
    rec = {m: race_table(ev, f's_{m}') for m in ['v15', 's2b', 'mkt']}
    allkeys = list(rec['v15'].keys())
    print(f"評価レース数: {len(allkeys)}")

    print("\n=== 買い方総当たり (全レース) ROI% / 的中% / N ===")
    print(f"{'券種':18s}" + "".join(f"{m:>22s}" for m in ['V15', 's2b', '市場(人気)']))
    grid = {}
    for name, fn in STRATS:
        line = f"{name:18s}"
        for m in ['v15', 's2b', 'mkt']:
            r, h, n = roi(rec[m], pay, allkeys, fn); grid[(name, m)] = (r, h, n)
            line += f"  {r*100:6.1f}%/{h*100:4.1f}%/{n}"
        print(line)

    # ===== レース選択 × 軍資金 =====
    print("\n=== レース選択: 自信度(top1-top2スコア差)上位% で複勝top1/三連複top4box ROI ===")
    for m in ['v15', 's2b']:
        confs = sorted(allkeys, key=lambda k: -rec[m][k]['conf'])
        print(f" [{m}]")
        for pct in [100, 50, 30, 20, 10]:
            sel = confs[:int(len(confs) * pct / 100)]
            r1, h1, n1 = roi(rec[m], pay, sel, S_fuku1)
            r4, h4, n4 = roi(rec[m], pay, sel, S_trio4)
            rt, ht, nt = roi(rec[m], pay, sel, S_tan)
            print(f"   上位{pct:3d}%(N={n1:5d}): 単勝{rt*100:6.1f}% 複勝top1{r1*100:6.1f}% 三連複top4box{r4*100:6.1f}%(的中{h4*100:.1f}%)")

    print("\n=== レース選択: 市場乖離(s2b top1の人気)別 ROI ===")
    for m in ['v15', 's2b']:
        print(f" [{m}]")
        for lo, hi, lab in [(1,1,'人気1'),(2,3,'人気2-3'),(4,6,'人気4-6'),(7,18,'人気7+(穴)')]:
            sel = [k for k in allkeys if lo <= rec[m][k]['top1_pop'] <= hi]
            rt,ht,nt=roi(rec[m],pay,sel,S_tan); r1,h1,n1=roi(rec[m],pay,sel,S_fuku1); r3,h3,n3=roi(rec[m],pay,sel,S_trio3)
            print(f"   {lab:9s}(N={nt:5d}): 単勝{rt*100:6.1f}% 複勝top1{r1*100:6.1f}% 三連複top3box{r3*100:6.1f}%")

    # ===== 最高ROIの特定(N>=200のみ信頼) =====
    print("\n=== ROI>100% かつ N>=200 の組合せ(全レース) ===")
    found = [(name, m, r, h, n) for (name, m), (r, h, n) in grid.items() if r > 1.0 and n >= 200]
    for name, m, r, h, n in sorted(found, key=lambda x: -x[2]):
        print(f"   {m} × {name}: ROI {r*100:.1f}% 的中{h*100:.1f}% N={n}")
    if not found: print("   なし(全レース・全券種でROI<100% または N<200)")

    json.dump({'auc': {'v15': a15, 's2b': asb}, 'grid': {f"{k[0]}|{k[1]}": [v[0], v[1], v[2]] for k, v in grid.items()}},
              open(os.path.join(DATA, 'v16_leakfree_roi_grid.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
