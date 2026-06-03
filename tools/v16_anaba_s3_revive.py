#!/usr/bin/env python3
"""大規模②: s2b に蘇生特徴(馬具変更/足元/バイアス×枠×脚質交互)を追加し leak-free v2 で厳密検証。
★leak-free v2 cache のみ使用・発走前確定特徴のみ・リーク版禁止★。
新特徴: bagu_change/is_bagu/ashimoto(TYB直前=発走前・再構築12桁race_idで100%結合)+ bias×枠×脚質交互。
ブリンカー(歴史raw KYI不在)・A/Bコース替わり(データ無/IP BAN)は見送り。
検証: s2b vs s3_revive で ROI(単勝/三連複t4/馬連)・★反市場好走率(base22%超えるか=本物の穴力)★・spearman・N・gain。
本番V15/V16不変・投票未使用。
"""
from __future__ import annotations
import os, sys, gzip, pickle, json, time
if sys.platform == "win32": sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from scipy.stats import spearmanr
sys.path.insert(0, os.path.abspath('tools'))
from v16_anaba_s2_eval import build_features, ODDS_REMOVE, PROXY_FAMILY, RAW_REPLACE, NEW
from v16_leakfree_roi_grid import load_payouts, make_oof, S_tan, S_trio4, S_umaren_t3box, LGB_P, XGB_P
EXTRA = ['paci_goal_rank', 'paci_goal_diff', 'paci_dochu_rank']
DATA = 'data'
BASHO = {'札幌':'01','函館':'02','福島':'03','新潟':'04','東京':'05','中山':'06','中京':'07','京都':'08','阪神':'09','小倉':'10'}
REVIVE = ['bagu_change', 'is_bagu', 'ashimoto', 'front_inner_bias3', 'bagu_x_front']


def add_revive_features(df):
    # --- TYB 直前データ(発走前確定): 馬具変更・足元 を再構築12桁race_idで結合 ---
    def rid12(r):
        b = BASHO.get(str(r['course']))
        if b is None: return None
        try: return f"{int(r['date_num'])//10000:04d}{b}{int(r['kai']):02d}{int(r['nichi']):02d}{int(r['race_num']):02d}"
        except Exception: return None
    df['_rid12'] = df.apply(rid12, axis=1)
    df['_uma'] = pd.to_numeric(df['horse_num'], errors='coerce')
    tyb = pd.read_csv(os.path.join(DATA, 'jrdb_tyb.csv'), dtype={'race_id': str}, usecols=['race_id', 'umaban', 'bagu_change', 'ashimoto'])
    tyb['umaban'] = pd.to_numeric(tyb['umaban'], errors='coerce')
    tyb['_k'] = tyb['race_id'].astype(str) + '_' + tyb['umaban'].astype('Int64').astype(str)
    tyb = tyb.drop_duplicates('_k')
    df['_k'] = df['_rid12'].astype(str) + '_' + df['_uma'].astype('Int64').astype(str)
    mp_bagu = dict(zip(tyb['_k'], pd.to_numeric(tyb['bagu_change'], errors='coerce')))
    mp_ashi = dict(zip(tyb['_k'], pd.to_numeric(tyb['ashimoto'], errors='coerce')))
    df['bagu_change'] = df['_k'].map(mp_bagu).fillna(0)
    df['is_bagu'] = (df['bagu_change'] > 0).astype(int)
    df['ashimoto'] = df['_k'].map(mp_ashi).fillna(0)
    # --- バイアス×枠×脚質 交互(cache内: jrdb_tb_homestr_inner + bracket/horse_num + running_style)---
    rs = pd.to_numeric(df['jrdb_running_style'], errors='coerce').fillna(0)
    is_front = rs.isin([1, 2]).astype(int)
    nh = pd.to_numeric(df['num_horses_val'], errors='coerce').fillna(0).clip(lower=1)
    hn = pd.to_numeric(df['horse_num'], errors='coerce').fillna(99)
    inner_draw = (hn <= np.ceil(0.35 * nh)).astype(int)
    tb = pd.to_numeric(df.get('jrdb_tb_homestr_inner', 0), errors='coerce').fillna(0)
    df['front_inner_bias3'] = is_front * inner_draw * tb         # 内有利日×内枠×先行 の3way
    df['bagu_x_front'] = df['is_bagu'] * is_front                 # 馬具変更×先行(変化×展開)
    return df


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
    return (hit/h if h else float('nan'), float(np.mean(sp)), nr, h)


def roi_of(ev, scol, pay, fn):
    ret = stake = hit = n = 0
    for k, g in ev.groupby('_rk'):
        if k not in pay or len(g) < 5: continue
        o = [int(x) for x in g.sort_values(scol, ascending=False)['horse_num'].tolist()]
        r, pts = fn(o, pay[k]); n += 1; ret += r; stake += 100 * pts; hit += (r > 0)
    return (ret/stake if stake else 0, hit/n if n else 0, n)


def gain_pct(model, feats, names):
    g = dict(zip(model.feature_name(), model.feature_importance(importance_type='gain')))
    if all(k.startswith('Column_') for k in list(g)[:3]):
        g = {feats[int(k.split('_')[1])]: v for k, v in g.items()}
    tot = sum(g.values()) or 1
    return {n: 100*g.get(n, 0)/tot for n in names}


def main():
    t0 = time.time()
    obj = pickle.load(gzip.open(os.path.join(DATA, '_v15_optuna_df_cache_leakfree_v2.pkl.gz'), 'rb'))
    df = obj['df']; v15 = obj['features']
    if 'target' not in df.columns: df['target'] = (df['finish'] <= 3).astype(int)
    df = build_features(df)
    df = add_revive_features(df)
    df['pop'] = pd.to_numeric(df['oz_base_pop_rank'], errors='coerce').fillna(df['num_horses_val'])
    df['_rk'] = [f"{int(a)}_{b}_{int(c)}_{int(e)}_{int(f)}" for a, b, c, e, f in
                 zip(df['date_num'], df['course'], df['kai'], df['nichi'], df['race_num'])]
    v16 = [f for f in v15 if f not in ODDS_REMOVE]
    s2b = [f for f in v16 if f not in (PROXY_FAMILY+EXTRA+RAW_REPLACE)] + NEW
    s3 = s2b + REVIVE
    for f in set(v15) | set(s2b) | set(REVIVE): df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    y = pd.to_numeric(df['year'], errors='coerce')
    pay = load_payouts()
    print(f"leak-free v2。 蘇生特徴: {REVIVE}。 s2b={len(s2b)} s3_revive={len(s3)}", flush=True)

    # OOF (V15 market ref, s2b, s3_revive) + s3 gain
    df['s_v15'], a15 = make_oof(df, v15)
    df['s_s2b'], asb = make_oof(df, s2b); print(f"  s2b AUC={asb:.4f}", flush=True)
    # s3: make_oof + 最終foldモデルのgain取得
    s3_oof = pd.Series(index=df.index, dtype=float); a3 = []; g3 = {}
    from sklearn.metrics import roc_auc_score
    for ty in [23, 24, 25]:
        tr = y < ty; te = y == ty
        ml = lgb.train(LGB_P, lgb.Dataset(df.loc[tr, s3].values, label=df.loc[tr, 'target'].values), num_boost_round=420)
        mx = xgb.train(XGB_P, xgb.DMatrix(df.loc[tr, s3].values, label=df.loc[tr, 'target'].values), num_boost_round=420)
        s3_oof.loc[te] = 0.5*ml.predict(df.loc[te, s3].values) + 0.5*mx.predict(xgb.DMatrix(df.loc[te, s3].values))
        a3.append(roc_auc_score(df.loc[te, 'target'].values, s3_oof[te]))
        for n, v in gain_pct(ml, s3, REVIVE).items(): g3[n] = g3.get(n, 0) + v/3
    df['s_s3'] = s3_oof; a3 = np.mean(a3); print(f"  s3_revive AUC={a3:.4f}", flush=True)
    ev = df[y >= 23]

    print("\n=== 蘇生特徴 gain%(s3_revive・生き返ったか) ===")
    for n in REVIVE: print(f"  {n:18s} {g3.get(n,0):.3f}%")

    print("\n=== leak-free v2 検証: s2b vs s3_revive ===")
    rows = []
    for name, sc in [('s2b', 's_s2b'), ('s3_revive', 's_s3')]:
        auc = asb if name == 's2b' else a3
        tan = roi_of(ev, sc, pay, S_tan); t4 = roi_of(ev, sc, pay, S_trio4); um = roi_of(ev, sc, pay, S_umaren_t3box)
        hr, sp, nr, hh = anaba(ev, sc)
        rows.append((name, auc, tan, t4, um, hr, sp, hh))
        print(f"  {name:10s} AUC={auc:.4f} 単勝{tan[0]*100:.1f}% 三連複t4 {t4[0]*100:.1f}% 馬連box {um[0]*100:.1f}% "
              f"反市場好走率{hr*100:.1f}%(base~22%) spvs V15 {sp:.4f} 反市場N{hh}")
    print("\n=== 診断 ===")
    s2, s3r = rows[0], rows[1]
    print(f"  ROI改善: 単勝 {s2[2][0]*100:.1f}→{s3r[2][0]*100:.1f}% / 三連複 {s2[3][0]*100:.1f}→{s3r[3][0]*100:.1f}%")
    print(f"  ★反市場好走率(本物の穴力): {s2[5]*100:.1f}→{s3r[5]*100:.1f}% (22%超えれば本物・未満なら全体ランキング由来)★")
    json.dump({'s2b': {'auc': s2[1], 'tan': s2[2][0], 't4': s2[3][0], 'um': s2[4][0], 'anaba': s2[5]},
               's3_revive': {'auc': s3r[1], 'tan': s3r[2][0], 't4': s3r[3][0], 'um': s3r[4][0], 'anaba': s3r[5]},
               'revive_gain': g3}, open(os.path.join(DATA, 'v16_anaba_s3_revive.json'), 'w'), ensure_ascii=False, indent=2)
    # 候補保存(全データ)
    mask = (y >= 20) & (y <= 25)
    ml = lgb.train(LGB_P, lgb.Dataset(df.loc[mask, s3].values, label=df.loc[mask, 'target'].values), num_boost_round=500)
    mx = xgb.train(XGB_P, xgb.DMatrix(df.loc[mask, s3].values, label=df.loc[mask, 'target'].values), num_boost_round=500)
    pickle.dump({'version': 'v16_anaba_s3_revive_candidate', 'model': ml, 'xgb_model': mx,
                 'ensemble_weights': {'lgb': 0.5, 'xgb': 0.5, 'mlp': 0}, 'features': s3, 'revive_features': REVIVE,
                 'wf_auc_mean': a3, 'is_live': False, 'is_candidate': True, 'parent': 'v16_anaba_s2b_candidate',
                 'trained_on': 'leakfree_v2'},
                gzip.open(os.path.join('models', 'v16_anaba_s3_revive_candidate.pkl.gz'), 'wb'), protocol=4)
    print(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
