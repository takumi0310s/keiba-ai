#!/usr/bin/env python3
"""V16 ステップ3 — s2b(人気代理族13件除去)を正式候補化 + ★ROI検証★。

s2b = V16能力 − 人気代理族13(jockey_exp×2/印4/cid/ls/training_idx/stable_idx/goal_rank/goal_diff/dochu_rank)
      + レース相対・交互特徴(front_advantage等)。one-hot保持。
WF out-of-fold(2023-25, リークなし)予測 × JRA公式払戻(jra_payouts.csv)で ROI を測る。
予測は発走前情報のみ。配当は結果払戻(ROI計算専用=予測未使用)。本番V15/V16不変・投票未使用。

★レースキー修正★: race_id_unique は course を含まず別場の同R番号を併合していた(2314 vs 正3455)。
本スクリプトは date_course_kai_nichi_racenum で正しくグループ化(払戻結合キーと一致)。
"""
from __future__ import annotations
import os, sys, gzip, pickle, json, time
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v16_anaba_s2_eval import build_features, ODDS_REMOVE, PROXY_FAMILY, RAW_REPLACE, NEW
from v16_anaba_s1_eval import train_predict, EVAL_YEARS, LGB_PARAMS, XGB_PARAMS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA = os.path.join(BASE, "data")
EXTRA = ['paci_goal_rank', 'paci_goal_diff', 'paci_dochu_rank']


def rk(date, course, kai, nichi, rnum):
    return f"{int(date)}_{course}_{int(kai)}_{int(nichi)}_{int(rnum)}"


def load_payouts():
    p = pd.read_csv(os.path.join(DATA, 'jra_payouts.csv'), low_memory=False)
    p = p[(p['race_date'] >= 20230101) & (p['race_date'] <= 20251231)]
    out = {}
    for _, r in p.iterrows():
        key = rk(r['race_date'], r['course'], r['kai'], r['nichi'], r['race_num'])
        def to_i(x):
            try: return int(float(x))
            except Exception: return 0
        # 複勝 nums→payout
        fuku = {}
        try:
            ns = str(r['fukusho_nums']).replace(' ', '').split('/')
            ps = str(r['fukusho_payouts']).replace(' ', '').split('/')
            for n, pay in zip(ns, ps):
                fuku[to_i(n)] = to_i(pay)
        except Exception: pass
        # 三連複 set→payout
        trio = None
        try:
            tn = str(r['trio_nums']).replace(' ', '').split('-')
            trio = (frozenset(to_i(x) for x in tn), to_i(r['trio_payout']))
        except Exception: pass
        out[key] = {'tansho_pay': to_i(r['tansho_payout']), 'fuku': fuku, 'trio': trio}
    return out


def compute_oof(df, v15, s2b):
    rows = []; aucs = {'v15': [], 's2b': []}
    for ty in EVAL_YEARS:
        tr = df['year'] < ty; te = df['year'] == ty
        print(f"[WF {2000+ty}] train={tr.sum()} test={te.sum()}", flush=True)
        p15, a15, _, _ = train_predict(df, v15, tr, te)
        ps, asb, _, _ = train_predict(df, s2b, tr, te)
        aucs['v15'].append(a15); aucs['s2b'].append(asb)
        print(f"  AUC V15={a15:.4f} s2b={asb:.4f}", flush=True)
        sub = df.loc[te, ['_rk', 'horse_num', 'finish', 'target', 'pop_rank']].copy()
        sub['s_v15'] = p15; sub['s_s2b'] = ps
        rows.append(sub)
    return pd.concat(rows, ignore_index=True), {k: float(np.mean(v)) for k, v in aucs.items()}


def roi_for(oof, pay, score_col, ascending=False, label=''):
    """score_col 降順(ascending=Falseでスコア最大が本命) top1/top3 で 単勝/複勝/三連複 ROI。
    market は pop_rank 昇順(ascending=True)。"""
    tan_ret = tan_n = fuk_ret = fuk_n = trio_ret = trio_n = 0
    for key, g in oof.groupby('_rk'):
        if key not in pay or len(g) < 4: continue
        pm = pay[key]
        gg = g.sort_values(score_col, ascending=ascending)
        top1 = gg.iloc[0]; top3 = gg.iloc[:3]
        # 単勝 top1
        tan_n += 1
        if int(top1['finish']) == 1: tan_ret += pm['tansho_pay']
        # 複勝 top1
        fuk_n += 1
        hn = int(top1['horse_num'])
        if hn in pm['fuku']: fuk_ret += pm['fuku'][hn]
        # 三連複 top3 box (1点)
        if pm['trio']:
            trio_n += 1
            tset = frozenset(int(x) for x in top3['horse_num'])
            if tset == pm['trio'][0]: trio_ret += pm['trio'][1]
    return {'label': label,
            'tansho_roi': tan_ret / (100 * tan_n) if tan_n else 0,
            'fukusho_roi': fuk_ret / (100 * fuk_n) if fuk_n else 0,
            'trio_roi': trio_ret / (100 * trio_n) if trio_n else 0, 'n': tan_n}


def anti_market_roi(oof, pay, score_col, pop_thresh=6):
    """s2b top1 が market(pop_rank)で pop_thresh 圏外 のレースだけ、その馬の 単勝/複勝 ROI。"""
    tan_ret = fuk_ret = n = hit_place = 0; payouts = []
    for key, g in oof.groupby('_rk'):
        if key not in pay or len(g) < 4: continue
        top1 = g.sort_values(score_col, ascending=False).iloc[0]
        if top1['pop_rank'] < pop_thresh: continue   # 市場本命寄りは除外=穴ピックのみ
        n += 1; pm = pay[key]; hn = int(top1['horse_num'])
        if int(top1['finish']) == 1: tan_ret += pm['tansho_pay']
        if hn in pm['fuku']:
            fuk_ret += pm['fuku'][hn]; hit_place += 1; payouts.append(pm['fuku'][hn])
    return {'n_anti_races': n, 'tansho_roi': tan_ret / (100 * n) if n else 0,
            'fukusho_roi': fuk_ret / (100 * n) if n else 0,
            'place_hit_rate': hit_place / n if n else 0,
            'place_payouts': payouts}


def anaba_quick(oof, score_col, market='s_v15', topk=6):
    hit = h = nr = 0; spear = []
    for key, g in oof.groupby('_rk'):
        if len(g) < 4: continue
        nr += 1
        s = g[score_col].values; m = g[market].values; t = g['target'].values
        anti = [i for i in set(np.argsort(-s)[:topk]) if i not in set(np.argsort(-m)[:topk])]
        for i in anti: h += 1; hit += int(t[i])
        if len(s) >= 3:
            r, _ = spearmanr(s, m)
            if not np.isnan(r): spear.append(r)
    return hit / h if h else float('nan'), float(np.mean(spear)), nr


def main():
    t0 = time.time()
    obj = pickle.load(gzip.open(os.path.join(DATA, '_v15_optuna_df_cache.pkl.gz'), 'rb'))
    df = obj['df']; v15 = obj['features']
    if 'target' not in df.columns: df['target'] = (df['finish'] <= 3).astype(int)
    df = build_features(df)
    df['pop_rank'] = pd.to_numeric(df['oz_base_pop_rank'], errors='coerce').fillna(df['num_horses_val'])
    df['_rk'] = [rk(d, c, k, n, r) for d, c, k, n, r in
                 zip(df['date_num'], df['course'], df['kai'], df['nichi'], df['race_num'])]
    v16 = [f for f in v15 if f not in ODDS_REMOVE]
    fam = PROXY_FAMILY + EXTRA
    s2b = [f for f in v16 if f not in (fam + RAW_REPLACE)] + NEW
    for f in set(v15) | set(s2b):
        if f in df.columns: df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    print(f"s2b={len(s2b)} feats, 除去族={len(fam)}件")

    oof, wf = compute_oof(df, v15, s2b)
    oof.to_parquet(os.path.join(DATA, 'v16_anaba_s3_oof.parquet'))
    print(f"\n=== WF AUC === V15={wf['v15']:.4f} s2b={wf['s2b']:.4f}")
    hr, sp, nr = anaba_quick(oof, 's_s2b')
    hr16 = None
    print(f"=== 穴(正しいレースキー) s2b: 反市場好走率={hr*100:.1f}% spearman={sp:.4f} ({nr}R) ===")

    pay = load_payouts()
    print(f"払戻ロード: {len(pay)}R\n")

    print("=== ROI (全レース・100円/点) ===")
    R = [roi_for(oof, pay, 's_s2b', False, 's2b'),
         roi_for(oof, pay, 's_v15', False, 'V15(市場追随)'),
         roi_for(oof, pay, 'pop_rank', True, '人気(オッズ上位)')]
    print(f"{'戦略':16s}{'単勝':>9s}{'複勝':>9s}{'三連複box':>11s}{'N':>7s}")
    for r in R:
        print(f"{r['label']:16s}{r['tansho_roi']*100:8.1f}%{r['fukusho_roi']*100:8.1f}%{r['trio_roi']*100:10.1f}%{r['n']:7d}")

    print("\n=== ★反市場ピック ROI (s2b top1 が pop>=6 圏外のレースのみ) ★ ===")
    for th in [6, 8]:
        a = anti_market_roi(oof, pay, 's_s2b', th)
        pa = np.array(a['place_payouts'])
        print(f"  [pop>={th}] {a['n_anti_races']}R: 単勝ROI {a['tansho_roi']*100:.1f}% / 複勝ROI {a['fukusho_roi']*100:.1f}% / 複勝的中率 {a['place_hit_rate']*100:.1f}%")
        if len(pa):
            print(f"      複勝的中時 配当: 中央値{np.median(pa):.0f}円 平均{pa.mean():.0f}円 最大{pa.max():.0f}円 (>=300円の割合 {(pa>=300).mean()*100:.0f}%)")

    print("\n=== レース選択: s2b top1 の人気別 単勝/複勝ROI ===")
    for lo, hi in [(1, 1), (2, 3), (4, 5), (6, 9), (10, 18)]:
        sel = []
        for key, g in oof.groupby('_rk'):
            if key not in pay or len(g) < 4: continue
            t1 = g.sort_values('s_s2b', ascending=False).iloc[0]
            if lo <= t1['pop_rank'] <= hi: sel.append((key, t1))
        if not sel: continue
        tr = sum(pay[k]['tansho_pay'] for k, t in sel if int(t['finish']) == 1)
        fr = sum(pay[k]['fuku'].get(int(t['horse_num']), 0) for k, t in sel)
        n = len(sel)
        print(f"  s2b top1が人気{lo}-{hi}: {n}R 単勝{tr/(100*n)*100:5.1f}% 複勝{fr/(100*n)*100:5.1f}%")

    # s2b 正式候補 保存
    print("\n=== s2b 正式候補 全データ学習+保存 ===", flush=True)
    mask = (df['year'] >= 20) & (df['year'] <= 25)
    X, y = df.loc[mask, s2b].values, df.loc[mask, 'target'].values
    m_lgb = lgb.train(LGB_PARAMS, lgb.Dataset(X, label=y), num_boost_round=500)
    m_xgb = xgb.train(XGB_PARAMS, xgb.DMatrix(X, label=y), num_boost_round=500,
                      evals=[(xgb.DMatrix(X, label=y), 't')], verbose_eval=False)
    out = os.path.join(BASE, 'models', 'v16_anaba_s2b_candidate.pkl.gz')
    pickle.dump({'version': 'v16_anaba_s2b_candidate',
                 'description': 'V16 ability minus popularity-proxy family(13) + race-relative/interaction features. Candidate/paper only. NOT for live voting.',
                 'model': m_lgb, 'xgb_model': m_xgb, 'ensemble_weights': {'lgb': 0.5, 'xgb': 0.5, 'mlp': 0},
                 'features': s2b, 'n_features': len(s2b), 'removed_proxy_family': fam,
                 'wf_auc_mean': wf['s2b'], 'wf_auc_v15': wf['v15'],
                 'anaba_hit_rate': hr, 'spearman_vs_v15': sp, 'roi': {r['label']: r for r in R},
                 'leak_free': True, 'is_live': False, 'is_candidate': True, 'parent': 'v16_anaba_s2_candidate'},
                gzip.open(out, 'wb'), protocol=4)
    print(f"  saved: {out}")
    json.dump({'wf_auc': wf, 'anaba_hit_rate': hr, 'spearman': sp,
               'roi_all': {r['label']: {'tansho': r['tansho_roi'], 'fukusho': r['fukusho_roi'], 'trio': r['trio_roi']} for r in R}},
              open(os.path.join(DATA, 'v16_anaba_s3_roi_summary.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
