#!/usr/bin/env python3
"""リーク監査 ステップ4: 集計族A/前走族B を ablation して、top1勝率の人気依存が正常化するか + leak-free ROI。
本番V15相当(145)と s2b で、A除去/AB除去版を test非依存(固定round)で WF-OOF 学習し、
①勝率vs人気の曲線(正常=人気薄ほど急落) ②単勝/複勝ROI を比較。リーク源を特定し leak-free 真値を出す。
本番V15/V16 .pkl.gz 不変・候補/評価のみ。"""
from __future__ import annotations
import os, sys, gzip, pickle, json, time
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v16_anaba_s2_eval import build_features, ODDS_REMOVE, PROXY_FAMILY, RAW_REPLACE, NEW
from v16_anaba_s1_eval import EVAL_YEARS, LGB_PARAMS, XGB_PARAMS
from v16_anaba_s3_roi import rk, load_payouts
EXTRA = ['paci_goal_rank', 'paci_goal_diff', 'paci_dochu_rank']
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ROUNDS = 450

# Group A: レート集計(非expandingなら当該/未来を含むリーク疑い)
GROUP_A = ['jockey_wr_calc', 'jockey_course_wr_calc', 'jockey_surface_wr', 'horse_career_wr',
           'horse_career_top3r', 'horse_surface_top3r', 'horse_dist_top3r', 'trainer_top3_calc',
           'sire_surface_wr', 'sire_dist_wr', 'bms_surface_wr', 'frame_course_dist_wr',
           'sib_top3_rate_exp', 'sib_shinba_wr_exp', 'sire_shinba_top3r_exp',
           'horse_career_races', 'sib_total_races_exp', 'sib_total_offspring_exp']
# Group B: 前走/近走 form(当該レース含む off-by-one なら リーク)
GROUP_B = ['prev_finish', 'prev2_finish', 'prev3_finish', 'avg_finish_3r', 'best_finish_3r',
           'top3_count_3r', 'finish_trend', 'avg_last3f_3r', 'prev2_last3f', 'prev_last3f',
           'prev_pass4', 'prev_prize']


def tp_clean(df, feats, tr, te):
    Xtr, ytr = df.loc[tr, feats].values, df.loc[tr, 'target'].values
    Xte, yte = df.loc[te, feats].values, df.loc[te, 'target'].values
    m_lgb = lgb.train(LGB_PARAMS, lgb.Dataset(Xtr, label=ytr), num_boost_round=ROUNDS)
    m_xgb = xgb.train(XGB_PARAMS, xgb.DMatrix(Xtr, label=ytr), num_boost_round=ROUNDS)
    p = 0.5 * m_lgb.predict(Xte) + 0.5 * m_xgb.predict(xgb.DMatrix(Xte))
    return p, roc_auc_score(yte, p)


def oof_for(df, feats):
    rows = []; aucs = []
    for ty in EVAL_YEARS:
        tr = df['year'] < ty; te = df['year'] == ty
        p, a = tp_clean(df, feats, tr, te); aucs.append(a)
        sub = df.loc[te, ['_rk', 'horse_num', 'finish', 'pop_rank']].copy(); sub['s'] = p
        rows.append(sub)
    return pd.concat(rows, ignore_index=True), float(np.mean(aucs))


def diag(oof, pay, label):
    rows = []
    for k, g in oof.groupby('_rk'):
        if k not in pay or len(g) < 4: continue
        t1 = g.sort_values('s', ascending=False).iloc[0]
        won = int(t1['finish']) == 1
        rows.append((int(t1['pop_rank']), won, pay[k]['tansho_pay'] if won else 0,
                     pay[k]['fuku'].get(int(t1['horse_num']), 0) if int(t1['finish']) <= 3 else 0))
    r = pd.DataFrame(rows, columns=['pop', 'won', 'tan', 'fuk'])
    tiers = []
    for lo, hi in [(1, 1), (2, 3), (4, 5), (6, 9), (10, 18)]:
        s = r[(r['pop'] >= lo) & (r['pop'] <= hi)]
        tiers.append(f"{lo}-{hi}:{s['won'].mean()*100:.0f}%(n{len(s)})" if len(s) else "")
    n = len(r)
    print(f"  {label:16s} AUC=?  全体勝率{r['won'].mean()*100:4.1f}%  単勝{r['tan'].sum()/(100*n)*100:5.1f}% 複勝{r['fuk'].sum()/(100*n)*100:5.1f}%  人気別[{' '.join(tiers)}]")
    return r


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
    s2b = [f for f in v16 if f not in (PROXY_FAMILY + EXTRA + RAW_REPLACE)] + NEW
    for f in set(v15) | set(s2b):
        if f in df.columns: df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    pay = load_payouts()
    A = [f for f in GROUP_A if f in v15]; B = [f for f in GROUP_B if f in v15]
    print(f"Group A(集計) {len(A)}件: {A}")
    print(f"Group B(前走) {len(B)}件: {B}")
    print("\n各構成: 勝率vs人気曲線(正常=人気薄ほど急落)+ROI。市場本命勝率=31.7%が健全な天井。\n")

    configs = [
        ('V15_full', v15),
        ('V15_noA', [f for f in v15 if f not in A]),
        ('V15_noAB', [f for f in v15 if f not in (A + B)]),
        ('s2b_full', s2b),
        ('s2b_noA', [f for f in s2b if f not in A]),
        ('s2b_noAB', [f for f in s2b if f not in (A + B)]),
    ]
    res = {}
    for name, feats in configs:
        oof, auc = oof_for(df, feats)
        print(f"[{name}] feats={len(feats)} AUC={auc:.4f}", flush=True)
        r = diag(oof, pay, name)
        # 人気依存度 = 人気1番勝率 - 人気10-18勝率 (大きいほど健全)
        p1 = r[r['pop'] == 1]['won'].mean(); pl = r[r['pop'] >= 10]['won'].mean()
        res[name] = {'auc': auc, 'overall_win': float(r['won'].mean()),
                     'win_p1': float(p1), 'win_p10plus': float(pl),
                     'pop_dependence': float(p1 - pl),
                     'tansho_roi': float(r['tan'].sum() / (100 * len(r))),
                     'fukusho_roi': float(r['fuk'].sum() / (100 * len(r)))}
    json.dump(res, open(os.path.join(DATA, 'v16_anaba_s4_leakaudit.json'), 'w'), ensure_ascii=False, indent=2)
    print("\n=== 人気依存度(p1勝率 − p10+勝率。健全なら大きい・リークなら~0) ===")
    for n, d in res.items():
        print(f"  {n:12s} 依存度={d['pop_dependence']*100:+5.1f}pt (p1 {d['win_p1']*100:.0f}% / p10+ {d['win_p10plus']*100:.0f}%)  単勝ROI {d['tansho_roi']*100:.0f}%")
    print(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
