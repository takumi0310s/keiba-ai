#!/usr/bin/env python3
"""ROI再検証(クリーン): WF OOF の early-stopping を test年で行うと test覗き見でROIが楽観化する。
本版は ★test非依存の固定round学習★(valid_sets/early_stopなし)で OOF を作り直し、ROIを再測定。
V15単勝ROIが160%→現実的水準に落ちれば test覗き見が主因。本番不変・候補は既存s3で保存済。"""
from __future__ import annotations
import os, sys, gzip, pickle, json, time
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v16_anaba_s2_eval import build_features, ODDS_REMOVE, PROXY_FAMILY, RAW_REPLACE, NEW
from v16_anaba_s1_eval import EVAL_YEARS, LGB_PARAMS, XGB_PARAMS
from v16_anaba_s3_roi import rk, load_payouts, roi_for, anti_market_roi
EXTRA = ['paci_goal_rank', 'paci_goal_diff', 'paci_dochu_rank']
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ROUNDS = 450  # 固定(early-stop点 424-606 の中庸)。test非依存。


def train_predict_clean(df, feats, tr, te):
    Xtr, ytr = df.loc[tr, feats].values, df.loc[tr, 'target'].values
    Xte, yte = df.loc[te, feats].values, df.loc[te, 'target'].values
    m_lgb = lgb.train(LGB_PARAMS, lgb.Dataset(Xtr, label=ytr), num_boost_round=ROUNDS)  # validなし
    m_xgb = xgb.train(XGB_PARAMS, xgb.DMatrix(Xtr, label=ytr), num_boost_round=ROUNDS)   # early_stopなし
    p = 0.5 * m_lgb.predict(Xte) + 0.5 * m_xgb.predict(xgb.DMatrix(Xte))
    return p, roc_auc_score(yte, p)


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

    rows = []; a15 = []; asb = []
    for ty in EVAL_YEARS:
        tr = df['year'] < ty; te = df['year'] == ty
        p15, r15 = train_predict_clean(df, v15, tr, te)
        ps, rs = train_predict_clean(df, s2b, tr, te)
        a15.append(r15); asb.append(rs)
        print(f"[WF {2000+ty}] CLEAN AUC V15={r15:.4f} s2b={rs:.4f}", flush=True)
        sub = df.loc[te, ['_rk', 'horse_num', 'finish', 'target', 'pop_rank']].copy()
        sub['s_v15'] = p15; sub['s_s2b'] = ps
        rows.append(sub)
    oof = pd.concat(rows, ignore_index=True)
    print(f"\n=== CLEAN WF AUC === V15={np.mean(a15):.4f} s2b={np.mean(asb):.4f}")
    pay = load_payouts()
    print("\n=== CLEAN ROI (test非依存・固定round) ===")
    R = [roi_for(oof, pay, 's_s2b', False, 's2b'),
         roi_for(oof, pay, 's_v15', False, 'V15'),
         roi_for(oof, pay, 'pop_rank', True, '人気')]
    print(f"{'戦略':10s}{'単勝':>9s}{'複勝':>9s}{'三連複':>10s}{'N':>7s}")
    for r in R:
        print(f"{r['label']:10s}{r['tansho_roi']*100:8.1f}%{r['fukusho_roi']*100:8.1f}%{r['trio_roi']*100:9.1f}%{r['n']:7d}")
    print("\n=== CLEAN 反市場ピックROI (s2b top1 pop>=6/8) ===")
    for th in [6, 8]:
        a = anti_market_roi(oof, pay, 's_s2b', th)
        pa = np.array(a['place_payouts'])
        print(f"  [pop>={th}] {a['n_anti_races']}R 単勝{a['tansho_roi']*100:.1f}% 複勝{a['fukusho_roi']*100:.1f}% 複勝的中{a['place_hit_rate']*100:.1f}%"
              + (f" 配当中央{np.median(pa):.0f}円" if len(pa) else ""))
    json.dump({'clean_wf_auc': {'v15': float(np.mean(a15)), 's2b': float(np.mean(asb))},
               'roi': {r['label']: {'tansho': r['tansho_roi'], 'fukusho': r['fukusho_roi'], 'trio': r['trio_roi']} for r in R}},
              open(os.path.join(DATA, 'v16_anaba_s3b_clean_roi.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
