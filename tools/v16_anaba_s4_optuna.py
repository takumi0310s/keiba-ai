#!/usr/bin/env python3
"""大規模③: s2b のハイパラを Optuna で最適化(★ROIで最適化・AUCで最適化しない★)+ leak-free v2 厳密検証。

罠回避:
  - ★leak-free v2 cache のみ★(リーク版禁止)。
  - ★目的関数 = WF test期間の複合ROI(0.4単勝+0.3複勝top1+0.3三連複top4box)。AUCで最適化しない(人気をなぞる方向を避ける)★。
  - 過学習防止: Optuna は train=2020-2023 / test=2024 で選ぶ。 ★2025 は Optuna 未使用 = held-out で最終検証★。
本番V15/V16不変・投票未使用。
"""
from __future__ import annotations
import os, sys, gzip, pickle, json, time, itertools
if sys.platform == "win32": sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb, optuna
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
sys.path.insert(0, os.path.abspath('tools'))
from v16_anaba_s2_eval import build_features, ODDS_REMOVE, PROXY_FAMILY, RAW_REPLACE, NEW
from v16_leakfree_roi_grid import load_payouts, S_tan, S_fuku1, S_trio4, S_umaren_t3box, LGB_P, XGB_P
EXTRA = ['paci_goal_rank', 'paci_goal_diff', 'paci_dochu_rank']
DATA = 'data'
optuna.logging.set_verbosity(optuna.logging.WARNING)


def roi(ev, scol, pay, fn):
    ret = stake = n = 0
    for k, g in ev.groupby('_rk'):
        if k not in pay or len(g) < 5: continue
        o = [int(x) for x in g.sort_values(scol, ascending=False)['horse_num'].tolist()]
        r, pts = fn(o, pay[k]); ret += r; stake += 100 * pts
    return ret / stake if stake else 0


def composite(ev, scol, pay):
    return 0.4 * roi(ev, scol, pay, S_tan) + 0.3 * roi(ev, scol, pay, S_fuku1) + 0.3 * roi(ev, scol, pay, S_trio4)


def main():
    t0 = time.time()
    N_TRIALS = int(os.environ.get('N_TRIALS', '25'))
    obj = pickle.load(gzip.open(os.path.join(DATA, '_v15_optuna_df_cache_leakfree_v2.pkl.gz'), 'rb'))
    df = obj['df']; v15 = obj['features']
    if 'target' not in df.columns: df['target'] = (df['finish'] <= 3).astype(int)
    df = build_features(df)
    df['_rk'] = [f"{int(a)}_{b}_{int(c)}_{int(e)}_{int(f)}" for a, b, c, e, f in
                 zip(df['date_num'], df['course'], df['kai'], df['nichi'], df['race_num'])]
    v16 = [f for f in v15 if f not in ODDS_REMOVE]
    s2b = [f for f in v16 if f not in (PROXY_FAMILY + EXTRA + RAW_REPLACE)] + NEW
    for f in set(s2b): df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    y = pd.to_numeric(df['year'], errors='coerce')
    pay = load_payouts()
    # Optuna分割: train 20-23 / valid 24 (選択) / 25 = held-out(未使用)
    tr_o = (y >= 20) & (y < 24); va_o = (y == 24)
    Xtr = df.loc[tr_o, s2b].values; ytr = df.loc[tr_o, 'target'].values
    ev24 = df[va_o].copy()
    print(f"Optuna: train20-23({tr_o.sum()}) valid24({va_o.sum()}) / 25=held-out。 目的=複合ROI(単勝.4+複勝.3+三連複.3)。 trials={N_TRIALS}", flush=True)

    def objective(trial):
        p = {'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'seed': 42,
             'num_leaves': trial.suggest_int('num_leaves', 15, 127),
             'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.12, log=True),
             'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
             'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
             'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
             'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
             'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
             'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)}
        nbr = trial.suggest_int('num_boost_round', 200, 600)
        m = lgb.train(p, lgb.Dataset(Xtr, label=ytr), num_boost_round=nbr)
        ev24['_s'] = m.predict(ev24[s2b].values)
        return composite(ev24, '_s', pay)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)
    bp = study.best_params
    vals = [t.value for t in study.trials if t.value is not None]
    print(f"\nOptuna完了: best valid24複合ROI={study.best_value:.4f}")
    print(f"  trial分散: mean={np.mean(vals):.4f} std={np.std(vals):.4f} min={min(vals):.4f} max={max(vals):.4f}")
    print(f"  best params: {bp}")
    # baseline(s2b現行param)の valid24 複合ROI(比較用)
    mb = lgb.train(LGB_P, lgb.Dataset(Xtr, label=ytr), num_boost_round=420)
    ev24['_sb'] = mb.predict(ev24[s2b].values)
    base24 = composite(ev24, '_sb', pay)
    print(f"  s2b現行param valid24複合ROI={base24:.4f} → optuna {study.best_value:.4f} (差{study.best_value-base24:+.4f})")
    # 過学習チェック: best params の train複合ROI vs valid24
    mbest = lgb.train({**bp, 'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'seed': 42},
                      lgb.Dataset(Xtr, label=ytr), num_boost_round=bp['num_boost_round'])
    tr_ev = df[tr_o].copy(); tr_ev['_s'] = mbest.predict(tr_ev[s2b].values)
    tr_roi = composite(tr_ev, '_s', pay)
    print(f"  過学習チェック: train複合ROI={tr_roi:.4f} vs valid24={study.best_value:.4f} (乖離大なら過学習)")

    # ===== 最終: s2b(現行param) vs s4_optuna(best param) を leak-free v2 WF(2023-25)+ held-out 2025 ===
    def wf_oof(params_lgb, nbr):
        s = pd.Series(index=df.index, dtype=float); aucs = []
        for ty in [23, 24, 25]:
            tr = y < ty; te = y == ty
            ml = lgb.train({**params_lgb, 'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'seed': 42},
                           lgb.Dataset(df.loc[tr, s2b].values, label=df.loc[tr, 'target'].values), num_boost_round=nbr)
            mx = xgb.train(XGB_P, xgb.DMatrix(df.loc[tr, s2b].values, label=df.loc[tr, 'target'].values), num_boost_round=nbr)
            s.loc[te] = 0.5 * ml.predict(df.loc[te, s2b].values) + 0.5 * mx.predict(xgb.DMatrix(df.loc[te, s2b].values))
            aucs.append(roc_auc_score(df.loc[te, 'target'].values, s[te]))
        return s, float(np.mean(aucs))
    df['s_s2b'], a_s2b = wf_oof({k: v for k, v in LGB_P.items() if k not in ('objective', 'metric', 'verbose', 'seed')}, 420)
    df['s_s4'], a_s4 = wf_oof({k: v for k, v in bp.items() if k != 'num_boost_round'}, bp['num_boost_round'])

    def anaba(ev, scol):
        hit = h = 0; sp = []
        for k, g in ev.groupby('_rk'):
            if len(g) < 4: continue
            sc = g[scol].values; m = g['s_s2b'].values; t = g['target'].values
            for i in [i for i in set(np.argsort(-sc)[:6]) if i not in set(np.argsort(-m)[:6])]: h += 1; hit += int(t[i])
            if len(sc) >= 3:
                r, _ = spearmanr(sc, g['s_s2b'].values)
                if not np.isnan(r): sp.append(r)
        return (hit/h if h else float('nan'), float(np.mean(sp)))

    print("\n=== leak-free v2 最終比較 (s2b現行 vs s4_optuna) ===")
    for label, sc, auc in [('s2b', 's_s2b', a_s2b), ('s4_optuna', 's_s4', a_s4)]:
        for yr, tag in [(None, '全期間2023-25'), (25, '★held-out 2025(Optuna未使用)★')]:
            ev = df[y >= 23] if yr is None else df[y == yr]
            tan = roi(ev, sc, pay, S_tan); fk = roi(ev, sc, pay, S_fuku1); t4 = roi(ev, sc, pay, S_trio4); um = roi(ev, sc, pay, S_umaren_t3box)
            extra = f" AUC={auc:.4f}" if yr is None else ""
            hr = anaba(ev, sc)[0] if yr is None else None
            print(f"  {label:10s}[{tag:28s}] 単勝{tan*100:6.1f}% 複勝{fk*100:6.1f}% 三連複t4{t4*100:7.1f}% 馬連box{um*100:6.1f}%"
                  + (f" 反市場{hr*100:.1f}%{extra}" if yr is None else ""))
    json.dump({'best_params': bp, 'best_valid24': study.best_value, 'base_valid24': base24,
               'trial_std': float(np.std(vals)), 'train_roi': tr_roi, 'auc_s2b': a_s2b, 'auc_s4': a_s4},
              open(os.path.join(DATA, 'v16_anaba_s4_optuna.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
