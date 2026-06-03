#!/usr/bin/env python3
"""V16 穴特化 候補 c1 — 学習 + 穴発見力検証 (開発3/3 第1歩)。

設計:
  現V16(v16_ability_candidate, 137能力特徴)は paci_jockey_exp_*(人気とr=0.96)が
  支配的で V15(市場)と spearman 0.966 → 穴ピックを約9%しか出さない。
  c1 = V16 と同じ137特徴・同じLGB+XGBだが、★人気デコリレーション sample weight★
  (w = 1 + BETA*ln(pop_rank)) で人気馬を down-weight・穴馬を up-weight。
  → jockey_exp の実効 importance を下げ、穴根拠特徴(dist_apt/ze/脚質/調教/前走不利)を
    longshot 識別で効かせる。特徴は1つも消さない(下げる+足す)。
  pop_rank(基準人気=発走前)は学習 weight のみに使用 = モデル特徴・推論には未使用 = リークなし。

検証(同じ土俵 = WF out-of-fold 2023-2025 ≈ 6,909R):
  - 反市場好走率: model top6 かつ V15(市場) top6 圏外 の馬の3着内率 (base ~22% / 現V16 29.2%)
  - spearman(model, V15) 平均 (現V16 0.966)
  - 穴ピック頻度: 反市場ピックを出すレース割合 (現V16 ~9%)
  - WF AUC (現V16 0.868)

本番 V15/V16 不変・predict_core 不変。出力: models/v16_anaba_c1_candidate.pkl.gz (候補・投票未使用)。
"""
from __future__ import annotations
import os, sys, io, gzip, pickle, json, time, argparse
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

ODDS_FEATURES_REMOVE = [
    'paci_ninki_idx', 'odds_change_rate', 'odds_sharp_drop', 'oz_base_pop_rank',
    'oz_fukusho_base_log', 'oz_tansho_base_log', 'pop_rank_change', 'prev_odds_log',
]
JOCKEY_EXP = ['paci_jockey_exp_wr', 'paci_jockey_exp_3rd']
ANABA_FEATURES = [  # 穴根拠 (既に137内・診断用に gain を追う)
    'jrdb_dist_apt', 'jrdb_running_style', 'jrdb_ze_idm_avg', 'jrdb_ze_ten_avg',
    'jrdb_ze_agari_avg', 'jrdb_oikiri_idx', 'jrdb_prev_interference',
    'jrdb_prev_late_start', 'jrdb_prev_track_bias',
]
LGB_PARAMS = {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
              'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
              'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
              'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'seed': 42}
XGB_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 6,
              'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
              'min_child_weight': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
              'seed': 42, 'tree_method': 'hist', 'verbosity': 0}
EVAL_YEARS = [23, 24, 25]
BETA = float(os.environ.get('C1_BETA', '0.6'))  # 人気デコリレーション強度
TOPK = 6


def load_cache():
    obj = pickle.load(gzip.open(os.path.join(DATA, '_v15_optuna_df_cache.pkl.gz'), 'rb'))
    df = obj['df']; feats = obj['features']
    if 'target' not in df.columns:
        df['target'] = (df['finish'] <= 3).astype(int)
    df['rid'] = df['race_id_unique'].astype(str)
    # pop_rank (基準人気=発走前) — weight + 市場判定の補助。欠損は最下位扱い。
    pr = pd.to_numeric(df.get('oz_base_pop_rank'), errors='coerce')
    df['pop_rank'] = pr.fillna(df['num_horses_val']).clip(lower=1)
    return df, feats


def train_predict(df, features, train_mask, test_mask, weight=None):
    """LGB+XGB を学習し test の ensemble スコアを返す。weight は train 行の sample weight。"""
    Xtr = df.loc[train_mask, features].values
    ytr = df.loc[train_mask, 'target'].values
    Xte = df.loc[test_mask, features].values
    yte = df.loc[test_mask, 'target'].values
    wtr = None if weight is None else weight[train_mask.values]
    dt = lgb.Dataset(Xtr, label=ytr, weight=wtr)
    dv = lgb.Dataset(Xte, label=yte, reference=dt)
    m_lgb = lgb.train(LGB_PARAMS, dt, num_boost_round=1000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    p_lgb = m_lgb.predict(Xte)
    dxtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
    dxte = xgb.DMatrix(Xte, label=yte)
    m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=1000, evals=[(dxte, 'v')],
                      early_stopping_rounds=50, verbose_eval=False)
    p_xgb = m_xgb.predict(dxte)
    p = 0.5 * p_lgb + 0.5 * p_xgb
    return p, roc_auc_score(yte, p), m_lgb, m_xgb


def compute_oof(df, v15_feats, v16_feats):
    """WF OOF スコア (V15 / V16 / c1) を 2023-2025 で生成。"""
    weight_c1 = (1.0 + BETA * np.log(df['pop_rank'].values)).astype(float)
    rows = []
    aucs = {'v15': [], 'v16': [], 'c1': []}
    gains_v16 = {}; gains_c1 = {}
    for ty in EVAL_YEARS:
        tr = df['year'] < ty
        te = df['year'] == ty
        print(f"\n[WF {2000+ty}] train={tr.sum()} test={te.sum()}", flush=True)
        p15, a15, _, _ = train_predict(df, v15_feats, tr, te)
        p16, a16, m16, _ = train_predict(df, v16_feats, tr, te)
        pc1, ac1, mc1, _ = train_predict(df, v16_feats, tr, te, weight=weight_c1)
        aucs['v15'].append(a15); aucs['v16'].append(a16); aucs['c1'].append(ac1)
        print(f"  AUC  V15={a15:.4f}  V16={a16:.4f}  c1={ac1:.4f}", flush=True)
        # gain (last fold で代表)
        for f, g in zip(v16_feats, m16.feature_importance(importance_type='gain')):
            gains_v16[f] = gains_v16.get(f, 0) + g
        for f, g in zip(v16_feats, mc1.feature_importance(importance_type='gain')):
            gains_c1[f] = gains_c1.get(f, 0) + g
        sub = df.loc[te, ['rid', 'target', 'pop_rank']].copy()
        sub['s_v15'] = p15; sub['s_v16'] = p16; sub['s_c1'] = pc1
        sub['year'] = ty
        rows.append(sub)
    oof = pd.concat(rows, ignore_index=True)
    return oof, aucs, gains_v16, gains_c1


def gain_pct(gains, names):
    tot = sum(gains.values()) or 1.0
    return {n: 100.0 * gains.get(n, 0) / tot for n in names}


def anaba_metrics(oof, score_col, market_col='s_v15', topk=TOPK):
    """反市場好走率 / spearman / 穴頻度 / base率 を算出。
    反市場ピック = score_col top-k かつ market_col top-k 圏外 の馬。
    """
    base = oof['target'].mean()
    hit_n = hit_h = 0          # 反市場ピックの 3着内 / 総数
    races_with_pick = 0
    spear = []
    grp = oof.groupby('rid')
    nrace = 0
    for rid, g in grp:
        if len(g) < 4:
            continue
        nrace += 1
        s = g[score_col].values; m = g[market_col].values
        t = g['target'].values
        order_s = np.argsort(-s)
        order_m = np.argsort(-m)
        top_s = set(order_s[:topk])
        top_m = set(order_m[:topk])
        anti = [i for i in top_s if i not in top_m]   # model好む / 市場好まない
        if anti:
            races_with_pick += 1
            for i in anti:
                hit_h += 1
                hit_n += int(t[i])
        # spearman(model, market)
        if len(s) >= 3:
            r, _ = spearmanr(s, m)
            if not np.isnan(r):
                spear.append(r)
    return {
        'base_top3': float(base),
        'anti_market_hit_rate': float(hit_n / hit_h) if hit_h else float('nan'),
        'anti_market_picks': int(hit_h),
        'pick_freq': float(races_with_pick / nrace) if nrace else float('nan'),
        'spearman_vs_v15': float(np.mean(spear)) if spear else float('nan'),
        'n_races': int(nrace),
    }


def main():
    t0 = time.time()
    df, v15_feats = load_cache()
    v16_feats = [f for f in v15_feats if f not in ODDS_FEATURES_REMOVE]
    for f in set(v15_feats) | set(v16_feats):
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    print(f"V15={len(v15_feats)} V16={len(v16_feats)} feats. BETA={BETA}", flush=True)

    oof, aucs, gv16, gc1 = compute_oof(df, v15_feats, v16_feats)
    oof.to_parquet(os.path.join(DATA, 'v16_anaba_c1_oof.parquet'))

    wf = {k: float(np.mean(v)) for k, v in aucs.items()}
    print(f"\n=== WF AUC (2023-25 mean) ===  V15={wf['v15']:.4f}  V16={wf['v16']:.4f}  c1={wf['c1']:.4f}")

    print("\n=== gain% (jockey_exp / anaba 抜粋) ===")
    g16 = gain_pct(gv16, JOCKEY_EXP + ANABA_FEATURES)
    gc = gain_pct(gc1, JOCKEY_EXP + ANABA_FEATURES)
    je16 = sum(g16[f] for f in JOCKEY_EXP); jec1 = sum(gc[f] for f in JOCKEY_EXP)
    an16 = sum(g16[f] for f in ANABA_FEATURES); anc1 = sum(gc[f] for f in ANABA_FEATURES)
    print(f"  jockey_exp 合計 gain%: V16={je16:.2f}  c1={jec1:.2f}")
    print(f"  anaba根拠 合計 gain%: V16={an16:.2f}  c1={anc1:.2f}")
    for f in JOCKEY_EXP + ANABA_FEATURES:
        print(f"    {f:28s} V16={g16[f]:5.2f}%  c1={gc[f]:5.2f}%")

    print("\n=== 穴発見力 (反市場=top6 & V15 top6 圏外) ===")
    m_v16 = anaba_metrics(oof, 's_v16')
    m_c1 = anaba_metrics(oof, 's_c1')
    print(f"  base 3着内率: {m_v16['base_top3']*100:.1f}% (理論≈22%)")
    print(f"  {'metric':24s} {'V16(現)':>10s} {'c1':>10s}")
    print(f"  {'反市場好走率':24s} {m_v16['anti_market_hit_rate']*100:9.1f}% {m_c1['anti_market_hit_rate']*100:9.1f}%")
    print(f"  {'反市場ピック数':24s} {m_v16['anti_market_picks']:10d} {m_c1['anti_market_picks']:10d}")
    print(f"  {'穴ピック頻度':24s} {m_v16['pick_freq']*100:9.1f}% {m_c1['pick_freq']*100:9.1f}%")
    print(f"  {'spearman vs V15':24s} {m_v16['spearman_vs_v15']:10.4f} {m_c1['spearman_vs_v15']:10.4f}")
    print(f"  {'n_races':24s} {m_v16['n_races']:10d} {m_c1['n_races']:10d}")

    # ===== c1 候補モデル本体を全データ(2020-25)で学習し保存 =====
    print("\n=== c1 候補モデル 全データ学習 + 保存 ===", flush=True)
    mask = (df['year'] >= 20) & (df['year'] <= 25)
    w = (1.0 + BETA * np.log(df.loc[mask, 'pop_rank'].values)).astype(float)
    Xtr = df.loc[mask, v16_feats].values; ytr = df.loc[mask, 'target'].values
    dt = lgb.Dataset(Xtr, label=ytr, weight=w)
    m_lgb = lgb.train(LGB_PARAMS, dt, num_boost_round=500)
    dxtr = xgb.DMatrix(Xtr, label=ytr, weight=w)
    m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=500, evals=[(dxtr, 't')], verbose_eval=False)
    tag = 'candidate' if abs(BETA - 0.6) < 1e-9 else f'b{BETA:g}'
    out = os.path.join(BASE, 'models', f'v16_anaba_c1_{tag}.pkl.gz')
    pkl = {
        'version': 'v16_anaba_c1_candidate',
        'description': 'V16 137 ability features + popularity-decorrelation sample weight (w=1+BETA*ln(pop_rank)). Candidate/paper only.',
        'model': m_lgb, 'xgb_model': m_xgb,
        'ensemble_weights': {'lgb': 0.5, 'xgb': 0.5, 'mlp': 0},
        'features': v16_feats, 'n_features': len(v16_feats),
        'beta': BETA, 'sample_weight': 'w = 1 + BETA*ln(pop_rank)  [pop_rank=oz_base_pop_rank, pre-race, train-only]',
        'wf_auc_mean': wf['c1'], 'wf_auc_v16': wf['v16'], 'wf_auc_v15': wf['v15'],
        'anaba_metrics_c1': m_c1, 'anaba_metrics_v16': m_v16,
        'leak_free': True, 'is_live': False, 'is_candidate': True,
        'parent': 'v16_ability_candidate',
    }
    with gzip.open(out, 'wb') as f:
        pickle.dump(pkl, f, protocol=4)
    print(f"  saved: {out}")

    summary = {'beta': BETA, 'wf_auc': wf, 'metrics_v16': m_v16, 'metrics_c1': m_c1,
               'jockey_exp_gain_pct': {'v16': je16, 'c1': jec1},
               'anaba_gain_pct': {'v16': an16, 'c1': anc1}}
    json.dump(summary, open(os.path.join(DATA, f'v16_anaba_c1_summary_b{BETA:g}.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
