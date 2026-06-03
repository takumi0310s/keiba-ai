#!/usr/bin/env python3
"""V16 作り直し ステップ1 — 脚質・距離適性の正しいエンコード(one-hot)+ 候補s1 学習・検証。

設計(本日の解剖・棚卸しで確定):
  jrdb_running_style(1逃2先3差4追)/ jrdb_dist_apt(1短2中3長5マイル6万能)は
  ★カテゴリコードを数値の大小として扱い★ gain≒0.1%/0.05% で死亡。CSV充足は90%/70%。
  → one-hot化(大小関係を廃す)+「今回距離×距離適性の合致フラグ」で蘇生を試す。

s1 = 現V16(137能力特徴)から jrdb_running_style / jrdb_dist_apt(死んだ数値版)を抜き、
     one-hot版(脚質4 + 距離適性5 + 合致1 = 10特徴)に差し替え。★騎手指数はまだ残す★
     (エンコード修正単独の効果を純粋に測る。騎手指数を消すのは次ステップ)。

リーク: 脚質・距離適性は前日KYI、距離は事前確定 = 発走前確定 = リークなし。
本番 V15/V16 不変・predict_core 不変。出力: models/v16_anaba_s1_candidate.pkl.gz(候補/検証専用)。
"""
from __future__ import annotations
import os, sys, gzip, pickle, json, time
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
ODDS_REMOVE = ['paci_ninki_idx', 'odds_change_rate', 'odds_sharp_drop', 'oz_base_pop_rank',
               'oz_fukusho_base_log', 'oz_tansho_base_log', 'pop_rank_change', 'prev_odds_log']
RAW_REPLACE = ['jrdb_running_style', 'jrdb_dist_apt']  # 死んだ数値版を抜く
RS_ONEHOT = ['rs_nige', 'rs_senko', 'rs_sashi', 'rs_oikomi']           # 脚質 1逃2先3差4追
DA_ONEHOT = ['da_short', 'da_mid', 'da_long', 'da_mile', 'da_banno']   # 距離適性 1短2中3長5マイル6万能
MATCH = ['dist_apt_match']                                             # 今回距離×適性 合致
NEW_FEATS = RS_ONEHOT + DA_ONEHOT + MATCH
LGB_PARAMS = {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'num_leaves': 63,
              'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
              'min_child_samples': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'seed': 42}
XGB_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 6, 'learning_rate': 0.05,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 50, 'reg_alpha': 0.1,
              'reg_lambda': 0.1, 'seed': 42, 'tree_method': 'hist', 'verbosity': 0}
EVAL_YEARS = [23, 24, 25]
TOPK = 6


def build_onehots(df):
    rs = pd.to_numeric(df['jrdb_running_style'], errors='coerce').fillna(0)
    df['rs_nige']  = (rs == 1).astype(int)
    df['rs_senko'] = (rs == 2).astype(int)
    df['rs_sashi'] = (rs == 3).astype(int)
    df['rs_oikomi']= (rs == 4).astype(int)
    da = pd.to_numeric(df['jrdb_dist_apt'], errors='coerce').fillna(0)
    df['da_short'] = (da == 1).astype(int)
    df['da_mid']   = (da == 2).astype(int)
    df['da_long']  = (da == 3).astype(int)
    df['da_mile']  = (da == 5).astype(int)
    df['da_banno'] = (da == 6).astype(int)
    # 今回距離 → バケット → 距離適性との合致 (万能は任意一致)
    dist = pd.to_numeric(df['distance'], errors='coerce').fillna(0)
    b_short = dist <= 1400
    b_mile  = (dist > 1400) & (dist <= 1800)
    b_mid   = (dist > 1800) & (dist <= 2200)
    b_long  = dist > 2200
    match = ((da == 1) & b_short) | ((da == 5) & b_mile) | ((da == 2) & b_mid) | \
            ((da == 3) & b_long) | (da == 6)
    df['dist_apt_match'] = match.astype(int)
    return df


def train_predict(df, features, tr, te):
    Xtr, ytr = df.loc[tr, features].values, df.loc[tr, 'target'].values
    Xte, yte = df.loc[te, features].values, df.loc[te, 'target'].values
    dt = lgb.Dataset(Xtr, label=ytr); dv = lgb.Dataset(Xte, label=yte, reference=dt)
    m_lgb = lgb.train(LGB_PARAMS, dt, num_boost_round=1000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    p_lgb = m_lgb.predict(Xte)
    dxtr = xgb.DMatrix(Xtr, label=ytr); dxte = xgb.DMatrix(Xte, label=yte)
    m_xgb = xgb.train(XGB_PARAMS, dxtr, num_boost_round=1000, evals=[(dxte, 'v')],
                      early_stopping_rounds=50, verbose_eval=False)
    p = 0.5 * p_lgb + 0.5 * m_xgb.predict(dxte)
    return p, roc_auc_score(yte, p), m_lgb, m_xgb


def anaba_metrics(oof, score_col, market='s_v15', topk=TOPK):
    base = oof['target'].mean()
    hit_n = hit_h = races_pick = nrace = 0
    spear = []
    pick_match = []; pick_t = []  # 反市場ピックの dist_apt_match と target
    for rid, g in oof.groupby('rid'):
        if len(g) < 4: continue
        nrace += 1
        s = g[score_col].values; m = g[market].values; t = g['target'].values
        dm = g['dist_apt_match'].values
        top_s = set(np.argsort(-s)[:topk]); top_m = set(np.argsort(-m)[:topk])
        anti = [i for i in top_s if i not in top_m]
        if anti:
            races_pick += 1
            for i in anti:
                hit_h += 1; hit_n += int(t[i]); pick_match.append(int(dm[i])); pick_t.append(int(t[i]))
        if len(s) >= 3:
            r, _ = spearmanr(s, m)
            if not np.isnan(r): spear.append(r)
    pm = np.array(pick_match); pt = np.array(pick_t)
    return {'base_top3': float(base),
            'anti_market_hit_rate': float(hit_n / hit_h) if hit_h else float('nan'),
            'anti_market_picks': int(hit_h),
            'pick_freq': float(races_pick / nrace) if nrace else float('nan'),
            'spearman_vs_v15': float(np.mean(spear)) if spear else float('nan'),
            'n_races': int(nrace),
            'anti_pick_dist_match_rate': float(pm.mean()) if len(pm) else float('nan'),
            'anti_pick_hit_when_match': float(pt[pm == 1].mean()) if (pm == 1).any() else float('nan'),
            'anti_pick_hit_when_nomatch': float(pt[pm == 0].mean()) if (pm == 0).any() else float('nan')}


def gain_map(model, feats):
    g = dict(zip(model.feature_name(), model.feature_importance(importance_type='gain')))
    if all(k.startswith('Column_') for k in list(g)[:3]):
        g = {feats[int(k.split('_')[1])]: v for k, v in g.items()}
    tot = sum(g.values()) or 1.0
    return {f: 100.0 * g.get(f, 0) / tot for f in feats}


def main():
    t0 = time.time()
    obj = pickle.load(gzip.open(os.path.join(DATA, '_v15_optuna_df_cache.pkl.gz'), 'rb'))
    df = obj['df']; v15 = obj['features']
    if 'target' not in df.columns: df['target'] = (df['finish'] <= 3).astype(int)
    df['rid'] = df['race_id_unique'].astype(str)
    df = build_onehots(df)
    v16 = [f for f in v15 if f not in ODDS_REMOVE]
    s1 = [f for f in v16 if f not in RAW_REPLACE] + NEW_FEATS  # 死んだ数値版を抜き one-hot を足す
    allf = set(v15) | set(v16) | set(s1)
    for f in allf:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    print(f"V15={len(v15)} V16={len(v16)} s1={len(s1)} (V16から-2死特徴+{len(NEW_FEATS)}one-hot)")

    rows = []; aucs = {'v15': [], 'v16': [], 's1': []}; g16 = {}; gs1 = {}
    for ty in EVAL_YEARS:
        tr = df['year'] < ty; te = df['year'] == ty
        print(f"\n[WF {2000+ty}] train={tr.sum()} test={te.sum()}", flush=True)
        p15, a15, _, _ = train_predict(df, v15, tr, te)
        p16, a16, m16, _ = train_predict(df, v16, tr, te)
        ps1, as1, ms1, _ = train_predict(df, s1, tr, te)
        aucs['v15'].append(a15); aucs['v16'].append(a16); aucs['s1'].append(as1)
        print(f"  AUC V15={a15:.4f} V16={a16:.4f} s1={as1:.4f}", flush=True)
        for f, v in gain_map(m16, v16).items(): g16[f] = g16.get(f, 0) + v / len(EVAL_YEARS)
        for f, v in gain_map(ms1, s1).items(): gs1[f] = gs1.get(f, 0) + v / len(EVAL_YEARS)
        sub = df.loc[te, ['rid', 'target', 'dist_apt_match']].copy()
        sub['s_v15'] = p15; sub['s_v16'] = p16; sub['s_s1'] = ps1
        rows.append(sub)
    oof = pd.concat(rows, ignore_index=True)
    oof.to_parquet(os.path.join(DATA, 'v16_anaba_s1_oof.parquet'))

    wf = {k: float(np.mean(v)) for k, v in aucs.items()}
    print(f"\n=== WF AUC (2023-25) === V15={wf['v15']:.4f} V16={wf['v16']:.4f} s1={wf['s1']:.4f}")

    rs_old = g16.get('jrdb_running_style', 0); da_old = g16.get('jrdb_dist_apt', 0)
    rs_new = sum(gs1.get(f, 0) for f in RS_ONEHOT)
    da_new = sum(gs1.get(f, 0) for f in DA_ONEHOT)
    match_new = gs1.get('dist_apt_match', 0)
    print("\n=== エンコード前後 gain% ===")
    print(f"  脚質   : 数値版 {rs_old:.3f}%  →  one-hot合計 {rs_new:.3f}%")
    for f in RS_ONEHOT: print(f"      {f:12s} {gs1.get(f,0):.3f}%")
    print(f"  距離適性: 数値版 {da_old:.3f}%  →  one-hot合計 {da_new:.3f}% (+合致 {match_new:.3f}%)")
    for f in DA_ONEHOT: print(f"      {f:12s} {gs1.get(f,0):.3f}%")
    print(f"      {'dist_apt_match':12s} {match_new:.3f}%")
    print(f"  脚質指数(参考) jockey_exp合計 V16={g16.get('paci_jockey_exp_wr',0)+g16.get('paci_jockey_exp_3rd',0):.2f}% "
          f"s1={gs1.get('paci_jockey_exp_wr',0)+gs1.get('paci_jockey_exp_3rd',0):.2f}%")

    print("\n=== 穴発見力 (反市場=top6 & V15 top6圏外) ===")
    mv = anaba_metrics(oof, 's_v16'); ms = anaba_metrics(oof, 's_s1')
    print(f"  base 3着内率: {mv['base_top3']*100:.1f}%")
    print(f"  {'metric':26s}{'V16(現)':>10s}{'s1':>10s}")
    print(f"  {'反市場好走率':26s}{mv['anti_market_hit_rate']*100:9.1f}%{ms['anti_market_hit_rate']*100:9.1f}%")
    print(f"  {'反市場ピック数':26s}{mv['anti_market_picks']:10d}{ms['anti_market_picks']:10d}")
    print(f"  {'穴ピック頻度':26s}{mv['pick_freq']*100:9.1f}%{ms['pick_freq']*100:9.1f}%")
    print(f"  {'spearman vs V15':26s}{mv['spearman_vs_v15']:10.4f}{ms['spearman_vs_v15']:10.4f}")
    print(f"\n  [s1] 反市場ピックの距離適性合致率: {ms['anti_pick_dist_match_rate']*100:.1f}%")
    print(f"  [s1] 反市場ピック 3着内率: 合致時 {ms['anti_pick_hit_when_match']*100:.1f}% / 非合致時 {ms['anti_pick_hit_when_nomatch']*100:.1f}%")

    # s1 候補 全データ学習・保存
    print("\n=== s1 候補 全データ学習+保存 ===", flush=True)
    mask = (df['year'] >= 20) & (df['year'] <= 25)
    X, y = df.loc[mask, s1].values, df.loc[mask, 'target'].values
    m_lgb = lgb.train(LGB_PARAMS, lgb.Dataset(X, label=y), num_boost_round=500)
    m_xgb = xgb.train(XGB_PARAMS, xgb.DMatrix(X, label=y), num_boost_round=500,
                      evals=[(xgb.DMatrix(X, label=y), 't')], verbose_eval=False)
    out = os.path.join(BASE, 'models', 'v16_anaba_s1_candidate.pkl.gz')
    pickle.dump({'version': 'v16_anaba_s1_candidate',
                 'description': 'V16 137 ability feats with running_style/dist_apt re-encoded as one-hot + dist match flag. Jockey-index kept. Candidate/paper only.',
                 'model': m_lgb, 'xgb_model': m_xgb, 'ensemble_weights': {'lgb': 0.5, 'xgb': 0.5, 'mlp': 0},
                 'features': s1, 'n_features': len(s1), 'new_features': NEW_FEATS, 'replaced': RAW_REPLACE,
                 'wf_auc_mean': wf['s1'], 'wf_auc_v16': wf['v16'], 'wf_auc_v15': wf['v15'],
                 'gain_running_style': {'numeric_v16': rs_old, 'onehot_s1': rs_new},
                 'gain_dist_apt': {'numeric_v16': da_old, 'onehot_s1': da_new, 'match': match_new},
                 'anaba_metrics_s1': ms, 'anaba_metrics_v16': mv,
                 'leak_free': True, 'is_live': False, 'is_candidate': True, 'parent': 'v16_ability_candidate'},
                gzip.open(out, 'wb'), protocol=4)
    print(f"  saved: {out}")
    json.dump({'wf_auc': wf, 'gain_rs': {'v16': rs_old, 's1': rs_new},
               'gain_da': {'v16': da_old, 's1': da_new, 'match': match_new},
               'metrics_v16': mv, 'metrics_s1': ms},
              open(os.path.join(DATA, 'v16_anaba_s1_summary.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
