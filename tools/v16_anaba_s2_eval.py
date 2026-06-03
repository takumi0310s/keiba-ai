#!/usr/bin/env python3
"""V16 作り直し ステップ2 — 人気代理"族"一括除去 + 脚質/距離適性のレース相対・交互特徴化。

族の確定(フルオッズ基底の残差回帰, r2_by_odds>0.5 & 残差target保持<25% = ほぼ純人気代理):
  paci_jockey_exp_wr/_3rd, paci_jockey_mark/sogo_mark/train_mark/idm_mark(JRDB印),
  jrdb_cid_idx(retain0.01%), jrdb_ls_idx(0.56%), jrdb_training_idx(11.6%), jrdb_stable_idx(10.5%) = 10件。
  ※生の調教実測(training_time_filled 等)・ze_idm・career・idm は能力として残す。

レース相対・交互特徴(per-horse単一コードは死=ステップ1で確認 → レース文脈で作り直す):
  脚質構成: n_nige/n_front/front_ratio/is_lone_nige/is_lone_front/front_advantage(レース内相対前進度)
  距離適性: n_apt_match/is_rare_apt_match
  交互: inner_draw(内枠flag)・front_x_inner・front_x_innerbias・draw_x_innerbias (脚質×バイアス×枠)

リーク: 脚質(前日KYI)・距離適性(KYI)・距離(事前)・枠(確定)・トラックバイアス(KAB前日) = 全て発走前確定。
本番 V15/V16 不変。出力 models/v16_anaba_s2_candidate.pkl.gz(候補/検証専用・投票未使用)。
"""
from __future__ import annotations
import os, sys, gzip, pickle, json, time
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v16_anaba_s1_eval import train_predict, anaba_metrics, gain_map, EVAL_YEARS, LGB_PARAMS, XGB_PARAMS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA = os.path.join(BASE, "data")
ODDS_REMOVE = ['paci_ninki_idx', 'odds_change_rate', 'odds_sharp_drop', 'oz_base_pop_rank',
               'oz_fukusho_base_log', 'oz_tansho_base_log', 'pop_rank_change', 'prev_odds_log']
PROXY_FAMILY = ['paci_jockey_exp_wr', 'paci_jockey_exp_3rd', 'paci_jockey_mark', 'paci_sogo_mark',
                'paci_train_mark', 'paci_idm_mark', 'jrdb_cid_idx', 'jrdb_ls_idx',
                'jrdb_training_idx', 'jrdb_stable_idx']
RAW_REPLACE = ['jrdb_running_style', 'jrdb_dist_apt']
ONEHOT = ['rs_nige', 'rs_senko', 'rs_sashi', 'rs_oikomi',
          'da_short', 'da_mid', 'da_long', 'da_mile', 'da_banno', 'dist_apt_match']
RELATIVE = ['n_nige', 'n_front', 'front_ratio', 'is_lone_nige', 'is_lone_front', 'front_advantage',
            'n_apt_match', 'is_rare_apt_match', 'inner_draw',
            'front_x_inner', 'front_x_innerbias', 'draw_x_innerbias']
NEW = ONEHOT + RELATIVE


def build_features(df):
    df['rid'] = df['race_id_unique'].astype(str)
    rs = pd.to_numeric(df['jrdb_running_style'], errors='coerce').fillna(0)
    da = pd.to_numeric(df['jrdb_dist_apt'], errors='coerce').fillna(0)
    dist = pd.to_numeric(df['distance'], errors='coerce').fillna(0)
    nh = pd.to_numeric(df['num_horses_val'], errors='coerce').fillna(0).clip(lower=1)
    hn = pd.to_numeric(df['horse_num'], errors='coerce').fillna(99)
    tb = pd.to_numeric(df.get('jrdb_tb_homestr_inner', 0), errors='coerce').fillna(0)
    # --- one-hot (脚質/距離適性) ---
    for v, n in [(1, 'rs_nige'), (2, 'rs_senko'), (3, 'rs_sashi'), (4, 'rs_oikomi')]:
        df[n] = (rs == v).astype(int)
    for v, n in [(1, 'da_short'), (2, 'da_mid'), (3, 'da_long'), (5, 'da_mile'), (6, 'da_banno')]:
        df[n] = (da == v).astype(int)
    b_short = dist <= 1400; b_mile = (dist > 1400) & (dist <= 1800)
    b_mid = (dist > 1800) & (dist <= 2200); b_long = dist > 2200
    df['dist_apt_match'] = (((da == 1) & b_short) | ((da == 5) & b_mile) | ((da == 2) & b_mid) |
                            ((da == 3) & b_long) | (da == 6)).astype(int)
    # --- 前進度スコア(逃3先2差1追0、不明=1) ---
    front_score = rs.map({1: 3.0, 2: 2.0, 3: 1.0, 4: 0.0, 0: 1.0}).fillna(1.0)
    df['_fs'] = front_score
    df['_is_nige'] = (rs == 1).astype(int)
    df['_is_front'] = rs.isin([1, 2]).astype(int)
    g = df.groupby('rid')
    df['n_nige'] = g['_is_nige'].transform('sum')
    df['n_front'] = g['_is_front'].transform('sum')
    df['front_ratio'] = df['n_front'] / nh
    df['is_lone_nige'] = ((rs == 1) & (df['n_nige'] == 1)).astype(int)
    df['is_lone_front'] = ((df['_is_front'] == 1) & (df['n_front'] <= 2)).astype(int)
    df['front_advantage'] = front_score - g['_fs'].transform('mean')   # レース内で自分がどれだけ前か
    df['n_apt_match'] = g['dist_apt_match'].transform('sum')
    df['is_rare_apt_match'] = ((df['dist_apt_match'] == 1) & (df['n_apt_match'] <= 3)).astype(int)
    # --- 枠 × バイアス 交互 ---
    inner_th = np.ceil(0.35 * nh)
    df['inner_draw'] = (hn <= inner_th).astype(int)
    df['front_x_inner'] = front_score * df['inner_draw']
    df['front_x_innerbias'] = front_score * tb        # 符号は木に学習させる
    df['draw_x_innerbias'] = df['inner_draw'] * tb
    return df


def main():
    t0 = time.time()
    obj = pickle.load(gzip.open(os.path.join(DATA, '_v15_optuna_df_cache.pkl.gz'), 'rb'))
    df = obj['df']; v15 = obj['features']
    if 'target' not in df.columns: df['target'] = (df['finish'] <= 3).astype(int)
    df = build_features(df)
    v16 = [f for f in v15 if f not in ODDS_REMOVE]
    s2 = [f for f in v16 if f not in (PROXY_FAMILY + RAW_REPLACE)] + NEW
    for f in set(v15) | set(s2):
        if f in df.columns: df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    print(f"V15={len(v15)} V16={len(v16)} s2={len(s2)}  (V16 -{len(PROXY_FAMILY)}proxy -2raw +{len(NEW)}new)")
    print(f"除去した人気代理族: {PROXY_FAMILY}")

    rows = []; aucs = {'v15': [], 'v16': [], 's2': []}; gs2 = {}
    for ty in EVAL_YEARS:
        tr = df['year'] < ty; te = df['year'] == ty
        print(f"\n[WF {2000+ty}] train={tr.sum()} test={te.sum()}", flush=True)
        p15, a15, _, _ = train_predict(df, v15, tr, te)
        p16, a16, _, _ = train_predict(df, v16, tr, te)
        ps2, as2, ms2, _ = train_predict(df, s2, tr, te)
        aucs['v15'].append(a15); aucs['v16'].append(a16); aucs['s2'].append(as2)
        print(f"  AUC V15={a15:.4f} V16={a16:.4f} s2={as2:.4f}", flush=True)
        for f, v in gain_map(ms2, s2).items(): gs2[f] = gs2.get(f, 0) + v / len(EVAL_YEARS)
        sub = df.loc[te, ['rid', 'target', 'dist_apt_match']].copy()
        sub['s_v15'] = p15; sub['s_v16'] = p16; sub['s_s2'] = ps2
        rows.append(sub)
    oof = pd.concat(rows, ignore_index=True)
    oof.to_parquet(os.path.join(DATA, 'v16_anaba_s2_oof.parquet'))
    wf = {k: float(np.mean(v)) for k, v in aucs.items()}
    print(f"\n=== WF AUC === V15={wf['v15']:.4f} V16={wf['v16']:.4f} s2={wf['s2']:.4f}")

    print("\n=== レース相対・交互特徴の gain% (s2) ===")
    for f in RELATIVE:
        print(f"  {f:20s} {gs2.get(f,0):.3f}%")
    print(f"  脚質one-hot合計 {sum(gs2.get(f,0) for f in ONEHOT[:4]):.3f}% / 距離適性one-hot {sum(gs2.get(f,0) for f in ONEHOT[4:9]):.3f}% / 合致 {gs2.get('dist_apt_match',0):.3f}%")
    print("  gain TOP15:")
    for f, v in sorted(gs2.items(), key=lambda x: -x[1])[:15]:
        print(f"      {f:24s} {v:.2f}%")

    mv = anaba_metrics(oof, 's_v16'); ms = anaba_metrics(oof, 's_s2')
    print("\n=== 穴発見力 (反市場=top6 & V15 top6圏外) ===")
    print(f"  base={mv['base_top3']*100:.1f}%")
    print(f"  {'metric':24s}{'V16(現)':>11s}{'s2':>11s}")
    print(f"  {'反市場好走率':24s}{mv['anti_market_hit_rate']*100:10.1f}%{ms['anti_market_hit_rate']*100:10.1f}%")
    print(f"  {'反市場ピック数':24s}{mv['anti_market_picks']:11d}{ms['anti_market_picks']:11d}")
    print(f"  {'穴ピック頻度':24s}{mv['pick_freq']*100:10.1f}%{ms['pick_freq']*100:10.1f}%")
    print(f"  {'spearman vs V15':24s}{mv['spearman_vs_v15']:11.4f}{ms['spearman_vs_v15']:11.4f}")
    print(f"\n  [s2] 反市場ピック 距離適性合致率 {ms['anti_pick_dist_match_rate']*100:.1f}% / 3着内 合致{ms['anti_pick_hit_when_match']*100:.1f}% 非合致{ms['anti_pick_hit_when_nomatch']*100:.1f}%")

    print("\n=== s2 候補 全データ学習+保存 ===", flush=True)
    mask = (df['year'] >= 20) & (df['year'] <= 25)
    X, y = df.loc[mask, s2].values, df.loc[mask, 'target'].values
    m_lgb = lgb.train(LGB_PARAMS, lgb.Dataset(X, label=y), num_boost_round=500)
    m_xgb = xgb.train(XGB_PARAMS, xgb.DMatrix(X, label=y), num_boost_round=500,
                      evals=[(xgb.DMatrix(X, label=y), 't')], verbose_eval=False)
    out = os.path.join(BASE, 'models', 'v16_anaba_s2_candidate.pkl.gz')
    pickle.dump({'version': 'v16_anaba_s2_candidate',
                 'description': 'V16 ability minus popularity-proxy family (10) + race-relative/interaction pace & aptitude & draw-bias features. Candidate/paper only.',
                 'model': m_lgb, 'xgb_model': m_xgb, 'ensemble_weights': {'lgb': 0.5, 'xgb': 0.5, 'mlp': 0},
                 'features': s2, 'n_features': len(s2), 'removed_proxy_family': PROXY_FAMILY,
                 'new_relative_features': RELATIVE, 'new_onehot': ONEHOT,
                 'wf_auc_mean': wf['s2'], 'wf_auc_v16': wf['v16'], 'wf_auc_v15': wf['v15'],
                 'anaba_metrics_s2': ms, 'anaba_metrics_v16': mv, 'gain_relative': {f: gs2.get(f, 0) for f in RELATIVE},
                 'leak_free': True, 'is_live': False, 'is_candidate': True, 'parent': 'v16_ability_candidate'},
                gzip.open(out, 'wb'), protocol=4)
    print(f"  saved: {out}")
    json.dump({'wf_auc': wf, 'removed_family': PROXY_FAMILY, 'gain_relative': {f: gs2.get(f, 0) for f in RELATIVE},
               'metrics_v16': mv, 'metrics_s2': ms},
              open(os.path.join(DATA, 'v16_anaba_s2_summary.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
