#!/usr/bin/env python3
"""診断(次ステップ先取り): 騎手指数を消したら one-hot 脚質/距離適性 は蘇るか。
s1(one-hot版・騎手指数あり)から paci_jockey_exp_wr/_3rd を抜いた s2 を WF 学習し、
脚質/距離適性 one-hot の gain・spearman・反市場好走率を測る。診断専用=候補保存しない・投票未使用。
本番不変。market参照(s_v15)と target は既存 data/v16_anaba_s1_oof.parquet を再利用(同一concat順)。
"""
from __future__ import annotations
import os, sys, gzip, pickle, json
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v16_anaba_s1_eval import (build_onehots, train_predict, anaba_metrics, gain_map,
                               ODDS_REMOVE, RAW_REPLACE, NEW_FEATS, RS_ONEHOT, DA_ONEHOT,
                               EVAL_YEARS, LGB_PARAMS, XGB_PARAMS)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA = os.path.join(BASE, "data")
JOCKEY = ['paci_jockey_exp_wr', 'paci_jockey_exp_3rd']

obj = pickle.load(gzip.open(os.path.join(DATA, '_v15_optuna_df_cache.pkl.gz'), 'rb'))
df = obj['df']; v15 = obj['features']
if 'target' not in df.columns: df['target'] = (df['finish'] <= 3).astype(int)
df['rid'] = df['race_id_unique'].astype(str); df = build_onehots(df)
v16 = [f for f in v15 if f not in ODDS_REMOVE]
s1 = [f for f in v16 if f not in RAW_REPLACE] + NEW_FEATS
s2 = [f for f in s1 if f not in JOCKEY]   # ★ 騎手指数を消す ★
for f in set(v15) | set(s2):
    if f in df.columns: df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
print(f"s2 = s1({len(s1)}) - jockey_exp({len(JOCKEY)}) = {len(s2)} feats")

oof_prev = pd.read_parquet(os.path.join(DATA, 'v16_anaba_s1_oof.parquet'))  # s_v15, target, rid, dist_apt_match (te-concat順)
preds = []; aucs = []; gs2 = {}
for ty in EVAL_YEARS:
    tr = df['year'] < ty; te = df['year'] == ty
    p, a, m, _ = train_predict(df, s2, tr, te)
    preds.append(p); aucs.append(a)
    for f, v in gain_map(m, s2).items(): gs2[f] = gs2.get(f, 0) + v / len(EVAL_YEARS)
    print(f"[WF {2000+ty}] s2 AUC={a:.4f}")
s2_pred = np.concatenate(preds)
assert len(s2_pred) == len(oof_prev), f"len mismatch {len(s2_pred)} vs {len(oof_prev)}"
oof = oof_prev.copy(); oof['s_s2'] = s2_pred

print(f"\n=== s2 WF AUC = {np.mean(aucs):.4f} ===")
rs = sum(gs2.get(f, 0) for f in RS_ONEHOT); da = sum(gs2.get(f, 0) for f in DA_ONEHOT); mt = gs2.get('dist_apt_match', 0)
print("=== 騎手指数を消した後の gain% ===")
print(f"  脚質 one-hot 合計   = {rs:.3f}%")
for f in RS_ONEHOT: print(f"      {f:12s} {gs2.get(f,0):.3f}%")
print(f"  距離適性 one-hot 合計= {da:.3f}%  (+合致 {mt:.3f}%)")
for f in DA_ONEHOT: print(f"      {f:12s} {gs2.get(f,0):.3f}%")
print(f"      dist_apt_match {mt:.3f}%")
print("  gain TOP12:")
for f, v in sorted(gs2.items(), key=lambda x: -x[1])[:12]: print(f"      {f:26s} {v:.2f}%")

m = anaba_metrics(oof, 's_s2')
mv = anaba_metrics(oof, 's_v16')
print("\n=== 穴発見力 (反市場=top6 & V15圏外) ===")
print(f"  base={mv['base_top3']*100:.1f}%")
print(f"  反市場好走率: V16 {mv['anti_market_hit_rate']*100:.1f}% → s2(騎手指数なし) {m['anti_market_hit_rate']*100:.1f}%")
print(f"  spearman vs V15: V16 {mv['spearman_vs_v15']:.4f} → s2 {m['spearman_vs_v15']:.4f}")
print(f"  穴ピック頻度: V16 {mv['pick_freq']*100:.1f}% → s2 {m['pick_freq']*100:.1f}%")
print(f"  反市場ピック数: {m['anti_market_picks']}")
print(f"  [s2] 反市場ピック距離適性合致率 {m['anti_pick_dist_match_rate']*100:.1f}% / 3着内 合致{m['anti_pick_hit_when_match']*100:.1f}% 非合致{m['anti_pick_hit_when_nomatch']*100:.1f}%")
json.dump({'s2_wf_auc': float(np.mean(aucs)), 'gain_rs': rs, 'gain_da': da, 'gain_match': mt,
           'metrics_s2': m}, open(os.path.join(DATA, 'v16_anaba_s2_preview.json'), 'w'),
          ensure_ascii=False, indent=2)
print("\nDONE (診断のみ・候補保存なし・投票未使用)")
