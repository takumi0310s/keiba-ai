#!/usr/bin/env python3
"""診断: 人気代理族に paci_goal_rank/goal_diff(JRDB予想着順=市場アンカー)も加えて消すと
spearman が更に落ちるか・罠回避を保てるか。s2 の WF を再利用(s_v15/s_v16 は既存 parquet)。
診断専用=候補保存しない・投票未使用・本番不変。"""
from __future__ import annotations
import os, sys, gzip, pickle, json
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v16_anaba_s2_eval import (build_features, ODDS_REMOVE, PROXY_FAMILY, RAW_REPLACE, NEW, RELATIVE)
from v16_anaba_s1_eval import train_predict, anaba_metrics, gain_map, EVAL_YEARS
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA = os.path.join(BASE, "data")
EXTRA = ['paci_goal_rank', 'paci_goal_diff', 'paci_dochu_rank']   # JRDB予想着順/道中=市場アンカー

obj = pickle.load(gzip.open(os.path.join(DATA, '_v15_optuna_df_cache.pkl.gz'), 'rb'))
df = obj['df']; v15 = obj['features']
if 'target' not in df.columns: df['target'] = (df['finish'] <= 3).astype(int)
df = build_features(df)
v16 = [f for f in v15 if f not in ODDS_REMOVE]
fam = PROXY_FAMILY + [e for e in EXTRA if e in v16]
s2b = [f for f in v16 if f not in (fam + RAW_REPLACE)] + NEW
for f in set(v15) | set(s2b):
    if f in df.columns: df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
print(f"s2b = V16 -{len(fam)}proxy(族+予想着順) -2raw +{len(NEW)}new = {len(s2b)} feats")
print(f"追加除去: {[e for e in EXTRA if e in v16]}")

prev = pd.read_parquet(os.path.join(DATA, 'v16_anaba_s2_oof.parquet'))  # s_v15, s_v16, target, dist_apt_match (te順)
preds = []; aucs = []; g = {}
for ty in EVAL_YEARS:
    tr = df['year'] < ty; te = df['year'] == ty
    p, a, m, _ = train_predict(df, s2b, tr, te)
    preds.append(p); aucs.append(a)
    for f, v in gain_map(m, s2b).items(): g[f] = g.get(f, 0) + v / len(EVAL_YEARS)
    print(f"[WF {2000+ty}] s2b AUC={a:.4f}")
oof = prev.copy(); oof['s_s2b'] = np.concatenate(preds)
print(f"\n=== s2b WF AUC = {np.mean(aucs):.4f} (V16=0.8690) ===")
print("gain TOP12:")
for f, v in sorted(g.items(), key=lambda x: -x[1])[:12]: print(f"  {f:24s} {v:.2f}%")
print("レース相対 gain:", {f: round(g.get(f, 0), 3) for f in RELATIVE})
mv = anaba_metrics(oof, 's_v16'); m = anaba_metrics(oof, 's_s2b')
print("\n=== 穴発見力 ===")
print(f"  base={mv['base_top3']*100:.1f}%")
print(f"  反市場好走率: V16 {mv['anti_market_hit_rate']*100:.1f}% → s2b {m['anti_market_hit_rate']*100:.1f}%")
print(f"  spearman vs V15: V16 {mv['spearman_vs_v15']:.4f} → s2 0.9497 → s2b {m['spearman_vs_v15']:.4f}")
print(f"  穴ピック頻度: V16 {mv['pick_freq']*100:.1f}% → s2b {m['pick_freq']*100:.1f}%")
print(f"  反市場ピック数 {m['anti_market_picks']}")
json.dump({'s2b_wf_auc': float(np.mean(aucs)), 'metrics_s2b': m, 'family': fam},
          open(os.path.join(DATA, 'v16_anaba_s2b_preview.json'), 'w'), ensure_ascii=False, indent=2)
print("\nDONE (診断のみ・候補保存なし)")
