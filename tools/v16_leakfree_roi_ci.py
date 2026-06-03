#!/usr/bin/env python3
"""headline ROI のブートストラップCI(leak-free)。三連複top4box/単勝 が本物の正ROIか分散かを判定。
leak-free cache使用・本番不変・投票未使用。"""
from __future__ import annotations
import os,sys,gzip,pickle,itertools,time
if sys.platform=="win32": sys.stdout.reconfigure(encoding="utf-8")
import numpy as np,pandas as pd,lightgbm as lgb,xgboost as xgb
sys.path.insert(0,os.path.abspath('tools'))
from v16_anaba_s2_eval import build_features,ODDS_REMOVE,PROXY_FAMILY,RAW_REPLACE,NEW
from v16_leakfree_roi_grid import load_payouts,LGB_P,XGB_P
EXTRA=['paci_goal_rank','paci_goal_diff','paci_dochu_rank']; DATA='data'
obj=pickle.load(gzip.open(os.path.join(DATA,'_v15_optuna_df_cache_leakfree.pkl.gz'),'rb')); df=obj['df']; v15=obj['features']
if 'target' not in df: df['target']=(df['finish']<=3).astype(int)
df=build_features(df)
df['pop']=pd.to_numeric(df['oz_base_pop_rank'],errors='coerce').fillna(df['num_horses_val'])
df['_rk']=[f"{int(a)}_{b}_{int(c)}_{int(e)}_{int(f)}" for a,b,c,e,f in zip(df['date_num'],df['course'],df['kai'],df['nichi'],df['race_num'])]
v16=[f for f in v15 if f not in ODDS_REMOVE]; s2b=[f for f in v16 if f not in (PROXY_FAMILY+EXTRA+RAW_REPLACE)]+NEW
for f in set(v15)|set(s2b): df[f]=pd.to_numeric(df[f],errors='coerce').fillna(0)
y=pd.to_numeric(df['year'],errors='coerce')
def oof(feats):
    s=pd.Series(index=df.index,dtype=float)
    for ty in [23,24,25]:
        tr=y<ty; te=y==ty
        ml=lgb.train(LGB_P,lgb.Dataset(df.loc[tr,feats].values,label=df.loc[tr,'target'].values),num_boost_round=420)
        mx=xgb.train(XGB_P,xgb.DMatrix(df.loc[tr,feats].values,label=df.loc[tr,'target'].values),num_boost_round=420)
        s.loc[te]=0.5*ml.predict(df.loc[te,feats].values)+0.5*mx.predict(xgb.DMatrix(df.loc[te,feats].values))
    return s
print("WF学習...",flush=True)
df['s_s2b']=oof(s2b); df['s_v15']=oof(v15)
pay=load_payouts(); ev=df[y>=23]
def per_race(score):
    # 各レース: (単勝return, 三連複top4box return, 三連複top4 points)
    rows=[]
    for k,g in ev.groupby('_rk'):
        if k not in pay or len(g)<5: continue
        o=[int(x) for x in g.sort_values(score,ascending=False)['horse_num'].tolist()]
        pm=pay[k]; tan=pm['tan'][1] if pm['tan'][0]==o[0] else 0
        t4=0
        if pm['trio']:
            won=pm['trio'][0]
            if any(frozenset(c)==won for c in itertools.combinations(o[:4],3)): t4=pm['trio'][1]
        rows.append((tan, t4))
    return np.array(rows)
def boot(arr,points,n=2000):
    # ROI = sum(return)/(100*points*N). bootstrap over races
    N=len(arr); rois=[]
    idx=np.arange(N)
    rng=np.random.RandomState(0)
    for _ in range(n):
        s=rng.choice(idx,N,replace=True)
        rois.append(arr[s].sum()/(100*points*N))
    return np.mean(arr)/( 100*points)*len(arr)/len(arr), np.percentile(rois,2.5),np.percentile(rois,97.5)
for m in ['s2b','v15']:
    a=per_race(f's_{m}')
    tan=a[:,0]; t4=a[:,1]
    # 単勝 (1点)
    r=tan.sum()/(100*len(tan)); lo,hi=np.percentile([tan[np.random.RandomState(i).choice(len(tan),len(tan),True)].sum()/(100*len(tan)) for i in range(1500)],[2.5,97.5])
    print(f"[{m}] 単勝 ROI {r*100:.1f}%  95%CI [{lo*100:.0f}, {hi*100:.0f}]%  (N={len(tan)})")
    r4=t4.sum()/(100*4*len(t4)); lo4,hi4=np.percentile([t4[np.random.RandomState(i+999).choice(len(t4),len(t4),True)].sum()/(100*4*len(t4)) for i in range(1500)],[2.5,97.5])
    print(f"[{m}] 三連複top4box ROI {r4*100:.1f}%  95%CI [{lo4*100:.0f}, {hi4*100:.0f}]%  (N={len(t4)})")
print("DONE")
