# -*- coding: utf-8 -*-
"""v15r-B: CYB7追加の147特徴 WF学習 (PREREG_V15R_B.md 準拠)。"""
import json,os,sys
BASE=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
HERE=os.path.dirname(os.path.abspath(__file__))
if sys.platform=='win32': sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd,lightgbm as lgb,xgboost as xgb
from sklearn.metrics import roc_auc_score
LGB_P=dict(objective='binary',metric='auc',num_leaves=63,learning_rate=0.05,feature_fraction=0.8,
 bagging_fraction=0.8,bagging_freq=5,min_child_samples=50,reg_alpha=0.1,reg_lambda=0.1,verbose=-1,seed=42)
XGB_P=dict(objective='binary:logistic',eval_metric='auc',max_depth=6,learning_rate=0.05,subsample=0.8,
 colsample_bytree=0.8,min_child_weight=50,reg_alpha=0.1,reg_lambda=0.1,seed=42,tree_method='hist')
CYB=['cyb_train_type','cyb_course_type','cyb_baba','cyb_mark','cyb_amount','cyb_change','cyb_eval']
def wf(df,feats):
    aucs=[]
    for Y in [21,22,23,24,25]:
        tr=df[df['year']<Y]; va=df[df['year']==Y-1]; te=df[df['year']==Y]
        y=lambda s:(s['finish']<=3).astype(int); X=lambda s:s[feats].apply(pd.to_numeric,errors='coerce')
        lm=lgb.train(LGB_P,lgb.Dataset(X(tr),y(tr)),num_boost_round=1000,
            valid_sets=[lgb.Dataset(X(va),y(va))],callbacks=[lgb.early_stopping(50,verbose=False)])
        xm=xgb.train(XGB_P,xgb.DMatrix(X(tr),label=y(tr)),num_boost_round=1000,
            evals=[(xgb.DMatrix(X(va),label=y(va)),'v')],early_stopping_rounds=50,verbose_eval=False)
        p=0.5*lm.predict(X(te))+0.5*xm.predict(xgb.DMatrix(X(te)))
        a=roc_auc_score(y(te),p); aucs.append(a); print(f"    fold20{Y}: {a:.4f}",flush=True)
    return float(np.mean(aucs))
def main():
    df=pd.read_parquet(os.path.join(HERE,'v15r_train.parquet'))
    df['year']=pd.to_numeric(df['year'])
    # CYB join
    cyb=pd.read_csv(os.path.join(BASE,'data','jrdb_cyb.csv'),dtype=str,encoding='utf-8-sig')
    cyb['_key']=cyb['race_id'].astype(str)+'_'+pd.to_numeric(cyb['umaban'],errors='coerce').fillna(0).astype(int).astype(str)
    src=['train_type','train_course_type','train_baba','train_mark','train_amount','train_change','train_eval']
    for s,d in zip(src,CYB): cyb[d]=pd.to_numeric(cyb[s],errors='coerce')
    cyb=cyb.drop_duplicates('_key',keep='last').set_index('_key')
    key=df['nk_race_id']+'_'+df['umaban'].astype(int).astype(str)
    add=cyb.reindex(key.values)[CYB].reset_index(drop=True).set_index(df.index)
    print(f"CYB充足: {add['cyb_eval'].notna().mean():.2%}")
    df=pd.concat([df,add],axis=1)
    fj=json.load(open(os.path.join(HERE,'v15r_features.json'),encoding='utf-8'))
    full=fj['base_from_v15']+fj['jv']+fj['srb']+fj['kka']
    res={}
    print("=== (aB) full147 ==="); res['aB']=wf(df,full+CYB)
    print("=== (bB) -CYB 140 ==="); res['bB']=wf(df,full)
    res['contrib_cyb']=round(res['aB']-res['bB'],5)
    json.dump(res,open(os.path.join(HERE,'v15r_b_results.json'),'w'),indent=1)
    thr=0.8357
    print(f"\naB={res['aB']:.4f} bB={res['bB']:.4f} CYB寄与={res['contrib_cyb']:+.4f}")
    print(f"合格(≥{thr}): {'★合格★' if res['aB']>=thr else '★不合格★'}")
    print(f"CYB: {'保持' if res['contrib_cyb']>0 else '★落とす★'}")
if __name__=='__main__': main()
